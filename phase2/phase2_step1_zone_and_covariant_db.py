"""
PHASE 2 - STEP 1 (v2)
Structural zone decomposer + distributed relational covariant scanner.

v2 fixes (post phase2_mathematicalfix.md):
  * reads the repaired/whitened cache with mmap (never loads 2.6 GB into RAM)
  * top-K capped peer evaluation  -> O(N*D) GEMM + O(K*D) per row, no more
    materialising 500k-element peer sets or R_mat[peers] copies
  * growable numpy edge buffers instead of multi-million python tuples
  * parquet via pyarrow when available, else fastparquet
  * periodic checkpointing + ETA so a long shard can be resumed/monitored
"""
import os
import sys
import json
import time
import argparse

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIB_TICKS = [35, 57, 75, 93, 118]


# ---------------------------------------------------------------------------
# parquet helper (pyarrow is blocked by Application Control on this host)
# ---------------------------------------------------------------------------
import pqcompat

ENGINE = pqcompat.engine()


def write_parquet(df, path):
    if ENGINE is not None:
        pqcompat.write_parquet(df, path)
    else:
        df.to_csv(path.replace(".parquet", ".csv"), index=False)
    return ENGINE


# ---------------------------------------------------------------------------
# zone extraction (kept for provenance / cold start; not used when cache exists)
# ---------------------------------------------------------------------------
def zone_whitened_fft(y_slice, k_max=12):
    L = y_slice.shape[1]
    if L < 4:
        return np.zeros((y_slice.shape[0], max(1, k_max * 2 + 3)), dtype=np.float32)
    t_lin = np.linspace(0.0, 1.0, L, dtype=np.float32)[None, :]
    slope = y_slice[:, -1:] - y_slice[:, 0:1]
    bridge = y_slice - (y_slice[:, 0:1] + t_lin * slope)
    fft_c = np.fft.rfft(bridge, axis=1)[:, 1:k_max + 1] / max(L, 1)
    k_weights = np.arange(1, fft_c.shape[1] + 1, dtype=np.float32)[None, :]
    fft_w = fft_c * k_weights
    vec = np.hstack([fft_w.real, fft_w.imag, slope.astype(np.float32)]).astype(np.float32)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return vec / norms


# ---------------------------------------------------------------------------
# distributed relational covariant scan
# ---------------------------------------------------------------------------
def _boundaries(N, weights):
    b = [0]
    s = 0.0
    for w in weights:
        s += w
        b.append(int(round(N * s)))
    b[-1] = N
    return b


class EdgeBuffer:
    """Growable (n,4) float32 edge store - no multi-million python tuples."""

    def __init__(self, cap_hint=1_000_000):
        self.blocks = []
        self.n = 0
        self.block = np.empty((cap_hint, 4), dtype=np.float32)
        self.fill = 0

    def add(self, src, dst, cov, lock):
        if self.fill >= self.block.shape[0]:
            self.blocks.append(self.block[:self.fill])
            self.block = np.empty((max(65536, self.block.shape[0]), 4), dtype=np.float32)
            self.fill = 0
        self.block[self.fill] = (src, dst, cov, lock)
        self.fill += 1
        self.n += 1

    def array(self):
        parts = self.blocks + [self.block[:self.fill]]
        if not parts:
            return np.empty((0, 4), dtype=np.float32)
        return np.vstack(parts)


def scan_relational_covariant_shard(
    cache_path,
    node_id=0,
    num_nodes=3,
    sim_threshold=0.85,
    min_covariant_peers=12,
    chunk_size=256,
    top_peers=512,
    max_edges=24,
    output_dir="C:/cluster/phase2/shards",
    checkpoint_secs=120,
    max_rows=None,
    row_start_override=None,
    row_end_override=None,
):
    os.makedirs(output_dir, exist_ok=True)
    z = np.load(cache_path, mmap_mode="r")
    keys = set(z.files)

    if "trigger_vec_dense" in keys:
        T_mat = z["trigger_vec_dense"]
        R_mat = z["resolution_vec_dense"]
        mode = "dense(whitened)"
    else:
        T_mat = z["trigger_vec"]
        R_mat = z["resolution_vec"]
        mode = "fft(whitened)"

    N, D = T_mat.shape
    held_all = z["protected_held"]
    pol_all = z["ext_polarity"]
    ret_all = z["q4_net_return"]
    tick_all = z["ext_tick"]
    bico_key = "phase_closure_abs_cos" if "phase_closure_abs_cos" in keys else "bicoherence"
    bico_all = z[bico_key]

    if row_start_override is not None and row_end_override is not None:
        row_start, row_end = int(row_start_override), int(row_end_override)
    else:
        weights = [0.40, 0.30, 0.30] if num_nodes == 3 else [1.0 / num_nodes] * num_nodes
        bounds = _boundaries(N, weights)
        row_start, row_end = bounds[node_id], bounds[node_id + 1]
    if max_rows:
        row_end = min(row_end, row_start + max_rows)

    print(f"[*] Node {node_id}/{num_nodes}: rows {row_start:,}..{row_end:,} of {N:,} "
          f"| T={T_mat.shape} D={D} | thr={sim_threshold} min_peers={min_covariant_peers} "
          f"top_peers={top_peers} chunk={chunk_size} | mode={mode} | engine={ENGINE}", flush=True)

    T_T = np.asarray(T_mat).T.copy()  # (D,N) contiguous for fast GEMM
    node_rows = []   # list of tuples (compact)
    edges = EdgeBuffer()

    n_pass = 0
    n_scanned = 0
    t_start = time.time()
    t_ckpt = time.time()
    stats = {"max_abs_r": 0.0, "max_abs_r_cell": 0, "max_lock": -2.0, "max_lock_cell": 0,
             "max_abs_r_edge": 0.0, "edges": 0, "anomalies": 0}
    n_chunks = (row_end - row_start + chunk_size - 1) // chunk_size
    chunk_i = 0

    def checkpoint(tag="partial"):
        if not node_rows:
            return
        cols = ["anchor_cell_id", "covariant_peer_count", "peers_evaluated",
                "pos_corr_peers", "neg_mirror_peers", "mean_0_50_similarity",
                "max_0_50_similarity", "mean_50_100_lock_corr", "frac_75_100_exact_lock",
                "protected_ext_win_rate", "directional_50_100_win_rate",
                "direction_consistency", "anomaly_z_score", "bicoherence_phase_lock",
                "target_ext_tick_median", "recommended_action"]
        dfn = pd.DataFrame.from_records(node_rows, columns=cols)
        write_parquet(dfn, os.path.join(output_dir, f"covariant_anomalies_node{node_id}_{tag}.parquet"))
        e = edges.array()
        dfe = pd.DataFrame({"src_idx": e[:, 0].astype(np.int64),
                            "dst_idx": e[:, 1].astype(np.int64),
                            "cov_0_50": e[:, 2], "lock_50_100": e[:, 3]})
        write_parquet(dfe, os.path.join(output_dir, f"covariant_edges_node{node_id}_{tag}.parquet"))
        with open(os.path.join(output_dir, f"covariant_stats_node{node_id}_{tag}.json"), "w") as f:
            json.dump({"node": node_id, "rows_done": n_scanned, "row_end": row_end,
                       "anomalies": n_pass, "source": mode, "threshold": sim_threshold,
                       **stats}, f, indent=2)

    for i0 in range(row_start, row_end, chunk_size):
        i1 = min(i0 + chunk_size, row_end)
        Q = np.asarray(T_mat[i0:i1], dtype=np.float32)
        sim = Q @ T_T  # (rows_in_chunk, N) float32

        for lr in range(i1 - i0):
            g = i0 + lr
            row = sim[lr]
            lo, hi = max(0, g - 2), min(N, g + 3)
            row[lo:hi] = 0.0
            absr = np.abs(row)
            cand = np.flatnonzero(absr >= sim_threshold)
            n_cand = int(cand.size)
            if n_cand < min_covariant_peers:
                continue

            if n_cand > top_peers:
                part = cand[np.argpartition(absr[cand], -top_peers)[-top_peers:]]
                sel = part[np.argsort(absr[part])[::-1]]
            else:
                sel = cand[np.argsort(absr[cand])[::-1]]
            k = sel.size

            signed = row[sel]
            pol = np.sign(signed)
            z_res = R_mat[g]
            y_res = np.asarray(R_mat[sel], dtype=np.float32) * pol[:, None]
            cont = y_res @ np.asarray(z_res, dtype=np.float32)  # (k,)

            mean_cont = float(cont.mean())
            frac_lock = float(np.mean(cont >= 0.65))
            p_out = np.asarray(held_all[sel])
            p_dirs = np.asarray(pol_all[sel], dtype=np.int32) * pol
            dir_cons = float(abs(p_dirs.mean()))
            prot_win = float(p_out.mean())
            aligned_ret = np.asarray(ret_all[sel]) * pol
            med_ret = float(np.median(aligned_ret))
            dir_win = float(np.mean(np.sign(aligned_ret + 1e-18) == np.sign(med_ret + 1e-18)))
            edge_eff = max(prot_win, dir_win * dir_cons)
            z_sc = (edge_eff - 0.50) * np.sqrt(k) / 0.50
            mean_sim = float(absr[sel].mean())
            max_sim = float(absr[sel].max())

            stats["max_abs_r"] = max(stats["max_abs_r"], max_sim)
            if max_sim >= stats["max_abs_r"]:
                stats["max_abs_r_cell"] = int(g + 1)
            if mean_cont > stats["max_lock"]:
                stats["max_lock"] = mean_cont
                stats["max_lock_cell"] = int(g + 1)

            if edge_eff >= 0.53 or mean_cont >= 0.30:
                node_rows.append((
                    int(g + 1), n_cand, int(k),
                    int(np.sum(pol > 0)), int(np.sum(pol < 0)),
                    round(mean_sim, 4), round(max_sim, 4),
                    round(mean_cont, 4), round(frac_lock, 4),
                    round(prot_win, 4), round(dir_win, 4), round(dir_cons, 4),
                    round(float(z_sc), 3), round(float(bico_all[g]), 4),
                    int(np.median(np.asarray(tick_all[sel]))),
                    "SELL_PROTECTED_HIGH" if float(p_dirs.mean()) < 0 else "BUY_PROTECTED_LOW",
                ))
                n_pass += 1
                for j in range(min(max_edges, k)):
                    src, dst = int(g), int(sel[j])
                    if src == dst:
                        continue
                    if src > dst:
                        src, dst = dst, src
                    lock = float(cont[j])
                    edges.add(src, dst, float(signed[j]), lock)
                    if abs(float(signed[j])) > stats["max_abs_r_edge"]:
                        stats["max_abs_r_edge"] = abs(float(signed[j]))

        n_scanned += (i1 - i0)
        chunk_i += 1
        now = time.time()
        if now - t_ckpt >= checkpoint_secs:
            stats["anomalies"] = n_pass
            stats["edges"] = edges.n
            checkpoint()
            t_ckpt = now

        if chunk_i % 5 == 0 or chunk_i == n_chunks:
            el = time.time() - t_start
            rate = n_scanned / max(el, 1e-9)
            eta = (row_end - i1) / max(rate, 1e-9)
            print(f"    -> node {node_id}: {i1:,}/{row_end:,} rows "
                  f"({100.0 * (i1 - row_start) / max(1, row_end - row_start):5.1f}%) "
                  f"| {rate:,.0f} rows/s | elapsed {el/60:.1f}m | eta {eta/60:.1f}m "
                  f"| anomalies {n_pass:,} | edges {edges.n:,}", flush=True)

    stats["anomalies"] = n_pass
    stats["edges"] = edges.n
    checkpoint("final")

    # compact final artifacts (also overwrite the unsuffixed names used downstream)
    cols = ["anchor_cell_id", "covariant_peer_count", "peers_evaluated",
            "pos_corr_peers", "neg_mirror_peers", "mean_0_50_similarity",
            "max_0_50_similarity", "mean_50_100_lock_corr", "frac_75_100_exact_lock",
            "protected_ext_win_rate", "directional_50_100_win_rate",
            "direction_consistency", "anomaly_z_score", "bicoherence_phase_lock",
            "target_ext_tick_median", "recommended_action"]
    dfn = pd.DataFrame.from_records(node_rows, columns=cols)
    e = edges.array()
    dfe = pd.DataFrame({"src_idx": e[:, 0].astype(np.int64),
                        "dst_idx": e[:, 1].astype(np.int64),
                        "cov_0_50": e[:, 2], "lock_50_100": e[:, 3]})
    write_parquet(dfn, os.path.join(output_dir, f"covariant_anomalies_node{node_id}.parquet"))
    write_parquet(dfe, os.path.join(output_dir, f"covariant_edges_node{node_id}.parquet"))
    with open(os.path.join(output_dir, f"covariant_stats_node{node_id}.json"), "w") as f:
        json.dump({"node": node_id, "rows_scanned": n_scanned, "row_start": row_start,
                   "row_end": row_end, "anomalies": n_pass, "source": mode,
                   "threshold": sim_threshold, "top_peers": top_peers,
                   "minutes": round((time.time() - t_start) / 60.0, 2), **stats}, f, indent=2)
    print(f"[+] node {node_id} done: {n_pass:,} anomaly hubs, {edges.n:,} edges, "
          f"{(time.time() - t_start)/60:.1f} min -> {output_dir}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz_path", type=str, default="C:/cluster/phase2/jump10_phase2_subzones.npz")
    p.add_argument("--out_dir", type=str, default="C:/cluster/phase2")
    p.add_argument("--node_id", type=int, default=0)
    p.add_argument("--num_nodes", type=int, default=3)
    p.add_argument("--sim_threshold", type=float, default=0.85)
    p.add_argument("--min_peers", type=int, default=12)
    p.add_argument("--top_peers", type=int, default=512)
    p.add_argument("--max_edges", type=int, default=24)
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--checkpoint_secs", type=int, default=120)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--row_start", type=int, default=-1)
    p.add_argument("--row_end", type=int, default=-1)
    a = p.parse_args()

    if not os.path.exists(a.npz_path):
        print(f"[-] cache not found: {a.npz_path}", flush=True)
        sys.exit(1)
    scan_relational_covariant_shard(
        a.npz_path,
        node_id=a.node_id, num_nodes=a.num_nodes,
        sim_threshold=a.sim_threshold, min_covariant_peers=a.min_peers,
        chunk_size=a.chunk_size, top_peers=a.top_peers, max_edges=a.max_edges,
        output_dir=os.path.join(a.out_dir, "shards"),
        checkpoint_secs=a.checkpoint_secs,
        max_rows=(a.max_rows or None),
        row_start_override=(a.row_start if a.row_start >= 0 else None),
        row_end_override=(a.row_end if a.row_end >= 0 else None),
    )
