"""
PHASE 2 - COLLINEAR-COLLAPSE REPAIR
Implements C:\\cluster\\phase2_mathematicalfix.md:

  Bug A: single-shot bicoherence identity  |A*B*conj(C)| / (|A*B|*|C|) == 1.0 exactly
  Bug B: common-mode positive cone  ->  every cell at |r| >= 0.85 with every anchor

Fixes:
  * cross-sectional standardization (per feature)
  * SVD/PCA strip of the common-mode component(s) + Mahalanobis whitening
  * L2 unit-sphere projection    =>  E[r] = 0.0, sigma_r ~ 1/sqrt(D)
  * bicoherence replaced by phase-triad closure terms cos(phi1+phi2-phi3)

Memory-safe: mmap reads, two-pass chunked standardization, disk-backed outputs,
streamed zip writing. Never holds more than one float32 block of the matrix.
"""
import os
import sys
import io
import json
import time
import zipfile
import numpy as np

RAW = r"C:\cluster\phase2\jump10_phase2_subzones_raw.npz"
OUT = r"C:\cluster\phase2\jump10_phase2_subzones.npz"
TMP = r"C:\cluster\phase2\_white_tmp"
WNPZ = r"C:\cluster\phase2\whiten_transforms.npz"
REPORT = r"C:\cluster\phase2\whiten_report.json"

WHITEN_KEYS = [
    "trigger_vec", "resolution_vec",
    "trigger_vec_dense", "resolution_vec_dense",
    "dense_q1", "dense_q2", "dense_q3", "dense_q4", "dense_h1", "dense_h2",
]
CHUNK = 65536
SUBSAMPLE = 50000
STRIP_TOP_PC = 1
EPS = 1e-5
SEED = 42


def log(msg):
    print(msg, flush=True)


def two_pass_stats(A, chunk=CHUNK):
    n, d = A.shape
    s1 = np.zeros(d, dtype=np.float64)
    s2 = np.zeros(d, dtype=np.float64)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        b = np.asarray(A[i:j], dtype=np.float64)
        s1 += b.sum(axis=0)
        s2 += (b * b).sum(axis=0)
    mu = s1 / n
    var = np.maximum(s2 / n - mu * mu, 0.0)
    return mu.astype(np.float32), np.sqrt(var).astype(np.float32)


def fit_whitener(A, valid_cols, mu, sd):
    """Covariance on a random subsample of standardized data -> whitening W."""
    n = A.shape[0]
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n, size=min(SUBSAMPLE, n), replace=False))
    mu_sub = np.asarray(mu, dtype=np.float32)[valid_cols]
    sd_sub = np.asarray(sd, dtype=np.float32)[valid_cols]
    sub = np.empty((len(idx), int(valid_cols.sum())), dtype=np.float32)
    pos = 0
    for k in range(0, len(idx), CHUNK):
        sl = idx[k:k + CHUNK]
        block = np.asarray(A[sl], dtype=np.float32)[:, valid_cols]
        block = np.clip((block - mu_sub) / sd_sub, -6.0, 6.0)
        sub[pos:pos + block.shape[0]] = block
        pos += block.shape[0]

    cov = (sub.T @ sub) / sub.shape[0]
    cov = 0.5 * (cov + cov.T)
    U, S, _ = np.linalg.svd(cov, full_matrices=False)
    var_share = (S / max(S.sum(), 1e-12)) * 100.0
    start_pc = STRIP_TOP_PC if (S[0] / S.sum() > 0.35) else 0
    keep = np.arange(len(S)) >= start_pc
    keep &= S > 1e-3
    U_keep, S_keep = U[:, keep], S[keep]
    W = (U_keep @ np.diag(1.0 / np.sqrt(S_keep + EPS))).astype(np.float32)
    info = {
        "n_features_in": int(A.shape[1]),
        "n_constant_cols_dropped": int((~valid_cols).sum()),
        "n_features_kept": int(valid_cols.sum()),
        "n_components_out": int(W.shape[1]),
        "top5_eigenvalue_variance_share_pct": [round(float(x), 3) for x in var_share[:5]],
        "stripped_top_pc": int(start_pc),
    }
    return W, info


def transform_chunked(A, valid_cols, mu, sd, W, out_path, chunk=CHUNK):
    """y = L2normalize( clip((x-mu)/sd, +-6) @ W ) written to a disk-backed .npy"""
    n = A.shape[0]
    d_out = W.shape[1]
    mu_sub = np.asarray(mu, dtype=np.float32)[valid_cols]
    sd_sub = np.asarray(sd, dtype=np.float32)[valid_cols]
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, d_out))
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        b = np.asarray(A[i:j], dtype=np.float32)[:, valid_cols]
        b = np.clip((b - mu_sub) / sd_sub, -6.0, 6.0)
        y = b @ W
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm < 1e-9] = 1.0
        out[i:j] = y / nrm
    out.flush()
    del out
    return out_path


def main():
    t0 = time.time()
    if not os.path.exists(RAW):
        log(f"[-] raw npz not found: {RAW}")
        sys.exit(1)
    os.makedirs(TMP, exist_ok=True)

    log(f"[*] Opening {RAW} (mmap, no full load)")
    src = np.load(RAW, mmap_mode="r")
    keys = list(src.files)
    log(f"[*] {len(keys)} arrays: " + ", ".join(keys))
    report = {"input": RAW, "output": OUT, "arrays": {}, "whiteners": {}}

    # ---------------- pass 1: fix bicoherence BEFORE touching desc_Q1 --------
    log("\n[*] Replacing bicoherence with true phase-triad closure ...")
    q1 = np.asarray(src["desc_Q1"], dtype=np.float32)
    phi1 = np.arctan2(q1[:, 8], q1[:, 0] + 1e-9)
    phi2 = np.arctan2(q1[:, 9], q1[:, 1] + 1e-9)
    phi3 = np.arctan2(q1[:, 10], q1[:, 2] + 1e-9)
    triad = phi1 + phi2 - phi3
    bico_fixed = (0.5 * (np.abs(np.cos(triad)) + np.abs(np.sin(triad)))).astype(np.float32)
    phase_closure_cos = np.abs(np.cos(triad)).astype(np.float32)
    bico_old = np.asarray(src["bicoherence"], dtype=np.float32)
    report["bicoherence"] = {
        "old_min": float(bico_old.min()), "old_median": float(np.median(bico_old)),
        "old_max": float(bico_old.max()), "old_std": float(bico_old.std()),
        "new_min": float(bico_fixed.min()), "new_median": float(np.median(bico_fixed)),
        "new_max": float(bico_fixed.max()), "new_std": float(bico_fixed.std()),
        "phase_closure_abs_cos_median": float(np.median(phase_closure_cos)),
    }
    log(f"    old bico: min={bico_old.min():.6f} med={np.median(bico_old):.6f} "
        f"max={bico_old.max():.6f} (std={bico_old.std():.2e})")
    log(f"    new bico: min={bico_fixed.min():.6f} med={np.median(bico_fixed):.6f} "
        f"max={bico_fixed.max():.6f} (std={bico_fixed.std():.6f})")
    del bico_old, triad

    # ---------------- pass 2: whiten each descriptor matrix ------------------
    whitened_paths = {}
    for key in WHITEN_KEYS:
        if key not in src.files:
            log(f"[!] skip missing {key}")
            continue
        t1 = time.time()
        A = src[key]
        n, d = A.shape
        log(f"\n[*] Whitening `{key}` shape={A.shape} ...")
        mu, sd = two_pass_stats(A)
        sd_safe = np.where(sd < 1e-6, 1.0, sd)
        valid = sd >= 1e-6
        if valid.sum() < 2:
            log(f"[!] {key}: <2 non-constant columns, passing through")
            continue
        W, info = fit_whitener(A, valid, mu, sd_safe)
        p = os.path.join(TMP, f"{key}.npy")
        transform_chunked(A, valid, mu, sd_safe, W, p)
        whitened_paths[key] = p
        info["seconds"] = round(time.time() - t1, 2)
        report["whiteners"][key] = info
        report["arrays"][key] = {"in_shape": [n, d], "out_shape": [n, info["n_components_out"]]}
        log(f"    -> kept {info['n_features_kept']}/{d} cols, {info['n_components_out']} comps, "
            f"strip_top_pc={info['stripped_top_pc']}, top5 var%={info['top5_eigenvalue_variance_share_pct']}, "
            f"{info['seconds']}s")
        # persist W for later live/outbound scoring
        np.save(os.path.join(TMP, f"W_{key}.npy"), W)
        np.save(os.path.join(TMP, f"mu_{key}.npy"), mu)
        np.save(os.path.join(TMP, f"sd_{key}.npy"), sd_safe)
        np.save(os.path.join(TMP, f"cols_{key}.npy"), valid)
        del A, mu, sd, W

    # ---------------- pass 3: stream-write the repaired npz -----------------
    log(f"\n[*] Writing {OUT} ...")
    big_out = {k: np.load(p, mmap_mode="r") for k, p in whitened_paths.items()}
    tmp_out = OUT + ".tmp"
    order = []
    for k in keys:
        if k in whitened_paths:
            order.append(k)
    order += [k for k in keys if k not in whitened_paths]
    with zipfile.ZipFile(tmp_out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for k in order:
            if k in big_out:
                with zf.open(f"{k}.npy", "w", force_zip64=True) as fh:
                    np.save(fh, big_out[k])
                continue
            if k == "bicoherence":
                arr = bico_fixed
            else:
                arr = np.asarray(src[k])
            with zf.open(f"{k}.npy", "w", force_zip64=True) as fh:
                np.save(fh, arr)
            if k == "bicoherence":
                with zf.open("phase_closure_abs_cos.npy", "w", force_zip64=True) as fh:
                    np.save(fh, phase_closure_cos)
    os.replace(tmp_out, OUT)
    size = os.path.getsize(OUT)
    log(f"[+] wrote {OUT} ({size/1e9:.2f} GB)")
    report["output_bytes"] = size
    report["elapsed_seconds"] = round(time.time() - t0, 1)

    # ---------------- pass 4: verification ---------------------------------
    log("\n--- VERIFYING DE-COLLAPSED COVARIANT SCORES ACROSS 591,171 CELLS ---")
    new = np.load(OUT, mmap_mode="r")
    T = new["trigger_vec_dense"] if "trigger_vec_dense" in new.files else new["trigger_vec"]
    R = new["resolution_vec_dense"] if "resolution_vec_dense" in new.files else new["resolution_vec"]
    q4_ret = new["q4_net_return"]
    held = new["protected_held"]
    d = T.shape[1]
    log(f"[*] T shape={T.shape}  (expected sigma_r ~ {1/np.sqrt(d):.4f})")

    anchors = [100, 500, 1000, 5000, 50000, 250000]
    verify = []
    for a in anchors:
        t1 = time.time()
        sims = T @ T[a]
        sims[max(0, a - 2): min(T.shape[0], a + 3)] = 0.0
        abs_sims = np.abs(sims)
        top5 = np.sort(abs_sims)[::-1][:5]
        rec = {
            "anchor": a,
            "top5_abs_r": [round(float(x), 4) for x in top5],
            "peers_ge_070": int((abs_sims >= 0.70).sum()),
            "peers_ge_080": int((abs_sims >= 0.80).sum()),
            "peers_ge_085": int((abs_sims >= 0.85).sum()),
            "mean_r": round(float(sims.mean()), 6),
            "p999_abs_r": round(float(np.percentile(abs_sims, 99.9)), 4),
            "seconds": round(time.time() - t1, 2),
        }
        verify.append(rec)
        log(f"Anchor {a:6d}: top5 |r|={rec['top5_abs_r']} | peers>=0.70: {rec['peers_ge_070']:5d} "
            f"| >=0.80: {rec['peers_ge_080']:4d} | >=0.85: {rec['peers_ge_085']:3d} "
            f"| mean r={rec['mean_r']:+.5f} | p99.9={rec['p999_abs_r']:.3f}")
    report["anchor_verification"] = verify

    # population scan: how many rows would pass min_peers / pathology check
    rng = np.random.default_rng(7)
    sample = np.sort(rng.choice(T.shape[0], size=300, replace=False))
    counts85, counts70, means = [], [], []
    for a in sample:
        s = T @ T[int(a)]
        s[max(0, int(a) - 2): min(T.shape[0], int(a) + 3)] = 0.0
        aa = np.abs(s)
        counts85.append(int((aa >= 0.85).sum()))
        counts70.append(int((aa >= 0.70).sum()))
        means.append(float(s.mean()))
    counts85 = np.array(counts85)
    counts70 = np.array(counts70)
    report["population_sample"] = {
        "rows_sampled": int(len(sample)),
        "mean_r_of_means": round(float(np.mean(means)), 6),
        "median_peers_ge_085": int(np.median(counts85)),
        "p90_peers_ge_085": int(np.percentile(counts85, 90)),
        "p99_peers_ge_085": int(np.percentile(counts85, 99)),
        "median_peers_ge_070": int(np.median(counts70)),
        "frac_rows_with_ge12_peers_085": round(float((counts85 >= 12).mean()), 4),
        "frac_rows_with_ge12_peers_070": round(float((counts70 >= 12).mean()), 4),
    }
    log("\n[*] Population sample of 300 random anchors:")
    for k, v in report["population_sample"].items():
        log(f"      {k}: {v}")
    log(f"\n[+] total elapsed {report['elapsed_seconds']}s")
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"[+] report -> {REPORT}")


if __name__ == "__main__":
    main()
