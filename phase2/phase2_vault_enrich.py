"""Phase 2 vault enrichment: items 08 (macro fractal), 09 (RSI/VWAP), 10 (null test).

Inputs : jump10_phase2_scan.npz, jump10_phase1_fourier_functions.npz (raw_tick_cells),
         phase2_covariant_db.parquet, phase2_strategy_vault.parquet
Outputs: vault_triggers.npz, indicator_features.npz, phase2_strategy_vault.parquet (enriched),
         phase2_anomaly_strategies.xlsx, phase2_enrichment_report.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pqcompat
from phase2_dense_anomaly_bank import compute_quartile_brute_force_features

BASE = r"C:\cluster\phase2"
P1 = r"C:\cluster\phase1\final\jump10_phase1_fourier_functions.npz"
N = 591171
N_IN = 502495
BLOCK = 12
CHUNK = 40000


def read_table(p):
    return pqcompat.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)


def recompute_triggers(thr=0.85, min_hits=10, top_hubs=5000, batch=256):
    scan = np.load(os.path.join(BASE, "jump10_phase2_scan.npz"), mmap_mode="r")
    T = scan["trigger_vec_dense"]
    db = read_table(os.path.join(BASE, "phase2_covariant_db.parquet"))
    db = db[db.anchor_cell_id <= N_IN].sort_values("anomaly_z_score", ascending=False).reset_index(drop=True)
    hubs = db.head(top_hubs)
    vault = read_table(os.path.join(BASE, "phase2_strategy_vault.parquet"))
    pos = {int(a): i for i, a in enumerate(hubs.anchor_cell_id.to_numpy())}
    want = [pos[int(a)] for a in vault.anchor_cell_id]
    if any(int(a) not in pos for a in vault.anchor_cell_id):
        raise RuntimeError("vault anchor not in top_hubs set")
    Th = np.ascontiguousarray(T[[int(a) - 1 for a in hubs.anchor_cell_id.to_numpy()]])

    out_idx, out_pol, out_off = [], [], [0]
    for b0 in range(0, len(want), batch):
        sel = want[b0:b0 + batch]
        hits = [[] for _ in sel]
        for s in range(N_IN, N, 20000):
            e = min(N, s + 20000)
            sims = T[s:e] @ Th[sel].T
            big = np.abs(sims) >= thr
            if big.any():
                ri, ci = np.nonzero(big)
                for r, c in zip(ri, ci):
                    hits[c].append((s + r, sims[r, c]))
        for lst in hits:
            idx = np.fromiter((x[0] for x in lst), dtype=np.int64, count=len(lst))
            pol = np.fromiter((np.sign(x[1]) for x in lst), dtype=np.float32, count=len(lst))
            out_idx.append(idx)
            out_pol.append(pol)
            out_off.append(out_off[-1] + len(idx))
        print(f"    triggers {b0 + len(sel)}/{len(want)}", flush=True)
    np.savez(os.path.join(BASE, "vault_triggers.npz"),
             idx=np.concatenate(out_idx) if out_idx else np.zeros(0, np.int64),
             pol=np.concatenate(out_pol) if out_pol else np.zeros(0, np.float32),
             off=np.asarray(out_off, np.int64),
             hubs=vault.anchor_cell_id.to_numpy(np.int64))
    return vault, out_idx, out_pol


def indicator_block():
    cache = os.path.join(BASE, "indicator_features.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    z1 = np.load(P1, mmap_mode="r")
    px = z1["raw_tick_cells"]
    out = {}
    names = ["rsi_end1", "rsi_end2", "rsi_max1", "rsi_max2", "rsi_min1", "rsi_min2",
             "px_high1", "px_high2", "px_low1", "px_low2", "vwap_z1", "vwap_z2", "vwap_stretch1"]
    for n in names:
        out[n] = np.lib.format.open_memmap(os.path.join(BASE, f"_ind_{n}.npy"), mode="w+",
                                           dtype=np.float32, shape=(N,))
    W = 14
    for s in range(0, N, CHUNK):
        e = min(N, s + CHUNK)
        p = np.asarray(px[s:e], dtype=np.float64)
        d = np.diff(p, axis=1)
        up = np.clip(d, 0.0, None)
        dn = np.clip(-d, 0.0, None)
        cu = np.cumsum(up, axis=1)
        cd = np.cumsum(dn, axis=1)
        cs_u = np.concatenate([np.zeros((e - s, 1)), cu], axis=1)
        cs_d = np.concatenate([np.zeros((e - s, 1)), cd], axis=1)
        su = cs_u[:, W:] - cs_u[:, :-W]
        sd = cs_d[:, W:] - cs_d[:, :-W]
        rsi = 100.0 * su / (su + sd + 1e-12)
        full = np.full((e - s, p.shape[1]), 50.0)
        full[:, W:] = rsi
        full[:, 1:W] = rsi[:, :1]
        h1, h2 = full[:, :75], full[:, 75:]
        out["rsi_end1"][s:e] = full[:, 74]
        out["rsi_end2"][s:e] = full[:, -1]
        out["rsi_max1"][s:e] = h1.max(1)
        out["rsi_max2"][s:e] = h2.max(1)
        out["rsi_min1"][s:e] = h1.min(1)
        out["rsi_min2"][s:e] = h2.min(1)
        out["px_high1"][s:e] = p[:, :75].max(1)
        out["px_high2"][s:e] = p[:, 75:].max(1)
        out["px_low1"][s:e] = p[:, :75].min(1)
        out["px_low2"][s:e] = p[:, 75:].min(1)
        csum = np.cumsum(p, axis=1)
        cs2 = np.cumsum(p * p, axis=1)
        cs_pad = np.concatenate([np.zeros((e - s, 1)), csum], axis=1)
        cs2_pad = np.concatenate([np.zeros((e - s, 1)), cs2], axis=1)
        VW = 55
        m = (cs_pad[:, VW:] - cs_pad[:, :-VW]) / VW
        v = (cs2_pad[:, VW:] - cs2_pad[:, :-VW]) / VW - m * m
        sdv = np.sqrt(np.maximum(v, 1e-12))
        dev = p[:, VW - 1:] - m
        zsc = dev / sdv
        ncols = zsc.shape[1]
        fullz = np.zeros((e - s, p.shape[1]))
        fullz[:, VW - 1:] = zsc
        fullz[:, :VW - 1] = zsc[:, :1]
        out["vwap_z1"][s:e] = fullz[:, 74]
        out["vwap_z2"][s:e] = fullz[:, -1]
        out["vwap_stretch1"][s:e] = np.abs(fullz[:, :75]).max(1)
        print(f"    indicators {e:,}/{N:,} ({ncols} cols)", flush=True)
    for n in names:
        out[n].flush()
    np.savez(os.path.join(BASE, "indicator_features.npz"), **{n: np.asarray(out[n]) for n in names})
    return {n: np.asarray(out[n]) for n in names}


def macro_fractal(vault_anchor):
    z1 = np.load(P1, mmap_mode="r")
    px = z1["raw_tick_cells"]
    n_blocks = int(np.ceil(N / BLOCK))
    cached = os.path.join(BASE, "_macro_desc.npy")
    if os.path.exists(cached):
        macro_desc = np.load(cached)
    else:
      macro_desc = np.lib.format.open_memmap(cached, mode="w+", dtype=np.float32, shape=(n_blocks, 64))
      for b0 in range(0, n_blocks, 4000):
        b1 = min(n_blocks, b0 + 4000)
        rows = np.empty((b1 - b0, 150), dtype=np.float64)
        for i, bi in enumerate(range(b0, b1)):
            s = bi * BLOCK
            e = min(N, s + BLOCK)
            blk = np.asarray(px[s:e], dtype=np.float64)
            if blk.shape[0] < BLOCK:
                blk = np.vstack([blk, np.repeat(blk[-1:], BLOCK - blk.shape[0], axis=0)])
            rows[i] = blk.reshape(BLOCK, 150).mean(axis=0)
        macro_desc[b0:b1] = compute_quartile_brute_force_features(rows)
        print(f"    macro desc {b1:,}/{n_blocks:,}", flush=True)
    macro_desc = np.asarray(macro_desc)

    hub_idx = vault_anchor.astype(np.int64) - 1
    micro = np.empty((len(hub_idx), 150), dtype=np.float64)
    for i in range(0, len(hub_idx), 2000):
        sl = hub_idx[i:i + 2000]
        micro[i:i + 2000] = np.asarray(px[sl], dtype=np.float64)
    micro_desc = compute_quartile_brute_force_features(micro)
    blocks = hub_idx // BLOCK

    def cos64(a, b):
        an = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)
        bn = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-9)
        return np.sum(an * bn, axis=1) / a.shape[1]

    agree = cos64(micro_desc, macro_desc[blocks])
    return agree, micro_desc, macro_desc


def main():
    vault, trig_idx, trig_pol = recompute_triggers()
    print(f"[*] vault rows={len(vault)} triggers={len(trig_idx)}", flush=True)
    scan = np.load(os.path.join(BASE, "jump10_phase2_scan.npz"), mmap_mode="r")
    held = np.asarray(scan["protected_held"])
    ret = np.asarray(scan["q4_net_return"])

    print("[*] item 08 macro fractal ...", flush=True)
    agree, _, _ = macro_fractal(vault.anchor_cell_id.to_numpy(np.int64))
    is_frac = agree >= float(np.quantile(agree, 0.90))
    print(f"    agreement med={np.median(agree):.4f} p90={np.quantile(agree,0.9):.4f} "
          f"frac_anomalies={int(is_frac.sum())}", flush=True)

    print("[*] item 09 indicator block ...", flush=True)
    ind = indicator_block()

    print("[*] item 10 block-permutation null ...", flush=True)
    rng = np.random.default_rng(20261025)
    nb = N - N_IN
    n_perm = 200
    n_full = nb // 150
    idx_base = (np.arange(n_full)[:, None] * 150 + np.arange(150)[None, :]).ravel()
    tail = np.arange(n_full * 150, nb, dtype=np.int64)
    perms = np.empty((n_perm, len(trig_idx)), dtype=np.float32)
    perm_held = np.empty((n_perm, len(trig_idx)), dtype=np.float32)
    held_o = held[N_IN:]
    ret_o = ret[N_IN:]
    li_all = [idx - N_IN for idx in trig_idx]
    for p in range(n_perm):
        pm = rng.permutation(n_full)
        newpos = np.concatenate([(pm[:, None] * 150 + np.arange(150)[None, :]).ravel(), tail])
        ph = held_o[newpos]
        pr = ret_o[newpos]
        for i, li in enumerate(li_all):
            perms[p, i] = float(np.mean(np.sign(pr[li] * trig_pol[i]) == 1))
            perm_held[p, i] = float(np.mean(ph[li]))
        if (p + 1) % 50 == 0:
            print(f"    perm {p + 1}/{n_perm}", flush=True)

    obs_dir, obs_prot, pvals = [], [], []
    for i, idx in enumerate(trig_idx):
        li = idx - N_IN
        od = float(np.mean(np.sign(ret[li] * trig_pol[i]) == 1))
        op = float(np.mean(held[li]))
        obs_dir.append(od)
        obs_prot.append(op)
        null = np.maximum(perms[:, i], perm_held[:, i])
        pvals.append(float((np.sum(null >= max(od, op)) + 1) / (n_perm + 1)))

    vault = vault.copy()
    vault["fractal_macro_agreement"] = np.round(agree, 4)
    vault["is_fractal_anomaly"] = is_frac
    vault["rsi_divergence_confluence"] = np.round(
        np.where((ind["px_high2"][vault.anchor_cell_id - 1] > ind["px_high1"][vault.anchor_cell_id - 1])
                 & (ind["rsi_max2"][vault.anchor_cell_id - 1] < ind["rsi_max1"][vault.anchor_cell_id - 1]), 1.0,
                 np.where((ind["px_low2"][vault.anchor_cell_id - 1] < ind["px_low1"][vault.anchor_cell_id - 1])
                          & (ind["rsi_min2"][vault.anchor_cell_id - 1] > ind["rsi_min1"][vault.anchor_cell_id - 1]), -1.0, 0.0)), 2)
    vault["vwap_band_anomaly_score"] = np.round(ind["vwap_stretch1"][vault.anchor_cell_id - 1], 3)
    vault["rsi_end1"] = np.round(ind["rsi_end1"][vault.anchor_cell_id - 1], 2)
    vault["vwap_z1"] = np.round(ind["vwap_z1"][vault.anchor_cell_id - 1], 3)
    vault["null_pval"] = np.round(pvals, 4)
    vault["null_p99"] = np.round(np.quantile(np.maximum(perms, perm_held), 0.99, axis=0), 4)
    vault["beats_null"] = vault["null_pval"] < 0.01
    vault["outbound_protected_win_rate"] = np.round(obs_prot, 4)
    vault["outbound_directional_win_rate"] = np.round(obs_dir, 4)
    vault["outbound_verified_edge"] = np.round(np.maximum(obs_prot, obs_dir), 4)

    pqcompat.write_parquet(vault, os.path.join(BASE, "phase2_strategy_vault.parquet"))
    try:
        with pd.ExcelWriter(os.path.join(BASE, "phase2_anomaly_strategies.xlsx"), engine="openpyxl") as w:
            vault.to_excel(w, sheet_name="Phase3_Ready_Strategies", index=False)
            vault[vault["beats_null"]].to_excel(w, sheet_name="Null_Verified", index=False)
    except Exception as ex:
        print(f"[-] xlsx: {type(ex).__name__} {ex}", flush=True)

    top300 = vault.nlargest(min(300, len(vault)), "outbound_win_lb95")
    rep = {
        "vault_rows": int(len(vault)),
        "fractal": {"median_agreement": round(float(np.median(agree)), 4),
                    "p90_agreement": round(float(np.quantile(agree, 0.9)), 4),
                    "is_fractal_anomalies": int(is_frac.sum())},
        "indicators": {
            "rsi_divergence_bearish": int((vault["rsi_divergence_confluence"] == 1).sum()),
            "rsi_divergence_bullish": int((vault["rsi_divergence_confluence"] == -1).sum()),
            "vwap_stretch_median": round(float(vault["vwap_band_anomaly_score"].median()), 3),
        },
        "null_test": {
            "permutations": n_perm, "block_size_ticks": 150,
            "top300_observed_mean_win": round(float(top300["outbound_verified_edge"].mean()), 4),
            "top300_null_p99_mean": round(float(top300["null_p99"].mean()), 4),
            "top300_beats_null": int(top300["beats_null"].sum()),
            "vault_beats_null": int(vault["beats_null"].sum()),
            "median_pval": round(float(np.median(pvals)), 4),
        },
    }
    with open(os.path.join(BASE, "phase2_enrichment_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
