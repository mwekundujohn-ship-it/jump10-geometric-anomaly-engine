"""Selection-aware block-permutation null test for the Phase-2 vault (item 10).

The naive per-strategy p-value is meaningless here because the 893 strategies were
*selected* on the outbound window. This script therefore re-selects the top-300 in
every permutation and compares the selected-set mean win rate, which is the only
honest test of "the vault beats a block-shuffled null".

Outputs: null_test.json, phase2_strategy_vault.parquet (null columns updated)
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pqcompat

BASE = r"C:\cluster\phase2"
N, N_IN = 591171, 502495
NPERM = 400
TOP = 300


def wilson(p, n, z=1.959964):
    p = np.clip(p, 0, 1)
    d = 1 + z ** 2 / n
    c = p + z ** 2 / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (c - r) / d


def main():
    tg = np.load(os.path.join(BASE, "vault_triggers.npz"))
    idx_all, pol_all, off = tg["idx"], tg["pol"], tg["off"]
    scan = np.load(os.path.join(BASE, "jump10_phase2_scan.npz"), mmap_mode="r")
    held_o = np.asarray(scan["protected_held"])[N_IN:].astype(np.float32)
    ret_o = np.asarray(scan["q4_net_return"])[N_IN:].astype(np.float32)
    nb = len(held_o)
    n_full = nb // 150
    tail = np.arange(n_full * 150, nb, dtype=np.int64)
    base = np.arange(n_full)[:, None] * 150 + np.arange(150)[None, :]
    lis = [idx_all[off[i]:off[i + 1]] - N_IN for i in range(len(off) - 1)]
    pols = [pol_all[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    ns = np.array([len(x) for x in lis], dtype=float)
    print(f"[*] {len(lis)} strategies, triggers {int(ns.sum()):,}", flush=True)

    def win_arrays(h, r):
        w = np.empty(len(lis), dtype=np.float64)
        for i, (li, po) in enumerate(zip(lis, pols)):
            w[i] = max(float(np.mean(h[li])), float(np.mean(np.sign(r[li] * po) == 1)))
        return w

    obs = win_arrays(held_o, ret_o)
    obs_lb = wilson(obs, ns)
    obs_sel = np.argsort(-obs_lb)[:TOP]
    obs_mean = float(obs[obs_sel].mean())
    obs_pool_n = int(ns[obs_sel].sum())

    rng = np.random.default_rng(20261025)
    null_mean = np.empty(NPERM)
    null_pool = np.empty(NPERM)
    for p in range(NPERM):
        pm = rng.permutation(n_full)
        pos = np.concatenate([(pm[:, None] * 150 + np.arange(150)[None, :]).ravel(), tail])
        w = win_arrays(held_o[pos], ret_o[pos])
        sel = np.argsort(-wilson(w, ns))[:TOP]
        null_mean[p] = w[sel].mean()
        null_pool[p] = np.sum(w[sel] * ns[sel]) / ns[sel].sum() if ns[sel].sum() else 0
        if (p + 1) % 100 == 0:
            print(f"    perm {p + 1}/{NPERM} | null top300 mean {null_mean[p]:.4f}", flush=True)

    p_agg = float((np.sum(null_mean >= obs_mean) + 1) / (NPERM + 1))
    obs_pool = float(np.sum(obs[obs_sel] * ns[obs_sel]) / obs_pool_n)
    p_pool = float((np.sum(null_pool >= obs_pool) + 1) / (NPERM + 1))
    per_p = np.array([float((np.sum(np.abs(null_mean - null_mean.mean()) >= abs(obs[i] - null_mean.mean())) + 1) / (NPERM + 1))
                      for i in range(len(lis))])

    out = {
        "permutations": NPERM, "block_size_ticks": 150, "top": TOP,
        "observed_top300_mean_win": round(obs_mean, 4),
        "null_top300_mean_median": round(float(np.median(null_mean)), 4),
        "null_top300_mean_p99": round(float(np.quantile(null_mean, 0.99)), 4),
        "p_selection_aware": round(p_agg, 4),
        "observed_pooled_win": round(obs_pool, 4),
        "null_pooled_median": round(float(np.median(null_pool)), 4),
        "p_pooled": round(p_pool, 4),
        "beats_null_p01": bool(p_agg < 0.01),
        "verdict": ("vault beats the block-permutation null" if p_agg < 0.01
                    else "OUTBOUND EDGE IS NOT DISTINGUISHABLE FROM A BLOCK-SHUFFLED NULL "
                         "(selection on the outbound window is the likely cause)"),
    }
    print(json.dumps(out, indent=2), flush=True)
    with open(os.path.join(BASE, "null_test.json"), "w") as f:
        json.dump(out, f, indent=2)

    v = pqcompat.read_parquet(os.path.join(BASE, "phase2_strategy_vault.parquet"))
    v["null_pval"] = np.round(per_p, 4)
    v["null_pval_selection_aware"] = round(p_agg, 4)
    v["beats_null"] = bool(p_agg < 0.01)
    v["null_verdict"] = out["verdict"]
    pqcompat.write_parquet(v, os.path.join(BASE, "phase2_strategy_vault.parquet"))


if __name__ == "__main__":
    main()
