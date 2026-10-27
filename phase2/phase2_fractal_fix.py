"""Recompute the macro/micro fractal agreement in a standardized descriptor space.

The first pass correlated raw 64-d descriptors, whose scale-dependent terms (s_y,
int_y_dt, quad_var ...) carry the absolute price level and forced every pair to ~1.
Here both vectors are z-scored per feature against the macro population first, so the
agreement measures *shape*, not price level.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pqcompat
from phase2_dense_anomaly_bank import compute_quartile_brute_force_features

BASE = r"C:\cluster\phase2"
P1 = r"C:\cluster\phase1\final\jump10_phase1_fourier_functions.npz"
BLOCK = 12


def main():
    vault = pqcompat.read_parquet(os.path.join(BASE, "phase2_strategy_vault.parquet"))
    macro = np.load(os.path.join(BASE, "_macro_desc.npy"))
    print("[*] macro descriptors", macro.shape, flush=True)

    px = np.load(P1, mmap_mode="r")["raw_tick_cells"]
    hub_idx = vault.anchor_cell_id.to_numpy(np.int64) - 1
    micro_paths = np.empty((len(hub_idx), 150), dtype=np.float64)
    for i in range(0, len(hub_idx), 2000):
        micro_paths[i:i + 2000] = np.asarray(px[hub_idx[i:i + 2000]], dtype=np.float64)
    micro = compute_quartile_brute_force_features(micro_paths)
    np.save(os.path.join(BASE, "_micro_desc_hubs.npy"), micro.astype(np.float32))

    mu = macro.mean(0, keepdims=True)
    sd = macro.std(0, keepdims=True) + 1e-9
    Zm = np.clip((macro - mu) / sd, -6, 6)
    Zi = np.clip((micro - mu) / sd, -6, 6)
    blocks = hub_idx // BLOCK
    Mb = Zm[blocks]
    Zi_c = Zi - Zi.mean(1, keepdims=True)
    Mb_c = Mb - Mb.mean(1, keepdims=True)
    agree = np.sum(Zi_c * Mb_c, axis=1) / (
        np.sqrt(np.sum(Zi_c ** 2, axis=1) * np.sum(Mb_c ** 2, axis=1)) + 1e-12)

    # scale-free path shape agreement: z-scored 150-point paths
    mp = micro_paths - micro_paths.mean(1, keepdims=True)
    mp /= (mp.std(1, keepdims=True) + 1e-12)
    macro_paths = np.empty((len(set(blocks.tolist())), 150))
    pos = {b: i for i, b in enumerate(sorted(set(blocks.tolist())))}
    for i, b in enumerate(sorted(pos)):
        s = b * BLOCK
        e = min(len(px), s + BLOCK)
        blk = np.asarray(px[s:e], dtype=np.float64)
        if blk.shape[0] < BLOCK:
            blk = np.vstack([blk, np.repeat(blk[-1:], BLOCK - blk.shape[0], axis=0)])
        macro_paths[i] = blk.reshape(BLOCK, 150).mean(0)
    macro_paths -= macro_paths.mean(1, keepdims=True)
    macro_paths /= (macro_paths.std(1, keepdims=True) + 1e-12)
    mpb = macro_paths[[pos[b] for b in blocks.tolist()]]
    shape = np.sum(mp * mpb, axis=1) / 150.0

    q90 = float(np.quantile(agree, 0.90))
    vault["fractal_macro_agreement"] = np.round(agree, 4)
    vault["fractal_shape_agreement"] = np.round(shape, 4)
    vault["is_fractal_anomaly"] = agree >= q90
    pqcompat.write_parquet(vault, os.path.join(BASE, "phase2_strategy_vault.parquet"))

    rep = {
        "descriptor_agreement": {"min": round(float(agree.min()), 4), "median": round(float(np.median(agree)), 4),
                                 "p75": round(float(np.quantile(agree, .75)), 4), "max": round(float(agree.max()), 4),
                                 "std": round(float(agree.std()), 4), "p90_threshold": round(q90, 4)},
        "path_shape_agreement": {"median": round(float(np.median(shape)), 4),
                                 "p90": round(float(np.quantile(shape, .9)), 4)},
        "is_fractal_anomalies": int((agree >= q90).sum()),
        "agreement_vs_verified_edge": round(float(np.corrcoef(agree, vault.outbound_verified_edge.to_numpy(float))[0, 1]), 4),
        "fractal_anomalies_mean_edge": round(float(vault.loc[agree >= q90, "outbound_verified_edge"].mean()), 4),
        "non_fractal_mean_edge": round(float(vault.loc[agree < q90, "outbound_verified_edge"].mean()), 4),
    }
    print(json.dumps(rep, indent=2), flush=True)
    with open(os.path.join(BASE, "fractal_report.json"), "w") as f:
        json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
