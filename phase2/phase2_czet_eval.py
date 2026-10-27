"""Out-of-sample evaluation of the trained CZET (runs on the X Flip, CPU-light)."""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_step2_inbound_outbound_ai import CrossZoneEpicycleTransformer, read_table

BASE = r"C:\cluster\phase2"


def main(n_pairs=8000, n_rows=4000, seed=7):
    np.random.seed(seed)
    ck = torch.load(os.path.join(BASE, "czet_phase2.pt"), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = CrossZoneEpicycleTransformer(**cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print("[*] loaded", cfg)

    cache = np.load(os.path.join(BASE, "czet_train_cache.npz"), mmap_mode="r")
    Q = cache["Q_seq"]
    N = Q.shape[0]
    n_in = int(N * 0.85)
    out = {"cfg": cfg, "train_rows": 150000}

    # 1) out-of-sample quartile reconstruction (last 15% never trained)
    idx = np.random.permutation(np.arange(n_in, N))[:n_rows]
    idx.sort()
    q = torch.from_numpy(np.ascontiguousarray(Q[idx]))
    with torch.no_grad():
        p34, p12, z, anom = model(q)
    for tag, pred, tgt in (("q34", p34, q[:, 2:4, :]), ("q12", p12, q[:, 0:2, :])):
        sse = float(((pred - tgt) ** 2).sum())
        sst = float(((tgt - tgt.mean(0, keepdim=True)) ** 2).sum())
        out[f"{tag}_oos_r2"] = round(1 - sse / sst, 5)
        out[f"{tag}_oos_mse"] = round(float(((pred - tgt) ** 2).mean()), 6)

    # 2) covariant metric quality on labelled pairs
    edges = read_table(os.path.join(BASE, "phase2_covariant_edges.parquet"))
    e = edges.sample(n=min(n_pairs, len(edges)), random_state=seed)
    src = e["src_idx"].to_numpy(np.int64)
    dst = e["dst_idx"].to_numpy(np.int64)
    sign = np.sign(e["cov_0_50"].to_numpy(np.float32))
    lock = e["lock_50_100"].to_numpy(np.float32)
    with torch.no_grad():
        _, _, zs, _ = model(torch.from_numpy(np.ascontiguousarray(Q[src])))
        _, _, zd, _ = model(torch.from_numpy(np.ascontiguousarray(Q[dst])))
    cos = (zs * (zd * torch.from_numpy(sign)[:, None])).sum(-1).numpy()
    lockc = np.clip(lock, -1, 1)
    out["pairs"] = int(len(e))
    out["pair_cos_vs_lock_pearson"] = round(float(np.corrcoef(cos, lockc)[0, 1]), 4)
    if len(cos) > 2:
        from scipy.stats import spearmanr
        out["pair_cos_vs_lock_spearman"] = round(float(spearmanr(cos, lockc).statistic), 4)
    pos = lockc >= 0.65
    if pos.sum() and (~pos).sum():
        order = np.argsort(-cos)
        hits = pos[order]
        tp = np.cumsum(hits)
        prec = tp / np.arange(1, len(hits) + 1)
        out["precision_at_10pct"] = round(float(prec[max(0, int(0.1 * len(hits)) - 1)]), 4)
        out["base_rate_lock_ge_065"] = round(float(pos.mean()), 4)
    print(json.dumps(out, indent=2))
    with open(os.path.join(BASE, "czet_eval.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
