"""Runs on Kali - independent edge-graph audit from the CSV (no parquet engine needed)."""
import os
import json
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/cluster/phase2")
ed = pd.read_csv(os.path.join(BASE, "phase2_covariant_edges.csv"))
db = pd.read_csv(os.path.join(BASE, "phase2_covariant_db.csv"))

es = ed.src_idx.to_numpy(np.int64)
ds = ed.dst_idx.to_numpy(np.int64)
lo = np.minimum(es, ds)
hi = np.maximum(es, ds)
pairs = np.stack([lo, hi], 1)
uniq = np.unique(pairs, axis=0)

deg = np.bincount(np.concatenate([lo, hi]))
res = {
    "node": "kali",
    "edge_rows": int(len(ed)),
    "unique_undirected_edges": int(len(uniq)),
    "self_loops": int((es == ds).sum()),
    "duplicate_pairs": int(len(ed) - len(uniq)),
    "nodes_touched": int((deg > 0).sum()),
    "max_degree": int(deg.max()),
    "median_degree_nonzero": float(np.median(deg[deg > 0])),
    "cov_min": round(float(ed.cov_0_50.abs().min()), 4),
    "cov_median": round(float(ed.cov_0_50.abs().median()), 4),
    "cov_max": round(float(ed.cov_0_50.abs().max()), 4),
    "lock_median_abs": round(float(ed.lock_50_100.abs().median()), 4),
    "lock_ge_065": int((ed.lock_50_100.abs() >= 0.65).sum()),
    "db_rows": int(len(db)),
    "db_hubs": int(db.anchor_cell_id.nunique()),
    "hubs_appearing_in_edges": int(np.isin(db.anchor_cell_id.to_numpy(np.int64) - 1, np.unique(np.concatenate([lo, hi]))).sum()),
    "peers_mean": round(float(db.covariant_peer_count.mean()), 3),
    "peers_median": float(db.covariant_peer_count.median()),
    "mean_sim_median": round(float(db.mean_0_50_similarity.median()), 4),
    "cells_above_085_mean_sim": int((db.mean_0_50_similarity >= 0.85).sum()),
}
pad = lambda s, n=40: s + " " * max(0, n - len(s))
for k, val in res.items():
    print(pad(k) + ": " + str(val))
with open(os.path.join(BASE, "node_verify_kali.json"), "w") as f:
    json.dump(res, f, indent=2)
print("[+] kali audit written")
