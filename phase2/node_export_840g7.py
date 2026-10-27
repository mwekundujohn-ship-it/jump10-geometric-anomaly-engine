"""Runs on 840g7 - independent re-verification of the Phase 2 relational DB / vault
and regeneration of the strategy workbook (offloads openpyxl work from XFlip)."""
import os
import sys
import json
import numpy as np

BASE = r"C:\cluster\phase2"
sys.path.insert(0, BASE)
import pqcompat
import pandas as pd

db = pqcompat.read_parquet(os.path.join(BASE, "phase2_covariant_db.parquet"))
ed = pqcompat.read_parquet(os.path.join(BASE, "phase2_covariant_edges.parquet"))
v = pqcompat.read_parquet(os.path.join(BASE, "phase2_strategy_vault.parquet"))

n_und = len({(min(a, b), max(a, b)) for a, b in zip(ed.src_idx, ed.dst_idx)})
sym = len({(a, b) for a, b in zip(ed.src_idx, ed.dst_idx)} & {(b, a) for a, b in zip(ed.src_idx, ed.dst_idx)})
loops = int((ed.src_idx == ed.dst_idx).sum())
hubs = int(db.anchor_cell_id.nunique())
dup_cols = int(ed.duplicated().sum())
res = {
    "node": "840g7",
    "db_rows": int(len(db)),
    "db_unique_anchors": hubs,
    "edge_rows": int(len(ed)),
    "edge_unique_undirected": int(n_und),
    "edge_rows_that_are_mirrored": int(sym),
    "edge_self_loops": loops,
    "edge_exact_duplicate_rows": dup_cols,
    "vault_rows": int(len(v)),
    "vault_ok_rows": int((v.sample_flag == "OK").sum()),
    "vault_thin_rows": int((v.sample_flag == "THIN").sum()),
    "vault_min_edge": round(float(v.outbound_verified_edge.min()), 4),
    "vault_max_edge": round(float(v.outbound_verified_edge.max()), 4),
    "vault_rows_above_55pct": int((v.outbound_verified_edge >= 0.55).sum()),
    "beats_null_rows": int(v.beats_null.astype(bool).sum()),
    "czet_spearman": None,
}
ce = os.path.join(BASE, "czet_eval.json")
if os.path.exists(ce):
    res["czet_spearman"] = json.load(open(ce))["spearman"]

pad = lambda s, n=44: s + " " * max(0, n - len(s))
for k, val in res.items():
    print(pad(k) + ": " + str(val))

with pd.ExcelWriter(os.path.join(BASE, "phase2_anomaly_strategies.xlsx"), engine="openpyxl") as w:
    v.to_excel(w, sheet_name="Phase3_Ready_Strategies", index=False)
    v[v.beats_null.astype(bool)].to_excel(w, sheet_name="Null_Verified", index=False)
    v[(v.sample_flag == "OK") & (v.outbound_verified_edge >= 0.55)].sort_values(
        "outbound_win_lb95", ascending=False).to_excel(w, sheet_name="Ranked_by_Wilson_LB", index=False)
    db.sort_values("anomaly_z_score", ascending=False).head(2000).to_excel(w, sheet_name="Inbound_Covariant_Hubs", index=False)
print("[+] strategies workbook written")

with open(os.path.join(BASE, "node_verify_840g7.json"), "w") as f:
    json.dump(res, f, indent=2)
