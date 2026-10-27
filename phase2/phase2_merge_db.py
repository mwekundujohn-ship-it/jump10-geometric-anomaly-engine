"""Merge the 3 node shards into the Phase-2 Relational Covariant Database.

Runs on the X Flip (cheap: only the hub/edge tables move, never the matrices).
Produces:
  C:\\cluster\\phase2\\phase2_covariant_db.parquet
  C:\\cluster\\phase2\\phase2_covariant_edges.parquet
  C:\\cluster\\phase2\\phase2_covariant_summary.json
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\cluster\phase2")
import pqcompat

BASE = r"C:\cluster\phase2"
SHARDS = os.path.join(BASE, "shards")
RECV = r"C:\cluster\phase1\out\phase2\shards_received"
N_TOTAL = 591171

COLS = ["anchor_cell_id", "covariant_peer_count", "peers_evaluated",
        "pos_corr_peers", "neg_mirror_peers", "mean_0_50_similarity",
        "max_0_50_similarity", "mean_50_100_lock_corr", "frac_75_100_exact_lock",
        "protected_ext_win_rate", "directional_50_100_win_rate",
        "direction_consistency", "anomaly_z_score", "bicoherence_phase_lock",
        "target_ext_tick_median", "recommended_action"]


def load_any(path):
    if path.endswith(".parquet"):
        return pqcompat.read_parquet(path)
    df = pd.read_csv(path)
    if "recommended_action" not in df.columns:
        return df
    return df


def main():
    node_files = {}
    for ext in ("parquet", "csv"):
        for pat in (os.path.join(SHARDS, f"covariant_anomalies_node?.{ext}"),
                    os.path.join(RECV, f"covariant_anomalies_node?.{ext}")):
            for f in sorted(glob.glob(pat)):
                key = os.path.basename(f).split("_node")[1].split(".")[0]
                node_files.setdefault(key, f)
    if not node_files:
        print("[-] no shard files found", flush=True)
        sys.exit(1)
    print("[*] hub shards:", node_files, flush=True)

    parts = []
    for k, f in sorted(node_files.items()):
        df = load_any(f)
        df["shard"] = int(k)
        parts.append(df)
        print(f"    node {k}: {len(df):,} anomaly hubs  <- {f}", flush=True)
    db = pd.concat(parts, ignore_index=True)
    before = len(db)
    db = db.sort_values("anomaly_z_score", ascending=False)
    db = db.drop_duplicates(subset="anchor_cell_id", keep="first").reset_index(drop=True)
    print(f"[*] merged {before:,} -> {len(db):,} unique anomaly hubs", flush=True)

    for c in COLS:
        if c not in db.columns:
            db[c] = np.nan
    db = db[COLS + ["shard"]]

    all_anchors = set(db["anchor_cell_id"].astype(np.int64))
    missing = N_TOTAL - len(all_anchors)
    out_db = os.path.join(BASE, "phase2_covariant_db.parquet")
    pqcompat.write_parquet(db, out_db)
    print(f"[+] {out_db}  ({os.path.getsize(out_db)/1e6:.2f} MB)", flush=True)

    # ---- edges ------------------------------------------------------------
    eparts = []
    for k in sorted(node_files):
        for ext in ("parquet", "csv"):
            for pat in (os.path.join(SHARDS, f"covariant_edges_node{k}.{ext}"),
                        os.path.join(RECV, f"covariant_edges_node{k}.{ext}")):
                for f in glob.glob(pat):
                    e = load_any(f)
                    if len(e):
                        eparts.append(e)
    if eparts:
        edges = pd.concat(eparts, ignore_index=True)
        edges["abs_cov"] = edges["cov_0_50"].abs()
        edges = edges.sort_values("abs_cov", ascending=False)
        s = edges["src_idx"].to_numpy(np.int64)
        d = edges["dst_idx"].to_numpy(np.int64)
        edges["pair"] = np.minimum(s, d) * np.int64(N_TOTAL + 1) + np.maximum(s, d)
        edges = edges.drop_duplicates(subset="pair", keep="first").drop(columns=["pair", "abs_cov"])
        out_e = os.path.join(BASE, "phase2_covariant_edges.parquet")
        pqcompat.write_parquet(edges, out_e)
        n_edges = len(edges)
        print(f"[+] {out_e}  ({n_edges:,} unique undirected edges, {os.path.getsize(out_e)/1e6:.2f} MB)", flush=True)
    else:
        n_edges = 0
        edges = pd.DataFrame(columns=["src_idx", "dst_idx", "cov_0_50", "lock_50_100"])

    # ---- summary ----------------------------------------------------------
    top = db.head(25)
    summary = {
        "cells_total": N_TOTAL,
        "anomaly_hubs": int(len(db)),
        "hub_rate_pct": round(100.0 * len(db) / N_TOTAL, 3),
        "cells_not_anomalous": int(missing),
        "unique_undirected_edges": int(n_edges),
        "shards": {k: int(len(p)) for k, p in zip(sorted(node_files), parts)},
        "covariant_similarity": {
            "max_single_peer_abs_r": round(float(db["max_0_50_similarity"].max()), 4),
            "max_single_peer_abs_r_anchor": int(db.loc[db["max_0_50_similarity"].idxmax(), "anchor_cell_id"]),
            "median_mean_abs_r": round(float(db["mean_0_50_similarity"].median()), 4),
            "p99_mean_abs_r": round(float(np.percentile(db["mean_0_50_similarity"], 99)), 4),
        },
        "continuation_lock": {
            "max_mean_lock_corr": round(float(db["mean_50_100_lock_corr"].max()), 4),
            "max_mean_lock_corr_anchor": int(db.loc[db["mean_50_100_lock_corr"].idxmax(), "anchor_cell_id"]),
            "max_frac_exact_lock": round(float(db["frac_75_100_exact_lock"].max()), 4),
            "median_mean_lock_corr": round(float(db["mean_50_100_lock_corr"].median()), 4),
        },
        "edge_quality": {
            "max_protected_ext_win_rate": round(float(db["protected_ext_win_rate"].max()), 4),
            "median_protected_ext_win_rate": round(float(db["protected_ext_win_rate"].median()), 4),
            "max_directional_win_rate": round(float(db["directional_50_100_win_rate"].max()), 4),
            "median_directional_win_rate": round(float(db["directional_50_100_win_rate"].median()), 4),
            "max_anomaly_z_score": round(float(db["anomaly_z_score"].max()), 3),
            "hubs_z_ge_3": int((db["anomaly_z_score"] >= 3).sum()),
            "hubs_protected_win_ge_60pct": int((db["protected_ext_win_rate"] >= 0.60).sum()),
            "hubs_directional_win_ge_60pct": int((db["directional_50_100_win_rate"] >= 0.60).sum()),
        },
        "peer_counts": {
            "median": int(db["covariant_peer_count"].median()),
            "p90": int(np.percentile(db["covariant_peer_count"], 90)),
            "max": int(db["covariant_peer_count"].max()),
            "median_evaluated": int(db["peers_evaluated"].median()),
        },
        "actions": db["recommended_action"].value_counts().to_dict(),
        "mean_ext_tick": round(float(db["target_ext_tick_median"].mean()), 1),
        "top_10_hubs": top.head(10)[["anchor_cell_id", "covariant_peer_count",
                                    "mean_0_50_similarity", "max_0_50_similarity",
                                    "mean_50_100_lock_corr", "protected_ext_win_rate",
                                    "directional_50_100_win_rate", "anomaly_z_score",
                                    "recommended_action"]].to_dict(orient="records"),
    }
    with open(os.path.join(BASE, "phase2_covariant_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
