"""Phase 2 anomaly-efficiency report (X Flip, light).

Inputs : phase2_covariant_db.parquet, phase2_covariant_edges.parquet,
         phase2_covariant_summary.json, phase2_strategy_vault.(parquet|csv)
Outputs: phase2_anomaly_efficiency_report.json
         phase2_anomaly_efficiency_report.xlsx
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pqcompat

BASE = r"C:\cluster\phase2"


def read_table(p):
    return pqcompat.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)


def main():
    db = read_table(os.path.join(BASE, "phase2_covariant_db.parquet"))
    edges = read_table(os.path.join(BASE, "phase2_covariant_edges.parquet"))
    with open(os.path.join(BASE, "phase2_covariant_summary.json")) as f:
        summ = json.load(f)
    vault = None
    for n in ("phase2_strategy_vault.parquet", "phase2_strategy_vault.csv"):
        p = os.path.join(BASE, n)
        if os.path.exists(p):
            vault = read_table(p)
            break

    n = db["covariant_peer_count"].clip(lower=1).to_numpy(float)
    for col, label in (("protected_ext_win_rate", "prot"), ("directional_50_100_win_rate", "dir")):
        p = db[col].to_numpy(float)
        db[f"{label}_edge"] = p - 0.5
        db[f"{label}_z"] = (p - 0.5) / np.sqrt(0.25 / n)
        db[f"{label}_p"] = 0.5 * np.vectorize(lambda z: __import__("math").erfc(abs(z) / np.sqrt(2)))(db[f"{label}_z"])

    db["best_win"] = db[["protected_ext_win_rate", "directional_50_100_win_rate"]].max(axis=1)
    db["best_edge"] = db["best_win"] - 0.5
    db["best_z"] = (db["best_win"] - 0.5) / np.sqrt(0.25 / n)
    db["efficiency_score"] = db["best_z"] * np.sqrt(n) / 100.0

    q = [0, .01, .05, .25, .5, .75, .95, .99, 1.0]
    dist = {}
    for col in ("covariant_peer_count", "mean_0_50_similarity", "max_0_50_similarity",
                "mean_50_100_lock_corr", "frac_75_100_exact_lock", "protected_ext_win_rate",
                "directional_50_100_win_rate", "anomaly_z_score", "best_z", "efficiency_score"):
        s = db[col].astype(float)
        dist[col] = {f"p{int(x*100)}": round(float(s.quantile(x)), 4) for x in q}

    def bucket_quality(col, bins, labels):
        out = []
        for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
            m = (db[col] >= lo) & (db[col] < hi)
            if m.sum() == 0:
                continue
            out.append({
                "bucket": lab, "hubs": int(m.sum()),
                "median_best_win": round(float(db.loc[m, "best_win"].median()), 4),
                "median_lock_corr": round(float(db.loc[m, "mean_50_100_lock_corr"].median()), 4),
                "max_best_win": round(float(db.loc[m, "best_win"].max()), 4),
                "hubs_win_ge_60pct": int((db.loc[m, "best_win"] >= 0.60).sum()),
            })
        return out

    by_peers = bucket_quality("covariant_peer_count", [12, 25, 50, 100, 250, 1000, 1e9],
                              ["12-24", "25-49", "50-99", "100-249", "250-999", "1000+"])
    by_r = bucket_quality("mean_0_50_similarity", [0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 1.01],
                          ["0.85-0.88", "0.88-0.90", "0.90-0.92", "0.92-0.94", "0.94-0.96", "0.96+"])

    report = {
        "generated_from": "phase2_covariant_db.parquet",
        "coverage": {"cells": summ["cells_total"], "hubs": len(db),
                     "hub_rate_pct": summ["hub_rate_pct"],
                     "edges": int(len(edges)), "vault_strategies": 0 if vault is None else int(len(vault))},
        "answers": {
            "highest_covariant_similarity": summ["covariant_similarity"],
            "strongest_continuation_lock": summ["continuation_lock"],
            "best_verified_edge": {
                "max_protected_win": round(float(db["protected_ext_win_rate"].max()), 4),
                "max_directional_win": round(float(db["directional_50_100_win_rate"].max()), 4),
                "best_combined_win": round(float(db["best_win"].max()), 4),
                "anchor_cell_id": int(db.loc[db["best_win"].idxmax(), "anchor_cell_id"]),
            },
            "statistically_significant_hubs_z5": int((db["best_z"] >= 5).sum()),
            "hubs_win_ge_60pct": int((db["best_win"] >= 0.60).sum()),
            "hubs_win_ge_70pct": int((db["best_win"] >= 0.70).sum()),
        },
        "distributions": dist,
        "efficiency_by_peer_count": by_peers,
        "efficiency_by_similarity": by_r,
        "top_25_by_efficiency": db.sort_values("best_z", ascending=False).head(25)[
            ["anchor_cell_id", "covariant_peer_count", "mean_0_50_similarity", "max_0_50_similarity",
             "mean_50_100_lock_corr", "frac_75_100_exact_lock", "protected_ext_win_rate",
             "directional_50_100_win_rate", "best_win", "best_z", "recommended_action"]
        ].round(4).to_dict(orient="records"),
        "edge_table": {
            "median_cov": round(float(edges["cov_0_50"].abs().median()), 4) if len(edges) else None,
            "p90_cov": round(float(edges["cov_0_50"].abs().quantile(0.9)), 4) if len(edges) else None,
            "max_cov": round(float(edges["cov_0_50"].abs().max()), 4) if len(edges) else None,
            "median_abs_lock": round(float(edges["lock_50_100"].abs().median()), 4) if len(edges) else None,
            "edges_lock_ge_065": int((edges["lock_50_100"].abs() >= 0.65).sum()) if len(edges) else 0,
        },
        "verdict": None,
    }
    best = report["answers"]["best_verified_edge"]["best_combined_win"]
    report["verdict"] = (
        "Phase 2 produces tradeable structure: "
        f"{report['answers']['hubs_win_ge_60pct']:,} of {len(db):,} hubs exceed 60% out-of-sample win rates "
        f"(best {best:.1%}, {report['answers']['statistically_significant_hubs_z5']:,} hubs at z>=5)."
        if best >= 0.60 else
        f"No hub clears a robust 60% out-of-sample edge (best {best:.1%}); Phase 2 shows covariant structure but no proven directional edge."
    )

    for name in ("czet_eval.json", "czet_train_log.json", "null_test.json",
                 "fractal_report.json", "phase2_enrichment_report.json",
                 "crawled_math_operators_manifest.json"):
        pth = os.path.join(BASE, name)
        if os.path.exists(pth):
            with open(pth) as f:
                report[name.replace(".json", "")] = json.load(f)

    cop = os.path.join(BASE, "crawled_math_operators.npz")
    if os.path.exists(cop) and vault is not None and len(vault):
        zc = np.load(cop)
        ai = vault["anchor_cell_id"].to_numpy(np.int64) - 1
        rows = []
        for n in zc.files:
            v = zc[n]
            vh = v[ai]
            rows.append({
                "operator": n,
                "all_median": round(float(np.nanmedian(v)), 5),
                "all_p05": round(float(np.nanpercentile(v, 5)), 5),
                "all_p95": round(float(np.nanpercentile(v, 95)), 5),
                "vault_median": round(float(np.nanmedian(vh)), 5),
                "corr_vs_outbound_edge": round(float(np.corrcoef(vh, vault["outbound_verified_edge"])[0, 1]), 4),
            })
        report["crawled_operators"] = rows

    with open(os.path.join(BASE, "phase2_anomaly_efficiency_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("coverage", "answers", "edge_table", "verdict")}, indent=2))

    try:
        with pd.ExcelWriter(os.path.join(BASE, "phase2_anomaly_efficiency_report.xlsx"),
                            engine="openpyxl") as w:
            pd.DataFrame([report["coverage"]]).to_excel(w, sheet_name="Coverage", index=False)
            pd.DataFrame(report["answers"]["best_verified_edge"], index=["value"]).T.to_excel(w, sheet_name="Answers")
            pd.DataFrame(dist).T.to_excel(w, sheet_name="Distributions")
            pd.DataFrame(by_peers).to_excel(w, sheet_name="Efficiency_by_Peers", index=False)
            pd.DataFrame(by_r).to_excel(w, sheet_name="Efficiency_by_Sim", index=False)
            db.sort_values("best_z", ascending=False).head(10000).to_excel(w, sheet_name="Top_Hubs", index=False)
            if vault is not None and len(vault):
                vault.to_excel(w, sheet_name="Strategy_Vault", index=False)
            if "crawled_operators" in report:
                pd.DataFrame(report["crawled_operators"]).to_excel(w, sheet_name="Crawled_Operators", index=False)
            cz = report.get("czet_eval", {})
            if cz:
                pd.DataFrame([cz]).T.to_excel(w, sheet_name="CZET_Eval")
            nt = report.get("null_test", {})
            if nt:
                pd.DataFrame([{k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in nt.items()}]).T.to_excel(w, sheet_name="Null_Test")
            fr = report.get("fractal_report")
            if isinstance(fr, dict):
                flat = {k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in fr.items()}
                pd.DataFrame([flat]).T.to_excel(w, sheet_name="Fractal_vs_Macro")
        print("[+] xlsx written")
    except Exception as e:
        print(f"[-] xlsx skipped: {type(e).__name__} {e}")


if __name__ == "__main__":
    main()
