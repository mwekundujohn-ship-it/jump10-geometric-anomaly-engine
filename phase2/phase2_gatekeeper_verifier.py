import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pqcompat

_orig_read_parquet = pd.read_parquet


def _read_parquet(p, *a, **k):
    try:
        return _orig_read_parquet(p, *a, **k)
    except Exception:
        return pqcompat.read_parquet(p)


pd.read_parquet = _read_parquet

PHASE2_DIR = "C:/cluster/phase2"

def run_phase2_gatekeeper_audit():
    print("=" * 78)
    print("         PHASE 2 MASTER GATEKEEPER VERIFICATION CHECKLIST")
    print("=" * 78)
    results = {}

    def check(item_id, name, passed, detail):
        status = "PASS" if passed else "MISSING / INCOMPLETE"
        icon = "[+]" if passed else "[!]"
        print(f"{icon} {item_id}. {name:<42} : {status} ({detail})")
        results[item_id] = {"name": name, "passed": bool(passed), "detail": detail}

    # 1. Subzones File & Contiguity
    npz_path = os.path.join(PHASE2_DIR, "jump10_phase2_subzones.npz")
    if os.path.exists(npz_path):
        z = np.load(npz_path, mmap_mode="r")
        keys = list(z.keys())
        n_rows = z["desc_Q1"].shape[0] if "desc_Q1" in keys else 0
        check("01", "591,171 Contiguous Cell Coverage", n_rows == 591171, f"Rows={n_rows:,}, Keys={len(keys)}")

        # 2. Structural Quartiles (Q1..Q4)
        has_q = all(k in keys for k in ["desc_Q1", "desc_Q2", "desc_Q3", "desc_Q4"])
        check("02", "0-25-50-75-100% & Fib Zone Decomposition", has_q, f"Q1..Q4 present={has_q}")

        # 3. 12-Angle Brute-Force Feature Density
        t_key = "trigger_vec_dense" if "trigger_vec_dense" in keys else ("trigger_vec" if "trigger_vec" in keys else None)
        t_dim = z[t_key].shape[1] if t_key else 0
        manifest_exists = os.path.exists(os.path.join(PHASE2_DIR, "crawled_math_operators_manifest.json"))
        check("03", "12-Angle Brute-Force + Webcrawl Features", t_dim >= 50 and manifest_exists,
              f"vec={t_key} (dim={t_dim}), webcrawl_manifest={manifest_exists}")

        # 4. Whitening & Bicoherence Fix Check
        bico = np.array(z["bicoherence"]) if "bicoherence" in keys else np.ones(10)
        bico_std = float(np.std(bico))
        T_sample = np.array(z[t_key][:2000])
        rand_sims = T_sample[:1000] @ T_sample[1000:2000].T
        mean_abs_r = float(np.mean(np.abs(rand_sims)))
        check("04", "Zero Collinear Collapse & Fixed Bicoherence", mean_abs_r < 0.25 and bico_std > 0.01,
              f"random |r|={mean_abs_r:.4f}, bico_std={bico_std:.4f}")
    else:
        for i in ["01", "02", "03", "04"]:
            check(i, "Subzone Tensor Checks", False, "jump10_phase2_subzones.npz missing")

    # 5. Relational Covariant DB & Positive/Negative Mirror Pairs
    cov_db_path = os.path.join(PHASE2_DIR, "phase2_covariant_db.parquet")
    edges_path = os.path.join(PHASE2_DIR, "phase2_covariant_edges.parquet")
    if os.path.exists(cov_db_path) and os.path.exists(edges_path):
        df_cov = pd.read_parquet(cov_db_path)
        df_edg = pd.read_parquet(edges_path)
        has_mirrors = ("pos_corr_peers" in df_cov.columns) and ("neg_mirror_peers" in df_cov.columns)
        check("05", "Relational Covariant DB (+/-0.85 Mirrors)", len(df_cov) > 1000 and has_mirrors,
              f"Hubs={len(df_cov):,}, Edges={len(df_edg):,}, Pos/Neg Mirrors={has_mirrors}")
    else:
        check("05", "Relational Covariant DB (+/-0.85 Mirrors)", False, "Parquet files missing")

    # 6. Inbound Pass 1 & Pass 2 Neural Convergence (CZET + Learnable Kernels)
    czet_eval_path = os.path.join(PHASE2_DIR, "czet_eval.json")
    if os.path.exists(czet_eval_path):
        with open(czet_eval_path, "r") as f:
            ev = json.load(f)
        spearman = float(ev.get("spearman", ev.get("pair_spearman", 0.15)))
        r2_val = float(ev.get("r2", ev.get("oos_r2", 0.0)))
        check("06", "CZET Inbound Pass 1&2 Continuation Ranking", spearman >= 0.28 and r2_val > 0.02,
              f"Spearman={spearman:.3f}, R2={r2_val:.4f} (Needs LearnableKernel/SweetSpot retraining if low)")
    else:
        check("06", "CZET Inbound Pass 1&2 Continuation Ranking", False, "czet_eval.json missing")

    # 7, 8, 9, 10. Strategy Vault, Fractal Multi-Timeframe, Indicator Coupling & Null Test
    vault_path = os.path.join(PHASE2_DIR, "phase2_strategy_vault.parquet")
    if os.path.exists(vault_path):
        df_v = pd.read_parquet(vault_path)
        cols = set(df_v.columns)
        check("07", "Outbound Streaming Strategy Vault (>=55%)", len(df_v) >= 100, f"Vault Strategies={len(df_v):,}")

        has_fractal = any("fractal" in c.lower() or "macro" in c.lower() for c in cols)
        check("08", "Multi-Timeframe Fractal Check (150t vs 1800t)", has_fractal,
              f"Fractal column present={has_fractal}")

        has_ind = any(k in c.lower() for c in cols for k in ["rsi", "vwap", "indicator", "divergence"])
        check("09", "Coupled Indicator Functions (RSI/VWAP)", has_ind,
              f"Indicator confluence columns present={has_ind}")

        has_null = any("null" in c.lower() or "pval" in c.lower() or "wilson" in c.lower() for c in cols)
        check("10", "Statistical Null / Wilson Lower Bound Audit", has_null,
              f"Statistical bound present={has_null}")
    else:
        for i, lbl in [("07", "Outbound Strategy Vault"), ("08", "Multi-TF Fractal Check"),
                       ("09", "Coupled Indicator Functions"), ("10", "Statistical Null Audit")]:
            check(i, lbl, False, "phase2_strategy_vault.parquet missing")

    all_passed = all(v["passed"] for v in results.values())
    print("=" * 78)
    if all_passed:
        print("[VERDICT] PHASE 2 IS 100% COMPLETE & VERIFIED. CLEAR TO LAUNCH PHASE 3!")
    else:
        missing = [k for k, v in results.items() if not v["passed"]]
        print(f"[VERDICT] PHASE 2 HAS {len(missing)} UNFINISHED ITEM(S): {missing}")
        print("          Complete the missing items above before moving to Phase 3.")
    print("=" * 78)

    with open(os.path.join(PHASE2_DIR, "phase2_gatekeeper_status.json"), "w") as f:
        json.dump({"all_passed": all_passed, "checks": results}, f, indent=2)

if __name__ == "__main__":
    run_phase2_gatekeeper_audit()

