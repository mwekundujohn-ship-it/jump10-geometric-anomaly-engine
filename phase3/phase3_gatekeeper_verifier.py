"""PHASE 3 GATEKEEPER - final verification of Stage 3A deliverables and Stage 3B readiness."""
import os
import sys
import json
import sqlite3
import subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\cluster\phase2")
import pqcompat

P3 = r"C:\cluster\phase3"
P2 = r"C:\cluster\phase2"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ::  {detail}" if detail else ""))


# 1 - deliverables exist
files = ["phase3_strategy_ev_leaderboard.parquet", "phase3_deduplicated_equity_curve.parquet",
         "phase3_master_backtest_report.xlsx", "phase3_live_forward_config.json",
         "phase3_whitening_transform.npz", "phase3_deriv_live_forward_tester.py",
         "phase3_gatekeeper_verifier.py", "phase3_executive_summary.json"]
missing = [f for f in files if not os.path.exists(os.path.join(P3, f))]
check("Deliverables present (8)", not missing, f"missing={missing}")

summ = json.load(open(os.path.join(P3, "phase3_executive_summary.json")))
cfg = json.load(open(os.path.join(P3, "phase3_live_forward_config.json")))

# 2 - leaderboard schema + anchor count
lb = pqcompat.read_parquet(os.path.join(P3, "phase3_strategy_ev_leaderboard.parquet"))
need = {"anchor_cell_id", "cons_ev_rise_fall", "cons_ev_onetouch", "cons_ev_barrier",
        "cons_ev_notouch", "cons_ev_multiplier", "inbound_wlb_dir", "mode_dir_win",
        "mode_touch_win", "mode_barrier_win", "mode_notouch_win", "safe_sigma_bins",
        "verified_profitable"}
check("Leaderboard columns", need <= set(lb.columns), f"rows={len(lb)}")
check("Leaderboard covers every sweet-spot anchor", len(lb) == summ["vault_rows_in_sweet_spot"],
      f"{len(lb)} vs {summ['vault_rows_in_sweet_spot']}")

# 3 - orthogonal pruning really holds (|r_anchor| >= 0.82 suppressed)
T = np.load(os.path.join(P3, "phase3_inputs.npz"), mmap_mode="r")["trigger_vec"]
anc = lb.anchor_cell_id.to_numpy(np.int64) - 1
A = np.asarray(T[anc], dtype=np.float64)
An = A / np.linalg.norm(A, axis=1, keepdims=True)
S = np.abs(An @ An.T)
np.fill_diagonal(S, 0.0)
check("Anchor orthogonality |r|<0.82", float(S.max()) < 0.82, f"max_offdiag={S.max():.4f}")

# 4 - kindex leakage: sigma must be causal, i.e. uncorrelated with the forward move
eq = pqcompat.read_parquet(os.path.join(P3, "phase3_deduplicated_equity_curve.parquet"))
check("Equity curve per-trade rows", len(eq) == summ["portfolio_trades"],
      f"{len(eq)} vs {summ['portfolio_trades']}")
if "outbound_cell_id" in eq.columns and "sigma_bps" in eq.columns:
    r = np.corrcoef(eq.sigma_bps.to_numpy(float), eq.gross_dir_149.to_numpy(float))[0, 1]
    check("Sigma causal (|corr| with forward move < 0.10)", abs(r) < 0.10, f"corr={r:+.4f}")
else:
    check("Sigma causal (columns present)", False, "sigma_bps/gross_dir_149 missing")

# 5 - one trade per cell (dedup)
if "outbound_cell_id" in eq.columns:
    dup = int(eq.outbound_cell_id.duplicated().sum())
    check("One trade per cell", dup == 0, f"duplicates={dup}")

# 6 - friction ladder monotonicity
c = summ["cum_equity_sigma"]
check("Friction ladder monotone", c["gross_00"] >= c["net_03"] >= c["net_06"] >= c["net_10"],
      f"{c}")

# 7 - Monte Carlo sanity
check("MC runs & ordering", summ["mc95_worst_dd_sigma"] >= summ["mc99_worst_dd_sigma"],
      f"mc95={summ['mc95_worst_dd_sigma']} mc99={summ['mc99_worst_dd_sigma']}")
check("Kelly non-negative logic", summ["kelly_fraction"] == 0.0 and
      summ["portfolio_profit_factor_net03"] < 1.0,
      "kelly=0 because PF<1 (correct)")

# 8 - whitening bridge / live parity artifacts
w = np.load(os.path.join(P3, "phase3_whitening_transform.npz"))
bridge = json.load(open(os.path.join(P3, "phase3_whitening_bridge.json")))
check("Whitening transform arrays", all(k in w.files for k in
      ("q_mu", "q_sd", "anchor_vecs", "anchor_ids", "anchor_actions")),
      f"anchors={len(w['anchor_ids'])}")
check("Live bridge trained (report parity honestly)", os.path.exists(
    os.path.join(P3, "phase3_live_bridge.pt")),
    f"match_recall={bridge['match_recall']:.3f} precision={bridge['match_precision']:.3f} "
    f"IoU={bridge['match_iou']:.3f}  <- live triggers are a SUBSET of backtest triggers")

# 9 - forward tester: compiles, self-checks, 7 guards, SQLite
src = open(os.path.join(P3, "phase3_deriv_live_forward_tester.py"), encoding="utf-8").read()
guards = ["SIGMA_GATE", "PAYOUT_MONITOR", "LAG_GUARD", "LIVE_CALIBRATOR", "DRAWDOWN_HALT",
          "KellyStakeSizer", "HOUR_FILTER"]
check("7 Fable guards present", all(g in src for g in guards),
      ",".join(g for g in guards if g not in src) or "all")
rc = subprocess.run([sys.executable, "-c",
                     "import py_compile;py_compile.compile(r'%s',doraise=True)" %
                     os.path.join(P3, "phase3_deriv_live_forward_tester.py")],
                    capture_output=True, text=True)
check("Forward tester compiles", rc.returncode == 0, rc.stderr.strip()[:120])
rc = subprocess.run([sys.executable, os.path.join(P3, "phase3_deriv_live_forward_tester.py"),
                     "--self_check"], capture_output=True, text=True)
check("Forward tester self-check", rc.returncode == 0 and "self-check OK" in rc.stdout,
      (rc.stdout or rc.stderr).strip().splitlines()[-1][:120] if (rc.stdout or rc.stderr) else "")
db = os.path.join(P3, "fable_forward_cal.db")
ok_db = False
if os.path.exists(db):
    con = sqlite3.connect(db)
    tabs = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    ok_db = "forward_trades" in tabs
check("SQLite LiveCalibrator journal", ok_db, db)

# 10 - credential hygiene: no token material in any Phase 3 artifact
hits = []
for fn in os.listdir(P3):
    p = os.path.join(P3, fn)
    if os.path.isfile(p) and os.path.getsize(p) < 8_000_000 and fn.endswith(
            (".py", ".json", ".txt", ".log", ".csv", ".md")):
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if ("pat_" + "a9c5f3d4") in t or ("34NbJFTA3U4VL" + "tchImAaP") in t:
            hits.append(fn)
check("No credentials in Phase 3 artifacts", not hits, f"leaks={hits}")

# 11 - live config contents
check("Live config has anchors+hours+sigma edges",
      len(cfg.get("approved_anchor_cell_ids", [])) > 0 and
      len(cfg.get("blocked_utc_hours", [])) > 0 and
      len(cfg.get("sigma_quantile_edges_bps", [])) == 6,
      f"anchors={len(cfg.get('approved_anchor_cell_ids', []))} "
      f"hours={len(cfg.get('blocked_utc_hours', []))}")

print("\n" + "=" * 72)
npass = sum(1 for _, ok, _ in results if ok)
print(f"PHASE 3 GATEKEEPER: {npass}/{len(results)} PASS")
for n, ok, d in results:
    if not ok:
        print(f"  !! {n}  ::  {d}")
print("=" * 72)
json.dump([{"check": n, "pass": ok, "detail": d} for n, ok, d in results],
          open(os.path.join(P3, "phase3_gatekeeper_results.json"), "w"), indent=2)
