# Jump 10 Geometric Anomaly Engine

A three-node distributed quant-research cluster that decomposes Deriv **Jump 10 Index** tick
streams into 150-tick structural cells, learns a relational *covariant anomaly graph* between
them, and then stress-tests the resulting strategy vault through a Fable-style
sigma-survival / contract-EV auction backtester and a live Deriv WebSocket forward tester.

Everything here was produced by the cluster itself; every number in the PDF is read from the
artifacts in this repository (see `docs/`).

## TL;DR findings

| Phase | Question | Answer |
|---|---|---|
| 1 | Can 591,171 Jump-10 cells be decomposed + Fourier-encoded exactly? | Yes — 100% reconstruction, 0 verification failures |
| 2 | Do cells form a real relational graph with predictive edges? | Yes structurally (11,266 hubs, 241,703 undirected edges, median similarity 0.874), but the *directional* edge is not tradeable |
| 2 item 03 | Do crawled complexity measures add signal? | Weakly (\|r\| ≤ 0.07 vs edge strength) |
| 2 item 06 | Can CZET v2 rank the sweet-spot universe? | Yes — Spearman **0.511** (cov baseline 0.101), R² **0.028**; ranking only, not magnitude |
| 3A | Does the vault survive tick-exact replay + friction + MC? | **No** — PF **0.958**, win **45.2%**, cum equity **−86.7σ gross / −209σ net**, MC P(positive) **18.6%** |
| 3A diagnostic | Is the Phase-2 "55–93% win" real? | No — Phase 2's `verified_edge` is `max(protected_held, directional)`, a selection artifact; directional out-of-sample is **48.8%** |
| 3B | Is the live tester ready? | Built, self-checked, 7 guards, SQLite journal — but Deriv's public WS returns **HTTP 520** from all three nodes, so live payouts are unverified |

**Verdict: do not trade this live.** The signal is payout-agnostic negative; contract-level
profitability appears only under the brief's assumed payouts (ONETOUCH 2.15×), which would
need to clear a **1.42×** breakeven that could not be verified.

## Repository layout

```
.
├── AGENTS.md                     session rules for the 3-node cluster
├── cluster_ctl.py                dispatch a task to kali / 840g7 / all
├── dispatch.py, benchmark.py     orchestration helpers
├── xflip_spawner.py              visible-terminal spawner
├── orchestrator_master.bat
├── phase1/                       tick-cell extraction + Fourier feature build
│   ├── scripts/                  worker, merger, verifier, launcher
│   ├── jobs/                     per-node job specs
│   ├── analysis/                 exactness checks (sample5)
│   └── final/                    verification JSON + SHA256SUMS
├── phase2/                       zone decomposition, covariant graph, vault, CZET, reports
├── phase3/                       Fable EV survival engine, live forward tester, gatekeeper
└── docs/                         this documentation set + the generated PDF
    ├── PROJECT_DOCUMENTATION.pdf (100+ pages)
    ├── PHASE3_BRIEF_REDACTED.md
    └── DATA_MANIFEST.json        SHA256 of every large regenerable artifact
```

## Quick start

```powershell
# health check the cluster
python cluster_ctl.py all "python -c \"import sys,numpy,pandas;print(sys.version)\"" fast_cmd

# phase 1 (already verified): rebuild verification only
python phase1\scripts\verify_phase1.py --final_dir C:\cluster\phase1\final

# phase 2 gatekeeper (10 checks)
python phase2\phase2_gatekeeper_verifier.py

# phase 3A: shard on each node then merge locally
python phase3\phase3_master_engine.py --node_id 0 --num_nodes 3
python cluster_ctl.py 840g7 "C:\Python314\python.exe C:\cluster\phase3\p3_worker.py 1"
python cluster_ctl.py kali "cd /home/user/cluster/phase3 && /home/user/cluster/venv/bin/python -u p3_worker.py 2"
python phase3\phase3_master_engine.py --merge

# phase 3B readiness + live probe
python phase3\phase3_deriv_live_forward_tester.py --self_check
$env:DERIV_APP_ID="..."; $env:DERIV_API_TOKEN="..."
python phase3\phase3_deriv_live_forward_tester.py --ping_test 20
```

## Data policy

Large binary artifacts (`*.npz`, `*.npy`, `*.pt`, raw parquet extracts) are **not** committed.
`docs/DATA_MANIFEST.json` records their byte sizes and SHA-256 hashes so any clone can verify a
regenerated copy. Every one of them is reproducible from the scripts in `phase1/`, `phase2/`,
`phase3/` given the original tick export.

## Security

No credentials are stored in this repository. The Phase-3 brief was committed as
`docs/PHASE3_BRIEF_REDACTED.md` with the API token and app-id removed. The live tester reads
credentials from environment variables or `--creds_file` only, and masks them in every log line.

## License

Private research. All rights reserved.
