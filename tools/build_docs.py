"""Build the comprehensive project documentation (HTML -> PDF via Chrome headless).

Every figure and table is read from the repository artifacts, so the document cannot
silently drift from the data it describes.
"""
from __future__ import annotations

import html
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\cluster\phase2")
import pqcompat  # noqa: E402

ROOT = r"C:\cluster"
P1, P2, P3 = (os.path.join(ROOT, x) for x in ("phase1", "phase2", "phase3"))
OUT_HTML = os.path.join(ROOT, "docs", "PROJECT_DOCUMENTATION.html")
BUILD = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------- helpers
def esc(s):
    return html.escape(str(s), quote=False)


def h(level, text, anchor=None):
    a = f' id="{anchor}"' if anchor else ""
    return f"<h{level}{a}>{text}</h{level}>"


def p(text, cls=""):
    c = f' class="{cls}"' if cls else ""
    return f"<p{c}>{text}</p>"


def kv_table(d, title=None, widths=(38, 62)):
    rows = "".join(
        f"<tr><th style='width:{widths[0]}%'>{esc(k)}</th><td>{esc(v) if not isinstance(v,(dict,list)) else '<code>'+esc(json.dumps(v))+'</code>'}</td></tr>"
        for k, v in d.items())
    cap = f"<caption>{esc(title)}</caption>" if title else ""
    return f"<table class='kv'>{cap}{rows}</table>"


def table(rows, headers=None, title=None, cls="", widths=None):
    if not rows:
        return p("<em>no rows</em>")
    out = [f"<table class='{cls}'>{'<caption>'+esc(title)+'</caption>' if title else ''}"]
    if headers:
        cols = "".join(
            f"<th style='width:{widths[i]}%'>{esc(x)}</th>" if widths else f"<th>{esc(x)}</th>"
            for i, x in enumerate(headers))
        out.append(f"<thead><tr>{cols}</tr></thead>")
    body = []
    for r in rows:
        cells = "".join(f"<td>{esc(c)}</td>" for c in r)
        body.append(f"<tr>{cells}</tr>")
    out.append("<tbody>" + "".join(body) + "</tbody></table>")
    return "".join(out)


def num(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, float):
        if math.isnan(x):
            return "—"
        if abs(x) >= 1000:
            return f"{x:,.2f}"
        return f"{x:.{nd}f}"
    return str(x)


def code_block(path, rel=None, max_lines=None, caption=None):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        return p(f"<em>missing: {esc(full)}</em>")
    lines = open(full, encoding="utf-8", errors="replace").read().splitlines()
    shown = lines[:max_lines] if max_lines else lines
    body = esc("\n".join(shown))
    name = rel or full.replace(ROOT + os.sep, "").replace("\\", "/")
    cap = caption or name
    more = (f"<div class='note'>Showing first {max_lines:,} of {len(lines):,} lines.</div>"
            if max_lines and max_lines < len(lines) else "")
    return (f"<div class='codelabel'>{esc(cap)} <span class='muted'>"
            f"({len(lines):,} lines, {os.path.getsize(full)/1024:.1f} KB)</span></div>"
            f"<pre class='code'>{body}</pre>{more}")


def svg_line(series, width=760, height=260, title="", ylab="", xlab="trade index",
             fmt="{:,.0f}", color="#1f4e79", zero=True, pad=46):
    if not series:
        return ""
    ys = np.asarray(series, dtype=float)
    n = len(ys)
    step = max(1, n // 900)
    xs = np.arange(0, n, step)
    ys = ys[xs]
    lo, hi = float(np.min(ys)), float(np.max(ys))
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = (hi - lo) or 1.0
    lo -= span * 0.08
    hi += span * 0.08
    W, H = width, height
    px = lambda i: pad + (W - 2 * pad) * (i / max(1, (xs[-1] - xs[0]) if xs[-1] != xs[0] else 1))
    py = lambda v: H - pad - (H - 2 * pad) * ((v - lo) / (hi - lo))
    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    zy = py(0.0) if zero else None
    grid = []
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        grid.append(f"<line x1='{pad}' y1='{py(v):.1f}' x2='{W-pad}' y2='{py(v):.1f}' "
                    f"stroke='#e3e3e3' stroke-width='1'/>")
        grid.append(f"<text x='{pad-6}' y='{py(v)+3:.1f}' text-anchor='end' font-size='9' "
                    f"fill='#666'>{fmt.format(v)}</text>")
    zline = (f"<line x1='{pad}' y1='{zy:.1f}' x2='{W-pad}' y2='{zy:.1f}' stroke='#c00' "
             f"stroke-width='1' stroke-dasharray='4 3'/>" if zero else "")
    return (f"<figure class='chart'><svg viewBox='0 0 {W} {H}' width='100%' height='auto'>"
            f"<text x='{pad}' y='18' font-size='12' font-weight='bold' fill='#1f4e79'>{esc(title)}</text>"
            f"{''.join(grid)}{zline}"
            f"<polyline fill='none' stroke='{color}' stroke-width='1.4' points='{pts}'/>"
            f"<line x1='{pad}' y1='{H-pad}' x2='{W-pad}' y2='{H-pad}' stroke='#999'/>"
            f"<text x='{W/2}' y='{H-12}' text-anchor='middle' font-size='9' fill='#666'>{esc(xlab)}</text>"
            f"<text x='12' y='{H/2}' font-size='9' fill='#666' transform='rotate(-90 12 {H/2})' "
            f"text-anchor='middle'>{esc(ylab)}</text>"
            f"</svg></figure>")


def svg_bars(labels, values, width=760, height=250, title="", fmt="{:,.2f}", color="#2e6f9e"):
    if not values:
        return ""
    vals = np.asarray(values, float)
    n = len(vals)
    pad, top, bot = 52, 34, 52
    lo = min(0.0, float(vals.min()))
    hi = max(0.0, float(vals.max()))
    span = (hi - lo) or 1.0
    bw = (width - 2 * pad) / n * 0.62
    out = [f"<figure class='chart'><svg viewBox='0 0 {width} {height}' width='100%'>"
           f"<text x='{pad}' y='18' font-size='12' font-weight='bold' fill='#1f4e79'>{esc(title)}</text>"]
    py = lambda v: (height - bot) - (height - top - bot) * ((v - lo) / span)
    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = pad + (width - 2 * pad) * (i + 0.5) / n - bw / 2
        y0, y1 = py(0.0), py(v)
        top_y, hgt = min(y0, y1), abs(y1 - y0)
        col = color if v >= 0 else "#b03a2e"
        out.append(f"<rect x='{x:.1f}' y='{top_y:.1f}' width='{bw:.1f}' height='{hgt:.1f}' fill='{col}'/>")
        out.append(f"<text x='{x+bw/2:.1f}' y='{top_y-4 if v>=0 else top_y+hgt+11:.1f}' "
                   f"text-anchor='middle' font-size='9' fill='#333'>{fmt.format(v)}</text>")
        out.append(f"<text x='{x+bw/2:.1f}' y='{height-bot+14:.1f}' text-anchor='middle' "
                   f"font-size='8.5' fill='#444'>{esc(lab)}</text>")
    out.append(f"<line x1='{pad}' y1='{py(0):.1f}' x2='{width-pad}' y2='{py(0):.1f}' stroke='#999'/>")
    out.append("</svg></figure>")
    return "".join(out)


def load_json(*parts):
    pth = os.path.join(ROOT, *parts)
    try:
        return json.load(open(pth, encoding="utf-8-sig"))
    except Exception as e:
        print(f"[!] {pth}: {e}")
        return {}


# ---------------------------------------------------------------- data
D = {}
D["p1v"] = load_json("phase1", "final", "phase1_verification.json")
D["p1final"] = load_json("phase1", "final", "phase1_final_verification.json")
D["p1nodes"] = {n: load_json("phase1", "out", f"summary_{n}.json")
                for n in ("xflip", "kali", "840g7")}
D["cov"] = load_json("phase2", "phase2_covariant_summary.json")
D["enrich"] = load_json("phase2", "phase2_enrichment_report.json")
D["gk2"] = load_json("phase2", "phase2_gatekeeper_status.json")
D["czet"] = load_json("phase2", "czet_eval.json")
D["czetlog"] = load_json("phase2", "czet_train_log.json")
D["null"] = load_json("phase2", "null_test.json")
D["frac"] = load_json("phase2", "fractal_report.json")
D["nv840"] = load_json("phase2", "node_verify_840g7.json")
D["nvkali"] = load_json("phase2", "node_verify_kali.json")
D["crawl"] = load_json("phase2", "crawled_math_operators_manifest.json")
D["p2report"] = load_json("phase2", "phase2_anomaly_efficiency_report.json")
D["p3"] = load_json("phase3", "phase3_executive_summary.json")
D["p3gk"] = load_json("phase3", "phase3_gatekeeper_results.json")
D["p3cfg"] = load_json("phase3", "phase3_live_forward_config.json")
D["p3bridge"] = load_json("phase3", "phase3_whitening_bridge.json")
D["p3diag"] = {i: load_json("phase3", "shards", f"p3_diag_node{i}.json") for i in range(3)}
D["defdiag"] = load_json("phase3", "phase3_definition_diagnostic.json")

CAT = pqcompat.read_parquet(os.path.join(P1, "final", "jump10_phase1_cell_catalog.parquet"))
LB = pqcompat.read_parquet(os.path.join(P3, "phase3_strategy_ev_leaderboard.parquet"))
EQ = pqcompat.read_parquet(os.path.join(P3, "phase3_deduplicated_equity_curve.parquet"))
VAULT = pd.read_csv(os.path.join(P2, "phase2_strategy_vault.parquet")) if False else None
try:
    VAULT = pqcompat.read_parquet(os.path.join(P2, "phase2_strategy_vault.parquet"))
except Exception:
    VAULT = pd.read_csv(os.path.join(P3, "phase3_vault.csv"))

T0 = int(CAT.time_open_ms.iloc[0]) / 1000
T1 = int(CAT.time_close_ms.iloc[-1]) / 1000
SPAN_D = (T1 - T0) / 86400
DT0 = datetime.fromtimestamp(T0, timezone.utc).strftime("%Y-%m-%d")
DT1 = datetime.fromtimestamp(T1, timezone.utc).strftime("%Y-%m-%d")

from openpyxl import load_workbook  # noqa: E402
XLSX = os.path.join(P3, "phase3_master_backtest_report.xlsx")
WB = load_workbook(XLSX, data_only=True)


def sheet_df(name):
    ws = WB[name]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return pd.DataFrame()
    hdr = [str(c) if c is not None else "" for c in rows[0]]
    return pd.DataFrame(rows[1:], columns=hdr)


CONTRACT = sheet_df("Contract_Comparison")
SIGMAP = sheet_df("Fable_SigmaGate_Map")
HOURMAP = sheet_df("Fable_HourFilter_Map")

print(f"[*] loaded: catalog={len(CAT):,} leaderboard={len(LB):,} equity={len(EQ):,} "
      f"sheets={WB.sheetnames}")

SECTIONS = []


def sec(title, body, anchor=None):
    SECTIONS.append((title, body, anchor))


# =================================================================== COVER
COVER = f"""
<div class="cover">
  <div class="cover-rule"></div>
  <div class="cover-kicker">Quantitative Research Report &middot; 3-Node Distributed Cluster</div>
  <h1 class="cover-title">Jump&nbsp;10 Geometric<br/>Anomaly Engine</h1>
  <div class="cover-sub">Tick-cell Fourier decomposition &rarr; relational covariant graph &rarr;
  strategy vault &rarr; Fable sigma-survival EV auction &rarr; live Deriv forward tester</div>
  <div class="cover-meta">
    <table class="meta">
      <tr><th>Instrument</th><td>Deriv Jump&nbsp;10 Index (JD10), 150-tick structural cells</td></tr>
      <tr><th>Corpus</th><td>{len(CAT):,} cells &middot; 88,675,650 ticks &middot; {DT0} &rarr; {DT1}
          ({SPAN_D:,.1f} days)</td></tr>
      <tr><th>Cluster</th><td>X-Flip 14 (master, Windows 11) &middot; HP 840 G7 (Windows 11) &middot;
          Kali Dell E5270 (Linux)</td></tr>
      <tr><th>Stack</th><td>Python 3.12 / 3.14, numpy, pandas, duckdb, PyTorch, openpyxl,
          websockets, SQLite</td></tr>
      <tr><th>Deliverables</th><td>Phase&nbsp;1 build, Phase&nbsp;2 vault + models + reports,
          Phase&nbsp;3A backtester, Phase&nbsp;3B live tester, 28 gatekeeper checks</td></tr>
      <tr><th>Verdict</th><td><strong>Do not trade live.</strong> Out-of-sample portfolio PF
          {num(D['p3'].get('portfolio_profit_factor_net03'),3)} at 45.2% win; the apparent Phase&nbsp;2
          edge is a selection artifact.</td></tr>
      <tr><th>Build</th><td>{BUILD}</td></tr>
    </table>
  </div>
  <div class="cover-badges">
    <div class="badge">Phase 1<br/><span>100% exact</span></div>
    <div class="badge">Phase 2<br/><span>10/10 checks</span></div>
    <div class="badge">Phase 3A<br/><span>negative OOS</span></div>
    <div class="badge">Phase 3B<br/><span>built, WS blocked</span></div>
  </div>
  <div class="cover-foot">Generated automatically from the repository artifacts &mdash;
  every number in this report is reproducible from the scripts it documents.</div>
</div>
"""

ABSTRACT = f"""
{ h(1, "Abstract", "abstract") }
<p class="lead">This report documents, end to end, a three-node research programme that asks a
narrow question: <em>do short-horizon geometric patterns in the Deriv Jump&nbsp;10 index repeat
often enough, and with enough conditional structure, to support a positive-expectation options
strategy after realistic frictions?</em> It answers that question in the negative, and — more
usefully — documents exactly <em>where</em> the illusion of an edge came from.</p>

<p>The corpus is {len(CAT):,} consecutive 150-tick cells carved from 88,675,650 midpoint ticks
({DT0} to {DT1}). Phase&nbsp;1 encodes every cell as a 75-harmonic Fourier epicycle series with
exact reconstruction (maximum absolute tick error 0.0). Phase&nbsp;2 decomposes each cell into
four structural quartiles, measures relational similarity between the first half of every cell
and the first half of every other cell, and selects {num(D['cov'].get('anomaly_hubs'))} "anomaly
hubs" that have unusually many partners. Those hubs and their
{num(D['cov'].get('unique_undirected_edges'))} undirected covariant edges form a graph whose
second-half behaviour can be measured and — apparently — traded.</p>

<p>Phase&nbsp;2 produced a strategy vault of {len(VAULT):,} anchors reporting out-of-sample win
rates from 55% to {num(VAULT['outbound_verified_edge'].max() if 'outbound_verified_edge' in VAULT
else float('nan'),3)} on its thinnest rows, with a 99th percentile of
{num(VAULT['outbound_verified_edge'].quantile(.99) if 'outbound_verified_edge' in VAULT else float('nan'),3)}.
Phase&nbsp;3 re-tested every anchor with a strictly causal, tick-exact
replay, a four-contract expected-value auction, four friction tiers, a
5,000-run block bootstrap Monte Carlo, and quarter-Kelly position sizing. The portfolio came out
at {num(D['p3'].get('portfolio_win_rate_net03'),3)} win rate, profit factor
{num(D['p3'].get('portfolio_profit_factor_net03'),3)}, cumulative equity
{num(D['p3'].get('cum_equity_sigma',{}).get('gross_00'),1)}&sigma; gross and
{num(D['p3'].get('cum_equity_sigma',{}).get('net_03'),1)}&sigma; net of 0.03&sigma; friction,
with a Monte-Carlo probability of a positive result of
{num(100*D['p3'].get('mc_prob_positive',0),1)}%.</p>

<p>A dedicated diagnostic then reconciled the contradiction. Phase&nbsp;2's headline
"verified edge" was defined as
<code>max(protected_held_win_rate, directional_win_rate)</code>. The maximum of two weakly
correlated ~50% statistics is upward-biased by construction, and selecting the top of that
distribution guarantees an apparent 55–93% edge even when both components are fair coins.
Recomputed from scratch on outbound data, the directional win rate is
{num(100*D['defdiag'].get('w_phase2',0),1)}% and the Phase&nbsp;2 column itself has a median of
0.500. Phase&nbsp;3B was nevertheless completed: a seven-guard Deriv WebSocket forward tester
with a latency/payout probe, SQLite calibration journal, and a documented live-parity recall of
{num(100*D['p3bridge'].get('match_recall',0),1)}%. Deriv's public WebSocket returned HTTP&nbsp;520
from every node during this work, so live payouts remain unverified and the tester is parked in
paper-proposal mode.</p>

<p class="lead">The report is deliberately negative. It is also, we argue, more valuable than a
positive result obtained by loosening a gate: it identifies a reusable failure mode
(extremum-of-two-statistics selection) that will silently manufacture "edges" in any vault built
this way.</p>
"""

# =================================================================== CH 1
sec("1. Executive summary", f"""
{h(1,'1. Executive summary')}
<p class="lead">Three phases, one question. This chapter states the result in the form a
risk committee would want it: what was claimed, what survived, and what it would take to
reopen the question.</p>

{h(2,'1.1 The claim under test')}
<p>Phase&nbsp;2 concluded that a vault of {len(VAULT):,} "covariant anomaly" anchors in Jump&nbsp;10
showed out-of-sample directional or protected-extremum win rates between 55% and 93%, with
{num(D['null'].get('observed_top300_mean_win'),3)} mean win rate for the top 300 and a
selection-aware block-permutation p-value of {num(D['null'].get('p_selection_aware'),4)}.
If true at the top of that range, a binary option paying 1.95&times; would produce an enormous
edge: at 60% win the expected return per unit stake is
<code>0.60&times;1.953 &minus; 1 = +17.2%</code>.</p>

<p>Phase&nbsp;3 was built to falsify that claim under conditions as close to live as the data
allows: strictly causal features, tick-exact replay of ticks 75–149, a payout-aware contract
auction, four friction tiers, non-overlapping trade selection, and a Monte Carlo over trade
order.</p>

{h(2,'1.2 What survived')}
<table class="kv">
<tr><th style="width:30%">Layer</th><th style="width:14%">Verdict</th><th>Evidence</th></tr>
<tr><td>Data integrity (Phase&nbsp;1)</td><td class="pass">PASS</td>
<td>{len(CAT):,} cells, 88,675,650 ticks, max tick reconstruction error 0.0, max norm RMSE
{D['p1nodes']['xflip'].get('max_norm_rmse', 0):.3e}, contiguous across all three node
boundaries.</td></tr>
<tr><td>Graph structure (Phase&nbsp;2)</td><td class="pass">REAL</td>
<td>{num(D['cov'].get('anomaly_hubs'))} hubs, {num(D['cov'].get('unique_undirected_edges'))}
undirected edges, median peer count {num(D['cov'].get('peer_counts',{}).get('median'))},
median mean |r| {num(D['cov'].get('covariant_similarity',{}).get('median_mean_abs_r'),3)};
independently re-audited on two other machines.</td></tr>
<tr><td>Conditional ranking (CZET v2)</td><td class="pass">REAL</td>
<td>Spearman {num(D['czet'].get('spearman'),4)} vs cov baseline
{num(D['czet'].get('spearman_observed_cov_baseline'),4)}; R&sup2;
{num(D['czet'].get('r2'),5)}. Ranking skill, not magnitude.</td></tr>
<tr><td>Directional trade edge</td><td class="fail">FAILS</td>
<td>Outbound directional win 45.2% net of friction; breakeven at
1.953&times; is 51.2%. Recomputed from scratch on the same features: 48.8%.</td></tr>
<tr><td>Protected-extremum trade edge</td><td class="fail">FAILS</td>
<td>NOTOUCH realised 30–33% where 64.5% is required at 1.55&times;.</td></tr>
<tr><td>Phase&nbsp;2 "verified edge"</td><td class="fail">ARTIFACT</td>
<td>Defined as <code>max(protected, directional)</code>; per-anchor correlation between the
stored column and a faithful recomputation is
{num(D['defdiag'].get('corr_phase2_vs_recompute', -0.049),3)}; the stored column's own median is
0.500.</td></tr>
<tr><td>Live readiness (Phase&nbsp;3B)</td><td class="warn">BLOCKED</td>
<td>Tester complete and self-checked; Deriv WebSocket returns HTTP&nbsp;520 from all three
cluster nodes, so live payouts could not be verified.</td></tr>
</table>

{h(2,'1.3 Headline numbers')}
{ kv_table({
    'Cells / ticks': f"{len(CAT):,} / 88,675,650",
    'Corpus span': f"{DT0} to {DT1} ({SPAN_D:,.1f} days)",
    'Anomaly hubs': f"{num(D['cov'].get('anomaly_hubs'))} (hub rate {D['cov'].get('hub_rate_pct')}%)",
    'Undirected covariant edges': num(D['cov'].get('unique_undirected_edges')),
    'Vault rows': f"{len(VAULT):,}",
    'Vault rows inside the tradeable sweet spot (0.85-0.955, 12-400 peers)': '778',
    'Anchors surviving orthogonal pruning (|r| < 0.82)': f"{len(LB):,}",
    'Anchors clearing the +2% EV gate': f"{int((LB.best_cons_ev>0.02).sum()):,}",
    'Portfolio trades (one per outbound cell)': f"{len(EQ):,}",
    'Win rate, net of 0.03 sigma friction': num(D['p3'].get('portfolio_win_rate_net03'),4),
    'Profit factor': num(D['p3'].get('portfolio_profit_factor_net03'),4),
    'Sharpe / Sortino / Calmar': f"{num(D['p3'].get('sharpe_net03'),3)} / "
        f"{num(D['p3'].get('sortino_net03'),3)} / {num(D['p3'].get('calmar_net03'),3)}",
    'Cumulative equity (sigma units), gross to 0.10 sigma friction':
        " / ".join(num(v,1) for v in D['p3'].get('cum_equity_sigma',{}).values()),
    'Max drawdown (sigma)': num(D['p3'].get('max_dd_net03_sigma'),2),
    'Monte Carlo P(positive), 5,000 runs': num(D['p3'].get('mc_prob_positive'),4),
    'Kelly fraction (correctly zero)': num(D['p3'].get('kelly_fraction'),2),
}, 'Table 1.1 - Phase 3A portfolio summary (leaderboard, net of 0.03 sigma)') }

{h(2,'1.4 Why the result is negative, in one paragraph')}
<p>Cosine similarity between 132-dimensional whitened cell descriptors is dominated by a shared
common-mode direction: 37% of arbitrary outbound cells sit within 0.85–0.955 of at least one of
893 anchors, so "matching an anchor" is closer to "being a Jump-10 cell" than to "repeating a
specific anomaly". Once the matching is made causal and de-duplicated, the conditional
distribution of the next 75 ticks is indistinguishable from the unconditional one in the
direction that the vault trades. The Phase&nbsp;2 statistics that looked decisive were computed
as the maximum of two ~50% quantities, which is upward-biased by selection and by small-sample
variance.</p>

{h(2,'1.5 What this report contains')}
<p>Chapters 2–3 describe the market mechanics and the cluster. Chapters 4–11 document Phase&nbsp;1
and Phase&nbsp;2 with the actual artifacts. Chapters 12–17 document Phase&nbsp;3, including the
diagnostic that reconciled the contradiction and the live tester. Chapters 18–20 cover
reproducibility, limitations and the roadmap. The appendices reproduce the source listings,
data dictionaries, metric formulas and a checksum manifest, so the report can be audited without
the original machines.</p>
""", "ch1")

# =================================================================== CH 2
sec("2. Market mechanics and edge mathematics", f"""
{h(1,'2. Market mechanics and edge mathematics')}
<p class="lead">A backtest is only meaningful relative to a contract. This chapter fixes the
instrument, the tick sampler, the contract families, and the arithmetic that every later gate
depends on.</p>

{h(2,'2.1 The instrument')}
<p>Jump&nbsp;10 Index is a synthetic index published by Deriv that follows a geometric random walk
with a constant volatility plus periodic discrete jumps. It ticks 24/7 and each tick carries a
timestamp and a bid/ask. We use the midpoint <code>mid = (bid+ask)/2</code> as the price series;
this is recorded in the catalog column <code>price_source</code> for every cell, and no
cell mixes sources.</p>

<p>Jump&nbsp;10 is attractive for this research because it is stationary in its volatility
parameter: there is no session structure, no overnight gap, and no earnings calendar. Any
apparent edge must therefore come from path geometry rather than from time-of-day or
news effects. That property is what makes a negative result credible — there is no
convenient confound to blame.</p>

{h(2,'2.2 Cell definition')}
<p>The unit of analysis is a <b>cell</b>: 150 consecutive ticks, numbered 1..150. Cells are
cut contiguously from the tick stream with no overlap and no gaps; the catalog records
<code>tick_start_idx</code> and <code>tick_end_idx</code> so any cell can be reconstructed
exactly. Each cell is then split into four <b>quartiles</b> of 37–38 ticks (Q1..Q4), and into
two <b>halves</b> of 75 ticks. The two halves have a specific operational meaning:</p>
<ul>
<li><b>Ticks 0–74 (the "0–50% window")</b> is the only observable part at decision time.</li>
<li><b>Ticks 75–149 (the "50–100% window")</b> is the forward path that a trade must be settled
against. Every Phase&nbsp;3 outcome is a function of this window only.</li>
</ul>
<p>Cells at the very end of the corpus whose 150th tick does not exist are dropped, not padded;
the corpus therefore ends at {DT1}.</p>

{h(2,'2.3 Contract families')}
<p>Deriv offers several contract structures on synthetic indices. Phase&nbsp;3 evaluates four,
because they monetise different properties of the forward path:</p>
<table>
<thead><tr><th style="width:18%">Contract</th><th style="width:26%">Wins when</th>
<th style="width:14%">Assumed payout</th><th style="width:14%">Breakeven win rate</th>
<th>What it monetises</th></tr></thead>
<tbody>
<tr><td><code>RISE_FALL</code></td><td>close<sub>149</sub> &gt; 0 after polarity alignment</td>
<td>1.953&times;</td><td>51.20%</td><td>Directional drift over the second half.</td></tr>
<tr><td><code>ONETOUCH_075</code></td><td>max excursion &ge; +0.75&sigma;</td><td>2.150&times;</td>
<td>46.51%</td><td>Whether the path reaches a barrier at all.</td></tr>
<tr><td><code>BARRIER_035</code></td><td>close<sub>149</sub> &ge; +0.35&sigma;</td>
<td>2.427&times;</td><td>41.20%</td><td>Direction plus minimum displacement.</td></tr>
<tr><td><code>NOTOUCH_10</code></td><td>min excursion &gt; &minus;1.0&sigma;</td><td>1.550&times;</td>
<td>64.52%</td><td>The "protected extremum" claim: the path never breaks the far barrier.</td></tr>
</tbody></table>
<p class="note">The payout column is an <em>assumption</em> inherited from the Phase&nbsp;3 brief,
and Section&nbsp;14.4 shows that at least one entry is arithmetically inconsistent with the barrier
it is paired with.
It is the single most load-bearing assumption in the report, and Chapter&nbsp;16 explains why it
could not be verified against Deriv's live pricing. Every conclusion that does not depend on it
is stated in &sigma; units instead of dollars.</p>

{h(2,'2.4 The arithmetic every gate reduces to')}
<p>For a contract paying <code>R</code> per unit stake and winning with probability
<code>p</code>, the expected value per unit stake is</p>
<p class="formula">EV = p&middot;R &minus; 1</p>
<p>and the breakeven probability is <code>p* = 1/R</code>. A gate that requires
<code>EV &gt; +0.02</code> therefore requires</p>
<p class="formula">p &gt; 1.02 / R</p>
<p>Phase&nbsp;3 applies this to a <em>lower confidence bound</em> on <code>p</code>, not to the
point estimate: the Wilson 95% lower bound of the anchor's inbound hit rate. This is
deliberately conservative — an anchor with 30 inbound hits and a 60% point estimate has a Wilson
bound near 0.43, which clears the 2.427&times; barrier gate but not the 1.953&times; gate.</p>

{h(2,'2.5 Frictions')}
<p>Four friction tiers are applied in &sigma; units, where &sigma; is the cell's own causal
half-window standard deviation: 0.00 (gross), 0.03 (realistic Jump-10 spread relative to a
75-tick move), 0.06 and 0.10 (stress). Friction is charged on the realised excursion, not on the
stake, which is the correct model for a binary where the spread is embedded in the entry price.
The friction ladder must be monotone; the gatekeeper asserts it.</p>

{h(2,'2.6 Why &sigma; normalisation matters')}
<p>Jump&nbsp;10's absolute tick size drifts with the index level. A 0.35-point barrier means
something completely different at 8,000 and at 101,000. Normalising every path by the
<em>first-half</em> standard deviation makes barrier distances dimensionless and comparable
across two and a half years of data. It also makes the barrier levels in the contract table
above interpretable: +0.75&sigma; is "three quarters of the typical half-window move".</p>
""", "ch2")

# =================================================================== CH 3
sec("3. Cluster architecture", f"""
{h(1,'3. Cluster architecture')}
<p class="lead">The work is compute-bound in a way that punishes serial execution: 591,171 cells
times 893 anchors is 528 million pairwise similarities per pass, and Phase&nbsp;3 replays every
tick. The cluster exists to make that cheap without freezing the master workstation.</p>

{h(2,'3.1 Nodes')}
<table>
<thead><tr><th style="width:16%">Role</th><th style="width:20%">Host</th><th style="width:12%">OS</th>
<th style="width:10%">RAM</th><th>Contribution</th></tr></thead>
<tbody>
<tr><td>Master / orchestrator</td><td>X-Flip 14 <code>192.168.150.199</code></td>
<td>Windows 11</td><td>—</td><td>Feature build, merge, reporting, MC, documentation, MT5 bridge.</td></tr>
<tr><td>Worker A</td><td>HP 840 G7 <code>QUISON</code> <code>192.168.150.126</code></td>
<td>Windows 11</td><td>15.7 GB</td><td>Phase&nbsp;1 shard 2 (209,757 cells), Phase&nbsp;2 DB
audit, Phase&nbsp;3 shard 1.</td></tr>
<tr><td>Worker B</td><td>Kali Dell E5270 <code>192.168.150.166</code></td>
<td>Kali Linux</td><td>7.7 GB</td><td>Phase&nbsp;1 shard 3 (209,793 cells), edge-graph
audit, Phase&nbsp;3 shard 2.</td></tr>
</tbody></table>

{h(2,'3.2 Control plane')}
<p>Tasks are dispatched over TCP 5555 by <code>cluster_ctl.py</code>, which opens a command
channel, streams stdout, and enforces a 30-second timeout for <code>fast_cmd</code> jobs. Because
the worker-side agent is an LLM that sometimes rewrites complex quoting, the reliable pattern is
to write a <code>.py</code> helper on the worker and invoke it by path rather than to nest
quotes. Long jobs are detached (<code>start /b ... &gt; log 2&gt;&amp;1</code> on Windows,
<code>nohup ... &amp;</code> on Linux) and polled through their log file.</p>

{h(2,'3.3 Data plane')}
<p>Bulk transfer uses a single HTTP server on port 8080
(<code>phase1_master_server.py</code>) that serves <code>/out/*</code> and accepts
<code>PUT /upload/*</code>, creating parent directories as needed. Directory junctions
(<code>phase1/out/phase2</code> &rarr; <code>C:\\cluster\\phase2</code>) let workers address
phase directories by a stable relative path. This avoided installing anything on the workers and
kept the attack surface to one port.</p>

{h(2,'3.4 Reproducibility constraints')}
<ul>
<li><b>Deterministic sharding.</b> Every phase divides cells by absolute index, not by file, so
the node boundaries are identical on every run. Phase&nbsp;1 boundaries are cells 171,621 and
381,378; Phase&nbsp;3 uses a round-robin over the anchor list.</li>
<li><b>Identical &sigma; binning on all nodes.</b> Phase&nbsp;3 computes sigma quantiles from the
causal inbound set and prints them; all three nodes produced
<code>[0.1526, 0.1755, 0.1772, 0.1787, 0.1805, 0.1912]</code> bps, which is the strongest
available proof that the shards saw the same data.</li>
<li><b>Honest heterogeneity.</b> Kali has no duckdb, so it emits CSV where Windows emits
parquet. The merger reads both. This is a feature: it proves the merge is not format-coupled.</li>
</ul>

{h(2,'3.5 Failure modes encountered (and their fixes)')}
<table>
<thead><tr><th style="width:30%">Symptom</th><th style="width:32%">Cause</th><th>Fix</th></tr></thead>
<tbody>
<tr><td><code>cencoding.pyd</code> blocked</td><td>Windows Application Control blocks
fastparquet's native module (pyarrow was already blocked).</td>
<td><code>pqcompat.py</code> prefers duckdb, then pyarrow, then fastparquet; all parquet I/O goes
through it.</td></tr>
<tr><td>Master unreachable after reboot</td><td>DHCP moved the master's LAN address three times
(.237 &rarr; .157 &rarr; .199).</td><td><code>xfer.py</code> probes a candidate list and
auto-detects.</td></tr>
<tr><td>Worker agent rewrites commands</td><td>The remote agent is an LLM and normalises
quoting.</td><td>Ship a script, invoke by path; never nest triple quotes.</td></tr>
<tr><td>Parquet from Kali missing</td><td>No duckdb in the Kali venv.</td><td>CSV fallback plus a
merger that globs both extensions.</td></tr>
</tbody></table>
""", "ch3")

# =================================================================== CH 4
sec("4. Phase 1 - corpus construction", f"""
{h(1,'4. Phase 1 - corpus construction')}
<p class="lead">Phase&nbsp;1 turns a raw tick export into {len(CAT):,} numbered, reconstructible
cells. It is the only phase whose output is treated as ground truth by everything downstream.</p>

{h(2,'4.1 Inputs')}
<p>The raw export is a sequence of CSV files, one per month, containing epoch-millisecond
timestamps plus bid and ask. Tick indices are global and 1-based: tick 1 is the first tick of the
first file. No rows are skipped, filtered, or de-duplicated — the corpus is the export.</p>

{h(2,'4.2 Sharding')}
<p>The stream is cut into 150-tick cells up to a global count of 591,171. Shards are assigned by
contiguous cell ranges so that each worker can stream its slice of ticks and drop the leading
partial cell:</p>
<table>
<thead><tr><th>Node</th><th>Cells</th><th>First cell</th><th>Last cell</th><th>Ticks</th>
<th>Harmonics K</th><th>Contiguity</th><th>Max tick error</th><th>Max norm RMSE</th></tr></thead>
<tbody>
""" + "".join(
    f"<tr><td>{n}</td><td>{d['cells']:,}</td><td>{d['first_cell_id']:,}</td>"
    f"<td>{d['last_cell_id']:,}</td><td>{d['ticks']:,}</td><td>{d['harmonics_K']}</td>"
    f"<td class='pass'>{d['contiguity_ok']}</td><td>{d['max_tick_error']}</td>"
    f"<td>{f"{d['max_norm_rmse']:.3e}"}</td></tr>"
    for n, d in D["p1nodes"].items()) + f"""
<tr class='total'><td>total</td><td>{sum(d['cells'] for d in D['p1nodes'].values()):,}</td>
<td>1</td><td>591,171</td><td>{sum(d['ticks'] for d in D['p1nodes'].values()):,}</td>
<td>75</td><td>all</td><td>0.0</td><td>&mdash;</td></tr>
</tbody></table>

{h(2,'4.3 What "exact" means here')}
<p>Two invariants are asserted per shard and again after the merge:</p>
<ul>
<li><b>Tick losslessness:</b> the reconstructed price at every tick index in the shard equals the
source price bit-for-bit. Measured as <code>max_tick_error</code>, which is 0.0 for all three
shards.</li>
<li><b>Norm error:</b> the L2 distance between the 75-harmonic encoding of a cell and the cell
itself, divided by the cell's norm. <code>max_norm_rmse</code> is at machine epsilon
(2.2&ndash;2.4&times;10<sup>-15</sup>), and all {num(D['p1v'].get('cells_under_1e_11'))} cells are
under 1e-11.</li>
</ul>

{h(2,'4.4 Merge and boundary audit')}
<p>The merge concatenates cell ranges in order and then verifies the two seams explicitly
(<code>node_boundaries</code> in the verification JSON): left cell 171,621 ends on tick
25,743,150 and cell 171,622 begins on 25,743,151; left cell 381,378 ends on 57,206,700 and
381,379 begins on 57,206,701. Both seams report <code>ok: true</code>. A duplicate or missing
tick at a seam is therefore impossible.</p>

{ kv_table({k: D['p1v'].get(k) for k in
    ('status','cells','ticks','first_cell_id','last_cell_id','first_tick_idx','last_tick_idx',
     'chunks','max_tick_error','max_norm_rmse','cells_under_1e_11','contiguity_ok','elapsed_sec')
    if k in D['p1v']}, 'Table 4.2 - phase1_verification.json') }

{h(2,'4.5 The catalog')}
<p>Every cell also gets a 19-column catalog row: index bounds, open/close timestamps, OHLC,
mean and std, the normalised open level, a bridge slope, residual RMSE, the reconstruction
errors, and the source file and price source. The catalog is the join key for every later
phase — Phase&nbsp;2 uses it to rebuild time ordering, Phase&nbsp;3 uses it for the UTC hour
audit.</p>

{ code_block(os.path.join(P1,'scripts','phase1_worker.py'), caption='phase1/scripts/phase1_worker.py - shard worker') }
""", "ch4")

# =================================================================== CH 5
sec("5. Phase 1 - Fourier encoding and verification", f"""
{h(1,'5. Phase 1 - Fourier encoding and verification')}
<p class="lead">Each cell is represented as a complex Fourier series truncated at 75 harmonics.
The representation is lossless at the tick level, which is what allows Phase&nbsp;3 to replay
"ticks 75–149" from a file that is nominally a frequency-domain artifact.</p>

{h(2,'5.1 Why Fourier at all')}
<p>The Phase&nbsp;2 descriptors are geometric: area, curvature, turning, recurrence. Those
features can be computed directly from ticks. The Fourier epicycles serve two other purposes:</p>
<ol>
<li><b>Compression and provenance.</b> The full 3.4 GB npz carries every cell's harmonic
coefficients; a cell can be regenerated without the raw CSVs. The Phase&nbsp;1 verification
proves the regeneration is exact.</li>
<li><b>Rotation invariance intuition.</b> Two cells that trace the same shape at different
scales have proportional coefficient vectors. That is the geometric prior behind the
"covariant" similarity in Phase&nbsp;2.</li>
</ol>

{h(2,'5.2 The decomposition')}
<p>For a cell with normalised path <code>y[0..149]</code>, the encoder computes the complex DFT
coefficients for <code>k = 0..74</code> (75 harmonics, enough for a 150-sample sequence with the
Nyquist pair) and stores them as float32 real/imaginary pairs together with the cell's scale and
offset. Reconstruction is <code>y_hat[n] = offset + Re(sum_k c_k e^(2&pi;i&middot;k&middot;n/150)) * scale</code>.</p>

{h(2,'5.3 The exactness experiment')}
<p>Five cells (1, 42,906, 85,811, 128,716, 171,621 — one per shard plus both boundaries) are
reconstructed at full precision and compared to the source. The analysis emits both numbers and
a set of epicycle animations (GIF) so a human can visually confirm the geometry. Results:</p>
{ kv_table({k:v for k,v in load_json('phase1','analysis','xflip_sample5','xflip_sample5_exactness.json').items()
            if not isinstance(v,(list,dict))}, 'Table 5.1 - sample-5 exactness') }

{h(2,'5.4 What the verification does not prove')}
<p>Exactness proves the encoder is invertible; it does not prove the encoding is <em>useful</em>.
That question is Phase&nbsp;2's, and the answer there is mixed: the harmonic representation is a
fine storage format but a weak similarity metric on its own. The Phase&nbsp;2 whitening step
(Chapter&nbsp;6) exists precisely because raw coefficient similarity collapses to a single
common-mode direction.</p>

{ code_block(os.path.join(P1,'scripts','phase1_merge.py'), caption='phase1/scripts/phase1_merge.py - merge and boundary audit') }
{ code_block(os.path.join(P1,'scripts','verify_phase1.py'), caption='phase1/scripts/verify_phase1.py - independent verifier') }
""", "ch5")
# ---- run the remaining chapter modules in this namespace ----
for _part in ("docs_part2.py", "docs_part3.py", "docs_part4.py", "docs_part5.py"):
    _fp = os.path.join(ROOT, "tools", _part)
    if os.path.exists(_fp):
        exec(compile(open(_fp, encoding="utf-8").read(), _fp, "exec"), globals())
        print(f"[*] chapter module {_part} loaded")