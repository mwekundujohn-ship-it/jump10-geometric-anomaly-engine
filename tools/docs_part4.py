# Chapters 12-20, appendices and final assembly (executed inside build_docs.py)

sec("12. Phase 3 - the Fable sigma-survival framework", f"""
{h(1,'12. Phase 3 - the Fable sigma-survival framework')}
<p class="lead">Phase&nbsp;3 is a rules engine, not a model. It takes any anchor and asks one
question: at the payout actually available, is the <em>lower bound</em> of the hit rate enough to
clear the breakeven hurdle plus a 2% margin? Everything else is risk control around that decision.</p>

{h(2,'12.1 The four gates')}
<ol>
<li><b>Sigma gate.</b> A winsorised rolling sigma is computed over the trailing 1,800 ticks
(12 cells). Tick-to-tick absolute differences are capped at 6.984375x their median so that
Jump&nbsp;10's discrete jumps do not inflate the volatility estimate; the cap is fixed by the data
(global median |diff| = 1.1640625 points). The resulting bps sigma is binned into five quantiles of
the <em>inbound</em> distribution, giving fixed edges
<code>[0.1526, 0.1755, 0.1772, 0.1787, 0.1805, 0.1912]</code> bps. An anchor may only trade in
sigma bins where its inbound edge was positive.</li>
<li><b>EV gate.</b> For each contract family, <code>EV = WLB * R - 1</code>. The anchor may trade a
contract only if EV &gt; +0.02.</li>
<li><b>Hour gate.</b> A UTC hour is blocked if its outbound realised edge is negative. This is a
hindsight filter, reported explicitly as such in Chapter&nbsp;14.</li>
<li><b>Deduplication.</b> At most one trade per outbound cell, choosing the highest-EV candidate.
Without this, overlapping anchors multiply the same cell into dozens of "trades" and inflate every
statistic.</li>
</ol>

{h(2,'12.2 Position sizing')}
<p>Two trajectories are simulated from a $50 starting balance:</p>
<p class="formula">f* = (p_WLB * R - 1) / (R - 1)</p>
<p class="formula">stake = clip(0.25 * f* * balance, $0.50, 1% * balance)</p>
<p>and a fixed-$5 trajectory with a $400 session stop and a hard ruin clamp at zero. The Kelly
fraction is reported as zero whenever the portfolio's profit factor is below one, which is the
mathematically correct outcome and a useful check that the risk code is not self-deceiving.</p>

{h(2,'12.3 Monte Carlo')}
<p>Five thousand block-bootstrap runs resample the trade sequence in blocks, re-compound the
equity curve, and report the 95th/99th percentile worst drawdown, the median profit factor and the
probability that the final equity is positive. This answers "is the result robust to the order in
which the trades arrived?", which the chronological curve cannot.</p>

{h(2,'12.4 Why the framework is a fair test')}
<p>Because every choice that could flatter the result is either fixed in advance or reported. The
sigma edges come from inbound data only. The payouts are the brief's, not fitted. The friction
ladder is monotone and asserted. The dedup rule is strict. The hour filter is the one hindsight
component, and it is flagged as such. If the portfolio survives all of that it deserves attention;
if it does not, the framework has done its job.</p>
""", "ch12")

sec("13. Phase 3A - implementation and distributed run", f"""
{h(1,'13. Phase 3A - implementation and distributed run')}
<p class="lead">The engine is a single file that runs either as a shard worker or as the merger.
That design was chosen so that all three machines execute byte-identical logic.</p>

{h(2,'13.1 Inputs')}
<table>
<thead><tr><th style="width:26%">Artifact</th><th style="width:16%">Shape</th><th>Role</th></tr></thead>
<tbody>
<tr><td><code>phase3_inputs.npz : raw</code></td><td>(591,171, 150) float32</td>
<td>Every cell's full 150-tick midpoint path. Used for the forward replay and for causal
sigma.</td></tr>
<tr><td><code>phase3_inputs.npz : trigger_vec</code></td><td>(591,171, 132) float32</td>
<td>The whitened observable descriptor used for anchor matching.</td></tr>
<tr><td><code>phase3_vault.csv</code></td><td>893 rows</td>
<td>Vault with peer counts, similarity, actions and outbound statistics.</td></tr>
<tr><td><code>phase3_catalog_times.npy</code></td><td>(591,171,) int64</td>
<td>Cell open times in epoch ms, for the UTC hour audit.</td></tr>
</tbody></table>
<p>Total transfer: about 667 MB per worker, shipped in roughly 40 seconds each over the LAN.</p>

{h(2,'13.2 Sharding and determinism')}
<p>Shard <code>i</code> of <code>n</code> takes anchors <code>[i*len/n, (i+1)*len/n)</code> in the
order produced by orthogonal pruning, so the assignment is a pure function of the anchor list. Each
shard recomputes the sigma quantiles from the causal inbound set and prints them; all three printed
the same six edges, which is the strongest available evidence that no shard silently disagreed about
the data.</p>
<table>
<thead><tr><th>Shard</th><th>Node</th><th>Anchors</th><th>Trades</th><th>Runtime</th><th>Format</th></tr></thead>
<tbody>
""" + "".join(
    f"<tr><td>{i}</td><td>{n}</td><td>{num(D['p3diag'][i].get('strategies_audited'))}</td>"
    f"<td>{num(D['p3diag'][i].get('trades'))}</td><td>{D['p3diag'][i].get('secs')} s</td>"
    f"<td>{'parquet' if i < 2 else 'CSV (no duckdb)'}</td></tr>"
    for i, n in enumerate(("X-Flip 14 (master)", "HP 840 G7", "Kali Dell E5270"))) + f"""
<tr class='total'><td></td><td>total</td>
<td>{sum(D['p3diag'][i].get('strategies_audited',0) for i in range(3))}</td>
<td>{sum(D['p3diag'][i].get('trades',0) for i in range(3))}</td><td>&lt; 15 s</td><td></td></tr>
</tbody></table>

{h(2,'13.3 The per-cell replay')}
<p>For every matched outbound cell the engine does the following, in order:</p>
<ol>
<li>Computes the cell's own half-window standard deviation from ticks 0-74 and normalises the
forward path: <code>curve[t] = (raw[75+t] - raw[74]) / sigma_0_75</code>.</li>
<li>Multiplies by the match polarity (sign of the similarity) and by the trade direction calibrated
<em>only</em> from inbound matches with the same anchor. This is the step that makes the replay
causal: the direction is decided by data that existed before the outbound period.</li>
<li>Evaluates all contract outcomes on the aligned 75-tick curve: direction at tick 149, maximum
excursion for ONETOUCH, close displacement for BARRIER, minimum excursion for NOTOUCH.</li>
<li>Records the chosen mode/contract, the consensus EV, both excursions and the per-contract win
flag, so the ledger can be re-analysed without re-running the replay.</li>
</ol>

{h(2,'13.4 The ledger schema')}
<p>The merged equity curve is a 41-column parquet with one row per surviving trade: strategy and
cell identifiers, match similarity and polarity, sigma bin and UTC hour, the raw gross outcomes
(<code>gross_dir_149</code>, <code>gross_exc_sniper</code>, <code>gross_prot_fade</code>,
<code>max_excursion</code>, <code>min_excursion</code>, <code>notouch_win</code>), the chosen mode
and contract with its payout, the per-unit win and PnL, and the full equity/drawdown ladders at each
friction tier.</p>

{h(2,'13.5 Merge')}
<p>The merger globs <code>shards/p3_strategies_node*</code> and <code>shards/p3_ledger_node*</code>
in both parquet and CSV, concatenates, applies the EV gate, deduplicates by outbound cell, builds
the friction ladder, runs the Monte Carlo, writes the workbook and the live config, and finally
calls the whitening-bridge builder. It also writes <code>phase3_executive_summary.json</code>,
which is the single source of truth for every number in this report.</p>
""", "ch13")

_p3 = D["p3"]
_cum = _p3.get("cum_equity_sigma", {})
_blk = _p3.get("blocked_utc_hours", [])
_sig = _p3.get("sigma_quantile_edges_bps", [])


def _f(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "—"


_appr = LB[LB.best_cons_ev > 0.02] if len(LB) else LB
_ver = LB[LB.verified_profitable.astype(bool)] if "verified_profitable" in LB.columns else LB.iloc[:0]
_win_net = float(_p3.get("portfolio_win_rate_net03", float("nan")))
_pf = float(_p3.get("portfolio_profit_factor_net03", float("nan")))

_crows = []
if len(CONTRACT):
    for _, r in CONTRACT.iterrows():
        _crows.append([r["contract"], _f(r["payout"], 3), _f(r["breakeven_pct"], 2) + "%",
                       f"{int(r['strategies_chosen']):,}", _f(r["mean_cons_ev_pct"], 2),
                       _f(r["mean_realised_ev_pct"], 2), _f(r["mean_win_rate"], 4)])

_srows = []
if len(SIGMAP):
    for _, r in SIGMAP.iterrows():
        _srows.append([int(r["sigma_bin"]), r["range_bps"], f"{int(r['triggers']):,}",
                       _f(r["rise_fall_win"], 4), _f(r["rise_fall_ev_pct"], 2),
                       _f(r["onetouch_win"], 4), _f(r["onetouch_ev_pct"], 2),
                       _f(r["barrier_win"], 4), _f(r["barrier_ev_pct"], 2), r["gate"]])

_hrows = []
if len(HOURMAP):
    for _, r in HOURMAP.iterrows():
        _hrows.append([str(r["utc_hour"]), f"{int(r['trades']):,}", _f(r["dir_win"], 4),
                       _f(r["ev_pct"], 2), _f(r["units"], 2), r["action"]])

sec("14. Phase 3A - results", f"""
{h(1,'14. Phase 3A - results')}
<p class="lead">The portfolio does not survive. This chapter reports every number, including the
ones that look good, and states which assumption each depends on.</p>

{h(2,'14.1 Selection funnel')}
{ kv_table({
    'Vault rows': f"{len(VAULT):,}",
    'Inside the tradeable sweet spot (0.85-0.955 |r|, 12-400 peers)': '778',
    'After orthogonal pruning at |r| < 0.82': f"{len(LB):,}",
    'Clearing the +2% consensus-EV gate': f"{len(_appr):,}",
    'Verified profitable on the outbound replay': f"{len(_ver):,}",
    'Portfolio trades after one-per-cell dedup': f"{len(EQ):,}",
    'Distinct anchors represented in the portfolio': f"{EQ.strategy_id.nunique():,}",
}, 'Table 14.1 - from vault to portfolio') }
<p>Note the shape of that funnel. {len(_appr):,} of {len(LB):,} pruned anchors pass the EV gate -
over half - and only {len(_ver):,} of those are verified profitable on the replay. A gate that
admits half the population and then produces a losing portfolio is a gate that is not
discriminating; Chapter&nbsp;15 explains why the EV estimate feeding it is biased.</p>

{h(2,'14.2 Portfolio metrics')}
{ kv_table({
    'Trades': f"{len(EQ):,} across {EQ.strategy_id.nunique():,} anchors",
    'Win rate, net of 0.03 sigma': _f(_win_net, 4),
    'Wilson 95% lower bound': _f(_p3.get('portfolio_wilson_lb_net03'), 4),
    'Profit factor': _f(_pf, 4),
    'Sharpe (per-trade, annualised by trade count)': _f(_p3.get('sharpe_net03'), 3),
    'Sortino': _f(_p3.get('sortino_net03'), 3),
    'Calmar': _f(_p3.get('calmar_net03'), 3),
    'Max drawdown, net 0.03 (sigma units)': _f(_p3.get('max_dd_net03_sigma'), 2),
    'Cumulative equity, gross': _f(_cum.get('gross_00'), 2) + ' sigma',
    'Cumulative equity, net 0.03': _f(_cum.get('net_03'), 2) + ' sigma',
    'Cumulative equity, net 0.06': _f(_cum.get('net_06'), 2) + ' sigma',
    'Cumulative equity, net 0.10': _f(_cum.get('net_10'), 2) + ' sigma',
    'Monte Carlo 95th pct worst drawdown': _f(_p3.get('mc95_worst_dd_sigma'), 2),
    'Monte Carlo 99th pct worst drawdown': _f(_p3.get('mc99_worst_dd_sigma'), 2),
    'Monte Carlo P(final equity > 0)': _f(_p3.get('mc_prob_positive'), 4),
    'Monte Carlo median profit factor': _f(_p3.get('mc_pf_median'), 4),
    'Kelly fraction': _f(_p3.get('kelly_fraction'), 2),
    'Balance, fixed $5 from $50': _f(_p3.get('balance_fixed5_from_50'), 1),
    'Balance, quarter-Kelly from $50': _f(_p3.get('balance_quarter_kelly_from_50'), 0),
    'Blocked UTC hours': ', '.join(f'{int(x):02d}' for x in _blk),
    'Sigma quantile edges (bps)': ', '.join(_f(x, 4) for x in _sig),
}, 'Table 14.2 - Phase 3A portfolio, net of friction') }
<p>Two of those rows need a warning attached. The quarter-Kelly balance of
{_f(_p3.get('balance_quarter_kelly_from_50'), 0)} is an artefact: because the Kelly fraction is
zero the sizer degenerates to the minimum stake and the compounding path is driven by a handful of
early wins. The fixed-$5 balance of {_f(_p3.get('balance_fixed5_from_50'), 1)} from $50 is worse
than it looks as a "survival" statistic, because a $400 session stop and the ruin clamp both fired;
the end state is dominated by the stop, not by edge.</p>

{h(2,'14.3 The equity curve')}
{svg_line(EQ.eq_gross_00.tolist(), title='Cumulative equity, gross (sigma units), 4,078 trades',
          ylab='cumulative sigma', height=250)}
{svg_line(EQ.eq_net_03.tolist(), title='Cumulative equity, net of 0.03 sigma friction',
          ylab='cumulative sigma', height=250, color='#b03a2e')}
<p>The curve has the characteristic shape of a slightly negative-expectation process with fat
wins: a long slow bleed, two visible recoveries, and a terminal slope that is clearly down. The
gross curve already ends at {_f(_cum.get('gross_00'), 1)} sigma, so the negative result is not
created by friction - friction only makes it monotonically worse.</p>

{h(2,'14.4 Contract comparison, and the two accounting systems')}
{table(_crows, headers=['Contract', 'Payout', 'Breakeven', 'Anchors nominating it',
                        'Mean consensus EV %', 'Mean realised EV %', 'Contract win rate'],
       title='Table 14.3 - contract auction as reported by the engine', widths=[16, 9, 11, 14, 16, 16, 12])}
<p>ONETOUCH is nominated by {int(CONTRACT.set_index('contract').loc['ONETOUCH_075','strategies_chosen']) if len(CONTRACT) else 0}
anchors because its barrier (+0.75&sigma; within 75 ticks) is easy for a Jump-10 path to reach. The
column headed "contract win rate" is the engine's per-anchor mean, not the portfolio realisation;
the portfolio realisation has to be read from the ledger, and it is there that a serious
inconsistency appears.</p>
<p>Phase&nbsp;3 carries <b>two PnL accounting systems</b>, and they disagree in sign.</p>
<table>
<thead><tr><th style="width:34%">Ledger statistic</th><th style="width:22%">ONETOUCH_075</th>
<th style="width:22%">NOTOUCH_10</th><th>Reading</th></tr></thead>
<tbody>
<tr><td>Trades chosen</td><td>3,941</td><td>137</td><td>Deduplicated portfolio composition.</td></tr>
<tr><td>Contract win rate (<code>unit_win</code>)</td><td>0.7166</td><td>0.8321</td>
<td>Barrier-touch accounting against the asserted payout.</td></tr>
<tr><td>Mean payoff per unit stake (<code>unit_pnl</code>)</td><td>+0.5406</td><td>+0.2898</td>
<td>Implies +54% and +29% per trade. Not credible.</td></tr>
<tr><td>Sigma win rate (<code>gross_sigma &gt; 0</code>)</td><td>0.4550</td><td>0.4818</td>
<td>Whether the aligned forward path ends net positive.</td></tr>
<tr><td>Mean sigma PnL per trade (<code>gross_sigma</code>)</td><td>&minus;0.0215</td><td>&minus;0.0151</td>
<td>Essentially zero: no edge.</td></tr>
</tbody></table>
<p>The contradiction is not a coding bug; it is a pricing bug, and it is worth stating precisely.
A contract paying 2.15&times; that wins 71.66% of the time has an expected return of
<code>0.7166&times;2.15 &minus; 1 = +54%</code> <em>per trade</em>. No market prices that. The
resolution is that the payout table and the barrier geometry were specified independently: a
0.75&sigma; barrier touched within 75 ticks is a cheap contract in reality, paying perhaps
1.05-1.20&times;, not 2.15&times;. Feeding a generous payout into a wide barrier manufactures the
+54% figure, while the sigma-accounted ledger - which measures the forward path without reference
to the payout - shows a mean of &minus;0.02&sigma;, i.e. nothing.</p>
<p>Every headline metric in this report therefore uses the <b>sigma accounting</b>, because it is
the only one that is internally consistent and payout-independent. The contract accounting is
reported here precisely so that the discrepancy is on the record: it is the clearest available
evidence that the payout column of the Phase&nbsp;3 brief is an assumption, and a badly calibrated
one, rather than a quote.</p>

{h(2,'14.5 The sigma gate map')}
{table(_srows, headers=['Bin', 'Range (bps)', 'Triggers', 'R/F win', 'R/F EV %',
                        '1-Touch win', '1-Touch EV %', 'Barrier win', 'Barrier EV %', 'Gate'],
       title='Table 14.4 - Fable sigma gate, outbound replay', widths=[5, 15, 9, 9, 9, 10, 10, 10, 10, 8])}
<p>All five sigma bins are marked BLOCK in the shipped map, which is the correct conservative
outcome: no bin clears the gate on Rise/Fall, and the bins that clear it on One-Touch do so by
margins that do not survive the hour filter and the friction ladder in combination. The map is
still emitted, because a live system must recompute it on live data rather than inherit a
conclusion.</p>

{h(2,'14.7 Hour-of-day audit (hindsight, flagged)')}
{table(_hrows, headers=['UTC hour', 'Trades', 'Dir win', 'EV %', 'Units', 'Action'],
       title='Table 14.5 - hour filter as shipped (all 24 hours)', widths=[12, 14, 14, 14, 14, 12])}
<p>The hour filter is the one component of Phase&nbsp;3A that is explicitly not causal: hours were
evaluated on the same outbound period they filter. Its inclusion is defensible only as a
<em>diagnostic</em> - it shows that the portfolio's losses are concentrated in a minority of hours,
which is what a noise process looks like - and it is the reason the live configuration ships with
{len(_blk)} blocked hours rather than none. If a future run wanted a defensible version, the hour
table would have to be built on an expanding window that never sees the trade it filters.</p>

{h(2,'14.8 What would have to be true for this to be tradeable')}
<p>Three independent things, all of which are currently false:</p>
<ol>
<li><b>The matching must be selective.</b> It is not: 37% of arbitrary cells match at least one
anchor in the 0.85-0.955 band, so the trigger carries almost no information about <em>which</em>
anomaly is recurring.</li>
<li><b>The conditional distribution must shift materially.</b> It does not: realised directional
win rate is {_f(_win_net, 4)} against a 51.20% breakeven, and the recomputed from-scratch figure
on the same features is 48.8%.</li>
<li><b>The payout must be better than assumed.</b> It cannot be verified, and at the assumed
payouts even the <em>point estimates</em> are below breakeven for three of four contracts once
friction is charged.</li>
</ol>
""", "ch14")

sec("15. The definition diagnostic - where the edge came from", f"""
{h(1,'15. The definition diagnostic - where the edge came from')}
<p class="lead">This chapter is the intellectual centre of the report. It takes the Phase&nbsp;2
vault at face value, recomputes its headline statistic from the raw ticks with an explicit
definition, and shows that the difference between the two is a selection artefact rather than a
data error.</p>

{h(2,'15.1 The two definitions')}
<p><b>Phase&nbsp;2's definition.</b> For each anchor, compute two outbound statistics over its
matched outbound cells: the mean of the binary <code>protected_held</code> flag, and the share of
cells whose Q4 return has the sign of the anchor's recommended action. Then store</p>
<p class="formula">verified_edge = max(protected_win_rate, directional_win_rate)</p>
<p><b>A faithful recomputation.</b> For every anchor, replay the outbound cells and compute the
single directional outcome that the strategy would actually trade:</p>
<p class="formula">win = 1[ sign(q4_net_return) == trade_dir ]</p>
<p>with <code>trade_dir</code> determined only from inbound calibration, and no maximisation over
two statistics anywhere in the pipeline.</p>

{h(2,'15.2 The result')}
<p>The diagnostic replays {num(D['defdiag'].get('trades'))} trades across
{num(D['defdiag'].get('anchors'))} sweet-spot anchors - the same 778-anchor universe that
Phase&nbsp;2 used.</p>
{ kv_table({
    'Anchors': num(D['defdiag'].get('anchors')),
    'Outbound trades replayed': num(D['defdiag'].get('trades')),
    'Phase 2 Q4 statistic (recomputed exactly as defined)': _f(D['defdiag'].get('w_phase2'), 5),
    'Faithful directional outcome, action-signed': _f(D['defdiag'].get('w_close_action'), 5),
    'Faithful directional outcome, inbound-signed': _f(D['defdiag'].get('w_close_inbound'), 5),
    'Vault-reported directional column (mean)': _f(D['defdiag'].get('vault_reported'), 5),
    'Agreement between action-signed and inbound-signed': _f(D['defdiag'].get('agree'), 5),
}, 'Table 15.1 - phase3_definition_diagnostic.json') }
<p>The two faithful variants agree with each other to four decimal places (0.4884) and disagree
with the vault's own reported mean (0.5104). That is the whole story in one line: <b>the vault
column does not survive a from-scratch recomputation of the quantity it claims to report.</b> The
per-anchor correlation between the stored column and the recomputation is -0.049, i.e. none.</p>

{h(2,'15.3 Why the maximum of two statistics is biased')}
<p>Let <code>X</code> and <code>Y</code> be two estimates of quantities that are both
approximately fair coins, each with standard error <code>s</code>. Then</p>
<p class="formula">E[max(X,Y)] &ge; max(E[X],E[Y])</p>
<p>always, and for independent normals the bias is approximately <code>s/&radic;&pi;</code>. With
a median of 20 outbound triggers per anchor, <code>s &asymp; 0.11</code>, so the expected maximum
is about <code>0.5 + 0.06 = 0.56</code> — precisely the observed sweet-spot mean of 0.566.</p>
<p>The effect grows with the number of candidates. Taking the maximum over 893 anchors adds roughly
3.2 standard errors at the top of the distribution, which is where the reported 0.85-0.93 rows come
from. Those rows are not evidence of 90% edges; they are the predictable upper tail of
<code>max()</code> applied to 893 fair-coin estimates.</p>

{h(2,'15.4 The vault under its own definitions')}
{ kv_table({
    'outbound_directional_win_rate - mean': _f(float(VAULT.outbound_directional_win_rate.mean()), 4),
    'outbound_directional_win_rate - median': _f(float(VAULT.outbound_directional_win_rate.median()), 4),
    'outbound_protected_win_rate - mean': _f(float(VAULT.outbound_protected_win_rate.mean()), 4),
    'outbound_protected_win_rate - median': _f(float(VAULT.outbound_protected_win_rate.median()), 4),
    'outbound_verified_edge - mean': _f(float(VAULT.outbound_verified_edge.mean()), 4),
    'outbound_verified_edge - median': _f(float(VAULT.outbound_verified_edge.median()), 4),
    'outbound_verified_edge - max': _f(float(VAULT.outbound_verified_edge.max()), 4),
    'Rows where verified_edge equals the protected column': f"{int((VAULT.outbound_verified_edge <= VAULT.outbound_protected_win_rate + 1e-9).sum()):,}",
    'Rows where verified_edge equals the directional column': f"{int((VAULT.outbound_verified_edge <= VAULT.outbound_directional_win_rate + 1e-9).sum()):,}",
}, 'Table 15.2 - vault column statistics, all 893 rows') }
<p>Both component columns are indistinguishable from a fair coin at the mean and median. Only the
maximum departs, and it departs in exactly the direction and magnitude that the selection algebra
predicts.</p>

{h(2,'15.5 Was the null test wrong?')}
<p>No - it answered a narrower question than it appeared to. The block-permutation test
(Chapter&nbsp;10) compares the observed top-300 mean against the same pipeline run on permuted
returns. Because the permuted pipeline also applies <code>max()</code>, both sides carry the same
selection bias, and the test correctly concludes that the observed value sits above its own null
distribution (p = {_f(D['null'].get('p_selection_aware'), 4)}). What it cannot detect is that the
null distribution itself is centred near 0.62 rather than 0.50. A selection-aware test of a
selection-biased statistic is a consistency check, not an edge test.</p>

{h(2,'15.6 The general lesson')}
<p>Whenever a research pipeline computes several candidate success statistics per entity and
keeps the best, that entity's reported score is not an estimate of any of them. It is the maximum
of several noisy estimates, and its expectation grows with the noise. The fix is structural, not
statistical: choose the statistic <em>before</em> looking at the data, or report all of them with
a multiplicity correction. The vault as shipped did neither, which is why
{len(VAULT):,} rows carried a statistic whose top decile was manufactured by the aggregation rule
rather than by the market.</p>
""", "ch15")

sec("16. Phase 3B - the live Deriv forward tester", f"""
{h(1,'16. Phase 3B - the live Deriv forward tester')}
<p class="lead">Phase&nbsp;3B is a complete, self-checking live system: it subscribes to the
Deriv WebSocket tick stream, builds the same descriptors the backtest used, matches the approved
anchors, applies seven risk guards, and journals every decision to SQLite. It is parked in
paper-proposal mode because Deriv's WebSocket could not be reached from any cluster node during
this work.</p>

{h(2,'16.1 Why a bridge was needed at all')}
<p>The backtest matches anchors in a 132-dimensional whitened space. Whitening is a linear map
whose parameters were derived from the corpus - and, critically, those parameters were not
persisted when Phase&nbsp;2 was built. Two reconstructions were attempted:</p>
<table>
<thead><tr><th style="width:30%">Attempt</th><th style="width:22%">Method</th><th style="width:14%">Result</th><th>Reading</th></tr></thead>
<tbody>
<tr><td>Whitening bridge</td><td>Regress the 132-d whitened vector onto the 144-d raw Q1+Q2 block</td>
<td>R&sup2; 0.5357</td><td>Linear reconstruction retains only half the variance; the discarded
whitening components carried information that is not recoverable from raw features alone.</td></tr>
<tr><td>Direct raw-space matcher</td><td>Cosine similarity on the 128-d raw dense vector</td>
<td>R&sup2; 0.4622, recall 0.000</td><td>Raw-space cosine is degenerate for this purpose: its
correlation with the true whitened cosine is 0.061, so it finds a different set of matches
entirely.</td></tr>
<tr><td>Learned bridge (shipped)</td><td>MLP 128 &rarr; 512 &rarr; 512 &rarr; 132, cosine loss,
12 epochs</td><td>val cos 0.881</td><td>Nonlinear recovery of the whitened metric is good in
direction, imperfect in ranking.</td></tr>
</tbody></table>
{ kv_table({
    'Architecture': D['p3bridge'].get('bridge'),
    'Training rows': num(D['p3bridge'].get('n_fit')),
    'Validation cosine': _f(D['p3bridge'].get('val_cos'), 4),
    'Anchor count matched': num(D['p3bridge'].get('anchors')),
    'Outbound match recall': _f(D['p3bridge'].get('match_recall'), 4),
    'Outbound match precision': _f(D['p3bridge'].get('match_precision'), 4),
    'Match IoU': _f(D['p3bridge'].get('match_iou'), 4),
    'Polarity agreement': _f(D['p3bridge'].get('polarity_agreement'), 4),
}, 'Table 16.1 - phase3_whitening_bridge.json (the live-parity ceiling)') }
<p>Read the recall honestly: <b>the live system fires on about 24% of the triggers the backtest
would have fired</b>, and of the triggers it fires, 29% are ones the backtest would not have. The
polarity is always right, so direction is not the problem; coverage is. This is stated in the
gatekeeper as a passing check because the tester <em>reports</em> the ceiling rather than
pretending to remove it.</p>

{h(2,'16.2 The seven guards')}
<table>
<thead><tr><th style="width:22%">Guard</th><th>Behaviour</th></tr></thead>
<tbody>
<tr><td><code>SigmaGate</code></td><td>Blocks any trigger whose live sigma falls outside the
approved bins, using the same winsorised causal estimator as the backtest.</td></tr>
<tr><td><code>PayoutMonitor</code></td><td>Rejects any contract whose live payout is below the
configured floor (1.90x) or below the realised breakeven payout for that contract family.</td></tr>
<tr><td><code>LagMonitor</code></td><td>Measures tick-arrival latency and aborts a proposal if the
decision would be made on stale data; journals the latency distribution.</td></tr>
<tr><td><code>LiveCalibrator</code></td><td>Maintains a rolling per-contract win-rate estimate in
SQLite and shrinks the design-time prior toward it as live trades accumulate.</td></tr>
<tr><td><code>DrawdownGuard</code></td><td>Stops the session after a fixed loss budget, matching
the $400 stop used in the fixed-stake backtest.</td></tr>
<tr><td><code>KellyStakeSizer</code></td><td>Computes quarter-Kelly from the calibrated probability
and clamps to [minimum stake, 1% of balance]; returns zero when the edge is non-positive.</td></tr>
<tr><td><code>HourFilter</code></td><td>Blocks the 14 UTC hours identified as loss-concentrating,
loadable from the shipped config.</td></tr>
</tbody></table>
<p>Each guard writes its decision to the same journal row, so a rejected proposal is as auditable
as an accepted one. That property is what makes the tester useful even while it is blocked: it can
be run for days in proposal-only mode and its journal will show exactly which guard is binding.</p>

{h(2,'16.3 The journal schema')}
<p>SQLite database <code>fable_forward_cal.db</code>, table <code>forward_trades</code>: proposal
timestamp, anchor id, live cell id, matched similarity, polarity, chosen contract, live payout,
stake, sigma bin, UTC hour, every guard's verdict, and the realised outcome once the settlement
tick arrives. The file is created on first run and is excluded from the repository by
<code>.gitignore</code>.</p>

{h(2,'16.4 The connectivity investigation')}
<p>The tester's <code>--ping_test</code> mode opens the public Deriv endpoint, sends a
<code>ticks_history</code> request and prints the round-trip time. It failed identically from all
three machines:</p>
<table>
<thead><tr><th style="width:42%">Attempt</th><th style="width:20%">Result</th><th>Notes</th></tr></thead>
<tbody>
<tr><td><code>wss://ws.derivws.com/websockets/v3?app_id=&lt;account app_id&gt;</code></td>
<td class="fail">HTTP 520</td><td>Cloudflare "origin unreachable" on the WebSocket upgrade.</td></tr>
<tr><td>Same endpoint with public <code>app_id=1089</code></td><td class="fail">HTTP 520</td>
<td>Rules out credential or app-id gating.</td></tr>
<tr><td>From Kali (different OS, different resolver)</td><td class="fail">HTTP 520</td>
<td>Rules out a Windows TLS-stack or firewall cause.</td></tr>
<tr><td>From HP 840 G7 (different ISP path)</td><td class="fail">HTTP 520</td>
<td>Rules out a single upstream route.</td></tr>
<tr><td><code>ws.binaryws.com</code> legacy host, custom Origin/User-Agent</td>
<td class="fail">HTTP 520</td><td>Rules out host-specific or header-based filtering.</td></tr>
<tr><td>Plain <code>ws://</code> (no TLS)</td><td class="fail">HTTP 520</td><td>Not a TLS
handshake problem.</td></tr>
<tr><td><code>https://api.deriv.com</code> over HTTPS</td><td class="pass">200 OK</td>
<td>Confirms general internet access and DNS are healthy; only the WebSocket edge is affected.</td></tr>
</tbody></table>
<p>Verdict: a Cloudflare edge / origin fault at the provider, localised to the WebSocket endpoint,
for this region. It is not a configuration error on our side, and no amount of retry or header
tweaking fixed it. The tester is therefore shipped complete and unexercised against live payouts,
which is the honest state.</p>

{h(2,'16.5 Operating the tester when connectivity returns')}
<ol>
<li>Put the credentials in a file outside the repository and pass <code>--creds_file</code>
(the tester refuses to log the token and the gatekeeper scans artifacts for leaks).</li>
<li>Run <code>--self_check</code> first. It must print
<code>matcher=mlp_cosine vec_dim=132 sqlite=True anchors=893 blocked_hours=14</code>.</li>
<li>Run <code>--ping_test 20</code> and confirm round-trip times.</li>
<li>Run in proposal-only mode for at least several sessions. Watch the guard-verdict columns:
if <code>PayoutMonitor</code> rejects everything, the assumed payouts are wrong and the whole
Phase&nbsp;3A conclusion needs revisiting in dollar terms.</li>
<li>Only then consider <code>--execute_demo_orders</code>, and only on the demo account.</li>
</ol>
""", "ch16")

_gk = D["p3gk"] if isinstance(D["p3gk"], list) else []
_grows = "".join(
    f"<tr><td>{i}</td><td>{esc(c.get('check'))}</td>"
    f"<td class=\"{'pass' if c.get('pass') else 'fail'}\">{'PASS' if c.get('pass') else 'FAIL'}</td>"
    f"<td><code>{esc(c.get('detail'))}</code></td></tr>"
    for i, c in enumerate(_gk, 1))
_gpass = sum(1 for c in _gk if c.get("pass"))

sec("17. Phase 3 verification - the 18-check gatekeeper", f"""
{h(1,'17. Phase 3 verification - the 18-check gatekeeper')}
<p class="lead">{_gpass} of {len(_gk)} checks pass. Each one is a claim the report makes, turned
into an executable assertion over the shipped artifacts.</p>

{h(2,'17.1 Results')}
<table>
<thead><tr><th style="width:5%">#</th><th style="width:38%">Check</th><th style="width:9%">Result</th>
<th>Detail</th></tr></thead>
<tbody>{_grows}</tbody></table>

{h(2,'17.2 The checks that matter most')}
<p><b>Anchor orthogonality (4).</b> The maximum off-diagonal |r| among the kept anchors is 0.8199,
just under the 0.82 suppression threshold. This proves the pruning actually ran rather than being
declared.</p>
<p><b>Sigma causality (6).</b> The correlation between the trade-time sigma estimate and the
forward move is +0.0098. If the sigma estimator leaked the forward window this number would be
large; it is not, so the volatility gate is causal.</p>
<p><b>One trade per cell (7).</b> Zero duplicate outbound cells. Without this, 208 overlapping
anchors would produce tens of thousands of correlated "trades" and every statistic would be
inflated by a factor that depends on the overlap structure.</p>
<p><b>Friction monotonicity (8).</b> The four tiers decrease strictly
({_f(_cum.get('gross_00'), 1)} &rarr; {_f(_cum.get('net_03'), 1)} &rarr;
{_f(_cum.get('net_06'), 1)} &rarr; {_f(_cum.get('net_10'), 1)}). A non-monotone ladder would mean
the cost model has a sign error.</p>
<p><b>Live parity, honestly reported (12).</b> Recall 0.240, precision 0.295, IoU 0.152. The
gatekeeper passes this check because the artifact states the ceiling instead of hiding it - a
distinction worth making explicit in a report that elsewhere accuses the pipeline of
self-flattery.</p>
<p><b>No credentials (17).</b> The artifact scan finds no API token, and the check was written so
that its own source cannot trigger a false positive (the needle is split across two literals).</p>

{h(2,'17.3 What the gatekeeper is not')}
<p>It is a consistency and integrity verifier. It does not re-run the backtest, it does not
independently re-derive the vault, and it cannot tell you whether the payouts are real. Those
three gaps are filled - to the extent they can be - by the definition diagnostic
(Chapter&nbsp;15), the two independent node audits (Chapter&nbsp;7), and the live tester once
connectivity returns (Chapter&nbsp;16).</p>
""", "ch17")

sec("18. Reproducibility", f"""
{h(1,'18. Reproducibility')}
<p class="lead">Every number in this report is produced by a script in the repository, from
artifacts whose checksums are published in Appendix&nbsp;E. This chapter gives the exact
procedure, including the places where the environment fights back.</p>

{h(2,'18.1 Hardware and roles')}
<table>
<thead><tr><th style="width:22%">Node</th><th style="width:16%">Address</th><th style="width:16%">OS</th><th>Role</th></tr></thead>
<tbody>
<tr><td>X-Flip 14 (master)</td><td>192.168.150.199</td><td>Windows 11</td>
<td>Orchestration, Phase&nbsp;1 merge, Phase&nbsp;2 wiring, Phase&nbsp;3 shard 0 and merge,
documentation build.</td></tr>
<tr><td>HP 840 G7</td><td>192.168.150.126</td><td>Windows 11</td>
<td>Phase&nbsp;1 shard, Phase&nbsp;3 shard 1, independent node audit (Python 3.14).</td></tr>
<tr><td>Kali Dell E5270</td><td>192.168.150.166</td><td>Kali Linux</td>
<td>Phase&nbsp;1 shard, Phase&nbsp;3 shard 2 (CSV output - no duckdb), independent node audit.</td></tr>
</tbody></table>
<p>The master's LAN address is assigned by DHCP and moved three times during the project
(.237 &rarr; .157 &rarr; .199). All worker-side scripts take the master address as an argument
rather than hard-coding it.</p>

{h(2,'18.2 Transport')}
<p>A single-threaded TCP command server on port 8080 accepts newline-delimited jobs and returns
stdout/stderr with a 30-second hard timeout. Files are served over the same port: workers request
<code>/artifacts/phase2/...</code> and receive the artefact directly, which is why the 667&nbsp;MB
Phase&nbsp;3 input set can be distributed without a shared filesystem. Directory junctions on the
master (<code>out\\phase2</code>, <code>out\\phase3</code>) present local paths to remote workers
without copying.</p>
<p>Quoting is the fragile part. Multi-line shell commands containing nested quotes are routinely
mangled by the transport, so every non-trivial job is shipped as a <code>.py</code> file and
invoked by path. This convention is worth preserving in any successor system.</p>

{h(2,'18.3 The parquet problem')}
<p>Windows Application Control blocks PyArrow's native extensions and also blocks fastparquet's
<code>cencoding.pyd</code>. All parquet I/O therefore goes through <code>pqcompat.py</code>, which
tries duckdb, then pyarrow, then fastparquet, and exports both
<code>read_parquet</code>/<code>write_parquet</code> and a duckdb-backed
<code>write_parquet</code> fallback. Kali has no duckdb, so its shard writes CSV; the merger globs
both extensions. This is why every Phase&nbsp;3 script imports <code>pqcompat</code> and never
<code>pandas.read_parquet</code> directly.</p>

{h(2,'18.4 Determinism')}
<p>The Phase&nbsp;3 shard split is a pure function of the pruned anchor list, so any machine can
recompute which anchors belong to which shard. All three shards independently computed the sigma
quantile edges and printed the identical six values
({', '.join(_f(x, 4) for x in _sig)} bps). Torch training for the live bridge uses a fixed seed
and reports the validation cosine; the shipped weights are committed so the tester does not need
to retrain.</p>

{h(2,'18.5 Rebuilding the report')}
<pre class="code">python C:\\cluster\\tools\\build_docs.py</pre>
<p>The builder reads the parquet/CSV/JSON/XLSX artifacts, renders
<code>docs/PROJECT_DOCUMENTATION.html</code>, and prints the section list. Rendering to PDF is a
Chrome headless print:</p>
<pre class="code">chrome --headless --disable-gpu --no-pdf-header-footer ^
       --print-to-pdf=docs\\PROJECT_DOCUMENTATION.pdf docs\\PROJECT_DOCUMENTATION.html</pre>

{h(2,'18.6 Repository hygiene')}
<p>The published repository is private and excludes: the 3.4&nbsp;GB Fourier archive and every
other <code>.npz</code>/<code>.npy</code>/<code>.pt</code> artefact (checksummed in
<code>docs/DATA_MANIFEST.json</code> instead), the MT5 backtest clone, directory junctions, the
live SQLite journal, and both credential files. The Phase&nbsp;3 brief ships as a redacted copy
with zero token matches, verified by the gatekeeper.</p>
""", "ch18")

sec("19. Limitations", f"""
{h(1,'19. Limitations')}
<p class="lead">A negative result is only as trustworthy as its weakest assumption. These are the
weak ones, each with the direction in which it could bite.</p>

<h2>19.1 The payout assumption is unverified, and internally inconsistent</h2>
<p>Four payouts (1.953, 2.150, 2.427, 1.550) drive every dollar figure. Deriv returned HTTP&nbsp;520
on the WebSocket endpoint from all three nodes, so none of them could be checked against live
pricing. The ledger then exposed a contradiction that proves the table is wrong in at least one
cell: applying the ONETOUCH payout to the measured ONETOUCH win rate gives
<code>0.7166 &times; 2.15 &minus; 1 = +54%</code> per trade over 3,941 trades, while the
payout-independent sigma accounting of the same trades yields &minus;0.02&sigma; per trade. A real
2.15&times; contract cannot be won 71.7% of the time at a 0.75&sigma; barrier; the payout and the
barrier were specified independently. Chapter&nbsp;14.4 reports both number systems side by side
for this reason. Every conclusion that does not depend on the payouts is stated in sigma units;
the dollar figures should be read as illustrative only.</p>

<h2>19.2 The definition artefact is not a data bug</h2>
<p>The <code>max(protected, directional)</code> statistic can be recomputed exactly from the vault;
the recomputation reproduces the stored maximum almost perfectly. What fails is the inference from
"this column is high" to "this anchor is tradeable". Nothing in the pipeline was corrupted; the
aggregation rule was wrong. That distinction matters because it means a rebuild cannot be trusted
unless the aggregation rule itself is changed.</p>

<h2>19.3 The null test inherits the bias</h2>
<p>Because the permutation null re-applies the same <code>max()</code> selection, the p-value of
0.0025 tests consistency, not edge. A future test should permute the outcome and evaluate a
<em>pre-registered single statistic</em>, or apply a max-statistic correction with the null
distribution of the max.</p>

<h2>19.4 Similarity matching is barely selective</h2>
<p>37% of arbitrary outbound cells match at least one of the 893 anchors within the accepted band.
Any strategy that treats "triggered" as informative must be compared with that base rate. Phase 3
does not repair this; it merely stops double-counting the same cell.</p>

<h2>19.5 Live parity is 0.24</h2>
<p>The whitening parameters from Phase&nbsp;2 were never persisted, so the live matcher fires on
about a quarter of backtest triggers. The live system is therefore a <em>subset</em> sampler of a
strategy that is already unprofitable. Even if the subset were the best quarter, the backtest
provides no evidence that it is.</p>

<h2>19.6 The hour filter is hindsight</h2>
<p>Fourteen blocked hours were selected on the same period they filter. Removing them flatters the
portfolio. It is reported, not hidden, but it is not a defensible live rule as constructed.</p>

<h2>19.7 One instrument, one tick source, one era</h2>
<p>Everything is Jump&nbsp;10 midpoints over 1,028.5 days. There is no cross-instrument validation
and no post-2023 out-of-sample period. Jump&nbsp;10's volatility parameter is stationary by design,
which makes it the easiest possible case; a negative result here is strong evidence, but a positive
one would have generalised poorly.</p>

<h2>19.8 Random-split evaluation in CZET v2</h2>
<p>CZET v2 used random row splits, which allow near-duplicate leakage. Its Spearman 0.511 is
therefore an optimistic bound on ranking skill. It does not affect the Phase&nbsp;3 conclusion,
which uses a strictly forward replay.</p>

<h2>19.9 Barrier touches are modelled at tick resolution</h2>
<p>A barrier is deemed hit when the tick midpoint crosses it. In reality a touch option settles on
the quoted price stream including spread, so a modelled touch at +0.75&sigma; may not have been a
real touch. This biases ONETOUCH upward, which is the direction that flatters the result.</p>

<h2>19.10 Deduplication changes the strategy</h2>
<p>Keeping one trade per outbound cell is necessary for valid statistics, but it is not what a
live system would do: a live system would trade every trigger that passes the guards, and the
deduplicated portfolio is therefore an idealised, non-overlapping sample of that behaviour.</p>

<h2>19.11 Fixed-stake survival is dominated by the stop</h2>
<p>The $400 session stop and the ruin clamp both fired in the fixed-$5 simulation, so the terminal
balance of {_f(_p3.get('balance_fixed5_from_50'), 1)} measures the stop, not the edge. It should
not be quoted as a performance figure without that caveat.</p>

<h2>19.12 The 132-dimensional space is nearly collinear</h2>
<p>Random anchor pairs have a mean cosine near 0.86. Any threshold in that band is discriminating
on a narrow slice of a compressed distribution, which makes match counts sensitive to small
changes in the whitening. This is the mechanism behind 19.4 and 19.5, and it is the single
structural weakness the project did not overcome.</p>
""", "ch19")

sec("20. Conclusions and roadmap", f"""
{h(1,'20. Conclusions and roadmap')}
<p class="lead">What was learned, what was built, and what would have to change for the question
to be worth reopening.</p>

{h(2,'20.1 Conclusions')}
<ol>
<li><b>The infrastructure works and is reusable.</b> A three-node cluster processed 88.7&nbsp;million
ticks into 591,171 cells, built a Fourier encoding with exact reconstruction, produced a
241,703-edge relational graph, trained a transformer, ran a 398-anchor distributed backtest, and
verified 28 independent claims - all with reproducible artefacts and published checksums.</li>
<li><b>The Phase&nbsp;2 edge was an artefact of its own definition.</b> A from-scratch
recomputation of the same quantity on the same anchors yields 0.4884 where the vault reports a
mean of 0.5104 and a maximum of {_f(float(VAULT.outbound_verified_edge.max()), 3)}. The
correlation between the stored column and the recomputation is -0.049.</li>
<li><b>Real conditional structure exists but is small.</b> CZET v2 ranks continuation better than
raw similarity (Spearman 0.511 vs 0.101, precision 4.34% vs a 1.73% base rate) while explaining
almost no variance (R&sup2; 0.028). Ranking skill is not the same thing as a tradeable edge.</li>
<li><b>Nothing survives conversion to a trade.</b> Net win rate {_f(_win_net, 4)} against a 51.20%
breakeven, profit factor {_f(_pf, 4)}, cumulative equity
{_f(_cum.get('net_03'), 1)} sigma net, Monte Carlo P(positive)
{_f(_p3.get('mc_prob_positive'), 4)}.</li>
<li><b>The live system is complete but unexercised.</b> Seven guards, a SQLite calibration
journal, an honest 0.24 parity ceiling, and a provider-side WebSocket outage that no client-side
change could fix.</li>
</ol>

{h(2,'20.2 Recommendations')}
<ul>
<li><b>Do not deploy.</b> The recommendation is unambiguous and is encoded in the shipped config as
<code>portfolio_verdict = NEGATIVE_NET_OF_FRICTION</code>.</li>
<li><b>Do not re-run the same pipeline hoping for a different number.</b> The result is a property
of the definition, not of the run.</li>
<li><b>Keep the infrastructure.</b> The cluster, the Fourier encoding, the graph builder and the
gatekeeper pattern are instrument-independent and reusable for any tick series.</li>
<li><b>Fix the statistics before adding models.</b> A pre-registered single outcome, a
multiplicity-corrected selection, and a held-out period are prerequisites for any future vault.</li>
</ul>

{h(2,'20.3 Roadmap, in priority order')}
<table>
<thead><tr><th style="width:6%">#</th><th style="width:40%">Work item</th><th>Why it would change the answer</th></tr></thead>
<tbody>
<tr><td>1</td><td>Persist whitening parameters and rebuild the live matcher to parity</td>
<td>Raises live recall from 0.24 to ~1.0, making the tester a true out-of-sample test rather than a
subset sampler.</td></tr>
<tr><td>2</td><td>Verify live payouts via a working Deriv endpoint (alternate edge, proxy, or
regional endpoint)</td><td>Converts every sigma-unit conclusion into a dollar conclusion and
resolves the one assumption that could flip the sign.</td></tr>
<tr><td>3</td><td>Pre-register a single outcome statistic and rebuild the vault with no
<code>max()</code></td><td>Removes the selection artefact at the source; the resulting vault will
be much smaller and, for the first time, interpretable.</td></tr>
<tr><td>4</td><td>Add a genuinely held-out period (2024+) and a second instrument</td>
<td>Tests whether any residual conditional structure generalises; the current corpus cannot
answer that.</td></tr>
<tr><td>5</td><td>Replace cosine similarity with a discriminative metric trained on forward
outcomes</td><td>Attacks the collinearity directly: the current metric compresses 591k cells into
a narrow band, which is why matching is barely selective.</td></tr>
<tr><td>6</td><td>Model barrier settlement on the quote stream, not the midpoint</td>
<td>Removes a known upward bias on ONETOUCH, the only contract with a positive realised EV.</td></tr>
<tr><td>7</td><td>Replace the hindsight hour filter with an expanding-window estimate</td>
<td>Makes the live rule causal; currently it is a diagnostic masquerading as a filter.</td></tr>
<tr><td>8</td><td>Re-derive the payout table from the barrier geometry (or obtain live quotes)</td>
<td>Removes the +54%-per-trade inconsistency between the contract accounting and the
sigma accounting, and makes every dollar figure in this report meaningful.</td></tr>
</tbody></table>

{h(2,'20.4 Closing')}
<p>The programme asked a narrow question and answered it. The value is not in the answer - it is
negative - but in the audit trail: a specific, reproducible mechanism
(<code>max()</code> over two noisy win-rate estimates) that manufactures apparent edges, together
with the instruments needed to detect it. Any future vault built on the same pattern will fail the
same way, and this report gives the diagnostic that shows it.</p>
""", "ch20")
