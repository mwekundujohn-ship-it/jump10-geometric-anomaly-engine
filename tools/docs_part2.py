# Chapters 6-8 (executed inside build_docs.py's namespace)

sec("6. Phase 2 - quartile decomposition and descriptors", """
<h1>6. Phase 2 - quartile decomposition and descriptors</h1>
<p class="lead">Phase 2 asks whether two cells can be said to be <em>the same</em> in a way that
predicts the future. That requires a descriptor that is rich enough to distinguish shapes, robust
enough to survive noise, and free of the collinearity that makes every cell look like every other
cell.</p>

<h2>6.1 The zone lattice</h2>
<p>Each cell is cut at 0-25-50-75-100% and additionally at the Fibonacci tick marks
35, 57, 75, 93 and 118. The quartile cuts give the four Q-zones used by the descriptors; the Fib
cuts give a finer lattice used by the fractal and confluence checks. The 50% cut is special: it is
the decision boundary between "observable" and "forward".</p>

<h2>6.2 The dense descriptor (64 raw dimensions per quartile)</h2>
<p>For a quartile path <code>y</code> of length 37-38, <code>compute_quartile_brute_force_features</code>
builds a 64-dimensional vector in six families:</p>
<table>
<thead><tr><th style="width:16%">Family</th><th style="width:10%">Dims</th><th>Contents</th></tr></thead>
<tbody>
<tr><td>Rough-path</td><td>9</td><td>end displacement, It&ocirc;-style time-weighted
integral, area (L&eacute;vy area), quadratic variation, cubic variation, squared integral, second
time-weighted integral, lead-lag covariance.</td></tr>
<tr><td>Recurrence (RQA)</td><td>~14</td><td>determinism, laminarity, mean/max diagonal and
vertical line lengths, recurrence rate, entropy of line lengths.</td></tr>
<tr><td>Ordinal / permutation</td><td>~10</td><td>permutation entropy at orders 3-5, ordinal
pattern frequencies, forbidden-pattern share.</td></tr>
<tr><td>Scattering (wavelet-like)</td><td>~18</td><td>multi-scale energy ratios, band-power
shares, second-order scattering moments.</td></tr>
<tr><td>Fren&eacute;t / curvature</td><td>6</td><td>arc length, elastic energy, mean signed
curvature, max |curvature|, inflection rate, total turning.</td></tr>
<tr><td>Scale-free shape</td><td>7</td><td>sorted |dy| quantiles, Hurst-like exponent, jump
share, skew of increments, kurtosis.</td></tr>
</tbody></table>
<p>The concatenation of Q1 and Q2 dense descriptors is the <em>observable</em> feature set: 128
raw dimensions computable from the first 75 ticks alone. This matters later, in Chapter&nbsp;16,
where the live tester must reproduce it in real time.</p>

<h2>6.3 The 17-dimensional summary descriptor</h2>
<p>Each quartile also gets a compact 17-vector (moments, trend, curvature summary, extrema
positions, phase proxies). It exists because the full descriptor is expensive and because the
CZET model needs a fixed-width sequence. The 17-vector is what Phase&nbsp;2's gatekeeper check 02
verifies as "Q1..Q4 present".</p>

<h2>6.4 The 132-dimensional trigger vector</h2>
<p>The similarity engine does not use raw descriptors. It uses a 132-dimensional whitened vector
built from <code>[trigger_vec(62) | dense_q1(64) | dense_q2(64)]</code> plus scalars, then
whitened with a covariance-based transform that drops three constant columns, strips the top
principal component, and projects onto 132 components. The top component carries 70.3% of the
variance - a warning sign that was later confirmed as the core problem with the similarity
metric.</p>
<table class="kv">
<tr><th style="width:44%">Whitening statistic</th><th>Value</th></tr>
<tr><td>Input width</td><td>196</td></tr>
<tr><td>Constant columns dropped</td><td>3</td></tr>
<tr><td>Components kept</td><td>132</td></tr>
<tr><td>Variance share of PC1 (stripped)</td><td>70.306%</td></tr>
<tr><td>Variance share PC2..PC5 (kept)</td><td>5.185 / 3.129 / 2.425 / 1.683%</td></tr>
<tr><td>Random-pair |r| after whitening</td><td>0.1790 (gatekeeper check 04)</td></tr>
</table>

<h2>6.5 Collinear collapse and the bicoherence fix</h2>
<p>An earlier build collapsed: descriptors of unrelated cells were almost perfectly correlated
because a few high-variance channels dominated the inner product. Two repairs were applied:</p>
<ol>
<li><b>Per-column standardisation plus covariance whitening</b>, with the dominant common-mode
component removed. This is what turns "everything correlates with everything" into a usable
metric (random-pair |r| falls to 0.179).</li>
<li><b>True phase-triad bicoherence.</b> The original bicoherence column was degenerate
(std&nbsp;&asymp;&nbsp;0). It was replaced by the bispectral closure
<code>0.5(|cos(&phi;1+&phi;2-&phi;3)| + |sin(&phi;1+&phi;2-&phi;3)|)</code> computed from the
first three complex coefficients, which has std 0.0626 and is no longer a constant.</li>
</ol>
<p>Both fixes are asserted by gatekeeper check 04, and both are necessary: without the whitening
the graph is a single clique; without the bicoherence fix one entire descriptor column carries no
information.</p>
""", "ch6")

sec("7. Phase 2 - the relational covariant graph", f"""
<h1>7. Phase 2 - the relational covariant graph</h1>
<p class="lead">The graph is the heart of the project: {num(D['cov'].get('unique_undirected_edges'))}
undirected edges among {num(D['cov'].get('anomaly_hubs'))} hubs, discovered by comparing the
observable half of every hub to the observable half of every other cell.</p>

<h2>7.1 Hub selection</h2>
<p>A cell is a hub if its inbound observable descriptor has an unusual number of highly similar
partners. Concretely, for each inbound cell the scanner counts peers with |r| &ge; 0.85 and
computes an anomaly z-score against the peer-count distribution. The top 5,000 by z-score are
retained, then filtered to those with |r| &ge; 0.85, giving
<b>{num(D['cov'].get('anomaly_hubs'))}</b> hubs ({D['cov'].get('hub_rate_pct')}% of the corpus,
{num(D['cov'].get('cells_not_anomalous'))} cells not anomalous).</p>

<h2>7.2 Edge construction</h2>
<p>For each hub, every inbound cell with |r| in [0.85, 0.999] to the hub's observable half
becomes an edge. The sign of r is kept: positive edges are "same-shape" (continuation) relations,
negative edges are "mirror" relations. The symmetric relation is de-duplicated so that
(hub, peer) and (peer, hub) count once. The final graph has
<b>{num(D['cov'].get('unique_undirected_edges'))}</b> undirected edges.</p>

<h2>7.3 Graph statistics</h2>
<div class="two-col">
{ kv_table({
    'Cells scanned': num(D['cov'].get('cells_total')),
    'Hubs': num(D['cov'].get('anomaly_hubs')),
    'Hub rate': f"{D['cov'].get('hub_rate_pct')}%",
    'Undirected edges': num(D['cov'].get('unique_undirected_edges')),
    'Median peers per hub': num(D['cov'].get('peer_counts',{}).get('median')),
    '90th pct peers': num(D['cov'].get('peer_counts',{}).get('p90')),
    'Max peers': num(D['cov'].get('peer_counts',{}).get('max')),
    'Median mean |r|': num(D['cov'].get('covariant_similarity',{}).get('median_mean_abs_r'),4),
    '99th pct mean |r|': num(D['cov'].get('covariant_similarity',{}).get('p99_mean_abs_r'),4),
    'Max single-peer |r|': num(D['cov'].get('covariant_similarity',{}).get('max_single_peer_abs_r'),4),
}, 'Graph summary') }
{ kv_table({
    'Median mean lock corr': num(D['cov'].get('continuation_lock',{}).get('median_mean_lock_corr'),4),
    'Max mean lock corr': num(D['cov'].get('continuation_lock',{}).get('max_mean_lock_corr'),4),
    'Max frac exact lock': num(D['cov'].get('continuation_lock',{}).get('max_frac_exact_lock'),4),
    'Median protected win rate': num(D['cov'].get('edge_quality',{}).get('median_protected_ext_win_rate'),4),
    'Median directional win rate': num(D['cov'].get('edge_quality',{}).get('median_directional_win_rate'),4),
    'Hubs with protected win &ge; 60%': num(D['cov'].get('edge_quality',{}).get('hubs_protected_win_ge_60pct')),
    'Hubs with directional win &ge; 60%': num(D['cov'].get('edge_quality',{}).get('hubs_directional_win_ge_60pct')),
    'Hubs with z &ge; 3': num(D['cov'].get('edge_quality',{}).get('hubs_z_ge_3')),
    'Mean extremum tick': num(D['cov'].get('mean_ext_tick'),1),
    'Action split (BUY low / SELL high)': f"{num(D['cov'].get('actions',{}).get('BUY_PROTECTED_LOW'))} / {num(D['cov'].get('actions',{}).get('SELL_PROTECTED_HIGH'))}",
}, 'Second-half behaviour') }
</div>

<h2>7.4 Independent audits</h2>
<p>The graph was re-derived from the raw files on two other machines with independent code paths.
Both reproduced every aggregate exactly:</p>
<table>
<thead><tr><th>Audit</th><th>Result</th></tr></thead>
<tbody>
<tr><td>840 G7 (Windows)</td><td>{num(D['nv840'].get('db_rows'))} hubs / {num(D['nv840'].get('db_unique_anchors'))} unique anchors;
{num(D['nv840'].get('edge_rows'))} edge rows = {num(D['nv840'].get('edge_unique_undirected'))} unique undirected;
0 mirrored rows, 0 self-loops, 0 duplicates; vault {num(D['nv840'].get('vault_rows'))} rows
({num(D['nv840'].get('vault_ok_rows'))} with enough triggers, {num(D['nv840'].get('vault_thin_rows'))} thin);
vault edge range {num(D['nv840'].get('vault_min_edge'),4)}-{num(D['nv840'].get('vault_max_edge'),4)};
{num(D['nv840'].get('vault_rows_above_55pct'))} rows at or above 55%.</td></tr>
<tr><td>Kali (Linux)</td><td>{num(D['nvkali'].get('edge_rows'))} edges; {num(D['nvkali'].get('nodes_touched'))} nodes touched;
max degree {num(D['nvkali'].get('max_degree'))}; median degree {num(D['nvkali'].get('median_degree_nonzero'))};
cov min/median/max {num(D['nvkali'].get('cov_min'),3)} / {num(D['nvkali'].get('cov_median'),4)} / {num(D['nvkali'].get('cov_max'),4)};
median |lock| {num(D['nvkali'].get('lock_median_abs'),4)}; {num(D['nvkali'].get('lock_ge_065'))} edges with |lock| &ge; 0.65;
all {num(D['nvkali'].get('db_hubs'))} hubs appear in the edge list.</td></tr>
</tbody></table>

<h2>7.5 Interpretation</h2>
<p>The graph is real, and it is also nearly uninformative. Median similarity is 0.862 and the
99th percentile is 0.973, which means the metric compresses almost the entire corpus into a narrow
band just below the threshold. A hub with 60 median peers is not a rare configuration; it is
typical. The graph's apparent density is a property of the descriptor geometry, not of the market
repeating itself.</p>
<p>This is the first place where the project's central difficulty becomes visible, and it is worth
stating plainly: <b>high cosine similarity in this space is cheap.</b> A random outbound cell has
a 37% chance of being within 0.85-0.955 of at least one of the 893 vault anchors. Any strategy
selection that treats "matched an anchor" as evidence must therefore be compared against that
base rate, not against zero.</p>
""", "ch7")

sec("8. Phase 2 - the strategy vault and the selection artifact", f"""
<h1>8. Phase 2 - the strategy vault and the selection artifact</h1>
<p class="lead">This is the chapter that matters most. Phase&nbsp;2 produced a vault of
{len(VAULT):,} anchors with out-of-sample win rates up to
{num(float(VAULT['outbound_verified_edge'].max()),3)}. Phase&nbsp;3 could not reproduce them.
This chapter explains why, with the exact definitions and the exact numbers.</p>

<h2>8.1 What goes into the vault</h2>
<p>For each hub, the inbound peer set is re-evaluated on outbound cells. For every outbound cell
whose observable half matches the hub at |r| &ge; 0.85, three quantities are accumulated:</p>
<ul>
<li><code>outbound_50_100_lock_corr</code> - the mean correlation between the hub's forward
half and the peer's forward half (the "lock"): does the forward path continue the same way?</li>
<li><code>outbound_protected_win_rate</code> - the mean of <code>protected_held</code>, a binary
flag computed in Phase&nbsp;1 that is true when the cell's forward extremum stays on the
protected side.</li>
<li><code>outbound_directional_win_rate</code> - the share of peers whose forward 50-100%
quarter return has the sign implied by the hub's <code>recommended_action</code>.</li>
</ul>
<p>Then the vault stores:</p>
<p class="formula">verified_edge = max(protected_win_rate, directional_win_rate)</p>
<p>and keeps every hub with <code>verified_edge &ge; 0.55</code> and at least 10 outbound
triggers. That is the {len(VAULT):,}-row vault.</p>

<h2>8.2 The distribution already told the story</h2>
<div class="two-col">
{ kv_table({
    'Vault rows': f"{len(VAULT):,}",
    'outbound_directional_win_rate mean': num(float(VAULT.outbound_directional_win_rate.mean()),4),
    'outbound_directional_win_rate median': num(float(VAULT.outbound_directional_win_rate.median()),4),
    'outbound_directional_win_rate p90': num(float(VAULT.outbound_directional_win_rate.quantile(.9)),4),
    'outbound_protected_win_rate mean': num(float(VAULT.outbound_protected_win_rate.mean()),4),
    'outbound_protected_win_rate median': num(float(VAULT.outbound_protected_win_rate.median()),4),
    'outbound_verified_edge mean': num(float(VAULT.outbound_verified_edge.mean()),4),
    'outbound_verified_edge max': num(float(VAULT.outbound_verified_edge.max()),4),
}, 'Vault columns (all 893 rows)') }
{ svg_bars(['directional','protected','verified=max'], [
    float(VAULT.outbound_directional_win_rate.mean()),
    float(VAULT.outbound_protected_win_rate.mean()),
    float(VAULT.outbound_verified_edge.mean())],
    title='Mean of each vault column (893 rows)', fmt='{:.3f}', height=230) }
</div>
<p>Both components average 0.511 and 0.487 - fair coins. The maximum of two fair-coin estimates
averages 0.566 purely because of selection. Nothing in the market produced that; the
<code>max()</code> did.</p>

<h2>8.3 The bias, formally</h2>
<p>Let <code>X</code> and <code>Y</code> be two win-rate estimates, each with expectation
<code>p &asymp; 0.5</code> and standard error <code>s</code>. Then
<code>E[max(X,Y)] &ge; max(E[X],E[Y])</code> always, and for independent normals</p>
<p class="formula">E[max(X,Y)] &asymp; p + s/&radic;&pi;</p>
<p>With 20 outbound triggers, <code>s &asymp; 0.11</code>, so the expected maximum is
<code>0.5 + 0.06 = 0.56</code> - exactly the observed mean. The tail is worse: the maximum of
893 such draws has an expected value near <code>p + 3.2s &asymp; 0.85</code>, which is where the
reported 0.85-0.93 rows come from. The apparent edge is a predictable function of
<code>n_triggers</code>, not of the market.</p>

<h2>8.4 Choosing the tradeable subset does not rescue it</h2>
<p>Phase&nbsp;2's own report noticed that the top of the similarity distribution (|r| &ge; 0.96)
had "zero hubs at 60%", and Phase&nbsp;3 therefore restricted to the sweet spot
0.85 &le; |r| &le; 0.955 with 12-400 peers. That leaves 778 rows, and the same pattern persists:</p>
{ kv_table({
    'Sweet-spot rows': '778',
    'protected mean / median / p90': "0.487 / 0.489 / 0.625",
    'directional mean / median / p90': "0.511 / 0.500 / 0.667",
    'verified mean / median / p90': "0.566 / 0.556 / 0.700",
    'rows with protected &ge; 0.60': '119',
    'rows with directional &ge; 0.55': '289',
    'rows with directional &ge; 0.667': '38',
}, 'Sweet-spot subset, before orthogonal pruning') }
<p>Note what those numbers mean operationally: 289 of 778 anchors have a directional rate above
0.55, and 38 are above 0.667. Under a fair coin, selecting the top 37% of 778 estimates will
always produce numbers like this. The correct null is not "0.5 with no variance"; it is "0.5 with
the variance implied by the trigger counts". Phase&nbsp;2's permutation test (Chapter&nbsp;10)
was selection-aware for the top 300, and it did reject the null at p = 0.0025 - but it was
testing the <em>same</em> max-of-two statistic, so it inherited the same bias.</p>

<h2>8.5 Orthogonal pruning</h2>
<p>To stop the vault from counting the same anomaly many times, Phase&nbsp;3 suppresses any anchor
whose 132-dimensional vector correlates at |r| &ge; 0.82 with an already-kept anchor, in
descending order of inbound Wilson lower bound. The vault shrinks from 778 to
<b>{len(LB)}</b> anchors. The gatekeeper re-verifies that the maximum pairwise |r| among kept
anchors is 0.8199 - i.e. the pruning is real, not nominal.</p>

<h2>8.6 What the vault is actually good for</h2>
<p>The vault is a catalogue of recurring shapes with measured continuation statistics. It is a
legitimate description of the corpus. It is not, by itself, a trading signal, because its
selection procedure guarantees that some rows will look profitable. Phase&nbsp;3's job was to
tell the difference, and Chapter&nbsp;14 shows how it did.</p>
""", "ch8")
