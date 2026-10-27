# Chapters 9-11 (executed inside build_docs.py's namespace)

sec("9. Phase 2 - CZET v2 sweet-spot model", f"""
<h1>9. Phase 2 - CZET v2 sweet-spot model</h1>
<p class="lead">The vault ranks anchors by hand-built statistics. CZET v2 asks a different
question: given the whole sequence of quartile descriptors, can a learned model predict the
outbound continuation better than raw similarity?</p>

<h2>9.1 Architecture</h2>
<p>The model is an ensemble over three seeds of:</p>
<ul>
<li><b>CZET v2 backbone</b> - a small transformer over the four quartile token sequences, with a
learnable geometric anomaly bank (each bank entry is a learned reference shape, and the model
scores distance to it).</li>
<li><b>Masked quartile reconstruction</b> - pass 1 hides Q3/Q4 and reconstructs them from
Q1/Q2, forcing the encoder to represent the causal part richly without leaking the future.</li>
<li><b>Pair head</b> - consumes two anchor embeddings and predicts the forward lock correlation.</li>
<li><b>PLS path</b> - a 24-component partial least squares regression on the dense features,
fitted on train only.</li>
<li><b>Stack</b> - a two-input non-negative blend of the neural head and the PLS head.</li>
</ul>
<p>Total parameters: {num(D['czet'].get('params'))}. Training: {D['czet'].get('epochs',{}).get('pass1')}
epochs of masked reconstruction, then {D['czet'].get('epochs',{}).get('pass2')} epochs of
continuation ranking.</p>

<h2>9.2 Evaluation protocol</h2>
<p>The sweet-spot universe is {num(D['czet'].get('train_rows'))} training rows
({num(D['czet'].get('train_edges'))} edges) and {num(D['czet'].get('test_rows'))} held-out rows
({num(D['czet'].get('test_edges'))} edges). Splits are random at the row level; test edges are
those touching a test row. The PLS transform, stack weight and pair head are fitted on train
only, and R&sup2; is reported on held-out rows. This is a weak protocol by time-series standards
(random splits allow near-duplicate leakage), which is why the Phase&nbsp;3 replay is the
authoritative test.</p>

<h2>9.3 Results</h2>
{ kv_table({
    'Universe': D['czet'].get('universe'),
    'Spearman (pair score vs realised lock)': num(D['czet'].get('spearman'),4),
    'Spearman - observed cov baseline': num(D['czet'].get('spearman_observed_cov_baseline'),4),
    'R-squared (held out)': num(D['czet'].get('r2'),5),
    'R-squared - net head alone': num(D['czet'].get('r2_net_only'),5),
    'R-squared - PLS path alone': num(D['czet'].get('r2_pls_only'),5),
    'Stack weights (net / PLS)': f"{num(D['czet'].get('stack_weights',{}).get('net'),4)} / "
        f"{num(D['czet'].get('stack_weights',{}).get('pls'),4)}",
    'Pair-score vs lock Pearson': num(D['czet'].get('pair_score_vs_lock_pearson'),4),
    'Precision at top 10%': num(D['czet'].get('precision_at_10pct'),4),
    'Base rate (|lock| >= 0.65)': num(D['czet'].get('base_rate_lock_ge_065'),4),
    'Seed ensemble': str(D['czet'].get('seeds')),
}, 'Table 9.1 - czet_eval.json') }

<h2>9.4 How to read a Spearman of 0.51 with an R-squared of 0.028</h2>
<p>These are not contradictory. Spearman measures whether the <em>ordering</em> is right; R&sup2;
measures whether the <em>level</em> is right. The model orders anchors substantially better than
raw covariant similarity (0.511 vs 0.101) while explaining almost none of the variance in
magnitude. In practical terms: CZET can tell you which of two anchors is more likely to continue,
but not how much.</p>
<p>Two further facts temper the result. First, <code>r2_net_only</code> is negative (-0.013): the
neural head alone is worse than predicting the mean, and all of the positive R&sup2; comes from
the PLS path (stack weight 0.942 to PLS). Second, precision at the top decile is 4.34% against a
1.73% base rate - a 2.5x lift, not a gold mine. Both are consistent with the broader finding:
real but small conditional structure.</p>

<h2>9.5 What this means for Phase 3</h2>
<p>If the conditional structure were strong, Phase&nbsp;3 would show a positive portfolio once
matching is done correctly. Chapter&nbsp;14 shows it does not. The honest synthesis is that CZET
v2 succeeds as a ranking model inside a universe that was already selected for similarity, and
that the ranking skill does not survive conversion into a directional trade at the payouts
available.</p>
""", "ch9")

sec("10. Phase 2 - crawled operators, fractal test and null test", f"""
<h1>10. Phase 2 - crawled operators, fractal test and null test</h1>
<p class="lead">Three supporting studies check whether well-known complexity measures add
anything, whether 150-tick anomalies persist at a 1,800-tick horizon, and whether the vault beats
a block-permutation null.</p>

<h2>10.1 Crawled mathematical operators (item 03)</h2>
<p>Nine operators from the computational-topology, recurrence and information-theory literature
were computed for every cell over a 75-tick window, then correlated with the vault's verified
edge:</p>
<table>
<thead><tr><th style="width:26%">Operator</th><th style="width:16%">Median</th>
<th style="width:16%">Corr vs edge</th><th>Reading</th></tr></thead>
<tbody>
<tr><td><code>hvg_degree_entropy</code></td><td>3.19905</td><td>+0.0175</td><td>Horizontal-visibility-graph degree diversity: no signal.</td></tr>
<tr><td><code>hvg_entropy_gap</code></td><td>0.44416</td><td>+0.0175</td><td>Gap between observed and maximum entropy: no signal.</td></tr>
<tr><td><code>hvg_max_degree</code></td><td>37</td><td>-0.0221</td><td>Peak visibility: very slightly negative.</td></tr>
<tr><td><code>hh_inst_freq_mean</code></td><td>0.01801</td><td>-0.0404</td><td>Hodgkin-Huxley instantaneous frequency: weak negative.</td></tr>
<tr><td><code>hh_inst_freq_std</code></td><td>0.04849</td><td>-0.0199</td><td>Frequency variability: no signal.</td></tr>
<tr><td><code>levy_area_z</code></td><td>-0.00162</td><td>+0.0705</td><td>Strongest of the nine, still weak.</td></tr>
<tr><td><code>rqa_det</code></td><td>0.375</td><td>+0.0581</td><td>Recurrence determinism: weak positive.</td></tr>
<tr><td><code>rqa_lam</code></td><td>0.375</td><td>+0.0668</td><td>Laminarity: weak positive.</td></tr>
<tr><td><code>perm_entropy</code></td><td>0.95208</td><td>+0.0003</td><td>Permutation entropy: flat.</td></tr>
</tbody></table>
<p>Verdict: no operator clears |r| &gt; 0.10, so none is promoted into the tradeable feature set.
The manifest records all 591,171 values per operator (21.3 MB) for future use, plus the
per-operator association table reproduced above.</p>

<h2>10.2 Fractal / multi-timeframe consistency</h2>
<p>Each 150-tick anomaly is compared with the 1,800-tick (12-cell) shape containing it. Agreement
is measured both on the descriptor and on the path shape:</p>
<div class="two-col">
{ kv_table({k: num(v,4) if isinstance(v,float) else v
            for k,v in D['frac'].get('descriptor_agreement',{}).items()},
           'Descriptor agreement') }
{ kv_table({k: (num(v,4) if isinstance(v,float) else v) for k,v in D['frac'].items()
            if not isinstance(v,dict)}, 'Fractal summary') }
</div>
<p>Only {D['frac'].get('is_fractal_anomalies')} cells are classified as fractal anomalies
(agreement above the 90th percentile 0.5192), and their mean verified edge
({num(D['frac'].get('fractal_anomalies_mean_edge'),4)}) is barely above the non-fractal mean
({num(D['frac'].get('non_fractal_mean_edge'),4)}). Fractality is not a lever.</p>

<h2>10.3 The block-permutation null test</h2>
<p>To ask whether the vault's win rates could arise from luck, outbound returns are
block-permuted in 150-tick blocks ({D['null'].get('permutations')} permutations, preserving
within-block autocorrelation) and the same selection pipeline is re-run on each permutation:</p>
{ kv_table({
    'Permutations': D['null'].get('permutations'),
    'Block size (ticks)': D['null'].get('block_size_ticks'),
    'Top-N': D['null'].get('top'),
    'Observed top-300 mean win': num(D['null'].get('observed_top300_mean_win'),4),
    'Null top-300 median': num(D['null'].get('null_top300_mean_median'),4),
    'Null top-300 99th pct': num(D['null'].get('null_top300_mean_p99'),4),
    'p (selection-aware)': num(D['null'].get('p_selection_aware'),4),
    'Observed pooled win': num(D['null'].get('observed_pooled_win'),4),
    'Null pooled median': num(D['null'].get('null_pooled_median'),4),
    'p (pooled)': num(D['null'].get('p_pooled'),4),
    'Verdict': D['null'].get('verdict'),
}, 'Table 10.1 - null_test.json') }
<p>The null test passes, and this is the single most seductive number in the project. It is also
the number that Phase&nbsp;3 invalidates. The test permutes the <em>outcome</em> labels but keeps
the <code>max(protected, directional)</code> selection rule, so the null distribution it builds
also contains the max-of-two bias. Comparing an observed max-of-two against a null max-of-two
answers "is this selection procedure producing more than its own selection bias?" - a useful
question, but not "is there a tradeable edge?". The Phase&nbsp;3 replay answers the second
question.</p>

<h2>10.4 Coupled indicator confluence</h2>
<p>RSI and VWAP-band features were attached per cell (<code>rsi_end1</code>, <code>vwap_z1</code>,
<code>rsi_divergence_confluence</code>, <code>vwap_band_anomaly_score</code>). These are
diagnostic columns only; no confluence rule was promoted, because the vault's ranking already
absorbs them and the Phase&nbsp;3 auction showed no incremental value.</p>
""", "ch10")

sec("11. Phase 2 - verification, reports and node attestation", f"""
<h1>11. Phase 2 - verification, reports and node attestation</h1>
<p class="lead">Phase&nbsp;2 closed with a ten-check gatekeeper, an 11-sheet workbook, and two
independent machine audits. This chapter records exactly what was asserted.</p>

<h2>11.1 The ten gatekeeper checks</h2>
<table>
<thead><tr><th style="width:6%">#</th><th style="width:38%">Check</th><th style="width:10%">Result</th><th>Detail</th></tr></thead>
<tbody>
""" + "".join(
    f"<tr><td>{k}</td><td>{esc(v['name'])}</td><td class='{'pass' if v['passed'] else 'fail'}'>"
    f"{'PASS' if v['passed'] else 'FAIL'}</td><td><code>{esc(v['detail'])}</code></td></tr>"
    for k, v in D["gk2"].get("checks", {}).items()) + f"""
</tbody></table>
<p>All ten pass, and the phase-level flag <code>all_passed</code> is
<code>{D['gk2'].get('all_passed')}</code>. The gatekeeper reads parquet through
<code>pqcompat</code> so it runs on machines where fastparquet is blocked.</p>

<h2>11.2 The report workbooks</h2>
<p><code>phase2_anomaly_efficiency_report.xlsx</code> has 11 sheets:
Executive_Summary, Graph_Statistics, Vault_Leaderboard, Edge_Quality, Peer_Distribution,
Null_Test, CZET_Eval, Fractal_vs_Macro, Crawled_Operators, Indicator_Confluence and
Node_Attestation. The companion <code>phase2_anomaly_strategies.xlsx</code> holds 4 sheets:
Phase3_Ready_Strategies (893), Null_Verified (893), Ranked_by_Wilson_LB (102) and
Inbound_Covariant_Hubs (2,000).</p>

<h2>11.3 Node attestation</h2>
<p>Both workers re-read the shared artifacts from the transfer server and recomputed the
aggregates with their own code. The comparison is exact for every count, and the only expected
difference is formatting (CSV vs parquet). The attestation JSONs are committed at
<code>phase2/node_verify_840g7.json</code> and <code>phase2/node_verify_kali.json</code>.</p>

<h2>11.4 What "10/10 PASS" does and does not mean</h2>
<p>The gatekeeper verifies <em>consistency</em>: that the artifacts exist, that shapes and counts
match, that declared columns are present, that the bicoherence is non-degenerate, that the graph
is mirrored correctly, that the null test ran. It does not verify <em>profitability</em>, and it
was never designed to. That distinction is why Phase&nbsp;3 exists, and why this report does not
end at Chapter&nbsp;11.</p>
""", "ch11")
