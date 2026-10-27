# Appendices A-E and final HTML assembly (executed inside build_docs.py)

D['manifest'] = load_json('docs', 'DATA_MANIFEST.json')

_ck_rows = [["<code>" + esc(r["file"]) + "</code>", f"{int(r['bytes']):,}",
             f"{r['mb']:,.2f}", "<code>" + r["sha256"] + "</code>"] for r in D.get("manifest", [])]

sec("Appendix A - source listings", """
""" + h(1, "Appendix A - source listings", "appx-a") + """
<p class="lead">The complete source of every script that produced a number in this report. The
listings are reproduced verbatim from the repository at build time, so they cannot drift from the
artefacts.</p>

<h2>A.1 Phase 1 - corpus construction</h2>
""" + code_block(os.path.join(P1, "scripts", "phase1_worker.py"),
                caption="phase1/scripts/phase1_worker.py - per-node 150-tick cell builder and Fourier encoder") + """
<p>The worker is the only place where the 150-tick cell boundary is defined. It reads raw ticks,
validates continuity, writes the cell record and computes the 75-harmonic epicycle encoding. Every
downstream phase depends on the invariant it establishes: cell <code>n+1</code> starts exactly one
tick after cell <code>n</code> ends.</p>
""" + code_block(os.path.join(P1, "scripts", "phase1_merge.py"),
                caption="phase1/scripts/phase1_merge.py - node merge, seam audit, catalog assembly") + """
<p>The merger is also the seam auditor. The two internal node boundaries (cells 171,621|171,622 and
381,378|381,379) are checked for tick continuity explicitly, because a corrupted seam would
silently create a fake 150-tick "anomaly" at two points in the corpus and nothing downstream would
notice.</p>
""" + code_block(os.path.join(P1, "scripts", "verify_phase1.py"),
                caption="phase1/scripts/verify_phase1.py - independent reconstruction verifier") + """

<h2>A.2 Phase 2 - relational graph, vault and models</h2>
""" + code_block(os.path.join(P2, "pqcompat.py"),
                caption="phase2/pqcompat.py - parquet shim (duckdb -> pyarrow -> fastparquet)") + """
<p>This 143-line file is why the project runs at all on the master machine. Application Control
blocks PyArrow's native modules and fastparquet's <code>cencoding.pyd</code>; duckdb is the only
working engine there, and it is absent on Kali. Every other script imports this shim instead of
pandas' parquet readers.</p>
""" + code_block(os.path.join(P2, "phase2_step1_zone_and_covariant_db.py"),
                caption="phase2/phase2_step1_zone_and_covariant_db.py - quartile cut, descriptor bank, hub selection") + """
<p>This is where the 64-dimensional dense descriptor is built and where the hub rule lives: an
inbound cell is a hub if its peer count at |r| &ge; 0.85 puts its anomaly z-score in the top 5,000
of the corpus.</p>
""" + code_block(os.path.join(P2, "phase2_dense_anomaly_bank.py"),
                caption="phase2/phase2_dense_anomaly_bank.py - the six descriptor families") + """
<p>Read this carefully in conjunction with Chapter&nbsp;15. The descriptor is a good description
of a path and a poor discriminator between paths: the six families overlap in what they measure,
which is the root of the collinearity documented in Chapter&nbsp;6.</p>
""" + code_block(os.path.join(P2, "phase2_step2_inbound_outbound_ai.py"),
                caption="phase2/phase2_step2_inbound_outbound_ai.py - outbound evaluation and vault construction") + """
<p>Line for line, this is the most consequential file in the project: the
<code>verified_edge = max(protected, directional)</code> assignment appears here, together with the
sweet-spot filter and the 10-trigger minimum.</p>
""" + code_block(os.path.join(P2, "fix_collapse_and_whiten.py"),
                caption="phase2/fix_collapse_and_whiten.py - covariance whitening that produced the 132-d space") + """
<p>The whitening transform built here was applied in memory and never written to disk. That single
omission is why the live tester cannot reproduce the backtest's matches (Chapter&nbsp;16).</p>
""" + code_block(os.path.join(P2, "phase2_czet_sweetspot.py"),
                caption="phase2/phase2_czet_sweetspot.py - CZET v2 architecture, masked reconstruction, PLS blend") + """
""" + code_block(os.path.join(P2, "phase2_czet_eval.py"),
                caption="phase2/phase2_czet_eval.py - held-out evaluation of the pair head") + """
""" + code_block(os.path.join(P2, "phase2_null_test.py"),
                caption="phase2/phase2_null_test.py - selection-aware block permutation test") + """
<p>Note what the permutation loop permutes: the outcome. Note what it does not permute: the
selection rule. This is the design decision discussed in Chapters&nbsp;10 and&nbsp;15.</p>
""" + code_block(os.path.join(P2, "phase2_fractal_fix.py"),
                caption="phase2/phase2_fractal_fix.py - multi-timeframe consistency test") + """
""" + code_block(os.path.join(P2, "phase2_crawled_operators.py"),
                caption="phase2/phase2_crawled_operators.py - nine complexity operators") + """
""" + code_block(os.path.join(P2, "phase2_merge_db.py"),
                caption="phase2/phase2_merge_db.py - statement database assembly") + """
""" + code_block(os.path.join(P2, "phase2_report.py"),
                caption="phase2/phase2_report.py - eleven-sheet workbook writer") + """
""" + code_block(os.path.join(P2, "phase2_gatekeeper_verifier.py"),
                caption="phase2/phase2_gatekeeper_verifier.py - the ten Phase 2 checks") + """
""" + code_block(os.path.join(P2, "phase2_vault_enrich.py"),
                caption="phase2/phase2_vault_enrich.py - indicator confluence columns") + """
""" + code_block(os.path.join(P2, "node_export_840g7.py"),
                caption="phase2/node_export_840g7.py - HP 840 G7 independent audit") + """
""" + code_block(os.path.join(P2, "node_verify_kali.py"),
                caption="phase2/node_verify_kali.py - Kali independent audit") + """

<h2>A.3 Phase 3 - replay engine, tester, verifier</h2>
""" + code_block(os.path.join(P3, "phase3_master_engine.py"),
                caption="phase3/phase3_master_engine.py - shard/merge engine, contract auction, friction ladder, Monte Carlo") + """
<p>The engine is 651 lines and runs in two modes from one file. The contract auction, the
deduplication pass, the friction ladder and the Monte Carlo all live in the merge path, which is
why a shard can be re-run without invalidating the portfolio statistics.</p>
""" + code_block(os.path.join(P3, "phase3_definition_diagnostic.py"),
                caption="phase3/phase3_definition_diagnostic.py - the recomputation that removed the edge") + """
""" + code_block(os.path.join(P3, "phase3_gatekeeper_verifier.py"),
                caption="phase3/phase3_gatekeeper_verifier.py - the eighteen Phase 3 checks") + """
""" + code_block(os.path.join(P3, "build_whitening_bridge.py"),
                caption="phase3/build_whitening_bridge.py - linear whitening reconstruction") + """
""" + code_block(os.path.join(P3, "build_live_bridge.py"),
                caption="phase3/build_live_bridge.py - raw-space cosine matcher (rejected: recall 0.000)") + """
""" + code_block(os.path.join(P3, "train_live_bridge.py"),
                caption="phase3/train_live_bridge.py - MLP 128-512-512-132 cosine bridge (shipped)") + """
""" + code_block(os.path.join(P3, "phase3_deriv_live_forward_tester.py"),
                caption="phase3/phase3_deriv_live_forward_tester.py - seven guards, SQLite journal, upgrade probe") + """
<p>Read the guard classes in order: each one is a single predicate over a proposal, and each one
journals its verdict. The tester's value in its current blocked state is that a week of
proposal-only operation will show which guard binds first on live data.</p>
""", "appx-a")

_cat_cols = [str(c) for c in CAT.columns]
_cat_rows = "".join(f"<tr><td><code>{esc(c)}</code></td></tr>" for c in _cat_cols)
_lb_cols = [str(c) for c in LB.columns]
_eq_cols = [str(c) for c in EQ.columns]


def _three_col(cols):
    per = (len(cols) + 2) // 3
    chunks = [cols[i:i + per] for i in range(0, len(cols), per)]
    rows = []
    for r in range(per):
        cells = "".join(f"<td><code>{esc(ch[r])}</code></td>" if r < len(ch) else "<td></td>"
                        for ch in chunks)
        rows.append(f"<tr>{cells}</tr>")
    return ("<table><thead><tr><th>Column</th><th>Column</th><th>Column</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


sec("Appendix B - data dictionary", f"""
{h(1,'Appendix B - data dictionary','appx-b')}
<p class="lead">Every artifact the report reads, with its shape and the meaning of each column.</p>

<h2>B.1 Cell catalog - <code>jump10_phase1_cell_catalog.parquet</code></h2>
<p>{len(CAT):,} rows, one per cell, written by the Phase&nbsp;1 merger. It is the spine of the
entire project: every other artifact is keyed by <code>cell_index</code>.</p>
{_three_col(_cat_cols)}

<h2>B.2 Strategy ev leaderboard - <code>phase3_strategy_ev_leaderboard.parquet</code></h2>
<p>{len(LB):,} rows, one per pruned anchor. This is the object the Phase&nbsp;3 auction operates
on; the equity curve is one row per trade derived from it.</p>
{_three_col(_lb_cols)}

<h2>B.3 Deduplicated equity curve - <code>phase3_deduplicated_equity_curve.parquet</code></h2>
<p>{len(EQ):,} rows, one per trade, one trade per outbound cell. The <code>gross_*</code> columns
are sigma-unit outcomes before friction; the <code>pnl_*</code>/<code>eq_*</code>/<code>dd_*</code>
triples are the cumulative ladders at 0.00, 0.03, 0.06 and 0.10 sigma friction.</p>
{_three_col(_eq_cols)}

<h2>B.4 Vault - <code>phase2_strategy_vault.parquet</code> (893 rows)</h2>
{ kv_table({
    'hub_cell_id': 'Cell index of the anchor (inbound, before the 50% cut).',
    'best_peer_id / partner': 'Highest-similarity outbound partner.',
    'inbound_peer_count': 'How many inbound cells match this anchor at |r| >= 0.85.',
    'mean_0_50_similarity': 'Mean observable-half correlation across peers.',
    'recommended_action': 'BUY_PROTECTED_LOW or SELL_PROTECTED_HIGH, from the inbound distribution.',
    'out_trigger_count': 'Number of outbound cells that matched. Minimum 10.',
    'outbound_50_100_lock_corr': 'Mean forward-half correlation between anchor and peers.',
    'outbound_protected_win_rate': 'Mean of the binary protected_held flag over outbound peers.',
    'outbound_directional_win_rate': 'Share of outbound peers whose Q4 return matched the action sign.',
    'outbound_verified_edge': 'max(protected_win_rate, directional_win_rate) - the artefact.',
}, 'Vault columns') }

<h2>B.5 Other artifacts</h2>
<table>
<thead><tr><th style="width:40%">File</th><th>Contents</th></tr></thead>
<tbody>
<tr><td><code>phase3_inputs.npz</code></td><td><code>raw</code> (591171, 150) float32 midpoints;
<code>trigger_vec</code> (591171, 132) float32 whitened descriptors. 667 MB.</td></tr>
<tr><td><code>czet_train_cache.npz</code></td><td><code>Q_seq</code> (591171, 4, 72) quartile token
sequences for CZET. 658 MB.</td></tr>
<tr><td><code>jump10_phase2_subzones.npz</code></td><td>Computed descriptors: <code>trigger_vec</code>
(62), <code>dense_q1..h2</code> (64 raw each), <code>desc_Q1..Q4</code> (17 each),
<code>y_half1/2</code> (75 each), <code>y_q1..q4</code> (38 each), <code>protected_held</code>,
<code>q4_net_return</code>.</td></tr>
<tr><td><code>jump10_phase1_fourier_functions.npz</code></td><td>75-harmonic complex coefficients per
cell. 3.4 GB. Not shipped; checksummed.</td></tr>
<tr><td><code>phase3_live_bridge.pt</code></td><td>MLP weights for the live matcher (128-512-512-132).</td></tr>
<tr><td><code>phase3_whitening_transform.npz</code></td><td>Mean/scale arrays used by the bridge.</td></tr>
<tr><td><code>phase3_live_forward_config.json</code></td><td>208 approved anchor ids, 14 blocked
hours, sigma edges, payout floors, verdict.</td></tr>
<tr><td><code>fable_forward_cal.db</code></td><td>SQLite journal of live proposals (not shipped).</td></tr>
</tbody></table>
""", "appx-b")

sec("Appendix C - metric definitions", f"""
{h(1,'Appendix C - metric definitions','appx-c')}
<p class="lead">Every statistic in this report, defined exactly as the code computes it, so that a
reader can recompute any of them from the parquet.</p>

<h2>C.1 Win rates</h2>
<p class="formula">win_rate = (1/N) &sum; 1[unit_win_i]</p>
<p>For a directional contract <code>unit_win = sign(gross_dir_149) == trade_dir</code>. For
ONETOUCH <code>unit_win = max_excursion &ge; 0.75</code>. For NOTOUCH
<code>unit_win = min_excursion &gt; &minus;1.0</code>. For BARRIER
<code>unit_win = gross_dir_149 &ge; 0.35</code>.</p>

<h2>C.2 Wilson lower bound</h2>
<p class="formula">LB = (p&#770; + z&sup2;/2n &minus; z&radic;(p&#770;(1&minus;p&#770;)/n + z&sup2;/4n&sup2;)) / (1 + z&sup2;/n)</p>
<p>with <code>z = 1.959964</code>. Used everywhere a hit rate is gated, because it penalises small
samples automatically: an anchor with 12 hits needs a much higher point estimate to clear the same
gate as one with 400.</p>

<h2>C.3 Expected value and the gate</h2>
<p class="formula">EV = p_WLB &middot; R &minus; 1,  gate: EV &gt; 0.02</p>
<p>where <code>R</code> is the contract payout and <code>p_WLB</code> the Wilson bound of the
<em>inbound</em> hit rate for that contract family.</p>

<h2>C.4 Profit factor</h2>
<p class="formula">PF = &sum; pnl_i [pnl_i &gt; 0] / |&sum; pnl_i [pnl_i &lt; 0]|</p>
<p>Computed on the friction-tier PnL column, not on gross sigma.</p>

<h2>C.5 Sharpe, Sortino, Calmar</h2>
<p class="formula">Sharpe = mean(pnl) / std(pnl) &middot; &radic;N
Sortino = mean(pnl) / std(min(pnl, 0)) &middot; &radic;N
Calmar = total_return / |max_drawdown|</p>
<p>Per-trade series, annualised by trade count rather than by calendar time, because the corpus is
not uniformly sampled in wall-clock terms.</p>

<h2>C.6 Kelly and the stake clamp</h2>
<p class="formula">f* = (p_WLB &middot; R &minus; 1) / (R &minus; 1)
stake = clip(0.25 &middot; f* &middot; balance, 0.50, 0.01 &middot; balance)</p>
<p><code>f*</code> is set to zero when <code>p_WLB &middot; R &le; 1</code>, which is why the
portfolio reports a Kelly fraction of zero.</p>

<h2>C.7 Winsorised sigma</h2>
<p class="formula">&sigma; = 1.4826 &middot; median(|&Delta;mid| capped at 6.984375 &middot; median(|&Delta;mid|))</p>
<p>over the trailing 1,800 ticks, expressed in bps of price. The 1.4826 factor converts a median
absolute deviation to a normal-consistent standard deviation; the cap is set at six median absolute
tick moves so that Jump-10's discrete 10-point jumps do not dominate.</p>

<h2>C.8 Monte Carlo</h2>
<p>5,000 block-bootstrap runs with block length 50 trades, sampling with replacement from the
realised trade sequence, re-compounding equity from zero and reporting the 95th/99th percentile
worst drawdown, the median profit factor and P(final &gt; 0).</p>

<h2>C.9 Whitening</h2>
<p class="formula">z = &Lambda;<sup>&minus;1/2</sup> U<sup>T</sup> (x &minus; &mu;)</p>
<p>with <code>&mu;</code> the column means, <code>U</code> the eigenvectors of the covariance and
<code>&Lambda;</code> the eigenvalues, keeping the 132 components after dropping three constant
columns and removing the first (70.3% variance) component.</p>

<h2>C.10 Anchor orthogonality</h2>
<p>Two anchors are considered the same anomaly when <code>|cos(z_i, z_j)| &ge; 0.82</code> in the
132-d space. Suppression keeps the anchor with the higher inbound Wilson bound. The gatekeeper
asserts the surviving maximum is below the threshold.</p>
""", "appx-c")

sec("Appendix D - glossary", f"""
{h(1,'Appendix D - glossary','appx-d')}
<table>
<thead><tr><th style="width:24%">Term</th><th>Definition as used in this project</th></tr></thead>
<tbody>
<tr><td>Cell</td><td>150 consecutive ticks. The unit of everything: one cell, one possible trade.</td></tr>
<tr><td>Quartile (Q1-Q4)</td><td>37-38 tick sub-sequences of a cell. Q1+Q2 are observable; Q3+Q4 are forward.</td></tr>
<tr><td>Observable half</td><td>Ticks 0-74. The only data available at decision time.</td></tr>
<tr><td>Forward half</td><td>Ticks 75-149. Where every outcome is measured.</td></tr>
<tr><td>Hub</td><td>An inbound cell with an unusually large number of similar peers (top 5,000 by anomaly z-score).</td></tr>
<tr><td>Covariant edge</td><td>|r| &ge; 0.85 between the observable halves of two cells.</td></tr>
<tr><td>Peer count</td><td>Number of cells matching an anchor above the similarity threshold.</td></tr>
<tr><td>Trigger</td><td>An outbound cell that matches an anchor and therefore generates a candidate trade.</td></tr>
<tr><td>Lock correlation</td><td>Correlation between the forward halves of an anchor and a peer: does the future continue the same way?</td></tr>
<tr><td>Protected extremum</td><td>The forward path never breaks the far barrier; the quantity ONETOUCH/NOTOUCH monetise.</td></tr>
<tr><td>Sigma</td><td>Causal cell volatility in bps, from the winsorised MAD estimator over the trailing 1,800 ticks.</td></tr>
<tr><td>Contract auction</td><td>Choosing, per anchor, the contract family with the highest gated EV.</td></tr>
<tr><td>Friction tier</td><td>A cost level (0.00/0.03/0.06/0.10 sigma) subtracted from the realised excursion.</td></tr>
<tr><td>Deduplication</td><td>Keeping only the highest-EV candidate per outbound cell.</td></tr>
<tr><td>Selection artefact</td><td>An apparent edge produced by taking a maximum over noisy estimates rather than by the data.</td></tr>
<tr><td>Wilson bound</td><td>Lower confidence bound on a binomial rate, used as the gate input instead of the point estimate.</td></tr>
<tr><td>Live parity</td><td>Recall of the live matcher against the backtest's trigger set. Currently 0.240.</td></tr>
<tr><td>Gatekeeper</td><td>An executable checklist over shipped artefacts; 10 checks in Phase 2, 18 in Phase 3.</td></tr>
<tr><td>Fable</td><td>The name for the Phase 3 sigma-survival framework and its seven live guards.</td></tr>
</tbody></table>
""", "appx-d")

sec("Appendix E - artifact inventory", f"""
{h(1,'Appendix E - artifact inventory','appx-e')}
<p class="lead">SHA-256 checksums for every artifact too large to ship. A reader with the source
repository and these hashes can verify that a rebuilt artifact matches the one this report
describes.</p>
<table>
<thead><tr><th style="width:44%">File</th><th style="width:12%">Bytes</th>
<th style="width:10%">MB</th><th>SHA-256</th></tr></thead>
<tbody>
""" + "".join(
    f"<tr><td><code>{esc(r['file'])}</code></td><td>{int(r['bytes']):,}</td>"
    f"<td>{r['mb']:,.2f}</td><td class='hash'>{r['sha256']}</td></tr>"
    for r in D.get("manifest", [])) + f"""
</tbody></table>

<h2>E.2 Shipped in the repository</h2>
<table>
<thead><tr><th style="width:50%">Path</th><th>Role</th></tr></thead>
<tbody>
<tr><td><code>README.md</code></td><td>Top-level findings and reproduction commands.</td></tr>
<tr><td><code>docs/PHASE3_BRIEF_REDACTED.md</code></td><td>The Phase 3 specification, credentials stripped.</td></tr>
<tr><td><code>docs/DATA_MANIFEST.json</code></td><td>The checksum table above, machine-readable.</td></tr>
<tr><td><code>docs/PROJECT_DOCUMENTATION.pdf</code></td><td>This report.</td></tr>
<tr><td><code>tools/build_docs.py</code> + <code>docs_part2..5.py</code></td><td>The generator that produced this report.</td></tr>
<tr><td><code>phase1/scripts/*.py</code></td><td>Corpus construction and verification.</td></tr>
<tr><td><code>phase2/*.py</code></td><td>Graph, vault, CZET, audits, reports, gatekeeper.</td></tr>
<tr><td><code>phase3/*.py</code></td><td>Engine, tester, verifier, diagnostic, bridges.</td></tr>
<tr><td><code>phase3/*.json, *.xlsx, *.parquet</code></td><td>All numbers quoted in this report.</td></tr>
</tbody></table>

<h2>E.3 Build provenance</h2>
<p>This document was generated on {BUILD} from the working tree at commit
<code>b71de45</code> (Phase&nbsp;3) plus the uncommitted documentation files. It contains
{len(SECTIONS) + 2} top-level sections and is rendered from HTML by Chrome headless print-to-PDF.</p>
""", "appx-e")

# ------------------------------------------------------------------ assemble
TOC = "".join(
    f"<li><a href='#{a}'>{esc(t)}</a></li>" for t, _b, a in SECTIONS if a)

CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: "Georgia","Times New Roman",serif; font-size: 10.2pt; line-height: 1.52;
       color: #1b1b1b; margin: 0; }
h1 { font-family: "Segoe UI",Helvetica,Arial,sans-serif; font-size: 19pt; color: #123a5c;
     margin: 0 0 10pt 0; padding-bottom: 6pt; border-bottom: 2.2pt solid #123a5c;
     page-break-before: always; page-break-after: avoid; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-family: "Segoe UI",Helvetica,Arial,sans-serif; font-size: 12.6pt; color: #1d4f78;
     margin: 15pt 0 5pt 0; page-break-after: avoid; }
h3 { font-family: "Segoe UI",Helvetica,Arial,sans-serif; font-size: 11pt; color: #24486b;
     margin: 11pt 0 4pt 0; page-break-after: avoid; }
p { margin: 0 0 7.5pt 0; text-align: justify; hyphens: auto; }
p.lead { font-size: 10.8pt; color: #2c2c2c; border-left: 3pt solid #9ec3dd;
         padding: 2pt 0 2pt 9pt; background: #f6fafd; }
ul, ol { margin: 0 0 8pt 0; padding-left: 18pt; }
li { margin-bottom: 3.5pt; text-align: justify; }
code { font-family: "Consolas","Courier New",monospace; font-size: 8.6pt;
       background: #f2f4f6; padding: 0.5pt 2pt; border-radius: 2pt; }
pre.code { font-family: "Consolas","Courier New",monospace; font-size: 7.15pt; line-height: 1.28;
       background: #fbfcfd; border: 0.6pt solid #d8dee4; border-left: 3pt solid #7fa8c4;
       padding: 6pt 7pt; white-space: pre-wrap; word-break: break-word;
       page-break-inside: auto; margin: 3pt 0 10pt 0; }
div.codelabel { font-family: "Segoe UI",sans-serif; font-size: 7.6pt; color: #33506b;
       font-weight: 600; margin-top: 9pt; page-break-after: avoid; }
span.muted { color: #7b8b99; font-weight: 400; }
table { width: 100%; border-collapse: collapse; margin: 6pt 0 11pt 0; font-size: 8.7pt;
        page-break-inside: avoid; }
caption { caption-side: top; text-align: left; font-family: "Segoe UI",sans-serif;
          font-size: 8.4pt; font-weight: 700; color: #123a5c; padding-bottom: 3pt; }
th { background: #e8eff5; color: #123a5c; font-family: "Segoe UI",sans-serif;
     font-size: 8.2pt; text-align: left; padding: 3.5pt 4pt; border: 0.5pt solid #c3d2de;
     vertical-align: bottom; }
td { padding: 3.2pt 4pt; border: 0.5pt solid #d8e0e7; vertical-align: top; }
tbody tr:nth-child(even) td { background: #fafcfd; }
table.kv th { width: 40%; background: #f0f5f9; font-weight: 600; font-family: "Segoe UI",sans-serif;
     font-size: 8.4pt; }
tr.total td { font-weight: 700; background: #eef4f8; }
td.pass, .pass { color: #14663a; font-weight: 700; }
td.fail, .fail { color: #9c2018; font-weight: 700; }
td.warn, .warn { color: #8a5a00; font-weight: 700; }
td.hash { font-family: "Consolas",monospace; font-size: 6.6pt; word-break: break-all; }
p.formula { font-family: "Cambria Math",Georgia,serif; font-size: 10.4pt; text-align: center;
            background: #f7f9fb; border-top: 0.5pt solid #dde5ec;
            border-bottom: 0.5pt solid #dde5ec; padding: 5pt; margin: 7pt 0 9pt 0; }
figure.chart { margin: 8pt 0 12pt 0; page-break-inside: avoid; }
.two-col { display: flex; gap: 12pt; align-items: flex-start; }
.two-col > * { flex: 1; }
div.note { font-family: "Segoe UI",sans-serif; font-size: 8pt; color: #6b7a88; margin: -6pt 0 9pt 0; }
/* cover */
.cover { page-break-after: always; padding-top: 26mm; text-align: left; }
.cover-rule { height: 5pt; background: #123a5c; width: 62mm; margin-bottom: 12mm; }
.cover-kicker { font-family: "Segoe UI",sans-serif; font-size: 9.6pt; letter-spacing: 2.4pt;
       text-transform: uppercase; color: #5d7689; }
h1.cover-title { font-size: 34pt; line-height: 1.08; border: none; padding: 0; margin: 4mm 0 5mm 0;
       page-break-before: avoid; color: #0f2f4b; }
.cover-sub { font-size: 12.2pt; color: #34536e; max-width: 148mm; line-height: 1.45; margin-bottom: 12mm; }
table.meta { font-size: 8.9pt; margin-bottom: 10mm; }
table.meta th { width: 27mm; background: #f2f6f9; font-family: "Segoe UI",sans-serif; }
.cover-badges { display: flex; gap: 6mm; margin-bottom: 9mm; }
.badge { flex: 1; border: 0.8pt solid #b9cbda; border-top: 3pt solid #123a5c; padding: 7pt;
       font-family: "Segoe UI",sans-serif; font-size: 8.6pt; font-weight: 700; color: #123a5c; }
.badge span { font-weight: 400; font-size: 8pt; color: #4a6a83; }
.cover-foot { font-size: 8pt; color: #6b7a88; border-top: 0.6pt solid #cfdae3; padding-top: 4pt; }
/* toc */
.toc { page-break-after: always; }
.toc h1 { page-break-before: avoid; }
.toc ol { list-style: none; padding-left: 0; font-family: "Segoe UI",sans-serif; font-size: 9.6pt; }
.toc li { padding: 3.1pt 0; border-bottom: 0.4pt dotted #c8d5df; }
.toc a { color: #1b3f5e; text-decoration: none; }
.toc-sub { font-size: 8.4pt; color: #6b7a88; margin-top: 8pt; }
"""

_sec_html = "".join(f"<section id=\"{_a}\">{b}</section>" for _t, b, _a in SECTIONS if _a) + "".join(f"<section>{b}</section>" for _t, b, _a in SECTIONS if not _a)
_HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Jump 10 Geometric Anomaly Engine - Project Documentation</title>
<style>{CSS}</style></head>
<body>
{COVER}
{ABSTRACT}
<div class="toc">
<h1>Contents</h1>
<ol>{TOC}</ol>
<div class="toc-sub">Generated {BUILD} &middot; {len(CAT):,} cells &middot; {len(SECTIONS) + 2}
sections &middot; every figure read from the repository artifacts at build time.</div>
</div>
{_sec_html}
</body></html>
"""

os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
with open(OUT_HTML, "w", encoding="utf-8") as fh:
    fh.write(_HTML)

_words = len(" ".join(t + " " + b for t, b, _a in SECTIONS).split())
print(f"[+] wrote {OUT_HTML}  ({len(_HTML)/1024:.0f} KB, {len(SECTIONS)+2} sections, "
      f"~{_words:,} words)")
print(f"[+] chapter order: " + " | ".join(t.split(' - ')[0] for t, _b, _a in SECTIONS))
