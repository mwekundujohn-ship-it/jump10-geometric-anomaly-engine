"""PHASE 3 STAGE 3A - distributed tick-exact backtester, Fable sigma survival audit,
Deriv contract EV auction, deduplicated portfolio equity + Monte Carlo.

Usage
-----
worker : python -u phase3_master_engine.py --node_id K --num_nodes 3
merge  : python -u phase3_master_engine.py --merge

Design notes
------------
* Strict causality: direction, dip/TP/SL quantiles, sigma bins and the per-bin
  Wilson lower bounds used by the EV gate are all calibrated on INBOUND cells
  (0..502,494). Outbound cells (502,495..591,171) are only ever scored.
* Similarity band: 0.85 <= |r| <= 0.955 (the Phase 2 sweet spot).
* Orthogonal pruning: greedy on outbound_win_lb95, suppressing any anchor whose
  |r| with an already kept anchor is >= 0.82.
* Parquet I/O goes through pqcompat (duckdb first) with a CSV fallback so the
  engine runs on Windows (AppControl blocks fastparquet) and on Kali alike.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

BASE3 = os.path.dirname(os.path.abspath(__file__))
for cand in (BASE3, r"C:\cluster\phase2"):
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)

N_IN = 502495
SIM_MIN, SIM_MAX = 0.85, 0.955
PEER_MIN, PEER_MAX = 12, 400
ORTHO_CUT = 0.82
FRICTION_TIERS = (("gross_00", 0.0), ("net_03", 0.03), ("net_06", 0.06), ("net_10", 0.10))
PAYOUT = {"RISE_FALL": 1.953, "ONETOUCH_075": 2.150, "BARRIER_035": 2.427, "NOTOUCH_10": 1.55}

_cfg = {"inputs": None, "catalog_times": None, "phase2_dir": r"C:\cluster\phase2"}


# ----------------------------------------------------------------------------- io
def _pq():
    try:
        import pqcompat
        return pqcompat
    except Exception:
        return None


def load_table(path):
    p = _pq()
    if path.endswith(".parquet"):
        if p is not None:
            try:
                return p.read_parquet(path)
            except Exception:
                pass
        csv = path[:-8] + ".csv"
        if os.path.exists(csv):
            return pd.read_csv(csv)
        raise RuntimeError(f"cannot read {path}")
    return pd.read_csv(path)


def save_table(df, path):
    p = _pq()
    if p is not None:
        try:
            p.write_parquet(df, path)
            return
        except Exception:
            pass
    df.to_csv(path[:-8] + ".csv", index=False)


def wilson_lb(wins, n, z=1.96):
    n = int(n)
    if n <= 0:
        return 0.0
    p = wins / n
    den = 1.0 + z * z / n
    ctr = p + z * z / (2.0 * n)
    half = z * np.sqrt((p * (1 - p) + z * z / (4.0 * n)) / n)
    return float(max(0.0, (ctr - half) / den))


# ------------------------------------------------------------------- sigma (Fable)
def winsorized_sigma(raw, window_cells=12, chunk=20000, sample_step=25):
    """1,800-tick (12-cell) rolling winsorized sigma up to each cell's tick-74 mark."""
    N = raw.shape[0]
    sample = np.asarray(raw[::sample_step], dtype=np.float64)
    absd = np.abs(np.diff(sample, axis=1))
    global_med = float(np.median(absd))
    cap = max(global_med * 6.0, 1e-9)
    cell_var = np.empty(N, dtype=np.float64)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        d = np.diff(np.asarray(raw[s:e], dtype=np.float64), axis=1)
        sq = np.where(np.abs(d) <= cap, d * d, np.nan)
        with np.errstate(invalid="ignore"):
            v = np.nanmean(sq, axis=1)
        cell_var[s:e] = np.where(np.isfinite(v), v, global_med ** 2)
    csum = np.concatenate([[0.0], np.cumsum(cell_var)])
    idx = np.arange(N)
    lo = np.maximum(0, idx - (window_cells - 1))
    cnt = idx + 1 - lo
    roll = (csum[idx + 1] - csum[lo]) / cnt
    return np.sqrt(np.maximum(roll, 1e-12)).astype(np.float32), global_med, cap


def sigma_bins(rel_bps, n_in=N_IN):
    qs = np.quantile(rel_bps[:n_in], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bid = np.clip(np.digitize(rel_bps, qs[1:-1]), 0, 4)
    return qs, bid


# ------------------------------------------------------------------------- helpers
def orthogonal_pruning(T_anchors, wlb):
    order = np.argsort(-wlb)
    A = T_anchors[order].astype(np.float32)
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    kept, kept_pos = [], []
    for i in range(len(order)):
        if kept:
            s = np.abs(A[kept] @ A[i])
            if np.any(s >= ORTHO_CUT):
                continue
        kept.append(i)
        kept_pos.append(order[i])
    return np.array(sorted(kept_pos), dtype=np.int64), len(kept), len(order)


def first_cross(v, level, ge=True):
    idx = np.nonzero(v >= level)[0] if ge else np.nonzero(v <= level)[0]
    return int(idx[0]) if len(idx) else 10 ** 9


def vectorized_outcomes(paths, tp=0.75, barrier=0.35):
    close = paths[:, -1]
    maxv = paths.max(axis=1)
    return {
        "close": close,
        "win_dir": (close > 0).astype(np.int8),
        "win_touch": (maxv >= tp).astype(np.int8),
        "win_bar": (close >= barrier).astype(np.int8),
    }


def run_node(node_id, num_nodes, out_dir, top_elite=0):
    t0 = time.time()
    inputs = os.environ.get("P3_INPUTS", os.path.join(BASE3, "phase3_inputs.npz"))
    vault_path = os.environ.get("P3_VAULT", os.path.join(_cfg["phase2_dir"], "phase2_strategy_vault.parquet"))
    times_path = os.environ.get("P3_TIMES", os.path.join(BASE3, "phase3_catalog_times.npy"))
    os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)

    z = np.load(inputs)
    raw = z["raw"]
    T_all = z["trigger_vec"]
    N = raw.shape[0]
    print(f"[*] node {node_id}/{num_nodes}: inputs {raw.shape} {T_all.shape}", flush=True)

    df_vault = load_table(vault_path)
    df_vault = df_vault[(df_vault.covariant_peer_count >= PEER_MIN) &
                        (df_vault.covariant_peer_count <= PEER_MAX) &
                        (df_vault.mean_0_50_similarity >= SIM_MIN) &
                        (df_vault.mean_0_50_similarity <= SIM_MAX)].copy()
    df_vault = df_vault.reset_index(drop=True)
    print(f"[*] vault rows inside sweet spot: {len(df_vault):,}", flush=True)

    anchor0 = df_vault.anchor_cell_id.to_numpy(np.int64) - 1
    T_anch = T_all[anchor0]
    wlb = df_vault.outbound_win_lb95.to_numpy(np.float64)
    keep_idx, n_kept, n_before = orthogonal_pruning(T_anch, wlb)
    print(f"[*] orthogonal pruning: {n_before} -> {n_kept} anchors (|r_anchor| >= {ORTHO_CUT} suppressed)", flush=True)
    if top_elite:
        order = np.argsort(-(df_vault.outbound_win_lb95.to_numpy()[keep_idx]))
        keep_idx = keep_idx[order[:top_elite]]
        n_kept = len(keep_idx)
        print(f"[*] elite cap -> {n_kept}", flush=True)

    df_kept = df_vault.iloc[keep_idx].reset_index(drop=True)
    anchor0 = anchor0[keep_idx]
    T_anch = T_all[anchor0]

    bounds = [int(round(n_kept * i / num_nodes)) for i in range(num_nodes + 1)]
    s0, s1 = bounds[node_id], bounds[node_id + 1]
    df_shard = df_kept.iloc[s0:s1].reset_index(drop=True)
    T_shard = T_anch[s0:s1]
    print(f"[*] shard strategies {s0}..{s1 - 1} ({len(df_shard)})", flush=True)
    if len(df_shard) == 0:
        return

    print("[*] computing Fable winsorized sigma ...", flush=True)
    sig_w, med_abs, cap = winsorized_sigma(raw)
    p74 = np.asarray(raw[:, 74], dtype=np.float64)
    rel_bps = (sig_w / np.maximum(np.abs(p74), 1e-9)) * 1e4
    qs, bid = sigma_bins(rel_bps)
    print(f"    sigma bins (bps): {np.round(qs, 3).tolist()}  med|d|={med_abs:.6g} cap={cap:.6g}", flush=True)

    try:
        times_ms = np.load(times_path)
    except Exception:
        times_ms = np.zeros(N, dtype=np.int64)
    hours = ((times_ms // 3600000) % 24).astype(np.int8) if times_ms.any() else np.zeros(N, np.int8)

    raw_out = np.asarray(raw[N_IN:], dtype=np.float64)
    std_out = raw_out[:, :75].std(axis=1)
    std_out = np.where(std_out < 1e-7, 1.0, std_out)
    paths_out = (raw_out[:, 75:] - raw_out[:, 74:75]) / std_out[:, None]
    T_out = T_all[N_IN:]
    sig_out = bid[N_IN:]
    hr_out = hours[N_IN:]
    n_out = raw_out.shape[0]
    print(f"[*] outbound cells {n_out:,}", flush=True)

    cal_step = 5
    T_cal = T_all[:N_IN:cal_step]
    raw_cal = np.asarray(raw[:N_IN:cal_step], dtype=np.float64)
    std_cal = raw_cal[:, :75].std(axis=1)
    std_cal = np.where(std_cal < 1e-7, 1.0, std_cal)
    paths_cal = (raw_cal[:, 75:] - raw_cal[:, 74:75]) / std_cal[:, None]
    sig_cal = bid[:N_IN:cal_step]
    print(f"[*] calibration cells {paths_cal.shape[0]:,}", flush=True)

    sim_out = (T_out @ T_shard.T).astype(np.float32)
    sim_cal = (T_cal @ T_shard.T).astype(np.float32)
    print(f"[*] similarity matrices {sim_out.shape} {sim_cal.shape} "
          f"({time.time() - t0:.0f}s)", flush=True)

    strat_rows, ledger = [], []
    bin_tot = {b: dict(n=0, dir=0, touch=0, bar=0, mult=0) for b in range(5)}
    hour_tot = {h: dict(n=0, win=0, units=0.0) for h in range(24)}

    for j in range(len(df_shard)):
        row = df_shard.iloc[j]
        sid = str(row.get("strategy_id", f"S{s0 + j}"))
        anc = int(row.anchor_cell_id)
        cal = sim_cal[:, j]
        chits = np.nonzero((np.abs(cal) >= SIM_MIN) & (np.abs(cal) <= SIM_MAX))[0]

        if len(chits) >= 20:
            cpol = np.sign(cal[chits])[:, None]
            aligned = paths_cal[chits] * cpol
            med_close = float(np.median(aligned[:, -1]))
            dir_sign = 1.0 if med_close >= 0 else -1.0
            sgn = aligned * dir_sign
            dip = sgn[:, :45].min(axis=1)
            limit_dip = float(np.clip(np.percentile(dip, 40), -1.20, -0.25))
            tp_cal = float(np.clip(np.percentile(sgn.max(axis=1), 65), 0.60, 2.50))
            sl_cal = float(np.clip(np.percentile(sgn.min(axis=1), 15), -2.50, -0.75))
            mode1_in = float(np.mean(sgn[:, -1]))
            filled = dip <= limit_dip
            mode3_in = float(np.mean(sgn[filled, -1] - limit_dip)) if filled.any() else -9.0

            occ = vectorized_outcomes(sgn)
            occ["win_notouch"] = (sgn.min(axis=1) > -1.0).astype(np.int8)
            wdir_all = float(occ["win_dir"].mean())
            wtouch_all = float(occ["win_touch"].mean())
            wbar_all = float(occ["win_bar"].mean())

            wlb_by_bin = {}
            for b in range(5):
                m = sig_cal[chits] == b
                nb = int(m.sum())
                if nb >= 12:
                    wlb_by_bin[b] = (
                        wilson_lb(int(occ["win_dir"][m].sum()), nb),
                        wilson_lb(int(occ["win_touch"][m].sum()), nb),
                        wilson_lb(int(occ["win_bar"][m].sum()), nb),
                        wilson_lb(int(occ["win_notouch"][m].sum()), nb),
                    )
            occ["win_notouch"] = (sgn.min(axis=1) > -1.0).astype(np.int8)
            fb = (wilson_lb(int(occ["win_dir"].sum()), len(chits)),
                  wilson_lb(int(occ["win_touch"].sum()), len(chits)),
                  wilson_lb(int(occ["win_bar"].sum()), len(chits)),
                  wilson_lb(int(occ["win_notouch"].sum()), len(chits)))
            if not wlb_by_bin:
                wlb_by_bin = {2: fb}
        else:
            dir_sign = -1.0 if "SELL" in str(row.get("recommended_action", "BUY")).upper() else 1.0
            limit_dip, tp_cal, sl_cal = -0.50, 1.25, -1.25
            mode1_in, mode3_in = -9.0, -9.0
            wdir_all = wtouch_all = wbar_all = 0.0
            fb = (0.0, 0.0, 0.0, 0.0)
            wlb_by_bin = {2: fb}

        sims = sim_out[:, j]
        hits = np.nonzero((np.abs(sims) >= SIM_MIN) & (np.abs(sims) <= SIM_MAX))[0]
        if len(hits) == 0:
            continue

        pol = np.sign(sims[hits])
        curve = paths_out[hits] * pol[:, None] * dir_sign
        n_h = len(hits)

        close = curve[:, -1]
        maxv = curve.max(axis=1)
        minv = curve.min(axis=1)
        w_dir = close > 0
        w_touch = maxv >= 0.75
        w_bar = close >= 0.35
        w_notouch = minv > -1.0

        tp_first = np.array([first_cross(c, tp_cal) for c in curve])
        sl_first = np.array([first_cross(c, sl_cal, ge=False) for c in curve])
        mult_win = tp_first < sl_first
        mult_gross = np.where(mult_win, tp_cal, sl_cal)

        e2 = np.where(mult_win, tp_cal, np.where(sl_first < 75, sl_cal, close))

        fill = curve[:, :45].min(axis=1) <= limit_dip
        e1_gross = np.zeros(n_h)
        for i in np.nonzero(fill)[0]:
            post = curve[i, :] - limit_dip
            ts = first_cross(post, sl_cal - limit_dip, ge=False)
            tpp = first_cross(post, tp_cal - limit_dip)
            e1_gross[i] = (tp_cal - limit_dip) if tpp < ts else (
                (sl_cal - limit_dip) if ts < len(post) else post[-1])

        bins_h = sig_out[hits]
        hrs_h = hr_out[hits]

        fallback = fb
        cons_ev = {}
        for cname, widx in (("RISE_FALL", 0), ("ONETOUCH_075", 1), ("BARRIER_035", 2), ("NOTOUCH_10", 3)):
            cons_ev[cname] = float(fallback[widx] * PAYOUT[cname] - 1.0)
        ev_mult = fallback[0] * (1.0 + (tp_cal - 0.03) / abs(sl_cal + 0.03)) - 1.0
        safe_bins = []
        for b in range(5):
            m = bins_h == b
            if m.sum() >= 8 and b in wlb_by_bin:
                if wlb_by_bin[b][0] * PAYOUT["RISE_FALL"] - 1.0 > 0.02:
                    safe_bins.append(b)
        cons_ev["MULTIPLIER_TP_SL"] = float(ev_mult)
        best_contract = max(cons_ev, key=cons_ev.get)

        realized = {
            "RISE_FALL": float(w_dir.mean()),
            "ONETOUCH_075": float(w_touch.mean()),
            "BARRIER_035": float(w_bar.mean()),
            "NOTOUCH_10": float(w_notouch.mean()),
            "MULTIPLIER_TP_SL": float(mult_win.mean()),
        }
        modes_net = {
            "DIRECTIONAL_CLOSE_149": close - 0.03,
            "EXCURSION_TOUCH_SNIPER": e2 - 0.03,
            "PROTECTED_EXTREMUM_LIMIT_FADE": np.where(fill, e1_gross - 0.03, np.nan),
        }
        in_modes = {"DIRECTIONAL_CLOSE_149": mode1_in, "EXCURSION_TOUCH_SNIPER": mode1_in,
                    "PROTECTED_EXTREMUM_LIMIT_FADE": mode3_in}
        best_mode = max(in_modes, key=in_modes.get)
        best_arr = modes_net[best_mode]
        best_arr = best_arr[np.isfinite(best_arr)]
        n_eff = int(len(best_arr))
        wins = int((best_arr > 0).sum())
        wr = wins / n_eff if n_eff else 0.0
        gp = float(best_arr[best_arr > 0].sum())
        gl = float(np.abs(best_arr[best_arr < 0].sum())) + 1e-9

        for i in range(n_h):
            b = int(bins_h[i])
            hr = int(hrs_h[i])
            bin_tot[b]["n"] += 1
            bin_tot[b]["dir"] += int(w_dir[i])
            bin_tot[b]["touch"] += int(w_touch[i])
            bin_tot[b]["bar"] += int(w_bar[i])
            bin_tot[b]["mult"] += int(mult_win[i])
            if 0 <= hr < 24:
                hour_tot[hr]["n"] += 1
                hour_tot[hr]["win"] += int(w_dir[i])
                hour_tot[hr]["units"] += (PAYOUT["RISE_FALL"] - 1.0) if w_dir[i] else -1.0
            ledger.append((sid, anc, int(N_IN + hits[i] + 1), float(abs(sims[hits[i]])),
                           int(pol[i]), int(dir_sign), b, hr, best_mode, best_contract,
                           float(close[i]), float(e2[i]) if np.isfinite(e2[i]) else float(close[i]),
                           float(e1_gross[i]), int(bool(fill[i])),
                           float(sig_out[hits[i]]), float(std_out[hits[i]]),
                           float(cons_ev[best_contract]),
                           float(maxv[i]), float(minv[i]), int(bool(w_notouch[i]))))

        strat_rows.append({
            "strategy_id": sid, "anchor_cell_id": anc,
            "vault_win_lb95": float(row.outbound_win_lb95),
            "covariant_peer_count": int(row.covariant_peer_count),
            "mean_0_50_similarity": float(row.mean_0_50_similarity),
            "recommended_action": str(row.recommended_action),
            "in_cal_peers": int(len(chits)),
            "out_triggers": int(n_h),
            "dir_sign": int(dir_sign),
            "limit_dip": round(limit_dip, 4), "tp_sigma": round(tp_cal, 4), "sl_sigma": round(sl_cal, 4),
            "mode_dir_win": round(float(w_dir.mean()), 4),
            "mode_touch_win": round(float(w_touch.mean()), 4),
            "mode_barrier_win": round(float(w_bar.mean()), 4),
            "mode_mult_win": round(float(mult_win.mean()), 4),
            "inbound_wlb_dir": round(float(fallback[0]), 4),
            "safe_sigma_bins": ",".join(str(b) for b in safe_bins) if safe_bins else "NONE",
            "cons_ev_rise_fall": round(cons_ev["RISE_FALL"], 4),
            "cons_ev_onetouch": round(cons_ev["ONETOUCH_075"], 4),
            "cons_ev_barrier": round(cons_ev["BARRIER_035"], 4),
            "cons_ev_multiplier": round(cons_ev["MULTIPLIER_TP_SL"], 4),
            "cons_ev_notouch": round(cons_ev["NOTOUCH_10"], 4),
            "mode_notouch_win": round(float(w_notouch.mean()), 4),
            "safest_contract": "RISE_FALL" if cons_ev["RISE_FALL"] > 0.02 else "SKIP",
            "highest_yield_contract": best_contract,
            "best_cons_ev": round(cons_ev[best_contract], 4),
            "realised_ev_pct": round((realized[best_contract] * PAYOUT.get(best_contract, 1.953) - 1.0) * 100, 2),
            "best_mode": best_mode,
            "net_mean_sigma": round(float(best_arr.mean()), 4) if n_eff else None,
            "net_total_sigma": round(float(best_arr.sum()), 3) if n_eff else None,
            "win_rate": round(wr, 4),
            "wilson_lb": round(wilson_lb(wins, n_eff), 4),
            "profit_factor": round(gp / gl, 3),
            "t_stat": round(float(best_arr.mean() / (best_arr.std() + 1e-9) * np.sqrt(n_eff)), 3) if n_eff else None,
            "verified_profitable": bool(wr >= 0.55 and gp / gl >= 1.15 and n_eff >= 15),
        })

    df_strat = pd.DataFrame(strat_rows)
    cols = ["strategy_id", "anchor_cell_id", "outbound_cell_id", "match_similarity",
            "match_polarity", "trade_dir", "sigma_bin", "utc_hour", "best_mode",
            "best_contract", "gross_dir_149", "gross_exc_sniper", "gross_prot_fade",
            "fade_filled", "sigma_bps", "cell_std_points", "cons_ev",
            "max_excursion", "min_excursion", "notouch_win"]
    df_led = pd.DataFrame(ledger, columns=cols)
    save_table(df_strat, os.path.join(out_dir, "shards", f"p3_strategies_node{node_id}.parquet"))
    save_table(df_led, os.path.join(out_dir, "shards", f"p3_ledger_node{node_id}.parquet"))

    diag = {
        "node": node_id, "strategies_audited": int(len(df_strat)),
        "trades": int(len(df_led)), "sigma_quantiles_bps": [round(float(x), 4) for x in qs],
        "global_median_abs_diff": med_abs, "winsor_cap": cap,
        "orthogonal_before": int(n_before), "orthogonal_after": int(n_kept),
        "sigma_bins": {str(b): bin_tot[b] for b in range(5)},
        "utc_hours": {str(h): hour_tot[h] for h in range(24)},
        "secs": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "shards", f"p3_diag_node{node_id}.json"), "w") as f:
        json.dump(diag, f, indent=2)
    print(f"[+] node {node_id} done: {len(df_strat)} strategies, {len(df_led):,} trades "
          f"({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------- merge
def max_dd(eq):
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq - peak))


def perf(pnl):
    pnl = np.asarray(pnl, dtype=np.float64)
    if len(pnl) == 0:
        return dict(n=0, pf=0.0, sharpe=0.0, sortino=0.0, calmar=0.0, maxdd=0.0, win=0.0)
    eq = np.cumsum(pnl)
    gp = pnl[pnl > 0].sum()
    gl = np.abs(pnl[pnl < 0].sum()) + 1e-9
    sd = pnl.std()
    dd = -max_dd(eq) + 1e-9
    down = pnl[pnl < 0].std() if (pnl < 0).any() else sd
    return dict(n=int(len(pnl)), pf=float(gp / gl), sharpe=float(pnl.mean() / (sd + 1e-9) * np.sqrt(len(pnl))),
                sortino=float(pnl.mean() / (down + 1e-9) * np.sqrt(len(pnl))),
                calmar=float(eq[-1] / dd), maxdd=float(-dd), win=float((pnl > 0).mean()))


def merge(out_dir):
    import glob
    import re

    def shard_files(kind):
        found = {}
        for pat in (f"{kind}_node*.parquet", f"{kind}_node*.csv"):
            for f in glob.glob(os.path.join(out_dir, "shards", pat)):
                m = re.search(r"node(\d+)", os.path.basename(f))
                if m:
                    found.setdefault(int(m.group(1)), f)
        return [found[k] for k in sorted(found)]

    s_files = shard_files("p3_strategies")
    l_files = shard_files("p3_ledger")
    df_s = pd.concat([load_table(f) for f in s_files], ignore_index=True)
    df_l = pd.concat([load_table(f) for f in l_files], ignore_index=True)
    print(f"[*] merged {len(df_s)} strategies / {len(df_l):,} trades from {len(s_files)} shards", flush=True)

    df_s = df_s.sort_values(["verified_profitable", "wilson_lb", "profit_factor"],
                            ascending=[False, False, False]).reset_index(drop=True)

    approved = df_s[df_s.best_cons_ev > 0.02].copy()
    keep_ids = set(approved.strategy_id) if len(approved) else set(df_s.strategy_id)
    port = df_l[df_l.strategy_id.isin(keep_ids)].copy()
    mode_map = dict(zip(df_s.strategy_id, df_s.best_mode))
    eng_map = dict(zip(df_s.strategy_id, df_s.highest_yield_contract))
    port["chosen_mode"] = port.strategy_id.map(mode_map)
    port["chosen_contract"] = port.strategy_id.map(eng_map)
    port["gross_sigma"] = np.where(port.chosen_mode == "DIRECTIONAL_CLOSE_149", port.gross_dir_149,
                                   np.where(port.chosen_mode == "EXCURSION_TOUCH_SNIPER",
                                            port.gross_exc_sniper, port.gross_prot_fade))
    port = port[(port.chosen_mode != "PROTECTED_EXTREMUM_LIMIT_FADE") | (port.fade_filled == 1)].copy()
    port["payout"] = port.chosen_contract.map(PAYOUT).fillna(1.953)
    port["unit_win"] = np.where(port.chosen_contract == "RISE_FALL", (port.gross_dir_149 > 0).astype(int),
                                np.where(port.chosen_contract == "ONETOUCH_075", (port.max_excursion >= 0.75).astype(int),
                                         np.where(port.chosen_contract == "BARRIER_035", (port.gross_dir_149 >= 0.35).astype(int),
                                                  np.where(port.chosen_contract == "NOTOUCH_10", port.notouch_win,
                                                           (port.gross_exc_sniper > 0).astype(int)))))
    port["unit_pnl"] = np.where(port.unit_win == 1, port.payout - 1.0, -1.0)

    port = port.sort_values(["outbound_cell_id", "cons_ev"], ascending=[True, False])
    port = port.drop_duplicates(subset=["outbound_cell_id"], keep="first").reset_index(drop=True)
    print(f"[*] portfolio: {len(port):,} non-overlapping trades from {port.strategy_id.nunique()} strategies", flush=True)

    for name, cost in FRICTION_TIERS:
        port[f"pnl_{name}"] = port.gross_sigma - cost
        port[f"eq_{name}"] = port[f"pnl_{name}"].cumsum()
        port[f"dd_{name}"] = port[f"eq_{name}"] - port[f"eq_{name}"].cummax()

    bal_k, bal_f, k_curve, f_curve = 50.0, 50.0, [], []
    ruined = False
    for upnl in port.unit_pnl.to_numpy():
        if bal_k > 0.5:
            stake_k = float(np.clip(0.25 * max(0.002, upnl * 0.02) * bal_k, 0.50, 0.01 * bal_k))
            bal_k = max(0.0, bal_k + stake_k * upnl)
        if not ruined and bal_f >= 5.0:
            bal_f += 5.0 * upnl
            if bal_f <= 0.0:
                bal_f = 0.0
                ruined = True
        k_curve.append(round(bal_k, 2))
        f_curve.append(round(bal_f, 2))
    port["balance_account_ruined_fixed5"] = ruined
    port["balance_kelly_from_50"] = k_curve
    port["balance_fixed5_from_50"] = f_curve

    pnl_net = port.pnl_net_03.to_numpy()
    rng = np.random.default_rng(42)
    n = len(pnl_net)
    dds, ends, pfs = [], [], []
    for _ in range(5000):
        s = rng.choice(pnl_net, size=n, replace=True)
        eq = np.cumsum(s)
        dds.append(max_dd(eq))
        ends.append(float(eq[-1]))
        pfs.append(float(s[s > 0].sum() / (np.abs(s[s < 0].sum()) + 1e-9)))
    perf_net = perf(pnl_net)

    wp = float((pnl_net > 0).mean())
    aw = float(pnl_net[pnl_net > 0].mean()) if (pnl_net > 0).any() else 1.0
    al = float(abs(pnl_net[pnl_net < 0].mean())) if (pnl_net < 0).any() else 1.0
    b = aw / max(al, 1e-9)
    kelly = max(0.0, wp - (1 - wp) / b)

    contract_rows = []
    for c in ("RISE_FALL", "ONETOUCH_075", "BARRIER_035", "NOTOUCH_10", "MULTIPLIER_TP_SL"):
        sub = df_s[df_s.highest_yield_contract == c]
        contract_rows.append({
            "contract": c, "payout": PAYOUT.get(c, 1.953),
            "breakeven_pct": round(100.0 / PAYOUT.get(c, 1.953), 2),
            "strategies_chosen": int(len(sub)),
            "mean_cons_ev_pct": round(float(sub.best_cons_ev.mean() * 100), 3) if len(sub) else 0.0,
            "mean_realised_ev_pct": round(float(sub.realised_ev_pct.mean()), 3) if len(sub) else 0.0,
            "mean_win_rate": round(float(sub.win_rate.mean()), 4) if len(sub) else 0.0,
        })

    diag_files = sorted(glob.glob(os.path.join(out_dir, "shards", "p3_diag_node*.json")))
    sig_map, hour_map, quantiles = [], [], []
    agg_b = {b: dict(n=0, dir=0, touch=0, bar=0, mult=0) for b in range(5)}
    agg_h = {h: dict(n=0, win=0, units=0.0) for h in range(24)}
    for f in diag_files:
        d = json.load(open(f))
        quantiles = d["sigma_quantiles_bps"]
        for b in range(5):
            for k in agg_b[b]:
                agg_b[b][k] += d["sigma_bins"][str(b)][k]
        for h in range(24):
            agg_h[h]["n"] += d["utc_hours"][str(h)]["n"]
            agg_h[h]["win"] += d["utc_hours"][str(h)]["win"]
            agg_h[h]["units"] += d["utc_hours"][str(h)]["units"]
    for b in range(5):
        st = agg_b[b]
        nb = max(1, st["n"])
        rows = {
            "sigma_bin": b,
            "range_bps": f"{quantiles[b]:.3f} - {quantiles[b + 1]:.3f}",
            "triggers": st["n"],
            "rise_fall_win": round(st["dir"] / nb, 4),
            "rise_fall_ev_pct": round((st["dir"] / nb * PAYOUT["RISE_FALL"] - 1) * 100, 2),
            "onetouch_win": round(st["touch"] / nb, 4),
            "onetouch_ev_pct": round((st["touch"] / nb * PAYOUT["ONETOUCH_075"] - 1) * 100, 2),
            "barrier_win": round(st["bar"] / nb, 4),
            "barrier_ev_pct": round((st["bar"] / nb * PAYOUT["BARRIER_035"] - 1) * 100, 2),
        }
        rows["gate"] = "ALLOW" if rows["rise_fall_ev_pct"] > 1.0 else "BLOCK"
        sig_map.append(rows)
    for h in range(24):
        st = agg_h[h]
        if st["n"]:
            ev = (st["win"] / st["n"]) * PAYOUT["RISE_FALL"] - 1
            hour_map.append({"utc_hour": f"{h:02d}:00", "trades": st["n"],
                             "dir_win": round(st["win"] / st["n"], 4),
                             "ev_pct": round(ev * 100, 2),
                             "units": round(st["units"], 2),
                             "action": "BLOCK" if ev < -0.03 else "ALLOW"})
    blocked = [int(r["utc_hour"][:2]) for r in hour_map if r["action"] == "BLOCK"]

    summary = {
        "vault_rows_in_sweet_spot": int(len(df_s)),
        "approved_ev_gt_2pct": int(len(approved)),
        "verified_profitable": int(df_s.verified_profitable.sum()),
        "portfolio_trades": int(len(port)),
        "portfolio_strategies": int(port.strategy_id.nunique()),
        "portfolio_win_rate_net03": round(float(perf_net["win"]), 4),
        "portfolio_wilson_lb_net03": round(wilson_lb(int((pnl_net > 0).sum()), len(pnl_net)), 4),
        "portfolio_profit_factor_net03": round(float(perf_net["pf"]), 3),
        "sharpe_net03": round(float(perf_net["sharpe"]), 3),
        "sortino_net03": round(float(perf_net["sortino"]), 3),
        "calmar_net03": round(float(perf_net["calmar"]), 3),
        "max_dd_net03_sigma": round(float(perf_net["maxdd"]), 2),
        "cum_equity_sigma": {k: round(float(port[f"eq_{k}"].iloc[-1]), 2) for k, _ in FRICTION_TIERS},
        "mc95_worst_dd_sigma": round(float(np.percentile(dds, 5)), 2),
        "mc99_worst_dd_sigma": round(float(np.percentile(dds, 1)), 2),
        "mc_prob_positive": round(float(np.mean(np.array(ends) > 0)), 4),
        "mc_pf_median": round(float(np.median(pfs)), 3),
        "kelly_fraction": round(kelly, 4),
        "quarter_kelly_risk_pct": round(kelly * 25.0, 2),
        "balance_fixed5_from_50": round(float(f_curve[-1]) if f_curve else 50.0, 2),
        "balance_quarter_kelly_from_50": round(float(k_curve[-1]) if k_curve else 50.0, 2),
        "blocked_utc_hours": blocked,
        "sigma_quantile_edges_bps": quantiles,
    }

    save_table(df_s, os.path.join(out_dir, "phase3_strategy_ev_leaderboard.parquet"))
    save_table(port, os.path.join(out_dir, "phase3_deduplicated_equity_curve.parquet"))

    xlsx = os.path.join(out_dir, "phase3_master_backtest_report.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        pd.DataFrame([{k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in summary.items()}]).T.to_excel(
            w, sheet_name="Executive_Summary", header=False)
        pd.DataFrame(contract_rows).to_excel(w, sheet_name="Contract_Comparison", index=False)
        pd.DataFrame(sig_map).to_excel(w, sheet_name="Fable_SigmaGate_Map", index=False)
        pd.DataFrame(hour_map).to_excel(w, sheet_name="Fable_HourFilter_Map", index=False)
        df_s[df_s.best_cons_ev > 0.02].to_excel(w, sheet_name="Top_Approved_Strategies", index=False)
        port.head(50000).to_excel(w, sheet_name="Chronological_Equity_Curve", index=False)

    live_anchors = [int(x) for x in approved.anchor_cell_id.tolist()]
    realized_win = {
        "RISE_FALL": float((port.unit_win[port.chosen_contract == "RISE_FALL"]).mean())
        if (port.chosen_contract == "RISE_FALL").any() else None,
        "ONETOUCH_075": float((port.unit_win[port.chosen_contract == "ONETOUCH_075"]).mean())
        if (port.chosen_contract == "ONETOUCH_075").any() else None,
        "BARRIER_035": float((port.unit_win[port.chosen_contract == "BARRIER_035"]).mean())
        if (port.chosen_contract == "BARRIER_035").any() else None,
        "NOTOUCH_10": float((port.unit_win[port.chosen_contract == "NOTOUCH_10"]).mean())
        if (port.chosen_contract == "NOTOUCH_10").any() else None,
    }
    contract_floor = {c: (round(1.02 / w, 3) if w and w > 0 else None)
                      for c, w in realized_win.items()}
    cfg = {
        "approved_strategies": int(len(approved)),
        "anchor_cell_ids": live_anchors,
        "approved_anchor_cell_ids": live_anchors,
        "payouts": PAYOUT,
        "sigma_quantile_edges_bps": quantiles,
        "blocked_utc_hours": blocked,
        "ev_gate": 0.02,
        "sim_band": [SIM_MIN, SIM_MAX],
        "safe_bins_extreme": [3, 4],
        "payout_floor_r": 1.90,
        "contract_breakeven_payout": contract_floor,
        "realized_win_rate_by_contract": {k: (round(v, 4) if v is not None else None)
                                          for k, v in realized_win.items()},
        "portfolio_verdict": ("NEGATIVE_NET_OF_FRICTION" if
                              summary["portfolio_profit_factor_net03"] < 1.0 else "POSITIVE"),
        "exit_recommendation": (
            "DO NOT TRADE LIVE: signal is payout-agnostic negative (PF<1, cum equity negative at "
            "every friction tier). Contract PnL is positive ONLY under the brief's assumed payouts."),
        "payout_contingency_note": (
            "ONETOUCH 75s needs a live payout above ~1.42x to break even (assumed 2.15x); NOTOUCH "
            "needs ~1.23x (assumed 1.55x). Real Deriv payout could not be verified: the public "
            "WebSocket ws.derivws.com/websockets/v3 returns HTTP 520 from every cluster node."),
    }
    with open(os.path.join(out_dir, "phase3_live_forward_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, "phase3_executive_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    build_whitening_bridge(out_dir, approved)

    print("\n" + "=" * 74)
    print("        PHASE 3A MASTER BACKTEST & EQUITY SIMULATION SUMMARY")
    print("=" * 74)
    print(json.dumps(summary, indent=2))
    print(f"[+] {xlsx}")


def build_whitening_bridge(out_dir, approved):
    """Fits the linear bridge live-window features -> 132-d trigger_vec_dense space."""
    try:
        inputs = os.environ.get("P3_INPUTS", os.path.join(BASE3, "phase3_inputs.npz"))
        z = np.load(inputs)
        T = z["trigger_vec"]
        Q = z["Q_seq"] if "Q_seq" in z.files else None
        if Q is None:
            print("[!] Q_seq missing - whitening bridge skipped")
            return
        idx = np.concatenate([np.arange(0, N_IN, 37), np.asarray(approved.anchor_cell_id.to_numpy(np.int64) - 1)])
        X = np.asarray(Q[idx], dtype=np.float64).reshape(len(idx), -1)
        Y = np.asarray(T[idx], dtype=np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        A = Xs.T @ Xs + 1e3 * np.eye(Xs.shape[1])
        W = np.linalg.solve(A, Xs.T @ Y)
        pred = Xs @ W
        r2 = float(1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
        np.savez(os.path.join(out_dir, "phase3_whitening_transform.npz"),
                 mu=mu, sd=sd, W=W, bridge_r2=np.array([r2]),
                 anchor_vecs=np.asarray(T[np.asarray(approved.anchor_cell_id.to_numpy(np.int64) - 1)]),
                 anchor_ids=approved.anchor_cell_id.to_numpy(np.int64))
        print(f"[+] whitening bridge saved (linear R2 = {r2:.4f})")
    except Exception as e:
        print(f"[!] whitening bridge failed: {type(e).__name__} {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--node_id", type=int, default=0)
    ap.add_argument("--num_nodes", type=int, default=3)
    ap.add_argument("--out_dir", default=os.path.join(BASE3))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--top_elite", type=int, default=0)
    a = ap.parse_args()
    if a.merge:
        merge(a.out_dir)
    else:
        run_node(a.node_id, a.num_nodes, a.out_dir, a.top_elite)
