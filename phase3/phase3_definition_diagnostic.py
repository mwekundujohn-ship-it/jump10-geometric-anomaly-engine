"""Diagnostic: does the Phase 2 outbound edge survive as a tick-75->149 directional trade?

Compares three definitions on the SAME outbound matches:
  A) Phase 2 statistic      : mean( sign(q4_net_return * pol) == expected_action )
  B) tick-149 close sign    : mean( sign(path_149 * pol * action_sign) > 0 )
  C) inbound-calibrated dir : mean( sign(path_149 * pol * dir_sign_inbound) > 0 )
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\cluster\phase2")
import pqcompat

P2 = r"C:\cluster\phase2"
N_IN = 502495
SIM_MIN, SIM_MAX = 0.85, 0.955

z = np.load(os.path.join(P2, "jump10_phase2_subzones.npz"), mmap_mode="r")
T = np.asarray(z["trigger_vec_dense"], dtype=np.float32)
scan = np.load(os.path.join(P2, "jump10_phase2_scan.npz"), mmap_mode="r")
qret = np.asarray(scan["q4_net_return"], dtype=np.float32)
raw = np.load(os.path.join(r"C:\cluster\phase3", "phase3_inputs.npz"))["raw"]

vault = pd.read_csv(os.path.join(r"C:\cluster\phase3", "phase3_vault.csv"))
vault = vault[(vault.covariant_peer_count >= 12) & (vault.covariant_peer_count <= 400) &
              (vault.mean_0_50_similarity >= 0.85) & (vault.mean_0_50_similarity <= 0.955)]
anchors = (vault.anchor_cell_id.to_numpy(np.int64) - 1)
anchors = anchors[anchors < N_IN]

raw_out = np.asarray(raw[N_IN:], dtype=np.float64)
std_out = raw_out[:, :75].std(axis=1)
std_out = np.where(std_out < 1e-7, 1.0, std_out)
paths = (raw_out[:, 75:] - raw_out[:, 74:75]) / std_out[:, None]
ret_out = qret[N_IN:]
T_out = T[N_IN:]
T_anch = T[anchors]

raw_cal = np.asarray(raw[:N_IN:5], dtype=np.float64)
std_cal = raw_cal[:, :75].std(axis=1)
std_cal = np.where(std_cal < 1e-7, 1.0, std_cal)
paths_cal = (raw_cal[:, 75:] - raw_cal[:, 74:75]) / std_cal[:, None]
T_cal = T[:N_IN:5]

rows = []
for j, a in enumerate(anchors):
    act = str(vault.recommended_action.iloc[j])
    expected = -1.0 if "SELL" in act.upper() else 1.0
    sims = T_out @ T_anch[j]
    hits = np.nonzero((np.abs(sims) >= SIM_MIN) & (np.abs(sims) <= SIM_MAX))[0]
    if len(hits) < 8:
        continue
    pol = np.sign(sims[hits])
    ca = T_cal @ T_anch[j]
    ch = np.nonzero((np.abs(ca) >= SIM_MIN) & (np.abs(ca) <= SIM_MAX))[0]
    if len(ch) >= 40:
        med = float(np.median(paths_cal[ch] * np.sign(ca[ch])[:, None]))
        dsign = 1.0 if med >= 0 else -1.0
    else:
        dsign = expected
    curve = paths[hits] * pol[:, None] * expected
    curve_in = paths[hits] * pol[:, None] * dsign
    rows.append({
        "anchor_cell_id": int(a + 1),
        "n": int(len(hits)),
        "win_phase2_q4stat": float(np.mean(np.sign(ret_out[hits] * pol) == expected)),
        "win_close149_action": float(np.mean(curve[:, -1] > 0)),
        "win_close149_inbound": float(np.mean(curve_in[:, -1] > 0)),
        "agree_action_vs_inbound": float(np.mean(np.sign(curve[:, -1]) == np.sign(curve_in[:, -1]))),
        "vault_reported_dirwin": float(vault.outbound_directional_win_rate.iloc[j]),
        "mean_abs_sim": float(np.abs(sims[hits]).mean()),
    })

df = pd.DataFrame(rows)
w = df.n.to_numpy(float)
print(f"anchors with >=8 matches: {len(df)}  trades: {int(df.n.sum()):,}")
for c in ("win_phase2_q4stat", "win_close149_action", "win_close149_inbound", "vault_reported_dirwin"):
    print(f"  {c:26s} trades-weighted = {np.average(df[c], weights=w):.4f}   median={df[c].median():.4f}")
print(f"  action vs inbound directional agreement = {np.average(df.agree_action_vs_inbound, weights=w):.4f}")
print(f"  corr(phase2 stat, close149_action)  = {df.win_phase2_q4stat.corr(df.win_close149_action):.3f}")
print(f"  corr(phase2 stat, vault reported)   = {df.win_phase2_q4stat.corr(df.vault_reported_dirwin):.3f}")
df.to_csv(r"C:\cluster\phase3\phase3_definition_diagnostic.csv", index=False)
json.dump({"anchors": len(df), "trades": int(df.n.sum()),
           "w_phase2": float(np.average(df.win_phase2_q4stat, weights=w)),
           "w_close_action": float(np.average(df.win_close149_action, weights=w)),
           "w_close_inbound": float(np.average(df.win_close149_inbound, weights=w)),
           "vault_reported": float(np.average(df.vault_reported_dirwin, weights=w)),
           "agree": float(np.average(df.agree_action_vs_inbound, weights=w))},
          open(r"C:\cluster\phase3\phase3_definition_diagnostic.json", "w"), indent=2)
print("[+] saved diagnostic")
