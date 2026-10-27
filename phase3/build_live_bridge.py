"""Live-parity bridge: raw 75-tick window -> Phase 3 anchor space (132-d).

Live path (reproducible from ticks alone):
    y = last 75 ticks of the cell (0-50% window)
    F = [dense(y[:38]) | dense(y[38:])]          (Phase 2 dense extractor, raw 64-d each)
    F = (F - mu) / sd
    T_hat = F @ W   (ridge, or MLP)              -> 132-d, cosine vs anchor_vecs

Reports R2, cosine error, and trigger-match recall/IoU vs the true Phase 2 space.
"""
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, r"C:\cluster\phase2")
from phase2_dense_anomaly_bank import compute_quartile_brute_force_features

P2 = r"C:\cluster\phase2"
P3 = r"C:\cluster\phase3"
N_IN = 502495
CHUNK = 50000

z = np.load(os.path.join(P2, "jump10_phase2_subzones.npz"), mmap_mode="r")
T = np.load(os.path.join(P3, "phase3_inputs.npz"), mmap_mode="r")["trigger_vec"]
N = len(T)
t0 = time.time()

cache = os.path.join(P3, "_live_feat.npy")
if not os.path.exists(cache):
    q1 = np.asarray(z["y_q1"], dtype=np.float64)
    q2 = np.asarray(z["y_q2"], dtype=np.float64)
    F = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32, shape=(N, 128))
    for s in range(0, N, CHUNK):
        e = min(N, s + CHUNK)
        a = compute_quartile_brute_force_features(q1[s:e])
        b = compute_quartile_brute_force_features(q2[s:e])
        F[s:e] = np.hstack([a, b]).astype(np.float32)
        if (s // CHUNK) % 4 == 0:
            print(f"    {e:,}/{N:,}  ({time.time()-t0:.0f}s)", flush=True)
    F.flush()
    del F
print(f"[*] live feature matrix ready ({time.time()-t0:.0f}s)", flush=True)

F = np.load(cache, mmap_mode="r")
F_in = np.asarray(F[:N_IN], dtype=np.float64)
mu = F_in.mean(0)
sd = F_in.std(0)
sd = np.where(sd < 1e-9, 1.0, sd)
X = (F_in - mu) / sd
T_in = np.asarray(T[:N_IN], dtype=np.float64)

G = X.T @ X / len(X)
G.flat[:: G.shape[0] + 1] += 1e-3
W = np.linalg.solve(G, X.T @ T_in / len(X))
pred = X @ W
r2 = 1.0 - float(((T_in - pred) ** 2).sum() / ((T_in - T_in.mean(0)) ** 2).sum())
print(f"[*] linear bridge R2 = {r2:.4f}", flush=True)
del X, T_in, pred

vault = __import__("pandas").read_csv(os.path.join(P3, "phase3_vault.csv"))
anchors = vault.anchor_cell_id.to_numpy(np.int64) - 1
A = np.asarray(T[anchors], dtype=np.float64)
An = A / np.linalg.norm(A, axis=1, keepdims=True)

idx = np.arange(N_IN, N, 2)
Tt = np.asarray(T[idx], dtype=np.float64)
Ft = (np.asarray(F[idx], dtype=np.float64) - mu) / sd
Tc = Ft @ W
st = (Tt / np.linalg.norm(Tt, axis=1, keepdims=True)) @ An.T
sb = (Tc / np.linalg.norm(Tc, axis=1, keepdims=True)) @ An.T
bt = (np.abs(st) >= 0.85) & (np.abs(st) <= 0.955)
bb = (np.abs(sb) >= 0.85) & (np.abs(sb) <= 0.955)
recall = float((bt & bb).sum() / max(1, bt.sum()))
iou = float((bt & bb).sum() / max(1, (bt | bb).sum()))
pol = float((np.sign(st[bt & bb]) == np.sign(sb[bt & bb])).mean()) if (bt & bb).any() else 0.0
print(f"[*] outbound match fidelity: recall={recall:.3f} IoU={iou:.3f} polarity={pol:.3f} "
      f"true_hits={int(bt.sum()):,} bridge_hits={int(bb.sum()):,}", flush=True)

np.savez_compressed(
    os.path.join(P3, "phase3_whitening_transform.npz"),
    q_mu=mu.astype(np.float32), q_sd=sd.astype(np.float32), W=W.astype(np.float32),
    anchor_vecs=A.astype(np.float32), anchor_ids=vault.anchor_cell_id.to_numpy(np.int64),
    anchor_actions=vault.recommended_action.to_numpy().astype("U32"),
    bridge_r2=np.float32(r2), n_fit=np.int64(N_IN),
    match_recall=np.float32(recall), match_iou=np.float32(iou),
)
json.dump({"bridge_r2": r2, "match_recall": recall, "match_iou": iou,
           "polarity_agreement": pol, "n_fit": N_IN, "anchors": int(len(anchors)),
           "feature": "dense_q1_raw64+dense_q2_raw64 -> 128 -> 132"},
          open(os.path.join(P3, "phase3_whitening_bridge.json"), "w"), indent=2)
print("[+] phase3_whitening_transform.npz written")
