"""Build phase3_whitening_transform.npz: live 75-tick -> 132-d trigger vector bridge.

Fits a ridge-regularised linear map on the INBOUND half only (no lookahead) from the
Phase 2 dense quartile features (Q_seq, 4x72 = 288-d) to the Phase 2 trigger vector
(132-d cosine-state descriptor) that the Phase 3 anchor matching uses.
"""
import os
import json
import numpy as np

P2 = r"C:\cluster\phase2"
P3 = r"C:\cluster\phase3"
N_IN = 502495
RIDGE = 1e-2

z = np.load(os.path.join(P2, "czet_train_cache.npz"), mmap_mode="r")
Q = np.asarray(z["Q_seq"][:, :2, :], dtype=np.float64).reshape(len(z["Q_seq"]), -1)
T = np.load(os.path.join(P3, "phase3_inputs.npz"), mmap_mode="r")["trigger_vec"]
T = np.asarray(T, dtype=np.float64)

Q_in, T_in = Q[:N_IN], T[:N_IN]
q_mu = Q_in.mean(axis=0)
q_sd = Q_in.std(axis=0)
q_sd = np.where(q_sd < 1e-9, 1.0, q_sd)
X = (Q_in - q_mu) / q_sd

G = X.T @ X / len(X)
G.flat[:: G.shape[0] + 1] += RIDGE
W = np.linalg.solve(G, X.T @ T_in / len(X)).astype(np.float32)

pred = X @ W
r2 = 1.0 - float(((T_in - pred) ** 2).sum() / ((T_in - T_in.mean(0)) ** 2).sum())

vault = __import__("pandas").read_csv(os.path.join(P3, "phase3_vault.csv"))
anchors = vault.anchor_cell_id.to_numpy(np.int64) - 1

np.savez_compressed(
    os.path.join(P3, "phase3_whitening_transform.npz"),
    q_mu=q_mu.astype(np.float32), q_sd=q_sd.astype(np.float32), W=W,
    anchor_vecs=T[anchors].astype(np.float32), anchor_ids=vault.anchor_cell_id.to_numpy(np.int64),
    anchor_actions=vault.recommended_action.to_numpy().astype("U32"),
    bridge_r2=np.float32(r2), n_fit=np.int64(N_IN),
)
print(f"[+] bridge R2 (inbound, Q1+Q2 144->132) = {r2:.4f}   params={W.size:,}   anchors={len(anchors)}")
json.dump({"bridge_r2": r2, "n_fit": N_IN, "anchors": int(len(anchors))},
          open(os.path.join(P3, "phase3_whitening_bridge.json"), "w"), indent=2)
