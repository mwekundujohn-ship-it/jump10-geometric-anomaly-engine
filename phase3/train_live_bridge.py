"""Train a nonlinear live-parity bridge (75-tick dense features -> Phase 3 132-d anchor space).

Cosine-loss MLP. Trains on INBOUND only; validates on the last 10% of inbound, then measures
outbound trigger-match recall / IoU against the true Phase 2 similarity.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"C:\cluster\phase3")
sys.path.insert(0, r"C:\cluster\phase2")

P3 = r"C:\cluster\phase3"
P2 = r"C:\cluster\phase2"
N_IN = 502495
torch.manual_seed(0)
torch.set_num_threads(8)

F = np.load(os.path.join(P3, "_live_feat.npy"), mmap_mode="r")
T = np.load(os.path.join(P3, "phase3_inputs.npz"), mmap_mode="r")["trigger_vec"]
N = len(T)

Fi = np.asarray(F[:N_IN], dtype=np.float32)
mu = Fi.mean(0)
sd = Fi.std(0)
sd = np.where(sd < 1e-9, 1.0, sd)
Fi = (Fi - mu) / sd
Ti = np.asarray(T[:N_IN], dtype=np.float32)
Ti = Ti / np.maximum(np.linalg.norm(Ti, axis=1, keepdims=True), 1e-9)

n_val = 50000
Xtr, Ytr = torch.from_numpy(Fi[:-n_val]), torch.from_numpy(Ti[:-n_val])
Xva, Yva = torch.from_numpy(Fi[-n_val:]), torch.from_numpy(Ti[-n_val:])
del Fi, Ti


class Net(nn.Module):
    def __init__(self, d_in=128, d_out=132, h=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, d_out),
        )

    def forward(self, x):
        return self.net(x)


net = Net().to("cpu")
opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
BS = 4096
n = len(Xtr)
best = 9.9
t0 = time.time()

for ep in range(12):
    perm = torch.randperm(n)
    tot = 0.0
    for s in range(0, n, BS):
        b = perm[s:s + BS]
        xb, yb = Xtr[b], Ytr[b]
        pred = net(xb)
        cos = (pred * yb).sum(1) / (pred.norm(dim=1) * yb.norm(dim=1) + 1e-9)
        loss = (1.0 - cos).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss) * len(b)
    sched.step()
    with torch.no_grad():
        pv = net(Xva)
        cv = ((pv * Yva).sum(1) / (pv.norm(dim=1) * Yva.norm(dim=1) + 1e-9))
        vloss = float((1.0 - cv).mean())
        acc = float((cv > 0.995).float().mean())
    print(f"  epoch {ep+1:2d}  train_cos={1-tot/n:.5f}  val_cos={float(cv.mean()):.5f}  "
          f"val>0.995={acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if vloss < best:
        best = vloss
        torch.save(net.state_dict(), os.path.join(P3, "phase3_live_bridge.pt"))

net.load_state_dict(torch.load(os.path.join(P3, "phase3_live_bridge.pt")))
net.eval()

import pandas as pd
vault = pd.read_csv(os.path.join(P3, "phase3_vault.csv"))
anc = vault.anchor_cell_id.to_numpy(np.int64) - 1
A = np.asarray(T[anc], dtype=np.float64)
An = A / np.linalg.norm(A, axis=1, keepdims=True)

idx = np.arange(N_IN, N, 2)
Tt = np.asarray(T[idx], dtype=np.float64)
Tt_n = Tt / np.linalg.norm(Tt, axis=1, keepdims=True)
st = Tt_n @ An.T

Fc = (np.asarray(F[idx], dtype=np.float32) - mu) / sd
with torch.no_grad():
    Tc = net(torch.from_numpy(Fc)).numpy().astype(np.float64)
Tc_n = Tc / np.maximum(np.linalg.norm(Tc, axis=1, keepdims=True), 1e-9)
sb = Tc_n @ An.T

bt = (np.abs(st) >= 0.85) & (np.abs(st) <= 0.955)
bb = (np.abs(sb) >= 0.85) & (np.abs(sb) <= 0.955)
recall = float((bt & bb).sum() / max(1, bt.sum()))
prec = float((bt & bb).sum() / max(1, bb.sum()))
iou = float((bt & bb).sum() / max(1, (bt | bb).sum()))
pol = float((np.sign(st[bt & bb]) == np.sign(sb[bt & bb])).mean()) if (bt & bb).any() else 0.0
print(f"[*] outbound parity: recall={recall:.3f} precision={prec:.3f} IoU={iou:.3f} "
      f"polarity={pol:.3f} true_hits={int(bt.sum()):,} bridge_hits={int(bb.sum()):,}", flush=True)

torch.save({"state_dict": net.state_dict(), "d_in": 128, "d_out": 132, "h": 512,
            "feat": "dense_q1(64)+dense_q2(64) raw -> standardize"},
           os.path.join(P3, "phase3_live_bridge.pt"))
np.savez_compressed(
    os.path.join(P3, "phase3_whitening_transform.npz"),
    q_mu=mu.astype(np.float32), q_sd=sd.astype(np.float32),
    anchor_vecs=A.astype(np.float32), anchor_ids=vault.anchor_cell_id.to_numpy(np.int64),
    anchor_actions=vault.recommended_action.to_numpy().astype("U32"),
    bridge_r2=np.float32(np.nan), n_fit=np.int64(N_IN),
    match_recall=np.float32(recall), match_iou=np.float32(iou), match_precision=np.float32(prec),
)
json.dump({"bridge": "mlp_128_512_512_132_cosine", "match_recall": recall,
           "match_precision": prec, "match_iou": iou, "polarity_agreement": pol,
           "val_cos": float(cv.mean()), "n_fit": N_IN, "anchors": int(len(anc))},
          open(os.path.join(P3, "phase3_whitening_bridge.json"), "w"), indent=2)
print("[+] phase3_whitening_transform.npz + phase3_live_bridge.pt written")
