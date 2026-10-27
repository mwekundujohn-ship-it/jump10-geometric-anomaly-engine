"""Item 06 - CZET v2 on the covariant sweet-spot manifold, LearnableGeometricAnomalyBank attached.

Universe   : hubs with 12 <= peers <= 240 and 0.85 <= mean|r| <= 0.955, plus every cell that
             is an endpoint of an edge inside that set (all cells < n_in).
Predictors : self q1/q2 (144) + peer-family mean of q1/q2 (144) + block macro descriptor (64).
Pass 1     : masked quartile reconstruction (q3/q4 from q1/q2) + protected/return heads.
Pass 2     : covariant metric learning. score = cosine + pair head(zz, zy, zz*zy, |zz-zy|,
             cos, observed cov). loss = value + BCE + weighted q3/q4 alignment + rank loss.
Linear path: 24-component PLS on the 352-d predictor block, fitted on train rows only,
             stacked with the ensembled net output by a train-fitted scalar blend.
Evaluation : R2 on held-out rows, Spearman / precision@10% on held-out edges
             (edges touching a test row; no test row is ever trained on).
Ensemble   : 3 seeds, scores and reconstructions averaged.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pqcompat
from phase2_step2_inbound_outbound_ai import CrossZoneEpicycleTransformer
from phase2_dense_anomaly_bank import LearnableGeometricAnomalyBank

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = r"C:\cluster\phase2"
N_IN = 502495
BLOCK = 12
N_PLS = 24
SEEDS = (11, 22, 33)


class CZETv2(nn.Module):
    def __init__(self, zone_dim=72, d_model=192, nhead=4, num_layers=3, dim_ff=512,
                 bank_out=64, seq_len=38):
        super().__init__()
        self.core = CrossZoneEpicycleTransformer(zone_dim=zone_dim, d_model=d_model,
                                                 nhead=nhead, num_layers=num_layers,
                                                 dim_ff=dim_ff)
        self.bank = LearnableGeometricAnomalyBank(seq_len=seq_len, num_wavelets=16,
                                                  rqa_kernels=8, out_dim=bank_out)
        self.fuse = nn.Sequential(
            nn.Linear(128 + bank_out * 4, d_model * 2),
            nn.GELU(), nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, 128),
        )
        self.pair = nn.Sequential(
            nn.Linear(4 * 128 + 2, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, q, yq=None):
        p34, p12, z, anom = self.core(q)
        if yq is None:
            return p34, p12, z, anom
        b = torch.cat([self.bank(yq[:, i, :]) for i in range(4)], dim=-1)
        z = F.normalize(self.fuse(torch.cat([z, b], dim=-1)), dim=-1)
        return p34, p12, z, anom

    def score_pairs(self, zs, zy, cos, cov):
        return self.pair(torch.cat([zs, zy, zs * zy, (zs - zy).abs(),
                                    cos.unsqueeze(-1), cov.unsqueeze(-1)], dim=-1)).squeeze(-1)


def build():
    db = pqcompat.read_parquet(os.path.join(BASE, "phase2_covariant_db.parquet"))
    ss = db[(db.covariant_peer_count >= 12) & (db.covariant_peer_count <= 240) &
            (db.mean_0_50_similarity >= 0.85) & (db.mean_0_50_similarity <= 0.955)]
    hub_ids = ss.anchor_cell_id.to_numpy(np.int64) - 1
    hub_ids = hub_ids[(hub_ids >= 0) & (hub_ids < N_IN)]
    edges = pqcompat.read_parquet(os.path.join(BASE, "phase2_covariant_edges.parquet"))
    hset = set(hub_ids.tolist())
    m = edges.src_idx.isin(hset) & edges.dst_idx.isin(hset)
    e = edges[m & (edges.src_idx < N_IN) & (edges.dst_idx < N_IN)].reset_index(drop=True)
    ids = np.unique(np.concatenate([hub_ids, e.src_idx.to_numpy(np.int64),
                                    e.dst_idx.to_numpy(np.int64)]))
    ids = ids[(ids >= 0) & (ids < N_IN)]
    print(f"[*] rows={len(ids):,} hubs={len(hub_ids):,} edges={len(e):,}", flush=True)

    Q = np.load(os.path.join(BASE, "czet_train_cache.npz"), mmap_mode="r")
    q = np.ascontiguousarray(Q["Q_seq"][ids])
    held = np.ascontiguousarray(Q["protected_held"][ids].astype(np.float32))
    ret = np.ascontiguousarray(Q["q4_net_return"][ids])
    z = np.load(os.path.join(BASE, "jump10_phase2_subzones.npz"), mmap_mode="r")
    yq = np.stack([np.ascontiguousarray(z[f"y_q{i}"][ids]) for i in range(1, 5)], axis=1)

    loc = {int(v): i for i, v in enumerate(ids)}
    es = np.array([loc[int(v)] for v in e.src_idx], dtype=np.int64)
    ed = np.array([loc[int(v)] for v in e.dst_idx], dtype=np.int64)
    sign = np.sign(e.cov_0_50.to_numpy(np.float32))
    cov = np.abs(e.cov_0_50.to_numpy(np.float32))
    lock = np.clip(e.lock_50_100.to_numpy(np.float32), -1, 1)

    n = len(ids)
    acc = np.zeros((n, q.shape[2] * 2))
    cnt = np.zeros(n)
    q2 = q[:, :2, :].astype(np.float64)
    for a, b in zip(es, ed):
        acc[a] += q2[b].ravel()
        acc[b] += q2[a].ravel()
        cnt[a] += 1
        cnt[b] += 1
    fam = np.divide(acc, np.maximum(cnt, 1)[:, None])
    mac = np.nan_to_num(np.load(os.path.join(BASE, "_macro_desc.npy"))[ids // BLOCK].astype(np.float64))
    Xp = np.concatenate([q2.reshape(n, -1), fam, mac], axis=1)
    print(f"[*] predictor block {Xp.shape}", flush=True)
    return ids, q, yq, held, ret, es, ed, sign, cov, lock, Xp


def fit_pls(X, Y, tr, kc=N_PLS):
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-9
    Xs = (X - mu) / sd
    my = Y[tr].mean(0)
    Xr = Xs[tr].copy()
    Yr = (Y - my)[tr].copy()
    W, P, C = [], [], []
    for _ in range(kc):
        u = Yr[:, 0]
        w = Xr.T @ u
        w /= np.linalg.norm(w) + 1e-12
        t = Xr @ w
        pp = Xr.T @ t / (t @ t)
        cc = Yr.T @ t / (t @ t)
        Xr -= np.outer(t, pp)
        Yr -= np.outer(t, cc)
        W.append(w)
        P.append(pp)
        C.append(cc)
    Wm, Pm, Cm = np.column_stack(W), np.column_stack(P), np.column_stack(C)
    B = Wm @ np.linalg.pinv(Pm.T @ Wm) @ Cm.T
    return {"mu": mu, "sd": sd, "my": my, "B": B, "k": kc}


def pls_predict(pls, X):
    return ((X - pls["mu"]) / pls["sd"]) @ pls["B"] + pls["my"]


def train_one(seed, D, epochs1, epochs2, batch):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_rows, te_rows, tr_e, te_e = D["tr_rows"], D["te_rows"], D["tr_e"], D["te_e"]
    Qt, Yt, Ht, Rt = D["Qt"], D["Yt"], D["Ht"], D["Rt"]
    model = CZETv2()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    log = {"pass1": [], "pass2": []}

    steps = max(1, len(tr_rows) // batch)
    for ep in range(1, epochs1 + 1):
        p = tr_rows[np.random.permutation(len(tr_rows))]
        tot, t0 = 0.0, time.time()
        model.train()
        for s in range(steps):
            idx = np.sort(p[s * batch:(s + 1) * batch])
            bq, by, bh, br = Qt[idx], Yt[idx], Ht[idx], Rt[idx]
            opt.zero_grad()
            p34, p12, _, anom = model(bq, by)
            loss = (F.mse_loss(p34, bq[:, 2:4, :]) + F.mse_loss(p12, bq[:, 0:2, :])
                    + 0.25 * (F.binary_cross_entropy_with_logits(anom[:, 0], bh)
                              + F.smooth_l1_loss(anom[:, 2], br)))
            loss.backward()
            opt.step()
            tot += loss.item()
        log["pass1"].append({"seed": seed, "epoch": ep, "loss": round(tot / steps, 5),
                             "secs": round(time.time() - t0, 1)})
        print(f"    [s{seed}] pass1 {ep}/{epochs1} loss {tot / steps:.5f} ({time.time() - t0:.0f}s)", flush=True)

    opt2 = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs2, eta_min=2e-5)
    S, Dd = D["S"], D["D"]
    SG, LK, CV = D["SG"], D["LK"], D["CV"]
    steps2 = max(1, len(tr_e) // batch)
    for ep in range(1, epochs2 + 1):
        p = np.random.permutation(len(tr_e))
        tot, t0 = 0.0, time.time()
        model.train()
        for s in range(steps2):
            j = torch.from_numpy(p[s * batch:(s + 1) * batch])
            sg, lk, cv = SG[j], LK[j], CV[j]
            opt2.zero_grad()
            p34z, _, zz, _ = model(Qt[S[j]], Yt[S[j]])
            _, _, zy, _ = model(Qt[Dd[j]], Yt[Dd[j]])
            cos = F.cosine_similarity(zz, zy * sg[:, None], dim=-1)
            sc = model.score_pairs(zz, zy, cos, cv)
            loss = (F.mse_loss(sc, lk) + 0.5 * F.mse_loss(cos, lk)
                    + 0.5 * F.binary_cross_entropy_with_logits(4.0 * sc, (lk >= 0.65).float())
                    + torch.mean(torch.sigmoid(lk * 3.0)[:, None, None]
                                 * (p34z - Qt[S[j]][:, 2:4, :]) ** 2))
            dl = lk[:, None] - lk[None, :]
            order = sc[:, None] - sc[None, :]
            mask = dl > 0.25
            if bool(mask.any()):
                loss = loss + 0.75 * F.softplus(-order[mask]).mean()
            loss.backward()
            opt2.step()
            tot += loss.item()
        sched.step()
        log["pass2"].append({"seed": seed, "epoch": ep, "loss": round(tot / steps2, 5),
                             "secs": round(time.time() - t0, 1)})
        print(f"    [s{seed}] pass2 {ep}/{epochs2} loss {tot / steps2:.5f} ({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    with torch.no_grad():
        p34, _, _, _ = model(Qt[tr_rows], Yt[tr_rows])
        Pnet_tr = p34.numpy().reshape(len(tr_rows), -1)
        p34, _, _, _ = model(Qt[te_rows], Yt[te_rows])
        Pnet_te = p34.numpy().reshape(len(te_rows), -1)

        def scores(e_idx):
            zs_ = D["Qt"][D["es"][e_idx]]
            zy_ = D["Qt"][D["ed"][e_idx]]
            _, _, zz_, _ = model(zs_, D["Yt"][D["es"][e_idx]])
            _, _, zy2_, _ = model(zy_, D["Yt"][D["ed"][e_idx]])
            cs_ = F.cosine_similarity(zz_, zy2_ * torch.from_numpy(D["sign"][e_idx])[:, None], dim=-1)
            return model.score_pairs(zz_, zy2_, cs_, torch.from_numpy(D["cov"][e_idx])).numpy()

        s_tr = scores(tr_e)
        s_te = scores(te_e)
    return model, log, Pnet_tr, Pnet_te, s_tr, s_te


def main(epochs1=8, epochs2=30, batch=128, threads=4):
    torch.set_num_threads(threads)
    ids, q, yq, held, ret, es, ed, sign, cov, lock, Xp = build()

    n = len(ids)
    perm = np.random.default_rng(11).permutation(n)
    n_tr = int(0.8 * n)
    tr_rows = np.sort(perm[:n_tr])
    te_rows = np.sort(perm[n_tr:])
    tr_set = np.zeros(n, bool)
    tr_set[tr_rows] = True
    te_set = ~tr_set
    tr_e = np.where(tr_set[es] & tr_set[ed])[0]
    te_e = np.where(te_set[es] | te_set[ed])[0]
    print(f"[*] train rows={len(tr_rows):,} test rows={len(te_rows):,} "
          f"train edges={len(tr_e):,} test edges={len(te_e):,}", flush=True)

    Y = q[:, 2:4, :].reshape(n, -1).astype(np.float64)
    pls = fit_pls(Xp, Y, tr_rows)
    np.savez(os.path.join(BASE, "czet_phase2_pls.npz"),
             mu=pls["mu"], sd=pls["sd"], my=pls["my"], B=pls["B"], k=np.array([pls["k"]]))
    Ppls_tr = pls_predict(pls, Xp[tr_rows])
    Ppls_te = pls_predict(pls, Xp[te_rows])
    r2_pls = float(1 - ((Ppls_te - Y[te_rows]) ** 2).sum() /
                   ((Y[te_rows] - Y[te_rows].mean(0)) ** 2).sum())
    print(f"[*] PLS({pls['k']}) held-out R2 = {r2_pls:.5f}", flush=True)

    D = {"tr_rows": tr_rows, "te_rows": te_rows, "tr_e": tr_e, "te_e": te_e,
         "Qt": torch.from_numpy(q), "Yt": torch.from_numpy(yq),
         "Ht": torch.from_numpy(held), "Rt": torch.from_numpy(ret),
         "es": es, "ed": ed, "sign": sign, "cov": cov, "lock": lock,
         "S": torch.from_numpy(es[tr_e]), "D": torch.from_numpy(ed[tr_e]),
         "SG": torch.from_numpy(sign[tr_e]), "LK": torch.from_numpy(lock[tr_e]),
         "CV": torch.from_numpy(cov[tr_e])}

    logs, Ptr, Pte, Str, Ste, models = [], [], [], [], [], []
    for sd in SEEDS:
        m, lg, a, b, c, d = train_one(sd, D, epochs1, epochs2, batch)
        models.append(m)
        logs.append(lg)
        Ptr.append(a)
        Pte.append(b)
        Str.append(c)
        Ste.append(d)
        from scipy.stats import spearmanr
        print(f"    [s{sd}] seed held-out Spearman = "
              f"{float(spearmanr(d, lock[te_e]).statistic):.4f}", flush=True)

    Pnet_tr = np.mean(Ptr, axis=0)
    Pnet_te = np.mean(Pte, axis=0)
    s_te = np.mean(Ste, axis=0)
    s_tr = np.mean(Str, axis=0)
    Ytr, Yte = Y[tr_rows], Y[te_rows]

    num = float(np.sum((Pnet_tr - Ppls_tr) * (Ytr - Ppls_tr)))
    den = float(np.sum((Pnet_tr - Ppls_tr) ** 2)) + 1e-12
    lam = float(np.clip(num / den, 0.0, 1.0))
    w = np.array([lam, 1.0 - lam])
    Pst_te = w[0] * Pnet_te + w[1] * Ppls_te
    r2_net = float(1 - ((Pnet_te - Yte) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum())
    r2_st = float(1 - ((Pst_te - Yte) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum())

    lk_te = lock[te_e]
    from scipy.stats import spearmanr
    rho = float(spearmanr(s_te, lk_te).statistic)
    pear = float(np.corrcoef(s_te, lk_te)[0, 1])
    pos = lk_te >= 0.65
    prec = None
    if pos.sum() > 5:
        k10 = max(1, int(0.1 * len(s_te)))
        prec = float(pos[np.argsort(-s_te)[:k10]].mean())
    rho_cov = float(spearmanr(cov[te_e], lk_te).statistic)

    out = {
        "model": "CZETv2 + LearnableGeometricAnomalyBank + pair head + PLS path (3-seed ensemble)",
        "universe": "sweet spot: 12-240 peers, 0.85-0.955 mean|r|",
        "train_rows": int(len(tr_rows)), "test_rows": int(len(te_rows)),
        "train_edges": int(len(tr_e)), "test_edges": int(len(te_e)),
        "spearman": round(rho, 4), "pair_spearman": round(rho, 4),
        "spearman_observed_cov_baseline": round(rho_cov, 4),
        "r2": round(r2_st, 5), "oos_r2": round(r2_st, 5),
        "r2_net_only": round(r2_net, 5), "r2_pls_only": round(r2_pls, 5),
        "stack_weights": {"net": round(float(w[0]), 4), "pls": round(float(w[1]), 4)},
        "pair_score_vs_lock_pearson": round(pear, 4),
        "precision_at_10pct": None if prec is None else round(prec, 4),
        "base_rate_lock_ge_065": round(float(pos.mean()), 4),
        "params": int(sum(p.numel() for p in models[0].parameters())),
        "seeds": list(SEEDS), "epochs": {"pass1": epochs1, "pass2": epochs2},
        "protocol": "rows split 80/20 at random; test edges are those touching a test row; "
                    "PLS, stack weight and pair head are fitted on train only; R2 on held-out rows.",
    }
    print(json.dumps(out, indent=2), flush=True)
    with open(os.path.join(BASE, "czet_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(BASE, "czet_train_log.json"), "w") as f:
        json.dump({"runs": logs, "config": {"d_model": 192, "layers": 3, "bank_out": 64,
                                            "pls_components": N_PLS, "lr_pass1": 3e-4,
                                            "lr_pass2": 1.5e-4, "batch": batch, "seeds": list(SEEDS),
                                            "universe": "sweetspot 12-240 peers, 0.85-0.955 |r|"}}, f, indent=2)
    torch.save({"state_dicts": [m.state_dict() for m in models],
                "cfg": {"zone_dim": 72, "d_model": 192, "nhead": 4, "num_layers": 3,
                        "dim_ff": 512, "bank_out": 64, "seq_len": 38},
                "stack_weights": w.tolist(), "seeds": list(SEEDS)},
               os.path.join(BASE, "czet_phase2_sweetspot.pt"))


if __name__ == "__main__":
    main()
