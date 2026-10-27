"""Phase 2 Step 2 - CZET inbound foundation training + outbound vault verification.

Adapted to the whitened feature bank (v2):
  Q_seq(cell, quartile, 72) = [desc_Qk (17) | dense_qk whitened, padded to 55]
  trigger_vec_dense (N,132) / resolution_vec_dense (N,138) for the similarity engine

Modes:
  --mode train   train pass 1 (bidirectional masked quartile reconstruction) + pass 2
                 (covariant-pair metric learning) -> czet_phase2.pt
  --mode vault   score outbound triggers against the trained hubs -> strategy vault
  --mode both    (default)

Outputs (in --phase2_dir):
  czet_phase2.pt, czet_train_log.json, phase2_strategy_vault.parquet,
  phase2_anomaly_strategies.xlsx (if openpyxl), phase2_step2_report.json
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pqcompat
except Exception:
    pqcompat = None

import torch
import torch.nn as nn
import torch.nn.functional as F


def read_table(path):
    if path.endswith(".parquet"):
        try:
            if pqcompat is not None:
                return pqcompat.read_parquet(path)
            return pd.read_parquet(path)
        except Exception:
            alt = path[:-len(".parquet")] + ".csv"
            if os.path.exists(alt):
                return pd.read_csv(alt)
            raise
    return pd.read_csv(path)


class CrossZoneEpicycleTransformer(nn.Module):
    def __init__(self, zone_dim=72, d_model=256, nhead=4, num_layers=4, dim_ff=768):
        super().__init__()
        self.zone_proj = nn.Linear(zone_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.zone_decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, zone_dim),
        )
        self.covariant_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 128),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model * 2, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 3),
        )

    def encode(self, q, mask=None):
        x = self.zone_proj(q) + self.pos_embed
        if mask is not None:
            x = x.clone()
            x[:, mask, :] = 0.0
        return self.transformer(x)

    def forward(self, q):
        h_fwd = self.encode(q, [2, 3])
        pred_q34 = self.zone_decoder(h_fwd[:, 2:4, :])
        h_bwd = self.encode(q, [0, 1])
        pred_q12 = self.zone_decoder(h_bwd[:, 0:2, :])
        h_early = torch.cat([h_fwd[:, 0, :], h_fwd[:, 1, :]], dim=-1)
        z_cov = F.normalize(self.covariant_proj(h_early), dim=-1)
        anom = self.anomaly_head(h_early)
        return pred_q34, pred_q12, z_cov, anom


def train(phase2_dir, epochs1, epochs2, batch_size, d_model, layers, nhead, ff,
          max_pairs, seed=1337, max_train_rows=0, resume=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cache = np.load(os.path.join(phase2_dir, "czet_train_cache.npz"), mmap_mode="r")
    Q = cache["Q_seq"]
    N = Q.shape[0]
    zone_dim = Q.shape[2]
    n_in = int(N * 0.85)
    if max_train_rows and max_train_rows < n_in:
        train_idx = np.sort(np.random.permutation(n_in)[:max_train_rows]).astype(np.int64)
    else:
        train_idx = np.arange(n_in, dtype=np.int64)
    n_tr = len(train_idx)
    print(f"[*] Q_seq={Q.shape} inbound={n_in:,} outbound={N - n_in:,} train_rows={n_tr:,} "
          f"| threads={torch.get_num_threads()}", flush=True)

    model = CrossZoneEpicycleTransformer(zone_dim=zone_dim, d_model=d_model, nhead=nhead,
                                         num_layers=layers, dim_ff=ff)
    if resume and os.path.exists(os.path.join(phase2_dir, "czet_phase2.pt")):
        ck = torch.load(os.path.join(phase2_dir, "czet_phase2.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print("[*] resumed weights from czet_phase2.pt", ck.get("cfg"), flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] CZET params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    held = cache["protected_held"]
    ret = cache["q4_net_return"]
    log = {"params": int(n_params), "zone_dim": int(zone_dim), "inbound": n_in, "train_rows": int(n_tr),
           "batch_size": batch_size, "pass1": [], "pass2": []}

    steps = n_tr // batch_size
    model.train()
    for ep in range(1, epochs1 + 1):
        perm = train_idx[np.random.permutation(n_tr)]
        tot, t0 = 0.0, time.time()
        for s in range(steps):
            idx = np.sort(perm[s * batch_size:(s + 1) * batch_size])
            bq = torch.from_numpy(np.ascontiguousarray(Q[idx]))
            b_held = torch.from_numpy(np.ascontiguousarray(held[idx].astype(np.float32)))
            b_ret = torch.from_numpy(np.ascontiguousarray(ret[idx]))
            opt.zero_grad()
            p34, p12, _, anom = model(bq)
            loss = (F.mse_loss(p34, bq[:, 2:4, :]) + F.mse_loss(p12, bq[:, 0:2, :])
                    + 0.25 * (F.binary_cross_entropy_with_logits(anom[:, 0], b_held)
                              + F.smooth_l1_loss(anom[:, 2], b_ret)))
            loss.backward()
            opt.step()
            tot += loss.item()
        msg = f"    pass1 epoch {ep}/{epochs1} | loss {tot / steps:.5f} | {time.time() - t0:.0f}s"
        print(msg, flush=True)
        log["pass1"].append({"epoch": ep, "loss": round(tot / steps, 5), "secs": round(time.time() - t0, 1)})
        torch.save({"state_dict": model.state_dict(),
                    "cfg": {"zone_dim": zone_dim, "d_model": d_model, "nhead": nhead,
                            "num_layers": layers, "dim_ff": ff}},
                   os.path.join(phase2_dir, "czet_phase2.pt"))

    edges = None
    for name in ("phase2_covariant_edges.parquet", "phase2_covariant_edges.csv"):
        p = os.path.join(phase2_dir, name)
        if os.path.exists(p):
            edges = read_table(p)
            break
    if edges is not None and len(edges):
        e = edges[(edges["src_idx"] < n_in) & (edges["dst_idx"] < n_in)]
        if len(e) > max_pairs > 0:
            e = e.sample(n=max_pairs, random_state=seed)
        print(f"[*] pass2 pairs={len(e):,}", flush=True)
        opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        src = e["src_idx"].to_numpy(np.int64)
        dst = e["dst_idx"].to_numpy(np.int64)
        sign = np.sign(e["cov_0_50"].to_numpy(np.float32))
        lock = np.clip(e["lock_50_100"].to_numpy(np.float32), -1, 1)
        steps2 = len(e) // batch_size
        for ep in range(1, epochs2 + 1):
            perm = np.random.permutation(len(e))
            tot, t0 = 0.0, time.time()
            for s in range(steps2):
                j = np.sort(perm[s * batch_size:(s + 1) * batch_size])
                qz = torch.from_numpy(np.ascontiguousarray(Q[src[j]]))
                qy = torch.from_numpy(np.ascontiguousarray(Q[dst[j]]))
                b_sign = torch.from_numpy(sign[j])
                b_lock = torch.from_numpy(lock[j])
                opt2.zero_grad()
                p34z, _, zz, anomz = model(qz)
                _, _, zy, _ = model(qy)
                cos = F.cosine_similarity(zz, zy * b_sign[:, None], dim=-1)
                loss_cov = F.mse_loss(cos, b_lock)
                w = torch.sigmoid(b_lock * 3.0)[:, None, None]
                loss_lock = torch.mean(w * (p34z - qz[:, 2:4, :]) ** 2)
                loss_prob = F.binary_cross_entropy_with_logits(anomz[:, 1], (b_lock >= 0.65).float())
                loss = loss_cov + 2.0 * loss_lock + 0.5 * loss_prob
                loss.backward()
                opt2.step()
                tot += loss.item()
            msg = f"    pass2 epoch {ep}/{epochs2} | loss {tot / max(1, steps2):.5f} | {time.time() - t0:.0f}s"
            print(msg, flush=True)
            log["pass2"].append({"epoch": ep, "loss": round(tot / max(1, steps2), 5),
                                 "secs": round(time.time() - t0, 1), "pairs": int(len(e))})
    torch.save({"state_dict": model.state_dict(),
                "cfg": {"zone_dim": zone_dim, "d_model": d_model, "nhead": nhead,
                        "num_layers": layers, "dim_ff": ff}},
               os.path.join(phase2_dir, "czet_phase2.pt"))
    with open(os.path.join(phase2_dir, "czet_train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    return model, log


def vault(phase2_dir, top_hubs=5000, thr=0.85, min_hits=10):
    scan = np.load(os.path.join(phase2_dir, "jump10_phase2_scan.npz"), mmap_mode="r")
    T = scan["trigger_vec_dense"]
    R = scan["resolution_vec_dense"]
    N = T.shape[0]
    n_in = int(N * 0.85)

    db = None
    for name in ("phase2_covariant_db.parquet", "phase2_covariant_db.csv"):
        p = os.path.join(phase2_dir, name)
        if os.path.exists(p):
            db = read_table(p)
            break
    if db is None or not len(db):
        print("[-] no covariant db; skipping vault", flush=True)
        return 0

    db = db[db["anchor_cell_id"] <= n_in].sort_values("anomaly_z_score", ascending=False)
    hubs = db.head(top_hubs).reset_index(drop=True)
    hub_idx = (hubs["anchor_cell_id"].to_numpy(np.int64) - 1)
    hub_idx = hub_idx[(hub_idx >= 0) & (hub_idx < n_in)]
    Th = np.ascontiguousarray(T[hub_idx])
    Rh = np.ascontiguousarray(R[hub_idx])
    held = scan["protected_held"]
    ret = scan["q4_net_return"]
    print(f"[*] vault: {len(hub_idx):,} hubs x {N - n_in:,} outbound triggers", flush=True)

    rows = []
    CH = 20000
    for h0 in range(0, len(hub_idx), 256):
        h1 = min(len(hub_idx), h0 + 256)
        sim_hits = [[] for _ in range(h1 - h0)]
        for s in range(n_in, N, CH):
            e = min(N, s + CH)
            sims = T[s:e] @ Th[h0:h1].T
            big = np.abs(sims) >= thr
            if big.any():
                ri, ci = np.nonzero(big)
                for r, c in zip(ri, ci):
                    sim_hits[c].append((s + r, sims[r, c]))
        for ci in range(h1 - h0):
            if not sim_hits[ci]:
                continue
            idx = np.fromiter((x[0] for x in sim_hits[ci]), dtype=np.int64, count=len(sim_hits[ci]))
            if len(idx) < min_hits:
                continue
            pol = np.fromiter((np.sign(x[1]) for x in sim_hits[ci]), dtype=np.float32, count=len(idx))
            h = h0 + ci
            cont = float(np.mean((R[idx] * pol[:, None]) @ Rh[h]))
            prot = float(np.mean(held[idx]))
            dirret = ret[idx] * pol
            hrow = hubs.iloc[h]
            expected = -1.0 if str(hrow["recommended_action"]) == "SELL_PROTECTED_HIGH" else 1.0
            dirwin = float(np.mean(np.sign(dirret) == expected))
            rows.append({
                "anchor_cell_id": int(hrow["anchor_cell_id"]),
                "outbound_triggers": int(len(idx)),
                "outbound_50_100_lock_corr": round(cont, 4),
                "outbound_protected_win_rate": round(prot, 4),
                "outbound_directional_win_rate": round(dirwin, 4),
                "outbound_verified_edge": round(max(prot, dirwin), 4),
            })
        print(f"    hubs {h0}-{h1} done", flush=True)

    v = pd.DataFrame(rows)
    if not v.empty:
        v = v.merge(hubs, on="anchor_cell_id", how="left", suffixes=("", "_db"))
        v = v[v["outbound_verified_edge"] >= 0.55].copy()
        nn = v["outbound_triggers"].to_numpy(float)
        pp = v["outbound_verified_edge"].clip(0.0, 1.0).to_numpy(float)
        zc = 1.959964
        lb = (pp + zc ** 2 / (2 * nn) - zc * np.sqrt(pp * (1 - pp) / nn + zc ** 2 / (4 * nn ** 2))) / (1 + zc ** 2 / nn)
        v["outbound_win_lb95"] = np.round(lb, 4)
        v["sample_flag"] = np.where(nn >= 30, "OK", "THIN")
        v = v.sort_values(["outbound_win_lb95", "outbound_triggers"], ascending=False).reset_index(drop=True)
        v.insert(0, "strategy_id", [f"STRAT_J10_{i + 1:04d}" for i in range(len(v))])
        v["phase3_status"] = "READY_FOR_EQUITY_SIM"
    out_p = os.path.join(phase2_dir, "phase2_strategy_vault.parquet")
    if pqcompat is not None and not v.empty:
        pqcompat.write_parquet(v, out_p)
    else:
        out_p = out_p.replace(".parquet", ".csv")
        v.to_csv(out_p, index=False)
    print(f"[+] vault: {len(v):,} verified strategies -> {out_p}", flush=True)
    try:
        with pd.ExcelWriter(os.path.join(phase2_dir, "phase2_anomaly_strategies.xlsx"),
                            engine="openpyxl") as w:
            v.to_excel(w, sheet_name="Phase3_Ready_Strategies", index=False)
            db.head(10000).to_excel(w, sheet_name="Inbound_Covariant_Hubs", index=False)
        print("[+] xlsx written", flush=True)
    except Exception as ex:
        print(f"[-] xlsx skipped: {type(ex).__name__} {ex}", flush=True)
    return len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2_dir", default=r"C:\cluster\phase2")
    ap.add_argument("--mode", default="both", choices=["train", "vault", "both"])
    ap.add_argument("--epochs_pass1", type=int, default=4)
    ap.add_argument("--epochs_pass2", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--dim_ff", type=int, default=768)
    ap.add_argument("--max_pairs", type=int, default=300000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--top_hubs", type=int, default=5000)
    ap.add_argument("--max_train_rows", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    if a.mode in ("train", "both"):
        train(a.phase2_dir, a.epochs_pass1, a.epochs_pass2, a.batch_size,
              a.d_model, a.layers, a.nhead, a.dim_ff, a.max_pairs,
              max_train_rows=a.max_train_rows, resume=a.resume)
    if a.mode in ("vault", "both"):
        vault(a.phase2_dir, a.top_hubs)


if __name__ == "__main__":
    main()
