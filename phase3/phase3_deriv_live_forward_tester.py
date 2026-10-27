"""PHASE 3B - Deriv live WebSocket forward tester with the 7 Fable adaptive guards.

Default mode is paper_proposal: subscribes to live ticks, evaluates the Phase 3 anchor
matches on every completed 75-tick window, queries REAL payouts through the Deriv
`proposal` endpoint, settles 75 ticks later from the live stream, and journals every
decision to SQLite. Live orders are only placed with --execute_demo_orders.

Credentials: never hardcoded.
  precedence: --api_token/--app_id  >  $env:DERIV_API_TOKEN/$env:DERIV_APP_ID  >  --creds_file
  --creds_file accepts any text file containing "API TOKEN:<tok>" and "APP ID:<id>" lines
  (e.g. the Phase 3 brief). The token is masked in every log line.

Usage
  python -u phase3_deriv_live_forward_tester.py --ping_test 20 --creds_file "...brief.md"
  python -u phase3_deriv_live_forward_tester.py --symbol JD10 --max_minutes 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, r"C:\cluster\phase2")
sys.path.insert(0, r"C:\cluster\phase3")
from phase2_dense_anomaly_bank import compute_quartile_brute_force_features  # noqa: E402

P3 = r"C:\cluster\phase3"
DB_PATH = os.path.join(P3, "fable_forward_cal.db")
SIM_MIN, SIM_MAX = 0.85, 0.955
SIGMA_LOOKBACK = 1800
SETTLE_TICKS = 75


def mask(tok: str) -> str:
    if not tok:
        return "<unset>"
    return tok[:8] + "..." + tok[-4:] if len(tok) > 14 else "***"


def load_creds(args) -> tuple[str, str]:
    app_id = args.app_id or os.environ.get("DERIV_APP_ID", "")
    token = args.api_token or os.environ.get("DERIV_API_TOKEN", "")
    if (not app_id or not token) and args.creds_file:
        try:
            text = open(args.creds_file, "r", encoding="utf-8", errors="ignore").read()
            if not app_id:
                m = re.search(r"APP\s*ID\s*[:\-]\s*([A-Za-z0-9]+)", text)
                app_id = m.group(1) if m else ""
            if not token:
                m = re.search(r"API\s*TOKEN\s*[:\-]\s*(pat_[A-Za-z0-9]+|[A-Za-z0-9]{32,})", text)
                token = m.group(1) if m else ""
        except OSError as e:
            print(f"[!] creds_file unreadable: {e}")
    return app_id, token


def wilson_lb(p: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    return (c - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


class FableAdaptiveGuards:
    """All 7 adaptive circuit breakers (SigmaGate, PayoutMonitor, LagMonitor,
    LiveCalibrator, DrawdownGuard, KellyStakeSizer, HourFilter)."""

    def __init__(self, cfg_path: str, db_path: str = DB_PATH):
        self.cfg = json.load(open(cfg_path, "r"))
        self.sigma_edges = np.array(self.cfg.get("sigma_quantile_edges_bps", [0.15, 0.20]))
        self.blocked_hours = set(int(h) for h in self.cfg.get("blocked_utc_hours", []))
        self.payout_floor = float(self.cfg.get("payout_floor_r", 1.90))
        self.session_stop = -400.0
        self.daily_stop = -800.0
        self.max_rtt_ms = 250.0
        self.consec_loss_limit = 8
        self.cooldown_s = 300.0

        self.rtts = deque(maxlen=40)
        self.session_pnl = 0.0
        self.daily_pnl = 0.0
        self.day = datetime.now(timezone.utc).date()
        self.consec_losses = 0
        self.pause_until = 0.0
        self.trades = []
        self.live_hits, self.live_exp = 0.0, 0.0
        self.db_path = db_path
        self._init_sqlite()

    def _init_sqlite(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        con = sqlite3.connect(self.db_path)
        con.execute("""CREATE TABLE IF NOT EXISTS forward_trades(
            ts_utc TEXT, symbol TEXT, anchor_cell_id INTEGER, contract TEXT,
            sigma_bps REAL, sim REAL, live_payout REAL, rtt_ms REAL,
            stake REAL, won INTEGER, pnl REAL, settle_spot REAL, note TEXT)""")
        con.execute("CREATE INDEX IF NOT EXISTS ix_ts ON forward_trades(ts_utc)")
        con.commit()
        con.close()
        self.db_ok = True

    def note_payout(self, r: float) -> bool:
        return r >= self.payout_floor

    def check_all(self, sigma_bps: float, rtt_ms: float, live_payout: float) -> tuple[bool, str]:
        now = time.time()
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day, self.daily_pnl = today, 0.0
        if self.session_pnl <= self.session_stop:
            return False, f"DRAWDOWN_HALT session={self.session_pnl:.2f}"
        if self.daily_pnl <= self.daily_stop:
            return False, f"DAILY_HALT daily={self.daily_pnl:.2f}"
        if now < self.pause_until:
            return False, f"CONSEC_LOSS_COOLDOWN {int(self.pause_until - now)}s"
        hr = datetime.now(timezone.utc).hour
        if hr in self.blocked_hours:
            return False, f"HOUR_FILTER {hr:02d}Z"
        self.rtts.append(rtt_ms)
        if len(self.rtts) >= 5 and float(np.median(self.rtts)) > self.max_rtt_ms:
            return False, f"LAG_GUARD median RTT={np.median(self.rtts):.0f}ms"
        if not (self.sigma_edges[0] <= sigma_bps <= self.sigma_edges[-1]):
            return False, f"SIGMA_GATE {sigma_bps:.3f}bps"
        if not self.note_payout(live_payout):
            return False, f"PAYOUT_MONITOR {live_payout:.3f}x < {self.payout_floor:.2f}x"
        if self.live_exp >= 25 and self.live_hits / self.live_exp < 0.40:
            return False, f"LIVE_CALIBRATOR hit={self.live_hits/self.live_exp:.2f} (degraded)"
        return True, "ALL_7_GUARDS_GREEN"

    def stake(self, balance: float, p: float, r: float) -> float:
        if r <= 1.0 or p <= 0:
            return 0.0
        f_star = (p * r - 1.0) / (r - 1.0)
        return float(np.clip(0.25 * f_star * balance, 0.35, 0.01 * balance))

    def record(self, row: dict):
        self.trades.append(row)
        self.session_pnl += row["pnl"]
        self.daily_pnl += row["pnl"]
        self.live_hits += row["won"]
        self.live_exp += 1
        if row["won"]:
            self.consec_losses = 0
        else:
            self.consec_losses += 1
            if self.consec_losses >= self.consec_loss_limit:
                self.pause_until = time.time() + self.cooldown_s
                self.consec_losses = 0
        con = sqlite3.connect(self.db_path)
        con.execute("INSERT INTO forward_trades VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["ts_utc"], row["symbol"], row["anchor_cell_id"], row["contract"],
                     row["sigma_bps"], row["sim"], row["live_payout"], row["rtt_ms"],
                     row["stake"], row["won"], row["pnl"], row["settle_spot"], row["note"]))
        con.commit()
        con.close()


class LiveMatcher:
    def __init__(self):
        w = np.load(os.path.join(P3, "phase3_whitening_transform.npz"))
        self.mu, self.sd = w["q_mu"], w["q_sd"]
        self.A = w["anchor_vecs"].astype(np.float64)
        self.An = self.A / np.linalg.norm(self.A, axis=1, keepdims=True)
        self.ids = w["anchor_ids"]
        self.actions = w["anchor_actions"]
        self.mode = "linear_ridge"
        self.W = w["W"] if "W" in w.files else None
        pt = os.path.join(P3, "phase3_live_bridge.pt")
        self.net = None
        if os.path.exists(pt):
            import torch
            ck = torch.load(pt, map_location="cpu")
            sd = ck["state_dict"] if "state_dict" in ck else ck
            sd = {k[4:] if k.startswith("net.") else k: v for k, v in sd.items()}
            self.net = torch.nn.Sequential(
                torch.nn.Linear(128, 512), torch.nn.GELU(), torch.nn.LayerNorm(512),
                torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.LayerNorm(512),
                torch.nn.Linear(512, 132))
            self.net.load_state_dict(sd)
            self.net.eval()
            self.mode = "mlp_cosine"
        meta = os.path.join(P3, "phase3_whitening_bridge.json")
        self.parity = json.load(open(meta)) if os.path.exists(meta) else {}

    def vector(self, y75: np.ndarray) -> np.ndarray:
        f = np.hstack([compute_quartile_brute_force_features(np.asarray(y75[:38])[None, :]),
                       compute_quartile_brute_force_features(np.asarray(y75[38:75])[None, :])])
        f = (f - self.mu) / self.sd
        if self.net is not None:
            import torch
            with torch.no_grad():
                v = self.net(torch.from_numpy(f.astype(np.float32))).numpy()[0].astype(np.float64)
            return v / max(np.linalg.norm(v), 1e-9)
        return (f @ self.W)[0]

    def match(self, v: np.ndarray):
        s = self.An @ (v / max(np.linalg.norm(v), 1e-9))
        a = np.abs(s)
        k = np.nonzero((a >= SIM_MIN) & (a <= SIM_MAX))[0]
        if not len(k):
            return None
        j = k[np.argmax(a[k])]
        return int(self.ids[j]), float(a[j]), float(np.sign(s[j])), str(self.actions[j])


def winsorized_sigma_bps(prices: np.ndarray) -> float:
    d = np.abs(np.diff(prices))
    if len(d) < 30:
        return float("nan")
    med = float(np.median(d))
    cap = 6.98438 * max(med, 1e-9)
    dw = np.clip(d, 0.0, cap)
    return float(np.std(dw) / max(abs(prices[-1]), 1e-12) * 1e4)


async def ws_request(ws, payload: dict, timeout=10.0) -> dict:
    await ws.send(json.dumps(payload))
    end = time.time() + timeout
    while time.time() < end:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=max(0.1, end - time.time())))
        if "error" in msg:
            return {"error": msg["error"]}
        for k in ("proposal", "ping", "authorize", "tick", "buy"):
            if k in msg:
                return msg
    return {"error": {"message": "timeout"}}


def proposal_payload(contract: str, symbol: str, stake: float, spot: float | None) -> dict | None:
    if contract in ("CALL", "PUT"):
        return {"proposal": 1, "amount": stake, "basis": "stake",
                "contract_type": contract, "currency": "USD", "duration": 75,
                "duration_unit": "s", "symbol": symbol}
    if contract == "ONETOUCH":
        return {"proposal": 1, "amount": stake, "basis": "stake",
                "contract_type": "ONETOUCH", "currency": "USD", "duration": 1,
                "duration_unit": "d", "symbol": symbol,
                "barrier": f"+{max(spot * 0.00015, 0.5):.2f}" if spot else "+1.00"}
    return None


async def ping_test(app_id: str, token: str, symbol: str, n: int = 20):
    import websockets
    url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    print(f"[*] PING TEST  url={url}  token={mask(token)}  symbol={symbol}")
    rtts = []
    async with websockets.connect(url, ping_interval=20, max_size=2**22) as ws:
        if token:
            r = await ws_request(ws, {"authorize": token})
            if "error" in r:
                print(f"[!] authorize failed: {r['error'].get('message')}")
            else:
                print(f"[+] authorized: loginid={r['authorize'].get('loginid')} "
                      f"balance={r['authorize'].get('balance')} {r['authorize'].get('currency')}")
        for i in range(n):
            t0 = time.perf_counter()
            r = await ws_request(ws, {"ping": 1})
            dt = (time.perf_counter() - t0) * 1000.0
            if "error" in r:
                print(f"  ping {i+1}: {r['error']}")
                continue
            rtts.append(dt)
            print(f"  ping {i+1:2d}: {dt:7.1f} ms")
            await asyncio.sleep(0.25)
        r = await ws_request(ws, {"ticks": symbol, "subscribe": 1})
        spot = None
        if "tick" in r:
            spot = float(r["tick"]["quote"])
            print(f"[+] {symbol} spot={spot}")
            await ws_request(ws, {"forget": r["tick"].get("id", "")}, timeout=2)
        for c in ("CALL", "PUT", "ONETOUCH"):
            p = proposal_payload(c, symbol, 1.0, spot or 1000.0)
            if not p:
                continue
            t0 = time.perf_counter()
            r = await ws_request(ws, p)
            dt = (time.perf_counter() - t0) * 1000.0
            if "error" in r:
                print(f"  proposal {c:9s}: ERROR {r['error'].get('message')}")
            else:
                pr = r["proposal"]
                payout = float(pr.get("payout", 0.0))
                print(f"  proposal {c:9s}: payout={payout:.3f} -> R={payout/1.0:.3f}x "
                      f"(dur={pr.get('display_value')}) {dt:.0f}ms")
    if rtts:
        med = float(np.median(rtts))
        verdict = "PASS" if med < 250 else "FAIL"
        print(f"[{verdict}] median RTT={med:.1f}ms  p95={np.percentile(rtts,95):.1f}ms  n={len(rtts)}")
        return med < 250
    return False


async def run_stream(app_id, token, symbol, cfg_path, minutes, execute_orders):
    import websockets
    guards = FableAdaptiveGuards(cfg_path)
    matcher = LiveMatcher()
    print(f"[*] matcher={matcher.mode}  parity={ {k: round(v,3) for k,v in matcher.parity.items() if isinstance(v,(int,float))} }")
    url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    ticks = deque(maxlen=SIGMA_LOOKBACK + SETTLE_TICKS + 5)
    pending = []
    balance = 50.0
    stop_at = time.time() + minutes * 60
    n_eval = n_trig = n_settled = 0

    async with websockets.connect(url, ping_interval=20, max_size=2**22) as ws:
        if token:
            await ws_request(ws, {"authorize": token})
        r = await ws_request(ws, {"ticks": symbol, "subscribe": 1})
        if "error" in r:
            print(f"[!] subscribe failed: {r['error']}")
            return
        print(f"[+] subscribed {symbol}; buffer warmup {SIGMA_LOOKBACK} ticks")
        while time.time() < stop_at:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            if "tick" not in msg:
                continue
            q = float(msg["tick"]["quote"])
            ticks.append(q)
            for p in list(pending):
                p["age"] += 1
                if p["age"] >= SETTLE_TICKS:
                    pending.remove(p)
                    if p["contract"] in ("CALL", "PUT"):
                        won = (q > p["spot"]) if p["contract"] == "CALL" else (q < p["spot"])
                    else:
                        won = abs(q - p["spot"]) >= p["barrier_pts"]
                    pnl = p["stake"] * (p["live_payout"] - 1.0) if won else -p["stake"]
                    balance += pnl
                    guards.record({"ts_utc": p["ts"], "symbol": symbol,
                                   "anchor_cell_id": p["anchor"], "contract": p["contract"],
                                   "sigma_bps": p["sigma"], "sim": p["sim"],
                                   "live_payout": p["live_payout"], "rtt_ms": p["rtt"],
                                   "stake": p["stake"], "won": int(won), "pnl": pnl,
                                   "settle_spot": q, "note": p["note"]})
                    n_settled += 1
                    print(f"  [SETTLE] {p['contract']:9s} anchor={p['anchor']:6d} "
                          f"{'WIN ' if won else 'LOSS'} pnl={pnl:+.2f} bal={balance:.2f}")
            if len(ticks) < 76:
                continue
            n_eval += 1
            if n_eval % 8:
                continue
            window = np.fromiter(list(ticks)[-75:], dtype=np.float64, count=75)
            sigma = winsorized_sigma_bps(np.fromiter(list(ticks)[-SIGMA_LOOKBACK:],
                                                     dtype=np.float64,
                                                     count=min(len(ticks), SIGMA_LOOKBACK)))
            if not np.isfinite(sigma):
                continue
            v = matcher.vector(window)
            m = matcher.match(v)
            if m is None:
                continue
            anchor, sim, polarity, action = m
            want = "CALL" if action.upper().startswith("BUY") else "PUT"
            t0 = time.perf_counter()
            pr = await ws_request(ws, proposal_payload(want, symbol, 1.0, q))
            rtt = (time.perf_counter() - t0) * 1000.0
            if "error" in pr:
                continue
            live_R = float(pr["proposal"].get("payout", 0.0)) / 1.0
            ok, reason = guards.check_all(sigma, rtt, live_R)
            if not ok:
                print(f"  [BLOCK] anchor={anchor} sim={sim:.3f} R={live_R:.3f} -> {reason}")
                continue
            n_trig += 1
            stake = guards.stake(balance, guards.live_hits / max(1, guards.live_exp) if guards.live_exp else 0.5, live_R)
            pending.append({"age": 0, "spot": q, "stake": stake if execute_orders else 0.0,
                            "anchor": anchor, "contract": want, "sigma": sigma, "sim": sim,
                            "live_payout": live_R, "rtt": rtt, "note": action,
                            "barrier_pts": q * 0.00015, "ts": datetime.now(timezone.utc).isoformat()})
            print(f"  [TRIGGER] {want} anchor={anchor} sim={sim:.3f} R={live_R:.3f}x "
                  f"sigma={sigma:.3f}bps rtt={rtt:.0f}ms stake={stake:.2f}")

    p = guards.session_pnl
    print(f"[+] session done: evals={n_eval} triggers={n_trig} settled={n_settled} "
          f"pnl={p:+.2f} balance={balance:.2f} db={DB_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app_id", default=None)
    ap.add_argument("--api_token", default=None)
    ap.add_argument("--creds_file", default=None)
    ap.add_argument("--symbol", default="JD10")
    ap.add_argument("--ping_test", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--execute_demo_orders", action="store_true")
    ap.add_argument("--self_check", action="store_true")
    a = ap.parse_args()

    if a.self_check:
        g = FableAdaptiveGuards(os.path.join(P3, "phase3_live_forward_config.json"))
        mt = LiveMatcher()
        y = np.cumsum(np.random.default_rng(0).normal(0, 1, 75))
        v = mt.vector(y)
        print(f"[+] self-check OK: matcher={mt.mode} vec_dim={v.shape[0]} "
              f"sqlite={g.db_ok} anchors={len(mt.ids)} blocked_hours={len(g.blocked_hours)}")
        return

    app_id, token = load_creds(a)
    if not app_id or not token:
        print("[!] no credentials: pass --app_id/--api_token, set DERIV_APP_ID/DERIV_API_TOKEN, "
              "or --creds_file <file containing 'API TOKEN:' and 'APP ID:'>")
        return 2
    print(f"[*] app_id={app_id} token={mask(token)}")
    if a.ping_test:
        ok = asyncio.run(ping_test(app_id, token, a.symbol, a.ping_test))
        return 0 if ok else 1
    asyncio.run(run_stream(app_id, token, a.symbol,
                           os.path.join(P3, "phase3_live_forward_config.json"),
                           a.minutes, a.execute_demo_orders))
    return 0


if __name__ == "__main__":
    sys.exit(main())
