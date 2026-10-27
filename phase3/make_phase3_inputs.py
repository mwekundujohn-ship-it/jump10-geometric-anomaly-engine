"""Builds the compact Phase 3 worker inputs (one file, no parquet dependency)."""
import os
import sys
import numpy as np

BASE3 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE3)
sys.path.insert(0, r"C:\cluster\phase2")
import pqcompat

P1 = r"C:\cluster\phase1\final\jump10_phase1_fourier_functions.npz"
P2 = r"C:\cluster\phase2"
CAT = r"C:\cluster\phase1\final\jump10_phase1_cell_catalog.parquet"

out = os.path.join(BASE3, "phase3_inputs.npz")
raw = np.load(P1, mmap_mode="r")["raw_tick_cells"]
z = np.load(os.path.join(P2, "jump10_phase2_subzones.npz"), mmap_mode="r")
T = z["trigger_vec_dense"]
N = raw.shape[0]
print(f"[*] raw {raw.shape} T {T.shape}")

np.savez(out,
         raw=np.asarray(raw, dtype=np.float32),
         trigger_vec=np.asarray(T, dtype=np.float32))
print(f"[+] {out} {os.path.getsize(out) / 1e6:.0f} MB")

cat = pqcompat.read_parquet(CAT)
t = cat["time_open_ms"].to_numpy(np.int64) if "time_open_ms" in cat.columns else cat.iloc[:, 4].to_numpy(np.int64)
np.save(os.path.join(BASE3, "phase3_catalog_times.npy"), t)
print(f"[+] times {t.shape} {t[:3].tolist()}")

v = pqcompat.read_parquet(os.path.join(P2, "phase2_strategy_vault.parquet"))
v.to_csv(os.path.join(BASE3, "phase3_vault.csv"), index=False)
ss = v[(v.covariant_peer_count >= 12) & (v.covariant_peer_count <= 400) &
       (v.mean_0_50_similarity >= 0.85) & (v.mean_0_50_similarity <= 0.955)]
print(f"[+] vault {len(v)} rows; sweet-spot {len(ss)}; OK-flag {int((ss.sample_flag == 'OK').sum())}")
