"""Build the compact (~640 MB) scan-only cache that the workers need.

Only the arrays the covariant scanner touches are copied, so the 2.59 GB full
cache never has to cross the LAN.
"""
import json
import os
import numpy as np

SRC = r"C:\cluster\phase2\jump10_phase2_subzones.npz"
DST = r"C:\cluster\phase2\jump10_phase2_scan.npz"
KEYS = ["trigger_vec_dense", "resolution_vec_dense", "protected_held",
        "ext_polarity", "q4_net_return", "ext_tick", "phase_closure_abs_cos"]

z = np.load(SRC, mmap_mode="r")
want = [k for k in KEYS if k in z.files]
print("[*] source keys used:", want, flush=True)
meta = {}
for k in want:
    meta[k] = {"shape": list(z[k].shape), "dtype": str(z[k].dtype)}
np.savez(DST, **{k: np.asarray(z[k]) for k in want})
size = os.path.getsize(DST)
print(f"[+] {DST}  {size/1e6:.1f} MB", flush=True)
meta["bytes"] = size
with open(r"C:\cluster\phase2\scan_cache_manifest.json", "w") as f:
    json.dump(meta, f, indent=2)
chk = np.load(DST, mmap_mode="r")
print("[+] verify:", {k: (chk[k].shape, str(chk[k].dtype)) for k in chk.files}, flush=True)
