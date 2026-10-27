"""Build the compact CZET training cache from the whitened Phase-2 bank.

Q_seq(cell, quartile, feature) = [desc_Qk (17) | dense_qk padded to 55]
Out: C:\\cluster\\phase2\\czet_train_cache.npz (float32, uncompressed for mmap)
"""
import os
import numpy as np

BASE = r"C:\cluster\phase2"
SRC = os.path.join(BASE, "jump10_phase2_subzones.npz")
OUT = os.path.join(BASE, "czet_train_cache.npz")
CHUNK = 100_000
DENSE_WIDTH = 55


def main():
    z = np.load(SRC)
    N = z["desc_Q1"].shape[0]
    W = z["desc_Q1"].shape[1] + DENSE_WIDTH
    print(f"[*] N={N:,} per-quartile width={W}", flush=True)
    arrs = {
        "Q_seq": np.lib.format.open_memmap(os.path.join(BASE, "_q_seq_tmp.npy"),
                                          mode="w+", dtype=np.float32, shape=(N, 4, W)),
    }
    for k in ("protected_held", "ext_polarity"):
        arrs[k] = np.lib.format.open_memmap(os.path.join(BASE, f"_{k}_tmp.npy"),
                                            mode="w+", dtype=np.int8, shape=(N,))
    arrs["ext_tick"] = np.lib.format.open_memmap(os.path.join(BASE, "_ext_tick_tmp.npy"),
                                                mode="w+", dtype=np.int16, shape=(N,))
    for k in ("q4_net_return", "phase_closure_abs_cos", "bicoherence"):
        arrs[k] = np.lib.format.open_memmap(os.path.join(BASE, f"_{k}_tmp.npy"),
                                            mode="w+", dtype=np.float32, shape=(N,))

    for s in range(0, N, CHUNK):
        e = min(N, s + CHUNK)
        qs = []
        for i in range(1, 5):
            d = z[f"desc_Q{i}"][s:e]
            dq = z[f"dense_q{i}"][s:e]
            pad = np.zeros((e - s, W - d.shape[1]), dtype=np.float32)
            pad[:, :dq.shape[1]] = dq
            qs.append(np.concatenate([d, pad], axis=1))
        arrs["Q_seq"][s:e] = np.stack(qs, axis=1)
        for k in ("protected_held", "ext_polarity", "ext_tick",
                  "q4_net_return", "phase_closure_abs_cos", "bicoherence"):
            arrs[k][s:e] = z[k][s:e]
        print(f"    {e:,}/{N:,}", flush=True)

    for k, a in arrs.items():
        a.flush()
    np.savez(OUT, **{k: np.asarray(a) for k, a in arrs.items()})
    print(f"[+] {OUT} ({os.path.getsize(OUT)/1e9:.2f} GB) keys={list(arrs)}", flush=True)
    for k, a in arrs.items():
        p = os.path.join(BASE, os.path.basename(a.filename))
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
