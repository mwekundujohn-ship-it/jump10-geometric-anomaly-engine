"""Item 03 - crawled mathematical trajectory operators.

Computes the literature-derived invariants that were NOT in the original 12-angle
brute-force bank, over the strictly causal 0-50% window of every cell, and writes
crawled_math_operators.npz + crawled_math_operators_manifest.json.

Operators
  HVG degree entropy + entropy gap vs the iid-exponential law  (Luque et al., PRE 80, 046103)
  Hilbert-Huang instantaneous frequency mean/std               (Huang et al., Proc. R. Soc. A 454, 903)
  Normalized Levy area of the (t, y) path                      (Chevyrev & Kormilitzin, arXiv:1603.03788, eq. 12)
  RQA determinism / laminarity                                 (Marwan et al., Phys. Rep. 438, 237)
  Ordinal permutation entropy                                  (Bandt & Pompe, PRL 88, 174102)
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pqcompat
from scipy.signal import hilbert
from phase2_dense_anomaly_bank import _rqa_features, _ordinal_features

BASE = r"C:\cluster\phase2"
P1 = r"C:\cluster\phase1\final\jump10_phase1_fourier_functions.npz"
N = 591171
WIN = 75
CHUNK = 4000

K = np.arange(2, 200)
P_TH = (1.0 / 3.0) * (2.0 / 3.0) ** (K - 2)
H_TH = float(-np.sum(P_TH * np.log2(P_TH)))


def hvg_degree(y):
    """Exact horizontal-visibility degree (all lags) via next/prev greater-or-equal."""
    n, L = y.shape
    i = np.arange(L)
    upper = np.triu(np.ones((L, L), dtype=bool), 1)[None]
    lower = np.tril(np.ones((L, L), dtype=bool), -1)[None]
    nxt = (y[:, None, :] >= y[:, :, None]) & upper
    prv = (y[:, :, None] >= y[:, None, :]) & lower
    has_n = nxt.any(2)
    has_p = prv.any(2)
    jn = np.argmax(nxt, axis=2)
    jp = L - 1 - np.argmax(prv[:, :, ::-1], axis=2)
    deg = (jn - i[None, :] - 1) + (i[None, :] - jp - 1) + has_n.astype(np.int32) + has_p.astype(np.int32)
    return np.clip(deg, 0, L - 1).astype(np.float32)


def hvg_entropy(deg):
    n, length = deg.shape
    hist = np.zeros((n, length + 1), dtype=np.float32)
    rows = np.repeat(np.arange(n), length)
    np.add.at(hist, (rows, deg.astype(np.int32).ravel()), 1.0)
    p = hist / np.maximum(hist.sum(1, keepdims=True), 1e-9)
    return -np.sum(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0.0), axis=1)


def main():
    npz_out = os.path.join(BASE, "crawled_math_operators.npz")
    px = np.load(P1, mmap_mode="r")["raw_tick_cells"]
    names = ["hvg_degree_entropy", "hvg_entropy_gap", "hvg_max_degree",
             "hh_inst_freq_mean", "hh_inst_freq_std",
             "levy_area_z", "rqa_det", "rqa_lam", "perm_entropy"]
    arrs = {}
    cached = os.path.exists(npz_out) and "--force" not in sys.argv
    if cached:
        d = np.load(npz_out)
        arrs = {n: d[n] for n in names}
        print("[*] using cached crawled_math_operators.npz", flush=True)
    for n in names:
        if not cached:
            arrs[n] = np.lib.format.open_memmap(os.path.join(BASE, f"_op_{n}.npy"), mode="w+",
                                                dtype=np.float32, shape=(N,))
    t = np.linspace(0.0, 1.0, WIN)
    for s in ([] if cached else range(0, N, CHUNK)):
        e = min(N, s + CHUNK)
        p = np.asarray(px[s:e, :WIN], dtype=np.float64)
        y = p - p.mean(1, keepdims=True)
        y = y / (y.std(1, keepdims=True) + 1e-12)
        dy = np.diff(y, axis=1)
        dt = 1.0 / (WIN - 1)
        area = 0.5 * (np.sum(t[:-1] * dy, axis=1) - np.sum(y[:, :-1] * dt, axis=1))
        arrs["levy_area_z"][s:e] = area / (np.abs(dy).sum(1) + 1e-9)

        ana = hilbert(y, axis=1)
        inst = np.diff(np.unwrap(np.angle(ana), axis=1), axis=1) / (2 * np.pi * dt)
        arrs["hh_inst_freq_mean"][s:e] = inst.mean(1) / 100.0
        arrs["hh_inst_freq_std"][s:e] = inst.std(1) / 100.0

        rqa, _ = _rqa_features(y)
        arrs["rqa_det"][s:e] = rqa[:, 1]
        arrs["rqa_lam"][s:e] = rqa[:, 2]

        ord_feat = _ordinal_features(y)
        arrs["perm_entropy"][s:e] = ord_feat[:, 6]

        deg = hvg_degree(y)
        h = hvg_entropy(deg)
        arrs["hvg_degree_entropy"][s:e] = h
        arrs["hvg_entropy_gap"][s:e] = h - H_TH
        arrs["hvg_max_degree"][s:e] = deg.max(1)
        print(f"    operators {e:,}/{N:,}", flush=True)

    if not cached:
        for n in names:
            arrs[n].flush()
        np.savez(npz_out, **{n: np.asarray(arrs[n]) for n in names})

    vault = pqcompat.read_parquet(os.path.join(BASE, "phase2_strategy_vault.parquet"))
    ai = vault.anchor_cell_id.to_numpy(np.int64) - 1
    assoc = {}
    for n in names:
        v = np.asarray(arrs[n])[ai]
        assoc[n] = {
            "min": round(float(np.nanmin(v)), 5), "median": round(float(np.nanmedian(v)), 5),
            "max": round(float(np.nanmax(v)), 5),
            "corr_vs_outbound_edge": round(float(np.corrcoef(v, vault.outbound_verified_edge.to_numpy(float))[0, 1]), 4),
        }
        ptmp = os.path.join(BASE, f"_op_{n}.npy")
        if os.path.exists(ptmp):
            try:
                os.remove(ptmp)
            except Exception:
                pass

    manifest = {
        "generated": "2026-10-25",
        "purpose": "Literature-sourced trajectory invariants crawled for Phase 2 item 03 and computed on the "
                   "causal 0-50% window (ticks 1..75) of all 591,171 cells.",
        "window": {"start_tick": 1, "end_tick": 75, "length": WIN,
                   "preprocessing": "per-window mean-centering and unit-variance scaling (df=74)"},
        "operators": [
            {"name": "hvg_degree_entropy", "family": "complex networks",
             "formula": "H(d) = -sum_k p_k log2 p_k over the horizontal-visibility-graph degree distribution",
             "reference": "Luque, Lacasa, Ballesteros & Luque, Phys. Rev. E 80, 046103 (2009), doi:10.1103/PhysRevE.80.046103",
             "note": "iid series produce P(k)=(1/3)(2/3)^(k-2) regardless of amplitude law"},
            {"name": "hvg_entropy_gap", "family": "complex networks",
             "formula": "H_empirical - H_theory with H_theory = %.6f bits for the iid exponential law" % H_TH,
             "reference": "same as hvg_degree_entropy", "note": "positive gap = excess order vs white noise"},
            {"name": "hvg_max_degree", "family": "complex networks", "formula": "max_i deg(i)",
             "reference": "same as hvg_degree_entropy"},
            {"name": "hh_inst_freq_mean", "family": "Hilbert-Huang",
             "formula": "mean of d/dt arg(Hilbert(y)) / 2*pi", "reference": "Huang et al., Proc. R. Soc. A 454, 903 (1998)"},
            {"name": "hh_inst_freq_std", "family": "Hilbert-Huang",
             "formula": "std of the instantaneous frequency", "reference": "same as hh_inst_freq_mean"},
            {"name": "levy_area_z", "family": "rough paths",
             "formula": "A = 0.5*(int t dy - int y dt), normalized by total variation",
             "reference": "Chevyrev & Kormilitzin, arXiv:1603.03788 eq. (12); Lyons (1998) rough path theory"},
            {"name": "rqa_det", "family": "recurrence quantification",
             "formula": "DET = sum(diagonal recurrence lengths >= 2) / total recurrence (16-pt delay embedding)",
             "reference": "Marwan, Romano, Thiel & Kurths, Phys. Rep. 438, 237 (2007)"},
            {"name": "rqa_lam", "family": "recurrence quantification",
             "formula": "LAM = vertical-line recurrence / total recurrence", "reference": "same as rqa_det"},
            {"name": "perm_entropy", "family": "ordinal statistics",
             "formula": "H = -sum p(pi) ln p(pi) / ln 6 over the 6 ordinal patterns of order 3",
             "reference": "Bandt & Pompe, Phys. Rev. Lett. 88, 174102 (2002)"},
        ],
        "arrays": {"path": "crawled_math_operators.npz", "rows": N, "dtype": "float32", "columns": names},
        "vault_association": assoc,
    }
    with open(os.path.join(BASE, "crawled_math_operators_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"H_theory": round(H_TH, 4), "association": assoc}, indent=2), flush=True)


if __name__ == "__main__":
    main()
