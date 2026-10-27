import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

CELL_SIZE = 150
FPS = 20
HOLD_FRAMES = 24
TOP_CIRCLES = 24
TOP_TERMS = 8
SAMPLE_CELLS = [1, 42906, 85811, 128716, 171621]


def fmt_k(k):
    k = int(k)
    return f"+{k}" if k > 0 else str(k)


def load_catalog(path, cells):
    catalog = pd.read_parquet(path, engine="fastparquet")
    sample = catalog[catalog["cell_id"].isin(cells)].copy()
    sample = sample.set_index("cell_id").loc[cells].reset_index()
    return catalog, sample


def load_sample_arrays(npz_path, cells):
    with np.load(npz_path, allow_pickle=False) as npz:
        ids = npz["cell_id"].astype(np.int64)
        positions = np.searchsorted(ids, np.asarray(cells, dtype=np.int64))
        ck = npz["c_k"][positions].copy()
        raw = npz["raw_tick_cells"][positions].copy()
        stored_ids = ids[positions]
    if not np.array_equal(stored_ids, np.asarray(cells, dtype=np.int64)):
        raise ValueError(f"cell id mismatch: {stored_ids} != {cells}")
    return ck, raw


def analyze_cell(ck, raw, p_mean, p_std, y_open, slope):
    n = np.arange(CELL_SIZE, dtype=np.float64)
    t = n / (CELL_SIZE - 1)
    theta = 2.0 * np.pi * n / CELL_SIZE
    y_norm = (raw - p_mean) / p_std
    y_bridge = y_norm - (y_norm[0] + t * (y_norm[-1] - y_norm[0]))
    z = np.cos(theta) + 1j * y_bridge
    ck_direct = np.fft.fft(z) / CELL_SIZE
    coeff_err = float(np.max(np.abs(ck_direct - ck)))
    z_rec = np.fft.ifft(ck * CELL_SIZE)
    y_bridge_rec = z_rec.imag
    y_norm_rec = y_bridge_rec + y_open + t * slope
    price_rec = y_norm_rec * p_std + p_mean
    price_err = float(np.max(np.abs(price_rec - raw)))
    norm_rmse = float(np.sqrt(np.mean((y_norm_rec - y_norm) ** 2)))
    meta_err = float(max(abs(y_norm[0] - y_open), abs((y_norm[-1] - y_norm[0]) - slope)))
    return {
        "ck_direct": ck_direct,
        "y_norm": y_norm,
        "y_bridge": y_bridge,
        "z": z,
        "z_rec": z_rec,
        "y_norm_rec": y_norm_rec,
        "price_rec": price_rec,
        "coeff_err": coeff_err,
        "price_err": price_err,
        "norm_rmse": norm_rmse,
        "meta_err": meta_err,
        "t": t,
        "theta": theta,
    }


def top_terms(ck, k_idx, count):
    order = np.argsort(-np.abs(ck))[:count]
    terms = []
    for j in order:
        k = int(k_idx[j])
        c = complex(ck[j])
        terms.append({
            "k": k,
            "radius": float(abs(c)),
            "phase_rad": float(np.angle(c)),
            "phase_deg": float(np.degrees(np.angle(c))),
            "real": float(c.real),
            "imag": float(c.imag),
        })
    return terms


def figure_price_paths(sample, data, out_png):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 15), constrained_layout=True)
    gs = fig.add_gridspec(6, 1, height_ratios=[1.15, 1, 1, 1, 1, 1])
    ax_c = fig.add_subplot(gs[0])
    for i, (_, row) in enumerate(sample.iterrows()):
        color = "#00ff9d" if row["p_close"] >= row["p_open"] else "#ff4d6d"
        ax_c.vlines(i, row["p_low"], row["p_high"], color=color, lw=2.0)
        ax_c.vlines(i, min(row["p_open"], row["p_close"]), max(row["p_open"], row["p_close"]), color=color, lw=9.0)
        ax_c.scatter([i], [row["p_open"]], color="#c7f9ff", s=22, zorder=3)
        ax_c.scatter([i], [row["p_close"]], color="#ffd166", s=22, zorder=3)
        ax_c.text(i, row["p_high"], f"{row['p_close']:.1f}", ha="center", va="bottom", fontsize=8, color="#dddddd")
    ax_c.set_xticks(range(len(sample)))
    ax_c.set_xticklabels([f"Cell_{int(c)}" for c in sample["cell_id"]], rotation=0)
    ax_c.set_ylabel("Price")
    ax_c.set_title("XFlip sample: one candle per 150-tick cell (open / high / low / close)")
    ax_c.grid(alpha=0.2)

    for i, (_, row) in enumerate(sample.iterrows()):
        ax = fig.add_subplot(gs[i + 1])
        d = data[i]
        ticks = np.arange(1, CELL_SIZE + 1)
        ax.plot(ticks, row["raw"], color="#8f8f8f", lw=2.2, label="raw mid price")
        ax.plot(ticks, d["price_rec"], color="#00e5ff", lw=1.2, ls="--", label="Fourier reconstruction")
        ax.scatter([1], [row["p_open"]], color="#00ff9d", s=34, zorder=4, label="open")
        ax.scatter([CELL_SIZE], [row["p_close"]], color="#ff4d6d", s=34, zorder=4, label="close")
        ax.text(
            0.01, 0.95, f"Cell_{int(row['cell_id'])}   max|fit-raw| = {d['price_err']:.3e}",
            transform=ax.transAxes, va="top", fontsize=9, color="#ffffff",
        )
        ax.set_xlim(0.5, CELL_SIZE + 0.5)
        pad = (row["p_high"] - row["p_low"]) * 0.12 + 1e-9
        ax.set_ylim(row["p_low"] - pad, row["p_high"] + pad)
        ax.set_ylabel("Price")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        if i == len(sample) - 1:
            ax.set_xlabel("Tick within cell (1 = open, 150 = close)")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def figure_functions(sample, data, k_idx, out_png):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(len(sample), 2, figsize=(16, 14), constrained_layout=True)
    for i, (_, row) in enumerate(sample.iterrows()):
        d = data[i]
        ck = row["ck"]
        order = np.argsort(k_idx)
        k_sorted = k_idx[order]
        r_sorted = np.abs(ck[order])
        terms = top_terms(ck, k_idx, TOP_TERMS)
        top_k = np.array([t["k"] for t in terms])
        top_r = np.array([t["radius"] for t in terms])
        top_p = np.array([t["phase_rad"] for t in terms])

        ax = axes[i, 0]
        ax.semilogy(k_sorted, np.maximum(r_sorted, 1e-16), ".", color="#8f8f8f", ms=4)
        ax.stem(top_k, np.maximum(top_r, 1e-16), linefmt="#ffd166", markerfmt="o", basefmt=" ")
        ax.set_ylabel("|c_k|")
        ax.set_xlabel("k")
        ax.set_title(f"Cell_{int(row['cell_id'])}  |c_k| spectrum (all 150)")
        ax.grid(alpha=0.2)

        ax = axes[i, 1]
        ax.stem(top_k, top_p, linefmt="#00e5ff", markerfmt="o", basefmt=" ")
        ax.axhline(0, color="#666666", lw=0.6)
        ax.set_ylabel("phase (rad)")
        ax.set_xlabel("k")
        ax.set_title(f"Top {TOP_TERMS} phases | max|coeff err|={d['coeff_err']:.2e}")
        ax.grid(alpha=0.2)
        text = "\n".join(
            f"k={fmt_k(t['k'])}  R={t['radius']:.4f}  phi={t['phase_rad']:+.3f}"
            for t in terms[:4]
        )
        ax.text(
            0.02, 0.03, text, transform=ax.transAxes, va="bottom", fontsize=8,
            family="monospace", color="#dddddd",
        )
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def figure_epicycles(cell_id, row, d, ck, k_idx, out_gif):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])
    ax_price = fig.add_subplot(gs[0, 0])
    ax_bridge = fig.add_subplot(gs[1, 0])
    ax_epi = fig.add_subplot(gs[:, 1])
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08, wspace=0.18, hspace=0.25)

    ticks = np.arange(1, CELL_SIZE + 1)
    ax_price.plot(ticks, row["raw"], color="#8f8f8f", lw=2.0, label="raw")
    price_line, = ax_price.plot([], [], color="#00e5ff", lw=1.8, label="Fourier")
    ax_price.scatter([1], [row["p_open"]], color="#00ff9d", s=38, zorder=4)
    ax_price.scatter([CELL_SIZE], [row["p_close"]], color="#ff4d6d", s=38, zorder=4)
    ax_price.set_xlim(0.5, CELL_SIZE + 0.5)
    pad = (row["p_high"] - row["p_low"]) * 0.15 + 1e-9
    ax_price.set_ylim(row["p_low"] - pad, row["p_high"] + pad)
    ax_price.set_ylabel("Price")
    ax_price.set_title(f"Cell_{cell_id}: price path  (open -> close)")
    ax_price.grid(alpha=0.2)
    ax_price.legend(loc="upper right", fontsize=8)

    ax_bridge.plot(ticks, d["y_bridge"], color="#8f8f8f", lw=1.6, label="bridge target")
    bridge_line, = ax_bridge.plot([], [], color="#ffd166", lw=1.6, label="Fourier")
    ax_bridge.set_xlim(0.5, CELL_SIZE + 0.5)
    bpad = max(abs(d["y_bridge"]).max(), 0.1) * 0.25
    ax_bridge.set_ylim(d["y_bridge"].min() - bpad, d["y_bridge"].max() + bpad)
    ax_bridge.set_ylabel("y_bridge (normalized)")
    ax_bridge.set_xlabel("Tick within cell")
    ax_bridge.set_title("Fourier signal being reconstructed")
    ax_bridge.grid(alpha=0.2)
    ax_bridge.legend(loc="upper right", fontsize=8)

    z_all = d["z"]
    lim = max(1.35, 1.15 * np.max(np.abs(z_all)), 1.1 * np.max(np.abs(ck)) + np.max(np.abs(z_all)))
    ax_epi.set_xlim(-lim, lim)
    ax_epi.set_ylim(-lim, lim)
    ax_epi.set_aspect("equal", adjustable="box")
    ax_epi.set_title("Epicycles: z(t) = sum c_k exp(i 2*pi*k*t/150)")
    ax_epi.grid(alpha=0.12)
    ax_epi.plot(z_all.real, z_all.imag, color="#ffffff", alpha=0.18, lw=1.0, ls="--")
    chain_line, = ax_epi.plot([], [], color="#ffffff", alpha=0.45, lw=0.9)
    trail_line, = ax_epi.plot([], [], color="#00e5ff", alpha=0.85, lw=1.5)
    point, = ax_epi.plot([], [], "o", color="#00e5ff", ms=6, zorder=8)

    top_idx = np.argsort(-np.abs(ck))[:TOP_CIRCLES]
    circles = []
    vec_lines = []
    for j in top_idx:
        circ = Circle((0.0, 0.0), float(abs(ck[j])), fill=False, ec="#ffd166", lw=0.8, alpha=0.55)
        ax_epi.add_patch(circ)
        circles.append(circ)
        line, = ax_epi.plot([], [], color="#ffd166", lw=1.0, alpha=0.8)
        vec_lines.append(line)

    terms = top_terms(ck, k_idx, 6)
    formula = "\n".join(
        f"k={fmt_k(t['k'])}  R={t['radius']:.4f}  phi={t['phase_rad']:+.3f}"
        for t in terms
    )
    info = ax_epi.text(
        0.02, 0.98,
        f"Cell_{cell_id}\nmax|coeff err|={d['coeff_err']:.2e}\nmax|price err|={d['price_err']:.2e}\n\n{formula}",
        transform=ax_epi.transAxes, va="top", ha="left", fontsize=8,
        family="monospace", color="#ffffff",
        bbox=dict(boxstyle="round,pad=0.45", fc="#000000", ec="#444444", alpha=0.65),
    )
    progress = ax_epi.text(
        0.98, 0.02, "", transform=ax_epi.transAxes, va="bottom", ha="right",
        fontsize=9, color="#00e5ff",
    )

    def update(frame):
        n = min(frame, CELL_SIZE - 1)
        vecs = ck * np.exp(1j * 2.0 * np.pi * k_idx * n / CELL_SIZE)
        pts = np.concatenate([[0.0 + 0.0j], np.cumsum(vecs)])
        chain_line.set_data(pts.real, pts.imag)
        for j, line, circ in zip(top_idx, vec_lines, circles):
            x0, y0 = pts[j].real, pts[j].imag
            x1, y1 = pts[j + 1].real, pts[j + 1].imag
            line.set_data([x0, x1], [y0, y1])
            circ.center = (x0, y0)
        trail_line.set_data(z_all[:n + 1].real, z_all[:n + 1].imag)
        point.set_data([pts[-1].real], [pts[-1].imag])
        price_line.set_data(ticks[:n + 1], d["price_rec"][:n + 1])
        y_open = float(row["y_open_norm"])
        slope = float(row["bridge_slope"])
        bridge_target = y_open + d["t"] * slope
        bridge_line.set_data(ticks[:n + 1], (d["y_norm_rec"] - bridge_target)[:n + 1])
        progress.set_text(f"t = {n + 1:3d}/150")
        return chain_line, trail_line, point, price_line, bridge_line, progress, *vec_lines, *circles

    total_frames = CELL_SIZE + HOLD_FRAMES
    anim = FuncAnimation(
        fig, update, frames=total_frames, interval=1000.0 / FPS,
        blit=False, repeat=False, cache_frame_data=False,
    )
    anim.save(out_gif, writer=PillowWriter(fps=FPS), dpi=85)
    plt.close(fig)


def write_functions_csv(sample, data, k_idx, path):
    rows = []
    for i, (_, row) in enumerate(sample.iterrows()):
        ck = row["ck"]
        order = np.argsort(-np.abs(ck))
        rank = np.empty(len(ck), dtype=int)
        rank[order] = np.arange(1, len(ck) + 1)
        for j, k in enumerate(k_idx):
            c = complex(ck[j])
            rows.append({
                "cell_id": int(row["cell_id"]),
                "tick_start": int(row["tick_start_idx"]),
                "tick_end": int(row["tick_end_idx"]),
                "k": int(k),
                "rank": int(rank[j]),
                "radius": float(abs(c)),
                "phase_rad": float(np.angle(c)),
                "phase_deg": float(np.degrees(np.angle(c))),
                "c_real": float(c.real),
                "c_imag": float(c.imag),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def write_functions_text(sample, data, k_idx, path, exactness):
    lines = []
    lines.append("XFLIP SAMPLE â€” 150-TICK FOURIER FUNCTIONS")
    lines.append("")
    lines.append("z_n = cos(2*pi*n/150) + i*y_bridge(n)")
    lines.append("c_k = (1/150) * sum_{n=0}^{149} z_n * exp(-i*2*pi*k*n/150)")
    lines.append("z(t) = sum_{k=-75}^{74} c_k * exp(i*2*pi*k*t/150)")
    lines.append("y_bridge(t) = Im(z(t))")
    lines.append("y_norm(t) = y_bridge(t) + y_open + (t/149)*bridge_slope")
    lines.append("price(t) = y_norm(t)*p_std + p_mean")
    lines.append("")
    lines.append(f"XFlip global: {exactness['xflip_cells']:,} cells, "
                 f"max_tick_error={exactness['max_tick_error']:.3e}, "
                 f"max_norm_rmse={exactness['max_norm_rmse']:.3e}")
    lines.append("")
    for i, (_, row) in enumerate(sample.iterrows()):
        d = data[i]
        terms = top_terms(row["ck"], k_idx, TOP_TERMS)
        lines.append(f"Cell_{int(row['cell_id'])}  ticks {int(row['tick_start_idx'])}..{int(row['tick_end_idx'])}")
        lines.append(f"  open={row['p_open']:.6f} high={row['p_high']:.6f} low={row['p_low']:.6f} close={row['p_close']:.6f}")
        lines.append(f"  p_mean={row['p_mean']:.12f} p_std={row['p_std']:.12f}")
        lines.append(f"  y_open={row['y_open_norm']:.12f} bridge_slope={row['bridge_slope']:.12f}")
        lines.append(f"  max|coeff_recompute_err|={d['coeff_err']:.3e}")
        lines.append(f"  max|price_reconstruction_err|={d['price_err']:.3e}")
        lines.append(f"  max|norm_rmse|={d['norm_rmse']:.3e}")
        lines.append("  top terms:")
        for t in terms:
            lines.append(
                f"    k={fmt_k(t['k']):>3s}  R={t['radius']:.8f}  "
                f"phi={t['phase_rad']:+.8f} rad  c={t['real']:+.8f}{t['imag']:+.8f}i"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=r"C:\cluster\phase1\final")
    ap.add_argument("--out", default=r"C:\cluster\phase1\analysis\xflip_sample5")
    ap.add_argument("--animate", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    catalog_path = os.path.join(args.final, "jump10_phase1_cell_catalog.parquet")
    npz_path = os.path.join(args.final, "jump10_phase1_fourier_functions.npz")
    catalog, sample = load_catalog(catalog_path, SAMPLE_CELLS)
    ck_all, raw_all = load_sample_arrays(npz_path, SAMPLE_CELLS)
    sample = sample.reset_index(drop=True)
    sample["ck"] = list(ck_all)
    sample["raw"] = list(raw_all)

    k_idx = np.fft.fftfreq(CELL_SIZE, d=1.0 / CELL_SIZE).astype(np.int64)
    sample_rows = []
    data = []
    for i, (_, row) in enumerate(sample.iterrows()):
        d = analyze_cell(
            ck_all[i], raw_all[i], float(row["p_mean"]), float(row["p_std"]),
            float(row["y_open_norm"]), float(row["bridge_slope"]),
        )
        d.update({"ck": ck_all[i], "raw": raw_all[i]})
        data.append(d)
        sample_rows.append({
            "cell_id": int(row["cell_id"]),
            "tick_start": int(row["tick_start_idx"]),
            "tick_end": int(row["tick_end_idx"]),
            "open": float(row["p_open"]),
            "high": float(row["p_high"]),
            "low": float(row["p_low"]),
            "close": float(row["p_close"]),
            "max_coeff_error": d["coeff_err"],
            "max_price_error": d["price_err"],
            "max_norm_rmse": d["norm_rmse"],
            "max_meta_error": d["meta_err"],
            "top_terms": top_terms(ck_all[i], k_idx, TOP_TERMS),
        })

    xflip = catalog[catalog["cell_id"] <= 171621]
    exactness = {
        "scope": "XFlip 2021-03..2021-12",
        "xflip_cells": int(len(xflip)),
        "xflip_ticks": int(xflip["tick_end_idx"].iloc[-1]),
        "max_tick_error": float(xflip["max_tick_error"].max()),
        "max_norm_rmse": float(xflip["norm_rmse"].max()),
        "cells_under_1e_11": int((xflip["max_tick_error"] < 1e-11).sum()),
        "sample_cells": SAMPLE_CELLS,
        "sample": sample_rows,
        "all_sample_coefficients_exact": bool(max(r["max_coeff_error"] for r in sample_rows) < 1e-12),
        "all_sample_prices_exact": bool(max(r["max_price_error"] for r in sample_rows) < 1e-11),
    }

    with open(os.path.join(args.out, "xflip_sample5_exactness.json"), "w", encoding="utf-8") as f:
        json.dump(exactness, f, indent=2)

    write_functions_csv(sample, data, k_idx, os.path.join(args.out, "xflip_sample5_functions.csv"))
    write_functions_text(
        sample, data, k_idx,
        os.path.join(args.out, "xflip_sample5_functions.txt"),
        exactness,
    )

    figure_price_paths(sample, data, os.path.join(args.out, "xflip_sample5_price_paths.png"))
    figure_functions(sample, data, k_idx, os.path.join(args.out, "xflip_sample5_functions.png"))

    if args.animate:
        for i, (_, row) in enumerate(sample.iterrows()):
            gif = os.path.join(args.out, f"xflip_cell_{int(row['cell_id']):06d}_epicycles.gif")
            figure_epicycles(int(row["cell_id"]), row, data[i], ck_all[i], k_idx, gif)

    summary = {
        "out": args.out,
        "exactness": os.path.join(args.out, "xflip_sample5_exactness.json"),
        "price_paths_png": os.path.join(args.out, "xflip_sample5_price_paths.png"),
        "functions_png": os.path.join(args.out, "xflip_sample5_functions.png"),
        "functions_csv": os.path.join(args.out, "xflip_sample5_functions.csv"),
        "functions_txt": os.path.join(args.out, "xflip_sample5_functions.txt"),
        "gifs": [
            os.path.join(args.out, f"xflip_cell_{int(c)}_epicycles.gif")
            for c in SAMPLE_CELLS
        ] if args.animate else [],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

