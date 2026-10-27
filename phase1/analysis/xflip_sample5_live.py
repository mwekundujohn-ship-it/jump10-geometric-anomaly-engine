import argparse
import itertools
import json
import os
import sys
import traceback

import matplotlib
if "--preview" in sys.argv:
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import numpy as np
import pandas as pd

CELL_SIZE = 150
SAMPLE_CELLS = [1, 42906, 85811, 128716, 171621]
TOP_CIRCLES = 24
TOP_TERMS = 6
FPS = 24

BG = "#050807"
PANEL = "#08100c"
WHITE = "#f4fff8"
GREEN = "#8dffb0"
GREEN_DIM = "#4f9f70"
GREEN_FAINT = "#1e3a2e"
GOLD = "#ffd166"
RED = "#ff6b7a"
GRID = "#173026"

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xflip_sample5_live.log")


def log(msg):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def fmt_k(k):
    k = int(k)
    return f"+{k}" if k > 0 else str(k)


def load_data(final_dir):
    catalog_path = os.path.join(final_dir, "jump10_phase1_cell_catalog.parquet")
    npz_path = os.path.join(final_dir, "jump10_phase1_fourier_functions.npz")
    catalog = pd.read_parquet(catalog_path, engine="fastparquet")
    sample = catalog[catalog["cell_id"].isin(SAMPLE_CELLS)].copy()
    sample = sample.set_index("cell_id").loc[SAMPLE_CELLS].reset_index()

    with np.load(npz_path, allow_pickle=False) as npz:
        ids = npz["cell_id"].astype(np.int64)
        positions = np.searchsorted(ids, np.asarray(SAMPLE_CELLS, dtype=np.int64))
        ck_all = npz["c_k"][positions].copy()
        raw_all = npz["raw_tick_cells"][positions].copy()

    k_idx = np.fft.fftfreq(CELL_SIZE, d=1.0 / CELL_SIZE).astype(np.int64)
    data = []
    for i, row in sample.iterrows():
        ck = ck_all[i]
        raw = raw_all[i]
        t = np.arange(CELL_SIZE, dtype=np.float64) / (CELL_SIZE - 1)
        n = np.arange(CELL_SIZE, dtype=np.float64)
        theta = 2.0 * np.pi * n / CELL_SIZE
        p_mean = float(row["p_mean"])
        p_std = float(row["p_std"])
        y_open = float(row["y_open_norm"])
        slope = float(row["bridge_slope"])
        y_norm = (raw - p_mean) / p_std
        y_bridge = y_norm - (y_norm[0] + t * (y_norm[-1] - y_norm[0]))
        z = np.cos(theta) + 1j * y_bridge
        ck_direct = np.fft.fft(z) / CELL_SIZE
        z_rec = np.fft.ifft(ck * CELL_SIZE)
        y_bridge_rec = z_rec.imag
        y_norm_rec = y_bridge_rec + y_open + t * slope
        price_rec = y_norm_rec * p_std + p_mean
        price_err = np.abs(price_rec - raw)
        coeff_err = float(np.max(np.abs(ck_direct - ck)))
        order = np.argsort(-np.abs(ck))
        terms = []
        for j in order[:TOP_TERMS]:
            c = complex(ck[j])
            terms.append({
                "k": int(k_idx[j]),
                "radius": float(abs(c)),
                "phase": float(np.angle(c)),
                "real": float(c.real),
                "imag": float(c.imag),
            })
        data.append({
            "ck": ck,
            "raw": raw,
            "y_bridge": y_bridge,
            "z": z,
            "price_rec": price_rec,
            "price_err": price_err,
            "max_price_err": float(price_err.max()),
            "max_norm_rmse": float(np.sqrt(np.mean((y_norm_rec - y_norm) ** 2))),
            "coeff_err": coeff_err,
            "order": order,
            "k_sorted": k_idx[order],
            "ck_sorted": ck[order],
            "terms": terms,
            "z_lim": float(max(1.25, 1.2 * np.max(np.abs(z)) + 0.25 * np.max(np.abs(ck)))),
        })
    return sample, data, k_idx


def style_axis(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GREEN, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GREEN_FAINT)
    ax.grid(color=GRID, alpha=0.75, lw=0.6)
    ax.xaxis.label.set_color(GREEN)
    ax.yaxis.label.set_color(GREEN)
    ax.title.set_color(WHITE)


class LiveViewer:
    def __init__(self, sample, data, k_idx, preview=False):
        self.sample = sample
        self.data = data
        self.k_idx = k_idx
        self.index = 0
        self.frame = 0.0
        self.paused = False
        self.completed = False
        self.speed = 1.0
        self.preview = preview
        self.build_figure()
        self.load_cell(0)
        if not preview:
            self.anim = FuncAnimation(
                self.fig, self.tick, frames=itertools.count(),
                interval=1000.0 / FPS, blit=True, repeat=False, cache_frame_data=False,
            )
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)
            self.fig.canvas.mpl_connect("close_event", lambda _evt: log("window closed"))
            try:
                self.fig.canvas.manager.set_window_title("XFlip - 5 Cell Fourier Epicycle Derivation")
            except Exception:
                pass

    def build_figure(self):
        plt.rcParams["font.family"] = "DejaVu Sans"
        self.fig = plt.figure(figsize=(14, 8), facecolor=BG)
        gs = self.fig.add_gridspec(
            2, 2, width_ratios=[1.28, 1.0], height_ratios=[1.25, 1.0],
            left=0.06, right=0.975, top=0.885, bottom=0.16, wspace=0.16, hspace=0.25,
        )
        self.ax_price = self.fig.add_subplot(gs[0, 0])
        self.ax_error = self.fig.add_subplot(gs[1, 0])
        self.ax_epi = self.fig.add_subplot(gs[0, 1])
        self.ax_formula = self.fig.add_subplot(gs[1, 1])
        for ax in (self.ax_price, self.ax_error, self.ax_epi):
            style_axis(ax)
        self.ax_formula.set_facecolor(PANEL)
        self.ax_formula.axis("off")

        self.fig.suptitle(
            "XFLIP - 5-CELL FOURIER EPICYCLE DERIVATION",
            color=WHITE, fontsize=16, fontweight="bold", y=0.965,
        )
        self.status = self.fig.text(
            0.5, 0.925, "", ha="center", va="center", color=GREEN, fontsize=10,
            family="monospace",
        )
        self.fig.text(
            0.5, 0.012,
            "SPACE pause   |   LEFT / RIGHT change cell   |   R replay   |   UP / DOWN speed   |   all 150 coefficients summed",
            ha="center", va="bottom", color=GREEN_DIM, fontsize=8,
        )

        self.cell_buttons = []
        labels = [f"Cell {c}" for c in SAMPLE_CELLS]
        x0, width, gap = 0.045, 0.103, 0.007
        for i, label in enumerate(labels):
            ax = self.fig.add_axes([x0 + i * (width + gap), 0.055, width, 0.045])
            btn = Button(ax, label, color="#10201a", hovercolor="#1e3a2e")
            btn.label.set_color(GREEN)
            btn.label.set_fontsize(8)
            btn.on_clicked(lambda _evt, idx=i: self.load_cell(idx, reset=True))
            self.cell_buttons.append(btn)

        ax_pause = self.fig.add_axes([x0 + 5 * (width + gap) + 0.012, 0.055, 0.085, 0.045])
        self.pause_button = Button(ax_pause, "Pause", color="#10201a", hovercolor="#1e3a2e")
        self.pause_button.label.set_color(GREEN)
        self.pause_button.label.set_fontsize(8)
        self.pause_button.on_clicked(lambda _evt: self.toggle_pause())

        ax_replay = self.fig.add_axes([x0 + 5 * (width + gap) + 0.012 + 0.095, 0.055, 0.085, 0.045])
        self.replay_button = Button(ax_replay, "Replay", color="#10201a", hovercolor="#1e3a2e")
        self.replay_button.label.set_color(GREEN)
        self.replay_button.label.set_fontsize(8)
        self.replay_button.on_clicked(lambda _evt: self.replay())

        ax_speed = self.fig.add_axes([x0 + 5 * (width + gap) + 0.012 + 0.190, 0.055, 0.085, 0.045])
        self.speed_button = Button(ax_speed, "Speed 1x", color="#10201a", hovercolor="#1e3a2e")
        self.speed_button.label.set_color(GREEN)
        self.speed_button.label.set_fontsize(8)
        self.speed_button.on_clicked(lambda _evt: self.cycle_speed())

    def load_cell(self, index, reset=True):
        self.index = int(index)
        self.frame = 0.0
        self.paused = False
        self.completed = False
        if hasattr(self, "pause_button"):
            self.pause_button.label.set_text("Pause")
        for i, btn in enumerate(self.cell_buttons):
            btn.ax.set_facecolor("#1e3a2e" if i == self.index else "#10201a")
        row = self.sample.iloc[self.index]
        d = self.data[self.index]
        self.row = row
        self.d = d
        self.ticks = np.arange(1, CELL_SIZE + 1)
        self.draw_price(row, d)
        self.draw_error(row, d)
        self.draw_epicycles(row, d)
        self.draw_formulas(row, d)
        self.update(0)
        if hasattr(self, "anim"):
            self.anim._blit_cache.clear()
            self.fig.canvas.draw()
            self.anim.event_source.start()

    def draw_price(self, row, d):
        ax = self.ax_price
        ax.cla()
        style_axis(ax)
        raw = d["raw"]
        ax.plot(self.ticks, raw, color=WHITE, lw=2.4, label="Actual mid price (150 ticks)", zorder=2)
        self.fit_line, = ax.plot([], [], color=GREEN, lw=1.6, ls="--", label="Fourier reconstruction", zorder=4)
        self.cursor = ax.axvline(1, color=GOLD, lw=0.9, alpha=0.7, zorder=1)
        self.actual_dot = ax.scatter([1], [raw[0]], s=60, facecolors="none", edgecolors=WHITE, linewidths=1.4, zorder=6)
        self.fit_dot = ax.scatter([1], [d["price_rec"][0]], s=34, marker="x", color=GREEN, linewidths=1.4, zorder=7)
        ax.scatter([1], [row["p_open"]], s=42, color="#00ff9d", zorder=5)
        ax.scatter([CELL_SIZE], [row["p_close"]], s=42, color=RED, zorder=5)
        ax.annotate(
            f"OPEN\n{row['p_open']:.4f}", xy=(1, row["p_open"]),
            xytext=(8, 12), textcoords="offset points", color="#00ff9d", fontsize=8,
            arrowprops=dict(arrowstyle="-", color="#00ff9d", lw=0.7),
        )
        ax.annotate(
            f"CLOSE\n{row['p_close']:.4f}", xy=(CELL_SIZE, row["p_close"]),
            xytext=(-10, 12), textcoords="offset points", ha="right", color=RED, fontsize=8,
            arrowprops=dict(arrowstyle="-", color=RED, lw=0.7),
        )
        ax.set_xlim(0.5, CELL_SIZE + 0.5)
        pad = (row["p_high"] - row["p_low"]) * 0.16 + 1e-9
        ax.set_ylim(row["p_low"] - pad, row["p_high"] + pad)
        ax.set_xlabel("Tick within cell  (1 = open, 150 = close)")
        ax.set_ylabel("Price")
        ax.set_title(
            f"Cell_{int(row['cell_id'])}   ticks {int(row['tick_start_idx'])}..{int(row['tick_end_idx'])}"
            f"   |   max|fit-raw| = {d['max_price_err']:.3e}",
            fontsize=10, loc="left",
        )
        ax.legend(loc="upper right", fontsize=8, facecolor=PANEL, edgecolor=GREEN_FAINT, labelcolor=WHITE)

    def draw_error(self, row, d):
        ax = self.ax_error
        ax.cla()
        style_axis(ax)
        scale = max(1e-12, d["max_price_err"] * 1.25)
        ax.axhline(0, color=GREEN_DIM, lw=0.8, alpha=0.7)
        self.err_line, = ax.plot([], [], color=GREEN, lw=1.2, marker=".", ms=3, zorder=3)
        ax.set_xlim(0.5, CELL_SIZE + 0.5)
        ax.set_ylim(-0.10 * scale, scale)
        ax.set_xlabel("Tick within cell")
        ax.set_ylabel("|actual - Fourier|")
        ax.set_title(
            f"Live error   |   max price err = {d['max_price_err']:.3e}"
            f"   |   max norm RMSE = {d['max_norm_rmse']:.3e}",
            fontsize=10, loc="left",
        )
        ax.text(
            0.99, 0.06,
            f"coefficient recompute error = {d['coeff_err']:.3e}",
            transform=ax.transAxes, ha="right", va="bottom", color=GREEN_DIM, fontsize=8,
            family="monospace",
        )

    def draw_epicycles(self, row, d):
        ax = self.ax_epi
        ax.cla()
        style_axis(ax)
        lim = d["z_lim"]
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Epicycles: all 150 terms summed, {TOP_CIRCLES} largest circles drawn",
            fontsize=10, loc="left",
        )
        ax.plot(d["z"].real, d["z"].imag, color=WHITE, alpha=0.22, lw=1.0, ls="--", zorder=1)
        self.chain_line, = ax.plot([], [], color=WHITE, alpha=0.55, lw=0.8, zorder=3)
        self.trail_line, = ax.plot([], [], color=GREEN, lw=1.6, zorder=5)
        self.tip_dot, = ax.plot([], [], "o", color=GREEN, ms=6, zorder=8)
        self.circles = []
        self.vector_lines = []
        for j in range(min(TOP_CIRCLES, len(d["ck_sorted"]))):
            circ = Circle((0, 0), float(abs(d["ck_sorted"][j])), fill=False, ec=GREEN, lw=0.7, alpha=0.38, zorder=2)
            ax.add_patch(circ)
            self.circles.append(circ)
            line, = ax.plot([], [], color=WHITE, lw=0.9, alpha=0.75, zorder=4)
            self.vector_lines.append(line)
        ax.text(
            0.01, 0.99,
            "z_n = cos(2*pi*n/150) + i*y_bridge(n)\n"
            "z(t) = sum c_k * exp(i*2*pi*k*t/150)",
            transform=ax.transAxes, va="top", ha="left", color=GREEN, fontsize=8,
            family="monospace",
        )

    def draw_formulas(self, row, d):
        ax = self.ax_formula
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.axis("off")
        formulas = (
            "EXACT FORWARD / INVERSE MAP\n"
            "z_n = cos(2*pi*n/150) + i*y_bridge(n)\n"
            "c_k = (1/150) * sum_{n=0}^{149} z_n * exp(-i*2*pi*k*n/150)\n"
            "z(t) = sum_{k=-75}^{74} c_k * exp(i*2*pi*k*t/150)\n"
            "y_bridge(t) = Im(z(t))\n"
            "y_norm(t) = y_bridge(t) + y_open + (t/149)*bridge_slope\n"
            "price(t) = y_norm(t)*p_std + p_mean\n"
        )
        ax.text(
            0.02, 0.97, formulas, transform=ax.transAxes, va="top", ha="left",
            color=GREEN, fontsize=8.5, family="monospace", linespacing=1.45,
        )
        terms = "\n".join(
            f"k={fmt_k(t['k']):>3s}  R={t['radius']:.6f}  phi={t['phase']:+.6f}"
            for t in d["terms"]
        )
        cell = (
            f"Cell_{int(row['cell_id'])}   open={row['p_open']:.4f}   close={row['p_close']:.4f}\n"
            f"p_mean={row['p_mean']:.6f}   p_std={row['p_std']:.6f}\n"
            f"max|coeff err|={d['coeff_err']:.3e}   max|price err|={d['max_price_err']:.3e}\n"
            f"top {TOP_TERMS} terms by radius:\n{terms}"
        )
        ax.text(
            0.02, 0.02, cell, transform=ax.transAxes, va="bottom", ha="left",
            color=WHITE, fontsize=8, family="monospace", linespacing=1.35,
        )

    def update(self, frame):
        row = self.row
        d = self.d
        n = min(int(round(self.frame)), CELL_SIZE - 1)
        fit = d["price_rec"][:n + 1]
        self.fit_line.set_data(self.ticks[:n + 1], fit)
        self.cursor.set_xdata([n + 1, n + 1])
        self.actual_dot.set_offsets([[n + 1, d["raw"][n]]])
        self.fit_dot.set_offsets([[n + 1, fit[n]]])
        self.err_line.set_data(self.ticks[:n + 1], d["price_err"][:n + 1])

        vecs = d["ck_sorted"] * np.exp(1j * 2.0 * np.pi * d["k_sorted"] * n / CELL_SIZE)
        pts = np.concatenate([[0.0 + 0.0j], np.cumsum(vecs)])
        self.chain_line.set_data(pts.real, pts.imag)
        for j, (line, circ) in enumerate(zip(self.vector_lines, self.circles)):
            line.set_data([pts[j].real, pts[j + 1].real], [pts[j].imag, pts[j + 1].imag])
            circ.center = (pts[j].real, pts[j].imag)
        self.trail_line.set_data(d["z"][:n + 1].real, d["z"][:n + 1].imag)
        self.tip_dot.set_data([pts[-1].real], [pts[-1].imag])

        err = abs(d["raw"][n] - fit[n])
        state = "COMPLETE" if self.completed else ("PAUSED" if self.paused else "DRAWING")
        self.status.set_text(
            f"{state}   t={n + 1:3d}/150   actual={d['raw'][n]:.6f}   Fourier={fit[n]:.6f}   "
            f"|err|={err:.3e}   speed={self.speed:g}x"
        )
        return (
            self.fit_line, self.cursor, self.actual_dot, self.fit_dot, self.err_line,
            self.chain_line, self.trail_line, self.tip_dot, self.status,
            *self.vector_lines, *self.circles,
        )

    def tick(self, _frame):
        if self.paused or self.completed:
            return self.update(0)
        self.frame += self.speed
        if self.frame >= CELL_SIZE - 1:
            self.frame = CELL_SIZE - 1
            self.completed = True
            if hasattr(self, "anim"):
                self.anim.event_source.stop()
        return self.update(0)

    def toggle_pause(self):
        if self.completed:
            self.replay()
            return
        self.paused = not self.paused
        if hasattr(self, "anim"):
            if self.paused:
                self.anim.event_source.stop()
            else:
                self.anim.event_source.start()
        if hasattr(self, "pause_button"):
            self.pause_button.label.set_text("Play" if self.paused else "Pause")

    def replay(self):
        self.frame = 0.0
        self.completed = False
        self.paused = False
        if hasattr(self, "anim"):
            self.anim.event_source.start()
        if hasattr(self, "pause_button"):
            self.pause_button.label.set_text("Pause")

    def cycle_speed(self):
        options = [0.5, 1.0, 2.0, 4.0]
        i = options.index(self.speed) if self.speed in options else 1
        self.speed = options[(i + 1) % len(options)]
        if hasattr(self, "speed_button"):
            self.speed_button.label.set_text(f"Speed {self.speed:g}x")

    def on_key(self, event):
        if event.key == " ":
            self.toggle_pause()
        elif event.key == "right":
            self.load_cell((self.index + 1) % len(self.sample))
        elif event.key == "left":
            self.load_cell((self.index - 1) % len(self.sample))
        elif event.key in ("r", "R"):
            self.replay()
        elif event.key == "up":
            self.cycle_speed()
        elif event.key == "down":
            self.cycle_speed()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=r"C:\cluster\phase1\final")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--out", default=r"C:\cluster\phase1\analysis\xflip_sample5")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log("starting live viewer")
    sample, data, k_idx = load_data(args.final)
    viewer = LiveViewer(sample, data, k_idx, preview=args.preview)
    if args.preview:
        viewer.frame = CELL_SIZE - 1
        viewer.completed = True
        viewer.update(0)
        preview_path = os.path.join(args.out, "xflip_sample5_live_preview.png")
        viewer.fig.savefig(preview_path, dpi=150, facecolor=BG)
        print(preview_path)
        log(f"preview saved {preview_path}")
    else:
        plt.show()
        log("viewer returned")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        raise



