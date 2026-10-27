import os
import sys
import json
import time
import math
import gc
import shutil
import subprocess
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CELL_SIZE = 150


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def json_dump(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def upload_file(local_path, url, retries=5):
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if not curl:
        raise RuntimeError("curl not found")
    last = None
    for attempt in range(1, retries + 1):
        p = subprocess.run(
            [curl, "-sS", "--fail", "--retry", "3", "--retry-delay", "2",
             "-X", "PUT", "--data-binary", "@" + str(local_path), url],
            capture_output=True, text=True, timeout=3600,
        )
        if p.returncode == 0:
            return p.stdout.strip()
        last = (p.returncode, p.stdout.strip(), p.stderr.strip())
        log(f"upload attempt {attempt} failed for {os.path.basename(str(local_path))}: {last}")
        time.sleep(3)
    raise RuntimeError(f"upload failed: {last}")


def download_file(url, dest, expected_size=None, retries=4):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "phase1-worker"})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                t0 = time.time()
                last_log = t0
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    now = time.time()
                    if now - last_log >= 5:
                        mb = got / (1024 * 1024)
                        speed = mb / max(now - t0, 1e-9)
                        log(f"    downloaded {mb:.1f} MB / {total/(1024*1024):.1f} MB ({speed:.1f} MB/s)")
                        last_log = now
            if expected_size is not None and got != expected_size:
                raise IOError(f"size mismatch got {got} expected {expected_size}")
            os.replace(part, dest)
            return got
        except Exception as e:
            last = e
            try:
                if os.path.exists(part):
                    os.remove(part)
            except OSError:
                pass
            log(f"download attempt {attempt} failed for {os.path.basename(dest)}: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"download failed: {url}: {last}")


def upload_json(obj, filename, upload_url):
    tmp = os.path.join(os.path.dirname(__file__), filename)
    json_dump(tmp, obj)
    try:
        upload_file(tmp, upload_url + filename)
    except Exception as e:
        log(f"progress upload failed: {e}")


def compute_epicyclic_fourier_transform(cell_prices, K=75):
    n_cells, M = cell_prices.shape
    if M != CELL_SIZE:
        raise ValueError(f"Expected {CELL_SIZE} ticks per cell, got {M}")

    p_open = cell_prices[:, 0:1]
    p_close = cell_prices[:, -1:]
    p_high = cell_prices.max(axis=1, keepdims=True)
    p_low = cell_prices.min(axis=1, keepdims=True)
    p_mean = cell_prices.mean(axis=1, keepdims=True)
    p_std_raw = cell_prices.std(axis=1, keepdims=True)
    p_std = np.where(p_std_raw < 1e-12, 1.0, p_std_raw)

    y_norm = (cell_prices - p_mean) / p_std

    t_linear = np.linspace(0.0, 1.0, M, dtype=np.float64)[None, :]
    bridge_slope = y_norm[:, -1:] - y_norm[:, 0:1]
    y_bridge = y_norm - (y_norm[:, 0:1] + t_linear * bridge_slope)

    theta = 2.0 * np.pi * np.arange(M, dtype=np.float64) / M
    z_loop = np.cos(theta)[None, :] + 1j * y_bridge

    fft_full = np.fft.fft(z_loop, axis=1) / M
    freqs = np.fft.fftfreq(M, d=1.0 / M).astype(int)

    if K >= 75:
        k_indices = freqs.copy()
        c_k = fft_full
        fft_recon_buf = fft_full
    else:
        k_indices = np.array([k for k in range(-K, K + 1)], dtype=int)
        col_idx = [int(np.where(freqs == k)[0][0]) for k in k_indices]
        c_k = fft_full[:, col_idx]
        fft_recon_buf = np.zeros_like(fft_full)
        for idx, k in enumerate(k_indices):
            f_col = int(np.where(freqs == k)[0][0])
            fft_recon_buf[:, f_col] = c_k[:, idx]

    radii = np.abs(c_k)
    phases = np.angle(c_k)
    omegas = 2.0 * np.pi * k_indices / M

    z_reconstructed = np.fft.ifft(fft_recon_buf * M, axis=1)
    y_bridge_rec = z_reconstructed.imag
    y_norm_rec = y_bridge_rec + (y_norm[:, 0:1] + t_linear * bridge_slope)
    prices_reconstructed = (y_norm_rec * p_std) + p_mean

    abs_err = np.abs(cell_prices - prices_reconstructed)
    max_abs_err = np.max(abs_err, axis=1)
    rmse_err = np.sqrt(np.mean((cell_prices - prices_reconstructed) ** 2, axis=1))
    norm_err = np.sqrt(np.mean((y_norm - y_norm_rec) ** 2, axis=1))

    return {
        "p_open": p_open.ravel(),
        "p_close": p_close.ravel(),
        "p_high": p_high.ravel(),
        "p_low": p_low.ravel(),
        "p_mean": p_mean.ravel(),
        "p_std": p_std.ravel(),
        "y_open_norm": y_norm[:, 0].ravel(),
        "bridge_slope": bridge_slope.ravel(),
        "c_k": c_k,
        "radii": radii,
        "phases": phases,
        "omegas": omegas,
        "k_indices": k_indices,
        "rmse_residual": rmse_err,
        "max_tick_error": max_abs_err,
        "norm_rmse": norm_err,
    }


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--job":
        print("usage: phase1_worker.py --job job.json", file=sys.stderr)
        return 2

    job_path = sys.argv[2]
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)

    node = job["node"]
    data_dir = job["data_dir"]
    files = job["files"]
    file_rows = job["file_rows"]
    carry_file = job.get("carry_file")
    carry_rows = int(job.get("carry_rows", 0))
    carry_file_rows = int(job.get("carry_file_rows", 0))
    K = int(job.get("harmonics_K", 75))
    out_dir = job["output_dir"]
    upload_url = job.get("upload_url")
    do_upload = bool(job.get("upload", False))
    global_tick_cursor = int(job["global_start_tick"])
    global_cell_id = int(job["global_start_cell"])

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    download_files = job.get("download_files", [])
    data_url_base = job.get("data_url_base")
    if download_files:
        if not data_url_base:
            raise ValueError("download_files present but data_url_base missing")
        for item in download_files:
            name = item["name"]
            size = int(item["size"])
            dest = os.path.join(data_dir, name)
            if os.path.isfile(dest) and os.path.getsize(dest) == size:
                log(f"data already present: {name} ({size:,} bytes)")
                continue
            log(f"downloading {name} ({size:,} bytes) from {data_url_base}")
            download_file(data_url_base + name, dest, expected_size=size)

    run_started = time.time()
    progress = {
        "node": node,
        "status": "starting",
        "cells_done": 0,
        "ticks_done": 0,
        "files_done": 0,
        "files_total": len(files),
        "global_start_tick": global_tick_cursor,
        "global_start_cell": global_cell_id,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if do_upload and upload_url:
        upload_json(progress, f"progress_{node}.json", upload_url)

    def progress_update(**kw):
        progress.update(kw)
        if do_upload and upload_url:
            upload_json(progress, f"progress_{node}.json", upload_url)

    carry_prices = np.empty(0, dtype=np.float64)
    carry_times = np.empty(0, dtype=np.int64)

    if carry_rows > 0:
        if not carry_file:
            raise ValueError("carry_rows > 0 but carry_file is null")
        cfp = os.path.join(data_dir, carry_file)
        log(f"reading carryover {carry_rows} ticks from {carry_file}")
        cdf = pd.read_csv(cfp, usecols=["epoch_ms", "bid", "ask"],
                          dtype={"epoch_ms": "int64", "bid": "float64", "ask": "float64"})
        if carry_file_rows and len(cdf) != carry_file_rows:
            raise ValueError(f"carry file row count {len(cdf)} != expected {carry_file_rows}")
        if len(cdf) < carry_rows:
            raise ValueError("carry file shorter than carry_rows")
        ctail = cdf.tail(carry_rows)
        carry_prices = ((ctail["bid"].to_numpy(np.float64) + ctail["ask"].to_numpy(np.float64)) / 2.0)
        carry_times = ctail["epoch_ms"].to_numpy(np.int64)
        del cdf, ctail
        log(f"carryover ready: {len(carry_prices)} ticks, first={carry_times[0]}, last={carry_times[-1]}")

    for old in os.listdir(out_dir):
        if old.startswith(f"chunk_{node}_") or old.startswith(f"summary_{node}"):
            try:
                os.remove(os.path.join(out_dir, old))
            except OSError:
                pass

    chunks = []
    k_indices_ref = None
    total_cells = 0
    total_ticks = 0
    max_tick_error_all = 0.0
    max_norm_rmse_all = 0.0
    cells_under_1e_11 = 0
    contiguity_ok = True
    rowcount_mismatches = []
    expected_tick = int(job["global_start_tick"])
    expected_cell = int(job["global_start_cell"])

    for fi, (fname, expected_rows) in enumerate(zip(files, file_rows), start=1):
        fp = os.path.join(data_dir, fname)
        t_file = time.time()
        log(f"[{fi}/{len(files)}] reading {fname} (expect {expected_rows:,} rows)")
        df = pd.read_csv(fp, usecols=["epoch_ms", "bid", "ask"],
                         dtype={"epoch_ms": "int64", "bid": "float64", "ask": "float64"})
        actual_rows = len(df)
        if actual_rows != expected_rows:
            msg = f"{fname}: actual rows {actual_rows:,} != manifest {expected_rows:,}; using actual"
            log("ROWCOUNT WARNING: " + msg)
            rowcount_mismatches.append({
                "file": fname, "actual": int(actual_rows), "manifest": int(expected_rows)
            })

        prices = ((df["bid"].to_numpy(np.float64) + df["ask"].to_numpy(np.float64)) / 2.0)
        times = df["epoch_ms"].to_numpy(np.int64)
        del df
        gc.collect()

        if len(carry_prices) > 0:
            prices = np.concatenate([carry_prices, prices])
            times = np.concatenate([carry_times, times])
            carry_prices = np.empty(0, dtype=np.float64)
            carry_times = np.empty(0, dtype=np.int64)

        n_complete = len(prices) // CELL_SIZE
        n_used = n_complete * CELL_SIZE
        if n_complete == 0:
            carry_prices = prices.copy()
            carry_times = times.copy()
            log(f"    no complete cell yet; carry={len(carry_prices)}")
            continue

        cell_prices = prices[:n_used].reshape(n_complete, CELL_SIZE)
        cell_times = times[:n_used].reshape(n_complete, CELL_SIZE)
        carry_prices = prices[n_used:].copy()
        carry_times = times[n_used:].copy()

        ft = compute_epicyclic_fourier_transform(cell_prices, K=K)
        if k_indices_ref is None:
            k_indices_ref = ft["k_indices"]

        cell_ids = np.arange(global_cell_id, global_cell_id + n_complete, dtype=np.int64)
        start_ticks = global_tick_cursor + np.arange(n_complete, dtype=np.int64) * CELL_SIZE
        end_ticks = start_ticks + (CELL_SIZE - 1)

        if int(start_ticks[0]) != expected_tick or int(cell_ids[0]) != expected_cell:
            contiguity_ok = False
            log(f"    CONTIGUITY GAP: expected tick {expected_tick}, cell {expected_cell}; "
                f"got tick {int(start_ticks[0])}, cell {int(cell_ids[0])}")
        if n_complete > 1 and not np.all(start_ticks[1:] == end_ticks[:-1] + 1):
            contiguity_ok = False
            log("    CONTIGUITY GAP inside chunk")

        chunk = pd.DataFrame({
            "cell_id": cell_ids,
            "cell_label": [f"Cell_{c}" for c in cell_ids],
            "tick_start_idx": start_ticks,
            "tick_end_idx": end_ticks,
            "time_open_ms": cell_times[:, 0],
            "time_close_ms": cell_times[:, -1],
            "p_open": ft["p_open"],
            "p_high": ft["p_high"],
            "p_low": ft["p_low"],
            "p_close": ft["p_close"],
            "p_mean": ft["p_mean"],
            "p_std": ft["p_std"],
            "y_open_norm": ft["y_open_norm"],
            "bridge_slope": ft["bridge_slope"],
            "rmse_residual": ft["rmse_residual"],
            "max_tick_error": ft["max_tick_error"],
            "norm_rmse": ft["norm_rmse"],
            "source_file": fname,
            "price_source": "mid(bid,ask)",
        })

        chunk_catalog_name = f"chunk_{node}_{fi:02d}_catalog.csv"
        chunk_npz_name = f"chunk_{node}_{fi:02d}_functions.npz"
        chunk_catalog_path = os.path.join(out_dir, chunk_catalog_name)
        chunk_npz_path = os.path.join(out_dir, chunk_npz_name)

        log(f"    writing {chunk_catalog_name} and {chunk_npz_name}")
        chunk.to_csv(chunk_catalog_path, index=False)
        np.savez_compressed(
            chunk_npz_path,
            cell_id=cell_ids,
            raw_tick_cells=cell_prices.astype(np.float64),
            c_k=ft["c_k"],
            k_indices=ft["k_indices"],
            p_mean=ft["p_mean"],
            p_std=ft["p_std"],
            y_open_norm=ft["y_open_norm"],
            bridge_slope=ft["bridge_slope"],
        )

        max_err = float(np.max(ft["max_tick_error"]))
        max_norm = float(np.max(ft["norm_rmse"]))
        under = int(np.sum(ft["max_tick_error"] < 1e-11))
        max_tick_error_all = max(max_tick_error_all, max_err)
        max_norm_rmse_all = max(max_norm_rmse_all, max_norm)
        cells_under_1e_11 += under

        chunks.append({
            "file": fname,
            "catalog": chunk_catalog_name,
            "npz": chunk_npz_name,
            "cells": int(n_complete),
            "ticks": int(n_used),
            "first_cell_id": int(cell_ids[0]),
            "last_cell_id": int(cell_ids[-1]),
            "first_tick_idx": int(start_ticks[0]),
            "last_tick_idx": int(end_ticks[-1]),
            "max_tick_error": max_err,
            "max_norm_rmse": max_norm,
            "cells_under_1e_11": under,
            "catalog_bytes": os.path.getsize(chunk_catalog_path),
            "npz_bytes": os.path.getsize(chunk_npz_path),
        })

        total_cells += n_complete
        total_ticks += n_used
        global_cell_id += n_complete
        global_tick_cursor += n_used
        expected_tick = int(end_ticks[-1]) + 1
        expected_cell = int(cell_ids[-1]) + 1

        log(f"    built {n_complete:,} cells | Cell_{cell_ids[0]}..Cell_{cell_ids[-1]} | "
            f"max_err={max_err:.3e} | carry={len(carry_prices)} | {time.time()-t_file:.1f}s")

        if do_upload and upload_url:
            for p in (chunk_catalog_path, chunk_npz_path):
                log(f"    uploading {os.path.basename(p)} ({os.path.getsize(p):,} bytes)")
                upload_file(p, upload_url + os.path.basename(p))

        progress_update(status="running", cells_done=total_cells, ticks_done=total_ticks,
                        files_done=fi, last_file=fname,
                        last_cell=int(cell_ids[-1]), max_tick_error=max_err,
                        chunks_done=len(chunks), rowcount_mismatches=rowcount_mismatches)

        del ft, chunk, cell_prices, cell_times, prices, times
        gc.collect()

    if total_cells == 0:
        raise RuntimeError("no complete cells built")
    if k_indices_ref is None:
        raise RuntimeError("no harmonic indices captured")

    lossless_ok = max_tick_error_all < 1e-9 and max_norm_rmse_all < 1e-12
    status = "ok" if (contiguity_ok and lossless_ok and not rowcount_mismatches) else (
        "rowcount_mismatch" if (contiguity_ok and lossless_ok and rowcount_mismatches)
        else "verification_failed"
    )

    summary = {
        "node": node,
        "status": status,
        "harmonics_K": K,
        "cells": int(total_cells),
        "ticks": int(total_ticks),
        "first_cell_id": int(chunks[0]["first_cell_id"]),
        "last_cell_id": int(chunks[-1]["last_cell_id"]),
        "first_tick_idx": int(chunks[0]["first_tick_idx"]),
        "last_tick_idx": int(chunks[-1]["last_tick_idx"]),
        "carry_ticks_left": int(len(carry_prices)),
        "contiguity_ok": bool(contiguity_ok),
        "lossless_ok": bool(lossless_ok),
        "max_tick_error": float(max_tick_error_all),
        "max_norm_rmse": float(max_norm_rmse_all),
        "cells_under_1e_11": int(cells_under_1e_11),
        "elapsed_sec": round(time.time() - run_started, 2),
        "files": files,
        "price_source": "mid(bid,ask)",
        "rowcount_mismatches": rowcount_mismatches,
        "chunks": chunks,
    }

    summary_path = os.path.join(out_dir, f"summary_{node}.json")
    json_dump(summary_path, summary)
    log("summary: " + json.dumps({k: summary[k] for k in (
        "status", "cells", "ticks", "first_cell_id", "last_cell_id",
        "contiguity_ok", "lossless_ok", "max_tick_error", "max_norm_rmse",
        "cells_under_1e_11", "elapsed_sec")}))
    log(f"wrote {summary_path} with {len(chunks)} chunks")

    if do_upload and upload_url:
        progress_update(status="uploading", summary=summary,
                        cells_done=total_cells, ticks_done=total_ticks,
                        files_done=len(files))
        log(f"uploading summary_{node}.json")
        upload_file(summary_path, upload_url + os.path.basename(summary_path))
        upload_json(progress, f"progress_{node}.json", upload_url)

    if not (contiguity_ok and lossless_ok):
        log("VERIFICATION FAILED")
        return 1
    if rowcount_mismatches:
        log("COMPLETE WITH ROWCOUNT WARNINGS")
        return 0
    log("PHASE 1 WORKER COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
