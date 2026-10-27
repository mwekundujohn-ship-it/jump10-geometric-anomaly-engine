import os
import sys
import json
import time
import shutil
import zipfile
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap, write_array_header_1_0, dtype_to_descr


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_chunk_index(out_dir):
    rows = []
    for name in sorted(os.listdir(out_dir)):
        if not (name.startswith("chunk_") and name.endswith("_catalog.csv")):
            continue
        path = os.path.join(out_dir, name)
        df = pd.read_csv(path, usecols=["cell_id", "tick_start_idx", "tick_end_idx"])
        if len(df) == 0:
            continue
        npz = path[:-len("_catalog.csv")] + "_functions.npz"
        if not os.path.isfile(npz):
            raise FileNotFoundError(npz)
        rows.append({
            "catalog": path,
            "npz": npz,
            "cells": int(len(df)),
            "first_cell": int(df["cell_id"].iloc[0]),
            "last_cell": int(df["cell_id"].iloc[-1]),
            "first_tick": int(df["tick_start_idx"].iloc[0]),
            "last_tick": int(df["tick_end_idx"].iloc[-1]),
        })
    rows.sort(key=lambda r: r["first_cell"])
    return rows


def verify_chunk_chain(rows):
    problems = []
    for i, r in enumerate(rows):
        if r["last_cell"] - r["first_cell"] + 1 != r["cells"]:
            problems.append(f"cell count mismatch in {os.path.basename(r['catalog'])}")
        if r["last_tick"] - r["first_tick"] + 1 != r["cells"] * 150:
            problems.append(f"tick span mismatch in {os.path.basename(r['catalog'])}")
        if i > 0:
            p = rows[i - 1]
            if r["first_cell"] != p["last_cell"] + 1:
                problems.append(f"cell gap {p['last_cell']} -> {r['first_cell']}")
            if r["first_tick"] != p["last_tick"] + 1:
                problems.append(f"tick gap {p['last_tick']} -> {r['first_tick']}")
    return problems


def write_npz_streaming(npz_path, rows, catalog, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    n = len(catalog)
    m = 150
    ck_path = os.path.join(temp_dir, "c_k.npy")
    raw_path = os.path.join(temp_dir, "raw_tick_cells.npy")
    rad_path = os.path.join(temp_dir, "radii.npy")
    pha_path = os.path.join(temp_dir, "phases.npy")

    ck_mm = open_memmap(ck_path, mode="w+", dtype=np.complex128, shape=(n, m))
    raw_mm = open_memmap(raw_path, mode="w+", dtype=np.float64, shape=(n, m))
    rad_mm = open_memmap(rad_path, mode="w+", dtype=np.float64, shape=(n, m))
    pha_mm = open_memmap(pha_path, mode="w+", dtype=np.float64, shape=(n, m))

    k_indices_ref = None
    p_mean = np.empty(n, dtype=np.float64)
    p_std = np.empty(n, dtype=np.float64)
    y_open = np.empty(n, dtype=np.float64)
    slope = np.empty(n, dtype=np.float64)
    cell_id = np.empty(n, dtype=np.int64)

    pos = 0
    for r in rows:
        with np.load(r["npz"], allow_pickle=False) as z:
            ids = z["cell_id"].astype(np.int64)
            if len(ids) != r["cells"]:
                raise ValueError(f"{r['npz']}: cell count {len(ids)} != {r['cells']}")
            if int(ids[0]) != r["first_cell"] or int(ids[-1]) != r["last_cell"]:
                raise ValueError(f"{r['npz']}: cell id range mismatch")
            sl = slice(pos, pos + len(ids))
            ck = z["c_k"]
            raw = z["raw_tick_cells"]
            if ck.shape != (len(ids), m) or raw.shape != (len(ids), m):
                raise ValueError(f"{r['npz']}: bad shapes ck={ck.shape} raw={raw.shape}")
            ck_mm[sl] = ck
            raw_mm[sl] = raw
            rad_mm[sl] = np.abs(ck)
            pha_mm[sl] = np.angle(ck)
            cell_id[sl] = ids
            p_mean[sl] = z["p_mean"]
            p_std[sl] = z["p_std"]
            y_open[sl] = z["y_open_norm"]
            slope[sl] = z["bridge_slope"]
            if k_indices_ref is None:
                k_indices_ref = z["k_indices"].astype(np.int64)
            pos += len(ids)
        log(f"  streamed {os.path.basename(r['npz'])} ({len(ids):,} cells)")

    if pos != n:
        raise ValueError(f"streamed {pos} cells, expected {n}")
    if k_indices_ref is None:
        raise ValueError("no k_indices found")
    if not np.array_equal(cell_id, catalog["cell_id"].to_numpy(np.int64)):
        raise ValueError("npz cell_id order does not match catalog order")

    for mm in (ck_mm, raw_mm, rad_mm, pha_mm):
        mm.flush()
    del ck_mm, raw_mm, rad_mm, pha_mm

    log(f"packing {npz_path}")
    with zipfile.ZipFile(npz_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.write(ck_path, arcname="c_k.npy")
        zf.write(raw_path, arcname="raw_tick_cells.npy")
        zf.write(rad_path, arcname="radii.npy")
        zf.write(pha_path, arcname="phases.npy")

        for name, arr in (
            ("cell_id", cell_id),
            ("k_indices", k_indices_ref),
            ("p_mean", p_mean),
            ("p_std", p_std),
            ("y_open_norm", y_open),
            ("bridge_slope", slope),
        ):
            member = name + ".npy"
            with zf.open(member, "w", force_zip64=True) as fp:
                write_array_header_1_0(fp, {
                    "shape": arr.shape,
                    "fortran_order": False,
                    "descr": dtype_to_descr(arr.dtype),
                })
                fp.write(arr.tobytes(order="C"))

    for p in (ck_path, raw_path, rad_path, pha_path):
        try:
            os.remove(p)
        except OSError:
            pass


def write_excel(path, catalog, rows, n_rows):
    rows = list(rows)
    pieces = []
    raw_pieces = []
    got = 0
    for r in rows:
        take = min(r["cells"], n_rows - got)
        if take <= 0:
            break
        with np.load(r["npz"], allow_pickle=False) as z:
            pieces.append(z["c_k"][:take])
            raw_pieces.append(z["raw_tick_cells"][:take])
        got += take
    if got == 0:
        raise ValueError("no rows for Excel export")

    ck = np.vstack(pieces)
    raw = np.vstack(raw_pieces)
    meta = catalog.iloc[:got].copy()

    df_ticks = pd.DataFrame(raw, columns=[f"Tick_{i+1}" for i in range(raw.shape[1])])
    df_ticks.insert(0, "cell_label", meta["cell_label"].values)

    k_idx = np.load(rows[0]["npz"], allow_pickle=False)["k_indices"].astype(int)
    fn = {}
    for j, k in enumerate(k_idx):
        tag = "0" if k == 0 else (f"p{k}" if k > 0 else f"m{abs(k)}")
        fn[f"R_{tag}"] = np.abs(ck[:, j])
        fn[f"Phi_{tag}"] = np.angle(ck[:, j])
    df_fn = pd.DataFrame(fn)
    df_fn.insert(0, "cell_label", meta["cell_label"].values)

    log(f"writing Excel {path} ({got:,} rows, {df_fn.shape[1]} function columns)")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        meta.to_excel(writer, sheet_name="Cell_Catalog", index=False)
        df_fn.to_excel(writer, sheet_name="Epicycle_Math_Functions", index=False)
        df_ticks.to_excel(writer, sheet_name="Raw_150_Tick_Cells", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\cluster\phase1\out")
    ap.add_argument("--final", default=r"C:\cluster\phase1\final")
    ap.add_argument("--excel-rows", type=int, default=10000)
    ap.add_argument("--skip-excel", action="store_true")
    ap.add_argument("--expected-first-cell", type=int, default=1)
    ap.add_argument("--expected-last-cell", type=int, default=591171)
    ap.add_argument("--expected-cells", type=int, default=591171)
    ap.add_argument("--skip-boundaries", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.final, exist_ok=True)
    temp_dir = os.path.join(args.final, "_tmp_npy")
    started = time.time()

    rows = load_chunk_index(args.out)
    if not rows:
        raise RuntimeError(f"no chunks found in {args.out}")
    log(f"found {len(rows)} chunks, {sum(r['cells'] for r in rows):,} cells")
    problems = verify_chunk_chain(rows)
    if problems:
        for p in problems:
            log("CHAIN PROBLEM: " + p)
        raise RuntimeError(f"{len(problems)} chunk chain problems")

    catalog_parts = []
    for r in rows:
        catalog_parts.append(pd.read_csv(r["catalog"]))
    catalog = pd.concat(catalog_parts, ignore_index=True)
    catalog = catalog.sort_values("cell_id", kind="stable").reset_index(drop=True)
    del catalog_parts

    ids = catalog["cell_id"].to_numpy(np.int64)
    starts = catalog["tick_start_idx"].to_numpy(np.int64)
    ends = catalog["tick_end_idx"].to_numpy(np.int64)
    if ids[0] != args.expected_first_cell:
        raise ValueError(f"first cell {ids[0]} != {args.expected_first_cell}")
    if ids[-1] != args.expected_last_cell:
        raise ValueError(f"last cell {ids[-1]} != {args.expected_last_cell}")
    if len(catalog) != args.expected_cells:
        raise ValueError(f"cell count {len(catalog)} != {args.expected_cells}")
    if not np.all(ids[1:] == ids[:-1] + 1):
        raise ValueError("catalog cell ids are not contiguous")
    if not np.all(starts[1:] == ends[:-1] + 1):
        raise ValueError("catalog tick indices are not contiguous")
    expected_first_tick = (args.expected_first_cell - 1) * 150 + 1
    if starts[0] != expected_first_tick:
        raise ValueError(f"first tick {starts[0]} != {expected_first_tick}")

    expected_last_tick = int(ids[-1]) * 150
    tail = int(catalog["tick_end_idx"].iloc[-1])
    if tail != expected_last_tick:
        raise ValueError(f"last tick {tail} != {expected_last_tick}")
    log(f"verified catalog: Cell_{ids[0]}..Cell_{ids[-1]}, ticks {starts[0]}..{tail}")

    parquet_path = os.path.join(args.final, "jump10_phase1_cell_catalog.parquet")
    log(f"writing {parquet_path}")
    catalog.to_parquet(parquet_path, index=False, engine="fastparquet")

    npz_path = os.path.join(args.final, "jump10_phase1_fourier_functions.npz")
    write_npz_streaming(npz_path, rows, catalog, temp_dir)

    excel_path = os.path.join(args.final, "jump10_phase1_inspection.xlsx")
    if not args.skip_excel:
        write_excel(excel_path, catalog, rows, args.excel_rows)

    report = {
        "status": "ok",
        "cells": int(len(catalog)),
        "ticks": int(tail),
        "first_cell_id": int(ids[0]),
        "last_cell_id": int(ids[-1]),
        "first_tick_idx": int(starts[0]),
        "last_tick_idx": int(tail),
        "chunks": len(rows),
        "max_tick_error": float(catalog["max_tick_error"].max()),
        "max_norm_rmse": float(catalog["norm_rmse"].max()),
        "cells_under_1e_11": int((catalog["max_tick_error"] < 1e-11).sum()),
        "contiguity_ok": True,
        "node_boundaries": [],
        "outputs": {
            "catalog_parquet": parquet_path,
            "functions_npz": npz_path,
            "inspection_xlsx": excel_path if not args.skip_excel else None,
        },
        "elapsed_sec": round(time.time() - started, 2),
    }

    if not args.skip_boundaries:
        for cid in (171621, 381378):
            row = catalog.loc[catalog["cell_id"] == cid]
            nxt = catalog.loc[catalog["cell_id"] == cid + 1]
            if len(row) and len(nxt):
                report["node_boundaries"].append({
                    "left_cell": int(cid),
                    "left_last_tick": int(row["tick_end_idx"].iloc[0]),
                    "right_cell": int(cid + 1),
                    "right_first_tick": int(nxt["tick_start_idx"].iloc[0]),
                    "ok": bool(int(nxt["tick_start_idx"].iloc[0]) == int(row["tick_end_idx"].iloc[0]) + 1),
                })

    report_path = os.path.join(args.final, "phase1_verification.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log("verification: " + json.dumps(report, indent=2))
    log("PHASE 1 MERGE COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
