import argparse
import json
import os
import zipfile

import numpy as np
import pandas as pd
from openpyxl import load_workbook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=r"C:\cluster\phase1\final")
    args = ap.parse_args()

    npz_path = os.path.join(args.final, "jump10_phase1_fourier_functions.npz")
    parquet_path = os.path.join(args.final, "jump10_phase1_cell_catalog.parquet")
    xlsx_path = os.path.join(args.final, "jump10_phase1_inspection.xlsx")

    report = {"status": "ok", "checks": {}}

    with zipfile.ZipFile(npz_path) as zf:
        bad = zf.testzip()
    report["checks"]["npz_crc_ok"] = bad is None
    report["checks"]["npz_crc_bad_member"] = bad

    catalog = pd.read_parquet(parquet_path, engine="fastparquet")
    n = len(catalog)
    expected = np.arange(1, n + 1, dtype=np.int64)
    first = catalog["tick_start_idx"].to_numpy(np.int64)
    last = catalog["tick_end_idx"].to_numpy(np.int64)
    report["checks"]["catalog_rows"] = n
    report["checks"]["catalog_columns"] = int(catalog.shape[1])
    report["checks"]["cell_ids_contiguous"] = bool(np.array_equal(catalog["cell_id"].to_numpy(np.int64), expected))
    report["checks"]["cell_labels_match"] = bool(
        np.array_equal(catalog["cell_label"].astype(str).to_numpy(), np.char.add("Cell_", expected.astype(str)))
    )
    report["checks"]["tick_ranges_contiguous"] = bool(
        np.all(first == 1 + 150 * np.arange(n, dtype=np.int64))
        and np.all(last == 150 * np.arange(1, n + 1, dtype=np.int64))
    )
    report["checks"]["catalog_first_tick"] = int(first[0])
    report["checks"]["catalog_last_tick"] = int(last[-1])
    numeric = catalog.select_dtypes(include=[np.number])
    report["checks"]["catalog_all_finite"] = bool(np.isfinite(numeric.to_numpy()).all())

    sample = np.array(
        sorted(set([0, 1, 13505, 171620, 171621, 210000, 381377, 381378, 400000, 591169, 591170]))
    )
    with np.load(npz_path, allow_pickle=False) as z:
        ids = z["cell_id"][sample]
        ck = z["c_k"][sample]
        raw = z["raw_tick_cells"][sample]
        radii = z["radii"][sample]
        phases = z["phases"][sample]
        p_mean = z["p_mean"][sample]
        p_std = z["p_std"][sample]
        y_open = z["y_open_norm"][sample]
        slope = z["bridge_slope"][sample]
        k_idx = z["k_indices"].astype(np.int64)
        report["checks"]["npz_cell_id_count"] = int(z["cell_id"].shape[0])
        report["checks"]["npz_cell_id_ok"] = bool(np.array_equal(z["cell_id"].astype(np.int64), expected))

    t = np.linspace(0.0, 1.0, 150, dtype=np.float64)[None, :]
    z_rec = np.fft.ifft(ck * 150.0, axis=1)
    y_norm_rec = z_rec.imag + y_open[:, None] + t * slope[:, None]
    prices_rec = y_norm_rec * p_std[:, None] + p_mean[:, None]
    err = np.abs(prices_rec - raw)
    report["checks"]["sample_rows"] = sample.tolist()
    report["checks"]["sample_cell_ids"] = ids.astype(int).tolist()
    report["checks"]["sample_max_abs_error"] = float(err.max())
    report["checks"]["sample_max_norm_rmse"] = float(
        np.sqrt(np.mean((prices_rec - raw) ** 2 / (p_std[:, None] ** 2), axis=1)).max()
    )
    report["checks"]["sample_all_finite"] = bool(
        np.isfinite(ck).all()
        and np.isfinite(raw).all()
        and np.isfinite(radii).all()
        and np.isfinite(phases).all()
        and np.isfinite(prices_rec).all()
    )
    report["checks"]["radii_match"] = bool(np.allclose(radii, np.abs(ck), rtol=0.0, atol=0.0))
    report["checks"]["phases_match"] = bool(np.allclose(phases, np.angle(ck), rtol=0.0, atol=0.0))
    report["checks"]["k_indices_ok"] = bool(np.array_equal(k_idx, np.fft.fftfreq(150, d=1.0 / 150).astype(np.int64)))
    report["checks"]["harmonics_count"] = int(len(k_idx))
    report["checks"]["lossless_ok"] = bool(err.max() < 1e-11)

    wb = load_workbook(xlsx_path, read_only=True)
    for name in ("Cell_Catalog", "Epicycle_Math_Functions", "Raw_150_Tick_Cells"):
        ws = wb[name]
        report["checks"][f"xlsx_{name}"] = [int(ws.max_row), int(ws.max_column)]
    wb.close()

    report["outputs"] = {
        "catalog_parquet_bytes": os.path.getsize(parquet_path),
        "functions_npz_bytes": os.path.getsize(npz_path),
        "inspection_xlsx_bytes": os.path.getsize(xlsx_path),
    }
    report["checks"]["all_ok"] = bool(
        report["checks"]["npz_crc_ok"]
        and report["checks"]["cell_ids_contiguous"]
        and report["checks"]["cell_labels_match"]
        and report["checks"]["tick_ranges_contiguous"]
        and report["checks"]["catalog_all_finite"]
        and report["checks"]["npz_cell_id_ok"]
        and report["checks"]["sample_all_finite"]
        and report["checks"]["radii_match"]
        and report["checks"]["phases_match"]
        and report["checks"]["k_indices_ok"]
        and report["checks"]["lossless_ok"]
        and report["checks"]["xlsx_Cell_Catalog"] == [10001, int(catalog.shape[1] + 0)]
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

