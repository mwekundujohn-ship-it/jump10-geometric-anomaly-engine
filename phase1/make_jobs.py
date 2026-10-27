import json
import os
from pathlib import Path

BASE = r"C:\Users\dalto\mt5-opencode\data\ticks\Jump_10_Index"
OUT = r"C:\cluster\phase1\jobs"
MASTER = "192.168.150.157"
PORT = 8080

with open(os.path.join(BASE, "manifest.json"), "r", encoding="utf-8") as f:
    manifest = json.load(f)

months = {m["file"]: m for m in manifest["monthly"]}
ordered = sorted(months)

ranges = {
    "xflip": ("2021-03", "2021-12"),
    "840g7": ("2022-01", "2022-12"),
    "kali": ("2023-01", "2023-12"),
}

data_dirs = {
    "xflip": BASE,
    "840g7": r"C:\cluster\phase1\data",
    "kali": "/home/user/cluster/phase1/data",
}
out_dirs = {
    "xflip": r"C:\cluster\phase1\out",
    "840g7": r"C:\cluster\phase1\out",
    "kali": "/home/user/cluster/phase1/out",
}

# cumulative rows before each file
cum = {}
running = 0
for f in ordered:
    cum[f] = running
    running += months[f]["rows"]

os.makedirs(OUT, exist_ok=True)

for node, (start, end) in ranges.items():
    files = [f for f in ordered if start <= f.split("_")[-1][:7] <= end]
    prev = None
    idx = ordered.index(files[0])
    if idx > 0:
        prev = ordered[idx - 1]
    O = cum[files[0]]
    R = O % 150
    job = {
        "node": node,
        "data_dir": data_dirs[node],
        "files": files,
        "file_rows": [months[f]["rows"] for f in files],
        "carry_file": prev,
        "carry_rows": R,
        "carry_file_rows": months[prev]["rows"] if prev else 0,
        "harmonics_K": 75,
        "output_dir": out_dirs[node],
        "upload": node != "xflip",
        "upload_url": f"http://{MASTER}:{PORT}/upload/",
        "data_url_base": f"http://{MASTER}:{PORT}/data/Jump_10_Index/",
        "global_start_tick": O - R + 1,
        "global_start_cell": (O - R) // 150 + 1,
    }
    dl = []
    if prev:
        dl.append(prev)
    dl.extend(files)
    job["download_files"] = [
        {"name": f, "size": os.path.getsize(os.path.join(BASE, f))} for f in dl
    ] if node != "xflip" else []
    path = os.path.join(OUT, f"job_{node}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)
    print(f"{node}: {len(files)} files, {sum(months[f]['rows'] for f in files):,} ticks, "
          f"carry={R} from {prev}, cells {job['global_start_cell']}..., job={path}")

print(f"total ticks assigned: {sum(months[f]['rows'] for node,(s,e) in ranges.items() for f in ordered if s <= f.split('_')[-1][:7] <= e):,}")
