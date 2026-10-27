"""Worker bootstrap: pulls nothing (data already local), sets env and runs the engine."""
import os
import sys
import subprocess

here = os.path.dirname(os.path.abspath(__file__))
node = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["P3_INPUTS"] = os.path.join(here, "phase3_inputs.npz")
os.environ["P3_VAULT"] = os.path.join(here, "phase3_vault.csv")
os.environ["P3_TIMES"] = os.path.join(here, "phase3_catalog_times.npy")
cmd = [sys.executable, "-u", os.path.join(here, "phase3_master_engine.py"),
       "--node_id", str(node), "--num_nodes", "3", "--out_dir", here]
print("[*] " + " ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
