import sys, os, platform, shutil
print("python", sys.version.split()[0], "exe", sys.executable)
print("machine", platform.machine(), "cpus", os.cpu_count())
try:
    import numpy; print("numpy", numpy.__version__)
except Exception as e:
    print("numpy MISSING", e)
try:
    import pandas; print("pandas", pandas.__version__)
except Exception as e:
    print("pandas MISSING", type(e).__name__, e)
try:
    import torch; print("torch", torch.__version__, "threads", torch.get_num_threads())
except Exception as e:
    print("torch MISSING", type(e).__name__)
try:
    import openpyxl; print("openpyxl", openpyxl.__version__)
except Exception as e:
    print("openpyxl MISSING", type(e).__name__)
try:
    import fastparquet; print("fastparquet", fastparquet.__version__)
except Exception as e:
    print("fastparquet MISSING", type(e).__name__)
t, u, f = shutil.disk_usage(".")
print("disk free GB", round(f / 1e9, 1))
try:
    import psutil; print("ram free GB", round(psutil.virtual_memory().available / 1e9, 2))
except Exception:
    print("ram free GB n/a")
