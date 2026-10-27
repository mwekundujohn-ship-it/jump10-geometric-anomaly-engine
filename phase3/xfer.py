"""Tiny transfer helper for the 8080 file bridge.

The master's LAN address is DHCP-assigned, so every call probes a candidate list and
uses the first host that answers GET /out/phase2/pqcompat.py.

  python xfer.py put <local> <relpath> [area=phase2]
  python xfer.py get <relpath> <local> [area=phase2]
  python xfer.py url
"""
import os
import sys
import socket
import urllib.request

CANDIDATES = ["192.168.150.199", "192.168.150.237", "192.168.150.157",
              "192.168.150.110", "192.168.150.100", "192.168.150.101"]
PORT = 8080
PROBE = "/out/phase2/pqcompat.py"


def base():
    for ip in CANDIDATES:
        s = socket.socket()
        s.settimeout(0.6)
        ok = s.connect_ex((ip, PORT)) == 0
        s.close()
        if not ok:
            continue
        try:
            with urllib.request.urlopen(f"http://{ip}:{PORT}{PROBE}", timeout=3) as r:
                if r.status == 200:
                    return f"http://{ip}:{PORT}"
        except Exception:
            continue
    raise SystemExit("no reachable master on port 8080")


def put(local, rel, area="phase2"):
    url = f"{base()}/upload/{area}/{rel}"
    data = open(local, "rb").read()
    req = urllib.request.Request(url, data=data, method="PUT")
    with urllib.request.urlopen(req, timeout=1800) as r:
        return r.status


def get(rel, local, area="phase2"):
    url = f"{base()}/out/{area}/{rel}"
    with urllib.request.urlopen(url, timeout=1800) as r:
        data = r.read()
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
    with open(local, "wb") as f:
        f.write(data)
    return len(data)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "url":
        print(base())
    elif sys.argv[1] == "put":
        print("put", sys.argv[3], put(sys.argv[2], sys.argv[3], *(sys.argv[4:5] or [])))
    elif sys.argv[1] == "get":
        print("get", sys.argv[2], get(sys.argv[2], sys.argv[3], *(sys.argv[4:5] or [])), "bytes")
    else:
        raise SystemExit(__doc__)
