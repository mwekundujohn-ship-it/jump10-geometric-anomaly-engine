import os
import sys
import json
import time
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote

ROOT = r"C:\cluster\phase1"
DATA = r"C:\Users\dalto\mt5-opencode\data\ticks"
SCRIPTS = os.path.join(ROOT, "scripts")
JOBS = os.path.join(ROOT, "jobs")
OUT = os.path.join(ROOT, "out")
LOGS = os.path.join(ROOT, "logs")
PORT = 8080

for d in (SCRIPTS, JOBS, OUT, LOGS):
    os.makedirs(d, exist_ok=True)


def safe_join(base, rel):
    rel = rel.replace("\\", "/").lstrip("/")
    parts = [p for p in rel.split("/") if p not in ("", ".", "..")]
    return os.path.join(base, *parts)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        line = "%s - %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), fmt % args)
        with open(os.path.join(LOGS, "http_access.log"), "a", encoding="utf-8") as f:
            f.write(line)

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])
        if path == "/health":
            self._send_json(200, {"ok": True, "root": ROOT, "data": DATA})
            return
        if path.startswith("/data/"):
            base, rel = DATA, path[len("/data/"):]
        elif path.startswith("/scripts/"):
            base, rel = SCRIPTS, path[len("/scripts/"):]
        elif path.startswith("/jobs/"):
            base, rel = JOBS, path[len("/jobs/"):]
        elif path.startswith("/out/"):
            base, rel = OUT, path[len("/out/"):]
        else:
            self.send_error(404)
            return
        fp = safe_join(base, rel)
        if not os.path.isfile(fp):
            self.send_error(404)
            return
        size = os.path.getsize(fp)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(fp, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def do_PUT(self):
        path = unquote(self.path.split("?", 1)[0])
        if not path.startswith("/upload/"):
            self.send_error(404)
            return
        rel = path[len("/upload/"):]
        fp = safe_join(OUT, rel)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        length = int(self.headers.get("Content-Length", "0"))
        tmp = fp + ".part"
        got = 0
        with open(tmp, "wb") as f:
            while got < length:
                chunk = self.rfile.read(min(1024 * 1024, length - got))
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
        if got != length:
            try:
                os.remove(tmp)
            except OSError:
                pass
            self._send_json(400, {"ok": False, "error": "short read", "got": got, "expected": length})
            return
        os.replace(tmp, fp)
        self._send_json(200, {"ok": True, "path": fp, "bytes": got})

    def do_POST(self):
        self.do_PUT()


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[PHASE1 MASTER HTTP] listening on 0.0.0.0:{PORT}", flush=True)
    print(f"[PHASE1 MASTER HTTP] data={DATA}", flush=True)
    print(f"[PHASE1 MASTER HTTP] out={OUT}", flush=True)
    srv.serve_forever()
