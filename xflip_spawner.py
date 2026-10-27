import socket, subprocess, json

HOST = '0.0.0.0'
PORT = 9999

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(5)

print(f"[X-FLIP COCKPIT SPAWNER ONLINE] Listening on {HOST}:{PORT} (IP: 192.168.150.110)...")

while True:
    conn, addr = server.accept()
    try:
        raw = conn.recv(65536).decode('utf-8')
        if not raw:
            continue
        req = json.loads(raw)
        cmd = req.get("prompt", req.get("cmd", "echo Hello from Cluster"))
        title = req.get("title", f"Worker_{addr[0]}")
        print(f"\n[SPAWN REQUEST FROM {addr[0]}]: {cmd}")
        subprocess.Popen(f'start "{title}" cmd /k "{cmd}"', shell=True)
        conn.sendall(json.dumps({"node": "X-FLIP", "status": "spawned", "output": f"Spawned window on X-Flip: {title}"}).encode('utf-8'))
    except Exception as e:
        conn.sendall(json.dumps({"node": "X-FLIP", "status": "error", "output": str(e)}).encode('utf-8'))
    finally:
        conn.close()
