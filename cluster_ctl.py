import socket, json, sys, re, time, concurrent.futures

def resolve_host(hostname, fallback_ip):
    try:
        return socket.gethostbyname(hostname)
    except Exception:
        return fallback_ip

NODES = {
    "kali":  ("192.168.150.166", 5555),
    "840g7": (resolve_host("QUISON", "192.168.150.126"), 5555),
    "xflip": ("127.0.0.1", 9999)
}

ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

def send_task(target, prompt, action="opencode", timeout=120):
    t0 = time.perf_counter()
    ip, port = NODES[target]
    with socket.create_connection((ip, port), timeout=10) as s:
        s.settimeout(timeout)
        s.sendall(json.dumps({"action": action, "prompt": prompt}).encode('utf-8'))
        chunks = []
        while True:
            data = s.recv(65536)
            if not data:
                break
            chunks.append(data.decode('utf-8', errors='replace'))
        ms = round((time.perf_counter() - t0) * 1000, 1)
        resp = json.loads("".join(chunks))
        clean_out = ANSI_RE.sub('', resp.get("output", "")).strip()
        return target, ip, resp.get("status"), ms, clean_out

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python C:\\cluster\\cluster_ctl.py [kali|840g7|all] \"prompt\" [opencode|fast_cmd|spawn_terminal]")
        sys.exit(1)

    target_arg = sys.argv[1].lower()
    prompt_arg = sys.argv[2]
    action_arg = sys.argv[3] if len(sys.argv) > 3 else "opencode"
    targets = ["kali", "840g7"] if target_arg == "all" else [target_arg]

    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures = {ex.submit(send_task, t, prompt_arg, action_arg): t for t in targets}
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            try:
                node, ip, status, ms, out = fut.result()
                print(f"\n=== [{node.upper()} ({ip}) | {status} | {ms} ms] ===\n{out}")
            except Exception as e:
                print(f"\n=== [{t.upper()} | ERROR] ===\n{e}")
