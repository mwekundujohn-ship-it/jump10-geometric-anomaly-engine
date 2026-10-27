import socket, time, json, re, concurrent.futures

NODES = {
    "KALI-DELL":       ("192.168.150.166", 5555, "opencode", "Reply with ONLY your hostname in one word, nothing else."),
    "840-G7 (QUISON)": ("192.168.150.126", 5555, "opencode", "Reply with ONLY your hostname in one word, nothing else."),
    "840-G7 (FAST_CMD)": ("192.168.150.126", 5555, "fast_cmd", "hostname")
}

ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def clean_out(text):
    text = ANSI_RE.sub('', text or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:35] if lines else "EMPTY"

def run_bench(label, ip, port, action, prompt):
    # 1. Measure raw TCP handshake latency
    t_ping0 = time.perf_counter()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ps:
            ps.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            ps.settimeout(5.0)
            ps.connect((ip, port))
        tcp_ms = (time.perf_counter() - t_ping0) * 1000.0
    except Exception as e:
        return {
            "label": label, "tcp_ms": -1.0, "worker_ms": -1.0,
            "total_ms": -1.0, "output": f"ERR: {e}"
        }

    # 2. Measure actual task dispatch + execution round-trip
    t_start = time.perf_counter()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(120.0)
            s.connect((ip, port))
            payload = {"action": action, "prompt": prompt, "cmd": prompt}
            s.sendall(json.dumps(payload).encode('utf-8'))
            # Signal EOF on write side so the worker never hangs waiting for more bytes
            s.shutdown(socket.SHUT_WR)

            chunks = []
            while True:
                data = s.recv(65536)
                if not data:
                    break
                chunks.append(data.decode('utf-8', errors='replace'))

        total_ms = (time.perf_counter() - t_start) * 1000.0
        raw_resp = "".join(chunks)
        resp = json.loads(raw_resp)
        # Use worker's internal timer if available, otherwise total minus TCP ping
        worker_ms = float(resp.get("worker_ms", max(0.0, total_ms - tcp_ms)))
        output = clean_out(resp.get("output", ""))
        return {
            "label": label, "tcp_ms": tcp_ms, "worker_ms": worker_ms,
            "total_ms": total_ms, "output": output
        }
    except Exception as e:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "label": label, "tcp_ms": tcp_ms, "worker_ms": total_ms,
            "total_ms": total_ms, "output": f"ERR: {e}"
        }

if __name__ == "__main__":
    print("\nDispatching identical task to KALI (.166) and 840-G7 (.126) simultaneously...\n")
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futs = {
            ex.submit(run_bench, label, ip, port, act, prm): label
            for label, (ip, port, act, prm) in NODES.items()
        }
        for f in concurrent.futures.as_completed(futs):
            res = f.result()
            results[res["label"]] = res

    print("=" * 92)
    print(f"{'MACHINE':<20} | {'TCP PING (ms)':<14} | {'WORKER EXEC (ms)':<17} | {'TOTAL TIME (ms)':<16} | OUTPUT")
    print("-" * 92)
    for label in ["KALI-DELL", "840-G7 (QUISON)", "840-G7 (FAST_CMD)"]:
        r = results[label]
        print(f"{r['label']:<20} | {r['tcp_ms']:<14.1f} | {r['worker_ms']:<17.1f} | {r['total_ms']:<16.1f} | {r['output']}")
    print("=" * 92)

    k = results["KALI-DELL"]
    q = results["840-G7 (QUISON)"]
    f = results["840-G7 (FAST_CMD)"]
    if k["total_ms"] > 0 and q["total_ms"] > 0:
        tcp_diff = q["tcp_ms"] - k["tcp_ms"]
        tot_diff = q["total_ms"] - k["total_ms"]
        ratio = q["total_ms"] / k["total_ms"] if k["total_ms"] else 0
        print("\n[LAG SUMMARY]")
        print(f" -> Network (TCP) Difference : 840-G7 is {tcp_diff:+.1f} ms vs Kali")
        print(f" -> Total Execution Delay    : 840-G7 took {tot_diff:+.1f} ms ({tot_diff/1000:+.2f} sec) vs Kali ({ratio:.2f}x)")
        print(f" -> 840-G7 Direct Shell Time : {f['total_ms']:.1f} ms (raw Windows speed without OpenCode cold-start)\n")
