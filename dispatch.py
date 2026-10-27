import socket, sys, json, concurrent.futures

NODES = {
    "840g7": ("192.168.150.216", 5555),
    "dell":  ("192.168.150.166", 5555)
}

def call_worker(node_name, prompt):
    ip, port = NODES[node_name]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(600)
            s.connect((ip, port))
            s.sendall(json.dumps({"prompt": prompt}).encode('utf-8'))
            chunks = []
            while True:
                data = s.recv(65536)
                if not data:
                    break
                chunks.append(data.decode('utf-8'))
            return json.loads("".join(chunks))
    except Exception as e:
        return {"node": node_name, "status": "unreachable", "output": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python C:\\cluster\\dispatch.py [840g7|dell|all] \"<task prompt>\"")
        sys.exit(1)

    target = sys.argv[1].lower()
    prompt = " ".join(sys.argv[2:])
    targets = ["840g7", "dell"] if target == "all" else [target]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(call_worker, t, prompt): t for t in targets}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            print(f"\n================ [{res.get('node')} | {res.get('status')}] ================")
            print(res.get("output", ""))
            print("================================================================")
