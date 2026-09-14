import json, time, urllib.request, sys
port, label = int(sys.argv[1]), sys.argv[2]
body = json.dumps({"messages":[{"role":"user","content":"Explain how speculative decoding works in large language model inference, in three short paragraphs."}],"max_tokens":256,"temperature":0.0,"stream":False}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type":"application/json"})
t0 = time.time()
r = json.loads(urllib.request.urlopen(req, timeout=300).read())
dt = time.time() - t0
n = r.get("usage",{}).get("completion_tokens",0)
print(f"{label}: {n} tokens in {dt:.1f}s = {n/dt:.2f} tok/s")