"""Drive an already-running vLLM OpenAI server with isolated warm 256-tok prefills
(max_tokens=1) and report server-side latency. Used to test whether golden's async
serving engine runs the isolated <=512 prefill at ~16ms (vs offline generate 40ms).
"""
import sys, time, json, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8765"
URL = f"http://127.0.0.1:{PORT}/v1/completions"
MODEL = "qwen36"
def req(ntok):
    body = json.dumps({"model": MODEL, "prompt": list(range(ntok)), "max_tokens": 1,
                       "temperature": 0.0, "stream": False}).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=120) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000.0

for ntok in (256, 512):
    for _ in range(8): req(ntok)               # warm
    lat = sorted(req(ntok) for _ in range(20))
    med = lat[len(lat)//2]
    print(f"[{ntok}-tok] server-side isolated latency: median={med:.1f}ms min={lat[0]:.1f}ms "
          f"[{', '.join(f'{x:.1f}' for x in lat[:6])}]", flush=True)
print("(collector isolated execute_model_gpu ~33-40ms; golden FPM <=512 ~16ms)")
