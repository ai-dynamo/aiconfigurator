"""Sample h200 SILICON FAILs, re-run each in HYBRID via the support-matrix in-process
path, and tally FAIL->PASS by failure category. A fast proxy for the full re-run."""

import collections
import csv
import os

os.environ["AIC_SM_DATABASE_MODE"] = "HYBRID"
import logging

logging.disable(logging.WARNING)
from tools.support_matrix.support_matrix import SupportMatrix, _get_test_constraints

SYS = "h200_sxm"


def category(msg):
    if "Failed to query moe data" in msg:
        return "moe-data"
    if "Failed to query context attention" in msg:
        return "ctx-attn"
    if "mHC" in msg or "DeepSeek-V4" in msg or ("DeepSeek-V3.2" in msg and "mHC" in msg):
        return "dsv4/mHC"
    if "dsa_context_module" in msg or "dsa_generation_module" in msg:
        return "dsa-file"
    if "Unsupported" in msg and "quant mode" in msg:
        return "unsup-quant"
    if "does not fit" in msg:
        return "config-OOM"
    if "non-wideep" in msg or "TP>1 and DP>1" in msg:
        return "topology"
    if "weight_block_size" in msg or "Invalid quantized MoE" in msg:
        return "block-align"
    return "other"


rows = []
with open(f"src/aiconfigurator/systems/support_matrix/{SYS}.csv") as _fh:
    for r in csv.DictReader(_fh):
        if r["Status"] == "FAIL":
            rows.append((r["HuggingFaceID"], r["Backend"], r["Version"], r["Mode"], category(r["ErrMsg"] or "")))

# stratified sample: up to 3 per category
bycat = collections.defaultdict(list)
for x in rows:
    bycat[x[4]].append(x)
sample = []
for c, xs in bycat.items():
    sample += xs[:3]
print(f"h200 SILICON FAIL rows={len(rows)}; categories={ {c: len(v) for c, v in bycat.items()} }")
print(f"sampling {len(sample)} combos (<=3/category)\n")

res = collections.Counter()
for model, bk, ver, mode, cat in sample:
    try:
        df = SupportMatrix._run_mode(
            mode=mode,
            model=model,
            system=SYS,
            backend=bk,
            version=ver,
            constraints=_get_test_constraints(model),
            engine_step_backend=None,
        )
        ok = df is not None and not df.empty
        verdict = "PASS" if ok else "FAIL(empty)"
    except Exception as e:
        verdict = f"FAIL({type(e).__name__})"
    res[(cat, verdict.split("(")[0])] += 1
    print(f"  [{cat:>11}] {mode:>6} {model.split('/')[-1][:34]:<34} {bk}/{ver} -> {verdict}")

print("\n=== HYBRID outcome by category (sampled) ===")
cats = sorted({c for c, _ in res})
for c in cats:
    p = res[(c, "PASS")]
    f = sum(n for (cc, v), n in res.items() if cc == c and v.startswith("FAIL"))
    print(f"  {c:>11}: PASS {p} / FAIL {f}")
