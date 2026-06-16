"""MiniMax-M3 pure-hybrid run_static vs the sglang B200 cookbook number.

sglang cookbook (docs.sglang.io, MiniMax-M3, B200 mxfp8, TP8, single node,
ISL 2048 / OSL 256, max-concurrency 64): TTFT 1580 ms, TPOT 24.1 ms,
265 tok/s/GPU.

M3 here runs HYBRID: MoE from real silicon data; MSA has no data so it transfers
from DSA's measured util (k = msa_dsa_scale_k, default 1.0). TPOT ~ one decode
step latency at batch=concurrency, kv ~ isl + osl/2. fp8_block ~ mxfp8.
"""

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database

SGLANG_TPOT_MS = 24.1


def main():
    db = get_database("b200_sxm", "trtllm", "1.3.0rc10")
    be = TRTLLMBackend()
    mc = config.ModelConfig(
        tp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
    )
    model = get_model("MiniMaxAI/MiniMax-M3", mc, backend_name="trtllm")  # full 60 layers
    db.set_default_database_mode(common.DatabaseMode.HYBRID)

    bs, kv = 64, 2048 + 128  # concurrency, avg kv length over the 256-token decode
    rc = config.RuntimeConfig(batch_size=bs, beam_width=1, isl=kv, osl=2, prefix=0)
    summ = be.run_static(model, db, rc, mode="static_gen", stride=1)
    tpot = sum(summ.get_generation_latency_dict().values())
    print(f"M3 HYBRID TPOT (bs={bs}, kv={kv}): {tpot:.2f} ms")
    print(f"sglang B200 mxfp8 TPOT       : {SGLANG_TPOT_MS} ms")
    print(f"ratio (est/ref)             : {tpot / SGLANG_TPOT_MS:.2f}")
    top = sorted(summ.get_generation_latency_dict().items(), key=lambda x: -x[1])[:4]
    print("gen breakdown:", {k: round(v, 3) for k, v in top})


if __name__ == "__main__":
    main()
