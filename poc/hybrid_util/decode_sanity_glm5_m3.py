"""Decode sanity check: is M3's flat TPOT real, and is the transfer actually firing?

Compares decode TPOT (bs=64, b200/trtllm) across kv for:
  - GLM-5 SILICON (real DSA sparse attention -> ground-truth "is sparse decode flat?")
  - M3 fp8_block HYBRID (MSA transfers from DSA util -- the FAITHFUL path)
  - M3 NVFP4 HYBRID  (b200 has no NVFP4 DSA data -> MSA falls back to the constant)

Finding: all three are flat in kv (sparse decode is context-independent, top-k cap),
so M3's flat decode is physical. But NVFP4 M3 is the constant fallback, not a DSA
transfer -- use fp8_block (or collect NVFP4 DSA data) for a faithful number.
"""

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database


def main():
    db = get_database("b200_sxm", "trtllm", "1.3.0rc10")
    be = TRTLLMBackend()
    par = dict(tp_size=8, moe_tp_size=1, moe_ep_size=8)
    glm = get_model("zai-org/GLM-5-FP8", config.ModelConfig(**par), backend_name="trtllm")
    m3_fp8 = get_model(
        "MiniMaxAI/MiniMax-M3",
        config.ModelConfig(
            **par,
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
            moe_quant_mode=common.MoEQuantMode.fp8_block,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        ),
        backend_name="trtllm",
    )
    m3_nvfp4 = get_model(
        "MiniMaxAI/MiniMax-M3",
        config.ModelConfig(
            **par,
            gemm_quant_mode=common.GEMMQuantMode.nvfp4,
            moe_quant_mode=common.MoEQuantMode.nvfp4,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        ),
        backend_name="trtllm",
    )

    def tpot(model, mode, kv):
        db.set_default_database_mode(mode)
        s = be.run_static(
            model,
            db,
            config.RuntimeConfig(batch_size=64, beam_width=1, isl=kv, osl=2, prefix=0),
            mode="static_gen",
            stride=1,
        )
        return sum(s.get_generation_latency_dict().values())

    print(f"{'kv':>7}{'GLM5 sil':>11}{'M3 fp8 xfer':>13}{'M3 nvfp4 const':>16}")
    for kv in (2048, 4096, 8192, 16384):
        g = tpot(glm, common.DatabaseMode.SILICON, kv)
        a = tpot(m3_fp8, common.DatabaseMode.HYBRID, kv)
        b = tpot(m3_nvfp4, common.DatabaseMode.HYBRID, kv)
        print(f"{kv:>7}{g:>11.2f}{a:>13.2f}{b:>16.2f}")


if __name__ == "__main__":
    main()
