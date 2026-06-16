"""Compare SILICON vs EMPIRICAL run_static latency across (bs, seq, pastkv).

Reports MAPE / p95 / p99 / max of |empirical - silicon| / silicon, treating
SILICON as the reference -- the model-level accuracy of empirical mode.

Usage:  python run_static_compare.py [config_name]
configs: llama70b (default), glm5

pastkv = prefix (cached kv length); isl = seq + pastkv. Silicon may raise on
incomplete grids -- those points are counted separately, not silently dropped.
"""

import sys

import numpy as np

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database

BS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEQ = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]  # new prompt tokens to prefill

CONFIGS = {
    "llama70b": dict(
        model="meta-llama/Meta-Llama-3.1-70B",
        system="h100_sxm",
        version="1.3.0rc10",
        pastkv=[0, 1024, 4096],  # prefix / cached kv length; isl = seq + pastkv
        model_config=lambda: config.ModelConfig(tp_size=8, gemm_quant_mode=common.GEMMQuantMode.fp8),
    ),
    # GLM-5: GlmMoeDsaForCausalLM -> MoE + DSA. tp*attn_dp == moe_tp*moe_ep.
    # pastkv=0 only: DSA-context perf table has no prefix>0 silicon data yet.
    "glm5": dict(
        model="zai-org/GLM-5-FP8",
        system="b200_sxm",
        version="1.3.0rc10",
        pastkv=[0],
        model_config=lambda: config.ModelConfig(tp_size=8, moe_tp_size=1, moe_ep_size=8),
    ),
    # GLM-5 WideEP (trtllm): TrtLLMWideEPMoE + TrtLLMWideEPMoEDispatch + DSA.
    "glm5_wideep": dict(
        model="zai-org/GLM-5-FP8",
        system="b200_sxm",
        version="1.3.0rc10",
        pastkv=[0],
        model_config=lambda: config.ModelConfig(
            tp_size=8, attention_dp_size=8, moe_tp_size=1, moe_ep_size=64, enable_wideep=True
        ),
    ),
    # DeepSeek-V3: MLA (FallbackOp primary = MLAModule ctx/gen) + MoE.
    # bf16 quant: MLA-module/context perf tables only collected mla_dtype=bfloat16.
    "deepseek_v3": dict(
        model="deepseek-ai/DeepSeek-V3",
        system="h200_sxm",
        version="1.3.0rc10",
        pastkv=[0],
        model_config=lambda: config.ModelConfig(
            tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        ),
    ),
}


def collect(name):
    """Run the bs x seq x pastkv sweep for a config; return per-point records.

    apes is a list of (ape, bs, seq, pk, silicon, empirical) -- ape=|emp-sil|/sil.
    Used both by main() (summary) and plot_gate_errors.py (distribution).
    """
    cfg = CONFIGS[name]
    backend_name = "trtllm"
    db = get_database(cfg["system"], backend_name, cfg["version"])  # shared layer OFF for both modes
    model = get_model(cfg["model"], cfg["model_config"](), backend_name=backend_name)
    backend = TRTLLMBackend()

    def latency(mode_enum, rc):
        db.set_default_database_mode(mode_enum)
        return backend.run_static_latency_only(model, db, rc, mode="static", stride=8)

    pastkv = cfg["pastkv"]
    apes, sil_fail, both_ok = [], 0, 0
    fail_types = {}
    for bs in BS:
        for seq in SEQ:
            for pk in pastkv:
                rc = config.RuntimeConfig(batch_size=bs, beam_width=1, isl=seq + pk, osl=8, prefix=pk)
                try:
                    sil = latency(common.DatabaseMode.SILICON, rc)
                except Exception as e:
                    sil_fail += 1
                    fail_types[type(e).__name__] = fail_types.get(type(e).__name__, 0) + 1
                    continue
                emp = latency(common.DatabaseMode.EMPIRICAL, rc)
                if sil and sil > 0 and emp and emp > 0:
                    # SIGNED relative error (emp-sil)/sil: >0 over-predict, <0 under-predict.
                    apes.append(((emp - sil) / sil, bs, seq, pk, sil, emp))
                    both_ok += 1
    return {
        "name": name,
        "cfg": cfg,
        "backend": backend_name,
        "pastkv": pastkv,
        "apes": apes,
        "sil_fail": sil_fail,
        "fail_types": fail_types,
        "both_ok": both_ok,
    }


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "llama70b"
    r = collect(name)
    cfg, backend_name, pastkv = r["cfg"], r["backend"], r["pastkv"]
    apes, sil_fail, fail_types, both_ok = r["apes"], r["sil_fail"], r["fail_types"], r["both_ok"]

    print(f"\n=== {name}: {cfg['model']} on {cfg['system']}/{backend_name}/{cfg['version']}")
    print(f"grid: bs{len(BS)} x seq{len(SEQ)} x pastkv{len(pastkv)} = {len(BS) * len(SEQ) * len(pastkv)} points")
    print(f"silicon failures: {sil_fail} {fail_types}   scored points: {both_ok}")
    if apes:
        signed = np.array([a for a, *_ in apes]) * 100  # signed (emp-sil)/sil
        av = np.abs(signed)
        print("\n  empirical-vs-silicon  (emp-sil)/sil")
        print(f"    MAPE%  {av.mean():7.2f}    median|e| {np.median(av):7.2f}    bias(signed) {signed.mean():+7.2f}")
        print(
            f"    p95|e| {np.percentile(av, 95):7.2f}    p99|e| {np.percentile(av, 99):7.2f}    max|e| {av.max():7.2f}"
        )
        print("\n  worst 5 points (bs, seq, pastkv | silicon -> empirical | signed%):")
        for a, bs, seq, pk, sil, emp in sorted(apes, key=lambda t: abs(t[0]), reverse=True)[:5]:
            print(f"    bs={bs:<4} seq={seq:<6} pk={pk:<5} | {sil:8.3f} -> {emp:8.3f} | {a * 100:+7.1f}")


if __name__ == "__main__":
    sys.exit(main())
