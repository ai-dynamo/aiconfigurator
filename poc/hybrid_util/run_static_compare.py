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

BS = [1, 4, 16, 64, 256]
SEQ = [128, 512, 2048, 8192]  # new prompt tokens to prefill
PASTKV = [0, 1024, 4096]  # prefix / cached kv length; isl = seq + pastkv

CONFIGS = {
    "llama70b": dict(
        model="meta-llama/Meta-Llama-3.1-70B",
        system="h100_sxm",
        version="1.3.0rc10",
        model_config=lambda: config.ModelConfig(tp_size=8, gemm_quant_mode=common.GEMMQuantMode.fp8),
    ),
    # GLM-5: GlmMoeDsaForCausalLM -> MoE + DSA. tp*attn_dp == moe_tp*moe_ep.
    "glm5": dict(
        model="zai-org/GLM-5-FP8",
        system="b200_sxm",
        version="1.3.0rc10",
        model_config=lambda: config.ModelConfig(tp_size=8, moe_tp_size=1, moe_ep_size=8),
    ),
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "llama70b"
    cfg = CONFIGS[name]
    backend_name = "trtllm"
    db = get_database(cfg["system"], backend_name, cfg["version"])  # shared layer OFF for both modes
    model = get_model(cfg["model"], cfg["model_config"](), backend_name=backend_name)
    backend = TRTLLMBackend()

    def latency(mode_enum, rc):
        db.set_default_database_mode(mode_enum)
        return backend.run_static_latency_only(model, db, rc, mode="static", stride=8)

    apes, sil_fail, both_ok = [], 0, 0
    fail_types = {}
    for bs in BS:
        for seq in SEQ:
            for pk in PASTKV:
                rc = config.RuntimeConfig(batch_size=bs, beam_width=1, isl=seq + pk, osl=8, prefix=pk)
                try:
                    sil = latency(common.DatabaseMode.SILICON, rc)
                except Exception as e:
                    sil_fail += 1
                    fail_types[type(e).__name__] = fail_types.get(type(e).__name__, 0) + 1
                    if sil_fail <= 3:
                        print(f"  FAIL bs={bs} seq={seq} pk={pk}: {type(e).__name__}: {str(e)[:200]}")
                    continue
                emp = latency(common.DatabaseMode.EMPIRICAL, rc)
                if sil and sil > 0 and emp and emp > 0:
                    apes.append(abs(emp - sil) / sil)
                    both_ok += 1

    apes = np.array(apes) * 100
    print(f"\n=== {name}: {cfg['model']} on {cfg['system']}/{backend_name}/{cfg['version']}")
    print(f"grid: {len(BS)}x{len(SEQ)}x{len(PASTKV)} = {len(BS) * len(SEQ) * len(PASTKV)} points")
    print(f"silicon failures: {sil_fail} {fail_types}   scored points: {both_ok}")
    if len(apes):
        print("\n  empirical-vs-silicon  |emp-sil|/sil")
        print(f"    MAPE%  {apes.mean():7.2f}")
        print(f"    median {np.median(apes):7.2f}")
        print(f"    p95%   {np.percentile(apes, 95):7.2f}")
        print(f"    p99%   {np.percentile(apes, 99):7.2f}")
        print(f"    max%   {apes.max():7.2f}")


if __name__ == "__main__":
    sys.exit(main())
