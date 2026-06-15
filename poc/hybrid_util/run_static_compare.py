"""Compare SILICON vs EMPIRICAL run_static latency across (bs, seq, pastkv).

Reports MAPE / p95 / p99 / max of |empirical - silicon| / silicon, treating
SILICON as the reference. Establishes the model-level accuracy of empirical
mode. Run unchanged today (SOL/const) to get the baseline bar, then again
after the util-enhanced empirical lands.

Silicon may raise PerfDataNotAvailableError on incomplete grids (observation
1) -- those points are counted separately, not silently dropped.
"""

import sys

import numpy as np

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database

SYSTEM, BACKEND, VERSION = "h100_sxm", "trtllm", "1.3.0rc10"
MODEL = "meta-llama/Meta-Llama-3.1-70B"  # dense: GEMM + attention + elementwise + comm

BS = [1, 4, 16, 64, 256]
SEQ = [128, 512, 2048, 8192]  # new prompt tokens to prefill
PASTKV = [0, 1024, 4096]  # prefix / cached kv length; isl = seq + pastkv


def main():
    db = get_database(SYSTEM, BACKEND, VERSION)  # shared layer OFF for both modes
    model_config = config.ModelConfig(tp_size=8, gemm_quant_mode=common.GEMMQuantMode.fp8)
    model = get_model(MODEL, model_config, backend_name=BACKEND)
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
                        print(f"  FAIL bs={bs} seq={seq} pk={pk}: {type(e).__name__}: {str(e)[:300]}")
                    continue
                emp = latency(common.DatabaseMode.EMPIRICAL, rc)
                if sil and sil > 0 and emp and emp > 0:
                    apes.append(abs(emp - sil) / sil)
                    both_ok += 1

    apes = np.array(apes) * 100
    print(f"\n=== {MODEL} on {SYSTEM}/{BACKEND}/{VERSION}  (tp8, fp8)")
    print(f"grid: {len(BS)}x{len(SEQ)}x{len(PASTKV)} = {len(BS) * len(SEQ) * len(PASTKV)} points")
    print(f"silicon failures (obs 1): {sil_fail} {fail_types}   scored points: {both_ok}")
    if len(apes):
        print("\n  empirical-vs-silicon  |emp-sil|/sil")
        print(f"    MAPE%  {apes.mean():7.2f}")
        print(f"    median {np.median(apes):7.2f}")
        print(f"    p95%   {np.percentile(apes, 95):7.2f}")
        print(f"    p99%   {np.percentile(apes, 99):7.2f}")
        print(f"    max%   {apes.max():7.2f}")


if __name__ == "__main__":
    sys.exit(main())
