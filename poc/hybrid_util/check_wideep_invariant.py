"""Per-op invariant check for WideEP ops whose run_static gate is blocked by
silicon data gaps (trtllm alltoall table absent; sglang deepep has no SOL).

At a *collected* shape, util-empirical must recover the silicon value
(SOL / (SOL/measured) = measured), unlike the fixed scale_factor. This proves
the WideEP wiring end-to-end without needing a full WideEP model run_static.
"""

import pandas as pd

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import get_database


def _check(label, db, qfn, old_scale):
    sil = qfn(common.DatabaseMode.SILICON)
    emp = qfn(common.DatabaseMode.EMPIRICAL)
    sol = qfn(common.DatabaseMode.SOL)
    print(f"{label}")
    print(f"  SILICON={sil:.5f}  EMPIRICAL={emp:.5f}  oldconst(SOL/{old_scale})={sol / old_scale:.5f}")
    print(f"  emp/sil={emp / sil:.4f} (≈1.0 = util recovered silicon)  oldconst/sil={(sol / old_scale) / sil:.4f}\n")


def main():
    # TrtLLMWideEPMoE compute (b200/trtllm) — wideep_moe table exists
    df = pd.read_parquet("src/aiconfigurator/systems/data/b200_sxm/trtllm/1.3.0rc10/wideep_moe_perf.parquet")
    r = df.iloc[len(df) // 2]
    db = get_database("b200_sxm", "trtllm", "1.3.0rc10")
    kw = dict(
        num_tokens=int(r["num_tokens"]),
        hidden_size=int(r["hidden_size"]),
        inter_size=int(r["inter_size"]),
        topk=int(r["topk"]),
        num_experts=int(r["num_experts"]),
        num_slots=int(r["num_slots"]),
        moe_tp_size=int(r["moe_tp_size"]),
        moe_ep_size=int(r["moe_ep_size"]),
        quant_mode=common.MoEQuantMode.nvfp4,
        workload_distribution=str(r["distribution"]),
    )

    def _moe(mode):
        db.set_default_database_mode(mode)
        return float(db.query_wideep_moe_compute(**kw))

    _check(f"TrtLLMWideEPMoE @ num_tokens={kw['num_tokens']}", db, _moe, 0.4)

    # WideEPGenerationMLA (h100/sglang) — sglang-only table
    df = pd.read_parquet("src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.10/wideep_generation_mla_perf.parquet")
    r = df.iloc[len(df) // 2]
    nh, b, s = int(r["num_heads"]), int(r["batch_size"]), int(r["isl"])
    tp = 128 // nh
    db2 = get_database("h100_sxm", "sglang", "0.5.10")

    def _mla(mode):
        db2.set_default_database_mode(mode)
        return float(
            db2.query_wideep_generation_mla(
                b, s, tp, common.KVCacheQuantMode.fp8, common.FMHAQuantMode.bfloat16, "flashinfer"
            )
        )

    _check(f"WideEPGenerationMLA @ num_heads={nh}(tp={tp}), b={b}, s={s}", db2, _mla, 0.7)

    # DeepSeekV4MHCModule (b200/sglang) — mhc_module table exists (DSV4 attention
    # modules have no collected data anywhere yet, so only MHC is checkable).
    df = pd.read_parquet("src/aiconfigurator/systems/data/b200_sxm/sglang/0.5.10/mhc_module_perf.parquet")
    g = df[df.op_name == "pre"]
    nt = int(sorted(g.num_tokens.unique())[len(g.num_tokens.unique()) // 2])
    hc, h = int(g.hc_mult.iloc[0]), int(g.hidden_size.iloc[0])
    db3 = get_database("b200_sxm", "sglang", "0.5.10")

    def _mhc(mode):
        db3.set_default_database_mode(mode)
        return float(db3.query_mhc_module(nt, h, hc, 1, "pre", common.GEMMQuantMode.bfloat16))

    _check(f"DeepSeekV4MHCModule(pre) @ num_tokens={nt}, hc={hc}, hidden={h}", db3, _mhc, 0.55)


if __name__ == "__main__":
    main()
