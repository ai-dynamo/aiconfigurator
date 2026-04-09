# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for KV cache oversubscription detection.

The KV capacity check is embedded in _get_memory_usage() via the max_seq_len
parameter.  When max_seq_len is provided:
  kvcache = batch_size * max_seq_len * kvcache_per_token
A batch_size exceeds KV capacity when memory["kvcache"] > available_kv_gib.

Regular OOM (model doesn't fit at all) is a separate condition: the total
memory check in run_agg catches it; when available_kv_gib <= 0 the model
doesn't fit and any batch_size is KV OOM.

All benchmark data was collected with --max_num_tokens $((2 * ISL)).  This
value determines the activation memory TRT-LLM pre-allocates at engine build
time (see BuildConfig.max_num_tokens, default 8192).  The activation memory
in turn determines how much GPU memory remains for KV cache.  Tests pass
max_num_tokens=2*isl explicitly to match the benchmark configuration.

All tests use the real production pipeline with no mocks.
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.backends.trtllm_backend import (
    KV_CACHE_MEMORY_RESERVED_FRACTION,
    KV_CACHE_MEMORY_TOLERANCE,
)
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database, get_latest_database_version

pytestmark = pytest.mark.integration

_BACKEND = "trtllm"
_BENCHMARK_SLACK = 1000  # TRT-LLM benchmark used --max_seq_len isl+osl+1000


def _load(system, model_path, tp, kvcache_quant_mode=None, moe_ep_size=None):
    version = get_latest_database_version(system=system, backend=_BACKEND)
    assert version is not None, f"No database for {system}/{_BACKEND}"
    database = get_database(system, _BACKEND, version)
    assert database is not None

    moe_tp = tp if moe_ep_size is None else (tp // moe_ep_size if tp >= moe_ep_size else 1)
    moe_ep = moe_ep_size or 1
    model_config = ModelConfig(
        tp_size=tp,
        pp_size=1,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
    )
    if kvcache_quant_mode is not None:
        model_config.kvcache_quant_mode = kvcache_quant_mode
    model = get_model(model_path, model_config, _BACKEND)
    backend = get_backend(_BACKEND)
    return model, database, backend


def _is_kv_oom(
    backend,
    model,
    database,
    batch_size,
    isl,
    osl,
    free_gpu_memory_fraction,
    *,
    max_seq_len,
    max_num_tokens,
    kv_cache_capacity_tolerance=KV_CACHE_MEMORY_TOLERANCE,
):
    """KV OOM check using InferenceSummary with a single unified memory dict."""
    memory = backend._get_memory_usage(
        model,
        database,
        batch_size,
        1,
        isl,
        osl,
        num_tokens=max_num_tokens,
        max_seq_len=max_seq_len,
    )
    summary = InferenceSummary(RuntimeConfig(isl=isl, osl=osl))
    summary.set_memory_and_check_oom(
        memory,
        database.system_spec["gpu"]["mem_capacity"],
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        kv_cache_reserved_fraction=KV_CACHE_MEMORY_RESERVED_FRACTION,
        kv_cache_tolerance=kv_cache_capacity_tolerance,
    )
    return summary.check_oom() or summary.check_kv_cache_oom()


class TestKvCacheOomProperties:
    """Property-based sanity tests using real models and systems.

    Batch sizes are chosen relative to known benchmark thresholds so that
    each assertion is expected to be stable across formula changes within
    the validated ±2% accuracy range.
    """

    def test_fraction_reduces_capacity(self):
        """Lower free_gpu_memory_fraction reduces max KV batch size.

        h100_sxm / Qwen3-32B-FP8 / tp=1 / isl=osl=2048:
          threshold at f=0.9 ≈ 78, at f=0.5 ≈ 43.
          bs=60: fits at f=0.9, OOM at f=0.5.
        """
        isl, osl = 2048, 2048
        model, db, backend = _load("h100_sxm", "Qwen/Qwen3-32B-FP8", tp=1)
        assert not _is_kv_oom(
            backend, model, db, 60, isl, osl, 0.9, max_seq_len=isl + osl + _BENCHMARK_SLACK, max_num_tokens=2 * isl
        )
        assert _is_kv_oom(
            backend, model, db, 60, isl, osl, 0.5, max_seq_len=isl + osl + _BENCHMARK_SLACK, max_num_tokens=2 * isl
        )

    def test_longer_sequences_reduce_capacity(self):
        """Longer ISL+OSL means fewer concurrent sequences fit.

        h100_sxm / Qwen3-32B-FP8 / tp=1 / f=0.9:
          threshold at isl=osl=512 ≈ 319, at isl=osl=4096 ≈ 38.
          bs=100: fits for short seqs, OOM for long seqs.
        """
        model, db, backend = _load("h100_sxm", "Qwen/Qwen3-32B-FP8", tp=1)
        assert not _is_kv_oom(
            backend, model, db, 100, 512, 512, 0.9, max_seq_len=512 + 512 + _BENCHMARK_SLACK, max_num_tokens=2 * 512
        )
        assert _is_kv_oom(
            backend,
            model,
            db,
            100,
            4096,
            4096,
            0.9,
            max_seq_len=4096 + 4096 + _BENCHMARK_SLACK,
            max_num_tokens=2 * 4096,
        )

    def test_tp_increases_capacity(self):
        """More TP shards reduce KV heads per GPU, increasing KV capacity.

        h100_sxm / Qwen3-32B-FP8 / isl=osl=2048 / f=0.9:
          tp=1 threshold ≈ 78, tp=2 threshold ≈ 215.
          bs=120: OOM at tp=1, fits at tp=2.
        """
        isl, osl = 2048, 2048
        model_tp1, db, backend = _load("h100_sxm", "Qwen/Qwen3-32B-FP8", tp=1)
        model_tp2, _, _ = _load("h100_sxm", "Qwen/Qwen3-32B-FP8", tp=2)
        assert _is_kv_oom(
            backend, model_tp1, db, 120, isl, osl, 0.9, max_seq_len=isl + osl + _BENCHMARK_SLACK, max_num_tokens=2 * isl
        )
        assert not _is_kv_oom(
            backend, model_tp2, db, 120, isl, osl, 0.9, max_seq_len=isl + osl + _BENCHMARK_SLACK, max_num_tokens=2 * isl
        )

    def test_larger_gpu_increases_capacity(self):
        """GB300 (277 GiB) fits more sequences than H100 (80 GiB).

        Qwen3-32B-FP8 / tp=1 / isl=osl=2048 / f=0.9:
          h100 threshold ≈ 78, gb300 threshold ≈ 434.
          bs=200: OOM on h100, fits on gb300.
        """
        isl, osl = 2048, 2048
        model_h100, db_h100, backend = _load("h100_sxm", "Qwen/Qwen3-32B-FP8", tp=1)
        model_gb300, db_gb300, _ = _load("gb300", "Qwen/Qwen3-32B-FP8", tp=1)
        assert _is_kv_oom(
            backend,
            model_h100,
            db_h100,
            200,
            isl,
            osl,
            0.9,
            max_seq_len=isl + osl + _BENCHMARK_SLACK,
            max_num_tokens=2 * isl,
        )
        assert not _is_kv_oom(
            backend,
            model_gb300,
            db_gb300,
            200,
            isl,
            osl,
            0.9,
            max_seq_len=isl + osl + _BENCHMARK_SLACK,
            max_num_tokens=2 * isl,
        )

    def test_smaller_model_has_more_kv_capacity(self):
        """8B model has much more KV capacity than 32B on the same GPU.

        Both models use float16 KV cache to match benchmark conditions and ensure
        a fair comparison (default KV quant mode differs between model families).

        h100_sxm / tp=1 / isl=osl=2048 / f=0.9 / float16 KV:
          32B threshold ≈ 30, 8B threshold ≈ 88.
          bs=60: OOM for 32B, fits for 8B.
        """
        isl, osl = 2048, 2048
        model_8b, db, backend = _load(
            "h100_sxm", "meta-llama/Meta-Llama-3.1-8B", tp=1, kvcache_quant_mode=common.KVCacheQuantMode.float16
        )
        model_32b, _, _ = _load(
            "h100_sxm", "Qwen/Qwen3-32B-FP8", tp=1, kvcache_quant_mode=common.KVCacheQuantMode.float16
        )
        assert _is_kv_oom(
            backend, model_32b, db, 60, isl, osl, 0.9, max_seq_len=isl + osl + _BENCHMARK_SLACK, max_num_tokens=2 * isl
        )
        assert not _is_kv_oom(
            backend, model_8b, db, 60, isl, osl, 0.9, max_seq_len=isl + osl + _BENCHMARK_SLACK, max_num_tokens=2 * isl
        )

    def test_deepseek_fits_on_gb300(self):
        """DeepSeek-V3 (MLA) fits on GB300 TP=4: bs=1 is not KV OOM, bs=10000 is."""
        model_ds, db, backend = _load("gb300", "deepseek-ai/DeepSeek-V3", tp=4, moe_ep_size=4)
        assert not _is_kv_oom(
            backend,
            model_ds,
            db,
            1,
            2048,
            2048,
            0.9,
            max_seq_len=2048 + 2048 + _BENCHMARK_SLACK,
            max_num_tokens=2 * 2048,
        )
        assert not _is_kv_oom(
            backend,
            model_ds,
            db,
            1,
            8192,
            8192,
            0.9,
            max_seq_len=8192 + 8192 + _BENCHMARK_SLACK,
            max_num_tokens=2 * 8192,
        )
        assert _is_kv_oom(
            backend,
            model_ds,
            db,
            10000,
            2048,
            2048,
            0.9,
            max_seq_len=2048 + 2048 + _BENCHMARK_SLACK,
            max_num_tokens=2 * 2048,
        )

    def test_model_too_large_is_regular_oom(self):
        """405B on single H100 (TP=1): model doesn't fit → KV OOM returns True for bs=1."""
        model, db, backend = _load("h100_sxm", "meta-llama/Meta-Llama-3.1-405B", tp=1)
        assert _is_kv_oom(
            backend, model, db, 1, 1024, 1024, 0.9, max_seq_len=1024 + 1024 + _BENCHMARK_SLACK, max_num_tokens=2 * 1024
        )


# ---------------------------------------------------------------------------
# Benchmark-anchored KV OOM boundary tests
#
# bench_max_bs: real TRT-LLM max concurrent sequences measured with:
#   --max_seq_len $((ISL + OSL + 1000))
#   --max_num_tokens $((2 * ISL))
#   --free_gpu_memory_fraction FRAC
#   KV cache dtype = float16 (TRT-LLM default)
#
# We call _is_kv_oom with the same parameters and assert:
#   - batch_size = bench_max_bs - tolerance → NOT KV OOM
#   - batch_size = bench_max_bs + tolerance + 1 → IS KV OOM
# tolerance = max(1, int(bench_max_bs * KV_CACHE_MEMORY_TOLERANCE))
# ---------------------------------------------------------------------------

# fmt: off
# (system, model, tp, isl, osl, frac, bench_max_bs)
_BENCHMARK_DATA = [
    # ===== Qwen/Qwen3-32B-FP8 =====
    # --- H100_SXM, fraction=0.8 ---
    ("h100_sxm", "Qwen/Qwen3-32B-FP8", 1, 2048, 2048, 0.8,  27),
    ("h100_sxm", "Qwen/Qwen3-32B-FP8", 2, 2048, 2048, 0.8,  75),
    ("h100_sxm", "Qwen/Qwen3-32B-FP8", 4, 2048, 2048, 0.8, 169),
    ("h100_sxm", "Qwen/Qwen3-32B-FP8", 1, 2048, 4096, 0.8,  19),
    ("h100_sxm", "Qwen/Qwen3-32B-FP8", 2, 2048, 4096, 0.8,  53),
    ("h100_sxm", "Qwen/Qwen3-32B-FP8", 4, 2048, 4096, 0.8, 121),
    ("h100_sxm", "Qwen/Qwen3-32B-FP8", 1, 4096, 2048, 0.8,  19),
    ("h100_sxm", "Qwen/Qwen3-32B-FP8", 2, 4096, 2048, 0.8,  53),
    ("h100_sxm", "Qwen/Qwen3-32B-FP8", 4, 4096, 2048, 0.8, 120),
    ("h100_sxm", "Qwen/Qwen3-32B-FP8", 1, 4096, 4096, 0.8,  14),
    ("h100_sxm", "Qwen/Qwen3-32B-FP8", 2, 4096, 4096, 0.8,  41),
    ("h100_sxm", "Qwen/Qwen3-32B-FP8", 4, 4096, 4096, 0.8,  93),
    # --- GB300, Qwen3-32B-FP8, fraction=0.8 ---
    ("gb300", "Qwen/Qwen3-32B-FP8", 1, 2048, 2048, 0.8, 153),
    ("gb300", "Qwen/Qwen3-32B-FP8", 1, 2048, 4096, 0.8, 109),
    ("gb300", "Qwen/Qwen3-32B-FP8", 1, 4096, 2048, 0.8, 108),
    ("gb300", "Qwen/Qwen3-32B-FP8", 1, 4096, 4096, 0.8,  84),
    ("gb300", "Qwen/Qwen3-32B-FP8", 2, 2048, 2048, 0.8, 326),
    ("gb300", "Qwen/Qwen3-32B-FP8", 2, 2048, 4096, 0.8, 233),
    ("gb300", "Qwen/Qwen3-32B-FP8", 2, 4096, 2048, 0.8, 232),
    ("gb300", "Qwen/Qwen3-32B-FP8", 2, 4096, 4096, 0.8, 181),
    ("gb300", "Qwen/Qwen3-32B-FP8", 4, 2048, 2048, 0.8, 673),
    ("gb300", "Qwen/Qwen3-32B-FP8", 4, 2048, 4096, 0.8, 480),
    ("gb300", "Qwen/Qwen3-32B-FP8", 4, 4096, 2048, 0.8, 479),
    ("gb300", "Qwen/Qwen3-32B-FP8", 4, 4096, 4096, 0.8, 373),
    # --- GB300, Qwen3-32B-FP8, fraction=0.9 ---
    ("gb300", "Qwen/Qwen3-32B-FP8", 1, 2048, 2048, 0.9, 172),
    ("gb300", "Qwen/Qwen3-32B-FP8", 1, 2048, 4096, 0.9, 123),
    ("gb300", "Qwen/Qwen3-32B-FP8", 1, 4096, 2048, 0.9, 122),
    ("gb300", "Qwen/Qwen3-32B-FP8", 1, 4096, 4096, 0.9,  95),
    ("gb300", "Qwen/Qwen3-32B-FP8", 2, 2048, 2048, 0.9, 367),
    ("gb300", "Qwen/Qwen3-32B-FP8", 2, 2048, 4096, 0.9, 262),
    ("gb300", "Qwen/Qwen3-32B-FP8", 2, 4096, 2048, 0.9, 261),
    ("gb300", "Qwen/Qwen3-32B-FP8", 2, 4096, 4096, 0.9, 203),
    ("gb300", "Qwen/Qwen3-32B-FP8", 4, 2048, 2048, 0.9, 757),
    ("gb300", "Qwen/Qwen3-32B-FP8", 4, 2048, 4096, 0.9, 541),
    ("gb300", "Qwen/Qwen3-32B-FP8", 4, 4096, 2048, 0.9, 539),
    ("gb300", "Qwen/Qwen3-32B-FP8", 4, 4096, 4096, 0.9, 419),
    # ===== meta-llama/Llama-3.1-8B =====
    # --- H100_SXM, fraction=0.8 ---
    ("h100_sxm", "meta-llama/Meta-Llama-3.1-8B", 1, 2048, 2048, 0.8,  78),
    ("h100_sxm", "meta-llama/Meta-Llama-3.1-8B", 2, 2048, 2048, 0.8, 173),
    ("h100_sxm", "meta-llama/Meta-Llama-3.1-8B", 4, 2048, 2048, 0.8, 363),
    ("h100_sxm", "meta-llama/Meta-Llama-3.1-8B", 1, 2048, 4096, 0.8,  56),
    ("h100_sxm", "meta-llama/Meta-Llama-3.1-8B", 2, 2048, 4096, 0.8, 124),
    ("h100_sxm", "meta-llama/Meta-Llama-3.1-8B", 4, 2048, 4096, 0.8, 259),
    ("h100_sxm", "meta-llama/Meta-Llama-3.1-8B", 1, 4096, 2048, 0.8,  55),
    ("h100_sxm", "meta-llama/Meta-Llama-3.1-8B", 2, 4096, 2048, 0.8, 123),
    ("h100_sxm", "meta-llama/Meta-Llama-3.1-8B", 4, 4096, 2048, 0.8, 258),
    ("h100_sxm", "meta-llama/Meta-Llama-3.1-8B", 1, 4096, 4096, 0.8,  43),
    ("h100_sxm", "meta-llama/Meta-Llama-3.1-8B", 2, 4096, 4096, 0.8,  96),
    ("h100_sxm", "meta-llama/Meta-Llama-3.1-8B", 4, 4096, 4096, 0.8, 201),
    # --- GB300, Llama-3.1-8B, fraction=0.8 ---
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 1, 2048, 2048, 0.8, 330),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 1, 2048, 4096, 0.8, 236),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 1, 4096, 2048, 0.8, 235),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 1, 4096, 4096, 0.8, 183),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 2, 2048, 2048, 0.8, 677),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 2, 2048, 4096, 0.8, 484),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 2, 4096, 2048, 0.8, 483),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 2, 4096, 4096, 0.8, 376),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 4, 2048, 2048, 0.8, 1371),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 4, 2048, 4096, 0.8, 979),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 4, 4096, 2048, 0.8, 978),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 4, 4096, 4096, 0.8, 761),
    # --- GB300, Llama-3.1-8B, fraction=0.9 ---
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 1, 2048, 2048, 0.9, 372),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 1, 2048, 4096, 0.9, 265),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 1, 4096, 2048, 0.9, 265),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 1, 4096, 4096, 0.9, 206),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 2, 2048, 2048, 0.9, 762),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 2, 2048, 4096, 0.9, 544),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 2, 4096, 2048, 0.9, 544),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 2, 4096, 4096, 0.9, 423),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 4, 2048, 2048, 0.9, 1542),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 4, 2048, 4096, 0.9, 1101),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 4, 4096, 2048, 0.9, 1100),
    ("gb300", "meta-llama/Meta-Llama-3.1-8B", 4, 4096, 4096, 0.9, 856),
]
# fmt: on


def _short_model(model_path):
    return model_path.rsplit("/", 1)[-1].lower().replace("-", "").replace(".", "")


@pytest.mark.parametrize(
    "system, model_path, tp, isl, osl, frac, bench_max_bs",
    _BENCHMARK_DATA,
    ids=[f"{s}-{_short_model(m)}-tp{t}-isl{i}-osl{o}-f{f}" for s, m, t, i, o, f, _ in _BENCHMARK_DATA],
)
def test_kv_oom_boundary(system, model_path, tp, isl, osl, frac, bench_max_bs):
    """Validate KV OOM boundary against real TRT-LLM benchmark data.

    Calls _is_kv_oom with the same parameters used during benchmarking
    (max_seq_len=isl+osl+1000, max_num_tokens=2*isl) and asserts:
      - batch_size = bench_max_bs - tol → NOT KV OOM
      - batch_size = bench_max_bs + tol + 1 → IS KV OOM
    where tol = max(1, int(bench_max_bs * KV_CACHE_MEMORY_TOLERANCE))

    Uses float16 KV cache to match benchmark conditions.
    """
    model, database, backend = _load(
        system,
        model_path,
        tp,
        kvcache_quant_mode=common.KVCacheQuantMode.float16,
    )

    tol = max(1, int(bench_max_bs * KV_CACHE_MEMORY_TOLERANCE))
    bs_below = bench_max_bs - tol
    bs_above = bench_max_bs + tol + 1

    assert not _is_kv_oom(
        backend,
        model,
        database,
        bs_below,
        isl,
        osl,
        frac,
        max_seq_len=isl + osl + _BENCHMARK_SLACK,
        max_num_tokens=2 * isl,
        kv_cache_capacity_tolerance=0,
    ), f"bs={bs_below} (bench={bench_max_bs}, tol={tol}) should NOT be KV OOM"

    assert _is_kv_oom(
        backend,
        model,
        database,
        bs_above,
        isl,
        osl,
        frac,
        max_seq_len=isl + osl + _BENCHMARK_SLACK,
        max_num_tokens=2 * isl,
        kv_cache_capacity_tolerance=0,
    ), f"bs={bs_above} (bench={bench_max_bs}, tol={tol}) should be KV OOM"
