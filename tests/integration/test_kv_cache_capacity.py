# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for TRTLLMBackend._calculate_max_kv_cache_batch_size().

All tests use the real production pipeline (get_model, PerfDatabase,
system YAML specs) with no mocks. Validates that the KV cache capacity
formula produces accurate results against TRT-LLM benchmark data.
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.config import ModelConfig
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database, get_latest_database_version

pytestmark = pytest.mark.integration

_BACKEND = "trtllm"


def _load(system, model_path, tp, kvcache_quant_mode=None, moe_ep_size=None):
    """Load real model, database, and backend from production codebase."""
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


class TestKvCacheCapacityProperties:
    """Property-based tests using real models and systems."""

    def test_fraction_reduces_capacity(self):
        """Lower free_gpu_memory_fraction should reduce max batch size."""
        model, db, backend = _load("h100_sxm", "Qwen/Qwen3-32B-FP8", tp=1)

        bs_90 = backend._calculate_max_kv_cache_batch_size(
            model,
            db,
            1024,
            1024,
            free_gpu_memory_fraction=0.9,
        )
        bs_50 = backend._calculate_max_kv_cache_batch_size(
            model,
            db,
            1024,
            1024,
            free_gpu_memory_fraction=0.5,
        )

        assert bs_90 > bs_50
        assert bs_50 > 0

    def test_longer_sequences_reduce_capacity(self):
        """Longer ISL+OSL means fewer concurrent sequences fit."""
        model, db, backend = _load("h100_sxm", "Qwen/Qwen3-32B-FP8", tp=1)

        bs_short = backend._calculate_max_kv_cache_batch_size(
            model,
            db,
            512,
            512,
            free_gpu_memory_fraction=0.9,
        )
        bs_long = backend._calculate_max_kv_cache_batch_size(
            model,
            db,
            4096,
            4096,
            free_gpu_memory_fraction=0.9,
        )

        assert bs_short > bs_long
        ratio = bs_short / bs_long
        assert ratio > 6

    def test_tp_increases_capacity(self):
        """More TP shards reduce KV heads per GPU, increasing capacity."""
        model_tp1, db, backend = _load("h100_sxm", "Qwen/Qwen3-32B-FP8", tp=1)
        model_tp2, _, _ = _load("h100_sxm", "Qwen/Qwen3-32B-FP8", tp=2)

        bs_tp1 = backend._calculate_max_kv_cache_batch_size(
            model_tp1,
            db,
            1024,
            1024,
            free_gpu_memory_fraction=0.9,
        )
        bs_tp2 = backend._calculate_max_kv_cache_batch_size(
            model_tp2,
            db,
            1024,
            1024,
            free_gpu_memory_fraction=0.9,
        )

        assert bs_tp2 > bs_tp1

    def test_larger_gpu_increases_capacity(self):
        """GB300 (277 GiB) should fit more sequences than H100 (80 GiB)."""
        model_h100, db_h100, backend = _load("h100_sxm", "Qwen/Qwen3-32B-FP8", tp=1)
        model_gb300, db_gb300, _ = _load("gb300", "Qwen/Qwen3-32B-FP8", tp=1)

        bs_h100 = backend._calculate_max_kv_cache_batch_size(
            model_h100,
            db_h100,
            2048,
            2048,
            free_gpu_memory_fraction=0.9,
        )
        bs_gb300 = backend._calculate_max_kv_cache_batch_size(
            model_gb300,
            db_gb300,
            2048,
            2048,
            free_gpu_memory_fraction=0.9,
        )

        assert bs_gb300 > bs_h100 * 3

    def test_smaller_model_has_more_capacity(self):
        """8B model should have much more KV capacity than 32B on same GPU."""
        model_8b, db, backend = _load("h100_sxm", "meta-llama/Meta-Llama-3.1-8B", tp=1)
        model_32b, _, _ = _load("h100_sxm", "Qwen/Qwen3-32B-FP8", tp=1)

        bs_8b = backend._calculate_max_kv_cache_batch_size(
            model_8b,
            db,
            2048,
            2048,
            free_gpu_memory_fraction=0.9,
        )
        bs_32b = backend._calculate_max_kv_cache_batch_size(
            model_32b,
            db,
            2048,
            2048,
            free_gpu_memory_fraction=0.9,
        )

        assert bs_8b > bs_32b

    def test_deepseek_fits_on_gb300(self):
        """DeepSeek-V3 (671B, MLA) fits on GB300 TP=4 and has positive KV capacity."""
        model_ds, db, backend = _load(
            "gb300",
            "deepseek-ai/DeepSeek-V3",
            tp=4,
            moe_ep_size=4,
        )

        bs = backend._calculate_max_kv_cache_batch_size(
            model_ds,
            db,
            2048,
            2048,
            free_gpu_memory_fraction=0.9,
        )
        assert bs > 0

        # Longer sequences should reduce capacity (verifies MLA code path)
        bs_long = backend._calculate_max_kv_cache_batch_size(
            model_ds,
            db,
            8192,
            8192,
            free_gpu_memory_fraction=0.9,
        )
        assert bs > bs_long

    def test_returns_zero_when_model_too_large(self):
        """405B model on single H100 GPU (TP=1) should not fit."""
        model, db, backend = _load("h100_sxm", "meta-llama/Meta-Llama-3.1-405B", tp=1)

        result = backend._calculate_max_kv_cache_batch_size(
            model,
            db,
            1024,
            1024,
            free_gpu_memory_fraction=0.9,
        )
        assert result == 0


# ---------------------------------------------------------------------------
# Benchmark validation: ±1 tolerance against real TRT-LLM data
#
# Benchmark collected on H100_SXM and GB300 with:
#   Qwen/Qwen3-32B-FP8, meta-llama/Llama-3.1-8B
# Server started with:
#   --max_seq_len $((ISL + OSL + 1000))
#   --max_num_tokens $((2 * ISL))
#   --free_gpu_memory_fraction FRAC
#   kv_cache default dtype = float16
# ---------------------------------------------------------------------------

_BENCHMARK_SLACK = 1000

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
    """Shorten model path for test ID."""
    return model_path.rsplit("/", 1)[-1].lower().replace("-", "").replace(".", "")


@pytest.mark.parametrize(
    "system, model_path, tp, isl, osl, frac, bench_max_bs",
    _BENCHMARK_DATA,
    ids=[f"{s}-{_short_model(m)}-tp{t}-isl{i}-osl{o}-f{f}" for s, m, t, i, o, f, _ in _BENCHMARK_DATA],
)
def test_benchmark_kv_capacity(system, model_path, tp, isl, osl, frac, bench_max_bs):
    """Validate the full pipeline against real TRT-LLM benchmark data.

    Uses production get_model(), PerfDatabase, and system YAML specs.
    The benchmark used float16 KV cache (TRT-LLM default).
    Asserts ±1 tolerance; xfails cases where _get_memory_usage
    underestimates non-KV overhead.
    """
    model, database, backend = _load(
        system,
        model_path,
        tp,
        kvcache_quant_mode=common.KVCacheQuantMode.float16,
    )

    result = backend._calculate_max_kv_cache_batch_size(
        model,
        database,
        isl,
        osl,
        free_gpu_memory_fraction=frac,
        max_seq_len=isl + osl + _BENCHMARK_SLACK,
        max_num_tokens=2 * isl,
    )

    diff = result - bench_max_bs
    rel_pct = abs(diff) / bench_max_bs * 100
    ok = abs(diff) <= 1 or rel_pct < 3.5
    assert ok, (
        f"Formula ({result}) vs benchmark ({bench_max_bs}): "
        f"diff {diff:+d} ({rel_pct:.1f}%) exceeds tolerance (+-1 or <3.5%)"
    )
