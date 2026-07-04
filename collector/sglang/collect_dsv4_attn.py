# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4-Flash module-level attention collector for SGLang.

ONE file containing both:

  1. The bench engine — builds an sglang ``ModelRunner`` for a single
     attn_kind (CSA / HCA) layer and times CUDA-Graph replay of
     ``layer.self_attn(...)`` (Q/KV proj + norm/rope + cache store +
     compressor + C4 indexer/topk for CSA + final FlashMLA).
  2. The registry-facing entrypoints — ``run_dsv4_attn_worker``
     (per-(kind, tp, gemm, bs) test case) which spawns a subprocess that
     internally sweeps every valid sl for that bs.

Test cases (sweep grids + ``get_*_test_cases`` functions) live in
``dsv4_test_cases`` and are re-exported below for registry use.

Manual CLI use::

    python collect_dsv4_attn.py --mode generation --attn-kind csa
    python collect_dsv4_attn.py --mode context --attn-kind hca \
        --batch-sizes 1,4 --seq-lens 128,1024
"""

# Requires stock SGLang 0.5.14 with its matching ``sgl-kernel`` package.
from __future__ import annotations

__compat__ = "sglang==0.5.14"

import argparse
import contextlib
import copy
import gc
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Iterable
from fractions import Fraction
from importlib.metadata import version as get_version

import torch

# DSV4 local forks default to replacing small patched configs with packaged
# config_backup_small.json.  Suppress so collector's per-kind 1-layer config
# isn't overwritten.
os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")
# Hard-disable DeepGEMM bulk pre-compile.  Each test case touches only a
# few shapes which the bench's own warmup JIT-compiles on first use.
os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
try:
    from helper import _resolve_local_model_path, benchmark_with_power, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import _resolve_local_model_path, benchmark_with_power, log_perf

try:
    from collector.sglang.runtime_limits import (
        alloc_prefix_indices as _alloc_prefix_indices,
    )
    from collector.sglang.runtime_limits import (
        kv_pool_capacity_tokens as _kv_pool_capacity_tokens,
    )
    from collector.sglang.runtime_limits import (
        kv_pool_page_size as _kv_pool_page_size,
    )
    from collector.sglang.runtime_limits import (
        required_kv_alloc_tokens,
        required_kv_tokens,
        required_prefill_extend_tokens,
        required_swa_kv_alloc_tokens,
    )
    from collector.sglang.runtime_limits import (
        runtime_chunk_size as _runtime_chunk_size,
    )
    from collector.sglang.runtime_limits import (
        swa_kv_pool_capacity_tokens as _swa_kv_pool_capacity_tokens,
    )
    from collector.sglang.runtime_limits import (
        temporarily_chunked_alloc_extend as _temporarily_chunked_alloc_extend,
    )
except ModuleNotFoundError:
    from runtime_limits import (
        alloc_prefix_indices as _alloc_prefix_indices,
    )
    from runtime_limits import (
        kv_pool_capacity_tokens as _kv_pool_capacity_tokens,
    )
    from runtime_limits import (
        kv_pool_page_size as _kv_pool_page_size,
    )
    from runtime_limits import (
        required_kv_alloc_tokens,
        required_kv_tokens,
        required_prefill_extend_tokens,
        required_swa_kv_alloc_tokens,
    )
    from runtime_limits import (
        runtime_chunk_size as _runtime_chunk_size,
    )
    from runtime_limits import (
        swa_kv_pool_capacity_tokens as _swa_kv_pool_capacity_tokens,
    )
    from runtime_limits import (
        temporarily_chunked_alloc_extend as _temporarily_chunked_alloc_extend,
    )


# Re-export test case generators from the centralized case generator module so
# collect.py's registry (``module="collector.sglang.collect_dsv4_attn"``) can
# resolve them via getattr.
try:
    from collector.case_generator import (
        _DSV4_MODULE_BATCH_SIZES as _BATCH_SIZES,
    )
    from collector.case_generator import (
        _DSV4_MODULE_PAST_KV_LIST as _PREFIX_LENGTHS,
    )
    from collector.case_generator import (
        _DSV4_MODULE_SEQ_LENGTHS as _SEQ_LENGTHS,
    )
    from collector.case_generator import (
        _DSV4_MODULE_TP_SIZES as _TP_SIZES,
    )
    from collector.case_generator import (
        DSV4_ATTN_KINDS as ATTN_KINDS,
    )
    from collector.case_generator import (
        _dsv4_module_filter_pairs as _filter_pairs,
    )
    from collector.case_generator import (
        _dsv4_module_is_valid_shape as _is_valid_shape,
    )
    from collector.case_generator import get_dsv4_csa_context_test_cases as _get_dsv4_csa_context_test_cases_impl
    from collector.case_generator import get_dsv4_csa_generation_test_cases as _get_dsv4_csa_generation_test_cases_impl
    from collector.case_generator import get_dsv4_hca_context_test_cases as _get_dsv4_hca_context_test_cases_impl
    from collector.case_generator import get_dsv4_hca_generation_test_cases as _get_dsv4_hca_generation_test_cases_impl
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import (
        _DSV4_MODULE_BATCH_SIZES as _BATCH_SIZES,
    )
    from case_generator import (
        _DSV4_MODULE_PAST_KV_LIST as _PREFIX_LENGTHS,
    )
    from case_generator import (
        _DSV4_MODULE_SEQ_LENGTHS as _SEQ_LENGTHS,
    )
    from case_generator import (
        _DSV4_MODULE_TP_SIZES as _TP_SIZES,
    )
    from case_generator import (
        DSV4_ATTN_KINDS as ATTN_KINDS,
    )
    from case_generator import (
        _dsv4_module_filter_pairs as _filter_pairs,
    )
    from case_generator import (
        _dsv4_module_is_valid_shape as _is_valid_shape,
    )
    from case_generator import get_dsv4_csa_context_test_cases as _get_dsv4_csa_context_test_cases_impl
    from case_generator import get_dsv4_csa_generation_test_cases as _get_dsv4_csa_generation_test_cases_impl
    from case_generator import get_dsv4_hca_context_test_cases as _get_dsv4_hca_context_test_cases_impl
    from case_generator import get_dsv4_hca_generation_test_cases as _get_dsv4_hca_generation_test_cases_impl


def _expand_grid():
    """Return ``(batch_sizes, seq_lens)`` for the module-level sweep."""
    return list(_BATCH_SIZES), list(_SEQ_LENGTHS)


def _dsv4_max_position_embeddings(model_path: str) -> int:
    """Read the context boundary from the selected model config."""
    try:
        from collector.sglang.deepseekv4_sparse_modules import _dsv4_model_config
    except ModuleNotFoundError:
        from deepseekv4_sparse_modules import _dsv4_model_config

    value = _dsv4_model_config(model_path).get("max_position_embeddings")
    if value in (None, "") or int(value) <= 0:
        raise ValueError(f"invalid max_position_embeddings={value!r} for {model_path!r}")
    return int(value)


def get_dsv4_csa_context_test_cases():
    return _get_dsv4_csa_context_test_cases_impl()


def get_dsv4_csa_generation_test_cases():
    return _get_dsv4_csa_generation_test_cases_impl()


def get_dsv4_hca_context_test_cases():
    return _get_dsv4_hca_context_test_cases_impl()


def get_dsv4_hca_generation_test_cases():
    return _get_dsv4_hca_generation_test_cases_impl()


get_dsv4_flash_csa_context_test_cases = get_dsv4_csa_context_test_cases
get_dsv4_flash_csa_generation_test_cases = get_dsv4_csa_generation_test_cases
get_dsv4_flash_hca_context_test_cases = get_dsv4_hca_context_test_cases
get_dsv4_flash_hca_generation_test_cases = get_dsv4_hca_generation_test_cases


__all__ = [
    "ATTN_KINDS",
    "_BATCH_SIZES",
    "_SEQ_LENGTHS",
    "_TP_SIZES",
    "_filter_pairs",
    "get_dsv4_csa_context_test_cases",
    "get_dsv4_csa_generation_test_cases",
    "get_dsv4_flash_csa_context_test_cases",
    "get_dsv4_flash_csa_generation_test_cases",
    "get_dsv4_flash_hca_context_test_cases",
    "get_dsv4_flash_hca_generation_test_cases",
    "get_dsv4_hca_context_test_cases",
    "get_dsv4_hca_generation_test_cases",
    "run_dsv4_attn_worker",
]


NATIVE_HEADS = 64

_DSV4_CUDA_PAGE_SIZE = 256
_DSV4_SWA_WINDOW_SIZE = 128
_DSV4_SWA_TO_FULL_SCALE = 10

ATTN_KIND_TO_COMPRESS_RATIO = {
    "csa": 4,
    "hca": 128,
}


CLI_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


_PORTS_PER_GPU = 1000
_DSV4_PORT_RETRIES = 5


def _canonical_dsv4_model_id(model_path: str) -> str:
    normalized = str(model_path).rstrip("/")
    basename = normalized.rsplit("/", 1)[-1]
    if basename == "DeepSeek-V4-Pro":
        return "deepseek-ai/DeepSeek-V4-Pro"
    if basename == "DeepSeek-V4-Flash":
        return "deepseek-ai/DeepSeek-V4-Flash"
    return str(model_path)


def _port_is_available(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # TCPStore may listen on any local interface; check the same address
        # family instead of only 127.0.0.1.
        s.bind(("0.0.0.0", port))
        s.listen(1)
        return True
    except OSError:
        return False
    finally:
        s.close()


def _nccl_port_for_attempt(gpu_id: int, attempt: int) -> int:
    """Use a deterministic, GPU-scoped TCPStore port."""
    return 40000 + gpu_id * _PORTS_PER_GPU + attempt


def _pick_free_port(gpu_id: int) -> int:
    """Return a free TCP port from a ``gpu_id``-scoped 1000-port range.

    Used as ``nccl_port`` for the per-subprocess torch.distributed
    rendezvous.  Up to 8 collector workers run in parallel, each pinned
    to one GPU.  Partitioning the port space by ``gpu_id`` makes
    cross-worker collision impossible: worker N's candidate set is
    [40000 + N*1000, 40000 + N*1000 + 999].  Kept as a fallback for
    direct/manual use; normal collect.py entrypoints pass
    ``AIC_DSV4_NCCL_PORT`` explicitly.
    """
    base = _nccl_port_for_attempt(gpu_id, 0)
    for offset in range(_PORTS_PER_GPU):
        port = _nccl_port_for_attempt(gpu_id, offset)
        if _port_is_available(port):
            return port
    raise RuntimeError(f"no free port in [{base}, {base + 999}] for gpu_id={gpu_id}")


def _kv_dtype_db_to_sglang(kv_dtype_db: str) -> str:
    """Map perf-database kv dtype string to SGLang's ServerArgs value."""
    return {"bfloat16": "bfloat16", "fp8": "fp8_e4m3"}[kv_dtype_db]


# ═══════════════════════════════════════════════════════════════════════
# Bench engine — model load, forward batch, CUDA-graph timing, perf log
# ═══════════════════════════════════════════════════════════════════════


def _resolve_perf_path(output_path: str | None, default_name: str) -> str:
    if not output_path:
        return default_name
    if output_path.endswith(".txt"):
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, default_name)


def _copy_non_weight_files(src_dir: str, dst_dir: str) -> None:
    """Mirror model assets into the patched-config temp dir.

    - Non-weight files (tokenizer, generation_config, etc.) are copied.
    - Weight files (``.safetensors`` etc.) are *symlinked* so that
      ``load_format=auto`` can read real weights from the original model dir
      while the temp dir's patched ``config.json`` controls the architecture.
      This is required to reproduce production score distributions in the
      indexer's ``topk_512_transform`` (dummy weights produce uniformly
      random logits which take a different radix path and clock in higher
      than the structured logits a trained checkpoint produces).
    - ``config.json`` is intentionally skipped here; the caller writes the
      patched config in its place.
    """
    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        if not os.path.isfile(src_path):
            continue
        if fname == "config.json":
            continue
        dst_path = os.path.join(dst_dir, fname)
        if os.path.exists(dst_path) or os.path.islink(dst_path):
            continue
        if fname.endswith(_WEIGHT_SUFFIXES) or fname.endswith(".safetensors.index.json"):
            os.symlink(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def _download_non_weight_model_files(model_id: str) -> tuple[str, dict]:
    from huggingface_hub import hf_hub_download, list_repo_files

    try:
        files = list_repo_files(model_id)
    except Exception:
        files = ["config.json"]

    config_file = None
    for fname in files:
        if fname.endswith(_WEIGHT_SUFFIXES):
            continue
        try:
            path = hf_hub_download(model_id, fname)
            if fname == "config.json":
                config_file = path
        except Exception:
            continue

    if config_file is None:
        config_file = hf_hub_download(model_id, "config.json")

    with open(config_file) as f:
        config = json.load(f)
    return os.path.dirname(config_file), config


def _resolve_model_path(
    model_path: str,
    *,
    attn_kind: str,
    num_layers: int,
    shrink_unused_moe: bool,
    disable_weight_quant: bool,
    strip_auto_map: bool = True,
    gemm_type: str = "bfloat16",
    load_format: str = "dummy",
) -> str:
    """Create a local config dir patched for a single DSV4 attention kind.

    ``gemm_type`` controls which GEMM path the projection layers take:
        - ``"bfloat16"`` (default): drops fp8 ``quantization_config`` so weights
          load as bf16 and projections dispatch to cuBLASLt nvjet kernels.  This
          matches the historical collector behavior and is fast/light to load.
        - ``"fp8_block"``: keeps the upstream V4-Flash fp8 block-quantized
          ``quantization_config``.  Combined with ServerArgs ``quantization="fp8"``
          this routes projection GEMMs through DeepGEMM's
          ``sm90_fp8_gemm_1d2d_impl`` kernel — the same path the production
          server uses, so kernel-by-kernel the latency lines up with a real run.

    TP simulation is NOT done at this layer (do not patch num_attention_heads).
    Use ``_tp_load_model_patch`` instead, which sets ``_TP.world_size`` and
    ``_ATTN_TP_SIZE`` to N at model construction.  That keeps FMLA's required
    h_q=64 (Q is zero-padded with only the rank's tp_slice filled) while
    projection GEMMs (wq_b, wo_a, wo_b, ColumnParallel/RowParallel) allocate
    1/N shards.  Patching ``num_attention_heads`` directly would bypass the
    zero-pad path and trip FlashMLA's "Unsupported h_q: 8" template guard.
    """

    if os.path.isdir(model_path):
        src_dir = model_path
        with open(os.path.join(src_dir, "config.json")) as f:
            config = json.load(f)
    elif load_format == "dummy":
        src_dir = _resolve_local_model_path(model_path)
        with open(os.path.join(src_dir, "config.json")) as f:
            config = json.load(f)
    else:
        src_dir, config = _download_non_weight_model_files(model_path)

    config = copy.deepcopy(config)
    if strip_auto_map:
        config.pop("auto_map", None)

    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO[attn_kind]
    config["num_hidden_layers"] = num_layers
    config["num_key_value_heads"] = 1
    if config.get("architectures") != ["DeepseekV4ForCausalLM"]:
        config["architectures"] = ["DeepseekV4ForCausalLM"]

    # transformers has no "deepseek_v4" entry in CONFIG_MAPPING.  The V4
    # sglang fork's get_config only triggers its V4 loader when AutoConfig
    # fails with "deepseek_ref" or "deepseek_v32" in the message.  Rewriting
    # model_type to "deepseek_v3" mirrors what sglang's
    # _load_deepseek_temp_model produces internally, so AutoConfig succeeds
    # and the V4 model class is still selected via the architectures field.
    config["model_type"] = "deepseek_v3"

    # gemm_type "fp8_block" overrides disable_weight_quant: we MUST keep the
    # fp8 quantization_config so sglang dispatches projections to DeepGEMM.
    drop_quant = disable_weight_quant and gemm_type != "fp8_block"
    if drop_quant:
        config.pop("quantization_config", None)
        config.pop("compression_config", None)

    old_ratios = config.get("compress_ratios") or []
    if old_ratios:
        config["compress_ratios"] = [compress_ratio] * num_layers
    else:
        config["compress_ratios"] = [compress_ratio] * num_layers

    if shrink_unused_moe:
        # V4 DeepseekV4DecoderLayer always constructs ``self.mlp = DeepseekV2MoE``
        # (no dense-MLP fallback like V2's ``first_k_dense_replace`` toggles), so
        # the MoE weights *are* allocated even though forward only calls
        # ``layer.self_attn``.  Shrink only the count of experts; keep the per-
        # expert intermediate dim and the shared-experts count at production
        # values, because:
        #   - ``moe_intermediate_size`` shows up as the ``output_size`` of
        #     ColumnParallelLinear in fp8 block-quant; per-partition size must
        #     be divisible by ``block_n=128`` (``fp8.py:validate_block_quant_shapes``).
        #     Production 2048 / TP=8 = 256 (ok); shrinking to 256 would give
        #     32 at TP=8 and trigger a quantization shape error.
        #   - Setting ``n_shared_experts=0`` makes ``DeepseekV2MoE`` build a
        #     shared expert with intermediate=0, which divides-by-zero in
        #     ``validate_block_quant_shapes``.
        # 8 routed experts x 2048 inter x 7168 hidden x 1 byte fp8 ≈ 230 MB
        # per layer, comfortable on one H20.
        config["n_routed_experts"] = min(int(config.get("n_routed_experts", 8)), 8)
        config["num_experts_per_tok"] = min(int(config.get("num_experts_per_tok", 2)), 2)

    tmp_dir = os.path.join(
        tempfile.gettempdir(),
        f"aic_dsv4_{attn_kind}_{model_path.replace('/', '_')}_{os.getpid()}",
    )
    os.makedirs(tmp_dir, exist_ok=True)
    _copy_non_weight_files(src_dir, tmp_dir)
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)
    return tmp_dir


@contextlib.contextmanager
def _tp_load_model_patch(tp_size: int):
    """Single-process simulation of TP=N rank-0 attention.

    Runs sglang with a real torch.distributed group of world_size=1 (no NCCL
    setup) but lies to the model-construction code so ColumnParallelLinear /
    RowParallelLinear allocate weights as if running on N ranks.

    Mechanics:
      1.  ``ModelRunner.load_model`` is wrapped.  Just before it constructs
          the model, set ``ps._TP.world_size = N`` and ``rank_in_group = 0``.
          ``get_tensor_model_parallel_world_size()`` returns N, so projection
          ``ColumnParallelLinear``/``RowParallelLinear`` allocate ``out//N``
          shards.  ``dp_attention._ATTN_TP_SIZE/RANK`` are set to (N, 0).
      2.  After ``load_model`` returns, ``_TP.world_size`` is restored to 1.
          Any forward-time ``tensor_model_parallel_all_reduce`` /
          ``_all_gather`` then short-circuits at ``world_size == 1`` (the real
          group only has 1 rank, so a real collective would hang/fail anyway).
      3.  ``_ATTN_TP_SIZE`` is **NOT** restored.  V4's ``_forward_prepare``
          reads ``get_attention_tp_size()`` at forward time for the
          ``q_padded[..., n_heads]`` / ``q_out = q_padded[:, tp_slice, :]``
          zero-pad logic; keeping it at N is what makes FlashMLA receive the
          fixed h_q=64 with only the rank-0 slice filled (matching prod TP=N
          rank-0 byte-for-byte).

    Why this is safe:
      - FlashMLA's ``Unsupported h_q: 8`` error is avoided because Q is always
        zero-padded to h_q=64 before FMLA — at any N, FMLA sees h_q=64.
      - V4 main attention's ``wq_b`` is ``ColumnParallelLinear`` and stores
        ``self.tp_size`` at construction (read once), so forward uses N
        without re-querying _TP.world_size.
      - Indexer / Compressor are ``ReplicatedLinear`` (no sharding); they are
        unaffected by the patch.

    What the measured kernel time represents: the cost of attention module
    forward on **one** rank of a real TP=N deployment, including projection
    GEMMs at the correctly sharded shape and full-resolution attention
    kernels (FMLA/paged_mqa_logits/compressor — TP-invariant).
    """
    if tp_size <= 1:
        yield
        return

    import sglang.srt.distributed.parallel_state as ps
    import sglang.srt.layers.dp_attention as dp_attn
    from sglang.srt.model_executor.model_runner import ModelRunner

    orig_load = ModelRunner.load_model

    def patched_load(self):
        tp_group = ps._TP
        assert tp_group is not None, (
            "_TP not initialized; ModelRunner.load_model called before init_distributed_environment ran."
        )
        orig_world_size = tp_group.world_size
        orig_rank = tp_group.rank_in_group
        tp_group.world_size = tp_size
        tp_group.rank_in_group = 0
        dp_attn._ATTN_TP_SIZE = tp_size
        dp_attn._ATTN_TP_RANK = 0
        try:
            return orig_load(self)
        finally:
            # Restore _TP for forward-time collective short-circuit; leave
            # _ATTN_TP_SIZE at N because V4 forward re-reads it for tp_slice.
            tp_group.world_size = orig_world_size
            tp_group.rank_in_group = orig_rank

    ModelRunner.load_model = patched_load
    try:
        yield
    finally:
        ModelRunner.load_model = orig_load


def _effective_prefill_chunk_size(model_runner) -> int:
    """Return the query-token limit enforced by this collector path."""
    runtime_chunk = _runtime_chunk_size(model_runner)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    smem_optin = int(properties.shared_memory_per_block_optin)
    sched_meta_max_queries = (smem_optin // 4 - 1) // 5
    if sched_meta_max_queries <= 0:
        raise RuntimeError(f"invalid sched-meta query limit from shared_memory_per_block_optin={smem_optin}")
    return min(runtime_chunk, sched_meta_max_queries)


def _derive_csa_context_pool_cap(
    model_runner,
    shapes: Iterable[tuple[int, int, int]],
) -> tuple[int, int, int]:
    """Derive a DSV4 CSA-context pool cap while reserving its logits output."""
    from sglang.srt.environ import envs
    from sglang.srt.model_executor.pool_configurator import (
        DSV4PoolConfigurator,
        create_memory_pool_configurator,
    )

    sm_major, sm_minor = torch.cuda.get_device_capability(model_runner.device)
    sm_version = sm_major * 10 + sm_minor
    if sm_version == 120:
        # Exact SGLang 0.5.14 forces a Torch logits leaf on SM120.  That leaf
        # materializes additional FP32 gather/BMM intermediates, so the
        # DeepGEMM output-workspace formula below is not its memory contract.
        raise RuntimeError(
            "DSV4 CSA context on SM120 uses the Torch indexer and needs a separately "
            "validated workspace policy; the SM90/100/103 DeepGEMM formula is disabled"
        )
    if sm_version not in {90, 100, 103}:
        raise RuntimeError(
            "DSV4 CSA context pool derivation supports the DeepGEMM indexer on "
            f"SM90/100/103; SM{sm_version} needs a separately validated workspace policy"
        )

    alternate_indexers = {
        "tilelang": envs.SGLANG_OPT_USE_TILELANG_INDEXER.get(),
        "aiter": envs.SGLANG_OPT_USE_AITER_INDEXER.get(),
        "torch": envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get(),
    }
    selected_alternates = [name for name, enabled in alternate_indexers.items() if enabled]
    if selected_alternates:
        raise RuntimeError(
            "DSV4 CSA context pool derivation is not validated for indexer leaf/leaves " + ",".join(selected_alternates)
        )
    if bool(getattr(model_runner.server_args, "enable_deepseek_v4_fp4_indexer", False)):
        raise RuntimeError("DSV4 CSA context pool derivation is not validated for the FP4 indexer")

    configurator = create_memory_pool_configurator(model_runner)
    if not isinstance(configurator, DSV4PoolConfigurator):
        raise TypeError(f"expected SGLang 0.5.14 DSV4PoolConfigurator, got {type(configurator).__name__}")

    page_size = int(model_runner.server_args.page_size)
    if page_size != _DSV4_CUDA_PAGE_SIZE:
        raise RuntimeError(f"expected DSV4 CUDA page_size={_DSV4_CUDA_PAGE_SIZE}, got {page_size}")
    window_size = int(model_runner.model_config.window_size)
    if window_size <= 0:
        raise RuntimeError(f"invalid DSV4 SWA window_size={window_size}")

    effective_chunk = _effective_prefill_chunk_size(model_runner)
    chunk_eligible_shapes = [
        (int(batch_size), int(seq_len), int(prefix_len))
        for batch_size, seq_len, prefix_len in shapes
        if required_prefill_extend_tokens(int(batch_size), int(seq_len)) <= effective_chunk
    ]
    if not chunk_eligible_shapes:
        raise RuntimeError(f"DSV4 CSA context pool derivation has no shape within effective_chunk={effective_chunk}")

    profiled_bytes = int(model_runner._profile_available_bytes(model_runner.pre_model_load_memory))
    profiled_config = configurator.calculate_pool_sizes(profiled_bytes, page_size)
    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO["csa"]
    if page_size % compress_ratio != 0:
        raise RuntimeError(f"page_size={page_size} is not divisible by CSA ratio={compress_ratio}")
    c4_page_size = page_size // compress_ratio
    # sgl-deep-gemm 0.1.3 aligns the FP8 logits stride to split_kv=256 on
    # architecture major 9/10 (csrc/apis/attention.hpp).
    logits_alignment = 256
    fp32_bytes = 4
    eligible_shapes = []
    source_capacity_excluded = 0
    workspace_shape = None
    workspace_bytes = 0
    required_full_tokens = 0
    required_swa_tokens = 0
    for batch_size, seq_len, prefix_len in chunk_eligible_shapes:
        full_tokens = required_kv_alloc_tokens(
            batch_size,
            seq_len,
            prefix_len,
            page_size,
            is_prefill=True,
        )
        swa_tokens = required_swa_kv_alloc_tokens(
            batch_size,
            seq_len,
            prefix_len,
            page_size,
            window_size,
            is_prefill=True,
        )
        if (
            full_tokens > profiled_config.full_max_total_num_tokens
            or swa_tokens > profiled_config.swa_max_total_num_tokens
        ):
            source_capacity_excluded += 1
            continue

        raw_c4_width = ((prefix_len + seq_len + page_size - 1) // page_size) * c4_page_size
        logits_stride = ((raw_c4_width + logits_alignment - 1) // logits_alignment) * logits_alignment
        shape_workspace_bytes = batch_size * seq_len * logits_stride * fp32_bytes
        if shape_workspace_bytes > workspace_bytes:
            workspace_shape = (batch_size, seq_len, prefix_len)
            workspace_bytes = shape_workspace_bytes
        eligible_shapes.append((batch_size, seq_len, prefix_len))
        required_full_tokens = max(required_full_tokens, full_tokens)
        required_swa_tokens = max(required_swa_tokens, swa_tokens)

    if not eligible_shapes:
        raise RuntimeError(
            "DSV4 CSA context pool derivation has no shape within the framework-profiled "
            f"full/SWA capacities {profiled_config.full_max_total_num_tokens}/"
            f"{profiled_config.swa_max_total_num_tokens}"
        )

    swa_ratio = Fraction(str(configurator.swa_ratio))
    if not 0 < swa_ratio <= 1:
        raise RuntimeError(f"invalid DSV4 swa_full_tokens_ratio={configurator.swa_ratio}")
    required_from_swa = (required_swa_tokens * swa_ratio.denominator + swa_ratio.numerator - 1) // swa_ratio.numerator
    required_cap = max(required_full_tokens, required_from_swa)
    required_cap = ((required_cap + page_size - 1) // page_size) * page_size
    required_config = configurator.calculate_pool_sizes_from_max_tokens(required_cap, page_size)
    if (
        required_config.full_max_total_num_tokens < required_full_tokens
        or required_config.swa_max_total_num_tokens < required_swa_tokens
    ):
        raise RuntimeError(
            "SGLang DSV4 configurator could not preserve the retained full/SWA requirements: "
            f"required_full={required_full_tokens}, configured_full="
            f"{required_config.full_max_total_num_tokens}, required_swa={required_swa_tokens}, "
            f"configured_swa={required_config.swa_max_total_num_tokens}"
        )

    if workspace_bytes >= profiled_bytes:
        raise RuntimeError(
            f"DSV4 CSA logits workspace={workspace_bytes} leaves no pool budget from "
            f"profiled_bytes={profiled_bytes}; shape={workspace_shape}"
        )
    workspace_config = configurator.calculate_pool_sizes(profiled_bytes - workspace_bytes, page_size)
    workspace_cap = int(workspace_config.max_total_num_tokens)
    if required_cap > workspace_cap:
        raise RuntimeError(
            "one DSV4 CSA context worker cannot preserve all framework-feasible cells and "
            f"reserve its maximum logits workspace: required_cap={required_cap}, "
            f"workspace_cap={workspace_cap}; split only the measured Pareto-distinct groups"
        )

    print(
        "[dsv4-collector] CSA context derived pool: "
        f"sm={sm_version}, indexer_leaf=deep_gemm, page_size={page_size}, "
        f"swa_ratio={float(swa_ratio)}, effective_chunk={effective_chunk}, "
        f"eligible_shapes={len(eligible_shapes)}, source_capacity_excluded={source_capacity_excluded}, "
        f"profiled_bytes={profiled_bytes}, "
        f"profiled_full={profiled_config.full_max_total_num_tokens}, "
        f"profiled_swa={profiled_config.swa_max_total_num_tokens}, "
        f"workspace_bytes={workspace_bytes}, workspace_shape={workspace_shape}, "
        f"bytes_per_full_token={configurator.bytes_per_full_token:.6f}, "
        f"workspace_cap={workspace_cap}, required_full={required_full_tokens}, "
        f"required_swa={required_swa_tokens}, final_cap={required_cap}"
    )
    return required_cap, required_full_tokens, required_swa_tokens


def _load_model_runner(
    model_path: str,
    *,
    attn_kind: str,
    num_layers: int,
    kv_cache_dtype: str,
    device: str,
    shrink_unused_moe: bool,
    disable_weight_quant: bool,
    gemm_type: str = "bfloat16",
    tp_size: int = 1,
    max_total_tokens: int | None = None,
    csa_context_shapes: Iterable[tuple[int, int, int]] | None = None,
):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()
    torch.cuda.set_device(device)

    load_format = os.environ.get("SGLANG_LOAD_FORMAT", "dummy")
    local_model_path = _resolve_model_path(
        model_path,
        attn_kind=attn_kind,
        num_layers=num_layers,
        shrink_unused_moe=shrink_unused_moe,
        disable_weight_quant=disable_weight_quant,
        gemm_type=gemm_type,
        load_format=load_format,
    )
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    # CUDA_VISIBLE_DEVICES remaps every child to cuda:0; keep the physical GPU
    # id for NCCL port sharding so parallel workers do not collide.
    port_shard = int(os.environ.get("AIC_DSV4_PORT_SHARD", gpu_id))
    nccl_port = int(os.environ.get("AIC_DSV4_NCCL_PORT") or _pick_free_port(port_shard))
    # Mirror collect_mla_module.py's "grab sglang's own defaults, only override
    # what is strictly needed" approach: construct ServerArgs and let its
    # ``__post_init__`` derive the runtime-sizing knobs from sglang's own logic.
    # Do NOT pass collector-derived values for mem_fraction_static,
    # chunked_prefill_size, or max_prefill_tokens -- leaving them unset means:
    #   * mem_fraction_static=None -> derived from GPU memory in __post_init__
    #   * chunked_prefill_size=None -> derived from GPU memory
    #   * max_prefill_tokens         -> sglang default (16384)
    # An explicit ``max_total_tokens`` is reserved for direct diagnostics.  The
    # normal CSA-context path starts with None and, after model load, derives a
    # workspace-aware bound through SGLang's live DSV4PoolConfigurator.  HCA and
    # generation keep the framework-derived None path.

    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format=load_format,
        tp_size=1,
        trust_remote_code=True,
        disable_radix_cache=True,
        # The module benchmark below captures its own CUDA Graph and fails if
        # capture is not possible.  Keep SGLang's serving-level graph runner off
        # so it does not add unrelated full-model graph state to this collector.
        disable_decode_cuda_graph=True,
        disable_prefill_cuda_graph=True,
        kv_cache_dtype=kv_cache_dtype,
        # The bench sweep includes batch_size up to 1024 (collector's
        # ``_BATCH_SIZES``).  sglang's ``alloc_req_slots`` exposes
        # ``available_size = max_running_requests - 1`` (one slot is
        # reserved internally), so a bs=1024 cell with
        # ``max_running_requests=1024`` raises
        # ``alloc_req_slots runs out of memory: available=1023, num_reqs=1024``.
        # Bump to 1100 for headroom over the largest tested bs.  This is the
        # only sizing knob the collector must override (per-cell request slots
        # for the bs sweep); all memory/token knobs come from sglang.
        max_running_requests=1100,
        # None -> SGLang default sizing.  mem_fraction_static and
        # chunked_prefill_size also stay unset and are derived by
        # ServerArgs.__post_init__.
        max_total_tokens=max_total_tokens,
    )
    # gemm_type controls projection GEMM dispatch.  "fp8_block" → DeepGEMM
    # (matches production V4-Flash-FP8); anything else → cuBLASLt bf16.
    server_args.quantization = "fp8" if gemm_type == "fp8_block" else None
    server_args.attention_backend = "dsv4"

    print(
        f"[dsv4-collector] model_path {model_path} -> {local_model_path}; "
        f"attn_kind={attn_kind}, backend=dsv4, kv_cache_dtype={kv_cache_dtype}, "
        f"mem_fraction_static={server_args.mem_fraction_static} (sglang-derived), "
        f"chunked_prefill_size={server_args.chunked_prefill_size} (sglang-derived), "
        f"max_prefill_tokens={server_args.max_prefill_tokens} (sglang-derived), "
        f"max_total_tokens_input={server_args.max_total_tokens}, "
        f"shrink_unused_moe={shrink_unused_moe}, "
        f"disable_weight_quant={disable_weight_quant}, gemm_type={gemm_type}, "
        f"quantization={server_args.quantization}, tp_size={tp_size}, nccl_port={nccl_port}"
    )

    _set_envs_and_config(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    with _tp_load_model_patch(tp_size):
        model_runner = ModelRunner(
            model_config=model_config,
            # Use sglang's own __post_init__-derived value, not a collector knob.
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            moe_ep_rank=0,
            moe_ep_size=1,
            nccl_port=nccl_port,
            server_args=server_args,
        )
    derived_requirements = None
    if csa_context_shapes is not None:
        if max_total_tokens is not None:
            raise RuntimeError("CSA context live pool derivation cannot be combined with an explicit token cap")
        derived_requirements = _derive_csa_context_pool_cap(
            model_runner,
            csa_context_shapes,
        )
        model_runner.server_args.max_total_tokens = derived_requirements[0]
    # SGLang 0.5.14 separates model construction from serving-state setup.
    model_runner.alloc_memory_pool()
    if derived_requirements is not None:
        _, required_full_tokens, required_swa_tokens = derived_requirements
        actual_full_tokens = int(model_runner.full_max_total_num_tokens)
        actual_swa_tokens = int(model_runner.swa_max_total_num_tokens)
        if actual_full_tokens < required_full_tokens or actual_swa_tokens < required_swa_tokens:
            raise RuntimeError(
                "SGLang DSV4 pool shrank below the derived retained-case requirements: "
                f"actual_full={actual_full_tokens}, required_full={required_full_tokens}, "
                f"actual_swa={actual_swa_tokens}, required_swa={required_swa_tokens}"
            )
    model_runner.init_attention_backends()
    # --- AIC proper init (env-gated) -------------------------------------
    # Dummy load uses uniform(-1e-3, 1e-3) (sglang initialize_dummy_weights).
    # Those tiny weights make the C4 indexer's q.k logits land in the fp16
    # subnormal range (<6.1e-5). extract_coarse_bin casts scores to fp16
    # first, so subnormal logits collapse into very few coarse bins -> the
    # topK threshold bin holds hundreds of ties (num_equal >> 64) -> Small
    # path takes its O(n^2) block tie-break -> topk_short_transform is huge
    # and non-representative vs a trained checkpoint.
    # Re-init projection weights with a normal of larger scale so logits keep
    # full fp16 precision (spread). norm/scale params are forced to 1.0 to
    # avoid scaling activations back down (norm) or producing inf on fp8
    # dequant (scale). Only safe because the collector runs num_layers==1.
    _proper = os.environ.get("AIC_DSV4_PROPER_INIT", "")
    if _proper and _proper != "0":
        _std = float(os.environ.get("AIC_DSV4_PROPER_INIT_STD", "0.05"))
        torch.manual_seed(1234)
        _n_normal = _n_norm = _n_scale = 0
        with torch.no_grad():
            for _name, _p in model_runner.model.state_dict().items():
                if not torch.is_floating_point(_p):
                    continue
                _lname = _name.lower()
                if "scale" in _lname:
                    _p.fill_(1.0)
                    _n_scale += 1
                elif "norm" in _lname:
                    _p.fill_(1.0)
                    _n_norm += 1
                elif torch.finfo(_p.dtype).bits < 16:
                    _tmp = torch.empty_like(_p, dtype=torch.float16).normal_(0.0, _std)
                    _p.copy_(_tmp.to(_p.dtype))
                    _n_normal += 1
                else:
                    _p.normal_(0.0, _std)
                    _n_normal += 1
        print(
            f"[dsv4-collector] AIC_DSV4_PROPER_INIT std={_std}: "
            f"normal={_n_normal} norm->1={_n_norm} scale->1={_n_scale}",
            flush=True,
        )

    allocator = model_runner.token_to_kv_pool_allocator
    pool_parts = []
    for name in (
        "max_total_num_tokens",
        "full_max_total_num_tokens",
        "swa_max_total_num_tokens",
        "c4_max_total_num_tokens",
        "c128_max_total_num_tokens",
        "c4_state_pool_size",
        "c128_state_pool_size",
    ):
        if hasattr(model_runner, name):
            pool_parts.append(f"{name}={getattr(model_runner, name)}")
    if hasattr(allocator, "debug_print"):
        pool_parts.append(allocator.debug_print().strip())
    elif hasattr(allocator, "available_size"):
        pool_parts.append(f"available_size={allocator.available_size()}")
    print("[dsv4-collector] pool " + ", ".join(pool_parts))
    return model_runner


def _make_reqs(
    batch_size: int,
    seq_len: int,
    *,
    decode: bool,
    prefix_len: int = 0,
    prefix_indices: list[torch.Tensor] | None = None,
    swa_evicted_seqlen: int = 0,
):
    from array import array

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    full_len = prefix_len + seq_len
    prefix_indices = prefix_indices or [torch.empty((0,), dtype=torch.int64, device="cuda") for _ in range(batch_size)]

    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=str(i),
            origin_input_text="",
            origin_input_ids=list(torch.randint(0, 10000, (full_len,)).tolist()),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.prefix_indices = prefix_indices[i]
        req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
        req.fill_len = full_len
        req.logprob_start_len = 0
        req.set_extend_input_len(seq_len if prefix_len else full_len)
        req.swa_evicted_seqlen = swa_evicted_seqlen
        if decode:
            req.cached_tokens = 0
            req.already_computed = 0
        reqs.append(req)
    return reqs


def _build_forward_batch(
    model_runner,
    batch_size: int,
    seq_len: int,
    *,
    is_prefill: bool,
    prefix_len: int = 0,
):
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.chunk_cache import ChunkCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    cached_prefix_len = prefix_len if is_prefill else seq_len
    extend_seq_len = seq_len if is_prefill else 0
    prefix_indices = _alloc_prefix_indices(model_runner, batch_size, cached_prefix_len)
    swa_evicted_seqlen = 0
    allocator = model_runner.token_to_kv_pool_allocator
    window_size = getattr(model_runner.model_config, "window_size", None)
    page_size = getattr(allocator, "page_size", None)
    if (
        cached_prefix_len > 0
        and callable(getattr(allocator, "alloc_extend_swa_tail", None))
        and isinstance(window_size, int)
        and window_size > 0
        and isinstance(page_size, int)
        and page_size > 1
    ):
        swa_evicted_seqlen = max(0, cached_prefix_len - window_size)
        swa_evicted_seqlen = (swa_evicted_seqlen // page_size) * page_size
    reqs = _make_reqs(
        batch_size,
        extend_seq_len,
        decode=not is_prefill,
        prefix_len=cached_prefix_len,
        prefix_indices=prefix_indices,
        swa_evicted_seqlen=swa_evicted_seqlen,
    )
    cache_params = CacheInitParams(
        disable=True,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        page_size=model_runner.token_to_kv_pool_allocator.page_size,
    )
    tree_cache = ChunkCache(cache_params)
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )

    with _temporarily_chunked_alloc_extend(model_runner, batch_size * extend_seq_len):
        if is_prefill:
            batch.prepare_for_extend()
        else:
            batch.prepare_for_extend()
            for req in batch.reqs:
                req.output_ids.append(0)
            batch.prepare_for_decode()

    forward_batch = ForwardBatch.init_new(batch, model_runner)
    model_runner.attn_backend.init_forward_metadata(forward_batch)
    return forward_batch


def _make_inputs(
    model_runner,
    *,
    batch_size: int,
    seq_len: int,
    is_prefill: bool,
    device: str,
    prefix_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = model_runner.model.config.hidden_size
    max_pos = getattr(model_runner.model_config.hf_config, "max_position_embeddings", None)
    if is_prefill:
        full_len = prefix_len + seq_len
        if max_pos is not None and full_len > max_pos:
            raise ValueError(
                f"context full_len={full_len} exceeds max_position_embeddings={max_pos} "
                f"(seq_len={seq_len}, prefix_len={prefix_len})"
            )
        n_tokens = batch_size * seq_len
        positions = (
            torch.arange(prefix_len, prefix_len + seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
            .flatten()
        )
    else:
        if max_pos is not None and seq_len >= max_pos:
            raise ValueError(
                f"decode seq_len={seq_len} >= max_position_embeddings={max_pos}; "
                f"max valid decode seq_len is {max_pos - 1}"
            )
        n_tokens = batch_size
        positions = torch.full((batch_size,), seq_len, dtype=torch.int64, device=device)

    hidden_states = torch.randn(
        n_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    return hidden_states, positions


def _bench_cuda_events(
    kernel_func,
    num_warmup: int,
    num_iterations: int,
    graph_repeat: int = 1,
    device: str = "cuda:0",
) -> dict[str, float]:
    """Benchmark through AIC's benchmark_with_power helper.

    benchmark_with_power handles warmup, CUDA Graph capture/replay, optional
    power sampling, and graph-private-pool teardown.  Capture failure is a hard
    error: allow_graph_fail=False and used_cuda_graph is checked explicitly.
    """

    if num_iterations < 3:
        raise ValueError("num_iterations must be at least 3")
    if graph_repeat < 1:
        raise ValueError("graph_repeat must be at least 1")

    def timed_kernel():
        with torch.no_grad():
            return kernel_func()

    with benchmark_with_power(
        device=torch.device(device),
        kernel_func=timed_kernel,
        num_warmups=num_warmup,
        num_runs=num_iterations,
        repeat_n=graph_repeat,
        allow_graph_fail=False,
    ) as result:
        pass

    if not result.get("used_cuda_graph", False):
        raise RuntimeError("benchmark_with_power did not use CUDA Graph")

    latency_ms = float(result["latency_ms"])
    return {
        "mean_ms": latency_ms,
        "median_ms": latency_ms,
        "min_ms": latency_ms,
        "max_ms": latency_ms,
        "std_ms": 0.0,
        "n": int(result.get("num_runs_executed", num_iterations)),
        "used_cuda_graph": True,
        "power_stats": result.get("power_stats"),
        "throttled": bool(result.get("throttled", False)),
    }


def _log_result(
    *,
    output_path: str | None,
    model_path: str,
    mode: str,
    attn_kind: str,
    compress_ratio: int,
    batch_size: int,
    seq_len: int,
    kv_cache_dtype: str,
    latency_ms: float,
    version: str,
    device_name: str,
    power_stats: dict | None = None,
    perf_filename_prefix: str = "dsv4",
    gemm_type: str = "bfloat16",
    tp_size: int = 1,
    step: int | None = None,
    num_heads: int | None = None,
) -> None:
    # V4-Flash output layout: ONE CSV per (attn_kind, mode) — 3 kinds x 2
    # modes = 6 files total, regardless of how many (tp_size, gemm_type)
    # subprocesses run.  Within each file, rows are disambiguated by the
    # ``tp_size``, ``gemm_type``, ``batch_size``, ``isl`` columns.
    # ``log_perf`` is file-locked so concurrent appends from different
    # subprocesses to the same kind+mode file are safe.
    # Non-V4-Flash callers (legacy ``dsv4`` MLA module) still use the old
    # per-(prefix, kind) filename layout to avoid behavior breaks.
    if perf_filename_prefix.startswith("dsv4"):
        consolidated_filename = f"dsv4_{attn_kind}_{mode}_module_perf.txt"
    else:
        consolidated_filename = f"{perf_filename_prefix}_{attn_kind}_{mode}_module_perf.txt"
    perf_filename = _resolve_perf_path(output_path, consolidated_filename)
    is_prefill = mode == "context"
    step_value = step if step is not None else (0 if is_prefill else seq_len)
    if not log_perf(
        item_list=[
            {
                "model": _canonical_dsv4_model_id(model_path),
                "architecture": "DeepseekV4ForCausalLM",
                "mla_dtype": "bfloat16",
                "kv_cache_dtype": kv_cache_dtype,
                "gemm_type": gemm_type,
                "num_heads": num_heads if num_heads is not None else max(1, NATIVE_HEADS // tp_size),
                "batch_size": batch_size,
                "isl": seq_len if is_prefill else 1,
                "tp_size": tp_size,
                "step": step_value,
                "compress_ratio": compress_ratio,
                "latency": f"{latency_ms:.4f}",
            }
        ],
        framework="SGLang",
        version=version,
        device_name=device_name,
        # op_name still encodes the run config so a single-CSV view can group
        # by op_name when needed (e.g. for plotting per-(kind, tp, gemm)).
        op_name=f"{perf_filename_prefix}_{attn_kind}_{mode}_module",
        kernel_source="compressed_flashmla",
        perf_filename=perf_filename,
        power_stats=power_stats,
    ):
        raise RuntimeError(f"failed to persist DeepSeek-V4 attention row to {perf_filename}")


def run_dsv4_mla_module(
    *,
    model_path: str = CLI_DEFAULT_MODEL,
    mode: str,
    attn_kind: str,
    batch_sizes: Iterable[int],
    seq_lens: Iterable[int],
    layer_id: int = 0,
    num_layers: int = 1,
    kv_cache_dtype: str = "fp8_e4m3",
    num_warmup: int = 5,
    num_iterations: int = 20,
    graph_repeat: int = 1,
    device: str = "cuda:0",
    output_path: str | None = None,
    shrink_unused_moe: bool = True,
    disable_weight_quant: bool = True,
    perf_filename_prefix: str = "dsv4",
    gemm_type: str = "bfloat16",
    tp_size: int = 1,
    prefix_len: int = 0,
    prefix_lens: Iterable[int] | None = None,
    seq_lens_by_prefix: dict[int, list[int]] | None = None,
    max_total_tokens: int | None = None,
) -> list[dict[str, float]]:
    is_prefill = mode == "context"
    batch_sizes = list(batch_sizes)
    default_seq_lens = list(seq_lens)
    if prefix_lens is None:
        prefix_values = [prefix_len]
    else:
        prefix_values = list(prefix_lens)
    if not is_prefill:
        prefix_values = [0]
    if any(p > 0 for p in prefix_values) and not is_prefill:
        raise ValueError("prefix_len is only supported for context/extend collection")

    case_shapes = [
        (batch_size, seq_len, cur_prefix)
        for cur_prefix in prefix_values
        for batch_size in batch_sizes
        for seq_len in (
            seq_lens_by_prefix.get(cur_prefix, default_seq_lens) if seq_lens_by_prefix is not None else default_seq_lens
        )
    ]

    # SGLang derives compressed-pool sizes from max_total_tokens. Direct
    # probes with an explicit value must account for both the full pool and
    # the smaller page-rounded SWA tail pool before ModelRunner construction.
    if max_total_tokens is not None:
        max_full_alloc = 0
        max_swa_alloc = 0
        for batch_size, seq_len, cur_prefix in case_shapes:
            max_full_alloc = max(
                max_full_alloc,
                required_kv_alloc_tokens(
                    batch_size,
                    seq_len,
                    cur_prefix,
                    _DSV4_CUDA_PAGE_SIZE,
                    is_prefill=is_prefill,
                ),
            )
            max_swa_alloc = max(
                max_swa_alloc,
                required_swa_kv_alloc_tokens(
                    batch_size,
                    seq_len,
                    cur_prefix,
                    _DSV4_CUDA_PAGE_SIZE,
                    _DSV4_SWA_WINDOW_SIZE,
                    is_prefill=is_prefill,
                ),
            )
        max_total_tokens = max(max_total_tokens, max_full_alloc, max_swa_alloc * _DSV4_SWA_TO_FULL_SCALE)
        max_total_tokens = (
            (max_total_tokens + _DSV4_CUDA_PAGE_SIZE - 1) // _DSV4_CUDA_PAGE_SIZE
        ) * _DSV4_CUDA_PAGE_SIZE

    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO[attn_kind]
    if tp_size not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"tp_size must be a power of 2 in [1, 32]; got {tp_size}")
    model_runner = _load_model_runner(
        model_path,
        attn_kind=attn_kind,
        num_layers=max(num_layers, layer_id + 1),
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        shrink_unused_moe=shrink_unused_moe,
        disable_weight_quant=disable_weight_quant,
        gemm_type=gemm_type,
        tp_size=tp_size,
        max_total_tokens=max_total_tokens,
        csa_context_shapes=(case_shapes if is_prefill and attn_kind == "csa" and max_total_tokens is None else None),
    )

    attention_module = model_runner.model.model.layers[layer_id].self_attn
    actual_ratio = getattr(attention_module, "compress_ratio", None)
    if actual_ratio != compress_ratio:
        raise RuntimeError(f"target layer compress_ratio mismatch: expected {compress_ratio}, got {actual_ratio}")
    native_attention_heads = int(getattr(attention_module, "n_heads", NATIVE_HEADS))
    local_attention_heads = max(1, native_attention_heads // tp_size)

    print(
        f"[dsv4-collector] layer={layer_id}, attn_kind={attn_kind}, "
        f"compress_ratio={actual_ratio}, mode={mode}, prefix_lens={prefix_values}"
    )

    version = get_version("sglang")
    device_name = torch.cuda.get_device_name(device)
    results = []
    skipped_shapes: list[tuple[int, int, int, str]] = []
    sweep_label = f"kind={attn_kind} mode={mode} tp={tp_size} gemm={gemm_type}"
    try:
        kv_capacity = _kv_pool_capacity_tokens(model_runner)
        kv_page_size = _kv_pool_page_size(model_runner)
        swa_capacity = _swa_kv_pool_capacity_tokens(model_runner)
        swa_window_size = getattr(getattr(model_runner, "model_config", None), "window_size", None)
        # The current 0.5.14 collector preserves its existing sched-meta limit.
        # Use the same live value for pool derivation and execution so shapes
        # that are later skipped cannot inflate the reserved workspace.
        runtime_chunk = _effective_prefill_chunk_size(model_runner) if is_prefill else None
        for cur_prefix in prefix_values:
            seq_lens_for_prefix = (
                seq_lens_by_prefix.get(cur_prefix, default_seq_lens)
                if seq_lens_by_prefix is not None
                else default_seq_lens
            )
            for batch_size in batch_sizes:
                for seq_len in seq_lens_for_prefix:
                    fresh_tokens = required_prefill_extend_tokens(batch_size, seq_len)
                    if runtime_chunk is not None and fresh_tokens > runtime_chunk:
                        print(
                            f"[SKIP] dsv4-flash {sweep_label} bs={batch_size} sl={seq_len} "
                            f"prefix={cur_prefix}: fresh_tokens={fresh_tokens} exceeds "
                            f"SGLang runtime chunked_prefill_size={runtime_chunk}"
                        )
                        skipped_shapes.append((batch_size, seq_len, cur_prefix, "ChunkedPrefillSize"))
                        continue
                    total_tokens = required_kv_tokens(
                        batch_size,
                        seq_len,
                        cur_prefix,
                        is_prefill=is_prefill,
                    )
                    if kv_capacity is not None and total_tokens > kv_capacity:
                        print(
                            f"[SKIP] dsv4-flash {sweep_label} bs={batch_size} sl={seq_len} "
                            f"prefix={cur_prefix}: total_tokens={total_tokens} exceeds actual "
                            f"KV pool capacity={kv_capacity}"
                        )
                        skipped_shapes.append((batch_size, seq_len, cur_prefix, "KVPoolCapacity"))
                        continue
                    # Paged alloc_extend over-estimate: each request rounds its KV
                    # span up to a page and reserves one extra page (bs*page_size),
                    # so large-bs cases pass the naive capacity check above yet still
                    # fail alloc_extend with "Prefill out of memory". Skip them here
                    # instead of letting the forward raise (read page_size at runtime;
                    # this is a no-op for page_size==1 token allocators).
                    alloc_tokens = required_kv_alloc_tokens(
                        batch_size,
                        seq_len,
                        cur_prefix,
                        kv_page_size,
                        is_prefill=is_prefill,
                    )
                    if kv_capacity is not None and alloc_tokens > kv_capacity:
                        print(
                            f"[SKIP] dsv4-flash {sweep_label} bs={batch_size} sl={seq_len} "
                            f"prefix={cur_prefix}: alloc_tokens={alloc_tokens} (page_size="
                            f"{kv_page_size}) exceeds KV pool capacity={kv_capacity}"
                        )
                        skipped_shapes.append((batch_size, seq_len, cur_prefix, "KVPoolAllocPaged"))
                        continue
                    if swa_capacity is not None and isinstance(swa_window_size, int) and swa_window_size > 0:
                        swa_alloc_tokens = required_swa_kv_alloc_tokens(
                            batch_size,
                            seq_len,
                            cur_prefix,
                            kv_page_size,
                            swa_window_size,
                            is_prefill=is_prefill,
                        )
                        if swa_alloc_tokens > swa_capacity:
                            print(
                                f"[SKIP] dsv4-flash {sweep_label} bs={batch_size} sl={seq_len} "
                                f"prefix={cur_prefix}: swa_alloc_tokens={swa_alloc_tokens} exceeds "
                                f"SWA KV pool capacity={swa_capacity}"
                            )
                            skipped_shapes.append((batch_size, seq_len, cur_prefix, "KVPoolSWACapacity"))
                            continue
                    print(f"\n{mode}: batch_size={batch_size}, seq_len={seq_len}, prefix_len={cur_prefix}")
                    try:
                        forward_batch = _build_forward_batch(
                            model_runner,
                            batch_size,
                            seq_len,
                            is_prefill=is_prefill,
                            prefix_len=cur_prefix,
                        )
                        hidden_states, positions = _make_inputs(
                            model_runner,
                            batch_size=batch_size,
                            seq_len=seq_len,
                            is_prefill=is_prefill,
                            device=device,
                            prefix_len=cur_prefix,
                        )

                        def kernel_func(model_runner=model_runner):
                            from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

                            with forward_context(ForwardContext(attn_backend=model_runner.attn_backend)):
                                return attention_module(
                                    x=hidden_states,
                                    positions=positions,
                                    forward_batch=forward_batch,
                                )

                        stats = _bench_cuda_events(
                            kernel_func,
                            num_warmup=num_warmup,
                            num_iterations=num_iterations,
                            graph_repeat=graph_repeat,
                            device=device,
                        )
                        print(
                            f"  latency mean={stats['mean_ms']:.4f} ms, "
                            f"median={stats['median_ms']:.4f} ms, "
                            f"min={stats['min_ms']:.4f} ms, max={stats['max_ms']:.4f} ms, "
                            f"std={stats['std_ms']:.4f} ms, n={stats['n']}"
                        )
                        _log_result(
                            output_path=output_path,
                            model_path=model_path,
                            mode=mode,
                            attn_kind=attn_kind,
                            compress_ratio=compress_ratio,
                            batch_size=batch_size,
                            seq_len=seq_len,
                            kv_cache_dtype=kv_cache_dtype,
                            latency_ms=stats["mean_ms"],
                            version=version,
                            device_name=device_name,
                            power_stats=stats.get("power_stats"),
                            perf_filename_prefix=perf_filename_prefix,
                            gemm_type=gemm_type,
                            tp_size=tp_size,
                            step=cur_prefix if is_prefill else None,
                            num_heads=local_attention_heads,
                        )
                        stats.update(
                            {
                                "batch_size": batch_size,
                                "seq_len": seq_len,
                                "compress_ratio": compress_ratio,
                                "prefix_len": cur_prefix,
                            }
                        )
                        results.append(stats)
                    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                        print(
                            f"[WARN] dsv4-flash {sweep_label} bs={batch_size} sl={seq_len} "
                            f"prefix={cur_prefix}: OOM; skipping this shape"
                        )
                        skipped_shapes.append((batch_size, seq_len, cur_prefix, "OOM"))
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    except Exception as exc:
                        traceback.print_exc()
                        print(
                            f"[WARN] dsv4-flash {sweep_label} bs={batch_size} sl={seq_len} "
                            f"prefix={cur_prefix}: {type(exc).__name__}; skipping this shape"
                        )
                        skipped_shapes.append((batch_size, seq_len, cur_prefix, type(exc).__name__))
                    finally:
                        cleanup_errors = []
                        for _cleanup_label, _cleanup_step in (
                            ("req_to_token_pool.clear", model_runner.req_to_token_pool.clear),
                            ("token_to_kv_pool_allocator.clear", model_runner.token_to_kv_pool_allocator.clear),
                            ("torch.cuda.empty_cache", torch.cuda.empty_cache),
                            ("gc.collect", gc.collect),
                        ):
                            try:
                                _cleanup_step()
                            except Exception as _cleanup_exc:
                                cleanup_errors.append(
                                    f"{_cleanup_label}: {type(_cleanup_exc).__name__}: {_cleanup_exc}"
                                )
                        if cleanup_errors:
                            raise RuntimeError(
                                f"dsv4-flash {sweep_label} bs={batch_size} sl={seq_len} "
                                f"prefix={cur_prefix}: cleanup failed: {'; '.join(cleanup_errors)}"
                            )
    finally:
        final_cleanup_errors = []
        del model_runner
        for cleanup_name, cleanup_fn in (
            ("torch.cuda.empty_cache", torch.cuda.empty_cache),
            ("gc.collect", gc.collect),
        ):
            try:
                cleanup_fn()
            except Exception as cleanup_exc:
                final_cleanup_errors.append(f"{cleanup_name}: {type(cleanup_exc).__name__}: {cleanup_exc}")
        if final_cleanup_errors:
            raise RuntimeError(f"dsv4-flash {sweep_label}: final cleanup failed: {'; '.join(final_cleanup_errors)}")
    capacity_skip_reasons = {
        "ChunkedPrefillSize",
        "KVPoolCapacity",
        "KVPoolAllocPaged",
        "KVPoolSWACapacity",
    }
    skip_count = sum(reason in capacity_skip_reasons for _, _, _, reason in skipped_shapes)
    error_count = len(skipped_shapes) - skip_count
    ok_count = len(results)
    total_count = ok_count + error_count + skip_count
    summary = f"ok={ok_count} error={error_count} skip={skip_count} total={total_count}"
    if skipped_shapes:
        skipped_str = ", ".join(f"(bs={b},sl={s},prefix={p},reason={r})" for b, s, p, r in skipped_shapes)
        print(f"[WARN] dsv4-flash {sweep_label}: skipped or errored shapes: {skipped_str}")
    print(f"[dsv4-collector] {sweep_label}: {summary}")
    if ok_count == 0 or error_count > 0:
        raise RuntimeError(f"dsv4-flash {sweep_label}: {summary}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Subprocess-isolated worker (registry path)
# ═══════════════════════════════════════════════════════════════════════


def _run_subprocess(
    *,
    mode: str,
    attn_kind: str,
    model_path: str,
    kv_cache_dtype_sglang: str,
    batch_size: int,
    output_path: str,
    gpu_id: int,
    gemm_type: str = "bfloat16",
    tp_size: int = 1,
    prefix_len: int = 0,
    prefix_lens: Iterable[int] | None = None,
    smoke: bool = False,
):
    """Run one (attn_kind, tp, gemm, bs) subprocess that sweeps valid sl/prefix.

    Builds one ``ModelRunner`` sized for ``(bs, max_sl_for_this_bs)`` and
    iterates every valid sl for that bs.  Per-sl crash isolation is
    handled by ``run_dsv4_mla_module``'s try/except per forward.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["AIC_DSV4_PORT_SHARD"] = str(gpu_id)
    env.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")
    env.setdefault("SGLANG_LOAD_FORMAT", "dummy")
    # Hard-disable DeepGEMM bulk pre-compile.  First sl in this sweep
    # triggers runtime lazy JIT for the (M, N, K) shapes it needs;
    # subsequent sl within the same subprocess hit in-memory cache.
    env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
    # Newer DeepSeek-V4 defaults route wo_a through an FP8 GEMM.  The bf16
    # collector path deliberately drops quantization_config, so that FP8-only
    # wo_a path would fail model construction due to missing weight_scale_inv.
    env["SGLANG_OPT_FP8_WO_A_GEMM"] = "1" if gemm_type == "fp8_block" else "0"

    prefix_lens_arg = list(prefix_lens) if prefix_lens is not None else None
    code = (
        f'import sys; sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")\n'
        f"from collect_dsv4_attn import _subprocess_entry\n"
        f"_subprocess_entry(\n"
        f'    mode="{mode}",\n'
        f'    attn_kind="{attn_kind}",\n'
        f'    model_path="{model_path}",\n'
        f'    kv_cache_dtype="{kv_cache_dtype_sglang}",\n'
        f"    batch_size={batch_size},\n"
        f'    output_path="{output_path}",\n'
        f'    gemm_type="{gemm_type}",\n'
        f"    tp_size={tp_size!r},\n"
        f"    prefix_len={prefix_len!r},\n"
        f"    prefix_lens={prefix_lens_arg!r},\n"
        f"    smoke={smoke!r},\n"
        f")\n"
    )

    # Persist subprocess output to a per-task log so we can inspect failures
    # even when the child dies before stdout is streamed (e.g. OOM kill).
    log_dir = os.path.join(tempfile.gettempdir(), "dsv4_subproc_logs")
    os.makedirs(log_dir, exist_ok=True)
    prefix_label = "sweep" if prefix_lens_arg is not None else str(prefix_len)
    log_path = os.path.join(
        log_dir,
        f"{attn_kind}_{mode}_prefix{prefix_label}_bs{batch_size}_tp{tp_size}_{gemm_type}_gpu{gpu_id}.log",
    )

    def _run_once(nccl_port: int) -> tuple[int, str]:
        attempt_env = env.copy()
        attempt_env["AIC_DSV4_NCCL_PORT"] = str(nccl_port)
        with open(log_path, "wb") as logf:
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                env=attempt_env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            try:
                proc.wait(timeout=3600)  # up to 1 hour per (kind, tp, gemm, bs)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        try:
            with open(log_path, encoding="utf-8", errors="replace") as logf:
                log_text = logf.read()
        except OSError:
            log_text = ""

        # Echo the log so it shows up in the parent's collector log.
        if log_text:
            print(log_text)
        return proc.returncode, log_text

    max_attempts = _DSV4_PORT_RETRIES
    for attempt in range(max_attempts):
        nccl_port = _nccl_port_for_attempt(gpu_id, attempt)
        returncode, log_text = _run_once(nccl_port)

        if returncode == 0:
            return

        is_port_race = "EADDRINUSE" in log_text or "address already in use" in log_text
        if is_port_race and attempt + 1 < max_attempts:
            print(
                f"[dsv4-collector] retrying after NCCL/TCPStore port collision "
                f"on nccl_port={nccl_port} ({attempt + 1}/{max_attempts}); log: {log_path}"
            )
            continue

        raise RuntimeError(
            f"dsv4_{attn_kind}_{mode} subprocess failed for "
            f"(bs={batch_size}, prefix={prefix_label}, tp={tp_size}, gemm={gemm_type}); "
            f"exit={returncode}; log: {log_path}"
        )


def _subprocess_entry(
    *,
    mode: str,
    attn_kind: str,
    model_path: str,
    kv_cache_dtype: str,
    batch_size: int,
    output_path: str,
    gemm_type: str = "bfloat16",
    tp_size: int = 1,
    prefix_len: int = 0,
    prefix_lens: Iterable[int] | None = None,
    smoke: bool = False,
):
    """In-subprocess runner: build model once for fixed bs, sweep valid sl/prefix."""
    _, sl_grid = _expand_grid()
    if smoke:
        sl_grid = [sl for sl in sl_grid if sl in (1, 128)]

    prefix_values = list(prefix_lens) if prefix_lens is not None else [prefix_len]
    if mode == "context" and smoke:
        prefix_values = [p for p in prefix_values if p in (0, 512)]
    elif mode != "context":
        prefix_values = [0]

    max_position_embeddings = _dsv4_max_position_embeddings(model_path) if mode == "context" else None

    seq_lens_by_prefix: dict[int, list[int]] = {}
    for cur_prefix in prefix_values:
        pairs = [
            (bs, sl)
            for bs, sl in _filter_pairs(mode, [batch_size], sl_grid)
            if _is_valid_shape(mode, bs, sl, cur_prefix)
            and (max_position_embeddings is None or cur_prefix + sl <= max_position_embeddings)
        ]
        if not pairs:
            print(f"[dsv4-flash] no valid sl values for mode={mode}, bs={batch_size}, prefix_len={cur_prefix}")
            continue
        seq_lens_by_prefix[cur_prefix] = sorted({sl for _, sl in pairs}, reverse=True)

    if not seq_lens_by_prefix:
        raise RuntimeError(
            f"dsv4-flash mode={mode} bs={batch_size}: no valid prefix/sl values; ok=0 error=0 skip=0 total=0"
        )

    # Let SGLang derive its full/SWA/DSV4 auxiliary pools, then use the live
    # allocator capacities and page size to skip over-capacity cells.
    max_total_tokens = None

    run_dsv4_mla_module(
        model_path=model_path,
        mode=mode,
        attn_kind=attn_kind,
        batch_sizes=[batch_size],
        seq_lens=[],
        kv_cache_dtype=kv_cache_dtype,
        device="cuda:0",
        output_path=output_path,
        perf_filename_prefix="dsv4",
        gemm_type=gemm_type,
        tp_size=tp_size,
        prefix_lens=seq_lens_by_prefix.keys(),
        seq_lens_by_prefix=seq_lens_by_prefix,
        max_total_tokens=max_total_tokens,
    )


def run_dsv4_attn_worker(
    seq_len: int,
    batch_size: int,
    tp_size: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_kind: str,
    attention_backend: str | None = None,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """collect.py-compatible worker — runs ONE (kind, tp, gemm, bs) test case.

    Test case tuple is 9 elements (``perf_filename`` is bound by collect.py
    via OpEntry, NOT in the tuple).  Worker spawns a subprocess that builds
    a fresh ``ModelRunner`` for that bs and sweeps every valid sl internally.

    ``tp_size`` triggers single-process TP simulation in the spawned subprocess
    via ``collect_dsv4_mla_module._tp_load_model_patch``: ColumnParallel /
    RowParallel weights allocate at 1/N shape; FMLA sees h_q=64 (zero-padded).
    """
    del seq_len, attention_backend  # context sweeps prefix/isl inside subprocess.

    if attn_kind not in ATTN_KINDS:
        raise ValueError(f"unknown attn_kind={attn_kind}; expected one of {ATTN_KINDS}")
    if tp_size not in _TP_SIZES:
        raise ValueError(f"unsupported tp_size={tp_size}; expected one of {_TP_SIZES}")

    is_prefill = "context" in perf_filename
    mode = "context" if is_prefill else "generation"
    prefix_lens = list(_PREFIX_LENGTHS) if is_prefill else None

    device_str = str(device)
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    print(
        f"[dsv4-flash {mode}] kind={attn_kind} tp={tp_size} gemm={gemm_type} "
        f"bs={batch_size} prefix_lens={'swept' if prefix_lens is not None else [0]} "
        f"(sl swept internally) GPU={gpu_id}"
    )

    output_path = os.path.dirname(perf_filename) or os.getcwd()
    kv_dtype_sglang = _kv_dtype_db_to_sglang(kv_cache_dtype)
    smoke = "--smoke" in sys.argv

    _run_subprocess(
        mode=mode,
        attn_kind=attn_kind,
        model_path=model_path,
        kv_cache_dtype_sglang=kv_dtype_sglang,
        batch_size=batch_size,
        output_path=output_path,
        gpu_id=gpu_id,
        gemm_type=gemm_type,
        tp_size=tp_size,
        prefix_len=0,
        prefix_lens=prefix_lens,
        smoke=smoke,
    )


# ═══════════════════════════════════════════════════════════════════════
# CLI (manual / smoke test)
# ═══════════════════════════════════════════════════════════════════════


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect DeepSeek-V4-Flash HCA/CSA attention-module latency on SGLang."
    )
    parser.add_argument("--model-path", default=CLI_DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument(
        "--attn-kind",
        choices=ATTN_KINDS,
        default=None,
        help="If unset, sweeps csa/hca in turn.",
    )
    parser.add_argument("--batch-sizes", default=None)
    parser.add_argument("--seq-lens", default=None)
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)
    parser.add_argument(
        "--gemm-type",
        choices=["bfloat16", "fp8_block"],
        default="bfloat16",
        help="Projection-GEMM dispatch path.  fp8_block matches production.",
    )
    parser.add_argument(
        "--tp-sizes",
        default=",".join(str(t) for t in _TP_SIZES),
        help=(
            f"Comma-separated TP sizes to sweep.  Default '{','.join(str(t) for t in _TP_SIZES)}'.  "
            "Each value runs the in-process TP simulation; FMLA always sees "
            "h_q=64 (V4 zero-pads), so any TP power-of-2 in [1, 32] is valid."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.batch_sizes is not None:
        batch_sizes = _parse_int_list(args.batch_sizes)
    else:
        batch_sizes, _ = _expand_grid()
    if args.seq_lens is not None:
        seq_lens = _parse_int_list(args.seq_lens)
    else:
        _, seq_lens = _expand_grid()

    pairs = _filter_pairs(args.mode, batch_sizes, seq_lens)
    _bs_grid = sorted({bs for bs, _ in pairs})
    kinds = [args.attn_kind] if args.attn_kind else list(ATTN_KINDS)
    tp_sizes = _parse_int_list(args.tp_sizes)
    for tp_size in tp_sizes:
        if tp_size not in _TP_SIZES and tp_size not in (16, 32):
            raise ValueError(f"tp_size={tp_size} not in supported set; pick from 1/2/4/8/16/32")

    device_str = str(args.device)
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
    output_path = args.output_path or os.getcwd()
    # Each (kind, tp, bs) is one subprocess that internally sweeps all valid
    # sl values for that bs.  Mirrors the registry-driven path used by
    # collect.py (one test case per (kind, tp, gemm, bs)).
    bs_unique = sorted({bs for bs, _ in pairs})
    for kind in kinds:
        for tp_size in tp_sizes:
            for bs in bs_unique:
                try:
                    _run_subprocess(
                        mode=args.mode,
                        attn_kind=kind,
                        model_path=args.model_path,
                        kv_cache_dtype_sglang=args.kv_cache_dtype,
                        batch_size=bs,
                        output_path=output_path,
                        gpu_id=gpu_id,
                        gemm_type=args.gemm_type,
                        tp_size=tp_size,
                    )
                except Exception:
                    traceback.print_exc()
                    print(f"[dsv4-flash] FAILED kind={kind} tp={tp_size} bs={bs}; continuing")


if __name__ == "__main__":
    main()
