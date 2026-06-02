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

# Requires an SGLang build with DeepSeek-V4 support. Stock lmsysorg/sglang:v*
# images may not include the required deepseek_v4 modules; use a
# deepseek-v4-blackwell/deepseek-v4-grace-blackwell image or matching Dynamo
# sglang-runtime:*deepseek-v4* image.
from __future__ import annotations

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
from importlib.metadata import version as get_version

import torch

# DSV4 local forks default to replacing small patched configs with packaged
# config_backup_small.json.  Suppress so collector's per-kind 1-layer config
# isn't overwritten.
os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")
# Hard-disable DeepGEMM bulk pre-compile.  Each test case touches only a
# few shapes which the bench's own warmup JIT-compiles on first use.
os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
# Older DeepSeek-V4 SGLang containers ship the CUDA kernels as ``sgl-kernel``
# instead of the newer ``sglang-kernel`` package name.  The collector should
# run on those pinned environments, so skip only SGLang's package-name check.
os.environ.setdefault("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "1")

try:
    from helper import benchmark_with_power, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import benchmark_with_power, log_perf

try:
    from collector.sglang.runtime_limits import (
        alloc_prefix_indices as _alloc_prefix_indices,
    )
    from collector.sglang.runtime_limits import (
        kv_pool_capacity_tokens as _kv_pool_capacity_tokens,
    )
    from collector.sglang.runtime_limits import (
        required_kv_tokens,
        required_prefill_extend_tokens,
    )
    from collector.sglang.runtime_limits import (
        runtime_chunk_size as _runtime_chunk_size,
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
        required_kv_tokens,
        required_prefill_extend_tokens,
    )
    from runtime_limits import (
        runtime_chunk_size as _runtime_chunk_size,
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

ATTN_KIND_TO_COMPRESS_RATIO = {
    "csa": 4,
    "hca": 128,
}


CLI_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


_PORTS_PER_GPU = 1000
_DSV4_PORT_RETRIES = 5


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


def _load_model_runner(
    model_path: str,
    *,
    attn_kind: str,
    num_layers: int,
    kv_cache_dtype: str,
    device: str,
    mem_fraction_static: float,
    max_total_tokens: int | None,
    shrink_unused_moe: bool,
    disable_weight_quant: bool,
    gemm_type: str = "bfloat16",
    tp_size: int = 1,
):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()
    torch.cuda.set_device(device)

    local_model_path = _resolve_model_path(
        model_path,
        attn_kind=attn_kind,
        num_layers=num_layers,
        shrink_unused_moe=shrink_unused_moe,
        disable_weight_quant=disable_weight_quant,
        gemm_type=gemm_type,
    )
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    # CUDA_VISIBLE_DEVICES remaps every child to cuda:0; keep the physical GPU
    # id for NCCL port sharding so parallel workers do not collide.
    port_shard = int(os.environ.get("AIC_DSV4_PORT_SHARD", gpu_id))
    nccl_port = int(os.environ.get("AIC_DSV4_NCCL_PORT") or _pick_free_port(port_shard))

    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format=os.environ.get("SGLANG_LOAD_FORMAT", "dummy"),
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=mem_fraction_static,
        disable_radix_cache=True,
        # The module benchmark below captures its own CUDA Graph and fails if
        # capture is not possible.  Keep SGLang's serving-level graph runner off
        # so it does not add unrelated full-model graph state to this collector.
        disable_cuda_graph=True,
        kv_cache_dtype=kv_cache_dtype,
        max_total_tokens=max_total_tokens,
        # The bench sweep includes batch_size up to 1024 (collector's
        # ``_BATCH_SIZES``).  sglang's ``alloc_req_slots`` exposes
        # ``available_size = max_running_requests - 1`` (one slot is
        # reserved internally), so a bs=1024 cell with
        # ``max_running_requests=1024`` raises
        # ``alloc_req_slots runs out of memory: available=1023, num_reqs=1024``.
        # Bump to 1100 for headroom over the largest tested bs.
        max_running_requests=1100,
        max_prefill_tokens=max(max_total_tokens or 4096, 2048),
    )
    # gemm_type controls projection GEMM dispatch.  "fp8_block" → DeepGEMM
    # (matches production V4-Flash-FP8); anything else → cuBLASLt bf16.
    server_args.quantization = "fp8" if gemm_type == "fp8_block" else None
    server_args.enable_piecewise_cuda_graph = False
    server_args.attention_backend = "dsv4"

    print(
        f"[dsv4-collector] model_path {model_path} -> {local_model_path}; "
        f"attn_kind={attn_kind}, backend=dsv4, kv_cache_dtype={kv_cache_dtype}, "
        f"max_total_tokens={max_total_tokens}, shrink_unused_moe={shrink_unused_moe}, "
        f"disable_weight_quant={disable_weight_quant}, gemm_type={gemm_type}, "
        f"quantization={server_args.quantization}, tp_size={tp_size}, nccl_port={nccl_port}"
    )

    _set_envs_and_config(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    with _tp_load_model_patch(tp_size):
        model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=mem_fraction_static,
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
):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    if decode and prefix_len:
        raise ValueError("prefix_len is only supported for context/extend collection")
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
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = seq_len if prefix_len else len(req.fill_ids)
        req.logprob_start_len = 0
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

    prefix_indices = _alloc_prefix_indices(model_runner, batch_size, prefix_len)
    reqs = _make_reqs(
        batch_size,
        seq_len,
        decode=not is_prefill,
        prefix_len=prefix_len,
        prefix_indices=prefix_indices,
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

    with _temporarily_chunked_alloc_extend(model_runner, batch_size * seq_len):
        if is_prefill:
            batch.prepare_for_extend()
        else:
            batch.prepare_for_extend()
            batch.output_ids = torch.randint(0, 10000, (batch_size,), dtype=torch.int64, device="cuda")
            batch.prepare_for_decode()

    if hasattr(batch, "get_model_worker_batch"):
        batch_for_forward = batch.get_model_worker_batch()
    else:
        batch_for_forward = batch
    forward_batch = ForwardBatch.init_new(batch_for_forward, model_runner)
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
    native_heads: int = NATIVE_HEADS,
    step: int | None = None,
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
    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": "DeepseekV4ForCausalLM",
                "mla_dtype": "bfloat16",
                "kv_cache_dtype": kv_cache_dtype,
                "gemm_type": gemm_type,
                "num_heads": native_heads,
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
    )


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
    mem_fraction_static: float = 0.5,
    max_total_tokens: int | None = 4096,
    shrink_unused_moe: bool = True,
    disable_weight_quant: bool = True,
    perf_filename_prefix: str = "dsv4",
    gemm_type: str = "bfloat16",
    tp_size: int = 1,
    prefix_len: int = 0,
    prefix_lens: Iterable[int] | None = None,
    seq_lens_by_prefix: dict[int, list[int]] | None = None,
) -> list[dict[str, float]]:
    is_prefill = mode == "context"
    if prefix_lens is None:
        prefix_values = [prefix_len]
    else:
        prefix_values = list(prefix_lens)
    if not is_prefill:
        prefix_values = [0]
    if any(p > 0 for p in prefix_values) and not is_prefill:
        raise ValueError("prefix_len is only supported for context/extend collection")

    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO[attn_kind]
    if tp_size not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"tp_size must be a power of 2 in [1, 32]; got {tp_size}")
    model_runner = _load_model_runner(
        model_path,
        attn_kind=attn_kind,
        num_layers=max(num_layers, layer_id + 1),
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        mem_fraction_static=mem_fraction_static,
        max_total_tokens=max_total_tokens,
        shrink_unused_moe=shrink_unused_moe,
        disable_weight_quant=disable_weight_quant,
        gemm_type=gemm_type,
        tp_size=tp_size,
    )

    attention_module = model_runner.model.model.layers[layer_id].self_attn
    native_heads = int(getattr(model_runner.model_config.hf_config, "num_attention_heads", NATIVE_HEADS))
    actual_ratio = getattr(attention_module, "compress_ratio", None)
    if actual_ratio != compress_ratio:
        raise RuntimeError(f"target layer compress_ratio mismatch: expected {compress_ratio}, got {actual_ratio}")

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
        default_seq_lens = list(seq_lens)
        kv_capacity = _kv_pool_capacity_tokens(model_runner)
        runtime_chunk = _runtime_chunk_size(model_runner) if is_prefill else None
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

                        def kernel_func():
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
                            native_heads=native_heads,
                            step=cur_prefix if is_prefill else None,
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
                        for _cleanup_label, _cleanup_step in (
                            ("req_to_token_pool.clear", model_runner.req_to_token_pool.clear),
                            ("token_to_kv_pool_allocator.clear", model_runner.token_to_kv_pool_allocator.clear),
                            ("torch.cuda.empty_cache", torch.cuda.empty_cache),
                            ("gc.collect", gc.collect),
                        ):
                            try:
                                _cleanup_step()
                            except Exception as _cleanup_exc:
                                print(
                                    f"[WARN] dsv4-flash {sweep_label} bs={batch_size} sl={seq_len} "
                                    f"prefix={cur_prefix}: cleanup step '{_cleanup_label}' failed with "
                                    f"{type(_cleanup_exc).__name__}; CUDA context likely poisoned, "
                                    "remaining shapes in this sweep may be unreliable"
                                )
    finally:
        try:
            del model_runner
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
    if skipped_shapes:
        skipped_str = ", ".join(f"(bs={b},sl={s},prefix={p},reason={r})" for b, s, p, r in skipped_shapes)
        print(
            f"[WARN] dsv4-flash {sweep_label}: SWEEP SUMMARY - {len(skipped_shapes)} of "
            f"{len(skipped_shapes) + len(results)} shapes failed: {skipped_str}"
        )
    if not results:
        raise RuntimeError(f"dsv4-flash {sweep_label}: all shapes failed")
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
    env.setdefault("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "1")
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
):
    """In-subprocess runner: build model once for fixed bs, sweep valid sl/prefix."""
    _, sl_grid = _expand_grid()
    if "--smoke" in sys.argv:
        sl_grid = [sl for sl in sl_grid if sl in (1, 128)]

    prefix_values = list(prefix_lens) if prefix_lens is not None else [prefix_len]
    if mode == "context" and "--smoke" in sys.argv:
        prefix_values = [p for p in prefix_values if p in (0, 512)]
    elif mode != "context":
        prefix_values = [0]

    seq_lens_by_prefix: dict[int, list[int]] = {}
    for cur_prefix in prefix_values:
        pairs = [
            (bs, sl)
            for bs, sl in _filter_pairs(mode, [batch_size], sl_grid)
            if _is_valid_shape(mode, bs, sl, cur_prefix)
        ]
        if not pairs:
            print(f"[dsv4-flash] no valid sl values for mode={mode}, bs={batch_size}, prefix_len={cur_prefix}")
            continue
        seq_lens_by_prefix[cur_prefix] = sorted({sl for _, sl in pairs}, reverse=True)

    if not seq_lens_by_prefix:
        print(f"[dsv4-flash] no valid prefix/sl values for mode={mode}, bs={batch_size}")
        return

    # The shape filter caps bs * (sl + prefix) at 1M.  Allocate 16x because
    # DSV4 compressed/SWA sub-pools are carved from the full KV pool.
    GLOBAL_MAX_PAIR = 1024 * 1024  # noqa: N806
    max_total_tokens = GLOBAL_MAX_PAIR * 16

    run_dsv4_mla_module(
        model_path=model_path,
        mode=mode,
        attn_kind=attn_kind,
        batch_sizes=[batch_size],
        seq_lens=[],
        kv_cache_dtype=kv_cache_dtype,
        device="cuda:0",
        output_path=output_path,
        mem_fraction_static=0.7,
        max_total_tokens=max_total_tokens,
        perf_filename_prefix="dsv4",
        gemm_type=gemm_type,
        tp_size=tp_size,
        prefix_lens=seq_lens_by_prefix.keys(),
        seq_lens_by_prefix=seq_lens_by_prefix,
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
