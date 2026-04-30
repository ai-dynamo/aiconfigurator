# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4-Flash module-level attention collector for SGLang.

ONE file containing both:

  1. The bench engine — builds an sglang ``ModelRunner`` for a single
     attn_kind (CSA / HCA) layer and times CUDA-Graph replay of
     ``layer.self_attn(...)`` (Q/KV proj + norm/rope + cache store +
     compressor + C4 indexer/topk for CSA + final FlashMLA).
  2. The registry-facing entrypoints — ``run_dsv4_flash_attn_worker``
     (per-(kind, tp, gemm, bs) test case) which spawns a subprocess that
     internally sweeps every valid sl for that bs.

Test cases (sweep grids + ``get_*_test_cases`` functions) live in
``dsv4_flash_test_cases`` and are re-exported below for registry use.

Manual CLI use::

    python collect_dsv4_flash_attn.py --mode generation --attn-kind csa
    python collect_dsv4_flash_attn.py --mode context --attn-kind hca \
        --batch-sizes 1,4 --seq-lens 128,1024
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import os
import random
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

try:
    from helper import benchmark_with_power, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import benchmark_with_power, log_perf


# Re-export test case generators from the dedicated test_cases module so
# collect.py's registry (``module="collector.sglang.collect_dsv4_flash_attn"``)
# can resolve them via getattr.
try:
    from collector.common_test_cases import (
        _DSV4_FLASH_MODULE_BATCH_SIZES as _BATCH_SIZES,
    )
    from collector.common_test_cases import (
        _DSV4_FLASH_MODULE_SEQ_LENGTHS as _SEQ_LENGTHS,
    )
    from collector.common_test_cases import (
        _DSV4_FLASH_MODULE_TP_SIZES as _TP_SIZES,
    )
    from collector.common_test_cases import (
        DSV4_FLASH_ATTN_KINDS as ATTN_KINDS,
    )
    from collector.common_test_cases import (
        _dsv4_flash_module_filter_pairs as _filter_pairs,
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common_test_cases import (
        _DSV4_FLASH_MODULE_BATCH_SIZES as _BATCH_SIZES,
    )
    from common_test_cases import (
        _DSV4_FLASH_MODULE_SEQ_LENGTHS as _SEQ_LENGTHS,
    )
    from common_test_cases import (
        _DSV4_FLASH_MODULE_TP_SIZES as _TP_SIZES,
    )
    from common_test_cases import (
        DSV4_FLASH_ATTN_KINDS as ATTN_KINDS,
    )
    from common_test_cases import (
        _dsv4_flash_module_filter_pairs as _filter_pairs,
    )


def _expand_grid():
    """Return ``(batch_sizes, seq_lens)`` for the module-level sweep."""
    return list(_BATCH_SIZES), list(_SEQ_LENGTHS)


def get_dsv4_flash_csa_context_test_cases():
    from collector.common_test_cases import get_dsv4_flash_csa_context_test_cases as _impl

    return _impl()


def get_dsv4_flash_csa_generation_test_cases():
    from collector.common_test_cases import get_dsv4_flash_csa_generation_test_cases as _impl

    return _impl()


def get_dsv4_flash_hca_context_test_cases():
    from collector.common_test_cases import get_dsv4_flash_hca_context_test_cases as _impl

    return _impl()


def get_dsv4_flash_hca_generation_test_cases():
    from collector.common_test_cases import get_dsv4_flash_hca_generation_test_cases as _impl

    return _impl()


__all__ = [
    "ATTN_KINDS",
    "_BATCH_SIZES",
    "_SEQ_LENGTHS",
    "_TP_SIZES",
    "_filter_pairs",
    "get_dsv4_flash_csa_context_test_cases",
    "get_dsv4_flash_csa_generation_test_cases",
    "get_dsv4_flash_hca_context_test_cases",
    "get_dsv4_flash_hca_generation_test_cases",
    "run_dsv4_flash_attn_worker",
]


NATIVE_HEADS = 64

ATTN_KIND_TO_COMPRESS_RATIO = {
    "csa": 4,
    "hca": 128,
}


CLI_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def _pick_free_port(gpu_id: int) -> int:
    """Return a free TCP port from a ``gpu_id``-scoped 1000-port range.

    Used as ``nccl_port`` for the per-subprocess torch.distributed
    rendezvous.  Up to 8 collector workers run in parallel, each pinned
    to one GPU.  Partitioning the port space by ``gpu_id`` makes
    cross-worker collision impossible: worker N's candidate set is
    [40000 + N*1000, 40000 + N*1000 + 999], disjoint from every other
    worker's.  The bind / close / subprocess re-bind window inside one
    worker is harmless because no peer worker can race for the same
    port (unrelated system services landing on our specific freed port
    in <1ms is negligibly rare).
    """
    base = 40000 + gpu_id * 1000
    for _ in range(100):
        port = random.randint(base, base + 999)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            s.close()
            continue
        s.close()
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
    server_args.attention_backend = "compressed"

    print(
        f"[dsv4-collector] model_path {model_path} -> {local_model_path}; "
        f"attn_kind={attn_kind}, backend=compressed, kv_cache_dtype={kv_cache_dtype}, "
        f"max_total_tokens={max_total_tokens}, shrink_unused_moe={shrink_unused_moe}, "
        f"disable_weight_quant={disable_weight_quant}, gemm_type={gemm_type}, "
        f"quantization={server_args.quantization}, tp_size={tp_size}"
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
            nccl_port=_pick_free_port(gpu_id),
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


def _make_reqs(batch_size: int, seq_len: int, *, decode: bool):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=str(i),
            origin_input_text="",
            origin_input_ids=list(torch.randint(0, 10000, (seq_len,)).tolist()),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids)
        req.logprob_start_len = 0
        if decode:
            req.cached_tokens = 0
            req.already_computed = 0
        reqs.append(req)
    return reqs


# Fallback chunk size used only when ``server_args.chunked_prefill_size``
# is unavailable.  The real value is read from sglang's server args at
# wrap-time so we follow whatever sglang's GPU-memory-based default would
# pick (8192 on H20/H100, 16384 on B200, 4096 fallback — see
# ``server_args.py`` ``_finalize_resource_allocation``).
_ALLOC_EXTEND_CHUNK_FALLBACK = 8192


def _chunked_alloc_extend(orig_alloc_extend, chunk_size: int = _ALLOC_EXTEND_CHUNK_FALLBACK):
    """Wrap ``alloc_extend`` to chunk large extends in sglang-aligned style.

    Sglang's scheduler chunks any prefill into ≤``chunked_prefill_size`` token
    rounds, with each round invoking ``alloc_extend`` with the **total** token
    count distributed across all in-flight requests.  This wrapper mimics
    that: every ``alloc_extend`` call has ``extend_num_tokens ≤ chunk_size``
    by issuing N ≥ 1 sub-calls, each round advancing every request by a
    proportional share of its remaining extend.  Per-call constexpr stays at
    ≤ chunk_size so Triton's PTX size stays bounded.

    Per-round token distribution:
      * ``per_round_share[i] = extend_per_req[i] // n_rounds`` plus an extra
        +1 in the first ``remainder[i]`` rounds to absorb the remainder.
      * ``sum(chunk_extends_for_round) ≤ chunk_size`` (matches sglang's
        per-round token budget).

    Output index layout: each round's indices come back as
    ``[req0_extends_for_round, req1_extends_for_round, ...]`` (same
    contract as ``alloc_extend_kernel``).  We accumulate per-request lists
    across rounds and concat them in order at the end so the final result
    matches what a single big ``alloc_extend`` would have returned —
    necessary for callers that index into ``out_cache_loc`` per request.
    """

    def wrapped(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens):
        bs = prefix_lens.shape[0]

        # Fast path: small total extend, no chunking needed.
        if extend_num_tokens <= chunk_size or bs == 0:
            return orig_alloc_extend(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens)

        extend_per_req = (seq_lens_cpu - prefix_lens_cpu).tolist()
        # Per-request token budget per round.  ``chunk_size // bs`` ensures
        # ``sum(chunk_extends) ≤ chunk_size`` always — Triton constexpr
        # ``next_pow_2(chunk_total) ≤ next_pow_2(chunk_size) = 8192``.
        # ``max(1, ...)`` handles bs > chunk_size (shouldn't happen with our
        # filter cap, but defensive).
        chunk_size_per_req = max(1, chunk_size // bs)

        cur_prefix = prefix_lens.clone()
        cur_prefix_cpu = prefix_lens_cpu.clone()
        cur_last_loc = last_loc.clone()
        advanced = [0] * bs
        per_req_indices: list[list[torch.Tensor]] = [[] for _ in range(bs)]

        while True:
            # Each round: every request advances by ≤ chunk_size_per_req,
            # capped at its remaining extend.  Sum naturally ≤ chunk_size.
            chunk_extends = [min(chunk_size_per_req, extend_per_req[i] - advanced[i]) for i in range(bs)]
            chunk_total = sum(chunk_extends)
            if chunk_total == 0:
                break

            chunk_tensor = torch.tensor(chunk_extends, dtype=cur_prefix_cpu.dtype)
            new_seq_cpu = cur_prefix_cpu + chunk_tensor
            new_seq = new_seq_cpu.to(seq_lens.device)

            indices = orig_alloc_extend(cur_prefix, cur_prefix_cpu, new_seq, new_seq_cpu, cur_last_loc, chunk_total)
            if indices is None:
                return None

            # Distribute the round's flat ``indices`` back to per-request
            # buckets.  ``alloc_extend_kernel`` writes
            # ``[req0_n, req1_n, ..., req(bs-1)_n]`` in order.
            offset = 0
            for i in range(bs):
                n = chunk_extends[i]
                if n > 0:
                    req_chunk = indices[offset : offset + n]
                    per_req_indices[i].append(req_chunk)
                    offset += n
                    cur_last_loc[i] = req_chunk[-1]
                    advanced[i] += n
            cur_prefix = new_seq
            cur_prefix_cpu = new_seq_cpu

        # Concat each request's pieces, then concat across requests in
        # order — preserves single-call alloc_extend's index layout.
        final = []
        for lst in per_req_indices:
            if lst:
                final.append(torch.cat(lst))
        if not final:
            return torch.empty((0,), dtype=torch.int64, device=prefix_lens.device)
        return torch.cat(final)

    return wrapped


def _build_forward_batch(model_runner, batch_size: int, seq_len: int, *, is_prefill: bool):
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.chunk_cache import ChunkCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    reqs = _make_reqs(batch_size, seq_len, decode=not is_prefill)
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

    # Read sglang's actual ``chunked_prefill_size`` from server_args — this
    # is the same per-call token budget the production scheduler uses, set
    # by sglang based on GPU memory (8192 on H20/H100, 16384 on B200,
    # 4096 fallback).  Our chunked alloc_extend wrapper splits requests
    # into rounds of this size so each Triton constexpr stays bounded.
    server_args = getattr(model_runner, "server_args", None)
    sglang_chunk = getattr(server_args, "chunked_prefill_size", None) if server_args else None
    chunk_size = (
        int(sglang_chunk) if isinstance(sglang_chunk, int) and sglang_chunk > 0 else _ALLOC_EXTEND_CHUNK_FALLBACK
    )
    allocator = model_runner.token_to_kv_pool_allocator
    needs_chunking = batch_size * seq_len > chunk_size
    saved_alloc_extend = None
    if needs_chunking and hasattr(allocator, "alloc_extend"):
        saved_alloc_extend = allocator.alloc_extend
        allocator.alloc_extend = _chunked_alloc_extend(saved_alloc_extend, chunk_size=chunk_size)

    try:
        if is_prefill:
            batch.prepare_for_extend()
        else:
            batch.prepare_for_extend()
            batch.output_ids = torch.randint(0, 10000, (batch_size,), dtype=torch.int64, device="cuda")
            batch.prepare_for_decode()
    finally:
        if saved_alloc_extend is not None:
            allocator.alloc_extend = saved_alloc_extend

    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    model_runner.attn_backend.init_forward_metadata(forward_batch)
    return forward_batch


def _make_inputs(
    model_runner,
    *,
    batch_size: int,
    seq_len: int,
    is_prefill: bool,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = model_runner.model.config.hidden_size
    # freqs_cis is precomputed at shape [max_position_embeddings, rope_dim/2]
    # (deepseek_v4_rope.precompute_freqs_cis: t = torch.arange(seqlen)), so
    # any rope position must be in [0, max_position_embeddings - 1].
    # Production sglang enforces this via clamp_position(seq_lens) = seq_lens-1
    # where seq_lens = past_kv + 1, i.e. decode_pos = past_kv < max_pos.
    # Reject inputs that would index past freqs_cis instead of silently
    # clamping, so the perf log's seq_len always matches the actual rope
    # position used.
    max_pos = getattr(model_runner.model_config.hf_config, "max_position_embeddings", None)
    if is_prefill:
        if max_pos is not None and seq_len > max_pos:
            raise ValueError(f"context seq_len={seq_len} exceeds max_position_embeddings={max_pos}")
        n_tokens = batch_size * seq_len
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1).contiguous().flatten()
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
) -> None:
    # V4-Flash output layout: ONE CSV per (attn_kind, mode) — 3 kinds x 2
    # modes = 6 files total, regardless of how many (tp_size, gemm_type)
    # subprocesses run.  Within each file, rows are disambiguated by the
    # ``tp_size``, ``gemm_type``, ``batch_size``, ``isl`` columns.
    # ``log_perf`` is file-locked so concurrent appends from different
    # subprocesses to the same kind+mode file are safe.
    # Non-V4-Flash callers (legacy ``dsv4`` MLA module) still use the old
    # per-(prefix, kind) filename layout to avoid behavior breaks.
    if perf_filename_prefix.startswith("dsv4_flash"):
        consolidated_filename = f"dsv4_flash_{attn_kind}_{mode}_module_perf.txt"
    else:
        consolidated_filename = f"{perf_filename_prefix}_{attn_kind}_{mode}_module_perf.txt"
    perf_filename = _resolve_perf_path(output_path, consolidated_filename)
    is_prefill = mode == "context"
    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": "DeepseekV4ForCausalLM",
                "mla_dtype": "bfloat16",
                "kv_cache_dtype": kv_cache_dtype,
                "gemm_type": gemm_type,
                "num_heads": NATIVE_HEADS,
                "batch_size": batch_size,
                "isl": seq_len if is_prefill else 1,
                "tp_size": tp_size,
                "step": 0 if is_prefill else seq_len,
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
) -> list[dict[str, float]]:
    is_prefill = mode == "context"
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
    actual_ratio = getattr(attention_module, "compress_ratio", None)
    if actual_ratio != compress_ratio:
        raise RuntimeError(f"target layer compress_ratio mismatch: expected {compress_ratio}, got {actual_ratio}")

    print(f"[dsv4-collector] layer={layer_id}, attn_kind={attn_kind}, compress_ratio={actual_ratio}, mode={mode}")

    version = get_version("sglang")
    device_name = torch.cuda.get_device_name(device)
    results = []
    try:
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                print(f"\n{mode}: batch_size={batch_size}, seq_len={seq_len}")
                try:
                    forward_batch = _build_forward_batch(model_runner, batch_size, seq_len, is_prefill=is_prefill)
                    hidden_states, positions = _make_inputs(
                        model_runner,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        is_prefill=is_prefill,
                        device=device,
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
                    )
                    stats.update(
                        {
                            "batch_size": batch_size,
                            "seq_len": seq_len,
                            "compress_ratio": compress_ratio,
                        }
                    )
                    results.append(stats)
                except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                    print(f"  OOM: batch_size={batch_size}, seq_len={seq_len}; skipping")
                    torch.cuda.empty_cache()
                except Exception:
                    traceback.print_exc()
                    print("  failed; skipping this shape")
                finally:
                    model_runner.req_to_token_pool.clear()
                    model_runner.token_to_kv_pool_allocator.clear()
                    torch.cuda.empty_cache()
                    gc.collect()
    finally:
        del model_runner
        torch.cuda.empty_cache()
        gc.collect()
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
):
    """Run one (attn_kind, tp, gemm, bs) subprocess that sweeps all valid sl.

    Builds one ``ModelRunner`` sized for ``(bs, max_sl_for_this_bs)`` and
    iterates every valid sl for that bs.  Per-sl crash isolation is
    handled by ``run_dsv4_mla_module``'s try/except per forward.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")
    env.setdefault("SGLANG_LOAD_FORMAT", "dummy")
    # Hard-disable DeepGEMM bulk pre-compile.  First sl in this sweep
    # triggers runtime lazy JIT for the (M, N, K) shapes it needs;
    # subsequent sl within the same subprocess hit in-memory cache.
    env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"

    code = (
        f'import sys; sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")\n'
        f"from collect_dsv4_flash_attn import _subprocess_entry\n"
        f"_subprocess_entry(\n"
        f'    mode="{mode}",\n'
        f'    attn_kind="{attn_kind}",\n'
        f'    model_path="{model_path}",\n'
        f'    kv_cache_dtype="{kv_cache_dtype_sglang}",\n'
        f"    batch_size={batch_size},\n"
        f'    output_path="{output_path}",\n'
        f'    gemm_type="{gemm_type}",\n'
        f"    tp_size={tp_size!r},\n"
        f")\n"
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    try:
        stdout, _ = proc.communicate(timeout=3600)  # up to 1 hour per (kind, tp, gemm, bs)
        if stdout:
            print(stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"dsv4_flash_{attn_kind}_{mode} subprocess failed for "
            f"(bs={batch_size}, tp={tp_size}, gemm={gemm_type}); exit={proc.returncode}"
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
):
    """In-subprocess runner: build model once for fixed bs, sweep all valid sl.

    The KV pool is sized for ``(bs, max_sl_for_this_bs)`` so every sl
    forward reuses the same allocator without re-init.
    """
    bs_grid, sl_grid = _expand_grid()
    pairs = _filter_pairs(mode, [batch_size], sl_grid)
    if not pairs:
        print(f"[dsv4-flash] no valid sl values for mode={mode}, bs={batch_size}")
        return
    # Sort sl DESCENDING — start from the largest case so:
    #   1. OOM fails fast (don't waste time on small sl before discovering
    #      max_sl can't fit)
    #   2. CUDA / DeepGEMM workspace grows monotonically; allocator settles
    #      at max state immediately, smaller sl never triggers further
    #      grow / re-alloc.
    sl_for_bs = sorted({sl for _, sl in pairs}, reverse=True)

    # Both kinds (csa/hca) write the swa_k_cache sub-pool — see
    # ``deepseek_v4_backend_radix.py`` line ~1020.  Sub-pool ratios out of
    # ``max_total`` (page-256 aligned, not exact):
    #   swa_pool / max_total: ~1/10 at max_total≥100k, ~1/16 at smaller
    # so we need ``max_total ≥ ~16 * max(bs*sl)``.
    #
    # The global ceiling on bs*sl in our sweep is bs=1024 x sl=1024 = 1M
    # (other (bs, sl) pairs all stay ≤ 1M).  Hard-code max_total off this
    # global cap so every subprocess gets the same KV pool, regardless of
    # which (bs, sl_list) it owns.  H20 80GB easily fits 16M tokens of
    # single-layer fp8 KV.
    GLOBAL_MAX_PAIR = 1024 * 1024  # noqa: N806
    max_total_tokens = GLOBAL_MAX_PAIR * 16

    run_dsv4_mla_module(
        model_path=model_path,
        mode=mode,
        attn_kind=attn_kind,
        batch_sizes=[batch_size],
        seq_lens=sl_for_bs,
        kv_cache_dtype=kv_cache_dtype,
        device="cuda:0",
        output_path=output_path,
        mem_fraction_static=0.7,
        max_total_tokens=max_total_tokens,
        perf_filename_prefix="dsv4_flash",
        gemm_type=gemm_type,
        tp_size=tp_size,
    )


def run_dsv4_flash_attn_worker(
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
    del seq_len, attention_backend  # placeholders; sl swept inside subprocess

    if attn_kind not in ATTN_KINDS:
        raise ValueError(f"unknown attn_kind={attn_kind}; expected one of {ATTN_KINDS}")
    if tp_size not in _TP_SIZES:
        raise ValueError(f"unsupported tp_size={tp_size}; expected one of {_TP_SIZES}")

    is_prefill = "context" in perf_filename
    mode = "context" if is_prefill else "generation"

    device_str = str(device)
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    print(
        f"[dsv4-flash {mode}] kind={attn_kind} tp={tp_size} gemm={gemm_type} "
        f"bs={batch_size} (sl swept internally) GPU={gpu_id}"
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
