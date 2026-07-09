# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 module-level attention collector for vLLM.

Full CSA/HCA attention constructs one ``DeepseekV4Attention`` layer, binds the
DSV4 main/SWA/indexer/compressor metadata and KV caches, then benchmarks the
full attention wrapper forward.  Sparse HCA is also isolated from that module.
Paged MQA logits is collected as a kernel-level benchmark with directly
constructed vLLM DeepGEMM inputs, matching the sparse correction model.

The attention path uses dummy weights. With dummy FP8 block weights, vLLM's
DeepGEMM path may require real checkpoint layouts/scales to be representative;
the collector does not override vLLM's backend selection at import time.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.nn import Parameter
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.deepseek_compressor import CompressorBackend
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.models.deepseek_v4 import DeepseekV4Attention
from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.mla.flashmla_sparse import DeepseekV4FlashMLASparseBackend
from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec
from vllm.version import __version__ as vllm_version

from collector.case_generator import (
    _DSV4_DEFAULT_MODELS,
    _DSV4_MODULE_BATCH_SIZES,
    _DSV4_MODULE_SEQ_LENGTHS,
    _DSV4_MODULE_TP_SIZES,
    _DSV4_SPARSE_BS_LIST,
    _DSV4_SPARSE_ISL_LIST,
    _DSV4_SPARSE_MAX_FULL_S,
    _DSV4_SPARSE_PAST_KV_LIST,
    _DSV4_SPARSE_TP_LIST_ATTN,
    _DSV4_SPARSE_TP_LIST_INDEXER,
    DSV4_ATTN_KINDS,
    DSV4_SPARSE_KERNELS,
    _selected_dsv4_models,
)
from collector.helper import benchmark_with_power, log_perf
from collector.registry_types import PerfFile
from collector.vllm.utils import BatchSpec, create_common_attn_metadata, create_vllm_config, setup_distributed

__compat__ = "vllm>=0.20.1"


DEFAULT_MODEL = _DSV4_DEFAULT_MODELS[0]
ARCHITECTURE = "DeepseekV4ForCausalLM"
ATTN_KIND_TO_COMPRESS_RATIO = {"csa": 4, "hca": 128}
SPARSE_KERNEL_TO_ATTN_KIND = {"paged_mqa_logits": "csa", "hca_attn": "hca"}
SPARSE_KERNEL_TO_OP_NAME = {
    "paged_mqa_logits": "dsv4_paged_mqa_logits_module",
    "hca_attn": "dsv4_hca_attn_module",
}
SPARSE_KERNEL_TO_PERF_FILE = {
    "paged_mqa_logits": PerfFile.DSV4_PAGED_MQA_LOGITS_MODULE,
    "hca_attn": PerfFile.DSV4_HCA_ATTN_MODULE,
}
MODEL_CONFIGS_DIR = (
    Path(__file__).resolve().parents[2].parent / "aiconfigurator-core" / "src" / "aiconfigurator_core" / "model_configs"
)
SUPPORTED_GEMM_TYPES = {"fp8_block"}
DEFAULT_MAX_SEQ_LEN = 65536
MAX_SEQ_LEN = int(os.environ.get("AIC_VLLM_DSV4_MAX_SEQ_LEN", DEFAULT_MAX_SEQ_LEN))
MAX_SPARSE_QUERY_TOKENS = int(os.environ.get("AIC_VLLM_DSV4_SPARSE_MAX_QUERY_TOKENS", "8192"))
MAX_CONTEXT_QUERY_TOKENS = 262144
MAX_GENERATION_KV_TOKENS = 1024 * 1024


def _resolve_perf_path(output_path: str | None, filename: str | None) -> str:
    if filename is None:
        raise ValueError("filename is required")
    if not output_path:
        return filename
    if output_path.endswith(".txt"):
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, filename)


def _read_model_config(model_id: str) -> dict:
    if os.path.isdir(model_id):
        with open(os.path.join(model_id, "config.json"), encoding="utf-8") as f:
            return json.load(f)

    config_file = MODEL_CONFIGS_DIR / f"{model_id.replace('/', '--')}_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"AIC packaged config not found for model_id={model_id!r}: {config_file}")
    with open(config_file, encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def _patched_config_dir(model_id: str, *, compress_ratio: int):
    config = dict(_read_model_config(model_id))
    config.pop("auto_map", None)

    # Transformers does not recognize the shipped deepseek_ref model_type, but
    # vLLM selects the DSV4 model class from architectures. DeepSeekV3Config can
    # carry the extra DSV4 fields.
    config["model_type"] = "deepseek_v3"
    config["architectures"] = [ARCHITECTURE]
    config["num_hidden_layers"] = 1
    config["num_key_value_heads"] = 1
    config["compress_ratios"] = [compress_ratio]

    with tempfile.TemporaryDirectory(prefix=f"aic_vllm_dsv4_{compress_ratio}_{os.getpid()}_") as tmp_dir:
        with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)
        yield tmp_dir


def _init_cuda(device: str) -> None:
    setup_distributed(device)
    torch.cuda.set_device(device)
    try:
        from vllm.v1.worker.workspace import init_workspace_manager

        init_workspace_manager(torch.device(device))
    except (ImportError, RuntimeWarning):
        pass


@contextmanager
def _tp_simulation(tp_size: int):
    if tp_size == 1:
        yield
        return

    import vllm.model_executor.layers.deepseek_v4_attention as dsv4_attn_mod
    import vllm.model_executor.layers.linear as linear_mod
    import vllm.model_executor.models.deepseek_v4 as dsv4_model_mod

    def world_size() -> int:
        return tp_size

    def rank() -> int:
        return 0

    def identity_collective(tensor, *args, **kwargs):
        del args, kwargs
        return tensor

    patches = [
        (linear_mod, "get_tensor_model_parallel_world_size", world_size),
        (linear_mod, "get_tensor_model_parallel_rank", rank),
        (linear_mod, "tensor_model_parallel_all_reduce", identity_collective),
        (linear_mod, "tensor_model_parallel_all_gather", identity_collective),
        (dsv4_model_mod, "get_tensor_model_parallel_world_size", world_size),
        (dsv4_model_mod, "get_tensor_model_parallel_rank", rank),
        (dsv4_attn_mod, "get_tensor_model_parallel_world_size", world_size),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in patches]
    try:
        for module, name, replacement in patches:
            setattr(module, name, replacement)
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def _init_dummy_module_tensors(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            if tensor.is_meta:
                continue
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8):
                tensor.zero_()
            elif tensor.dtype == torch.float32 and "scale" in name:
                tensor.fill_(1.0)
            else:
                tensor.fill_(0.01)


def _process_quantized_weights(module: torch.nn.Module, vllm_config) -> None:
    with set_current_vllm_config(vllm_config):
        for _, child in module.named_modules():
            quant_method = getattr(child, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(child)


def _reshape_wo_a_for_fp8_einsum(attn_module: DeepseekV4Attention) -> None:
    """Match the layout expected by ``deepseek_v4_fp8_einsum``.

    vLLM's real loader materializes ``wo_a`` as grouped BMM data. In this
    dummy module path we construct the layer directly, so the ColumnParallel
    parameter is still 2D. The custom einsum expects:
        weight: [groups, o_lora_rank, heads_per_group * head_dim]
        scale:  [groups, o_lora_rank / 128, input_dim / 128]
    """
    wrapper = attn_module.mla_attn
    wo_a = wrapper.wo_a
    groups = wrapper.n_local_groups
    out_per_group = wrapper.o_lora_rank
    in_per_group = wrapper.n_local_heads // groups * wrapper.head_dim
    if wo_a.weight.dim() == 2:
        wo_a.weight = Parameter(wo_a.weight.view(groups, out_per_group, in_per_group), requires_grad=False)
    if getattr(wo_a, "weight_scale_inv", None) is not None and wo_a.weight_scale_inv.dim() == 2:
        wo_a.weight_scale_inv = Parameter(
            wo_a.weight_scale_inv.view(groups, out_per_group // 128, in_per_group // 128),
            requires_grad=False,
        )


@contextmanager
def _create_dsv4_attention_module(
    *,
    model_path: str,
    attn_kind: str,
    batch_size: int,
    seq_len: int,
    tp_size: int,
    is_context: bool,
    device: str,
    query_len: int | None = None,
):
    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO[attn_kind]
    with _patched_config_dir(model_path, compress_ratio=compress_ratio) as local_model:
        max_model_len = max(seq_len, 4096)
        if query_len is None:
            query_len = seq_len if is_context else 1
        query_tokens = batch_size * query_len
        max_num_batched_tokens = max(query_tokens, 2048)
        block_size = 256
        cache_blocks = _cache_blocks(batch_size, seq_len)

        vllm_config = create_vllm_config(
            model_name=local_model,
            tensor_parallel_size=tp_size,
            # The collector simulates TP shard shapes in one process instead of
            # launching one vLLM executor rank per shard.
            distributed_executor_backend="mp",
            max_model_len=max_model_len,
            block_size=block_size,
            num_gpu_blocks=cache_blocks,
            max_num_seqs=batch_size,
            max_num_batched_tokens=max_num_batched_tokens,
            use_fp8_kv_cache=True,
            trust_remote_code=True,
        )
        vllm_config.cache_config.cache_dtype = "fp8_ds_mla"

        hf_config = vllm_config.model_config.hf_config
        hf_config.num_hidden_layers = 1
        hf_config.compress_ratios = [compress_ratio]
        hf_config.num_key_value_heads = 1

        # DeepSeek rotary construction creates CPU tensors internally; setting the
        # default device only for module construction keeps those tensors aligned.
        torch.set_default_device(device)
        try:
            with set_current_vllm_config(vllm_config), set_default_torch_dtype(vllm_config.model_config.dtype):
                topk_indices_buffer = torch.empty(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    hf_config.index_topk,
                    dtype=torch.int32,
                    device=device,
                )
                aux_streams = [torch.cuda.Stream() for _ in range(3)]
                attn_module = DeepseekV4Attention(
                    vllm_config,
                    prefix="model.layers.0.attn",
                    topk_indices_buffer=topk_indices_buffer,
                    aux_stream_list=aux_streams,
                )
        finally:
            torch.set_default_device("cpu")

        if any(p.is_meta for p in attn_module.parameters()):
            attn_module = attn_module.to_empty(device=torch.device(device))
        else:
            attn_module = attn_module.to(device)
        attn_module.eval()
        attn_module.requires_grad_(False)
        _init_dummy_module_tensors(attn_module)
        _process_quantized_weights(attn_module, vllm_config)
        _reshape_wo_a_for_fp8_einsum(attn_module)
        yield attn_module, vllm_config


def _cache_blocks_for_block_size(batch_size: int, seq_len: int, block_size: int) -> int:
    logical_blocks = batch_size * max(1, math.ceil(seq_len / block_size))
    return max(256, logical_blocks + 64)


def _cache_blocks(batch_size: int, seq_len: int) -> int:
    return _cache_blocks_for_block_size(batch_size, seq_len, 64)


def _make_common_metadata(
    *,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    device: str,
    query_len: int | None = None,
):
    if query_len is None:
        query_len = seq_len if is_context else 1
    batch_spec = BatchSpec(
        seq_lens=[seq_len] * batch_size,
        query_lens=[query_len] * batch_size,
    )
    common = create_common_attn_metadata(
        batch_spec,
        block_size=64,
        device=torch.device(device),
        arange_block_indices=True,
    )
    if getattr(common, "_seq_lens_cpu", None) is not None:
        common.seq_lens_cpu_upper_bound = common._seq_lens_cpu
    common.positions = _positions(batch_size, seq_len, is_context, query_len=query_len, device=device)

    context_len = seq_len - query_len
    slot_values: list[int] = []
    for req_idx in range(batch_size):
        for q_idx in range(query_len):
            pos = context_len + q_idx
            block_id = int(common.block_table_tensor[req_idx, pos // 64].item())
            slot_values.append(block_id * 64 + pos % 64)
    common.slot_mapping.copy_(torch.tensor(slot_values, dtype=torch.int64, device=device))
    return common


def _remap_common_metadata(common, *, block_size: int, device: str):
    seq_lens_cpu = getattr(common, "_seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = common.seq_lens_cpu_upper_bound
    seq_lens = [int(x) for x in seq_lens_cpu.tolist()]
    query_start_loc_cpu = common.query_start_loc_cpu
    query_lens = [int((query_start_loc_cpu[i + 1] - query_start_loc_cpu[i]).item()) for i in range(len(seq_lens))]
    remapped = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=block_size,
        device=torch.device(device),
        arange_block_indices=True,
    )
    if getattr(remapped, "_seq_lens_cpu", None) is not None:
        remapped.seq_lens_cpu_upper_bound = remapped._seq_lens_cpu
    remapped.positions = common.positions
    return remapped


def _allocate_attention_kv_cache(backend, spec, num_blocks: int, cache_dtype: str, *, device: str) -> torch.Tensor:
    shape_block_size = spec.storage_block_size if spec.storage_block_size != spec.block_size else spec.block_size
    shape = backend.get_kv_cache_shape(
        num_blocks,
        shape_block_size,
        spec.num_kv_heads,
        spec.head_size,
        cache_dtype,
    )
    if spec.page_size_padded is None:
        return torch.zeros(shape, dtype=spec.dtype, device=device)

    dtype_size = torch.empty((), dtype=spec.dtype).element_size()
    strides = list(torch.empty(shape).stride())
    strides[0] = spec.page_size_bytes // dtype_size
    return torch.empty_strided(shape, tuple(strides), dtype=spec.dtype, device=device).zero_()


def _build_metadata_and_bind_caches(attn_module: DeepseekV4Attention, vllm_config, common, *, device: str):
    hf_config = vllm_config.model_config.hf_config
    compress_ratio = attn_module.mla_attn.compress_ratio
    cache_blocks = _cache_blocks(int(common.num_reqs), int(common.max_seq_len))
    metadata = {}

    attn_prefix = "model.layers.0.attn"
    if compress_ratio > 1:
        main_spec = MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=hf_config.head_dim,
            dtype=torch.uint8,
            compress_ratio=compress_ratio,
            cache_dtype_str="fp8_ds_mla",
            alignment=576,
            model_version="deepseek_v4",
        )
        metadata[attn_prefix] = DeepseekV4FlashMLASparseBackend.get_builder_cls()(
            main_spec,
            [attn_prefix],
            vllm_config,
            torch.device(device),
        ).build(0, common)

    swa_prefix = f"{attn_prefix}.swa_cache"
    swa_spec = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=hf_config.head_dim,
        dtype=torch.uint8,
        sliding_window=hf_config.sliding_window,
        cache_dtype_str="fp8_ds_mla",
        alignment=576,
        model_version="deepseek_v4",
    )
    metadata[swa_prefix] = DeepseekSparseSWABackend.get_builder_cls()(
        swa_spec,
        [swa_prefix],
        vllm_config,
        torch.device(device),
    ).build(0, common)

    if compress_ratio == 4 and attn_module.indexer is not None:
        indexer_prefix = f"{attn_prefix}.indexer.k_cache"
        indexer_spec = MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=hf_config.index_head_dim + 4,
            dtype=torch.uint8,
            compress_ratio=compress_ratio,
            alignment=576,
            model_version="deepseek_v4",
        )
        metadata[indexer_prefix] = DeepseekV4IndexerBackend.get_builder_cls()(
            indexer_spec,
            [indexer_prefix],
            vllm_config,
            torch.device(device),
        ).build(0, common)

    for compressor in filter(None, [attn_module.mla_attn.compressor, getattr(attn_module.indexer, "compressor", None)]):
        spec = compressor.state_cache.get_kv_cache_spec(vllm_config)
        compressor_common = _remap_common_metadata(common, block_size=spec.block_size, device=device)
        metadata[compressor.state_cache.prefix] = CompressorBackend.get_builder_cls()(
            spec,
            [compressor.state_cache.prefix],
            vllm_config,
            torch.device(device),
        ).build(0, compressor_common)

    static_ctx = vllm_config.compilation_config.static_forward_context
    if compress_ratio > 1:
        static_ctx[attn_prefix].kv_cache = _allocate_attention_kv_cache(
            DeepseekV4FlashMLASparseBackend,
            main_spec,
            cache_blocks,
            "fp8_ds_mla",
            device=device,
        )
    static_ctx[swa_prefix].kv_cache = _allocate_attention_kv_cache(
        DeepseekSparseSWABackend,
        swa_spec,
        cache_blocks,
        "fp8_ds_mla",
        device=device,
    )
    if compress_ratio == 4 and attn_module.indexer is not None:
        static_ctx[f"{attn_prefix}.indexer.k_cache"].kv_cache = _allocate_attention_kv_cache(
            DeepseekV4IndexerBackend,
            indexer_spec,
            cache_blocks,
            "auto",
            device=device,
        )
    for compressor in filter(None, [attn_module.mla_attn.compressor, getattr(attn_module.indexer, "compressor", None)]):
        spec = compressor.state_cache.get_kv_cache_spec(vllm_config)
        state_cache_blocks = _cache_blocks_for_block_size(
            int(common.num_reqs),
            int(common.max_seq_len),
            spec.block_size,
        )
        static_ctx[compressor.state_cache.prefix].kv_cache = torch.zeros(
            CompressorBackend.get_kv_cache_shape(
                state_cache_blocks,
                spec.block_size,
                1,
                spec.head_size,
            ),
            dtype=spec.dtype,
            device=device,
        )

    return metadata


def _positions(
    batch_size: int,
    seq_len: int,
    is_context: bool,
    *,
    device: str,
    query_len: int | None = None,
) -> torch.Tensor:
    if query_len is None:
        query_len = seq_len if is_context else 1
    start_pos = seq_len - query_len
    return (start_pos + torch.arange(query_len, device=device, dtype=torch.long)).repeat(batch_size)


def _bench_attention_shape(
    *,
    model_path: str,
    attn_kind: str,
    mode: str,
    batch_size: int,
    seq_len: int,
    tp_size: int,
    gemm_type: str,
    device: str,
    perf_filename: str,
    warming_up: int,
    test_ite: int,
) -> float | None:
    is_context = mode == "context"
    # Module-generation ``seq_len`` is the decode step / past KV length used
    # as the perf DB key.  vLLM metadata expects the sequence length including
    # the current decode token, so construct with +1 while logging the original
    # step.  This matches the SGLang collector's generation convention.
    metadata_seq_len = seq_len if is_context else seq_len + 1
    with (
        _tp_simulation(tp_size),
        _create_dsv4_attention_module(
            model_path=model_path,
            attn_kind=attn_kind,
            batch_size=batch_size,
            seq_len=metadata_seq_len,
            tp_size=tp_size,
            is_context=is_context,
            device=device,
        ) as (attn_module, vllm_config),
    ):
        common = _make_common_metadata(
            batch_size=batch_size,
            seq_len=metadata_seq_len,
            is_context=is_context,
            device=device,
        )
        metadata = _build_metadata_and_bind_caches(attn_module, vllm_config, common, device=device)

        hf_config = vllm_config.model_config.hf_config
        num_tokens = batch_size * seq_len if is_context else batch_size
        hidden_states = torch.full(
            (num_tokens, hf_config.hidden_size),
            0.01,
            dtype=torch.bfloat16,
            device=device,
        )
        positions = common.positions

        with set_current_vllm_config(vllm_config), set_forward_context(metadata, vllm_config), torch.inference_mode():
            attn_module(positions, hidden_states, None)
            torch.cuda.synchronize()

            def kernel_func(
                attn_module=attn_module,
                positions=positions,
                hidden_states=hidden_states,
            ):
                attn_module(positions, hidden_states, None)

            with benchmark_with_power(
                device=torch.device(device),
                kernel_func=kernel_func,
                num_warmups=warming_up,
                num_runs=test_ite,
                repeat_n=1,
                allow_graph_fail=True,
                use_cuda_graph=True,
            ) as result:
                pass

    latency = float(result["latency_ms"])
    log_perf(
        item_list=[
            {
                "model": model_path,
                "mla_dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "gemm_type": gemm_type,
                "num_heads": int(hf_config.num_attention_heads),
                "batch_size": batch_size,
                "isl": seq_len if is_context else 1,
                "tp_size": tp_size,
                "step": 0 if is_context else seq_len,
                "compress_ratio": ATTN_KIND_TO_COMPRESS_RATIO[attn_kind],
                "latency": f"{latency:.4f}",
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name=f"dsv4_{attn_kind}_{mode}_module",
        kernel_source="vllm_deepseek_v4_flashmla_sparse_dummy",
        perf_filename=perf_filename,
        power_stats=result.get("power_stats"),
    )
    print(f"[vllm-dsv4] {attn_kind} {mode} b={batch_size} s={seq_len} latency={latency:.4f} ms")
    del attn_module, vllm_config, hidden_states, positions
    torch.cuda.empty_cache()
    gc.collect()
    return latency


def _kv_cache_cast_to_fp8_indexer(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    scale = x_amax / 448.0
    x_scaled = (x * (1.0 / scale)).to(torch.float8_e4m3fn)

    out = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    out[:, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(torch.uint8)
    out[:, block_size * head_dim :] = scale.view(num_blocks, block_size).view(torch.uint8)
    return out.view(num_blocks, block_size, 1, head_dim + 4)


def _bench_paged_mqa_logits_kernel(
    *,
    hf_config,
    num_query_rows: int,
    past_kv: int,
    device: str,
    warming_up: int,
    test_ite: int,
) -> dict:
    """Benchmark paged MQA logits with packed M rows across the batch."""
    m = num_query_rows
    full_s = m + past_kv
    full_c4 = max(1, full_s // 4)
    block_kv = 64
    n_heads = int(hf_config.index_n_heads)
    head_dim = int(hf_config.index_head_dim)

    q_bf16 = torch.randn(m, 1, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    q_quant = q_bf16.to(torch.float8_e4m3fn)

    blocks_per_req = (full_c4 + block_kv - 1) // block_kv
    kv_bf16 = torch.randn(
        blocks_per_req,
        block_kv,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = _kv_cache_cast_to_fp8_indexer(kv_bf16)

    weights = torch.randn(m, n_heads, dtype=torch.float32, device=device)
    causal_seq = torch.arange(
        past_kv + 1,
        past_kv + m + 1,
        dtype=torch.int32,
        device=device,
    )
    context_lens = (causal_seq // 4).clamp(min=1).view(m, 1)

    block_table = torch.arange(blocks_per_req, dtype=torch.int32, device=device)
    block_table = block_table.unsqueeze(0).expand(m, blocks_per_req).contiguous()
    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens,
        block_kv,
        num_compute_units(),
    )

    def kernel_func():
        return fp8_fp4_paged_mqa_logits(
            (q_quant, None),
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_metadata,
            max_model_len=full_c4,
            clean_logits=False,
        )

    with torch.inference_mode():
        kernel_func()
        torch.cuda.synchronize()
        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=kernel_func,
            num_warmups=warming_up,
            num_runs=test_ite,
            repeat_n=1,
            allow_graph_fail=False,
            use_cuda_graph=True,
        ) as result:
            pass
    if not result.get("used_cuda_graph", False):
        raise RuntimeError("benchmark_with_power did not use CUDA Graph")
    return result


def _bench_mla_sparse_op(
    *,
    attn_module: DeepseekV4Attention,
    vllm_config,
    metadata,
    positions: torch.Tensor,
    num_tokens: int,
    device: str,
    warming_up: int,
    test_ite: int,
) -> dict:
    sparse_attn = attn_module.mla_attn.mla_attn
    q = torch.full(
        (num_tokens, sparse_attn.padded_heads, sparse_attn.head_dim),
        0.01,
        dtype=torch.bfloat16,
        device=device,
    )
    kv = torch.zeros((num_tokens, sparse_attn.head_dim), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(q)

    with set_current_vllm_config(vllm_config), set_forward_context(metadata, vllm_config), torch.inference_mode():
        sparse_attn(q, kv, positions, output)
        torch.cuda.synchronize()

        def kernel_func():
            sparse_attn(q, kv, positions, output)

        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=kernel_func,
            num_warmups=warming_up,
            num_runs=test_ite,
            repeat_n=1,
            allow_graph_fail=True,
            use_cuda_graph=True,
        ) as result:
            pass
    return result


def _bench_sparse_kernel_shape(
    *,
    model_path: str,
    kernel: str,
    batch_size: int,
    isl: int,
    past_kv: int,
    tp_size: int,
    device: str,
    perf_filename: str,
    warming_up: int,
    test_ite: int,
) -> float | None:
    if kernel not in SPARSE_KERNEL_TO_ATTN_KIND:
        raise ValueError(f"unknown sparse kernel={kernel}")
    full_seq_len = past_kv + isl
    if full_seq_len <= 0:
        return None

    attn_kind = SPARSE_KERNEL_TO_ATTN_KIND[kernel]
    is_context = isl > 1
    query_len = isl
    num_tokens = batch_size * query_len

    if kernel == "paged_mqa_logits":
        hf_config = SimpleNamespace(**_read_model_config(model_path))
        result = _bench_paged_mqa_logits_kernel(
            hf_config=hf_config,
            num_query_rows=num_tokens,
            past_kv=past_kv,
            device=device,
            warming_up=warming_up,
            test_ite=test_ite,
        )
    else:
        with (
            _tp_simulation(tp_size),
            _create_dsv4_attention_module(
                model_path=model_path,
                attn_kind=attn_kind,
                batch_size=batch_size,
                seq_len=full_seq_len,
                tp_size=tp_size,
                is_context=is_context,
                device=device,
                query_len=query_len,
            ) as (attn_module, vllm_config),
        ):
            common = _make_common_metadata(
                batch_size=batch_size,
                seq_len=full_seq_len,
                is_context=is_context,
                query_len=query_len,
                device=device,
            )
            metadata = _build_metadata_and_bind_caches(attn_module, vllm_config, common, device=device)
            positions = common.positions
            hf_config = vllm_config.model_config.hf_config
            result = _bench_mla_sparse_op(
                attn_module=attn_module,
                vllm_config=vllm_config,
                metadata=metadata,
                positions=positions,
                num_tokens=num_tokens,
                device=device,
                warming_up=warming_up,
                test_ite=test_ite,
            )
            del attn_module, vllm_config, metadata, common, positions

    latency = float(result["latency_ms"])
    if kernel == "paged_mqa_logits":
        kernel_source = "vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits"
    else:
        sparse_kernel_name = "flash_mla_with_kvcache" if query_len == 1 else "flash_mla_sparse_fwd"
        kernel_source = f"vllm.DeepseekV4MLAAttention.{sparse_kernel_name}"

    log_perf(
        item_list=[
            {
                "model": model_path,
                "mla_dtype": "fp8_e4m3" if kernel == "paged_mqa_logits" else "bfloat16",
                "kv_cache_dtype": "fp8",
                "gemm_type": "fp8_block",
                "num_heads": int(hf_config.num_attention_heads),
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": tp_size,
                "step": past_kv,
                "compress_ratio": ATTN_KIND_TO_COMPRESS_RATIO[attn_kind],
                "latency": f"{latency:.4f}",
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name=SPARSE_KERNEL_TO_OP_NAME[kernel],
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=result.get("power_stats"),
    )
    print(f"[vllm-dsv4] {kernel} b={batch_size} isl={isl} past_kv={past_kv} tp={tp_size} latency={latency:.4f} ms")
    torch.cuda.empty_cache()
    gc.collect()
    return latency


def _vllm_dsv4_attention_filter_pairs(mode: str, batch_sizes, seq_lens) -> list[tuple[int, int]]:
    """Enumerate vLLM DSV4 attention-module shapes.

    Context allows DSV4 module sweeps up to ``bs * isl <= 262144``
    budget. Generation follows the DSV4 common KV budget, while still allowing
    bs=1 through the configured max sequence length.
    """
    is_context = mode == "context"
    pairs = []
    for bs in batch_sizes:
        for sl in seq_lens:
            if sl > MAX_SEQ_LEN:
                continue
            if is_context:
                if bs * sl > MAX_CONTEXT_QUERY_TOKENS:
                    continue
            else:
                if bs * sl > MAX_GENERATION_KV_TOKENS:
                    continue
                if sl >= 524288 and bs > 1:
                    continue
                if sl >= 262144 and bs > 2:
                    continue
                if sl >= 131072 and bs > 4:
                    continue
                if sl >= 65536 and bs > 8:
                    continue
                if sl >= 32768 and bs > 16:
                    continue
                if sl >= 8192 and bs > 64:
                    continue
            pairs.append((bs, sl))
    return pairs


def _build_dsv4_test_cases(mode: str, attn_kind: str) -> list[list]:
    model_paths = _selected_dsv4_models()
    if not model_paths:
        return []
    seq_lens = [64] if "--smoke" in sys.argv else list(_DSV4_MODULE_SEQ_LENGTHS)
    pairs = _vllm_dsv4_attention_filter_pairs(mode, _DSV4_MODULE_BATCH_SIZES, seq_lens)
    return [
        [seq_len, batch_size, tp_size, "fp8", "bfloat16", "fp8_block", model_path, attn_kind, None]
        for model_path in model_paths
        for tp_size in _DSV4_MODULE_TP_SIZES
        for batch_size, seq_len in pairs
    ]


def _build_vllm_dsv4_sparse_test_cases(kernel: str) -> list[list]:
    if kernel not in DSV4_SPARSE_KERNELS:
        raise ValueError(f"unknown sparse kernel={kernel}")
    model_paths = _selected_dsv4_models()
    if not model_paths:
        return []

    tp_list = _DSV4_SPARSE_TP_LIST_ATTN if kernel == "hca_attn" else _DSV4_SPARSE_TP_LIST_INDEXER
    if "--smoke" in sys.argv:
        smoke_shapes = [
            (1, 1, 8192),
            (1, 64, 8192),
        ]
        return [
            [bs, isl, past_kv, 1, kernel, model_path] for model_path in model_paths for bs, isl, past_kv in smoke_shapes
        ]

    cases: list[list] = []
    for model_path in model_paths:
        for tp_size in tp_list:
            for bs in _DSV4_SPARSE_BS_LIST:
                for isl in _DSV4_SPARSE_ISL_LIST:
                    if bs * isl > MAX_SPARSE_QUERY_TOKENS:
                        continue
                    for past_kv in _DSV4_SPARSE_PAST_KV_LIST:
                        full_s = isl + past_kv
                        if bs * full_s > _DSV4_SPARSE_MAX_FULL_S:
                            continue
                        if kernel == "paged_mqa_logits" and full_s < 4:
                            continue
                        if kernel == "hca_attn" and full_s < 64:
                            continue
                        cases.append([bs, isl, past_kv, tp_size, kernel, model_path])
    return cases


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
) -> None:
    del kv_cache_dtype, compute_dtype, attention_backend
    if attn_kind not in DSV4_ATTN_KINDS:
        raise ValueError(f"unknown attn_kind={attn_kind}")
    if tp_size not in _DSV4_MODULE_TP_SIZES:
        raise ValueError(f"unsupported tp_size={tp_size}")
    if gemm_type not in SUPPORTED_GEMM_TYPES:
        print(f"[vllm-dsv4] gemm_type={gemm_type} not implemented for vLLM DSV4; skipping")
        return

    mode = "context" if "context" in os.path.basename(perf_filename) else "generation"
    _init_cuda(device)
    try:
        _bench_attention_shape(
            model_path=model_path,
            attn_kind=attn_kind,
            mode=mode,
            batch_size=batch_size,
            seq_len=seq_len,
            tp_size=tp_size,
            gemm_type=gemm_type,
            device=device,
            perf_filename=perf_filename,
            warming_up=3 if "--smoke" in sys.argv else 5,
            test_ite=3 if "--smoke" in sys.argv else 10,
        )
    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        print(f"[vllm-dsv4] OOM: {attn_kind} {mode} b={batch_size} s={seq_len}; skipping")
        torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc()
        raise


def run_dsv4_sparse_kernel_worker(
    batch_size: int,
    isl: int,
    past_kv: int,
    tp_size: int,
    kernel: str,
    model_path: str,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    if kernel not in SPARSE_KERNEL_TO_ATTN_KIND:
        raise ValueError(f"unknown sparse kernel={kernel}")
    if tp_size not in _DSV4_MODULE_TP_SIZES:
        raise ValueError(f"unsupported tp_size={tp_size}")
    full_s = isl + past_kv
    if full_s > _DSV4_SPARSE_MAX_FULL_S:
        print(
            f"[vllm-dsv4] skip {kernel} b={batch_size} isl={isl} past_kv={past_kv}: "
            f"full_s={full_s} > max_position_embeddings={_DSV4_SPARSE_MAX_FULL_S}"
        )
        return
    if kernel == "paged_mqa_logits" and full_s < 4:
        print(f"[vllm-dsv4] skip paged_mqa_logits b={batch_size} isl={isl} past_kv={past_kv}: full_s < 4")
        return
    if kernel == "hca_attn" and full_s < 64:
        print(f"[vllm-dsv4] skip hca_attn b={batch_size} isl={isl} past_kv={past_kv}: full_s < 64")
        return

    _init_cuda(device)
    try:
        _bench_sparse_kernel_shape(
            model_path=model_path,
            kernel=kernel,
            batch_size=batch_size,
            isl=isl,
            past_kv=past_kv,
            tp_size=tp_size,
            device=device,
            perf_filename=perf_filename,
            warming_up=3 if "--smoke" in sys.argv else 5,
            test_ite=3 if "--smoke" in sys.argv else 10,
        )
    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        print(f"[vllm-dsv4] OOM: {kernel} b={batch_size} isl={isl} past_kv={past_kv} tp={tp_size}; skipping")
        torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc()
        raise


def get_dsv4_csa_context_test_cases():
    return _build_dsv4_test_cases("context", "csa")


def get_dsv4_hca_context_test_cases():
    return _build_dsv4_test_cases("context", "hca")


def get_dsv4_csa_generation_test_cases():
    return _build_dsv4_test_cases("generation", "csa")


def get_dsv4_hca_generation_test_cases():
    return _build_dsv4_test_cases("generation", "hca")


def get_dsv4_paged_mqa_logits_test_cases():
    return _build_vllm_dsv4_sparse_test_cases("paged_mqa_logits")


def get_dsv4_hca_attn_test_cases():
    return _build_vllm_dsv4_sparse_test_cases("hca_attn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect vLLM DeepSeek-V4 attention module latency.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["context", "generation"], default="context")
    parser.add_argument("--attn-kind", choices=list(DSV4_ATTN_KINDS), default="csa")
    parser.add_argument("--sparse-kernel", choices=list(DSV4_SPARSE_KERNELS), default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--past-kv", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gemm-type", default="fp8_block")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)

    args = parser.parse_args()
    filename = {
        ("csa", "context"): PerfFile.DSV4_CSA_CONTEXT_MODULE.value,
        ("hca", "context"): PerfFile.DSV4_HCA_CONTEXT_MODULE.value,
        ("csa", "generation"): PerfFile.DSV4_CSA_GENERATION_MODULE.value,
        ("hca", "generation"): PerfFile.DSV4_HCA_GENERATION_MODULE.value,
    }[(args.attn_kind, args.mode)]
    if args.gemm_type not in SUPPORTED_GEMM_TYPES:
        raise ValueError(f"unsupported vLLM DSV4 gemm_type={args.gemm_type}; supported={sorted(SUPPORTED_GEMM_TYPES)}")
    _init_cuda(args.device)
    if args.sparse_kernel is not None:
        sparse_filename = SPARSE_KERNEL_TO_PERF_FILE[args.sparse_kernel].value
        perf_filename = _resolve_perf_path(args.output_path, sparse_filename)
        _bench_sparse_kernel_shape(
            model_path=args.model_path,
            kernel=args.sparse_kernel,
            batch_size=args.batch_size,
            isl=args.seq_len,
            past_kv=args.past_kv,
            tp_size=args.tp_size,
            device=args.device,
            perf_filename=perf_filename,
            warming_up=3,
            test_ite=3,
        )
        return

    perf_filename = _resolve_perf_path(args.output_path, filename)
    _bench_attention_shape(
        model_path=args.model_path,
        attn_kind=args.attn_kind,
        mode=args.mode,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        tp_size=args.tp_size,
        gemm_type=args.gemm_type,
        device=args.device,
        perf_filename=perf_filename,
        warming_up=3,
        test_ite=3,
    )


if __name__ == "__main__":
    main()
