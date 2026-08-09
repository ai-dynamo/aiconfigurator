# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MiniMax-M3 support landed in SGLang v0.5.16 (models/minimax_m3.py +
# layers/attention/minimax_sparse_backend.py + minimax_sparse_ops/); the
# collector-default 0.5.14 pin has no M3. The framework manifest pins the
# msa family to the official v0.5.16 image.
__compat__ = "sglang>=0.5.16"

"""
MSA Module Collector for SGLang — MiniMax-M3 sparse-attention benchmarking.

Profiles the complete MiniMax-M3 sparse-attention module forward pass
(fused qkv+index projections + per-head Gemma QK norm + partial RoPE +
index attention + top-k block selection + sparse GQA + output projection)
at the ModelRunner level, using SGLang's own ServerArgs → ModelRunner →
ScheduleBatch → ForwardBatch pipeline with dummy weights — the same
construction pattern as ``collect_mla_module.py`` (DSA/MLA modules).

Module construction is serving-faithful for the sparse layers (layers 3+
of the real checkpoint): ``MiniMaxM3DecoderLayer`` builds
``MiniMaxM3Attention(is_sparse_attention_layer=True, disable_index_value=True)``
for every layer whose ``sparse_attention_freq``/``sparse_disable_index_value``
entries are set (models/minimax_m3.py:1146-1165@v0.5.16); the collector
truncates those per-layer arrays so its benchmark layer takes exactly that
construction path. Attention dispatch is SGLang's own:
``attn_backend_wrapper`` wraps the platform dense backend with
``MiniMaxHybridAttnBackend``/``MiniMaxSparseAttnBackend`` for every
``is_minimax_sparse`` model (layers/attention/attention_registry.py:272-283
@v0.5.16), and the benchmarked layer id is verified to route to the sparse
backend.

On SM90 (Hopper) SGLang's own M3 server-args override selects
attention_backend=fa3 + page_size=128 and the sparse layers run the Triton
sparse path (arg_groups/overrides.py:521-537@v0.5.16 — "MSA is SM100-only;
sparse attention runs on the Triton path"); on SM100/103 the main sparse
attention step upgrades to the fmha_sm100 MSA kernel when available
(minimax_sparse_backend.py:68-88, minimax_sparse_ops/msa.py:41-56
@v0.5.16). On CC major 12 (SM120/121) the M3 override has NO branch and
SGLang's generic default (flashinfer + page 1) crashes at backend init on
the M3 KV pool, so the collector passes the serving user's own escape
hatch ``attention_backend="triton"`` — the one owner-authorized
explicit-backend exception (2026-08-09; full evidence at the ServerArgs
construction in ``load_model_runner``). Beyond that knob the collector
never pins a kernel: it records what the backend actually selected in
``kernel_source``.

Op names / perf schema are aligned with collector/trtllm/collect_msa_module.py
(msa_context_module / msa_generation_module, architecture
MiniMaxM3ForCausalLM) so perf_database queries work across frameworks.

Usage (inside the pinned lmsysorg/sglang:v0.5.16 image):
    python3 -m collector.sglang.collect_msa_module --mode context --quick \
        --batch-size 2 --seq-len 512 --num-heads 64
    python3 -m collector.sglang.collect_msa_module --mode generation --quick \
        --batch-size 4 --seq-len 2048 --num-heads 64
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import traceback
from importlib.metadata import version as get_version

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from collector.case_generator import (
    get_mla_module_model_specs,
    get_mla_module_precision_specs,
    get_mla_module_sweep_spec,
)
from collector.helper import (
    _resolve_local_model_path,
    benchmark_with_power,
    get_sm_version,
    log_perf,
)

try:
    from collector.sglang.runtime_limits import (
        alloc_prefix_indices,
        kv_pool_capacity_tokens,
        kv_pool_page_size,
        required_kv_alloc_tokens,
        runtime_chunk_size,
        temporarily_chunked_alloc_extend,
    )
except ModuleNotFoundError:  # direct-script import inside the subprocess
    from runtime_limits import (
        alloc_prefix_indices,
        kv_pool_capacity_tokens,
        kv_pool_page_size,
        required_kv_alloc_tokens,
        runtime_chunk_size,
        temporarily_chunked_alloc_extend,
    )

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

MSA_ARCHITECTURE = "MiniMaxM3ForCausalLM"  # AIC-canonical persisted architecture
# SGLang's serving identity for the M3 text decoder. The official
# MiniMaxAI/MiniMax-M3 checkpoint is a VL repo whose text_config declares
# architectures=["MiniMaxM3SparseForCausalLM"] (HF config.json), and SGLang
# keys every M3 code path on that name: is_minimax_sparse
# (configs/model_config.py:136-141@v0.5.16), the model EntryClass
# (models/minimax_m3.py:1690), the server-args override handler
# (arg_groups/overrides.py:465), and the KV-pool / attention-backend wiring
# (mem_cache/kv_cache_configurator.py:842-845,
# layers/attention/attention_registry.py:272-283).
SGLANG_MSA_ARCHITECTURE = "MiniMaxM3SparseForCausalLM"

# Perf-database dtype strings → SGLang ServerArgs kv_cache_dtype values
# (same mapping as collect_mla_module.py).
SGLANG_KV_DTYPE = {"bfloat16": "bfloat16", "fp8": "fp8_e4m3"}

# MiniMax-M3 native GQA geometry (bundled MiniMaxAI--MiniMax-M3_config.json,
# identical to the HF checkpoint text_config): 64 query heads / 4 KV heads /
# 4 index heads. Serving TP sharding:
#   q  heads: num_heads = total // attn_tp_size    (minimax_m3.py:475-477)
#   kv heads: max(1, total_kv // attn_tp_size)     (minimax_m3.py:478-488,
#             replicated when attn_tp_size > total_kv)
#   index heads: idx_head_tp_size = min(attn_tp_size, total_idx);
#             num_idx_heads = total_idx // idx_head_tp_size — i.e.
#             max(1, total_idx // attn_tp_size), replica ranks share one
#             head when attn_tp_size > total_idx (minimax_m3.py:509-521).
# All three are emulated on one GPU by shrinking the config head counts.


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ═══════════════════════════════════════════════════════════════════════
# Precision combos
# ═══════════════════════════════════════════════════════════════════════


def _get_precision_combos(phase: str):
    """Return (compute_dtype, kv_cache_dtype, gemm_type) triples for MSA.

    Starts from the YAML mla_module_sglang combos and keeps only what
    SGLang v0.5.16 MiniMax-M3 actually supports:

    gemm_type — ``bfloat16`` only. The declared MiniMaxAI/MiniMax-M3
      artifact is a BF16 checkpoint (MiniMaxM3ForCausalLM_cases.yaml moe
      framework_quantization allowed_modes=[bfloat16]; no quantization_config
      in the bundled config), so serving runs unquantized projections.
      SGLang's quantized-M3 flow is MXFP8 (checkpoint-driven:
      arg_groups/overrides.py:471-476 quant resolution;
      models/minimax_m3.py:172-203 _FusedQKVIndexProj mxfp8 scales), not the
      DSA-style fp8_block — the YAML ``fp8_block`` combos belong to
      block-FP8 DSA checkpoints and are not declared for this artifact.

    kv_cache_dtype — ``bfloat16`` and ``fp8`` (min_sm 90 from YAML).
      The M3 sparse main-attention kernels accept an fp8 main K/V cache and
      widen it to the Q compute dtype in-kernel
      (sglang/kernels/ops/attention/minimax_sparse/common/utils.py:25-45
      check_sparse_kv_fp8; prefill/topk_sparse.py:278, decode/topk_sparse.py:313
      @v0.5.16); the fmha_sm100 MSA fast path is bf16-only, so an fp8 main
      KV cache stays on the Triton sparse path on every SM
      (minimax_sparse_backend.py:75-88). The index K cache is always the
      model dtype (kv_cache_configurator.py:1246 index_dtype=model_dtype).

    compute_dtype — ``bfloat16`` (checkpoint torch_dtype; the sparse kernels
      assert q.dtype in (bf16, fp16), decode/flash_with_topk_idx.py:778-782).
    """
    combos = []
    for spec in get_mla_module_precision_specs(
        "sglang", phase=phase, sm_version=get_sm_version(), attention_type="msa"
    ):
        if spec.gemm_type != "bfloat16":
            # BF16 artifact: no fp8_block M3 projections exist in serving
            # (see docstring). Declaration-level filter, mirroring
            # collect_mla_module._build_module_test_cases operator-gemm gate.
            continue
        combos.append((spec.compute_dtype, spec.kv_cache_dtype, spec.gemm_type))
    return combos


# ═══════════════════════════════════════════════════════════════════════
# Test-case generation (collect.py entrypoints)
# ═══════════════════════════════════════════════════════════════════════


def _parse_int_list_env(name: str) -> set[int] | None:
    value = os.environ.get(name)
    if not value:
        return None
    return {int(item) for item in value.split(",") if item}


def _filter_shapes_from_env(shapes, *, is_prefill: bool):
    """Targeted-run env pins (dev/healing tool, mirrors the DSA collector's
    AIC_DSA_CONTEXT_* filters — runtime subset selection, never persisted).

    Context: AIC_MSA_CONTEXT_SEQ_LENS / AIC_MSA_CONTEXT_PREFIX_LENS over
    (batch, seq, prefix). Generation: AIC_MSA_GENERATION_BATCH_SIZES /
    AIC_MSA_GENERATION_KV_LENS over (batch, kv, 0).
    """
    if is_prefill:
        seq_set = _parse_int_list_env("AIC_MSA_CONTEXT_SEQ_LENS")
        prefix_set = _parse_int_list_env("AIC_MSA_CONTEXT_PREFIX_LENS")
        if seq_set is None and prefix_set is None:
            return shapes
        filtered = [
            (b, s, p)
            for (b, s, p) in shapes
            if (seq_set is None or s in seq_set) and (prefix_set is None or p in prefix_set)
        ]
    else:
        bs_set = _parse_int_list_env("AIC_MSA_GENERATION_BATCH_SIZES")
        kv_set = _parse_int_list_env("AIC_MSA_GENERATION_KV_LENS")
        if bs_set is None and kv_set is None:
            return shapes
        filtered = [
            (b, kv, p) for (b, kv, p) in shapes if (bs_set is None or b in bs_set) and (kv_set is None or kv in kv_set)
        ]
    print(f"[MSA] Env-filtered shapes: {len(filtered)}/{len(shapes)}")
    return filtered


def _model_max_position_embeddings(model_id: str) -> int | None:
    """Max context (RoPE table size) from the bundled config — the same value
    SGLang uses for the M3 rotary cache (get_rope max_position,
    models/minimax_m3.py:546-552) and default context_length."""
    try:
        config_dir = _resolve_local_model_path(model_id)
        with open(os.path.join(config_dir, "config.json")) as f:
            value = json.load(f).get("max_position_embeddings")
        return int(value) if value else None
    except Exception:
        return None


def _context_shapes(batch_size: int, max_pos: int | None):
    """(seq_len, prefix_len) grid for one context batch size, mirroring the
    trtllm msa getter filters plus the per-request RoPE-table cap."""
    sweep = get_mla_module_sweep_spec("sglang")
    shapes = []
    for s in sweep.context_sequence_lengths:
        if batch_size * s > sweep.context_max_tokens:
            continue
        if (
            sweep.context_large_sequence_min
            and s >= sweep.context_large_sequence_min
            and batch_size > sweep.context_large_sequence_max_batch_size
        ):
            continue
        prefixes = list(sweep.context_prefix_lengths)
        # Per-seq ceiling: land a real point at prefix + seq == max_position
        # (last position = max_pos - 1 still indexes the RoPE table), so the
        # top of the valid range interpolates instead of extrapolating —
        # same rationale as collect_mla_module's DSA ceiling points.
        if max_pos is not None and (max_pos - s) > 0 and (max_pos - s) not in prefixes:
            prefixes.append(max_pos - s)
        for prefix in sorted(set(prefixes)):
            if prefix < 0:
                continue
            # positions run [prefix, prefix + s); beyond max_position they
            # index past the rotary cos/sin cache SGLang sized with
            # max_position_embeddings (models/minimax_m3.py:546-552).
            if max_pos is not None and prefix + s > max_pos:
                continue
            shapes.append((s, prefix))
    return shapes


def _generation_shapes(max_pos: int | None):
    """(batch_size, kv_len) grid for the generation phase."""
    sweep = get_mla_module_sweep_spec("sglang")
    shapes = []
    for b in sweep.generation_batch_sizes:
        for kv in sweep.generation_sequence_lengths:
            if b * kv > sweep.generation_max_tokens:
                continue
            if (
                sweep.generation_large_sequence_min
                and kv >= sweep.generation_large_sequence_min
                and b > sweep.generation_large_sequence_max_batch_size
            ):
                continue
            # decode next-token position == kv must stay < max_position
            if max_pos is not None and kv >= max_pos:
                continue
            shapes.append((b, kv))
    # Ceiling: one bs=1 point at the last valid KV length (position
    # max_pos - 1) so near-max decode interpolates within data — mirrors
    # collect_mla_module's DSA generation ceiling.
    if max_pos is not None and (1, max_pos - 1) not in shapes:
        shapes.append((1, max_pos - 1))
    return shapes


def _build_module_test_cases(mode: str):
    """One case per (model, precision, target TP shard[, context batch]).

    Case tuple layout keeps the trtllm collector's positional prefix
    [seq_len, batch_size, num_heads, kv_cache_dtype, compute_dtype,
     gemm_type, model_path] and appends target_tp_size. Like the sibling
    collect_mla_module.py, each case is a subprocess that sweeps its whole
    (batch, seq[, prefix]) grid internally — an SGLang ModelRunner load
    costs tens of seconds, so one subprocess per shape point is infeasible.
    seq_len is a 0 placeholder; context uses batch_size to shard the
    prefix x seq sweep across GPU workers, generation uses one task per
    (precision, heads) with batch_size 0.
    """
    model_specs = get_mla_module_model_specs(attention_type="msa", backend="sglang")
    sweep = get_mla_module_sweep_spec("sglang")
    cases = []
    for model_spec in model_specs:
        for compute_dtype, kv_dtype, gemm_type in _get_precision_combos(mode):
            for target_tp in sweep.module_tp_sizes:
                if model_spec.native_num_heads % target_tp != 0:
                    continue
                num_heads = model_spec.native_num_heads // target_tp
                if num_heads not in sweep.inner_sweep_head_counts:
                    continue
                batch_sizes = sweep.context_batch_sizes if mode == "context" else [0]
                for batch_size in batch_sizes:
                    cases.append(
                        [
                            0,
                            batch_size,
                            num_heads,
                            kv_dtype,
                            compute_dtype,
                            gemm_type,
                            model_spec.model_path,
                            target_tp,
                        ]
                    )
    return cases


def get_msa_context_module_test_cases():
    """collect.py entrypoint for MSA context module collection."""
    return _build_module_test_cases(mode="context")


def get_msa_generation_module_test_cases():
    """collect.py entrypoint for MSA generation module collection."""
    return _build_module_test_cases(mode="generation")


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════


def _register_minimax_m3_text_config():
    """Make transformers AutoConfig accept model_type="minimax_m3".

    The bundled text-only config keeps the checkpoint's raw schema
    (model_type "minimax_m3"), which neither transformers 5.12 nor SGLang's
    _CONFIG_REGISTRY knows. Serving materialises the M3 text config as a
    generic attribute bag: SGLang's own VL config coerces the checkpoint's
    text_config dict through CONFIG_MAPPING.get(model_type, PretrainedConfig)
    — an unknown model_type "falls back to PretrainedConfig so dict keys
    still become real attributes" (configs/minimax_vl.py:9-23@v0.5.16), and
    the M3 modeling code reads plain attributes only. Registering a bare
    PretrainedConfig alias therefore changes HOW the config object is
    constructed, never which kernel runs (M3 dispatch keys on
    architectures[0] + sparse_attention_config, both preserved verbatim).
    Same shim precedent as collector/trtllm/collect_msa_module.py.
    """
    import transformers

    class _MiniMaxM3TextConfig(transformers.PretrainedConfig):
        model_type = "minimax_m3"

    try:
        transformers.AutoConfig.register("minimax_m3", _MiniMaxM3TextConfig)
    except ValueError as e:  # already registered by an earlier case in this proc
        if "already" not in str(e).lower():
            raise


def _resolve_sglang_model_dir(model_id: str) -> str:
    """Resolve the bundled M3 config and present it under SGLang's identity.

    ``collector.helper._resolve_local_model_path`` materialises the bundled
    AIC config (auto_map stripped). That config carries AIC's canonical
    ``architectures=["MiniMaxM3ForCausalLM"]``; SGLang's text-decoder entry
    class is ``MiniMaxM3SparseForCausalLM`` — exactly what the official
    checkpoint's own text_config.architectures declares (see
    SGLANG_MSA_ARCHITECTURE above). Rewrite only that field so SGLang loads
    the same model class serving uses; every other field stays verbatim.
    """
    base_dir = _resolve_local_model_path(model_id)
    with open(os.path.join(base_dir, "config.json")) as f:
        config = json.load(f)

    if config.get("architectures") not in (
        [MSA_ARCHITECTURE],
        [SGLANG_MSA_ARCHITECTURE],
    ):
        raise ValueError(
            f"bundled config for {model_id} declares architectures="
            f"{config.get('architectures')!r}; expected the MiniMax-M3 text decoder"
        )
    if not config.get("sparse_attention_config"):
        raise ValueError(
            f"model {model_id} has no sparse_attention_config; the MSA module "
            "collector only supports MiniMax-M3 sparse-attention checkpoints"
        )
    config["architectures"] = [SGLANG_MSA_ARCHITECTURE]

    tmp_dir = os.path.join(
        tempfile.gettempdir(),
        f"aic_sglang_msa_config_{model_id.replace('/', '_')}",
    )
    os.makedirs(tmp_dir, exist_ok=True)
    target = os.path.join(tmp_dir, "config.json")
    tmp_target = f"{target}.{os.getpid()}.tmp"
    with open(tmp_target, "w") as f:
        json.dump(config, f)
    os.replace(tmp_target, target)
    return tmp_dir


def _build_model_override_args(config: dict, num_heads: int, target_tp_size: int, num_layers: int) -> dict:
    """Single-GPU emulation of one target-TP rank + benchmark-layer surgery.

    Head shrinking mirrors SGLang's own TP sharding math (citations at the
    top of this file: models/minimax_m3.py:475-488 q/kv, :509-521 index
    heads). The per-layer pattern arrays are truncated so every loaded layer
    takes the sparse construction path of serving layers 3+
    (is_sparse_attention_layer=True, disable_index_value=True —
    models/minimax_m3.py:1146-1165 reading get_minimax_sparse_layer_ids /
    get_minimax_sparse_disable_value_layer_ids,
    configs/model_config.py:158-170). The MLP is made dense
    (moe_layer_freq=0 → MiniMaxM3MLP, models/minimax_m3.py:1167-1193) —
    it is outside the timed attention module; same single-layer surgery as
    the trtllm MSA collector. All sparse scalar fields stay
    checkpoint-verbatim.
    """
    native_q = int(config["num_attention_heads"])
    native_kv = int(config["num_key_value_heads"])
    if native_q % target_tp_size != 0 or native_q // target_tp_size != num_heads:
        raise ValueError(f"num_heads={num_heads} does not equal native {native_q} / target_tp {target_tp_size}")
    local_kv = max(1, native_kv // target_tp_size)

    sparse_cfg = dict(config["sparse_attention_config"])
    native_idx = int(sparse_cfg["sparse_num_index_heads"])
    local_idx = max(1, native_idx // target_tp_size)
    sparse_cfg["sparse_attention_freq"] = [1] * num_layers
    # Serving layers 3+ all disable the index value path
    # (sparse_disable_index_value[3:] == 1 in the checkpoint config); the
    # benchmark layer replicates that flavor.
    sparse_cfg["sparse_disable_index_value"] = [1] * num_layers
    sparse_cfg["sparse_num_index_heads"] = local_idx

    return {
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": local_kv,
        "moe_layer_freq": [0] * num_layers,
        "sparse_attention_config": sparse_cfg,
    }


def _expect(actual, expected, what: str):
    if actual != expected:
        raise RuntimeError(f"{what}={actual!r}, expected {expected!r}")


def _validate_msa_module(model_runner, attn, num_heads: int, target_tp_size: int, head_dim: int):
    """Verify the single-GPU module emulates one target-TP rank of serving.

    Shapes per models/minimax_m3.py@v0.5.16: qkv_proj packs
    (num_heads + 2*num_kv_heads)*head_dim (:523-533), o_proj input
    num_heads*head_dim (:535-544), index_qkv_proj packs
    (num_idx_heads + 1)*idx_head_dim with v_head_size=0 when
    disable_index_value (:554-566), index_o_proj is None (:568-569).
    """
    local_kv = max(1, num_heads // 16)  # native 64q/4kv → GQA ratio 16
    local_idx = max(1, 4 // target_tp_size)
    if not getattr(attn, "is_sparse_attention_layer", False):
        raise RuntimeError("benchmark layer came out dense; sparse_attention_freq surgery did not take")
    if not getattr(attn, "disable_index_value", False):
        raise RuntimeError("benchmark layer keeps the index-value path; serving layers 3+ disable it")
    _expect(int(attn.num_heads), num_heads, "self_attn.num_heads")
    _expect(int(attn.num_kv_heads), local_kv, "self_attn.num_kv_heads")
    _expect(int(attn.num_idx_heads), local_idx, "self_attn.num_idx_heads")
    _expect(int(attn.head_dim), head_dim, "self_attn.head_dim")
    _expect(
        int(attn.qkv_proj.output_size_per_partition),
        (num_heads + 2 * local_kv) * head_dim,
        "qkv_proj.output_size_per_partition",
    )
    _expect(
        int(attn.o_proj.input_size_per_partition),
        num_heads * head_dim,
        "o_proj.input_size_per_partition",
    )
    _expect(
        int(attn.index_qkv_proj.output_size_per_partition),
        (local_idx + 1) * int(attn.idx_head_dim),
        "index_qkv_proj.output_size_per_partition",
    )
    if attn.index_o_proj is not None:
        raise RuntimeError("index_o_proj exists despite disable_index_value=True")

    backend = model_runner.attn_backend
    sparse = getattr(backend, "sparse", None)
    if sparse is None or attn.attn.layer_id not in getattr(backend, "sparse_layer_ids", ()):
        raise RuntimeError(
            "benchmark layer does not route to MiniMaxSparseAttnBackend; "
            "attn_backend_wrapper did not wrap the runner (attention_registry.py:272-283)"
        )


def cleanup_distributed():
    """Clean up SGLang distributed process-group state between subprocesses."""
    import sglang.srt.distributed.parallel_state as parallel_state

    try:
        parallel_state.destroy_model_parallel()
    except Exception:
        pass


def load_model_runner(
    model_path: str,
    num_heads: int,
    kv_cache_dtype: str,
    target_tp_size: int,
    max_total_tokens: int | None,
    chunked_prefill_size: int | None,
    max_running_requests: int | None,
    device: str = "cuda:0",
):
    """Load an SGLang ModelRunner hosting the M3 sparse module (dummy weights).

    Serving-selected defaults are deliberately NOT overridden: the M3
    server-args handler resolves attention_backend / page_size per platform
    (arg_groups/overrides.py:465-537@v0.5.16 — fa3 + page 128 on SM90,
    fa4 + page 128 on SM100) inside ServerArgs.__post_init__
    (server_args.py:2969 _handle_model_specific_adjustments). Sole
    exception: on CC major 12 (SM120/121) that handler has no branch and
    the generic default crashes, so the explicit ``--attention-backend
    triton`` serving knob is passed — see the owner-authorized block at
    the ServerArgs construction below.

    chunked_prefill_size / max_running_requests / max_total_tokens /
    mem_fraction_static are benchmark-harness resource knobs (user-tunable
    ServerArgs, not kernel dispatch): chunked_prefill_size is raised to the
    largest planned one-shot chunk so every planned (bs, seq) is a single
    extend batch — the exact batch serving forms at that setting
    (managers/schedule_batch.py prepare_for_extend); mem_fraction_static is
    pinned because SGLang's heuristic couples it to chunked_prefill_size
    (server_args.py _handle_gpu_memory_settings) and would go negative.
    """
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()

    device_str = str(device)
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    num_layers = int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2"))
    load_format = os.environ.get("SGLANG_LOAD_FORMAT", "dummy")

    sglang_kv_dtype = SGLANG_KV_DTYPE.get(kv_cache_dtype)
    if sglang_kv_dtype is None:
        raise ValueError(f"unsupported kv_cache_dtype {kv_cache_dtype!r} for MSA (see _get_precision_combos)")

    _register_minimax_m3_text_config()
    local_model_path = _resolve_sglang_model_dir(model_path)
    with open(os.path.join(local_model_path, "config.json")) as f:
        raw_config = json.load(f)
    override_args = _build_model_override_args(raw_config, num_heads, target_tp_size, num_layers)

    # ── SM120/121 (CC major 12, e.g. RTX PRO 6000 Blackwell): explicit
    # attention_backend="triton" — owner-authorized exception (2026-08-09).
    #
    # Failure evidence: v0.5.16's M3 server-args handler has no CC-12 branch
    # (arg_groups/overrides.py:465-548 — is_hip() → triton :480-482,
    # elif is_sm100_supported() → fa4 + page 128 :502-511,
    # elif is_sm90_supported() → fa3 + page 128 :520-529; the predicates are
    # capability-major-exact, utils/common.py:296/:286 majors [9]/[10], and
    # is_sm120_supported :281-285 is never consulted by the handler), so
    # ServerArgs falls to the generic MHA default = flashinfer + page_size 1
    # (server_args.py:4880-4931 _get_default_attn_backend — its docstring
    # even notes "trtllm_mha does not support SM120, which will fall back to
    # flashinfer"; overrides.py:1917-1936 _page_size_default). FlashInfer
    # then crashes at backend init on EVERY M3 case: the M3 KV pool
    # MiniMaxSparseKVPool (memory_pool.py:4626) carries no quant_method, so
    # get_kv_cache_quant_method() (memory_pool.py:1661-1676) returns None
    # and flashinfer_backend.py:328 raises AttributeError: 'NoneType' object
    # has no attribute 'resolve_attention_access' — observed as 16/16
    # context + 8/8 generation failures in the SM120 smoke run (2026-08-09).
    # Default-config serving crashes identically, so there is no
    # framework-selected default to record on this platform.
    #
    # The pin is a legitimate serving knob, not a collector invention:
    # --attention-backend triton is the user-facing ServerArgs escape hatch
    # (server_args.py:1398; explicit values are respected via the
    # value-based is_attention_backend_not_set :7503-7508), triton is the
    # backend the M3 handler itself selects on the HIP platform — with no
    # page_size override there either (overrides.py:480-482) — and
    # attn_backend_wrapper wraps ANY dense backend with the MiniMax sparse
    # backend (attention_registry.py:263-283). page_size is deliberately
    # left to the generic default (1): the only page==128 requirement in
    # the M3 sparse path gates use_msa (minimax_sparse_backend.py:81-88),
    # unreachable on CC 12 because msa_available() is SM100-family-only
    # (minimax_sparse_ops/msa.py:41-56); the Triton sparse kernels address
    # per-token req_to_token slots and constrain only block_size_k
    # (kernels/ops/attention/minimax_sparse/prefill/topk_sparse.py:279-292).
    #
    # Dispatch-not-skip: this changes HOW the case is constructed (a
    # ServerArgs knob), never WHETHER it runs; kernel_source still records
    # the sparse path the backend actually selected. Smoke-validated on
    # SM120; SM121 shares the identical source-level gap (major-12
    # predicates) and stays registry-gated (unverified_sms) until validated.
    attention_backend = "triton" if get_sm_version() in (120, 121) else None

    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format=load_format,
        tp_size=1,
        trust_remote_code=True,
        disable_radix_cache=True,
        kv_cache_dtype=sglang_kv_dtype,
        attention_backend=attention_backend,
        max_total_tokens=max_total_tokens,
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=(chunked_prefill_size or 16384),
        mem_fraction_static=0.80,
        max_running_requests=max_running_requests,
        json_model_override_args=json.dumps(override_args),
    )

    print(
        f"ServerArgs resolved: attention_backend={server_args.attention_backend}, "
        f"page_size={server_args.page_size}, kv_cache_dtype={sglang_kv_dtype}, "
        f"chunked_prefill_size={server_args.chunked_prefill_size}, "
        f"max_total_tokens={max_total_tokens}, max_running_requests={max_running_requests}"
    )

    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)

    model_config = ModelConfig.from_server_args(server_args)
    actual_architecture = (model_config.hf_config.architectures or [None])[0]
    if actual_architecture != SGLANG_MSA_ARCHITECTURE:
        raise RuntimeError(
            f"SGLang loaded architecture={actual_architecture!r} for {model_path}, expected {SGLANG_MSA_ARCHITECTURE!r}"
        )

    import random

    nccl_port = 29500 + random.randint(0, 10000) + gpu_id * 100

    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        ps=ParallelState.trivial(gpu_id=gpu_id),
        nccl_port=nccl_port,
        server_args=server_args,
    )
    model_runner.alloc_memory_pool()
    model_runner.init_attention_backends()

    head_dim = int(raw_config.get("head_dim", raw_config["hidden_size"] // raw_config["num_attention_heads"]))
    attn = model_runner.model.model.layers[0].self_attn
    _validate_msa_module(model_runner, attn, num_heads, target_tp_size, head_dim)

    return model_runner


# ═══════════════════════════════════════════════════════════════════════
# ForwardBatch construction + benchmarking
# ═══════════════════════════════════════════════════════════════════════


class PerfLogWriteError(RuntimeError):
    """Fail the subprocess when a measured row was not durably persisted."""


def _forward_ctx(model_runner):
    """Publish SGLang's per-forward control context, exactly as serving does
    around every model forward (model_executor/model_runner.py:1383-1386
    @v0.5.16 _forward_raw). RadixAttention resolves the attention backend
    through it (layers/radix_attention.py get_attn_backend)."""
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    return forward_context(ForwardContext(attn_backend=model_runner.attn_backend))


def _make_reqs(batch_size: int, full_length: int, extend_len: int, prefix_indices=None):
    from array import array

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=str(i),
            origin_input_text="",
            origin_input_ids=list(torch.randint(0, 10000, (full_length,)).tolist()),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.prefix_indices = prefix_indices[i] if prefix_indices is not None else torch.empty((0,), dtype=torch.int64)
        req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
        # v0.5.16 admission contract: extend_range spans the tokens this
        # extend batch processes; prepare_for_extend derives input_ids /
        # seq_lens / prefix_lens from it (managers/schedule_batch.py:
        # 2157-2163 get_fill_ids()[len(prefix_indices):], extend_range.end,
        # extend_range.length).
        req.set_extend_range(full_length - extend_len, full_length)
        req.logprob_start_len = 0
        req.cached_tokens = 0
        req.already_computed = 0
        reqs.append(req)
    return reqs


def _make_schedule_batch(model_runner, reqs):
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.chunk_cache import ChunkCache
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    cache_params = CacheInitParams(
        disable=True,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        page_size=model_runner.token_to_kv_pool_allocator.page_size,
    )
    tree_cache = ChunkCache(cache_params)
    return ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )


def _decode_graph_covered(model_runner, num_tokens: int) -> bool:
    """Whether serving would run this decode batch under a captured CUDA
    graph: cuda_graph_config.decode coverage (backend != disabled; bs list
    or max_bs — resolved per GPU tier by server_args
    _handle_gpu_memory_settings, e.g. 256 on the SM90 validation node at tp<4)."""
    decode_cfg = model_runner.server_args.cuda_graph_config.decode
    if decode_cfg.backend == "disabled":
        return False
    if decode_cfg.bs:
        return int(num_tokens) in set(decode_cfg.bs)
    max_bs = decode_cfg.max_bs or 256
    return 0 < int(num_tokens) <= int(max_bs)


def _decode_topk_variant(sparse_backend, max_seqlen_k: int) -> str:
    """Report which decode top-k kernel the sparse indexer selects — the
    same gate as decode/flash_with_topk_idx.py:822-826@v0.5.16
    (SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX && score blocks <= 4096 &&
    topk <= 32 → JIT radix select; otherwise 2-stage Triton top-k)."""
    from sglang.srt.environ import envs

    score_blocks = _ceil_div(max_seqlen_k, sparse_backend.block_size_k)
    use_radix = (
        envs.SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX.get()
        and score_blocks <= 4096
        and sparse_backend.topk_blocks <= 32
    )
    return "topk_radix" if use_radix else "topk_split"


def _prefill_kernel_source(model_runner) -> str:
    sparse = model_runner.attn_backend.sparse
    main = "msa_fmha_sm100" if sparse.use_msa else "triton_sparse"
    return f"sglang_minimax_prefill_{main}"


def _decode_kernel_source(model_runner, use_graph: bool, actual_kv: int) -> str:
    sparse = model_runner.attn_backend.sparse
    main = "msa_fmha_sm100" if getattr(sparse, "_use_msa_decode", False) else "triton_sparse"
    # Under a captured decode graph the indexer metadata is built with
    # in_capture=True → _max_seqlen_k = max_context_len
    # (minimax_sparse_backend.py:175-178), which is what gates the top-k
    # kernel choice; eager decode uses the live KV length.
    bound = sparse.max_context_len if use_graph else actual_kv
    return f"sglang_minimax_decode_{main}_{_decode_topk_variant(sparse, bound)}"


def _log_msa_row(
    *,
    perf_filename: str,
    model_path: str,
    compute_dtype: str,
    kv_cache_dtype: str,
    gemm_type: str,
    num_heads: int,
    batch_size: int,
    isl: int,
    step: int,
    target_tp_size: int,
    latency_ms: float,
    op_name: str,
    kernel_source: str,
    device_name: str,
    power_stats,
):
    if not log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": MSA_ARCHITECTURE,
                "mla_dtype": compute_dtype,
                "kv_cache_dtype": kv_cache_dtype,
                "gemm_type": gemm_type,
                "num_heads": num_heads,
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": target_tp_size,
                "step": step,
                "latency": f"{latency_ms:.4f}",
            }
        ],
        framework="SGLang",
        version=get_version("sglang"),
        device_name=device_name,
        op_name=op_name,
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=power_stats,
    ):
        raise PerfLogWriteError(f"failed to persist MSA row to {perf_filename}")


def _run_prefill_point(
    model_runner,
    attention_module,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    *,
    num_warmup: int,
    num_iterations: int,
    device: str,
):
    """Benchmark one context (prefill) point; returns (latency_ms, power_stats).

    The module is timed EAGERLY: SGLang prefill runs the attention op
    outside any captured graph segment even under the piecewise prefill
    CUDA graph (RadixAttention extend routes through the
    unified_attention_with_output split op —
    layers/radix_attention.py:161-243, @register_split_op), so eager module
    timing is the serving prefill attention path. Same choice as
    collect_mla_module's DSA prefill.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    prefix_indices = alloc_prefix_indices(model_runner, batch_size, prefix_len)
    full_length = prefix_len + seq_len
    reqs = _make_reqs(
        batch_size,
        full_length,
        seq_len if prefix_len else full_length,
        prefix_indices=prefix_indices,
    )
    batch = _make_schedule_batch(model_runner, reqs)
    with temporarily_chunked_alloc_extend(model_runner, batch_size * seq_len):
        batch.prepare_for_extend()
    # return_hidden_states_before_norm=False mirrors the serving worker
    # (managers/tp_worker.py:267,548@v0.5.16).
    forward_batch = ForwardBatch.init_new(batch, model_runner, return_hidden_states_before_norm=False)
    # Serving metadata init for the eager forward (base
    # AttentionBackend.init_forward_metadata → out_graph + in_graph;
    # MiniMaxHybrid fans out to sparse+dense —
    # minimax_sparse_backend.py:505-507, base_attn_backend.py:48-54).
    with _forward_ctx(model_runner):
        model_runner.attn_backend.init_forward_metadata(forward_batch)

    hidden_states = torch.randn(
        batch_size * seq_len,
        model_runner.model.config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    positions = (
        torch.arange(prefix_len, prefix_len + seq_len, device="cuda")
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
        .flatten()
    )

    def kernel_func():
        with _forward_ctx(model_runner):
            return attention_module(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

    try:
        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=kernel_func,
            num_warmups=num_warmup,
            num_runs=num_iterations,
            repeat_n=1,
            allow_graph_fail=False,
            use_cuda_graph=False,
        ) as results:
            pass
        return results["latency_ms"], results["power_stats"]
    finally:
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
        torch.cuda.empty_cache()


def _run_decode_point(
    model_runner,
    attention_module,
    batch_size: int,
    kv_len: int,
    *,
    num_warmup: int,
    num_iterations: int,
    device: str,
):
    """Benchmark one generation (decode) point.

    Returns (latency_ms, power_stats, used_graph). Decode is timed under a
    captured CUDA graph exactly where serving covers the batch size with a
    decode graph. For graphed points, capture-parity metadata is built with
    in_capture=True — the decode graph runner's pre-capture call
    (model_executor/runner/decode_cuda_graph_runner.py:942@v0.5.16) — which
    fixes the sparse indexer's KV bound at max_context_len
    (minimax_sparse_backend.py:175-178) and therefore the top-k kernel
    choice, matching what serving replays.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    reqs = _make_reqs(batch_size, kv_len, kv_len)
    batch = _make_schedule_batch(model_runner, reqs)
    with temporarily_chunked_alloc_extend(model_runner, batch_size * kv_len):
        batch.prepare_for_extend()
    for req in batch.reqs:
        req.output_ids.append(0)
    batch.prepare_for_decode()
    forward_batch = ForwardBatch.init_new(batch, model_runner, return_hidden_states_before_norm=False)

    use_graph = _decode_graph_covered(model_runner, batch_size)
    with _forward_ctx(model_runner):
        model_runner.attn_backend.init_forward_metadata(forward_batch)
        if use_graph:
            # Capture parity for the SPARSE backend only: the benchmarked
            # layer routes exclusively to MiniMaxSparseAttnBackend
            # (hybrid dispatch by layer id, minimax_sparse_backend.py:585-597),
            # so only its capture-time metadata is consumed. The dense fa3
            # side of the hybrid wrapper needs its full graph-runner state
            # (init_cuda_graph_state buffers) for an in_capture call, which
            # the serving decode graph runner owns — the module bench never
            # executes a dense layer, so its eager metadata is kept.
            model_runner.attn_backend.sparse.init_forward_metadata_out_graph(forward_batch, in_capture=True)

    hidden_states = torch.randn(
        batch_size,
        model_runner.model.config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    positions = torch.full((batch_size,), kv_len, device="cuda", dtype=torch.int64)

    def kernel_func():
        with _forward_ctx(model_runner):
            return attention_module(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

    try:
        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=kernel_func,
            num_warmups=num_warmup,
            num_runs=num_iterations,
            repeat_n=1,
            allow_graph_fail=False,
            use_cuda_graph=use_graph,
        ) as results:
            pass
        return results["latency_ms"], results["power_stats"], use_graph
    finally:
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Subprocess orchestration
# ═══════════════════════════════════════════════════════════════════════


def run_msa_module(
    num_heads: int,
    model_path: str,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    is_prefill: bool,
    gpu_id: int,
    output_path: str | None = None,
    batch_size_filter: int | None = None,
    target_tp_size: int = 1,
    quick_shape: tuple | None = None,
):
    """Run the MSA module benchmark sweep — called inside a subprocess.

    ``quick_shape``: optional single (batch, seq, prefix) point for CLI
    smoke runs; replaces the sweep but keeps every construction path.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    if compute_dtype != "bfloat16":
        raise ValueError(
            f"M3 sparse attention at SGLang v0.5.16 computes in bf16 "
            f"(kernels assert q.dtype in bf16/fp16, decode/flash_with_topk_idx.py:778-782); "
            f"got compute={compute_dtype}"
        )
    if gemm_type != "bfloat16":
        raise ValueError(
            f"MiniMax-M3 is a BF16 artifact; SGLang's quantized-M3 flow is MXFP8, "
            f"not {gemm_type!r} (see _get_precision_combos)"
        )
    if kv_cache_dtype not in SGLANG_KV_DTYPE:
        raise ValueError(f"unsupported kv_cache_dtype {kv_cache_dtype!r}")

    max_pos = _model_max_position_embeddings(model_path)
    phase = "context" if is_prefill else "generation"

    if quick_shape is not None:
        b, s, prefix = quick_shape
        shapes = [(b, s, prefix)] if is_prefill else [(b, s, 0)]
    elif is_prefill:
        if not batch_size_filter or batch_size_filter <= 0:
            raise ValueError("context collection shards by batch_size; got none")
        shapes = [(batch_size_filter, s, p) for (s, p) in _context_shapes(batch_size_filter, max_pos)]
        shapes = _filter_shapes_from_env(shapes, is_prefill=True)
    else:
        shapes = [(b, kv, 0) for (b, kv) in _generation_shapes(max_pos)]
        shapes = _filter_shapes_from_env(shapes, is_prefill=False)

    if not shapes:
        raise RuntimeError(
            f"MSA module {phase} has no runnable shapes; model={model_path}, "
            f"heads={num_heads}, batch_filter={batch_size_filter}"
        )

    # Pre-load allocation-feasibility bound. 128 is the M3 serving page on
    # SM90/SM100 (overrides.py:502-529); on CC 12 the triton escape hatch
    # resolves page_size=1 (see load_model_runner), for which page-128
    # rounding is a strict upper bound — post-load drops re-check against
    # the real kv_pool_page_size(model_runner) below.
    page_size_guess = 128
    max_total_tokens = max(
        required_kv_alloc_tokens(b, s, p, page_size_guess, is_prefill=is_prefill) for (b, s, p) in shapes
    )
    max_total_tokens += max(1024, max_total_tokens // 20)
    chunk_needed = max(b * s for (b, s, _p) in shapes) if is_prefill else None
    if chunk_needed is not None:
        chunk_needed = max(chunk_needed, 8192)
    max_running_requests = max(b for (b, _s, _p) in shapes)

    print(
        f"\n{'=' * 60}\nMSA Module {phase}: model={model_path}, heads={num_heads}, "
        f"target_tp={target_tp_size}, kv={kv_cache_dtype}, gemm={gemm_type}, "
        f"shapes={len(shapes)}, GPU={gpu_id}\n{'=' * 60}"
    )

    cleanup_distributed()
    torch.cuda.empty_cache()

    ok = 0
    failures: list[str] = []
    try:
        model_runner = load_model_runner(
            model_path=model_path,
            num_heads=num_heads,
            kv_cache_dtype=kv_cache_dtype,
            target_tp_size=target_tp_size,
            max_total_tokens=max_total_tokens,
            chunked_prefill_size=chunk_needed,
            max_running_requests=max_running_requests,
            device=device,
        )

        # Post-load drops against real runtime limits (logged, mirroring
        # collect_mla_module): actual KV-pool capacity and the runtime
        # chunk size (serving forms one extend batch per chunk).
        page_size = kv_pool_page_size(model_runner)
        capacity = kv_pool_capacity_tokens(model_runner)
        if capacity is not None:
            before = len(shapes)
            shapes = [
                (b, s, p)
                for (b, s, p) in shapes
                if required_kv_alloc_tokens(b, s, p, page_size, is_prefill=is_prefill) <= capacity
            ]
            if before - len(shapes):
                print(f"[MSA] dropped {before - len(shapes)} shapes beyond KV pool capacity={capacity} tokens")
        if is_prefill:
            chunk = runtime_chunk_size(model_runner)
            before = len(shapes)
            shapes = [(b, s, p) for (b, s, p) in shapes if b * s <= chunk]
            if before - len(shapes):
                print(
                    f"[MSA] dropped {before - len(shapes)} shapes with bs*seq > "
                    f"chunked_prefill_size={chunk} (multi-chunk in serving)"
                )
        if not shapes:
            raise RuntimeError(f"MSA module {phase} has no runnable shapes after runtime checks")

        attention_module = model_runner.model.model.layers[0].self_attn
        device_name = torch.cuda.get_device_name(device)
        perf_filename = _resolve_perf_path(output_path, f"msa_{phase}_module_perf.txt")
        op_name = f"msa_{phase}_module"

        for i, (b, s, p) in enumerate(shapes):
            label = f"b={b}, s={s}, prefix={p}" if is_prefill else f"b={b}, kv={s}"
            print(f"[{i + 1}/{len(shapes)}] {phase} {label}, heads={num_heads}")
            try:
                if is_prefill:
                    latency, power_stats = _run_prefill_point(
                        model_runner,
                        attention_module,
                        b,
                        s,
                        p,
                        num_warmup=8,
                        num_iterations=10,
                        device=device,
                    )
                    kernel_source = _prefill_kernel_source(model_runner)
                    isl, step = s, p
                else:
                    latency, power_stats, used_graph = _run_decode_point(
                        model_runner,
                        attention_module,
                        b,
                        s,
                        num_warmup=8,
                        num_iterations=10,
                        device=device,
                    )
                    kernel_source = _decode_kernel_source(model_runner, used_graph, s)
                    isl, step = 1, s
                _log_msa_row(
                    perf_filename=perf_filename,
                    model_path=model_path,
                    compute_dtype=compute_dtype,
                    kv_cache_dtype=kv_cache_dtype,
                    gemm_type=gemm_type,
                    num_heads=num_heads,
                    batch_size=b,
                    isl=isl,
                    step=step,
                    target_tp_size=target_tp_size,
                    latency_ms=latency,
                    op_name=op_name,
                    kernel_source=kernel_source,
                    device_name=device_name,
                    power_stats=power_stats,
                )
                print(f"  {label}: {latency:.4f} ms [{kernel_source}]")
                ok += 1
            except PerfLogWriteError:
                raise
            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
                print(f"  OOM at {label}: {e}")
                failures.append(f"{label}: OOM: {e}")
                torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                error_str = str(e).lower()
                if "cuda" in error_str and "illegal" in error_str:
                    # CUDA context is poisoned; stop to preserve prior rows.
                    failures.append(f"{label}: CUDA illegal access: {e}")
                    raise RuntimeError(
                        f"CUDA illegal access at {label}; aborting subprocess. "
                        f"progress ok={ok} failed={len(failures)} total={len(shapes)}"
                    ) from e
                failures.append(f"{label}: {type(e).__name__}: {e}")

        summary = f"ok={ok} error={len(failures)} skip=0 total={len(shapes)}"
        print(f"[MSA] {phase} {summary}")
        if ok == 0:
            raise RuntimeError(f"MSA module {phase} persisted no rows; {summary}")
        if failures:
            details = "\n- ".join(failures)
            raise RuntimeError(f"MSA module {phase} failed strict completeness: {summary}; failures:\n- {details}")
    finally:
        cleanup_distributed()
        torch.cuda.empty_cache()
        gc.collect()


def _resolve_perf_path(output_path: str | None, filename: str) -> str:
    if output_path is not None:
        return os.path.join(output_path, filename)
    collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(collector_dir, filename)


def _run_msa_subprocess(
    num_heads: int,
    model_path: str,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    is_prefill: bool,
    gpu_id: int,
    output_path: str | None,
    batch_size_filter: int | None,
    target_tp_size: int,
):
    """Run one MSA sweep in a subprocess with CUDA_VISIBLE_DEVICES isolation
    (same pattern as collect_mla_module._run_mla_subprocess: SGLang's
    ModelRunner/NCCL state cannot be re-initialized in-process)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    phase = "context" if is_prefill else "generation"
    output_repr = f'"{output_path}"' if output_path else "None"
    batch_repr = "None" if batch_size_filter is None else str(batch_size_filter)
    code = (
        f'import sys; sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")\n'
        f"from collect_msa_module import run_msa_module\n"
        f'run_msa_module({num_heads}, "{model_path}", "{kv_cache_dtype}", '
        f'"{compute_dtype}", "{gemm_type}", {is_prefill}, 0, {output_repr}, '
        f"{batch_repr}, {target_tp_size})\n"
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    _timeout_env = os.environ.get("AIC_MLA_MODULE_SUBPROCESS_TIMEOUT_SEC")
    subprocess_timeout = int(_timeout_env) if _timeout_env else None
    try:
        stdout, _ = proc.communicate(timeout=subprocess_timeout)
        if stdout:
            print(stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.wait()
        raise RuntimeError(
            f"MSA module {phase} subprocess timed out after {subprocess_timeout}s "
            f"(heads={num_heads}, model={model_path}, kv={kv_cache_dtype})"
        ) from exc

    if proc.returncode != 0:
        tail = ""
        if stdout:
            lines = stdout.decode("utf-8", errors="replace").strip().splitlines()
            tail = "\n".join(lines[-30:])
        raise RuntimeError(
            f"MSA module {phase} subprocess failed (exit code {proc.returncode})\n"
            f"--- subprocess output (last 30 lines) ---\n{tail}"
        )


def run_msa_module_worker(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    target_tp_size: int = 1,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """Worker-compatible positional wrapper used by collector/collect.py.

    Positional prefix matches the trtllm MSA worker; seq_len is a 0
    placeholder (the subprocess sweeps its grid — see
    _build_module_test_cases). Context cases shard by batch_size.
    """
    device_str = str(device) if not isinstance(device, str) else device
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
    is_prefill = "context" in perf_filename
    batch_size_filter = batch_size if is_prefill and batch_size > 0 else None

    print(f"\n{'=' * 60}")
    print(
        f"MSA Module {'Context' if is_prefill else 'Generation'}: model={model_path}, "
        f"heads={num_heads}, target_tp={target_tp_size}, kv={kv_cache_dtype}, "
        f"compute={compute_dtype}, gemm={gemm_type}, "
        f"batch_filter={batch_size_filter or 'all'}, GPU={gpu_id}"
    )
    print(f"{'=' * 60}")

    output_path = os.path.dirname(perf_filename) or os.getcwd()

    _run_msa_subprocess(
        num_heads=num_heads,
        model_path=model_path,
        kv_cache_dtype=kv_cache_dtype,
        compute_dtype=compute_dtype,
        gemm_type=gemm_type,
        is_prefill=is_prefill,
        gpu_id=gpu_id,
        output_path=output_path,
        batch_size_filter=batch_size_filter,
        target_tp_size=target_tp_size,
    )


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="MiniMax-M3 MSA module-level collector for SGLang")
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument("--model", type=str, default="MiniMaxAI/MiniMax-M3")
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--target-tp-size", type=int, default=None, help="defaults to 64 // num_heads")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument("--kv-cache-dtype", choices=["bfloat16", "fp8"], default="bfloat16")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quick", action="store_true", help="run a single (batch, seq[, prefix]) point in-process")
    args = parser.parse_args()

    gpu_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    target_tp = args.target_tp_size or max(1, 64 // args.num_heads)
    is_prefill = args.mode == "context"

    if args.quick:
        run_msa_module(
            num_heads=args.num_heads,
            model_path=args.model,
            kv_cache_dtype=args.kv_cache_dtype,
            compute_dtype="bfloat16",
            gemm_type="bfloat16",
            is_prefill=is_prefill,
            gpu_id=gpu_id,
            output_path=args.output_path,
            batch_size_filter=None,
            target_tp_size=target_tp,
            quick_shape=(
                args.batch_size or (2 if is_prefill else 4),
                args.seq_len or (512 if is_prefill else 2048),
                args.prefix_len,
            ),
        )
        return

    if is_prefill:
        if not args.batch_size:
            raise SystemExit("--batch-size is required for full context sweeps (shards the prefix grid)")
        batch_filter = args.batch_size
    else:
        batch_filter = None

    run_msa_module(
        num_heads=args.num_heads,
        model_path=args.model,
        kv_cache_dtype=args.kv_cache_dtype,
        compute_dtype="bfloat16",
        gemm_type="bfloat16",
        is_prefill=is_prefill,
        gpu_id=gpu_id,
        output_path=args.output_path,
        batch_size_filter=batch_filter,
        target_tp_size=target_tp,
    )


if __name__ == "__main__":
    main()
