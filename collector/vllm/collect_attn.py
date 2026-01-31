# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os

import torch
import vllm

try:
    from vllm.attention.backends.registry import AttentionBackendEnum
except ImportError:
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:
        AttentionBackendEnum = None  # type: ignore
try:
    from vllm.platforms import _Backend as LegacyBackendEnum  # type: ignore
except Exception:
    LegacyBackendEnum = None  # type: ignore
from vllm.platforms import current_platform

try:
    from vllm.utils import is_torch_equal_or_newer
except ImportError:
    from vllm.utils.torch_utils import is_torch_equal_or_newer

from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.version import __version__ as vllm_version

try:
    from vllm.utils import resolve_obj_by_qualname
except ImportError:
    from vllm.utils.import_utils import resolve_obj_by_qualname  # type: ignore

from vllm.config import set_current_vllm_config

from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.vllm.utils import (
    BatchSpec,
    create_and_prepopulate_kv_cache,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
    get_attention_backend,
    with_exit_stack,
)

compatible_version = ["0.11.0", "0.12.0"]


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        # Add float versions for flashinfer
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


compatible_versions = ["0.11.0", "0.12.0"]

# https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends
# support MHA GQA MQA fp16 tensor and float16/fp8 kv cache


@with_exit_stack
def run_attention_torch(
    exit_stack,
    batch_size,
    input_len,
    num_heads,
    num_kv_heads,  # keep same as num_heads for MHA
    head_dim,
    use_fp8_kv_cache,
    is_context_phase,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)

    dtype = torch.float16
    model = os.path.join(os.path.dirname(__file__), "fake_hf_model")
    block_size = 64

    # Let vLLM choose the backend. Handle multiple historical signatures:
    # newest: (... use_mm_prefix=..., use_v1=True/False defaulted to True)
    # mid:    (... use_mm_prefix omitted)
    # old:    (... use_v1 required, no use_mm_prefix)
    try:
        backend = current_platform.get_attn_backend_cls(
            None,
            head_dim,
            dtype,
            kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
            block_size=block_size,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
        )
    except TypeError:
        try:
            backend = current_platform.get_attn_backend_cls(
                None,
                head_dim,
                dtype,
                kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
                block_size=block_size,
                use_mla=False,
                has_sink=False,
                use_sparse=False,
            )
        except TypeError:
            try:
                backend = current_platform.get_attn_backend_cls(
                    None,
                    head_dim,
                    dtype,
                    kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
                    block_size=block_size,
                    use_mla=False,
                    has_sink=False,
                    use_sparse=False,
                    use_v1=True,
                )
            except TypeError:
                from vllm.v1.attention.selector import AttentionSelectorConfig

                attn_selector_config = AttentionSelectorConfig(
                    head_size=head_dim,
                    dtype=dtype,
                    kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
                    block_size=block_size,
                    use_mla=False,
                    has_sink=False,
                    use_sparse=False,
                )
                backend = current_platform.get_attn_backend_cls(None, attn_selector_config)

    backend_name_obj = resolve_obj_by_qualname(backend)
    backend_name_str = backend_name_obj.get_name()
    if AttentionBackendEnum is not None:
        backend_name = AttentionBackendEnum[backend_name_str]
    elif LegacyBackendEnum is not None:
        backend_name = LegacyBackendEnum[backend_name_str]
    else:
        backend_name = backend_name_str

    if is_context_phase:
        batch_spec = BatchSpec(
            seq_lens=[input_len] * batch_size,
            query_lens=[input_len] * batch_size,
        )
    else:
        batch_spec = BatchSpec(
            seq_lens=[input_len] * batch_size,
            query_lens=[1] * batch_size,
        )

    try:
        vllm.utils.torch_utils.set_random_seed(42)
    except AttributeError:
        current_platform.seed_everything(42)

    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        block_size=block_size,
        num_gpu_blocks=8192,
        max_num_seqs=batch_size,
        use_fp8_kv_cache=use_fp8_kv_cache,
    )

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config, use_fp8_kv_cache)

    # Ensure the KV cache has enough blocks for all sequences.
    required_blocks = 1 + sum((s_len + block_size - 1) // block_size for s_len in batch_spec.seq_lens)
    num_blocks = vllm_config.cache_config.num_gpu_blocks or required_blocks
    num_blocks = max(num_blocks, required_blocks)

    # Generate data and compute SDPA reference output
    all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
    k_contexts, v_contexts = [], []

    for i in range(batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        # Generate Q, K, V for the whole sequence
        q = torch.randn(q_len, num_heads, head_dim, dtype=dtype, device=device)
        k_full = torch.randn(s_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        # Inputs for vLLM backends are just the new tokens
        all_q_vllm.append(q)
        all_k_vllm.append(k_full[context_len:])
        all_v_vllm.append(v_full[context_len:])

        # Contextual K/V data used to populate the paged cache
        k_contexts.append(k_full[:context_len])
        v_contexts.append(v_full[:context_len])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    key_vllm = torch.cat(all_k_vllm, dim=0)
    value_vllm = torch.cat(all_v_vllm, dim=0)

    common_attn_metadata = create_common_attn_metadata(batch_spec, vllm_config.cache_config.block_size, device)

    # 3. Simulate Paged KV Cache and a realistic slot_mapping
    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_contexts,
        v_contexts=v_contexts,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=current_platform.fp8_dtype() if use_fp8_kv_cache else dtype,
        device=device,
        num_blocks=num_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=True,
    )

    # Fix backend-specific kv cache layout.
    backend_name_str = backend_name if isinstance(backend_name, str) else backend_name.name

    if backend_name_str == "FLASHINFER":
        kv_cache = kv_cache.transpose(0, 1)

        # For FlashInfer default to HND layout
        kv_cache = kv_cache.transpose(2, 3).contiguous().transpose(2, 3)
        set_kv_cache_layout("HND")

    # Handle special case for FLEX_ATTENTION_SLOW
    actual_backend = backend_name
    use_direct_block_mask = is_torch_equal_or_newer("2.9.0.dev0")
    if backend_name_str == "FLEX_ATTENTION_SLOW":
        if AttentionBackendEnum is not None:
            actual_backend = AttentionBackendEnum.FLEX_ATTENTION
        elif LegacyBackendEnum is not None:
            actual_backend = LegacyBackendEnum.FLEX_ATTENTION
        else:
            actual_backend = backend_name
        use_direct_block_mask = False

    builder_cls, impl_cls = get_attention_backend(actual_backend)
    layer_names = ["placeholder"]

    exit_stack.enter_context(set_current_vllm_config(vllm_config))

    # Mock flashinfer's get_per_layer_parameters if needed
    if backend_name_str == "FLASHINFER":
        import unittest.mock

        from vllm.v1.attention.backends.utils import PerLayerParameters

        def mock_get_per_layer_parameters(vllm_config, layer_names, impl_cls):
            # Return mock parameters for a single layer
            return {
                layer_name: PerLayerParameters(
                    window_left=-1,  # No sliding window
                    logits_soft_cap=0.0,  # No soft cap
                    sm_scale=1.0 / (head_dim**0.5),  # Standard scale
                )
                for layer_name in layer_names
            }

        with unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters", mock_get_per_layer_parameters
        ):
            builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
    else:
        # Build metadata
        builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
        if backend_name_str == "FLEX_ATTENTION":
            builder.direct_build = use_direct_block_mask
        attn_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

    # Instantiate implementation
    sliding_window = vllm_config.model_config.get_sliding_window()
    scale = 1.0 / (head_dim**0.5)
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_dim,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="fp8" if use_fp8_kv_cache else "auto",
    )

    # Create mock layer and output buffer
    mock_layer = MockAttentionLayer(device)
    output = torch.empty_like(query_vllm)

    # Run forward pass

    test_ite = 6
    warm_up = 3

    if use_fp8_kv_cache and backend_name_str == "FLASH_ATTN":
        query_vllm = query_vllm.to(current_platform.fp8_dtype())
        output = output.to(torch.bfloat16)

    def run():
        impl.forward(
            mock_layer,
            query_vllm,
            key_vllm,
            value_vllm,
            kv_cache,
            attn_metadata,
            output=output,
        )

    # Use benchmark_with_power context manager
    with benchmark_with_power(
        device=device,
        kernel_func=run,
        num_warmups=warm_up,
        num_runs=test_ite,
        repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]
    print(f"attn latency: {latency}")

    if is_context_phase:
        isl = input_len
        step = 0
        op_name = "context_attention"
    else:
        isl = 1
        step = input_len
        op_name = "generation_attention"

    kv_cache_dtype_str = "float16" if not use_fp8_kv_cache else "fp8"
    dtype_str = "float16"
    kernel_source = f"vllm_{backend_name_str}".lower()

    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "isl": isl,
                "num_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_dim,
                "beam_width": 1,
                "attn_dtype": dtype_str,
                "kv_cache_dtype": kv_cache_dtype_str,
                "step": step,
                "latency": latency,
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


def get_context_attention_test_cases(if_unit_test=False):
    test_cases = []

    if not if_unit_test:
        b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        s_list = [
            1,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
            10240,
            12288,
            16384,
            262144,
        ]
        n_list = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64]
        n_kv_list = [0, 1, 2, 4, 8]
        # n_kv_list = [64]
    else:
        b_list = [1]
        s_list = [64]
        n_list = [4]
        n_kv_list = [0]

    kv_cache_dtype_list = [False]
    if get_sm_version() > 86:
        kv_cache_dtype_list.append(True)

    # DEBUG
    # print(f"b_list: {b_list}, s_list: {s_list}, n_list: {n_list}, n_kv_list: {n_kv_list}")
    for n in sorted(n_list, reverse=True):
        for s in sorted(s_list, reverse=True):
            for b in sorted(b_list, reverse=True):
                for n_kv in n_kv_list:
                    if n_kv != 0 and (n_kv > n or n % n_kv != 0):
                        continue
                    num_kv_heads = n_kv if n_kv != 0 else n
                    # Only keep self-attention case
                    # if n != num_kv_heads:
                    #    continue
                    if num_kv_heads == n:
                        if b * s > 65536 or b > 128:
                            continue
                    else:
                        if b * s > 131072:
                            continue
                    if b * s * num_kv_heads * 128 * 2 >= 2147483647:
                        continue

                    for is_fp8_kv_cache in kv_cache_dtype_list:
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                num_kv_heads,
                                128,
                                is_fp8_kv_cache,
                                True,
                                "context_attention_perf.txt",
                            ]
                        )

    return test_cases


def get_generation_attention_test_cases():
    test_cases = []

    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # b_list_xqa = [1,2,4,8,16,32,64,128,256,512,1024,2048]
    n_list = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64]
    # n_list_xqa = [4,8,16,32,64,128]
    s_list = [
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
    ]
    n_kv_list = [1, 2, 4, 8]

    kv_cache_dtype_list = [False]
    if get_sm_version() > 86:
        kv_cache_dtype_list.append(True)

    max_bsn = 8192 * 1024
    for n in sorted(n_list, reverse=True):
        b_s_dict = {}
        s_b_dict = {}
        for s in s_list:
            max_b = max_bsn // s // n
            for b in b_list:
                if b > max_b:
                    break
                if s not in s_b_dict:
                    s_b_dict[s] = {b}
                else:
                    s_b_dict[s].add(b)
        for s, b_set in s_b_dict.items():
            if len(b_set) < 4:
                continue
            for b in b_set:
                if b not in b_s_dict:
                    b_s_dict[b] = {s - 1}
                b_s_dict[b].add(s - 1)
        for b, s_list_limited in b_s_dict.items():
            target_s_list = sorted(s_list_limited)
            if b >= 256:
                target_s_list = target_s_list[:-1]
            for n_kv in n_kv_list:
                if n_kv > n or n % n_kv != 0:
                    continue
                for s in target_s_list:
                    for is_fp8_kv_cache in kv_cache_dtype_list:
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                n_kv,
                                128,
                                is_fp8_kv_cache,
                                False,
                                "generation_attention_perf.txt",
                            ]
                        )
    return test_cases


if __name__ == "__main__":
    test_cases = get_context_attention_test_cases()
    test_cases = test_cases[:10]
    for test_case in test_cases:
        print(f"Running context attention test case: {test_case}")
        run_attention_torch(*test_case)

    test_cases = get_generation_attention_test_cases()
    test_cases = test_cases[:10]
    for test_case in test_cases:
        print(f"Running generation attention test case: {test_case}")
        run_attention_torch(*test_case)
