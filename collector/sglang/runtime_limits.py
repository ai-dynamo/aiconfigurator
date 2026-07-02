# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SGLang collector helpers for runtime token limits and KV allocation."""

import importlib
import inspect
from collections.abc import Callable
from contextlib import contextmanager

import torch

DSA_INDEXER_TOTAL_KV_TOKEN_LIMIT = 1 << 25


def kv_pool_capacity_tokens(model_runner) -> int | None:
    """Best-effort token capacity from the actual SGLang KV allocator."""
    allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
    if allocator is None:
        return None
    # Hybrid SWA allocators report min(full, swa) from available_size().
    # Logical sequence capacity follows the full-attention pool instead.
    if hasattr(allocator, "full_available_size"):
        try:
            return int(allocator.full_available_size())
        except Exception:
            return None
    if hasattr(allocator, "available_size"):
        try:
            return int(allocator.available_size())
        except Exception:
            return None
    for attr in ("max_total_num_tokens", "size", "pool_size"):
        value = getattr(allocator, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def swa_kv_pool_capacity_tokens(model_runner) -> int | None:
    """Best-effort token capacity of SGLang's hybrid SWA KV pool."""
    allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
    available_size = getattr(allocator, "swa_available_size", None)
    if not callable(available_size):
        return None
    try:
        return int(available_size())
    except Exception:
        return None


def required_kv_tokens(batch_size: int, seq_len: int, prefix_len: int, *, is_prefill: bool) -> int:
    """Return total KV tokens needed by a collector case."""
    return batch_size * (seq_len + prefix_len) if is_prefill else batch_size * seq_len


def kv_pool_page_size(model_runner) -> int:
    """Page size of the SGLang KV allocator (1 for token-level allocators)."""
    allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
    page = getattr(allocator, "page_size", None)
    return int(page) if isinstance(page, int) and page > 0 else 1


def required_kv_alloc_tokens(
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    page_size: int,
    *,
    is_prefill: bool,
) -> int:
    """KV tokens SGLang's paged ``alloc_extend`` must find free for this case.

    The naive :func:`required_kv_tokens` (``bs*(sl+prefix)``) only counts logical
    KV tokens, but SGLang's PAGED allocator rounds EACH request's KV span up to a
    full page independently. With a large page (e.g. 256) and many requests, even
    tiny ``seq_len`` forces ``batch_size`` whole pages: a bs=512 / sl=1 decode
    needs 512 pages = ``512 * 256`` slots, which can exceed the KV pool even
    though ``bs*sl`` is tiny — so ``alloc_extend`` fails with "Prefill out of
    memory" while the naive check passes. This returns that true paged
    requirement so such shapes are skipped BEFORE launching. With ``page_size==1``
    it equals the logical span plus the fresh decode token (a no-op for
    token-level page rounding).

    Decode also allocates one fresh token per request. When ``seq_len`` lands
    exactly on a page boundary, that token requires one additional page for
    every request.

    NOTE: this is the real allocation size only, NOT SGLang's eviction
    over-estimate (``extend_num_tokens + bs*page_size``). The collector clears
    the pool between shapes, so the cache is empty and nothing needs eviction;
    adding that slack would over-skip valid shapes (e.g. bs=256/sl<=256).
    """
    if batch_size <= 0:
        return 0
    per_req = required_kv_tokens(batch_size, seq_len, prefix_len, is_prefill=is_prefill) // batch_size
    if not is_prefill:
        per_req += 1
    pages_per_req = (per_req + page_size - 1) // page_size
    return batch_size * pages_per_req * page_size


def required_swa_kv_alloc_tokens(
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    page_size: int,
    window_size: int,
    *,
    is_prefill: bool,
) -> int:
    """Page-rounded hybrid SWA allocation needed by one collector shape."""
    if batch_size <= 0:
        return 0
    past_len = prefix_len if is_prefill else seq_len
    fresh_len = seq_len if is_prefill else 1
    window_start = max(0, past_len - window_size)
    window_start = (window_start // page_size) * page_size
    swa_tail_len = past_len - window_start
    rounded_tail = ((swa_tail_len + page_size - 1) // page_size) * page_size
    rounded_past = ((past_len + page_size - 1) // page_size) * page_size
    rounded_total = ((past_len + fresh_len + page_size - 1) // page_size) * page_size
    return batch_size * (rounded_tail + rounded_total - rounded_past)


def dsa_indexer_workspace_bytes(
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    num_heads: int,
    target_tp_size: int,
    *,
    is_prefill: bool,
) -> int:
    """Estimate DSA indexer temporary logits workspace in bytes.

    SGLang's NSA indexer builds an MQA logits workspace over q tokens and KV
    tokens before top-k selection.  With synthetic reduced-TP module collection,
    the reduced-head paths can materialize a per-head logits workspace; this
    catches shapes that fit the KV pool but cannot fit the indexer workspace.
    """
    if is_prefill:
        q_tokens = required_prefill_extend_tokens(batch_size, seq_len)
        kv_tokens = required_kv_tokens(batch_size, seq_len, prefix_len, is_prefill=True)
    else:
        q_tokens = batch_size
        kv_tokens = required_kv_tokens(batch_size, seq_len, prefix_len, is_prefill=False)

    head_factor = max(1, int(num_heads)) if target_tp_size > 1 else 1
    return q_tokens * kv_tokens * head_factor * 4


def dsa_indexer_workspace_limit_bytes(free_mem: int, total_mem: int) -> int:
    """Return a conservative per-process DSA indexer workspace limit."""
    # The NSA indexer can hit illegal memory accesses before allocator OOM when
    # its temporary logits workspace gets close to the currently free memory.
    # Keep this intentionally conservative so unsupported long-prefix probes are
    # skipped before launching kernels.
    return int(min(free_mem * 0.30, total_mem * 0.15))


def sglang_dsa_mqa_logits_chunking_supported() -> bool:
    """Return whether the imported SGLang DSA/NSA indexer chunks MQA logits."""
    candidates = (
        "sglang.srt.layers.attention.dsa.dsa_indexer",
        "sglang.srt.layers.attention.nsa.nsa_indexer",
    )
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            indexer_cls = getattr(module, "Indexer", None)
            should_chunk = getattr(indexer_cls, "_should_chunk_mqa_logits", None)
            get_topk = getattr(indexer_cls, "_get_topk_ragged", None)
            if should_chunk is None or get_topk is None:
                continue
            source = "\n".join(
                (
                    inspect.getsource(should_chunk),
                    inspect.getsource(get_topk),
                )
            )
        except Exception:
            continue

        has_chunk_decision = "_should_chunk_mqa_logits" in source
        has_chunk_loop = "while start < q_offset" in source
        has_chunk_kernel = "logits_chunk" in source
        if has_chunk_decision and has_chunk_loop and has_chunk_kernel:
            return True
    return False


def dsa_indexer_prefill_shape_is_supported(batch_size: int, seq_len: int) -> bool:
    """Return whether SGLang's DSA prefill indexer supports this query shape."""
    return batch_size > 0 and seq_len > 1


def dsa_indexer_total_kv_tokens_supported(
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    *,
    is_prefill: bool,
) -> bool:
    """Return whether the DSA indexer offset path supports this KV span."""
    total_kv_tokens = required_kv_tokens(batch_size, seq_len, prefix_len, is_prefill=is_prefill)
    return total_kv_tokens <= DSA_INDEXER_TOTAL_KV_TOKEN_LIMIT


def required_prefill_extend_tokens(batch_size: int, seq_len: int) -> int:
    """Return the number of fresh prefill tokens in a collector case."""
    return batch_size * seq_len


def runtime_chunk_size(model_runner) -> int:
    server_args = getattr(model_runner, "server_args", None)
    sglang_chunk = getattr(server_args, "chunked_prefill_size", None) if server_args else None
    if isinstance(sglang_chunk, int) and sglang_chunk > 0:
        return sglang_chunk
    raise RuntimeError("SGLang did not initialize server_args.chunked_prefill_size")


def chunked_alloc_extend(orig_alloc_extend: Callable, chunk_size: int) -> Callable:
    """Wrap ``alloc_extend`` using SGLang's runtime chunk size."""

    def wrapped(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens):
        bs = prefix_lens.shape[0]
        if extend_num_tokens <= chunk_size or bs == 0:
            return orig_alloc_extend(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens)

        extend_per_req = (seq_lens_cpu - prefix_lens_cpu).tolist()
        chunk_size_per_req = max(1, chunk_size // bs)

        cur_prefix = prefix_lens.clone()
        cur_prefix_cpu = prefix_lens_cpu.clone()
        cur_last_loc = last_loc.clone()
        advanced = [0] * bs
        per_req_indices: list[list] = [[] for _ in range(bs)]

        while True:
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

        final = [torch.cat(lst) for lst in per_req_indices if lst]
        if not final:
            return torch.empty((0,), dtype=torch.int64, device=prefix_lens.device)
        return torch.cat(final)

    return wrapped


@contextmanager
def temporarily_chunked_alloc_extend(model_runner, extend_num_tokens: int):
    """Patch SGLang ``alloc_extend`` while a synthetic collector batch is built."""
    chunk_size = runtime_chunk_size(model_runner)
    allocator = model_runner.token_to_kv_pool_allocator
    saved_alloc_extend = None
    if extend_num_tokens > chunk_size:
        saved_alloc_extend = allocator.alloc_extend
        allocator.alloc_extend = chunked_alloc_extend(saved_alloc_extend, chunk_size=chunk_size)

    try:
        yield
    finally:
        if saved_alloc_extend is not None:
            allocator.alloc_extend = saved_alloc_extend


def alloc_prefix_indices(model_runner, batch_size: int, prefix_len: int) -> list:
    """Allocate per-request prefix KV indices using SGLang's runtime chunk size."""
    device = getattr(model_runner, "device", "cuda")
    if prefix_len <= 0:
        return [torch.empty((0,), dtype=torch.int64, device=device) for _ in range(batch_size)]

    allocator = model_runner.token_to_kv_pool_allocator
    alloc_swa_tail = getattr(allocator, "alloc_extend_swa_tail", None)
    window_size = getattr(getattr(model_runner, "model_config", None), "window_size", None)
    page_size = getattr(allocator, "page_size", None)
    if callable(alloc_swa_tail) and isinstance(window_size, int) and window_size > 0:
        if not isinstance(page_size, int) or page_size <= 1:
            raise RuntimeError("SGLang SWA tail allocation requires a paged KV allocator")
        window_start = max(0, prefix_len - window_size)
        window_start = (window_start // page_size) * page_size
        swa_tail_len = prefix_len - window_start
        per_request = []
        for _ in range(batch_size):
            prefix_lens_cpu = torch.zeros(1, dtype=torch.int64)
            seq_lens_cpu = torch.full((1,), prefix_len, dtype=torch.int64)
            prefix_lens = prefix_lens_cpu.to(device, non_blocking=True)
            seq_lens = seq_lens_cpu.to(device, non_blocking=True)
            last_loc = torch.full((1,), -1, dtype=torch.int64, device=device)
            indices = alloc_swa_tail(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                prefix_len,
                swa_tail_len,
            )
            if indices is None:
                raise RuntimeError(
                    f"failed to allocate SWA-tail prefix cache: prefix_len={prefix_len}, swa_tail_len={swa_tail_len}"
                )
            per_request.append(indices.contiguous())
        return per_request

    chunk_size = runtime_chunk_size(model_runner)
    alloc_extend = allocator.alloc_extend
    if batch_size * prefix_len > chunk_size:
        alloc_extend = chunked_alloc_extend(alloc_extend, chunk_size=chunk_size)

    prefix_lens_cpu = torch.zeros(batch_size, dtype=torch.int64)
    seq_lens_cpu = torch.full((batch_size,), prefix_len, dtype=torch.int64)
    prefix_lens = prefix_lens_cpu.to(device, non_blocking=True)
    seq_lens = seq_lens_cpu.to(device, non_blocking=True)
    last_loc = torch.full((batch_size,), -1, dtype=torch.int64, device=device)

    flat = alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        batch_size * prefix_len,
    )
    if flat is None:
        raise RuntimeError(f"failed to allocate prefix cache: batch_size={batch_size}, prefix_len={prefix_len}")
    return [flat[i * prefix_len : (i + 1) * prefix_len].contiguous() for i in range(batch_size)]
