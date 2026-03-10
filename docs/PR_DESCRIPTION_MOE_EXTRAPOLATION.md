# PR Description: MoE extrapolation refactor

Copy the sections below into your PR description to satisfy the template.

---

#### Overview:

This PR refactors MoE performance query behavior so that:

- **Overflow extrapolation** (when `num_tokens` > max collected token) uses a single helper that always returns a `PerformanceResult`; the decision to use extrapolation vs interpolation is made at the call site based on `query_tokens > token_points[-1]`.
- **SOL/roofline** for MoE respects **gated vs non-gated** activation: `query_moe(..., is_gated=False)` now uses a 2-GEMM roofline for SOL and for overflow extrapolation (non-gated / Relu2), matching the behavior of `query_wideep_moe_compute()`.
- **Collector fixes**: trtllm 1.0/1.1 MoE collector (v2) places router logits on the correct device to fix "token_selected_experts must be a CUDA tensor" and "renorm_moe_routing_op CPU backend" errors; `load_cache` only passes `rank` when the API supports it (trtllm >= 1.3); and `torch.AcceleratorError` is guarded for older PyTorch.

Unit tests are added for `query_moe` SILICON path (within-range interpolation, overflow extrapolation, boundary at max tokens). Docstrings are added where missing to meet coverage.

#### Details:

- **`src/aiconfigurator/sdk/perf_database.py`**
  - `query_moe`: Introduce `num_gemms = 3 if is_gated else 2` and use it in the inner `get_sol` (ops and mem_bytes) so SOL and overflow extrapolation use the correct roofline for gated (SwiGLU) vs non-gated (Relu2) MoE.
  - `_estimate_overflow_with_last_token_util`: No longer returns `None`; callers only invoke it when `query_tokens > token_points[-1]`. Removed in-function check and `sol_last <= 0` branch per domain assumptions.
  - Call sites (tensorrt_llm, trtllm, vllm): If `num_tokens > token_points[-1]`, return `_estimate_overflow_with_last_token_util(...)`; else use 1D interpolation and return that result.
- **`tests/unit/sdk/database/test_moe_mla.py`**
  - New tests: `test_query_moe_silicon_within_range_uses_interpolation`, `test_query_moe_silicon_overflow_uses_util_extrapolation`, `test_query_moe_silicon_boundary_at_max_tokens`. Restored `class TestMLABMM` (was accidentally removed in an earlier edit).
- **`collector/trtllm/collect_moe_v2.py`**
  - Router logits moved to device: `power_law_logits_v3(...).to(device=device, dtype=router_logits_dtype)` and `balanced_logits(...).to(device=device, dtype=router_logits_dtype)` so trtllm 1.0/1.1 runs without CUDA/CPU tensor errors.
  - Docstrings added for `get_moe_test_cases` and `run_moe_torch`.
- **`collector/trtllm/collect_moe_v3.py`**
  - `load_cache`: use `inspect.signature(load_cache).parameters` to pass `rank` only when supported (trtllm >= 1.3).
  - Docstrings added for `gc_collect`, `cleanup_empty_json_files`, `get_moe_test_cases`, `run_moe_torch`.
- **`collector/trtllm/collect_wideep_moe_compute.py`**
  - Same `load_cache` rank check; docstrings for `cleanup_empty_json_files` and `moe_op` property.
- **`collector/collect.py`**
  - Use `getattr(torch, "AcceleratorError", None)` before `isinstance(e, ...)` so older PyTorch without `AcceleratorError` does not raise.
- **`tools/sanity_check/validate_database.ipynb`**
  - MoE `weight_bits` derived from `quant_mode.value.memory * 8`; removed undefined `size_of` usage and redundant print.

#### Where should the reviewer start?

1. **`src/aiconfigurator/sdk/perf_database.py`** — `query_moe` and `num_gemms` / `get_sol` / `_estimate_overflow_with_last_token_util` and the three backend branches (sglang, trtllm, vllm) that decide extrapolation vs interpolation.
2. **`tests/unit/sdk/database/test_moe_mla.py`** — New MoE SILICON tests and `TestMLABMM` class.
3. **`collector/trtllm/collect_moe_v2.py`** — Device fix for logits and docstrings (trtllm 1.0/1.1 runnability).

#### Related Issues: (use one of the action keywords Closes / Fixes / Resolves / Relates to)

- Relates to MoE overflow extrapolation and gated/non-gated SOL consistency.
- (Add a specific issue number if applicable, e.g. `Closes #123`.)
