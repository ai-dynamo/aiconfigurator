# Feature Pitfalls

Use this checklist before enabling advanced AIC features in CLI or YAML.

## MTP / `nextn`

- MTP changes generation modeling by accounting for draft tokens.
- AIC defaults `nextn` to `1` only for model families currently treated as
  native MTP users in the task factory: `DEEPSEEK`, `DEEPSEEKV32`, `KIMIK25`,
  and `QWEN35`. Other models default to `0`.
- Only set `nextn > 0` when the model and backend runtime are expected to use
  MTP/speculative decoding.
- `nextn_accept_rates` must have five values; only the first `nextn` entries are
  used.
- Higher `nextn` also affects activation memory in backend memory estimates.
- Always report the chosen `nextn` and accept rates when comparing results.

## WideEP

- WideEP is for MoE models. Do not enable it for dense models.
- `enable_wideep: true` changes the MoE search space. It generally forces
  `moe_tp_list: [1]` and uses larger `moe_ep_list` candidates.
- In SDK construction, `enable_wideep` implies `moe_backend="deepep_moe"` when no
  MoE backend is explicitly supplied.
- SGLang WideEP is modeled as inter-node DeepEP with EP candidates starting at
  8 in the default search space.
- SGLang `moe_backend="deepep_moe"` without `enable_wideep` represents an
  intra-node DeepEP path with EP candidates up to 8.
- TRT-LLM WideEP expects DP and MoE EP to be greater than 1 in the default MoE
  search space.
- For SGLang non-WideEP disaggregated serving, keep an eye on prefill/decode TP
  compatibility; the webapp enforces same TP for that path.

## EPLB

- EPLB is tied to WideEP-style expert-parallel operation. Do not enable EPLB
  independently unless the WideEP path is intended.
- For webapp-style behavior, `enable_eplb` is only passed through when
  `enable_wideep` is true.
- If `wideep_num_slots` is used, explain whether it equals `num_experts` or adds
  redundant slots.

## Quantization

- Quantization defaults are inferred from model config when possible.
- Explicit YAML quantization overrides inferred defaults. Do not add explicit
  quantization in agent-authored YAML unless the user asks for a quantization
  study or a known runtime requires a specific mode.
- Validate quant mode support against the selected system/backend/version. AIC
  may reject unsupported modes before the sweep.
- `fp8_static` GEMM mode is currently TRT-LLM-only.
- WideEP SGLang paths use dedicated WideEP attention/MoE perf tables, so normal
  MoE quant support is not enough.

## Database Mode

- Use `SILICON` for final experiments, analysis, deployment sizing, and config
  generation.
- Use `SOL` only as an upper-bound or feasibility sanity check.
- Avoid `HYBRID` and `EMPIRICAL` in agent-led experiments. They are not mature
  enough to mix into actual experiment results.
- If silicon data is missing, report the coverage gap instead of substituting
  another database mode into the final result.
