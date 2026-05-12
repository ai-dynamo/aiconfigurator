# Shareability audit

Classifies every `(system, op_file, kernel_source)` triple in the perf database into one of two tiers:

- **`shared`** — named kernel_source. The SDK loader inherits these rows from sibling backend/version directories (cross-version and cross-backend) when the database is loaded in HYBRID mode.
- **`shared_fallback`** — `kernel_source = default`. Framework-implicit, low-fidelity. Inherited alongside `shared` rows in HYBRID mode (HYBRID already accepts coarser fallbacks).

Rows with a blank/`<unknown>` kernel_source are skipped during audit (the current corpus has none).

## Headline numbers

- Total rows scanned: **8,925,338**
- Within-framework cross-version dedup-able rows: **2,955** (~0.0%)
- Tier distribution (groups / rows):
  - `shared`: 344 groups · 8,607,323 rows
  - `shared_fallback`: 47 groups · 318,015 rows


## `computescale_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `torch_ops` | shared | trtllm | trtllm:2068 | 0 / 1649 | 0 | — | — | — |
| b300_sxm | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| gb200 | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| gb300 | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| h100_sxm | `torch_ops` | shared | trtllm | trtllm:2069 | 0 / 1649 | 0 | — | — | — |
| h200_sxm | `torch_ops` | shared | trtllm | trtllm:2069 | 0 / 1649 | 0 | — | — | — |
| l40s | `torch_ops` | shared | trtllm | trtllm:439 | 0 / 439 | 0 | — | — | — |

## `context_attention_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `flash_attention` | shared | sglang | sglang:11254 | 0 / 5627 | 11 | — | — | — |
| a100_sxm | `torch_flow` | shared | trtllm | trtllm:4864 | 0 / 4864 | 0 | — | — | — |
| a100_sxm | `vllm_flash_attn` | shared | vllm | vllm:5457 | 0 / 5049 | 0 | — | — | — |
| b200_sxm | `torch_flow` | shared | trtllm | trtllm:106930 | 0 / 63124 | 87 | — | — | — |
| b200_sxm | `trtllm_mha` | shared | sglang | sglang:67428 | 0 / 33714 | 85 | — | — | — |
| b200_sxm | `vllm_flashinfer` | shared | vllm | vllm:21816 | 0 / 20184 | 0 | — | — | — |
| b300_sxm | `torch_flow` | shared | trtllm | trtllm:63133 | 0 / 63133 | 0 | — | — | — |
| b300_sxm | `trtllm_mha` | shared | sglang | sglang:50595 | 0 / 33714 | 16 | — | — | — |
| b300_sxm | `vllm_flashinfer` | shared | vllm | vllm:21816 | 0 / 20184 | 0 | — | — | — |
| b60 | `vllm_flash_attn` | shared | vllm | vllm:6900 | 0 / 6900 | 0 | — | — | — |
| gb200 | `torch_flow` | shared | trtllm | trtllm:63120 | 0 / 63120 | 0 | — | — | — |
| gb200 | `trtllm_mha` | shared | sglang | sglang:50595 | 0 / 33714 | 37 | — | — | — |
| gb200 | `vllm_flashinfer` | shared | vllm | vllm:32586 | 0 / 20184 | 0 | — | — | — |
| gb300 | `torch_flow` | shared | trtllm | trtllm:63121 | 0 / 63121 | 0 | — | — | — |
| gb300 | `trtllm_mha` | shared | sglang | sglang:50595 | 0 / 33714 | 25 | — | — | — |
| gb300 | `vllm_flashinfer` | shared | vllm | vllm:22506 | 0 / 20184 | 0 | — | — | — |
| h100_sxm | `flash_attention` | shared | sglang | sglang:49356 | 0 / 16881 | 2 | — | — | — |
| h100_sxm | `torch_flow` | shared | trtllm | trtllm:97282 | 0 / 50578 | 29 | — | — | — |
| h100_sxm | `vllm_flash_attn` | shared | vllm | vllm:10914 | 0 / 10098 | 0 | — | — | — |
| h200_sxm | `flash_attention` | shared | sglang | sglang:50643 | 0 / 16881 | 1 | — | — | — |
| h200_sxm | `torch_flow` | shared | trtllm | trtllm:97283 | 0 / 50578 | 10 | — | — | — |
| h200_sxm | `vllm_flash_attn` | shared | vllm | vllm:60866 | 0 / 47530 | 49 | — | — | — |
| l40s | `flash_attention` | shared | sglang | sglang:11254 | 0 / 5627 | 12 | — | — | — |
| l40s | `torch_flow` | shared | trtllm | trtllm:29323 | 0 / 29323 | 0 | — | — | — |
| l40s | `vllm_flash_attn` | shared | vllm | vllm:5457 | 0 / 5049 | 0 | — | — | — |
| l40s | `vllm_flashinfer` | shared | vllm | vllm:5457 | 0 / 5049 | 0 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:33522 | 0 / 33522 | 0 | — | — | — |

## `context_mla_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `default` | shared_fallback | trtllm | trtllm:2436 | 0 / 2436 | 0 | — | — | — |
| a100_sxm | `vllm_triton_mla` | shared | vllm | vllm:880 | 0 / 880 | 0 | — | — | — |
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:3152 | 0 / 1760 | 0 | — | — | — |
| b200_sxm | `trtllm_mla` | shared | sglang | sglang:6160 | 0 / 3080 | 3 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:1760 | 0 / 1760 | 0 | — | — | — |
| b300_sxm | `trtllm_mla` | shared | sglang | sglang:6160 | 0 / 3080 | 4 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:1760 | 0 / 1760 | 0 | — | — | — |
| gb200 | `trtllm_mla` | shared | sglang | sglang:6160 | 0 / 3080 | 1 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:1760 | 0 / 1760 | 0 | — | — | — |
| gb300 | `trtllm_mla` | shared | sglang | sglang:6160 | 0 / 3080 | 4 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:3520 | 0 / 1760 | 2 | — | — | — |
| h100_sxm | `flash_attention` | shared | sglang | sglang:8596 | 0 / 3080 | 1 | — | — | — |
| h100_sxm | `vllm_flash_attn_mla` | shared | vllm | vllm:1650 | 0 / 1650 | 0 | — | — | — |
| h100_sxm | `vllm_flashmla` | shared | vllm | vllm:1515 | 0 / 1515 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:3152 | 0 / 1760 | 2 | — | — | — |
| h200_sxm | `flash_attention` | shared | sglang | sglang:9240 | 0 / 3080 | 5 | — | — | — |
| h200_sxm | `vllm_flash_attn_mla` | shared | vllm | vllm:1650 | 0 / 1650 | 0 | — | — | — |
| h200_sxm | `vllm_flashmla` | shared | vllm | vllm:1515 | 0 / 1515 | 0 | — | — | — |
| l40s | `default` | shared_fallback | trtllm | trtllm:2436 | 0 / 2436 | 0 | — | — | — |
| l40s | `vllm_triton_mla` | shared | vllm | vllm:880 | 0 / 880 | 0 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:1540 | 0 / 1540 | 0 | — | — | — |

## `custom_allreduce_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:138 | 0 / 69 | 0 | — | — | — |
| a100_sxm | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:138 | 0 / 69 | 0 | — | — | — |
| a100_sxm | `TRTLLM` | shared | trtllm | trtllm:69 | 0 / 69 | 0 | — | — | — |
| a100_sxm | `vLLM_custom_eager` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| a100_sxm | `vLLM_custom_graph` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| b200_sxm | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:138 | 0 / 69 | 0 | — | — | — |
| b200_sxm | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:138 | 0 / 69 | 0 | — | — | — |
| b200_sxm | `TRTLLM` | shared | trtllm | trtllm:276 | 0 / 69 | 0 | — | — | — |
| b200_sxm | `vLLM_custom_eager` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| b200_sxm | `vLLM_custom_graph` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| b300_sxm | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:138 | 0 / 69 | 0 | — | — | — |
| b300_sxm | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:138 | 0 / 69 | 0 | — | — | — |
| b300_sxm | `TRTLLM` | shared | trtllm | trtllm:207 | 0 / 69 | 0 | — | — | — |
| b300_sxm | `vLLM_custom_eager` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| b300_sxm | `vLLM_custom_graph` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| b60 | `vLLM_custom_eager` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| gb200 | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:138 | 0 / 46 | 0 | — | — | — |
| gb200 | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:138 | 0 / 46 | 0 | — | — | — |
| gb200 | `TRTLLM` | shared | trtllm | trtllm:230 | 0 / 46 | 0 | — | — | — |
| gb200 | `vLLM_custom_eager` | shared | vllm | vllm:92 | 0 / 46 | 0 | — | — | — |
| gb200 | `vLLM_custom_graph` | shared | vllm | vllm:92 | 0 / 46 | 0 | — | — | — |
| gb300 | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:138 | 0 / 46 | 0 | — | — | — |
| gb300 | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:138 | 0 / 46 | 0 | — | — | — |
| gb300 | `TRTLLM` | shared | trtllm | trtllm:184 | 0 / 46 | 0 | — | — | — |
| gb300 | `vLLM_custom_eager` | shared | vllm | vllm:92 | 0 / 46 | 1 | — | — | — |
| gb300 | `vLLM_custom_graph` | shared | vllm | vllm:92 | 0 / 46 | 0 | — | — | — |
| h100_sxm | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:138 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:138 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `TRTLLM` | shared | trtllm | trtllm:276 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `vLLM_custom_eager` | shared | vllm | vllm:138 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `vLLM_custom_graph` | shared | vllm | vllm:138 | 0 / 69 | 0 | — | — | — |
| h200_sxm | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:138 | 0 / 69 | 0 | — | — | — |
| h200_sxm | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:138 | 0 / 69 | 0 | — | — | — |
| h200_sxm | `TRTLLM` | shared | trtllm | trtllm:276 | 0 / 69 | 0 | — | — | — |
| h200_sxm | `vLLM_custom_eager` | shared | vllm | vllm:138 | 0 / 69 | 0 | — | — | — |
| h200_sxm | `vLLM_custom_graph` | shared | vllm | vllm:138 | 0 / 69 | 0 | — | — | — |
| l40s | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:69 | 0 / 69 | 0 | — | — | — |
| l40s | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:69 | 0 / 69 | 0 | — | — | — |
| l40s | `TRTLLM` | shared | trtllm | trtllm:138 | 0 / 69 | 0 | — | — | — |
| l40s | `vLLM_custom_eager` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| l40s | `vLLM_custom_graph` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| rtx_pro_6000_server | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:23 | 0 / 23 | 0 | — | — | — |
| rtx_pro_6000_server | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:23 | 0 / 23 | 0 | — | — | — |

## `dsa_context_module_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `default` | shared_fallback | trtllm, vllm | trtllm:15502, vllm:17568 | 11626 / 17568 | 8 | 68.7 | 177.9 | 185.5 |
| b200_sxm | `dsa_nsa` | shared | sglang | sglang:4031 | 0 / 4031 | 0 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:11564 | 0 / 11564 | 0 | — | — | — |
| b300_sxm | `dsa_nsa` | shared | sglang | sglang:882 | 0 / 786 | 0 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:11553 | 0 / 11553 | 0 | — | — | — |
| gb200 | `dsa_nsa` | shared | sglang | sglang:151 | 0 / 151 | 0 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:11563 | 0 / 11563 | 0 | — | — | — |
| gb300 | `dsa_nsa` | shared | sglang | sglang:151 | 0 / 151 | 0 | — | — | — |
| h100_sxm | `dsa_nsa` | shared | sglang | sglang:1287 | 0 / 1068 | 0 | — | — | — |
| h200_sxm | `dsa_nsa` | shared | sglang | sglang:3917 | 0 / 3917 | 0 | — | — | — |

## `dsa_generation_module_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `default` | shared_fallback | trtllm, vllm | trtllm:26453, vllm:17664 | 17636 / 17664 | 40 | 53.3 | 92.2 | 116.3 |
| b200_sxm | `dsa_nsa` | shared | sglang | sglang:4092 | 0 / 4092 | 0 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:17643 | 0 / 17643 | 0 | — | — | — |
| b300_sxm | `dsa_nsa` | shared | sglang | sglang:1473 | 0 / 1334 | 0 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:17649 | 0 / 17649 | 0 | — | — | — |
| gb200 | `dsa_nsa` | shared | sglang | sglang:396 | 0 / 396 | 0 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:17650 | 0 / 17650 | 0 | — | — | — |
| gb300 | `dsa_nsa` | shared | sglang | sglang:396 | 0 / 396 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:4804 | 0 / 4804 | 0 | — | — | — |
| h100_sxm | `dsa_nsa` | shared | sglang | sglang:2244 | 0 / 1848 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:2944 | 0 / 2944 | 0 | — | — | — |
| h200_sxm | `dsa_nsa` | shared | sglang | sglang:4752 | 0 / 4752 | 0 | — | — | — |

## `gdn_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `causal_conv1d_fn` | shared | sglang | sglang:824 | 0 / 824 | 0 | — | — | — |
| a100_sxm | `causal_conv1d_update` | shared | sglang | sglang:88 | 0 / 88 | 0 | — | — | — |
| a100_sxm | `fused_recurrent_gated_delta_rule` | shared | sglang | sglang:86 | 0 / 86 | 0 | — | — | — |
| b200_sxm | `causal_conv1d_fn` | shared | sglang, trtllm | sglang:854, trtllm:854 | 854 / 854 | 0 | 9.0 | 22.0 | 30.8 |
| b200_sxm | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:88, trtllm:88, vllm:88 | 88 / 88 | 0 | 7.8 | 36.0 | 37.9 |
| b200_sxm | `chunk_gated_delta_rule` | shared | trtllm | trtllm:854 | 0 / 854 | 0 | — | — | — |
| b200_sxm | `fused_recurrent_gated_delta_rule` | shared | sglang, trtllm | sglang:86, trtllm:86 | 86 / 86 | 0 | 20.5 | 39.5 | 48.1 |
| b300_sxm | `causal_conv1d_fn` | shared | sglang, trtllm | sglang:1674, trtllm:665 | 665 / 837 | 5 | 8.6 | 22.1 | 29.1 |
| b300_sxm | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:165, trtllm:22, vllm:88 | 88 / 88 | 1 | 15.5 | 37.3 | 38.9 |
| b300_sxm | `chunk_gated_delta_rule` | shared | trtllm | trtllm:659 | 0 / 659 | 0 | — | — | — |
| b300_sxm | `fused_recurrent_gated_delta_rule` | shared | sglang, trtllm | sglang:161, trtllm:22 | 22 / 86 | 1 | 20.7 | 45.3 | 46.8 |
| gb200 | `causal_conv1d_fn` | shared | sglang, trtllm | sglang:1708, trtllm:854 | 854 / 854 | 2 | 6.6 | 22.0 | 27.3 |
| gb200 | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:176, trtllm:88, vllm:88 | 88 / 88 | 0 | 25.1 | 38.4 | 45.1 |
| gb200 | `chunk_gated_delta_rule` | shared | trtllm | trtllm:854 | 0 / 854 | 0 | — | — | — |
| gb200 | `fused_recurrent_gated_delta_rule` | shared | sglang, trtllm | sglang:172, trtllm:86 | 86 / 86 | 0 | 18.6 | 38.3 | 40.3 |
| gb300 | `causal_conv1d_fn` | shared | sglang, trtllm | sglang:1600, trtllm:843 | 800 / 858 | 1 | 7.5 | 20.5 | 28.7 |
| gb300 | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:132, trtllm:77, vllm:88 | 88 / 88 | 0 | 34.3 | 77.4 | 105.2 |
| gb300 | `chunk_gated_delta_rule` | shared | trtllm | trtllm:836 | 0 / 836 | 0 | — | — | — |
| gb300 | `fused_recurrent_gated_delta_rule` | shared | sglang, trtllm | sglang:131, trtllm:75 | 65 / 86 | 0 | 19.7 | 27.8 | 31.8 |
| h100_sxm | `causal_conv1d_fn` | shared | sglang, trtllm | sglang:1648, trtllm:824 | 824 / 824 | 9 | 22.8 | 29.1 | 117.9 |
| h100_sxm | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:176, trtllm:88, vllm:88 | 88 / 88 | 0 | 24.8 | 38.4 | 40.0 |
| h100_sxm | `chunk_gated_delta_rule` | shared | trtllm | trtllm:806 | 0 / 806 | 0 | — | — | — |
| h100_sxm | `fused_recurrent_gated_delta_rule` | shared | sglang, trtllm | sglang:172, trtllm:86 | 86 / 86 | 0 | 14.0 | 40.2 | 43.9 |
| h200_sxm | `causal_conv1d_fn` | shared | sglang, trtllm | sglang:848, trtllm:848 | 848 / 848 | 0 | 23.4 | 38.0 | 56.5 |
| h200_sxm | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:88, trtllm:88, vllm:88 | 88 / 88 | 0 | 17.9 | 37.6 | 48.9 |
| h200_sxm | `chunk_gated_delta_rule` | shared | trtllm | trtllm:840 | 0 / 840 | 0 | — | — | — |
| h200_sxm | `fused_recurrent_gated_delta_rule` | shared | sglang, trtllm | sglang:86, trtllm:86 | 86 / 86 | 0 | 19.2 | 38.7 | 40.3 |
| l40s | `causal_conv1d_fn` | shared | sglang | sglang:786 | 0 / 786 | 0 | — | — | — |
| l40s | `causal_conv1d_update` | shared | sglang | sglang:88 | 0 / 88 | 0 | — | — | — |
| l40s | `fused_recurrent_gated_delta_rule` | shared | sglang | sglang:86 | 0 / 86 | 0 | — | — | — |
| rtx_pro_6000_server | `causal_conv1d_fn` | shared | sglang | sglang:824 | 0 / 824 | 0 | — | — | — |
| rtx_pro_6000_server | `causal_conv1d_update` | shared | sglang | sglang:88 | 0 / 88 | 0 | — | — | — |
| rtx_pro_6000_server | `fused_recurrent_gated_delta_rule` | shared | sglang | sglang:86 | 0 / 86 | 0 | — | — | — |

## `gemm_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `sglang` | shared | sglang | sglang:44982 | 0 / 36182 | 3 | — | — | — |
| a100_sxm | `torch_flow` | shared | trtllm | trtllm:9240 | 0 / 9240 | 0 | — | — | — |
| a100_sxm | `trt_flow_/smooth_quant_gemm_L96/PLUGIN_V2_SmoothQuantGemm_0` | shared | trtllm | trtllm:6048 | 0 / 6048 | 0 | — | — | — |
| a100_sxm | `trt_flow_/weight_only_quant_matmul_L257/PLUGIN_V2_WeightOnlyQuantMatmul_0` | shared | trtllm | trtllm:12096 | 0 / 12096 | 0 | — | — | — |
| a100_sxm | `vllm_default` | shared | vllm | vllm:9240 | 0 / 9240 | 0 | — | — | — |
| b200_sxm | `deepgemm` | shared | trtllm | trtllm:58650 | 0 / 29412 | 3 | — | — | — |
| b200_sxm | `sglang` | shared | sglang | sglang:261072 | 0 / 130536 | 276 | — | — | — |
| b200_sxm | `torch_flow` | shared | trtllm | trtllm:202020 | 0 / 101010 | 128 | — | — | — |
| b200_sxm | `vllm_default` | shared | vllm | vllm:114566 | 0 / 114566 | 0 | — | — | — |
| b300_sxm | `deepgemm` | shared | trtllm | trtllm:29242 | 0 / 29242 | 0 | — | — | — |
| b300_sxm | `sglang` | shared | sglang | sglang:261072 | 0 / 130536 | 274 | — | — | — |
| b300_sxm | `torch_flow` | shared | trtllm | trtllm:101010 | 0 / 101010 | 0 | — | — | — |
| b300_sxm | `vllm_default` | shared | vllm | vllm:114569 | 0 / 114569 | 0 | — | — | — |
| b60 | `vllm_default` | shared | vllm | vllm:15162 | 0 / 15162 | 0 | — | — | — |
| gb200 | `deepgemm` | shared | trtllm | trtllm:29378 | 0 / 29370 | 0 | — | — | — |
| gb200 | `sglang` | shared | sglang | sglang:261072 | 0 / 130536 | 222 | — | — | — |
| gb200 | `torch_flow` | shared | trtllm | trtllm:101010 | 0 / 101010 | 0 | — | — | — |
| gb200 | `vllm_default` | shared | vllm | vllm:147454 | 0 / 116174 | 12 | — | — | — |
| gb300 | `deepgemm` | shared | trtllm | trtllm:29382 | 0 / 29382 | 0 | — | — | — |
| gb300 | `sglang` | shared | sglang | sglang:261072 | 0 / 130536 | 260 | — | — | — |
| gb300 | `torch_flow` | shared | trtllm | trtllm:101014 | 0 / 101010 | 0 | — | — | — |
| gb300 | `vllm_default` | shared | vllm | vllm:200105 | 0 / 115059 | 3 | — | — | — |
| h100_sxm | `sglang` | shared | sglang | sglang:237294 | 0 / 111487 | 89 | — | — | — |
| h100_sxm | `torch_flow` | shared | trtllm | trtllm:202902 | 0 / 101892 | 103 | — | — | — |
| h100_sxm | `vllm_default` | shared | vllm | vllm:127043 | 0 / 102243 | 12 | — | — | — |
| h200_sxm | `sglang` | shared | sglang | sglang:228060 | 0 / 102250 | 90 | — | — | — |
| h200_sxm | `torch_flow` | shared | trtllm | trtllm:127050 | 0 / 102250 | 33 | — | — | — |
| h200_sxm | `trt_flow_/smooth_quant_gemm_L96/PLUGIN_V2_SmoothQuantGemm_0` | shared | trtllm | trtllm:9240 | 0 / 9240 | 0 | — | — | — |
| h200_sxm | `trt_flow_/weight_only_quant_matmul_L257/PLUGIN_V2_WeightOnlyQuantMatmul_0` | shared | trtllm | trtllm:18480 | 0 / 18480 | 0 | — | — | — |
| h200_sxm | `vllm_default` | shared | vllm | vllm:202014 | 0 / 101010 | 75 | — | — | — |
| l40s | `sglang` | shared | sglang | sglang:89847 | 0 / 72247 | 9 | — | — | — |
| l40s | `torch_flow` | shared | trtllm | trtllm:26040 | 0 / 26040 | 0 | — | — | — |
| l40s | `trt_flow_/smooth_quant_gemm_L96/PLUGIN_V2_SmoothQuantGemm_0` | shared | trtllm | trtllm:6048 | 0 / 6048 | 0 | — | — | — |
| l40s | `trt_flow_/weight_only_quant_matmul_L257/PLUGIN_V2_WeightOnlyQuantMatmul_0` | shared | trtllm | trtllm:12096 | 0 / 12096 | 0 | — | — | — |
| l40s | `vllm_default` | shared | vllm | vllm:18480 | 0 / 18480 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang` | shared | sglang | sglang:101010 | 0 / 101010 | 0 | — | — | — |

## `generation_attention_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `flash_attention` | shared | sglang | sglang:10186 | 0 / 5093 | 13 | — | — | — |
| a100_sxm | `torch_flow` | shared | trtllm | trtllm:5026 | 0 / 5026 | 0 | — | — | — |
| a100_sxm | `vllm_flash_attn` | shared | vllm | vllm:5431 | 0 / 5431 | 0 | — | — | — |
| b200_sxm | `torch_flow` | shared | trtllm | trtllm:69586 | 0 / 40240 | 56 | — | — | — |
| b200_sxm | `trtllm_mha` | shared | sglang | sglang:38968 | 0 / 19484 | 111 | — | — | — |
| b200_sxm | `vllm_flashinfer` | shared | vllm | vllm:16908 | 0 / 16908 | 0 | — | — | — |
| b300_sxm | `torch_flow` | shared | trtllm | trtllm:40240 | 0 / 40240 | 0 | — | — | — |
| b300_sxm | `trtllm_mha` | shared | sglang | sglang:29670 | 0 / 19484 | 16 | — | — | — |
| b300_sxm | `vllm_flashinfer` | shared | vllm | vllm:16908 | 0 / 16908 | 0 | — | — | — |
| b60 | `vllm_flash_attn` | shared | vllm | vllm:10870 | 0 / 10870 | 0 | — | — | — |
| gb200 | `torch_flow` | shared | trtllm | trtllm:40240 | 0 / 40240 | 0 | — | — | — |
| gb200 | `trtllm_mha` | shared | sglang | sglang:29670 | 0 / 19484 | 42 | — | — | — |
| gb200 | `vllm_flashinfer` | shared | vllm | vllm:25682 | 0 / 16908 | 3 | — | — | — |
| gb300 | `torch_flow` | shared | trtllm | trtllm:40240 | 0 / 40240 | 0 | — | — | — |
| gb300 | `trtllm_mha` | shared | sglang | sglang:29670 | 0 / 19484 | 21 | — | — | — |
| gb300 | `vllm_flashinfer` | shared | vllm | vllm:25682 | 0 / 16908 | 5 | — | — | — |
| h100_sxm | `flash_attention` | shared | sglang | sglang:30558 | 0 / 10186 | 0 | — | — | — |
| h100_sxm | `torch_flow` | shared | trtllm | trtllm:54168 | 0 / 29730 | 142 | — | — | — |
| h100_sxm | `vllm_flash_attn` | shared | vllm | vllm:10870 | 0 / 10870 | 0 | — | — | — |
| h200_sxm | `flash_attention` | shared | sglang | sglang:30558 | 0 / 10186 | 0 | — | — | — |
| h200_sxm | `torch_flow` | shared | trtllm | trtllm:54168 | 0 / 29730 | 49 | — | — | — |
| h200_sxm | `vllm_flash_attn` | shared | vllm | vllm:65217 | 0 / 54348 | 72 | — | — | — |
| l40s | `flash_attention` | shared | sglang | sglang:10186 | 0 / 5093 | 21 | — | — | — |
| l40s | `torch_flow` | shared | trtllm | trtllm:20104 | 0 / 20104 | 0 | — | — | — |
| l40s | `vllm_flash_attn` | shared | vllm | vllm:5432 | 0 / 5432 | 0 | — | — | — |
| l40s | `vllm_flashinfer` | shared | vllm | vllm:5435 | 0 / 5435 | 0 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:19484 | 0 / 19484 | 0 | — | — | — |

## `generation_mla_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `default` | shared_fallback | trtllm | trtllm:5068 | 0 / 5068 | 0 | — | — | — |
| a100_sxm | `vllm_triton_mla` | shared | vllm | vllm:1365 | 0 / 1365 | 0 | — | — | — |
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:5792 | 0 / 2896 | 2 | — | — | — |
| b200_sxm | `trtllm_mla` | shared | sglang | sglang:9296 | 0 / 4648 | 30 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:2896 | 0 / 2896 | 0 | — | — | — |
| b300_sxm | `trtllm_mla` | shared | sglang | sglang:9296 | 0 / 4648 | 29 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:3615 | 0 / 2896 | 0 | — | — | — |
| gb200 | `trtllm_mla` | shared | sglang | sglang:9296 | 0 / 4648 | 14 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:2896 | 0 / 2896 | 0 | — | — | — |
| gb300 | `trtllm_mla` | shared | sglang | sglang:9296 | 0 / 4648 | 5 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:5792 | 0 / 2896 | 0 | — | — | — |
| h100_sxm | `flash_attention` | shared | sglang | sglang:13944 | 0 / 4648 | 5 | — | — | — |
| h100_sxm | `vllm_flash_attn_mla` | shared | vllm | vllm:2715 | 0 / 2715 | 0 | — | — | — |
| h100_sxm | `vllm_flashmla` | shared | vllm | vllm:2715 | 0 / 2715 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:5792 | 0 / 2896 | 0 | — | — | — |
| h200_sxm | `flash_attention` | shared | sglang | sglang:13944 | 0 / 4648 | 3 | — | — | — |
| h200_sxm | `vllm_flash_attn_mla` | shared | vllm | vllm:2715 | 0 / 2715 | 0 | — | — | — |
| h200_sxm | `vllm_flashmla` | shared | vllm | vllm:2715 | 0 / 2715 | 0 | — | — | — |
| l40s | `default` | shared_fallback | trtllm | trtllm:5068 | 0 / 5068 | 0 | — | — | — |
| l40s | `vllm_triton_mla` | shared | vllm | vllm:1401 | 0 / 1401 | 0 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:2324 | 0 / 2324 | 0 | — | — | — |

## `mamba2_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `causal_conv1d_fn` | shared | trtllm | trtllm:425 | 0 / 425 | 0 | — | — | — |
| b200_sxm | `causal_conv1d_update` | shared | trtllm | trtllm:44 | 0 / 44 | 0 | — | — | — |
| b300_sxm | `causal_conv1d_fn` | shared | trtllm | trtllm:436 | 0 / 436 | 0 | — | — | — |
| b300_sxm | `causal_conv1d_update` | shared | trtllm | trtllm:44 | 0 / 44 | 0 | — | — | — |
| gb200 | `causal_conv1d_fn` | shared | trtllm | trtllm:425 | 0 / 425 | 0 | — | — | — |
| gb200 | `causal_conv1d_update` | shared | trtllm | trtllm:44 | 0 / 44 | 0 | — | — | — |
| gb300 | `causal_conv1d_fn` | shared | trtllm | trtllm:436 | 0 / 436 | 0 | — | — | — |
| gb300 | `causal_conv1d_update` | shared | trtllm | trtllm:44 | 0 / 44 | 0 | — | — | — |
| h100_sxm | `causal_conv1d_fn` | shared | trtllm | trtllm:410 | 0 / 410 | 0 | — | — | — |
| h100_sxm | `causal_conv1d_update` | shared | trtllm | trtllm:44 | 0 / 44 | 0 | — | — | — |
| h200_sxm | `causal_conv1d_fn` | shared | trtllm | trtllm:425 | 0 / 425 | 0 | — | — | — |
| h200_sxm | `causal_conv1d_update` | shared | trtllm | trtllm:44 | 0 / 44 | 0 | — | — | — |

## `mhc_module_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| rtx_pro_6000_server | `sglang_mhc` | shared | sglang | sglang:139 | 0 / 70 | 0 | — | — | — |

## `mla_bmm_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `default` | shared_fallback | sglang, trtllm | sglang:848, trtllm:424 | 224 / 224 | 2 | 1.3 | 5.0 | 15.5 |
| b200_sxm | `default` | shared_fallback | sglang, trtllm | sglang:1696, trtllm:848 | 224 / 448 | 7 | 1.5 | 9.0 | 32.4 |
| b300_sxm | `default` | shared_fallback | sglang, trtllm | sglang:1696, trtllm:424 | 224 / 448 | 6 | 2.0 | 14.6 | 27.7 |
| gb200 | `default` | shared_fallback | sglang, trtllm | sglang:1696, trtllm:424 | 224 / 448 | 1 | 3.4 | 30.9 | 34.3 |
| gb300 | `default` | shared_fallback | sglang, trtllm | sglang:1696, trtllm:424 | 224 / 448 | 0 | 3.3 | 27.8 | 63.3 |
| h100_sxm | `default` | shared_fallback | sglang, trtllm | sglang:2544, trtllm:1696 | 448 / 448 | 8 | 5.3 | 25.0 | 34.8 |
| h200_sxm | `default` | shared_fallback | sglang, trtllm | sglang:2544, trtllm:1696 | 448 / 448 | 4 | 4.1 | 22.4 | 35.0 |
| l40s | `default` | shared_fallback | sglang, trtllm | sglang:1696, trtllm:848 | 448 / 448 | 4 | 10.6 | 52.7 | 81.9 |
| rtx_pro_6000_server | `default` | shared_fallback | sglang | sglang:848 | 0 / 448 | 0 | — | — | — |

## `mla_context_module_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `default` | shared_fallback | trtllm, vllm | trtllm:2928, vllm:8784 | 2928 / 8784 | 0 | 40.5 | 63.4 | 96.8 |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:2928 | 0 / 2928 | 0 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:2928 | 0 / 2928 | 0 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:2928 | 0 / 2928 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:3878 | 0 / 3878 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:3873 | 0 / 3873 | 0 | — | — | — |

## `mla_generation_module_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `default` | shared_fallback | trtllm, vllm | trtllm:4415, vllm:8832 | 4415 / 8832 | 0 | 47.1 | 76.3 | 94.9 |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:4415 | 0 / 4415 | 0 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:4415 | 0 / 4415 | 0 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:4415 | 0 / 4415 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:5888 | 0 / 5888 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:5888 | 0 / 5888 | 0 | — | — | — |

## `moe_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `moe_torch_flow` | shared | trtllm | trtllm:8940 | 0 / 8940 | 0 | — | — | — |
| a100_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:39385 | 0 / 33553 | 0 | — | — | — |
| a100_sxm | `sglang_marlin_moe` | shared | sglang | sglang:19035 | 0 / 16686 | 0 | — | — | — |
| a100_sxm | `vllm_fused_moe` | shared | vllm | vllm:6360 | 0 / 6360 | 0 | — | — | — |
| b200_sxm | `deepgemm` | shared | trtllm | trtllm:14451 | 0 / 14451 | 0 | — | — | — |
| b200_sxm | `moe_torch_flow` | shared | trtllm | trtllm:2268 | 0 / 2268 | 0 | — | — | — |
| b200_sxm | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:147177 | 0 / 78165 | 16 | — | — | — |
| b200_sxm | `moe_torch_flow_min_latency` | shared | trtllm | trtllm:34605 | 0 / 19440 | 11 | — | — | — |
| b200_sxm | `moe_torch_flow_nongated` | shared | trtllm | trtllm:18954 | 0 / 9558 | 2 | — | — | — |
| b200_sxm | `sglang_flashinfer_cutedsl_moe` | shared | sglang | sglang:61480 | 0 / 29331 | 41 | — | — | — |
| b200_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:133606 | 0 / 63828 | 20 | — | — | — |
| b200_sxm | `sglang_marlin_moe` | shared | sglang | sglang:27803 | 0 / 12717 | 6 | — | — | — |
| b200_sxm | `vllm_flashinfer_trtllm_moe_fp4` | shared | vllm | vllm:10287 | 0 / 9234 | 0 | — | — | — |
| b200_sxm | `vllm_fused_moe` | shared | vllm | vllm:36288 | 0 / 33210 | 0 | — | — | — |
| b200_sxm | `vllm_marlin_int4_moe` | shared | vllm | vllm:10206 | 0 / 9234 | 0 | — | — | — |
| b300_sxm | `moe_torch_flow` | shared | trtllm | trtllm:2268 | 0 / 2268 | 0 | — | — | — |
| b300_sxm | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:78165 | 0 / 78165 | 0 | — | — | — |
| b300_sxm | `moe_torch_flow_nongated` | shared | trtllm | trtllm:9558 | 0 / 9558 | 0 | — | — | — |
| b300_sxm | `sglang_flashinfer_cutedsl_moe` | shared | sglang | sglang:51829 | 0 / 29346 | 17 | — | — | — |
| b300_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:112463 | 0 / 63827 | 2 | — | — | — |
| b300_sxm | `sglang_marlin_moe` | shared | sglang | sglang:19027 | 0 / 16686 | 0 | — | — | — |
| b300_sxm | `vllm_flashinfer_trtllm_moe_fp4` | shared | vllm | vllm:10287 | 0 / 9234 | 0 | — | — | — |
| b300_sxm | `vllm_fused_moe` | shared | vllm | vllm:36288 | 0 / 33210 | 0 | — | — | — |
| b300_sxm | `vllm_marlin_int4_moe` | shared | vllm | vllm:10206 | 0 / 9234 | 0 | — | — | — |
| b60 | `vllm_fused_moe` | shared | vllm | vllm:1296 | 0 / 1296 | 0 | — | — | — |
| gb200 | `moe_torch_flow` | shared | trtllm | trtllm:2268 | 0 / 2268 | 0 | — | — | — |
| gb200 | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:78192 | 0 / 78165 | 0 | — | — | — |
| gb200 | `moe_torch_flow_min_latency` | shared | trtllm | trtllm:15552 | 0 / 15552 | 0 | — | — | — |
| gb200 | `moe_torch_flow_nongated` | shared | trtllm | trtllm:9558 | 0 / 9558 | 0 | — | — | — |
| gb200 | `sglang_flashinfer_cutedsl_moe` | shared | sglang | sglang:61560 | 0 / 29351 | 18 | — | — | — |
| gb200 | `sglang_fused_moe_triton` | shared | sglang | sglang:133616 | 0 / 63828 | 1 | — | — | — |
| gb200 | `sglang_marlin_moe` | shared | sglang | sglang:32207 | 0 / 16686 | 0 | — | — | — |
| gb200 | `vllm_flashinfer_trtllm_moe_fp4` | shared | vllm | vllm:19521 | 0 / 9234 | 0 | — | — | — |
| gb200 | `vllm_fused_moe` | shared | vllm | vllm:69336 | 0 / 33210 | 1 | — | — | — |
| gb200 | `vllm_marlin_int4_moe` | shared | vllm | vllm:10206 | 0 / 9234 | 0 | — | — | — |
| gb200 | `vllm_marlin_moe` | shared | vllm | vllm:4708 | 0 / 4123 | 0 | — | — | — |
| gb300 | `moe_torch_flow` | shared | trtllm | trtllm:2268 | 0 / 2268 | 0 | — | — | — |
| gb300 | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:78189 | 0 / 78165 | 0 | — | — | — |
| gb300 | `moe_torch_flow_nongated` | shared | trtllm | trtllm:9558 | 0 / 9558 | 0 | — | — | — |
| gb300 | `sglang_flashinfer_cutedsl_moe` | shared | sglang | sglang:61553 | 0 / 29352 | 32 | — | — | — |
| gb300 | `sglang_fused_moe_triton` | shared | sglang | sglang:133624 | 0 / 63828 | 5 | — | — | — |
| gb300 | `sglang_marlin_moe` | shared | sglang | sglang:32199 | 0 / 16686 | 5 | — | — | — |
| gb300 | `vllm_flashinfer_trtllm_moe_fp4` | shared | vllm | vllm:19521 | 0 / 9234 | 0 | — | — | — |
| gb300 | `vllm_fused_moe` | shared | vllm | vllm:49094 | 0 / 33210 | 1 | — | — | — |
| gb300 | `vllm_marlin_int4_moe` | shared | vllm | vllm:10206 | 0 / 9234 | 0 | — | — | — |
| gb300 | `vllm_marlin_moe` | shared | vllm | vllm:4732 | 0 / 4147 | 0 | — | — | — |
| h100_sxm | `moe_torch_flow` | shared | trtllm | trtllm:6876 | 0 / 6390 | 0 | — | — | — |
| h100_sxm | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:149040 | 0 / 74439 | 6 | — | — | — |
| h100_sxm | `moe_torch_flow_nongated` | shared | trtllm | trtllm:15066 | 0 / 7533 | 2 | — | — | — |
| h100_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:138018 | 0 / 63984 | 1 | — | — | — |
| h100_sxm | `sglang_marlin_moe` | shared | sglang | sglang:32238 | 0 / 16686 | 0 | — | — | — |
| h100_sxm | `vllm_fused_moe` | shared | vllm | vllm:69336 | 0 / 33210 | 7 | — | — | — |
| h100_sxm | `vllm_marlin_int4_moe` | shared | vllm | vllm:10206 | 0 / 9234 | 0 | — | — | — |
| h100_sxm | `vllm_marlin_moe` | shared | vllm | vllm:4714 | 0 / 4129 | 0 | — | — | — |
| h200_sxm | `moe_torch_flow` | shared | trtllm | trtllm:64764 | 0 / 63630 | 0 | — | — | — |
| h200_sxm | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:65691 | 0 / 65691 | 0 | — | — | — |
| h200_sxm | `moe_torch_flow_nongated` | shared | trtllm | trtllm:7533 | 0 / 7533 | 0 | — | — | — |
| h200_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:138018 | 0 / 63984 | 4 | — | — | — |
| h200_sxm | `sglang_marlin_moe` | shared | sglang | sglang:32238 | 0 / 16686 | 0 | — | — | — |
| h200_sxm | `vllm_fused_moe` | shared | vllm | vllm:69336 | 0 / 33210 | 3 | — | — | — |
| h200_sxm | `vllm_marlin_int4_moe` | shared | vllm | vllm:10206 | 0 / 9234 | 0 | — | — | — |
| h200_sxm | `vllm_marlin_moe` | shared | vllm | vllm:4709 | 0 / 4124 | 0 | — | — | — |
| h200_sxm | `vllm_mxfp4_moe` | shared | vllm | vllm:1701 | 0 / 1296 | 0 | — | — | — |
| l40s | `moe_torch_flow` | shared | trtllm | trtllm:1960 | 0 / 1960 | 0 | — | — | — |
| l40s | `sglang_fused_moe_triton` | shared | sglang | sglang:39374 | 0 / 33552 | 1 | — | — | — |
| l40s | `sglang_marlin_moe` | shared | sglang | sglang:19035 | 0 / 16686 | 0 | — | — | — |
| l40s | `vllm_fused_moe` | shared | vllm | vllm:12718 | 0 / 12718 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_fused_moe_triton` | shared | sglang | sglang:67982 | 0 / 57792 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_marlin_moe` | shared | sglang | sglang:14661 | 0 / 12717 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_fused_moe` | shared | vllm | vllm:17766 | 0 / 17766 | 0 | — | — | — |

## `nccl_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b300_sxm | `NCCL` | shared | trtllm | trtllm:1512 | 0 / 126 | 0 | — | — | — |

## `scale_matrix_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `torch_ops` | shared | trtllm | trtllm:2069 | 0 / 1649 | 1 | — | — | — |
| b300_sxm | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| gb200 | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| gb300 | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| h100_sxm | `torch_ops` | shared | trtllm | trtllm:2069 | 0 / 1649 | 1 | — | — | — |
| h200_sxm | `torch_ops` | shared | trtllm | trtllm:2069 | 0 / 1649 | 0 | — | — | — |
| l40s | `torch_ops` | shared | trtllm | trtllm:441 | 0 / 441 | 0 | — | — | — |

## `trtllm_alltoall_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| gb200 | `NVLinkOneSided` | shared | trtllm | trtllm:296 | 0 / 148 | 0 | — | — | — |
| gb200 | `NVLinkTwoSided` | shared | trtllm | trtllm:1800 | 0 / 540 | 0 | — | — | — |

## `wideep_context_mla_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `trtllm_mla` | shared | sglang | sglang:500 | 0 / 500 | 0 | — | — | — |
| b300_sxm | `trtllm_mla` | shared | sglang | sglang:1125 | 0 / 625 | 0 | — | — | — |
| gb200 | `trtllm_mla` | shared | sglang | sglang:625 | 0 / 625 | 0 | — | — | — |
| gb300 | `trtllm_mla` | shared | sglang | sglang:625 | 0 / 625 | 0 | — | — | — |
| h100_sxm | `fa3` | shared | sglang | sglang:1552 | 0 / 1052 | 0 | — | — | — |
| h100_sxm | `flashinfer` | shared | sglang | sglang:1577 | 0 / 1077 | 0 | — | — | — |
| h200_sxm | `fa3` | shared | sglang | sglang:960 | 0 / 960 | 0 | — | — | — |
| h200_sxm | `flashinfer` | shared | sglang | sglang:960 | 0 / 960 | 0 | — | — | — |

## `wideep_context_mlp_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| h100_sxm | `deepseek_v3` | shared | sglang | sglang:18 | 0 / 18 | 0 | — | — | — |
| h200_sxm | `deepseek_v3` | shared | sglang | sglang:18 | 0 / 18 | 0 | — | — | — |

## `wideep_context_moe_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deepepmoe` | shared | sglang | sglang:426 | 0 / 426 | 0 | — | — | — |
| b300_sxm | `deepepmoe` | shared | sglang | sglang:426 | 0 / 426 | 0 | — | — | — |
| gb200 | `deepepmoe` | shared | sglang | sglang:426 | 0 / 426 | 0 | — | — | — |
| gb300 | `deepepmoe` | shared | sglang | sglang:426 | 0 / 426 | 0 | — | — | — |
| h100_sxm | `deepepmoe` | shared | sglang | sglang:892 | 0 / 461 | 0 | — | — | — |
| h200_sxm | `deepepmoe` | shared | sglang | sglang:954 | 0 / 492 | 0 | — | — | — |

## `wideep_deepep_ll_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| h100_sxm | `deepep` | shared | sglang | sglang:95 | 0 / 95 | 0 | — | — | — |
| h200_sxm | `deepep` | shared | sglang | sglang:95 | 0 / 95 | 0 | — | — | — |

## `wideep_generation_mla_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `trtllm_mla` | shared | sglang | sglang:528 | 0 / 528 | 0 | — | — | — |
| b300_sxm | `trtllm_mla` | shared | sglang | sglang:1188 | 0 / 660 | 20 | — | — | — |
| gb200 | `trtllm_mla` | shared | sglang | sglang:660 | 0 / 660 | 0 | — | — | — |
| gb300 | `trtllm_mla` | shared | sglang | sglang:660 | 0 / 660 | 0 | — | — | — |
| h100_sxm | `fa3` | shared | sglang | sglang:1680 | 0 / 1152 | 9 | — | — | — |
| h100_sxm | `flashinfer` | shared | sglang | sglang:1732 | 0 / 1204 | 5 | — | — | — |
| h200_sxm | `fa3` | shared | sglang | sglang:1056 | 0 / 1056 | 0 | — | — | — |
| h200_sxm | `flashinfer` | shared | sglang | sglang:1056 | 0 / 1056 | 0 | — | — | — |

## `wideep_generation_mlp_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| h100_sxm | `deepseek_v3` | shared | sglang | sglang:18 | 0 / 18 | 0 | — | — | — |
| h200_sxm | `deepseek_v3` | shared | sglang | sglang:18 | 0 / 18 | 0 | — | — | — |

## `wideep_generation_moe_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deepepmoe` | shared | sglang | sglang:336 | 0 / 336 | 0 | — | — | — |
| b300_sxm | `deepepmoe` | shared | sglang | sglang:336 | 0 / 336 | 0 | — | — | — |
| gb200 | `deepepmoe` | shared | sglang | sglang:336 | 0 / 336 | 0 | — | — | — |
| gb300 | `deepepmoe` | shared | sglang | sglang:336 | 0 / 336 | 0 | — | — | — |
| h100_sxm | `deepepmoe` | shared | sglang | sglang:719 | 0 / 384 | 0 | — | — | — |
| h200_sxm | `deepepmoe` | shared | sglang | sglang:720 | 0 / 384 | 0 | — | — | — |

## `wideep_moe_perf.txt`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `wideep_compute_cutlass` | shared | trtllm | trtllm:4158 | 0 / 4158 | 0 | — | — | — |
| b300_sxm | `wideep_compute_cutlass` | shared | trtllm | trtllm:4158 | 0 / 4158 | 0 | — | — | — |
| gb200 | `wideep_compute_cutlass` | shared | trtllm | trtllm:4158 | 0 / 4158 | 0 | — | — | — |
| gb300 | `wideep_compute_cutlass` | shared | trtllm | trtllm:4158 | 0 / 4158 | 0 | — | — | — |
| h100_sxm | `wideep_compute_cutlass` | shared | trtllm | trtllm:4158 | 0 / 4158 | 0 | — | — | — |
| h200_sxm | `wideep_compute_cutlass` | shared | trtllm | trtllm:4158 | 0 / 4158 | 0 | — | — | — |

## Appendix: all kernel sources

Each row is one distinct `kernel_source` value seen in the corpus, with the union of frameworks, op files, and systems it appears in. Tier is determined by the kernel_source name alone, so a single kernel_source has one tier across the whole corpus.


### `shared` (48 kernel sources)

| kernel_source | frameworks | op files | systems | rows |
|---|---|---|---|---|
| `causal_conv1d_fn` | sglang, trtllm | gdn_perf.txt, mamba2_perf.txt | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 18,211 |
| `causal_conv1d_update` | sglang, trtllm, vllm | gdn_perf.txt, mamba2_perf.txt | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 2,332 |
| `chunk_gated_delta_rule` | trtllm | gdn_perf.txt | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 4,849 |
| `deepep` | sglang | wideep_deepep_ll_perf.txt | h100_sxm, h200_sxm | 190 |
| `deepepmoe` | sglang | wideep_context_moe_perf.txt, wideep_generation_moe_perf.txt | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 6,333 |
| `deepgemm` | trtllm | gemm_perf.txt, moe_perf.txt | b200_sxm, b300_sxm, gb200, gb300 | 161,103 |
| `deepseek_v3` | sglang | wideep_context_mlp_perf.txt, wideep_generation_mlp_perf.txt | h100_sxm, h200_sxm | 72 |
| `dsa_nsa` | sglang | dsa_context_module_perf.txt, dsa_generation_module_perf.txt | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 23,772 |
| `fa3` | sglang | wideep_context_mla_perf.txt, wideep_generation_mla_perf.txt | h100_sxm, h200_sxm | 5,248 |
| `flash_attention` | sglang | context_attention_perf.txt, context_mla_perf.txt, generation_attention_perf.txt, generation_mla_perf.txt | a100_sxm, h100_sxm, h200_sxm, l40s | 249,719 |
| `flashinfer` | sglang | wideep_context_mla_perf.txt, wideep_generation_mla_perf.txt | h100_sxm, h200_sxm | 5,325 |
| `fused_recurrent_gated_delta_rule` | sglang, trtllm | gdn_perf.txt | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,507 |
| `moe_torch_flow` | trtllm | moe_perf.txt | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s | 91,612 |
| `moe_torch_flow_cutlass` | trtllm | moe_perf.txt | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 596,454 |
| `moe_torch_flow_min_latency` | trtllm | moe_perf.txt | b200_sxm, gb200 | 50,157 |
| `moe_torch_flow_nongated` | trtllm | moe_perf.txt | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 70,227 |
| `NCCL` | trtllm | nccl_perf.txt | b300_sxm | 1,512 |
| `NVLinkOneSided` | trtllm | trtllm_alltoall_perf.txt | gb200 | 296 |
| `NVLinkTwoSided` | trtllm | trtllm_alltoall_perf.txt | gb200 | 1,800 |
| `sglang` | sglang | gemm_perf.txt | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,745,481 |
| `SGLang_CustomAllReduce_eager` | sglang | custom_allreduce_perf.txt | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,058 |
| `SGLang_CustomAllReduce_graph` | sglang | custom_allreduce_perf.txt | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,058 |
| `sglang_flashinfer_cutedsl_moe` | sglang | moe_perf.txt | b200_sxm, b300_sxm, gb200, gb300 | 236,422 |
| `sglang_fused_moe_triton` | sglang | moe_perf.txt | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 936,086 |
| `sglang_marlin_moe` | sglang | moe_perf.txt | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 228,443 |
| `sglang_mhc` | sglang | mhc_module_perf.txt | rtx_pro_6000_server | 139 |
| `torch_flow` | trtllm | context_attention_perf.txt, gemm_perf.txt, generation_attention_perf.txt | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s | 1,719,114 |
| `torch_ops` | trtllm | computescale_perf.txt, scale_matrix_perf.txt | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s | 23,061 |
| `triton` | sglang | context_attention_perf.txt, context_mla_perf.txt, generation_attention_perf.txt, generation_mla_perf.txt | rtx_pro_6000_server | 56,870 |
| `trt_flow_/smooth_quant_gemm_L96/PLUGIN_V2_SmoothQuantGemm_0` | trtllm | gemm_perf.txt | a100_sxm, h200_sxm, l40s | 21,336 |
| `trt_flow_/weight_only_quant_matmul_L257/PLUGIN_V2_WeightOnlyQuantMatmul_0` | trtllm | gemm_perf.txt | a100_sxm, h200_sxm, l40s | 42,672 |
| `TRTLLM` | trtllm | custom_allreduce_perf.txt | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s | 1,656 |
| `trtllm_mha` | sglang | context_attention_perf.txt, generation_attention_perf.txt | b200_sxm, b300_sxm, gb200, gb300 | 347,191 |
| `trtllm_mla` | sglang | context_mla_perf.txt, generation_mla_perf.txt, wideep_context_mla_perf.txt, wideep_generation_mla_perf.txt | b200_sxm, b300_sxm, gb200, gb300 | 67,735 |
| `vLLM_custom_eager` | vllm | custom_allreduce_perf.txt | a100_sxm, b200_sxm, b300_sxm, b60, gb200, gb300, h100_sxm, h200_sxm, l40s | 805 |
| `vLLM_custom_graph` | vllm | custom_allreduce_perf.txt | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s | 736 |
| `vllm_default` | vllm | gemm_perf.txt | a100_sxm, b200_sxm, b300_sxm, b60, gb200, gb300, h100_sxm, h200_sxm, l40s | 948,633 |
| `vllm_flash_attn` | vllm | context_attention_perf.txt, generation_attention_perf.txt | a100_sxm, b60, h100_sxm, h200_sxm, l40s | 187,414 |
| `vllm_flash_attn_mla` | vllm | context_mla_perf.txt, generation_mla_perf.txt | h100_sxm, h200_sxm | 8,730 |
| `vllm_flashinfer` | vllm | context_attention_perf.txt, generation_attention_perf.txt | b200_sxm, b300_sxm, gb200, gb300, l40s | 194,796 |
| `vllm_flashinfer_trtllm_moe_fp4` | vllm | moe_perf.txt | b200_sxm, b300_sxm, gb200, gb300 | 59,616 |
| `vllm_flashmla` | vllm | context_mla_perf.txt, generation_mla_perf.txt | h100_sxm, h200_sxm | 8,460 |
| `vllm_fused_moe` | vllm | moe_perf.txt | a100_sxm, b200_sxm, b300_sxm, b60, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 367,818 |
| `vllm_marlin_int4_moe` | vllm | moe_perf.txt | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 61,236 |
| `vllm_marlin_moe` | vllm | moe_perf.txt | gb200, gb300, h100_sxm, h200_sxm | 18,863 |
| `vllm_mxfp4_moe` | vllm | moe_perf.txt | h200_sxm | 1,701 |
| `vllm_triton_mla` | vllm | context_mla_perf.txt, generation_mla_perf.txt | a100_sxm, l40s | 4,526 |
| `wideep_compute_cutlass` | trtllm | wideep_moe_perf.txt | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 24,948 |

### `shared_fallback` (1 kernel sources)

| kernel_source | frameworks | op files | systems | rows |
|---|---|---|---|---|
| `default` | sglang, trtllm, vllm | context_mla_perf.txt, dsa_context_module_perf.txt, dsa_generation_module_perf.txt, generation_mla_perf.txt, mla_bmm_perf.txt, mla_context_module_perf.txt, mla_generation_module_perf.txt | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 318,015 |
