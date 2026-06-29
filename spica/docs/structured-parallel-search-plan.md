<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Spica 结构化并行配置搜索计划

## 目标

将 Vizier 当前使用的 opaque `parallel_config_index` 替换为结构化搜索参数，让
Bayesian model 能看到 GPU allocation、单 engine GPU 数量及并行模式。现有 parallel
enumerator 继续作为合法性来源；Vizier 提议的点在 replay 前确定性吸附到最近的 valid
config。

Agg 与 disagg 继续使用独立 study，backend 仍是 study 内的 categorical knob。YAML、
unrolled sample、deployment 和 replay 接口均不改变。

## 搜索空间与默认值

### Agg

| Knob | Vizier 类型 | 候选或范围 | 默认值 |
|---|---|---|---|
| `backend` | Categorical | 当前 branch 中可行的用户配置 backends | 用户配置顺序中的第一个可行 backend；当前配置默认是 `vllm` |
| `used_gpu_ratio` | Double | valid pool 中 `used_gpus / gpu_budget` 的最小值到最大值 | 最大可行值，通常为 `1.0` |
| `agg_num_gpus_per_engine_target` | Discrete + Log | valid pool 中所有 agg `gpus_per_worker` 的并集 | log 空间中点对应的最近候选；`[1,2,4,8,16]` 时为 `4` |
| `agg_attention_mode` | Categorical | `tp`, `dp` | `tp`；不存在时使用唯一可行值 |
| `agg_ffn_mode` | Categorical | MoE 模型为 `tp`, `ep`；dense 不创建该参数 | `ep`；不存在时使用唯一可行值 |

### Disagg

| Knob | Vizier 类型 | 候选或范围 | 默认值 |
|---|---|---|---|
| `backend` | Categorical | 当前 branch 中可行的用户配置 backends | 用户配置顺序中的第一个可行 backend；当前配置默认是 `vllm` |
| `used_gpu_ratio` | Double | valid pool 中 `used_gpus / gpu_budget` 的最小值到最大值 | 最大可行值，通常为 `1.0` |
| `prefill_gpu_share` | Double | valid pool 中 `prefill_total_gpus / used_gpus` 的最小值到最大值 | `0.5` clamp 到范围内 |
| `prefill_num_gpus_per_engine_target` | Discrete + Log | valid prefill shapes 的 `gpus_per_worker` 并集 | log 空间中点对应的最近候选；`[1,2,4,8,16]` 时为 `4` |
| `decode_num_gpus_per_engine_target` | Discrete + Log | valid decode shapes 的 `gpus_per_worker` 并集 | log 空间中点对应的最近候选；`[1,2,4,8,16]` 时为 `4` |
| `prefill_attention_mode` | Categorical | `tp`, `dp` | `tp`；不存在时使用唯一可行值 |
| `decode_attention_mode` | Categorical | `tp`, `dp` | `tp`；不存在时使用唯一可行值 |
| `prefill_ffn_mode` | Categorical | MoE 模型为 `tp`, `ep`；dense 不创建该参数 | `ep`；不存在时使用唯一可行值 |
| `decode_ffn_mode` | Categorical | MoE 模型为 `tp`, `ep`；dense 不创建该参数 | `ep`；不存在时使用唯一可行值 |

`num_gpus_per_engine_target` 的动态默认值按以下规则计算：

```text
target = exp((log(min_candidate) + log(max_candidate)) / 2)
default = log distance 最接近 target 的合法候选，平局时取较小值
```

其他规则：

- 当前 dense model 固定 attention TP，不创建 FFN mode。
- `pp=1`、replicas、实际 GPU 数量、`tp`、`attention_dp`、`moe_tp`、`moe_ep`
  和 strategy 继续作为 derived fields。
- 单候选参数不加入 Vizier search space，直接作为 constant 注入。
- 默认值只决定初始建议和无历史数据时的中心，不绕过 valid-point projection。

## 用户 Pin 与自定义候选菜单

保留现有 `parallel_configs` YAML schema，不要求用户迁移配置。在
`SPICA_PARALLEL_ENCODING=structured` 下，其行为为：

| `parallel_configs` | 行为 |
|---|---|
| 省略或 `[]` | 使用当前 enumerator 自动生成完整 valid pool，并搜索 structured parallel dimensions |
| 只有一项 | 严格 pin 到该 parallel config，不创建任何 parallel Vizier 参数，也不执行 projection |
| 多项 | 作为用户定义的 valid menu；projector 只能从这些条目中选择 |

Pin 或自定义 menu 时继续要求 `deployment_mode` 只有一个值。每个条目仍通过现有
model、hardware、backend、GPU budget 和 KV-capacity 校验；任何 backend 都无法运行的
显式条目继续 fail fast。

Agg 保持当前扁平写法。省略的 `attention_dp`、`moe_tp`、`moe_ep` 和 `pp` 默认为 `1`，
`replicas` 默认为 `1`：

```yaml
search_space:
  deployment_mode: [agg]
  backend: [vllm]
  gpu_budget: 32
  parallel_configs:
    - {tp: 4, moe_ep: 4, replicas: 8}
```

Disagg 保持 `prefill` 和 `decode` 两个子配置的写法：

```yaml
search_space:
  deployment_mode: [disagg]
  backend: [sglang]
  gpu_budget: 32
  parallel_configs:
    - prefill: {tp: 8, moe_ep: 8, replicas: 1}
      decode: {tp: 1, attention_dp: 8, moe_ep: 8, replicas: 2}
```

单条 pin 时 backend 仍可包含多个候选，但 `enumerate_branches` 只保留支持该 config 的
backends。Router、planner、batching 和 workload knobs 仍可正常搜索；只有 parallel config
被固定。多个条目时，结构化参数的候选范围和默认值只从用户 menu 计算，而不是从自动
enumerator 的完整结果计算。

## Valid-Point Projection

新增独立 projector，输入 `BranchSpace.parallel_configs` 与 `supported_backends`。`backend`
继续作为普通 Vizier categorical knob，并在 projection 时作为硬过滤条件；projector 为每个
合法配置生成以下结构化 features：

```text
used_gpu_ratio
prefill_gpu_share                 # disagg only
per-role num_gpus_per_engine
per-role attention mode
per-role FFN mode                 # MoE only
```

每个 Vizier suggestion 按以下固定顺序投影：

1. 按请求的 backend 硬过滤，绝不改变 backend。
2. 优先保留完全匹配的 attention 和 FFN modes。
3. 若该 mode family 没有合法配置，选择 mode mismatch 数量最少的 family。
4. 在剩余配置中最小化归一化数值距离：

```text
d =
    delta(used_gpu_ratio)^2
  + delta(prefill_gpu_share)^2
  + delta(log2(prefill_num_gpus_per_engine))^2
  + delta(log2(decode_num_gpus_per_engine))^2
```

Agg 忽略 disagg-only 项。每个数值维度先按 valid pool 的实际范围归一化到 `[0,1]`；
零跨度维度的距离为 `0`。初始权重均为 `1`。

5. 使用完整 parallel tuple 的字典序作为稳定 tie-break。

相同 requested point 必须始终映射到相同 actual config，不允许改为“最近的未测试配置”。
Replay 使用 actual config，Vizier 仍以原始 requested parameters 接收 measurement。

Trial metadata 使用根 key `spica_projection` 记录：

```text
requested_features
actual_features
projection_distance
mode_projected
actual_parallel_config
```

结构性非法点不再调用 `observe_infeasible`；该接口只用于 replay、AIC 或 candidate build
的真实失败。现有 backend 和 GPU-budget gate 暂时保留为防御性检查，命中时视为
projector bug。

## 重复点与 Sweep Budget

Sweep budget 按成功完成的 unique replay configurations 计数，不按 Vizier trials 计数。
只有完整 sample 相同才视为重复；parallel config 相同但 batching、router、planner、
backend 或 workload 不同，仍需独立 replay。

处理流程：

1. Project suggestion，并为完整 sample 生成 canonical cache key。
2. 未命中 cache 时执行 replay；只缓存成功结果。
3. 命中 cache 时不执行 replay，使用缓存 metrics 完成当前 Vizier trial。
4. Cache hit 不加入最终 candidates，也不消耗 unique replay budget；继续请求 suggestion，
   直到本轮获得 `candidates_per_round` 个新配置。
5. 每轮最多额外请求 `10 * candidates_per_round` 个 trials。达到上限仍无法补齐时，提前
   结束 sweep 并报告 projection stalled，不使用重复点填充预算。

Cache key 必须包含 selection、actual parallel config、workload 和 load-predictor preset。
Timeout、candidate build error 和 replay error 不缓存，后续 suggestion 可以重试。

## 集成与兼容

- Sampler 使用 structured parameters 替代 `parallel_config_index`，但
  `Suggestion.parallel_config` 仍携带 projector 返回的现有 dataclass。
- `unroll_sample`、`build_deployment`、Candidate schema 和输出格式不变。
- Pinned 或 custom `parallel_configs` 直接作为 projector pool；单一配置自然退化为常量。
- 保留现有 `parallel_configs` 配置格式和单项 pin、多项 custom-menu 语义，不增加新的
  YAML parallel pin schema。
- 增加私有环境变量 `SPICA_PARALLEL_ENCODING=structured|opaque`，初始默认保持
  `opaque` 用于 A/B；验证通过后改为 `structured` 并移除临时开关。
- 保留请求值和实际值的结构化日志，支持统计 projection 和 collision 行为。

## 测试与验收

单元和 property tests 覆盖：

- 合法点投影到自身，投影结果始终属于原 valid pool。
- 单项 `parallel_configs` 不创建 parallel Vizier dimensions，并始终返回 pinned config。
- 多项 `parallel_configs` 的 projection 永远不会离开用户提供的 custom menu。
- Backend 永远保持一致。
- Agg 和 disagg 均不超过 GPU budget，disagg 两个 role 都至少有一个 replica。
- Dense 不生成 FFN mode，`gpus_per_worker=1` 使用 canonical mode。
- Unsupported mode family 的 fallback 确定且可复现。
- 相同输入在不同进程和调用顺序下产生相同 projection。
- 随机生成大量 latent suggestions，不产生结构性 infeasible replay。
- Cache hit 正确 tell Vizier，但不重复 replay、不重复输出 Candidate、不消耗 unique budget。
- 达到 duplicate retry 上限时 sweep 可控地提前退出。
- Vizier default value、numeric discrete、categorical、metadata 和 ask/tell 正常 round-trip。

使用 GLM-5 disagg Pareto sweep 对比 opaque 与 structured encoding，保持 Replay/AIC 版本、
workload、目标、并发数和 unique replay budget 相同。记录：

```text
hypervolume
SA frontier coverage
unique actual parallel configs
projection rate
mode projection rate
average projection distance
cache-hit and collision rate
backend and concurrency coverage
```

验收条件：

- 结构性 infeasible replay 数量为零。
- 不再出现绝大多数 trial 集中到单个 parallel config。
- Unique config coverage 明显高于 opaque baseline。
- Hypervolume 不低于当前 sweep。
- Mode projection 保持低频；主要吸附发生在 GPU size、replicas 和 allocation 上。

如果 mode projection 或重复映射仍然较多，下一阶段改为 custom candidate-set Vizier
designer，使 GP 直接用 actual valid-config features 建模，并只在 valid pool 上计算
acquisition。
