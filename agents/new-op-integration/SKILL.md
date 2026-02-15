---
name: new-op-integration
description: |
  Integrate new operations into aiconfigurator to support new models. 
  Use when adding support for new model architectures or new attention/compute mechanisms.
  This skill guides through: Situation Assessment → Analysis → Mock → Profile → Integration → Collection.
---

# New Operation Integration Skill

> **Official reference**: `docs/add_a_new_model.md` — read it first for the overall decision tree.

## Step 0: 判断 Situation（必须先做）

在开始前，必须先判断属于以下哪种情况。不同情况所需的工作量差别很大。

| Situation | 描述 | 需要的 Phase |
|-----------|------|-------------|
| **S1: 简单变体** | 模型架构已存在（如新的 Llama 变体），只需添加 architecture 映射 | Phase 1 (部分) |
| **S2: 需新数据** | 架构映射已存在，但 Op 参数不同（如新 MoE 模型有不同 num_experts/topk） | Phase 1 + 5 |
| **S3: 需新 Op** | 模型有全新的计算算子（如 Convolution、新的 Attention 机制），需要完整流程 | Phase 1 → 5 全部 |

**判断方法**:

```
1. 读取目标模型的 config.json
2. 检查 config["architectures"][0] 是否在 ARCHITECTURE_TO_MODEL_FAMILY 中
   → 如果在，可能是 S1 或 S2
3. 分析模型结构，看是否有 aiconfigurator 不支持的计算单元
   → 没有不支持的 Op → S1 或 S2
   → 有不支持的 Op → S3
4. 对于 S2，检查现有 perf 数据是否覆盖新模型的参数组合
   → 已覆盖 → S1
   → 未覆盖 → S2
```

---

## Key File Paths（速查表）

| 文件 | 路径 | 作用 |
|------|------|------|
| 架构映射 & 枚举 | `src/aiconfigurator/sdk/common.py` | `ModelFamily` 枚举, `ARCHITECTURE_TO_MODEL_FAMILY` 字典 |
| 配置解析 | `src/aiconfigurator/sdk/utils.py` | `_parse_hf_config_json()`, `get_model_config_from_model_path()` |
| 操作定义 | `src/aiconfigurator/sdk/operations.py` | `Operation` 基类及所有子类 |
| 性能数据库 | `src/aiconfigurator/sdk/perf_database.py` | 数据加载 + 查询方法 |
| 模型定义 | `src/aiconfigurator/sdk/models.py` | `BaseModel` 及各 Model 子类 |
| 性能结果 | `src/aiconfigurator/sdk/performance_result.py` | `PerformanceResult` (float-like, 含 power) |
| 性能数据目录 | `src/aiconfigurator/systems/data/{system}/{backend}/{version}/` | `*_perf.txt` 数据文件 |
| 测试用例定义 | `collector/common_test_cases.py` | 标准 benchmark 参数组合 |
| 采集框架 | `collector/helper.py` | `benchmark_with_power()`, `log_perf()` |
| 采集入口 | `collector/collect.py` | 统一采集入口 |
| 后端采集脚本 | `collector/{backend}/collect_{op}.py` | 各 backend 的采集实现 |

---

## Phase 1: 模型架构分析

**目标**: 理解模型配置，添加架构识别支持

**详见**: `agents/new-op-integration/references/phase1-analysis.md`

**完成标志**: `aiconfigurator cli support --model-path <model> --system h200_sxm` 能正确识别模型

---

## Phase 2: Mock Layer 构建

**目标**: 创建独立可运行的算子层，用于后续 profiling 和 data collection

**详见**: `agents/new-op-integration/references/phase2-mock-layer.md`

**完成标志**: Mock layer 能独立 forward，输入输出 shape 合理

---

## Phase 3: nsys Profile 对齐

**目标**: 验证 mock layer 的 kernel 行为与完整模型 E2E 推理一致

**详见**: `agents/new-op-integration/references/phase3-nsys-alignment.md`

**完成标志**: Kernel 名称匹配，latency 在合理范围内 (< 2x 差距)

---

## Phase 4: SDK 集成

**目标**: 将新算子添加到 aiconfigurator 建模体系中

**详见**: `agents/new-op-integration/references/phase4-integration.md`

**完成标志**: `operation.query(db, x=...) ` 能返回 `PerformanceResult`；CLI 能输出完整建模结果

---

## Phase 5: 性能数据采集

**目标**: 收集覆盖各种 batch/seq 组合的 benchmark 数据

**详见**: `agents/new-op-integration/references/phase5-collection.md`

**完成标志**: 生成 `*_perf.txt` 数据文件，数据点数量符合预期

---

## 完整集成 Checklist

完成所有 Phase 后，做最终验证：

```
□ common.py: ModelFamily 枚举值已添加（如需要）
□ common.py: ARCHITECTURE_TO_MODEL_FAMILY 映射已添加
□ utils.py: config.json 解析逻辑已添加
□ operations.py: 新 Operation 子类已实现（query + get_weights）
□ perf_database.py: 数据加载 + 查询方法已实现
□ models.py: 新 Model 类已实现（或修改已有类）
□ systems/data/: 性能数据文件已放置
□ collector/: 采集脚本可正常运行
□ CLI 验证: aiconfigurator cli support/default 正常运行
□ 测试: 现有测试未被破坏
```

---

## Lessons Learned（实战经验）

以下经验来自 DeepSeek-V3.2 DSA 集成，适用于任何新 Op。

### 1. Op 粒度必须与已有 Op 对齐

**问题**: 新 Op 如果包含了模型中已有的独立 Op（如 GEMMs），会导致双重计算。
**规则**: 参考最相似的已有 Op（如 MLA），让新 Op 的采集粒度与之一致。
- 如果 MLA 只采 attention kernel，新 Op 也应只采 attention kernel
- 如果新 Op 不得不采更粗粒度（整个 attention block），必须在 `models.py` 中去掉被包含的 GEMMs
- 验证方式：用 `cli default` 对比新旧模型同配置下的 static latency，检查每层延迟是否合理

### 2. 数据格式必须与已有 Op 完全一致

**问题**: `perf_database.py` 的插值和查询基础设施对 dict 结构有严格假设。
**规则**: 新 Op 的 load/query/interpolation 必须严格复制最相似的已有 Op 的模式：
- **dict 嵌套层级**: MLA context 是 5 层 `data[quant][kv_dtype][num_heads][s][b]`
- **defaultdict leaf**: 最内层必须是 `defaultdict()`（不是 `defaultdict(dict)`），否则 `try/except KeyError` 的去重逻辑会失败——defaultdict(dict) 的 `data[key]` 不会抛 KeyError，而是静默创建空 dict
- **插值维度顺序**: context 是 `(x=num_heads, y=s, z=b)`，generation 是 `(x=num_heads, y=b, z=s)`，跟 dict 嵌套顺序一一对应
- **`_interp_3d` 返回 dict**: 返回值是 `{"latency": ..., "energy": ...}`，需要手动包装成 `PerformanceResult`
- **TP 模拟**: 单卡采集，用 `local_num_heads = global_heads // tp_size` 模拟 TP，`Mapping(world_size=1, tp_size=1)`

### 3. 查询不要 fallback 到不同粒度的 Op

**问题**: 如果新 Op 的粒度跟 fallback Op 不同，fallback 会给出错误的估算。
**规则**: 如果新 Op 数据不可用，应该抛 `PerfDataNotAvailableError` 而不是静默 fallback 到粒度不同的 Op。只有粒度完全一致时才允许 fallback。

### 4. FP8/量化支持需要在框架层面验证

**问题**: 模型 config 可能自动推断出 FP8 量化，但框架的新 Op 实现可能不支持。
**规则**:
- 在 collector 中实际测试 FP8 KV cache 能否跑通（`try/except`）
- 如果不支持，在 `models.py` 中强制 override 到支持的 dtype，并打 log
- 检查不同 SM 版本的支持情况（如 SM90 vs SM100+ 可能走不同的 kernel 路径）

### 5. Phase 3 对齐的正确做法

**要做的**: 从完整 decoder layer 中取出新 Op 对应的子模块（如 `decoder.self_attn`），跟 collector 独立创建的 mock layer 用相同输入做 benchmark，对比 latency。
**不是**: 只对比 kernel name 列表（必要但不充分）。
**验证标准**: 两者 latency 差异 < 1%，且 CUDA graph capture 行为一致。

### 6. 用 static mode 做分层对比

发现建模异常时，用 `op.query(db, ...)` 逐 Op 对比新旧模型的每层延迟，快速定位哪个 Op 的值不合理。参考命令：
```python
for op in model.context_ops:
    r = op.query(db, x=b*s, batch_size=b, s=s, prefix=0, beam_width=1)
    print(f"{op._name}: {float(r)/op._scale_factor:.4f} ms/layer")
```
