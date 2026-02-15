# Phase 1: 模型架构分析

## 目标

理解新模型的配置参数，让 aiconfigurator 能识别并正确解析该模型。

## 步骤

### 1. 获取并分析 config.json

```bash
# 从 HuggingFace 下载
wget https://huggingface.co/<org>/<model>/raw/main/config.json

# 或使用本地模型目录
ls /path/to/model/config.json
```

重点关注以下字段：
- `architectures` — 模型架构标识（如 `["NewModelForCausalLM"]`）
- `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`
- `intermediate_size`, `vocab_size`, `max_position_embeddings`
- MoE 相关: `num_experts_per_tok`, `num_local_experts`/`n_routed_experts`, `moe_intermediate_size`
- 特殊字段: 任何不在上述列表中的新字段

### 2. 对比已知模型，识别新增字段

与已有模型的 config.json 对比，标记出新增或不同的字段：

```json
{
  "architectures": ["NewModelForCausalLM"],
  "num_hidden_layers": 32,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,        // 标准字段
  "new_specific_field_1": 64,      // ← 新增字段
  "new_specific_field_2": 2048     // ← 新增字段
}
```

### 3. 添加 ModelFamily（如需要新 family）

**文件**: `src/aiconfigurator/sdk/common.py`

如果新模型需要全新的 Model 类（Situation S3），需要在 `ModelFamily` 集合中添加新值：

```python
# common.py (约第 280 行)
ModelFamily = {"GPT", "LLAMA", "MOE", "DEEPSEEK", "NEMOTRONNAS", "NEMOTRONH", "NEW_MODEL"}  # ← 添加
```

> **注意**: 如果新模型只是已有 family 的变体（如另一个 Llama 架构），不需要这步。

### 4. 添加架构映射

**文件**: `src/aiconfigurator/sdk/common.py`

在 `ARCHITECTURE_TO_MODEL_FAMILY` 字典中添加映射：

```python
# common.py (约第 281 行)
ARCHITECTURE_TO_MODEL_FAMILY = {
    "LlamaForCausalLM": "LLAMA",
    "Qwen2ForCausalLM": "LLAMA",
    # ... 已有映射 ...
    "NewModelForCausalLM": "NEW_MODEL",  # ← 添加
}
```

### 5. 添加 Config Dataclass（如有特殊配置）

**文件**: `src/aiconfigurator/sdk/common.py`

只有当模型有**独特的结构化配置**时才需要（参考 `NemotronHConfig`）。
大多数模型不需要这步，参数直接在解析时提取。

```python
# common.py — 仅当有多个相关联的特殊参数时使用
@dataclass(frozen=True)
class NewModelConfig:
    """Configuration for NewModel-specific parameters."""
    new_field_1: int
    new_field_2: int
    # ...
```

### 6. 添加解析逻辑

**文件**: `src/aiconfigurator/sdk/utils.py`

在 `_parse_hf_config_json()` 函数中添加新模型的参数解析：

```python
# utils.py — _parse_hf_config_json() 函数内 (约第 466 行之后)

# 现有的 NemotronH 处理代码之后：
extra_params = None
if architecture == "NemotronHForCausalLM":
    extra_params = common.NemotronHConfig(...)
elif architecture == "NewModelForCausalLM":                    # ← 添加
    extra_params = common.NewModelConfig(                       # ← 添加
        new_field_1=config["new_field_1"],                      # ← 添加
        new_field_2=config.get("new_field_2", 2048),            # ← 添加 (有默认值)
    )                                                           # ← 添加
```

> **重要**: 解析后的数据通过 `extra_params` 字段传递到 `models.py`。
> 标准字段（layers, hidden_size, n, n_kv 等）已经由 `_parse_hf_config_json` 自动解析。
> 只需要处理**新增的非标准字段**。

### 7. 在 models.py 中接收新参数（如需新 Model 类）

**文件**: `src/aiconfigurator/sdk/models.py`

在 `get_model()` 函数中添加新模型的实例化逻辑。参考已有模型：

```python
# models.py — get_model() 函数内

# 参考 DeepSeekModel 或 NemotronHModel 的初始化方式
if model_family == "NEW_MODEL":
    model = NewModel(
        topk, num_experts, moe_inter_size,  # 如果是 MoE
        model_path, model_family, architecture, layers, n, n_kv, d,
        hidden, inter, vocab, context, model_config,
    )
    if extra_params is not None:
        model.set_extra_config(extra_params)  # 传入特殊配置
```

## 验证

```bash
# 验证 CLI 能识别新模型
aiconfigurator cli support --model-path <new-model-path-or-hf-id> --system h200_sxm --backend trtllm

# 预期输出：应该能识别模型架构，即使还没有性能数据
# 如果报 "architecture is not supported" 说明映射还没加好
# 如果报 "data not found" 说明映射成功，但还需要采集数据（Phase 5）
```

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `architecture is not supported` | `ARCHITECTURE_TO_MODEL_FAMILY` 缺少映射 | 添加映射到 `common.py` |
| `num_key_value_heads is None` | config.json 中该字段为 null | `utils.py` 中用 `config.get(field) or 0` |
| `extra_params` 无法传递 | 新 dataclass 未被 `_parse_hf_config_json` 返回 | 检查 `extra_params` 赋值逻辑 |
| 解析出的参数有误 | config.json 字段名与代码中不匹配 | 仔细检查 HuggingFace config 的 key 拼写 |
