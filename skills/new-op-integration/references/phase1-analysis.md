# Phase 1: 模型架构分析模板

## 步骤

### 1. 获取配置

```bash
# HuggingFace
wget https://huggingface.co/<org>/<model>/raw/main/config.json

# 或本地模型
cat /path/to/model/config.json
```

### 2. 识别新字段

对比已知模型，找出新增配置：

```json
{
  "architectures": ["NewModelForCausalLM"],
  "known_field": "...",
  "new_field_1": 64,      // 新增
  "new_field_2": 2048     // 新增
}
```

### 3. 添加架构映射

```python
# common.py
ARCH_MAPPING = {
    "ExistingModel": ModelFamily.EXISTING,
    "NewModelForCausalLM": ModelFamily.NEW,  # 添加
}
```

### 4. 添加 Config Dataclass

```python
@dataclass
class NewModelConfig:
    new_field_1: int = 64
    new_field_2: int = 2048
```

### 5. 添加解析逻辑

```python
# utils.py
def _parse_hf_config_json(config: dict) -> dict:
    # ... existing parsing ...
    
    # 新模型参数
    if "new_field_1" in config:
        extra_params["new_field_1"] = config["new_field_1"]
        extra_params["new_field_2"] = config.get("new_field_2", 2048)
    
    return {..., "extra_params": extra_params}
```

## 验证

```bash
aiconfigurator cli support --model-path <new-model> --system h200_sxm --backend trtllm
```
