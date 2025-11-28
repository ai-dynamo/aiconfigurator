# AI Generator - Refactored Architecture

This document describes the refactored AI generator architecture that provides better separation of concerns while maintaining full backward compatibility with the original API.

## Architecture Overview

The new architecture separates concerns into three main modules:

### 1. API Layer (`api.py`)
- **Purpose**: Unified parameter collection and normalization
- **Key Functions**:
  - `collect_generator_params()`: Aggregates parameters from different sources
  - `generate_config_from_input_dict()`: Generates configuration from dictionaries
  - `generate_backend_config()` / `generate_backend_artifacts()`: Backend-specific rendering helpers
  - CLI argument parsing and parameter extraction utilities

### 2. Configuration Assets (`config/`)
- **Purpose**: Data-only inputs that describe schemas, mappings, and templates
- **Key Components**:
  - `deployment_config.yaml`: Unified parameter schema + defaults metadata
  - `backend_config_mapping.yaml`: Backend flag translation table
  - `backend_templates/`: Jinja2 templates for each backend (e.g., `trtllm/`, `vllm/`)

### 3. Rendering Module (`rendering/`)
- **Purpose**: Code that interprets schemas/templates and generates backend artifacts
- **Key Components**:
  - `schemas.py`: Default/value resolution on top of `deployment_config.yaml`
  - `engine.py`: Parameter mapping + template rendering helpers
  - `rule_engine.py`: Evaluates `.rule` DSL files from `rule_plugin/`

### 4. Compatibility Layer (`compat_api.py`)
- **Purpose**: Maintains backward compatibility with original API
- **Key Classes**:
  - `DynamoConfig`: User-provided override configuration
  - `RuntimeView`: Runtime properties for generation
  - `ModeConfig`: Configuration for specific serving modes
  - `GeneratorContext`: Validated context for backend generators
  - `_GenerateAPI`: Main compatibility API class

## Key Features

### Multiple Input Methods
The new architecture supports three different input methods:

1. **Object-based API**: Direct configuration objects
```python
from api import generate_config_from_dicts
from config import ModelConfig, K8sConfig, RuntimeEnvConfig

model_config = ModelConfig(
    model_path="meta-llama/Llama-2-7b-hf",
    served_model_name="llama-2-7b"
)

k8s_config = K8sConfig(
    name_prefix="trtllm",
    mode="disagg"
)

params = generate_config_from_dicts(
    model_config=model_config,
    k8s_config=k8s_config,
    # ... other parameters
)
```

2. **Dictionary-based API**: Configuration as dictionaries
```python
params = generate_config_from_dicts(
    model_config={
        "model_path": "meta-llama/Llama-2-7b-hf",
        "served_model_name": "llama-2-7b"
    },
    k8s_config={
        "name_prefix": "trtllm",
        "mode": "disagg"
    },
    # ... other parameters
)
```

3. **CLI Interface**: Command-line parameter parsing

```bash
# Render backend flag mappings for every detected worker role
python -m aic_generator.main render-config \
  --backend trtllm \
  --config examples/sample.yaml \
  --role all

# Render artifacts by mixing YAML + inline overrides (dotted paths)
python -m aic_generator.main render-artifacts \
  --backend trtllm \
  --config examples/sample.yaml \
  --set Workers.prefill.tensor_parallel_size=4 \
  --output ./out
```

- `--config` points to a unified YAML input.
- `--set` accepts dotted keys (`ServiceConfig.model_path`, `Workers.prefill.tensor_parallel_size`, etc.) and merges inline overrides before validation.
- `render-config` prints a JSON object keyed by worker role (`prefill`, `decode`, `agg`), each containing the backend-specific flags returned by `generate_backend_config`.

### Backward Compatibility
The architecture maintains full backward compatibility with the original API through the compatibility layer:

```python
# Original API calls continue to work unchanged
from aiconfigurator.generator.api import generate_backend_config

artifacts = generate_backend_config.from_runtime(
    cfg={"model_name": "llama-2-7b", "serving_mode": "agg"},
    backend="vllm",
    version="0.5.0",
    overrides=DynamoConfig({"tp": 2, "bs": 32})
)
```

## File Structure

```
aic_generator/
├── api.py                          # Main API layer
├── compat_api.py                   # Compatibility layer
├── main.py                         # CLI entry point
├── config/                         # Data-only inputs (no Python)
│   ├── backend_config_mapping.yaml
│   ├── deployment_config.yaml
│   └── backend_templates/
│       ├── trtllm/
│       ├── vllm/
│       └── sglang/
├── rendering/                      # Schema + template rendering logic
│   ├── __init__.py
│   ├── engine.py
│   ├── rule_engine.py
│   └── schemas.py
├── rule_plugin/                    # DSL files consumed by rule_engine
│   ├── trtllm.rule
│   ├── vllm.rule
│   └── sglang.rule
└── ...
```

## Testing

The architecture includes comprehensive tests:

1. **Compatibility Tests** (`test_compatibility.py`):
   - Verifies original API calls work unchanged
   - Tests `report_and_save.py` integration
   - Validates backward compatibility

2. **New Architecture Tests** (`test_new_architecture.py`):
   - Tests object-based API
   - Tests dictionary-based API
   - Tests CLI interface

Additional unit tests cover CLI helpers (`tests/test_main.py`), artifact writers (`tests/test_artifacts.py`), and rule plugin safety (`tests/test_rules.py`). Run the whole suite with:

```bash
pytest aic_generator/tests
```

Run both test suites:
```bash
python test_compatibility.py
python test_new_architecture.py
```

## Migration Guide

### For Existing Users
No changes required! The compatibility layer ensures all existing code continues to work.

### For New Development
Use the new architecture for better separation of concerns:

1. **For programmatic use**: Use `generate_config_from_dicts()` with configuration objects
2. **For CLI use**: Use `main.py` with command-line arguments
3. **For template customization**: Modify files in `config/backend_templates/`

## Benefits

1. **Better Organization**: Clear separation between API, configuration, and mapping logic
2. **Multiple Input Methods**: Support for objects, dictionaries, and CLI
3. **Backward Compatibility**: Existing code continues to work unchanged
4. **Improved Testing**: Each module can be tested independently
5. **Easier Maintenance**: Modular design makes updates and debugging easier

## Configuration Mapping

The mapping system uses YAML templates with Jinja2 for backend-specific parameter transformation:

```yaml
# Example mapping for vLLM backend
vllm:
  run.sh.j2: |
    python -m vllm.entrypoints.openai.api_server \
      --model {{ model.model_path }} \
      --tensor-parallel-size {{ params.prefill.tp }} \
      --max-model-len {{ params.prefill.max_model_len | default(4096) }}
```

This allows flexible backend-specific configuration generation while maintaining a unified parameter interface.

## Rule Plugin DSL

Each backend can define lightweight tuning logic via `.rule` files (located in `rule_plugin/`) that are evaluated by `rendering/rule_engine.py`. See the inline docstrings in that module for syntax, best practices, and validation tips.