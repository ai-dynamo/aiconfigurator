# Config Schema Sync

Automatically detect and update generator templates when framework config formats change.

## When to Use

- User says "update trtllm to X version"
- User says "check config changes for vllm"
- User says "sync generator templates with new sglang version"

## Tool

```bash
aic-schema <command> [options]

Commands:
  current <backend>              Extract schema from currently installed version
  extract <backend> <version>    Extract schema for a specific version (creates venv)
  diff <backend> <old> <new>     Compare schemas between two versions
  diff-files <old.json> <new.json>  Compare two saved schema files
  list <backend>                 List cached versions
```

## Workflow

### Step 1: Extract Schema from Each Version

**Method A: Current Environment (Recommended)**

In each environment with a specific version installed:
```bash
aic-schema current trtllm -o schema-1.2.0rc5.json
```

**Method B: Venv (May have compatibility issues)**

```bash
aic-schema extract trtllm 1.2.0rc5 -o schema-1.2.0rc5.json
```

### Step 2: Compare Schemas

```bash
aic-schema diff-files schema-1.2.0rc5.json schema-1.3.0rc1.json
```

Output (JSON):
```json
{
  "backend": "trtllm",
  "old_version": "1.2.0rc5",
  "new_version": "1.3.0rc1",
  "changes": [
    {
      "type": "changed",
      "field": "BuildConfig.max_batch_size",
      "old": {"type": "integer", "default": 2048},
      "new": {"type": "integer", "default": 2048}
    }
  ],
  "summary": {
    "added": 2,
    "removed": 1,
    "changed": 3
  }
}
```

### Step 3: Update Templates

Based on the diff output, update:

1. **Jinja2 Templates**: `src/aiconfigurator/generator/config/backend_templates/<backend>/`
   - `extra_engine_args.<version>.yaml.j2`
   - `cli_args.<version>.j2`

2. **Mapping File**: `src/aiconfigurator/generator/config/backend_config_mapping.yaml`
   - Add new parameters
   - Update paths if field locations changed

### Step 4: Test

```bash
# Test the new templates
aic-generator render-config --backend trtllm --version <new_version> ...
```

## Template Update Rules

### Field Moved (path changed)

Old: `build_config.max_batch_size`
New: `max_batch_size`

Action:
```yaml
# Old template
build_config:
  max_batch_size: {{ max_batch_size }}

# New template
max_batch_size: {{ max_batch_size }}
```

### Field Added

New field: `new_option` with default `auto`

Action: Add to template with default:
```yaml
{% if new_option is defined %}
new_option: {{ new_option }}
{% endif %}
```

### Field Removed

Old field: `deprecated_option` removed

Action: Remove from template and `backend_config_mapping.yaml`

## File Locations

```
aiconfigurator/
├── tools/
│   ├── aic-schema              # CLI tool
│   └── .schema-cache/          # Cached schema files
└── src/aiconfigurator/generator/
    └── config/
        ├── backend_config_mapping.yaml   # Parameter mappings
        └── backend_templates/
            ├── trtllm/
            │   ├── extra_engine_args.yaml.j2
            │   ├── extra_engine_args.<version>.yaml.j2
            │   └── cli_args.j2
            ├── vllm/
            └── sglang/
```

## Supported Backends

| Backend | Package | Config Classes |
|---------|---------|----------------|
| trtllm | tensorrt-llm | BuildConfig, KvCacheConfig, CudaGraphConfig, ... |
| vllm | vllm | EngineArgs, ModelConfig, CacheConfig, ... |
| sglang | sglang | EngineArgs |

## Notes

- **Recommended**: Use `current` command in each environment to extract schema, then use `diff-files` to compare
- The `extract` command creates venvs in `~/.cache/aic-schema/venvs/` but may have dependency conflicts
- Schema files are JSON and can be version-controlled
