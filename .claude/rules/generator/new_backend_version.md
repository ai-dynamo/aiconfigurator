# Adding a New Backend Version Reference

Adding a new backend version is one of the most frequent generator tasks. It involves
coordinated changes across multiple files. Getting any step wrong results in deployment
failures that only surface when users try the new version.

## When to Use This Reference

- Adding support for a new backend engine version (TRT-LLM, vLLM, SGLang)
- Bumping the default Dynamo version
- Handling deprecated/renamed backend CLI flags
- Updating container image references

## Workflow

### 1. Gather Version Information

- What is the new backend version? (e.g., vLLM 0.17.0)
- What Dynamo release maps to it? (e.g., Dynamo 1.1.0)
- What changed in this backend version?
  - New CLI flags added
  - Deprecated/removed CLI flags
  - Renamed flags
  - Changed default behaviors
  - New model architecture support
- Source: backend release notes, changelog, arg_utils diff

### 2. Update Version Matrix

File: `src/aiconfigurator/generator/config/backend_version_matrix.yaml`

- Add new Dynamo version entry (or update existing)
- Map to new backend version
- Example:
  ```yaml
  - dynamo_version: "1.1.0"
    backends:
      trtllm: "1.4.0"
      vllm: "0.17.0"
      sglang: "0.6.0"
  ```

### 3. Create Version-Specific Templates (if needed)

Only create new version templates if the backend CLI interface changed.

1. Identify which artifact templates need new versions:
   - `cli_args.j2` -- if CLI flags changed
   - `extra_engine_args.yaml.j2` -- if engine config format changed (TRT-LLM)
   - `k8s_deploy.yaml.j2` -- usually version-independent
   - `run.sh.j2` -- if startup command changed

2. Copy the closest prior version template:
   ```bash
   cp cli_args.0.16.0.j2 cli_args.0.17.0.j2
   ```

3. Modify the new template:
   - Add new flags
   - Remove deprecated flags
   - Update renamed flags
   - Adjust conditionals for changed behavior

4. **DO NOT modify the prior version template** -- it must continue working for
   existing deployments.

### 4. Update Parameter Mapping (if needed)

File: `src/aiconfigurator/generator/config/backend_config_mapping.yaml`

If new parameters were added:
- Add `param_key` entry
- Set CLI flag names for each backend (null if unsupported)

If parameters were renamed:
- Update the backend-specific flag name
- Keep the unified `param_key` stable (don't rename the internal name)

### 5. Update Deployment Config (if needed)

File: `src/aiconfigurator/generator/config/deployment_config.yaml`

If new user-facing parameters were added:
- Add schema entry with section, default, required status
- Add `backend_defaults` if behavior differs by backend

### 6. Update Rule Files (if needed)

Files: `src/aiconfigurator/generator/rule_plugin/<backend>.rule` (production)
and `src/aiconfigurator/generator/rule_plugin/benchmark/<backend>.rule` (benchmark)

If parameter computation logic changed:
- Update expressions
- Add new rules for new parameters
- Check: did a previously-valid flag become invalid? (PR #613: `moe_dense_tp_size`)

### 7. Update Container Images

If the Dynamo release includes new container images:
- Verify default image expressions in `deployment_config.yaml`
- Check that version tag resolution works for new version

### 8. Test

- Generate config for the new version with a representative model
- Compare output against manually-written reference config
- Run generator validator if backend image available
- Test all modes: agg, disagg-prefill, disagg-decode
- Test with MoE model if backend version added MoE changes
- **MOST IMPORTANT: Test that OLD version still works**

## Deprecated Flag Handling

When a backend deprecates a CLI flag:

1. Identify the deprecation:
   - Old flag: `--connector` (deprecated in version X)
   - New flag: `--kv-transfer-config '{"connector": "..."}'`

2. Create version-specific template:
   - Old template (< boundary version): keeps old flag
   - New template (>= boundary version): uses new flag

3. Update version matrix to ensure correct version -> template mapping

4. **DO NOT modify old templates** -- users on old versions must still work

## Version Template Selection Logic

The rendering engine selects templates by version:

1. Look for exact version match: `cli_args.<version>.j2`
2. Fall back to closest prior version (highest version <= requested)
3. Fall back to base template: `cli_args.j2`

You only need a new version template when the interface CHANGES, not for every
version bump.

## Anti-Patterns

1. **Don't edit prior version templates** -- They serve existing deployments. Create
   a new version template instead.

2. **Don't create a new version template for no-op changes** -- If the CLI interface
   didn't change, the existing template works fine. The version fallback logic handles it.

3. **Don't forget to test the OLD version** -- Adding version 0.17.0 support must not
   break 0.16.0 generation.

4. **Don't hardcode version checks in rendering engine** -- Use the version template
   naming convention and fallback logic. No `if version >= "0.17.0"` in Python code.

## Checklist

```text
[ ] Gather new version info: what changed (new/deprecated/renamed flags)
[ ] Update backend_version_matrix.yaml
[ ] Determine if version-specific templates are needed
[ ] Create new version templates (copy closest prior, modify)
[ ] DO NOT modify prior version templates
[ ] Update backend_config_mapping.yaml for new/changed parameters
[ ] Update deployment_config.yaml for new user-facing parameters
[ ] Update rule files for changed computation logic
[ ] Handle deprecated flags with version-specific templates
[ ] Update container image references if needed
[ ] Test: generate config for new version, all modes
[ ] Test: generate config for OLD version still works
[ ] Run generator validator
```
