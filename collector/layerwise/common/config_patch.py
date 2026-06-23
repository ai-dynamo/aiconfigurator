"""HF config.json patcher for layerwise benchmarks.

Returns a local tmp dir containing a patched `config.json` plus every
non-weight file from the HF repo (tokenizer, generation_config,
preprocessor_config for multimodal archs, …). Auxiliary files are linked by
default, with copy fallback, so large tokenizer assets are not duplicated for
every patched TP/EP variant. vLLM / sglang's model loader treats the tmp dir as
a full local model path; weights come from `--load-format=dummy`.

Override forms accepted by `patch_model_path(hf_id, overrides)`:
    {"num_hidden_layers": 4}                      # top-level
    {"text_config.num_hidden_layers": 4}          # dotted, for nested LM params
    {"text_config": {"num_hidden_layers": 4}}     # nested dict, deep-merged

Also supports `model_type_rewrites` for arch families whose model_type is
not in the installed HF transformers (e.g. `glm_moe_dsa` → `deepseek_v3`).
`architectures` is left untouched so the framework registry still dispatches
to the correct subclass.
"""
import copy
import hashlib
import json
import os
import shutil
import tempfile

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_COPY_AUX_FILES_ENV = "AIC_LAYERWISE_PATCH_COPY_AUX_FILES"


def _deep_merge(dst: dict, src: dict):
    """Recursively merge src into dst, in place."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def _apply_dotted(dst: dict, path: str, value):
    """Set dst[a][b][c] = value given path='a.b.c'."""
    parts = path.split(".")
    for p in parts[:-1]:
        dst = dst.setdefault(p, {})
    dst[parts[-1]] = value


def _stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _cache_target(
    cache_dir: str | None,
    model_id: str,
    overrides: dict | None,
    strip_auto_map: bool,
    model_type_rewrites: dict[str, str] | None,
) -> str | None:
    if not cache_dir:
        return None
    key = {
        "model_id": model_id,
        "overrides": overrides or {},
        "strip_auto_map": strip_auto_map,
        "model_type_rewrites": model_type_rewrites or {},
    }
    digest = hashlib.sha256(_stable_json(key).encode("utf-8")).hexdigest()[:24]
    safe_model = model_id.replace("/", "_").replace(":", "_")
    return os.path.join(cache_dir, f"{safe_model}_{digest}")


def _install_aux_file(src_path: str, dst_path: str) -> None:
    """Install an auxiliary model file into a patched config dir."""

    if os.path.exists(dst_path) or os.path.islink(dst_path):
        return
    if os.environ.get(_COPY_AUX_FILES_ENV) == "1":
        shutil.copy2(src_path, dst_path)
        return
    try:
        os.symlink(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)


def patch_model_path(model_id: str, overrides: dict | None = None,
                     strip_auto_map: bool = True,
                     model_type_rewrites: dict[str, str] | None = None,
                     cache_dir: str | None = None) -> str:
    """Return a local directory containing a patched config.json.

    If `model_id` is already a local dir, patch in a copy; else download
    config.json via huggingface_hub and patch.

    Args:
      overrides: either a flat dict with dotted keys, or a nested dict; both
        supported (dotted keys take priority if both used).
      strip_auto_map: remove `auto_map` so trust_remote_code won't pull code.
      model_type_rewrites: map model_type strings to replacement (sglang may
        not recognise `glm_moe_dsa` / `deepseek_v32`; rewrite to `deepseek_v3`).
      cache_dir: optional deterministic cache root. When set, a patched model
        dir is reused across collector invocations with identical inputs.
    """
    target_dir = _cache_target(
        cache_dir, model_id, overrides, strip_auto_map, model_type_rewrites
    )
    if target_dir and os.path.exists(os.path.join(target_dir, ".complete")):
        return target_dir
    if target_dir and os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    if os.path.isdir(model_id):
        src_dir = model_id
        with open(os.path.join(src_dir, "config.json")) as f:
            config = json.load(f)
    else:
        from huggingface_hub import hf_hub_download, list_repo_files
        # Pull config.json plus every non-weight file — multimodal archs
        # need preprocessor_config etc even with skip_tokenizer_init.
        try:
            all_files = list_repo_files(model_id)
        except Exception:
            all_files = ["config.json"]
        config_file = None
        for f in all_files:
            if f.endswith(_WEIGHT_SUFFIXES):
                continue
            try:
                path = hf_hub_download(model_id, f)
                if f == "config.json":
                    config_file = path
            except Exception:
                pass  # `.gitattributes` etc can 404 across revisions
        if config_file is None:
            config_file = hf_hub_download(model_id, "config.json")
        src_dir = os.path.dirname(config_file)
        with open(config_file) as f:
            config = json.load(f)

    config = copy.deepcopy(config)

    if model_type_rewrites:
        mt = config.get("model_type")
        if mt in model_type_rewrites:
            # Rewrite model_type only; KEEP `architectures` as-is so the
            # framework registry still dispatches to the correct subclass
            # (e.g. GlmMoeDsaForCausalLM for GLM-5 even when model_type is
            # rewritten to deepseek_v3 to satisfy HF AutoConfig).
            config["model_type"] = model_type_rewrites[mt]

    if strip_auto_map:
        config.pop("auto_map", None)

    if overrides:
        # Separate dotted-key and nested-dict overrides.
        nested_overrides = {}
        for k, v in overrides.items():
            if "." in k:
                _apply_dotted(config, k, v)
            else:
                nested_overrides[k] = v
        _deep_merge(config, nested_overrides)

    if target_dir:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_dir = f"{target_dir}.tmp.{os.getpid()}"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
    else:
        tmp_dir = os.path.join(
            tempfile.gettempdir(),
            f"layerwise_cfg_{model_id.replace('/', '_')}_{os.getpid()}",
        )
    os.makedirs(tmp_dir, exist_ok=True)

    # Copy auxiliary (non-weight) files so the tmp dir is a complete
    # "model dir" for downstream loaders.
    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        if not os.path.isfile(src_path):
            continue
        if fname.endswith(_WEIGHT_SUFFIXES):
            continue
        if fname == "config.json":
            continue  # we write the patched version below
        dst_path = os.path.join(tmp_dir, fname)
        _install_aux_file(src_path, dst_path)

    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)
    if not target_dir:
        return tmp_dir

    with open(os.path.join(tmp_dir, ".complete"), "w") as f:
        f.write("ok\n")
    try:
        os.rename(tmp_dir, target_dir)
    except FileExistsError:
        shutil.rmtree(tmp_dir)
        if not os.path.exists(os.path.join(target_dir, ".complete")):
            raise
    return target_dir
