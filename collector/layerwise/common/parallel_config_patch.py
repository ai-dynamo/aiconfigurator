"""Patch HF config.json to simulate parallelism on a single GPU.

Computes the per-GPU dimensions for a given parallelism config and delegates
to config_patch.patch_model_path for the actual patching.

Supports three parallelism axes for MoE models:
  - attn_tp: tensor parallelism for attention (divides heads, intermediate_size)
  - moe_tp:  tensor parallelism within each expert (divides moe_intermediate_size)
  - ep:      expert parallelism (divides n_routed_experts / num_experts / num_local_experts)
  Constraint: attn_tp == moe_tp * ep

Reusable across trtllm, vllm, and sglang collectors.

Example:
    # Dense model, TP=4:
    tmp_dir = patch_for_parallelism("Qwen/Qwen3-8B", tp_size=4)

    # MoE model, attn_tp=8, ep=8, moe_tp=1 (pure EP for experts):
    tmp_dir = patch_for_parallelism("deepseek-ai/deepseek-r1", attn_tp=8, ep=8)

    # MoE model, attn_tp=8, ep=4, moe_tp=2 (combined EP+TP for experts):
    tmp_dir = patch_for_parallelism("deepseek-ai/deepseek-r1", attn_tp=8, moe_tp=2, ep=4)
"""
import json
import math
import os

from config_patch import patch_model_path

EXPERT_COUNT_KEYS = ("n_routed_experts", "num_experts", "num_local_experts")


def _decoder_config_view(config: dict) -> tuple[dict, str]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and "num_attention_heads" in text_config:
        return text_config, "text_config."
    return config, ""


def _load_original_config(model_id: str) -> dict:
    """Load the original HF config.json without patching."""
    if os.path.isdir(model_id):
        with open(os.path.join(model_id, "config.json")) as f:
            return json.load(f)
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(model_id, "config.json")
        with open(config_file) as f:
            return json.load(f)


def patch_for_parallelism(
    model_id: str,
    *,
    tp_size: int = 1,
    attn_tp: int | None = None,
    moe_tp: int = 1,
    ep: int | None = None,
    num_slots: int | None = None,
    num_hidden_layers: int = 1,
    extra_overrides: dict | None = None,
    strip_auto_map: bool = True,
    model_type_rewrites: dict[str, str] | None = None,
    cache_dir: str | None = None,
    original_config: dict | None = None,
) -> str:
    """Create a patched model dir simulating parallelism on a single GPU.

    Args:
        model_id: HF model name or local path.
        tp_size: Shorthand — sets attn_tp and ep when they are not given.
        attn_tp: Attention tensor parallelism (divides heads, dense MLP).
            Defaults to tp_size.
        moe_tp: MoE tensor parallelism within each expert (divides
            moe_intermediate_size). Defaults to 1.
        ep: Expert parallelism (divides expert count). Defaults to tp_size.
        num_slots: Optional EPLB-adjusted expert slot count. When supplied,
            local expert count is derived from this value instead of the
            original expert count.
        num_hidden_layers: Number of layers in the patched config (default 1).
        extra_overrides: Additional config overrides (merged last).
        strip_auto_map: Remove `auto_map` from the copied HF config.
        model_type_rewrites: Optional model_type rewrite map passed through to
            config_patch.patch_model_path.
        cache_dir: Optional deterministic patched-config cache root.
        original_config: Optional already-loaded HF config to avoid loading
            config.json again in collector loops.

    Returns:
        Path to a temp directory with patched config.json.
    """
    # Resolve defaults from tp_size for backward compatibility.
    if attn_tp is None:
        attn_tp = tp_size
    if ep is None:
        ep = tp_size

    root_config = original_config or _load_original_config(model_id)
    config, override_prefix = _decoder_config_view(root_config)
    is_moe = any((config.get(k, 0) or 0) > 0 for k in EXPERT_COUNT_KEYS)

    if is_moe and attn_tp != moe_tp * ep:
        raise ValueError(
            f"Constraint violated: attn_tp ({attn_tp}) != moe_tp ({moe_tp}) * ep ({ep})"
        )

    uses_intermediate_as_moe = (
        is_moe
        and config.get("model_type") == "gpt_oss"
        and (config.get("moe_intermediate_size", 0) or 0) <= 0
    )

    # --- Attention / dense MLP ---
    orig_heads = config["num_attention_heads"]
    orig_kv_heads = config.get("num_key_value_heads", orig_heads)
    orig_inter = config.get("intermediate_size", 0) or 0

    if orig_heads % attn_tp != 0:
        raise ValueError(
            f"num_attention_heads={orig_heads} not divisible by attn_tp={attn_tp}"
        )

    overrides = {
        f"{override_prefix}num_attention_heads": orig_heads // attn_tp,
        f"{override_prefix}num_key_value_heads": math.ceil(orig_kv_heads / attn_tp),
        f"{override_prefix}num_hidden_layers": num_hidden_layers,
    }
    if not uses_intermediate_as_moe and orig_inter > 0:
        overrides[f"{override_prefix}intermediate_size"] = orig_inter // attn_tp

    # --- MoE expert parallelism (EP) ---
    for expert_key in EXPERT_COUNT_KEYS:
        orig_experts = config.get(expert_key, 0)
        if orig_experts > 0 and ep > 1:
            source_experts = num_slots or orig_experts
            if source_experts % ep != 0:
                raise ValueError(
                    f"{expert_key} source={source_experts} not divisible by ep={ep}"
                )
            overrides[f"{override_prefix}{expert_key}"] = source_experts // ep

    # --- MoE tensor parallelism (moe_tp) ---
    if moe_tp > 1 or uses_intermediate_as_moe:
        moe_inter_key = "intermediate_size" if uses_intermediate_as_moe else "moe_intermediate_size"
        orig_moe_inter = config.get(moe_inter_key, 0)
        if orig_moe_inter <= 0:
            raise ValueError(
                f"moe_tp={moe_tp} but config has no {moe_inter_key}"
            )
        if orig_moe_inter % moe_tp != 0:
            raise ValueError(
                f"{moe_inter_key}={orig_moe_inter} not divisible by moe_tp={moe_tp}"
            )
        overrides[f"{override_prefix}{moe_inter_key}"] = orig_moe_inter // moe_tp

    if extra_overrides:
        for key, value in extra_overrides.items():
            if override_prefix and "." not in key:
                overrides[f"{override_prefix}{key}"] = value
            else:
                overrides[key] = value

    return patch_model_path(
        model_id,
        overrides=overrides,
        strip_auto_map=strip_auto_map,
        model_type_rewrites=model_type_rewrites,
        cache_dir=cache_dir,
    )
