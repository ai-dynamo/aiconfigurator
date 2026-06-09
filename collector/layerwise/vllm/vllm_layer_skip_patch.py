"""Identity-replace non-target DecoderLayer forwards + early PytHooks (vLLM 0.19.0).

Monkey-patches `GPUModelRunner.load_model`:
  1. For every DecoderLayer whose index is not in `LAYERWISE_TARGET_LAYERS`,
     swap `.forward` for an identity pass-through. Layer list discovered via
     `named_modules()` introspection — robust to multimodal wrappers
     (e.g. Qwen3.5 layers live at `language_model.model.layers`).
  2. When `compilation_config.mode == NONE`, register `PytHooks` early
     (before first `_dummy_run` / cuda graph capture). This is required for
     per-module NVTX → graphNodeId attribution in cuda-graph mode. Skipped
     under `VLLM_COMPILE` since dynamo can't trace the `"{}".format(dict)`
     inside `construct_marker_dict_and_push`.

Env:
  LAYERWISE_TARGET_LAYERS=3,4   # comma-separated layer indices (default 3)
  LAYERWISE_SKIP_ENABLE=1       # set 0 to disable
  LAYERWISE_MOE_NOOP=1          # replace target-layer MoE MLPs with identity
"""
import logging
import os
import types

logger = logging.getLogger(__name__)


def _parse_targets() -> set[int]:
    raw = os.environ.get("LAYERWISE_TARGET_LAYERS", "3")
    return {int(x) for x in raw.split(",") if x.strip()}


# vLLM DecoderLayer return arity. Unknown -> default 2.
_RETURN_ARITY = {
    "DeepseekV2DecoderLayer": 2,     # GLM-5 DSA uses this class
    "DeepseekV3DecoderLayer": 2,
    "Glm4MoeDecoderLayer": 2,
    "LlamaDecoderLayer": 2,
    "Qwen2MoeDecoderLayer": 2,
    "Qwen3DecoderLayer": 2,
    "Qwen3MoeDecoderLayer": 2,
    # Qwen3.5 / Qwen3-Next: single hybrid class dispatches linear_attention
    # vs full_attention internally. Signature is
    #   (hidden_states, residual, positions=None, **kw) — note hidden_states
    # is the FIRST positional (not second like DeepseekV2).
    "Qwen3NextDecoderLayer": 2,
    "Qwen3_5DecoderLayer": 2,
    # GPT-OSS names decoder blocks TransformerBlock in vLLM 0.20.1.
    "TransformerBlock": 2,
}


def _make_identity_forward(arity: int):
    def _identity_forward(self, *args, **kwargs):
        # Prefer kwargs since DecoderLayer classes use different positional
        # orderings (DeepseekV2: positions first; Qwen3Next: hidden_states first).
        hidden_states = kwargs.get("hidden_states")
        residual = kwargs.get("residual")

        if hidden_states is None:
            # Fallback: scan positional args for the first tensor with dim >= 2.
            # hidden_states is [num_tokens, hidden_dim] (2D); positions is 1D.
            for a in args:
                if hasattr(a, "dim") and a.dim() >= 2:
                    hidden_states = a
                    break
        if residual is None:
            # residual either shares hidden_states' shape or is None on the
            # first layer. Pick the second 2D tensor if present.
            seen_first = False
            for a in args:
                if hasattr(a, "dim") and a.dim() >= 2:
                    if not seen_first:
                        seen_first = True
                    else:
                        residual = a
                        break

        if arity == 2:
            # Decoder stacks use residual=None only for the first real layer.
            # If layer 0 is skipped and a later layer is kept, forwarding None
            # makes the kept layer run the first-layer RMSNorm branch instead
            # of the normal fused add+RMSNorm branch.  Preserve the residual
            # state transition while skipping compute so sampled later layers
            # keep real-stack control flow.
            if residual is None and hidden_states is not None:
                residual = hidden_states
            return hidden_states, residual
        return hidden_states
    return _identity_forward


def _find_layers(model):
    """Return (layers_module_list, dotted_path_used) or (None, None).

    Walks `model.named_modules()` to find the longest `nn.ModuleList` whose
    elements' class names are DecoderLayer-like (match `_RETURN_ARITY` keys,
    or fall back to ``*DecoderLayer``). Robust to multimodal wrapper shapes
    (Qwen3.5: `language_model.model.layers`; DeepSeek/GLM-5: `model.layers`;
    etc.) without hard-coding attribute paths.
    """
    import torch.nn as _nn
    best = None
    best_len = -1
    for name, mod in model.named_modules():
        if not isinstance(mod, _nn.ModuleList) or len(mod) == 0:
            continue
        first = mod[0]
        cls = type(first).__name__
        if (cls in _RETURN_ARITY or cls.endswith("DecoderLayer")) and len(mod) > best_len:
            # Prefer the longest such list (main transformer stack vs shorter
            # vision blocks that also use ModuleList).
            best = (mod, name)
            best_len = len(mod)
    return best if best is not None else (None, None)


def _find_first_hidden_tensor(args, kwargs):
    hidden_states = kwargs.get("hidden_states")
    if hidden_states is None:
        hidden_states = kwargs.get("x")
    if hidden_states is None:
        for a in args:
            if hasattr(a, "dim") and a.dim() >= 2:
                hidden_states = a
                break
    return hidden_states


def _make_moe_noop_forward():
    def _moe_noop_forward(self, *args, **kwargs):
        hidden_states = _find_first_hidden_tensor(args, kwargs)
        if hidden_states is None:
            raise RuntimeError(
                "LAYERWISE_MOE_NOOP could not find hidden_states tensor "
                f"for {type(self).__name__}.forward"
            )
        return hidden_states
    return _moe_noop_forward


def _looks_like_moe_mlp(name: str, mod) -> bool:
    leaf = name.rsplit(".", 1)[-1] if name else ""
    cls = type(mod).__name__
    cls_lower = cls.lower()
    if leaf == "experts":
        return False
    has_experts = hasattr(mod, "experts")
    has_router = hasattr(mod, "router") or hasattr(mod, "gate")
    if has_experts and (has_router or "moe" in cls_lower or cls == "MLPBlock"):
        return True
    return leaf in {"mlp", "moe"} and ("moe" in cls_lower or "sparse" in cls_lower)


def _patch_first_moe_mlp(layer) -> tuple[str, str] | None:
    for name, mod in layer.named_modules():
        if not name or getattr(mod, "_layerwise_moe_noop", False):
            continue
        if not _looks_like_moe_mlp(name, mod):
            continue
        mod._layerwise_original_forward = mod.forward
        mod.forward = types.MethodType(_make_moe_noop_forward(), mod)
        mod._layerwise_moe_noop = True
        return name, type(mod).__name__
    return None


def _apply_moe_noop(model, target_layers: set[int]) -> int:
    layers, path = _find_layers(model)
    if layers is None:
        logger.warning(
            "[vllm-layer-skip] LAYERWISE_MOE_NOOP requested but could not "
            "locate transformer layers"
        )
        return 0

    patched = []
    missing = []
    for i in sorted(target_layers):
        if i < 0 or i >= len(layers):
            missing.append(i)
            continue
        match = _patch_first_moe_mlp(layers[i])
        if match is None:
            missing.append(i)
            continue
        name, cls = match
        patched.append(f"{i}.{name}:{cls}")

    if missing:
        logger.warning(
            f"[vllm-layer-skip] LAYERWISE_MOE_NOOP found no MoE MLP in "
            f"target layer(s) {missing} under {path}"
        )
    logger.warning(
        f"[vllm-layer-skip] LAYERWISE_MOE_NOOP patched={patched}"
    )
    return len(patched)


def _apply_skip(model, target_layers: set[int]) -> int:
    layers, path = _find_layers(model)
    if layers is None:
        logger.warning(
            "[vllm-layer-skip] could not locate an nn.ModuleList of "
            "DecoderLayer — no-op. Add this model's DecoderLayer class "
            "name to _RETURN_ARITY if new."
        )
        return 0
    logger.warning(f"[vllm-layer-skip] found layers at {path}")
    skipped = 0
    kept = []
    unknown = set()
    for i, layer in enumerate(layers):
        if i in target_layers:
            kept.append(i)
            continue
        cls = type(layer).__name__
        if cls not in _RETURN_ARITY:
            unknown.add(cls)
        arity = _RETURN_ARITY.get(cls, 2)
        layer.forward = types.MethodType(_make_identity_forward(arity), layer)
        skipped += 1
    if unknown:
        logger.warning(
            f"[vllm-layer-skip] unknown DecoderLayer class(es) {unknown}; "
            f"defaulting to arity=2. Add to _RETURN_ARITY if incorrect."
        )
    logger.warning(
        f"[vllm-layer-skip] kept={sorted(kept)}, skipped={skipped} layers "
        f"(identity forward); total layers={len(layers)}"
    )
    return skipped


def _install_patch():
    if os.environ.get("LAYERWISE_SKIP_ENABLE", "1") != "1":
        logger.info("[vllm-layer-skip] disabled by LAYERWISE_SKIP_ENABLE=0")
        return

    from vllm.v1.worker import gpu_model_runner as _gmr

    orig = _gmr.GPUModelRunner.load_model
    targets = _parse_targets()

    def patched(self, *args, **kwargs):
        ret = orig(self, *args, **kwargs)
        try:
            # After CUDAGraphWrapper wrapping (load_model tail), `self.model`
            # may be a wrapper. Drill to underlying `.model` if needed.
            model = self.model
            inner = getattr(model, "model", None)
            # If the wrapper's .model is the *ForCausalLM, use it; else use model.
            root = inner if inner is not None and hasattr(inner, "model") else model
            _apply_skip(root, targets)
            if os.environ.get("LAYERWISE_MOE_NOOP", "0") == "1":
                _apply_moe_noop(root, targets)
            # Register PytHooks EARLY only when compile is disabled; under
            # VLLM_COMPILE mode dynamo can't trace `"{}".format(marker_dict)`
            # inside `construct_marker_dict_and_push` and the first forward
            # crashes. Compile-OFF + cudagraph=FULL is the sweet spot:
            # cuda graph capture still captures kernels, hooks fire at
            # capture time so capture-time cudaLaunchKernel calls land
            # inside Module NVTX → corr→module → graphNodeId→module, and
            # replay kernels inherit attribution.
            compile_mode = getattr(self.vllm_config.compilation_config, "mode", None)
            from vllm.config.compilation import CompilationMode
            if (
                compile_mode == CompilationMode.NONE
                and getattr(self.vllm_config.observability_config, "enable_layerwise_nvtx_tracing", False)
            ):
                from vllm.utils.nvtx_pytorch_hooks import PytHooks
                _pyt_hooks = PytHooks()
                _pyt_hooks.register_hooks(
                    self.model, self.model.__class__.__name__
                )
                self.layerwise_nvtx_hooks_registered = True
                logger.warning(
                    "[vllm-layer-skip] registered PytHooks early "
                    "(compile=NONE path, before graph capture)"
                )
        except Exception:
            logger.exception("[vllm-layer-skip] failed")
            raise
        return ret

    _gmr.GPUModelRunner.load_model = patched
    logger.warning(
        f"[vllm-layer-skip] installed GPUModelRunner.load_model patch, "
        f"targets={sorted(targets)}"
    )


_install_patch()
