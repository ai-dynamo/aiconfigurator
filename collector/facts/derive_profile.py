"""Checkpoint quant profile derivation — from the checkpoint's own quant
metadata, NEVER from the repo name (dummy fidelity rule 4).

Profile vocabulary matches targets.yaml kv_pairing keys."""
from __future__ import annotations

import json
from pathlib import Path


def derive_profile(repo: str, configs_dir: Path) -> str:
    stem = repo.replace('/', '_')
    p = configs_dir / f'{stem}.json'
    if not p.exists():
        return 'MISSING'
    c = json.loads(p.read_text())
    qc = c.get('quantization_config') or (c.get('text_config') or {}).get('quantization_config')
    hq_p = configs_dir / f'{stem}_hfquant.json'
    if hq_p.exists():
        algo = ((json.loads(hq_p.read_text()).get('quantization') or {}).get('quant_algo') or '').upper()
        if 'NVFP4' in algo or 'FP4' in algo:
            return 'nvfp4'
        if 'MXFP8' in algo:
            return 'mxfp8'
        if 'FP8' in algo:
            return 'fp8'
    if not qc:
        return 'bfloat16'

    def groups_have_4bit_float() -> bool:
        for g in (qc.get('config_groups') or {}).values():
            for part in ('weights', 'input_activations'):
                w = g.get(part) or {}
                if isinstance(w, dict) and w.get('num_bits') == 4 and w.get('type') == 'float':
                    return True
        # modelopt's other MIXED_PRECISION shape: flat per-layer dict
        for v in (qc.get('quantized_layers') or {}).values():
            if isinstance(v, dict) and 'FP4' in str(v.get('quant_algo', '')).upper():
                return True
        return False

    m = (qc.get('quant_method') or '').lower()
    algo = (qc.get('quant_algo') or '').upper()
    if m == 'mxfp8':
        return 'mxfp8'
    if m == 'mxfp4':
        return 'mxfp4'
    if m in ('modelopt', 'modelopt_mixed') or algo == 'MIXED_PRECISION':
        return 'nvfp4' if (groups_have_4bit_float() or 'FP4' in algo) else 'fp8'
    if m == 'compressed-tensors':
        # 4-bit float groups = fp4-family weights (e.g. Kimi-K3 native: w4 float
        # group-32); 4-bit int = packed w4 (marlin path), served fp8-activation
        if groups_have_4bit_float():
            return 'nvfp4'
        return 'fp8'
    if m == 'fp8':
        return 'fp8'
    return f'?{m}'
