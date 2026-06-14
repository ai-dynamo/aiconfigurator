"""Compare generated artifacts to the immutable pre-cutover reference.

Structured artifacts (YAML) -> compared by PARSED structure (semantic equality):
the typed builders serialize differently than Jinja but must mean the same thing.
Textual artifacts -> compared BYTE-for-byte: they stay on Jinja and must not change.
"""
from __future__ import annotations

import pathlib

import yaml

REF_DIR = pathlib.Path(__file__).parent / "precutover_ref"

_STRUCTURED_SUFFIXES = ("k8s_deploy.yaml",)
_STRUCTURED_PREFIXES = ("extra_engine_args_",)


def is_structured(name: str) -> bool:
    return name.endswith(_STRUCTURED_SUFFIXES) or name.startswith(_STRUCTURED_PREFIXES)


def _parse_multidoc(text: str):
    return list(yaml.safe_load_all(text))


def compare_artifact(name: str, generated: str, reference: str) -> tuple[bool, str]:
    if is_structured(name):
        if _parse_multidoc(generated) == _parse_multidoc(reference):
            return True, ""
        return False, f"semantic mismatch in {name}"
    if generated == reference:
        return True, ""
    return False, f"byte mismatch in {name}"
