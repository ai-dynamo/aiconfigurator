"""Compare generated artifacts to the immutable pre-cutover reference.

Structured artifacts (YAML) -> compared by PARSED structure (semantic equality):
the typed builders serialize differently than Jinja but must mean the same thing.
Textual artifacts -> compared BYTE-for-byte: they stay on Jinja and must not change.

Embedded YAML strings (e.g. ConfigMap block scalars in k8s_deploy.yaml that contain
engine configs) are recursively parsed so that serialization differences like
``false`` vs ``False`` are treated as semantically equivalent.
"""
from __future__ import annotations

import pathlib

import yaml

REF_DIR = pathlib.Path(__file__).parent / "precutover_ref"

_STRUCTURED_SUFFIXES = ("k8s_deploy.yaml", "sflow.yaml")
_STRUCTURED_PREFIXES = ("extra_engine_args_",)


def is_structured(name: str) -> bool:
    return name.endswith(_STRUCTURED_SUFFIXES) or name.startswith(_STRUCTURED_PREFIXES)


def _normalize_heredoc_yamls(text: str) -> str:
    """Replace heredoc-embedded YAML blocks with a canonicalized form.

    Finds patterns like ``<<'YAML'\\n<content>\\nYAML`` or ``<<YAML\\n...\\nYAML``
    within shell script strings, parses the content as YAML, and re-serializes it
    so that cosmetic differences (bool capitalization, blank lines, comments) do not
    cause false mismatches.
    """
    import re

    def replace_heredoc(m: re.Match) -> str:
        marker = m.group(1)
        content = m.group(2)
        try:
            parsed = yaml.safe_load(content)
            if isinstance(parsed, (dict, list)):
                canonical = yaml.safe_dump(parsed, sort_keys=False, default_flow_style=False)
                return f"<<'{marker}'\n{canonical}{marker}"
        except yaml.YAMLError:
            pass
        return m.group(0)

    return re.sub(r"<<'?(\w+)'?\n(.*?)\n\1", replace_heredoc, text, flags=re.DOTALL)


def _normalize(obj):
    """Recursively normalize a parsed-YAML object.

    Strings that are themselves valid YAML documents are parsed and normalized
    so that serialization differences (e.g. ``true``/``True``, blank lines,
    key ordering) produced by different renderers are treated as semantically
    equivalent.

    Strings containing shell heredocs with embedded YAML are also normalized
    by canonicalizing each heredoc's YAML content.
    """
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    if isinstance(obj, str):
        # First, try to parse the entire string as YAML
        try:
            parsed = yaml.safe_load(obj)
            # Only substitute when the result is a non-trivial structure (dict/list)
            # to avoid accidentally normalizing plain strings that happen to be
            # valid YAML scalars (e.g. "true", "null", "1").
            if isinstance(parsed, (dict, list)):
                return _normalize(parsed)
        except yaml.YAMLError:
            pass
        # Otherwise, normalize any heredoc-embedded YAML blocks within the string
        import re
        if re.search(r"<<'?\w+'?", obj):
            return _normalize_heredoc_yamls(obj)
    return obj


def _parse_multidoc(text: str):
    return [_normalize(doc) for doc in yaml.safe_load_all(text)]


def compare_artifact(name: str, generated: str, reference: str) -> tuple[bool, str]:
    if is_structured(name):
        if _parse_multidoc(generated) == _parse_multidoc(reference):
            return True, ""
        return False, f"semantic mismatch in {name}"
    if generated == reference:
        return True, ""
    return False, f"byte mismatch in {name}"
