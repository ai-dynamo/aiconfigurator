"""Shared codec for the PR-6 data-plane baseline (capture AND replay).

The pre-deletion capture (``capture_data_plane_baseline.py``) walks every
``PerfDatabase._<family>_data`` attribute produced by the Python loaders and
reduces each table to an order-sensitive digest.  After the loaders are
deleted and the attributes are backed by the engine table-view FFI, the
replay test recomputes the same digests over the rehydrated views — equality
is the bit-for-bit proof that the FFI view reproduces the Python data plane
(structure, key order, and leaf values).

Everything here must therefore keep working on BOTH trees: the old one with
Python parsers and the new one where the attributes come from the engine.
"""

from __future__ import annotations

import enum
import hashlib
import json
from collections import UserDict
from collections.abc import Callable
from typing import Any

# The frozen inventory of loader-backed table attributes on PerfDatabase.
# This is the deletion surface of PR-6: every attribute listed here is filled
# by a Python ``load_*_data`` parser before the migration and by the engine
# table-view FFI after it.  Alias attributes that bind the same object are
# recorded so the replay can assert the aliasing is preserved.
TABLE_ATTRIBUTES: tuple[str, ...] = (
    "_gemm_data",
    "_compute_scale_data",
    "_scale_matrix_data",
    "_context_attention_data",
    "_generation_attention_data",
    "_encoder_attention_data",
    "_context_mla_data",
    "_generation_mla_data",
    "_mla_bmm_data",
    "_context_mla_module_data",
    "_generation_mla_module_data",
    "_wideep_context_mla_data",
    "_wideep_generation_mla_data",
    "_moe_data",
    "_moe_low_latency_data",
    "_wideep_context_moe_data",
    "_wideep_generation_moe_data",
    "_wideep_deepep_normal_data",
    "_wideep_deepep_ll_data",
    "_wideep_moe_compute_data",
    "_trtllm_alltoall_data",
    "_moe_a2a_data",
    "_moe_ep_data",
    "_custom_allreduce_data",
    "_nccl_data",
    "_oneccl_data",
    "_context_dsa_module_data",
    "_context_dsa_module_skip_data",
    "_generation_dsa_module_data",
    "_generation_dsa_module_skip_data",
    "_mhc_module_data",
    "_context_deepseek_v4_attention_module_data",
    "_generation_deepseek_v4_attention_module_data",
    "_dsv4_sparse_kernel_data",
    "_dsv4_megamoe_module_data",
    "_mamba2_data",
    "_gdn_data",
    "_kda_data",
)

# attribute -> the attribute it must alias (same object identity).
ALIAS_ATTRIBUTES: dict[str, str] = {
    "_raw_generation_attention_data": "_generation_attention_data",
    "_raw_context_dsa_module_data": "_context_dsa_module_data",
    "_raw_context_dsa_module_skip_data": "_context_dsa_module_skip_data",
    "_raw_generation_dsa_module_data": "_generation_dsa_module_data",
    "_raw_generation_dsa_module_skip_data": "_generation_dsa_module_skip_data",
    "_raw_context_deepseek_v4_attention_module_data": "_context_deepseek_v4_attention_module_data",
}

MAX_SAMPLE_LEAVES = 8
MAX_L1_KEYS = 400


def encode_key(key: Any) -> str:
    """Type-tagged, order-preserving key encoding.

    The tag keeps an enum named ``fp8`` distinct from the plain string
    ``"fp8"`` so the replay also proves the rehydration layer restored the
    exact key TYPES the Python loaders used, not just equal spellings.
    """
    if isinstance(key, enum.Enum):
        return f"e:{type(key).__name__}:{key.name}"
    if isinstance(key, bool):
        return f"b:{key}"
    if isinstance(key, int):
        return f"i:{key}"
    if isinstance(key, str):
        return f"s:{key}"
    if isinstance(key, tuple):
        return "t:" + "|".join(encode_key(part) for part in key)
    raise TypeError(f"unhandled table key type {type(key).__name__}: {key!r}")


def _is_leaf(value: Any) -> bool:
    return isinstance(value, dict) and "latency" in value


def encode_leaf(leaf: dict) -> dict[str, str]:
    """Leaf field ORDER is not a consumer contract (fields are read by name),
    so leaves are canonicalized sorted; values via ``repr`` for bit-for-bit
    float identity."""
    return {str(k): repr(v) for k, v in sorted(leaf.items(), key=lambda kv: str(kv[0]))}


def _iter_mapping(node: Any):
    """Iterate (key, value) preserving insertion order for dict and
    LoadedOpData alike, without tripping LoadedOpData's not-loaded guard."""
    if isinstance(node, UserDict):
        return node.data.items()
    return node.items()


def walk_table(node: Any, visit: Callable[[tuple[str, ...], dict], None], path: tuple[str, ...] = ()) -> None:
    """Depth-first, insertion-ordered walk calling ``visit(path, leaf)``."""
    for key, value in _iter_mapping(node):
        sub_path = path + (encode_key(key),)
        if _is_leaf(value):
            visit(sub_path, value)
        elif isinstance(value, (dict, UserDict)):
            walk_table(value, visit, sub_path)
        else:
            raise TypeError(f"unhandled table node at {'/'.join(sub_path)}: {type(value).__name__}")


def table_digest(table: Any) -> dict[str, Any]:
    """Order-sensitive digest of one table: structure + key order + values.

    ``ordered_sha256`` hashes the FULL insertion-ordered structure — branch
    open/close markers included, so an empty subtree (e.g. the mamba2
    generation rows, which the Python loader vivifies but never fills) is
    part of the contract, and key order is too (notebook legends assign
    colors positionally from key order).
    ``sorted_sha256`` hashes just the leaves sorted by path — kept as a
    deliberate fallback so an order-only mismatch is distinguishable from a
    value mismatch without recapturing.
    """
    ordered = hashlib.sha256()
    leaves: list[tuple[tuple[str, ...], dict]] = []

    def hash_node(node: Any, path: tuple[str, ...]) -> None:
        ordered.update(b"{")
        for key, value in _iter_mapping(node):
            encoded = encode_key(key)
            ordered.update(encoded.encode())
            ordered.update(b"=")
            if _is_leaf(value):
                ordered.update(json.dumps(encode_leaf(value), sort_keys=True).encode())
                leaves.append((path + (encoded,), value))
            elif isinstance(value, (dict, UserDict)):
                hash_node(value, path + (encoded,))
            else:
                raise TypeError(f"unhandled table node at {'/'.join(path)}/{encoded}: {type(value).__name__}")
            ordered.update(b";")
        ordered.update(b"}")

    hash_node(table, ())

    sorted_hash = hashlib.sha256()
    for path, leaf in sorted(leaves, key=lambda item: item[0]):
        sorted_hash.update("/".join(path).encode())
        sorted_hash.update(b"=")
        sorted_hash.update(json.dumps(encode_leaf(leaf), sort_keys=True).encode())
        sorted_hash.update(b";")

    l1_keys = [encode_key(k) for k, _ in _iter_mapping(table)]
    leaf_fields = sorted({str(field) for _, leaf in leaves for field in leaf})

    n = len(leaves)
    if n <= MAX_SAMPLE_LEAVES:
        sample_indices = range(n)
    else:
        step = n / MAX_SAMPLE_LEAVES
        sample_indices = sorted({int(i * step) for i in range(MAX_SAMPLE_LEAVES)})
    samples = [["/".join(leaves[i][0]), encode_leaf(leaves[i][1])] for i in sample_indices]

    return {
        "ordered_sha256": ordered.hexdigest(),
        "sorted_sha256": sorted_hash.hexdigest(),
        "n_leaves": n,
        "l1_keys": l1_keys[:MAX_L1_KEYS],
        "l1_key_count": len(l1_keys),
        "leaf_fields": leaf_fields,
        "sample_leaves": samples,
    }


def snapshot_attribute(database: Any, attribute: str) -> dict[str, Any]:
    """Three-state snapshot of one table attribute.

    states: ``unbound`` (attribute absent), ``unloaded`` (LoadedOpData
    wrapping None), ``loaded`` (digest recorded).  ``_dsv4_sparse_kernel_data``
    is a plain dict of sub-tables and is digested per sub-key.
    """
    if attribute not in vars(database):
        return {"state": "unbound"}
    value = getattr(database, attribute)
    if value is None:
        return {"state": "none"}
    if isinstance(value, UserDict):
        if not getattr(value, "loaded", True):
            return {"state": "unloaded"}
        return {"state": "loaded", **table_digest(value)}
    if isinstance(value, dict) and value and all(isinstance(v, UserDict) for v in value.values()):
        sub = {}
        for sub_key, sub_table in value.items():
            if not getattr(sub_table, "loaded", True):
                sub[str(sub_key)] = {"state": "unloaded"}
            else:
                sub[str(sub_key)] = {"state": "loaded", **table_digest(sub_table)}
        return {"state": "loaded", "subtables": sub}
    if isinstance(value, dict):
        return {"state": "loaded", **table_digest(value)}
    return {"state": f"unexpected:{type(value).__name__}"}


def snapshot_database_tables(database: Any) -> dict[str, Any]:
    """Snapshot all frozen table attributes plus the alias-identity map."""
    tables = {attr: snapshot_attribute(database, attr) for attr in TABLE_ATTRIBUTES}
    aliases = {}
    for alias, target in ALIAS_ATTRIBUTES.items():
        if alias not in vars(database) and target not in vars(database):
            aliases[alias] = "both-unbound"
        elif alias in vars(database) and target in vars(database):
            aliases[alias] = "same-object" if getattr(database, alias) is getattr(database, target) else "DIFFERENT"
        else:
            aliases[alias] = "one-side-unbound"
    return {"tables": tables, "aliases": aliases}


def snapshot_support_matrix(database: Any) -> dict[str, Any]:
    """Both support-matrix paths, name lists order-preserved."""
    lazy = {}
    matrix = database.supported_quant_mode
    for key in list(matrix.keys()):
        lazy[key] = list(matrix[key])
    database._update_support_matrix()
    eager = {key: list(names) for key, names in database.supported_quant_mode.items()}
    return {"lazy": lazy, "eager": eager}
