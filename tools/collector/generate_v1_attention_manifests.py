#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate exact historical attention coverage manifests.

Context and generation attention are frozen from collector-v1 commit
``a4827ce203e9fbc24fe6c6779a7eaa2a7dc79f1a`` (the parent of collector-v2
commit ``1d12d321``). Encoder attention did not exist at that ref, so its
separate initial baseline is frozen from the feature's first commit,
``36808ecced9af9d0d71d944c716ae96d1d4a2a47`` inside PR #1092. The PR was
later squashed as ``1ce6ff602f18e5d6fa46955fe8ca71e540bbc60e`` after moving
the cases to YAML. Encoder's initial baseline must not be described as
collector-v1 coverage.

The enumerators are intentionally framework-free: generating a compatibility
baseline must not require importing, installing, or checking out the
historical SGLang, TensorRT-LLM, or vLLM runtime.

Each historical invocation is converted to the row its collector wrote, then
normalized through the current consumer-aligned ``physical_row_key`` registry.
The resulting manifests protect exact lookup keys rather than raw scheduler
case strings or aggregate counts.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from collector.planner.coverage import CoverageHeader, CoverageManifest, write
from collector.planner.physical_keys import PhysicalRowKey, physical_row_key

ATTENTION_V1_SOURCE_GIT_REF = "a4827ce203e9fbc24fe6c6779a7eaa2a7dc79f1a"
ENCODER_INITIAL_SOURCE_GIT_REF = "36808ecced9af9d0d71d944c716ae96d1d4a2a47"
ENCODER_FEATURE_SQUASH_COMMIT = "1ce6ff602f18e5d6fa46955fe8ca71e540bbc60e"
COLLECTOR_V2_COMMIT = "1d12d321"
MANIFEST_ROOT = _REPO_ROOT / "collector" / "planner" / "manifests" / "collector_v1"

FRAMEWORK_VERSIONS = {
    "sglang": "0.5.10",
    "trtllm": "1.3.0rc10",
    "vllm": "0.19.0",
    "vllm_xpu": "0.19.0",
}

# Minimum versions declared by the original encoder collector modules. The
# protected grid applies to every compatible later runtime; sglang 0.5.10 and
# vLLM 0.19.0 remain correctly out of scope because they cannot run the op.
ENCODER_BASELINE_FRAMEWORK_VERSIONS = {
    "sglang": ">=0.5.11",
    "trtllm": ">=1.3.0rc5",
    "vllm": ">=0.21.0",
}

HARDWARE_SCOPES = {
    "sglang": ("b200_sxm", 100),
    "trtllm": ("b200_sxm", 100),
    "vllm": ("b200_sxm", 100),
    "vllm_xpu": ("xpu", 0),
}

OPS = ("attention_context", "attention_generation", "encoder_attention")

SOURCE_GIT_REFS = {
    "attention_context": ATTENTION_V1_SOURCE_GIT_REF,
    "attention_generation": ATTENTION_V1_SOURCE_GIT_REF,
    "encoder_attention": ENCODER_INITIAL_SOURCE_GIT_REF,
}

EXPECTED_COUNTS = {
    ("sglang", "attention_context"): 33714,
    ("sglang", "attention_generation"): 19484,
    ("trtllm", "attention_context"): 63192,
    ("trtllm", "attention_generation"): 40240,
    ("vllm", "attention_context"): 40392,
    ("vllm", "attention_generation"): 36288,
    ("vllm_xpu", "attention_context"): 16188,
    ("vllm_xpu", "attention_generation"): 26322,
    ("sglang", "encoder_attention"): 7008,
    ("trtllm", "encoder_attention"): 7008,
    ("vllm", "encoder_attention"): 7008,
}

_CONTEXT_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
_CONTEXT_SEQUENCES = [
    1,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    10240,
    12288,
    16384,
    262144,
]
_CONTEXT_HEADS = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 96]
_CONTEXT_KV_OPTIONS = [0, 1, 2, 4, 8]

_GENERATION_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
_SGLANG_GENERATION_BATCHES = [1, 2, 4, 64, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
_GENERATION_SEQUENCES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
_GENERATION_MHA_HEADS = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64]
_GENERATION_XQA_HEADS = [1, 2, 4, 8, 16, 32, 64, 96, 128]
_GENERATION_KV_HEADS = [1, 2, 4, 8]

# Encoder attention's original hardcoded collector used this one shared sweep
# for all three CUDA backends. These values are copied from the three collector
# modules at PR #1092's internal commit ``36808ecc...`` rather than read from
# today's mutable case catalog. The later squash's YAML changed this grid.
_ENCODER_BATCHES = [1, 2, 4, 8, 16, 32, 64]
_ENCODER_SEQUENCES = [
    13,
    16,
    26,
    32,
    52,
    64,
    104,
    128,
    192,
    256,
    400,
    512,
    576,
    1024,
    1296,
    1500,
    1536,
    2048,
    2304,
    3072,
    3136,
    4096,
    5184,
    6144,
    6400,
    7744,
    8192,
    9216,
    10240,
    10816,
    12288,
    12544,
    14400,
    16384,
    24576,
    32768,
    49152,
    65536,
]
_ENCODER_HEADS = [2, 4, 5, 8, 10, 12, 14, 16, 20, 24, 32]
_ENCODER_HEAD_DIMS = [64, 72, 80]


@dataclass(frozen=True, slots=True)
class LegacyAttentionCase:
    batch_size: int
    input_len: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    window_size: int
    fp8_kv_cache: bool
    fp8_context_fmha: bool
    is_context: bool

    def output_row(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "isl": self.input_len if self.is_context else 1,
            "num_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "window_size": self.window_size,
            "beam_width": 1,
            "attn_dtype": "fp8" if self.fp8_context_fmha else "bfloat16",
            "kv_cache_dtype": "fp8" if self.fp8_kv_cache else "bfloat16",
            "step": 0 if self.is_context else self.input_len,
            # Deliberately irrelevant measurement field. Its presence
            # documents that the consumer key excludes measured values.
            "latency": 0.0,
        }


@dataclass(frozen=True, slots=True)
class InitialEncoderAttentionCase:
    """One case from encoder attention's initial YAML-driven baseline."""

    batch_size: int
    input_len: int
    num_heads: int
    head_dim: int

    def output_row(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "isl": self.input_len,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "attn_dtype": "bfloat16",
            "latency": 0.0,
        }


def _decode_targets(
    batches: Sequence[int],
    sequences: Sequence[int],
    max_batch_for_sequence: Callable[[int], int],
) -> dict[int, list[int]]:
    """Reproduce collector-v1's sparse decode batch/sequence selection."""

    sequence_batches: dict[int, set[int]] = {}
    for sequence in sequences:
        max_batch = max_batch_for_sequence(sequence)
        for batch in batches:
            if batch > max_batch:
                break
            sequence_batches.setdefault(sequence, set()).add(batch)

    batch_sequences: dict[int, set[int]] = {}
    for sequence, selected_batches in sequence_batches.items():
        if len(selected_batches) < 4:
            continue
        for batch in selected_batches:
            batch_sequences.setdefault(batch, {sequence - 1}).add(sequence - 1)

    result = {}
    for batch, selected_sequences in batch_sequences.items():
        targets = sorted(selected_sequences)
        if batch >= 256:
            targets = targets[:-1]
        result[batch] = targets
    return result


def _precision_cases(*, context: bool, include_fp8_fmha: bool = True) -> Iterable[tuple[bool, bool]]:
    yield False, False
    yield True, False
    if context and include_fp8_fmha:
        yield True, True


def _sglang_context() -> Iterable[LegacyAttentionCase]:
    for head_dim in (128, 256):
        for num_heads in sorted(_CONTEXT_HEADS, reverse=True):
            for sequence in sorted(_CONTEXT_SEQUENCES, reverse=True):
                for batch in sorted(_CONTEXT_BATCHES, reverse=True):
                    for kv_option in _CONTEXT_KV_OPTIONS:
                        if kv_option and (kv_option >= num_heads or num_heads % kv_option):
                            continue
                        num_kv_heads = kv_option or num_heads
                        if num_kv_heads == num_heads:
                            if batch * sequence > 65536 or batch > 128:
                                continue
                        elif batch * sequence > 131072:
                            continue
                        if batch * sequence * num_kv_heads * head_dim * 2 >= 2147483647:
                            continue
                        for fp8_kv, fp8_fmha in _precision_cases(context=True):
                            yield LegacyAttentionCase(
                                batch,
                                sequence,
                                num_heads,
                                num_kv_heads,
                                head_dim,
                                0,
                                fp8_kv,
                                fp8_fmha,
                                True,
                            )


def _sglang_generation() -> Iterable[LegacyAttentionCase]:
    for head_dim in (128, 256):
        for num_heads in sorted(_GENERATION_MHA_HEADS, reverse=True):
            targets = _decode_targets(
                _SGLANG_GENERATION_BATCHES,
                _GENERATION_SEQUENCES,
                lambda sequence, n=num_heads, h=head_dim: 8192 * 1024 // sequence // n * 128 // h,
            )
            for batch, sequences in targets.items():
                for sequence in sequences:
                    for fp8_kv, fp8_fmha in _precision_cases(context=False):
                        yield LegacyAttentionCase(
                            batch,
                            sequence,
                            num_heads,
                            num_heads,
                            head_dim,
                            0,
                            fp8_kv,
                            fp8_fmha,
                            False,
                        )

        for num_heads in sorted(_GENERATION_XQA_HEADS, reverse=True):
            targets = _decode_targets(
                _SGLANG_GENERATION_BATCHES,
                _GENERATION_SEQUENCES,
                lambda sequence, n=num_heads, h=head_dim: 8192 * 1024 * 2 // sequence // n * 128 // h,
            )
            for batch, sequences in targets.items():
                for num_kv_heads in _GENERATION_KV_HEADS:
                    if num_kv_heads >= num_heads:
                        continue
                    for sequence in sequences:
                        for fp8_kv, fp8_fmha in _precision_cases(context=False):
                            yield LegacyAttentionCase(
                                batch,
                                sequence,
                                num_heads,
                                num_kv_heads,
                                head_dim,
                                0,
                                fp8_kv,
                                fp8_fmha,
                                False,
                            )


def _trtllm_ratio_supported(num_heads: int, num_kv_heads: int) -> bool:
    ratio = num_heads // num_kv_heads
    return ratio < 32 or ratio % 32 == 0


def _trtllm_context() -> Iterable[LegacyAttentionCase]:
    for head_dim in (64, 128, 256):
        for num_heads in sorted(_CONTEXT_HEADS, reverse=True):
            for sequence in sorted(_CONTEXT_SEQUENCES, reverse=True):
                for batch in sorted(_CONTEXT_BATCHES, reverse=True):
                    for kv_option in _CONTEXT_KV_OPTIONS:
                        if kv_option and (kv_option >= num_heads or num_heads % kv_option):
                            continue
                        num_kv_heads = kv_option or num_heads
                        if num_kv_heads == num_heads:
                            if batch * sequence > 65536 or batch > 128:
                                continue
                        elif batch * sequence > 131072:
                            continue
                        if batch * sequence * num_kv_heads * head_dim * 2 >= 2147483647:
                            continue
                        if num_kv_heads != num_heads and not _trtllm_ratio_supported(num_heads, num_kv_heads):
                            continue
                        windows = (128, 0) if head_dim == 64 else (0,)
                        for window in windows:
                            for fp8_kv, fp8_fmha in _precision_cases(context=True):
                                yield LegacyAttentionCase(
                                    batch,
                                    sequence,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    window,
                                    fp8_kv,
                                    fp8_fmha,
                                    True,
                                )


def _trtllm_generation() -> Iterable[LegacyAttentionCase]:
    for num_heads in sorted(_GENERATION_MHA_HEADS, reverse=True):
        targets = _decode_targets(
            _GENERATION_BATCHES,
            _GENERATION_SEQUENCES,
            lambda sequence, n=num_heads: 8192 * 1024 // sequence // n,
        )
        for head_dim in (64, 128, 256):
            for batch, sequences in targets.items():
                for sequence in sequences:
                    for fp8_kv, fp8_fmha in _precision_cases(context=False):
                        yield LegacyAttentionCase(
                            batch,
                            sequence,
                            num_heads,
                            num_heads,
                            head_dim,
                            0,
                            fp8_kv,
                            fp8_fmha,
                            False,
                        )

    for num_heads in sorted(_GENERATION_XQA_HEADS, reverse=True):
        targets = _decode_targets(
            _GENERATION_BATCHES,
            _GENERATION_SEQUENCES,
            lambda sequence, n=num_heads: 8192 * 1024 * 2 // sequence // n,
        )
        for head_dim in (64, 128, 256):
            for batch, sequences in targets.items():
                for num_kv_heads in _GENERATION_KV_HEADS:
                    if num_kv_heads >= num_heads or not _trtllm_ratio_supported(num_heads, num_kv_heads):
                        continue
                    windows = (128, 0) if head_dim == 64 else (0,)
                    for sequence in sequences:
                        for window in windows:
                            for fp8_kv, fp8_fmha in _precision_cases(context=False):
                                yield LegacyAttentionCase(
                                    batch,
                                    sequence,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    window,
                                    fp8_kv,
                                    fp8_fmha,
                                    False,
                                )


def _vllm_context() -> Iterable[LegacyAttentionCase]:
    for head_dim in (128, 64):
        for num_heads in sorted(_GENERATION_MHA_HEADS, reverse=True):
            for sequence in sorted(_CONTEXT_SEQUENCES, reverse=True):
                for batch in sorted(_CONTEXT_BATCHES, reverse=True):
                    for kv_option in _CONTEXT_KV_OPTIONS:
                        if kv_option and (kv_option > num_heads or num_heads % kv_option):
                            continue
                        num_kv_heads = kv_option or num_heads
                        if num_kv_heads == num_heads:
                            if batch * sequence > 65536 or batch > 128:
                                continue
                        elif batch * sequence > 131072:
                            continue
                        if batch * sequence * num_kv_heads * head_dim * 2 >= 2147483647:
                            continue
                        for window in (0, 128):
                            for fp8_kv, fp8_fmha in _precision_cases(context=False):
                                yield LegacyAttentionCase(
                                    batch,
                                    sequence,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    window,
                                    fp8_kv,
                                    fp8_fmha,
                                    True,
                                )


def _vllm_generation() -> Iterable[LegacyAttentionCase]:
    for head_dim in (128, 64):
        for num_heads in sorted(_GENERATION_MHA_HEADS, reverse=True):
            targets = _decode_targets(
                _GENERATION_BATCHES,
                _GENERATION_SEQUENCES,
                lambda sequence, n=num_heads, h=head_dim: 8192 * 1024 * 128 // h // sequence // n,
            )
            for batch, sequences in targets.items():
                for num_kv_heads in _GENERATION_KV_HEADS:
                    if num_kv_heads > num_heads or num_heads % num_kv_heads:
                        continue
                    if num_heads // num_kv_heads > 16:
                        continue
                    for sequence in sequences:
                        for window in (0, 128):
                            for fp8_kv, fp8_fmha in _precision_cases(context=False):
                                yield LegacyAttentionCase(
                                    batch,
                                    sequence,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    window,
                                    fp8_kv,
                                    fp8_fmha,
                                    False,
                                )


def _vllm_xpu_context() -> Iterable[LegacyAttentionCase]:
    batches = [1, 2, 4, 8, 16, 32]
    sequences = _CONTEXT_SEQUENCES[:-1]
    for num_heads in sorted(_GENERATION_MHA_HEADS, reverse=True):
        for sequence in sorted(sequences, reverse=True):
            for batch in sorted(batches, reverse=True):
                for num_kv_heads in _GENERATION_KV_HEADS:
                    if num_kv_heads > num_heads or num_heads % num_kv_heads:
                        continue
                    if num_heads // num_kv_heads > 16:
                        continue
                    if num_kv_heads == num_heads:
                        if batch * sequence > 65536 or batch > 128:
                            continue
                    elif batch * sequence > 131072:
                        continue
                    for head_dim in (128, 64):
                        if batch * sequence * num_kv_heads * head_dim * 2 >= 2147483647:
                            continue
                        for window in (0, 128):
                            if window and head_dim == 128:
                                continue
                            for fp8_kv, fp8_fmha in _precision_cases(context=False):
                                yield LegacyAttentionCase(
                                    batch,
                                    sequence,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    window,
                                    fp8_kv,
                                    fp8_fmha,
                                    True,
                                )


def _vllm_xpu_generation() -> Iterable[LegacyAttentionCase]:
    for num_heads in sorted(_GENERATION_MHA_HEADS, reverse=True):
        targets = _decode_targets(
            _GENERATION_BATCHES,
            _GENERATION_SEQUENCES,
            lambda sequence, n=num_heads: 8192 * 1024 // sequence // n,
        )
        for batch, sequences in targets.items():
            for num_kv_heads in _GENERATION_KV_HEADS:
                if num_kv_heads > num_heads or num_heads % num_kv_heads:
                    continue
                if num_heads // num_kv_heads > 16:
                    continue
                for sequence in sequences:
                    for head_dim in (128, 64):
                        for window in (0, 128):
                            if window and head_dim == 128:
                                continue
                            for fp8_kv, fp8_fmha in _precision_cases(context=False):
                                yield LegacyAttentionCase(
                                    batch,
                                    sequence,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    window,
                                    fp8_kv,
                                    fp8_fmha,
                                    False,
                                )


def _initial_encoder_attention() -> Iterable[InitialEncoderAttentionCase]:
    """Reproduce encoder attention's first shared YAML-driven case loop."""

    for head_dim in _ENCODER_HEAD_DIMS:
        for num_heads in sorted(_ENCODER_HEADS):
            for sequence in sorted(_ENCODER_SEQUENCES):
                for batch in sorted(_ENCODER_BATCHES):
                    if batch * sequence > 131072:
                        continue
                    if 4 * batch * sequence * num_heads * head_dim * 2 >= 2**31:
                        continue
                    yield InitialEncoderAttentionCase(batch, sequence, num_heads, head_dim)


_ENUMERATORS = {
    ("sglang", "attention_context"): _sglang_context,
    ("sglang", "attention_generation"): _sglang_generation,
    ("trtllm", "attention_context"): _trtllm_context,
    ("trtllm", "attention_generation"): _trtllm_generation,
    ("vllm", "attention_context"): _vllm_context,
    ("vllm", "attention_generation"): _vllm_generation,
    ("vllm_xpu", "attention_context"): _vllm_xpu_context,
    ("vllm_xpu", "attention_generation"): _vllm_xpu_generation,
    ("sglang", "encoder_attention"): _initial_encoder_attention,
    ("trtllm", "encoder_attention"): _initial_encoder_attention,
    ("vllm", "encoder_attention"): _initial_encoder_attention,
}


def perf_table_for_op(op: str) -> str:
    tables = {
        "attention_context": "context_attention_perf.parquet",
        "attention_generation": "generation_attention_perf.parquet",
        "encoder_attention": "encoder_attention_perf.parquet",
    }
    if op not in tables:
        raise ValueError(f"unknown attention op: {op}")
    return tables[op]


def manifest_path(output_root: str | Path, backend: str, op: str) -> Path:
    return Path(output_root) / backend / f"{op}.jsonl.gz"


def framework_version_for(backend: str, op: str) -> str:
    if op == "encoder_attention":
        try:
            return ENCODER_BASELINE_FRAMEWORK_VERSIONS[backend]
        except KeyError as exc:
            raise ValueError(f"unsupported encoder attention backend: {backend}") from exc
    try:
        return FRAMEWORK_VERSIONS[backend]
    except KeyError as exc:
        raise ValueError(f"unsupported attention backend: {backend}") from exc


def build_historical_manifest(backend: str, op: str) -> CoverageManifest:
    """Build one exact-key manifest from its frozen historical enumerator."""

    try:
        enumerate_cases = _ENUMERATORS[(backend, op)]
    except KeyError as exc:
        raise ValueError(f"unsupported backend/op pair: {backend}/{op}") from exc

    perf_table = perf_table_for_op(op)
    keys: set[PhysicalRowKey] = set()
    for case in enumerate_cases():
        key = physical_row_key(perf_table, case.output_row())
        if key is None:  # pragma: no cover - the table is a checked-in registry invariant
            raise RuntimeError(f"physical key schema is not registered for {perf_table}")
        keys.add(key)

    expected_count = EXPECTED_COUNTS[(backend, op)]
    if len(keys) != expected_count:
        raise RuntimeError(
            f"historical baseline count mismatch for {backend}/{op}: generated={len(keys)}, expected={expected_count}"
        )

    gpu_type, sm_version = HARDWARE_SCOPES[backend]
    return CoverageManifest(
        header=CoverageHeader(
            source_git_ref=SOURCE_GIT_REFS[op],
            backend_variant=backend,
            framework_version=framework_version_for(backend, op),
            gpu_type=gpu_type,
            sm_version=sm_version,
            perf_table=perf_table,
        ),
        keys=frozenset(keys),
    )


def _canonical_bytes(manifest: CoverageManifest) -> bytes:
    with tempfile.TemporaryDirectory(prefix="aic-historical-attention-manifest-") as directory:
        path = Path(directory) / "manifest.jsonl.gz"
        write(path, manifest)
        return path.read_bytes()


def generate_manifests(output_root: str | Path, *, check: bool = False) -> list[str]:
    """Write or byte-check all frozen attention manifests."""

    messages = []
    errors = []
    for backend, op in EXPECTED_COUNTS:
        manifest = build_historical_manifest(backend, op)
        path = manifest_path(output_root, backend, op)
        if check:
            if not path.exists():
                errors.append(f"missing manifest: {path}")
                continue
            expected_bytes = _canonical_bytes(manifest)
            if path.read_bytes() != expected_bytes:
                errors.append(f"manifest is stale or nondeterministic: {path}")
                continue
            messages.append(f"checked {path}: {len(manifest.keys)} keys")
        else:
            write(path, manifest)
            messages.append(f"wrote {path}: {len(manifest.keys)} keys")

    if errors:
        raise RuntimeError("\n".join(errors))
    return messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MANIFEST_ROOT,
        help="Root containing canonical <backend>/<op>.jsonl.gz manifests",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare canonical generated bytes with checked-in manifests without rewriting them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        messages = generate_manifests(args.output_root, check=args.check)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for message in messages:
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
