# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang WideEP DeepEP normal-mode dispatch/combine collector.

Runs the DeepEP normal-mode (a.k.a. high-throughput / prefill EP) dispatch and
combine micro-benchmark once across all visible GPUs of a single node (full
node: 4 on GB200, 8 on B200/H200). It sweeps the MoE
``(hidden_size, num_experts, topk)`` shapes used by the MoE models under
``src/aiconfigurator/model_configs`` against a token list and an SM-count list,
and writes ``wideep_deepep_normal_perf.txt`` rows (``node_num=1``) that match
the schema consumed by ``operations.moe.load_wideep_deepep_normal_data``.

Building block: this wraps ``deepep/test_intranode.py`` — the DeepEP normal-mode
dispatch/combine benchmark for a single node. Per the DeepEP collection guide
(``deepep/README.md``), single-node (``node_num=1``) normal-mode data is
produced by ``test_intranode.py``; ``test_internode.py`` is the analogous
benchmark for the multi-node (``node_num>1``) RDMA path and asserts
``num_local_ranks == 8`` with ``num_ranks > 8``, so it cannot run in a
single-node full-node collective. Multi-node normal collection therefore remains
a documented follow-up that needs a SLURM multi-node launcher.

Unlike the per-GPU collectors, DeepEP normal is a single collective NCCL/DeepEP
job that must own every GPU at once, so ``collect.py`` invokes
``run_deepep_normal_fullnode`` directly (bypassing the per-GPU worker pool) and
then finalizes the produced CSV as parquet.
"""

import inspect
import json
import os
import socket
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTOR_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))  # collector/
REPO_ROOT = os.path.dirname(COLLECTOR_ROOT)
DEEPEP_DIR = os.path.join(THIS_DIR, "deepep")
MODEL_CONFIGS_DIR = os.path.join(REPO_ROOT, "src", "aiconfigurator", "model_configs")

# Pinned to the WideEP SGLang runtime (collector/framework_manifest.yaml -> 0.5.10).
# DeepEP normal has been collected on >= 0.5.0 runtimes, so allow the family.
__compat__ = ">=0.5.0"

# Token sweep, matching the existing SGLang 0.5.6.post2 normal dataset so the
# consumer's 1-D token interpolation has the same buckets it was built against.
DEEPEP_NORMAL_TOKENS = [
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    160,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
]

# SM-count sweep. The consumer keys the normal table on ``dispatch_sms``. Only an
# exact ``node_num==1 and sms==20`` query collapses to a 1-D token curve
# (``operations.moe._query_wideep_deepep_normal_table``); every other sms takes the
# 2-D ``(sms, num_tokens)`` grid branch, and ``MoEDispatch`` defaults to ``sms=12``
# (``operations.moe``: ``self._sms = kwargs.get("sms", 12)``), so the grid branch is
# the common path and needs the whole axis.
#
# The previous ``(20,)`` default, and the "only collect sm=20 for now" comment it
# cited, were accurate about the legacy 0.5.6.post2 dataset: that file really does
# hold sms=20 alone at ``node_num==1`` (the six-value grid appears there only at
# ``node_num`` 2/4/8). They went stale when the 0.5.10 data landed -- all six
# committed 0.5.10 files carry sms 4/8/12/16/20/24 at ``node_num==1``. Default to
# the grid the consumer actually reads; DEEPEP_NORMAL_SMS still narrows it for smoke
# runs, and ``_verify_axis_coverage`` asserts that what was swept is what landed.
DEEPEP_NORMAL_DEFAULT_SMS = (4, 8, 12, 16, 20, 24)

# Correctness checking. ``test_intranode.test_main(do_check=True)`` validates every
# dispatch/combine variant at a point; the collector previously hard-coded it off,
# which is what let a broken kernel reach a published table.
#
# The bound exists because the cost is dominated by allocation, not compute: the
# validation loop's ``num_worst_tokens`` path asks for ``num_tokens * num_ranks``
# extra tokens, and when the payload is FP8 that result goes through
# ``per_token_cast_back``, whose ``x_fp8.to(torch.float32)`` intermediate is 4
# bytes/element. At 131072 tokens x 8 ranks x hidden 8192 that single intermediate
# is 34 GB, on top of ~17 GB for ``recv_x``. Checks validate kernels, not batch
# sizes, so assert them across the part of the ladder where they are nearly free and
# leave the tail perf-only. Set to 0 to disable entirely, or very large to check
# every point.
DEEPEP_NORMAL_CHECK_MAX_TOKENS = int(os.environ.get("DEEPEP_NORMAL_CHECK_MAX_TOKENS", "1024"))

# Base seed, mirroring ``collect_deepep_ll.py``'s fixed ``seed=1``.
# ``test_intranode.test_main`` never seeds (``test_low_latency.test_main`` does), so
# without a per-point reseed every (sms, token) point draws a fresh routing and the
# same point on two systems is not measured against the same traffic pattern.
DEEPEP_NORMAL_SEED = int(os.environ.get("DEEPEP_NORMAL_SEED", "1"))

# DeepEP per-token FP8 cast requires the hidden dimension to be 128-divisible
# (deepep/utils.py per_token_cast_to_fp8). Unlike the low-latency kernels, the
# normal-mode dispatch is NOT template-specialized per hidden size and does not
# cap top-k, so no SWITCH_HIDDEN allowlist / kNumMaxTopK filter is applied here.
HIDDEN_DIVISOR = 128

# Intranode NVLink buffer size (bytes). Matches test_intranode.py's default.
# Override via DEEPEP_NORMAL_NVL_BYTES for very large token counts.
DEEPEP_NORMAL_NVL_BYTES = int(os.environ.get("DEEPEP_NORMAL_NVL_BYTES", str(int(2e9))))


def _sms_list():
    raw = os.environ.get("DEEPEP_NORMAL_SMS", "").strip()
    if not raw:
        return list(DEEPEP_NORMAL_DEFAULT_SMS)
    sms_list = [int(tok) for tok in raw.replace(",", " ").split()]
    # DeepEP splits the SM budget into ``num_sms / 2`` channels and
    # ``Buffer.set_num_sms`` asserts evenness. That assert only fires on the
    # default-config iteration of the tuning loop (the ``nvl_chunk_size == 0``
    # branch in test_intranode.test_main), i.e. roughly fifteen successful bench
    # passes into a shape and inside a live collective. Reject it at startup.
    odd = sorted({sms for sms in sms_list if sms % 2})
    if odd:
        raise RuntimeError(f"DEEPEP_NORMAL_SMS values must be even (DeepEP channels = num_sms/2); got {odd}")
    return sms_list


def _tokens_list():
    """Token ladder for the sweep, overridable via DEEPEP_NORMAL_TOKENS.

    The ladder is heavily top-weighted: reconstructed from the committed 0.5.10
    latencies and the exact call counts ``bench`` makes, the 131072-token point is
    roughly 47% of a full sweep and 65536 another 23%. Trimming the tail is the only
    large cost lever available, and a validation pass should not require editing this
    file to get one.
    """
    raw = os.environ.get("DEEPEP_NORMAL_TOKENS", "").strip()
    if not raw:
        return list(DEEPEP_NORMAL_TOKENS)
    return [int(tok) for tok in raw.replace(",", " ").split()]


def _verify_axis_coverage(output_path, sms_list, tokens):
    """Assert every shape present in the perf CSV carries the full (sms, token) grid.

    Nothing downstream can see these axes. ``get_deepep_normal_test_cases`` returns
    shape dicts, so ``provenance.case_plan_hash`` is byte-identical whether one sms
    or six were swept, and ``provenance.derive_table_status`` marks a table
    ``complete`` whenever no case failed -- coverage is never consulted. A run that
    swept ``(20,)`` instead of the six-sms grid therefore finalizes as a clean
    ``status: complete`` table with one sixth of the rows and a matching case-plan
    hash. This function is the only layer that knows what was asked for.

    Checked per shape, so shapes legitimately absent from the CSV -- skipped for
    ``num_experts % num_ranks`` or dropped on a buffer-allocation failure, both of
    which the worker logs -- do not raise here. Whole-shape loss is accounted for by
    the caller against the orchestrator's own succeeded/failed tally; this is about
    an axis being short *within* the shapes that did run.
    """
    import csv

    want_sms, want_tokens = set(sms_list), set(tokens)
    seen: dict[tuple[int, int, int], tuple[set[int], set[int]]] = {}
    with open(output_path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["hidden_size"]), int(row["num_experts"]), int(row["num_topk"]))
            got_sms, got_tokens = seen.setdefault(key, (set(), set()))
            got_sms.add(int(row["dispatch_sms"]))
            got_tokens.add(int(row["num_token"]))

    if not seen:
        raise RuntimeError(f"DeepEP normal collection wrote no data rows to {output_path}")

    short = [
        f"hidden={hidden} experts={num_experts} topk={num_topk}: "
        f"missing sms={sorted(want_sms - got_sms)} tokens={sorted(want_tokens - got_tokens)}"
        for (hidden, num_experts, num_topk), (got_sms, got_tokens) in sorted(seen.items())
        if (want_sms - got_sms) or (want_tokens - got_tokens)
    ]
    if short:
        raise RuntimeError(
            "DeepEP normal collection under-covers the swept axes; refusing to hand a partial table "
            "to finalization, which would attest it as complete:\n  " + "\n  ".join(short)
        )
    return len(seen)


def _iter_moe_configs():
    """Yield (hidden_size, num_experts, topk) from every MoE model config.

    Handles the field-name variants across DeepSeek / Qwen / Llama / MiniMax /
    GPT-OSS, including VL models that nest the language config under
    ``text_config``.
    """
    if not os.path.isdir(MODEL_CONFIGS_DIR):
        return
    for filename in sorted(os.listdir(MODEL_CONFIGS_DIR)):
        if not filename.endswith("_config.json"):
            continue
        try:
            with open(os.path.join(MODEL_CONFIGS_DIR, filename)) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            continue
        text_cfg = cfg.get("text_config") or {}
        num_experts = (
            cfg.get("n_routed_experts")
            or cfg.get("num_experts")
            or cfg.get("num_local_experts")
            or text_cfg.get("n_routed_experts")
            or text_cfg.get("num_experts")
            or text_cfg.get("num_local_experts")
        )
        topk = (
            cfg.get("num_experts_per_tok")
            or cfg.get("moe_topk")
            or cfg.get("num_experts_per_token")
            or cfg.get("topk")
            or text_cfg.get("num_experts_per_tok")
        )
        hidden = cfg.get("hidden_size") or text_cfg.get("hidden_size")
        if num_experts and topk and hidden:
            yield int(hidden), int(num_experts), int(topk)


def get_deepep_normal_test_cases():
    """Return unique structural ``(hidden_size, num_experts, topk)`` MoE combos.

    Tokens and SM counts are swept internally by the benchmark, so each case is
    one structural shape. Shapes DeepEP cannot run are pruned with evidence:

    * ``hidden % 128 != 0`` -- ``per_token_cast_to_fp8`` asserts a 128-divisible
      hidden dimension (collector/wideep/sglang/deepep/utils.py).
    * missing ``topk`` -- dense / non-MoE configs.

    (No SWITCH_HIDDEN allowlist or top-k cap is applied: those constrain the
    low-latency kernels, not the normal-mode dispatch.)
    """
    combos = set()
    for hidden, num_experts, topk in _iter_moe_configs():
        if hidden % HIDDEN_DIVISOR != 0:
            continue
        combos.add((hidden, num_experts, topk))
    return [
        {"hidden_size": hidden, "num_experts": num_experts, "topk": topk}
        for hidden, num_experts, topk in sorted(combos)
    ]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _normal_worker(local_rank, num_gpus, cases, tokens, sms_list, output_path, device_name, version):
    """Per-rank entrypoint spawned across all GPUs of the single node."""
    if DEEPEP_DIR not in sys.path:
        sys.path.insert(0, DEEPEP_DIR)

    import deep_ep
    import test_intranode
    import torch
    import torch.distributed as dist

    from utils import init_dist

    try:
        from helper import log_perf
    except ModuleNotFoundError:
        if COLLECTOR_ROOT not in sys.path:
            sys.path.append(COLLECTOR_ROOT)
        from helper import log_perf

    rank, num_ranks, group = init_dist(local_rank, num_gpus)
    if device_name is None:
        device_name = torch.cuda.get_device_name(local_rank)
    if rank == 0:
        # ``dispatch_transmit_us`` is the FP8 dispatch when this is True and the BF16
        # dispatch -- roughly twice the bytes -- when it is False: test_intranode
        # gates its FP8 tensor on ``is_sm90_compiled()`` and records whichever
        # variant comes first. No column distinguishes the two, and upstream's macro
        # is opt-in at build time, so record the build property in the run log rather
        # than assuming it across images.
        print(f"[deepep_normal] deep_ep is_sm90_compiled={deep_ep.Buffer.is_sm90_compiled()}", flush=True)
    written = 0

    def flush(shape_metrics):
        """Append one (shape, sms, token) row group to the perf CSV (rank 0).

        Writing incrementally keeps already-collected data on disk even if a
        later token/sms aborts the run.
        """
        nonlocal written
        if rank != 0 or not shape_metrics:
            return
        item_list = [
            {
                "node_num": 1,
                "hidden_size": m["hidden"],
                "num_token": m["num_tokens"],
                "num_topk": m["num_topk"],
                "num_experts": m["num_experts"],
                "dispatch_sms": m["dispatch_sms"],
                "dispatch_transmit_us": m["dispatch_transmit_us"],
                "dispatch_notify_us": m["dispatch_notify_us"],
                "combine_sms": m["combine_sms"],
                "combine_transmit_us": m["combine_transmit_us"],
                "combine_notify_us": m["combine_notify_us"],
            }
            for m in shape_metrics
        ]
        log_perf(
            item_list=item_list,
            framework="sglang",
            version=version,
            device_name=device_name,
            op_name="normal",
            kernel_source="deepep",
            perf_filename=output_path,
        )
        written += len(item_list)
        print(f"[deepep_normal] wrote {len(item_list)} rows (total {written}) to {output_path}", flush=True)

    for case in cases:
        hidden = int(case["hidden_size"])
        num_experts = int(case["num_experts"])
        num_topk = int(case["topk"])
        if num_experts % num_ranks != 0:
            if rank == 0:
                print(
                    f"[deepep_normal] skip hidden={hidden} experts={num_experts} topk={num_topk}: "
                    f"num_experts not divisible by num_ranks={num_ranks}",
                    flush=True,
                )
            continue

        # Buffer allocation can OOM for large shapes; that fails symmetrically
        # before any collective, so we can skip the shape and keep the group intact.
        if rank == 0:
            print(
                f"[deepep_normal] hidden={hidden} experts={num_experts} topk={num_topk} "
                f"ranks={num_ranks} nvl_buffer={DEEPEP_NORMAL_NVL_BYTES / 1e6:.0f}MB sms={sms_list}",
                flush=True,
            )
        try:
            buffer = deep_ep.Buffer(
                group,
                DEEPEP_NORMAL_NVL_BYTES,
                0,
                low_latency_mode=False,
                num_qps_per_rank=1,
                explicitly_destroy=True,
            )
        except Exception:
            if rank == 0:
                print(
                    f"[deepep_normal] SKIP (buffer alloc) hidden={hidden} experts={num_experts} "
                    f"topk={num_topk}:\n{traceback.format_exc()}",
                    flush=True,
                )
            dist.barrier()
            continue

        # A failure inside the benchmark leaves the process group in an undefined
        # state (a half-finished collective), so we cannot safely continue: log the
        # full traceback and re-raise. The orchestrator isolates each shape in its
        # own spawn, so this only tears down the current shape. Rows are flushed
        # per (sms, token) so a mid-shape crash keeps everything collected so far.
        try:
            for num_sms in sms_list:
                for num_tokens in tokens:
                    token_metrics: list[dict] = []
                    args = SimpleNamespace(
                        num_tokens=num_tokens,
                        hidden=hidden,
                        num_topk=num_topk,
                        num_experts=num_experts,
                    )
                    # Both hooks are required now that correctness checking is on.
                    # The previous inspect-gated skip degraded silently in both
                    # directions: without ``metrics_out`` the sweep runs to
                    # completion and writes ZERO rows while every shape reports
                    # success (only the final "produced no output" check notices,
                    # hours later), and without ``do_check`` upstream's default
                    # ``True`` applies -- the opposite of what the call site says.
                    # A stale checkout is how the GB200 TypeError happened in the
                    # first place, so fail on the first shape instead of guessing.
                    accepted = inspect.signature(test_intranode.test_main).parameters
                    missing = [name for name in ("do_check", "metrics_out") if name not in accepted]
                    if missing:
                        raise RuntimeError(
                            f"vendored test_intranode.test_main lacks {missing}; refusing to collect. "
                            "collector/wideep/sglang/deepep/test_intranode.py is stale."
                        )
                    # Reseed per point so a point's routing is a function of
                    # (seed, rank, num_tokens) alone: reproducible across reruns and
                    # identical across systems for the same point.
                    torch.manual_seed(DEEPEP_NORMAL_SEED + rank)
                    test_intranode.test_main(
                        args,
                        num_sms,
                        local_rank,
                        num_ranks,
                        rank,
                        buffer,
                        group,
                        do_check=num_tokens <= DEEPEP_NORMAL_CHECK_MAX_TOKENS,
                        metrics_out=token_metrics if rank == 0 else None,
                    )
                    dist.barrier()
                    flush(token_metrics)
        except Exception:
            if rank == 0:
                print(
                    f"[deepep_normal] FAILED hidden={hidden} experts={num_experts} topk={num_topk}:\n"
                    f"{traceback.format_exc()}",
                    flush=True,
                )
            try:
                buffer.destroy()
            except Exception:
                pass
            raise
        buffer.destroy()
        dist.barrier()

    if rank == 0:
        print(f"[deepep_normal] worker done, {written} rows this shape-group in {output_path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def run_deepep_normal_fullnode(perf_filename="wideep_deepep_normal_perf.txt", *, device=None, limit=None, cases=None):
    """Run the single-node, full-node DeepEP normal-mode sweep.

    Spawns one process per visible GPU (the full node), sweeps MoE shapes x SM
    counts x tokens, and writes ``wideep_deepep_normal_perf.txt`` to the current
    directory so ``collect.py`` can finalize it as parquet. ``device`` is
    accepted for registry-signature compatibility and ignored (the job owns all
    GPUs).
    """
    import torch
    import torch.multiprocessing as mp

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("DeepEP normal collection requires at least one visible CUDA device")

    if cases is None:
        cases = get_deepep_normal_test_cases()
        # Optional single-shape selection (DEEPEP_NORMAL_SHAPE_INDEX) so a SLURM job
        # array can run one shape per fresh node. A hard CUDA fault leaves node-level
        # NVSHMEM/IBGDA state dirty and cascades into later shapes on the same node;
        # isolating one shape per node sidesteps that entirely.
        shape_index = os.environ.get("DEEPEP_NORMAL_SHAPE_INDEX")
        if shape_index is not None and shape_index != "":
            idx = int(shape_index)
            if idx < 0 or idx >= len(cases):
                raise RuntimeError(f"DEEPEP_NORMAL_SHAPE_INDEX={idx} out of range (0..{len(cases) - 1})")
            cases = [cases[idx]]
        elif limit is not None:
            cases = cases[:limit]
    else:
        cases = list(cases)
    if not cases:
        raise RuntimeError("No MoE shapes resolved for DeepEP normal collection")

    sms_list = _sms_list()
    if not sms_list:
        raise RuntimeError("No SM counts resolved for DeepEP normal collection")

    tokens = _tokens_list()
    if not tokens:
        raise RuntimeError("No token counts resolved for DeepEP normal collection")

    # Resolve the device name inside the worker (rank 0) so the parent never
    # creates a CUDA context that would persist across the per-shape spawns.
    device_name = None
    try:
        from collector.wideep.sglang import dataset_version_label
    except ModuleNotFoundError:  # running with collector/ on sys.path
        from wideep.sglang import dataset_version_label

    version = dataset_version_label("DEEPEP_NORMAL_VERSION", "deepep_normal")

    output_path = os.path.join(os.getcwd(), str(perf_filename))

    # Single-node distributed bootstrap (WORLD_SIZE = number of NODES = 1).
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"

    print(
        f"[deepep_normal] starting full-node collection: {num_gpus} GPUs, {len(cases)} shapes, "
        f"sms={sms_list}, tokens={tokens}, "
        f"expecting {len(cases) * len(sms_list) * len(tokens)} rows, "
        f"checks on for num_tokens <= {DEEPEP_NORMAL_CHECK_MAX_TOKENS}",
        flush=True,
    )

    # Isolate each shape in its own spawn job: a hard CUDA fault (e.g. illegal
    # memory access) in one shape corrupts the context for every rank and cannot
    # be caught in-process, so a single mp.spawn over all shapes would lose every
    # later shape. One spawn per shape contains the blast radius; rows are flushed
    # per (sms, token), so completed work always lands on disk.
    succeeded, failed = 0, []
    for idx, case in enumerate(cases, start=1):
        os.environ["MASTER_PORT"] = str(_free_port())
        print(
            f"[deepep_normal] === shape {idx}/{len(cases)}: hidden={case['hidden_size']} "
            f"experts={case['num_experts']} topk={case['topk']} ===",
            flush=True,
        )
        try:
            mp.spawn(
                _normal_worker,
                args=(
                    num_gpus,
                    [case],
                    tokens,
                    sms_list,
                    output_path,
                    device_name,
                    version,
                ),
                nprocs=num_gpus,
                join=True,
            )
            succeeded += 1
        except Exception:
            failed.append(case)
            print(
                f"[deepep_normal] shape FAILED (isolated, continuing): {case}\n{traceback.format_exc()}",
                flush=True,
            )

    print(f"[deepep_normal] done: {succeeded}/{len(cases)} shapes ok, {len(failed)} failed", flush=True)
    if failed:
        print(f"[deepep_normal] failed shapes: {failed}", flush=True)

    if not Path(output_path).exists():
        raise RuntimeError(f"DeepEP normal collection produced no output at {output_path}")

    covered_shapes = _verify_axis_coverage(output_path, sms_list, tokens)
    print(
        f"[deepep_normal] axis coverage verified: {covered_shapes} shapes x {len(sms_list)} sms x {len(tokens)} tokens",
        flush=True,
    )
    if covered_shapes < len(cases):
        # Whole-shape loss, which `_verify_axis_coverage` deliberately does not
        # police. Not fatal -- a shape can be skipped for `num_experts % num_ranks`
        # or dropped on buffer allocation, both logged by the worker -- but it must
        # be visible next to the "complete" message, because provenance will not
        # show it.
        print(
            f"[deepep_normal] WARNING: {len(cases) - covered_shapes} of {len(cases)} shapes produced "
            "no rows (skipped or failed); see the per-shape SKIP/FAILED lines above",
            flush=True,
        )
    print(f"[deepep_normal] collection complete: {output_path}", flush=True)
    return {"succeeded": succeeded, "failed": failed}


if __name__ == "__main__":
    run_deepep_normal_fullnode()
