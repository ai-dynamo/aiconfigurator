# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-node collector orchestration.

Collectives such as DeepEP own every visible GPU and cannot run in the normal
per-GPU worker pool. This module keeps their case selection, checkpointing, and
failure mapping out of the top-level backend entrypoint.
"""

import os
import random
import traceback
from datetime import datetime

from collector.capabilities import filter_cases
from collector.helper import create_test_case_id
from collector.version_resolver import _check_compat

SGLANG_FULLNODE_OPS = {
    "deepep_ll": "DEEPEP_LL_SHAPE_INDEX",
    "deepep_normal": "DEEPEP_NORMAL_SHAPE_INDEX",
}


def _resume_errors(module_name, resume_tracker, resume_options, logger) -> list[dict]:
    if not (resume_options and resume_options.get("resume")):
        return []
    unresolved = resume_tracker.unresolved_failed_count()
    if not unresolved:
        return []
    logger.warning(
        f"{module_name}: checkpoint holds {unresolved} unresolved failed tasks "
        "(skipped on resume; rerun with --resume-retry-failed to retry)"
    )
    return [
        {
            "module": module_name,
            "task_id": "resume_unresolved",
            "error_type": "UnresolvedFailures",
            "error_message": f"checkpoint holds {unresolved} unresolved failed tasks",
            "classification": "unresolved_from_checkpoint",
            "timestamp": datetime.now().isoformat(),
        }
    ]


def select_cases(op: str, cases: list, limit: int | None) -> list:
    """Apply the explicit job-array shape selector or the common case limit."""
    shape_index_env = SGLANG_FULLNODE_OPS[op]
    shape_index = os.environ.get(shape_index_env)
    if shape_index is not None and shape_index != "":
        idx = int(shape_index)
        if idx < 0 or idx >= len(cases):
            raise RuntimeError(f"{shape_index_env}={idx} out of range (0..{len(cases) - 1})")
        return [cases[idx]]
    if limit is not None:
        return cases[:limit]
    return cases


def collect_sglang_fullnode_op(
    collection: dict,
    *,
    runtime_version: str,
    limit: int | None,
    shuffle: bool,
    shuffle_seed: int,
    backend: str,
    resume_options: dict | None,
    model_path: str | None,
    case_plan,
    sm_version: int | None,
    case_filters: list[str] | None,
    get_test_cases_for_model,
    resume_checkpoint_cls,
    logger,
) -> list[dict]:
    """Run one SGLang full-node op with normal V3 checkpoint evidence."""
    op = collection["type"]
    full_name = f"{collection['name']}.{op}"
    if case_plan is not None and not case_plan.has_op(op):
        logger.info(f"Skipping {full_name} — not in collector v2 case plan")
        return []

    module_name = collection["module"]
    get_module = __import__(module_name, fromlist=[collection["get_func"]])
    run_module = __import__(module_name, fromlist=[collection["run_func"]])

    declared = getattr(get_module, "__compat__", None)
    if declared:
        try:
            if not _check_compat(declared, runtime_version):
                raise RuntimeError(
                    f"module {module_name} declares __compat__={declared!r}, runtime is v{runtime_version}"
                )
        except ValueError as e:
            raise RuntimeError(f"invalid __compat__ {declared!r}: {e}") from e

    get_func = getattr(get_module, collection["get_func"])
    run_func = getattr(run_module, collection["run_func"])
    run_func_name = collection["run_func"]

    cases = get_test_cases_for_model(get_func, model_path)
    cases, _dropped = filter_cases(cases, op=op, sm_version=sm_version)
    if case_filters:
        before_count = len(cases)
        cases = [case for case in cases if any(fragment in str(case) for fragment in case_filters)]
        logger.info(f"{op}: --case-filter kept {len(cases)}/{before_count} cases")
    if shuffle:
        rng = random.Random(shuffle_seed)
        rng.shuffle(cases)
    cases = select_cases(op, cases, limit)
    if not cases:
        return [
            {
                "module": full_name,
                "error_type": "ModuleCollectionFailure",
                "error_message": f"No test cases resolved for full-node op {op}",
                "classification": "unexpected",
                "timestamp": datetime.now().isoformat(),
            }
        ]

    raw_task_infos = [
        {
            "id": create_test_case_id(case, run_func_name, full_name),
            "params": case,
            "index": i,
        }
        for i, case in enumerate(cases)
    ]

    checkpoint_dir = (
        resume_options.get("checkpoint_dir", ".collector_checkpoint") if resume_options else ".collector_checkpoint"
    )
    resume_tracker = resume_checkpoint_cls(
        backend=backend,
        module_name=full_name,
        run_func_name=run_func_name,
        checkpoint_dir=checkpoint_dir,
        framework_version=runtime_version,
        sm_version=sm_version,
    )
    if resume_options and resume_options.get("resume"):
        resume_tracker.load_existing()
        task_infos = resume_tracker.filter_done(raw_task_infos, retry_failed=resume_options.get("retry_failed", False))
    else:
        task_infos = raw_task_infos

    if not task_infos:
        logger.info(f"{full_name}: no full-node tasks to run")
        return _resume_errors(full_name, resume_tracker, resume_options, logger)

    logger.info(f"Running {full_name} full-node collection (bypassing per-GPU worker pool)")
    errors: list[dict] = []
    task_by_id = {task_info["id"]: task_info["params"] for task_info in task_infos}
    id_by_case = {str(task_info["params"]): task_info["id"] for task_info in task_infos}
    try:
        # The runner consumes the selected cases as one batch, so persist its
        # entire pending provenance set atomically before handing it control.
        # An interrupted resume then extends this set; only the successful
        # common parquet/sidecar finalizer closes it.
        resume_tracker.mark_attempted_many(task_by_id)
        result = run_func(
            perf_filename=collection["perf_filename"],
            limit=None,
            cases=[task_info["params"] for task_info in task_infos],
        )
        if not isinstance(result, dict) or "failed" not in result:
            raise RuntimeError(f"{full_name} runner must return a result containing its failed case list")
        failed_cases = result["failed"]
        unknown_failed = [case for case in failed_cases if str(case) not in id_by_case]
        if unknown_failed:
            raise RuntimeError(
                f"{full_name} runner reported failures outside the requested case plan: {unknown_failed}"
            )
        failed_ids = {id_by_case[str(case)] for case in failed_cases if str(case) in id_by_case}
        for task_id in task_by_id:
            if task_id in failed_ids:
                resume_tracker.mark_failed(task_id)
            else:
                resume_tracker.mark_passed(task_id)
        for task_id in sorted(failed_ids):
            errors.append(
                {
                    "module": full_name,
                    "task_id": task_id,
                    "task_params": str(task_by_id[task_id]),
                    "error_type": "FullNodeCaseFailure",
                    "error_message": "full-node runner reported failed case",
                    "classification": "unexpected",
                    "timestamp": datetime.now().isoformat(),
                }
            )
    except Exception as e:
        logger.exception(f"{full_name} full-node collection failed")
        for task_id in task_by_id:
            resume_tracker.mark_failed(task_id)
        errors.append(
            {
                "module": full_name,
                "task_id": "fullnode_collection",
                "error_type": "FullNodeCollectionFailure",
                "error_message": str(e),
                "classification": "unexpected",
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }
        )
    finally:
        resume_tracker.flush(force=True)

    errors.extend(_resume_errors(full_name, resume_tracker, resume_options, logger))
    return errors
