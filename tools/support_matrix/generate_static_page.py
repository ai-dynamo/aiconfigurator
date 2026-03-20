#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate a self-contained static HTML page for the support matrix.

Reads support_matrix.csv, processes the data (version comparison, matrix
building, error grouping), and outputs a single index.html with embedded
JSON data, CSS, and vanilla JS.

Usage:
    python tools/support_matrix/generate_static_page.py \
        --csv src/aiconfigurator/systems/support_matrix.csv \
        --output docs/support-matrix/index.html
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TARGET_VERSIONS = {
    "vllm": "0.14.0",
    "sglang": "0.5.9",
    "trtllm": "1.2.0rc6",
}


_PHASE_ORDER = {"dev": 0, "a": 1, "b": 2, "rc": 3, "release": 4, "post": 5}


def _parse_version_tuple(ver_str):
    """
    Parse a PEP 440-ish version string into a comparable tuple.
    Handles versions like '0.5.9', '1.2.0rc6', '0.5.8.post1'.
    Phase ordering: dev < a < b < rc < release < post.
    """
    if not ver_str:
        return (0, 0, 0, 0, 0, 0, 0)
    s = ver_str.strip()

    rc_match = re.match(r"^(\d[\d.]*)(?:rc|a|b)(\d+)(.*)$", s)
    post_match = re.match(r"^(\d[\d.]*)\.post(\d+)$", s)

    rc_post_match = re.match(r"^(\d[\d.]*)rc(\d+)\.post(\d+)$", s)
    if rc_post_match:
        base = rc_post_match.group(1)
        rc_num = int(rc_post_match.group(2))
        post_num = int(rc_post_match.group(3))
        parts = [int(x) for x in base.split(".") if x]
        while len(parts) < 3:
            parts.append(0)
        # rc6.post3 sorts between rc6 and rc7
        return tuple(parts) + (_PHASE_ORDER["rc"], rc_num, 1, post_num)

    if rc_match:
        base = rc_match.group(1)
        rc_num = int(rc_match.group(2))
        parts = [int(x) for x in base.split(".") if x]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts) + (_PHASE_ORDER["rc"], rc_num, 0, 0)
    elif post_match:
        base = post_match.group(1)
        post_num = int(post_match.group(2))
        parts = [int(x) for x in base.split(".") if x]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts) + (_PHASE_ORDER["post"], post_num, 0, 0)
    else:
        parts = []
        for x in s.split("."):
            try:
                parts.append(int(x))
            except ValueError:
                break
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts) + (_PHASE_ORDER["release"], 0, 0, 0)


def _version_ge(a, b):
    """Return True if version string a >= version string b."""
    return _parse_version_tuple(a) >= _parse_version_tuple(b)


def _version_sort_key(ver_str):
    return _parse_version_tuple(ver_str)


def _extract_error_signature(err_msg):
    """
    Extract a short error signature from a full traceback for grouping.
    Ported from support_matrix_tab.py.
    """
    if not err_msg:
        return "No error message"
    text = str(err_msg).strip()
    if not text:
        return "No error message"
    lines = [ln.strip() for ln in text.replace("\\n", "\n").split("\n") if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if re.match(r"^[A-Za-z][A-Za-z0-9_.]*Error[^:]*:", line) or re.match(
            r"^[A-Za-z][A-Za-z0-9_.]*Exception[^:]*:", line
        ):
            return line[:500]
    if lines:
        return lines[-1][:500]
    return text[:500] if len(text) > 500 else text


def load_csv(csv_path):
    """Load support_matrix.csv into a list of dicts."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def get_latest_supported_version(rows, huggingface_id, system, backend):
    """
    Determine the latest supported version for a (model, system, backend) tuple.
    Ported from support_matrix_tab.py but operates on plain dicts.

    Returns (version_str | "FAIL" | None, at_target: bool, error_msg | None).
    """
    subset = [
        r for r in rows if r["HuggingFaceID"] == huggingface_id and r["System"] == system and r["Backend"] == backend
    ]
    if not subset:
        return (None, False, None)

    version_has_fail = {}
    version_has_pass = {}
    version_error_msgs = {}

    for row in subset:
        version = row["Version"]
        status = row["Status"]
        error_msg = (row.get("ErrMsg") or "").strip() or None

        if status == "FAIL":
            version_has_fail[version] = True
            if error_msg:
                if version in version_error_msgs:
                    existing = version_error_msgs[version]
                    if existing and existing != error_msg:
                        version_error_msgs[version] = f"{existing} | {error_msg}"
                else:
                    version_error_msgs[version] = error_msg
        elif status == "PASS":
            version_has_pass[version] = True

    if not version_has_fail and not version_has_pass:
        return (None, False, None)

    target_ver = TARGET_VERSIONS.get(backend)

    def at_or_above_target(ver_str):
        if target_ver is None:
            return True
        return _version_ge(ver_str, target_ver)

    if target_ver is not None:
        if target_ver in version_has_pass:
            return (target_ver, at_or_above_target(target_ver), None)
        if target_ver in version_has_fail:
            msg = version_error_msgs.get(target_ver, "No error message available")
            return ("FAIL", False, msg)

    passing = sorted(version_has_pass.keys(), key=_version_sort_key, reverse=True)
    if passing:
        v = passing[0]
        return (v, at_or_above_target(v), None)

    all_msgs = []
    for version in sorted(version_has_fail.keys(), key=_version_sort_key, reverse=True):
        if version in version_error_msgs:
            all_msgs.append(f"{version}: {version_error_msgs[version]}")
    return ("FAIL", False, " | ".join(all_msgs) if all_msgs else "No error message available")


def build_matrix_for_mode(rows, system_name, mode_filter):
    """
    Build the matrix data for a single (system, mode) combination.
    Returns a list of row dicts and a list of backend names.
    """
    subset = [r for r in rows if r["System"] == system_name]
    if mode_filter != "all":
        subset = [r for r in subset if r["Mode"] == mode_filter]
    if not subset:
        return [], []

    models = sorted(set(r["HuggingFaceID"] for r in subset))
    backends = sorted(set(r["Backend"] for r in subset))

    matrix_rows = []
    for model in models:
        cells = {}
        for backend in backends:
            version, at_target, error_msg = get_latest_supported_version(subset, model, system_name, backend)

            # For "all" mode, determine if support is partial (one mode passes, other fails)
            status = "pass"
            if version is None or version == "FAIL":
                status = "fail"

            partial_error = None
            if mode_filter == "all" and status == "pass":
                agg_rows = [r for r in subset if r["Mode"] == "agg"]
                disagg_rows = [r for r in subset if r["Mode"] == "disagg"]
                agg_ver, _, agg_err = (
                    get_latest_supported_version(agg_rows, model, system_name, backend)
                    if agg_rows
                    else (None, False, None)
                )
                disagg_ver, _, disagg_err = (
                    get_latest_supported_version(disagg_rows, model, system_name, backend)
                    if disagg_rows
                    else (None, False, None)
                )
                agg_pass = agg_ver is not None and agg_ver != "FAIL"
                disagg_pass = disagg_ver is not None and disagg_ver != "FAIL"
                if agg_pass != disagg_pass:
                    status = "partial"
                    if not agg_pass:
                        partial_error = f"[agg FAIL] {agg_err or 'No error message available'}"
                    else:
                        partial_error = f"[disagg FAIL] {disagg_err or 'No error message available'}"

            if version is None:
                cells[backend] = {"version": "FAIL", "status": "fail", "error": "No data available"}
            elif version == "FAIL":
                cells[backend] = {"version": "FAIL", "status": "fail", "error": error_msg}
            else:
                cells[backend] = {"version": version, "status": status, "error": partial_error}
        matrix_rows.append({"model": model, "cells": cells})

    return matrix_rows, backends


def build_top_errors(rows, system_name, mode_filter, top_n=10):
    """Build top-N error signatures for a (system, mode) combination."""
    subset = [r for r in rows if r["System"] == system_name and r["Status"] == "FAIL"]
    if mode_filter != "all":
        subset = [r for r in subset if r["Mode"] == mode_filter]
    if not subset:
        return []

    signatures = [_extract_error_signature(r.get("ErrMsg", "")) for r in subset]
    counts = Counter(signatures)
    return [{"signature": sig, "count": cnt} for sig, cnt in counts.most_common(top_n)]


def build_full_data(rows):
    """Build the complete data structure for all systems and modes."""
    systems_set = sorted(set(r["System"] for r in rows))
    modes = ["all", "agg", "disagg"]

    systems_data = {}
    for system in systems_set:
        system_entry = {"matrix": {}, "top_errors": {}, "backends": []}
        for mode in modes:
            matrix_rows, backends = build_matrix_for_mode(rows, system, mode)
            system_entry["matrix"][mode] = matrix_rows
            system_entry["top_errors"][mode] = build_top_errors(rows, system, mode)
            if mode == "all" and backends:
                system_entry["backends"] = backends
        systems_data[system] = system_entry

    total_pass = sum(1 for r in rows if r["Status"] == "PASS")
    total_fail = sum(1 for r in rows if r["Status"] == "FAIL")

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_versions": TARGET_VERSIONS,
        "summary": {"total_rows": len(rows), "pass": total_pass, "fail": total_fail},
        "systems": systems_data,
    }


def generate_html(data, template_path):
    """Read the HTML template and inject JSON data."""
    with open(template_path) as f:
        template = f.read()

    json_str = json.dumps(data, ensure_ascii=False)
    html = template.replace("/*__MATRIX_DATA__*/null", json_str, 1)
    return html


def main():
    default_csv = os.path.join(_REPO_ROOT, "src", "aiconfigurator", "systems", "support_matrix.csv")
    default_output = os.path.join(_REPO_ROOT, "docs", "support-matrix", "index.html")
    default_template = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_page_template.html")

    parser = argparse.ArgumentParser(description="Generate static support matrix HTML page")
    parser.add_argument("--csv", default=default_csv, help="Path to support_matrix.csv")
    parser.add_argument("--output", default=default_output, help="Output HTML file path")
    parser.add_argument("--template", default=default_template, help="Path to HTML template")
    args = parser.parse_args()

    print(f"Reading CSV from {args.csv}")
    rows = load_csv(args.csv)
    print(f"  Loaded {len(rows)} rows")

    print("Processing data...")
    data = build_full_data(rows)
    systems = list(data["systems"].keys())
    print(f"  Systems: {systems}")
    print(f"  Summary: {data['summary']['pass']} PASS, {data['summary']['fail']} FAIL")

    print(f"Reading template from {args.template}")
    html = generate_html(data, args.template)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Wrote {len(html):,} bytes to {args.output}")


if __name__ == "__main__":
    main()
