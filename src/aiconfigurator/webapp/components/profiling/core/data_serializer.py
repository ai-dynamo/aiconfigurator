# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data serialization for frontend visualization.

Converts profiling results into JSON format for Chart.js and DataTables.
"""

import json

from aiconfigurator.webapp.components.profiling.constants import PLOTLY_COLORS


def serialize_profiling_data(
    prefill_results,
    decode_results,
    prefill_table_data,
    decode_table_data,
    cost_table_data,
    isl,
    osl,
    ttft,
    itl,
    allow_confirm_datapoint=False,
):
    """
    Serialize profiling data to JSON for frontend consumption.

    Args:
        allow_confirm_datapoint: If True, adds "Select" button for user confirmation

    Returns JSON string with all data for Chart.js and DataTables.
    """
    num_gpus_list, ttft_list, thpt_per_gpu_list = prefill_results

    data = {
        "settings": {
            "allow_confirm_datapoint": allow_confirm_datapoint,
        },
        "prefill": {
            "chart": {
                "labels": [f"{gpu} GPU{'s' if gpu > 1 else ''}" for gpu in num_gpus_list],
                "datasets": [
                    {
                        "label": "Prefill Performance",
                        "data": [
                            {
                                "x": ttft,
                                "y": thpt,
                                "gpu": gpu,
                                "tableIdx": idx,
                                "gpuLabel": f"{gpu} GPU{'s' if gpu > 1 else ''}",
                            }
                            for idx, (gpu, ttft, thpt) in enumerate(zip(num_gpus_list, ttft_list, thpt_per_gpu_list))
                        ],
                        "backgroundColor": PLOTLY_COLORS[0],
                        "borderColor": PLOTLY_COLORS[0],
                    }
                ],
                "target_line": {"value": ttft, "label": f"Target TTFT: {ttft} ms"},
                "axes": {
                    "x": {"title": "Time to First Token (ms)", "min": 0},
                    "y": {"title": "Prefill Throughput per GPU (tokens/s/GPU)", "min": 0},
                },
            },
            "table": {
                "columns": ["GPUs", "TTFT (ms)", "Throughput (tokens/s/GPU)", "Action"],
                "data": prefill_table_data,
            },
        },
        "decode": {
            "chart": {
                "datasets": [],
                "target_line": {"value": itl, "label": f"Target ITL: {itl} ms"},
                "axes": {
                    "x": {"title": "Inter Token Latency (ms)", "min": 0},
                    "y": {"title": "Decode Throughput per GPU (tokens/s/GPU)", "min": 0},
                },
            },
            "table": {
                "columns": ["GPUs", "ITL (ms)", "Throughput (tokens/s/GPU)", "Action"],
                "data": decode_table_data,
            },
        },
        "cost": {
            "chart": {
                "datasets": [],
                "axes": {
                    "x": {"title": "Tokens per User", "min": 0},
                    "y": {"title": "GPU Hours", "min": 0},
                },
                "title": f"GPU Hours Per 1000 i{isl}o{osl} requests",
            },
            "table": {
                "columns": [
                    "TTFT (ms)",
                    "Prefill Thpt (tokens/s/GPU)",
                    "ITL (ms)",
                    "Decode Thpt (tokens/s/GPU)",
                    "Tokens/User",
                    "GPU Hours",
                    "Action",
                ],
                "data": cost_table_data,
            },
        },
    }

    # Build decode chart datasets (one per GPU configuration)
    table_idx = 0
    for idx, decode_result in enumerate(decode_results):
        num_gpus = decode_result[0]
        itl_list = decode_result[1]
        thpt_per_gpu_list = decode_result[2]

        color = PLOTLY_COLORS[idx % len(PLOTLY_COLORS)]
        dataset = {
            "label": f"{num_gpus} GPU{'s' if num_gpus > 1 else ''}",
            "data": [
                {"x": itl, "y": thpt, "tableIdx": table_idx + i}
                for i, (itl, thpt) in enumerate(zip(itl_list, thpt_per_gpu_list))
            ],
            "backgroundColor": color,
            "borderColor": color,
        }
        data["decode"]["chart"]["datasets"].append(dataset)
        table_idx += len(itl_list)

    # Build cost chart datasets (one per TTFT curve)
    # Group cost_table_data by TTFT values
    cost_by_ttft = {}
    for idx, row in enumerate(cost_table_data):
        ttft_val = row[0]  # First column is TTFT
        if ttft_val not in cost_by_ttft:
            cost_by_ttft[ttft_val] = []
        cost_by_ttft[ttft_val].append(
            {
                "x": row[4],  # Tokens/User
                "y": row[5],  # Cost or GPU Hours
                "tableIdx": idx,
            }
        )

    for idx, (ttft_val, points) in enumerate(cost_by_ttft.items()):
        color = PLOTLY_COLORS[idx % len(PLOTLY_COLORS)]
        dataset = {
            "label": f"TTFT: {ttft_val:.2f}ms",
            "data": points,
            "backgroundColor": color,
            "borderColor": color,
        }
        data["cost"]["chart"]["datasets"].append(dataset)

    return json.dumps(data)
