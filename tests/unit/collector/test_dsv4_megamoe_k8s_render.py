# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from collector.sglang.dsv4_megamoe.render_k8s_indexed_job import _parse_node_selector

pytestmark = pytest.mark.unit


def test_parse_node_selector_accepts_comma_separated_key_value_pairs():
    assert _parse_node_selector("gpu=a,b=c") == [("gpu", "a"), ("b", "c")]


@pytest.mark.parametrize("value", ["missing_equals", "=value", "key="])
def test_parse_node_selector_rejects_malformed_entries(value):
    with pytest.raises(SystemExit, match="--node-selector entries must"):
        _parse_node_selector(value)
