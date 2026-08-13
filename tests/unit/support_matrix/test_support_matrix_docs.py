# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_support_matrix_ui_preserves_encoder_unsupported_replay_command():
    html = Path("docs/support-matrix/index.html").read_text()

    assert "if (command.includes('tools/support_matrix/generate_support_matrix.py')) return command;" in html
