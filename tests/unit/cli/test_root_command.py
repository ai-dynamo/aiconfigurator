# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.main import main

pytestmark = pytest.mark.unit


def test_root_help_omits_webapp_command(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "webapp" not in output
    assert "cli" in output
    assert "version" in output


def test_webapp_command_is_not_registered(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["webapp"])

    assert exc_info.value.code == 2
    assert "invalid choice: 'webapp'" in capsys.readouterr().err
