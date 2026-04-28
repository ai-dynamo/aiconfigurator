# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.agent import get_agent_text, list_agent_refs, list_agent_skills
from aiconfigurator.main import main

pytestmark = pytest.mark.unit


def test_agent_skill_text_uses_progressive_references():
    text = get_agent_text("usage")

    assert "name: aiconfigurator-usage" in text
    assert "Load References Only When Needed" in text
    assert "aiconfigurator agent usage --ref cli-modes" in text


def test_agent_reference_text_loads():
    text = get_agent_text("development", "testing")

    assert "python -m ruff check" in text
    assert "python -m pytest" in text


def test_agent_lists_skills_and_refs():
    assert list_agent_skills() == ("usage", "development")
    assert "examples" in list_agent_refs("usage")
    assert "generator" in list_agent_refs("development")


def test_agent_cli_prints_default_usage(capsys):
    main(["agent"])

    out = capsys.readouterr().out
    assert "name: aiconfigurator-usage" in out
    assert "aiconfigurator agent usage --ref examples" in out


def test_agent_cli_prints_reference(capsys):
    main(["agent", "development", "--ref", "repo-layout"])

    out = capsys.readouterr().out
    assert "src/aiconfigurator/sdk/" in out
