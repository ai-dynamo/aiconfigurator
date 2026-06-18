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
    assert "aiconfigurator agent usage --ref single-experiment-yaml" in text
    assert "aiconfigurator agent usage --ref experiment-template" in text
    assert "aiconfigurator agent usage --ref feature-pitfalls" in text


def test_agent_reference_text_loads():
    text = get_agent_text("development", "testing")

    assert "python -m ruff check" in text
    assert "python -m pytest" in text


def test_usage_reference_text_loads():
    yaml_text = get_agent_text("usage", "single-experiment-yaml")
    result_text = get_agent_text("usage", "result-interpretation")
    sdk_text = get_agent_text("usage", "sdk-step-breakdown")
    deployment_text = get_agent_text("usage", "deployment-bench")
    pitfalls_text = get_agent_text("usage", "feature-pitfalls")
    template_text = get_agent_text("usage", "experiment-template")

    assert "after a rough `default` run" in yaml_text
    assert "`tokens/s/gpu_cluster`" in result_text
    assert "Logs and Normalized Config" in result_text
    assert "--print-per-ops-latency" in sdk_text
    assert "--generated-config-version" in deployment_text
    assert "MTP / `nextn`" in pitfalls_text
    assert "Multi-Experiment Comparisons" in yaml_text
    assert "exp_disagg_full" in template_text
    assert 'database_mode: "SILICON"' in template_text


def test_agent_lists_skills_and_refs():
    assert list_agent_skills() == ("usage", "development")
    assert "examples" in list_agent_refs("usage")
    assert "single-experiment-yaml" in list_agent_refs("usage")
    assert "experiment-template" in list_agent_refs("usage")
    assert "feature-pitfalls" in list_agent_refs("usage")
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


def test_agent_cli_lists_new_usage_references(capsys):
    main(["agent", "--list"])

    out = capsys.readouterr().out
    assert "single-experiment-yaml" in out
    assert "experiment-template" in out
    assert "result-interpretation" in out
    assert "sdk-step-breakdown" in out
    assert "deployment-bench" in out
    assert "feature-pitfalls" in out
