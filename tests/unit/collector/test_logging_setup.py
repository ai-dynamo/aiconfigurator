# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from collector import helper

pytestmark = pytest.mark.unit


def test_setup_logging_honors_explicit_collector_log_dir(tmp_path, monkeypatch):
    log_dir = tmp_path / "stable-collector-logs"
    monkeypatch.setenv("COLLECTOR_LOG_DIR", str(log_dir))
    monkeypatch.setattr(helper, "_LOGGING_CONFIGURED", False)
    monkeypatch.setattr(helper, "_LOG_DIR", None)

    root_logger = helper.setup_logging(scope=["many"] * 40)

    try:
        assert log_dir.is_dir()
        assert helper.get_logging_config()["log_dir"] == log_dir
        assert (log_dir / "collector.log").exists()
        assert (log_dir / "collector_errors.log").exists()
    finally:
        for handler in list(root_logger.handlers):
            handler.close()
            root_logger.removeHandler(handler)
        logging.captureWarnings(False)
        helper._LOGGING_CONFIGURED = False
        helper._LOG_DIR = None
