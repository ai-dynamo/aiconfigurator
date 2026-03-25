# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.build]


@pytest.fixture
def set_cwd_to_sanity_check_dir():
    old_cwd = os.getcwd()
    os.chdir(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../../tools/sanity_check",
            )
        )
    )
    yield
    os.chdir(old_cwd)


def test_validate_database(set_cwd_to_sanity_check_dir):
    """
    Test that validate_database.ipynb runs successfully.
    """

    # Disable interactive backend for matplotlib
    os.environ["MPLBACKEND"] = "agg"

    # Import validate_database.ipynb jupyter notebook.
    # This will cause all the cells to run and any errors will be raised.
    import import_ipynb  # noqa: F401
    import validate_database  # noqa: F401
