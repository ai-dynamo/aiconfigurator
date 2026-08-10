# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.utils import get_model_config_from_model_path
from tools.support_matrix.support_matrix import SupportMatrix

pytestmark = pytest.mark.unit


def test_retired_models_are_excluded_only_from_default_matrix_generation():
    assert common.RetiredSupportMatrixHFModels
    assert common.RetiredSupportMatrixHFModels <= common.DefaultHFModels
    assert common.RetiredSupportMatrixHFModels.isdisjoint(common.SupportMatrixHFModels)
    assert common.SupportMatrixHFModels | common.RetiredSupportMatrixHFModels == common.DefaultHFModels

    matrix = SupportMatrix.__new__(SupportMatrix)
    assert matrix.get_models() == common.SupportMatrixHFModels


@pytest.mark.parametrize("model", sorted(common.RetiredSupportMatrixHFModels))
def test_retired_matrix_models_keep_bundled_offline_configs(model, monkeypatch):
    def fail_network(*_args, **_kwargs):
        raise AssertionError("retired bundled model unexpectedly attempted a network download")

    monkeypatch.setattr("aiconfigurator.sdk.utils._download_hf_config", fail_network)
    get_model_config_from_model_path.cache_clear()

    model_config = get_model_config_from_model_path(model)

    assert model_config["architecture"] in common.ARCHITECTURE_TO_MODEL_FAMILY
