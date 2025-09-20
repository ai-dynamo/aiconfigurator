import pytest
from unittest.mock import patch, MagicMock
from munch import Munch, DefaultMunch
import pandas as pd

from aiconfigurator.sdk.pareto_analysis import TaskConfig, TaskRunner
from aiconfigurator.sdk import common

@pytest.fixture
def base_task_config_args():
    return {
        "serving_mode": "agg",
        "model_name": "LLAMA2_7B",
        "system_name": "h200_sxm",
        "isl": 4000,
        "osl": 1000,
        "ttft": 1000,
        "tpot": 50,
    }

def test_task_config_disagg_init_no_yaml():
    with patch('aiconfigurator.sdk.pareto_analysis.get_latest_database_version', return_value='1.0'):
        with patch('aiconfigurator.sdk.pareto_analysis.get_database') as mock_get_db:
            mock_db = MagicMock()
            mock_db.system_spec = {'gpu': {'sm_version': 90}}
            mock_get_db.return_value = mock_db
            task_config = TaskConfig(
                serving_mode='disagg',
                model_name='LLAMA2_7B',
                system_name='h200_sxm',
                decode_system_name='h100_sxm'
            )
            assert task_config.config.serving_mode == 'disagg'
            assert task_config.config.model_name == 'LLAMA2_7B'
            assert task_config.config.prefill_worker_config.system_name == 'h200_sxm'
            assert task_config.config.decode_worker_config.system_name == 'h100_sxm'
            assert 'prefill_worker_config' in task_config.config
            assert 'decode_worker_config' in task_config.config

def test_task_config_with_yaml(base_task_config_args):
    yaml_config = {
        "runtime_config": {
            "isl": 5000
        },
        "worker_config": {
            "tp_list": [1, 2],
            "gemm_quant_mode": "float16"
        }
    }
    with patch('aiconfigurator.sdk.pareto_analysis.get_latest_database_version', return_value='1.0'):
        with patch('aiconfigurator.sdk.pareto_analysis.get_database') as mock_get_db:
            mock_db = MagicMock()
            mock_db.system_spec = {'gpu': {'sm_version': 90}}
            mock_get_db.return_value = mock_db
            
            task_config = TaskConfig(**base_task_config_args, yaml_config=yaml_config)

            assert task_config.config.runtime_config.isl == 5000
            assert task_config.config.worker_config.tp_list == [1, 2]
            assert task_config.config.worker_config.gemm_quant_mode == common.GEMMQuantMode.float16

def test_task_config_with_yaml_agg_thorough(base_task_config_args):
    yaml_config = {
        "runtime_config": {
            "isl": 5000,
            "osl": 1500,
            "ttft": 1200,
            "tpot": 60
        },
        "worker_config": {
            "tp_list": [1, 2, 8],
            "pp_list": [1, 4],
            "dp_list": [1, 2],
            "gemm_quant_mode": "float16",
            "kvcache_quant_mode": "fp8",
            "fmha_quant_mode": "fp8",
            "moe_quant_mode": "w4afp8",
            "comm_quant_mode": "fp8"
        }
    }
    with patch('aiconfigurator.sdk.pareto_analysis.get_latest_database_version', return_value='1.0'):
        with patch('aiconfigurator.sdk.pareto_analysis.get_database') as mock_get_db:
            mock_db = MagicMock()
            mock_db.system_spec = {'gpu': {'sm_version': 90}}
            mock_get_db.return_value = mock_db
            
            task_config = TaskConfig(**base_task_config_args, yaml_config=yaml_config)

            # Check runtime_config
            assert task_config.config.runtime_config.isl == 5000
            assert task_config.config.runtime_config.osl == 1500
            assert task_config.config.runtime_config.ttft == 1200
            assert task_config.config.runtime_config.tpot == 60

            # Check worker_config
            assert task_config.config.worker_config.tp_list == [1, 2, 8]
            assert task_config.config.worker_config.pp_list == [1, 4]
            assert task_config.config.worker_config.dp_list == [1, 2]
            assert task_config.config.worker_config.gemm_quant_mode == common.GEMMQuantMode.float16
            assert task_config.config.worker_config.kvcache_quant_mode == common.KVCacheQuantMode.fp8
            assert task_config.config.worker_config.fmha_quant_mode == common.FMHAQuantMode.fp8
            assert task_config.config.worker_config.moe_quant_mode == common.MoEQuantMode.w4afp8
            assert task_config.config.worker_config.comm_quant_mode == common.CommQuantMode.fp8

def test_task_config_with_yaml_disagg_thorough(base_task_config_args):
    disagg_args = base_task_config_args.copy()
    disagg_args["serving_mode"] = "disagg"
    disagg_args["decode_system_name"] = "h100_sxm"
    
    yaml_config = {
        "prefill_worker_config": {
            "tp_list": [1, 4],
            "gemm_quant_mode": "float16"
        },
        "decode_worker_config": {
            "tp_list": [1, 2],
            "gemm_quant_mode": "fp8_block"
        },
        "replica_config": {
            "max_gpu_per_replica": 256,
            "max_prefill_worker": 16
        },
        "advanced_tuning_config": {
            "prefill_correction_scale": 0.8,
            "decode_max_batch_size": 256
        }
    }
    with patch('aiconfigurator.sdk.pareto_analysis.get_latest_database_version', return_value='1.0'):
        with patch('aiconfigurator.sdk.pareto_analysis.get_database') as mock_get_db:
            mock_db = MagicMock()
            mock_db.system_spec = {'gpu': {'sm_version': 90}}
            mock_get_db.return_value = mock_db
            
            task_config = TaskConfig(**disagg_args, yaml_config=yaml_config)

            # Check prefill_worker_config
            assert task_config.config.prefill_worker_config.tp_list == [1, 4]
            assert task_config.config.prefill_worker_config.gemm_quant_mode == common.GEMMQuantMode.float16

            # Check decode_worker_config
            assert task_config.config.decode_worker_config.tp_list == [1, 2]
            assert task_config.config.decode_worker_config.gemm_quant_mode == common.GEMMQuantMode.fp8_block

            # Check replica_config
            assert task_config.config.replica_config.max_gpu_per_replica == 256
            assert task_config.config.replica_config.max_prefill_worker == 16

            # Check advanced_tuning_config
            assert task_config.config.advanced_tuning_config.prefill_correction_scale == 0.8
            assert task_config.config.advanced_tuning_config.decode_max_batch_size == 256

@patch('aiconfigurator.sdk.pareto_analysis.agg_pareto')
@patch('aiconfigurator.sdk.pareto_analysis.get_database')
def test_task_runner_agg(mock_get_db, mock_agg_pareto, base_task_config_args):
    mock_db = MagicMock()
    mock_db.system_spec = {'gpu': {'sm_version': 90}}
    mock_get_db.return_value = mock_db
    mock_agg_pareto.return_value = pd.DataFrame({'test': [1]})

    with patch('aiconfigurator.sdk.pareto_analysis.get_latest_database_version', return_value='1.0'):
        task_config = TaskConfig(**base_task_config_args)
        runner = TaskRunner()
        result = runner.run(task_config)

        assert not result.empty
        mock_agg_pareto.assert_called_once()

@patch('aiconfigurator.sdk.pareto_analysis.disagg_pareto')
@patch('aiconfigurator.sdk.pareto_analysis.get_database')
def test_task_runner_disagg(mock_get_db, mock_disagg_pareto, base_task_config_args):
    mock_db = MagicMock()
    mock_db.system_spec = {'gpu': {'sm_version': 90}}
    mock_get_db.return_value = mock_db
    mock_disagg_pareto.return_value = pd.DataFrame({'test': [1]})
    
    disagg_args = base_task_config_args.copy()
    disagg_args["serving_mode"] = "disagg"
    disagg_args["decode_system_name"] = "h100_sxm"

    with patch('aiconfigurator.sdk.pareto_analysis.get_latest_database_version', return_value='1.0'):
        task_config = TaskConfig(**disagg_args)
        runner = TaskRunner()
        result = runner.run(task_config)

        assert not result.empty
        mock_disagg_pareto.assert_called_once()
