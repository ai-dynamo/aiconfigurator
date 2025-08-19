# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from aiconfigurator.sdk import pareto_analysis
from aiconfigurator.sdk import config
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models import get_model_family, check_is_moe
from aiconfigurator.cli.helpers import DynamoConfig, add_dynamo_cli, build_dynamo_config, _dump_backend_file
from aiconfigurator.cli.backends import get_config_generator

import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.interpolate import interp1d
import yaml
import argparse
import plotext
import time
import os
import random
import matplotlib.pyplot as plt
import json
from prettytable import PrettyTable
from aiconfigurator import __version__


logger = logging.getLogger(__name__)

@dataclass
class WorkerConfig:
    """Configuration for a specific hardware and system setup."""
    system_name: str
    backend_name: str
    version: str
    gemm_quant_mode: common.GEMMQuantMode
    moe_quant_mode: common.MoEQuantMode
    kvcache_quant_mode: common.KVCacheQuantMode
    fmha_quant_mode: common.FMHAQuantMode
    comm_quant_mode: common.CommQuantMode    
    num_gpu_per_worker: List[int]
    tp_list: List[int]
    pp_list: List[int]
    dp_list: List[int] = field(default_factory=lambda: [1])
    moe_tp_list: List[int] = field(default_factory=lambda: [1])
    moe_ep_list: List[int] = field(default_factory=lambda: [1])

@dataclass
class DisaggReplicaConfig:
    num_gpu_per_replica: List[int] = None # None means no limit
    max_prefill_worker: int = 32
    max_decode_worker: int = 32
    max_gpu_per_replica: int = 0 # zero means no limit

@dataclass
class AIConfiguratorConfig:
    """Configuration for aiconfigurator"""
    # aiconfigurator yaml
    aiconfigurator_yaml_data: dict
    # model config section
    model_name: str
    is_moe: bool

    # MTP config section
    nextn: int # at most mtp5
    nextn_accept_rates: list # list of accept rates for nextn

    # deployment config section
    total_gpus: int # total GPUs you want to deploy

    # runtime config section
    isl: int  # input sequence length
    osl: int  # output sequence length
    ttft: float  # time to first token (ms)
    tpot: float  # time per output token (ms)

    # agg and disagg system config section
    agg_worker_config: Optional[WorkerConfig] = None
    disagg_replica_config: Optional[DisaggReplicaConfig] = None
    disagg_prefill_worker_config: Optional[WorkerConfig] = None
    disagg_decode_worker_config: Optional[WorkerConfig] = None

    # advanced tuning config section
    prefill_correction_scale: float = 1.0 # If you find the predicted prefill perf is too optimistic, you can set a scale factor to make it more realistic, throughput_corrected = throughput_predicted * prefill_correction_scale
    decode_correction_scale: float = 1.0 # If you find the predicted decode perf is too optimistic, you can set a scale factor to make it more realistic, throughput_corrected = throughput_predicted * decode_correction_scale
    prefill_max_batch_size: int = 1
    decode_max_batch_size: int = 512

@dataclass
class AIConfiguratorResult:
    """Result of Dynamo AIConfigurator"""
    chosen_system_type: str # "agg" or "disagg" or "none"    
    has_disagg_benefit: bool
    benefit_ratio: float # disagg_throughput / agg_throughput
    # Pareto frontiers
    agg_pareto: pd.DataFrame
    disagg_pareto: pd.DataFrame
    agg_interpolated_throughput: float
    disagg_interpolated_throughput: float
    agg_actual_best_throughput: float
    disagg_actual_best_throughput: float
    agg_best_config: pd.DataFrame
    disagg_best_config: pd.DataFrame
    backend_configs: Dict[common.BackendName, Dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Convert the aiconfigurator result to a dictionary.
        Returns:
            A dictionary containing the aiconfigurator result.
        """
        result_dict = {
            'chosen_system_type': self.chosen_system_type,
            'has_disagg_benefit': self.has_disagg_benefit,
            'benefit_ratio': self.benefit_ratio,
            'agg_actual_best_throughput': self.agg_actual_best_throughput,
            'disagg_actual_best_throughput': self.disagg_actual_best_throughput
        }
        return result_dict

class AIConfigurator:
    def __init__(self):
        pass
    
    def run(self, aiconfigurator_config: AIConfiguratorConfig) -> AIConfiguratorResult:
        """Main configuration workflow that executes all steps"""
        logger.info(f"Starting configuration for {aiconfigurator_config.model_name}")
        logger.info(f"Total GPUs to deploy on: {aiconfigurator_config.total_gpus}")
        logger.info(f"User scenario: ISL={aiconfigurator_config.isl}, OSL={aiconfigurator_config.osl}, "
                   f"TTFT={aiconfigurator_config.ttft}ms, TPOT={aiconfigurator_config.tpot}ms")
        if aiconfigurator_config.agg_worker_config:
            logger.info(f"Agg System: {aiconfigurator_config.agg_worker_config.system_name}")
        else:
            logger.info("Agg worker config is not provided, skipping agg configuration.")
        if aiconfigurator_config.disagg_replica_config:
            if aiconfigurator_config.disagg_prefill_worker_config and aiconfigurator_config.disagg_decode_worker_config:
                logger.info(f"Disagg Prefill Worker: {aiconfigurator_config.disagg_prefill_worker_config}")
                logger.info(f"Disagg Decode Worker: {aiconfigurator_config.disagg_decode_worker_config}")
        else:
            logger.info("Disagg prefill and decode config are not provided, skipping disagg configuration.")
        
        # Step 1: Run raw pareto analysis to get all data points
        logger.info("Running raw pareto analysis.It might take minutes to run...")
        raw_agg_df, raw_disagg_df = self._run_pareto_analysis(
            aiconfigurator_config=aiconfigurator_config
        )

        # Step 2: Get Pareto frontiers (tps/gpu vs. tps/user)
        logger.info("Generating Pareto frontiers for Agg and Disagg...")
        agg_pareto = pd.DataFrame()
        if raw_agg_df is not None and not raw_agg_df.empty:
            agg_pareto = pareto_analysis.get_pareto_front(raw_agg_df, 'tokens/s/user', 'tokens/s/gpu')
            logger.debug(f"Agg tps/gpu vs. tps/user Pareto front has {len(agg_pareto)} points.")
        else:
            logger.debug("No raw Agg data to generate Agg Pareto front.")

        disagg_pareto = pd.DataFrame()
        if raw_disagg_df is not None and not raw_disagg_df.empty:
            disagg_pareto = pareto_analysis.get_pareto_front(raw_disagg_df, 'tokens/s/user', 'tokens/s/gpu')
            logger.debug(f"Disagg tps/gpu vs. tps/user Pareto front has {len(disagg_pareto)} points.")
        else:
            logger.debug("No raw Disagg data to generate Disagg Pareto front.")

        # Step 3: Interpolate throughput at target TPOT, this is used as a reference
        logger.info("Interpolating throughput at target TPOT...")
        agg_interpolated_throughput = 0.0
        if not agg_pareto.empty:
            agg_interpolated_throughput = self._interpolate_throughput_at_tpot(agg_pareto, aiconfigurator_config.tpot)
            logger.debug(f"Agg: Interpolated tokens/s/gpu at {aiconfigurator_config.tpot}ms TPOT: {agg_interpolated_throughput:.2f}")
        else:
            logger.debug("Agg TPOT Pareto front is empty, skipping interpolation.")

        disagg_interpolated_throughput = 0.0
        if not disagg_pareto.empty:
            disagg_interpolated_throughput = self._interpolate_throughput_at_tpot(disagg_pareto, aiconfigurator_config.tpot)
            logger.debug(f"Disagg: Interpolated tokens/s/gpu at {aiconfigurator_config.tpot}ms TPOT: {disagg_interpolated_throughput:.2f}")
        else:
            logger.debug("Disagg TPOT Pareto front is empty, skipping interpolation.")

        # Step 4: Get top1 actual config under TPOT constraint for the whole cluster
        agg_actual_best_throughput = 0.0
        agg_best_config_df = self._get_best_config_under_tpot_constraint(aiconfigurator_config.total_gpus, agg_pareto, aiconfigurator_config.tpot)
        if not agg_best_config_df.empty:
            agg_actual_best_throughput = agg_best_config_df['tokens/s/gpu_cluster'].values[0]
            logger.debug(f"Agg: Actual best throughput under TPOT constraint: {agg_actual_best_throughput:.2f}")
        else:
            logger.debug("No actual Agg config met the TPOT constraint.")

        disagg_actual_best_throughput = 0.0
        disagg_best_config_df = self._get_best_config_under_tpot_constraint(aiconfigurator_config.total_gpus, disagg_pareto, aiconfigurator_config.tpot)
        if not disagg_best_config_df.empty:
            disagg_actual_best_throughput = disagg_best_config_df['tokens/s/gpu_cluster'].values[0]
            logger.debug(f"Disagg: Actual best throughput under TPOT constraint: {disagg_actual_best_throughput:.2f}")
        else:
            logger.debug("No actual Disagg config met the TPOT constraint.")

        # Step 5 (combines with decision): Compare performance and select overall best system/config
        has_disagg_benefit, benefit_ratio, chosen_system_type = self._determine_overall_best_system(
            agg_interpolated_throughput, disagg_interpolated_throughput,
            agg_actual_best_throughput, disagg_actual_best_throughput,
        )
        logger.info(f"Overall best system chosen: {chosen_system_type}")
        logger.info(f"Finished configuration for {aiconfigurator_config.model_name}")
        
        result = AIConfiguratorResult(
            chosen_system_type=chosen_system_type,
            has_disagg_benefit=has_disagg_benefit,
            benefit_ratio=benefit_ratio,
            agg_pareto=agg_pareto, # Store the tps/gpu vs tps/user Pareto
            disagg_pareto=disagg_pareto, # Store the tps/gpu vs tps/user Pareto
            agg_interpolated_throughput=agg_interpolated_throughput,
            disagg_interpolated_throughput=disagg_interpolated_throughput,
            agg_actual_best_throughput=agg_actual_best_throughput,
            disagg_actual_best_throughput=disagg_actual_best_throughput,
            agg_best_config=agg_best_config_df,
            disagg_best_config=disagg_best_config_df
        )
        
        return result
        
    def _run_pareto_analysis(self, aiconfigurator_config: AIConfiguratorConfig) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Run both Agg and Disagg pareto analysis based on provided hardware configs.
        Args:
            aiconfigurator_config: The config of the aiconfigurator.
        Returns:
            A tuple containing the Agg and Disagg Pareto analysis results.
            Return None if no pareto analysis is run.
        """
        
        agg_df: Optional[pd.DataFrame] = None
        disagg_df: Optional[pd.DataFrame] = None

        # Run Agg pareto analysis if config is provided
        if aiconfigurator_config.agg_worker_config:
            logger.info("Starting Agg Pareto Analysis...")
            agg_database = get_database(
                system=aiconfigurator_config.agg_worker_config.system_name,
                backend=aiconfigurator_config.agg_worker_config.backend_name,
                version=aiconfigurator_config.agg_worker_config.version
            )
            agg_parallel_config_list = pareto_analysis.enumerate_parallel_config(
                num_gpu_list=aiconfigurator_config.agg_worker_config.num_gpu_per_worker,
                tp_list=aiconfigurator_config.agg_worker_config.tp_list,
                pp_list=aiconfigurator_config.agg_worker_config.pp_list,
                dp_list=aiconfigurator_config.agg_worker_config.dp_list,
                moe_tp_list=aiconfigurator_config.agg_worker_config.moe_tp_list,
                moe_ep_list=aiconfigurator_config.agg_worker_config.moe_ep_list,
                is_moe=aiconfigurator_config.is_moe,
                backend=common.BackendName(aiconfigurator_config.agg_worker_config.backend_name)
            )
            
            if not agg_parallel_config_list:
                logger.error(f"No valid parallel config found for Agg on {aiconfigurator_config.model_name} with given constraints. Please double check your parallel configs.")
            else:
                for parallel_config in agg_parallel_config_list:
                    tp, pp, dp, moe_tp, moe_ep = parallel_config
                    logger.info(f"Enumerated Agg parallel config: tp={tp}, pp={pp}, dp={dp}, moe_tp={moe_tp}, moe_ep={moe_ep}")
                agg_df = pareto_analysis.ifb_pareto(
                    model_name=aiconfigurator_config.model_name,
                    runtime_config=config.RuntimeConfig(
                        isl=aiconfigurator_config.isl,
                        osl=aiconfigurator_config.osl,
                        ttft=aiconfigurator_config.ttft,
                        tpot=list(range(2,20,1))+list(range(20,300,5))
                    ),
                    database=agg_database,
                    backend_name=aiconfigurator_config.agg_worker_config.backend_name,
                    model_config=config.ModelConfig(
                        gemm_quant_mode=aiconfigurator_config.agg_worker_config.gemm_quant_mode,
                        kvcache_quant_mode=aiconfigurator_config.agg_worker_config.kvcache_quant_mode,
                        fmha_quant_mode=aiconfigurator_config.agg_worker_config.fmha_quant_mode,
                        moe_quant_mode=aiconfigurator_config.agg_worker_config.moe_quant_mode,
                        comm_quant_mode=aiconfigurator_config.agg_worker_config.comm_quant_mode,
                        nextn=aiconfigurator_config.nextn,
                        nextn_accept_rates=aiconfigurator_config.nextn_accept_rates
                    ),
                    parallel_config_list=agg_parallel_config_list
                )
            logger.info("Agg Pareto Analysis completed.")
        else:
            logger.info("No Agg config provided, skipping Agg Pareto Analysis.")

        # Run Disagg pareto analysis if config is provided
        if aiconfigurator_config.disagg_prefill_worker_config and aiconfigurator_config.disagg_decode_worker_config and aiconfigurator_config.disagg_replica_config:
            logger.info("Starting Disagg Pareto Analysis...")
            disagg_prefill_database = get_database(
                system=aiconfigurator_config.disagg_prefill_worker_config.system_name,
                backend=aiconfigurator_config.disagg_prefill_worker_config.backend_name,
                version=aiconfigurator_config.disagg_prefill_worker_config.version
            )
            disagg_prefill_parallel_config_list = pareto_analysis.enumerate_parallel_config(
                num_gpu_list=aiconfigurator_config.disagg_prefill_worker_config.num_gpu_per_worker,
                tp_list=aiconfigurator_config.disagg_prefill_worker_config.tp_list,
                pp_list=aiconfigurator_config.disagg_prefill_worker_config.pp_list,
                dp_list=aiconfigurator_config.disagg_prefill_worker_config.dp_list,
                moe_tp_list=aiconfigurator_config.disagg_prefill_worker_config.moe_tp_list,
                moe_ep_list=aiconfigurator_config.disagg_prefill_worker_config.moe_ep_list,
                is_moe=aiconfigurator_config.is_moe,
                backend=common.BackendName(aiconfigurator_config.disagg_prefill_worker_config.backend_name)
            )

            disagg_decode_database = get_database(
                system=aiconfigurator_config.disagg_decode_worker_config.system_name,
                backend=aiconfigurator_config.disagg_decode_worker_config.backend_name,
                version=aiconfigurator_config.disagg_decode_worker_config.version
            )
            disagg_decode_parallel_config_list = pareto_analysis.enumerate_parallel_config(
                num_gpu_list=aiconfigurator_config.disagg_decode_worker_config.num_gpu_per_worker,
                tp_list=aiconfigurator_config.disagg_decode_worker_config.tp_list,
                pp_list=aiconfigurator_config.disagg_decode_worker_config.pp_list,
                dp_list=aiconfigurator_config.disagg_decode_worker_config.dp_list,
                moe_tp_list=aiconfigurator_config.disagg_decode_worker_config.moe_tp_list,
                moe_ep_list=aiconfigurator_config.disagg_decode_worker_config.moe_ep_list,
                is_moe=aiconfigurator_config.is_moe,
                backend=common.BackendName(aiconfigurator_config.disagg_decode_worker_config.backend_name)
            )

            if not disagg_prefill_parallel_config_list or not disagg_decode_parallel_config_list:
                logger.error(f"No valid parallel config found for Disagg on {aiconfigurator_config.model_name} with given constraints. Please double check your parallel configs.")
            else:
                for prefill_parallel_config in disagg_prefill_parallel_config_list:
                    tp, pp, dp, moe_tp, moe_ep = prefill_parallel_config
                    logger.info(f"Enumerated Disagg prefill parallel config: tp={tp}, pp={pp}, dp={dp}, moe_tp={moe_tp}, moe_ep={moe_ep}")
                for decode_parallel_config in disagg_decode_parallel_config_list:
                    tp, pp, dp, moe_tp, moe_ep = decode_parallel_config
                    logger.info(f"Enumerated Disagg decode parallel config: tp={tp}, pp={pp}, dp={dp}, moe_tp={moe_tp}, moe_ep={moe_ep}")
                disagg_df = pareto_analysis.disagg_pareto(
                    model_name=aiconfigurator_config.model_name,
                    runtime_config=config.RuntimeConfig(
                        isl=aiconfigurator_config.isl,
                        osl=aiconfigurator_config.osl,
                        ttft=aiconfigurator_config.ttft,
                        tpot=list(range(1,20,1))+list(range(20,300,5))
                    ),
                    prefill_database=disagg_prefill_database,
                    prefill_backend_name=aiconfigurator_config.disagg_prefill_worker_config.backend_name,
                    prefill_model_config=config.ModelConfig(
                        gemm_quant_mode=aiconfigurator_config.disagg_prefill_worker_config.gemm_quant_mode,
                        kvcache_quant_mode=aiconfigurator_config.disagg_prefill_worker_config.kvcache_quant_mode,
                        fmha_quant_mode=aiconfigurator_config.disagg_prefill_worker_config.fmha_quant_mode,
                        moe_quant_mode=aiconfigurator_config.disagg_prefill_worker_config.moe_quant_mode,
                        comm_quant_mode=aiconfigurator_config.disagg_prefill_worker_config.comm_quant_mode,
                        nextn=aiconfigurator_config.nextn,
                        nextn_accept_rates=aiconfigurator_config.nextn_accept_rates
                    ),
                    prefill_parallel_config_list=disagg_prefill_parallel_config_list,
                    decode_database=disagg_decode_database,
                    decode_backend_name=aiconfigurator_config.disagg_decode_worker_config.backend_name,
                    decode_model_config=config.ModelConfig(
                        gemm_quant_mode=aiconfigurator_config.disagg_decode_worker_config.gemm_quant_mode,
                        kvcache_quant_mode=aiconfigurator_config.disagg_decode_worker_config.kvcache_quant_mode,
                        fmha_quant_mode=aiconfigurator_config.disagg_decode_worker_config.fmha_quant_mode,
                        moe_quant_mode=aiconfigurator_config.disagg_decode_worker_config.moe_quant_mode,
                        comm_quant_mode=aiconfigurator_config.disagg_decode_worker_config.comm_quant_mode,
                        nextn=aiconfigurator_config.nextn,
                        nextn_accept_rates=aiconfigurator_config.nextn_accept_rates
                    ),
                    decode_parallel_config_list=disagg_decode_parallel_config_list,
                    num_gpu_list=aiconfigurator_config.disagg_replica_config.num_gpu_per_replica,
                    max_num_gpu=aiconfigurator_config.disagg_replica_config.max_gpu_per_replica,
                    prefill_max_num_worker=aiconfigurator_config.disagg_replica_config.max_prefill_worker,
                    decode_max_num_worker=aiconfigurator_config.disagg_replica_config.max_decode_worker,
                    prefill_max_batch_size=aiconfigurator_config.prefill_max_batch_size,
                    decode_max_batch_size=aiconfigurator_config.decode_max_batch_size,
                    prefill_correction_scale=aiconfigurator_config.prefill_correction_scale,                    
                    decode_correction_scale=aiconfigurator_config.decode_correction_scale,
                )                
            logger.info("Disagg Pareto Analysis completed.")
        else:
            logger.info("No Disagg config provided, skipping Disagg Pareto Analysis.")
            
        return agg_df, disagg_df
    
    def _interpolate_throughput_at_tpot(self, df: Optional[pd.DataFrame], target_tpot: float) -> float:
        """
        Interpolates the throughput at a given TPOT. This is more for reference by reading the pareto frontier.
        Args:
            df: The DataFrame containing the throughput data.
            target_tpot: The target TPOT in ms.
        Returns:
            The interpolated throughput at the target TPOT.
        """
        if df is None or df.empty:
            return 0.0
        
        target_tps_user = 1000.0/target_tpot
        
        # Filter out points where tpot is not available or invalid
        df_filtered = df.dropna(subset=['tokens/s/user', 'tokens/s/gpu'])
        if df_filtered.empty or len(df_filtered) < 2:
            # Not enough points to interpolate, try to find closest or return 0
            if not df_filtered.empty:
                 # Fallback: find the point with tpot closest to target_tps_user
                closest_idx = (df_filtered['tokens/s/user'] - target_tps_user).abs().idxmin()
                return df_filtered.loc[closest_idx, 'tokens/s/gpu']
            return 0.0

        # Sort by tokens/s/user for interpolation
        df_sorted = df_filtered.sort_values(by='tokens/s/user')
        
        # Create interpolation functions
        # If target_tpot is outside the range, interp1d will extrapolate or error depending on fill_value
        # Using fill_value="extrapolate" can be risky.
        # It's often better to clamp to the nearest value if outside the range.
        min_tps_user, max_tps_user = df_sorted['tokens/s/user'].min(), df_sorted['tokens/s/user'].max()

        if target_tps_user < min_tps_user:
            return df_sorted.iloc[0]['tokens/s/gpu'] # Closest value at smallest tokens/s/user
        if target_tps_user > max_tps_user:
            return 0.0 # cannot meet the target tps_user
            
        interp_func = interp1d(df_sorted['tokens/s/user'], df_sorted['tokens/s/gpu'], kind='linear', fill_value="extrapolate")
        
        interpolated_throughput = float(interp_func(target_tps_user))
        return max(0.0, interpolated_throughput) # Ensure non-negative throughput

    def _get_best_config_under_tpot_constraint(self, 
                                               total_gpus: int,
                                               pareto_df: pd.DataFrame, 
                                               target_tpot: float) -> pd.DataFrame:
        """
        Finds the best actual config from a Pareto frontier DataFrame
        that meets the target_tpot constraint (tpot <= target_tpot)
        and maximizes 'tokens/s/gpu'.
        Args:
            pareto_df: The Pareto frontier DataFrame.
            target_tpot: The target TPOT in ms.
        Returns:
            A DataFrame containing the best config that meets the target_tpot constraint.
        """
        if pareto_df is None or pareto_df.empty:
            return pd.DataFrame()

        # Ensure 'tpot' and 'tokens/s/gpu' columns exist
        if 'tpot' not in pareto_df.columns or 'tokens/s/gpu' not in pareto_df.columns:
            logger.warning("Pareto DataFrame for _get_best_config_under_tpot_constraint is missing 'tpot' or 'tokens/s/gpu' columns.")
            return pd.DataFrame()

        candidate_configs = pareto_df[pareto_df['tpot'] <= target_tpot].copy()
        
        if not candidate_configs.empty:
            # compute achieved cluster-scale tokens/s/gpu
            candidate_configs['tokens/s/gpu_cluster'] = candidate_configs['tokens/s/gpu'] * \
                (total_gpus // candidate_configs['num_total_gpus']) * candidate_configs['num_total_gpus'] / total_gpus
            candidate_configs = candidate_configs.sort_values(by='tokens/s/gpu_cluster', ascending=False)
            logger.debug(f"actual replica-level throughputs: {candidate_configs['tokens/s/gpu'].iloc[0]:.2f} vs. actual cluster-level throughputs: {candidate_configs['tokens/s/gpu_cluster'].iloc[0]:.2f}")        
            return candidate_configs.head(1)
        else:
            # No config meets tpot <= target_tpot.
            # Optionally, one could return the one closest to target_tpot if no strict candidates exist.
            # For now, return empty if no config meets the criteria.
            logger.info(f"No config found on Pareto front with TPOT <= {target_tpot}ms.")
            return pd.DataFrame()

    def _determine_overall_best_system(self,
                                       agg_interpolated_throughput: float,
                                       disagg_interpolated_throughput: float,
                                       agg_actual_best_throughput: float,
                                       disagg_actual_best_throughput: float,
                                       ) -> Tuple[bool, float, str]:
        """
        Compares Agg and Disagg based on interpolated throughputs and selects the overall best
        actual configs.
        Args:
            agg_interpolated_throughput: The interpolated throughput of the Agg system.
            disagg_interpolated_throughput: The interpolated throughput of the Disagg system.
            agg_actual_best_throughput: The actual best throughput of the Agg system.
            disagg_actual_best_throughput: The actual best throughput of the Disagg system.
        Returns:
            has_disagg_benefit (bool): True if disagg offers higher throughput.
            benefit_ratio (float): Throughput ratio (disagg_throughput / agg_throughput).
            chosen_system_type (str): "agg", "disagg", or "none".
        """
        has_disagg_benefit = False
        benefit_ratio = 0.0
        chosen_system_type = "none"

        logger.info(f"Comparing systems: Agg interpolated throughput = {agg_interpolated_throughput:.2f}, Disagg interpolated throughput = {disagg_interpolated_throughput:.2f}")
        logger.info(f"Comparing systems: Agg actual best throughput = {agg_actual_best_throughput:.2f}, Disagg actual best throughput = {disagg_actual_best_throughput:.2f}")

        # Determine which system is better based on actual throughput
        if disagg_actual_best_throughput > 0 and agg_actual_best_throughput > 0:
            benefit_ratio = disagg_actual_best_throughput / agg_actual_best_throughput
            if disagg_actual_best_throughput > agg_actual_best_throughput:
                has_disagg_benefit = True
                chosen_system_type = "disagg"
                logger.debug(f"Disagg is preferred. Disagg vs. Agg benefit ratio: {benefit_ratio:.2f}")
            else: # Agg is better or equal
                chosen_system_type = "agg"
                logger.debug(f"Agg is preferred or equal to Disagg. Disagg vs. Agg benefit ratio: {benefit_ratio:.2f}")
        elif disagg_actual_best_throughput > 0: # Only Disagg matches the target TPOT
            has_disagg_benefit = True
            benefit_ratio = float('inf') # Disagg is infinitely better if Agg has zero/no throughput
            chosen_system_type = "disagg"
            logger.debug("Disagg meets target TPOT, Agg does not. Disagg chosen.")
        elif agg_actual_best_throughput > 0: # Only Agg has throughput
            # has_disagg_benefit remains False
            chosen_system_type = "agg"
            logger.debug("Agg meets target TPOT, Disagg does not. Agg chosen.")
        else: # Neither has throughput
            logger.debug("Neither Agg nor Disagg meets the target TPOT.")
            # chosen_system_type "none"
             
        return has_disagg_benefit, benefit_ratio, chosen_system_type
    
    def generate_backend_config(self,
                               aiconfigurator_result: AIConfiguratorResult,
                               aiconfigurator_config: AIConfiguratorConfig,
                               dynamo_config: DynamoConfig, generated_config_version: str) -> Dict[str, Dict[str, dict]]:
        """
        Generate backend-specific config based on the chosen system type (Agg or Disagg).
        Args:
            aiconfigurator_result: The result of the aiconfigurator.
            aiconfigurator_config: The config of the aiconfigurator.
        Returns:
            A dictionary containing the backend-specific configs. 'mode': {file_name: backend_config_yaml}
        """
        
        def _derive_backend(aiconfigurator_config: AIConfiguratorConfig) -> common.BackendName:
            """Derive the backend from the aiconfigurator config"""
            if aiconfigurator_config.agg_worker_config:
                return aiconfigurator_config.agg_worker_config.backend_name
            elif aiconfigurator_config.disagg_prefill_worker_config:
                return aiconfigurator_config.disagg_prefill_worker_config.backend_name
            else:
                raise ValueError("No backend found in the aiconfigurator config")

        def _derive_version(aiconfigurator_config: AIConfiguratorConfig) -> str:
            if aiconfigurator_config.agg_worker_config:
                return aiconfigurator_config.agg_worker_config.version
            elif aiconfigurator_config.disagg_prefill_worker_config:
                return aiconfigurator_config.disagg_prefill_worker_config.version
            else:
                raise ValueError("No version found in the aiconfigurator config")


        backend = _derive_backend(aiconfigurator_config)
        if generated_config_version:
            version = generated_config_version
        else:
            version = _derive_version(aiconfigurator_config)

        logger.info(f"Generating configs for deploying dynamo + {backend} with version {version}")
        config_generator = get_config_generator(backend)

        return config_generator(aiconfigurator_result, aiconfigurator_config, dynamo_config, version)


    def _plot_pareto_frontier(self, 
                              title: str, 
                              best_config_df: pd.DataFrame,
                              disagg_pareto_df: pd.DataFrame, 
                              agg_pareto_df: pd.DataFrame) -> str:
        """Plot Pareto frontier for Disagg and Agg"""
        plotext.plot_size(80, 30)
        plotext.theme("clear")
        if not disagg_pareto_df.empty:
            plotext.plot(
                disagg_pareto_df['tokens/s/user'],
                disagg_pareto_df['tokens/s/gpu'],
                label='Disagg',
                color=(144, 238, 144), # light green
                marker='d'
            )
        if not agg_pareto_df.empty:
            plotext.plot(
                agg_pareto_df['tokens/s/user'],
                agg_pareto_df['tokens/s/gpu'],
                label='Agg',
                color= (200, 200, 200), # gray
                marker='a'
            )
        
        if not best_config_df.empty:
            plotext.plot(
                best_config_df['tokens/s/user'],
                best_config_df['tokens/s/gpu'],
                label='Best',
                color=(255, 215, 0), # gold
                marker='X'
            )

        plotext.title(f"{title}: tokens/s/gpu vs tokens/s/user")
        plotext.xlabel("tokens/s/user")
        plotext.ylabel("tokens/s/gpu")
        plotext.grid(False)

        y_min = 0.0
        y_max = 0.0
        x_min = 0.0
        x_max = 0.0
        if not disagg_pareto_df.empty:
            y_max = max(disagg_pareto_df['tokens/s/gpu'].max(), y_max)
            x_max = max(disagg_pareto_df['tokens/s/user'].max(), x_max)
        if not agg_pareto_df.empty:
            y_max = max(agg_pareto_df['tokens/s/gpu'].max(), y_max)
            x_max = max(agg_pareto_df['tokens/s/user'].max(), x_max)
        y_max = y_max * 1.2
        y_max = ((y_max+49) // 50) * 50
        x_max = x_max * 1.1
        x_max = ((x_max+19) // 20) * 20
        x_max = min(x_max, 300)
        if y_max > 0.0 and x_max > 0.0:
            plotext.ylim(y_min, y_max)
            plotext.xlim(x_min, x_max)

        buf = plotext.build()
        plotext.clear_data()

        return buf
    
    def _plot_worker_setup_table(self, disagg_pareto: pd.DataFrame, agg_pareto: pd.DataFrame, total_gpus: int, tpot_target: float, top: int, is_moe: bool) -> str:
        """Plot worker setup table"""
        buf = []
        buf.append("    (p) stands for prefill, (d) stands for decode, bs stands for batch size, a replica stands for the smallest scalable unit xPyD of the disagg system")
        buf.append("    Some math: total gpus used = replicas * gpus/replica")
        buf.append("               gpus/replica = (p)gpus/worker * (p)workers + (d)gpus/worker * (d)workers; for Agg, gpus/replica = gpus/worker")
        buf.append("               gpus/worker = tp * pp * dp = etp * ep * pp for MoE models; tp * pp for dense models (underlined \033[4mnumbers\033[0m are the actual values in math)")

        # Filter and sort data
        if disagg_pareto is not None and not disagg_pareto.empty:
            disagg_pareto['tokens/s/gpu_cluster'] = disagg_pareto['tokens/s/gpu'] * (total_gpus // disagg_pareto['num_total_gpus']) \
                * disagg_pareto['num_total_gpus'] / total_gpus if total_gpus > 0 else 0
            disagg_top_configs = disagg_pareto[disagg_pareto['tpot'] <= tpot_target].sort_values(by='tokens/s/gpu_cluster', ascending=False).head(top).copy()
            # Calculate replicas and total GPUs used for each configuration
            disagg_top_configs['replicas'] = total_gpus // disagg_top_configs['num_total_gpus']
            disagg_top_configs['total_gpus_used'] = disagg_top_configs['num_total_gpus'] * disagg_top_configs['replicas']
            if not disagg_top_configs.empty:
                buf.append("\nDisagg Top Configurations: (Sorted by tokens/s/gpu)")
                table = PrettyTable()
                table.field_names = ["Rank", f"\033[1mtokens/s/gpu\033[0m", "tokens/s/user", "concurrency", "total_gpus(used)", "replicas", "gpus/replica", 
                                     "(p)workers", "(p)gpus/worker", "(p)parallel", "(p)bs",
                                     "(d)workers", "(d)gpus/worker", "(d)parallel", "(d)bs"]

                for i, row in enumerate(disagg_top_configs.to_dict('records')):
                    if is_moe:
                        p_parallel = f'tp\033[4m{row["(p)tp"]}\033[0mpp\033[4m{row["(p)pp"]}\033[0mdp\033[4m{row["(p)dp"]}\033[0metp{row["(p)moe_tp"]}ep{row["(p)moe_ep"]}'
                        d_parallel = f'tp\033[4m{row["(d)tp"]}\033[0mpp\033[4m{row["(d)pp"]}\033[0mdp\033[4m{row["(d)dp"]}\033[0metp{row["(d)moe_tp"]}ep{row["(d)moe_ep"]}'
                        p_gpus_worker = f'{row["(p)pp"]*row["(p)tp"]*row["(p)dp"]} (=\033[4m{row["(p)tp"]}\033[0mx\033[4m{row["(p)pp"]}\033[0mx\033[4m{row["(p)dp"]}\033[0m)'
                        d_gpus_worker = f'{row["(d)pp"]*row["(d)tp"]*row["(d)dp"]} (=\033[4m{row["(d)tp"]}\033[0mx\033[4m{row["(d)pp"]}\033[0mx\033[4m{row["(d)dp"]}\033[0m)'
                    else:
                        p_parallel = f'tp\033[4m{row["(p)tp"]}\033[0mpp\033[4m{row["(p)pp"]}\033[0m'
                        d_parallel = f'tp\033[4m{row["(d)tp"]}\033[0mpp\033[4m{row["(d)pp"]}\033[0m'
                        p_gpus_worker = f'{row["(p)pp"]*row["(p)tp"]} (=\033[4m{row["(p)tp"]}\033[0mx\033[4m{row["(p)pp"]}\033[0m)'
                        d_gpus_worker = f'{row["(d)pp"]*row["(d)tp"]} (=\033[4m{row["(d)tp"]}\033[0mx\033[4m{row["(d)pp"]}\033[0m)'
                    table.add_row([
                        i + 1, f"\033[1m{row['tokens/s/gpu_cluster']:.2f}\033[0m", f"{row['tokens/s/user']:.2f}", row['concurrency'],
                        f"{total_gpus} ({row['total_gpus_used']}={row['replicas']}x{row['num_total_gpus']})", row['replicas'],
                        f"{row['num_total_gpus']} (={row['(p)workers']}x{row['(p)pp']*row['(p)tp']*row['(p)dp']}+{row['(d)workers']}x{row['(d)pp']*row['(d)tp']*row['(d)dp']})",
                        row['(p)workers'], p_gpus_worker, p_parallel, row['(p)bs'],
                        row['(d)workers'], d_gpus_worker, d_parallel, row['(d)bs'],
                    ])
                buf.append(table.get_string())

        if agg_pareto is not None and not agg_pareto.empty:
            agg_pareto['tokens/s/gpu_cluster'] = agg_pareto['tokens/s/gpu'] * (total_gpus // agg_pareto['num_total_gpus']) \
                * agg_pareto['num_total_gpus'] / total_gpus if total_gpus > 0 else 0
            agg_top_configs = agg_pareto[agg_pareto['tpot'] <= tpot_target].sort_values(by='tokens/s/gpu_cluster', ascending=False).head(top).copy()
            # Calculate replicas and total GPUs used for each configuration
            agg_top_configs['replicas'] = total_gpus // agg_top_configs['num_total_gpus']
            agg_top_configs['total_gpus_used'] = agg_top_configs['num_total_gpus'] * agg_top_configs['replicas']
            if not agg_top_configs.empty:
                buf.append("Agg Top Configurations: (Sorted by tokens/s/gpu)")
                table = PrettyTable()
                table.field_names = ["Rank", f"\033[1mtokens/s/gpu\033[0m", "tokens/s/user", "concurrency", "total_gpus(used)", 
                                     "replicas", "gpus/replica", "gpus/worker", "parallel", "bs"]
                
                for i, row in enumerate(agg_top_configs.to_dict('records')):
                    if is_moe:
                        parallel = f'tp\033[4m{row["tp"]}\033[0mpp\033[4m{row["pp"]}\033[0mdp\033[4m{row["dp"]}\033[0metp{row["moe_tp"]}ep{row["moe_ep"]}'
                        gpus_worker = f'{row["pp"]*row["tp"]*row["dp"]} (=\033[4m{row["tp"]}\033[0mx\033[4m{row["pp"]}\033[0mx\033[4m{row["dp"]}\033[0m)'
                    else:
                        parallel = f'tp\033[4m{row["tp"]}\033[0mpp\033[4m{row["pp"]}\033[0m'
                        gpus_worker = f'{row["pp"]*row["tp"]} (=\033[4m{row["tp"]}\033[0mx\033[4m{row["pp"]}\033[0m)'
                    table.add_row([
                        i + 1, f"\033[1m{row['tokens/s/gpu_cluster']:.2f}\033[0m", f"{row['tokens/s/user']:.2f}",
                        row['concurrency'], f"{total_gpus} ({row['total_gpus_used']}={row['replicas']}x{row['num_total_gpus']})",
                        row['replicas'], row['num_total_gpus'],
                        gpus_worker, parallel, row['bs']
                    ])
                buf.append(table.get_string())

        return "\n".join(buf)
    
    def log_final_summary(self, aiconfigurator_result: AIConfiguratorResult, aiconfigurator_config: AIConfiguratorConfig):
        """Log final summary of configuration results"""
        logger.info(f"Configuration completed for {aiconfigurator_config.model_name}")
        
        # Consolidate and format results into a summary box for clear presentation
        summary_box = []
        summary_box.append("*" * 80)
        summary_box.append("*{:^78}*".format(" Dynamo aiconfigurator Final Results "))
        summary_box.append("*" * 80)

        summary_box.append("  " + "-" * 76)
        summary_box.append("  Input Configuration & SLA Target:")
        summary_box.append(f"    Model: {aiconfigurator_config.model_name} (is_moe: {aiconfigurator_config.is_moe})")
        summary_box.append(f"    Total GPUs: {aiconfigurator_config.total_gpus}")
        summary_box.append(f"    I/O Length (tokens): Input={aiconfigurator_config.isl}, Output={aiconfigurator_config.osl}")
        summary_box.append(f"    SLA Target: TTFT <= {aiconfigurator_config.ttft}ms, TPOT <= {aiconfigurator_config.tpot}ms")
        summary_box.append("  " + "-" * 76)


        # ============================= overall summary
        chosen_system_type = aiconfigurator_result.chosen_system_type
        if chosen_system_type != "none":
            benefit_ratio = aiconfigurator_result.benefit_ratio
            if chosen_system_type == "disagg":
                summary_box.append(f"  Overall best system chosen: \033[1m{chosen_system_type} at {aiconfigurator_result.disagg_actual_best_throughput:.2f} tokens/s/gpu ({benefit_ratio:.2f}x better)\033[0m")
            else:
                if benefit_ratio == 0:
                    benefit_ratio = float('inf')
                else:
                    benefit_ratio = 1.0 / benefit_ratio
                summary_box.append(f"  Overall best system chosen: \033[1m{chosen_system_type} at {aiconfigurator_result.agg_actual_best_throughput:.2f} tokens/s/gpu ({benefit_ratio:.2f}x better)\033[0m")
        else:
            summary_box.append("  No best system could be determined based on the criteria.")

        agg_best_str = f"    - Agg Actual Best: {aiconfigurator_result.agg_actual_best_throughput:.2f} tokens/s/gpu"
        if not aiconfigurator_result.agg_best_config.empty:
            best_conf_details = aiconfigurator_result.agg_best_config.iloc[0]
            agg_best_str += f"  {best_conf_details['tokens/s/user']:.2f} tokens/s/user | TTFT: {best_conf_details['ttft']:.2f}ms TPOT: {best_conf_details['tpot']:.2f}ms"
        summary_box.append(agg_best_str)

        disagg_best_str = f"    - Disagg Actual Best: {aiconfigurator_result.disagg_actual_best_throughput:.2f} tokens/s/gpu"
        if not aiconfigurator_result.disagg_best_config.empty:
            best_conf_details = aiconfigurator_result.disagg_best_config.iloc[0]
            disagg_best_str += f"  {best_conf_details['tokens/s/user']:.2f} tokens/s/user | TTFT: {best_conf_details['ttft']:.2f}ms TPOT: {best_conf_details['tpot']:.2f}ms"
        summary_box.append(disagg_best_str)
        summary_box.append("  " + "-" * 76)

        # ============================= pareto frontier
        summary_box.append("  Pareto Frontier:")
        pareto_plot_buf = self._plot_pareto_frontier(f"{aiconfigurator_config.model_name} Pareto Frontier", 
                                                     aiconfigurator_result.disagg_best_config if aiconfigurator_result.chosen_system_type == "disagg" else aiconfigurator_result.agg_best_config,
                                                     aiconfigurator_result.disagg_pareto, aiconfigurator_result.agg_pareto)
        summary_box.append(pareto_plot_buf)
        summary_box.append("  " + "-" * 76)

        # ============================= deployment details

        # Note: bs here is config reference. it means per worker batch size if attention dp is not used, otherwise, it means per attention dp rank batch size
        summary_box.append("  Worker Setup:")
        summary_box.append(f"    Model: {aiconfigurator_config.model_name} (is_moe: {aiconfigurator_config.is_moe})")

        if aiconfigurator_config.disagg_replica_config:
            prefill_sys = aiconfigurator_config.disagg_prefill_worker_config
            decode_sys = aiconfigurator_config.disagg_decode_worker_config
            summary_box.append(f"    Disagg Prefill: {prefill_sys.system_name} ({prefill_sys.backend_name})")
            summary_box.append(f"    Disagg Decode:  {decode_sys.system_name} ({decode_sys.backend_name})")
            quant_str = (f"GEMM: {prefill_sys.gemm_quant_mode.name}, KVCache: {prefill_sys.kvcache_quant_mode.name}, "
                    f"FMHA: {prefill_sys.fmha_quant_mode.name}")
            if aiconfigurator_config.is_moe:
                quant_str += f", MoE: {prefill_sys.moe_quant_mode.name}"
            summary_box.append(f"    Prefill Quantization: {quant_str}")
            quant_str = (f"GEMM: {decode_sys.gemm_quant_mode.name}, KVCache: {decode_sys.kvcache_quant_mode.name}, "
                    f"FMHA: {decode_sys.fmha_quant_mode.name}")
            if aiconfigurator_config.is_moe:
                quant_str += f", MoE: {decode_sys.moe_quant_mode.name}"
            summary_box.append(f"    Decode Quantization: {quant_str}")

        if aiconfigurator_config.agg_worker_config:
            agg_sys = aiconfigurator_config.agg_worker_config
            summary_box.append(f"    Agg: {agg_sys.system_name} ({agg_sys.backend_name})")
            quant_str = (f"GEMM: {agg_sys.gemm_quant_mode.name}, KVCache: {agg_sys.kvcache_quant_mode.name}, "
                    f"FMHA: {agg_sys.fmha_quant_mode.name}")
            if aiconfigurator_config.is_moe:
                quant_str += f", MoE: {agg_sys.moe_quant_mode.name}"
            summary_box.append(f"    Quantization: {quant_str}")

        summary_box.append("  " + "-" * 76)

        summary_box.append("  Deployment Details:")
        table_buf = self._plot_worker_setup_table(aiconfigurator_result.disagg_pareto, aiconfigurator_result.agg_pareto, aiconfigurator_config.total_gpus, aiconfigurator_config.tpot, 5, aiconfigurator_config.is_moe)
        summary_box.append(table_buf)

        summary_box.append("*" * 80)
        logger.info("\n" + "\n".join(summary_box))
    
    def save_aiconfigurator_result(self, aiconfigurator_result: AIConfiguratorResult, aiconfigurator_config: AIConfiguratorConfig, backend_configs: dict, dir_path: str) -> None:
        """
        Save the aiconfigurator result to a dir_path.
        Args:
            aiconfigurator_result: The aiconfigurator result to save.
            aiconfigurator_config: The config of the aiconfigurator.
            dir_path: The directory path to save the aiconfigurator result.
        Returns:
            None
        """
        result_prefix = f"{aiconfigurator_config.model_name}_isl{aiconfigurator_config.isl}_osl{aiconfigurator_config.osl}_ttft{int(aiconfigurator_config.ttft)}_tpot{int(aiconfigurator_config.tpot)}"
        result_dir_path = os.path.join(dir_path, f'{result_prefix}_{random.randint(0,1000000)}')
        logger.info('saving results to ' + result_dir_path)
        try:
            os.makedirs(result_dir_path, exist_ok=True)
            logger.debug(f"Saving aiconfigurator config to {os.path.join(result_dir_path, 'aiconfigurator_config.yaml')}")
            with open(os.path.join(result_dir_path, 'aiconfigurator_config.yaml'), 'w') as f:
                yaml.safe_dump(aiconfigurator_config.aiconfigurator_yaml_data, f, sort_keys=False, allow_unicode=True)

            logger.debug(f"Saving aiconfigurator result to {os.path.join(result_dir_path, 'aiconfigurator_result.yaml')}")
            with open(os.path.join(result_dir_path, 'aiconfigurator_result.json'), 'w') as f:
                json.dump(aiconfigurator_result.to_dict(), f, indent=4)

            logger.debug(f"Saving pareto frontiers to {os.path.join(result_dir_path, 'pareto_frontier.png')}")
            fig, ax = plt.subplots(1,1, figsize=(8,5))
            plt.title(f"{aiconfigurator_config.model_name} tokens/s/gpu vs tokens/s/user")
            if not aiconfigurator_result.agg_pareto.empty:
                pareto_analysis.draw_pareto(aiconfigurator_result.agg_pareto, 'tokens/s/user', 'tokens/s/gpu', ax, 'blue', 'Agg')
            if not aiconfigurator_result.disagg_pareto.empty:
                pareto_analysis.draw_pareto(aiconfigurator_result.disagg_pareto, 'tokens/s/user', 'tokens/s/gpu', ax, 'red', 'Disagg')
            plt.savefig(os.path.join(result_dir_path, 'pareto_frontier.png'))
            plt.close()

            logger.debug(f"Saving agg pareto frontier raw data to {os.path.join(result_dir_path, 'agg_pareto.csv')}")
            if not aiconfigurator_result.agg_pareto.empty:
                aiconfigurator_result.agg_pareto.to_csv(os.path.join(result_dir_path, 'agg_pareto.csv'), index=False)

            logger.debug(f"Saving disagg pareto frontier raw data to {os.path.join(result_dir_path, 'disagg_pareto.csv')}")
            if not aiconfigurator_result.disagg_pareto.empty:
                aiconfigurator_result.disagg_pareto.to_csv(os.path.join(result_dir_path, 'disagg_pareto.csv'), index=False)
            
            logger.debug(f"Saving backend configs to {os.path.join(result_dir_path, 'backend_configs')}")
            os.makedirs(os.path.join(result_dir_path, 'backend_configs'), exist_ok=True)
            backend_root = os.path.join(result_dir_path, "backend_configs")
            for mode, mode_backend_configs in backend_configs.items():
                logger.debug(f"Saving {mode} backend configs to {os.path.join(result_dir_path, 'backend_configs', mode)}")
                os.makedirs(os.path.join(backend_root, mode), exist_ok=True)
                mode_dir = os.path.join(backend_root, mode)
                for file_name, mode_backend_config in mode_backend_configs.items():
                    out_path = os.path.join(mode_dir, file_name)
                    _dump_backend_file(out_path, mode_backend_config)

        except Exception as e:
            logger.error(f"Failed to save aiconfigurator result to {result_dir_path}: {e}")

def load_config_from_yaml(model_name: str,
                            system_name: str,
                            backend_name: str,
                            version: str,
                            total_gpus: int,
                            isl: int,
                            osl: int,
                            ttft: float,
                            tpot: float,
                            yaml_path: Optional[str] = None) -> AIConfiguratorConfig:
    """
    Loads aiconfigurator config from a YAML file.
    Args:
        model_name: The name of the model.
        system_name: The name of the system.
        backend_name: The name of the backend.
        version: The version of the backend.
        total_gpus: The total number of GPUs.
        isl: The input sequence length.
        osl: The output sequence length.
        ttft: The time to first token.
        tpot: The time per output token.
    Returns:
        AIConfiguratorConfig: The loaded aiconfigurator config.
    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the config file is invalid.
        ValueError: If the config file is invalid.
    """
    def _get_worker_config(
            system_config: dict,
            worker_key: str) -> Optional[WorkerConfig]:
        def _translate_to_enum(quant_config: dict) -> dict:
            return {
                'gemm_quant_mode': common.GEMMQuantMode[quant_config['gemm_quant_mode']],
                'moe_quant_mode': common.MoEQuantMode[quant_config['moe_quant_mode']],
                'kvcache_quant_mode': common.KVCacheQuantMode[quant_config['kvcache_quant_mode']],
                'fmha_quant_mode': common.FMHAQuantMode[quant_config['fmha_quant_mode']],
                'comm_quant_mode': common.CommQuantMode[quant_config['comm_quant_mode']]
            }
        if system_config is not None and worker_key in system_config:
            return WorkerConfig(**system_config[worker_key]['system_config'],
                                **_translate_to_enum(system_config[worker_key]['quant_config']), 
                                **system_config[worker_key]['parallel_config'])

        return None

    def _get_replica_config(system_config, replica_key: str) -> Optional[DisaggReplicaConfig]:
        if system_config is not None and replica_key in system_config:
            return DisaggReplicaConfig(**system_config[replica_key])
        return None
    
    if yaml_path is None: # will use default template if yaml_path is not provided
        logger.info(f"No yaml path provided, using default template for model {model_name} using backend {backend_name}")
        if not check_is_moe(model_name):
            yaml_path = os.path.join(os.path.dirname(__file__), 'templates', backend_name, 'dense_default.yaml')
        elif get_model_family(model_name) == 'DEEPSEEK':
            yaml_path = os.path.join(os.path.dirname(__file__), 'templates', backend_name, 'deepseek_default.yaml')
        else:
            yaml_path = os.path.join(os.path.dirname(__file__), 'templates', backend_name, 'moe_default.yaml')

    assert os.path.exists(yaml_path), f"Config yaml file {yaml_path} for backend {backend_name} does not exist"
    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    config = AIConfiguratorConfig(
        aiconfigurator_yaml_data=config_data,
        model_name=model_name,
        is_moe=check_is_moe(model_name),
        nextn=config_data.get('nextn', 0),
        nextn_accept_rates=config_data.get('nextn_accept_rates', None),
        total_gpus=total_gpus,
        isl=config_data.get('isl', None),
        osl=config_data.get('osl', None),
        ttft=config_data.get('ttft', None),
        tpot=config_data.get('tpot', None),
        agg_worker_config=_get_worker_config(config_data.get('agg_config', None), 'agg_worker_config'),
        disagg_replica_config=_get_replica_config(config_data.get('disagg_config', None), 'replica_config'),
        disagg_prefill_worker_config=_get_worker_config(config_data.get('disagg_config', None), 'prefill_worker_config'),
        disagg_decode_worker_config=_get_worker_config(config_data.get('disagg_config', None), 'decode_worker_config'),
        prefill_correction_scale=config_data.get('prefill_correction_scale', 1.0),
        decode_correction_scale=config_data.get('decode_correction_scale', 1.0),
        prefill_max_batch_size=config_data.get('prefill_max_batch_size', 1),
        decode_max_batch_size=config_data.get('decode_max_batch_size', 512)
    )

    # update config based on input
    if isl is not None:
        config.isl = isl
    if osl is not None:
        config.osl = osl
    if ttft is not None:
        config.ttft = ttft
    if tpot is not None:
        config.tpot = tpot
    
    # validate args
    for item in [config.isl, config.osl, config.ttft, config.tpot]:
        assert item is not None, f"isl, osl, ttft, tpot must be set either in template or as input"
        assert item > 0, f"isl, osl, ttft, tpot must be greater than 0"
    assert config.total_gpus >= 2, f"Total GPUs {config.total_gpus} is less than 2, disagg deployment requires at least 2 gpus"

    for worker_config in [config.agg_worker_config, config.disagg_prefill_worker_config, config.disagg_decode_worker_config]:
        if worker_config is not None:
            # filter out those num_gpu_per_worker that is larger than total_gpus
            if worker_config.num_gpu_per_worker is not None:
                if config.total_gpus < max(worker_config.num_gpu_per_worker):
                    logger.debug(f"Total GPUs {config.total_gpus} is less than the max gpus per worker {max(worker_config.num_gpu_per_worker)}, filter out those num_gpu_per_worker that is larger than total_gpus")
                    worker_config.num_gpu_per_worker = [num_gpu for num_gpu in worker_config.num_gpu_per_worker if num_gpu <= config.total_gpus]
            # update system name, backend name, version based on args
            worker_config.system_name = system_name
            worker_config.backend_name = backend_name
            worker_config.version = version
    if config.disagg_replica_config is not None:
        if config.disagg_replica_config.max_gpu_per_replica > 0:
            logger.debug(f"Total GPUs {config.total_gpus} and max gpus per replica limit {config.disagg_replica_config.max_gpu_per_replica}")
            config.disagg_replica_config.max_gpu_per_replica = min(config.disagg_replica_config.max_gpu_per_replica, config.total_gpus)
        elif config.disagg_replica_config.num_gpu_per_replica is not None:
            logger.debug(f"Total GPUs {config.total_gpus} is less than the max gpus per replica {max(config.disagg_replica_config.num_gpu_per_replica)}, filter out those num_gpu_per_replica that is larger than total_gpus")
            config.disagg_replica_config.num_gpu_per_replica = [num_gpu for num_gpu in config.disagg_replica_config.num_gpu_per_replica if num_gpu <= config.total_gpus]
        else:
            logger.debug(f"Setting max_gpu_per_replica to {config.total_gpus} as no max_gpu_per_replica is set")
            config.disagg_replica_config.max_gpu_per_replica = total_gpus
    
    return config
    

def configure_parser(parser):
    """
    Configures the argument parser for the CLI.
    """
    parser.add_argument("--model", choices=common.SupportedModels.keys(), type=str, required=True, help="Model name") 
    parser.add_argument("--system", choices=['h100_sxm', 'h200_sxm'], type=str, required=True, help="System name")    
    parser.add_argument("--total_gpus", type=int, required=True, help="Total GPUs, no less than 2 as disagg deployment requires at least 2 gpus")
    # optional args, dedault according to templates
    parser.add_argument("--backend", choices=[backend.value for backend in common.BackendName], type=str, default=common.BackendName.trtllm.value, help="Backend name, suport trtllm for now")
    parser.add_argument("--version", type=str, default='0.20.0', help="Version, 0.20.0,1.0.0rc3 for trtllm")
    parser.add_argument("--isl", type=int, help="Input sequence length")
    parser.add_argument("--osl", type=int, help="Output sequence length")
    parser.add_argument("--ttft", type=float, help="Time to first token (ms)")
    parser.add_argument("--tpot", type=float, help="Time per output token (ms)")
    parser.add_argument("--yaml_path", type=str, default=None, help="Path to the aiconfigurator yaml file. Default is None, which means using default template.")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to the directory to save the aiconfigurator result. Default is None, which means not saving the aiconfigurator result.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

    chosen_backend = add_dynamo_cli(parser, common.BackendName.trtllm.value)

    parser.epilog = (
        "\nNOTE:\n"
        "  • The extra dynamo_config.* parameters shown here are for "
        f"backend '{chosen_backend}'.\n"
        "  • To see configuration for another backend run:\n"
        "        aiconfigurator cli --backend <backend_name> --help\n"
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter

def main(args):
    """
    Main function for the CLI.
    """
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, 
                        format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    
    logger.info(f"Loading Dynamo AIConfigurator version: {__version__}")

    # Create aiconfigurator and its config
    aiconfigurator = AIConfigurator()
    try:
        aiconfigurator_config = load_config_from_yaml(args.model,
                                               args.system,
                                               args.backend,
                                               args.version,
                                               args.total_gpus,
                                               args.isl,
                                               args.osl, 
                                               args.ttft, 
                                               args.tpot,
                                               args.yaml_path)
    except Exception as e:
        logger.error(f"Error init aiconfigurator config from: {e}")
        exit(1)
        
    dynamo_config = build_dynamo_config(args)

    try:
        start_time = time.time()
        # Run configuration
        aiconfigurator_result = aiconfigurator.run(aiconfigurator_config)
        aiconfigurator.log_final_summary(aiconfigurator_result, aiconfigurator_config)

        if args.save_dir is not None:
            backend_configs = aiconfigurator.generate_backend_config(aiconfigurator_result, aiconfigurator_config, dynamo_config, args.generated_config_version)
            aiconfigurator.save_aiconfigurator_result(aiconfigurator_result, aiconfigurator_config, backend_configs, args.save_dir)
    
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        logger.exception("Full traceback")
        exit(1)
    finally:
        end_time = time.time()
        logger.info(f"Configuration completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamo AIConfigurator for Disaggregated Serving Deployment")
    configure_parser(parser)
    args = parser.parse_args()
    main(args)
    