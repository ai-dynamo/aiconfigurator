# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import numpy as np
from scipy import interpolate
import math
from aiconfigurator.sdk import common
import os
from collections import defaultdict
import yaml
import logging
from typing import Optional, Dict, Any, List, Tuple
import importlib.resources as pkg_resources
import csv

databases_cache = defaultdict(lambda: defaultdict(lambda: defaultdict()))
logger = logging.getLogger(__name__)

def get_system_config_path():
    """
    Get the system config path
    """
    return pkg_resources.files('aiconfigurator') / 'systems'


def get_supported_databases(systems_dir: str = get_system_config_path()) -> Dict[str, Dict[str, List[str]]]:
    """
    Get all supported databases for all systems, backends and versions without loading them.
    """
    supported_dict = defaultdict(lambda: defaultdict(list))
    if not os.path.isdir(systems_dir):
        logger.warning(f"Systems directory not found: {systems_dir}")
        return supported_dict

    system_yamls = [f for f in os.listdir(systems_dir) if f.endswith('.yaml') and os.path.isfile(os.path.join(systems_dir, f))]
    for system_yaml in system_yamls:
        system = system_yaml.split('.')[0]
        try:
            with open(os.path.join(systems_dir, system_yaml), 'r') as f:
                system_spec = yaml.safe_load(f)
            
            data_dir = os.path.join(systems_dir, system_spec.get('data_dir', ''))
            if not os.path.isdir(data_dir):
                continue
                
            for backend in common.BackendName:
                backend_path = os.path.join(data_dir, backend.value)
                if not os.path.isdir(backend_path):
                    continue
                
                versions = sorted([v for v in os.listdir(backend_path) if not v.startswith('.') and os.path.isdir(os.path.join(backend_path, v))])
                if versions:
                    supported_dict[system][backend.value] = versions
        except Exception as e:
            logger.warning(f"Could not process system config {system_yaml}: {e}")
            
    return supported_dict


def get_latest_database_version(system : str,
                                backend : str, 
                                ) -> Optional[str]:
    """
    Get the latest database version for a given system and backend
    """
    import re
    
    supported_databases = get_supported_databases()
    try:
        database_versions = supported_databases[system][backend]
    except KeyError:
        logger.error(f"database not found for {system=}, {backend=}")
        return None
    
    def parse_version(version_str):
        """Parse version string into comparable tuple"""
        # Handle different version formats
        version_str = version_str.lower()
        
        # Extract numeric version pattern (e.g., "1.2.3" from "v1.2.3rc4" or "1.2.3_suffix")
        version_match = re.search(r'(\d+)\.(\d+)\.(\d+)', version_str)
        if version_match:
            major, minor, patch = map(int, version_match.groups())
            version_parts = [major, minor, patch]
            
            # Handle release candidates (lower priority than stable releases)
            if 'rc' in version_str:
                rc_match = re.search(r'rc(\d+)', version_str)
                if rc_match:
                    rc_num = int(rc_match.group(1))
                    version_parts.append(0)  # Stable release indicator
                    version_parts.append(rc_num)  # RC number
                else:
                    version_parts.append(0)  # Stable release indicator
                    version_parts.append(0)   # No RC number
            else:
                version_parts.append(1)  # Stable release (higher priority than RC)
                version_parts.append(0)  # No RC number
                
            return tuple(version_parts)
        
        # Try to extract version from other patterns (e.g., "v0.20_fix0719")
        version_match = re.search(r'v?(\d+)\.(\d+)', version_str)
        if version_match:
            major, minor = map(int, version_match.groups())
            version_parts = [major, minor, 0, 1, 0]  # Assume stable release
            return tuple(version_parts)
        
        # For completely non-standard versions, try to extract any numbers
        numbers = re.findall(r'\d+', version_str)
        if numbers:
            # Use first few numbers found, pad with zeros
            version_parts = [int(x) for x in numbers[:3]]
            while len(version_parts) < 3:
                version_parts.append(0)
            version_parts.extend([0, 0])  # Add RC indicators
            return tuple(version_parts)
        
        # If no numbers found, return a very low priority tuple
        return (0, 0, 0, -1, 0)
    
    # Convert version strings to comparable tuples
    versions_ids = []
    for version in database_versions:
        try:
            version_parts = parse_version(version)
            versions_ids.append((version_parts, version))
            logger.debug(f"Parsed version {version} as {version_parts}")
        except Exception as e:
            logger.warning(f"Failed to parse version {version}: {e}")
            continue
    
    if not versions_ids:
        logger.error(f"no valid versions parsed for {system=}, {backend=}")
        return None
    
    # Find the latest version by comparing version tuples.
    # The tuple format (major, minor, patch, is_stable, rc_num) ensures
    # correct sorting across stable and RC releases.
    latest_version = max(versions_ids, key=lambda x: x[0])
    
    logger.debug(f"Latest version for {system}/{backend}: {latest_version[1]} (parsed as {latest_version[0]})")
    return latest_version[1]
    

def get_database(system : str,
                 backend : str, 
                 version : str, 
                 systems_dir : str = get_system_config_path()) -> Optional[PerfDatabase]:
    """
    Get the database for a given system, backend and version

    Args:
        system (str): the system name
        backend (str): the backend name
        version (str): the version name
        systems_dir (str): the systems directory

    Returns:
        PerfDatabase: the database for the given system, backend and version
    """
    try:
        database = databases_cache[system][backend][version]
    except KeyError:
        logger.info(f"loading {system=}, {backend=}, {version=}")
        if os.path.exists(os.path.join(systems_dir, system+'.yaml')):
            system_spec = yaml.load(open(os.path.join(systems_dir, system+'.yaml')), Loader=yaml.SafeLoader)
            data_path = os.path.join(systems_dir, system_spec['data_dir'], backend, version)
            if os.path.exists(data_path):
                try:
                    database = PerfDatabase(system, backend, version, systems_dir)
                    databases_cache[system][backend][version] = database
                except Exception as e:
                    logger.error(f"failed to load {system=}, {backend=}, {version=}: {e}")
                    database = None
            else:
                logger.error(f"data path {data_path} not found")
                database = None
        else:
            logger.error(f"system yaml {os.path.join(systems_dir, system+'.yaml')} not found")
            database = None

    return database

def get_all_databases(systems_dir : str = get_system_config_path()) -> Dict[str, Dict[str, Dict[str, PerfDatabase]]]:
    """
    Get all the databases for all the systems, backends and versions
    """
    database_dict = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    system_yamls = [system_yaml for system_yaml in os.listdir(systems_dir) if system_yaml.endswith('.yaml')]
    for system_yaml in system_yamls:
        system = system_yaml.split('.')[0]
        system_spec = yaml.load(open(os.path.join(systems_dir, system_yaml)), Loader=yaml.SafeLoader)
        data_dir = os.path.join(systems_dir, system_spec['data_dir'])
        if not os.path.exists(data_dir):
            continue
        for backend in common.BackendName:
            if not os.path.exists(os.path.join(data_dir, backend.value)):
                continue
            for version in os.listdir(os.path.join(data_dir, backend.value)):
                if version.startswith('.'):
                    continue
                database = get_database(system, backend.value, version, systems_dir)
                if database is not None:
                    database_dict[system][backend.value][version] = database

    return database_dict

# by default float16
def load_custom_allreduce_data(custom_allreduce_file):
    """
    Load the custom allreduce data for trtllm
    """
    if not os.path.exists(custom_allreduce_file):
        logger.warning(f"Custom allreduce data file {custom_allreduce_file} not found.")
        return None
    custom_allreduce_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict())))

    with open(custom_allreduce_file, mode='r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    for row in rows:
        dtype, tp_size, message_size, latency = \
            row['allreduce_dtype'], row['num_gpus'], row['message_size'], row['latency']
        allreduce_strategy = 'AUTO'
        message_size  = int(message_size)
        latency = float(latency)
        tp_size = int(tp_size)
        dtype = common.CommQuantMode.half # TODO

        try:
            latency = custom_allreduce_data[dtype][tp_size][allreduce_strategy][message_size]
            logger.debug('value conflict in custom allreduce data: {} {} {} {} {}'.format(dtype, tp_size, allreduce_strategy, message_size, latency))
        except KeyError:
            custom_allreduce_data[dtype][tp_size][allreduce_strategy][message_size] = latency

    return custom_allreduce_data

def load_nccl_data(nccl_file):
    """
    Load the nccl data
    """
    if not os.path.exists(nccl_file):
        logger.warning(f"NCCL data file {nccl_file} not found.")
        return None
    nccl_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict())))

    with open(nccl_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    
    for row in rows:
        dtype, num_gpus, message_size, op_name, latency = \
            row['nccl_dtype'], row['num_gpus'], row['message_size'], row['op_name'], row['latency']
        message_size  = int(message_size)
        latency = float(latency)
        num_gpus = int(num_gpus)

        dtype = common.CommQuantMode[dtype]
        try:
            latency = nccl_data[dtype][op_name][num_gpus][message_size]
            logger.debug('value conflict in nccl data: {} {} {} {} {}'.format(dtype, op_name, num_gpus, message_size,latency))            
        except KeyError:
            nccl_data[dtype][op_name][num_gpus][message_size] = latency

    return nccl_data

def load_gemm_data(gemm_file):
    """
    Load the gemm data
    """
    if not os.path.exists(gemm_file):
        logger.warning(f"GEMM data file {gemm_file} not found.")
        return None
    gemm_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict())))

    with open(gemm_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    for row in rows:
        quant_mode, m, n, k, latency = row['gemm_dtype'], row['m'], row['n'], row['k'], row['latency']
        m = int(m)
        n = int(n)
        k = int(k)
        latency = float(latency)

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            latency = gemm_data[quant_mode][m][n][k]
            logger.debug('value conflict in gemm data: {} {} {} {} {}'.format(quant_mode, m, n, k, latency))
        except KeyError:
            gemm_data[quant_mode][m][n][k] = latency
    
    return gemm_data

def load_moe_data(moe_file):
    """
    Load the moe data
    """
    if not os.path.exists(moe_file):
        logger.warning(f"MOE data file {moe_file} not found.")
        return None
    
    moe_default_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))))))
    moe_low_latency_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))))))

    with open(moe_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    for row in rows:
        quant_mode, num_tokens, hidden_size, inter_size, topk, num_experts, moe_tp_size, moe_ep_size, workload_distribution,latency = \
            row['moe_dtype'], row['num_tokens'], row['hidden_size'], row['inter_size'], row['topk'], row['num_experts'], row['moe_tp_size'], row['moe_ep_size'], row['distribution'], row['latency']
        kernel_source = row['kernel_source'] # moe_torch_flow, moe_torch_flow_min_latency, moe_torch_flow
        num_tokens= int(num_tokens)
        hidden_size = int(hidden_size)
        inter_size = int(inter_size)
        topk = int(topk)
        num_experts = int(num_experts)
        moe_tp_size = int(moe_tp_size)
        moe_ep_size = int(moe_ep_size)
        latency = float(latency)

        quant_mode = common.MoEQuantMode[quant_mode]

        moe_data = moe_low_latency_data if 'moe_torch_flow_min_latency' == kernel_source else moe_default_data

        try:
            latency = moe_data[quant_mode][workload_distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size][num_tokens]
            logger.debug('value conflict in moe data: {} {} {} {} {} {} {} {} {} {}'.format(workload_distribution, quant_mode, topk, num_experts, hidden_size, inter_size, moe_tp_size, moe_ep_size,num_tokens,latency))            
        except KeyError:
            moe_data[quant_mode][workload_distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size][num_tokens] = latency
        
    return moe_default_data, moe_low_latency_data

def load_sglang_mlp_data(mlp_file):
    """
    Load the SGLang MLP data from context_mlp_perf.txt and generation_mlp_perf.txt
    """
    data_dir = os.path.dirname(mlp_file)
    prefill_mlp_file = os.path.join(data_dir, common.PerfDataFilename.context_mlp.value)
    generation_mlp_file = os.path.join(data_dir, common.PerfDataFilename.generation_mlp.value)
    
    context_mlp_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict())))
    generation_mlp_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict())))
    
    if os.path.exists(prefill_mlp_file):
        with open(prefill_mlp_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                quant_type, num_token, hidden_size, intermediate_size, avg_ms = \
                    row['quant_type'], row['num_token'], row['hidden_size'], row['intermediate_size'], row['avg_ms']
                
                num_token = int(num_token)
                hidden_size = int(hidden_size)
                intermediate_size = int(intermediate_size)
                avg_ms = float(avg_ms)
                quant_mode = common.MoEQuantMode[quant_type]
                
                try:
                    latency = context_mlp_data[quant_mode][hidden_size][intermediate_size][num_token]
                    logger.debug('value conflict in prefill mlp data: {} {} {} {} {}'.format(quant_mode, hidden_size, intermediate_size, num_token, latency))
                except KeyError:
                    context_mlp_data[quant_mode][hidden_size][intermediate_size][num_token] = avg_ms
    
    if os.path.exists(generation_mlp_file):
        with open(generation_mlp_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                quant_type, num_token, hidden_size, intermediate_size, avg_ms = \
                    row['quant_type'], row['num_token'], row['hidden_size'], row['intermediate_size'], row['avg_ms']
                
                num_token = int(num_token)
                hidden_size = int(hidden_size)
                intermediate_size = int(intermediate_size)
                avg_ms = float(avg_ms)
                quant_mode = common.MoEQuantMode[quant_type]

                try:
                    latency = generation_mlp_data[quant_mode][hidden_size][intermediate_size][num_token]
                    logger.debug('value conflict in generation mlp data: {} {} {} {} {}'.format(quant_mode, hidden_size, intermediate_size, num_token, latency))
                except KeyError:
                    generation_mlp_data[quant_mode][hidden_size][intermediate_size][num_token] = avg_ms
    
    return context_mlp_data, generation_mlp_data

def load_sglang_moe_data(moe_file):
    """
    Load the SGLang MoE data from context_moe_pref.txt and generation_moe_pref.txt
    """
    if not os.path.exists(moe_file):
        logger.warning(f"MoE data file {moe_file} not found.")
        return None
        
    data_dir = os.path.dirname(moe_file)
    context_moe_file = os.path.join(data_dir, common.PerfDataFilename.context_moe.value)
    generation_moe_file = os.path.join(data_dir, common.PerfDataFilename.generation_moe.value)
    
    moe_default_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))))))
    moe_low_latency_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))))))
    
    # Load context MoE data (normal mode)
    if os.path.exists(context_moe_file):
        logger.info(f"Loading context MoE data from: {context_moe_file}")
        with open(context_moe_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse the new CSV format with num_tokens instead of batch_size and input_len
                quant_mode = row['moe_dtype']
                num_tokens = int(row['num_tokens'])
                hidden_size = int(row['hidden_size'])
                inter_size = int(row['inter_size'])
                topk = int(row['topk'])
                num_experts = int(row['num_experts'])
                moe_tp_size = int(row['moe_tp_size'])
                moe_ep_size = int(row['moe_ep_size'])
                distribution = row['distribution']
                latency = float(row['latency'])
                quant_mode = common.MoEQuantMode[quant_mode]
                
                # Store the data, overwriting any previous entry with the same key
                moe_default_data[quant_mode][distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size][num_tokens] = latency
                logger.debug(f"Loaded context MoE data: {quant_mode}, {distribution}, {topk}, {num_experts}, {hidden_size}, {inter_size}, {moe_tp_size}, {moe_ep_size}, {num_tokens} -> {latency}")
    else:
        logger.warning(f"Context MoE file not found: {context_moe_file}")
    
    # Load generation MoE data (low latency mode)
    if os.path.exists(generation_moe_file):
        logger.info(f"Loading generation MoE data from: {generation_moe_file}")
        with open(generation_moe_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse the new CSV format with num_tokens instead of batch_size and input_len
                quant_mode = row['moe_dtype']
                num_tokens = int(row['num_tokens'])
                hidden_size = int(row['hidden_size'])
                inter_size = int(row['inter_size'])
                topk = int(row['topk'])
                num_experts = int(row['num_experts'])
                moe_tp_size = int(row['moe_tp_size'])
                moe_ep_size = int(row['moe_ep_size'])
                distribution = row['distribution']
                latency = float(row['latency'])
                quant_mode = common.MoEQuantMode[quant_mode]
                
                # Store the data, overwriting any previous entry with the same key
                moe_low_latency_data[quant_mode][distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size][num_tokens] = latency
                logger.debug(f"Loaded generation MoE data: {quant_mode}, {distribution}, {topk}, {num_experts}, {hidden_size}, {inter_size}, {moe_tp_size}, {moe_ep_size}, {num_tokens} -> {latency}")
    else:
        logger.warning(f"Generation MoE file not found: {generation_moe_file}")
    
    return moe_default_data, moe_low_latency_data

def load_context_attention_data(context_attention_file):
    """
    Load the context attention data
    """
    if not os.path.exists(context_attention_file):
        logger.warning(f"Context attention data file {context_attention_file} not found.")
        return None
    context_attention_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict())))))))
    with open(context_attention_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    for row in rows:
        try:
            window_size = row['window_size']
        except KeyError: #catch potential error for backward comptability
            window_size = 0
        quant_mode, kv_cache_dtype, b, s, n, kv_n, head_size, latency = \
            row['attn_dtype'], row['kv_cache_dtype'], row['batch_size'], row['isl'], row['num_heads'], row['num_key_value_heads'], row['head_dim'], row['latency']
        b=int(b)
        s=int(s)
        n=int(n)
        kv_n=int(kv_n)
        head_size=int(head_size)
        window_size=int(window_size)
        latency=float(latency)
        
        # we only have kv_n==n(MHA) and kv_n==1,2,4,8(XQA), interp/extrap all other num_kv_heads
        kv_n = 0 if n == kv_n else kv_n

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            latency = context_attention_data[quant_mode][kv_cache_dtype][kv_n][head_size][window_size][n][s][b]
            logger.debug('value conflict in context attention data: {} {} {} {} {} {} {}'.format(quant_mode, kv_cache_dtype, head_size, window_size, kv_n, n, s, b))
        except KeyError:
            context_attention_data[quant_mode][kv_cache_dtype][kv_n][head_size][window_size][n][s][b] = latency
    
    return context_attention_data

def load_generation_attention_data(generation_attention_file):
    """
    Load the generation attention data
    """
    if not os.path.exists(generation_attention_file):
        logger.warning(f"Generation attention data file {generation_attention_file} not found.")
        return None
    generation_attention_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict()))))))
    with open(generation_attention_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    for row in rows:
        try:
            window_size = row['window_size']
        except KeyError:
            window_size = 0
        quant_mode, kv_cache_dtype, b, s, n, kv_n, head_size, step, latency = \
            row['attn_dtype'], row['kv_cache_dtype'], row['batch_size'], row['isl'], row['num_heads'], row['num_key_value_heads'], row['head_dim'], row['step'], row['latency']
        b=int(b)
        s=int(s)
        n=int(n)
        kv_n=int(kv_n)
        head_size=int(head_size)
        window_size=int(window_size)
        step = int(step)
        latency=float(latency)

        # we only have kv_n==n(MHA) and kv_n==1,2,4,8(XQA), interp/extrap all other num_kv_heads
        kv_n = 0 if n == kv_n else kv_n
        s = s + step

        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            latency = generation_attention_data[kv_cache_dtype][kv_n][head_size][window_size][n][b][s]
            logger.debug('value conflict in generation attention data: {} {} {} {} {} {}'.format(kv_cache_dtype, kv_n, head_size, window_size, n, b, s))
        except KeyError:
            generation_attention_data[kv_cache_dtype][kv_n][head_size][window_size][n][b][s] = latency
        
    return generation_attention_data

def load_context_mla_data(context_mla_file):
    """
    Load the context mla data for trtllm
    """
    if not os.path.exists(context_mla_file):
        logger.warning(f"Context mla data file {context_mla_file} not found.")
        return None
    context_mla_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict()))))

    with open(context_mla_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    for row in rows:
        quant_mode, kv_cache_dtype, b, s, latency, = \
            row['mla_dtype'], row['kv_cache_dtype'], row['batch_size'], row['isl'], row['latency']

        if 'num_heads' not in row:
            tp_size = int(row['tp_size'])
            num_heads = 128 // tp_size
        else:
            num_heads = int(row['num_heads'])

        b=int(b)
        s=int(s)
        latency=float(latency)

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]
        
        try:
            latency = context_mla_data[quant_mode][kv_cache_dtype][num_heads][s][b]
            logger.debug('value conflict in context mla data: {} {} {} {} {} {}'.format(quant_mode, kv_cache_dtype, num_heads, s, b, latency))
        except KeyError:
            context_mla_data[quant_mode][kv_cache_dtype][num_heads][s][b] = latency
        
    return context_mla_data

def load_generation_mla_data(generation_mla_file):
    """
    Load the generation mla data for trtllm
    """
    if not os.path.exists(generation_mla_file):
        logger.warning(f"Generation mla data file {generation_mla_file} not found.")
        return None
    generation_mla_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict())))
    with open(generation_mla_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    for row in rows:
        quant_mode, kv_cache_dtype, b, s, step, latency = \
            row['mla_dtype'], row['kv_cache_dtype'], row['batch_size'], row['isl'], row['step'], row['latency']

        if 'num_heads' not in row:
            tp_size = int(row['tp_size'])
            num_heads = 128 // tp_size
        else:
            num_heads = int(row['num_heads'])
        
        b=int(b)
        s=int(s)
        step=int(step)
        latency=float(latency)

        s = s + step
        
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            latency = generation_mla_data[kv_cache_dtype][num_heads][b][s]
            logger.debug('value conflict in generation mla data: {} {} {} {} {} '.format(kv_cache_dtype, num_heads, b, s, latency))      
        except KeyError:
            generation_mla_data[kv_cache_dtype][num_heads][b][s] = latency
    
    return generation_mla_data

def load_sglang_context_mla_data(context_mla_file):
    """
    Load the context mla data for sglang
    """
    if not os.path.exists(context_mla_file):
        logger.warning(f"Context mla data file {context_mla_file} not found.")
        return None
    context_mla_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict())))))

    with open(context_mla_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    for row in rows:
        quant_mode, kv_cache_dtype, b, s, latency, = \
            row['mla_dtype'], row['kv_cache_dtype'], row['batch_size'], row['isl'], row['latency']
        
        kernel_source = row['kernel_source'] if 'kernel_source' in row else 'flashinfer'

        if 'num_heads' not in row:
            tp_size = int(row['tp_size'])
            num_heads = 128 // tp_size
        else:
            num_heads = int(row['num_heads'])

        b=int(b)
        s=int(s)
        latency=float(latency)

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]
        
        try:
            latency = context_mla_data[kernel_source][quant_mode][kv_cache_dtype][num_heads][s][b]
            logger.debug('value conflict in context mla data: {} {} {} {} {} {} {}'.format(kernel_source, quant_mode, kv_cache_dtype, num_heads, s, b, latency))
        except KeyError:
            context_mla_data[kernel_source][quant_mode][kv_cache_dtype][num_heads][s][b] = latency
        
    return context_mla_data

def load_sglang_generation_mla_data(generation_mla_file):
    """
    Load the generation mla data for sglang
    """
    if not os.path.exists(generation_mla_file):
        logger.warning(f"Generation mla data file {generation_mla_file} not found.")
        return None
    generation_mla_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict()))))
    with open(generation_mla_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    for row in rows:
        kv_cache_dtype, b, s, step, latency = \
            row['kv_cache_dtype'], row['batch_size'], row['isl'], row['step'], row['latency']
        
        kernel_source = row['kernel_source'] if 'kernel_source' in row else 'flashinfer'

        if 'num_heads' not in row:
            tp_size = int(row['tp_size'])
            num_heads = 128 // tp_size
        else:
            num_heads = int(row['num_heads'])
        
        b=int(b)
        s=int(s)
        step=int(step)
        latency=float(latency)

        s = s + step
        
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            latency = generation_mla_data[kernel_source][kv_cache_dtype][num_heads][b][s]
            logger.debug('value conflict in generation mla data: {} {} {} {} {} {} '.format(kernel_source, kv_cache_dtype, num_heads, b, s, latency))      
        except KeyError:
            generation_mla_data[kernel_source][kv_cache_dtype][num_heads][b][s] = latency
    
    return generation_mla_data

def load_mla_bmm_data(mla_bmm_file):
    """
    Load the mla bmm data for trtllm
    """
    if not os.path.exists(mla_bmm_file):
        logger.warning(f"MLA BMM data file {mla_bmm_file} not found.")
        return None
    mla_bmm_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict())))

    with open(mla_bmm_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    for row in rows:
        quant_mode, num_tokens, num_heads, latency, op_name = row['bmm_dtype'], row['num_tokens'], row['num_heads'], row['latency'], row['op_name']
        num_tokens=int(num_tokens)
        num_heads=int(num_heads)
        latency=float(latency)

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            latency = mla_bmm_data[quant_mode][op_name][num_heads][num_tokens]
            logger.debug('value conflict in mla bmm data: {} {} {} {} {} '.format(op_name, quant_mode, num_heads, num_tokens, latency))      
        except KeyError:
            mla_bmm_data[quant_mode][op_name][num_heads][num_tokens] = latency
    
    return mla_bmm_data

def load_deepep_ll_data(deepep_ll_file):
    """
    Load the DeepEP LL operation data
    """
    if not os.path.exists(deepep_ll_file):
        return None

    deepep_ll_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    with open(deepep_ll_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    
    for row in rows:
        hidden_size = int(row['hidden_size'])
        node_num = int(row['node_num'])
        num_token = int(row['num_token'])
        num_topk = int(row['num_topk'])
        num_experts = int(row['num_experts'])
        combine_avg_t_us = float(row['combine_avg_t_us'])
        dispatch_avg_t_us = float(row['dispatch_avg_t_us'])
        lat = combine_avg_t_us + dispatch_avg_t_us
        
        # Store the data with key structure: [hidden_size][num_topk][num_experts][num_token] -> timing data
        try:
            existing_data = deepep_ll_data[node_num][hidden_size][num_topk][num_experts][num_token]
            logger.debug('value conflict in deepep ll data: {} {} {} {}'.format(hidden_size, num_topk, num_experts, num_token))
        except KeyError:
            deepep_ll_data[node_num][hidden_size][num_topk][num_experts][num_token] = lat
    
    return deepep_ll_data

def load_deepep_normal_data(deepep_normal_file):
    """
    Load the DeepEP normal operation data
    """
    if not os.path.exists(deepep_normal_file):
        return None

    deepep_normal_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
    
    with open(deepep_normal_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    
    for row in rows:
        num_token = int(row['num_token'])
        topk = int(row['num_topk'])
        node_num = int(row['node_num'])
        num_experts = int(row['num_experts'])
        hidden_size = int(row['hidden_size'])
        dispatch_sms = int(row['dispatch_sms'])
        dispatch_transmit_us = float(row['dispatch_transmit_us'])
        dispatch_notify_us = float(row['dispatch_notify_us'])
        combine_transmit_us = float(row['combine_transmit_us'])
        combine_notify_us = float(row['combine_notify_us'])
        lat = dispatch_transmit_us + dispatch_notify_us + combine_transmit_us + combine_notify_us
        
        # Store the data with key structure: [hidden_size][topk][num_experts][dispatch_sms][num_token] -> timing data
        try:
            existing_data = deepep_normal_data[node_num][hidden_size][topk][num_experts][dispatch_sms][num_token]
            logger.debug('value conflict in deepep normal data: {} {} {} {} {}'.format(hidden_size, topk, num_experts, dispatch_sms, num_token))
        except KeyError:
            deepep_normal_data[node_num][hidden_size][topk][num_experts][dispatch_sms][num_token] = lat
        
    
    return deepep_normal_data

class PerfDatabase(object):
    """
    The perf database for a given system, backend and version

    Attributes:
        system (str): the system name
        backend (str): the backend name
        version (str): the version name
        system_spec (dict): the system spec
        _default_sol_mode (common.SOLMode): the default sol mode of the database
        _gemm_data (dict): the gemm data
        _context_attention_data (dict): the context attention data
        _generation_attention_data (dict): the generation attention data
        _custom_allreduce_data (dict): the custom allreduce data
        _moe_data (dict): the moe data
        _context_mla_data (dict): the context mla data
        _generation_mla_data (dict): the generation mla data
        _nccl_data (dict): the nccl data
        _mla_bmm_data (dict): the mla bmm data

    Methods:
        query_gemm: query the gemm data
        query_context_attention: query the context attention data
        query_generation_attention: query the generation attention data
        query_context_mla: query the context mla data
        query_generation_mla: query the generation mla data
        query_nccl: query the nccl data
        query_mla_bmm: query the mla bmm data
        query_mem_op: query the mem op data
        query_p2p: query the p2p data
        query_allreduce: query the allreduce data
        query_moe: query the moe data
    """
    def __init__(self, system : str, 
                 backend : str, 
                 version : str, 
                 systems_dir : str = './systems') -> None:
        """
        Initialize the perf database
        """
        self.system = system 
        self.backend = backend
        self.version = version
        self.system_spec = yaml.load(open(os.path.join(systems_dir, system+'.yaml')), Loader=yaml.SafeLoader)
        self._default_sol_mode = common.SOLMode.NON_SOL # non sol

        data_dir = os.path.join(systems_dir, self.system_spec['data_dir'], backend, version)
        nccl_data_dir = os.path.join(systems_dir, self.system_spec['data_dir'], 'nccl', self.system_spec['misc']['nccl_version'], common.PerfDataFilename.nccl.value)
        
        if backend == 'sglang':
            # For SGLang, only load MoE and MLP data and provide empty structures for other data
            self._gemm_data = {}
            self._context_attention_data = {}
            self._generation_attention_data = {}
            self._custom_allreduce_data = {}
            self._moe_data, self._generation_moe_data = load_sglang_moe_data(os.path.join(data_dir, common.PerfDataFilename.context_moe.value))
            self._context_mla_data = load_sglang_context_mla_data(os.path.join(data_dir, common.PerfDataFilename.context_mla.value))
            self._generation_mla_data = load_sglang_generation_mla_data(os.path.join(data_dir, common.PerfDataFilename.generation_mla.value))
            self._context_mlp_data, self._generation_mlp_data = load_sglang_mlp_data(os.path.join(data_dir, common.PerfDataFilename.context_mlp.value))
            self._deepep_normal_data = load_deepep_normal_data(os.path.join(data_dir, common.PerfDataFilename.deepep_normal.value))
            self._deepep_ll_data = load_deepep_ll_data(os.path.join(data_dir, common.PerfDataFilename.deepep_ll.value))
            self._nccl_data = {}
            self._mla_bmm_data = {}
        else:
            self._gemm_data = load_gemm_data(os.path.join(data_dir, common.PerfDataFilename.gemm.value))
            self._context_attention_data = load_context_attention_data(os.path.join(data_dir, common.PerfDataFilename.context_attention.value))
            self._generation_attention_data = load_generation_attention_data(os.path.join(data_dir, common.PerfDataFilename.generation_attention.value))
            self._custom_allreduce_data = load_custom_allreduce_data(os.path.join(data_dir, common.PerfDataFilename.custom_allreduce.value))
            self._moe_data, self._moe_low_latency_data = load_moe_data(os.path.join(data_dir, common.PerfDataFilename.moe.value))
            self._context_mla_data = load_context_mla_data(os.path.join(data_dir, common.PerfDataFilename.context_mla.value))
            self._generation_mla_data = load_generation_mla_data(os.path.join(data_dir, common.PerfDataFilename.generation_mla.value))
            self._nccl_data = load_nccl_data(nccl_data_dir)
            self._mla_bmm_data = load_mla_bmm_data(os.path.join(data_dir, common.PerfDataFilename.mla_bmm.value))

            # pre-correction
            self._correct_data()

        for quant_mode in self._context_attention_data.keys():
            for kv_cache_dtype in self._context_attention_data[quant_mode].keys():
                for num_kv_heads in self._context_attention_data[quant_mode][kv_cache_dtype]:
                    for head_size in self._context_attention_data[quant_mode][kv_cache_dtype][num_kv_heads]:
                        for window_size in self._context_attention_data[quant_mode][kv_cache_dtype][num_kv_heads][head_size]:
                            data_dict=self._context_attention_data[quant_mode][kv_cache_dtype][num_kv_heads][head_size][window_size]
                            min_x = min(data_dict.keys())
                            target_x_list=[4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 72, 96, 128] # n
                            # currently, support max seq to 1M. Because all the system is linear for now. it will be difficult to do square interpolation. Use more points to do the approximation
                            target_y_list=[16,32,64,128,256,512,1024,2048] + [4096+i*2048 for i in range(14)] + \
                                [32768 + 16384*i for i in range(6)] + [131072 + 32768*i for i in range(12)] + [524288 + 65536*i for i in range(9)]# s
                            target_z_list=[1,2,4,8,16,32,64,128,256,512,384,1024,2048] # b

                            filtered_x_list = []
                            for i in target_x_list:
                                if i >= min_x:
                                    filtered_x_list.append(i)
                            self._extrapolate_data_grid(data_dict=data_dict, #nsb
                                                        target_x_list=filtered_x_list,
                                                        target_y_list=target_y_list,
                                                        target_z_list=target_z_list, sqrt_y_value=True)

        for kv_cache_dtype in self._generation_attention_data.keys():
            for num_kv_heads in self._generation_attention_data[kv_cache_dtype]:
                for head_size in self._generation_attention_data[kv_cache_dtype][num_kv_heads]:
                    for window_size in self._generation_attention_data[kv_cache_dtype][num_kv_heads][head_size]:
                        target_x_list=[4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 72, 96, 128] # n
                        target_y_list=[1,2,4,8,16,32,64,128,256,384,512,1024,2048,8192] # b
                        target_z_list=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,2097152*8] # s
                        data_dict = self._generation_attention_data[kv_cache_dtype][num_kv_heads][head_size][window_size]
                        min_x = min(data_dict.keys())
                        filtered_x_list = []
                        for i in target_x_list:
                            if i >= min_x:
                                filtered_x_list.append(i)

                        self._extrapolate_data_grid(data_dict=data_dict, #nbs
                                                    target_x_list=filtered_x_list,
                                                    target_y_list=target_y_list,
                                                    target_z_list=target_z_list)

        for quant_mode, data_dict in self._gemm_data.items():
            target_x_list = [1,2,4,8,16,32,48,64,80,96,128,160,192,224,256,320,384,448,512,640,768,896,1024,2048,4096,8192,16384,32768,131072,524288,1048576,2097152*8] # num_tokens
            target_y_list = [32,64,128,256,512,768,1024,1536,2048,2560,3072,3584,4096,5120,6144,7168,8192,10240,12288,14336,16384,20480,24576,28672,32768,40960,49152,57344,65536,131072,262144] # to fit vocab gemm
            target_z_list = target_y_list
            self._extrapolate_data_grid(data_dict=data_dict, 
                                  target_x_list=target_x_list,
                                  target_y_list=target_y_list,
                                  target_z_list=target_z_list)
        
        # mla
        # For sglang backend, context_mla_data structure is [kernel_source][quant_mode][kv_cache_dtype][num_heads][s][b]
        # For other backends, it's [quant_mode][kv_cache_dtype][num_heads][s][b]
        if backend == 'sglang':
            for kernel_source in self._context_mla_data.keys():
                for quant_mode in self._context_mla_data[kernel_source].keys():
                    for kv_cache_dtype in self._context_mla_data[kernel_source][quant_mode].keys():
                        num_heads_list = list(self._context_mla_data[kernel_source][quant_mode][kv_cache_dtype].keys())
                        data_dict=self._context_mla_data[kernel_source][quant_mode][kv_cache_dtype]
                        target_x_list=num_heads_list # to reuse x dim
                        # currently, support max seq to 1M. Because all the system is linear for now. it will be difficult to do square interpolation. Use more points to do the approximation
                        target_y_list=[16,32,64,128,256,512,1024,2048] + [4096+i*2048 for i in range(14)] + \
                            [32768 + 16384*i for i in range(6)] + [131072 + 32768*i for i in range(12)] + [524288 + 65536*i for i in range(9)]# s
                        target_z_list=[1,2,4,8,16,32,64,128,256,384,512,1024,2048] # b

                        self._extrapolate_data_grid(data_dict=data_dict, #tpsize,sb
                                                    target_x_list=target_x_list,
                                                    target_y_list=target_y_list,
                                                    target_z_list=target_z_list, sqrt_y_value=True)
        else:
            for quant_mode in self._context_mla_data.keys():
                for kv_cache_dtype in self._context_mla_data[quant_mode].keys():
                    num_heads_list = list(self._context_mla_data[quant_mode][kv_cache_dtype].keys())
                    data_dict=self._context_mla_data[quant_mode][kv_cache_dtype]
                    target_x_list=num_heads_list # to reuse x dim
                    # currently, support max seq to 1M. Because all the system is linear for now. it will be difficult to do square interpolation. Use more points to do the approximation
                    target_y_list=[16,32,64,128,256,512,1024,2048] + [4096+i*2048 for i in range(14)] + \
                        [32768 + 16384*i for i in range(6)] + [131072 + 32768*i for i in range(12)] + [524288 + 65536*i for i in range(9)]# s
                    target_z_list=[1,2,4,8,16,32,64,128,256,384,512,1024,2048] # b

                    self._extrapolate_data_grid(data_dict=data_dict, #tpsize,sb
                                                target_x_list=target_x_list,
                                                target_y_list=target_y_list,
                                                target_z_list=target_z_list, sqrt_y_value=True)

        # For sglang backend, generation_mla_data structure is [kernel_source][kv_cache_dtype][num_heads][b][s]
        # For other backends, it's [kv_cache_dtype][num_heads][b][s]
        if backend == 'sglang':
            for kernel_source in self._generation_mla_data.keys():
                for kv_cache_dtype in self._generation_mla_data[kernel_source].keys():
                    tp_list =  list(self._generation_mla_data[kernel_source][kv_cache_dtype].keys())
                    data_dict=self._generation_mla_data[kernel_source][kv_cache_dtype]
                    target_x_list=tp_list # n
                    target_y_list=[1,2,4,8,16,32,64,128,256,384,512,1024,2048,8192] # b
                    target_z_list=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,2097152*8] # s

                    self._extrapolate_data_grid(data_dict=data_dict, #tpsize, bs
                                                target_x_list=target_x_list,
                                                target_y_list=target_y_list,
                                                target_z_list=target_z_list)
        else:
            for kv_cache_dtype in self._generation_mla_data.keys():
                tp_list =  list(self._generation_mla_data[kv_cache_dtype].keys())
                data_dict=self._generation_mla_data[kv_cache_dtype]
                target_x_list=tp_list # n
                target_y_list=[1,2,4,8,16,32,64,128,256,384,512,1024,2048,8192] # b
                target_z_list=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,2097152*8] # s

                self._extrapolate_data_grid(data_dict=data_dict, #tpsize, bs
                                            target_x_list=target_x_list,
                                            target_y_list=target_y_list,
                                            target_z_list=target_z_list)
        
        # post-correction
        self._correct_data()

        self._update_support_matrix()

    def _update_support_matrix(self):
        """
        Update the support matrix
        """
        # For sglang backend, context_mla_data and generation_mla_data have kernel_source as first level
        # We need to collect quant_modes from the nested structure
        if self.backend == 'sglang':
            context_mla_modes = set()
            for kernel_source in self._context_mla_data.keys():
                for quant_mode in self._context_mla_data[kernel_source].keys():
                    context_mla_modes.add(quant_mode.name)
            
            generation_mla_modes = set()
            for kernel_source in self._generation_mla_data.keys():
                for kv_cache_dtype in self._generation_mla_data[kernel_source].keys():
                    generation_mla_modes.add(kv_cache_dtype.name)
            
            self.supported_quant_mode = {
                'gemm': [key.name for key in self._moe_data.keys()],
                'context_attention': list(context_mla_modes),
                'generation_attention': list(generation_mla_modes),
                'context_mla': list(context_mla_modes),
                'generation_mla': list(generation_mla_modes),
                'mla_bmm': [key.name for key in self._mla_bmm_data.keys()],
                'nccl': [key.name for key in self._nccl_data.keys()],
                'moe': [key.name for key in self._moe_data.keys()],
            }
        elif self.backend == 'trtllm':
            self.supported_quant_mode = {
                'gemm': [key.name for key in self._gemm_data.keys()],
                'context_attention': [key.name for key in self._context_attention_data.keys()],
                'generation_attention': [key.name for key in self._generation_attention_data.keys()],
                'context_mla': [key.name for key in self._context_mla_data.keys()],
                'generation_mla': [key.name for key in self._generation_mla_data.keys()],
                'mla_bmm': [key.name for key in self._mla_bmm_data.keys()],
                'nccl': [key.name for key in self._nccl_data.keys()],
                'moe': [key.name for key in self._moe_data.keys()],
            }
        elif self.backend == 'vllm':
            self.supported_quant_mode = {
            }

    def is_inter_node(self, num_gpus: int) -> bool:
        """
        Check if the number of GPUs is an inter node
        """
        return num_gpus > self.system_spec['node']['num_gpus_per_node']

    def _extrapolate_data_grid(self, data_dict : Dict[int, Dict[int, Dict[int, float]]], 
                               target_x_list : List[int], 
                               target_y_list : List[int], 
                               target_z_list : List[int], 
                               sqrt_y_value : bool = False) -> None:
        """
        Extrapolate the data grid, we extrapolate the data grid at the initialization stage. Future query will based on interpolation.
        """
        x_list = sorted(list(data_dict.keys()))
        for x in x_list:
            # z_direction
            for y in sorted(list(data_dict[x].keys())):
                z_dict = data_dict[x][y]
                if len(z_dict) <=1:
                    logger.warning(f"only one data point for a given xy, might trigger error. Please revisit data collection. {x=}, {y=}, {z_dict=}")
                    continue
                for z in target_z_list:
                    if z not in z_dict.keys():
                        z_left, z_right = self._nearest_1d_point_helper(z, list(z_dict.keys()), False)
                        # Check if both left and right boundaries exist
                        if z_left not in z_dict.keys() or z_right not in z_dict.keys():
                            logger.warning(f"Skipping interpolation for z={z} as boundaries z_left={z_left} or z_right={z_right} do not exist in z_dict for x={x}, y={y}")
                            continue
                        value = self._interp_1d([z_left, z_right], [data_dict[x][y][z_left],data_dict[x][y][z_right]], z)
                        z_dict[z] = value
            
            # y_direction
            for y in target_y_list:
                if y not in data_dict[x].keys():
                    y_left, y_right = self._nearest_1d_point_helper(y, list(data_dict[x].keys()), False)
                    # Check if both left and right boundaries exist
                    if y_left not in data_dict[x].keys() or y_right not in data_dict[x].keys():
                        logger.warning(f"Skipping interpolation for y={y} as boundaries y_left={y_left} or y_right={y_right} do not exist in data_dict[{x}]")
                        continue
                    
                    z_list = sorted(list(data_dict[x][y_left].keys()))
                    for z in z_list:
                        # Check if z exists in both y_left and y_right
                        if z not in data_dict[x][y_left].keys() or z not in data_dict[x][y_right].keys():
                            logger.warning(f"Skipping interpolation for z={z} as it does not exist in both y_left={y_left} and y_right={y_right}")
                            continue
                            
                        y_left_value = data_dict[x][y_left][z]
                        y_right_value = data_dict[x][y_right][z]
                        assert(y_right_value is not None), "y_right_value cannot be None"
                        if sqrt_y_value:
                            y_left_value = math.sqrt(y_left_value)
                            y_right_value = math.sqrt(y_right_value)
                        value = self._interp_1d([y_left, y_right], [y_left_value, y_right_value], y)
                        if sqrt_y_value:
                            value = value*value

                        if y not in data_dict[x].keys():
                            data_dict[x][y] = {z:value}
                        else:
                            data_dict[x][y][z] = value

        for x in target_x_list:
            if x not in data_dict.keys():
                x_left, x_right = self._nearest_1d_point_helper(x, list(data_dict.keys()), False)
                # Check if both left and right boundaries exist
                if x_left not in data_dict.keys() or x_right not in data_dict.keys():
                    logger.warning(f"Skipping interpolation for x={x} as boundaries x_left={x_left} or x_right={x_right} do not exist in data_dict")
                    continue
                    
                for y in sorted(data_dict[x_left].keys()):
                    # Check if y exists in both x_left and x_right
                    if y not in data_dict[x_left].keys() or y not in data_dict[x_right].keys():
                        logger.warning(f"Skipping interpolation for y={y} as it does not exist in both x_left={x_left} and x_right={x_right}")
                        continue
                        
                    for z in sorted(data_dict[x_left][y].keys()):
                        # Check if z exists in both x_left and x_right for the given y
                        if z not in data_dict[x_left][y].keys() or z not in data_dict[x_right][y].keys():
                            logger.warning(f"Skipping interpolation for z={z} as it does not exist in both x_left={x_left} and x_right={x_right} for y={y}")
                            continue
                            
                        x_left_value = data_dict[x_left][y][z]
                        x_right_value = data_dict[x_right][y][z]
                        assert(x_right_value is not None), "x_right_value cannot be None"
                        value = self._interp_1d([x_left, x_right], [x_left_value, x_right_value], x)
                        if x not in data_dict.keys():
                            data_dict[x] = {y:{z:value}}
                        elif y not in data_dict[x].keys():
                            data_dict[x][y] = {z:value}
                        else:
                            data_dict[x][y][z] = value

    def _nearest_1d_point_helper(self, x:int, values:List[int], inner_only:bool=True) -> Tuple[int, int]:
        """
        Find the nearest 1d point
        """
        assert(values is not None and len(values) >= 2), f"values is None or len(values) < 2"
        sorted_values = sorted(values)

        if x < sorted_values[0]:
            if inner_only:
                raise ValueError(f"x is less than the smallest value in the list. {x=}, {sorted_values=}")
            else:
                return sorted_values[0], sorted_values[1]
        elif x > sorted_values[-1]:
            if inner_only:
                raise ValueError(f"x is greater than the largest value in the list. {x=}, {sorted_values=}")
            else:
                return sorted_values[-2], sorted_values[-1]

        for i, value in enumerate(sorted_values):
            if x >= value and i != len(sorted_values)-1:
                continue
            else:
                end = value
                start = sorted_values[i-1]
                break
        if start is None or end is None:
            raise ValueError(f"start or end is None. {x=}, {sorted_values=}, start={start=}, end={end=}")
        return start, end
    
    def _validate(self, value:float) -> float:
        """
        Validate the value
        """
        if value < 0.:
            logger.debug(f'Negative value detected {value}, pass')
        return value
    
    def _interp_3d_linear(self, x:int, y:int, z:int, data:dict) -> float:
        """
        Interpolate the 3d data using linear interpolation
        """
        points_list = []
        values_list = []
        x_left, x_right = self._nearest_1d_point_helper(x, list(data.keys()))
        for i in [x_left, x_right]:
            y_left, y_right = self._nearest_1d_point_helper(y, list(data[i].keys()))
            for j in [y_left, y_right]:
                z_left, z_right = self._nearest_1d_point_helper(z, list(data[i][j].keys()))
                points_list.append([i, j, z_left])
                points_list.append([i, j, z_right])
                values_list.append(data[i][j][z_left])
                values_list.append(data[i][j][z_right])
        
        return self._validate(interpolate.griddata(np.array(points_list), np.array(values_list), (x,y,z), method='linear'))

    def _interp_2d_linear(self, x:int, y:int, data:dict) -> float:
        """
        Interpolate the 3d data using linear interpolation
        """
        points_list = []
        values_list = []
        x_left, x_right = self._nearest_1d_point_helper(x, list(data.keys()))
        for i in [x_left, x_right]:
            y_left, y_right = self._nearest_1d_point_helper(y, list(data[i].keys()))
            for j in [y_left, y_right]:
                points_list.append([i, j])
                values_list.append(data[i][j])
        
        return self._validate(interpolate.griddata(np.array(points_list), np.array(values_list), (x,y), method='linear'))

    def _interp_3d(self, x:int, y:int, z:int, data:dict, method:str) -> float:
        """
        Interpolate the 3d data using the given method
        """
        if method == 'linear':
            return self._interp_3d_linear(x, y, z, data)
        else:
            return self._interp_2d_1d(x, y, z, data, method)
    
    def _bilinear_interpolation(self, x_list : List[int], y_list : List[int], x : int, y : int, data : dict) -> float:
        """
        Interpolate the 2d data using bilinear interpolation
        """
        x1, x2 = x_list
        # assure xy has a rectengle grid
        y1, y2 = y_list
        # Calculate the weights for the corners
        Q11,Q12,Q21,Q22 = data[x1][y1], data[x1][y2], data[x2][y1], data[x2][y2]
        f_x1_y1 = Q11 * (x2 - x) * (y2 - y)
        f_x1_y2 = Q12 * (x2 - x) * (y - y1)
        f_x2_y1 = Q21 * (x - x1) * (y2 - y)
        f_x2_y2 = Q22 * (x - x1) * (y - y1)
        # Calculate the total weight
        total_weight = (x2 - x1) * (y2 - y1)
        # Calculate the interpolated value
        interpolated_value = (f_x1_y1 + f_x1_y2 + f_x2_y1 + f_x2_y2) / total_weight
        return interpolated_value

    def _interp_2d_1d(self, x:int, y:int, z:int, data:dict, method='bilinear') -> float:
        """
        Interpolate the 3d data using the given method, 2d after 1d.
        """
        x_values = []
        x_left, x_right = self._nearest_1d_point_helper(x, list(data.keys()))

        for i in [x_left, x_right]:
            points_list = []
            values_list = []
            y_left, y_right = self._nearest_1d_point_helper(y, list(data[i].keys()))
            for j in [y_left, y_right]:
                z_left, z_right = self._nearest_1d_point_helper(z, list(data[i][j].keys()))
                points_list.append([j, z_left])
                points_list.append([j, z_right])
                values_list.append(data[i][j][z_left])
                values_list.append(data[i][j][z_right])
            if method == 'cubic':
                x_values.append(self._validate(interpolate.griddata(np.array(points_list), np.array(values_list), (y,z), method='cubic')))
            elif method == 'bilinear':
                x_values.append(self._validate(self._bilinear_interpolation([y_left, y_right],[z_left, z_right],y,z,data[i])))
            else:
                raise NotImplementedError

        return self._validate(self._interp_1d([x_left,x_right],x_values,x))

    def _interp_1d(self, x:List[int], y:List[int], value:int) -> float:
        """
        Interpolate the 1d data using linear interpolation
        """
        x0,x1 = x
        y0,y1 = y
        if (x0 - x1) * (y0 - y1) < 0 and (value - x0) * (value - x1) > 0:
            y1 = y0

        if y0 == y1:
            return y0

        return y0 + (y1 - y0) / (x1 - x0) * (value - x0)

    def set_default_sol_mode(self, mode:common.SOLMode) -> None:
        """
        Set the default sol mode
        """
        self._default_sol_mode = mode
    
    def get_default_sol_mode(self) -> common.SOLMode:
        """
        Get the default sol mode
        """
        return self._default_sol_mode
    
    def query_gemm(self, 
                   m : int, 
                   n : int, 
                   k : int, 
                   quant_mode : common.GEMMQuantMode, 
                   sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the gemm data
        """
        def get_sol(m : int, n : int, k : int, quant_mode : common.GEMMQuantMode) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            sol_math = 2 * m * n * k / (self.system_spec['gpu']['float16_tc_flops']*quant_mode.value.compute) * 1000
            sol_mem = quant_mode.value.memory * (m * n + m * k + n * k) / self.system_spec['gpu']['mem_bw'] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem
        
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(m, n, k, quant_mode)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(m, n, k, quant_mode)
        else:
            result = self._interp_3d(m, n, k, self._gemm_data[quant_mode], 'cubic')
            return result
    
    def query_context_attention(self, 
                                b : int, 
                                s : int, 
                                n : int, 
                                n_kv : int, 
                                kvcache_quant_mode : common.KVCacheQuantMode, 
                                fmha_quant_mode : common.FMHAQuantMode, 
                                sol_mode : Optional[common.SOLMode] = None,
                                window_size : int = 0,
                                head_size : int = 128) -> float:
        """
        Query the context attention data
        """
        def get_sol(b : int, s : int, n : int, n_kv : int, h : int, w : int,
                    kvcache_quant_mode : common.KVCacheQuantMode, fmha_quant_mode : common.FMHAQuantMode,) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            if w > 0 and s > w:
                # Sliding window attention
                # Each position attends to at most w previous positions
                ops = 2 * b * s * w * n * h * 2
            else:
                # Normal no sliding window
                ops = 2 * b * s * s * n * h * 2 / 2 # 2 for fma, 2 for q*k^t+*v, /2 for causality.
            mem_bytes = 2 * b * (
                n*s*h + # Q read, assuming 16 bits
                2*n_kv*s*h + # K,V read
                n*s*h) # Output write, assuming 16 bits
            sol_math = ops / self.system_spec['gpu']['float16_tc_flops'] * 1000 / fmha_quant_mode.value.compute
            sol_mem = mem_bytes / self.system_spec['gpu']['mem_bw'] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem
        assert(n_kv <= n), "n_kv must be less than or equal to n"

        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode)
        else:
            if head_size not in [64,128]:
                return get_sol(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode)[0]
            if n_kv == n:
                attention_dict = self._context_attention_data[fmha_quant_mode][kvcache_quant_mode][0][head_size][window_size]
            else:
                attention_dict = self._context_attention_data[fmha_quant_mode][kvcache_quant_mode][n_kv][head_size][window_size]
            latency = self._interp_3d(n, s, b, attention_dict, 'cubic')
            return latency
    
    def query_generation_attention(self, 
                                   b : int, 
                                   s : int, 
                                   n : int, 
                                   n_kv : int, 
                                   kvcache_quant_mode : common.KVCacheQuantMode, 
                                   sol_mode : Optional[common.SOLMode] = None,
                                   window_size : int = 0,
                                   head_size : int = 128) -> float:
        """
        Query the generation attention data
        """
        def get_sol(b : int, s : int, n : int, n_kv : int, h : int, w : int, kvcache_quant_mode : common.KVCacheQuantMode) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            if w > 0:
                kv_len = min(s - 1, w)
            else:
                kv_len = s - 1           
            # only consider fp16 mmha
            ops = 2 * b * n * h * 2 * (kv_len)   # 2 for fma, 2 for q*k^t+*v
            # kvcache load bytes will depend on kvcache quant. while input q and output might be in fp16.
            mem_bytes = b * (
                n*h*2 +                              # Query read, assuming 16bits
                2*n_kv*(kv_len)*h*kvcache_quant_mode.value.memory + # K, V cache read
                n*h*2                                # Output write, assuming 16bits
                )
            
            sol_math = ops / self.system_spec['gpu']['float16_tc_flops'] * 1000
            sol_mem = mem_bytes / self.system_spec['gpu']['mem_bw'] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem
        assert(n_kv <= n), "n_kv must be less than or equal to n"

        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode)
        else:
            if head_size not in [64,128]:
                return get_sol(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode)[0]
            else:
                attention_dict = self._generation_attention_data[kvcache_quant_mode][n_kv][head_size][window_size]

            latency =  self._interp_3d(n, b, s, attention_dict, 'bilinear')
            return latency

    def query_context_mla(self, 
                          b : int, 
                          s : int, 
                          num_heads : int, 
                          kvcache_quant_mode : common.KVCacheQuantMode, 
                          fmha_quant_mode : common.FMHAQuantMode, 
                          sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the context mla data
        """
        def get_sol(b : int, s : int, num_heads : int, kvcache_quant_mode : common.KVCacheQuantMode, fmha_quant_mode : common.FMHAQuantMode) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            ops = b * num_heads * 2 / 2 *(s * s * 192 + s * s * 128) # 2 for fma, 2 for causality. num_heads, for local heads
            mem_bytes = b * num_heads * 2 * (2*s*192 + 2*s*128) # 2 for fp16, TODO
            sol_math = ops / self.system_spec['gpu']['float16_tc_flops'] * 1000 / fmha_quant_mode.value.compute
            sol_mem = mem_bytes / self.system_spec['gpu']['mem_bw'] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem


        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(b, s, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(b, s, num_heads, kvcache_quant_mode, fmha_quant_mode)
        else:
            mla_dict = self._context_mla_data[fmha_quant_mode][kvcache_quant_mode]
            latency = self._interp_3d(num_heads, s, b, mla_dict, 'cubic')
            return latency
    
    def query_generation_mla(self, 
                             b : int, 
                             s : int, 
                             num_heads : int, 
                             kvcache_quant_mode : common.KVCacheQuantMode, 
                             sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the generation mla data
        """
        def get_sol(b : int, s : int, num_heads : int, kvcache_quant_mode : common.KVCacheQuantMode) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            # only consider fp16 mmha
            ops = 2 * b * num_heads * 1088 * s # 2 for fma
            # kvcache load bytes will depend on kvcache quant. while input q and output might be in fp16.
            mem_bytes = b * (num_heads * 1088 * 2 + (s-1)*1088 * kvcache_quant_mode.value.memory)
            
            sol_math = ops / self.system_spec['gpu']['float16_tc_flops'] * 1000 # only fp16
            sol_mem = mem_bytes / self.system_spec['gpu']['mem_bw'] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem
        
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(b, s, num_heads, kvcache_quant_mode)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(b, s, num_heads, kvcache_quant_mode)
        else:
            mla_dict = self._generation_mla_data[kvcache_quant_mode]
            latency =  self._interp_3d(num_heads, b, s, mla_dict, 'bilinear')
            return latency
    
    def query_generation_mla_sglang(self, 
                                   b : int, 
                                   s : int, 
                                   tp_size : int, 
                                   kvcache_quant_mode : common.KVCacheQuantMode, 
                                   fmha_quant_mode : common.FMHAQuantMode,
                                   attention_backend : Optional[str] = None,
                                   sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the generation mla data for SGLang backend with SOL calculation
        """
        def get_sol(b : int, s : int, tp_size : int, kvcache_quant_mode : common.KVCacheQuantMode, fmha_quant_mode : common.FMHAQuantMode) -> Tuple[float, float, float]:
            
            hidden_size = 7168
            q_lora_rank = 1536
            kv_lora_rank = 512
            qk_rope_head_dim = 64
            qk_nope_head_dim = 128
            v_head_dim = 128
            num_head = 128 // tp_size 
            
            # qkv_a projection (decode mode)
            qkv_a_flop = 2 * hidden_size * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * b
            qkv_a_mem = b * hidden_size + hidden_size * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) + 2 * b * (q_lora_rank + kv_lora_rank + qk_rope_head_dim)

            # q_b projection
            q_b_flop = 2 * q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim) * b
            q_b_mem = b * q_lora_rank + q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim) + 2 * b * num_head * (qk_rope_head_dim + qk_nope_head_dim)

            # q_w_kc (attention computation)
            q_w_kc_flop = 2 * num_head * qk_nope_head_dim * kv_lora_rank * b
            q_w_kc_mem = b * num_head * qk_nope_head_dim + num_head * kv_lora_rank * qk_nope_head_dim + 2 * b * num_head * kv_lora_rank

            attn_flop = 2 * b * s * num_head * (qk_rope_head_dim + kv_lora_rank * 2)
            attn_mem = b * num_head * (kv_lora_rank + qk_rope_head_dim) + b * s * (qk_rope_head_dim + kv_lora_rank) + b * num_head * kv_lora_rank

            # s_w_vc (attention output projection)
            s_w_vc_flop = 2 * b * num_head * kv_lora_rank * v_head_dim
            s_w_vc_mem = b * num_head * kv_lora_rank + num_head * v_head_dim * kv_lora_rank + 2 * b * num_head * v_head_dim

            # attention output projection
            attn_out_flop = 2 * num_head * v_head_dim * hidden_size * b
            attn_out_mem = b * num_head * v_head_dim + num_head * v_head_dim * hidden_size + 2 * b * hidden_size

            ops = qkv_a_flop + q_b_flop + q_w_kc_flop + s_w_vc_flop + attn_out_flop
            mem_bytes = (qkv_a_mem + q_b_mem + q_w_kc_mem + attn_mem*2 + s_w_vc_mem + attn_out_mem) * fmha_quant_mode.value.memory
            sol_math = ops / (self.system_spec['gpu']['float16_tc_flops'] * fmha_quant_mode.value.compute) * 1000
            sol_math += attn_flop / (self.system_spec['gpu']['float16_tc_flops']) * 1000
            sol_mem = mem_bytes / self.system_spec['gpu']['mem_bw'] * 1000
            sol_time = max(sol_math, sol_mem)
            
            return sol_time, sol_math, sol_mem
        
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode)
        else:
            if attention_backend is None:
                attention_backend = "flashinfer"
            if attention_backend == "flashinfer":
                attn_data = self._generation_mla_data["flashinfer"]
            elif attention_backend == "fa3":
                attn_data = self._generation_mla_data["fa3"]
            else:
                raise ValueError(f"Unsupported attention backend: {attention_backend}")
            # Convert tp_size to num_heads (assuming 128 total heads for DeepSeek)
            num_heads = 128 // tp_size
            mla_dict = attn_data[kvcache_quant_mode]
            latency =  self._interp_3d(num_heads, b, s, mla_dict, 'bilinear')
            return latency
    
    def query_context_mla_sglang(self, 
                                b : int, 
                                s : int, 
                                tp_size : int, 
                                kvcache_quant_mode : common.KVCacheQuantMode, 
                                fmha_quant_mode : common.FMHAQuantMode,
                                attention_backend : Optional[str] = None,
                                sol_mode : Optional[common.SOLMode] = None) -> float:
        def get_sol(b : int, s : int, tp_size : int, kvcache_quant_mode : common.KVCacheQuantMode, fmha_quant_mode : common.FMHAQuantMode) -> Tuple[float, float, float]:
            
            hidden_size = 7168
            q_lora_rank = 1536
            kv_lora_rank = 512
            qk_rope_head_dim = 64
            qk_nope_head_dim = 128
            v_head_dim = 128
            num_head = 128 // tp_size      

            # qkv_a projection (prefill mode)
            qkv_a_flop = 2 * hidden_size * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * b * s
            qkv_a_mem = b * hidden_size * s + hidden_size * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) + 2 * b * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * s
            
            # q_b projection
            q_b_flop = 2 * q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim) * b * s
            q_b_mem = b * q_lora_rank * s + q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim) + 2 * b * num_head * (qk_rope_head_dim + qk_nope_head_dim) * s
            
            # kv_b projection
            kv_b_flop = 2 * kv_lora_rank * num_head * (qk_nope_head_dim + v_head_dim) * b * s
            kv_b_mem = b * s * kv_lora_rank + num_head * (qk_nope_head_dim + v_head_dim) * kv_lora_rank + 2 * b * num_head * (qk_nope_head_dim + v_head_dim) * s
                
            # attention computation (prefill mode)
            attn_flop = 2 * num_head * (qk_nope_head_dim * 2 + qk_rope_head_dim) * b * s * s // 2
            attn_mem = b * s * num_head * (qk_nope_head_dim + qk_rope_head_dim) * 2+b * s * num_head * qk_nope_head_dim + b * s * num_head * qk_nope_head_dim
         
            # attention output projection
            attn_out_flop = 2 * num_head * v_head_dim * hidden_size * b * s
            attn_out_mem = b * num_head * v_head_dim * s + num_head * v_head_dim * hidden_size + 2 * b * hidden_size * s

            ops = qkv_a_flop + q_b_flop + kv_b_flop + attn_out_flop
            mem_bytes = (qkv_a_mem + q_b_mem + kv_b_mem + attn_mem * 2 + attn_out_mem) * fmha_quant_mode.value.memory
            sol_math = ops / (self.system_spec['gpu']['float16_tc_flops'] * fmha_quant_mode.value.compute) * 1000
            sol_math += attn_flop / (self.system_spec['gpu']['float16_tc_flops']) * 1000
            sol_mem = mem_bytes / self.system_spec['gpu']['mem_bw'] * 1000
            sol_time = max(sol_math, sol_mem)            
            return sol_time, sol_math, sol_mem
        
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode)
        else:
            if attention_backend is None:
                attention_backend = "flashinfer"
            if attention_backend == "flashinfer":
                attn_data = self._context_mla_data["flashinfer"]
            elif attention_backend == "fa3":
                attn_data = self._context_mla_data["fa3"]
            else:
                raise ValueError(f"Unsupported attention backend: {attention_backend}")

            # Convert tp_size to num_heads (assuming 128 total heads for DeepSeek)
            num_heads = 128 // tp_size
            mla_dict = attn_data[fmha_quant_mode][kvcache_quant_mode]
            latency = self._interp_3d(num_heads, s, b, mla_dict, 'cubic')
            return latency
        
    # to simplify, we no longer support allreduce_strategy
    def query_allreduce(self, 
                        quant_mode : common.CommQuantMode, 
                        tp_size : int, 
                        size : int, 
                        sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the allreduce data
        """
        def get_sol(quant_mode : common.CommQuantMode, tp_size : int, size : int) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            if tp_size == 1:
                return 0,0,0            
            # count, not size in bytes
            p2pBW = self.system_spec['node']['inter_node_bw'] if tp_size > self.system_spec['node']['num_gpus_per_node'] else self.system_spec['node']['intra_node_bw']

            # assume all are ring allreduce, ignore constant latency (~1us for hopper, ~2us for two-die blackwell)
            # assume float16
            sol_time = 2*size*2/tp_size*(tp_size-1)/p2pBW
            return sol_time*1000,0,0
        
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(quant_mode, tp_size, size)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(quant_mode, tp_size, size)
        else:
            if tp_size == 1:
                return 0.
            comm_dict = self._custom_allreduce_data[quant_mode][min(tp_size,8)]['AUTO'] # use AUTO for allreduce strategy
            size_left, size_right = self._nearest_1d_point_helper(size, list(comm_dict.keys()), inner_only=False)
            lat = self._interp_1d([size_left, size_right], [comm_dict[size_left], comm_dict[size_right]], size)
            if tp_size > 8: # FIXME, to collect real data, use inter-node and intra-node data seperately
                if tp_size > self.system_spec['node']['num_gpus_per_node']:
                    lat = lat * (tp_size-1)/tp_size * 8/7 * self.system_spec['node']['intra_node_bw'] / self.system_spec['node']['inter_node_bw']
                else:
                    lat = lat * (tp_size-1)/tp_size * 8/7
            return lat
    
    def query_nccl(self, dtype : common.CommQuantMode, 
                   num_gpus : int, 
                   operation : str, 
                   message_size : int, # element number
                   sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the nccl data

        message_size: element number
        """
        def get_sol(dtype : common.CommQuantMode, num_gpus : int, operation : str, message_size : int) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            message_size: element number
            """
            sol_time = 0.0
            p2p_bw = self.system_spec['node']['inter_node_bw'] if num_gpus > self.system_spec['node']['num_gpus_per_node'] else self.system_spec['node']['intra_node_bw']

            if operation == 'all_gather':
                sol_time = dtype.value.memory * message_size * (num_gpus - 1) / num_gpus / p2p_bw * 1000
            elif operation == 'alltoall':
                sol_time = dtype.value.memory * message_size * (num_gpus - 1) / num_gpus / p2p_bw * 1000
            elif operation == 'reduce_scatter':
                sol_time = dtype.value.memory * message_size * (num_gpus - 1) / num_gpus / p2p_bw * 1000
            elif operation == 'all_reduce':
                sol_time = 2 * dtype.value.memory * message_size * (num_gpus - 1) / num_gpus / p2p_bw * 1000
            return sol_time, 0, sol_time
            
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(dtype, num_gpus, operation, message_size)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(dtype, num_gpus, operation, message_size)
        else:
            if num_gpus == 1:
                return 0.

        max_num_gpus = max(self._nccl_data[dtype][operation].keys())
        nccl_dict = self._nccl_data[dtype][operation][min(num_gpus,max_num_gpus)]
        size_left, size_right = self._nearest_1d_point_helper(message_size, list(nccl_dict.keys()), inner_only=False)
        lat = self._interp_1d([size_left, size_right], [nccl_dict[size_left], nccl_dict[size_right]], message_size)

        if num_gpus > max_num_gpus: # need to do some correction
            logger.debug(f"nccl num_gpus {num_gpus} > max_num_gpus {max_num_gpus}, need to do some correction")
            if max_num_gpus > self.system_spec['node']['num_gpus_per_node']: # all inter node
                scale_factor = 1
            elif num_gpus > self.system_spec['node']['num_gpus_per_node']:
                scale_factor = self.system_spec['node']['intra_node_bw'] / self.system_spec['node']['inter_node_bw']
            else: # all intra node
                scale_factor = 1
            lat = lat * (num_gpus-1) / num_gpus * max_num_gpus / (max_num_gpus-1) * scale_factor

        return lat
    
    def query_moe(self, 
                  num_tokens : int, 
                  hidden_size : int, 
                  inter_size : int, 
                  topk : int, 
                  num_experts : int, 
                  moe_tp_size : int, 
                  moe_ep_size : int, 
                  quant_mode : common.MoEQuantMode, 
                  workload_distribution : str, 
                  is_context : bool = True,
                  moe_backend : Optional[str] = None,
                  sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the moe data
        """
        def get_sol(num_tokens : int, 
                    hidden_size : int, 
                    inter_size : int, 
                    topk : int, 
                    num_experts : int, 
                    moe_tp_size : int, 
                    moe_ep_size : int, 
                    quant_mode : common.MoEQuantMode, 
                    workload_distribution : str) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            # we ignore router part. only consider mlp
            # tp already impacted inter_size.
            # only consider even workload.
            total_tokens = num_tokens * topk 
            ops = total_tokens * hidden_size * inter_size * 3 * 2 // moe_ep_size // moe_tp_size # ffn1, ffn2, gate
            mem_bytes = quant_mode.value.memory * (total_tokens * hidden_size * 3 # input+output
                                                                        + total_tokens * inter_size * 3 // moe_tp_size # intermediate, assume ffn1/gate all need to write results.
                                                                        + hidden_size * inter_size * 3 // moe_tp_size * min(num_experts//moe_ep_size, total_tokens))
            sol_math = ops / (self.system_spec['gpu']['float16_tc_flops']*quant_mode.value.compute) * 1000
            sol_mem = mem_bytes / self.system_spec['gpu']['mem_bw'] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(num_tokens, hidden_size, inter_size, topk, num_experts, moe_tp_size, moe_ep_size, quant_mode, workload_distribution)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(num_tokens, hidden_size, inter_size, topk, num_experts, moe_tp_size, moe_ep_size, quant_mode, workload_distribution)
        else:
            if self.backend == common.BackendName.sglang.value:
                # Set default moe_backend if not specified
                if moe_backend is None:
                    moe_backend = "deepep_moe"
                
                if moe_backend == "deepep_moe":
                    if is_context:
                        moe_data = self._moe_data
                    else:
                        moe_data = self._generation_moe_data

                    moe_dict = moe_data[quant_mode][workload_distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size]
                    num_left, num_right = self._nearest_1d_point_helper(num_tokens, list(moe_dict.keys()), inner_only=False)
                    lat = self._interp_1d([num_left, num_right], [moe_dict[num_left], moe_dict[num_right]], num_tokens)
                    return lat 
            elif self.backend == common.BackendName.trtllm.value: 
                # aligned with trtllm, kernel source selection.
                if num_tokens <= 128 and self._moe_low_latency_data and quant_mode == common.MoEQuantMode.nvfp4:
                    try:
                        if workload_distribution in self._moe_low_latency_data[quant_mode].keys():
                            moe_dict = self._moe_low_latency_data[quant_mode][workload_distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size]
                        else:
                            moe_dict = self._moe_low_latency_data[quant_mode]["uniform"][topk][num_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size]
                        logger.debug(f"trying to find low latency data for moe {quant_mode} {workload_distribution} {topk} {num_experts} {hidden_size} {inter_size} {moe_tp_size} {moe_ep_size} but failed.")
                    except:
                        if workload_distribution in self._moe_data[quant_mode].keys():
                            moe_dict = self._moe_data[quant_mode][workload_distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size]
                        else:
                            moe_dict = self._moe_data[quant_mode]["uniform"][topk][num_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size]
                else:
                    if workload_distribution in self._moe_data[quant_mode].keys():
                        moe_dict = self._moe_data[quant_mode][workload_distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size]
                    else:
                        moe_dict = self._moe_data[quant_mode]["uniform"][topk][num_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size]
                if not moe_dict:
                    return get_sol(num_tokens, hidden_size, inter_size, topk, num_experts, moe_tp_size, moe_ep_size, quant_mode, workload_distribution)[0]
                num_left, num_right = self._nearest_1d_point_helper(num_tokens, list(moe_dict.keys()), inner_only=False)
                lat = self._interp_1d([num_left, num_right], [moe_dict[num_left], moe_dict[num_right]], num_tokens)
                return lat
            else:
                raise NotImplementedError(f"backend {self.backend} not supported for moe")

    def query_mla_bmm(self, 
                      num_tokens : int, 
                      num_heads : int, 
                      quant_mode : common.GEMMQuantMode, 
                      if_pre : bool = True, 
                      sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the mla bmm data
        """
        def get_sol(num_tokens : int, num_heads : int, quant_mode : common.GEMMQuantMode, if_pre : bool) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            ops = 2 * num_tokens * num_heads * 128 * 512 # 2 for fma
            mem_bytes = num_heads * (num_tokens * 640 + 128 * 512) * quant_mode.value.memory
            sol_math = ops / (self.system_spec['gpu']['float16_tc_flops'] * quant_mode.value.compute) * 1000
            sol_mem = mem_bytes / self.system_spec['gpu']['mem_bw'] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(num_tokens, num_heads, quant_mode, if_pre)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(num_tokens, num_heads, quant_mode, if_pre)
        else:
            if quant_mode not in self._mla_bmm_data:
                quant_mode = common.GEMMQuantMode.float16
            mla_bmm_dict = self._mla_bmm_data[quant_mode]['mla_gen_pre' if if_pre else 'mla_gen_post'][num_heads]
            num_left, num_right = self._nearest_1d_point_helper(num_tokens, list(mla_bmm_dict.keys()), inner_only=False)
            lat = self._interp_1d([num_left, num_right], [mla_bmm_dict[num_left], mla_bmm_dict[num_right]], num_tokens)
            return lat
        
    def query_mem_op(self, 
                     mem_bytes : int, 
                     sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the mem op data
        """
        def get_sol(mem_bytes : int) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            sol_time = mem_bytes / self.system_spec['gpu']['mem_bw'] * 1000
            return sol_time, 0, sol_time
        
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(mem_bytes)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(mem_bytes)
        else:
            lat = (mem_bytes / (self.system_spec['gpu']['mem_bw'] * self.system_spec['gpu']['mem_bw_empirical_scaling_factor']) + self.system_spec['gpu']['mem_empirical_constant_latency']) * 1000
            return lat
    
    def query_p2p(self, 
                  message_bytes : int, 
                  sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the p2p data
        """
        def get_sol(message_bytes : int) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            # TODO, use intra_node_bw if num_gpus < num_gpus_per_node
            sol_time = message_bytes / self.system_spec['node']['inter_node_bw'] * 1000
            return sol_time, 0, sol_time
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(message_bytes)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(message_bytes)
        else:
            return (message_bytes / self.system_spec['node']['inter_node_bw'] + self.system_spec['node']['p2p_latency']) * 1000
        
    def _correct_data(self) -> None:
        """
        Correct the data based on sol time reference.
        """
        # correct gemm
        for quant_mode in self._gemm_data.keys():
            for m in self._gemm_data[quant_mode].keys():
                for n in self._gemm_data[quant_mode][m].keys():
                    for k in self._gemm_data[quant_mode][m][n].keys():
                        sol = self.query_gemm(m, n, k, quant_mode, sol_mode=common.SOLMode.SOL)
                        if sol > self._gemm_data[quant_mode][m][n][k]:
                            logger.debug('gemm quant {} m{} n{} k{}: sol {} > perf_db {}'.format(quant_mode, m, n, k, sol, self._gemm_data[quant_mode][m][n][k]))
                            self._gemm_data[quant_mode][m][n][k] = max(sol, self._gemm_data[quant_mode][m][n][k])
        
        # correct generation attention
        for quant_mode in self._generation_attention_data.keys():
            for n_kv in self._generation_attention_data[quant_mode].keys():
                for head_size in self._generation_attention_data[quant_mode][n_kv].keys():
                    for window_size in self._generation_attention_data[quant_mode][n_kv][head_size].keys():
                        for n in self._generation_attention_data[quant_mode][n_kv][head_size][window_size].keys():
                            for b in self._generation_attention_data[quant_mode][n_kv][head_size][window_size][n].keys():
                                for s in self._generation_attention_data[quant_mode][n_kv][head_size][window_size][n][b].keys():
                                    if n_kv == 0:
                                        n_kv_local = n
                                    else:
                                        n_kv_local = n_kv
                                    sol = self.query_generation_attention(b, s, n, n_kv_local, quant_mode, sol_mode=common.SOLMode.SOL, window_size=window_size, head_size=head_size)
                                    if sol > self._generation_attention_data[quant_mode][n_kv][head_size][window_size][n][b][s]:
                                        logger.debug('generation attention quant {} n{} n_kv{} b{} s{}: sol {} > perf_db {}'.format(quant_mode, n, n_kv_local, b, s, window_size, sol, self._generation_attention_data[quant_mode][n_kv][head_size][window_size][n][b][s]))
                                        self._generation_attention_data[quant_mode][n_kv][head_size][window_size][n][b][s] = sol
                
    def query_mlp(self, 
                  num_tokens: int,
                  hidden_size: int,
                  intermediate_size: int,
                  quant_mode: common.MoEQuantMode,
                  is_context: bool = True,
                  sol_mode: Optional[common.SOLMode] = None) -> float:
        """
        Query the SGLang MLP data for DeepSeek shared expert operations
        """
        def get_sol(num_tokens: int, hidden_size: int, intermediate_size: int, quant_mode: common.MoEQuantMode) -> Tuple[float, float, float]:
            ops = 2 * num_tokens * hidden_size * intermediate_size * 3
            mem_bytes = quant_mode.value.memory * (
                num_tokens * hidden_size * 3 +  # input + output + intermediate
                num_tokens * intermediate_size * 3 +  # intermediate
                hidden_size * intermediate_size * 3  # weights for up + down projections
            )
            sol_math = ops / (self.system_spec['gpu']['float16_tc_flops'] * quant_mode.value.compute) * 1000
            sol_mem = mem_bytes / self.system_spec['gpu']['mem_bw'] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem
        
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(num_tokens, hidden_size, intermediate_size, quant_mode)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(num_tokens, hidden_size, intermediate_size, quant_mode)
        else:
            # Query the actual performance data
            if is_context:
                mlp_data = self._context_mlp_data
            else:
                mlp_data = self._generation_mlp_data

            mlp_dict = mlp_data[quant_mode][hidden_size][intermediate_size]
            num_left, num_right = self._nearest_1d_point_helper(num_tokens, list(mlp_dict.keys()), inner_only=False)
            lat = self._interp_1d([num_left, num_right], [mlp_dict[num_left], mlp_dict[num_right]], num_tokens)
            return lat
    
    def query_deepep_ll(self, 
                        node_num: int,
                        num_tokens: int,
                        num_experts: int,
                        topk: int,
                        hidden_size: int,
                        sol_mode: Optional[common.SOLMode] = None) -> float:
        """
        Query the DeepEP LL operation data
        """
        def get_sol(num_tokens: int, topk: int, num_experts: int) -> Tuple[float, float, float]:
            pass
            return 
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(num_tokens, topk, num_experts)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(num_tokens, topk, num_experts)
        else:
            data = self._deepep_ll_data[node_num][hidden_size][topk][num_experts]
            num_left, num_right = self._nearest_1d_point_helper(num_tokens, list(data.keys()), inner_only=False)
            lat = self._interp_1d([num_left, num_right], [data[num_left], data[num_right]], num_tokens)
            return lat / 1000.0 
                
    def query_deepep_normal(self, 
                           node_num: int,
                           num_tokens: int,
                           num_experts: int,
                           topk: int,
                           hidden_size: int,
                           sms: int,
                           sol_mode: Optional[common.SOLMode] = None) -> float:
        """
        Query the DeepEP normal operation data
        """
        def get_sol(num_tokens: int, num_experts: int, topk: int, hidden_size: int) -> Tuple[float, float, float]:
            pass
            return 
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(num_tokens, num_experts, topk, hidden_size)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(num_tokens, num_experts, topk, hidden_size)
        else:
            if node_num == 1 and sms == 20: # only collect sm=20 for now
                data = self._deepep_normal_data[node_num][hidden_size][topk][num_experts][sms]
                num_left, num_right = self._nearest_1d_point_helper(num_tokens, list(data.keys()), inner_only=False)
                lat = self._interp_1d([num_left, num_right], [data[num_left], data[num_right]], num_tokens)
            else:
                data = self._deepep_normal_data[node_num][hidden_size][topk][num_experts]
                lat = self._interp_2d_linear(sms, num_tokens, data)
            return lat / 1000.0 
        
if __name__ == '__main__':
    database_dict = get_all_databases()