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
                database = PerfDatabase(system, backend, version, systems_dir)
                databases_cache[system][backend][version] = database
            else:
                database = None
                logger.error(f"data path {data_path} not found")
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
        lines = f.readlines()
    
    for line in lines:
        dtype, tp_size, message_size, allreduce_strategy, layer_name, latency = line.split(',')
        message_size  = int(message_size)
        latency = float(latency)
        tp_size = int(tp_size)
        dtype = common.CommQuantMode.half # TODO

        try:
            latency = custom_allreduce_data[dtype][tp_size][allreduce_strategy][message_size]
            logger.debug('value conflict in custom allreduce data: {} {} {} {} {} {}'.format(dtype, tp_size, allreduce_strategy, message_size, latency))
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


def load_context_attention_data(context_attention_file):
    """
    Load the context attention data
    """
    if not os.path.exists(context_attention_file):
        logger.warning(f"Context attention data file {context_attention_file} not found.")
        return None
    context_attention_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict())))))
    with open(context_attention_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    for row in rows:
        quant_mode, kv_cache_dtype, b, s, n, kv_n, latency = \
            row['attn_dtype'], row['kv_cache_dtype'], row['batch_size'], row['isl'], row['num_heads'], row['num_key_value_heads'], row['latency']
        b=int(b)
        s=int(s)
        n=int(n)
        kv_n=int(kv_n)
        latency=float(latency)
        
        # we only have kv_n==n(MHA) and kv_n==1,2,4,8(XQA), interp/extrap all other num_kv_heads
        kv_n = 0 if n == kv_n else kv_n

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            latency = context_attention_data[quant_mode][kv_cache_dtype][kv_n][n][s][b]
            logger.debug('value conflict in context attention data: {} {} {} {} {} {}'.format(quant_mode, kv_cache_dtype, kv_n, n, s, b))
        except KeyError:
            context_attention_data[quant_mode][kv_cache_dtype][kv_n][n][s][b] = latency

    return context_attention_data


def load_generation_attention_data(generation_attention_file):
    """
    Load the generation attention data
    """
    if not os.path.exists(generation_attention_file):
        logger.warning(f"Generation attention data file {generation_attention_file} not found.")
        return None
    generation_attention_data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict()))))
    with open(generation_attention_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    for row in rows:
        quant_mode, kv_cache_dtype, b, s, n, kv_n, step, latency = \
            row['attn_dtype'], row['kv_cache_dtype'], row['batch_size'], row['isl'], row['num_heads'], row['num_key_value_heads'], row['step'], row['latency']
        b=int(b)
        s=int(s)
        n=int(n)
        kv_n=int(kv_n)
        step = int(step)
        latency=float(latency)

        # we only have kv_n==n(MHA) and kv_n==1,2,4,8(XQA), interp/extrap all other num_kv_heads
        kv_n = 0 if n == kv_n else kv_n
        s = s + step

        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            latency = generation_attention_data[kv_cache_dtype][kv_n][n][b][s]
            logger.debug('value conflict in generation attention data: {} {} {} {} {}'.format(kv_cache_dtype, kv_n, n, b, s))
        except KeyError:
            generation_attention_data[kv_cache_dtype][kv_n][n][b][s] = latency
        
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
        quant_mode, kv_cache_dtype, b, s, tp_size, latency = \
            row['mla_dtype'], row['kv_cache_dtype'], row['batch_size'], row['isl'], row['tp_size'], row['latency']
        b=int(b)
        s=int(s)
        tp_size=int(tp_size)
        latency=float(latency)

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]
        
        try:
            latency = context_mla_data[quant_mode][kv_cache_dtype][tp_size][s][b]
            logger.debug('value conflict in context mla data: {} {} {} {} {} {}'.format(quant_mode, kv_cache_dtype, tp_size, s, b, latency))
        except KeyError:
            context_mla_data[quant_mode][kv_cache_dtype][tp_size][s][b] = latency
        
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
        quant_mode, kv_cache_dtype, b, s, tp_size, step, latency = \
            row['mla_dtype'], row['kv_cache_dtype'], row['batch_size'], row['isl'], row['tp_size'], row['step'], row['latency']
        b=int(b)
        s=int(s)
        step=int(step)
        tp_size=int(tp_size)
        latency=float(latency)

        s = s + step
        
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            latency = generation_mla_data[kv_cache_dtype][tp_size][b][s]
            logger.debug('value conflict in generation mla data: {} {} {} {} {} '.format(kv_cache_dtype, tp_size, b, s, latency))      
        except KeyError:
            generation_mla_data[kv_cache_dtype][tp_size][b][s] = latency
    
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
                    data_dict=self._context_attention_data[quant_mode][kv_cache_dtype][num_kv_heads]
                    min_x = min(data_dict.keys())
                    target_x_list=[4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 72, 96, 128] # n
                    # currently, support max seq to 1M. Because all the system is linear for now. it will be difficult to do square interpolation. Use more points to do the approximation
                    target_y_list=[16,32,64,128,256,512,1024,2048] + [4096+i*2048 for i in range(14)] + \
                        [32768 + 16384*i for i in range(6)] + [131072 + 32768*i for i in range(12)] + [524288 + 65536*i for i in range(9)]# s
                    target_z_list=[1,2,4,8,16,32,64,128,256,384,512,1024,2048] # b

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
                target_x_list=[4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 72, 96, 128] # n
                target_y_list=[1,2,4,8,16,32,64,128,256,384,512,1024,2048,8192] # b
                target_z_list=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,2097152*8] # s
                data_dict = self._generation_attention_data[kv_cache_dtype][num_kv_heads]
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
        for quant_mode in self._context_mla_data.keys():
            for kv_cache_dtype in self._context_mla_data[quant_mode].keys():
                tp_list =  list(self._context_mla_data[quant_mode][kv_cache_dtype].keys())
                data_dict=self._context_mla_data[quant_mode][kv_cache_dtype]
                target_x_list=tp_list # to reuse x dim
                # currently, support max seq to 1M. Because all the system is linear for now. it will be difficult to do square interpolation. Use more points to do the approximation
                target_y_list=[16,32,64,128,256,512,1024,2048] + [4096+i*2048 for i in range(14)] + \
                    [32768 + 16384*i for i in range(6)] + [131072 + 32768*i for i in range(12)] + [524288 + 65536*i for i in range(9)]# s
                target_z_list=[1,2,4,8,16,32,64,128,256,384,512,1024,2048] # b

                self._extrapolate_data_grid(data_dict=data_dict, #tpsize,sb
                                            target_x_list=target_x_list,
                                            target_y_list=target_y_list,
                                            target_z_list=target_z_list, sqrt_y_value=True)

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
                        value = self._interp_1d([z_left, z_right], [data_dict[x][y][z_left],data_dict[x][y][z_right]], z)
                        z_dict[z] = value
            
            # y_direction
            for y in target_y_list:
                if y not in data_dict[x].keys():
                    y_left, y_right = self._nearest_1d_point_helper(y, list(data_dict[x].keys()), False)
                    z_list = sorted(list(data_dict[x][y_left].keys()))
                    for z in z_list:
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
                for y in sorted(data_dict[x_left].keys()):
                    for z in sorted(data_dict[x_left][y].keys()):
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
        model = np.polyfit(x, [y0,y1], 1)
        
        return self._validate(np.polyval(model, value))

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
                                sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the context attention data
        """
        def get_sol(b : int, s : int, n : int, n_kv : int, kvcache_quant_mode : common.KVCacheQuantMode, fmha_quant_mode : common.FMHAQuantMode) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            ops = 2 * b * s * s * n * 128 * 2 / 2 # 2 for fma, 2 for q*k^t+*v, 2 for causality.
            mem_bytes = 2 * b * (n*s*128 + 2*n_kv*s*128 + n*s*128) # 2 for fp16 TODO
            sol_math = ops / self.system_spec['gpu']['float16_tc_flops'] * 1000 / fmha_quant_mode.value.compute
            sol_mem = mem_bytes / self.system_spec['gpu']['mem_bw'] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem
        assert(n_kv <= n), "n_kv must be less than or equal to n"

        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(b, s, n, n_kv, kvcache_quant_mode, fmha_quant_mode)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(b, s, n, n_kv, kvcache_quant_mode, fmha_quant_mode)
        else:
            if n_kv == n:
                attention_dict = self._context_attention_data[fmha_quant_mode][kvcache_quant_mode][0]
            else:
                attention_dict = self._context_attention_data[fmha_quant_mode][kvcache_quant_mode][n_kv]

            latency = self._interp_3d(n, s, b, attention_dict, 'cubic')
            return latency
    
    def query_generation_attention(self, 
                                   b : int, 
                                   s : int, 
                                   n : int, 
                                   n_kv : int, 
                                   kvcache_quant_mode : common.KVCacheQuantMode, 
                                   sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the generation attention data
        """
        def get_sol(b : int, s : int, n : int, n_kv : int, kvcache_quant_mode : common.KVCacheQuantMode) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            # only consider fp16 mmha
            ops = 2 * b * n * 128 * 2 # 2 for fma, 2 for q*k^t+*v
            # kvcache load bytes will depend on kvcache quant. while input q and output might be in fp16.
            mem_bytes = b * (n*128*2 + 2*n_kv*(s-1)*128*kvcache_quant_mode.value.memory + n*128*2)
            
            sol_math = ops / self.system_spec['gpu']['float16_tc_flops'] * 1000
            sol_mem = mem_bytes / self.system_spec['gpu']['mem_bw'] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem
        assert(n_kv <= n), "n_kv must be less than or equal to n"

        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(b, s, n, n_kv, kvcache_quant_mode)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(b, s, n, n_kv, kvcache_quant_mode)
        else:
            if n_kv == n:
                attention_dict = self._generation_attention_data[kvcache_quant_mode][0]
            else:
                attention_dict = self._generation_attention_data[kvcache_quant_mode][n_kv]

            latency =  self._interp_3d(n, b, s, attention_dict, 'bilinear')
            return latency


    def query_context_mla(self, 
                          b : int, 
                          s : int, 
                          tp_size : int, 
                          kvcache_quant_mode : common.KVCacheQuantMode, 
                          fmha_quant_mode : common.FMHAQuantMode, 
                          sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the context mla data
        """
        def get_sol(b : int, s : int, tp_size : int, kvcache_quant_mode : common.KVCacheQuantMode, fmha_quant_mode : common.FMHAQuantMode) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            ops = b * 128 / tp_size * 2 / 2 *(s * s * 192 + s * s * 128) # 2 for fma, 2 for causality. 128/tp_size for local heads
            mem_bytes = b * 128 / tp_size * 2 * (2*s*192 + 2*s*128) # 2 for fp16, TODO
            sol_math = ops / self.system_spec['gpu']['float16_tc_flops'] * 1000 / fmha_quant_mode.value.compute
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
            mla_dict = self._context_mla_data[fmha_quant_mode][kvcache_quant_mode]
            latency = self._interp_3d(tp_size, s, b, mla_dict, 'cubic')
            return latency
    
    def query_generation_mla(self, 
                             b : int, 
                             s : int, 
                             tp_size : int, 
                             kvcache_quant_mode : common.KVCacheQuantMode, 
                             sol_mode : Optional[common.SOLMode] = None) -> float:
        """
        Query the generation mla data
        """
        def get_sol(b : int, s : int, tp_size : int, kvcache_quant_mode : common.KVCacheQuantMode) -> Tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            n = 128//tp_size
            # only consider fp16 mmha
            ops = 2 * b * n * 1088 * s # 2 for fma
            # kvcache load bytes will depend on kvcache quant. while input q and output might be in fp16.
            mem_bytes = b * (n * 1088 * 2 + (s-1)*1088 * kvcache_quant_mode.value.memory)
            
            sol_math = ops / self.system_spec['gpu']['float16_tc_flops'] * 1000 # only fp16
            sol_mem = mem_bytes / self.system_spec['gpu']['mem_bw'] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem
        
        if sol_mode is None:
            sol_mode = self._default_sol_mode
        if sol_mode == common.SOLMode.SOL:
            return get_sol(b, s, tp_size, kvcache_quant_mode)[0]
        elif sol_mode == common.SOLMode.SOL_FULL:
            return get_sol(b, s, tp_size, kvcache_quant_mode)
        else:
            mla_dict = self._generation_mla_data[kvcache_quant_mode]
            latency =  self._interp_3d(tp_size, b, s, mla_dict, 'bilinear')
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

            num_left, num_right = self._nearest_1d_point_helper(num_tokens, list(moe_dict.keys()), inner_only=False)
            lat = self._interp_1d([num_left, num_right], [moe_dict[num_left], moe_dict[num_right]], num_tokens)
            return lat

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
                for n in self._generation_attention_data[quant_mode][n_kv].keys():
                    for b in self._generation_attention_data[quant_mode][n_kv][n].keys():
                        for s in self._generation_attention_data[quant_mode][n_kv][n][b].keys():
                            if n_kv == 0:
                                n_kv_local = n
                            else:
                                n_kv_local = n_kv
                            sol = self.query_generation_attention(b, s, n, n_kv_local, quant_mode, sol_mode=common.SOLMode.SOL)
                            if sol > self._generation_attention_data[quant_mode][n_kv][n][b][s]:
                                logger.debug('generation attention quant {} n{} n_kv{} b{} s{}: sol {} > perf_db {}'.format(quant_mode, n, n_kv_local, b, s, sol, self._generation_attention_data[quant_mode][n_kv][n][b][s]))
                                self._generation_attention_data[quant_mode][n_kv][n][b][s] = sol
                
    
if __name__ == '__main__':
    database_dict = get_all_databases()