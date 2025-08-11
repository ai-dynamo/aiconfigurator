# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase

class Operation(object):
    """
    Base operation class.
    """
    def __init__(self, name: str, scale_factor: float) -> None:
        self._name = name
        self._scale_factor = scale_factor
    def query(self, database:PerfDatabase, **kwargs):
        raise NotImplementedError
    def get_weights(self, **kwargs):
        raise NotImplementedError

class AllReduce(Operation):
    """
    AllReduce operation. Now it's mapped to only trtllm custom allreduce.
    """
    def __init__(self, name: str, scale_factor: float, h: int, tp_size: int) -> None:
        super().__init__(name, scale_factor)
        self._h = h
        self._tp_size = tp_size
        self._weights = 0.0
        
    def query(self, database:PerfDatabase, **kwargs):
        if self._tp_size == 1:
            return 0.0
        # count, not size in bytes
        size = kwargs.get('x') * self._h

        return database.query_allreduce(common.CommQuantMode.half, self._tp_size, size)*self._scale_factor

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class P2P(Operation):
    """
    P2P operation.
    """
    def __init__(self, name: str, scale_factor: float, h: int, pp_size: int) -> None:
        super().__init__(name, scale_factor)
        self._h = h
        self._pp_size = pp_size
        self._bytes_per_element = 2
        #self._empirical_scaling_factor = 1.1
        self._weights = 0.0
  
    def query(self, database:PerfDatabase, **kwargs):
        if self._pp_size == 1:
            return 0.0

        size = kwargs.get('x') * self._h
        p2p_bytes = size * 2

        return database.query_p2p(p2p_bytes) * self._scale_factor
   
    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

class NCCL(Operation):
    """
    NCCL operation.
    """
    def __init__(self, name: str, scale_factor: float, nccl_op: str, num_elements_per_token: int, num_gpus: int, comm_quant_mode: common.CommQuantMode) -> None:
        super().__init__(name, scale_factor)
        self._nccl_op = nccl_op
        self._num_elements_per_token = num_elements_per_token
        self._num_gpus = num_gpus
        self._comm_quant_mode = comm_quant_mode
        self._weights = 0.0
  
    def query(self, database:PerfDatabase, **kwargs):
        message_size = kwargs.get('x') * self._num_elements_per_token

        return database.query_nccl(self._comm_quant_mode, self._num_gpus, self._nccl_op, message_size) * self._scale_factor
   
    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

class GEMM(Operation):
    """
    GEMM operation.
    """
    def __init__(self, name: str, scale_factor: float, n: int, k: int, quant_mode: common.GEMMQuantMode) -> None:
        super().__init__(name, scale_factor)
        self._n = n
        self._k = k
        self._quant_mode = quant_mode
        self._weights = self._n*self._k*quant_mode.value.memory 
    def query(self, database:PerfDatabase, **kwargs):
        x = kwargs.get('x')
        overwrite_quant_mode = kwargs.get('quant_mode', None)
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode
            
        return database.query_gemm(x, self._n, self._k, quant_mode)*self._scale_factor
    
    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor     

class MoE(Operation):
    """
    MoE operation.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 hidden_size: int, 
                 inter_size: int, 
                 topk: int, 
                 num_experts: int, 
                 moe_tp_size: int, 
                 moe_ep_size: int, 
                 quant_mode: common.MoEQuantMode, 
                 workload_distribution: str, 
                 attention_dp_size: int) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._quant_mode = quant_mode
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._workload_distribution = workload_distribution
        self._weights = self._hidden_size*self._inter_size*self._num_experts*quant_mode.value.memory*3 // self._moe_ep_size // self._moe_tp_size # 3 for ffn1,gate,ffn2; 2 for float16
    def query(self, database:PerfDatabase, **kwargs):
        # attention dp size will scale up the total input tokens. 
        x = kwargs.get('x') * self._attention_dp_size
        overwrite_quant_mode = kwargs.get('quant_mode', None)
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode
        return database.query_moe(num_tokens=x, 
                                 hidden_size=self._hidden_size, 
                                 inter_size=self._inter_size, 
                                 topk=self._topk, 
                                 num_experts=self._num_experts,
                                 moe_tp_size=self._moe_tp_size,
                                 moe_ep_size=self._moe_ep_size, 
                                 quant_mode=quant_mode, 
                                 workload_distribution=self._workload_distribution)*self._scale_factor

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

# a comm op to deduce the communication cost of MoE
class MoEDispatch(Operation):
    """
    MoE dispatch operation. For fine grained moe dispatch
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 hidden_size: int, 
                 topk: int, 
                 num_experts: int, 
                 moe_tp_size: int, 
                 moe_ep_size: int, 
                 attention_dp_size: int,
                 pre_dispatch: bool,
                 enable_fp4_all2all: bool = True) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._weights = 0.0
        self._enable_fp4_all2all = enable_fp4_all2all
        self._pre_dispatch = pre_dispatch
        self.num_gpus = self._moe_ep_size*self._moe_tp_size
        self._attention_tp_size = moe_tp_size*moe_ep_size // self._attention_dp_size
        
        
    def query(self, database:PerfDatabase, **kwargs):
        num_tokens = kwargs.get('x')
        volume = num_tokens * self._hidden_size
        _sm_version = database.system_spec['gpu']['sm_version']

        if database.backend == common.BackendName.trtllm.value:
            assert (self._attention_tp_size == 1 or self._attention_dp_size ==1), "trtllm does not support TP>1 and DP>1 for attn simultaneously"
            if _sm_version == 100:
                if self._pre_dispatch:
                    if self._attention_tp_size > 1: #tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        if self._enable_fp4_all2all:
                            # Calculate all2all communication volume for nvfp4 all2all operation
                            # Volume calculation considers the average case between best and worst scenarios:
                            # - Best case: volume * 1/4 (all selected experts are in one GPU for all tokens)
                            # - Worst case: volume * min(topk, attention_dp_size)/4 (every selected expert is in different GPU)
                            # - Final volume: average of best and worst cases, divided by 4 for nvfp4 quantization
                            all2all_volume = volume * (1 + min(self._topk, self._attention_dp_size))/2 / 4 # mean of best and worst
                            # to do: nvfp4 custom all2all
                            all2all_latency = database.query_nccl(common.CommQuantMode.half, self.num_gpus, 'alltoall', all2all_volume)
                            all2all_sf_latency = database.query_nccl(common.CommQuantMode.half, self.num_gpus, 'alltoall', all2all_volume/8) # volume_scale_factor = 1/8 volume
                            comm_latency = all2all_latency + all2all_sf_latency + 1e-2 # msg size static latency 10us
                        else:
                            all_gather_volume = volume * self._attention_dp_size / 4
                            all_gather_latency = database.query_nccl(common.CommQuantMode.half, self.num_gpus, 'all_gather', all_gather_volume) # nvfp4 allgather
                            all_gather_sf_latency = database.query_nccl(common.CommQuantMode.half, self.num_gpus, 'all_gather', all_gather_volume/8) # volume_scale_factor = 1/8 volume
                            comm_latency = all_gather_latency + all_gather_sf_latency
                    else:
                        comm_latency = 0
                else:
                    if self._attention_tp_size > 1: #tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        if self._enable_fp4_all2all:
                            # to do: nvfp4 all2all
                            all2all_volume = volume * (1 + min(self._topk, self._attention_dp_size))/2 / 4 #nvfp4 all2all
                            comm_latency = database.query_nccl(common.CommQuantMode.half, self.num_gpus, 'alltoall', all2all_volume)
                        else:
                            comm_latency = database.query_nccl(common.CommQuantMode.half, self.num_gpus, 'reduce_scatter', volume * self._attention_dp_size)
                    else:
                        comm_latency = 0
            else: # sm < 100 or > 100 (for now)
                if self._pre_dispatch:
                    if self._attention_tp_size > 1: #tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(common.CommQuantMode.half, self.num_gpus, 'all_gather', volume * self._attention_dp_size)
                    else:
                        comm_latency = 0
                else:
                    if self._attention_tp_size > 1: #tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(common.CommQuantMode.half, self.num_gpus, 'reduce_scatter', volume * self._attention_dp_size)
                    else:
                        comm_latency = 0
        elif database.backend == common.BackendName.vllm.value:
            raise NotImplementedError("Need to implement MoE dispatch for vllm")
        else: #sglang
            raise NotImplementedError("Need to implement MoE dispatch for sglang")
        
        return comm_latency * self._scale_factor

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

    def query_ideal(self, database:PerfDatabase, **kwargs):
        """
        Ideal communication cost for MoE dispatch. For reference only.
        """
        num_tokens = kwargs.get('x')
        volume = num_tokens * self._hidden_size

        if self._pre_dispatch:    
            reduce_scatter1_v = volume / self.num_gpus
            reduce_scatter1_num_gpus = self._attention_tp_size

            all2all1_v = volume * self._topk / self.num_gpus
            all2all1_num_gpus = self.num_gpus

            allgather1_v = volume / self._moe_tp_size
            allgather1_num_gpus = self._moe_tp_size

            comm_latency = database.query_nccl(common.CommQuantMode.half, reduce_scatter1_num_gpus, 'reduce_scatter', reduce_scatter1_v) + \
                            database.query_nccl(common.CommQuantMode.half, all2all1_num_gpus, 'alltoall', all2all1_v) + \
                            database.query_nccl(common.CommQuantMode.half, allgather1_num_gpus, 'all_gather', allgather1_v) 
        else:
            reduce_scatter2_v = volume
            reduce_scatter2_num_gpus = self._moe_tp_size

            all2all2_v = volume * self._topk / self.num_gpus
            all2all2_num_gpus = self.num_gpus

            allgather2_v = volume / self.num_gpus
            allgather2_num_gpus = self._attention_tp_size

            comm_latency = database.query_nccl(common.CommQuantMode.half, reduce_scatter2_num_gpus, 'reduce_scatter', reduce_scatter2_v) + \
                            database.query_nccl(common.CommQuantMode.half, all2all2_num_gpus, 'alltoall', all2all2_v) + \
                            database.query_nccl(common.CommQuantMode.half, allgather2_num_gpus, 'all_gather', allgather2_v)

        return comm_latency * self._scale_factor

class ContextAttention(Operation):
    """
    Context attention operation.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 n: int, 
                 n_kv: int, 
                 kvcache_quant_mode: common.KVCacheQuantMode, 
                 fmha_quant_mode: common.FMHAQuantMode) -> None:
        super().__init__(name, scale_factor)
        self._n = n
        self._weights = 0.0
        self._n_kv = n_kv
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode

    def query(self, database:PerfDatabase, **kwargs):
        batch_size = kwargs.get('batch_size')
        isl = kwargs.get('s')
        return database.query_context_attention(batch_size, isl, self._n, self._n_kv, self._kvcache_quant_mode, self._fmha_quant_mode)*self._scale_factor
    
    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
class GenerationAttention(Operation):
    """
    Generation attention operation.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 n: int, 
                 n_kv: int, 
                 kv_cache_dtype: common.KVCacheQuantMode) -> None:
        super().__init__(name, scale_factor)
        self._n = n
        self._weights = 0.0
        self._n_kv = n_kv
        self._kv_cache_dtype = kv_cache_dtype

    def query(self, database:PerfDatabase, **kwargs):
        beam_width = kwargs.get('beam_width')
        assert(beam_width == 1), "only support beam_width=1"
        batch_size = kwargs.get('batch_size')
        s = kwargs.get('s')
        return database.query_generation_attention(batch_size, s, self._n, self._n_kv, self._kv_cache_dtype)*self._scale_factor
 
    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

class ContextMLA(Operation):
    """
    Context MLA operation. now only contains MHA part.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 tp_size: int, 
                 kvcache_quant_mode: common.KVCacheQuantMode, fmha_quant_mode: common.FMHAQuantMode) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._weights = 0. #2*(1536*24576/tp_size + 128/tp_size*512*128+128/tp_size*512*128) # up q, up k, up v  float16 # 104MB / tpsize per layer
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode

    def query(self, database:PerfDatabase, **kwargs):
        batch_size = kwargs.get('batch_size')
        isl = kwargs.get('s')
        return database.query_context_mla(batch_size, isl, self._tp_size, self._kvcache_quant_mode, self._fmha_quant_mode)*self._scale_factor
    
    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

class GenerationMLA(Operation):
    """
    Generation MLA operation. now only contains MQA part.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 tp_size: int, 
                 kv_cache_dtype: common.KVCacheQuantMode) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._weights = 0. # 2*(1536*24576/tp_size + 128/tp_size*512*128+128/tp_size*512*128) # up q, up k, v up  float16
        self._kv_cache_dtype = kv_cache_dtype

    def query(self, database:PerfDatabase, **kwargs):
        beam_width = kwargs.get('beam_width')
        assert(beam_width == 1), "only support beam_width=1"
        batch_size = kwargs.get('batch_size')
        s = kwargs.get('s')
        return database.query_generation_mla(batch_size, s, self._tp_size, self._kv_cache_dtype)*self._scale_factor
 
    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

class MLABmm(Operation):
    """
    MLABmm operation. consider to be contained by mla op. for now, keep it as a separate op to show the cost of bmm
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 num_heads: int, 
                 quant_mode: common.GEMMQuantMode, 
                 if_pre: bool=True) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._weights = 0. 
        self._quant_mode = quant_mode
        self._if_pre = if_pre

    def query(self, database:PerfDatabase, **kwargs):
        beam_width = kwargs.get('beam_width')
        assert(beam_width == 1), "only support beam_width=1"
        batch_size = kwargs.get('batch_size')
        return database.query_mla_bmm(batch_size, self._num_heads, self._quant_mode, self._if_pre)*self._scale_factor
 
    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
    
class Embedding(Operation):
    """
    Embedding operation.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 row_size: int, 
                 column_size: int, 
                 empirical_bw_scaling_factor: float=0.3) -> None:
        super().__init__(name, scale_factor)
        self._row_size = row_size
        self._column_size = column_size
        self._weights = row_size * column_size * 2
        self._empirical_bw_scaling_factor = empirical_bw_scaling_factor
        self._constant_latency = 5e-6 # 5us

    #sol only
    def query(self, database: PerfDatabase, **kwargs):
        x = kwargs.get('x')
        d2d_bytes = x * self._column_size * 2
        
        return database.query_mem_op(d2d_bytes) * self._scale_factor
  
    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
    
class ElementWise(Operation):
    """
    Element-wise operation.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 dim_in: int, 
                 dim_out: int, 
                 empirical_bw_scaling_factor: float=0.8) -> None:
        super().__init__(name, scale_factor)
        self._weights = 0.
        self._empirical_bw_scaling_factor = empirical_bw_scaling_factor
        self._constant_latency = 5e-6 # 5us
        self._dim_in = dim_in
        self._dim_out = dim_out

    #sol only
    def query(self, database: PerfDatabase, **kwargs):
        x = kwargs.get('x') # num tokens
        read_bytes = x * self._dim_in * 2 # fp16 for act
        write_bytes = x * self._dim_out * 2
        
        return database.query_mem_op(read_bytes + write_bytes) * self._scale_factor

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor