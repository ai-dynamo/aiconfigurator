# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tensorrt as trt
import tensorrt_llm
import torch

try:
    from cuda import cudart
except:
    from cuda.bindings import runtime as cudart
from common_test_cases import get_gemm_common_test_cases
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.layers import Linear
from tensorrt_llm.quantization.layers import FP8Linear, SmoothQuantLinear, WeightOnlyQuantLinear
from tensorrt_llm.quantization.mode import QuantMode

from helper import log_perf, power_monitoring_only


def get_gemm_test_cases():
    gemm_list = ["int8_wo", "int4_wo", "sq"]
    use_plugin = True

    test_cases = []
    for gemm_common_testcase in get_gemm_common_test_cases():
        x = gemm_common_testcase.x
        n = gemm_common_testcase.n
        k = gemm_common_testcase.k
        for gemm_type in gemm_list:
            test_cases.append([gemm_type, use_plugin, x, n, k])

    return test_cases


class GEMMProfiler(trt.IProfiler):
    def __init__(self, m, n, k, device, perf_filename="gemm_perf.txt", gemm_type="float16"):
        trt.IProfiler.__init__(self)
        self._m = m
        self._n = n
        self._k = k
        self._device = device
        self._target_layers = {
            "float16": "PLUGIN_V2_Gemm_3",
            "sq": "PLUGIN_V2_SmoothQuantGemm_0",
            "int8_wo": "PLUGIN_V2_WeightOnlyQuantMatmul_0",
            "int4_wo": "PLUGIN_V2_WeightOnlyQuantMatmul_0",
            "fp8": "PLUGIN_V2_Gemm_3",
            "fp8_ootb": "myl0_1",
        }
        self._target_layer = self._target_layers[gemm_type]
        self._latency = None
        self._perf_filename = perf_filename
        self._gemm_type = gemm_type
        self._exclude_layers = {"CONSTANT", "Reformatting"}

    def report_layer_time(self, layer_name, ms):
        if self._target_layer.lower() in layer_name.lower():
            for keyword in self._exclude_layers:
                if keyword.lower() in layer_name.lower():
                    return
            self._layer_name = layer_name
            self._latency = ms

    def write_to_file(self, power_stats=None):
        log_perf(
            item_list=[
                {
                    "gemm_dtype": self._gemm_type,
                    "m": self._m,
                    "n": self._n,
                    "k": self._k,
                    "latency": self._latency,
                }
            ],
            framework="TRTLLM",
            version=tensorrt_llm.__version__,
            device_name=torch.cuda.get_device_name(self._device),
            op_name="gemm",
            kernel_source=f"trt_flow_{self._layer_name}",
            perf_filename=self._perf_filename,
            power_stats=power_stats,
        )


def run_gemm(gemm_type, use_plugin, m, n, k, device="cuda:0"):
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    # print("testing ", gemm_type, " ", m, " ", n,"  ", k)

    use_int8 = gemm_type == "int8_wo" or gemm_type == "int4_wo" or gemm_type == "sq"
    use_fp8 = gemm_type == "fp8" or gemm_type == "fp8_ootb"
    tensor_type = "float16"
    x_data = torch.randn(m, k, dtype=str_dtype_to_torch(tensor_type))

    # construct trt network
    builder = tensorrt_llm.Builder()
    builder.strongly_typed = False
    net = builder.create_network()
    # prepare plugin
    if use_plugin:
        if gemm_type == "fp8":
            net.plugin_config.gemm_plugin = "fp8"
        else:
            net.plugin_config.gemm_plugin = tensor_type
        if gemm_type == "int8_wo" or gemm_type == "int4_wo":
            net.plugin_config.weight_only_quant_matmul_plugin = tensor_type
        if gemm_type == "sq":
            net.plugin_config.set_smooth_quant_plugins(dtype=tensor_type)

    net.plugin_config.remove_input_padding = True
    with tensorrt_llm.net_guard(net):
        network = tensorrt_llm.default_trtnet()
        x = Tensor(name="x", shape=x_data.shape, dtype=tensorrt_llm.str_dtype_to_trt(tensor_type))

        # only test per-channel, per-token use static scale.
        sq_gemm_quant_mode = QuantMode.from_description(True, True, False, True)
        int8_wo_gemm_quant_mode = QuantMode.from_description(
            True, False, False, False, False, False, False, False, False
        )
        int4_wo_gemm_quant_mode = QuantMode.from_description(
            True, False, False, False, False, True, False, False, False
        )
        # fp8_gemm_quant_mode = QuantMode.from_description(
        #    False, False, False, False, False, False, False,False,True)

        gemm_list = []
        gemm_list.append(Linear(k, 12288, bias=False, dtype=tensor_type, gather_output=False, tp_size=1))
        gemm_list.append(Linear(12288, 12288, bias=False, dtype=tensor_type, gather_output=False, tp_size=1))
        gemm_list.append(Linear(12288, k, bias=False, dtype=tensor_type, gather_output=False, tp_size=1))
        gm = None
        if gemm_type == "float16":
            gm = Linear(k, n, bias=False, gather_output=False, dtype=tensor_type, tp_size=1)
        elif gemm_type == "int8_wo":
            gm = WeightOnlyQuantLinear(
                k,
                n,
                bias=False,
                gather_output=False,
                dtype=tensor_type,
                tp_size=1,
                quant_mode=int8_wo_gemm_quant_mode,
            )
        elif gemm_type == "int4_wo":
            gm = WeightOnlyQuantLinear(
                k,
                n,
                bias=False,
                gather_output=False,
                dtype=tensor_type,
                tp_size=1,
                quant_mode=int4_wo_gemm_quant_mode,
            )
        elif gemm_type == "sq":
            gm = SmoothQuantLinear(
                k,
                n,
                bias=False,
                gather_output=False,
                dtype=tensor_type,
                tp_size=1,
                quant_mode=sq_gemm_quant_mode,
            )
        elif gemm_type == "fp8" or gemm_type == "fp8_ootb":
            gm = FP8Linear(k, n, bias=False, gather_output=False, dtype=tensor_type, tp_size=1)

        inter = x
        if use_fp8 and not use_plugin:
            gemm_list = []
        for i, gemm in enumerate(gemm_list):
            inter = gemm.forward(inter)
        output = gm.forward(inter).trt_tensor
        output.name = "output"
        network.mark_output(output)

    # trt run
    build_engine = EngineFromNetwork(
        (builder.trt_builder, net.trt_network),
        CreateConfig(fp16=True, int8=use_int8, fp8=use_fp8, precision_constraints="obey", max_aux_streams=0),
    )

    profiler = GEMMProfiler(m=m, n=n, k=k, device=device, perf_filename="gemm_perf.txt", gemm_type=gemm_type)
    # l2_cache_flusher = L2CacheFlusher()
    cudart.cudaProfilerStart()
    x_data = x_data.to(torch.device(device))

    # Use power_monitoring_only context manager for TRT profiler pattern
    with power_monitoring_only(device) as power_monitor, TrtRunner(build_engine) as runner:
        runner.infer(feed_dict={"x": x_data}, check_inputs=False, copy_outputs_to_host=False)
        if use_fp8 and not use_plugin:  # additional warmup for fp8 OOTB
            for i in range(4):
                runner.infer(feed_dict={"x": x_data}, check_inputs=False, copy_outputs_to_host=False)
        runner.context.profiler = profiler
        runner.infer(feed_dict={"x": x_data}, check_inputs=False, copy_outputs_to_host=False)

    cudart.cudaProfilerStop()

    # Get power stats after monitoring completes
    power_stats = power_monitor.stop_sampling() if power_monitor else None
    profiler.write_to_file(power_stats=power_stats)
