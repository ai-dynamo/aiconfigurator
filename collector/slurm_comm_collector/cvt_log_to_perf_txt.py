# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
import os

class NCCLProfiler():
    def __init__(self, perf_filename='nccl_gb200.txt', prefix=''):
        self._prefix = prefix
        self._latency = 0.0
        self._layer_name = ''
        self._perf_filename = perf_filename

    def report_layer_time(self, layer_name, ms):
        self._layer_name = layer_name
        self._latency = ms
    
    def write_to_file(self):
        with open(self._perf_filename, 'a') as f:
            f.write(self._prefix + f',{self._layer_name},{self._latency}\n')


with open('16u64g', 'r', encoding='utf-8') as f:
    u16g64 = f.read()
    u16g64 = u16g64.split("\n")[:-1]
    u16g64 = [item + ' 64' for item in u16g64]

with open('12u48g', 'r', encoding='utf-8') as f:
    u12g48 = f.read()
    u12g48 = u12g48.split("\n")[:-1]
    u12g48 = [item + ' 48' for item in u12g48]

with open('8u32g', 'r', encoding='utf-8') as f:
    u8g32 = f.read()
    u8g32 = u8g32.split("\n")[:-1]
    u8g32 = [item + ' 32' for item in u8g32]

with open('4u16g', 'r', encoding='utf-8') as f:
    u4g16 = f.read()
    u4g16 = u4g16.split("\n")[:-1]
    u4g16 = [item + ' 16' for item in u4g16]

with open('2u8g', 'r', encoding='utf-8') as f:
    u2g8 = f.read()
    u2g8 = u2g8.split("\n")[:-1]
    u2g8 = [item + ' 8' for item in u2g8]

with open('1u4g', 'r', encoding='utf-8') as f:
    u1g4 = f.read()
    u1g4 = u1g4.split("\n")[:-1]
    u1g4 = [item + ' 4' for item in u1g4]

with open('1u2g', 'r', encoding='utf-8') as f:
    u1g2 = f.read()
    u1g2 = u1g2.split("\n")[:-1]
    u1g2 = [item + ' 2' for item in u1g2]

all_reduce_list = u1g2[:52] + u1g4[:52] + u2g8[:52] + u4g16[:52] + u8g32[:52] + u12g48[:52] + u16g64[:52]
all_to_all_list = u1g2[52:104] + u1g4[52:104] + u2g8[52:104] + u4g16[52:104] + u8g32[52:104] + u12g48[52:104] + u16g64[52:104]
reduce_scatter_list = u1g2[104:156] + u1g4[104:156] + u2g8[104:156] + u4g16[104:156] + u8g32[104:156] + u12g48[104:156] + u16g64[104:156]
all_gather_list = u1g2[156:208] + u1g4[156:208] + u2g8[156:208] + u4g16[156:208] + u8g32[156:208] + u12g48[156:208] + u16g64[156:208]

for i in range(len(all_reduce_list)):
    data_line = all_reduce_list[i].split()

    dtype = data_line[3]
    num_gpus = data_line[-1]
    size = data_line[2]
    latency = float(data_line[-5])*1e-3
    NCCL_op = "all_reduce"

    profiler = NCCLProfiler(prefix=f'{dtype},{num_gpus},{size},"NCCL"')
    profiler.report_layer_time(NCCL_op, latency)
    profiler.write_to_file()

for i in range(len(all_to_all_list)):
    data_line = all_to_all_list[i].split()

    dtype = data_line[3]
    num_gpus = data_line[-1]
    size = data_line[2]
    latency = float(data_line[-5])*1e-3
    NCCL_op = "alltoall"

    profiler = NCCLProfiler(prefix=f'{dtype},{num_gpus},{size},"NCCL"')
    profiler.report_layer_time(NCCL_op, latency)
    profiler.write_to_file()

for i in range(len(reduce_scatter_list)):
    data_line = reduce_scatter_list[i].split()

    dtype = data_line[3]
    num_gpus = data_line[-1]
    size = data_line[2]
    latency = float(data_line[-5])*1e-3
    NCCL_op = "reduce_scatter"

    profiler = NCCLProfiler(prefix=f'{dtype},{num_gpus},{size},"NCCL"')
    profiler.report_layer_time(NCCL_op, latency)
    profiler.write_to_file()

for i in range(len(all_gather_list)):
    data_line = all_gather_list[i].split()

    dtype = data_line[3]
    num_gpus = data_line[-1]
    size = data_line[2]
    latency = float(data_line[-5])*1e-3
    NCCL_op = "all_gather"

    profiler = NCCLProfiler(prefix=f'{dtype},{num_gpus},{size},"NCCL"')
    profiler.report_layer_time(NCCL_op, latency)
    profiler.write_to_file()

