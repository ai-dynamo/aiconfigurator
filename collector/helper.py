# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
 
import fcntl
import os
from cuda import cuda

def getSMVersion():
    # Init
    err, = cuda.cuInit(0)

    # Device
    err, cuDevice = cuda.cuDeviceGet(0)

    # Get target architecture
    err, sm_major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        cuDevice)
    err, sm_minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        cuDevice)

    return sm_major * 10 + sm_minor

def log_perf(item_list: list[dict], 
             framework: str, 
             version: str, 
             device_name: str, 
             op_name: str,
             kernel_source: str,
             perf_filename: str):
    
    content_prefix = f'{framework},{version},{device_name},{op_name},{kernel_source}'
    header_prefix = 'framework,version,device,op_name,kernel_source'
    for item in item_list:
        for key, value in item.items():
            content_prefix += f',{value}'
            header_prefix += f',{key}'

    with open(perf_filename, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        if os.fstat(f.fileno()).st_size == 0:
            f.write(header_prefix + '\n')

        f.write(content_prefix + '\n')