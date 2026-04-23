# oneCCL Communication Benchmarking for Intel XPU

This guide explains how to set up and run oneCCL collective communication benchmarks on Intel XPU (GPU) devices using `collect_oneccl_xpu.py` or `collect_comm.sh --device xpu`.

## Prerequisites

### 1. Intel oneAPI Packages

The following Intel oneAPI components must be installed:

- **oneCCL** (`intel-oneapi-ccl-devel`) — collective communications library and headers
- **Intel MPI** (`intel-oneapi-mpi`) — MPI runtime (`mpirun`)
- **Intel DPC++ Compiler** (`intel-oneapi-compiler-dpcpp-cpp`) — needed to compile the benchmark binary (`icpx`)

Verify installation:

```bash
dpkg -l | grep -i "intel-oneapi-ccl-devel\|intel-oneapi-mpi\|intel-oneapi-compiler"
```

### 2. Intel GPU Devices

At least 2 Intel GPU (XPU) devices must be available. Verify with:

```bash
sycl-ls | grep "level_zero:gpu"
```

### 3. Compile the oneCCL Benchmark Binary

The oneCCL package ships benchmark source code but **no pre-compiled binary**. You must compile it once.

```bash
# Source the oneAPI environment
source /opt/intel/oneapi/ccl/latest/env/vars.sh

# Create a temporary build directory
mkdir -p /tmp/oneccl_benchmark_build
cp -r /opt/intel/oneapi/ccl/latest/share/doc/ccl/examples/benchmark/* /tmp/oneccl_benchmark_build/
cd /tmp/oneccl_benchmark_build

# Compile with SYCL (GPU) support
icpx -std=c++17 -fsycl \
  -I./include -I./src \
  -I/opt/intel/oneapi/ccl/latest/share/doc/ccl/examples/include \
  -I/opt/intel/oneapi/ccl/latest/include \
  -I/opt/intel/oneapi/mpi/latest/include \
  -L/opt/intel/oneapi/ccl/latest/lib \
  -L/opt/intel/oneapi/mpi/latest/lib/release \
  -L/opt/intel/oneapi/mpi/latest/lib \
  src/benchmark.cpp \
  -lccl -lmpi -lpthread -lrt -lm -ldl \
  -o benchmark

# Install to a location on PATH
cp benchmark /usr/local/bin/oneccl_benchmark
```

Verify it works:

```bash
oneccl_benchmark --help
```

### 4. Environment Variables

The following environment variables must be set at runtime (the script sets them automatically):

| Variable | Value | Purpose |
|----------|-------|---------|
| `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK` | `0` | Avoids topology detection errors on PCIe-connected GPUs |
| `FI_PROVIDER` | `tcp` | Uses TCP fabric provider (avoids UCX assertion failures) |
| `I_MPI_OFI_PROVIDER` | `tcp` | Forces Intel MPI to use TCP OFI provider |

## Usage

### Option A: Run All Benchmarks via `collect_comm.sh`

This runs all collective operations (`all_gather`, `alltoall`, `reduce_scatter`, `all_reduce`) with both `half` and `int8` data types across all detected GPU counts:

```bash
cd collector/
bash collect_comm.sh --device xpu
```

### Option B: Run Individual Operations via `collect_oneccl_xpu.py`

```bash
cd collector/

# all_gather with 2 GPUs, half precision, default range (512B to 512MB)
python collect_oneccl_xpu.py --oneccl_op all_gather --dtype half --num_gpus 2

# reduce_scatter with 2 GPUs
python collect_oneccl_xpu.py --oneccl_op reduce_scatter --dtype half --num_gpus 2

# all_reduce with 4 GPUs, custom range
python collect_oneccl_xpu.py --oneccl_op all_reduce --dtype half --num_gpus 4 --range "1024,268435456,2"

# alltoall with int8
python collect_oneccl_xpu.py --oneccl_op alltoall --dtype int8 --num_gpus 4
```

### CLI Options for `collect_oneccl_xpu.py`

| Option | Default | Description |
|--------|---------|-------------|
| `--oneccl_op`, `-O` | `all_gather` | Collective operation: `all_gather`, `alltoall`, `reduce_scatter`, `all_reduce` |
| `--dtype`, `-t` | `half` | Data type: `half` (bf16, 2 bytes), `int8` (1 byte) |
| `--range`, `-r` | `512,536870913,2` | `min_bytes,max_bytes,multiplicative_ratio` |
| `--num_gpus`, `-n` | `2` | Number of GPUs (MPI ranks) |
| `--iters`, `-i` | `100` | Benchmark iterations per message size |
| `--warmup_iters`, `-w` | `20` | Warmup iterations per message size |

## Output

Results are appended to `oneccl_perf.txt` in the working directory with the following CSV format:

```
framework,version,device,op_name,kernel_source,nccl_dtype,num_gpus,message_size,latency
oneCCL,2021.17.2-5,Intel(R) Graphics [0xe211],all_gather,oneCCL,half,2,256,0.015538
```

## Troubleshooting

- **`oneCCL benchmark binary not found`** — Compile and install the benchmark binary (see step 3 above).
- **`BAD TERMINATION ... KILLED BY SIGNAL: 6`** — Usually a UCX assertion failure. Ensure `FI_PROVIDER=tcp` is set.
- **`topology recognition shows PCIe connection`** — Set `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`.
- **Only 1 GPU detected** — The script requires at least 2 GPUs. Check `sycl-ls` output.
