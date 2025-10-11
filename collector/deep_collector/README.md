Guidance for collecting deepep data in normal and low-latency modes.

Notes:
- MASTER_ADDR: IP address of the node with RANK=0.
- WORLD_SIZE: total number of nodes.
- RANK: 0-based index for this node.
- {num_node}: total number of nodes (e.g., 2 or 4).
- xxx: GPU type/model (e.g., A100, H100).

# Build Docker

Note: The test files under `deep_collector/` are sourced from [DeepEP](https://github.com/deepseek-ai/DeepEP/tree/main/tests) with some modifications applied.

```bashzai
docker build -t deepep:latest -f docker/Dockerfile.deepep .
docker run -it --network host --gpus all  -v aiconfigurator/collector/deep_collector:/new_workspace --privileged deepep:latest bash
```

# Two-node configuration

Server:
```bash
export MASTER_ADDR=10.6.131.20
export WORLD_SIZE=2
export MASTER_PORT=40303
export RANK=0
```

Client:
```bash
export MASTER_ADDR=10.6.131.20
export WORLD_SIZE=2
export MASTER_PORT=40303
export RANK=1
```

# Four-node configuration

Server:
```bash
export MASTER_ADDR=10.6.131.13
export WORLD_SIZE=4
export MASTER_PORT=40303
export RANK=0
```

Client:
```bash
export MASTER_ADDR=10.6.131.13
export WORLD_SIZE=4
export MASTER_PORT=40303
export RANK=1

export MASTER_ADDR=10.6.131.13
export WORLD_SIZE=4
export MASTER_PORT=40303
export RANK=2

export MASTER_ADDR=10.6.131.13
export WORLD_SIZE=4
export MASTER_PORT=40303
export RANK=3
```

# Test intra-node mode

Run the following command on a single node:
```bash
python /new_workspace/test_intranode.py |& tee deepep_node_1_mode_normal.log
```

# Test inter-node normal mode

On the first node:
Note: Replace {num_node} with the total number of nodes (e.g., 2 or 4).
```bash
python /new_workspace/test_internode.py |& tee deepep_node_{num_node}_mode_normal.log
```
On the other node(s):
```bash
python /new_workspace/test_internode.py
```

# Test low-latency mode

On the first node:
Note: Replace {num_node} with the total number of nodes (e.g., 2 or 4).
```bash
python /new_workspace/test_internode.py  --test-ll-compatibility |& tee deepep_node_{num_node}_mode_ll.log
```
On the other node(s):
```bash
python /new_workspace/test_internode.py  --test-ll-compatibility
```

# Post-process log files
Save the processed deepep data under path path/to/aiconfigurator/src/aiconfigurator/systems/data/xxx/sglang/0.5.0/.
Replace xxx with the GPU type (e.g., A100). Point --log-dir to that directory.
```bash
python aiconfigurator/collector/deep_collector/extract_data.py --log-dir path/to/aiconfigurator/src/aiconfigurator/systems/data/xxx/sglang/0.5.0/
```