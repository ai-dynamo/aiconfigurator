"""Tag a layerwise CSV with max_num_seqs=128 and max_num_batched_tokens=2048.

The golden tp4_ep4 eval keys the layerwise DB on max_num_batched_tokens for CTX
and max_num_seqs for GEN. batch<=16 latencies are independent of the scheduler
cap, so re-stamping the collected rows to the golden key is valid.
"""
import sys
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
df["max_num_seqs"] = 128
df["max_num_batched_tokens"] = 2048
df.to_csv(dst, index=False)
print(f"tagged {len(df)} rows -> {dst}")
print(df.groupby("phase").size().to_dict())
