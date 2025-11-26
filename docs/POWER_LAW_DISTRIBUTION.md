# Power-Law Distribution in MoE Workload Simulation

## Overview

This document describes the power-law distribution implementation used in `collect_moe.py` for simulating realistic Mixture-of-Experts (MoE) workload patterns during performance benchmarking.

## What is Power-Law Distribution?

A power-law distribution is a probability distribution where the probability of observing a value x is proportional to x^(-α), where α (alpha) is the power-law exponent:

```
P(x) ∝ x^(-α)
```

**Key characteristics:**
- **Heavy-tailed**: Few experts receive most tokens, many experts receive few tokens
- **Scale-free**: The pattern holds across different scales
- **Realistic**: Many real-world phenomena follow power-law distributions (Zipf's law, Pareto principle)

## Mathematical Derivation

This section derives the inverse CDF formula used in `sample_power_law()` for generating power-law distributed random samples.

### 1. Probability Density Function (PDF)

The power-law distribution's PDF over the interval \[x_min, x_max\] is:

$$
f(x) = \frac{\alpha - 1}{x_{\min}^{1-\alpha} - x_{\max}^{1-\alpha}} \cdot x^{-\alpha}, \quad x \in [x_{\min}, x_{\max}]
$$

where α > 1 is the power-law exponent.

### 2. Cumulative Distribution Function (CDF)

Integrating the PDF from x_min to x gives the CDF:

$$
F(x) = \int_{x_{\min}}^{x} f(t) \, dt
$$

Computing the integral:

$$
F(x) = \frac{\alpha - 1}{x_{\min}^{1-\alpha} - x_{\max}^{1-\alpha}} \int_{x_{\min}}^{x} t^{-\alpha} \, dt
$$

$$
= \frac{\alpha - 1}{x_{\min}^{1-\alpha} - x_{\max}^{1-\alpha}} \cdot \left[ \frac{t^{1-\alpha}}{1-\alpha} \right]_{x_{\min}}^{x}
$$

$$
= \frac{x^{1-\alpha} - x_{\min}^{1-\alpha}}{x_{\min}^{1-\alpha} - x_{\max}^{1-\alpha}}
$$

### 3. Inverse Transform Sampling

To generate random samples, we use the **inverse transform method**. Let u ~ Uniform(0,1), and solve for x such that:

$$
F(x) = u
$$

Substituting the CDF:

$$
\frac{x^{1-\alpha} - x_{\min}^{1-\alpha}}{x_{\min}^{1-\alpha} - x_{\max}^{1-\alpha}} = u
$$

### 4. Deriving the Inverse CDF

Multiply both sides by the denominator:

$$
x^{1-\alpha} - x_{\min}^{1-\alpha} = u \cdot (x_{\min}^{1-\alpha} - x_{\max}^{1-\alpha})
$$

Rearrange:

$$
x^{1-\alpha} = u \cdot (x_{\min}^{1-\alpha} - x_{\max}^{1-\alpha}) + x_{\min}^{1-\alpha}
$$

Adjust the order (note the sign change):

$$
x^{1-\alpha} = (x_{\max}^{1-\alpha} - x_{\min}^{1-\alpha}) \cdot u + x_{\min}^{1-\alpha}
$$

Take both sides to the power of 1/(1-α):

$$
\boxed{x = \left[ (x_{\max}^{1-\alpha} - x_{\min}^{1-\alpha}) \cdot u + x_{\min}^{1-\alpha} \right]^{\frac{1}{1-\alpha}}}
$$

### 5. Implementation in Code

The formula is implemented in the `sample_power_law()` function in `collect_moe.py`.

**Formula breakdown:**

| Code Component | Math Expression | Purpose |
|---------------|-----------------|---------|
| `xmax ** (1 - alpha)` | $x_{\max}^{1-\alpha}$ | Transform max value |
| `xmin ** (1 - alpha)` | $x_{\min}^{1-\alpha}$ | Transform min value |
| `(...) * u` | Multiply by uniform random | Linear interpolation in transformed space |
| `+ xmin ** (1 - alpha)` | Offset | Ensure x ≥ x_min |
| `** (1 / (1 - alpha))` | $\frac{1}{1-\alpha}$ power | Inverse transform back to original space |

### 6. Intuition

The formula works by:
1. **Transforming** the space: The (1-α) power compresses/stretches the probability space
2. **Linear interpolation**: In the transformed space, we do simple linear interpolation using u
3. **Inverse transform**: The 1/(1-α) power maps back to the original space, creating the power-law shape

### 7. Special Cases

**Case: α = 0 (Uniform Distribution)**

When α = 0, the power-law formula reduces to a uniform distribution. Let's verify this mathematically:

Starting from the inverse CDF:

$$
x = \left[ (x_{\max}^{1-\alpha} - x_{\min}^{1-\alpha}) \cdot u + x_{\min}^{1-\alpha} \right]^{\frac{1}{1-\alpha}}
$$

Substitute α = 0:

$$
x = \left[ (x_{\max}^{1-0} - x_{\min}^{1-0}) \cdot u + x_{\min}^{1-0} \right]^{\frac{1}{1-0}}
$$

$$
x = \left[ (x_{\max} - x_{\min}) \cdot u + x_{\min} \right]^{1}
$$

$$
\boxed{x = x_{\min} + u \cdot (x_{\max} - x_{\min})}
$$

This is exactly the formula for a **uniform distribution** over \[x_min, x_max\]!

**Numerical Example:**

With u = 0.5 (median), x_min = 1, x_max = 100, and α = 0.0:
- x = ((100 - 1) × 0.5 + 1) = 50.5

For a uniform distribution, the median equals the mean at (x_min + x_max)/2 = 50.5 ✓

**Practical Implication in MoE:**

In `collect_moe.py`, setting `alpha=0.0` triggers the use of `balanced_logits()` instead, ensuring perfectly balanced expert loads, representing the ideal baseline scenario with no imbalance.

### 8. Verification

The implementation can be verified by comparing generated samples against the theoretical PDF and checking statistical properties (mean, median, percentiles) and the Pareto principle (e.g., for α=1.2, expect top 20% to contribute ~60-70% of total load).

## Why Use Power-Law for MoE?

In production MoE models, token routing is rarely uniform. Real workloads exhibit:

1. **Expert Specialization**: Some experts become "generalists" handling many tokens
2. **Long-tail Distribution**: Most experts are "specialists" with lighter loads
3. **Dynamic Imbalance**: Load distribution varies with input data

Power-law simulation helps benchmark MoE performance under realistic conditions, not just ideal balanced scenarios.

## Alpha Parameter Guide

The `alpha` parameter controls the degree of load imbalance:

| Alpha Value | Distribution Type | Estimated For (by latency matching) |
|-------------|-------------------|----------|
| **0.0** | Perfectly balanced (theoretical ideal) | Synthetic baseline for benchmarking |
| **1.01** | Lower imbalance | DeepSeek-V3-R1 on CNNDaily/OpenOrca |
| **1.2** | Higher imbalance | Qwen3-235B on CNNDaily/OpenOrca，Typical production |
| **> 1.5** | Very high imbalance | Rare edge cases or models without load balancing |

**Important Context:**
- **α=0.0** represents a theoretical "speed of light" baseline with perfect uniformity—this is an ideal reference point, not achievable in real deployments
- **α=1.01** was estimated for DeepSeek-V3-R1 by comparing observed latency with benchmark results, suggesting relatively balanced distribution but still significant deviation from α=0.0
- **α=1.2** was estimated for Qwen3-235B by comparing observed latency with benchmark results, suggesting stronger expert specialization
- Different models and datasets may exhibit different alpha values; these are reference points, not universal categories

### Visual Representation

```
Alpha = 0.0 (Perfectly Balanced - Theoretical Ideal)
Experts:  E1   E2   E3   E4   E5   E6   E7   E8
Load:     ████ ████ ████ ████ ████ ████ ████ ████

Alpha = 1.01 (Lower Imbalance)
Experts:  E1   E2   E3   E4   E5   E6   E7   E8
Load:     █████████ ██████ █████ ████ ███ ███ ██ ██

Alpha = 1.2 (Higher Imbalance)
Experts:  E1   E2   E3   E4   E5   E6   E7   E8
Load:     ████████████████ ████████ ████ ██ ██ █ █ █
```

## Distribution Visualization

The following comprehensive visualization shows how different alpha values affect the distribution of expert loads in MoE models:

![Power-Law Distribution Comparison](power_law_comparison.png)

**Figure Description:**

The visualization contains six panels demonstrating different aspects of power-law distributions:

1. **Top Row (Individual Distributions)**: Shows probability density histograms for each alpha value with overlaid theoretical PDF curves
   - Left (α=0.0): Uniform distribution - all experts equally likely to receive tokens
   - Middle (α=1.01): Slight skew - some experts preferred but distribution remains relatively flat
   - Right (α=1.2): Heavy tail - few experts dominate, most receive minimal load

2. **Bottom Left (Overlay Comparison)**: All three distributions on log scale, clearly showing the different tail behaviors

3. **Bottom Middle (CDF Comparison)**: Cumulative distribution functions reveal how quickly probability mass accumulates
   - α=0.0: Linear CDF (uniform)
   - α=1.01: Slightly curved, gradual accumulation
   - α=1.2: Steep initial rise then long tail (most mass in small values)

4. **Bottom Right (Expert Rank Analysis)**: Rank-ordered expert loads in log-log scale
   - α=0.0: Flat line (all experts equal)
   - α=1.01: Gentle slope (mild hierarchy)
   - α=1.2: Steep power-law decay (strong Pareto effect)

### Real-World Model Observations

By comparing actual inference latency against benchmarks with different alpha values, we can estimate the load distribution pattern in production models:

#### Qwen3-235B on CNNDaily and OpenOrca Datasets

**Estimated α ≈ 1.2 (Higher Imbalance)**

*Method: Observed latency closely matches benchmark results with α=1.2*

- **Characteristics:**
  - Strong expert specialization with clear "hot" experts handling 60-70% of tokens
  - Significant load imbalance across the 128 experts
  - Benefits from auxiliary load balancing loss during training
  - Top 20% of experts handle ~65% of the workload (approaching Pareto 80/20)

- **Performance Implications:**
  - Higher EP degree recommended (e.g., EP=8) to distribute hot experts
  - AllReduce communication becomes bottleneck as few experts dominate
  - Cache-friendly: Hot experts stay in GPU cache
  - May benefit from dynamic load balancing strategies

- **Workload Pattern:**
  ```
  Expert 1:  ████████████████████████ (Hot - handles many tokens)
  Expert 2:  ██████████████
  Expert 3:  ████████
  Expert 4-10: ███
  Expert 11+:  █ (Cold - minimal load)
  ```

#### DeepSeek-V3-R1 on CNNDaily and OpenOrca Datasets

**Estimated α ≈ 1.01 (Lower Imbalance)**

*Method: Observed latency closely matches benchmark results with α=1.01*

- **Characteristics:**
  - Latency behavior suggests lower expert load skew relative to α=1.2 patterns
  - Effective auxiliary load balancing during training
  - Behavior consistent with top 20% of experts handling ~30-40% of workload

- **Performance Implications:**
  - Efficient hardware utilization - all experts contribute meaningfully
  - Lower EP degree sufficient (e.g., EP=2 or EP=4)
  - Reduced communication overhead compared to imbalanced scenarios
  - Better scalability to larger expert counts

- **Workload Pattern:**
  ```
  Expert 1:  ████████ (Slightly preferred)
  Expert 2:  ███████
  Expert 3:  ██████
  Expert 4:  ██████
  ...
  Expert 256: ████ (Still receives reasonable load)
  ```

### Key Takeaways

1. **α=0.0 (Perfectly Balanced)**: Theoretical ideal - unachievable baseline for comparison
2. **α=1.01 (Lower Imbalance)**: Estimated for DeepSeek-V3-R1 on CNNDaily/OpenOrca by latency matching
3. **α=1.2 (Higher Imbalance)**: Estimated for Qwen3-235B on CNNDaily/OpenOrca by latency matching

**Recommendation**: When benchmarking MoE models, test with multiple alpha values to understand performance across different load distribution patterns:
- **Theoretical limit** (α=0.0): Maximum possible throughput under perfect conditions (reference only)
- **Lower imbalance** (α=1.01): Representative of models with effective load balancing (still shows notable skew vs. α=0.0)
- **Higher imbalance** (α=1.2): Representative of models with significant expert specialization

**Note on Methodology**: The alpha values above were estimated by comparing observed inference latency against benchmark results with different alpha values. The appropriate alpha value for your use case depends on your specific model, dataset, and workload characteristics.

## Key Algorithms

### 1. EP Group Selection

The algorithm uses Conv1D to find the EP group with maximum token load and swaps it to group 0. This ensures the benchmark measures the worst-case (bottleneck) performance, as the EP group handling the most tokens determines the overall latency.

**Process (Example: 16 experts, EP=4):**

```
Step 1: Power-law distributed expert loads (unsorted - in original order)
┌────────────────────────────────────────────────────────────────────────┐
│ Expert:  E0   E1   E2   E3  │ E4   E5   E6   E7  │ E8   E9   E10  E11 │ E12  E13  E14  E15 │
│ Tokens:  180  110  210  140 │ 850  620  480  320 │ 70   280  55   40  │ 30   85   20   10  │
└────────────────────────────────────────────────────────────────────────┘
         ↑    ↑    ↑    ↑         ↑    ↑    ↑    ↑        ↑    ↑    ↑    ↑        ↑    ↑    ↑    ↑
         Expert loads follow power-law but are NOT pre-sorted

Step 2: Divide into EP groups (16 experts ÷ 4 EP = 4 experts/group)
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │   Group 0       │   │   Group 1       │   │   Group 2       │   │   Group 3       │
    │  E0  E1  E2 E3  │   │  E4  E5  E6 E7  │   │  E8  E9 E10 E11 │   │ E12 E13 E14 E15 │
    │ 180 110 210 140 │   │ 850 620 480 320 │   │  70 280  55  40 │   │  30  85  20  10 │
    └─────────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘

Step 3: Conv1D sums tokens per group (kernel_size=4, stride=4, weights=1)
           ↓                     ↓                     ↓                     ↓
    Total: 640              Total: 2270 ★          Total: 445           Total: 145
                                  (MAX)

Step 4: argmax finds Group 1 has maximum load → Swap to position 0
    Before swap:  [G0: 640]  [G1: 2270★]  [G2: 445]  [G3: 145]
                      ⇅          ⇅
    After swap:   [G1: 2270★]  [G0: 640]  [G2: 445]  [G3: 145]

    Experts after swap:
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │   Group 0       │   │   Group 1       │   │   Group 2       │   │   Group 3       │
    │  E4  E5  E6 E7  │   │  E0  E1  E2 E3  │   │  E8  E9 E10 E11 │   │ E12 E13 E14 E15 │
    │ 850 620 480 320 │   │ 180 110 210 140 │   │  70 280  55  40 │   │  30  85  20  10 │
    └─────────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘

Result: Benchmark measures Group 0 with highest token load (2270)
        → Captures worst-case latency (bottleneck performance)
```

### 2. Token Assignment Algorithm

After generating power-law distributed expert loads, we need to assign tokens to experts. This section compares two fundamentally different approaches.

**Notation:** `E` = num_experts, `T` = num_tokens, `K` = topk

#### 2.1 Naive Approach: Token Iteration with Sorted Experts

**Visual:** ![Naive Approach Flow](token_assignment_sorted.png)

A straightforward approach is to:
1. Sort experts by load (If we don't sort experts, it will need nextensive backtracking when remaining experts cannot fill topk slots without duplicates)
2. FOR each token, iterate through sorted experts and select if quota available
3. Continue until each token gets K experts

**Algorithm:**
```
SORT experts by load (descending)
FOR each token (T iterations):
    FOR each expert (up to E checks):
        IF quota > 0 AND not duplicate:
            SELECT expert
            DECREMENT quota
```

**Complexity:**
- **Computation:** `O(E log E + T*K)` ~ `O(E log E + T*E)` in worst case
- **Operations (E=10, T=1000, K=4):** ~33 (sort) + 4,000~10,000 (iteration) = **~4,033-10,033 operations**

**Critical Problem:** This approach **does not scale** when `num_tokens` is large:
- Requires T × K operations minimum (nested token loop)
- In typical MoE scenarios (T = thousands to millions), this becomes a major bottleneck
- **Not practical for production use with large token counts**

#### 2.2 AIC Assignment Algorithm (Current Implementation)

**Visual:** ![AIC Assignment Flow](token_assignment.png)

To address the scalability issue, we developed an optimized algorithm that **avoids iterating over tokens entirely** and **doesn't require sorting**.

**Algorithm Steps:**

1. **Expand** - FOR each expert (in original order), write `expert_id × quota` times: `O(E)` iterations
2. **Reshape** to `(K, T)` then **transpose** to `(T, K)`: `O(1)` metadata operation

**Pseudocode:**
```
expanded = []
FOR each expert in original order (E iterations):
    APPEND expert_id × quota times to expanded
assignments = expanded.reshape(K, T).transpose()  # Now (T, K)
```

**Complexity:**
- **Computation:** `O(E)` - **Independent of T and K!** No sorting needed!
- **I/O:** `O(T*K)` - Must write output array (unavoidable)
- **Operations (E=10, T=1000, K=4):** 10 (expand only) = **~10 operations**

**Key Guarantee:** Because `max_expert_tokens ≤ num_tokens`, the reshape operation mathematically guarantees no duplicate experts per token.

**Advantages over Naive Approach:**
- **~400× fewer operations** in typical scenarios (10 vs 4,000+)
- **Scales to large T:** Computation is `O(E)`, independent of token count
- **No sorting overhead:** Direct expansion without preprocessing
- **Vectorized:** Leverages NumPy's optimized operations
- **No nested loops:** Avoids token iteration entirely

#### Performance Comparison

**Typical MoE Scenario (E=10, T=1000, K=4):**

| Metric | Naive Approach | **AIC Assignment** |
|--------|----------------|-------------------|
| Algorithm | Sort → FOR token → FOR expert | Expand → reshape |
| Sort Comparisons | 33 | 0 (no sorting!) |
| Loop Iterations | 4,000 ~ 10,000 (T×K~T×E) | 10 (E only) |
| **Total Operations** | **~4,033-10,033** | **~10** |
| **Speedup** | baseline | **~400-1000×** |
| **Complexity** | O(E log E + T×K) | **O(E)** |
| **Scales with T?** | ❌ No | ✅ Yes |

**Key Insight:** When `E << T` (typical in MoE: E=8-64, T=thousands to millions), the AIC assignment algorithm exploits this asymmetry by **eliminating the token loop entirely**. It achieves `O(E)` complexity—independent of both the number of tokens and topk—by directly writing each expert's quota and using reshape's mathematical properties to guarantee no duplicates per token.

## Usage in Benchmarking

The `run_moe_torch()` function uses these distributions to benchmark MoE performance with multiple alpha values (0.0, 1.01, 1.2):

- **alpha=0.0** → Uses `balanced_logits()` for theoretical baseline (perfectly uniform distribution)
- **alpha>0.0** → Uses `power_law_logits_v3()` for different load distribution patterns
  - α=1.01: Lower imbalance pattern (observed in DeepSeek-V3-R1 on specific datasets)
  - α=1.2: Higher imbalance pattern (observed in Qwen3-235B on specific datasets)

## Debug Mode

Set environment variable `aic_moe_debug=2` before running to inspect token distribution details. The output will show the `num_tokens_per_expert` tensor with the actual token counts assigned to each expert.


---

**Last Updated**: 2025-11-21  
**Author**: NVIDIA AIC Team  
**Related Models**: DeepSeek-V3, Mixtral, Qwen3-MoE, Kimi K2

