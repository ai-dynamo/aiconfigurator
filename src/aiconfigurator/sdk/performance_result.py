# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PerformanceResult class for backward-compatible latency+energy tracking.
"""


class PerformanceResult(float):
    """
    Float-like class that stores both latency and energy.

    Behaves exactly like a float for backward compatibility, but stores energy
    instead of power internally. Power is derived as energy / latency.

    ⚠️ WARNING: Do NOT use for aggregation! Use separate latency_dict and energy_dict.
    PerformanceResult is for returning atomic operation results only.

    Units:
        - latency: milliseconds (ms)
        - energy: watt-milliseconds (W·ms) = millijoules (mJ)
        - power: watts (W) - derived property

    Note: 1 W·ms = 1 mJ. We use W·ms to match latency units (ms).
          To convert to Joules: divide by 1000 (J = W·s = W·ms / 1000)

    Example:
        result = PerformanceResult(10.5, energy=3675.0)  # 10.5ms latency, 3675 W·ms energy
        print(result)           # 10.5 (acts like float)
        print(result.energy)    # 3675.0 (energy in W·ms = 3.675 J)
        print(result.power)     # 350.0 (derived: 3675.0 / 10.5 = 350W)

        # ❌ WRONG: Do not sum PerformanceResults directly
        # total = sum([result1, result2])  # Might lose energy!

        # ✅ CORRECT: Extract and aggregate separately
        # latency_sum = float(result1) + float(result2)
        # energy_sum = result1.energy + result2.energy
    """

    def __new__(cls, latency, energy=0.0):
        """
        Create a new PerformanceResult.

        Args:
            latency: The latency value in milliseconds (acts as the float value)
            energy: The energy value in watt-milliseconds (W·ms)
        """
        instance = float.__new__(cls, latency)
        return instance

    def __init__(self, latency, energy=0.0):
        """
        Initialize the PerformanceResult.

        Args:
            latency: The latency value in milliseconds
            energy: The energy value in watt-milliseconds (W·ms)
                   Note: 1 W·ms = 1 millijoule (mJ)
        """
        self.energy = energy  # W·ms (watt-milliseconds)

    @property
    def power(self):
        """
        Derived power in watts (for backward compatibility).

        Power = Energy / Latency

        Returns:
            float: Power in watts
        """
        latency = float(self)
        if latency > 0:
            return self.energy / latency
        return 0.0

    def __repr__(self):
        """String representation showing latency, energy, and derived power."""
        return f"PerformanceResult(latency={float(self)}, energy={self.energy}, power={self.power})"

    def __add__(self, other):
        """Add two PerformanceResults or a PerformanceResult and a number."""
        if isinstance(other, PerformanceResult):
            # Add latencies and energies (both are additive!)
            return PerformanceResult(float(self) + float(other), energy=self.energy + other.energy)
        else:
            # Add to latency only, keep same energy
            return PerformanceResult(float(self) + other, energy=self.energy)

    def __radd__(self, other):
        """Right addition for sum() support."""
        return self.__add__(other)

    def __mul__(self, other):
        """Multiply PerformanceResult by a scalar."""
        # Scale both latency and energy
        return PerformanceResult(float(self) * other, energy=self.energy * other)

    def __rmul__(self, other):
        """Right multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide PerformanceResult by a scalar."""
        # Scale both latency and energy
        return PerformanceResult(float(self) / other, energy=self.energy / other)
