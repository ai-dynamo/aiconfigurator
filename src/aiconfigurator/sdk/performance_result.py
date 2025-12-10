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

    Supports all arithmetic and comparison operations for full float compatibility.

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

        # Comparisons work correctly
        if result > 10.0:  # Uses __gt__
            print("Latency exceeds threshold")

        # Aggregation with sum()
        results = [result1, result2, result3]
        total = sum(results)  # __radd__ handles sum() start value
        print(total.energy)   # Energy is preserved

        # Sorting works based on latency
        sorted_results = sorted(results)
    """

    # Note: We don't use __slots__ here because float subclasses cannot define __slots__

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
        Calculate average power (Watts) from energy and latency.

        Returns 0.0 if latency is too small to avoid division by zero.

        Power = Energy / Latency

        Returns:
            float: Power in watts, or 0.0 if latency < 1e-9
        """
        latency = float(self)
        if latency > 1e-9:  # Use threshold to avoid numerical issues
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
        """Right addition for sum() support.

        CRITICAL: Handle sum() which starts with 0.
        Without this special case, sum([r1, r2, r3]) would fail or lose energy data.
        """
        if other == 0:
            # sum() starts with 0, just return self
            return self
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

    def __rtruediv__(self, other):
        """Right division: other / self."""
        # Return plain float when dividing by PerformanceResult
        return other / float(self)

    # Comparison operators (CRITICAL - Python doesn't auto-infer from float inheritance)
    def __lt__(self, other):
        """Less than comparison based on latency."""
        return float(self) < float(other)

    def __gt__(self, other):
        """Greater than comparison based on latency."""
        return float(self) > float(other)

    def __le__(self, other):
        """Less than or equal comparison based on latency."""
        return float(self) <= float(other)

    def __ge__(self, other):
        """Greater than or equal comparison based on latency."""
        return float(self) >= float(other)

    def __eq__(self, other):
        """Equality comparison based on latency."""
        try:
            return float(self) == float(other)
        except (TypeError, ValueError):
            return False

    def __ne__(self, other):
        """Inequality comparison based on latency."""
        try:
            return float(self) != float(other)
        except (TypeError, ValueError):
            return True

    def __abs__(self):
        """Absolute value of latency and energy."""
        return PerformanceResult(abs(float(self)), energy=abs(self.energy))

    def __hash__(self):
        """Hash based on latency and energy for use in sets/dicts."""
        return hash((float(self), self.energy))

    def __str__(self):
        """String representation (acts like float for easy printing)."""
        return str(float(self))
