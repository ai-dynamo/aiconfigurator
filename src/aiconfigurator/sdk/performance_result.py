# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PerformanceResult class for backward-compatible latency+power tracking.
"""


class PerformanceResult(float):
    """
    Float-like class that stores both latency and power.
    
    Behaves exactly like a float for backward compatibility, but adds a .power attribute.
    This allows existing code to continue using the result as a float (latency in ms)
    while also making power data available when needed.
    
    Example:
        result = PerformanceResult(10.5, power=350.0)  # 10.5ms latency, 350W power
        print(result)  # 10.5 (acts like float)
        print(result + 5)  # 15.5 (math operations work)
        print(result.power)  # 350.0 (power data available)
    """
    
    def __new__(cls, latency, power=0.0):
        """
        Create a new PerformanceResult.
        
        Args:
            latency: The latency value in milliseconds (acts as the float value)
            power: The power value in watts (stored as attribute)
        """
        # Create the float with the latency value
        instance = float.__new__(cls, latency)
        return instance
    
    def __init__(self, latency, power=0.0):
        """
        Initialize the PerformanceResult.
        
        Args:
            latency: The latency value in milliseconds
            power: The power value in watts
        """
        # Store power as an attribute
        self.power = power
    
    def __repr__(self):
        """String representation showing both latency and power."""
        return f"PerformanceResult(latency={float(self)}, power={self.power})"
    
    def __add__(self, other):
        """Add two PerformanceResults or a PerformanceResult and a number."""
        if isinstance(other, PerformanceResult):
            # Add latencies and powers
            return PerformanceResult(
                float(self) + float(other),
                power=self.power + other.power
            )
        else:
            # Add to latency only, keep same power
            return PerformanceResult(float(self) + other, power=self.power)
    
    def __radd__(self, other):
        """Right addition for sum() support."""
        return self.__add__(other)
    
    def __mul__(self, other):
        """Multiply PerformanceResult by a scalar."""
        # Scale both latency and power
        return PerformanceResult(float(self) * other, power=self.power * other)
    
    def __rmul__(self, other):
        """Right multiplication."""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Divide PerformanceResult by a scalar."""
        # Scale both latency and power
        return PerformanceResult(float(self) / other, power=self.power / other)

