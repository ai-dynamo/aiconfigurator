# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NVML-based power monitoring - drop-in replacement for Zeus ZeusMonitor.

This module provides a Zeus-compatible API for GPU power monitoring using NVIDIA's
native NVML library (via pynvml). It replaces the discontinued Zeus package while
maintaining the same interface for minimal code changes.
"""

import time
from dataclasses import dataclass


@dataclass
class PowerMeasurement:
    """
    Zeus-compatible measurement result.

    Attributes:
        total_energy: Total energy consumed across all GPUs (Joules)
        time: Time elapsed during measurement (seconds)
        gpu_energy: Per-GPU energy consumption (dict[gpu_index, Joules])
    """

    total_energy: float  # Joules
    time: float  # Seconds
    gpu_energy: dict[int, float]  # Per-GPU energy in Joules


class NVMLPowerMonitor:
    """
    Drop-in replacement for Zeus ZeusMonitor using NVML.

    This class provides the same API as zeus.monitor.ZeusMonitor but uses
    pynvml (nvidia-ml-py) directly. It measures GPU energy consumption
    using NVML's hardware-backed energy counters.

    Example:
        >>> monitor = NVMLPowerMonitor(gpu_indices=[0, 1])
        >>> monitor.begin_window("benchmark", sync_execution=True)
        >>> # ... run workload ...
        >>> result = monitor.end_window("benchmark", sync_execution=True)
        >>> print(f"Energy: {result.total_energy:.2f} J, Time: {result.time:.2f} s")

    Args:
        gpu_indices: List of GPU indices to monitor
    """

    def __init__(self, gpu_indices: list[int]):
        try:
            import pynvml
        except ImportError as e:
            raise ImportError("pynvml is required. Install with: pip install nvidia-ml-py") from e

        self.pynvml = pynvml
        self.gpu_indices = gpu_indices
        self._window_data = {}
        self._initialized = False

        # Initialize NVML
        try:
            self.pynvml.nvmlInit()
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NVML: {e}") from e

        # Get and store device handles
        self.handles = []
        for idx in gpu_indices:
            try:
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(idx)
                self.handles.append(handle)
            except Exception as e:
                self._cleanup()
                raise RuntimeError(f"Failed to get handle for GPU {idx}: {e}") from e

    def begin_window(self, name: str, sync_execution: bool = False):
        """
        Start a power measurement window.

        Args:
            name: Unique identifier for this measurement window
            sync_execution: If True, synchronize CUDA before starting measurement
        """
        if sync_execution:
            try:
                import torch

                torch.cuda.synchronize()
            except ImportError:
                pass  # PyTorch not available, skip sync

        # Query energy for all GPUs
        start_energies = []
        for handle in self.handles:
            try:
                # Returns energy in millijoules (mJ)
                energy_mj = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                start_energies.append(energy_mj)
            except Exception as e:
                raise RuntimeError(f"Failed to query energy: {e}") from e

        # Store start state
        self._window_data[name] = {"start_time": time.time(), "start_energy": start_energies}

    def end_window(self, name: str, sync_execution: bool = False) -> PowerMeasurement:
        """
        End a power measurement window and return results.

        Args:
            name: Identifier of the window to end (must match begin_window call)
            sync_execution: If True, synchronize CUDA before ending measurement

        Returns:
            PowerMeasurement object containing energy and timing data

        Raises:
            KeyError: If no window with the given name was started
        """
        if sync_execution:
            try:
                import torch

                torch.cuda.synchronize()
            except ImportError:
                pass  # PyTorch not available, skip sync

        # Query energy for all GPUs
        end_time = time.time()
        end_energies = []
        for handle in self.handles:
            try:
                energy_mj = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                end_energies.append(energy_mj)
            except Exception as e:
                raise RuntimeError(f"Failed to query energy: {e}") from e

        # Retrieve start state
        if name not in self._window_data:
            raise KeyError(f"No window named '{name}' was started")

        start_data = self._window_data.pop(name)

        # Calculate deltas
        per_gpu_energy = {}
        total_energy_j = 0.0

        for i, (end_mj, start_mj) in enumerate(zip(end_energies, start_data["start_energy"])):
            # Handle potential counter rollover (unlikely for short measurements)
            if end_mj < start_mj:
                # Counter rolled over - this is rare but possible
                # NVML counter is 64-bit unsigned, max value is 2^64 - 1 mJ
                MAX_COUNTER_VALUE = 2**64 - 1  # noqa: N806
                energy_delta_mj = (MAX_COUNTER_VALUE - start_mj) + end_mj
            else:
                energy_delta_mj = end_mj - start_mj

            # Convert millijoules to joules
            energy_j = energy_delta_mj / 1000.0
            gpu_idx = self.gpu_indices[i]
            per_gpu_energy[gpu_idx] = energy_j
            total_energy_j += energy_j

        time_elapsed_s = end_time - start_data["start_time"]

        return PowerMeasurement(total_energy=total_energy_j, time=time_elapsed_s, gpu_energy=per_gpu_energy)

    def _cleanup(self):
        """Cleanup NVML resources."""
        if self._initialized:
            try:
                self.pynvml.nvmlShutdown()
                self._initialized = False
            except Exception:
                pass  # Best effort cleanup

    def __del__(self):
        """Ensure NVML is shut down when object is destroyed."""
        self._cleanup()

    def __enter__(self):
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager protocol."""
        self._cleanup()
        return False


def get_power_management_limit(device_id: int) -> int:
    """
    Get current GPU power management limit (TGP/TDP cap).

    Args:
        device_id: GPU index

    Returns:
        Current power limit in Watts

    Raises:
        RuntimeError: If getting power limit fails

    Example:
        >>> limit = get_power_management_limit(0)  # Get GPU 0 power limit
    """
    try:
        import pynvml
    except ImportError as e:
        raise ImportError("pynvml is required. Install with: pip install nvidia-ml-py") from e

    initialized = False
    try:
        # Initialize NVML if not already done
        try:
            pynvml.nvmlInit()
            initialized = True
        except Exception:
            # May already be initialized
            pass

        # Get device handle
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        # Get power limit (returned in milliwatts)
        power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        return power_limit_mw // 1000  # Convert milliwatts to watts

    except Exception as e:
        raise RuntimeError(f"Failed to get power limit on GPU {device_id}: {e}") from e
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def set_power_management_limit(device_id: int, power_limit_watts: int) -> None:
    """
    Set GPU power management limit (TGP/TDP cap).

    This is a convenience function to replace zeus.device.get_gpus().set_power_management_limit().
    Requires root/elevated privileges.

    Args:
        device_id: GPU index
        power_limit_watts: Power limit in Watts

    Raises:
        RuntimeError: If setting power limit fails
        PermissionError: If insufficient privileges

    Example:
        >>> set_power_management_limit(0, 300)  # Set GPU 0 to 300W
    """
    try:
        import pynvml
    except ImportError as e:
        raise ImportError("pynvml is required. Install with: pip install nvidia-ml-py") from e

    initialized = False
    try:
        # Initialize NVML if not already done
        try:
            pynvml.nvmlInit()
            initialized = True
        except Exception:
            # May already be initialized
            pass

        # Get device handle
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        # Set power limit (convert Watts to milliwatts)
        power_limit_mw = power_limit_watts * 1000
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, power_limit_mw)

    except pynvml.NVMLError_NoPermission as e:
        raise PermissionError(
            f"Insufficient privileges to set power limit on GPU {device_id}. "
            "Root access required (e.g., run in Docker with --cap-add SYS_ADMIN)"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to set power limit on GPU {device_id}: {e}") from e
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

