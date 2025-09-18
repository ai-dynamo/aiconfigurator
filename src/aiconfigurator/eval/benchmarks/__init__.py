from typing import Callable, Dict, Any
import warnings

_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register(name: str, *, parse=None):
    def _wrap(fn: Callable[..., None]):
        if name in _REGISTRY:
            warnings.warn(
                f"Benchmark runner '{name}' is being re-registered; "
                f"previous entry will be overwritten.",
                RuntimeWarning,
            )
        _REGISTRY[name] = {"run": fn, "parse": parse}
        return fn
    return _wrap

from . import genai_perf_runner as _genai_perf_runner
from . import bench_serving_runner as _bench_serving_runner

def get(name: str):
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"Benchmark runner '{name}' is not registered. "
            f"Registered runners: {', '.join(sorted(_REGISTRY.keys())) or '<none>'}."
        ) from e
