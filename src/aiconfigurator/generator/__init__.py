"""
generator:

The generator provide the config generation that can
be used by both CLI and WebApp. It offers:
- A API (api.py) that accepts runtime objects/saved files and optionally saves files.
- A pluggable backend generator registry (backends/*).
- A input schema and parser (inputs/*).
- Utilities for saving artifacts and node allocation (utils/*).
"""
from .api import generate_backend_config
