"""Environment profile schema.

The environment profile captures cluster facts that the generator cannot know —
namespace, storage class, mount root, NIC names, secret names, registry prefix.
It is the agent/discovery boundary: the generator core stays deterministic and
never queries the cluster; callers supply an EnvironmentProfile.

Only this file is the authoritative source; ENVIRONMENT_JSON_SCHEMA is derived
from the same field set so the dataclass and schema stay in sync.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

_KNOWN_FIELDS = frozenset(
    {
        "namespace",
        "storage_class_name",
        "mount_root",
        "image_pull_secrets",
        "hf_token_secret",
        "nccl_socket_ifname",
        "gloo_socket_ifname",
        "registry_prefix",
    }
)


@dataclasses.dataclass(frozen=True)
class EnvironmentProfile:
    """Flat bag of cluster-supplied facts consumed by the generator."""

    namespace: str = "default"
    storage_class_name: str = "standard"
    mount_root: str = "/opt/models"
    image_pull_secrets: list[str] = dataclasses.field(default_factory=list)
    hf_token_secret: str | None = None
    nccl_socket_ifname: str | None = None
    gloo_socket_ifname: str | None = None
    registry_prefix: str | None = None


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------

ENVIRONMENT_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EnvironmentProfile",
    "description": (
        "Cluster-supplied facts that the AI Configurator generator cannot discover "
        "on its own. Provide one profile per target cluster."
    ),
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "namespace": {
            "type": "string",
            "description": "Kubernetes namespace for all generated resources.",
            "default": "default",
        },
        "storage_class_name": {
            "type": "string",
            "description": "StorageClass used for PersistentVolumeClaims.",
            "default": "standard",
        },
        "mount_root": {
            "type": "string",
            "description": "Base path inside containers where volumes are mounted.",
            "default": "/opt/models",
        },
        "image_pull_secrets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of Kubernetes secrets used to pull images.",
            "default": [],
        },
        "hf_token_secret": {
            "type": ["string", "null"],
            "description": "Name of the Kubernetes secret holding the HuggingFace token.",
        },
        "nccl_socket_ifname": {
            "type": ["string", "null"],
            "description": "Network interface name for NCCL socket transport (e.g. eth0).",
        },
        "gloo_socket_ifname": {
            "type": ["string", "null"],
            "description": "Network interface name for Gloo socket transport (e.g. eth0).",
        },
        "registry_prefix": {
            "type": ["string", "null"],
            "description": (
                "Container registry prefix prepended to all image references "
                "(e.g. registry.example.com/myorg)."
            ),
        },
    },
}

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_environment_profile(path: str) -> EnvironmentProfile:
    """Load an EnvironmentProfile from a YAML file.

    Unknown keys are silently ignored so that future schema additions do not
    break older profiles loaded by a newer generator.  Fields absent from the
    file fall back to their dataclass defaults.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    known = {k: v for k, v in raw.items() if k in _KNOWN_FIELDS}
    # image_pull_secrets must be a list; guard against a bare null in YAML
    if "image_pull_secrets" in known and known["image_pull_secrets"] is None:
        known["image_pull_secrets"] = []
    return EnvironmentProfile(**known)
