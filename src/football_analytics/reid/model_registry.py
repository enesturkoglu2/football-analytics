"""Explicit ReID model registry (Market1501 vs SportsReID SoccerNet)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "reid"
    / "model_registry.yaml"
)

MODEL_ID_MARKET1501 = "osnet_x1_0_market1501"
MODEL_ID_SPORTSREID_SOCCERNET = "osnet_x1_0_sportsreid_soccernet"

SUPPORTED_MODEL_IDS = frozenset(
    {MODEL_ID_MARKET1501, MODEL_ID_SPORTSREID_SOCCERNET}
)


class ModelRegistryError(RuntimeError):
    """Raised when a ReID model registry lookup or contract fails."""


def default_registry_path() -> Path:
    return DEFAULT_REGISTRY_PATH


def load_model_registry(
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(registry_path or DEFAULT_REGISTRY_PATH).expanduser().resolve()
    if not path.is_file():
        raise ModelRegistryError(f"model registry not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — surface parse failures clearly
        raise ModelRegistryError(f"could not parse model registry: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelRegistryError("model registry root must be a mapping")
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ModelRegistryError("model registry 'models' must be a non-empty mapping")
    return payload


def get_reid_model_spec(
    model_id: str,
    *,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a deep-ish copy of one model entry. No silent fallback."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ModelRegistryError(
            "Requested ReID model could not be loaded; no fallback executed"
        )
    requested = model_id.strip()
    registry = load_model_registry(registry_path)
    models = registry["models"]
    if requested not in models:
        raise ModelRegistryError(
            "Requested ReID model could not be loaded; no fallback executed"
        )
    entry = models[requested]
    if not isinstance(entry, dict):
        raise ModelRegistryError(f"registry entry for {requested!r} must be a mapping")
    if str(entry.get("model_id", "")).strip() != requested:
        raise ModelRegistryError(
            f"registry model_id mismatch for key {requested!r}"
        )
    # Shallow copy is enough; nested lists/dicts are treated as read-only contracts.
    return dict(entry)


def list_reid_model_ids(*, registry_path: str | Path | None = None) -> list[str]:
    registry = load_model_registry(registry_path)
    return sorted(str(k) for k in registry["models"].keys())


def require_mapping_field(spec: Mapping[str, Any], key: str) -> Any:
    if key not in spec:
        raise ModelRegistryError(f"model spec missing required field: {key}")
    return spec[key]
