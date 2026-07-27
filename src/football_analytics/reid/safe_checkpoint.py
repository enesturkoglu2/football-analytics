"""Safe sanitized state-dict checkpoint loader (weights_only=True only)."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch

from football_analytics.reid.embedding import (
    EMBEDDING_DIM,
    MODEL_NAME,
    EmbeddingError,
    build_osnet_cpu_model,
    sha256_file,
    temporary_sys_path_prepend,
    verify_sn_reid_root,
)


class SafeCheckpointError(EmbeddingError):
    """Raised when a sanitized SportsReID checkpoint fails the safe-load contract."""


# Known full training checkpoint (quarantine). Must never be used for inference.
REJECTED_FULL_SPORTSREID_CHECKPOINT = Path(
    "/home/enesturkoglu2/projects/soccernet/checkpoints/"
    "_quarantine_sportsreid_osnet_20260727T205544_17463/osnet_checkpoint.pth.tar"
).resolve()


def _resolve_regular_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise SafeCheckpointError(f"checkpoint path must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if resolved.is_symlink():
        raise SafeCheckpointError(f"resolved checkpoint path is a symlink: {resolved}")
    if not resolved.is_file():
        raise SafeCheckpointError(
            f"checkpoint must be an existing regular file: {resolved}"
        )
    if resolved.stat().st_size <= 0:
        raise SafeCheckpointError(f"checkpoint is empty: {resolved}")
    return resolved


def reject_non_sanitized_sportsreid_checkpoint(path: Path) -> None:
    """Fail-closed if the original full SportsReID training checkpoint is requested."""
    try:
        if path.resolve() == REJECTED_FULL_SPORTSREID_CHECKPOINT:
            raise SafeCheckpointError(
                "Unsafe or non-sanitized SportsReID checkpoint rejected"
            )
    except SafeCheckpointError:
        raise
    except OSError as exc:
        raise SafeCheckpointError(f"could not resolve checkpoint path: {exc}") from exc
    # Size heuristic: sanitized backbone is ~9MB; full training ckpt is ~1GB.
    size = path.stat().st_size
    if size >= 100_000_000:
        raise SafeCheckpointError(
            "Unsafe or non-sanitized SportsReID checkpoint rejected"
        )


def verify_expected_sha256(path: Path, expected_sha256: str) -> str:
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise SafeCheckpointError(
            f"expected checkpoint SHA-256 must be 64 hex chars, got {expected_sha256!r}"
        )
    actual = sha256_file(path).lower()
    if actual != expected:
        raise SafeCheckpointError(
            f"checkpoint SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def load_sanitized_state_dict_only(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
) -> OrderedDict[str, torch.Tensor]:
    """Load a sanitized state-dict-only checkpoint with weights_only=True.

    No allowlist. No weights_only=False fallback.
    """
    path = _resolve_regular_file(checkpoint_path)
    reject_non_sanitized_sportsreid_checkpoint(path)
    verify_expected_sha256(path, expected_sha256)

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as exc:
        # Older torch without weights_only — fail closed (no unsafe fallback).
        raise SafeCheckpointError(
            "weights_only=True is required but unsupported by this torch build; "
            "unsafe fallback is forbidden"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise SafeCheckpointError(
            f"safe weights_only load failed for {path}: {exc}"
        ) from exc

    if not isinstance(obj, (dict, OrderedDict)):
        raise SafeCheckpointError(
            f"sanitized checkpoint top-level must be a dict/OrderedDict, got {type(obj)!r}"
        )

    forbidden_top = {"optimizer", "scheduler", "epoch", "rank1", "mAP"}
    if any(k in obj for k in forbidden_top) and not all(
        torch.is_tensor(v) for v in obj.values()
    ):
        raise SafeCheckpointError(
            "nested training state or non-tensor checkpoint object rejected"
        )

    # Accept pure state_dict OR accidental wrapper with only 'state_dict'
    if "state_dict" in obj and not all(torch.is_tensor(v) for v in obj.values()):
        inner = obj["state_dict"]
        if not isinstance(inner, (dict, OrderedDict)):
            raise SafeCheckpointError("checkpoint 'state_dict' must be a mapping")
        if any(k in obj for k in ("optimizer", "scheduler")):
            raise SafeCheckpointError(
                "nested training state or non-tensor checkpoint object rejected"
            )
        obj = inner

    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    seen: set[str] = set()
    for key, value in obj.items():
        if not isinstance(key, str) or not key:
            raise SafeCheckpointError(
                f"state_dict keys must be non-empty str, got {key!r}"
            )
        if key in seen:
            raise SafeCheckpointError(f"duplicate state_dict key: {key}")
        seen.add(key)
        if not torch.is_tensor(value):
            raise SafeCheckpointError(
                f"non-tensor state_dict value rejected for key {key!r}: {type(value)!r}"
            )
        if value.numel() > 0 and not torch.isfinite(value).all():
            raise SafeCheckpointError(f"non-finite tensor in state_dict key {key!r}")
        tensor = value.detach().to(device="cpu").contiguous()
        state[key] = tensor

    if not state:
        raise SafeCheckpointError("sanitized state_dict is empty")
    return state


def apply_sanitized_state_dict_to_osnet(
    model: Any,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Load matched backbone/feature weights; classifier-only mismatches may be ignored."""
    model_dict = model.state_dict()
    matched: list[str] = []
    missing: list[str] = []
    unexpected: list[str] = []
    shape_mismatch: list[dict[str, Any]] = []
    ignored_classifier: list[dict[str, Any]] = []

    normalized = OrderedDict()
    for key, tensor in state_dict.items():
        nk = key[7:] if key.startswith("module.") else key
        if nk in normalized:
            raise SafeCheckpointError(f"duplicate key after prefix strip: {nk}")
        normalized[nk] = tensor

    loadable = OrderedDict()
    for key, tensor in normalized.items():
        if key not in model_dict:
            if "classifier" in key:
                ignored_classifier.append(
                    {"key": key, "reason": "unexpected_classifier_not_in_model"}
                )
            else:
                unexpected.append(key)
            continue
        expected = model_dict[key]
        if tuple(tensor.shape) != tuple(expected.shape):
            if "classifier" in key:
                ignored_classifier.append(
                    {
                        "key": key,
                        "ckpt_shape": list(tensor.shape),
                        "model_shape": list(expected.shape),
                        "reason": "classifier_shape_mismatch",
                    }
                )
            else:
                shape_mismatch.append(
                    {
                        "key": key,
                        "ckpt_shape": list(tensor.shape),
                        "model_shape": list(expected.shape),
                    }
                )
            continue
        loadable[key] = tensor
        matched.append(key)

    for key in model_dict:
        if key not in loadable:
            missing.append(key)

    backbone_missing = [k for k in missing if "classifier" not in k]
    if unexpected:
        raise SafeCheckpointError(
            f"unexpected non-classifier checkpoint keys: {unexpected[:20]}"
        )
    if shape_mismatch:
        raise SafeCheckpointError(
            f"backbone/feature shape mismatches: {shape_mismatch[:20]}"
        )
    if backbone_missing:
        raise SafeCheckpointError(
            f"missing backbone/feature keys: {backbone_missing[:20]}"
        )

    model_dict.update(loadable)
    model.load_state_dict(model_dict)
    model.eval()
    model.cpu()

    backbone_keys = [k for k in model_dict if "classifier" not in k]
    backbone_matched = [k for k in matched if "classifier" not in k]
    coverage = {
        "decision": "SPORTSREID_OSNET_FEATURE_EXTRACTOR_COMPATIBLE",
        "architecture": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIM,
        "exact_matched_keys": matched,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatch,
        "ignored_classifier_keys": ignored_classifier,
        "backbone_key_count": len(backbone_keys),
        "backbone_matched_count": len(backbone_matched),
        "backbone_coverage": (
            len(backbone_matched) / len(backbone_keys) if backbone_keys else 0.0
        ),
        "feature_head_keys_matched": [
            k for k in matched if k.startswith("fc.") or k.startswith("conv5.")
        ],
    }
    if coverage["backbone_coverage"] != 1.0:
        raise SafeCheckpointError("backbone coverage incomplete")
    return coverage


def load_sportsreid_osnet_cpu_model(
    *,
    checkpoint_path: str | Path,
    expected_sha256: str,
    sn_reid_root: str | Path,
    expected_sn_reid_commit: str,
) -> tuple[Any, dict[str, Any]]:
    """Build OSNet on CPU and load sanitized SportsReID weights safely."""
    sn_info = verify_sn_reid_root(
        sn_reid_root, expected_commit=expected_sn_reid_commit
    )
    state = load_sanitized_state_dict_only(
        checkpoint_path, expected_sha256=expected_sha256
    )
    with temporary_sys_path_prepend(sn_info["root"]):
        model = build_osnet_cpu_model(model_name=MODEL_NAME)
        coverage = apply_sanitized_state_dict_to_osnet(model, state)
    meta = {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "checkpoint_sha256": expected_sha256.strip().lower(),
        "sn_reid_root": str(sn_info["root"]),
        "sn_reid_commit": sn_info["commit"],
        "weights_only": True,
        "allowlist_used": False,
        "architecture_coverage": coverage,
    }
    return model, meta
