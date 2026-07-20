"""ReID crop embedding extraction (Stage 4B-3).

Real ``torchreid`` imports happen only inside model build/load helpers so unit
tests can mock those callables without loading sn-reid or a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from football_analytics.reid.schema import (
    ReIDSchemaError,
    validate_manifest_row,
)
from football_analytics.reid.writers import (
    check_output_collision,
    cleanup_dir,
    write_manifest_jsonl,
)

MODEL_NAME = "osnet_x1_0"
EMBEDDING_DIM = 512
RESIZE_HEIGHT = 256
RESIZE_WIDTH = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_BATCH_SIZE = 8

INDEX_SCHEMA_VERSION = "reid_crop_embedding_index_v1"
SUMMARY_SCHEMA_VERSION = "reid_embedding_summary_v1"

NPZ_NAME = "crop_embeddings.npz"
INDEX_NAME = "crop_embeddings_index.jsonl"
SUMMARY_NAME = "embedding_summary.json"

PREPROCESSING: dict[str, Any] = {
    "color_conversion": "bgr_to_rgb",
    "resize_hw": [RESIZE_HEIGHT, RESIZE_WIDTH],
    "to_tensor": True,
    "normalize_mean": list(IMAGENET_MEAN),
    "normalize_std": list(IMAGENET_STD),
}

_CROP_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]
)


class EmbeddingError(RuntimeError):
    """Raised when ReID embedding extraction fails."""


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _reject_non_finite_json(value: str) -> None:
    raise EmbeddingError(f"NaN/Infinity forbidden in JSON ({value})")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(
    checkpoint_path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Validate checkpoint presence and SHA-256 before any model import."""
    path = Path(checkpoint_path).expanduser().resolve()
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise EmbeddingError(
            f"expected checkpoint SHA-256 must be 64 hex chars, got {expected_sha256!r}"
        )
    if not path.exists():
        raise EmbeddingError(f"checkpoint not found: {path}")
    if not path.is_file():
        raise EmbeddingError(f"checkpoint must be a regular file: {path}")
    if path.stat().st_size <= 0:
        raise EmbeddingError(f"checkpoint is empty: {path}")

    actual = sha256_file(path).lower()
    if actual != expected:
        raise EmbeddingError(
            f"checkpoint SHA-256 mismatch for {path}: "
            f"expected {expected}, got {actual}"
        )
    return {"path": path, "sha256": actual, "size_bytes": int(path.stat().st_size)}


def read_git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise EmbeddingError(
            f"could not read git HEAD for sn-reid root {repo_root}: {detail}"
        )
    commit = result.stdout.strip()
    if not commit:
        raise EmbeddingError(f"empty git HEAD for sn-reid root {repo_root}")
    return commit


def verify_sn_reid_root(
    sn_reid_root: str | Path, *, expected_commit: str
) -> dict[str, Any]:
    """Validate sn-reid checkout layout and HEAD commit before model import."""
    root = Path(sn_reid_root).expanduser().resolve()
    expected = str(expected_commit).strip().lower()
    if not expected:
        raise EmbeddingError("expected sn-reid commit must be non-empty")
    if not root.is_dir():
        raise EmbeddingError(f"sn-reid root not found or not a directory: {root}")
    torchreid_dir = root / "torchreid"
    if not torchreid_dir.is_dir():
        raise EmbeddingError(f"torchreid package directory missing under {root}")

    actual = read_git_head(root).lower()
    if actual != expected:
        raise EmbeddingError(
            f"sn-reid commit mismatch for {root}: expected {expected}, got {actual}"
        )
    return {"root": root, "commit": actual, "torchreid_dir": torchreid_dir}


@contextmanager
def temporary_sys_path_prepend(directory: Path) -> Iterator[None]:
    """Temporarily prepend a directory to ``sys.path`` for the current process."""
    entry = str(directory.resolve())
    inserted = False
    if entry not in sys.path:
        sys.path.insert(0, entry)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(entry)
            except ValueError:
                pass


def build_osnet_cpu_model(*, model_name: str = MODEL_NAME) -> Any:
    """Lazy-import torchreid and build OSNet on CPU with ``pretrained=False``."""
    from torchreid.models import build_model  # lazy: only when embedding actually runs

    if model_name != MODEL_NAME:
        raise EmbeddingError(
            f"unsupported model_name {model_name!r}; expected {MODEL_NAME!r}"
        )
    model = build_model(
        name=MODEL_NAME,
        num_classes=1,
        pretrained=False,
        use_gpu=False,
    )
    model.eval()
    model.cpu()
    return model


def load_osnet_checkpoint_weights(model: Any, checkpoint_path: str | Path) -> None:
    """Lazy-import torchreid weight loader and apply local checkpoint weights."""
    from torchreid.utils import load_pretrained_weights  # lazy

    path = Path(checkpoint_path).expanduser().resolve()
    load_pretrained_weights(model, str(path))
    model.eval()
    model.cpu()


def preprocess_crop_bgr(crop_bgr: np.ndarray) -> torch.Tensor:
    """Convert an OpenCV BGR crop to a CPU float32 tensor shaped (3, 256, 128)."""
    if not isinstance(crop_bgr, np.ndarray):
        raise EmbeddingError("crop must be a numpy array")
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        raise EmbeddingError(
            f"crop must be HxWx3, got shape {getattr(crop_bgr, 'shape', None)}"
        )
    if crop_bgr.size == 0 or crop_bgr.shape[0] <= 0 or crop_bgr.shape[1] <= 0:
        raise EmbeddingError("crop image is empty")
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = _CROP_TRANSFORM(pil)
    if not isinstance(tensor, torch.Tensor):
        raise EmbeddingError("preprocess transform did not return a torch.Tensor")
    tensor = tensor.to(dtype=torch.float32, device="cpu")
    if tuple(tensor.shape) != (3, RESIZE_HEIGHT, RESIZE_WIDTH):
        raise EmbeddingError(f"unexpected tensor shape {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise EmbeddingError("preprocessed tensor contains non-finite values")
    return tensor


def load_and_preprocess_crop(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise EmbeddingError(f"crop JPEG missing or not a file: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise EmbeddingError(f"could not decode crop JPEG: {path}")
    return preprocess_crop_bgr(image)


def resolve_crop_path(manifest_dir: Path, relative_path: Any) -> Path:
    """Resolve a crop relative path under the manifest directory with traversal guards."""
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise EmbeddingError(
            f"crop_relative_path must be a non-empty string, got {relative_path!r}"
        )
    rel = relative_path.replace("\\", "/")
    if rel.startswith("/") or Path(rel).is_absolute():
        raise EmbeddingError(
            f"absolute crop_relative_path is not allowed: {relative_path!r}"
        )
    parts = Path(rel).parts
    if any(part == ".." for part in parts):
        raise EmbeddingError(
            f"crop_relative_path must not contain '..': {relative_path!r}"
        )

    root = manifest_dir.expanduser().resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise EmbeddingError(
            f"crop path escapes manifest directory: {relative_path!r}"
        ) from exc
    if not candidate.is_file():
        raise EmbeddingError(
            f"crop JPEG missing or not a regular file: {relative_path}"
        )
    return candidate


def _assert_jpeg_decodable(path: Path) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise EmbeddingError(f"could not decode crop JPEG: {path}")
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise EmbeddingError(f"decoded crop JPEG is empty or invalid: {path}")


def load_crop_manifest_rows(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise EmbeddingError(f"crop manifest not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EmbeddingError(f"could not read crop manifest: {path}: {exc}") from exc

    if not text.strip():
        raise EmbeddingError("crop manifest is empty")

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line, parse_constant=_reject_non_finite_json)
        except EmbeddingError:
            raise
        except json.JSONDecodeError as exc:
            raise EmbeddingError(
                f"invalid JSON on crop manifest line {line_no}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise EmbeddingError(f"manifest line {line_no} must be a JSON object")
        try:
            validate_manifest_row(payload)
        except ReIDSchemaError as exc:
            raise EmbeddingError(f"manifest line {line_no}: {exc}") from exc
        rows.append(payload)

    if not rows:
        raise EmbeddingError("crop manifest is empty")

    crop_ids: set[str] = set()
    rel_paths: set[str] = set()
    manifest_dir = path.parent
    for row in rows:
        crop_id = str(row["crop_id"])
        rel = row["crop_relative_path"]
        if crop_id in crop_ids:
            raise EmbeddingError(f"duplicate crop_id in manifest: {crop_id}")
        if rel in rel_paths:
            raise EmbeddingError(f"duplicate crop_relative_path in manifest: {rel}")
        crop_ids.add(crop_id)
        rel_paths.add(str(rel))
        crop_path = resolve_crop_path(manifest_dir, rel)
        _assert_jpeg_decodable(crop_path)

    return rows


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if not isinstance(vectors, np.ndarray):
        raise EmbeddingError("embedding vectors must be a numpy array")
    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
        raise EmbeddingError(
            f"expected embedding shape (N, {EMBEDDING_DIM}), got {vectors.shape}"
        )
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32, copy=False)
    if not np.isfinite(vectors).all():
        raise EmbeddingError("embedding vectors contain NaN or Infinity")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all():
        raise EmbeddingError("embedding L2 norms contain NaN or Infinity")
    if np.any(norms <= 0.0):
        raise EmbeddingError("embedding row has non-positive L2 norm")
    normalized = (vectors / norms[:, None]).astype(np.float32, copy=False)
    if not np.isfinite(normalized).all():
        raise EmbeddingError("L2-normalized embeddings contain NaN or Infinity")
    return normalized


def _tensor_batch_to_embeddings(model: Any, batch: torch.Tensor) -> np.ndarray:
    if batch.device.type != "cpu":
        raise EmbeddingError("embedding batch must be on CPU")
    if batch.ndim != 4 or batch.shape[1:] != (3, RESIZE_HEIGHT, RESIZE_WIDTH):
        raise EmbeddingError(f"unexpected batch shape {tuple(batch.shape)}")
    if batch.dtype != torch.float32:
        raise EmbeddingError(f"batch dtype must be float32, got {batch.dtype}")

    with torch.inference_mode():
        output = model(batch)

    if not isinstance(output, torch.Tensor):
        raise EmbeddingError(
            f"model output must be a torch.Tensor, got {type(output)!r}"
        )
    if (
        output.ndim != 2
        or output.shape[0] != batch.shape[0]
        or output.shape[1] != EMBEDDING_DIM
    ):
        raise EmbeddingError(
            f"model output shape must be ({batch.shape[0]}, {EMBEDDING_DIM}), "
            f"got {tuple(output.shape)}"
        )
    array = output.detach().cpu().to(dtype=torch.float32).numpy()
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise EmbeddingError("model embeddings contain NaN or Infinity")
    return array


def embed_tensors(
    model: Any,
    tensors: Sequence[torch.Tensor],
    *,
    batch_size: int,
) -> np.ndarray:
    if not _is_positive_int(batch_size):
        raise EmbeddingError(f"batch_size must be a positive int, got {batch_size!r}")
    if not tensors:
        raise EmbeddingError("no crop tensors to embed")

    chunks: list[np.ndarray] = []
    for start in range(0, len(tensors), batch_size):
        piece = tensors[start : start + batch_size]
        batch = torch.stack(list(piece), dim=0).to(dtype=torch.float32, device="cpu")
        chunks.append(_tensor_batch_to_embeddings(model, batch))
    stacked = np.concatenate(chunks, axis=0)
    if stacked.shape != (len(tensors), EMBEDDING_DIM):
        raise EmbeddingError(
            f"concatenated embeddings shape {stacked.shape} != "
            f"({len(tensors)}, {EMBEDDING_DIM})"
        )
    return l2_normalize_rows(stacked)


def create_temp_embedding_dir(output_dir: Path) -> Path:
    final_dir = output_dir.expanduser().resolve()
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp_dir = parent / f"_tmp_reid_embeddings_{final_dir.name}_{token}"
    if tmp_dir.exists():
        raise EmbeddingError(f"temporary output path already exists: {tmp_dir}")
    tmp_dir.mkdir(parents=False, exist_ok=False)
    return tmp_dir


def finalize_embedding_dir(*, temp_dir: Path, final_dir: Path, overwrite: bool) -> Path:
    temp_path = temp_dir.expanduser().resolve()
    final_path = final_dir.expanduser().resolve()
    if not temp_path.is_dir():
        raise EmbeddingError(f"temporary output directory missing: {temp_path}")

    backup_path: Path | None = None
    try:
        if final_path.exists():
            if not overwrite:
                raise EmbeddingError(
                    f"output already exists: {final_path}; re-run with --overwrite"
                )
            backup_path = final_path.with_name(
                f"_backup_reid_embeddings_{final_path.name}_{uuid.uuid4().hex[:8]}"
            )
            os.rename(final_path, backup_path)

        os.rename(temp_path, final_path)

        if backup_path is not None and backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=False)
            backup_path = None
    except Exception:
        if backup_path is not None and backup_path.exists() and not final_path.exists():
            try:
                os.rename(backup_path, final_path)
                backup_path = None
            except OSError:
                pass
        raise

    parent = final_path.parent
    for stray in parent.glob(f"_tmp_reid_embeddings_{final_path.name}_*"):
        cleanup_dir(stray)
    for stray in parent.glob(f"_backup_reid_embeddings_{final_path.name}_*"):
        cleanup_dir(stray)

    return final_path


def write_embedding_artifacts(
    *,
    output_dir: Path,
    vectors: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    crop_manifest_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    sn_reid_root: Path,
    sn_reid_commit: str,
    batch_size: int,
    batch_count: int,
    elapsed_sec: float,
) -> dict[str, Any]:
    if vectors.shape != (len(rows), EMBEDDING_DIM):
        raise EmbeddingError("vectors/rows length mismatch while writing outputs")
    if vectors.dtype != np.float32:
        raise EmbeddingError("vectors must be float32 before writing")

    crop_ids = np.asarray([str(r["crop_id"]) for r in rows], dtype=np.str_)
    track_ids = np.asarray([int(r["track_id"]) for r in rows], dtype=np.int64)
    frame_indices = np.asarray([int(r["frame_index"]) for r in rows], dtype=np.int64)

    for name, array in (
        ("vectors", vectors),
        ("crop_ids", crop_ids),
        ("track_ids", track_ids),
        ("frame_indices", frame_indices),
    ):
        if array.dtype == object:
            raise EmbeddingError(f"NPZ array {name} must not use object dtype")

    npz_path = output_dir / NPZ_NAME
    np.savez(
        npz_path,
        vectors=vectors,
        crop_ids=crop_ids,
        track_ids=track_ids,
        frame_indices=frame_indices,
    )

    norms = np.linalg.norm(vectors, axis=1).astype(np.float64)
    index_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        norm = float(norms[i])
        if not math.isfinite(norm):
            raise EmbeddingError(f"non-finite l2_norm for row {i}")
        index_rows.append(
            {
                "crop_id": str(row["crop_id"]),
                "track_id": int(row["track_id"]),
                "frame_index": int(row["frame_index"]),
                "embedding_row": i,
                "embedding_shape": [EMBEDDING_DIM],
                "embedding_dtype": "float32",
                "l2_norm": norm,
                "model_name": MODEL_NAME,
                "checkpoint_sha256": checkpoint_sha256,
                "preprocessing": dict(PREPROCESSING),
                "schema_version": INDEX_SCHEMA_VERSION,
            }
        )
    write_manifest_jsonl(output_dir / INDEX_NAME, index_rows)

    summary = {
        "status": "ok",
        "crop_manifest": str(crop_manifest_path),
        "crop_count": len(rows),
        "embeddings_shape": [int(vectors.shape[0]), int(vectors.shape[1])],
        "embedding_dtype": "float32",
        "embedding_l2_norm_min": float(np.min(norms)),
        "embedding_l2_norm_max": float(np.max(norms)),
        "batch_size": int(batch_size),
        "batch_count": int(batch_count),
        "device": "cpu",
        "model_name": MODEL_NAME,
        "checkpoint_type": "general_person_reid",
        "checkpoint_training_dataset": "Market1501",
        "checkpoint_soccerNet_trained": False,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "pretrained_flag": False,
        "feature_extractor_used": False,
        "automatic_download_occurred": False,
        "preprocessing": dict(PREPROCESSING),
        "sn_reid_root": str(sn_reid_root),
        "sn_reid_commit": sn_reid_commit,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "numpy_version": str(np.__version__),
        "elapsed_sec": float(elapsed_sec),
        "schema_version": SUMMARY_SCHEMA_VERSION,
    }
    for key, value in summary.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise EmbeddingError(f"summary field {key} is not finite")

    summary_path = output_dir / SUMMARY_NAME
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )

    names = sorted(p.name for p in output_dir.iterdir())
    expected = sorted([NPZ_NAME, INDEX_NAME, SUMMARY_NAME])
    if names != expected:
        raise EmbeddingError(f"unexpected files in embedding output: {names}")

    return {
        "npz_path": str(npz_path),
        "index_path": str(output_dir / INDEX_NAME),
        "summary_path": str(summary_path),
        "summary": summary,
        "index_rows": index_rows,
    }


def run_extract_reid_embeddings(
    *,
    crop_manifest: str | Path,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    sn_reid_root: str | Path,
    expected_sn_reid_commit: str,
    output_dir: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
    model_builder: Callable[..., Any] | None = None,
    weight_loader: Callable[[Any, str | Path], None] | None = None,
) -> dict[str, Any]:
    """Embed crops from a manifest and write NPZ/JSONL/summary atomically.

    ``model_builder`` / ``weight_loader`` are injectable for unit tests. When
    omitted, torchreid is imported lazily only after checkpoint and sn-reid
    root checks succeed.
    """
    if not _is_positive_int(batch_size):
        raise EmbeddingError(f"batch_size must be a positive int, got {batch_size!r}")

    manifest_path = Path(crop_manifest).expanduser().resolve()
    final_dir = Path(output_dir).expanduser().resolve()
    check_output_collision(final_dir, overwrite=overwrite)

    # Validate manifest + JPEG paths before checkpoint/model work.
    rows = load_crop_manifest_rows(manifest_path)

    checkpoint_info = verify_checkpoint(checkpoint, expected_sha256=checkpoint_sha256)
    sn_info = verify_sn_reid_root(
        sn_reid_root, expected_commit=expected_sn_reid_commit
    )

    builder = model_builder or build_osnet_cpu_model
    loader = weight_loader or load_osnet_checkpoint_weights

    started = time.perf_counter()
    temp_dir: Path | None = None
    try:
        temp_dir = create_temp_embedding_dir(final_dir)

        with temporary_sys_path_prepend(sn_info["root"]):
            model = builder(model_name=MODEL_NAME)
            loader(model, checkpoint_info["path"])

        manifest_dir = manifest_path.parent
        tensors = [
            load_and_preprocess_crop(
                resolve_crop_path(manifest_dir, row["crop_relative_path"])
            )
            for row in rows
        ]
        batch_count = (len(tensors) + batch_size - 1) // batch_size
        vectors = embed_tensors(model, tensors, batch_size=batch_size)
        elapsed = time.perf_counter() - started

        artifacts = write_embedding_artifacts(
            output_dir=temp_dir,
            vectors=vectors,
            rows=rows,
            crop_manifest_path=manifest_path,
            checkpoint_path=checkpoint_info["path"],
            checkpoint_sha256=checkpoint_info["sha256"],
            sn_reid_root=sn_info["root"],
            sn_reid_commit=sn_info["commit"],
            batch_size=batch_size,
            batch_count=batch_count,
            elapsed_sec=elapsed,
        )

        finalized = finalize_embedding_dir(
            temp_dir=temp_dir, final_dir=final_dir, overwrite=overwrite
        )
        temp_dir = None
    except Exception:
        cleanup_dir(temp_dir)
        raise

    summary = artifacts["summary"]
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "status": "ok",
        "output_dir": str(finalized),
        "npz_path": str(finalized / NPZ_NAME),
        "index_path": str(finalized / INDEX_NAME),
        "summary_path": str(finalized / SUMMARY_NAME),
        "crop_count": len(rows),
        "batch_size": batch_size,
        "batch_count": batch_count,
        "embeddings_shape": [len(rows), EMBEDDING_DIM],
        "embedding_dtype": "float32",
        "embedding_l2_norm_min": float(np.min(norms)),
        "embedding_l2_norm_max": float(np.max(norms)),
        "model_name": MODEL_NAME,
        "checkpoint_path": str(checkpoint_info["path"]),
        "checkpoint_sha256": checkpoint_info["sha256"],
        "sn_reid_root": str(sn_info["root"]),
        "sn_reid_commit": sn_info["commit"],
        "elapsed_sec": float(summary["elapsed_sec"]),
        "device": "cpu",
        "summary": summary,
    }
