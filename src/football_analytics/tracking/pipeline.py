"""Orchestrate CPU person tracking with Ultralytics ByteTrack (single stream)."""

from __future__ import annotations

import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import yaml

from football_analytics.detection.annotate import sanitize_detection
from football_analytics.ingest.checksum import sha256_file
from football_analytics.ingest.validate import ValidationError, validate_video_path
from football_analytics.tracking.annotate import draw_tracks
from football_analytics.tracking.writers import (
    WritersError,
    check_output_collision,
    cleanup_partial_outputs,
    finalize_outputs,
    output_paths,
    write_json_file,
    write_jsonl_lines,
)

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.70
DEFAULT_IMGSZ = 640
DEFAULT_MODEL_PATH = "models/yolo11n.pt"
DEFAULT_TRACKER_PATH = "configs/tracking/bytetrack_stage3.yaml"

ModelLoader = Callable[[Path], Any]


class TrackingError(RuntimeError):
    """Raised when the tracking pipeline fails."""


def load_yolo_model(model_path: Path) -> Any:
    """Load a local YOLO weights file. Never downloads. Fresh instance per call."""
    path = model_path.expanduser().resolve()
    if not path.is_file():
        raise TrackingError(
            f"model file not found: {path}. "
            "Download is not automatic; place weights at this path after explicit approval."
        )

    from ultralytics import YOLO

    model = YOLO(str(path))
    _assert_person_class(model)
    return model


def load_tracker_config(tracker_path: Path) -> dict[str, Any]:
    """Load and validate the project-local ByteTrack YAML (no site-packages fallback)."""
    path = tracker_path.expanduser().resolve()
    if not path.is_file():
        raise TrackingError(
            f"tracker config not found: {path}. "
            "Expected project-local configs/tracking/bytetrack_stage3.yaml"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TrackingError(f"tracker config must be a mapping: {path}")
    if payload.get("tracker_type") != "bytetrack":
        raise TrackingError(
            f"tracker_type must be 'bytetrack', got {payload.get('tracker_type')!r}"
        )
    return payload


def run_tracking(
    video: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    tracker_path: str | Path = DEFAULT_TRACKER_PATH,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    max_frames: int | None = None,
    overwrite: bool = False,
    model_loader: ModelLoader | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """
    Track COCO person (class 0) on CPU for a single continuous video stream.

    Each call creates or receives a fresh model instance. Tracker state is not
    shared across pipeline runs. Prefer injecting ``model`` only in unit tests.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    paths = output_paths(out_dir)
    writer = None
    capture = None

    try:
        video_path = validate_video_path(Path(video))
        check_output_collision(out_dir, overwrite=overwrite)
        out_dir.mkdir(parents=True, exist_ok=True)
        cleanup_partial_outputs(out_dir)

        weights_path = Path(model_path).expanduser().resolve()
        tracker_cfg_path = Path(tracker_path).expanduser().resolve()
        tracker_params = load_tracker_config(tracker_cfg_path)
        tracker_sha256 = sha256_file(tracker_cfg_path)

        if model is not None:
            loaded_model = model
            _assert_person_class(loaded_model)
        else:
            loader = model_loader or load_yolo_model
            loaded_model = loader(weights_path)
            _assert_person_class(loaded_model)

        video_sha256 = sha256_file(video_path)
        model_sha256 = sha256_file(weights_path) if weights_path.is_file() else None
        ultralytics_version, torch_version = _environment_versions()

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise TrackingError(f"OpenCV could not open video: {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0:
            raise TrackingError(
                f"invalid video properties width={width} height={height} fps={fps}"
            )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(paths["tmp_tracked"]),
            fourcc,
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise TrackingError(
                f"OpenCV could not open VideoWriter: {paths['tmp_tracked']}"
            )

        track_rows: list[dict[str, Any]] = []
        frames_processed = 0
        frames_with_obs = 0
        skipped_invalid = 0
        # track_id -> list of frame indices where observed
        track_frames: dict[int, list[int]] = {}
        started = time.perf_counter()

        if max_frames is not None and max_frames <= 0:
            raise TrackingError("--max-frames must be a positive integer")

        while True:
            if max_frames is not None and frames_processed >= max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break

            frame_index = frames_processed
            timestamp_sec = frame_index / fps

            results = loaded_model.track(
                source=frame,
                device="cpu",
                classes=[0],
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                tracker=str(tracker_cfg_path),
                persist=True,
                save=False,
                verbose=False,
            )

            frame_obs: list[dict[str, Any]] = []
            result = results[0] if results else None
            boxes = getattr(result, "boxes", None) if result is not None else None
            if boxes is not None and len(boxes) > 0:
                xyxy_list = _to_list2d(boxes.xyxy)
                conf_list = _to_list1d(boxes.conf)
                cls_list = _to_list1d(boxes.cls)
                id_list = _extract_track_ids(boxes, expected=len(xyxy_list))

                for bbox, score, cls_id, track_id in zip(
                    xyxy_list, conf_list, cls_list, id_list
                ):
                    if int(cls_id) != 0:
                        continue
                    sanitized = sanitize_detection(
                        bbox_xyxy=bbox,
                        confidence=score,
                        frame_width=width,
                        frame_height=height,
                    )
                    if sanitized is None:
                        skipped_invalid += 1
                        continue
                    row = {
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp_sec,
                        "track_id": track_id,
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": sanitized["confidence"],
                        "bbox_xyxy": sanitized["bbox_xyxy"],
                    }
                    track_rows.append(row)
                    frame_obs.append(row)
                    if track_id is not None:
                        track_frames.setdefault(track_id, []).append(frame_index)

            if frame_obs:
                frames_with_obs += 1

            annotated = draw_tracks(frame, frame_obs)
            writer.write(annotated)
            frames_processed += 1

        elapsed_sec = time.perf_counter() - started
        avg_fps = (frames_processed / elapsed_sec) if elapsed_sec > 0 else 0.0

        writer.release()
        writer = None
        capture.release()
        capture = None

        with_id = sum(1 for r in track_rows if r["track_id"] is not None)
        without_id = sum(1 for r in track_rows if r["track_id"] is None)
        observation_stats, span_stats = _track_stats(track_frames)

        write_jsonl_lines(paths["tmp_tracks"], track_rows)

        summary = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": {
                "path": str(video_path),
                "filename": video_path.name,
                "sha256": video_sha256,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count_source": source_frame_count if source_frame_count > 0 else None,
            },
            "model": {
                "name": weights_path.name,
                "path": str(weights_path),
                "sha256": model_sha256,
            },
            "environment": {
                "ultralytics_version": ultralytics_version,
                "torch_version": torch_version,
            },
            "parameters": {
                "device": "cpu",
                "classes": [0],
                "conf": conf,
                "iou": iou,
                "detection_iou": iou,
                "imgsz": imgsz,
                "persist": True,
                "save": False,
                "verbose": False,
                "max_frames": max_frames,
            },
            "tracker": {
                "path": str(tracker_cfg_path),
                "sha256": tracker_sha256,
                "parameters": tracker_params,
            },
            "metrics": {
                "frames_processed": frames_processed,
                "total_box_observations": len(track_rows),
                "frames_with_box_observations": frames_with_obs,
                "box_observations_with_track_id": with_id,
                "box_observations_without_track_id": without_id,
                "unique_track_ids": len(track_frames),
                "track_observation_count": observation_stats,
                "track_span_frames": span_stats,
                "skipped_invalid_detections": skipped_invalid,
                "elapsed_sec": elapsed_sec,
                "avg_fps": avg_fps,
            },
            "outputs": {
                "tracked_video": str(paths["tracked"]),
                "tracks_jsonl": str(paths["tracks"]),
                "summary_json": str(paths["summary"]),
            },
            "notes": [
                "total_box_observations counts boxes returned by model.track after sanitization; not raw detector-only counts.",
                "Detector-to-tracker drop counts are not invented; only observed track output is reported.",
                "detection_iou is YOLO NMS IoU; tracker match_thresh is ByteTrack association — different purposes.",
                "No ground-truth IDs; ID-switch rate, MOTA, and HOTA are not reported.",
                "One pipeline run processes a single continuous video stream; tracker state is not reused across runs.",
                "Tracked video is silent; source audio is not copied.",
            ],
            "status": "ok",
            "errors": [],
        }
        write_json_file(paths["tmp_summary"], summary)

        finals = finalize_outputs(out_dir)
        return {
            "tracked_path": str(finals["tracked"]),
            "tracks_path": str(finals["tracks"]),
            "summary_path": str(finals["summary"]),
            "summary": summary,
        }
    except (ValidationError, WritersError, TrackingError) as exc:
        _safe_release(writer, capture)
        cleanup_partial_outputs(out_dir)
        if isinstance(exc, TrackingError):
            raise
        raise TrackingError(str(exc)) from exc
    except Exception as exc:
        _safe_release(writer, capture)
        cleanup_partial_outputs(out_dir)
        raise TrackingError(str(exc)) from exc


def _extract_track_ids(boxes: Any, *, expected: int) -> list[int | None]:
    """Return per-box track IDs using is_track / id guards; never invent IDs."""
    is_track = bool(getattr(boxes, "is_track", False))
    ids = getattr(boxes, "id", None)
    if (not is_track) or ids is None:
        return [None] * expected

    id_list = _to_optional_int_list(ids)
    if len(id_list) != expected:
        # Misaligned id tensor: treat as missing rather than inventing or crashing.
        return [None] * expected
    return id_list


def _to_optional_int_list(value: Any) -> list[int | None]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        data = value.tolist()
    else:
        data = list(value)
    result: list[int | None] = []
    for item in data:
        try:
            number = float(item)
        except (TypeError, ValueError):
            result.append(None)
            continue
        if math.isnan(number):
            result.append(None)
            continue
        result.append(int(number))
    return result


def _track_stats(
    track_frames: dict[int, list[int]],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    if not track_frames:
        empty = {"min": None, "max": None, "mean": None, "median": None}
        return empty, dict(empty)

    observation_counts = [len(frames) for frames in track_frames.values()]
    spans = [
        (max(frames) - min(frames) + 1) for frames in track_frames.values() if frames
    ]
    return _summary_stats(observation_counts), _summary_stats(spans)


def _summary_stats(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
    }


def _assert_person_class(model: Any) -> None:
    names = getattr(model, "names", None)
    if names is None:
        raise TrackingError("model.names is missing")
    person_name = names[0] if isinstance(names, dict) else names[0]
    if person_name != "person":
        raise TrackingError(
            f"model.names[0] must be 'person', got {person_name!r}"
        )


def _environment_versions() -> tuple[str | None, str | None]:
    try:
        import ultralytics

        ultra_ver = getattr(ultralytics, "__version__", None)
    except Exception:
        ultra_ver = None
    try:
        import torch

        torch_ver = getattr(torch, "__version__", None)
    except Exception:
        torch_ver = None
    return ultra_ver, torch_ver


def _safe_release(writer: Any, capture: Any) -> None:
    if writer is not None:
        try:
            writer.release()
        except Exception:
            pass
    if capture is not None:
        try:
            capture.release()
        except Exception:
            pass


def _to_list2d(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        data = value.tolist()
    else:
        data = list(value)
    return [list(map(float, row)) for row in data]


def _to_list1d(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        data = value.tolist()
    else:
        data = list(value)
    return [float(x) for x in data]
