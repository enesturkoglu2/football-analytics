"""Orchestrate CPU person detection without tracking."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2

from football_analytics.detection.annotate import draw_detections, sanitize_detection
from football_analytics.detection.writers import (
    WritersError,
    check_output_collision,
    cleanup_partial_outputs,
    finalize_outputs,
    output_paths,
    write_json_file,
    write_jsonl_lines,
)
from football_analytics.ingest.checksum import sha256_file
from football_analytics.ingest.validate import ValidationError, validate_video_path

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.70
DEFAULT_IMGSZ = 640
DEFAULT_MODEL_PATH = "models/yolo11n.pt"

ModelLoader = Callable[[Path], Any]


class DetectionError(RuntimeError):
    """Raised when the detection pipeline fails."""


def load_yolo_model(model_path: Path) -> Any:
    """
    Load a local YOLO weights file. Never downloads.

    Requires an existing file at model_path. Validates names[0] == 'person'.
    """
    path = model_path.expanduser().resolve()
    if not path.is_file():
        raise DetectionError(
            f"model file not found: {path}. "
            "Download is not automatic; place weights at this path after explicit approval."
        )

    # Import only when loading a real model so unit tests never need ultralytics.
    from ultralytics import YOLO

    model = YOLO(str(path))
    names = getattr(model, "names", None)
    if names is None:
        raise DetectionError("model.names is missing")
    person_name = names[0] if isinstance(names, dict) else names[0]
    if person_name != "person":
        raise DetectionError(
            f"model.names[0] must be 'person', got {person_name!r}"
        )
    return model


def run_detection(
    video: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    max_frames: int | None = None,
    overwrite: bool = False,
    model_loader: ModelLoader | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """
    Detect COCO person (class 0) on CPU and write annotated video + JSONL + summary.

    Provide ``model`` (injected) or ``model_loader`` for tests. If both are None,
    ``load_yolo_model`` is used and requires a local weights file (no download).
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
        if model is not None:
            loaded_model = model
            _assert_person_class(loaded_model)
        else:
            loader = model_loader or load_yolo_model
            loaded_model = loader(weights_path)
            _assert_person_class(loaded_model)

        video_sha256 = sha256_file(video_path)
        if weights_path.is_file():
            model_sha256 = sha256_file(weights_path)
        else:
            # Injected models in tests may not have a real weights file.
            model_sha256 = None

        ultralytics_version, torch_version = _environment_versions()

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise DetectionError(f"OpenCV could not open video: {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0:
            raise DetectionError(
                f"invalid video properties width={width} height={height} fps={fps}"
            )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(paths["tmp_annotated"]),
            fourcc,
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise DetectionError(
                f"OpenCV could not open VideoWriter: {paths['tmp_annotated']}"
            )

        detection_rows: list[dict[str, Any]] = []
        frames_processed = 0
        frames_with_detections = 0
        total_detections = 0
        skipped_invalid = 0
        started = time.perf_counter()

        frame_limit = max_frames if max_frames is not None else None
        if frame_limit is not None and frame_limit <= 0:
            raise DetectionError("--max-frames must be a positive integer")

        while True:
            if frame_limit is not None and frames_processed >= frame_limit:
                break
            ok, frame = capture.read()
            if not ok:
                break

            frame_index = frames_processed
            timestamp_sec = frame_index / fps

            results = loaded_model.predict(
                source=frame,
                device="cpu",
                classes=[0],
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                save=False,
                verbose=False,
            )
            frame_dets: list[dict[str, Any]] = []
            result = results[0] if results else None
            boxes = getattr(result, "boxes", None) if result is not None else None
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy
                confs = boxes.conf
                cls_ids = boxes.cls
                # Support torch tensors or plain lists from mocks.
                xyxy_list = _to_list2d(xyxy)
                conf_list = _to_list1d(confs)
                cls_list = _to_list1d(cls_ids)
                for bbox, score, cls_id in zip(xyxy_list, conf_list, cls_list):
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
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": sanitized["confidence"],
                        "bbox_xyxy": sanitized["bbox_xyxy"],
                    }
                    detection_rows.append(row)
                    frame_dets.append(row)

            if frame_dets:
                frames_with_detections += 1
                total_detections += len(frame_dets)

            annotated = draw_detections(frame, frame_dets)
            writer.write(annotated)
            frames_processed += 1

        elapsed_sec = time.perf_counter() - started
        avg_fps = (frames_processed / elapsed_sec) if elapsed_sec > 0 else 0.0
        avg_det = (
            (total_detections / frames_processed) if frames_processed > 0 else 0.0
        )

        writer.release()
        writer = None
        capture.release()
        capture = None

        write_jsonl_lines(paths["tmp_detections"], detection_rows)

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
                "imgsz": imgsz,
                "max_frames": max_frames,
                "save": False,
                "verbose": False,
            },
            "metrics": {
                "frames_processed": frames_processed,
                "total_detections": total_detections,
                "frames_with_detections": frames_with_detections,
                "avg_detections_per_frame": avg_det,
                "skipped_invalid_detections": skipped_invalid,
                "elapsed_sec": elapsed_sec,
                "avg_fps": avg_fps,
            },
            "outputs": {
                "annotated_video": str(paths["annotated"]),
                "detections_jsonl": str(paths["detections"]),
                "summary_json": str(paths["summary"]),
            },
            "notes": [
                "Annotated video is silent; source audio is not copied.",
            ],
            "status": "ok",
            "errors": [],
        }
        write_json_file(paths["tmp_summary"], summary)

        finals = finalize_outputs(out_dir)
        return {
            "annotated_path": str(finals["annotated"]),
            "detections_path": str(finals["detections"]),
            "summary_path": str(finals["summary"]),
            "summary": summary,
        }
    except (ValidationError, WritersError, DetectionError) as exc:
        _safe_release(writer, capture)
        cleanup_partial_outputs(out_dir)
        if isinstance(exc, DetectionError):
            raise
        raise DetectionError(str(exc)) from exc
    except Exception as exc:
        _safe_release(writer, capture)
        cleanup_partial_outputs(out_dir)
        raise DetectionError(str(exc)) from exc


def _assert_person_class(model: Any) -> None:
    names = getattr(model, "names", None)
    if names is None:
        raise DetectionError("model.names is missing")
    person_name = names[0] if isinstance(names, dict) else names[0]
    if person_name != "person":
        raise DetectionError(
            f"model.names[0] must be 'person', got {person_name!r}"
        )


def _environment_versions() -> tuple[str | None, str | None]:
    ultra_ver: str | None
    torch_ver: str | None
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
