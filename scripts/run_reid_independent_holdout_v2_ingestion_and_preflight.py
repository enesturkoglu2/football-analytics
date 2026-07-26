#!/usr/bin/env python3
"""Stage 5D-F3I — Independent holdout v2 ingestion and preflight.

Validates holdout technical integrity and independence versus sample.mp4 and
external enrollment via deterministic frame fingerprints. No detection,
tracking, crops, embeddings, scoring, GT, threshold, or identity assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.ingest.ffprobe import (  # noqa: E402
    parse_frame_rate,
    run_ffprobe,
    select_video_stream,
)

CONFIG_SCHEMA = "reid_independent_holdout_v2_ingestion_config_v1"
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FINAL_STATUS = (
    "COMPLETED_STAGE5D_F3I_TARGET_001_NEW_INDEPENDENT_HOLDOUT_INGESTED_AND_PREFLIGHT_PASSED"
)
READINESS = (
    "TARGET_001_INDEPENDENT_HOLDOUT_V2_READY_FOR_LABEL_BLIND_UNIVERSE_BUILD"
)
NEXT_GATE = (
    "STAGE5D-F3J_TARGET_001_NEW_INDEPENDENT_HOLDOUT_LABEL_BLIND_DETECTION_TRACKING_AND_SEGMENT_UNIVERSE_BUILD"
)
ALLOWED_DIRTY = {
    "scripts/run_reid_independent_holdout_v2_ingestion_and_preflight.py",
    "configs/reid/independent_holdout_v2_ingestion_stage5d_target_001.yaml",
    "tests/test_reid_independent_holdout_v2_ingestion_and_preflight.py",
    "docs/setup/stage5d-target-new-independent-holdout-ingestion-and-preflight.md",
}


class HoldoutPreflightError(RuntimeError):
    pass


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def listing_sha(root: Path) -> tuple[int, str]:
    files: list[tuple[str, int, str]] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            path = Path(dp) / fn
            rel = str(path.relative_to(root)).replace("\\", "/")
            files.append((rel, path.stat().st_size, sha256_file(path)))
    files.sort()
    blob = "\n".join(f"{a}\t{b}\t{c}" for a, b, c in files).encode()
    return len(files), hashlib.sha256(blob).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise HoldoutPreflightError("unexpected config schema")
    if not config.get("offline_required"):
        raise HoldoutPreflightError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise HoldoutPreflightError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise HoldoutPreflightError("BLOCKED_STAGE5D_F3I_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise HoldoutPreflightError("BLOCKED_STAGE5D_F3I_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            # Ignore untracked holdout mp4 if somehow visible; normally gitignored.
            if path.endswith("target_001_independent_holdout_v2.mp4"):
                continue
            if path not in ALLOWED_DIRTY:
                raise HoldoutPreflightError(
                    "BLOCKED_STAGE5D_F3I_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise HoldoutPreflightError("BLOCKED_STAGE5D_F3I_GIT_CONTRACT_MISMATCH message")
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_SCORING_DESIGN_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not listing.is_file() or not manifest.is_file():
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_SCORING_DESIGN_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(man_payload.get("sha256") or man_payload.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_SCORING_DESIGN_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def parse_fps_fraction(value: str | None) -> tuple[int, int, float]:
    text = str(value or "").strip()
    if "/" in text:
        a, b = text.split("/", 1)
        num, den = int(a), int(b)
        if den <= 0:
            raise HoldoutPreflightError("invalid fps fraction")
        return num, den, float(num) / float(den)
    rate = float(text)
    if rate <= 0:
        raise HoldoutPreflightError("invalid fps")
    return int(round(rate)), 1, rate


def validate_f3h(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = config["stage5d_f3h_package"]
    root = project_root / cfg["path"]
    summary = load_json(root / "stage5d_f3h_summary.json")
    if summary.get("final_status") != cfg["expected_final_status"]:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_SCORING_DESIGN_CONTRACT_MISMATCH status"
        )
    if summary.get("primary_formula") != cfg["expected_primary_formula"]:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_SCORING_DESIGN_CONTRACT_MISMATCH formula"
        )
    checks = {
        "target_gallery_v2_members": cfg["expected_target_members"],
        "distractor_gallery_v1_members": cfg["expected_distractor_members"],
        "primary_target_top_k": cfg["expected_primary_target_top_k"],
        "primary_distractor_top_k": cfg["expected_primary_distractor_top_k"],
        "secondary_top_k": cfg["expected_secondary_top_k"],
        "query_scoring_rows": 0,
        "rankings": 0,
        "metrics": 0,
        "identity_assignments": 0,
        "sample_score_row_read_count": 0,
    }
    for key, expected in checks.items():
        if int(summary.get(key) or 0) != int(expected):
            raise HoldoutPreflightError(
                f"BLOCKED_STAGE5D_F3I_SCORING_DESIGN_CONTRACT_MISMATCH {key}"
            )
    if summary.get("threshold_selected") is not False:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_SCORING_DESIGN_CONTRACT_MISMATCH threshold"
        )
    if summary.get("gallery_mutation") is not False:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_SCORING_DESIGN_CONTRACT_MISMATCH gallery_mutation"
        )
    if summary.get("sample_video_read") is not False:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_SCORING_DESIGN_CONTRACT_MISMATCH sample_reads"
        )
    if summary.get("new_independent_holdout_required") is not True:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_SCORING_DESIGN_CONTRACT_MISMATCH holdout_required"
        )
    if summary.get("holdout_input_pending") is not True:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_SCORING_DESIGN_CONTRACT_MISMATCH holdout_pending"
        )
    primary = load_json(
        root
        / "scoring"
        / "target_001_target_distractor_primary_scoring_contract.json"
    )
    if primary.get("formula_id") != "TARGET_DISTRACTOR_MAX_MARGIN":
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_SCORING_DESIGN_CONTRACT_MISMATCH primary_id"
        )
    if primary.get("S_primary", {}).get("formula") != "T_max(q) - D_max(q)":
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_SCORING_DESIGN_CONTRACT_MISMATCH primary_formula"
        )
    snap = resolve_snapshot_sha(
        Path(cfg["snapshot_path"]), cfg["expected_snapshot_sha256"]
    )
    return {"root": root, "summary": summary, "primary": primary, "snapshot_sha256": snap}


def validate_reference(
    project_root: Path, rel: str, exp: Mapping[str, Any], *, kind: str
) -> dict[str, Any]:
    assert_no_path_traversal(rel)
    path = project_root / rel
    if not path.is_file() or path.is_symlink():
        raise HoldoutPreflightError(
            f"BLOCKED_STAGE5D_F3I_REFERENCE_SOURCE_CONTRACT_MISMATCH {kind}_missing"
        )
    if path.stat().st_size != int(exp["expected_bytes"]):
        raise HoldoutPreflightError(
            f"BLOCKED_STAGE5D_F3I_REFERENCE_SOURCE_CONTRACT_MISMATCH {kind}_bytes"
        )
    digest = sha256_file(path)
    if digest != exp["expected_sha256"]:
        raise HoldoutPreflightError(
            f"BLOCKED_STAGE5D_F3I_REFERENCE_SOURCE_CONTRACT_MISMATCH {kind}_sha"
        )
    payload = run_ffprobe(path)
    video = select_video_stream(payload)
    frames = int(video.get("nb_frames") or 0)
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    fps = parse_frame_rate(str(video.get("r_frame_rate") or video.get("avg_frame_rate")))
    fdur = float((payload.get("format") or {}).get("duration") or 0.0)
    if frames != int(exp["expected_frames"]):
        raise HoldoutPreflightError(
            f"BLOCKED_STAGE5D_F3I_REFERENCE_SOURCE_CONTRACT_MISMATCH {kind}_frames"
        )
    if width != int(exp["expected_width"]) or height != int(exp["expected_height"]):
        raise HoldoutPreflightError(
            f"BLOCKED_STAGE5D_F3I_REFERENCE_SOURCE_CONTRACT_MISMATCH {kind}_wh"
        )
    if fps is None or abs(float(fps) - float(exp["expected_fps"])) > 1e-6:
        raise HoldoutPreflightError(
            f"BLOCKED_STAGE5D_F3I_REFERENCE_SOURCE_CONTRACT_MISMATCH {kind}_fps"
        )
    if abs(fdur - float(exp["expected_format_duration_sec"])) > 1e-3:
        raise HoldoutPreflightError(
            f"BLOCKED_STAGE5D_F3I_REFERENCE_SOURCE_CONTRACT_MISMATCH {kind}_dur"
        )
    if kind == "external":
        vdur = float(video.get("duration") or 0.0)
        if abs(vdur - float(exp["expected_video_duration_sec"])) > 1e-3:
            raise HoldoutPreflightError(
                "BLOCKED_STAGE5D_F3I_REFERENCE_SOURCE_CONTRACT_MISMATCH external_vdur"
            )
    return {
        "path": rel,
        "absolute_path": str(path.resolve()),
        "bytes": int(exp["expected_bytes"]),
        "sha256": digest,
        "frames": frames,
        "width": width,
        "height": height,
        "fps": float(fps),
        "format_duration_sec": fdur,
    }


def validate_holdout_input(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = config["holdout_source"]
    rel = cfg["path"]
    assert_no_path_traversal(rel)
    path = project_root / rel
    if not path.exists() or not path.is_file() or not os.access(path, os.R_OK):
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_HOLDOUT_INPUT_MISSING_OR_UNREADABLE"
        )
    if path.is_symlink():
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_HOLDOUT_INPUT_MISSING_OR_UNREADABLE symlink"
        )
    if path.name != cfg["exact_filename"]:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_HOLDOUT_INPUT_MISSING_OR_UNREADABLE filename"
        )
    if path.suffix.lower() != ".mp4":
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_HOLDOUT_INPUT_MISSING_OR_UNREADABLE extension"
        )
    resolved = path.resolve()
    clips_root = (project_root / "data" / "test_clips").resolve()
    try:
        resolved.relative_to(clips_root)
    except ValueError as exc:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_HOLDOUT_INPUT_MISSING_OR_UNREADABLE realpath"
        ) from exc
    size = path.stat().st_size
    if size <= 0:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_HOLDOUT_INPUT_MISSING_OR_UNREADABLE empty"
        )
    return {
        "path": rel,
        "absolute_path": str(resolved),
        "bytes": size,
        "exists": True,
        "regular_file": True,
        "readable": True,
        "symlink": False,
    }


def extract_holdout_metadata(path: Path) -> dict[str, Any]:
    payload = run_ffprobe(path)
    streams = payload.get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not video_streams:
        raise HoldoutPreflightError("no video stream")
    ambiguity = len(video_streams) > 1
    video = video_streams[0]
    abs_index = next(
        i for i, s in enumerate(streams) if s.get("codec_type") == "video"
    )
    fmt = payload.get("format") or {}
    fps_num, fps_den, fps_float = parse_fps_fraction(
        str(video.get("r_frame_rate") or video.get("avg_frame_rate"))
    )
    avg_fps = parse_frame_rate(str(video.get("avg_frame_rate")))
    vfr = False
    if avg_fps is not None and abs(avg_fps - fps_float) > 0.05:
        vfr = True
    nb_frames = video.get("nb_frames")
    frame_count = int(nb_frames) if nb_frames not in (None, "N/A", "") else None
    tags = video.get("tags") or {}
    fmt_tags = fmt.get("tags") or {}
    rotation = tags.get("rotate") or fmt_tags.get("rotate")
    creation = tags.get("creation_time") or fmt_tags.get("creation_time")
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    video_duration = float(video.get("duration") or duration or 0.0)
    if width <= 0 or height <= 0 or fps_float <= 0 or duration <= 0:
        raise HoldoutPreflightError("invalid technical metadata")
    return {
        "ffprobe_raw": payload,
        "canonical": {
            "codec": video.get("codec_name"),
            "codec_profile": video.get("profile"),
            "codec_tag": video.get("codec_tag_string"),
            "pixel_format": video.get("pix_fmt"),
            "width": width,
            "height": height,
            "sample_aspect_ratio": video.get("sample_aspect_ratio"),
            "display_aspect_ratio": video.get("display_aspect_ratio"),
            "r_frame_rate": video.get("r_frame_rate"),
            "avg_frame_rate": video.get("avg_frame_rate"),
            "fps_numerator": fps_num,
            "fps_denominator": fps_den,
            "fps_float": fps_float,
            "time_base": video.get("time_base"),
            "start_pts": video.get("start_pts"),
            "start_time": video.get("start_time"),
            "duration_ts": video.get("duration_ts"),
            "video_duration": video_duration,
            "format_duration": duration,
            "format_name": fmt.get("format_name"),
            "format_long_name": fmt.get("format_long_name"),
            "bit_rate": fmt.get("bit_rate"),
            "stream_count": len(streams),
            "video_stream_count": len(video_streams),
            "audio_stream_count": len(audio_streams),
            "audio_present": len(audio_streams) > 0,
            "nb_frames_metadata": frame_count,
            "variable_frame_rate_detected": vfr,
            "primary_video_stream_index": abs_index,
            "multiple_video_stream_ambiguity": ambiguity,
            "rotation_metadata": rotation,
            "creation_time_metadata": creation,
        },
    }


def resolve_frame_count_by_decode(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HoldoutPreflightError("cannot open for frame count")
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        count += 1
    cap.release()
    if count <= 0:
        raise HoldoutPreflightError("decoded frame count zero")
    return count


def full_decode_integrity(path: Path, expected_frames: int) -> dict[str, Any]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    ended = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stderr = completed.stderr or ""
    err_lines = [ln for ln in stderr.splitlines() if ln.strip()]
    # Count frames via OpenCV sequential read (already known) — reaffirm with CAP.
    cap = cv2.VideoCapture(str(path))
    decoded = 0
    last_ts = None
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        decoded += 1
        last_ts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    cap.release()
    pass_ok = (
        completed.returncode == 0
        and len(err_lines) == 0
        and decoded == expected_frames
        and decoded > 0
    )
    result = {
        "command": cmd,
        "decode_started_at": started,
        "decode_completed_at": ended,
        "return_code": completed.returncode,
        "stderr_line_count": len(err_lines),
        "corruption_error_count": len(err_lines),
        "decoded_frame_count": decoded,
        "expected_frame_count": expected_frames,
        "last_decoded_timestamp_sec": last_ts,
        "decode_integrity_pass": pass_ok,
        "stderr_preview": err_lines[:20],
    }
    if not pass_ok:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_HOLDOUT_DECODE_INTEGRITY_FAILURE "
            + json.dumps(
                {
                    "return_code": completed.returncode,
                    "stderr_line_count": len(err_lines),
                    "decoded": decoded,
                    "expected": expected_frames,
                }
            )
        )
    return result


def build_fingerprint_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    fp = config["fingerprint"]
    return {
        "schema_version": "reid_target_001_holdout_frame_fingerprint_contract_v1",
        "frozen_before_frame_decode": True,
        "sampling_rate_fps": float(fp["sampling_rate_fps"]),
        "timestamp_based_sampling": True,
        "primary_video_stream_only": True,
        "normalize_width": int(fp["normalize_width"]),
        "normalize_height": int(fp["normalize_height"]),
        "color_representation": "grayscale_uint8",
        "interpolation": "area",
        "maximum_samples_per_source": int(fp["max_samples_per_source"]),
        "sample_ordering": "timestamp_ascending",
        "raw_normalized_frame_sha256": True,
        "dhash_bits": 64,
        "edge_dhash_bits": 64,
        "normalized_frame_mean_absolute_pixel_difference_diagnostic": True,
        "no_identity_jersey_person_information_extracted": True,
        "no_frames_retained_after_finalization": True,
        "no_png_jpeg_during_successful_preflight": True,
        "match_thresholds": {
            "exact_normalized_sha_equal": True,
            "dhash_hamming_max": int(fp["dhash_hamming_max"]),
            "edge_dhash_hamming_max": int(fp["edge_dhash_hamming_max"]),
            "mad_max": int(fp["mad_max"]),
        },
        "confirmed_overlap_thresholds": {
            "min_consecutive_matched_samples": int(fp["confirmed_min_consecutive"]),
            "median_dhash_distance_max": int(fp["confirmed_median_dhash_max"]),
            "median_mad_max": int(fp["confirmed_median_mad_max"]),
            "offset_drift_max_samples": int(fp["offset_drift_max_samples"]),
        },
        "applies_to_sources": ["holdout", "sample", "external"],
    }


def dhash64(gray_u8: np.ndarray) -> int:
    # 8x9 horizontal difference → 64 bits.
    small = cv2.resize(gray_u8, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for i, flag in enumerate(diff.flatten().tolist()):
        if flag:
            bits |= 1 << i
    return bits


def edge_dhash64(gray_u8: np.ndarray) -> int:
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
    return dhash64(mag)


def hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


@dataclass
class FramePrint:
    sample_index: int
    frame_index: int
    timestamp_sec: float
    sha256: str
    dhash: int
    edge_dhash: int
    gray: np.ndarray  # 36x64 uint8 retained only in memory


def sample_timestamps(duration_sec: float, rate_fps: float, max_samples: int) -> list[float]:
    if duration_sec <= 0 or rate_fps <= 0:
        return []
    step = 1.0 / rate_fps
    times: list[float] = []
    t = 0.0
    # Sample up to last frame time exclusive of exact end if empty.
    while t < duration_sec - 1e-9 and len(times) < max_samples:
        times.append(t)
        t += step
    if not times:
        times = [0.0]
    return times


def maybe_downsample_uniform(
    times: list[float], max_samples: int
) -> tuple[list[float], dict[str, Any]]:
    if len(times) <= max_samples:
        return times, {"downsampled": False, "original_count": len(times)}
    idx = np.linspace(0, len(times) - 1, max_samples).round().astype(int)
    uniq = sorted({int(i) for i in idx.tolist()})
    out = [times[i] for i in uniq]
    return out, {
        "downsampled": True,
        "original_count": len(times),
        "effective_count": len(out),
        "strategy": "uniform_linspace_index",
    }


def extract_fingerprints(
    path: Path,
    *,
    fps: float,
    duration_sec: float,
    frame_count: int,
    rate_fps: float,
    width: int,
    height: int,
    max_samples: int,
) -> tuple[list[FramePrint], dict[str, Any]]:
    times = sample_timestamps(duration_sec, rate_fps, max_samples * 2)
    times, ds_meta = maybe_downsample_uniform(times, max_samples)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HoldoutPreflightError(f"cannot open for fingerprints: {path}")
    out: list[FramePrint] = []
    for si, ts in enumerate(times):
        fi = int(round(ts * fps))
        if fi >= frame_count:
            fi = frame_count - 1
        if fi < 0:
            fi = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        norm = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        out.append(
            FramePrint(
                sample_index=si,
                frame_index=fi,
                timestamp_sec=float(ts),
                sha256=sha256_bytes(norm.tobytes()),
                dhash=dhash64(norm),
                edge_dhash=edge_dhash64(norm),
                gray=norm,
            )
        )
    cap.release()
    meta = {
        "requested_sampling_rate_fps": rate_fps,
        "sample_count": len(out),
        "timestamps": [p.timestamp_sec for p in out],
        "downsample": ds_meta,
    }
    return out, meta


def pair_match(
    a: FramePrint, b: FramePrint, *, dhash_max: int, edge_max: int, mad_max: int
) -> Optional[dict[str, Any]]:
    exact = a.sha256 == b.sha256
    d_dist = hamming64(a.dhash, b.dhash)
    e_dist = hamming64(a.edge_dhash, b.edge_dhash)
    mad = float(np.mean(np.abs(a.gray.astype(np.int16) - b.gray.astype(np.int16))))
    perceptual = d_dist <= dhash_max and e_dist <= edge_max and mad <= mad_max
    if not (exact or perceptual):
        return None
    return {
        "exact_hash_match": exact,
        "dhash_distance": d_dist,
        "edge_hash_distance": e_dist,
        "mean_absolute_difference": mad,
        "perceptual_match": perceptual and not exact,
    }


def classify_overlap(
    hold: Sequence[FramePrint],
    ref: Sequence[FramePrint],
    *,
    fp_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    dhash_max = int(fp_cfg["dhash_hamming_max"])
    edge_max = int(fp_cfg["edge_dhash_hamming_max"])
    mad_max = int(fp_cfg["mad_max"])
    drift_max = int(fp_cfg["offset_drift_max_samples"])
    conf_min = int(fp_cfg["confirmed_min_consecutive"])
    amb_min = int(fp_cfg["ambiguous_min_consecutive"])
    amb_max = int(fp_cfg["ambiguous_max_consecutive"])
    med_dhash_max = int(fp_cfg["confirmed_median_dhash_max"])
    med_mad_max = int(fp_cfg["confirmed_median_mad_max"])
    near_cov = float(fp_cfg["near_duplicate_coverage_ge"])

    # Precompute best matches per hold index (all matching refs).
    matches: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    exact_pairs = 0
    perceptual_pairs = 0
    candidate_pairs: list[dict[str, Any]] = []
    for i, hp in enumerate(hold):
        for j, rp in enumerate(ref):
            info = pair_match(
                hp, rp, dhash_max=dhash_max, edge_max=edge_max, mad_max=mad_max
            )
            if info is None:
                continue
            matches.setdefault(i, []).append((j, info))
            if info["exact_hash_match"]:
                exact_pairs += 1
            else:
                perceptual_pairs += 1
            candidate_pairs.append(
                {
                    "holdout_sample_index": i,
                    "holdout_timestamp_sec": hp.timestamp_sec,
                    "holdout_frame_index": hp.frame_index,
                    "reference_sample_index": j,
                    "reference_timestamp_sec": rp.timestamp_sec,
                    "reference_frame_index": rp.frame_index,
                    **info,
                }
            )

    best_run: list[tuple[int, int, dict[str, Any]]] = []
    for i0, opts in matches.items():
        for j0, info0 in opts:
            run = [(i0, j0, info0)]
            i, j = i0, j0
            base_offset = j0 - i0
            while i + 1 in matches:
                nxt = None
                for j2, info2 in matches[i + 1]:
                    if j2 <= j:
                        continue
                    if abs((j2 - (i + 1)) - base_offset) > drift_max:
                        continue
                    nxt = (i + 1, j2, info2)
                    break
                if nxt is None:
                    break
                run.append(nxt)
                i, j = nxt[0], nxt[1]
            if len(run) > len(best_run):
                best_run = run

    run_len = len(best_run)
    dhash_dists = [r[2]["dhash_distance"] for r in best_run]
    mads = [r[2]["mean_absolute_difference"] for r in best_run]
    median_dhash = float(np.median(dhash_dists)) if dhash_dists else None
    median_mad = float(np.median(mads)) if mads else None
    est_duration = (
        (best_run[-1][0] - best_run[0][0]) / float(fp_cfg["sampling_rate_fps"])
        if run_len >= 2
        else 0.0
    )
    # Coverage of shorter side.
    shorter = max(1, min(len(hold), len(ref)))
    coverage = float(run_len) / float(shorter)
    offset = None
    offset_drift = None
    if best_run:
        offsets = [j - i for i, j, _ in best_run]
        offset = float(np.median(offsets))
        offset_drift = float(max(abs(o - offset) for o in offsets))

    classification = "no_overlap"
    if run_len == 0:
        classification = "no_overlap"
    elif coverage >= near_cov and run_len >= conf_min:
        classification = "near_duplicate_sequence"
    elif (
        run_len >= conf_min
        and median_dhash is not None
        and median_mad is not None
        and median_dhash <= med_dhash_max
        and median_mad <= med_mad_max
        and (offset_drift is None or offset_drift <= drift_max)
    ):
        classification = "confirmed_temporal_overlap"
    elif amb_min <= run_len <= amb_max:
        classification = "possible_ambiguous_overlap"
    elif 0 < run_len < amb_min:
        classification = "incidental_similarity"
    else:
        # long run but weak medians → ambiguous
        if run_len >= conf_min:
            classification = "possible_ambiguous_overlap"
        else:
            classification = "incidental_similarity"

    return {
        "matched_pair_count": len(candidate_pairs),
        "exact_match_pair_count": exact_pairs,
        "perceptual_match_pair_count": perceptual_pairs,
        "maximum_matched_run_length": run_len,
        "estimated_overlap_duration_sec": est_duration,
        "matched_coverage_shorter_source": coverage,
        "estimated_time_offset_samples": offset,
        "offset_drift_samples": offset_drift,
        "sequence_median_dhash_distance": median_dhash,
        "sequence_median_mad": median_mad,
        "best_run_holdout_sample_indices": [i for i, _, _ in best_run],
        "best_run_reference_sample_indices": [j for _, j, _ in best_run],
        "candidate_pair_count_recorded": min(len(candidate_pairs), 200),
        "candidate_pairs_head": candidate_pairs[:200],
        "final_classification": classification,
    }


def render_ambiguous_contact_sheet(
    hold_path: Path,
    ref_path: Path,
    audit: Mapping[str, Any],
    out_png: Path,
    *,
    hold_label: str,
    ref_label: str,
) -> None:
    hold_idx = list(audit.get("best_run_holdout_sample_indices") or [])
    ref_idx = list(audit.get("best_run_reference_sample_indices") or [])
    pairs = list(zip(hold_idx, ref_idx))[:8]
    if not pairs:
        return
    # Load by seeking to sample timestamps approx via frame index from audit head.
    # Use OpenCV CAP and sample indices as frame seekers from candidate pairs head.
    head = {
        (p["holdout_sample_index"], p["reference_sample_index"]): p
        for p in audit.get("candidate_pairs_head") or []
    }
    tiles = []
    for hi, ri in pairs:
        meta = head.get((hi, ri), {})
        h_fi = int(meta.get("holdout_frame_index", 0))
        r_fi = int(meta.get("reference_frame_index", 0))
        h_ts = float(meta.get("holdout_timestamp_sec", 0.0))
        r_ts = float(meta.get("reference_timestamp_sec", 0.0))

        def _read(p: Path, fi: int) -> np.ndarray:
            cap = cv2.VideoCapture(str(p))
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, fr = cap.read()
            cap.release()
            if not ok or fr is None:
                return np.zeros((180, 320, 3), dtype=np.uint8)
            return cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA)

        left = _read(hold_path, h_fi)
        right = _read(ref_path, r_fi)
        panel = np.concatenate([left, right], axis=1)
        label = (
            f"{hold_label} t={h_ts:.2f}s | {ref_label} t={r_ts:.2f}s | "
            f"dHash={meta.get('dhash_distance')} edge={meta.get('edge_hash_distance')} "
            f"MAD={meta.get('mean_absolute_difference')}"
        )
        cv2.putText(
            panel,
            label[:110],
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(panel)
    sheet = np.concatenate(tiles, axis=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), sheet)


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f3i_holdout_preflight_{final_dir.name}_{token}"
    if tmp.exists():
        raise HoldoutPreflightError("tmp_exists")
    tmp.mkdir(parents=False, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise HoldoutPreflightError("final_exists")
    os.rename(tmp, final_dir)


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = (project_root or _PROJECT_ROOT).resolve()
    config = load_config(config_path)
    assert_no_path_traversal(config["output"]["final_dir"])
    final_dir = project_root / config["output"]["final_dir"]
    blocked_dir = project_root / config["output"]["blocked_ambiguous_dir"]
    if final_dir.exists():
        raise HoldoutPreflightError("final_exists")

    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )
    f3h = validate_f3h(project_root, config)
    sample_ref = validate_reference(
        project_root,
        config["evaluation_source"]["path"],
        config["evaluation_source"],
        kind="sample",
    )
    external_ref = validate_reference(
        project_root,
        config["external_enrollment_source"]["path"],
        config["external_enrollment_source"],
        kind="external",
    )
    hold_input = validate_holdout_input(project_root, config)
    hold_path = Path(hold_input["absolute_path"])
    sha_before = sha256_file(hold_path)

    meta = extract_holdout_metadata(hold_path)
    canon = dict(meta["canonical"])
    frame_count = canon.get("nb_frames_metadata")
    if frame_count is None:
        frame_count = resolve_frame_count_by_decode(hold_path)
        frame_count_source = "decode_count"
    else:
        frame_count_source = "ffprobe_nb_frames"
    canon["frame_count"] = int(frame_count)
    canon["frame_count_source"] = frame_count_source
    canon["holdout_sha256"] = sha_before
    canon["holdout_bytes"] = hold_input["bytes"]

    decode = full_decode_integrity(hold_path, int(frame_count))

    # Gallery source SHAs from F3G lineage.
    f3g_root = project_root / config["stage5d_f3g_package"]["path"]
    t_lin = load_json(f3g_root / "target_gallery_v2" / "target_001_gallery_v2_lineage.json")
    d_lin = load_json(
        f3g_root
        / "distractor_gallery_v1"
        / "target_001_same_team_distractor_lineage.json"
    )
    gallery_shas = sorted(
        {
            t_lin.get("external_source_sha256"),
            d_lin.get("external_source_sha256"),
            config["stage5d_f3g_package"]["expected_external_source_sha256"],
        }
    )
    gallery_shas = [s for s in gallery_shas if s]
    if set(gallery_shas) != {
        config["stage5d_f3g_package"]["expected_external_source_sha256"]
    }:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_HOLDOUT_GALLERY_SOURCE_OVERLAP unexpected_universe"
        )

    exact_sample = sha_before == sample_ref["sha256"]
    exact_external = sha_before == external_ref["sha256"]
    exact_gallery = sha_before in set(gallery_shas)
    if exact_sample or exact_external or exact_gallery:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_HOLDOUT_EXACT_DUPLICATE "
            + json.dumps(
                {
                    "exact_duplicate_sample": exact_sample,
                    "exact_duplicate_external": exact_external,
                    "exact_duplicate_any_gallery_source": exact_gallery,
                }
            )
        )

    fp_contract = build_fingerprint_contract(config)
    # Contract must be conceptually frozen before fingerprint decode.
    fp_cfg = config["fingerprint"]
    hold_fps = float(canon["fps_float"])
    hold_prints, hold_fp_meta = extract_fingerprints(
        hold_path,
        fps=hold_fps,
        duration_sec=float(canon["format_duration"]),
        frame_count=int(frame_count),
        rate_fps=float(fp_cfg["sampling_rate_fps"]),
        width=int(fp_cfg["normalize_width"]),
        height=int(fp_cfg["normalize_height"]),
        max_samples=int(fp_cfg["max_samples_per_source"]),
    )
    sample_prints, sample_fp_meta = extract_fingerprints(
        Path(sample_ref["absolute_path"]),
        fps=float(sample_ref["fps"]),
        duration_sec=float(sample_ref["format_duration_sec"]),
        frame_count=int(sample_ref["frames"]),
        rate_fps=float(fp_cfg["sampling_rate_fps"]),
        width=int(fp_cfg["normalize_width"]),
        height=int(fp_cfg["normalize_height"]),
        max_samples=int(fp_cfg["max_samples_per_source"]),
    )
    external_prints, external_fp_meta = extract_fingerprints(
        Path(external_ref["absolute_path"]),
        fps=float(external_ref["fps"]),
        duration_sec=float(external_ref["format_duration_sec"]),
        frame_count=int(external_ref["frames"]),
        rate_fps=float(fp_cfg["sampling_rate_fps"]),
        width=int(fp_cfg["normalize_width"]),
        height=int(fp_cfg["normalize_height"]),
        max_samples=int(fp_cfg["max_samples_per_source"]),
    )

    vs_sample = classify_overlap(hold_prints, sample_prints, fp_cfg=fp_cfg)
    vs_external = classify_overlap(hold_prints, external_prints, fp_cfg=fp_cfg)

    # Drop in-memory frames before writing outputs.
    for seq in (hold_prints, sample_prints, external_prints):
        for item in seq:
            item.gray = np.zeros((1, 1), dtype=np.uint8)

    sha_after = sha256_file(hold_path)
    if sha_after != sha_before:
        raise HoldoutPreflightError("holdout mutated during preflight")

    sample_cls = vs_sample["final_classification"]
    external_cls = vs_external["final_classification"]
    blocker_overlap = {
        "confirmed_temporal_overlap",
        "near_duplicate_sequence",
    }
    if sample_cls in blocker_overlap or external_cls in blocker_overlap:
        raise HoldoutPreflightError(
            "BLOCKED_STAGE5D_F3I_HOLDOUT_TEMPORAL_OVERLAP_CONFIRMED "
            + json.dumps({"sample": sample_cls, "external": external_cls})
        )

    ambiguous = {
        "possible_ambiguous_overlap",
    }
    if sample_cls in ambiguous or external_cls in ambiguous:
        if blocked_dir.exists():
            shutil.rmtree(blocked_dir)
        blocked_dir.mkdir(parents=True, exist_ok=False)
        review = blocked_dir / "review"
        review.mkdir()
        if sample_cls in ambiguous:
            render_ambiguous_contact_sheet(
                hold_path,
                Path(sample_ref["absolute_path"]),
                vs_sample,
                review / "holdout_vs_sample_overlap_review.png",
                hold_label="holdout_v2",
                ref_label="sample",
            )
        if external_cls in ambiguous:
            render_ambiguous_contact_sheet(
                hold_path,
                Path(external_ref["absolute_path"]),
                vs_external,
                review / "holdout_vs_external_overlap_review.png",
                hold_label="holdout_v2",
                ref_label="external",
            )
        write_json(
            blocked_dir / "independence_ambiguous_summary.json",
            {
                "final_status": "BLOCKED_STAGE5D_F3I_HOLDOUT_INDEPENDENCE_AMBIGUOUS",
                "sample_classification": sample_cls,
                "external_classification": external_cls,
                "accepted_as_independent_holdout": False,
            },
        )
        raise HoldoutPreflightError("BLOCKED_STAGE5D_F3I_HOLDOUT_INDEPENDENCE_AMBIGUOUS")

    gallery_overlap = {
        "gallery_source_exact_overlap": False,
        "gallery_source_temporal_overlap": False,
        "gallery_crop_source_overlap": False,
        "gallery_source_independence_pass": True,
        "unique_gallery_source_video_shas": gallery_shas,
        "holdout_sha256": sha_before,
        "external_temporal_classification": external_cls,
    }
    # External is the gallery source video; confirmed overlap already blocked above.
    if exact_gallery:
        raise HoldoutPreflightError("BLOCKED_STAGE5D_F3I_HOLDOUT_GALLERY_SOURCE_OVERLAP")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = create_temp_root(final_dir)
    try:
        input_dir = tmp / "input"
        indep = tmp / "independence"
        runtime = tmp / "runtime"
        cfg_dir = tmp / "effective_configs"
        for d in (input_dir, indep, runtime, cfg_dir):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, cfg_dir / config_path.name)

        write_json(
            indep / "target_001_holdout_frame_fingerprint_contract_pre_decode.json",
            fp_contract,
        )
        write_json(
            input_dir / "target_001_independent_holdout_v2_input_manifest.json",
            {
                "schema_version": "reid_target_001_independent_holdout_v2_input_manifest_v1",
                **hold_input,
                "sha256": sha_before,
                "sha256_after_preflight": sha_after,
                "user_declaration": config["user_declaration"],
            },
        )
        write_json(
            input_dir / "target_001_independent_holdout_v2_ffprobe.json",
            meta["ffprobe_raw"],
        )
        write_json(
            input_dir / "target_001_independent_holdout_v2_technical_metadata.json",
            {
                "schema_version": "reid_target_001_independent_holdout_v2_technical_metadata_v1",
                **canon,
            },
        )
        write_json(
            input_dir / "target_001_independent_holdout_v2_decode_integrity.json",
            decode,
        )
        write_json(
            indep / "target_001_holdout_exact_duplicate_audit.json",
            {
                "exact_duplicate_sample": False,
                "exact_duplicate_external": False,
                "exact_duplicate_any_gallery_source": False,
                "holdout_sha256": sha_before,
                "sample_sha256": sample_ref["sha256"],
                "external_sha256": external_ref["sha256"],
                "gallery_source_shas": gallery_shas,
            },
        )
        write_json(
            indep / "target_001_holdout_vs_sample_fingerprint_audit.json",
            {
                "schema_version": "reid_target_001_holdout_vs_sample_fingerprint_audit_v1",
                "holdout_fingerprint_meta": hold_fp_meta,
                "sample_fingerprint_meta": sample_fp_meta,
                **vs_sample,
            },
        )
        write_json(
            indep / "target_001_holdout_vs_external_fingerprint_audit.json",
            {
                "schema_version": "reid_target_001_holdout_vs_external_fingerprint_audit_v1",
                "holdout_fingerprint_meta": hold_fp_meta,
                "external_fingerprint_meta": external_fp_meta,
                **vs_external,
            },
        )
        write_json(
            indep / "target_001_holdout_gallery_source_overlap_audit.json",
            gallery_overlap,
        )
        decision = {
            "schema_version": "reid_target_001_holdout_independence_decision_v1",
            "accepted_as_independent_holdout": True,
            "accepted_holdout_role": "frozen_evaluation_input",
            "holdout_version": "v2",
            "target_id": TARGET_ID,
            "enrollment_eligible": False,
            "gallery_growth_eligible": False,
            "threshold_calibration_eligible": False,
            "scoring_evaluation_eligible": True,
            "label_blind_universe_build_eligible": True,
            "file_exists_readable": True,
            "technical_metadata_valid": True,
            "full_decode_integrity_pass": True,
            "exact_duplicate_sample": False,
            "exact_duplicate_external": False,
            "exact_duplicate_gallery_source": False,
            "confirmed_overlap_sample": False,
            "confirmed_overlap_external": False,
            "possible_ambiguous_overlap_sample": False,
            "possible_ambiguous_overlap_external": False,
            "gallery_source_overlap": False,
            "user_declaration_present": True,
            "sample_classification": sample_cls,
            "external_classification": external_cls,
            "not_proof_of_target_presence": True,
            "not_proof_of_gt_support": True,
            "not_system_success": True,
            "not_deployment_approval": True,
        }
        write_json(indep / "target_001_holdout_independence_decision.json", decision)

        access = {
            "schema_version": "reid_stage5d_f3i_access_audit_v1",
            "holdout_metadata_read": True,
            "holdout_full_decode": True,
            "holdout_normalized_fingerprints": True,
            "sample_metadata_read": True,
            "sample_normalized_fingerprints": True,
            "external_metadata_read": True,
            "external_normalized_fingerprints": True,
            "sample_reference_fingerprint_decode": True,
            "external_reference_fingerprint_decode": True,
            "sample_identity_review": False,
            "external_identity_review": False,
            "sample_crop_read_count": 0,
            "sample_embedding_read_count": 0,
            "sample_score_row_read_count": 0,
            "f3_score_rank_read_count": 0,
            "person_detections": 0,
            "tracking_observations": 0,
            "segment_units": 0,
            "crop_exports": 0,
            "osnet_model_loads": 0,
            "embeddings": 0,
            "cosine_computations_for_reid": 0,
            "query_score_rows": 0,
            "rankings": 0,
            "gt_labels": 0,
            "metrics": 0,
            "threshold_candidates": 0,
            "identity_assignments": 0,
            "gallery_mutations": 0,
            "ocr_calls": 0,
            "team_classifier_calls": 0,
            "fingerprint_raw_frames_retained": 0,
        }
        write_json(runtime / "target_001_f3i_access_audit.json", access)
        write_json(
            runtime / "target_001_f3i_environment_and_network_audit.json",
            {
                "network_download": 0,
                "package_environment_changed": False,
                "conda_env_expected": "football-cv",
                "offline_required": True,
            },
        )

        if list(tmp.rglob("*.npy")) or list(tmp.rglob("*.mp4")):
            raise HoldoutPreflightError("forbidden media in output")
        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.jpg")):
            raise HoldoutPreflightError("png forbidden on successful pass")

        file_count, files_sha = listing_sha(tmp)
        summary = {
            "schema_version": "reid_stage5d_f3i_holdout_preflight_summary_v1",
            "final_status": FINAL_STATUS,
            "readiness": READINESS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "holdout_path": hold_input["path"],
            "holdout_sha256": sha_before,
            "holdout_bytes": hold_input["bytes"],
            "codec": canon["codec"],
            "width": canon["width"],
            "height": canon["height"],
            "fps": canon["fps_float"],
            "frame_count": int(frame_count),
            "duration_seconds": canon["format_duration"],
            "audio_present": canon["audio_present"],
            "full_decode_pass": True,
            "exact_duplicate_sample": False,
            "exact_duplicate_external": False,
            "sample_temporal_classification": sample_cls,
            "external_temporal_classification": external_cls,
            "gallery_overlap": False,
            "accepted_as_independent_holdout": True,
            "detections": 0,
            "tracks": 0,
            "crops": 0,
            "embeddings": 0,
            "score_rows": 0,
            "rankings": 0,
            "metrics": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "f3h_snapshot_sha256": f3h["snapshot_sha256"],
            "output_file_count": file_count,
            "output_listing_sha256": files_sha,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3i_summary.json", summary)
        write_json(
            tmp / "stage5d_f3i_contract.json",
            {
                "schema_version": "reid_stage5d_f3i_holdout_preflight_contract_v1",
                "final_status": FINAL_STATUS,
                "readiness": READINESS,
                "exact_next_gate": NEXT_GATE,
                "accepted_as_independent_holdout": True,
                "detections": 0,
                "tracks": 0,
                "crops": 0,
                "embeddings": 0,
                "score_rows": 0,
                "threshold_selected": False,
                "identity_assignments": 0,
                "gallery_mutation": False,
                "generated_at": generated_at,
                "project_head": head,
            },
        )
        write_json(
            tmp / "stage5d_f3i_manifest.json",
            {
                "schema_version": "reid_stage5d_f3i_holdout_preflight_manifest_v1",
                "final_status": FINAL_STATUS,
                "readiness": READINESS,
                "exact_next_gate": NEXT_GATE,
                "project_head": head,
                "output_file_count": file_count,
                "output_listing_sha256": files_sha,
                "artifact_budget": {
                    "mp4_copies": 0,
                    "png": 0,
                    "npy": 0,
                    "crops": 0,
                    "embeddings": 0,
                    "fingerprint_raw_frames_retained": 0,
                },
                "generated_at": generated_at,
            },
        )
        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    return load_json(final_dir / "stage5d_f3i_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/independent_holdout_v2_ingestion_stage5d_target_001.yaml",
    )
    args = parser.parse_args(argv)
    try:
        summary = run(args.config)
    except HoldoutPreflightError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "readiness": summary["readiness"],
                "exact_next_gate": summary["exact_next_gate"],
                "holdout_sha256": summary["holdout_sha256"],
                "accepted_as_independent_holdout": summary[
                    "accepted_as_independent_holdout"
                ],
                "sample_temporal_classification": summary[
                    "sample_temporal_classification"
                ],
                "external_temporal_classification": summary[
                    "external_temporal_classification"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
