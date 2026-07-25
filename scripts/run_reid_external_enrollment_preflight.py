#!/usr/bin/env python3
"""Stage 5D-B1E-A — external enrollment clip ingest and overlap preflight.

Validates immutable external source vs sample.mp4 with deterministic perceptual
overlap audit. No detection, tracking, embedding, OCR, similarity, or gallery.
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
CONFIG_SCHEMA = "reid_external_enrollment_preflight_config_v1"
FROZEN_SEED_CODE = "SEED_CANDIDATE_07"
MIN_ELIGIBLE_DURATION_SEC = 5.0


class ExternalPreflightError(RuntimeError):
    pass


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def listing_sha(root: Path) -> tuple[int, str]:
    files: list[tuple[str, int, str]] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            path = Path(dp) / fn
            rel = str(path.relative_to(root))
            files.append((rel, path.stat().st_size, sha256_file(path)))
    files.sort()
    blob = "\n".join(f"{a}\t{b}\t{c}" for a, b, c in files).encode()
    return len(files), hashlib.sha256(blob).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ExternalPreflightError("unexpected config schema")
    if not config.get("offline_required"):
        raise ExternalPreflightError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/"):
        raise ExternalPreflightError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise ExternalPreflightError("BLOCKED_STAGE5D_B1E_A_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_GIT_CONTRACT_MISMATCH origin"
        )
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    allowed = {
        "scripts/run_reid_external_enrollment_preflight.py",
        "configs/reid/external_enrollment_preflight_stage5d_target_001.yaml",
        "tests/test_reid_external_enrollment_preflight.py",
        "docs/setup/stage5d-target-external-enrollment-ingest-and-overlap-preflight.md",
    }
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path not in allowed:
                raise ExternalPreflightError(
                    "BLOCKED_STAGE5D_B1E_A_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != "Freeze target 001 bridge review with no selection":
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_GIT_CONTRACT_MISMATCH message"
        )
    return head


def ffprobe_json(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    raw = subprocess.check_output(cmd, text=True)
    return json.loads(raw)


def parse_fps(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            raise ExternalPreflightError(f"bad fps {rate}")
        return float(num) / den_f
    return float(rate)


def validate_handoff(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root / config["stage5d_b1d_package"]["path"]
    summary = load_json(root / "stage5d_b1d_summary.json")
    handoff = load_json(
        root
        / "external_enrollment_handoff"
        / "target_001_external_enrollment_requirements.json"
    )
    no_sel = load_json(
        root / "bridge_review_freeze" / "target_001_bridge_review_no_selection.json"
    )
    td = load_json(project_root / config["target_definition"]["path"])
    exp = config["stage5d_b1d_package"]["expected_final_status"]
    if summary.get("final_status") != exp:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_HANDOFF_CONTRACT_MISMATCH status"
        )
    if summary.get("target_id") != "target_001":
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_HANDOFF_CONTRACT_MISMATCH target"
        )
    if td.get("target_alias") != "sarı takım 5 numaralı oyuncu":
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_HANDOFF_CONTRACT_MISMATCH alias"
        )
    if int(handoff.get("human_verified_jersey_number")) != 5:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_HANDOFF_CONTRACT_MISMATCH jersey"
        )
    if summary.get("original_frozen_seed_code") != FROZEN_SEED_CODE:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_HANDOFF_CONTRACT_MISMATCH seed"
        )
    if summary.get("frozen_seed_enrollment_allowed") is not False:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_HANDOFF_CONTRACT_MISMATCH seed_enroll"
        )
    if summary.get("selected_bridge_candidate_code") not in ("", None):
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_HANDOFF_CONTRACT_MISMATCH selected"
        )
    if summary.get("current_video_eligible_source_search_closed") is not True:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_HANDOFF_CONTRACT_MISMATCH closed"
        )
    if summary.get("next_source_type") != "external_enrollment_clip":
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_HANDOFF_CONTRACT_MISMATCH next"
        )
    for key in ("derived_anchors", "gallery_members", "prototypes", "identity_assignments"):
        if int(summary.get(key) or 0) != 0:
            raise ExternalPreflightError(
                f"BLOCKED_STAGE5D_B1E_A_HANDOFF_CONTRACT_MISMATCH {key}"
            )
    if handoff.get("automatic_gallery_growth") is not False:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_HANDOFF_CONTRACT_MISMATCH auto_gallery"
        )
    if handoff.get("unknown_identity_preserved") is not True:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_HANDOFF_CONTRACT_MISMATCH unknown"
        )
    if no_sel.get("original_frozen_seed_preserved") is not True:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_HANDOFF_CONTRACT_MISMATCH seed_preserved"
        )
    return {
        "summary": summary,
        "handoff": handoff,
        "no_selection": no_sel,
        "target_definition": td,
    }


def validate_external_source(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    rel = config["external_enrollment_source"]["path"]
    assert_no_path_traversal(rel)
    path = project_root / rel
    exp = config["external_enrollment_source"]
    if not path.is_file() or path.is_symlink():
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY not_regular"
        )
    if path.stat().st_size != int(exp["expected_bytes"]):
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY bytes"
        )
    digest = sha256_file(path)
    if digest != exp["expected_sha256"]:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY sha"
        )
    meta = ffprobe_json(path)
    streams = meta.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY no_video"
        )
    if video.get("codec_name") != exp["expected_codec"]:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY codec"
        )
    if int(video.get("width")) != int(exp["expected_width"]):
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY width"
        )
    if int(video.get("height")) != int(exp["expected_height"]):
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY height"
        )
    fps = parse_fps(str(video.get("r_frame_rate")))
    if abs(fps - float(exp["expected_fps"])) > 1e-6:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY fps"
        )
    frames = int(video.get("nb_frames"))
    if frames != int(exp["expected_video_frames"]):
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY frames"
        )
    vdur = float(video.get("duration"))
    if abs(vdur - float(exp["expected_video_duration_sec"])) > 1e-3:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY vdur"
        )
    fdur = float((meta.get("format") or {}).get("duration"))
    if abs(fdur - float(exp["expected_format_duration_sec"])) > 1e-3:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY fdur"
        )
    if audio is None and not exp.get("allow_audio", True):
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY audio"
        )
    # OpenCV frame count cross-check (no mutate).
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY open"
        )
    cv_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cv_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if cv_frames != frames or cv_w != int(exp["expected_width"]) or cv_h != int(
        exp["expected_height"]
    ):
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY opencv"
        )
    if abs(cv_fps - fps) > 1e-6:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY opencv_fps"
        )
    return {
        "path": rel,
        "absolute_path": str(path),
        "bytes": int(exp["expected_bytes"]),
        "sha256": digest,
        "codec": exp["expected_codec"],
        "width": int(exp["expected_width"]),
        "height": int(exp["expected_height"]),
        "fps": fps,
        "video_frames": frames,
        "video_duration_sec": vdur,
        "format_duration_sec": fdur,
        "has_audio": audio is not None,
        "enrollment_only": True,
        "evaluation_use_forbidden": True,
    }


def validate_evaluation_source(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    rel = config["evaluation_source"]["path"]
    path = project_root / rel
    exp = config["evaluation_source"]
    if not path.is_file():
        raise ExternalPreflightError("BLOCKED_STAGE5D_B1E_A_EVAL_SOURCE_MISSING")
    if path.stat().st_size != int(exp["expected_bytes"]):
        raise ExternalPreflightError("BLOCKED_STAGE5D_B1E_A_EVAL_SOURCE_BYTES")
    digest = sha256_file(path)
    if digest != exp["expected_sha256"]:
        raise ExternalPreflightError("BLOCKED_STAGE5D_B1E_A_EVAL_SOURCE_SHA")
    meta = ffprobe_json(path)
    video = next(
        s for s in (meta.get("streams") or []) if s.get("codec_type") == "video"
    )
    frames = int(video.get("nb_frames"))
    width = int(video.get("width"))
    height = int(video.get("height"))
    fps = parse_fps(str(video.get("r_frame_rate")))
    fdur = float((meta.get("format") or {}).get("duration"))
    if frames != int(exp["expected_frames"]):
        raise ExternalPreflightError("BLOCKED_STAGE5D_B1E_A_EVAL_SOURCE_FRAMES")
    if width != int(exp["expected_width"]) or height != int(exp["expected_height"]):
        raise ExternalPreflightError("BLOCKED_STAGE5D_B1E_A_EVAL_SOURCE_WH")
    if abs(fps - float(exp["expected_fps"])) > 1e-6:
        raise ExternalPreflightError("BLOCKED_STAGE5D_B1E_A_EVAL_SOURCE_FPS")
    if abs(fdur - float(exp["expected_format_duration_sec"])) > 1e-3:
        raise ExternalPreflightError("BLOCKED_STAGE5D_B1E_A_EVAL_SOURCE_DUR")
    return {
        "path": rel,
        "bytes": int(exp["expected_bytes"]),
        "sha256": digest,
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
        "format_duration_sec": fdur,
        "evaluation_only": True,
        "enrollment_use_forbidden": True,
    }


def dhash_bytes(gray: np.ndarray, hash_size: int) -> bytes:
    resized = cv2.resize(
        gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA
    )
    diff = resized[:, 1:] > resized[:, :-1]
    return np.packbits(diff.flatten().astype(np.uint8)).tobytes()


def hamming_bytes(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def normalized_patch(frame_bgr: np.ndarray, size: Sequence[int]) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    patch = cv2.resize(
        gray, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    patch = (patch - float(patch.mean())) / (float(patch.std()) + 1e-6)
    return patch


def corr_and_mae(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    corr = float(np.mean(a * b))
    mae = float(np.mean(np.abs(a - b)))
    return corr, mae


@dataclass
class FrameHash:
    frame_index: int
    dhash: bytes


def sample_dhashes(
    path: Path, *, stride: int, hash_size: int, frame_indices: Optional[Sequence[int]] = None
) -> list[FrameHash]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ExternalPreflightError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_indices is None:
        indices = list(range(0, n, int(stride)))
        if n > 0 and (n - 1) not in indices:
            indices.append(n - 1)
    else:
        indices = sorted({int(i) for i in frame_indices if 0 <= int(i) < n})
    out: list[FrameHash] = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.append(FrameHash(fi, dhash_bytes(gray, hash_size)))
    cap.release()
    return out


def read_normalized(
    path: Path, frame_index: int, size: Sequence[int]
) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return normalized_patch(frame, size)


def merge_contiguous(
    pairs: Sequence[tuple[int, int]], *, gap_max: int
) -> list[dict[str, Any]]:
    """Merge verified (external_frame, sample_frame) into contiguous intervals."""
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda x: (x[0], x[1]))
    intervals: list[dict[str, Any]] = []
    cur_ext = [ordered[0][0], ordered[0][0]]
    cur_sam = [ordered[0][1], ordered[0][1]]
    members = [ordered[0]]
    for ext_i, sam_i in ordered[1:]:
        if ext_i - cur_ext[1] <= gap_max:
            cur_ext[1] = ext_i
            cur_sam[0] = min(cur_sam[0], sam_i)
            cur_sam[1] = max(cur_sam[1], sam_i)
            members.append((ext_i, sam_i))
        else:
            intervals.append(
                {
                    "external_frame_range": [cur_ext[0], cur_ext[1]],
                    "sample_frame_range": [cur_sam[0], cur_sam[1]],
                    "pair_count": len(members),
                    "pairs": [{"external_frame": a, "sample_frame": b} for a, b in members],
                }
            )
            cur_ext = [ext_i, ext_i]
            cur_sam = [sam_i, sam_i]
            members = [(ext_i, sam_i)]
    intervals.append(
        {
            "external_frame_range": [cur_ext[0], cur_ext[1]],
            "sample_frame_range": [cur_sam[0], cur_sam[1]],
            "pair_count": len(members),
            "pairs": [{"external_frame": a, "sample_frame": b} for a, b in members],
        }
    )
    return intervals


def complement_intervals(
    *, total_frames: int, excluded: Sequence[tuple[int, int]], fps: float
) -> list[dict[str, Any]]:
    """Return non-overlap intervals as [start,end] inclusive frame ranges."""
    if total_frames <= 0:
        return []
    blocked = np.zeros(total_frames, dtype=bool)
    for a, b in excluded:
        a0 = max(0, int(a))
        b0 = min(total_frames - 1, int(b))
        if a0 <= b0:
            blocked[a0 : b0 + 1] = True
    intervals: list[dict[str, Any]] = []
    i = 0
    while i < total_frames:
        if blocked[i]:
            i += 1
            continue
        j = i
        while j + 1 < total_frames and not blocked[j + 1]:
            j += 1
        start, end = i, j
        dur = (end - start + 1) / fps
        intervals.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_time": start / fps,
                "end_time": (end + 1) / fps,
                "duration": dur,
            }
        )
        i = j + 1
    return intervals


def run_overlap_audit(
    *,
    external_path: Path,
    sample_path: Path,
    external_meta: Mapping[str, Any],
    sample_meta: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    exact_duplicate = external_meta["sha256"] == sample_meta["sha256"]
    if exact_duplicate:
        raise ExternalPreflightError(
            "BLOCKED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INTEGRITY exact_duplicate"
        )

    coarse = int(cfg["coarse_sample_stride"])
    refine = int(cfg["refine_sample_stride"])
    radius = int(cfg["refine_window_radius"])
    hash_size = int(cfg["dhash_size"])
    ham_max = int(cfg["dhash_hamming_candidate_max"])
    verify_size = list(cfg["verify_resize"])
    corr_min = float(cfg["verify_corr_min"])
    mae_max = float(cfg["verify_mae_max"])
    gap_max = int(cfg["contiguous_gap_max_frames"])

    ext_hashes = sample_dhashes(external_path, stride=coarse, hash_size=hash_size)
    sam_hashes = sample_dhashes(sample_path, stride=coarse, hash_size=hash_size)

    fingerprint_candidates: list[dict[str, Any]] = []
    for eh in ext_hashes:
        best = None
        best_d = 10**9
        for sh in sam_hashes:
            d = hamming_bytes(eh.dhash, sh.dhash)
            if d < best_d:
                best_d = d
                best = sh.frame_index
        if best is not None and best_d <= ham_max:
            fingerprint_candidates.append(
                {
                    "external_frame": eh.frame_index,
                    "sample_frame": best,
                    "dhash_hamming": best_d,
                }
            )

    # Refine around candidates (and always keep empty path deterministic).
    refine_ext_indices: set[int] = set()
    refine_sam_indices: set[int] = set()
    for cand in fingerprint_candidates:
        for delta in range(-radius, radius + 1, refine):
            refine_ext_indices.add(cand["external_frame"] + delta)
            refine_sam_indices.add(cand["sample_frame"] + delta)

    verified_pairs: list[dict[str, Any]] = []
    if fingerprint_candidates:
        ext_ref = sample_dhashes(
            external_path,
            stride=refine,
            hash_size=hash_size,
            frame_indices=sorted(refine_ext_indices),
        )
        sam_ref = {
            h.frame_index: h
            for h in sample_dhashes(
                sample_path,
                stride=refine,
                hash_size=hash_size,
                frame_indices=sorted(refine_sam_indices),
            )
        }
        for eh in ext_ref:
            best = None
            best_d = 10**9
            for sj, sh in sam_ref.items():
                d = hamming_bytes(eh.dhash, sh.dhash)
                if d < best_d:
                    best_d = d
                    best = sj
            if best is None or best_d > ham_max:
                continue
            a = read_normalized(external_path, eh.frame_index, verify_size)
            b = read_normalized(sample_path, best, verify_size)
            if a is None or b is None:
                continue
            corr, mae = corr_and_mae(a, b)
            if corr >= corr_min and mae <= mae_max:
                verified_pairs.append(
                    {
                        "external_frame": eh.frame_index,
                        "sample_frame": best,
                        "dhash_hamming": best_d,
                        "normalized_correlation": corr,
                        "normalized_mae": mae,
                        "confidence": "high" if corr >= 0.85 else "medium",
                    }
                )

    pair_tuples = [(p["external_frame"], p["sample_frame"]) for p in verified_pairs]
    intervals = merge_contiguous(pair_tuples, gap_max=gap_max)
    fps = float(external_meta["fps"])
    for iv in intervals:
        a, b = iv["external_frame_range"]
        iv["external_time_range"] = [a / fps, (b + 1) / fps]
        sa, sb = iv["sample_frame_range"]
        iv["sample_time_range"] = [
            sa / float(sample_meta["fps"]),
            (sb + 1) / float(sample_meta["fps"]),
        ]
        iv["duration_sec"] = (b - a + 1) / fps

    total_overlap = float(sum(iv["duration_sec"] for iv in intervals))
    longest = float(max((iv["duration_sec"] for iv in intervals), default=0.0))

    excluded_ranges = [tuple(iv["external_frame_range"]) for iv in intervals]
    eligible_raw = complement_intervals(
        total_frames=int(external_meta["video_frames"]),
        excluded=excluded_ranges,
        fps=fps,
    )

    if not verified_pairs:
        source_decision = "EXTERNAL_ENROLLMENT_SOURCE_NONOVERLAPPING_READY"
    else:
        long_enough = [
            iv for iv in eligible_raw if iv["duration"] >= MIN_ELIGIBLE_DURATION_SEC
        ]
        if long_enough:
            source_decision = (
                "EXTERNAL_ENROLLMENT_SOURCE_PARTIALLY_OVERLAPPING_WITH_ELIGIBLE_INTERVALS"
            )
        else:
            source_decision = "EXTERNAL_ENROLLMENT_SOURCE_INELIGIBLE_OVERLAP"

    return {
        "schema_version": "reid_stage5d_b1e_a_overlap_audit_v1",
        "method": {
            "coarse_sample_stride": coarse,
            "refine_sample_stride": refine,
            "refine_window_radius": radius,
            "dhash_size": hash_size,
            "dhash_hamming_candidate_max": ham_max,
            "verify_resize": verify_size,
            "verify_corr_min": corr_min,
            "verify_mae_max": mae_max,
            "contiguous_gap_max_frames": gap_max,
            "notes": (
                "Deterministic perceptual dHash candidate search with normalized "
                "correlation + MAE verification. Not model inference."
            ),
        },
        "exact_file_duplicate": False,
        "external_sha_ne_sample_sha": True,
        "sampled_external_frame_count": len(ext_hashes),
        "sampled_sample_frame_count": len(sam_hashes),
        "potential_fingerprint_matches": fingerprint_candidates,
        "potential_fingerprint_match_count": len(fingerprint_candidates),
        "verified_overlapping_frame_pairs": verified_pairs,
        "verified_overlapping_pair_count": len(verified_pairs),
        "contiguous_overlapping_intervals": intervals,
        "total_verified_overlap_duration_sec": total_overlap,
        "longest_verified_overlap_duration_sec": longest,
        "source_overlap_decision": source_decision,
        "non_overlap_interval_candidates": eligible_raw,
    }


def build_interval_eligibility(
    *,
    audit: Mapping[str, Any],
    external_meta: Mapping[str, Any],
    min_duration: float,
) -> dict[str, Any]:
    fps = float(external_meta["fps"])
    intervals: list[dict[str, Any]] = []

    for iv in audit["contiguous_overlapping_intervals"]:
        a, b = iv["external_frame_range"]
        intervals.append(
            {
                "start_frame": a,
                "end_frame": b,
                "start_time": a / fps,
                "end_time": (b + 1) / fps,
                "duration": (b - a + 1) / fps,
                "overlap_status": "EXCLUDED_OVERLAP",
                "eligible_for_future_detection_tracking": False,
                "eligible_for_gallery_enrollment": False,
                "exclusion_reason": "verified_overlap_with_evaluation_sample",
            }
        )

    for iv in audit["non_overlap_interval_candidates"]:
        eligible = iv["duration"] >= min_duration
        intervals.append(
            {
                "start_frame": iv["start_frame"],
                "end_frame": iv["end_frame"],
                "start_time": iv["start_time"],
                "end_time": iv["end_time"],
                "duration": iv["duration"],
                "overlap_status": "ELIGIBLE_INTERVAL" if eligible else "UNRESOLVED",
                "eligible_for_future_detection_tracking": eligible,
                "eligible_for_gallery_enrollment": eligible,
                "exclusion_reason": (
                    None
                    if eligible
                    else "duration_below_min_eligible_threshold"
                ),
            }
        )

    intervals.sort(key=lambda x: (x["start_frame"], x["end_frame"]))
    eligible_intervals = [
        iv
        for iv in intervals
        if iv["eligible_for_gallery_enrollment"] and iv["overlap_status"] == "ELIGIBLE_INTERVAL"
    ]
    return {
        "schema_version": "reid_stage5d_b1e_a_interval_eligibility_v1",
        "external_source_enrollment_only": True,
        "external_source_evaluation_use_forbidden": True,
        "sample_evaluation_only": True,
        "sample_enrollment_use_forbidden": True,
        "min_eligible_duration_sec": min_duration,
        "intervals": intervals,
        "eligible_interval_count": len(eligible_intervals),
        "eligible_intervals": eligible_intervals,
        "has_min_duration_eligible_interval": len(eligible_intervals) > 0,
    }


def frame_overlap_label(
    frame_index: int, eligibility: Mapping[str, Any]
) -> str:
    for iv in eligibility["intervals"]:
        if iv["start_frame"] <= frame_index <= iv["end_frame"]:
            return str(iv["overlap_status"])
    return "UNRESOLVED"


def render_overview_sheet(
    *,
    video_path: Path,
    fps: float,
    frames: int,
    stride: int,
    eligibility: Mapping[str, Any],
) -> np.ndarray:
    indices = list(range(0, frames, int(stride)))
    if frames > 0 and (frames - 1) not in indices:
        indices.append(frames - 1)
    cols = 6
    rows_n = int(math.ceil(len(indices) / cols)) if indices else 1
    panel_w, panel_h = 280, 190
    sheet = np.full((rows_n * panel_h, cols * panel_w, 3), 18, dtype=np.uint8)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ExternalPreflightError("FAILED_STAGE5D_B1E_A_ATOMIC_OUTPUT open_sheet")
    for idx, fi in enumerate(indices):
        r, c = divmod(idx, cols)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        tile = np.full((panel_h, panel_w, 3), 36, dtype=np.uint8)
        label = frame_overlap_label(fi, eligibility)
        header = f"f={fi} t={fi/fps:.2f}s {label}"
        cv2.putText(
            tile,
            header,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        if ok and frame is not None:
            h, w = frame.shape[:2]
            scale = min((panel_w - 10) / max(w, 1), (panel_h - 34) / max(h, 1))
            nw = max(1, int(math.floor(w * scale)))
            nh = max(1, int(math.floor(h * scale)))
            disp = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            ox = (panel_w - nw) // 2
            oy = 28
            tile[oy : oy + nh, ox : ox + nw] = disp
        y0 = r * panel_h
        x0 = c * panel_w
        sheet[y0 : y0 + panel_h, x0 : x0 + panel_w] = tile
    cap.release()
    return sheet


def blank_seed_template(
    *,
    source_path: str,
    source_sha: str,
    eligible_intervals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_b1e_a_external_seed_review_template_v1",
        "target_id": "target_001",
        "source_video_path": source_path,
        "source_video_sha256": source_sha,
        "eligible_intervals": list(eligible_intervals),
        "selected_interval": None,
        "selected_reference_frame": None,
        "selected_neutral_detection_code": "",
        "manual_target_confirmed": "",
        "manual_human_verified_number_seen": "",
        "manual_crop_valid": "",
        "manual_target_dominant": "",
        "reviewer": "",
        "final_approver": "",
        "reviewed_at": "",
        "notes": "",
    }


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    head = assert_git_contract(project_root, config["project_head_expected"])
    final_rel = config["output"]["final_dir"]
    assert_no_path_traversal(final_rel)
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise ExternalPreflightError("FAILED_STAGE5D_B1E_A_ATOMIC_OUTPUT final_exists")

    handoff = validate_handoff(project_root, config)
    external = validate_external_source(project_root, config)
    evaluation = validate_evaluation_source(project_root, config)

    tmp = project_root / (
        "outputs/reid/"
        "_tmp_full_stage4b_rebuild_r2_stage5d_target_001_external_enrollment_preflight_"
        f"{uuid.uuid4().hex}"
    )
    if tmp.exists():
        raise ExternalPreflightError("FAILED_STAGE5D_B1E_A_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=True)

    try:
        ext_path = project_root / external["path"]
        eva_path = project_root / evaluation["path"]
        audit = run_overlap_audit(
            external_path=ext_path,
            sample_path=eva_path,
            external_meta=external,
            sample_meta=evaluation,
            cfg=config["overlap_audit"],
        )
        eligibility = build_interval_eligibility(
            audit=audit,
            external_meta=external,
            min_duration=float(config["overlap_audit"]["min_eligible_duration_sec"]),
        )

        decision = audit["source_overlap_decision"]
        if decision == "EXTERNAL_ENROLLMENT_SOURCE_NONOVERLAPPING_READY":
            if not eligibility["has_min_duration_eligible_interval"]:
                raise ExternalPreflightError(
                    "FAILED_STAGE5D_B1E_A_ATOMIC_OUTPUT nonoverlap_no_eligible"
                )
            final_status = "COMPLETED_STAGE5D_B1E_A_EXTERNAL_ENROLLMENT_PREFLIGHT_READY"
            exact_next = (
                "STAGE5D-B1E-B_TARGET_001_EXTERNAL_CLIP_DETECTION_TRACKING_AND_SEED_REVIEW_PACKAGE"
            )
        elif (
            decision
            == "EXTERNAL_ENROLLMENT_SOURCE_PARTIALLY_OVERLAPPING_WITH_ELIGIBLE_INTERVALS"
        ):
            final_status = (
                "COMPLETED_STAGE5D_B1E_A_PARTIAL_OVERLAP_ELIGIBLE_INTERVALS_READY"
            )
            exact_next = (
                "STAGE5D-B1E-B_TARGET_001_EXTERNAL_CLIP_ELIGIBLE_INTERVAL_PROCESSING"
            )
        else:
            final_status = "COMPLETED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INELIGIBLE_OVERLAP"
            exact_next = "STAGE5D-B1E_TARGET_001_REPLACEMENT_EXTERNAL_CLIP_REQUIRED"

        src_dir = tmp / "source_audit"
        src_dir.mkdir(parents=True)
        write_json(
            src_dir / "target_001_external_source_integrity.json",
            {
                "schema_version": "reid_stage5d_b1e_a_external_source_integrity_v1",
                "external": external,
                "evaluation": evaluation,
                "exact_file_duplicate": False,
                "external_mutated": False,
                "evaluation_mutated": False,
                "copied_to_outputs": False,
            },
        )

        ov_dir = tmp / "overlap_audit"
        ov_dir.mkdir(parents=True)
        write_json(ov_dir / "target_001_external_vs_sample_overlap_audit.json", audit)

        elig_dir = tmp / "eligibility"
        elig_dir.mkdir(parents=True)
        write_json(
            elig_dir / "target_001_external_enrollment_interval_eligibility.json",
            eligibility,
        )

        pkg = tmp / "review_packages" / "target_001_external_enrollment_preflight"
        pkg.mkdir(parents=True)
        sheet = render_overview_sheet(
            video_path=ext_path,
            fps=float(external["fps"]),
            frames=int(external["video_frames"]),
            stride=int(config["overview_sheet"]["sample_stride_frames"]),
            eligibility=eligibility,
        )
        sheet_path = pkg / "external_enrollment_overview_sheet_01.png"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise ExternalPreflightError("FAILED_STAGE5D_B1E_A_ATOMIC_OUTPUT png")

        tpl_dir = tmp / "templates"
        tpl_dir.mkdir(parents=True)
        write_json(
            tpl_dir / "target_001_external_enrollment_seed_review_template.json",
            blank_seed_template(
                source_path=external["path"],
                source_sha=external["sha256"],
                eligible_intervals=eligibility["eligible_intervals"],
            ),
        )

        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        write_json(
            tmp / "runtime" / "runtime.json",
            {
                "schema_version": "reid_stage5d_b1e_a_runtime_v1",
                "started_at": started,
                "offline_required": True,
                "network_download": 0,
                "new_detection": 0,
                "new_tracking": 0,
                "new_embedding": 0,
                "ocr": 0,
                "similarity_inference": 0,
                "target_selection": 0,
            },
        )
        eff = tmp / "effective_configs"
        eff.mkdir(parents=True)
        shutil.copy2(config_path, eff / config_path.name)

        if len(list(tmp.rglob("*.png"))) != 1:
            raise ExternalPreflightError("FAILED_STAGE5D_B1E_A_ATOMIC_OUTPUT png_count")
        if list(tmp.rglob("*.mp4")) or list(tmp.rglob("*.jpg")):
            raise ExternalPreflightError("FAILED_STAGE5D_B1E_A_ATOMIC_OUTPUT media")

        summary = {
            "schema_version": "reid_stage5d_b1e_a_summary_v1",
            "final_status": final_status,
            "project_head": head,
            "target_id": "target_001",
            "target_alias": handoff["target_definition"]["target_alias"],
            "human_verified_jersey_number": 5,
            "original_frozen_seed_code": FROZEN_SEED_CODE,
            "frozen_seed_enrollment_allowed": False,
            "source_overlap_decision": decision,
            "external_source": external,
            "evaluation_source": evaluation,
            "exact_file_duplicate": False,
            "verified_overlap_interval_count": len(
                audit["contiguous_overlapping_intervals"]
            ),
            "eligible_interval_count": eligibility["eligible_interval_count"],
            "has_min_duration_eligible_interval": eligibility[
                "has_min_duration_eligible_interval"
            ],
            "min_eligible_duration_sec": float(
                config["overlap_audit"]["min_eligible_duration_sec"]
            ),
            "png_count": 1,
            "mp4_copy_count": 0,
            "source_video_copy_count": 0,
            "detection_tracking_rows": 0,
            "crop_embedding_rows": 0,
            "ocr": 0,
            "similarity_ranking_rows": 0,
            "derived_anchors": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
            "new_detection": 0,
            "new_tracking": 0,
            "new_embedding": 0,
            "target_selection": 0,
            "exact_next_gate": exact_next,
        }
        write_json(tmp / "stage5d_b1e_a_summary.json", summary)

        n_files, listing = listing_sha(tmp)
        write_json(
            tmp / "stage5d_b1e_a_manifest.json",
            {
                "schema_version": "reid_stage5d_b1e_a_manifest_v1",
                "final_status": final_status,
                "project_head": head,
                "listing_file_count": n_files,
                "listing_sha256": listing,
                "external_sha256": external["sha256"],
                "sample_sha256": evaluation["sha256"],
                "source_overlap_decision": decision,
                "eligible_interval_count": eligibility["eligible_interval_count"],
                "png_count": 1,
                "gallery_members": 0,
            },
        )

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_b1e_a_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/external_enrollment_preflight_stage5d_target_001.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        summary = run(args.config.resolve(), args.project_root.resolve())
    except ExternalPreflightError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "overlap_decision": summary["source_overlap_decision"],
                "eligible_intervals": summary["eligible_interval_count"],
                "next": summary["exact_next_gate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
