#!/usr/bin/env python3
"""Stage 5D-B1E-D — external tracklet quality and anchor crop review package.

Uses only frozen EXT_004 / EXT_183 / EXT_198 bbox lineage. No new
detection/tracking, OSNet, OCR, similarity, or automatic anchor approval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
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

from football_analytics.reid.crop_select import (  # noqa: E402
    clamp_bbox_xyxy,
    float_bbox_to_int_crop,
)
from football_analytics.reid.quality import compute_image_metrics  # noqa: E402

CONFIG_SCHEMA = "reid_external_anchor_review_config_v1"
SELECTED_CODES = ("EXT_004", "EXT_183", "EXT_198")
FINAL_STATUS = "COMPLETED_STAGE5D_B1E_D_TARGET_001_EXTERNAL_ANCHOR_REVIEW_READY"
NEXT_GATE = "STAGE5D-B1E-E_TARGET_001_EXTERNAL_ANCHOR_MANUAL_REVIEW_AND_FREEZE"
TEMPLATE_FIELDS = (
    "anchor_candidate_id",
    "target_id",
    "source_occurrence_code",
    "frame_index",
    "video_time",
    "source_bbox",
    "crop_bbox",
    "crop_path",
    "crop_sha256",
    "manual_anchor_decision",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_confirmed",
    "manual_view_category",
    "manual_quality_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
ALLOWED_ANCHOR_DECISIONS = (
    "target_anchor_yes",
    "target_anchor_no",
    "uncertain",
    "invalid",
    "multi_person_ambiguous",
    "non_player",
)
ALLOWED_TRISTATE = ("yes", "no", "uncertain")
ALLOWED_VIEW = (
    "front",
    "rear",
    "left_side",
    "right_side",
    "front_oblique",
    "rear_oblique",
    "unknown",
)


class AnchorReviewError(RuntimeError):
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
        raise AnchorReviewError("unexpected config schema")
    if not config.get("offline_required"):
        raise AnchorReviewError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/"):
        raise AnchorReviewError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise AnchorReviewError("BLOCKED_STAGE5D_B1E_D_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise AnchorReviewError("BLOCKED_STAGE5D_B1E_D_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    allowed = {
        "scripts/run_reid_external_anchor_review_package.py",
        "configs/reid/external_anchor_review_stage5d_target_001.yaml",
        "tests/test_reid_external_anchor_review_package.py",
        "docs/setup/stage5d-target-external-tracklet-quality-and-anchor-review.md",
    }
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path not in allowed:
                raise AnchorReviewError(
                    "BLOCKED_STAGE5D_B1E_D_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != "Freeze target 001 external positive occurrences":
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_GIT_CONTRACT_MISMATCH message"
        )
    tracked = subprocess.check_output(
        [
            "git",
            "ls-files",
            "data/enrollment_clips/target_001_external_enrollment_v1.mp4",
        ],
        cwd=project_root,
        text=True,
    ).strip()
    if tracked:
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_GIT_CONTRACT_MISMATCH external_tracked"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not manifest.is_file():
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man = str(load_json(manifest).get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def edge_clipping_fraction(
    bbox: Sequence[float], *, width: int, height: int
) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    area = bbox_area(bbox)
    if area <= 0:
        return 1.0
    ix1, iy1 = max(x1, 0.0), max(y1, 0.0)
    ix2, iy2 = min(x2, float(width)), min(y2, float(height))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return float(max(0.0, min(1.0, 1.0 - (inter / area))))


def pad_bbox(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    fraction: float,
) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    px, py = fraction * bw, fraction * bh
    padded = [x1 - px, y1 - py, x2 + px, y2 + py]
    return clamp_bbox_xyxy(padded, video_width=width, video_height=height)


def dhash_hex(gray: np.ndarray, hash_size: int = 8) -> str:
    resized = cv2.resize(
        gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA
    )
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    bit_n = 0
    out = bytearray()
    for value in diff.flatten():
        bits = (bits << 1) | int(bool(value))
        bit_n += 1
        if bit_n == 8:
            out.append(bits)
            bits = 0
            bit_n = 0
    if bit_n:
        out.append(bits << (8 - bit_n))
    return out.hex()


def hamming_hex(a: str, b: str) -> int:
    ba = bytes.fromhex(a)
    bb = bytes.fromhex(b)
    if len(ba) != len(bb):
        return 10**9
    return sum(bin(x ^ y).count("1") for x, y in zip(ba, bb))


def temporal_bucket_name(
    *,
    frame_index: int,
    first: int,
    last: int,
    buckets: Sequence[str],
    representative_frame: int,
    rep_half_window: int,
) -> str:
    core = [b for b in buckets if b != "representative-support"]
    if not core:
        raise AnchorReviewError("empty temporal buckets")
    span = max(1, last - first)
    pos = (frame_index - first) / float(span)
    pos = max(0.0, min(1.0, pos))
    # Prefer representative-support near the frozen representative frame.
    if abs(frame_index - representative_frame) <= int(rep_half_window):
        if "representative-support" in buckets:
            return "representative-support"
    idx = min(len(core) - 1, int(pos * len(core)))
    if pos >= 1.0:
        idx = len(core) - 1
    return str(core[idx])


def validate_assets(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    ext = project_root / config["external_enrollment_source"]["path"]
    sample = project_root / config["evaluation_source"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if not ext.is_file() or ext.is_symlink():
        raise AnchorReviewError("external must be regular file")
    if ext.stat().st_size != int(config["external_enrollment_source"]["expected_bytes"]):
        raise AnchorReviewError("external bytes mismatch")
    if sha256_file(ext) != config["external_enrollment_source"]["expected_sha256"]:
        raise AnchorReviewError("external sha mismatch")
    if sha256_file(sample) != config["evaluation_source"]["expected_sha256"]:
        raise AnchorReviewError("sample sha mismatch")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise AnchorReviewError("yolo bytes mismatch")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise AnchorReviewError("yolo sha mismatch")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise AnchorReviewError("osnet sha mismatch")
    return {
        "path": config["external_enrollment_source"]["path"],
        "sha256": config["external_enrollment_source"]["expected_sha256"],
        "bytes": int(config["external_enrollment_source"]["expected_bytes"]),
        "width": int(config["external_enrollment_source"]["expected_width"]),
        "height": int(config["external_enrollment_source"]["expected_height"]),
        "fps": float(config["external_enrollment_source"]["expected_fps"]),
        "frames": int(config["external_enrollment_source"]["expected_frames"]),
        "yolo_sha256": config["yolo_checkpoint"]["expected_sha256"],
        "osnet_sha256": config["osnet_checkpoint"]["expected_sha256"],
    }


def validate_b1e_c(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root / config["stage5d_b1e_c_package"]["path"]
    summary = load_json(root / "stage5d_b1e_c_summary.json")
    freeze = load_json(
        root
        / "occurrence_freeze"
        / "target_001_external_positive_occurrence_freeze.json"
    )
    contract = load_json(
        root
        / "occurrence_freeze"
        / "target_001_external_occurrence_freeze_contract.json"
    )
    exp = config["stage5d_b1e_c_package"]
    if summary.get("final_status") != exp["expected_final_status"]:
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH status"
        )
    if summary.get("target_id") != "target_001":
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH target"
        )
    if summary.get("target_alias") != "sarı takım 5 numaralı oyuncu":
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH alias"
        )
    if summary.get("external_source_sha256") != config["external_enrollment_source"][
        "expected_sha256"
    ]:
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH source"
        )
    if int(summary["selected_positive_count"]) != 3:
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH positives"
        )
    if int(summary["reviewed_negative_count"]) != 0:
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH negatives"
        )
    if int(summary["unreviewed_count"]) != 245:
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH unreviewed"
        )
    if summary.get("review_scope") != "positive_occurrence_selection_only":
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH scope"
        )
    if tuple(summary["selected_external_candidate_codes"]) != SELECTED_CODES:
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH codes"
        )
    if list(summary["resolved_raw_track_ids"]) != [11, 388, 450]:
        raise AnchorReviewError(
            "BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH tracks"
        )
    for key in (
        "approved_anchor_crops",
        "embeddings",
        "gallery_members",
        "prototypes",
        "identity_assignments",
    ):
        if int(summary.get(key) or 0) != 0:
            raise AnchorReviewError(
                f"BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH {key}"
            )
    expected_lineage = {
        "EXT_004": (11, 0, 186, 185, 123),
        "EXT_183": (388, 456, 499, 44, 456),
        "EXT_198": (450, 511, 783, 272, 594),
    }
    for item in summary["observation_ranges"]:
        code = item["external_candidate_code"]
        exp_t = expected_lineage[code]
        got = (
            int(item["resolved_raw_track_id"]),
            int(item["first_frame"]),
            int(item["last_frame"]),
            int(item["observation_count"]),
            int(item["representative_frame"]),
        )
        if got != exp_t:
            raise AnchorReviewError(
                f"BLOCKED_STAGE5D_B1E_D_OCCURRENCE_FREEZE_CONTRACT_MISMATCH lineage {code}"
            )
    snap = resolve_snapshot_sha(
        Path(exp["snapshot_path"]), exp["expected_snapshot_sha256"]
    )
    return {
        "root": root,
        "summary": summary,
        "freeze": freeze,
        "contract": contract,
        "snapshot_sha256": snap,
    }


def validate_b1e_b(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root / config["stage5d_b1e_b_package"]["path"]
    summary = load_json(root / "stage5d_b1e_b_summary.json")
    exp = config["stage5d_b1e_b_package"]
    if summary.get("final_status") != (
        "COMPLETED_STAGE5D_B1E_B_EXTERNAL_TRACKING_SEED_REVIEW_READY"
    ):
        raise AnchorReviewError("B1E-B status mismatch")
    if int(summary["detection_total"]) != int(exp["expected_detection_total"]):
        raise AnchorReviewError("B1E-B detection_total mismatch")
    if int(summary["detection_frames_with_boxes"]) != int(
        exp["expected_detection_frames_with_boxes"]
    ):
        raise AnchorReviewError("B1E-B frame coverage mismatch")
    if int(summary["tracking_total_observations"]) != int(
        exp["expected_tracking_observations"]
    ):
        raise AnchorReviewError("B1E-B tracking observations mismatch")
    if int(summary["raw_track_count"]) != int(exp["expected_raw_track_count"]):
        raise AnchorReviewError("B1E-B raw track count mismatch")
    if summary.get("two_replay_determinism") is not True:
        raise AnchorReviewError("B1E-B determinism mismatch")
    return {"root": root, "summary": summary}


def load_selected_mapping(
    b1eb_root: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    path = b1eb_root / "inventory" / "target_001_external_track_candidate_mapping.jsonl"
    by_code: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        code = row["external_candidate_code"]
        if code in SELECTED_CODES:
            if code in by_code:
                raise AnchorReviewError(f"duplicate mapping for {code}")
            by_code[code] = row
    if set(by_code) != set(SELECTED_CODES):
        raise AnchorReviewError("selected mapping incomplete")
    for spec in config["selected_occurrences"]:
        code = spec["external_candidate_code"]
        row = by_code[code]
        if int(row["raw_external_track_id"]) != int(spec["raw_track_id"]):
            raise AnchorReviewError(f"raw track mismatch {code}")
        if int(row["first_frame"]) != int(spec["first_frame"]):
            raise AnchorReviewError(f"first_frame mismatch {code}")
        if int(row["last_frame"]) != int(spec["last_frame"]):
            raise AnchorReviewError(f"last_frame mismatch {code}")
        if int(row["observation_count"]) != int(spec["observation_count"]):
            raise AnchorReviewError(f"observation_count mismatch {code}")
        if int(row["representative_frame"]) != int(spec["representative_frame"]):
            raise AnchorReviewError(f"representative_frame mismatch {code}")
    total = sum(int(by_code[c]["observation_count"]) for c in SELECTED_CODES)
    if total != 501:
        raise AnchorReviewError(f"expected 501 observations, got {total}")
    return by_code


def load_tracks_by_frame(b1eb_root: Path) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with (b1eb_root / "tracking" / "tracks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            by_frame[int(row["frame_index"])].append(row)
    return by_frame


def hard_exclude(
    *,
    crop_w: int,
    crop_h: int,
    bbox_area_px: float,
    edge_clip: float,
    max_iou: float,
    hq: Mapping[str, Any],
    decode_ok: bool,
    lineage_ok: bool,
    sha_ok: bool,
) -> list[str]:
    reasons: list[str] = []
    if not decode_ok:
        reasons.append("crop_decode_failed")
    if not lineage_ok:
        reasons.append("source_lineage_incomplete")
    if not sha_ok:
        reasons.append("crop_sha_unavailable")
    if crop_w <= 0 or crop_h <= 0:
        reasons.append("non_positive_crop_geometry")
    if crop_h < int(hq["min_crop_height_px"]):
        reasons.append("crop_height_below_min")
    if crop_w < int(hq["min_crop_width_px"]):
        reasons.append("crop_width_below_min")
    if bbox_area_px < float(hq["min_bbox_area_px2"]):
        reasons.append("bbox_area_below_min")
    if edge_clip > float(hq["max_edge_clipping_fraction"]):
        reasons.append("edge_clipping_above_max")
    if max_iou > float(hq["max_person_iou"]):
        reasons.append("person_iou_above_max")
    return reasons


def rank_key(item: Mapping[str, Any]) -> tuple:
    q = item["quality"]
    # Prefer: low overlap, low edge clip, large area, high laplacian, earlier frame.
    return (
        float(q["max_person_iou"]),
        float(q["edge_clipping_fraction"]),
        -float(q["bbox_area"]),
        -float(q["laplacian_variance"]),
        int(item["frame_index"]),
    )


def select_candidates_for_occurrence(
    items: Sequence[Mapping[str, Any]],
    *,
    buckets: Sequence[str],
    max_n: int,
    min_desired: int,
    min_gap: int,
    dup_ham: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    passed = [i for i in items if i["hard_quality_pass"]]
    selected: list[dict[str, Any]] = []
    suppressed_dup = 0
    suppressed_gap = 0
    warnings: list[str] = []

    def conflicts(cand: Mapping[str, Any]) -> Optional[str]:
        for s in selected:
            if abs(int(cand["frame_index"]) - int(s["frame_index"])) < min_gap:
                return "gap"
            if (
                hamming_hex(cand["quality"]["perceptual_dhash"], s["quality"]["perceptual_dhash"])
                <= dup_ham
            ):
                return "dup"
        return None

    # First pass: one best per bucket.
    for bucket in buckets:
        pool = [i for i in passed if i["temporal_bucket"] == bucket]
        pool = sorted(pool, key=rank_key)
        for cand in pool:
            why = conflicts(cand)
            if why == "gap":
                suppressed_gap += 1
                continue
            if why == "dup":
                suppressed_dup += 1
                continue
            selected.append(dict(cand))
            break
        if len(selected) >= max_n:
            break

    # Fill remaining slots.
    if len(selected) < max_n:
        rest = sorted(
            [i for i in passed if i["frame_index"] not in {s["frame_index"] for s in selected}],
            key=rank_key,
        )
        for cand in rest:
            if len(selected) >= max_n:
                break
            why = conflicts(cand)
            if why == "gap":
                suppressed_gap += 1
                continue
            if why == "dup":
                suppressed_dup += 1
                continue
            selected.append(dict(cand))

    selected = sorted(selected, key=lambda x: int(x["frame_index"]))
    if len(passed) >= min_desired and len(selected) < min_desired:
        warnings.append(
            f"desired_min_{min_desired}_not_met_after_diversity_selected_{len(selected)}"
        )
    if len(passed) < min_desired:
        warnings.append(
            f"hard_quality_pass_below_min_desired pass={len(passed)} desired={min_desired}"
        )
    audit = {
        "hard_quality_pass": len(passed),
        "selected": len(selected),
        "suppressed_near_duplicate": suppressed_dup,
        "suppressed_min_frame_gap": suppressed_gap,
        "warnings": warnings,
        "selected_frames": [int(s["frame_index"]) for s in selected],
        "selected_buckets": [s["temporal_bucket"] for s in selected],
    }
    return selected, audit


def render_occurrence_sheet(
    candidates: Sequence[Mapping[str, Any]],
    *,
    occurrence_code: str,
) -> np.ndarray:
    n = len(candidates)
    cols = min(3, max(1, n))
    rows_n = int(math.ceil(n / cols)) if n else 1
    tile_w, tile_h = 420, 520
    sheet = np.full((rows_n * tile_h + 40, cols * tile_w, 3), 18, dtype=np.uint8)
    cv2.putText(
        sheet,
        f"{occurrence_code} external anchor crop review",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    if n == 0:
        cv2.putText(
            sheet,
            "no hard-quality candidates",
            (12, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return sheet
    for i, cand in enumerate(candidates):
        r, c = divmod(i, cols)
        tile = np.full((tile_h, tile_w, 3), 32, dtype=np.uint8)
        crop = cand["crop_image"]
        ctx = cand["context_image"].copy()
        x1, y1, x2, y2 = [int(v) for v in cand["source_bbox_int"]]
        cv2.rectangle(ctx, (x1, y1), (x2, y2), (0, 220, 255), 2)
        # Large crop.
        ch, cw = crop.shape[:2]
        crop_max_h, crop_max_w = 300, tile_w - 20
        scale = min(crop_max_w / max(1, cw), crop_max_h / max(1, ch))
        disp = cv2.resize(
            crop,
            (max(1, int(cw * scale)), max(1, int(ch * scale))),
            interpolation=cv2.INTER_AREA,
        )
        dh, dw = disp.shape[:2]
        ox = (tile_w - dw) // 2
        oy = 70
        tile[oy : oy + dh, ox : ox + dw] = disp
        # Small context.
        ctx_h, ctx_w = 120, 180
        cscale = min(ctx_w / ctx.shape[1], ctx_h / ctx.shape[0])
        cdisp = cv2.resize(
            ctx,
            (max(1, int(ctx.shape[1] * cscale)), max(1, int(ctx.shape[0] * cscale))),
            interpolation=cv2.INTER_AREA,
        )
        cdh, cdw = cdisp.shape[:2]
        tile[tile_h - cdh - 10 : tile_h - 10, 10 : 10 + cdw] = cdisp
        q = cand["quality"]
        lines = [
            str(cand["anchor_candidate_id"]),
            f"{occurrence_code} f={cand['frame_index']} t={cand['video_time']:.2f}s",
            f"{q['crop_width']}x{q['crop_height']} blur={q['laplacian_variance']:.1f}",
            f"iou={q['max_person_iou']:.2f} clip={q['edge_clipping_fraction']:.2f}",
        ]
        y = 18
        for text in lines:
            cv2.putText(
                tile,
                text,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            y += 14
        y0 = 40 + r * tile_h
        x0 = c * tile_w
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return sheet


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_b1e_d_contract_v1",
        "target_id": "target_001",
        "frozen_positive_occurrences_only": True,
        "unreviewed_occurrences_read": False,
        "existing_bbox_lineage_only": True,
        "no_new_detection": True,
        "no_new_tracking": True,
        "no_osnet": True,
        "no_ocr": True,
        "no_target_similarity": True,
        "deterministic_quality_selection": True,
        "temporal_diversity": True,
        "near_duplicate_suppression": True,
        "manual_approval_required": True,
        "candidate_is_not_anchor_until_freeze": True,
        "candidate_is_not_gallery_member": True,
        "unknown_identity_preserved": True,
        "automatic_gallery_growth": False,
        "reviewed_candidate_crops": 0,
        "approved_anchor_crops": 0,
        "embeddings": 0,
        "gallery_members": 0,
        "prototypes": 0,
        "identity_assignments": 0,
        "exact_next_gate": NEXT_GATE,
    }


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or _PROJECT_ROOT
    config = load_config(config_path)
    assert_no_path_traversal(config["output"]["final_dir"])
    head = assert_git_contract(project_root, config["project_head_expected"])
    final_dir = project_root / config["output"]["final_dir"]
    if final_dir.exists():
        raise AnchorReviewError("final_exists")

    assets = validate_assets(project_root, config)
    b1ec = validate_b1e_c(project_root, config)
    b1eb = validate_b1e_b(project_root, config)
    td = load_json(project_root / config["target_definition"]["path"])
    if td.get("target_id") != "target_001" or not td.get("target_definition_frozen"):
        raise AnchorReviewError("target definition mismatch")
    mapping = load_selected_mapping(b1eb["root"], config)
    tracks_by_frame = load_tracks_by_frame(b1eb["root"])

    width = assets["width"]
    height = assets["height"]
    fps = assets["fps"]
    pad_frac = float(config["crop_extraction"]["padding_fraction"])
    if pad_frac > float(config["crop_extraction"]["max_padding_fraction"]):
        raise AnchorReviewError("padding fraction exceeds max")
    hq = config["hard_quality"]
    sel_cfg = config["selection"]
    extraction_contract = {
        "schema_version": "reid_stage5d_b1e_d_crop_extraction_contract_v1",
        "padding_fraction": pad_frac,
        "max_padding_fraction": float(config["crop_extraction"]["max_padding_fraction"]),
        "clamp_to_frame": True,
        "bbox_source": "existing_bytetrack_observation",
        "no_bbox_refinement": True,
        "no_new_detection": True,
        "image_format": config["crop_extraction"]["image_format"],
    }
    extraction_contract_sha = sha256_bytes(
        json.dumps(extraction_contract, sort_keys=True).encode()
    )

    # Index track observations for selected raw IDs and validate against mapping.
    selected_raw = {int(s["raw_track_id"]): s for s in config["selected_occurrences"]}
    track_obs_index: dict[tuple[int, int], dict[str, Any]] = {}
    for fi, rows in tracks_by_frame.items():
        for row in rows:
            tid = int(row["raw_track_id"])
            if tid in selected_raw:
                key = (tid, int(fi))
                if key in track_obs_index:
                    raise AnchorReviewError(f"duplicate track observation {key}")
                track_obs_index[key] = row

    video_path = project_root / assets["path"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise AnchorReviewError("cannot open external video")

    quality_rows: list[dict[str, Any]] = []
    occurrence_items: dict[str, list[dict[str, Any]]] = {c: [] for c in SELECTED_CODES}
    exclusion_counts: dict[str, dict[str, int]] = {
        c: defaultdict(int) for c in SELECTED_CODES
    }
    pass_counts = {c: 0 for c in SELECTED_CODES}
    frame_cache: dict[int, np.ndarray] = {}

    def read_frame(fi: int) -> Optional[np.ndarray]:
        if fi in frame_cache:
            return frame_cache[fi]
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        frame_cache[fi] = frame
        # Bound cache size.
        if len(frame_cache) > 64:
            oldest = min(frame_cache)
            if oldest != fi:
                del frame_cache[oldest]
        return frame

    try:
        for spec in config["selected_occurrences"]:
            code = spec["external_candidate_code"]
            row = mapping[code]
            tid = int(spec["raw_track_id"])
            frames = [int(f) for f in row["observation_frames"]]
            bboxes = row["bbox_per_observation"]
            lineage = row["detection_lineage"]
            if len(frames) != len(bboxes) or len(frames) != len(lineage):
                raise AnchorReviewError(f"incomplete mapping lineage {code}")
            first, last = int(spec["first_frame"]), int(spec["last_frame"])
            buckets = list(spec["temporal_buckets"])
            for i, fi in enumerate(frames):
                bbox = [float(v) for v in bboxes[i]["bbox_xyxy"]]
                det_lin = lineage[i]
                track_row = track_obs_index.get((tid, fi))
                lineage_ok = track_row is not None and det_lin is not None
                if track_row is not None:
                    # Exact bbox agreement with tracking artifact.
                    tb = [float(v) for v in track_row["bbox_xyxy"]]
                    if any(abs(a - b) > 1e-3 for a, b in zip(bbox, tb)):
                        raise AnchorReviewError(
                            f"mapping/tracking bbox mismatch {code} f={fi}"
                        )
                    if track_row.get("source_video_sha256") != assets["sha256"]:
                        raise AnchorReviewError("track source sha mismatch")

                w0 = float(bbox[2] - bbox[0])
                h0 = float(bbox[3] - bbox[1])
                area0 = bbox_area(bbox)
                edge_clip = edge_clipping_fraction(bbox, width=width, height=height)
                others = [
                    r
                    for r in tracks_by_frame.get(fi, [])
                    if int(r["raw_track_id"]) != tid
                ]
                max_iou = 0.0
                for other in others:
                    max_iou = max(
                        max_iou, iou_xyxy(bbox, other["bbox_xyxy"])
                    )

                decode_ok = False
                sha_ok = False
                crop_img = None
                padded = None
                crop_int = None
                metrics = {
                    "grayscale_mean": 0.0,
                    "grayscale_std": 0.0,
                    "laplacian_variance": 0.0,
                }
                sat_mean = 0.0
                dhash = ""
                reasons: list[str] = []

                if w0 <= 0 or h0 <= 0:
                    reasons.append("non_positive_bbox")
                if edge_clip >= 1.0 - 1e-9:
                    reasons.append("bbox_completely_outside_frame")

                frame = read_frame(fi)
                if frame is None:
                    reasons.append("crop_decode_failed")
                else:
                    try:
                        padded = pad_bbox(
                            bbox, width=width, height=height, fraction=pad_frac
                        )
                        crop_int = float_bbox_to_int_crop(
                            padded, video_width=width, video_height=height
                        )
                        x1, y1, x2, y2 = crop_int
                        crop_img = frame[y1:y2, x1:x2].copy()
                        if crop_img.size == 0:
                            reasons.append("crop_decode_failed")
                        else:
                            decode_ok = True
                            metrics = compute_image_metrics(crop_img)
                            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                            sat_mean = float(np.mean(hsv[:, :, 1]))
                            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                            dhash = dhash_hex(gray, int(sel_cfg["dhash_size"]))
                            sha_ok = True
                    except Exception:
                        reasons.append("crop_decode_failed")
                        decode_ok = False
                        sha_ok = False

                crop_w = int(crop_int[2] - crop_int[0]) if crop_int else 0
                crop_h = int(crop_int[3] - crop_int[1]) if crop_int else 0
                more = hard_exclude(
                    crop_w=crop_w,
                    crop_h=crop_h,
                    bbox_area_px=area0,
                    edge_clip=edge_clip,
                    max_iou=max_iou,
                    hq=hq,
                    decode_ok=decode_ok and "crop_decode_failed" not in reasons,
                    lineage_ok=lineage_ok,
                    sha_ok=sha_ok,
                )
                for r in more:
                    if r not in reasons:
                        reasons.append(r)

                bucket = temporal_bucket_name(
                    frame_index=fi,
                    first=first,
                    last=last,
                    buckets=buckets,
                    representative_frame=int(spec["representative_frame"]),
                    rep_half_window=int(sel_cfg["representative_support_half_window"]),
                )
                span = max(1, last - first)
                norm_pos = (fi - first) / float(span)
                hard_pass = len(reasons) == 0
                if hard_pass:
                    pass_counts[code] += 1
                for r in reasons:
                    exclusion_counts[code][r] += 1

                source_bbox_int = None
                try:
                    source_bbox_int = float_bbox_to_int_crop(
                        clamp_bbox_xyxy(bbox, video_width=width, video_height=height),
                        video_width=width,
                        video_height=height,
                    )
                except Exception:
                    source_bbox_int = [0, 0, 0, 0]

                quality = {
                    "bbox_area": area0,
                    "bbox_width": w0,
                    "bbox_height": h0,
                    "bbox_area_frame_ratio": area0 / float(width * height),
                    "edge_clipping_fraction": edge_clip,
                    "crop_width": crop_w,
                    "crop_height": crop_h,
                    "laplacian_variance": float(metrics["laplacian_variance"]),
                    "brightness_mean": float(metrics["grayscale_mean"]),
                    "brightness_std": float(metrics["grayscale_std"]),
                    "saturation_mean": float(sat_mean),
                    "max_person_iou": float(max_iou),
                    "overlap_candidate_count": len(others),
                    "perceptual_dhash": dhash,
                    "temporal_distance_from_first": fi - first,
                    "normalized_temporal_position": float(norm_pos),
                }
                qrow = {
                    "target_id": "target_001",
                    "source_occurrence_code": code,
                    "raw_track_id": tid,
                    "frame_index": fi,
                    "video_time": fi / fps,
                    "original_bbox_xyxy": bbox,
                    "padded_crop_bbox_xyxy": padded,
                    "hard_quality_pass": hard_pass,
                    "exclusion_reasons": reasons,
                    "temporal_bucket": bucket,
                    "quality": quality,
                    "detection_lineage": det_lin,
                    "tracking_lineage": {
                        "raw_track_id": tid,
                        "frame_index": fi,
                        "present_in_tracks_jsonl": track_row is not None,
                    },
                    "source_video_sha256": assets["sha256"],
                    "extraction_contract_sha256": extraction_contract_sha,
                }
                quality_rows.append(qrow)
                occurrence_items[code].append(
                    {
                        **qrow,
                        "crop_image": crop_img,
                        "context_image": frame.copy() if frame is not None else None,
                        "source_bbox_int": source_bbox_int,
                        "crop_bbox_int": crop_int,
                    }
                )
    finally:
        cap.release()

    if len(quality_rows) != 501:
        raise AnchorReviewError(f"quality rows expected 501 got {len(quality_rows)}")

    # Selection.
    selected_all: list[dict[str, Any]] = []
    selection_audit: dict[str, Any] = {"occurrences": {}}
    for spec in config["selected_occurrences"]:
        code = spec["external_candidate_code"]
        chosen, audit = select_candidates_for_occurrence(
            occurrence_items[code],
            buckets=list(spec["temporal_buckets"]),
            max_n=int(sel_cfg["max_candidates_per_occurrence"]),
            min_desired=int(sel_cfg["min_desired_per_occurrence"]),
            min_gap=int(sel_cfg["min_frame_gap"]),
            dup_ham=int(sel_cfg["near_duplicate_dhash_hamming_max"]),
        )
        selection_audit["occurrences"][code] = audit
        selected_all.extend(chosen)

    if len(selected_all) > int(sel_cfg["max_total_candidates"]):
        raise AnchorReviewError("max total candidates exceeded")

    # Assign deterministic IDs by occurrence order then frame.
    selected_all = sorted(
        selected_all,
        key=lambda x: (SELECTED_CODES.index(x["source_occurrence_code"]), x["frame_index"]),
    )
    for i, item in enumerate(selected_all, start=1):
        item["anchor_candidate_id"] = f"target_001_ext_anchor_{i:03d}"

    tmp = project_root / (
        "_tmp_full_stage4b_rebuild_r2_stage5d_target_001_external_anchor_review_package_"
        + uuid.uuid4().hex
    )
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        inv = tmp / "inventory"
        quality_dir = tmp / "quality"
        anchor_review = tmp / "anchor_review"
        review_pkg = tmp / "review_packages" / "target_001_external_anchor_crop_review"
        crops_root = tmp / "crops"
        tpl_dir = tmp / "templates"
        runtime_dir = tmp / "runtime"
        cfg_dir = tmp / "effective_configs"
        for d in (
            inv,
            quality_dir,
            anchor_review,
            review_pkg,
            crops_root,
            tpl_dir,
            runtime_dir,
            cfg_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, cfg_dir / config_path.name)
        write_json(cfg_dir / "crop_extraction_contract.json", extraction_contract)

        # Persist quality inventory (no images).
        with (inv / "target_001_external_tracklet_observation_quality.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in quality_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Write selected crops + candidate inventory.
        candidate_rows: list[dict[str, Any]] = []
        for item in selected_all:
            code = item["source_occurrence_code"]
            crop_dir = crops_root / code
            crop_dir.mkdir(parents=True, exist_ok=True)
            cid = item["anchor_candidate_id"]
            rel = f"crops/{code}/{cid}.png"
            out = tmp / rel
            if item["crop_image"] is None:
                raise AnchorReviewError(f"missing crop image for {cid}")
            if not cv2.imwrite(str(out), item["crop_image"]):
                raise AnchorReviewError(f"failed crop write {cid}")
            crop_sha = sha256_file(out)
            cand = {
                "anchor_candidate_id": cid,
                "target_id": "target_001",
                "source_occurrence_code": code,
                "raw_track_id": item["raw_track_id"],
                "frame_index": item["frame_index"],
                "video_time": item["video_time"],
                "original_bbox_xyxy": item["original_bbox_xyxy"],
                "padded_crop_bbox_xyxy": item["padded_crop_bbox_xyxy"],
                "source_frame_width": width,
                "source_frame_height": height,
                "crop_width": item["quality"]["crop_width"],
                "crop_height": item["quality"]["crop_height"],
                "crop_path": rel,
                "crop_sha256": crop_sha,
                "source_video_path": assets["path"],
                "source_video_sha256": assets["sha256"],
                "detection_lineage": item["detection_lineage"],
                "tracking_lineage": item["tracking_lineage"],
                "extraction_contract_sha256": extraction_contract_sha,
                "temporal_bucket": item["temporal_bucket"],
                "quality": item["quality"],
                "manual_fields_blank": True,
                "manual_anchor_decision": "",
                "is_gallery_member": False,
                "is_approved_anchor": False,
            }
            candidate_rows.append(cand)
            # Attach for sheet rendering.
            item["crop_sha256"] = crop_sha
            item["crop_path"] = rel

        with (inv / "target_001_external_anchor_candidate_inventory.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in candidate_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Contact sheets.
        sheet_paths = []
        for code in SELECTED_CODES:
            cands = [c for c in selected_all if c["source_occurrence_code"] == code]
            sheet = render_occurrence_sheet(cands, occurrence_code=code)
            name = f"anchor_review_{code}.png"
            out = review_pkg / name
            if not cv2.imwrite(str(out), sheet):
                raise AnchorReviewError(f"failed sheet {name}")
            sheet_paths.append(
                f"review_packages/target_001_external_anchor_crop_review/{name}"
            )

        # Blank template.
        with (
            tpl_dir / "target_001_external_anchor_crop_review_template.csv"
        ).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TEMPLATE_FIELDS))
            writer.writeheader()
            for row in candidate_rows:
                writer.writerow(
                    {
                        "anchor_candidate_id": row["anchor_candidate_id"],
                        "target_id": "target_001",
                        "source_occurrence_code": row["source_occurrence_code"],
                        "frame_index": row["frame_index"],
                        "video_time": row["video_time"],
                        "source_bbox": json.dumps(row["original_bbox_xyxy"]),
                        "crop_bbox": json.dumps(row["padded_crop_bbox_xyxy"]),
                        "crop_path": row["crop_path"],
                        "crop_sha256": row["crop_sha256"],
                        "manual_anchor_decision": "",
                        "manual_crop_valid": "",
                        "manual_target_dominant": "",
                        "manual_single_person": "",
                        "manual_identity_confirmed": "",
                        "manual_view_category": "",
                        "manual_quality_notes": "",
                        "reviewer": "",
                        "final_approver": "",
                        "reviewed_at": "",
                    }
                )

        quality_summary = {
            "schema_version": "reid_stage5d_b1e_d_tracklet_quality_summary_v1",
            "source_observation_count": 501,
            "selected_occurrence_codes": list(SELECTED_CODES),
            "per_occurrence": {
                code: {
                    "observation_count": int(
                        next(
                            s["observation_count"]
                            for s in config["selected_occurrences"]
                            if s["external_candidate_code"] == code
                        )
                    ),
                    "hard_quality_pass": pass_counts[code],
                    "hard_quality_excluded": int(
                        next(
                            s["observation_count"]
                            for s in config["selected_occurrences"]
                            if s["external_candidate_code"] == code
                        )
                    )
                    - pass_counts[code],
                    "exclusion_reason_counts": dict(exclusion_counts[code]),
                }
                for code in SELECTED_CODES
            },
            "unreviewed_ext_source_reads": 0,
        }
        write_json(
            quality_dir / "target_001_external_tracklet_quality_summary.json",
            quality_summary,
        )
        selection_audit.update(
            {
                "schema_version": "reid_stage5d_b1e_d_anchor_selection_audit_v1",
                "max_per_occurrence": int(sel_cfg["max_candidates_per_occurrence"]),
                "max_total": int(sel_cfg["max_total_candidates"]),
                "min_frame_gap": int(sel_cfg["min_frame_gap"]),
                "near_duplicate_dhash_hamming_max": int(
                    sel_cfg["near_duplicate_dhash_hamming_max"]
                ),
                "total_selected": len(candidate_rows),
                "candidate_ids": [c["anchor_candidate_id"] for c in candidate_rows],
            }
        )
        write_json(
            quality_dir / "target_001_external_anchor_selection_audit.json",
            selection_audit,
        )

        contract = build_contract()
        write_json(tmp / "stage5d_b1e_d_contract.json", contract)
        write_json(
            anchor_review / "target_001_external_anchor_review_contract.json",
            {
                **contract,
                "extraction_contract_sha256": extraction_contract_sha,
                "hard_quality": dict(hq),
                "selection": dict(sel_cfg),
            },
        )

        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        write_json(
            runtime_dir / "runtime.json",
            {
                "schema_version": "reid_stage5d_b1e_d_runtime_v1",
                "started_at": started,
                "project_head": head,
                "osnet_loaded": False,
                "yolo_loaded": False,
                "yolo_inference": 0,
                "bytetrack_inference": 0,
                "network_download": 0,
            },
        )

        per_occ_counts = {
            code: sum(1 for c in candidate_rows if c["source_occurrence_code"] == code)
            for code in SELECTED_CODES
        }
        summary = {
            "schema_version": "reid_stage5d_b1e_d_summary_v1",
            "final_status": FINAL_STATUS,
            "project_head": head,
            "target_id": "target_001",
            "target_alias": "sarı takım 5 numaralı oyuncu",
            "external_source_sha256": assets["sha256"],
            "frozen_positive_occurrences": 3,
            "selected_occurrence_codes": list(SELECTED_CODES),
            "source_observation_count": 501,
            "unreviewed_ext_source_reads": 0,
            "hard_quality_pass_counts": pass_counts,
            "candidate_counts_per_occurrence": per_occ_counts,
            "total_candidate_count": len(candidate_rows),
            "contact_sheet_png_count": 3,
            "reviewed_candidate_crops": 0,
            "approved_anchor_crops": 0,
            "manual_decisions": 0,
            "embeddings": 0,
            "osnet_inference": 0,
            "ocr": 0,
            "similarity_ranking_rows": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
            "new_detection": 0,
            "new_tracking": 0,
            "b1e_c_snapshot_sha256": b1ec["snapshot_sha256"],
            "extraction_contract_sha256": extraction_contract_sha,
            "exact_next_gate": NEXT_GATE,
        }
        write_json(tmp / "stage5d_b1e_d_summary.json", summary)

        review_manifest = {
            "schema_version": "reid_stage5d_b1e_d_anchor_review_manifest_v1",
            "contact_sheets": sheet_paths,
            "candidate_count": len(candidate_rows),
            "crop_file_count": len(list((tmp / "crops").rglob("*.png"))),
            "template": "templates/target_001_external_anchor_crop_review_template.csv",
        }
        write_json(
            anchor_review / "target_001_external_anchor_review_manifest.json",
            review_manifest,
        )

        # Budget checks.
        crop_files = list((tmp / "crops").rglob("*.png"))
        if len(crop_files) > 18:
            raise AnchorReviewError("crop budget exceeded")
        if len(list(review_pkg.glob("*.png"))) != 3:
            raise AnchorReviewError("expected exactly 3 contact sheets")
        if list(tmp.rglob("*.mp4")):
            raise AnchorReviewError("mp4 forbidden")

        n_files, listing = listing_sha(tmp)
        manifest = {
            "schema_version": "reid_stage5d_b1e_d_manifest_v1",
            "final_status": FINAL_STATUS,
            "project_head": head,
            "listing_file_count_before_manifest": n_files,
            "listing_sha256_before_manifest": listing,
            "total_candidate_count": len(candidate_rows),
            "contact_sheet_png_count": 3,
            "approved_anchor_crops": 0,
            "gallery_members": 0,
            "embeddings": 0,
        }
        write_json(tmp / "stage5d_b1e_d_manifest.json", manifest)
        n_final, listing_final = listing_sha(tmp)
        manifest["listing_file_count"] = n_final
        manifest["listing_sha256"] = listing_final
        write_json(tmp / "stage5d_b1e_d_manifest.json", manifest)

        os.replace(str(tmp), str(final_dir))
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return {
        "final_status": FINAL_STATUS,
        "source_observations": 501,
        "candidates": len(selected_all),
        "per_occurrence": {
            c: sum(1 for x in selected_all if x["source_occurrence_code"] == c)
            for c in SELECTED_CODES
        },
        "next": NEXT_GATE,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 5D-B1E-D external anchor review package"
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.config.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
