#!/usr/bin/env python3
"""Stage 5D-F3E — external refinement crop quality + hard-negative review package.

Builds human crop-review artifacts from F3D-frozen EXT_161 and 35 same-team
distractor sources using existing B1E-B bbox lineage only. No sample access,
OSNet/OCR/similarity, approvals, or gallery mutation.
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

CONFIG_SCHEMA = "reid_external_refinement_crop_review_config_v1"
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FINAL_STATUS = (
    "COMPLETED_STAGE5D_F3E_TARGET_001_EXTERNAL_REFINEMENT_CROP_REVIEW_READY"
)
NEXT_GATE = (
    "STAGE5D-F3F_TARGET_001_EXTERNAL_REFINEMENT_CROP_MANUAL_REVIEW_AND_FREEZE"
)
ALLOWED_DIRTY = {
    "scripts/run_reid_external_refinement_crop_review_package.py",
    "configs/reid/external_refinement_crop_review_stage5d_target_001.yaml",
    "tests/test_reid_external_refinement_crop_review_package.py",
    "docs/setup/stage5d-target-external-refinement-crop-quality-and-hard-negative-review-package.md",
}
EXISTING_FROZEN_OCCURRENCES = ("EXT_004", "EXT_183", "EXT_198", "EXT_161")
SAME_TEAM_DISTRACTORS = (
    "EXT_001",
    "EXT_003",
    "EXT_005",
    "EXT_007",
    "EXT_010",
    "EXT_017",
    "EXT_028",
    "EXT_029",
    "EXT_034",
    "EXT_036",
    "EXT_042",
    "EXT_044",
    "EXT_047",
    "EXT_050",
    "EXT_055",
    "EXT_057",
    "EXT_061",
    "EXT_136",
    "EXT_138",
    "EXT_140",
    "EXT_158",
    "EXT_167",
    "EXT_175",
    "EXT_178",
    "EXT_182",
    "EXT_184",
    "EXT_208",
    "EXT_212",
    "EXT_215",
    "EXT_217",
    "EXT_218",
    "EXT_230",
    "EXT_242",
    "EXT_245",
    "EXT_247",
)
TRANSCRIPTION_CORRECTIONS = (
    ("EXT_016", "EXT_019"),
    ("EXT_048", "EXT_049"),
    ("EXT_065", "EXT_066"),
    ("EXT_080", "EXT_082"),
    ("EXT_228", "EXT_226"),
)
TARGET_TEMPLATE_FIELDS = (
    "target_crop_candidate_id",
    "source_occurrence_code",
    "raw_track_id",
    "frame_index",
    "video_time",
    "crop_path",
    "crop_sha256",
    "source_bbox",
    "crop_bbox",
    "quality_pass",
    "blur",
    "max_person_iou",
    "edge_clipping",
    "manual_target_crop_decision",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_confirmed",
    "manual_view_category",
    "manual_scale_category",
    "manual_quality_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
HN_TEMPLATE_FIELDS = (
    "hard_negative_candidate_id",
    "source_external_code",
    "raw_track_id",
    "selected_frame",
    "video_time",
    "crop_path",
    "crop_sha256",
    "source_bbox",
    "crop_bbox",
    "quality_pass",
    "quality_exception_review_only",
    "quality_exclusion_reasons",
    "blur",
    "max_person_iou",
    "edge_clipping",
    "frozen_source_decision",
    "human_visible_jersey_number",
    "manual_hard_negative_crop_decision",
    "manual_crop_valid",
    "manual_target_absent",
    "manual_same_team_confirmed",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_continuity_observed",
    "manual_view_category",
    "manual_scale_category",
    "manual_quality_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
ALLOWED_TARGET_DECISIONS = (
    "target_crop_yes",
    "target_crop_no",
    "uncertain",
    "invalid",
    "multi_person_ambiguous",
)
ALLOWED_HN_DECISIONS = (
    "hard_negative_crop_yes",
    "hard_negative_crop_no",
    "uncertain",
    "invalid",
    "multi_person_ambiguous",
)


class CropReviewError(RuntimeError):
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise CropReviewError("unexpected config schema")
    if not config.get("offline_required"):
        raise CropReviewError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise CropReviewError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise CropReviewError("BLOCKED_STAGE5D_F3E_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise CropReviewError("BLOCKED_STAGE5D_F3E_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise CropReviewError(
                    "BLOCKED_STAGE5D_F3E_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise CropReviewError("BLOCKED_STAGE5D_F3E_GIT_CONTRACT_MISMATCH message")
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise CropReviewError(
            "BLOCKED_STAGE5D_F3E_MANUAL_FREEZE_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not listing.is_file() or not manifest.is_file():
        raise CropReviewError(
            "BLOCKED_STAGE5D_F3E_MANUAL_FREEZE_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(man_payload.get("sha256") or man_payload.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise CropReviewError(
            "BLOCKED_STAGE5D_F3E_MANUAL_FREEZE_CONTRACT_MISMATCH snapshot_sha"
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


def hard_exclude(
    *,
    crop_w: int,
    crop_h: int,
    bbox_area_px: float,
    edge_clip: float,
    max_iou: float,
    hq: Mapping[str, Any],
    decode_ok: bool,
) -> list[str]:
    reasons: list[str] = []
    if not decode_ok:
        reasons.append("crop_decode_failed")
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


def observation_rank_key(item: Mapping[str, Any], *, midpoint: float) -> tuple:
    q = item["quality"]
    return (
        0 if item["hard_quality_pass"] else 1,
        float(q["max_person_iou"]),
        float(q["edge_clipping_fraction"]),
        -float(q["laplacian_variance"]),
        -float(q["bbox_area"]),
        abs(float(item["frame_index"]) - midpoint),
        int(item["frame_index"]),
    )


def temporal_bucket(
    frame_index: int, *, first: int, last: int, representative: int
) -> str:
    span = max(1, last - first)
    pos = (frame_index - first) / float(span)
    if abs(frame_index - representative) <= max(8, int(0.08 * span)):
        return "representative-support"
    if pos < 1.0 / 3.0:
        return "early"
    if pos < 2.0 / 3.0:
        return "middle"
    return "late"


def select_context_frames(
    obs_frames: Sequence[int],
    *,
    start_frame: int,
    end_frame: int,
    representative_frame: int,
) -> list[tuple[str, int]]:
    frames = sorted(set(int(f) for f in obs_frames))
    if not frames:
        raise CropReviewError("no observations")

    def nearest(target: int) -> int:
        return min(frames, key=lambda f: (abs(f - target), f))

    chosen: list[tuple[str, int]] = []
    for role, target in (
        ("START", start_frame),
        ("REP", representative_frame),
        ("END", end_frame),
    ):
        fi = nearest(target)
        if any(existing == fi for _, existing in chosen):
            continue
        chosen.append((role, fi))
    return chosen


def _fit(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / max(w, 1), box_h / max(h, 1))
    nw = max(1, int(math.floor(w * scale)))
    nh = max(1, int(math.floor(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def ext_numeric(code: str) -> int:
    return int(code.split("_", 1)[1])


def validate_f3d(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["stage5d_f3d_package"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_f3d_summary.json")
    freeze = load_json(
        root / "manual_freeze" / "target_001_external_refinement_manual_freeze.json"
    )
    cfg = config["stage5d_f3d_package"]
    if summary.get("final_status") != cfg["expected_final_status"]:
        raise CropReviewError(
            "BLOCKED_STAGE5D_F3E_MANUAL_FREEZE_CONTRACT_MISMATCH status"
        )
    checks = {
        "reviewed_target_view": 11,
        "target_anchor_expansion_yes": 2,
        "target_anchor_expansion_no": 7,
        "target_view_multi_person_ambiguous": 2,
        "reviewed_external_occurrence": 135,
        "additional_target_occurrence_yes": 1,
        "same_team_distractor_yes": 35,
        "other_team_player": 52,
        "non_player": 9,
        "uncertain": 4,
        "invalid": 5,
        "multi_person_ambiguous": 29,
        "inventory_coverage_missing": 0,
        "inventory_coverage_extra": 0,
        "total_frozen_target_occurrences_after_f3d": 4,
        "total_human_approved_target_anchor_sources_after_f3d": 9,
        "official_gallery_v1_members": 7,
        "hard_negative_gallery_members": 0,
        "new_embeddings": 0,
        "similarity_rows": 0,
    }
    for key, exp in checks.items():
        if summary.get(key) != exp:
            raise CropReviewError(
                f"BLOCKED_STAGE5D_F3E_MANUAL_FREEZE_CONTRACT_MISMATCH {key}"
            )
    if summary.get("sample_video_read") is not False:
        raise CropReviewError(
            "BLOCKED_STAGE5D_F3E_MANUAL_FREEZE_CONTRACT_MISMATCH sample"
        )
    if summary.get("gallery_mutation") is not False:
        raise CropReviewError(
            "BLOCKED_STAGE5D_F3E_MANUAL_FREEZE_CONTRACT_MISMATCH mutation"
        )
    if freeze.get("new_frozen_target_occurrence") != cfg["expected_additional_target"]:
        raise CropReviewError(
            "BLOCKED_STAGE5D_F3E_MANUAL_FREEZE_CONTRACT_MISMATCH EXT_161"
        )
    if list(freeze.get("newly_approved_target_expansion_candidates") or []) != list(
        cfg["expected_expansion_sources"]
    ):
        raise CropReviewError(
            "BLOCKED_STAGE5D_F3E_MANUAL_FREEZE_CONTRACT_MISMATCH expansion"
        )
    corr = [(c["from"], c["to"]) for c in freeze.get("transcription_corrections", [])]
    if corr != list(TRANSCRIPTION_CORRECTIONS):
        raise CropReviewError(
            "BLOCKED_STAGE5D_F3E_MANUAL_FREEZE_CONTRACT_MISMATCH transcription"
        )
    distractors = load_csv(
        root
        / "manual_freeze"
        / "target_001_external_same_team_distractor_sources_frozen.csv"
    )
    codes = [r["external_candidate_code"] for r in distractors]
    if codes != list(SAME_TEAM_DISTRACTORS):
        raise CropReviewError(
            "BLOCKED_STAGE5D_F3E_MANUAL_FREEZE_CONTRACT_MISMATCH distractors"
        )
    snap = resolve_snapshot_sha(
        Path(cfg["snapshot_path"]), cfg["expected_snapshot_sha256"]
    )
    return {
        "root": root,
        "summary": summary,
        "freeze": freeze,
        "distractors": distractors,
        "expansion": load_csv(
            root
            / "manual_freeze"
            / "target_001_external_target_anchor_expansion_sources_frozen.csv"
        ),
        "additional": load_csv(
            root
            / "manual_freeze"
            / "target_001_external_additional_target_occurrences_frozen.csv"
        ),
        "snapshot_sha256": snap,
    }


def validate_external_and_tracking(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    rel = config["external_enrollment_source"]["path"]
    assert_no_path_traversal(rel)
    path = project_root / rel
    exp = config["external_enrollment_source"]
    if not path.is_file() or path.is_symlink():
        raise CropReviewError("external file missing")
    if path.stat().st_size != int(exp["expected_bytes"]):
        raise CropReviewError("external bytes mismatch")
    if sha256_file(path) != exp["expected_sha256"]:
        raise CropReviewError("external sha mismatch")
    if int(exp["sample_overlap"]) != 0 or exp.get("future_evaluation_input") is not False:
        raise CropReviewError("external policy mismatch")
    b_rel = config["stage5d_b1e_b_package"]["path"]
    assert_no_path_traversal(b_rel)
    b_root = project_root / b_rel
    b_summary = load_json(b_root / "stage5d_b1e_b_summary.json")
    bexp = config["stage5d_b1e_b_package"]
    for key, exp_k in (
        ("detection_total", "expected_detection_total"),
        ("tracking_total_observations", "expected_tracking_observations"),
        ("raw_track_count", "expected_raw_track_count"),
        ("detection_frames_with_boxes", "expected_detection_frames_with_boxes"),
    ):
        if int(b_summary[key]) != int(bexp[exp_k]):
            raise CropReviewError(f"B1E-B {key} mismatch")
    if b_summary.get("two_replay_determinism") is not True:
        raise CropReviewError("B1E-B replay mismatch")
    mapping = {
        r["external_candidate_code"]: r
        for r in load_jsonl(
            b_root / "inventory" / "target_001_external_track_candidate_mapping.jsonl"
        )
    }
    if len(mapping) != 248:
        raise CropReviewError("EXT mapping count mismatch")
    if len(mapping) != len(set(mapping)):
        raise CropReviewError("EXT mapping not unique")
    return {
        "path": path,
        "sha256": exp["expected_sha256"],
        "width": int(exp["expected_width"]),
        "height": int(exp["expected_height"]),
        "fps": float(exp["expected_fps"]),
        "frames": int(exp["expected_frames"]),
        "b_root": b_root,
        "b_summary": b_summary,
        "mapping": mapping,
    }


def load_tracks_by_frame(b_root: Path) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with (b_root / "tracking" / "tracks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            by_frame[int(row["frame_index"])].append(row)
    return by_frame


def load_target_source_references(
    project_root: Path, config: Mapping[str, Any], f3d: Mapping[str, Any]
) -> list[dict[str, Any]]:
    b1ed = project_root / config["stage5d_b1e_d_package"]["path"]
    b1ee = project_root / config["stage5d_b1e_e_package"]["path"]
    f3c = project_root / config["stage5d_f3c_package"]["path"]
    approved = load_csv(
        b1ee / "anchor_freeze" / "target_001_external_approved_anchors_frozen.csv"
    )
    if [r["anchor_candidate_id"] for r in approved] != list(
        config["existing_gallery_anchor_ids"]
    ):
        raise CropReviewError("gallery anchor IDs mismatch")
    cand_inv = {
        r["anchor_candidate_id"]: r
        for r in load_jsonl(
            b1ed / "inventory" / "target_001_external_anchor_candidate_inventory.jsonl"
        )
    }
    view_inv = {
        r["target_view_candidate_id"]: r
        for r in load_jsonl(
            f3c
            / "inventory"
            / "target_001_external_target_view_candidate_inventory.jsonl"
        )
    }
    refs: list[dict[str, Any]] = []
    for row in approved:
        cid = row["anchor_candidate_id"]
        inv = cand_inv[cid]
        crop_path = b1ed / row["crop_path"]
        if not crop_path.is_file():
            raise CropReviewError(f"missing gallery crop {cid}")
        if sha256_file(crop_path) != row["crop_sha256"]:
            raise CropReviewError(f"gallery crop sha mismatch {cid}")
        img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if img is None:
            raise CropReviewError(f"cannot read gallery crop {cid}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        refs.append(
            {
                "source_id": cid,
                "source_kind": "gallery_v1_anchor",
                "source_occurrence_code": row["source_occurrence_code"],
                "frame_index": int(row["frame_index"]),
                "manual_view_category": row["manual_view_category"],
                "crop_path": str(Path(row["crop_path"])),
                "crop_sha256": row["crop_sha256"],
                "crop_image": img,
                "current_gallery_member": True,
                "perceptual_dhash": inv["quality"]["perceptual_dhash"],
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
            }
        )
    for row in f3d["expansion"]:
        cid = row["target_view_candidate_id"]
        inv = view_inv[cid]
        crop_path = f3c / inv["crop_path"]
        if not crop_path.is_file():
            raise CropReviewError(f"missing expansion crop {cid}")
        if sha256_file(crop_path) != row["crop_sha256"]:
            raise CropReviewError(f"expansion crop sha mismatch {cid}")
        img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if img is None:
            raise CropReviewError(f"cannot read expansion crop {cid}")
        refs.append(
            {
                "source_id": cid,
                "source_kind": "approved_expansion_source",
                "source_occurrence_code": row["source_occurrence_code"],
                "frame_index": int(row["frame_index"]),
                "manual_view_category": row["manual_view_category"],
                "crop_path": inv["crop_path"],
                "crop_sha256": row["crop_sha256"],
                "crop_image": img,
                "current_gallery_member": False,
                "perceptual_dhash": inv["quality"]["perceptual_dhash"],
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
            }
        )
    if len(refs) != 9:
        raise CropReviewError("expected 9 target source references")
    return refs


def compute_track_observations(
    *,
    mapping_row: Mapping[str, Any],
    tracks_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    read_frame,
    width: int,
    height: int,
    fps: float,
    pad_frac: float,
    hq: Mapping[str, Any],
    dhash_size: int,
    source_sha: str,
) -> list[dict[str, Any]]:
    tid = int(mapping_row["raw_external_track_id"])
    frames = [int(f) for f in mapping_row["observation_frames"]]
    bboxes = mapping_row["bbox_per_observation"]
    if len(frames) != len(bboxes):
        raise CropReviewError(f"incomplete lineage {mapping_row['external_candidate_code']}")
    first = int(mapping_row["first_frame"])
    last = int(mapping_row["last_frame"])
    rep = int(mapping_row["representative_frame"])
    midpoint = (first + last) / 2.0
    out: list[dict[str, Any]] = []
    for i, fi in enumerate(frames):
        bbox = [float(v) for v in bboxes[i]["bbox_xyxy"]]
        track_rows = [
            r for r in tracks_by_frame.get(fi, []) if int(r["raw_track_id"]) == tid
        ]
        if not track_rows:
            raise CropReviewError(f"missing track obs tid={tid} f={fi}")
        tb = [float(v) for v in track_rows[0]["bbox_xyxy"]]
        if any(abs(a - b) > 1e-3 for a, b in zip(bbox, tb)):
            raise CropReviewError(f"bbox mismatch tid={tid} f={fi}")
        others = [
            r for r in tracks_by_frame.get(fi, []) if int(r["raw_track_id"]) != tid
        ]
        max_iou = 0.0
        for other in others:
            max_iou = max(max_iou, iou_xyxy(bbox, other["bbox_xyxy"]))
        edge_clip = edge_clipping_fraction(bbox, width=width, height=height)
        area0 = bbox_area(bbox)
        decode_ok = False
        crop_img = None
        padded = None
        crop_int = None
        metrics = {"laplacian_variance": 0.0, "grayscale_mean": 0.0, "grayscale_std": 0.0}
        dhash = ""
        crop_bytes = b""
        reasons: list[str] = []
        frame = read_frame(fi)
        if frame is None:
            reasons.append("crop_decode_failed")
        else:
            try:
                padded = pad_bbox(bbox, width=width, height=height, fraction=pad_frac)
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
                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    dhash = dhash_hex(gray, dhash_size)
                    ok, buf = cv2.imencode(".png", crop_img)
                    if not ok:
                        raise CropReviewError("png encode failed")
                    crop_bytes = buf.tobytes()
            except CropReviewError:
                raise
            except Exception:
                reasons.append("crop_decode_failed")
                decode_ok = False
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
        )
        for r in more:
            if r not in reasons:
                reasons.append(r)
        hard_pass = len(reasons) == 0
        quality = {
            "bbox_area": area0,
            "edge_clipping_fraction": edge_clip,
            "crop_width": crop_w,
            "crop_height": crop_h,
            "laplacian_variance": float(metrics["laplacian_variance"]),
            "max_person_iou": float(max_iou),
            "perceptual_dhash": dhash,
        }
        out.append(
            {
                "frame_index": fi,
                "video_time": fi / fps,
                "source_bbox": bbox,
                "crop_bbox": padded,
                "hard_quality_pass": hard_pass,
                "exclusion_reasons": reasons,
                "quality": quality,
                "crop_image": crop_img,
                "crop_bytes": crop_bytes,
                "crop_sha256": sha256_bytes(crop_bytes) if crop_bytes else "",
                "temporal_bucket": temporal_bucket(
                    fi, first=first, last=last, representative=rep
                ),
                "midpoint_distance": abs(fi - midpoint),
                "raw_track_id": tid,
                "source_video_sha256": source_sha,
            }
        )
    return out


def select_ext161_candidates(
    observations: Sequence[Mapping[str, Any]],
    *,
    approved_refs: Sequence[Mapping[str, Any]],
    min_n: int,
    max_n: int,
    min_gap: int,
    dup_ham: int,
    buckets: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    approved_shas = {r["crop_sha256"] for r in approved_refs}
    approved_dhashes = [r["perceptual_dhash"] for r in approved_refs]
    midpoint = (
        min(int(o["frame_index"]) for o in observations)
        + max(int(o["frame_index"]) for o in observations)
    ) / 2.0
    pool = sorted(observations, key=lambda o: observation_rank_key(o, midpoint=midpoint))
    selected: list[dict[str, Any]] = []
    suppressed = {
        "exact_sha_duplicate": 0,
        "near_duplicate_dhash": 0,
        "min_frame_gap": 0,
        "not_hard_quality": 0,
    }

    def conflicts(cand: Mapping[str, Any]) -> Optional[str]:
        if cand["crop_sha256"] in approved_shas:
            return "sha"
        if any(
            hamming_hex(cand["quality"]["perceptual_dhash"], ad) <= dup_ham
            for ad in approved_dhashes
        ):
            return "dup_approved"
        for s in selected:
            if abs(int(cand["frame_index"]) - int(s["frame_index"])) < min_gap:
                return "gap"
            if (
                hamming_hex(
                    cand["quality"]["perceptual_dhash"],
                    s["quality"]["perceptual_dhash"],
                )
                <= dup_ham
            ):
                return "dup_self"
            if cand["crop_sha256"] == s["crop_sha256"]:
                return "sha"
        return None

    # Bucket pass: hard-quality only.
    for bucket in buckets:
        if len(selected) >= max_n:
            break
        for cand in pool:
            if cand["temporal_bucket"] != bucket:
                continue
            if not cand["hard_quality_pass"]:
                suppressed["not_hard_quality"] += 1
                continue
            why = conflicts(cand)
            if why == "sha":
                suppressed["exact_sha_duplicate"] += 1
                continue
            if why in ("dup_approved", "dup_self"):
                suppressed["near_duplicate_dhash"] += 1
                continue
            if why == "gap":
                suppressed["min_frame_gap"] += 1
                continue
            selected.append(dict(cand))
            break

    # Fill remaining from hard-quality only; never pad with redundant soft crops.
    if len(selected) < max_n:
        for cand in pool:
            if len(selected) >= max_n:
                break
            if not cand["hard_quality_pass"]:
                continue
            if int(cand["frame_index"]) in {int(s["frame_index"]) for s in selected}:
                continue
            why = conflicts(cand)
            if why:
                if why == "sha":
                    suppressed["exact_sha_duplicate"] += 1
                elif why in ("dup_approved", "dup_self"):
                    suppressed["near_duplicate_dhash"] += 1
                elif why == "gap":
                    suppressed["min_frame_gap"] += 1
                continue
            selected.append(dict(cand))

    selected = sorted(selected, key=lambda x: int(x["frame_index"]))
    if len(selected) < min_n:
        raise CropReviewError(
            f"EXT_161 produced {len(selected)} candidates; need >= {min_n}"
        )
    if len(selected) > max_n:
        raise CropReviewError("EXT_161 exceeded max candidates")
    audit = {
        "schema_version": "reid_target_001_EXT_161_target_crop_selection_audit_v1",
        "observation_count": len(observations),
        "hard_quality_pass_count": sum(1 for o in observations if o["hard_quality_pass"]),
        "selected_count": len(selected),
        "selected_frames": [int(s["frame_index"]) for s in selected],
        "selected_buckets": [s["temporal_bucket"] for s in selected],
        "suppressed": suppressed,
        "sample_used_for_candidate_selection": False,
        "identity_or_similarity_used": False,
    }
    return selected, audit


def select_distractor_candidate(
    observations: Sequence[Mapping[str, Any]],
    *,
    mapping_row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = int(mapping_row["first_frame"])
    last = int(mapping_row["last_frame"])
    midpoint = (first + last) / 2.0
    ordered = sorted(
        observations, key=lambda o: observation_rank_key(o, midpoint=midpoint)
    )
    hard = [o for o in ordered if o["hard_quality_pass"]]
    if hard:
        chosen = dict(hard[0])
        chosen["quality_exception_review_only"] = False
        mode = "hard_quality_pass"
    else:
        # Fallback: best available among all observations (still ranked).
        if not ordered:
            raise CropReviewError(
                f"no observations for {mapping_row['external_candidate_code']}"
            )
        chosen = dict(ordered[0])
        chosen["quality_exception_review_only"] = True
        mode = "quality_exception_review_only"
    audit = {
        "source_external_code": mapping_row["external_candidate_code"],
        "selection_mode": mode,
        "hard_quality_pass_count": len(hard),
        "selected_frame": int(chosen["frame_index"]),
        "quality_exception_review_only": chosen["quality_exception_review_only"],
        "exclusion_reasons": list(chosen.get("exclusion_reasons") or []),
    }
    return chosen, audit


def render_reference_sheet(refs: Sequence[Mapping[str, Any]], *, min_width: int) -> np.ndarray:
    cols = 3
    rows_n = int(math.ceil(len(refs) / cols))
    tile_w = max(900, int(math.ceil(min_width / cols)))
    tile_h = 520
    width = max(min_width, cols * tile_w)
    sheet = np.full((rows_n * tile_h + 56, width, 3), 18, dtype=np.uint8)
    cv2.putText(
        sheet,
        "Target source references (immutable; not reopened for review)",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    for i, ref in enumerate(refs):
        r, c = divmod(i, cols)
        tile = np.full((tile_h, tile_w, 3), 30, dtype=np.uint8)
        disp = _fit(ref["crop_image"], tile_w - 24, 360)
        dh, dw = disp.shape[:2]
        ox = (tile_w - dw) // 2
        tile[70 : 70 + dh, ox : ox + dw] = disp
        lines = [
            str(ref["source_id"]),
            f"{ref['source_occurrence_code']} f={ref['frame_index']}",
            f"view={ref['manual_view_category'] or 'n/a'}",
            f"gallery_member={str(ref['current_gallery_member']).lower()}",
            f"{ref['width']}x{ref['height']}",
        ]
        y = 18
        for text in lines:
            cv2.putText(
                tile, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA
            )
            y += 12
        y0 = 56 + r * tile_h
        x0 = c * tile_w
        sheet[y0 : y0 + tile_h, x0 : x0 + min(tile_w, sheet.shape[1] - x0)] = tile[
            :, : min(tile_w, sheet.shape[1] - x0)
        ]
    if sheet.shape[1] < min_width:
        raise CropReviewError("reference sheet width below minimum")
    return sheet


def render_candidate_sheet(
    candidates: Sequence[Mapping[str, Any]],
    *,
    title: str,
    min_width: int,
    label_prefix_lines: Optional[Sequence[str]] = None,
) -> np.ndarray:
    n = max(1, len(candidates))
    cols = min(4, n) if n else 1
    rows_n = int(math.ceil(len(candidates) / cols)) if candidates else 1
    tile_w = max(900, int(math.ceil(min_width / cols)))
    tile_h = 700
    width = max(min_width, cols * tile_w)
    sheet = np.full((rows_n * tile_h + 56, width, 3), 16, dtype=np.uint8)
    cv2.putText(
        sheet,
        title,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    for i, cand in enumerate(candidates):
        r, c = divmod(i, cols)
        tile = np.full((tile_h, tile_w, 3), 28, dtype=np.uint8)
        crop = cand["crop_image"]
        disp = _fit(crop, tile_w - 20, 300)
        dh, dw = disp.shape[:2]
        ox = (tile_w - dw) // 2
        tile[90 : 90 + dh, ox : ox + dw] = disp
        ctx_y = 90 + dh + 8
        ctxs = cand.get("context_images") or []
        if ctxs:
            ctx_w = (tile_w - 24) // len(ctxs)
            x = 8
            for role, img in ctxs:
                cdisp = _fit(img, ctx_w - 4, tile_h - ctx_y - 12)
                cdh, cdw = cdisp.shape[:2]
                tile[ctx_y : ctx_y + cdh, x : x + cdw] = cdisp
                cv2.putText(
                    tile,
                    role,
                    (x + 2, ctx_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (250, 250, 250),
                    1,
                    cv2.LINE_AA,
                )
                x += ctx_w
        q = cand["quality"]
        lines = list(label_prefix_lines or [])
        lines.extend(
            [
                str(cand.get("candidate_id") or cand.get("hard_negative_candidate_id")),
                f"{cand.get('source_occurrence_code') or cand.get('source_external_code')} "
                f"f={cand['frame_index']} t={cand['video_time']:.2f}s",
                f"{q['crop_width']}x{q['crop_height']} blur={q['laplacian_variance']:.1f}",
                f"iou={q['max_person_iou']:.2f} clip={q['edge_clipping_fraction']:.2f} "
                f"pass={str(cand['hard_quality_pass']).lower()}",
            ]
        )
        if cand.get("human_visible_jersey_number"):
            lines.append(f"HUMAN JERSEY METADATA={cand['human_visible_jersey_number']}")
        if cand.get("banner"):
            lines.append(str(cand["banner"]))
        y = 16
        for text in lines:
            cv2.putText(
                tile, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 245, 245), 1, cv2.LINE_AA
            )
            y += 12
        y0 = 56 + r * tile_h
        x0 = c * tile_w
        sheet[y0 : y0 + tile_h, x0 : x0 + min(tile_w, sheet.shape[1] - x0)] = tile[
            :, : min(tile_w, sheet.shape[1] - x0)
        ]
    if sheet.shape[1] < min_width:
        raise CropReviewError("candidate sheet width below minimum")
    return sheet


def draw_label(frame: np.ndarray, lines: Sequence[str], *, y0: int = 28) -> None:
    for i, text in enumerate(lines):
        y = y0 + i * 24
        cv2.putText(
            frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA
        )


def write_hn_review_mp4(
    path: Path,
    *,
    video_path: Path,
    candidates: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, Mapping[str, Any]],
    fps: float,
    half_window_sec: float,
    watermark: str,
    frozen_label: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CropReviewError("cannot open external video for HN mp4")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise CropReviewError("HN mp4 writer failed")
    half = int(round(half_window_sec * fps))
    total_frames = 0
    try:
        for cand in candidates:
            code = cand["source_external_code"]
            row = mapping[code]
            tid = int(row["raw_external_track_id"])
            bbox_by_f = {
                int(b["frame_index"]): [float(v) for v in b["bbox_xyxy"]]
                for b in row["bbox_per_observation"]
            }
            obs_frames = sorted(int(f) for f in row["observation_frames"])
            sel = int(cand["frame_index"])
            start = max(min(obs_frames), sel - half)
            end = min(max(obs_frames), sel + half)
            # Title card.
            card = np.full((h, w, 3), 24, dtype=np.uint8)
            draw_label(
                card,
                [
                    str(cand["hard_negative_candidate_id"]),
                    f"source {code}",
                    f"selected frame {sel}",
                    frozen_label,
                    "CROP APPROVAL PENDING",
                ],
                y0=80,
            )
            for _ in range(max(8, int(round(fps * 0.4)))):
                writer.write(card)
                total_frames += 1
            for fi in range(start, end + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise CropReviewError(f"failed read frame {fi}")
                out = frame.copy()
                if fi in bbox_by_f:
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox_by_f[fi]]
                    cv2.rectangle(out, (x1, y1), (x2, y2), (40, 200, 255), 2)
                q = cand["quality"]
                lines = [
                    str(cand["hard_negative_candidate_id"]),
                    f"{code} frame={fi} selected={sel}",
                    frozen_label,
                    f"blur={q['laplacian_variance']:.1f} iou={q['max_person_iou']:.2f} "
                    f"clip={q['edge_clipping_fraction']:.2f}",
                ]
                if cand.get("human_visible_jersey_number"):
                    lines.append(
                        f"HUMAN JERSEY METADATA={cand['human_visible_jersey_number']}"
                    )
                draw_label(out, lines)
                cv2.putText(
                    out,
                    watermark,
                    (12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    out,
                    watermark,
                    (12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(out)
                total_frames += 1
    finally:
        writer.release()
        cap.release()
    return {
        "frame_count": total_frames,
        "fps": fps,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "candidate_count": len(candidates),
        "bbox_interpolation": False,
        "existing_observations_only": True,
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_f3e_external_refinement_crop_review_contract_v1",
        "target_id": TARGET_ID,
        "external_only": True,
        "sample_read": False,
        "existing_bbox_lineage_only": True,
        "target_source_references": 9,
        "existing_gallery_members": 7,
        "approved_non_gallery_target_sources": 2,
        "distractor_source_count": 35,
        "distractor_candidate_count": 35,
        "one_candidate_per_distractor_source": True,
        "hard_negative_approvals": 0,
        "target_crop_approvals": 0,
        "new_embeddings": 0,
        "similarity_rows": 0,
        "gallery_mutation": False,
        "threshold": False,
        "identity_assignment": False,
        "human_crop_approval_required": True,
        "exact_next_gate": NEXT_GATE,
    }


def make_tmp(project_root: Path, final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f3e_crop_review_{final_dir.name}_{token}"
    tmp.mkdir(parents=True, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise CropReviewError("final_exists")
    tmp.rename(final_dir)


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or _PROJECT_ROOT
    config = load_config(config_path)
    assert_no_path_traversal(config["output"]["final_dir"])
    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )
    final_dir = project_root / config["output"]["final_dir"]
    if final_dir.exists():
        raise CropReviewError("final_exists")

    if config["evaluation_source"].get("decode_forbidden") is not True:
        raise CropReviewError("BLOCKED_STAGE5D_F3E_SAMPLE_LEAKAGE decode_policy")
    sample_path = project_root / config["evaluation_source"]["path"]
    runtime_audit = {
        "schema_version": "reid_stage5d_f3e_runtime_audit_v1",
        "sample_video_read": False,
        "sample_crop_read_count": 0,
        "sample_embedding_read_count": 0,
        "sample_score_row_read_count": 0,
        "sample_rank_row_read_count": 0,
        "sample_used_for_candidate_selection": False,
        "sample_used_for_quality_selection": False,
        "sample_used_for_gallery_optimization": False,
        "new_detection": False,
        "new_tracking": False,
        "osnet_inference": False,
        "ocr_inference": False,
        "similarity_scoring": 0,
        "new_embeddings": 0,
        "gallery_mutation": False,
        "network_download": 0,
        "sample_path_exists_but_unread": sample_path.is_file(),
    }

    f3d = validate_f3d(project_root, config)
    # Anti-overfit upstream readable for contract only.
    f3b_summary = load_json(
        project_root
        / config["stage5d_f3b_package"]["path"]
        / "stage5d_f3b_summary.json"
    )
    if f3b_summary.get("gallery_members") != 7:
        raise CropReviewError("F3B gallery members mismatch")
    gallery = load_json(
        project_root / config["gallery_v1"]["path"] / "stage5d_b1e_f_summary.json"
    )
    if int(gallery["individual_gallery_members"]) != 7:
        raise CropReviewError("gallery-v1 members mismatch")

    external = validate_external_and_tracking(project_root, config)
    refs = load_target_source_references(project_root, config, f3d)
    mapping = external["mapping"]
    if "EXT_161" not in mapping:
        raise CropReviewError("EXT_161 missing from B1E-B mapping")
    for code in SAME_TEAM_DISTRACTORS:
        if code not in mapping:
            raise CropReviewError(f"distractor {code} missing from mapping")
        if code in EXISTING_FROZEN_OCCURRENCES and code != "EXT_161":
            pass

    tracks_by_frame = load_tracks_by_frame(external["b_root"])
    width, height, fps = external["width"], external["height"], external["fps"]
    pad_frac = float(config["crop_extraction"]["padding_fraction"])
    hq = config["hard_quality"]
    sel = config["selection"]
    min_width = int(config["contact_sheets"]["min_width_px"])

    needed_codes = ["EXT_161", *SAME_TEAM_DISTRACTORS]
    needed_frames: set[int] = set()
    for code in needed_codes:
        for fi in mapping[code]["observation_frames"]:
            needed_frames.add(int(fi))

    cap = cv2.VideoCapture(str(external["path"]))
    if not cap.isOpened():
        raise CropReviewError("cannot open external enrollment video")
    frame_cache: dict[int, np.ndarray] = {}

    def read_frame(fi: int) -> Optional[np.ndarray]:
        if fi in frame_cache:
            return frame_cache[fi]
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        frame_cache[fi] = frame
        if len(frame_cache) > 96:
            oldest = min(k for k in frame_cache if k != fi)
            del frame_cache[oldest]
        return frame

    try:
        ext161_obs = compute_track_observations(
            mapping_row=mapping["EXT_161"],
            tracks_by_frame=tracks_by_frame,
            read_frame=read_frame,
            width=width,
            height=height,
            fps=fps,
            pad_frac=pad_frac,
            hq=hq,
            dhash_size=int(sel["dhash_size"]),
            source_sha=external["sha256"],
        )
        target_cands, target_audit = select_ext161_candidates(
            ext161_obs,
            approved_refs=refs,
            min_n=int(sel["ext_161_min_candidates"]),
            max_n=int(sel["ext_161_max_candidates"]),
            min_gap=int(sel["min_frame_gap"]),
            dup_ham=int(sel["near_duplicate_dhash_hamming_max"]),
            buckets=list(sel["temporal_buckets"]),
        )
        for i, cand in enumerate(target_cands, start=1):
            cand["candidate_id"] = f"target_001_ext_refine_target_candidate_{i:03d}"
            cand["source_occurrence_code"] = "EXT_161"
            cand["target_identity_source_level_human_confirmed"] = True
            cand["manual_crop_approval_pending"] = True
            cand["gallery_member"] = False
            cand["embedding_input"] = False
            cand["automatic_enrollment"] = False
            row = mapping["EXT_161"]
            bbox_by_f = {
                int(b["frame_index"]): [float(v) for v in b["bbox_xyxy"]]
                for b in row["bbox_per_observation"]
            }
            contexts = []
            for role, cfi in select_context_frames(
                [int(f) for f in row["observation_frames"]],
                start_frame=int(row["first_frame"]),
                end_frame=int(row["last_frame"]),
                representative_frame=int(cand["frame_index"]),
            ):
                cframe = read_frame(cfi)
                if cframe is None:
                    raise CropReviewError(f"context frame missing {cfi}")
                cframe = cframe.copy()
                cb = bbox_by_f[cfi]
                x1, y1, x2, y2 = [int(round(v)) for v in cb]
                cv2.rectangle(cframe, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(
                    cframe,
                    f"{role} f={cfi}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )
                contexts.append((role, cframe))
            cand["context_images"] = contexts

        distractor_meta = {
            r["external_candidate_code"]: r for r in f3d["distractors"]
        }
        hn_cands: list[dict[str, Any]] = []
        hn_audits: list[dict[str, Any]] = []
        silently_dropped = 0
        for idx, code in enumerate(SAME_TEAM_DISTRACTORS, start=1):
            obs = compute_track_observations(
                mapping_row=mapping[code],
                tracks_by_frame=tracks_by_frame,
                read_frame=read_frame,
                width=width,
                height=height,
                fps=fps,
                pad_frac=pad_frac,
                hq=hq,
                dhash_size=int(sel["dhash_size"]),
                source_sha=external["sha256"],
            )
            chosen, audit = select_distractor_candidate(obs, mapping_row=mapping[code])
            jersey = distractor_meta[code].get("manual_visible_jersey_number") or ""
            chosen["hard_negative_candidate_id"] = (
                f"target_001_ext_hard_negative_candidate_{idx:03d}"
            )
            chosen["source_external_code"] = code
            chosen["source_occurrence_code"] = code
            chosen["candidate_id"] = chosen["hard_negative_candidate_id"]
            chosen["human_visible_jersey_number"] = jersey
            chosen["frozen_source_decision"] = "same_team_distractor_yes"
            chosen["target_identity_negative_source"] = True
            chosen["manual_crop_approval_pending"] = True
            chosen["hard_negative_gallery_member"] = False
            chosen["embedding_input"] = False
            chosen["automatic_enrollment"] = False
            chosen["banner"] = "SAME-TEAM DISTRACTOR SOURCE — CROP APPROVAL PENDING"
            row = mapping[code]
            bbox_by_f = {
                int(b["frame_index"]): [float(v) for v in b["bbox_xyxy"]]
                for b in row["bbox_per_observation"]
            }
            contexts = []
            for role, cfi in select_context_frames(
                [int(f) for f in row["observation_frames"]],
                start_frame=int(row["first_frame"]),
                end_frame=int(row["last_frame"]),
                representative_frame=int(chosen["frame_index"]),
            ):
                cframe = read_frame(cfi)
                if cframe is None:
                    raise CropReviewError(f"context frame missing {cfi}")
                cframe = cframe.copy()
                cb = bbox_by_f[cfi]
                x1, y1, x2, y2 = [int(round(v)) for v in cb]
                cv2.rectangle(cframe, (x1, y1), (x2, y2), (40, 200, 255), 2)
                contexts.append((role, cframe))
            chosen["context_images"] = contexts
            hn_cands.append(chosen)
            hn_audits.append(audit)
        if len(hn_cands) != 35 or silently_dropped != 0:
            raise CropReviewError("distractor candidate coverage mismatch")
    finally:
        cap.release()

    tmp = make_tmp(project_root, final_dir)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        inv = tmp / "inventory"
        quality = tmp / "quality"
        review = tmp / "review"
        review_pkg = tmp / "review_packages" / "target_001_external_refinement_crop_review"
        videos = tmp / "videos"
        templates = tmp / "templates"
        runtime = tmp / "runtime"
        cfg_dir = tmp / "effective_configs"
        crops_t = review_pkg / "crops" / "EXT_161"
        crops_hn = review_pkg / "crops" / "hard_negative"
        for d in (
            inv,
            quality,
            review,
            review_pkg,
            videos,
            templates,
            runtime,
            cfg_dir,
            crops_t,
            crops_hn,
        ):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, cfg_dir / config_path.name)

        # Reference sheet.
        ref_sheet = render_reference_sheet(refs, min_width=min_width)
        if not cv2.imwrite(
            str(review_pkg / "target_001_existing_target_source_reference.png"), ref_sheet
        ):
            raise CropReviewError("failed reference sheet")

        # Target crops + inventory + sheet.
        with (
            inv / "target_001_EXT_161_target_crop_candidate_inventory.jsonl"
        ).open("w", encoding="utf-8") as handle:
            for cand in target_cands:
                rel = (
                    "review_packages/target_001_external_refinement_crop_review/crops/"
                    f"EXT_161/{cand['candidate_id']}.png"
                )
                (tmp / rel).write_bytes(cand["crop_bytes"])
                if sha256_file(tmp / rel) != cand["crop_sha256"]:
                    raise CropReviewError("target crop sha mismatch")
                rec = {
                    "target_crop_candidate_id": cand["candidate_id"],
                    "source_occurrence_code": "EXT_161",
                    "raw_track_id": int(cand["raw_track_id"]),
                    "frame_index": int(cand["frame_index"]),
                    "video_time": float(cand["video_time"]),
                    "crop_path": rel,
                    "crop_sha256": cand["crop_sha256"],
                    "source_bbox": cand["source_bbox"],
                    "crop_bbox": cand["crop_bbox"],
                    "quality": cand["quality"],
                    "hard_quality_pass": cand["hard_quality_pass"],
                    "temporal_bucket": cand["temporal_bucket"],
                    "gallery_member": False,
                    "embedding_input": False,
                    "automatic_enrollment": False,
                    "manual_crop_approval_pending": True,
                    "target_identity_source_level_human_confirmed": True,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        target_sheet = render_candidate_sheet(
            target_cands,
            title="EXT_161 target crop candidates (approval pending; not gallery)",
            min_width=min_width,
        )
        if not cv2.imwrite(
            str(review_pkg / "target_001_EXT_161_target_crop_candidates.png"), target_sheet
        ):
            raise CropReviewError("failed target sheet")

        # Hard-negative crops + inventory + sheets.
        with (
            inv / "target_001_external_hard_negative_crop_candidate_inventory.jsonl"
        ).open("w", encoding="utf-8") as handle:
            for cand in hn_cands:
                rel = (
                    "review_packages/target_001_external_refinement_crop_review/crops/"
                    f"hard_negative/{cand['hard_negative_candidate_id']}.png"
                )
                (tmp / rel).write_bytes(cand["crop_bytes"])
                if sha256_file(tmp / rel) != cand["crop_sha256"]:
                    raise CropReviewError("HN crop sha mismatch")
                rec = {
                    "hard_negative_candidate_id": cand["hard_negative_candidate_id"],
                    "source_external_code": cand["source_external_code"],
                    "raw_track_id": int(cand["raw_track_id"]),
                    "selected_frame": int(cand["frame_index"]),
                    "video_time": float(cand["video_time"]),
                    "crop_path": rel,
                    "crop_sha256": cand["crop_sha256"],
                    "source_bbox": cand["source_bbox"],
                    "crop_bbox": cand["crop_bbox"],
                    "quality": cand["quality"],
                    "hard_quality_pass": cand["hard_quality_pass"],
                    "quality_exception_review_only": cand["quality_exception_review_only"],
                    "quality_exclusion_reasons": cand.get("exclusion_reasons") or [],
                    "human_visible_jersey_number": cand.get("human_visible_jersey_number")
                    or "",
                    "frozen_source_decision": "same_team_distractor_yes",
                    "target_identity_negative_source": True,
                    "hard_negative_gallery_member": False,
                    "embedding_input": False,
                    "automatic_enrollment": False,
                    "manual_crop_approval_pending": True,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

        sheet_sizes = [12, 12, 11]
        offset = 0
        for i, n in enumerate(sheet_sizes, start=1):
            chunk = hn_cands[offset : offset + n]
            offset += n
            sheet = render_candidate_sheet(
                chunk,
                title=(
                    f"Hard-negative crop candidates sheet {i:02d} "
                    "(SAME-TEAM DISTRACTOR SOURCE — CROP APPROVAL PENDING)"
                ),
                min_width=min_width,
            )
            out = review_pkg / f"target_001_hard_negative_crop_candidates_{i:02d}.png"
            if not cv2.imwrite(str(out), sheet):
                raise CropReviewError(f"failed HN sheet {i}")
        if offset != 35:
            raise CropReviewError("HN sheet coverage mismatch")

        # Templates.
        with (
            templates / "target_001_EXT_161_target_crop_review_template.csv"
        ).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TARGET_TEMPLATE_FIELDS)
            writer.writeheader()
            for cand in target_cands:
                row = {k: "" for k in TARGET_TEMPLATE_FIELDS}
                row.update(
                    {
                        "target_crop_candidate_id": cand["candidate_id"],
                        "source_occurrence_code": "EXT_161",
                        "raw_track_id": int(cand["raw_track_id"]),
                        "frame_index": int(cand["frame_index"]),
                        "video_time": float(cand["video_time"]),
                        "crop_path": (
                            "review_packages/target_001_external_refinement_crop_review/crops/"
                            f"EXT_161/{cand['candidate_id']}.png"
                        ),
                        "crop_sha256": cand["crop_sha256"],
                        "source_bbox": json.dumps(cand["source_bbox"]),
                        "crop_bbox": json.dumps(cand["crop_bbox"]),
                        "quality_pass": str(cand["hard_quality_pass"]).lower(),
                        "blur": f"{cand['quality']['laplacian_variance']:.6f}",
                        "max_person_iou": f"{cand['quality']['max_person_iou']:.6f}",
                        "edge_clipping": f"{cand['quality']['edge_clipping_fraction']:.6f}",
                    }
                )
                writer.writerow(row)

        with (
            templates / "target_001_external_hard_negative_crop_review_template.csv"
        ).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HN_TEMPLATE_FIELDS)
            writer.writeheader()
            for cand in hn_cands:
                row = {k: "" for k in HN_TEMPLATE_FIELDS}
                row.update(
                    {
                        "hard_negative_candidate_id": cand["hard_negative_candidate_id"],
                        "source_external_code": cand["source_external_code"],
                        "raw_track_id": int(cand["raw_track_id"]),
                        "selected_frame": int(cand["frame_index"]),
                        "video_time": float(cand["video_time"]),
                        "crop_path": (
                            "review_packages/target_001_external_refinement_crop_review/crops/"
                            f"hard_negative/{cand['hard_negative_candidate_id']}.png"
                        ),
                        "crop_sha256": cand["crop_sha256"],
                        "source_bbox": json.dumps(cand["source_bbox"]),
                        "crop_bbox": json.dumps(cand["crop_bbox"]),
                        "quality_pass": str(cand["hard_quality_pass"]).lower(),
                        "quality_exception_review_only": str(
                            cand["quality_exception_review_only"]
                        ).lower(),
                        "quality_exclusion_reasons": json.dumps(
                            cand.get("exclusion_reasons") or []
                        ),
                        "blur": f"{cand['quality']['laplacian_variance']:.6f}",
                        "max_person_iou": f"{cand['quality']['max_person_iou']:.6f}",
                        "edge_clipping": f"{cand['quality']['edge_clipping_fraction']:.6f}",
                        "frozen_source_decision": "same_team_distractor_yes",
                        "human_visible_jersey_number": cand.get(
                            "human_visible_jersey_number"
                        )
                        or "",
                    }
                )
                writer.writerow(row)

        # Review video.
        mp4_meta = write_hn_review_mp4(
            videos / "target_001_external_hard_negative_crop_review.mp4",
            video_path=external["path"],
            candidates=hn_cands,
            mapping=mapping,
            fps=fps,
            half_window_sec=float(config["review_video"]["context_half_window_sec"]),
            watermark=str(config["review_video"]["watermark"]),
            frozen_label=str(config["review_video"]["frozen_source_label"]),
        )

        write_json(
            quality / "target_001_EXT_161_target_crop_selection_audit.json", target_audit
        )
        hn_pass = sum(1 for c in hn_cands if c["hard_quality_pass"])
        hn_exc = sum(1 for c in hn_cands if c["quality_exception_review_only"])
        write_json(
            quality / "target_001_external_hard_negative_crop_selection_audit.json",
            {
                "schema_version": "reid_target_001_external_hard_negative_crop_selection_audit_v1",
                "distractor_source_count": 35,
                "candidate_count": 35,
                "one_per_source": True,
                "source_silently_dropped": 0,
                "hard_quality_pass_count": hn_pass,
                "quality_exception_review_only_count": hn_exc,
                "per_source": hn_audits,
                "sample_used_for_candidate_selection": False,
                "jersey_metadata_used_for_selection": False,
                "identity_or_similarity_used": False,
            },
        )
        write_json(runtime / "target_001_external_refinement_crop_access_audit.json", runtime_audit)

        contract = {
            **build_contract(),
            "ext_161_target_candidate_count": len(target_cands),
            "generated_at": generated_at,
            "project_head": head,
            "f3d_snapshot_sha256": f3d["snapshot_sha256"],
            "external_source_sha256": external["sha256"],
        }
        write_json(
            review / "target_001_external_refinement_crop_review_contract.json", contract
        )
        write_json(tmp / "stage5d_f3e_contract.json", {**contract, "final_status": FINAL_STATUS})

        file_count, files_sha = listing_sha(tmp)
        summary = {
            "schema_version": "reid_stage5d_f3e_external_refinement_crop_review_summary_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "existing_frozen_target_occurrences": 4,
            "official_gallery_v1_members": 7,
            "approved_target_source_count": 9,
            "immutable_approved_expansion_sources": 2,
            "ext_161_target_crop_candidates": len(target_cands),
            "ext_161_selected_frames": [int(c["frame_index"]) for c in target_cands],
            "same_team_distractor_sources": 35,
            "distractor_crop_candidates": 35,
            "hard_quality_pass_distractor_candidates": hn_pass,
            "quality_exception_distractor_candidates": hn_exc,
            "source_silently_dropped": 0,
            "manual_crop_decisions": 0,
            "approved_new_target_crops": 0,
            "approved_hard_negative_crops": 0,
            "target_gallery_members": 7,
            "hard_negative_gallery_members": 0,
            "new_embeddings": 0,
            "similarity_rows": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "sample_video_read": False,
            "sample_crop_read_count": 0,
            "sample_embedding_read_count": 0,
            "sample_score_row_read_count": 0,
            "sample_rank_row_read_count": 0,
            "sample_used_for_candidate_selection": False,
            "target_reference_sheet": 1,
            "target_candidate_sheet": 1,
            "hard_negative_sheets": 3,
            "hard_negative_sheet_distribution": sheet_sizes,
            "diagnostic_mp4": 1,
            "target_candidate_crop_copies": len(target_cands),
            "hard_negative_candidate_crop_copies": 35,
            "source_video_copy": 0,
            "f3d_snapshot_sha256": f3d["snapshot_sha256"],
            "external_source_sha256": external["sha256"],
            "output_file_count": file_count,
            "output_listing_sha256": files_sha,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3e_summary.json", summary)
        write_json(
            review / "target_001_external_refinement_crop_review_manifest.json",
            {
                "schema_version": "reid_target_001_external_refinement_crop_review_manifest_v1",
                "reference_sheet": "target_001_existing_target_source_reference.png",
                "target_candidate_sheet": "target_001_EXT_161_target_crop_candidates.png",
                "hard_negative_sheets": [
                    f"target_001_hard_negative_crop_candidates_{i:02d}.png"
                    for i in range(1, 4)
                ],
                "review_video": "videos/target_001_external_hard_negative_crop_review.mp4",
                "review_video_sha256": mp4_meta["sha256"],
                "templates": [
                    "templates/target_001_EXT_161_target_crop_review_template.csv",
                    "templates/target_001_external_hard_negative_crop_review_template.csv",
                ],
            },
        )
        write_json(
            tmp / "stage5d_f3e_manifest.json",
            {
                "schema_version": "reid_stage5d_f3e_external_refinement_crop_review_manifest_v1",
                "final_status": FINAL_STATUS,
                "exact_next_gate": NEXT_GATE,
                "project_head": head,
                "output_file_count": file_count,
                "output_listing_sha256": files_sha,
                "artifact_budget": {
                    "target_source_reference_png": 1,
                    "target_candidate_png": 1,
                    "hard_negative_candidate_png": 3,
                    "diagnostic_mp4": 1,
                    "target_candidate_crop_copies": len(target_cands),
                    "hard_negative_candidate_crop_copies": 35,
                    "source_mp4_copies": 0,
                    "npy": 0,
                    "new_embeddings": 0,
                    "similarity_rows": 0,
                    "target_approvals": 0,
                    "hard_negative_approvals": 0,
                    "gallery_mutation": 0,
                    "identity_assignments": 0,
                },
                "generated_at": generated_at,
            },
        )

        if list(tmp.rglob("*.npy")) or list(tmp.rglob("*.npz")):
            raise CropReviewError("embedding artifacts forbidden")
        if list(tmp.rglob("sample.mp4")):
            raise CropReviewError("BLOCKED_STAGE5D_F3E_SAMPLE_LEAKAGE")
        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    return load_json(final_dir / "stage5d_f3e_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/external_refinement_crop_review_stage5d_target_001.yaml",
    )
    args = parser.parse_args(argv)
    summary = run(args.config)
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "exact_next_gate": summary["exact_next_gate"],
                "ext_161_target_crop_candidates": summary[
                    "ext_161_target_crop_candidates"
                ],
                "distractor_crop_candidates": summary["distractor_crop_candidates"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
