#!/usr/bin/env python3
"""Stage 5D-B1 — temporal context review package for target_001 anchors.

Visual context only. No manual decisions, gallery membership, OCR, similarity,
new detection/tracking/embedding, or identity assignment.
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
CONFIG_SCHEMA = "reid_target_temporal_context_review_config_v1"
ALLOWED_TEMPORAL_DECISIONS = (
    "target_anchor_yes",
    "target_anchor_no",
    "uncertain",
    "invalid",
    "multi_person_ambiguous",
    "non_player",
)
ALLOWED_TRISTATE = ("yes", "no", "uncertain")
TEMPLATE_FIELDS = (
    "anchor_candidate_id",
    "segment_id",
    "raw_track_id",
    "representative_frame",
    "context_start_frame",
    "context_end_frame",
    "temporal_review_decision",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_identity_continuity_observed",
    "manual_human_verified_number_seen_in_context",
    "manual_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
EXPECTED_CANDIDATE_IDS = [
    f"target_001_anchor_{i:03d}" for i in range(1, 10)
]


class TemporalContextError(RuntimeError):
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
        raise TemporalContextError("unexpected config schema")
    if not config.get("offline_required"):
        raise TemporalContextError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    parts = Path(rel).parts
    if ".." in parts or rel.startswith("/"):
        raise TemporalContextError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise TemporalContextError("BLOCKED_STAGE5D_B1_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise TemporalContextError("BLOCKED_STAGE5D_B1_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    allowed = {
        "scripts/run_reid_target_temporal_context_review_package.py",
        "configs/reid/target_temporal_context_review_stage5d_target_001.yaml",
        "tests/test_reid_target_temporal_context_review_package.py",
        "docs/setup/stage5d-target-temporal-context-review-package.md",
    }
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path not in allowed:
                raise TemporalContextError(
                    "BLOCKED_STAGE5D_B1_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != "Add target 001 anchor review package":
        raise TemporalContextError("BLOCKED_STAGE5D_B1_GIT_CONTRACT_MISMATCH message")
    return head


def resolve_snapshot_sha(snapshot_path: Path) -> str:
    if not snapshot_path.is_file():
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not manifest.is_file():
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH snapshot_sidecar"
        )
    sidecar_sha = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_sha = str(load_json(manifest).get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (sidecar_sha == man_sha == actual):
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def validate_stage5d_b_package(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = project_root / config["stage5d_b_package"]["path"]
    summary = load_json(root / "stage5d_b_summary.json")
    td = load_json(root / "target_definition" / "target_001_definition_frozen.json")
    man = load_json(root / "anchor_review" / "target_001_anchor_review_manifest.json")
    inv_sum = load_json(
        root / "inventory" / "target_001_anchor_candidate_summary.json"
    )
    inv_path = root / "inventory" / "target_001_anchor_candidate_inventory.jsonl"
    ann_path = root / "templates" / "target_001_anchor_review_annotation_template.csv"

    if summary.get("final_status") != config["stage5d_b_package"]["expected_final_status"]:
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH status"
        )
    if td.get("target_id") != "target_001" or td.get("target_definition_frozen") is not True:
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH target"
        )
    if td.get("target_alias") != "sarı takım 5 numaralı oyuncu":
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH alias"
        )
    if int(td.get("human_verified_jersey_number")) != 5:
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH jersey"
        )
    if td.get("jersey_number_provenance") != "human_verified_by_user_not_automated_ocr":
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH provenance"
        )
    if int(summary.get("embedded_segments_audited") or 0) != 150:
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH embedded"
        )
    if int(summary.get("eligible_anchor_candidates") or 0) != 9:
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH eligible"
        )
    if int(summary.get("contact_sheet_count") or 0) != 1:
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH sheets"
        )
    item_count = sum(
        int(d["item_count"]) for d in summary.get("contact_sheet_item_distribution") or []
    )
    if item_count != 9:
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH sheet_items"
        )
    for key in (
        "manual_decisions",
        "approved_anchors",
        "gallery_members",
        "prototypes",
        "identity_assignments",
        "similarity_ranking_rows",
    ):
        if int(summary.get(key) or 0) != 0:
            raise TemporalContextError(
                f"BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH {key}"
            )
    if summary.get("automated_jersey_used") is not False:
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH auto_jersey"
        )

    candidates = man.get("candidates") or []
    cids = [c["anchor_candidate_id"] for c in candidates]
    expected = list(config["stage5d_b_package"]["expected_candidate_ids"])
    if cids != expected:
        raise TemporalContextError(
            f"BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH cids={cids}"
        )

    with ann_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 9 or any(r.get("manual_anchor_decision") for r in rows):
        raise TemporalContextError(
            "BLOCKED_STAGE5D_B1_ANCHOR_PACKAGE_CONTRACT_MISMATCH annotation"
        )

    snap_sha = resolve_snapshot_sha(Path(config["stage5d_b_package"]["snapshot_path"]))
    return {
        "path": config["stage5d_b_package"]["path"],
        "candidates": candidates,
        "target_definition": td,
        "inventory_summary": inv_sum,
        "inventory_jsonl_sha256": sha256_file(inv_path),
        "annotation_template_blank": True,
        "snapshot_sha256": snap_sha,
        "summary": summary,
    }


def validate_stage5c_closure(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = project_root / config["stage5c_closure"]["path"]
    policy = load_json(root / config["stage5c_closure"]["policy_json"])
    required = {
        "stage5c_status": "closed",
        "stage5e_automated_jersey_channel_mode": "diagnostic_only",
        "automated_parseq_gallery_enrollment_allowed": False,
        "discovery_reserve_opened": False,
        "holdout_reserve_opened": False,
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise TemporalContextError(
                f"BLOCKED_STAGE5D_B1_TEMPORAL_EXCLUSION_CONTRACT {key}={policy.get(key)!r}"
            )
    return required


def validate_assets(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    video = project_root / config["source_video"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if not video.is_file() or video.is_symlink():
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY missing")
    if video.stat().st_size != int(config["source_video"]["expected_bytes"]):
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY bytes")
    if sha256_file(video) != config["source_video"]["expected_sha256"]:
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY sha")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY yolo_bytes")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY yolo_sha")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY osnet_sha")

    # Metadata-only duration (no full re-detection). Frame/fps from config contract
    # already validated against upstream in prior gates; confirm OpenCV props match.
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY open")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if frames != int(config["source_video"]["expected_frames"]):
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY frames")
    if width != int(config["source_video"]["expected_width"]):
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY width")
    if height != int(config["source_video"]["expected_height"]):
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY height")
    if abs(fps - float(config["source_video"]["expected_fps"])) > 1e-6:
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY fps")
    duration = frames / fps
    if abs(duration - 34.111979) > 0.05:
        raise TemporalContextError(
            f"BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY duration={duration}"
        )
    return {
        "path": config["source_video"]["path"],
        "sha256": config["source_video"]["expected_sha256"],
        "bytes": int(config["source_video"]["expected_bytes"]),
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": duration,
        "yolo_unchanged": True,
        "osnet_unchanged": True,
        "new_detection": 0,
        "new_tracking": 0,
    }


def load_segment_lineage(
    project_root: Path,
    config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    seg_view = (
        project_root
        / config["upstream"]["stage5_replay"]
        / config["upstream"]["segment_view_subdir"]
    )
    obs_path = seg_view / "segment_observations.jsonl"
    seg_path = seg_view / "track_segments.jsonl"
    if not obs_path.is_file() or not seg_path.is_file():
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SEGMENT_LINEAGE missing")

    needed = {str(c["segment_id"]) for c in candidates}
    segments: dict[str, dict[str, Any]] = {}
    for line in seg_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row["segment_id"])
        if sid in needed:
            segments[sid] = row
    if set(segments) != needed:
        raise TemporalContextError(
            f"BLOCKED_STAGE5D_B1_SEGMENT_LINEAGE missing_segments "
            f"{sorted(needed - set(segments))}"
        )

    obs_by_seg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in obs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row.get("segment_id") or "")
        if sid not in needed:
            continue
        src = row.get("source_observation") or {}
        bbox = src.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            raise TemporalContextError(
                f"BLOCKED_STAGE5D_B1_SEGMENT_LINEAGE bbox {sid}:{row.get('frame_index')}"
            )
        obs_by_seg[sid].append(
            {
                "frame_index": int(row["frame_index"]),
                "raw_track_id": int(row["raw_track_id"]),
                "bbox_xyxy": [float(x) for x in bbox],
                "timestamp_sec": float(src.get("timestamp_sec") or 0.0),
                "source_observation_sha256": row.get("source_observation_sha256"),
                "source_row_index": row.get("source_row_index"),
            }
        )

    out: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        sid = str(cand["segment_id"])
        seg = segments[sid]
        obs = sorted(obs_by_seg.get(sid, []), key=lambda r: r["frame_index"])
        if not obs:
            raise TemporalContextError(
                f"BLOCKED_STAGE5D_B1_SEGMENT_LINEAGE empty_obs {sid}"
            )
        if int(cand["raw_track_id"]) != int(seg["raw_track_id"]):
            raise TemporalContextError(
                f"BLOCKED_STAGE5D_B1_SEGMENT_LINEAGE track_mismatch {sid}"
            )
        if int(cand["raw_track_id"]) != int(obs[0]["raw_track_id"]):
            raise TemporalContextError(
                f"BLOCKED_STAGE5D_B1_SEGMENT_LINEAGE obs_track_mismatch {sid}"
            )
        frames = [o["frame_index"] for o in obs]
        seg_start = int(seg["first_observation_frame"])
        seg_end = int(seg["last_observation_frame"])
        if frames[0] != seg_start or frames[-1] != seg_end:
            raise TemporalContextError(
                f"BLOCKED_STAGE5D_B1_SEGMENT_LINEAGE range_mismatch {sid}"
            )
        rep = int(cand["frame_index"])
        if rep not in set(frames):
            raise TemporalContextError(
                f"BLOCKED_STAGE5D_B1_SEGMENT_LINEAGE rep_missing {sid}:{rep}"
            )
        # Representative crop still exists and SHA matches Stage 5D-B.
        crop_path = Path(cand["source_crop_path"])
        if not crop_path.is_file():
            raise TemporalContextError(
                f"BLOCKED_STAGE5D_B1_SEGMENT_LINEAGE crop_missing {sid}"
            )
        if sha256_file(crop_path) != cand["source_crop_sha256"]:
            raise TemporalContextError(
                f"BLOCKED_STAGE5D_B1_SEGMENT_LINEAGE crop_sha {sid}"
            )
        out[str(cand["anchor_candidate_id"])] = {
            "candidate": cand,
            "segment": seg,
            "observations": obs,
            "segment_start_frame": seg_start,
            "segment_end_frame": seg_end,
            "representative_frame": rep,
        }
    return out


def compute_context_window(
    *,
    segment_start: int,
    segment_end: int,
    representative: int,
    max_before: int,
    max_after: int,
) -> dict[str, Any]:
    if not (segment_start <= representative <= segment_end):
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SEGMENT_LINEAGE rep_out_of_segment")
    raw_start = representative - max_before
    raw_end = representative + max_after
    ctx_start = max(segment_start, raw_start)
    ctx_end = min(segment_end, raw_end)
    return {
        "selected_context_start_frame": int(ctx_start),
        "selected_context_end_frame": int(ctx_end),
        "truncated_left": raw_start < segment_start,
        "truncated_right": raw_end > segment_end,
        "source_time_start_sec": ctx_start / 30.0,
        "source_time_end_sec": ctx_end / 30.0,
    }


def select_sheet_observation_frames(
    observation_frames: Sequence[int],
    *,
    representative: int,
    ctx_start: int,
    ctx_end: int,
    offsets: Sequence[int],
) -> list[int]:
    """Deterministic nearest-observation selection with duplicate collapse."""
    in_window = [f for f in observation_frames if ctx_start <= f <= ctx_end]
    if not in_window:
        return []
    targets = [ctx_start]
    for off in offsets:
        if int(off) == 0:
            targets.append(representative)
        else:
            targets.append(representative + int(off))
    targets.append(ctx_end)

    selected: list[int] = []
    seen: set[int] = set()
    for target in targets:
        nearest = min(in_window, key=lambda f: (abs(f - target), f))
        if nearest not in seen:
            seen.add(nearest)
            selected.append(nearest)
    # Chronological display order after collapse.
    return sorted(selected)


def _fit_display(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / max(w, 1), box_h / max(h, 1))
    nw = max(1, int(math.floor(w * scale)))
    nh = max(1, int(math.floor(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def zoom_from_bbox(
    frame: np.ndarray,
    bbox_xyxy: Sequence[float],
    *,
    padding_ratio: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    xa = max(0, int(math.floor(x0 - pad_x)))
    ya = max(0, int(math.floor(y0 - pad_y)))
    xb = min(w, int(math.ceil(x1 + pad_x)))
    yb = min(h, int(math.ceil(y1 + pad_y)))
    crop = frame[ya:yb, xa:xb]
    if crop.size == 0:
        return np.full((40, 40, 3), 40, dtype=np.uint8)
    return crop


def annotate_full_frame(
    frame: np.ndarray,
    *,
    bbox_xyxy: Optional[Sequence[float]],
    anchor_candidate_id: str,
    frame_index: int,
    fps: float,
    is_representative: bool,
    bbox_color: Sequence[int],
    bbox_thickness: int,
) -> np.ndarray:
    out = frame.copy()
    if bbox_xyxy is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in bbox_xyxy]
        cv2.rectangle(
            out,
            (x0, y0),
            (x1, y1),
            tuple(int(c) for c in bbox_color),
            int(bbox_thickness),
        )
    label = "REPRESENTATIVE" if is_representative else "CONTEXT"
    time_s = frame_index / fps
    lines = [
        str(anchor_candidate_id),
        f"frame={frame_index}",
        f"t={time_s:.3f}s",
        label,
    ]
    y = 28
    for text in lines:
        cv2.putText(
            out,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y += 28
    return out


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise TemporalContextError(
            f"BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY read_frame={frame_index}"
        )
    return frame


def render_candidate_row(
    panels: Sequence[Mapping[str, Any]],
    *,
    panel_w: int = 360,
    panel_h: int = 420,
) -> np.ndarray:
    n = max(1, len(panels))
    row = np.full((panel_h, n * panel_w, 3), 20, dtype=np.uint8)
    for i, panel in enumerate(panels):
        tile = np.full((panel_h, panel_w, 3), 36, dtype=np.uint8)
        header = [
            f"f={panel['frame_index']}",
            f"t={panel['time_sec']:.2f}s",
            "REP" if panel["is_representative"] else "CTX",
        ]
        y = 18
        for text in header:
            cv2.putText(
                tile,
                text,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            y += 16
        full = _fit_display(panel["full_annotated"], panel_w - 12, 200)
        fh, fw = full.shape[:2]
        ox = (panel_w - fw) // 2
        oy = 60
        tile[oy : oy + fh, ox : ox + fw] = full
        zoom = _fit_display(panel["zoom"], panel_w - 12, 130)
        zh, zw = zoom.shape[:2]
        zx = (panel_w - zw) // 2
        zy = 280
        tile[zy : zy + zh, zx : zx + zw] = zoom
        x0 = i * panel_w
        row[:, x0 : x0 + panel_w] = tile
    return row


def render_temporal_sheet(
    candidate_rows: Sequence[tuple[str, np.ndarray]],
) -> np.ndarray:
    if not candidate_rows:
        raise TemporalContextError("empty temporal sheet")
    max_w = max(row.shape[1] for _, row in candidate_rows)
    label_h = 36
    total_h = sum(row.shape[0] + label_h for _, row in candidate_rows)
    sheet = np.full((total_h, max_w, 3), 12, dtype=np.uint8)
    y = 0
    for cid, row in candidate_rows:
        cv2.putText(
            sheet,
            cid,
            (12, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        y += label_h
        h, w = row.shape[:2]
        sheet[y : y + h, 0:w] = row
        y += h
    return sheet


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise TemporalContextError(f"FAILED_STAGE5D_B1_ATOMIC_OUTPUT png {path}")


def write_context_clip(
    path: Path,
    *,
    video_path: Path,
    ctx_start: int,
    ctx_end: int,
    bbox_by_frame: Mapping[int, Sequence[float]],
    anchor_candidate_id: str,
    representative: int,
    fps: float,
    bbox_color: Sequence[int],
    bbox_thickness: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY open_clip")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY dims")
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        cap.release()
        raise TemporalContextError("FAILED_STAGE5D_B1_ATOMIC_OUTPUT writer")
    frame_count = 0
    for frame_index in range(ctx_start, ctx_end + 1):
        # Re-seek each frame for deterministic positioning after first probe.
        frame = read_frame(cap, frame_index)
        bbox = bbox_by_frame.get(frame_index)
        annotated = annotate_full_frame(
            frame,
            bbox_xyxy=bbox,
            anchor_candidate_id=anchor_candidate_id,
            frame_index=frame_index,
            fps=fps,
            is_representative=(frame_index == representative),
            bbox_color=bbox_color,
            bbox_thickness=bbox_thickness,
        )
        writer.write(annotated)
        frame_count += 1
    writer.release()
    cap.release()
    if not path.is_file() or path.stat().st_size <= 0:
        raise TemporalContextError("FAILED_STAGE5D_B1_ATOMIC_OUTPUT empty_clip")
    return {
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "fps": fps,
        "width": w,
        "height": h,
    }


def build_contract(*, zoom_padding_ratio: float) -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_b1_temporal_context_contract_v1",
        "target_id": "target_001",
        "no_new_detection": True,
        "no_new_tracking": True,
        "no_new_embedding": True,
        "no_ocr": True,
        "no_similarity": True,
        "no_identity_prediction": True,
        "segment_bounded_context": True,
        "bbox_source": "segment_observations.source_observation.bbox_xyxy",
        "bbox_interpolation_forbidden": True,
        "deterministic_frame_selection": True,
        "zoom_padding_ratio": float(zoom_padding_ratio),
        "zoom_is_visualization_only": True,
        "zoom_not_gallery_or_embedding_input": True,
        "allowed_manual_decisions": list(ALLOWED_TEMPORAL_DECISIONS),
        "allowed_tristate": list(ALLOWED_TRISTATE),
        "no_gallery_membership": True,
        "no_prototypes": True,
        "no_identity_assignment": True,
        "human_approval_required": True,
        "manual_decisions": 0,
        "approved_anchors": 0,
        "rejected_anchors": 0,
        "gallery_members": 0,
        "prototypes": 0,
        "identity_assignments": 0,
        "similarity_ranking_rows": 0,
        "exact_next_gate": "STAGE5D-B2_TARGET_001_ANCHOR_MANUAL_REVIEW_AND_FREEZE",
    }


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    head = assert_git_contract(project_root, config["project_head_expected"])
    final_rel = config["output"]["final_dir"]
    assert_no_path_traversal(final_rel)
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise TemporalContextError("FAILED_STAGE5D_B1_ATOMIC_OUTPUT final_exists")

    stage5b = validate_stage5d_b_package(project_root, config)
    closure = validate_stage5c_closure(project_root, config)
    assets = validate_assets(project_root, config)
    lineage = load_segment_lineage(project_root, config, stage5b["candidates"])

    tmp = project_root / (
        "outputs/reid/"
        "_tmp_full_stage4b_rebuild_r2_stage5d_target_001_temporal_context_review_package_"
        f"{uuid.uuid4().hex}"
    )
    if tmp.exists():
        raise TemporalContextError("FAILED_STAGE5D_B1_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=True)

    video_path = project_root / config["source_video"]["path"]
    fps = float(assets["fps"])
    max_before = int(config["temporal_window"]["max_frames_before"])
    max_after = int(config["temporal_window"]["max_frames_after"])
    offsets = list(config["temporal_window"]["target_offsets_from_representative"])
    zoom_pad = float(config["visualization"]["zoom_padding_ratio"])
    bbox_color = list(config["visualization"]["bbox_color_bgr"])
    bbox_thickness = int(config["visualization"]["bbox_thickness"])

    try:
        pkg_dir = tmp / "review_packages" / "target_001_temporal_context_review"
        clips_dir = pkg_dir / "clips"
        pkg_dir.mkdir(parents=True)
        clips_dir.mkdir(parents=True)

        inventory_rows: list[dict[str, Any]] = []
        sheet_groups: dict[int, list[tuple[str, np.ndarray]]] = {
            1: [],
            2: [],
            3: [],
        }
        clip_records: list[dict[str, Any]] = []
        candidate_sheet_map: dict[str, str] = {}

        cap_sheet = cv2.VideoCapture(str(video_path))
        if not cap_sheet.isOpened():
            raise TemporalContextError("BLOCKED_STAGE5D_B1_SOURCE_VIDEO_INTEGRITY open")

        for cand in stage5b["candidates"]:
            cid = str(cand["anchor_candidate_id"])
            lin = lineage[cid]
            seg_start = lin["segment_start_frame"]
            seg_end = lin["segment_end_frame"]
            rep = lin["representative_frame"]
            obs = lin["observations"]
            obs_frames = [o["frame_index"] for o in obs]
            bbox_by_frame = {o["frame_index"]: o["bbox_xyxy"] for o in obs}

            window = compute_context_window(
                segment_start=seg_start,
                segment_end=seg_end,
                representative=rep,
                max_before=max_before,
                max_after=max_after,
            )
            ctx_start = window["selected_context_start_frame"]
            ctx_end = window["selected_context_end_frame"]
            # Strict segment bound already applied; eligible temporal window =
            # candidate segment observation range (Stage 5D-B eligible).
            if ctx_start < seg_start or ctx_end > seg_end:
                raise TemporalContextError(
                    "BLOCKED_STAGE5D_B1_TEMPORAL_EXCLUSION_CONTRACT window"
                )

            selected_frames = select_sheet_observation_frames(
                obs_frames,
                representative=rep,
                ctx_start=ctx_start,
                ctx_end=ctx_end,
                offsets=offsets,
            )
            if rep not in selected_frames:
                # Representative must be represented by nearest included obs;
                # nearest-to-rep target guarantees an observation near rep.
                nearest_rep = min(selected_frames or obs_frames, key=lambda f: abs(f - rep))
                if nearest_rep not in selected_frames:
                    selected_frames = sorted(set(selected_frames) | {nearest_rep})

            panels: list[dict[str, Any]] = []
            selected_obs_payload: list[dict[str, Any]] = []
            for frame_index in selected_frames:
                frame = read_frame(cap_sheet, frame_index)
                bbox = bbox_by_frame[frame_index]
                is_rep = frame_index == rep
                annotated = annotate_full_frame(
                    frame,
                    bbox_xyxy=bbox,
                    anchor_candidate_id=cid,
                    frame_index=frame_index,
                    fps=fps,
                    is_representative=is_rep,
                    bbox_color=bbox_color,
                    bbox_thickness=bbox_thickness,
                )
                zoom = zoom_from_bbox(frame, bbox, padding_ratio=zoom_pad)
                panels.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": frame_index / fps,
                        "is_representative": is_rep,
                        "full_annotated": annotated,
                        "zoom": zoom,
                    }
                )
                selected_obs_payload.append(
                    {
                        "frame_index": frame_index,
                        "bbox_xyxy": bbox,
                        "is_representative": is_rep,
                        "timestamp_sec": frame_index / fps,
                    }
                )

            row_img = render_candidate_row(panels)
            order = int(cid.rsplit("_", 1)[-1])
            sheet_idx = (order - 1) // 3 + 1
            sheet_groups[sheet_idx].append((cid, row_img))

            clip_name = f"{cid}_context.mp4"
            clip_path = clips_dir / clip_name
            clip_meta = write_context_clip(
                clip_path,
                video_path=video_path,
                ctx_start=ctx_start,
                ctx_end=ctx_end,
                bbox_by_frame=bbox_by_frame,
                anchor_candidate_id=cid,
                representative=rep,
                fps=fps,
                bbox_color=bbox_color,
                bbox_thickness=bbox_thickness,
            )
            clip_rel = (
                "review_packages/target_001_temporal_context_review/clips/" + clip_name
            )
            sheet_rel = (
                f"review_packages/target_001_temporal_context_review/"
                f"temporal_contact_sheet_{sheet_idx:02d}.png"
            )
            candidate_sheet_map[cid] = sheet_rel
            clip_records.append({"anchor_candidate_id": cid, "path": clip_rel, **clip_meta})

            inventory_rows.append(
                {
                    "anchor_candidate_id": cid,
                    "target_id": "target_001",
                    "segment_id": cand["segment_id"],
                    "raw_track_id": int(cand["raw_track_id"]),
                    "representative_frame": rep,
                    "segment_frame_range": [seg_start, seg_end],
                    "selected_context_frame_range": [ctx_start, ctx_end],
                    "selected_observation_frames": selected_frames,
                    "selected_observation_count": len(selected_frames),
                    "existing_bbox_per_selected_frame": selected_obs_payload,
                    "source_time_start_sec": window["source_time_start_sec"],
                    "source_time_end_sec": window["source_time_end_sec"],
                    "truncated_left": window["truncated_left"],
                    "truncated_right": window["truncated_right"],
                    "full_frame_source_lineage": {
                        "source_video": config["source_video"]["path"],
                        "source_video_sha256": assets["sha256"],
                        "decode_policy": "lineage_bounded_frame_seek_only",
                    },
                    "source_video_sha256": assets["sha256"],
                    "representative_crop_path": cand["source_crop_path"],
                    "representative_crop_sha256": cand["source_crop_sha256"],
                    "stage5d_b_candidate_inventory_sha256": stage5b[
                        "inventory_jsonl_sha256"
                    ],
                    "temporal_exclusion_audit": {
                        "context_within_segment": True,
                        "no_stage5c_window_expansion": True,
                        "bbox_interpolation": False,
                        "new_detection": False,
                    },
                    "sheet_path": sheet_rel,
                    "clip_path": clip_rel,
                    "clip_sha256": clip_meta["sha256"],
                    "clip_bytes": clip_meta["bytes"],
                    "clip_frame_count": clip_meta["frame_count"],
                    "clip_duration_sec": clip_meta["duration_sec"],
                    "decision_fields_blank": True,
                    "manual_decisions": 0,
                    "gallery_member": False,
                    "prototype": False,
                    "identity_assignment": None,
                }
            )

        cap_sheet.release()

        # Write three contact sheets
        sheet_paths: list[str] = []
        for sheet_idx in (1, 2, 3):
            rows = sheet_groups[sheet_idx]
            if len(rows) != 3:
                raise TemporalContextError(
                    f"FAILED_STAGE5D_B1_ATOMIC_OUTPUT sheet_rows={len(rows)}"
                )
            sheet = render_temporal_sheet(rows)
            name = f"temporal_contact_sheet_{sheet_idx:02d}.png"
            out = pkg_dir / name
            write_png(out, sheet)
            sheet_paths.append(
                f"review_packages/target_001_temporal_context_review/{name}"
            )

        if len(clip_records) != 9:
            raise TemporalContextError("FAILED_STAGE5D_B1_ATOMIC_OUTPUT clip_count")

        # Inventory
        inv_dir = tmp / "inventory"
        inv_dir.mkdir(parents=True)
        with (inv_dir / "target_001_temporal_context_inventory.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in inventory_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Contract + manifest
        tr_dir = tmp / "temporal_review"
        tr_dir.mkdir(parents=True)
        contract = build_contract(zoom_padding_ratio=zoom_pad)
        write_json(tr_dir / "target_001_temporal_context_contract.json", contract)
        write_json(
            tr_dir / "target_001_temporal_context_manifest.json",
            {
                "schema_version": "reid_stage5d_b1_temporal_context_manifest_v1",
                "target_id": "target_001",
                "candidates": [r["anchor_candidate_id"] for r in inventory_rows],
                "contact_sheets": sheet_paths,
                "clips": clip_records,
                "inventory_rows": len(inventory_rows),
                "manual_decisions": 0,
                "approved_anchors": 0,
                "gallery_members": 0,
                "prototypes": 0,
                "identity_assignments": 0,
                "similarity_ranking_rows": 0,
            },
        )

        # Blank template
        tpl_dir = tmp / "templates"
        tpl_dir.mkdir(parents=True)
        tpl_path = tpl_dir / "target_001_temporal_context_review_template.csv"
        with tpl_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TEMPLATE_FIELDS))
            writer.writeheader()
            for row in inventory_rows:
                writer.writerow(
                    {
                        "anchor_candidate_id": row["anchor_candidate_id"],
                        "segment_id": row["segment_id"],
                        "raw_track_id": row["raw_track_id"],
                        "representative_frame": row["representative_frame"],
                        "context_start_frame": row["selected_context_frame_range"][0],
                        "context_end_frame": row["selected_context_frame_range"][1],
                        "temporal_review_decision": "",
                        "manual_crop_valid": "",
                        "manual_target_dominant": "",
                        "manual_identity_continuity_observed": "",
                        "manual_human_verified_number_seen_in_context": "",
                        "manual_notes": "",
                        "reviewer": "",
                        "final_approver": "",
                        "reviewed_at": "",
                    }
                )
        with tpl_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["temporal_review_decision"]:
                    raise TemporalContextError("prefilled decision forbidden")

        # runtime / effective configs
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        write_json(
            tmp / "runtime" / "runtime.json",
            {
                "schema_version": "reid_stage5d_b1_runtime_v1",
                "started_at": started,
                "device": "cpu",
                "offline_required": True,
                "network_download": 0,
                "new_detection": 0,
                "new_tracking": 0,
                "new_embedding": 0,
                "ocr": 0,
                "similarity_inference": 0,
                "reserve_reads": 0,
            },
        )
        eff = tmp / "effective_configs"
        eff.mkdir(parents=True)
        shutil.copy2(config_path, eff / config_path.name)

        png_count = len(list(tmp.rglob("*.png")))
        mp4_count = len(list(tmp.rglob("*.mp4")))
        jpeg_count = len(list(tmp.rglob("*.jpg"))) + len(list(tmp.rglob("*.jpeg")))
        if png_count != 3 or mp4_count != 9 or jpeg_count != 0:
            raise TemporalContextError(
                f"FAILED_STAGE5D_B1_ATOMIC_OUTPUT budget png={png_count} "
                f"mp4={mp4_count} jpeg={jpeg_count}"
            )

        context_ranges = [
            {
                "anchor_candidate_id": r["anchor_candidate_id"],
                "segment_id": r["segment_id"],
                "segment_frame_range": r["segment_frame_range"],
                "representative_frame": r["representative_frame"],
                "selected_context_frame_range": r["selected_context_frame_range"],
                "selected_observation_frames": r["selected_observation_frames"],
                "truncated_left": r["truncated_left"],
                "truncated_right": r["truncated_right"],
            }
            for r in inventory_rows
        ]

        summary = {
            "schema_version": "reid_stage5d_b1_summary_v1",
            "final_status": (
                "COMPLETED_STAGE5D_B1_TARGET_001_TEMPORAL_CONTEXT_REVIEW_READY"
            ),
            "project_head": head,
            "target_id": "target_001",
            "target_definition_frozen": True,
            "target_alias": stage5b["target_definition"]["target_alias"],
            "human_verified_jersey_number": 5,
            "jersey_number_provenance": stage5b["target_definition"][
                "jersey_number_provenance"
            ],
            "candidate_count": 9,
            "candidate_ids": EXPECTED_CANDIDATE_IDS,
            "temporal_contact_sheet_count": 3,
            "temporal_context_clip_count": 9,
            "context_ranges": context_ranges,
            "annotation_template_blank": True,
            "stage5d_b_annotation_template_blank": True,
            "manual_decisions": 0,
            "approved_anchors": 0,
            "rejected_anchors": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
            "similarity_ranking_rows": 0,
            "new_detection": 0,
            "new_tracking": 0,
            "new_embedding": 0,
            "ocr": 0,
            "png_count": 3,
            "mp4_count": 9,
            "jpeg_count": 0,
            "source_crop_copies": 0,
            "stage5c_closure": closure,
            "stage5d_b_snapshot_sha256": stage5b["snapshot_sha256"],
            "source_video": assets,
            "exact_next_gate": "STAGE5D-B2_TARGET_001_ANCHOR_MANUAL_REVIEW_AND_FREEZE",
        }
        write_json(tmp / "stage5d_b1_summary.json", summary)

        files_n, files_sha = listing_sha(tmp)
        write_json(
            tmp / "stage5d_b1_manifest.json",
            {
                "schema_version": "reid_stage5d_b1_manifest_v1",
                "final_status": summary["final_status"],
                "project_head": head,
                "listing_file_count": files_n,
                "listing_sha256": files_sha,
                "contact_sheets": sheet_paths,
                "clips": [c["path"] for c in clip_records],
                "clip_sha256": {c["anchor_candidate_id"]: c["sha256"] for c in clip_records},
                "artifacts": {
                    "inventory": "inventory/target_001_temporal_context_inventory.jsonl",
                    "contract": "temporal_review/target_001_temporal_context_contract.json",
                    "manifest": "temporal_review/target_001_temporal_context_manifest.json",
                    "template": "templates/target_001_temporal_context_review_template.csv",
                    "summary": "stage5d_b1_summary.json",
                },
                "manual_decisions": 0,
                "gallery_members": 0,
                "prototypes": 0,
                "identity_assignments": 0,
            },
        )

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_b1_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/target_temporal_context_review_stage5d_target_001.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        summary = run(args.config.resolve(), args.project_root.resolve())
    except TemporalContextError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "sheets": summary["temporal_contact_sheet_count"],
                "clips": summary["temporal_context_clip_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
