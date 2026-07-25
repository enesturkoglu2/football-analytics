#!/usr/bin/env python3
"""Stage 5D-B1A — target_001 manual seed selection package.

Neutral visual review around human seed frame 290. No selection, anchor freeze,
gallery membership, OCR, similarity, or new detection/tracking/embedding.
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
CONFIG_SCHEMA = "reid_target_manual_seed_selection_config_v1"
TEMPLATE_FIELDS = (
    "target_id",
    "representative_frame",
    "selected_neutral_seed_code",
    "manual_target_confirmed",
    "manual_human_verified_number_seen",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
ALLOWED_TRISTATE = ("yes", "no", "uncertain")


class ManualSeedError(RuntimeError):
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
        raise ManualSeedError("unexpected config schema")
    if not config.get("offline_required"):
        raise ManualSeedError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/"):
        raise ManualSeedError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise ManualSeedError("BLOCKED_STAGE5D_B1A_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise ManualSeedError("BLOCKED_STAGE5D_B1A_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    allowed = {
        "scripts/run_reid_target_manual_seed_selection_package.py",
        "configs/reid/target_manual_seed_selection_stage5d_target_001.yaml",
        "tests/test_reid_target_manual_seed_selection_package.py",
        "docs/setup/stage5d-target-manual-seed-selection-package.md",
    }
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path not in allowed:
                raise ManualSeedError(
                    "BLOCKED_STAGE5D_B1A_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != "Add target 001 temporal context review package":
        raise ManualSeedError("BLOCKED_STAGE5D_B1A_GIT_CONTRACT_MISMATCH message")
    return head


def validate_target_and_zeros(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    td_path = project_root / config["target_definition"]["path"]
    td = load_json(td_path)
    exp = config["target_definition"]
    if td.get("target_id") != exp["expected_target_id"]:
        raise ManualSeedError("target_id mismatch")
    if td.get("target_alias") != exp["expected_alias"]:
        raise ManualSeedError("target_alias mismatch")
    if td.get("target_definition_frozen") is not True:
        raise ManualSeedError("target not frozen")
    if td.get("identity_basis") != exp["expected_identity_basis"]:
        raise ManualSeedError("identity_basis mismatch")
    if int(td.get("human_verified_jersey_number")) != int(exp["expected_jersey_number"]):
        raise ManualSeedError("jersey mismatch")
    if td.get("jersey_number_provenance") != exp["expected_jersey_provenance"]:
        raise ManualSeedError("jersey provenance mismatch")
    if td.get("automated_jersey_used") is not False:
        raise ManualSeedError("automated jersey used")

    roots = config["stage5d_roots"]
    listings: dict[str, dict[str, Any]] = {}
    for key, rel in roots.items():
        path = project_root / rel
        if not path.is_dir():
            raise ManualSeedError(f"missing stage root {key}")
        n, sha = listing_sha(path)
        listings[key] = {"path": rel, "listing_file_count": n, "listing_sha256": sha}

    b_sum = load_json(
        project_root / roots["stage5d_b"] / "stage5d_b_summary.json"
    )
    b1_sum = load_json(
        project_root / roots["stage5d_b1"] / "stage5d_b1_summary.json"
    )
    a_sum = load_json(
        project_root / roots["stage5d_a"] / "stage5d_preflight_summary.json"
    )
    for label, summary in (("A", a_sum), ("B", b_sum), ("B1", b1_sum)):
        if int(summary.get("gallery_members") or 0) != 0:
            raise ManualSeedError(f"gallery_members nonzero {label}")
        if int(summary.get("prototypes") or 0) != 0:
            raise ManualSeedError(f"prototypes nonzero {label}")
        if int(summary.get("identity_assignments") or 0) != 0:
            raise ManualSeedError(f"identity nonzero {label}")
    if int(b_sum.get("approved_anchors") or 0) != 0:
        raise ManualSeedError("approved_anchors nonzero")
    if int(b1_sum.get("approved_anchors") or 0) != 0:
        raise ManualSeedError("b1 approved_anchors nonzero")

    return {"target_definition": td, "stage5d_listings": listings}


def validate_stage5c(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root / config["stage5c_closure"]["path"]
    policy = load_json(root / config["stage5c_closure"]["policy_json"])
    required = {
        "automated_parseq_identity_assignment_allowed": False,
        "automated_parseq_gallery_enrollment_allowed": False,
        "stage5e_automated_jersey_channel_mode": "diagnostic_only",
        "stage5c_status": "closed",
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise ManualSeedError(f"stage5c mismatch {key}={policy.get(key)!r}")
    return required


def validate_assets(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    video = project_root / config["source_video"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if not video.is_file() or video.is_symlink():
        raise ManualSeedError("source video missing")
    if video.stat().st_size != int(config["source_video"]["expected_bytes"]):
        raise ManualSeedError("source video bytes")
    if sha256_file(video) != config["source_video"]["expected_sha256"]:
        raise ManualSeedError("source video sha")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise ManualSeedError("yolo bytes")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise ManualSeedError("yolo sha")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise ManualSeedError("osnet sha")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ManualSeedError("source video open")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if frames != int(config["source_video"]["expected_frames"]):
        raise ManualSeedError("frames")
    if width != int(config["source_video"]["expected_width"]):
        raise ManualSeedError("width")
    if height != int(config["source_video"]["expected_height"]):
        raise ManualSeedError("height")
    if abs(fps - float(config["source_video"]["expected_fps"])) > 1e-6:
        raise ManualSeedError("fps")
    rep = int(config["seed_window"]["representative_frame"])
    if not (0 <= rep < frames):
        raise ManualSeedError("representative frame out of range")
    return {
        "path": config["source_video"]["path"],
        "sha256": config["source_video"]["expected_sha256"],
        "bytes": int(config["source_video"]["expected_bytes"]),
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
    }


def load_stage5c_exclusion_keys(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, set[str]]:
    split = project_root / config["upstream"]["canonical_split"]
    keys = {"segment_id": set(), "raw_track_id": set(), "crop_id": set()}
    for batch in (
        "discovery_primary",
        "discovery_reserve",
        "holdout_primary",
        "holdout_reserve",
    ):
        man = split / batch / f"{batch}_manifest.jsonl"
        if not man.is_file():
            raise ManualSeedError(f"missing split batch {batch}")
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            keys["segment_id"].add(str(row.get("segment_id") or ""))
            keys["raw_track_id"].add(str(row.get("raw_track_id") or ""))
            keys["crop_id"].add(str(row.get("crop_id") or ""))
    return {k: {x for x in v if x} for k, v in keys.items()}


def load_crop_catalog(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Path]:
    catalog: dict[str, Path] = {}
    baseline = project_root / config["upstream"]["baseline_crops_root"]
    man = baseline / "crop_manifest.jsonl"
    if man.is_file():
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            catalog[str(row["crop_id"])] = baseline / str(row["crop_relative_path"])
    seg_root = (
        project_root
        / config["upstream"]["stage5_replay"]
        / config["upstream"]["segmented_reid_subdir"]
    )
    seg_man = seg_root / "segment_crop_manifest.jsonl"
    if seg_man.is_file():
        for line in seg_man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            catalog[str(row["crop_id"])] = seg_root / str(row["crop_relative_path"])
    return catalog


def assign_neutral_seed_codes(
    chains: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]]
) -> dict[tuple[int, str], str]:
    """Deterministic SEED_CANDIDATE_XX codes; no identity semantics."""
    ordered = sorted(
        chains.keys(),
        key=lambda k: (
            min(int(r["frame_index"]) for r in chains[k]),
            max(int(r["frame_index"]) for r in chains[k]),
            k[0],
            k[1],
        ),
    )
    return {key: f"SEED_CANDIDATE_{i:02d}" for i, key in enumerate(ordered, start=1)}


def select_sheet_frames(
    target_frames: Sequence[int],
    observation_frames: Sequence[int],
) -> list[int]:
    """Nearest observation frame per target; duplicates collapsed."""
    if not observation_frames:
        return []
    obs = sorted(set(int(f) for f in observation_frames))
    selected: list[int] = []
    seen: set[int] = set()
    for target in target_frames:
        nearest = min(obs, key=lambda f: (abs(f - int(target)), f))
        if nearest not in seen:
            seen.add(nearest)
            selected.append(nearest)
    return selected


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise ManualSeedError(f"failed to read frame {frame_index}")
    return frame


def _fit_display(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / max(w, 1), box_h / max(h, 1))
    nw = max(1, int(math.floor(w * scale)))
    nh = max(1, int(math.floor(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def annotate_seed_frame(
    frame: np.ndarray,
    *,
    items: Sequence[Mapping[str, Any]],
    frame_index: int,
    fps: float,
    representative_frame: int,
    nearest_to_representative: bool,
    bbox_color: Sequence[int],
    bbox_thickness: int,
) -> np.ndarray:
    out = frame.copy()
    for item in items:
        x0, y0, x1, y1 = [int(round(v)) for v in item["bbox_xyxy"]]
        cv2.rectangle(
            out,
            (x0, y0),
            (x1, y1),
            tuple(int(c) for c in bbox_color),
            int(bbox_thickness),
        )
        label = str(item["neutral_seed_code"])
        tx, ty = x0, max(18, y0 - 6)
        cv2.putText(
            out,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (10, 10, 10),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    header = [
        f"frame={frame_index}",
        f"t={frame_index / fps:.3f}s",
    ]
    if nearest_to_representative or frame_index == representative_frame:
        header.append("HUMAN SEED REFERENCE")
    y = 28
    for text in header:
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 1, cv2.LINE_AA
        )
        y += 30
    return out


def render_seed_sheet(
    panels: Sequence[Mapping[str, Any]],
    *,
    panel_w: int = 420,
    panel_h: int = 280,
) -> np.ndarray:
    cols = min(4, max(1, len(panels)))
    rows_n = int(math.ceil(len(panels) / cols)) if panels else 1
    sheet = np.full((rows_n * panel_h, cols * panel_w, 3), 16, dtype=np.uint8)
    for index, panel in enumerate(panels):
        r, c = divmod(index, cols)
        tile = np.full((panel_h, panel_w, 3), 32, dtype=np.uint8)
        label = f"f={panel['frame_index']}  t={panel['time_sec']:.2f}s"
        if panel.get("human_seed_reference"):
            label += "  | HUMAN SEED REFERENCE"
        cv2.putText(
            tile,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        disp = _fit_display(panel["image"], panel_w - 12, panel_h - 40)
        dh, dw = disp.shape[:2]
        ox = (panel_w - dw) // 2
        oy = 32
        tile[oy : oy + dh, ox : ox + dw] = disp
        y0 = r * panel_h
        x0 = c * panel_w
        sheet[y0 : y0 + panel_h, x0 : x0 + panel_w] = tile
    return sheet


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ManualSeedError(f"failed png write {path}")


def write_seed_clip(
    path: Path,
    *,
    video_path: Path,
    start: int,
    end: int,
    frame_items: Mapping[int, Sequence[Mapping[str, Any]]],
    fps: float,
    representative_frame: int,
    bbox_color: Sequence[int],
    bbox_thickness: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ManualSeedError("clip open failed")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        cap.release()
        raise ManualSeedError("clip writer failed")
    count = 0
    for frame_index in range(start, end + 1):
        frame = read_frame(cap, frame_index)
        annotated = annotate_seed_frame(
            frame,
            items=frame_items.get(frame_index, []),
            frame_index=frame_index,
            fps=fps,
            representative_frame=representative_frame,
            nearest_to_representative=(frame_index == representative_frame),
            bbox_color=bbox_color,
            bbox_thickness=bbox_thickness,
        )
        writer.write(annotated)
        count += 1
    writer.release()
    cap.release()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ManualSeedError("empty clip")
    return {
        "frame_count": count,
        "duration_sec": count / fps,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "fps": fps,
        "width": w,
        "height": h,
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_b1a_manual_seed_selection_contract_v1",
        "target_id": "target_001",
        "human_click_box_selection_required": True,
        "existing_bbox_only": True,
        "no_new_detection": True,
        "no_new_tracking": True,
        "no_embedding_recompute": True,
        "no_ocr": True,
        "no_similarity": True,
        "no_identity_prediction": True,
        "no_automatic_selection": True,
        "no_gallery_membership": True,
        "no_anchor_freeze": True,
        "selected_seed_requires_separate_human_approval_freeze_gate": True,
        "unknown_identity_preserved": True,
        "manual_selection": 0,
        "approved_anchors": 0,
        "gallery_members": 0,
        "prototypes": 0,
        "identity_assignments": 0,
        "exact_next_gate": (
            "STAGE5D-B1B_TARGET_001_MANUAL_SEED_SELECTION_FREEZE_AND_ANCHOR_DERIVATION"
        ),
    }


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    head = assert_git_contract(project_root, config["project_head_expected"])
    final_rel = config["output"]["final_dir"]
    assert_no_path_traversal(final_rel)
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise ManualSeedError("final root exists")

    target_info = validate_target_and_zeros(project_root, config)
    closure = validate_stage5c(project_root, config)
    assets = validate_assets(project_root, config)
    excl = load_stage5c_exclusion_keys(project_root, config)
    crop_catalog = load_crop_catalog(project_root, config)

    start = int(config["seed_window"]["start_frame"])
    end = int(config["seed_window"]["end_frame"])
    rep = int(config["seed_window"]["representative_frame"])
    sheet_targets = [int(x) for x in config["seed_window"]["sheet_target_frames"]]
    fps = float(assets["fps"])
    bbox_color = list(config["visualization"]["bbox_color_bgr"])
    bbox_thickness = int(config["visualization"]["bbox_thickness"])

    seg_view = (
        project_root
        / config["upstream"]["stage5_replay"]
        / config["upstream"]["segment_view_subdir"]
    )
    obs_path = seg_view / "segment_observations.jsonl"
    emb_index_path = (
        project_root
        / config["upstream"]["stage5_replay"]
        / config["upstream"]["segmented_reid_subdir"]
        / "segment_embedding_index.jsonl"
    )
    if not obs_path.is_file() or not emb_index_path.is_file():
        raise ManualSeedError("missing segment lineage")

    emb_by_seg: dict[str, dict[str, Any]] = {}
    for line in emb_index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        emb_by_seg[str(row["segment_id"])] = row

    chains: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    window_obs_count = 0
    frames_with_obs: set[int] = set()
    for line in obs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fi = int(row["frame_index"])
        if not (start <= fi <= end):
            continue
        src = row.get("source_observation") or {}
        bbox = src.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            raise ManualSeedError(f"missing bbox frame={fi}")
        key = (int(row["raw_track_id"]), str(row["segment_id"]))
        chains[key].append(
            {
                "frame_index": fi,
                "bbox_xyxy": [float(x) for x in bbox],
                "timestamp_sec": float(src.get("timestamp_sec") or fi / fps),
                "source_observation_sha256": row.get("source_observation_sha256"),
            }
        )
        window_obs_count += 1
        frames_with_obs.add(fi)

    if not chains:
        raise ManualSeedError("no observations in seed window")

    # Representative frame presence / nearest report.
    if rep in frames_with_obs:
        rep_obs_frame = rep
        rep_frame_delta = 0
    else:
        rep_obs_frame = min(frames_with_obs, key=lambda f: (abs(f - rep), f))
        rep_frame_delta = int(rep_obs_frame - rep)

    code_map = assign_neutral_seed_codes(chains)

    # Per-frame display items with stable codes.
    frame_items: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for key, rows in chains.items():
        code = code_map[key]
        for row in rows:
            frame_items[int(row["frame_index"])].append(
                {
                    "neutral_seed_code": code,
                    "bbox_xyxy": row["bbox_xyxy"],
                }
            )
    for fi in frame_items:
        frame_items[fi].sort(key=lambda x: x["neutral_seed_code"])

    # Mapping rows
    mapping_rows: list[dict[str, Any]] = []
    for key in sorted(code_map.keys(), key=lambda k: code_map[k]):
        tid, sid = key
        rows = sorted(chains[key], key=lambda r: r["frame_index"])
        emb = emb_by_seg.get(sid, {})
        crop_ids = list(emb.get("crop_ids") or [])
        crop_lineage = []
        for cid in crop_ids[:5]:
            path = crop_catalog.get(str(cid))
            entry = {"crop_id": cid, "path": str(path) if path else None, "exists": False, "sha256": None}
            if path and path.is_file():
                entry["exists"] = True
                entry["sha256"] = sha256_file(path)
            crop_lineage.append(entry)
        excluded = sid in excl["segment_id"] or str(tid) in excl["raw_track_id"]
        mapping_rows.append(
            {
                "neutral_seed_code": code_map[key],
                "raw_track_id": tid,
                "segment_id": sid,
                "observation_frames": [r["frame_index"] for r in rows],
                "bbox_per_frame": [
                    {"frame_index": r["frame_index"], "bbox_xyxy": r["bbox_xyxy"]}
                    for r in rows
                ],
                "first_frame": rows[0]["frame_index"],
                "last_frame": rows[-1]["frame_index"],
                "existing_embedding_available": bool(emb.get("embedding_available")),
                "embedding_dimension": emb.get("embedding_dimension"),
                "embedding_row": emb.get("embedding_row"),
                "representation_source": emb.get("representation_source"),
                "source_crop_lineage": crop_lineage,
                "stage5c_or_stage5d_exclusion_status": {
                    "excluded_by_stage5c_segment_or_track": excluded,
                    "in_stage5c_segment_id_set": sid in excl["segment_id"],
                    "in_stage5c_raw_track_id_set": str(tid) in excl["raw_track_id"],
                },
                "target_positive_decision": None,
                "selected_as_seed": False,
            }
        )

    sheet_frames = select_sheet_frames(sheet_targets, sorted(frames_with_obs))
    if not sheet_frames:
        raise ManualSeedError("no sheet frames")
    # Ensure representative (or nearest obs) is included.
    if rep_obs_frame not in sheet_frames:
        sheet_frames = sorted(set(sheet_frames) | {rep_obs_frame})

    tmp = project_root / (
        "outputs/reid/"
        "_tmp_full_stage4b_rebuild_r2_stage5d_target_001_manual_seed_selection_package_"
        f"{uuid.uuid4().hex}"
    )
    if tmp.exists():
        raise ManualSeedError("tmp exists")
    tmp.mkdir(parents=True)

    video_path = project_root / config["source_video"]["path"]
    try:
        pkg = tmp / "review_packages" / "target_001_manual_seed_selection"
        pkg.mkdir(parents=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ManualSeedError("video open")
        panels = []
        for fi in sheet_frames:
            frame = read_frame(cap, fi)
            annotated = annotate_seed_frame(
                frame,
                items=frame_items.get(fi, []),
                frame_index=fi,
                fps=fps,
                representative_frame=rep,
                nearest_to_representative=(fi == rep_obs_frame),
                bbox_color=bbox_color,
                bbox_thickness=bbox_thickness,
            )
            panels.append(
                {
                    "frame_index": fi,
                    "time_sec": fi / fps,
                    "human_seed_reference": fi == rep_obs_frame,
                    "image": annotated,
                }
            )
        cap.release()
        sheet = render_seed_sheet(panels)
        sheet_rel = (
            "review_packages/target_001_manual_seed_selection/seed_selection_sheet_01.png"
        )
        write_png(pkg / "seed_selection_sheet_01.png", sheet)

        clip_rel = (
            "review_packages/target_001_manual_seed_selection/"
            "target_001_manual_seed_window.mp4"
        )
        clip_meta = write_seed_clip(
            pkg / "target_001_manual_seed_window.mp4",
            video_path=video_path,
            start=start,
            end=end,
            frame_items=frame_items,
            fps=fps,
            representative_frame=rep,
            bbox_color=bbox_color,
            bbox_thickness=bbox_thickness,
        )

        inv_dir = tmp / "inventory"
        inv_dir.mkdir(parents=True)
        with (inv_dir / "target_001_manual_seed_candidate_mapping.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in mapping_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        seed_dir = tmp / "seed_selection"
        seed_dir.mkdir(parents=True)
        contract = build_contract()
        write_json(seed_dir / "target_001_manual_seed_selection_contract.json", contract)
        write_json(
            seed_dir / "target_001_manual_seed_selection_manifest.json",
            {
                "schema_version": "reid_stage5d_b1a_manual_seed_selection_manifest_v1",
                "target_id": "target_001",
                "representative_frame": rep,
                "representative_observation_frame": rep_obs_frame,
                "representative_frame_delta": rep_frame_delta,
                "seed_window": [start, end],
                "sheet_frames": sheet_frames,
                "sheet_path": sheet_rel,
                "clip_path": clip_rel,
                "clip": clip_meta,
                "neutral_seed_candidate_count": len(mapping_rows),
                "window_observation_count": window_obs_count,
                "manual_selection": 0,
                "gallery_members": 0,
                "prototypes": 0,
                "identity_assignments": 0,
                "approved_anchors": 0,
            },
        )

        tpl_dir = tmp / "templates"
        tpl_dir.mkdir(parents=True)
        tpl_path = tpl_dir / "target_001_manual_seed_selection_template.csv"
        with tpl_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TEMPLATE_FIELDS))
            writer.writeheader()
            writer.writerow(
                {
                    "target_id": "target_001",
                    "representative_frame": rep,
                    "selected_neutral_seed_code": "",
                    "manual_target_confirmed": "",
                    "manual_human_verified_number_seen": "",
                    "manual_crop_valid": "",
                    "manual_target_dominant": "",
                    "manual_notes": "",
                    "reviewer": "",
                    "final_approver": "",
                    "reviewed_at": "",
                }
            )
        with tpl_path.open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
            if row["selected_neutral_seed_code"] or row["manual_target_confirmed"]:
                raise ManualSeedError("prefilled selection forbidden")

        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        write_json(
            tmp / "runtime" / "runtime.json",
            {
                "schema_version": "reid_stage5d_b1a_runtime_v1",
                "started_at": started,
                "offline_required": True,
                "network_download": 0,
                "new_detection": 0,
                "new_tracking": 0,
                "new_embedding": 0,
                "ocr": 0,
                "similarity_inference": 0,
            },
        )
        eff = tmp / "effective_configs"
        eff.mkdir(parents=True)
        shutil.copy2(config_path, eff / config_path.name)

        png_count = len(list(tmp.rglob("*.png")))
        mp4_count = len(list(tmp.rglob("*.mp4")))
        jpeg_count = len(list(tmp.rglob("*.jpg"))) + len(list(tmp.rglob("*.jpeg")))
        if png_count != 1 or mp4_count != 1 or jpeg_count != 0:
            raise ManualSeedError(
                f"artifact budget png={png_count} mp4={mp4_count} jpeg={jpeg_count}"
            )

        summary = {
            "schema_version": "reid_stage5d_b1a_summary_v1",
            "final_status": (
                "COMPLETED_STAGE5D_B1A_TARGET_001_MANUAL_SEED_SELECTION_READY"
            ),
            "project_head": head,
            "target_id": "target_001",
            "target_alias": target_info["target_definition"]["target_alias"],
            "target_definition_frozen": True,
            "human_verified_jersey_number": 5,
            "jersey_number_provenance": target_info["target_definition"][
                "jersey_number_provenance"
            ],
            "automated_jersey_used": False,
            "representative_seed_frame": rep,
            "representative_observation_frame": rep_obs_frame,
            "representative_frame_delta": rep_frame_delta,
            "seed_window_frames": [start, end],
            "seed_window_time_sec": [start / fps, end / fps],
            "sheet_frames": sheet_frames,
            "window_observation_count": window_obs_count,
            "neutral_seed_candidate_count": len(mapping_rows),
            "stable_neutral_codes": True,
            "png_count": 1,
            "mp4_count": 1,
            "jpeg_count": 0,
            "manual_selection": 0,
            "approved_anchors": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
            "new_detection": 0,
            "new_tracking": 0,
            "new_embedding": 0,
            "ocr": 0,
            "similarity_ranking_rows": 0,
            "stage5c_closure": closure,
            "stage5d_listings": target_info["stage5d_listings"],
            "source_video": assets,
            "exact_next_gate": (
                "STAGE5D-B1B_TARGET_001_MANUAL_SEED_SELECTION_FREEZE_AND_ANCHOR_DERIVATION"
            ),
        }
        write_json(tmp / "stage5d_b1a_summary.json", summary)

        n_files, listing = listing_sha(tmp)
        write_json(
            tmp / "stage5d_b1a_manifest.json",
            {
                "schema_version": "reid_stage5d_b1a_manifest_v1",
                "final_status": summary["final_status"],
                "project_head": head,
                "listing_file_count": n_files,
                "listing_sha256": listing,
                "sheet_path": sheet_rel,
                "clip_path": clip_rel,
                "clip_sha256": clip_meta["sha256"],
                "neutral_seed_candidate_count": len(mapping_rows),
                "manual_selection": 0,
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

    return load_json(final_dir / "stage5d_b1a_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/target_manual_seed_selection_stage5d_target_001.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        summary = run(args.config.resolve(), args.project_root.resolve())
    except ManualSeedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "candidates": summary["neutral_seed_candidate_count"],
                "observations": summary["window_observation_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
