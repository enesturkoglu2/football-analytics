#!/usr/bin/env python3
"""Stage 5D-F3A — Retrieval error analysis and gallery diagnostics.

Read-only analysis of F3 independent retrieval results. Builds diagnostic
contact sheets/videos and refinement hypotheses. Does not recompute
scores, mutate gallery/GT, select thresholds, or assign identities.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "reid_target_retrieval_error_analysis_config_v1"
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FINAL_STATUS = "COMPLETED_STAGE5D_F3A_TARGET_001_RETRIEVAL_ERROR_ANALYSIS_READY"
NEXT_GATE = (
    "STAGE5D-F3B_TARGET_001_RETRIEVAL_ERROR_MANUAL_REVIEW_AND_REFINEMENT_DESIGN"
)
POSITIVE_IDS = (
    "SAMPLE_EVAL_003",
    "SAMPLE_EVAL_024",
    "SAMPLE_EVAL_028",
    "SAMPLE_EVAL_042",
    "SAMPLE_EVAL_046",
    "SAMPLE_EVAL_069",
    "SAMPLE_EVAL_100",
    "SAMPLE_EVAL_102",
)
CONFLICT_COMPONENTS = (
    "SAMPLE_COMPONENT_005",
    "SAMPLE_COMPONENT_018",
    "SAMPLE_COMPONENT_032",
    "SAMPLE_COMPONENT_055",
)
TARGET_PRESENT_AMBIGUOUS = ("SAMPLE_EVAL_108", "SAMPLE_EVAL_148")
GALLERY_IDS = (
    "target_001_ext_anchor_001",
    "target_001_ext_anchor_003",
    "target_001_ext_anchor_004",
    "target_001_ext_anchor_006",
    "target_001_ext_anchor_008",
    "target_001_ext_anchor_011",
    "target_001_ext_anchor_014",
)
OFFICIAL_AP = 0.5157291205956066
FRAME_W, FRAME_H = 1336, 744
ALLOWED_DIRTY = {
    "scripts/run_reid_target_retrieval_error_analysis.py",
    "configs/reid/target_retrieval_error_analysis_stage5d_target_001.yaml",
    "tests/test_reid_target_retrieval_error_analysis.py",
    "docs/setup/stage5d-target-retrieval-error-analysis-and-gallery-diagnostics.md",
}
TEMPLATE_FIELDS = (
    "sample_eval_code",
    "cohort_membership",
    "frozen_ground_truth_decision",
    "official_rank",
    "official_primary_score",
    "best_anchor_id",
    "segment_frame_range",
    "representative_crop_path",
    "manual_visible_team",
    "manual_visible_jersey_number",
    "manual_view_category",
    "manual_crop_quality",
    "manual_overlap_present",
    "manual_track_or_component_issue",
    "manual_primary_confusion_reason",
    "manual_gallery_gap_hypothesis",
    "manual_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
WATERMARK = "DIAGNOSTIC ONLY — NO THRESHOLD — NO IDENTITY ASSIGNMENT"


class ErrorAnalysisError(RuntimeError):
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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ErrorAnalysisError("unexpected config schema")
    if not config.get("offline_required"):
        raise ErrorAnalysisError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise ErrorAnalysisError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise ErrorAnalysisError(
                    "BLOCKED_STAGE5D_F3A_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_GIT_CONTRACT_MISMATCH message")
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH snapshot")
    sidecar = Path(str(snapshot_path) + ".sha256")
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not manifest.is_file():
        manifest = snapshot_path.with_name(
            snapshot_path.name.replace(".tar.gz", "_snapshot_manifest.json")
        )
    if not sidecar.is_file() or not listing.is_file() or not manifest.is_file():
        raise ErrorAnalysisError(
            "BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(man_payload.get("sha256") or man_payload.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH snapshot_sha")
    return actual


def validate_f3(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["stage5d_f3_evaluation"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_f3_summary.json")
    cfg = config["stage5d_f3_evaluation"]
    if summary.get("final_status") != cfg["expected_final_status"]:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH status")
    if summary.get("descriptive_outcome") != cfg["expected_descriptive_outcome"]:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH outcome")
    if summary.get("queries_scored") != 150:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH queries")
    if summary.get("individual_cosine_shape") != [150, 7]:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH matrix")
    if summary.get("official_segment_metric_rows") != 118:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH seg_rows")
    if summary.get("official_component_metric_rows") != 99:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH comp_rows")
    if summary.get("gallery_members") != 7:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH gallery")
    if summary.get("new_embeddings") != 0:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH embeddings")
    if summary.get("threshold_selected") is not False:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH threshold")
    if summary.get("identity_assignments") != 0:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH identity")
    if summary.get("gallery_mutation") is not False:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH mutation")
    if summary.get("two_pass_deterministic") is not True:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH twopass")
    if float(summary.get("two_pass_max_abs_diff", 1)) != 0.0:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH maxabs")
    seg = summary["segment_metrics"]
    for key, want in (
        ("Recall@1", 0.125),
        ("Recall@3", 0.25),
        ("Recall@5", 0.375),
        ("Recall@10", 0.375),
        ("Average_Precision", 0.5157291205956066),
        ("AUROC", 0.9068181818181819),
        ("separation_margin", -0.036077141761779785),
    ):
        if not math.isclose(float(seg[key]), want, rel_tol=0.0, abs_tol=1e-12):
            raise ErrorAnalysisError(
                f"BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH {key}"
            )
    comp = summary["component_metrics"]
    for key, want in (
        ("Recall@1", 0.25),
        ("Recall@3", 0.5),
        ("Recall@5", 0.5),
        ("Recall@10", 0.5),
        ("Average_Precision", 0.5299145299145299),
        ("AUROC", 0.9342105263157895),
        ("separation_margin", -0.018652677536010742),
    ):
        if not math.isclose(float(comp[key]), want, rel_tol=0.0, abs_tol=1e-12):
            raise ErrorAnalysisError(
                f"BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH c_{key}"
            )
    with (root / "rankings" / "target_001_segment_primary_ranking.csv").open(
        encoding="utf-8"
    ) as handle:
        ranking = list(csv.DictReader(handle))
    if len(ranking) != 118:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH ranking")
    expected_pos = {
        "SAMPLE_EVAL_100": (1, 0.970816),
        "SAMPLE_EVAL_003": (2, 0.967038),
        "SAMPLE_EVAL_028": (4, 0.964409),
        "SAMPLE_EVAL_046": (14, 0.953590),
        "SAMPLE_EVAL_102": (19, 0.947291),
        "SAMPLE_EVAL_024": (20, 0.946734),
        "SAMPLE_EVAL_042": (24, 0.942522),
        "SAMPLE_EVAL_069": (34, 0.929867),
    }
    for row in ranking:
        if row["binary_label"] != "1":
            continue
        code = row["sample_eval_code"]
        er, es = expected_pos[code]
        if int(row["rank"]) != er or abs(float(row["max_individual_cosine"]) - es) > 1e-5:
            raise ErrorAnalysisError(
                f"BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH pos_{code}"
            )
    scores = load_jsonl(root / "scores" / "target_001_sample_retrieval_scores.jsonl")
    if len(scores) != 150:
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH scores")
    # Shape-only check via mmap; values are not used to recompute scores.
    matrix_path = root / "scores" / "target_001_sample_individual_cosine.npy"
    matrix = np.load(matrix_path, mmap_mode="r")
    if tuple(matrix.shape) != (150, 7):
        raise ErrorAnalysisError("BLOCKED_STAGE5D_F3A_F3_CONTRACT_MISMATCH npy_shape")
    snap = resolve_snapshot_sha(
        Path(cfg["snapshot_path"]), cfg["expected_snapshot_sha256"]
    )
    return {
        "root": root,
        "summary": summary,
        "ranking": ranking,
        "scores": scores,
        "scores_by_code": {r["sample_eval_code"]: r for r in scores},
        "anchor_diag": load_json(
            root / "audit" / "target_001_gallery_anchor_scoring_diagnostics.json"
        ),
        "conflict_audit": load_json(
            root / "audit" / "target_001_conflicting_component_score_audit.json"
        ),
        "excluded_diag": load_json(
            root / "audit" / "target_001_excluded_item_score_diagnostics.json"
        ),
        "snapshot_sha256": snap,
        "listing_sha256": listing_sha(root)[1],
    }


def resolve_crop_path(project_root: Path, config: Mapping[str, Any], rel: str) -> Path:
    assert_no_path_traversal(rel)
    bases = [
        project_root / config["sample_baseline_crops"]["path"],
        project_root / config["sample_segmented_reid"]["path"],
    ]
    for base in bases:
        candidate = (base / rel).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError:
            continue
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise ErrorAnalysisError(f"crop missing: {rel}")


def edge_clipping(bbox: Sequence[float], *, margin: float = 2.0) -> dict[str, Any]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return {
        "left": x0 <= margin,
        "top": y0 <= margin,
        "right": x1 >= FRAME_W - margin,
        "bottom": y1 >= FRAME_H - margin,
        "any": x0 <= margin
        or y0 <= margin
        or x1 >= FRAME_W - margin
        or y1 >= FRAME_H - margin,
    }


def crop_quality_summary(bbox: Sequence[float], crop_shape: tuple[int, ...] | None) -> str:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ratio = area / float(FRAME_W * FRAME_H)
    clip = edge_clipping(bbox)
    parts = [f"area_ratio={ratio:.4f}"]
    if crop_shape is not None and len(crop_shape) >= 2:
        parts.append(f"crop={crop_shape[1]}x{crop_shape[0]}")
    if clip["any"]:
        parts.append("edge_clip")
    if ratio < 0.01:
        parts.append("small_scale")
    return ";".join(parts)


def select_context_frames(
    obs_frames: Sequence[int],
    *,
    start_frame: int,
    end_frame: int,
    representative_frame: int,
) -> list[tuple[str, int]]:
    frames = sorted(set(int(f) for f in obs_frames))
    if not frames:
        raise ErrorAnalysisError("no observations")

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


def _fit_bgr(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def annotate_context(
    frame: np.ndarray,
    bbox: Sequence[float],
    *,
    role: str,
    frame_index: int,
    time_sec: float,
) -> np.ndarray:
    out = frame.copy()
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 255), 2)
    for i, text in enumerate((role, f"frame={frame_index}", f"t={time_sec:.3f}s")):
        y = 28 + i * 26
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA
        )
    return out


def render_diagnostic_tile(
    *,
    lines: Sequence[str],
    crop_bgr: np.ndarray,
    contexts: Sequence[tuple[str, np.ndarray]],
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    tile = np.full((tile_h, tile_w, 3), 28, dtype=np.uint8)
    y = 34
    for i, text in enumerate(lines[:8]):
        scale = 0.85 if i == 0 else 0.5
        color = (245, 245, 245) if i == 0 else (210, 210, 210)
        cv2.putText(
            tile, text[:70], (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
        )
        y += 26 if i == 0 else 20
    crop_disp = _fit_bgr(crop_bgr, tile_w - 40, int(tile_h * 0.42))
    ch, cw = crop_disp.shape[:2]
    ox = (tile_w - cw) // 2
    oy = y + 8
    tile[oy : oy + ch, ox : ox + cw] = crop_disp
    ctx_y = oy + ch + 16
    ctx_slot_w = (tile_w - 48) // 3
    ctx_slot_h = tile_h - ctx_y - 16
    for i, (role, img) in enumerate(contexts[:3]):
        disp = _fit_bgr(img, ctx_slot_w - 8, ctx_slot_h - 8)
        dh, dw = disp.shape[:2]
        cx = 16 + i * ctx_slot_w + (ctx_slot_w - dw) // 2
        cy = ctx_y + (ctx_slot_h - dh) // 2
        tile[cy : cy + dh, cx : cx + dw] = disp
        cv2.putText(
            tile,
            role,
            (16 + i * ctx_slot_w + 4, ctx_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 120),
            1,
            cv2.LINE_AA,
        )
    return tile


def render_contact_sheet(
    tiles: Sequence[np.ndarray],
    *,
    title: str,
    min_width: int,
    cols: int,
) -> np.ndarray:
    n = len(tiles)
    if n > 12:
        raise ErrorAnalysisError("sheet exceeds 12 items")
    rows_n = int(math.ceil(n / cols)) if n else 1
    tile_w = max(900, int(math.ceil(min_width / cols)))
    tile_h = tiles[0].shape[0] if tiles else 780
    header_h = 64
    width = max(min_width, cols * tile_w)
    height = header_h + rows_n * tile_h
    sheet = np.full((height, width, 3), 14, dtype=np.uint8)
    cv2.putText(
        sheet, title[:90], (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = header_h + r * tile_h
        x0 = c * tile_w
        th, tw = tile.shape[:2]
        sheet[y0 : y0 + th, x0 : x0 + tw] = tile
    return sheet


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f3a_error_analysis_{final_dir.name}_{token}"
    if tmp.exists():
        raise ErrorAnalysisError(f"temp exists: {tmp}")
    tmp.mkdir(parents=True, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise ErrorAnalysisError(f"final root already exists: {final_dir}")
    os.rename(tmp, final_dir)


def hypothesis_positive(enriched: Mapping[str, Any], *, high: bool) -> list[dict[str, Any]]:
    reasons = []
    ratio = float(enriched["bbox_frame_area_ratio"])
    if ratio < 0.01:
        reasons.append("small_scale_candidate")
    if enriched["edge_clipping"]["any"]:
        reasons.append("partial_body_candidate")
    if enriched.get("component_label") == "conflicting_component":
        reasons.append("component_or_linking_issue_candidate")
    spread = float(enriched["seven_anchor_score_spread"])
    if spread > 0.05 and not high:
        reasons.append("gallery_view_gap_candidate")
    if not reasons:
        reasons.append("no_clear_metadata_cause" if high else "view_mismatch_candidate")
    return [
        {
            "reason": r,
            "hypothesis_only": True,
            "human_review_required": True,
            "automatic_ground_truth_change": False,
        }
        for r in reasons
    ]


def hypothesis_false_positive(enriched: Mapping[str, Any]) -> list[str]:
    reasons = []
    if enriched["best_gallery_anchor_id"] in {
        "target_001_ext_anchor_008",
        "target_001_ext_anchor_001",
        "target_001_ext_anchor_014",
    }:
        reasons.append("gallery_anchor_overgeneralization_candidate")
    reasons.append("same_uniform_confusion_candidate")
    if float(enriched["bbox_frame_area_ratio"]) < 0.01:
        reasons.append("scale_or_blur_confusion_candidate")
    if enriched["edge_clipping"]["any"]:
        reasons.append("partial_body_confusion_candidate")
    if enriched.get("component_label") == "conflicting_component":
        reasons.append("tracking_component_impurity_candidate")
    if not reasons:
        reasons.append("unknown_visual_confusion")
    return reasons


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or _PROJECT_ROOT
    config = load_config(config_path)
    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )
    final_rel = config["output"]["final_dir"]
    assert_no_path_traversal(final_rel)
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise ErrorAnalysisError(f"final root already exists: {final_dir}")

    f3 = validate_f3(project_root, config)

    # Immutable SHA checks (no vector load)
    sample_mp4 = project_root / config["evaluation_source"]["path"]
    if sha256_file(sample_mp4) != config["evaluation_source"]["expected_sha256"]:
        raise ErrorAnalysisError("sample.mp4 sha mismatch")
    emb = project_root / config["sample_embeddings"]["artifact_path"]
    if sha256_file(emb) != config["sample_embeddings"]["expected_artifact_sha256"]:
        raise ErrorAnalysisError("sample embeddings sha mismatch")
    if config["sample_embeddings"].get("vectors_load_forbidden") is not True:
        raise ErrorAnalysisError("vectors_load_forbidden required")
    if config["gallery_v1"].get("npy_load_forbidden") is not True:
        raise ErrorAnalysisError("gallery npy_load_forbidden required")

    f2a_root = project_root / config["stage5d_f2a_freeze"]["path"]
    f2a_listing = listing_sha(f2a_root)[1]
    gallery_root = project_root / config["gallery_v1"]["path"]
    gallery_listing = listing_sha(gallery_root)[1]
    f3_listing = f3["listing_sha256"]

    mapping = {
        r["sample_eval_code"]: r
        for r in load_jsonl(
            project_root
            / config["stage5d_f2_package"]["path"]
            / config["stage5d_f2_package"]["mapping_rel"]
        )
    }
    gallery_members = {
        r["anchor_candidate_id"]: r
        for r in load_jsonl(gallery_root / config["gallery_v1"]["members_jsonl_rel"])
    }

    # Ranking cohorts (official, no recompute)
    ranking = f3["ranking"]
    scores_by = f3["scores_by_code"]
    pos_rows = [r for r in ranking if r["binary_label"] == "1"]
    neg_rows = [r for r in ranking if r["binary_label"] == "0"]
    if len(pos_rows) != 8 or {r["sample_eval_code"] for r in pos_rows} != set(POSITIVE_IDS):
        raise ErrorAnalysisError("positive cohort mismatch")
    high_pos = [r for r in pos_rows if int(r["rank"]) <= 10]
    low_pos = [r for r in pos_rows if int(r["rank"]) > 10]
    if len(high_pos) != 3 or len(low_pos) != 5:
        raise ErrorAnalysisError("high/low positive split mismatch")
    top_fp = neg_rows[:24]
    if len(top_fp) != 24:
        raise ErrorAnalysisError("top FP != 24")
    min_pos_score = min(float(r["max_individual_cosine"]) for r in pos_rows)
    overlap_neg = [
        r for r in neg_rows if float(r["max_individual_cosine"]) >= min_pos_score
    ]
    excl_items = sorted(
        f3["excluded_diag"]["items"],
        key=lambda x: (-float(x["max_individual_cosine"]), x["sample_eval_code"]),
    )[:12]
    if len(excl_items) != 12:
        raise ErrorAnalysisError("excluded cohort != 12")
    for code in TARGET_PRESENT_AMBIGUOUS:
        if code not in {x["sample_eval_code"] for x in f3["excluded_diag"]["items"]}:
            raise ErrorAnalysisError(f"missing excluded special {code}")
    # Ensure 108/148 appear in excluded diagnostics (may or may not be in top12)
    special_excl = [
        x
        for x in f3["excluded_diag"]["items"]
        if x["sample_eval_code"] in TARGET_PRESENT_AMBIGUOUS
    ]

    # Load observations for needed segments only
    needed_segments = set()
    for code in (
        list(POSITIVE_IDS)
        + [r["sample_eval_code"] for r in top_fp]
        + [x["sample_eval_code"] for x in excl_items]
        + [m["sample_eval_code"] for c in f3["conflict_audit"]["components"] for m in c["members"]]
    ):
        needed_segments.add(mapping[code]["segment_id"])
    obs_by_seg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    obs_path = project_root / config["segment_observations"]["path"]
    with obs_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["segment_id"] in needed_segments:
                obs_by_seg[row["segment_id"]].append(row)

    def enrich(code: str, *, official_rank: int | None, cohort: str) -> dict[str, Any]:
        score = scores_by[code]
        mp = mapping[code]
        seg = mp["segment_id"]
        obs = obs_by_seg.get(seg, [])
        frames = [int(o["frame_index"]) for o in obs]
        bbox = mp["representative_bbox_xyxy"]
        x0, y0, x1, y1 = [float(v) for v in bbox]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        inds = score["individual_cosine_scores"]
        vals = list(inds.values())
        crop_path = resolve_crop_path(project_root, config, mp["representative_crop_path"])
        crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if crop is None:
            raise ErrorAnalysisError(f"crop unreadable: {crop_path}")
        start_f, end_f = int(mp["frame_range"][0]), int(mp["frame_range"][1])
        rep_f = int(mp["representative_frame"])
        start_t = start_f / 30.0
        end_t = end_f / 30.0
        best = score["best_gallery_anchor_id"]
        return {
            "sample_eval_code": code,
            "cohort_membership": cohort,
            "segment_id": seg,
            "evaluation_component_id": score["evaluation_component_id"],
            "component_label": score["component_label"],
            "frozen_ground_truth_decision": score["manual_occurrence_decision"],
            "retrieval_metric_eligible": score["retrieval_metric_eligible"],
            "metric_exclusion_reason": score.get("metric_exclusion_reason"),
            "official_rank": official_rank,
            "official_primary_score": float(score["max_individual_cosine"]),
            "best_gallery_anchor_id": best,
            "best_gallery_anchor_score": float(score["best_gallery_anchor_score"]),
            "second_best_gallery_anchor_id": score["second_best_gallery_anchor_id"],
            "second_best_gallery_anchor_score": float(
                score["second_best_gallery_anchor_score"]
            ),
            "best_second_score_gap": float(
                score["best_gallery_anchor_score"] - score["second_best_gallery_anchor_score"]
            ),
            "seven_individual_scores": inds,
            "seven_anchor_score_spread": float(max(vals) - min(vals)),
            "top3_mean_individual_cosine": float(score["top3_mean_individual_cosine"]),
            "centroid_cosine": float(score["centroid_cosine"]),
            "medoid_cosine": float(score["medoid_cosine"]),
            "mean_individual_cosine": float(score["mean_individual_cosine"]),
            "segment_start_frame": start_f,
            "segment_end_frame": end_f,
            "representative_frame": rep_f,
            "video_time_range_sec": [start_t, end_t],
            "observation_count": int(mp.get("observation_count") or len(obs)),
            "representative_crop_path": mp["representative_crop_path"],
            "representative_crop_sha256": mp.get("representative_crop_sha256"),
            "representative_crop_dimensions": [int(crop.shape[1]), int(crop.shape[0])],
            "representative_bbox_xyxy": bbox,
            "bbox_area": area,
            "bbox_frame_area_ratio": area / float(FRAME_W * FRAME_H),
            "edge_clipping": edge_clipping(bbox),
            "overlap_diagnostic": None,
            "blur_sharpness_diagnostic": None,
            "source_embedding_provenance": {
                "reused_existing_embedding": True,
                "recompute": False,
                "embedding_vector_sha256": score["embedding_vector_sha256"],
            },
            "documented_link_component_id": mp.get("documented_link_component_id"),
            "temporal_source_component": mp.get("temporal_source_component"),
            "source_observation_component": mp.get("source_observation_component"),
            "best_anchor_view_category": gallery_members.get(best, {}).get(
                "view_category"
            ),
            "crop_quality_summary": crop_quality_summary(bbox, crop.shape),
            "_crop_bgr": crop,
            "_obs": obs,
            "_map": mp,
        }

    generated_at = datetime.now(timezone.utc).isoformat()
    tmp = create_temp_root(final_dir)
    try:
        for sub in (
            "analysis",
            "gallery_diagnostics",
            "component_diagnostics",
            "review",
            "review_packages/target_001_retrieval_error_review",
            "videos",
            "templates",
            "runtime",
            "effective_configs",
        ):
            (tmp / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, tmp / "effective_configs" / Path(config_path).name)

        # Anti-overfit contract
        write_json(
            tmp / "runtime" / "target_001_f3a_anti_overfit_contract.json",
            {
                "schema_version": "reid_target_001_f3a_anti_overfit_contract_v1",
                "sample_used_for_error_analysis_only": True,
                "sample_authorized_for_gallery_optimization": False,
                "same_sample_revalidation_after_tuning_not_independent": True,
                "score_recompute": False,
                "gallery_mutation": False,
                "threshold_selected": False,
                "identity_assignment": False,
                "official_f3_result_unchanged": True,
            },
        )

        pos_enriched = []
        for r in sorted(pos_rows, key=lambda x: int(x["rank"])):
            e = enrich(
                r["sample_eval_code"],
                official_rank=int(r["rank"]),
                cohort="all_positive"
                + ("|high_ranked_positive" if int(r["rank"]) <= 10 else "|low_ranked_positive"),
            )
            e["diagnostic_reasons"] = hypothesis_positive(
                e, high=int(r["rank"]) <= 10
            )
            pos_enriched.append(e)

        fp_enriched = []
        for r in top_fp:
            e = enrich(
                r["sample_eval_code"],
                official_rank=int(r["rank"]),
                cohort="top_false_positive",
            )
            e["automatic_root_cause_candidates"] = hypothesis_false_positive(e)
            e["score_relative_to_positive_distribution"] = {
                "min_positive": min_pos_score,
                "delta_vs_min_positive": float(e["official_primary_score"] - min_pos_score),
                "above_min_positive": e["official_primary_score"] >= min_pos_score,
            }
            fp_enriched.append(e)

        excl_enriched = []
        for item in excl_items:
            e = enrich(
                item["sample_eval_code"],
                official_rank=None,
                cohort="excluded_high_score",
            )
            e["official_metric_inclusion"] = False
            excl_enriched.append(e)

        overlap_codes = [r["sample_eval_code"] for r in overlap_neg]
        overlap_scores = [float(r["max_individual_cosine"]) for r in overlap_neg]
        overlap_best = Counter(
            scores_by[c]["best_gallery_anchor_id"] for c in overlap_codes
        )

        # Inventory JSONL
        inventory = []
        for e in pos_enriched + fp_enriched + excl_enriched:
            inventory.append(
                {
                    k: v
                    for k, v in e.items()
                    if not k.startswith("_") and k != "diagnostic_reasons"
                    and k != "automatic_root_cause_candidates"
                }
            )
        write_jsonl(tmp / "analysis" / "target_001_error_cohort_inventory.jsonl", inventory)

        # Positive failure analysis
        write_json(
            tmp / "analysis" / "target_001_positive_rank_failure_analysis.json",
            {
                "schema_version": "reid_target_001_positive_rank_failure_analysis_v1",
                "high_ranked_positives": [
                    {
                        k: v
                        for k, v in e.items()
                        if not k.startswith("_")
                    }
                    for e in pos_enriched
                    if e["official_rank"] <= 10
                ],
                "low_ranked_positives": [
                    {k: v for k, v in e.items() if not k.startswith("_")}
                    for e in pos_enriched
                    if e["official_rank"] > 10
                ],
                "comparison_fields": [
                    "representative_crop_dimensions",
                    "bbox_frame_area_ratio",
                    "edge_clipping",
                    "observation_count",
                    "video_time_range_sec",
                    "best_gallery_anchor_id",
                    "seven_anchor_score_spread",
                    "centroid_cosine",
                    "medoid_cosine",
                    "source_embedding_provenance",
                    "documented_link_component_id",
                ],
            },
        )

        write_jsonl(
            tmp / "analysis" / "target_001_top_false_positive_analysis.jsonl",
            [{k: v for k, v in e.items() if not k.startswith("_")} for e in fp_enriched],
        )
        write_json(
            tmp / "analysis" / "target_001_false_positive_overlap_summary.json",
            {
                "schema_version": "reid_target_001_false_positive_overlap_summary_v1",
                "minimum_positive_score": min_pos_score,
                "overlap_negative_count": len(overlap_neg),
                "exact_item_list": [
                    {
                        "sample_eval_code": r["sample_eval_code"],
                        "official_rank": int(r["rank"]),
                        "primary_score": float(r["max_individual_cosine"]),
                        "decision": r["manual_occurrence_decision"],
                        "best_anchor": r["best_gallery_anchor_id"],
                    }
                    for r in overlap_neg
                ],
                "score_min": float(min(overlap_scores)) if overlap_scores else None,
                "score_median": float(np.median(overlap_scores)) if overlap_scores else None,
                "score_max": float(max(overlap_scores)) if overlap_scores else None,
                "best_anchor_distribution": dict(overlap_best),
            },
        )

        # Gallery diagnostics (extend F3 LOO read-only)
        loo_by = {
            r["dropped_anchor_id"]: r
            for r in f3["anchor_diag"]["leave_one_anchor_out_diagnostic"]["runs"]
        }
        best_all = f3["anchor_diag"]["best_anchor_counts_all_queries"]
        best_pos = f3["anchor_diag"]["best_anchor_counts_positive_eligible"]
        best_neg = f3["anchor_diag"]["best_anchor_counts_negative_eligible"]
        best_excl = Counter(
            scores_by[i["sample_eval_code"]]["best_gallery_anchor_id"]
            for i in f3["excluded_diag"]["items"]
        )
        anchor_rows = []
        for aid in GALLERY_IDS:
            pos_scores = [
                float(scores_by[c]["individual_cosine_scores"][aid]) for c in POSITIVE_IDS
            ]
            neg_scores = [
                float(scores_by[r["sample_eval_code"]]["individual_cosine_scores"][aid])
                for r in neg_rows
            ]
            loo = loo_by[aid]
            delta = float(loo["Average_Precision"] - OFFICIAL_AP)
            member = gallery_members[aid]
            row = {
                "anchor_id": aid,
                "occurrence_code": member.get("source_occurrence_code"),
                "view_category": member.get("view_category"),
                "best_match_count_all": int(best_all.get(aid, 0)),
                "best_match_count_positive": int(best_pos.get(aid, 0)),
                "best_match_count_negative": int(best_neg.get(aid, 0)),
                "best_match_count_excluded": int(best_excl.get(aid, 0)),
                "positive_coverage": int(best_pos.get(aid, 0)) / 8.0,
                "negative_dominance": int(best_neg.get(aid, 0))
                / max(1, int(best_all.get(aid, 0))),
                "mean_score_on_positives": float(np.mean(pos_scores)),
                "median_score_on_positives": float(np.median(pos_scores)),
                "mean_score_on_negatives": float(np.mean(neg_scores)),
                "median_score_on_negatives": float(np.median(neg_scores)),
                "maximum_negative_score": float(max(neg_scores)),
                "minimum_positive_score": float(min(pos_scores)),
                "positive_negative_mean_gap": float(
                    np.mean(pos_scores) - np.mean(neg_scores)
                ),
                "leave_one_anchor_out_AP": float(loo["Average_Precision"]),
                "leave_one_anchor_out_AP_delta_vs_official": delta,
                "official_gallery_member": True,
                "removal_authorized": False,
            }
            # Flags
            if aid == "target_001_ext_anchor_006":
                row["positive_critical_candidate"] = (
                    row["best_match_count_positive"] == 5 and delta < -0.05
                )
            if aid == "target_001_ext_anchor_008":
                row["broad_match_candidate"] = (
                    row["best_match_count_all"] >= 70
                    and row["best_match_count_negative"] >= 60
                )
            if aid == "target_001_ext_anchor_001":
                row["confusion_suspect_candidate"] = delta > 0.01
            if aid == "target_001_ext_anchor_014":
                row["confusion_suspect_candidate"] = (
                    row["best_match_count_positive"] == 0 and delta > 0.01
                )
            if aid == "target_001_ext_anchor_004":
                row["medoid_note"] = (
                    "Internal medoid representation strength is not the same as "
                    "independent discrimination power."
                )
            anchor_rows.append(row)

        write_json(
            tmp
            / "gallery_diagnostics"
            / "target_001_anchor_discrimination_diagnostics.json",
            {
                "schema_version": "reid_target_001_anchor_discrimination_diagnostics_v1",
                "official_AP": OFFICIAL_AP,
                "leave_one_anchor_out_source": "F3_audit_read_only",
                "anchors": anchor_rows,
            },
        )

        hypotheses = [
            {
                "hypothesis": "retain_critical_anchor_candidate",
                "anchor_ids": ["target_001_ext_anchor_006"],
                "evidence": "positive best-match=5; LOO AP drops sharply",
                "diagnostic_only": True,
                "action_authorized": False,
                "same_sample_revalidation_forbidden": True,
            },
            {
                "hypothesis": "review_for_overgeneralization",
                "anchor_ids": ["target_001_ext_anchor_008"],
                "evidence": "best-match all≈77 with large negative dominance",
                "diagnostic_only": True,
                "action_authorized": False,
                "same_sample_revalidation_forbidden": True,
            },
            {
                "hypothesis": "review_for_confusion",
                "anchor_ids": ["target_001_ext_anchor_001", "target_001_ext_anchor_014"],
                "evidence": "LOO AP improves when dropped; positive best-match low/zero",
                "diagnostic_only": True,
                "action_authorized": False,
                "same_sample_revalidation_forbidden": True,
            },
            {
                "hypothesis": "acquire_more_same_view_external_anchors",
                "anchor_ids": [],
                "evidence": "low-ranked positives may need matching external views",
                "diagnostic_only": True,
                "action_authorized": False,
                "same_sample_revalidation_forbidden": True,
            },
            {
                "hypothesis": "acquire_missing_view_external_anchors",
                "anchor_ids": [],
                "evidence": "gallery view coverage may leave gaps for some positives",
                "diagnostic_only": True,
                "action_authorized": False,
                "same_sample_revalidation_forbidden": True,
            },
            {
                "hypothesis": "create_external_hard-negative_protocol",
                "anchor_ids": [],
                "evidence": "many high-scoring same-kit negatives; external-only hard negatives",
                "diagnostic_only": True,
                "action_authorized": False,
                "same_sample_revalidation_forbidden": True,
            },
            {
                "hypothesis": "consider_view-aware_prototypes",
                "anchor_ids": [],
                "evidence": "medoid ≠ independent discriminator; view-aware prototypes may help",
                "diagnostic_only": True,
                "action_authorized": False,
                "same_sample_revalidation_forbidden": True,
            },
        ]
        write_json(
            tmp / "gallery_diagnostics" / "target_001_gallery_refinement_hypotheses.json",
            {
                "schema_version": "reid_target_001_gallery_refinement_hypotheses_v1",
                "hypotheses": hypotheses,
                "note": "No hypothesis is applied in F3A.",
            },
        )

        # Conflict lineage
        conflict_rows = []
        for comp in f3["conflict_audit"]["components"]:
            cid = comp["evaluation_component_id"]
            members = comp["members"]
            pos_m = [m for m in members if m["clean_positive"]]
            neg_m = [m for m in members if m["clean_negative"]]
            pos_map = mapping[pos_m[0]["sample_eval_code"]] if pos_m else None
            neg_map = mapping[neg_m[0]["sample_eval_code"]] if neg_m else None
            shared_raw = (
                pos_map
                and neg_map
                and pos_map.get("raw_track_id") == neg_map.get("raw_track_id")
            )
            shared_doc = (
                pos_map
                and neg_map
                and pos_map.get("documented_link_component_id")
                == neg_map.get("documented_link_component_id")
            )
            pr = [int(pos_map["frame_range"][0]), int(pos_map["frame_range"][1])] if pos_map else None
            nr = [int(neg_map["frame_range"][0]), int(neg_map["frame_range"][1])] if neg_map else None
            temporal_overlap = False
            if pr and nr:
                temporal_overlap = not (pr[1] < nr[0] or nr[1] < pr[0])
            cause = "unresolved"
            if shared_raw:
                cause = "raw_track_identity_switch_candidate"
            elif shared_doc:
                cause = "documented_link_impurity_candidate"
            elif temporal_overlap:
                cause = "overlapping_temporal_grouping_candidate"
            else:
                cause = "grouping_overmerge_candidate"
            conflict_rows.append(
                {
                    "evaluation_component_id": cid,
                    "positive_member": pos_m[0]["sample_eval_code"] if pos_m else None,
                    "negative_member": neg_m[0]["sample_eval_code"] if neg_m else None,
                    "positive_frame_range": pr,
                    "negative_frame_range": nr,
                    "temporal_overlap": temporal_overlap,
                    "shared_raw_track": bool(shared_raw),
                    "shared_documented_link_component": bool(shared_doc),
                    "exact_near_duplicate_relation": {
                        "positive_exact_crop_sha_group": pos_map.get("exact_crop_sha_group")
                        if pos_map
                        else None,
                        "negative_exact_crop_sha_group": neg_map.get("exact_crop_sha_group")
                        if neg_map
                        else None,
                    },
                    "members_score_rank": members,
                    "potential_cause": cause,
                    "ground_truth_unchanged": True,
                    "component_contract_unchanged": True,
                }
            )
        write_json(
            tmp
            / "component_diagnostics"
            / "target_001_conflicting_component_lineage_diagnostics.json",
            {
                "schema_version": "reid_target_001_conflicting_component_lineage_diagnostics_v1",
                "components": conflict_rows,
            },
        )

        # Decode frames needed for sheets/videos
        sheet_items = pos_enriched + fp_enriched + excl_enriched
        conflict_codes = []
        for comp in f3["conflict_audit"]["components"]:
            for m in sorted(comp["members"], key=lambda x: x["sample_eval_code"]):
                conflict_codes.append(m["sample_eval_code"])
        conflict_enriched = []
        for code in conflict_codes:
            # may already be enriched
            existing = next((e for e in sheet_items if e["sample_eval_code"] == code), None)
            if existing:
                conflict_enriched.append(existing)
            else:
                rank = next(
                    (int(r["rank"]) for r in ranking if r["sample_eval_code"] == code),
                    None,
                )
                conflict_enriched.append(
                    enrich(code, official_rank=rank, cohort="conflicting_component")
                )

        needed_frames: set[int] = set()
        for e in sheet_items + conflict_enriched:
            obs = e["_obs"]
            frames = [int(o["frame_index"]) for o in obs]
            ctx = select_context_frames(
                frames,
                start_frame=e["segment_start_frame"],
                end_frame=e["segment_end_frame"],
                representative_frame=e["representative_frame"],
            )
            e["_context_roles"] = ctx
            for _, fi in ctx:
                needed_frames.add(fi)
            # video windows: up to 4s of observation frames around rep
            if frames:
                rep = e["representative_frame"]
                window = sorted(frames, key=lambda f: (abs(f - rep), f))
                # take contiguous obs around nearest
                nearest = window[0]
                span = int(config["diagnostic_videos"]["max_seconds_per_item"] * 30)
                cand = [f for f in frames if abs(f - nearest) <= span // 2]
                if not cand:
                    cand = [nearest]
                cand = sorted(cand)[:span]
                e["_video_frames"] = cand
                for fi in cand:
                    needed_frames.add(fi)

        cap = cv2.VideoCapture(str(sample_mp4))
        if not cap.isOpened():
            raise ErrorAnalysisError("failed to open sample.mp4")
        decoded: dict[int, np.ndarray] = {}
        try:
            for fi in sorted(needed_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ErrorAnalysisError(f"decode failed frame {fi}")
                decoded[fi] = frame
        finally:
            cap.release()

        obs_bbox_index: dict[tuple[str, int], list[float]] = {}
        for e in sheet_items + conflict_enriched:
            for o in e["_obs"]:
                fi = int(o["frame_index"])
                bbox = o["source_observation"]["bbox_xyxy"]
                obs_bbox_index[(e["segment_id"], fi)] = [float(v) for v in bbox]

        def build_contexts(e: Mapping[str, Any]) -> list[tuple[str, np.ndarray]]:
            panels = []
            for role, fi in e["_context_roles"]:
                bbox = obs_bbox_index.get((e["segment_id"], fi))
                if bbox is None:
                    raise ErrorAnalysisError(f"missing bbox {e['segment_id']} {fi}")
                tsec = fi / 30.0
                panels.append(
                    (
                        role,
                        annotate_context(
                            decoded[fi], bbox, role=role, frame_index=fi, time_sec=tsec
                        ),
                    )
                )
            return panels

        min_w = int(config["contact_sheets"]["min_width_px"])
        cols = int(config["contact_sheets"]["grid_cols"])
        tile_w = max(900, int(math.ceil(min_w / cols)))
        tile_h = 820

        def make_tile(e: Mapping[str, Any], extra: str = "") -> np.ndarray:
            rank = e["official_rank"]
            gt = e["frozen_ground_truth_decision"]
            lines = [
                e["sample_eval_code"],
                f"GT={gt}" + (f" | {extra}" if extra else ""),
                f"rank={rank}" if rank is not None else "rank=n/a (excluded)",
                f"score={e['official_primary_score']:.6f}",
                f"best={e['best_gallery_anchor_id']}",
                e["crop_quality_summary"],
            ]
            if not e.get("retrieval_metric_eligible", True):
                lines.append("official_metric_inclusion=false")
            return render_diagnostic_tile(
                lines=lines,
                crop_bgr=e["_crop_bgr"],
                contexts=build_contexts(e),
                tile_w=tile_w,
                tile_h=tile_h,
            )

        review_dir = tmp / "review_packages" / "target_001_retrieval_error_review"
        sheets = {
            "target_001_all_positive_diagnostics.png": [
                make_tile(e, "POSITIVE") for e in pos_enriched
            ],
            "target_001_top_false_positives_01.png": [
                make_tile(e, "FP") for e in fp_enriched[:12]
            ],
            "target_001_top_false_positives_02.png": [
                make_tile(e, "FP") for e in fp_enriched[12:24]
            ],
            "target_001_excluded_high_score_diagnostics.png": [
                make_tile(e, "EXCLUDED") for e in excl_enriched
            ],
            "target_001_conflicting_components.png": [
                make_tile(e, "CONFLICT") for e in conflict_enriched
            ],
        }
        for name, tiles in sheets.items():
            sheet = render_contact_sheet(
                tiles, title=f"target_001 retrieval error — {name}", min_width=min_w, cols=cols
            )
            if sheet.shape[1] < 3600:
                raise ErrorAnalysisError("sheet width < 3600")
            out = review_dir / name
            if not cv2.imwrite(str(out), sheet):
                raise ErrorAnalysisError(f"imwrite failed {name}")

        # Diagnostic videos
        def write_diag_video(
            path: Path, items: Sequence[Mapping[str, Any]], *, gt_label_fn
        ) -> None:
            # title card + item frames
            frames_out: list[np.ndarray] = []
            fps = float(config["diagnostic_videos"]["fps"])
            for e in items:
                title = np.full((FRAME_H, FRAME_W, 3), 20, dtype=np.uint8)
                texts = [
                    WATERMARK,
                    e["sample_eval_code"],
                    f"GT: {gt_label_fn(e)}",
                    f"rank={e['official_rank']} score={e['official_primary_score']:.6f}",
                    f"best={e['best_gallery_anchor_id']}",
                    f"frames {e['segment_start_frame']}-{e['segment_end_frame']}",
                ]
                for i, text in enumerate(texts):
                    cv2.putText(
                        title,
                        text,
                        (40, 80 + i * 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0 if i else 0.7,
                        (240, 240, 240),
                        2,
                        cv2.LINE_AA,
                    )
                for _ in range(int(fps)):  # 1s title
                    frames_out.append(title.copy())
                for fi in e["_video_frames"]:
                    frame = decoded[fi].copy()
                    bbox = obs_bbox_index[(e["segment_id"], fi)]
                    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 220, 255), 2)
                    overlay_lines = [
                        WATERMARK,
                        e["sample_eval_code"],
                        f"GT:{gt_label_fn(e)}",
                        f"rank={e['official_rank']} score={e['official_primary_score']:.4f}",
                        f"best={e['best_gallery_anchor_id']}",
                        f"frame={fi}",
                    ]
                    for i, text in enumerate(overlay_lines):
                        cv2.putText(
                            frame,
                            text,
                            (12, 28 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (20, 20, 20),
                            3,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            text,
                            (12, 28 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (240, 240, 240),
                            1,
                            cv2.LINE_AA,
                        )
                    frames_out.append(frame)
            if not frames_out:
                raise ErrorAnalysisError("empty diagnostic video")
            h, w = frames_out[0].shape[:2]
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            if not writer.isOpened():
                raise ErrorAnalysisError(f"VideoWriter failed {path}")
            for fr in frames_out:
                writer.write(fr)
            writer.release()

        write_diag_video(
            tmp / "videos" / "target_001_positive_retrieval_diagnostics.mp4",
            pos_enriched,
            gt_label_fn=lambda e: "POSITIVE",
        )
        write_diag_video(
            tmp / "videos" / "target_001_top_false_positive_diagnostics.mp4",
            fp_enriched[:12],
            gt_label_fn=lambda e: (
                "NON_PLAYER"
                if e["frozen_ground_truth_decision"] == "non_player"
                else "NEGATIVE"
            ),
        )

        # Template
        template_rows = []
        seen = set()
        for e in pos_enriched + fp_enriched + excl_enriched + conflict_enriched:
            code = e["sample_eval_code"]
            if code in seen:
                # merge cohort membership
                for row in template_rows:
                    if row["sample_eval_code"] == code:
                        parts = set(row["cohort_membership"].split("|"))
                        parts.update(e["cohort_membership"].split("|"))
                        row["cohort_membership"] = "|".join(sorted(parts))
                continue
            seen.add(code)
            template_rows.append(
                {
                    "sample_eval_code": code,
                    "cohort_membership": e["cohort_membership"],
                    "frozen_ground_truth_decision": e["frozen_ground_truth_decision"],
                    "official_rank": "" if e["official_rank"] is None else e["official_rank"],
                    "official_primary_score": f"{e['official_primary_score']:.8f}",
                    "best_anchor_id": e["best_gallery_anchor_id"],
                    "segment_frame_range": f"{e['segment_start_frame']}-{e['segment_end_frame']}",
                    "representative_crop_path": e["representative_crop_path"],
                    "manual_visible_team": "",
                    "manual_visible_jersey_number": "",
                    "manual_view_category": "",
                    "manual_crop_quality": "",
                    "manual_overlap_present": "",
                    "manual_track_or_component_issue": "",
                    "manual_primary_confusion_reason": "",
                    "manual_gallery_gap_hypothesis": "",
                    "manual_notes": "",
                    "reviewer": "",
                    "final_approver": "",
                    "reviewed_at": "",
                }
            )
        with (tmp / "templates" / "target_001_retrieval_error_manual_review_template.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TEMPLATE_FIELDS))
            writer.writeheader()
            for row in template_rows:
                writer.writerow(row)

        # Review contract/manifest
        write_json(
            tmp / "review" / "target_001_retrieval_error_review_contract.json",
            {
                "schema_version": "reid_target_001_retrieval_error_review_contract_v1",
                "contact_sheets": 5,
                "diagnostic_videos": 2,
                "sample_used_for_error_analysis_only": True,
                "sample_authorized_for_gallery_optimization": False,
                "same_sample_revalidation_after_tuning_not_independent": True,
                "gallery_mutation": False,
                "threshold_selected": False,
                "identity_assignments": 0,
                "official_f3_result_unchanged": True,
            },
        )

        pngs = sorted((review_dir).glob("*.png"))
        mp4s = sorted((tmp / "videos").glob("*.mp4"))
        if len(pngs) != 5 or len(mp4s) != 2:
            raise ErrorAnalysisError(f"artifact budget sheets/videos {len(pngs)}/{len(mp4s)}")
        write_json(
            tmp / "review" / "target_001_retrieval_error_review_manifest.json",
            {
                "schema_version": "reid_target_001_retrieval_error_review_manifest_v1",
                "contact_sheets": [
                    {"path": str(p.relative_to(tmp)).replace("\\", "/"), "sha256": sha256_file(p)}
                    for p in pngs
                ],
                "diagnostic_videos": [
                    {"path": str(p.relative_to(tmp)).replace("\\", "/"), "sha256": sha256_file(p)}
                    for p in mp4s
                ],
                "generated_at": generated_at,
            },
        )

        contract = {
            "schema_version": "reid_stage5d_f3a_retrieval_error_analysis_contract_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "official_f3_result_unchanged": True,
            "gallery_members": 7,
            "ground_truth_decisions": 150,
            "ground_truth_unchanged": True,
            "new_embeddings": 0,
            "gallery_mutation": False,
            "threshold_selected": False,
            "identity_assignments": 0,
            "sample_used_for_error_analysis_only": True,
            "sample_authorized_for_gallery_optimization": False,
            "same_sample_revalidation_after_tuning_not_independent": True,
            "refinement_hypotheses_diagnostic_only": True,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3a_contract.json", contract)

        summary = {
            "schema_version": "reid_stage5d_f3a_retrieval_error_analysis_summary_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "official_f3_outcome_unchanged": True,
            "official_f3_descriptive_outcome": "INDEPENDENT_RETRIEVAL_PROMISING_BUT_OVERLAPPING",
            "high_ranked_positives": 3,
            "low_ranked_positives": 5,
            "top_false_positive_cohort": 24,
            "overlap_negative_cohort": len(overlap_neg),
            "minimum_positive_score": min_pos_score,
            "excluded_review_cohort": 12,
            "conflict_components": 4,
            "contact_sheets": 5,
            "diagnostic_videos": 2,
            "gallery_members": 7,
            "gallery_mutation": False,
            "threshold_selected": False,
            "identity_assignments": 0,
            "new_embeddings": 0,
            "score_recompute": False,
            "sample_decoded_for_diagnostic_render_only": True,
            "special_excluded_target_present": list(TARGET_PRESENT_AMBIGUOUS),
            "f3_snapshot_sha256": f3["snapshot_sha256"],
            "sample_sha256": config["evaluation_source"]["expected_sha256"],
            "network_used": False,
            "package_environment_changed": False,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3a_summary.json", summary)

        # budget checks
        if list(tmp.rglob("*.npy")):
            raise ErrorAnalysisError("npy forbidden in F3A")
        if len(list(tmp.rglob("*.png"))) != 5:
            raise ErrorAnalysisError("png budget")
        if len(list(tmp.rglob("*.mp4"))) != 2:
            raise ErrorAnalysisError("mp4 budget")

        n_before, listing_before = listing_sha(tmp)
        manifest = {
            "schema_version": "reid_stage5d_f3a_retrieval_error_analysis_manifest_v1",
            "final_status": FINAL_STATUS,
            "project_head": head,
            "output_root": final_rel,
            "listing_file_count_before_manifest": n_before,
            "listing_sha256_before_manifest": listing_before,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3a_manifest.json", manifest)
        n_final, listing_final = listing_sha(tmp)
        manifest["listing_file_count"] = n_final
        manifest["listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f3a_manifest.json", manifest)
        summary["output_file_count"] = n_final
        summary["output_listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f3a_summary.json", summary)

        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    if listing_sha(f3["root"])[1] != f3_listing:
        raise ErrorAnalysisError("F3 mutated")
    if listing_sha(f2a_root)[1] != f2a_listing:
        raise ErrorAnalysisError("F2A mutated")
    if listing_sha(gallery_root)[1] != gallery_listing:
        raise ErrorAnalysisError("gallery mutated")
    if sha256_file(emb) != config["sample_embeddings"]["expected_artifact_sha256"]:
        raise ErrorAnalysisError("embeddings mutated")
    if sha256_file(sample_mp4) != config["evaluation_source"]["expected_sha256"]:
        raise ErrorAnalysisError("sample.mp4 mutated")

    return load_json(final_dir / "stage5d_f3a_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze target_001 retrieval errors and gallery diagnostics."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to target_retrieval_error_analysis_stage5d_target_001.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(Path(args.config))
    except ErrorAnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"status={summary['final_status']} "
        f"high_pos={summary['high_ranked_positives']} "
        f"low_pos={summary['low_ranked_positives']} "
        f"top_fp={summary['top_false_positive_cohort']} "
        f"overlap_neg={summary['overlap_negative_cohort']} "
        f"sheets={summary['contact_sheets']} "
        f"videos={summary['diagnostic_videos']} "
        f"next_gate={summary['exact_next_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
