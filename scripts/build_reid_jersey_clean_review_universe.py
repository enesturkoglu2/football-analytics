#!/usr/bin/env python3
"""Build label-blind canonical jersey review universe (Stage 5C rebuild r2).

Reads Stage 5C visibility crop signals only. Does not render panels, run OCR,
create manual labels, or perform discovery/holdout selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from football_analytics.reid.writers import (  # noqa: E402
    ReIDWritersError,
    check_output_collision,
    cleanup_dir,
    create_temp_output_dir,
    finalize_output_dir,
    write_manifest_jsonl,
)

CONFIG_SCHEMA = "reid_jersey_clean_review_universe_config_v1"
ITEM_SCHEMA = "reid_jersey_clean_review_item_v1"
VIS_CROP_SCHEMA = "reid_jersey_visibility_crop_signal_v1"
SUMMARY_SCHEMA = "reid_jersey_clean_review_universe_summary_v1"

MANUAL_NULL_FIELDS = (
    "manual_crop_valid",
    "manual_back_facing",
    "manual_number_visible",
    "manual_number_readable",
    "manual_digit_count",
    "manual_jersey_number",
    "manual_contamination_affects_number_region",
    "manual_notes",
    "reviewer",
    "reviewed_at",
)

OCR_FORBIDDEN_FIELDS = (
    "ocr_prediction",
    "ocr_confidence",
    "ocr_exact_match",
    "ocr_wrong",
    "threshold_acceptance",
)

SELECTION_FORBIDDEN_FIELDS = (
    "discovery_member",
    "holdout_member",
    "previous_discovery_membership",
    "previous_holdout_membership",
)


class CleanUniverseError(RuntimeError):
    """Raised when clean review universe construction fails."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CleanUniverseError(f"config must be a mapping: {path}")
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise CleanUniverseError(f"invalid config schema_version: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CleanUniverseError(f"malformed JSONL {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise CleanUniverseError(f"JSONL row must be object: {path}:{line_no}")
            rows.append(row)
    return rows


def _finite(value: Any, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CleanUniverseError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise CleanUniverseError(f"{field} is NaN/Inf")
    return out


def _index_by_crop_id(path: Path | None, *, id_field: str = "crop_id") -> dict[str, dict]:
    if path is None or not path.is_file():
        return {}
    out: dict[str, dict] = {}
    for row in _load_jsonl(path):
        cid = row.get(id_field)
        if not isinstance(cid, str) or not cid:
            continue
        if cid in out:
            raise CleanUniverseError(f"duplicate {id_field} in join source {path}: {cid}")
        out[cid] = row
    return out


def _global_map(path: Path | None) -> dict[int, int]:
    if path is None or not path.is_file():
        return {}
    mapping: dict[int, int] = {}
    for row in _load_jsonl(path):
        tid = row.get("raw_track_id", row.get("track_id"))
        gid = row.get("global_candidate_id")
        if isinstance(tid, int) and isinstance(gid, int):
            mapping[tid] = gid
    return mapping


def _canonical_sort_key(row: Mapping[str, Any]) -> tuple:
    return (
        int(row["raw_track_id"]),
        str(row["segment_id"]),
        int(row["frame_index"]),
        int(row["selection_rank"]),
        str(row["crop_id"]),
    )


def build_clean_items(
    *,
    visibility_rows: Sequence[Mapping[str, Any]],
    quality_by_crop: Mapping[str, Mapping[str, Any]],
    kit_by_crop: Mapping[str, Mapping[str, Any]],
    global_by_track: Mapping[int, int],
    project_root: Path,
    require_sha_match: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(visibility_rows, key=_canonical_sort_key)
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_crop_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_shas: set[str] = set()
    quality_missing = 0
    kit_missing = 0
    roi_invalid = 0

    for index, row in enumerate(ordered, start=1):
        if row.get("schema_version") != VIS_CROP_SCHEMA:
            raise CleanUniverseError(
                f"visibility schema mismatch for crop {row.get('crop_id')}"
            )
        crop_id = str(row["crop_id"])
        if crop_id in seen_crop_ids:
            raise CleanUniverseError(f"duplicate crop_id: {crop_id}")
        seen_crop_ids.add(crop_id)

        review_item_id = f"review_{crop_id}"
        if review_item_id in seen_ids:
            raise CleanUniverseError(f"duplicate review_item_id: {review_item_id}")
        seen_ids.add(review_item_id)

        source_path = Path(str(row["source_crop_path"])).expanduser().resolve()
        try:
            source_path.relative_to(project_root.resolve())
        except ValueError as exc:
            raise CleanUniverseError(
                f"source crop escapes project root: {source_path}"
            ) from exc
        if not source_path.is_file():
            raise CleanUniverseError(f"missing source crop: {source_path}")
        path_key = str(source_path)
        if path_key in seen_paths:
            raise CleanUniverseError(f"duplicate source crop path: {path_key}")
        seen_paths.add(path_key)

        expected_sha = str(row["source_crop_sha256"])
        if expected_sha in seen_shas:
            raise CleanUniverseError(f"duplicate source crop SHA: {expected_sha}")
        seen_shas.add(expected_sha)
        if require_sha_match:
            actual = sha256_file(source_path)
            if actual != expected_sha:
                raise CleanUniverseError(f"source crop SHA mismatch: {crop_id}")

        width = int(row["crop_width_px"])
        height = int(row["crop_height_px"])
        if width <= 0 or height <= 0:
            raise CleanUniverseError(f"invalid crop dimensions: {crop_id}")

        x0 = int(row["roi_x_min"])
        y0 = int(row["roi_y_min"])
        x1 = int(row["roi_x_max"])
        y1 = int(row["roi_y_max"])
        roi_w = int(row["roi_width_px"])
        roi_h = int(row["roi_height_px"])
        roi_ok = (
            0 <= x0 < x1 <= width
            and 0 <= y0 < y1 <= height
            and roi_w == (x1 - x0)
            and roi_h == (y1 - y0)
            and roi_w > 0
            and roi_h > 0
        )
        if not roi_ok:
            roi_invalid += 1
            raise CleanUniverseError(f"invalid ROI bounds: {crop_id}")

        for field in (
            "laplacian_variance",
            "tenengrad_mean",
            "local_contrast",
            "entropy",
            "edge_density",
            "roi_other_person_union_coverage",
        ):
            _finite(row[field], field=f"{crop_id}.{field}")

        for field in MANUAL_NULL_FIELDS:
            if field in row and row[field] is not None:
                raise CleanUniverseError(
                    f"manual field must be null on visibility row: {field}"
                )
        for field in OCR_FORBIDDEN_FIELDS + SELECTION_FORBIDDEN_FIELDS:
            if field in row and row[field] not in (None, False, "", []):
                raise CleanUniverseError(
                    f"forbidden field present on visibility row: {field}"
                )

        q = quality_by_crop.get(crop_id)
        k = kit_by_crop.get(crop_id)
        if q is None:
            quality_missing += 1
        if k is None:
            kit_missing += 1

        tid = int(row["raw_track_id"])
        gid = global_by_track.get(tid)
        crop_area = width * height
        roi_rel = float(row["roi_area_px"]) / float(crop_area)

        item: dict[str, Any] = {
            "schema_version": ITEM_SCHEMA,
            "review_item_id": review_item_id,
            "canonical_order": index,
            "crop_id": crop_id,
            "segment_id": str(row["segment_id"]),
            "raw_track_id": tid,
            "frame_index": int(row["frame_index"]),
            "source_crop_path": path_key,
            "source_crop_sha256": expected_sha,
            "source_crop_width": width,
            "source_crop_height": height,
            "crop_source_type": str(row["crop_source_kind"]),
            "crop_rank_within_segment": int(row["selection_rank"]),
            "selection_rank": int(row["selection_rank"]),
            "number_roi_x": x0,
            "number_roi_y": y0,
            "number_roi_width": roi_w,
            "number_roi_height": roi_h,
            "number_roi_x_max": x1,
            "number_roi_y_max": y1,
            "number_roi_relative_area": roi_rel,
            "roi_valid": True,
            "laplacian_variance": float(row["laplacian_variance"]),
            "tenengrad_mean": float(row["tenengrad_mean"]),
            "local_contrast": float(row["local_contrast"]),
            "entropy": float(row["entropy"]),
            "edge_density": float(row["edge_density"]),
            "roi_other_person_union_coverage": float(
                row["roi_other_person_union_coverage"]
            ),
            "roi_other_person_center_inside_count": int(
                row["roi_other_person_center_inside_count"]
            ),
            "quality_signal_joined": q is not None,
            "kit_descriptor_joined": k is not None,
            "quality_laplacian_variance": (
                None if q is None else q.get("laplacian_variance")
            ),
            "kit_dominant_color_family": (
                None if k is None else k.get("dominant_color_family")
            ),
            "documented_global_candidate_id": gid,
            "documented_global_candidate_id_semantics": (
                "candidate_id_not_player_identity_not_jersey_not_team"
            ),
            "rebuild_generation": "r2",
            "stage5c_universe_generation": "r2",
            "annotation_status": "unreviewed",
            "review_universe_label_blind": True,
            "selection_performed": False,
            "ocr_predictions_used": False,
            "old_manual_labels_recovered": False,
            "new_manual_labels_generated": False,
        }
        for field in MANUAL_NULL_FIELDS:
            item[field] = None
        for field in OCR_FORBIDDEN_FIELDS:
            item[field] = None
        items.append(item)

    stats = {
        "quality_missing": quality_missing,
        "kit_missing": kit_missing,
        "roi_invalid": roi_invalid,
        "unique_review_item_ids": len(seen_ids),
        "unique_crop_ids": len(seen_crop_ids),
        "unique_paths": len(seen_paths),
        "unique_shas": len(seen_shas),
    }
    return items, stats


def run_build_clean_review_universe(
    *,
    visibility_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    quality_signals: str | Path | None = None,
    kit_descriptors: str | Path | None = None,
    global_id_map: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    vis_root = Path(visibility_dir).expanduser().resolve()
    cfg_path = Path(config_path).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    try:
        check_output_collision(out_root, overwrite=overwrite)
    except ReIDWritersError as exc:
        raise CleanUniverseError(str(exc)) from exc

    crop_signals = vis_root / "jersey_visibility_crop_signals.jsonl"
    vis_summary_path = vis_root / "jersey_visibility_summary.json"
    if not crop_signals.is_file() or not vis_summary_path.is_file():
        raise CleanUniverseError("visibility measurement artifacts missing")

    vis_summary = json.loads(vis_summary_path.read_text(encoding="utf-8"))
    if vis_summary.get("status") != "ok":
        raise CleanUniverseError("visibility summary status is not ok")

    rows = _load_jsonl(crop_signals)
    quality_by = _index_by_crop_id(
        Path(quality_signals).expanduser().resolve() if quality_signals else None
    )
    kit_by = _index_by_crop_id(
        Path(kit_descriptors).expanduser().resolve() if kit_descriptors else None
    )
    global_by = _global_map(
        Path(global_id_map).expanduser().resolve() if global_id_map else None
    )

    require_sha = bool(cfg.get("input", {}).get("require_source_crop_sha_match", True))
    items, stats = build_clean_items(
        visibility_rows=rows,
        quality_by_crop=quality_by,
        kit_by_crop=kit_by,
        global_by_track=global_by,
        project_root=project,
        require_sha_match=require_sha,
    )

    source_type_counts = Counter(item["crop_source_type"] for item in items)
    per_segment: Counter[str] = Counter()
    per_track: Counter[int] = Counter()
    for item in items:
        per_segment[str(item["segment_id"])] += 1
        per_track[int(item["raw_track_id"])] += 1

    def _dist(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"min": 0, "median": 0, "max": 0}
        values = sorted(values)
        mid = values[len(values) // 2]
        return {"min": values[0], "median": mid, "max": values[-1]}

    temp_dir: Path | None = None
    try:
        temp_dir = create_temp_output_dir(out_root)
        items_path = temp_dir / "clean_review_items.jsonl"
        write_manifest_jsonl(items_path, items)

        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "ok",
            "created_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "canonical_item_count": len(items),
            "visibility_signal_count": len(rows),
            "source_type_counts": dict(source_type_counts),
            "segments_with_items": len(per_segment),
            "raw_tracks_with_items": len(per_track),
            "items_per_segment": _dist(list(per_segment.values())),
            "items_per_raw_track": _dist(list(per_track.values())),
            "frame_index_min": min(int(i["frame_index"]) for i in items),
            "frame_index_max": max(int(i["frame_index"]) for i in items),
            "quality_join_missing_count": stats["quality_missing"],
            "kit_join_missing_count": stats["kit_missing"],
            "roi_invalid_count": stats["roi_invalid"],
            "duplicate_review_item_id": 0,
            "duplicate_crop_id": 0,
            "duplicate_source_path": 0,
            "duplicate_source_sha": 0,
            "missing_source_crop": 0,
            "sha_mismatch": 0,
            "manual_labels_seen": False,
            "OCR_predictions_seen": False,
            "OCR_confidence_seen": False,
            "old_manual_label_artifact_available": False,
            "old_manual_labels_reconstructed": False,
            "previous_discovery_membership_used": False,
            "previous_holdout_membership_used": False,
            "selection_performed": False,
            "review_universe_label_blind": True,
            "png_count": 0,
            "jpeg_count": 0,
            "mp4_count": 0,
            "rebuild_generation": "r2",
            "stage5c_universe_generation": "r2",
            "source_visibility_dir": str(vis_root),
            "source_visibility_summary_sha256": sha256_file(vis_summary_path),
            "source_config": str(cfg_path),
            "source_config_sha256": sha256_file(cfg_path),
            "review_item_id_contract": "review_{crop_id}",
            "deterministic_ordering": [
                "raw_track_id",
                "segment_id",
                "frame_index",
                "selection_rank",
                "crop_id",
            ],
        }
        (temp_dir / "clean_review_universe_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        finalize_output_dir(temp_dir=temp_dir, final_dir=out_root, overwrite=overwrite)
        temp_dir = None
        return summary
    except Exception:
        if temp_dir is not None:
            cleanup_dir(temp_dir)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a label-blind canonical jersey review universe from Stage 5C "
            "visibility crop signals. No panel rendering, OCR, or selection."
        )
    )
    parser.add_argument("--visibility-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quality-signals", default=None)
    parser.add_argument("--kit-descriptors", default=None)
    parser.add_argument("--global-id-map", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_build_clean_review_universe(
            visibility_dir=args.visibility_dir,
            config_path=args.config,
            project_root=args.project_root,
            output_dir=args.output_dir,
            quality_signals=args.quality_signals,
            kit_descriptors=args.kit_descriptors,
            global_id_map=args.global_id_map,
            overwrite=args.overwrite,
        )
    except (CleanUniverseError, ReIDWritersError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        "status=ok "
        f"canonical_item_count={summary['canonical_item_count']} "
        f"source_type_counts={summary['source_type_counts']} "
        f"output_dir={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
