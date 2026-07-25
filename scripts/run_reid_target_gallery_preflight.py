#!/usr/bin/env python3
"""Stage 5D-A target gallery enrollment design and asset preflight.

Design/preflight only. No target selection, gallery membership, identity
assignment, prototypes, ranking, contact sheets, or model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "reid_target_gallery_preflight_config_v1"


class PreflightError(RuntimeError):
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
        raise PreflightError("unexpected config schema")
    if not config.get("offline_required"):
        raise PreflightError("offline_required")
    return config


def assert_git_contract(project_root: Path, expected_head: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise PreflightError("BLOCKED_STAGE5D_A_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise PreflightError("BLOCKED_STAGE5D_A_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    # Allow only this triad while developing; at final run expect clean OR
    # the three new files + docs being written in same gate. During pipeline
    # execution tracked docs may already be dirty after edits — caller runs
    # after code exists. For atomic preflight publish we accept only:
    # empty porcelain OR only the exact Stage 5D-A triad/docs paths.
    allowed_prefixes = (
        "?? scripts/run_reid_target_gallery_preflight.py",
        "?? configs/reid/target_gallery_preflight_stage5d.yaml",
        "?? tests/test_reid_target_gallery_preflight.py",
        "?? docs/setup/stage5d-target-gallery-enrollment-design-and-preflight.md",
        " M README.md",
        " M PROJECT_CONTEXT.md",
        " M docs/setup/stage5-identity-signals-plan.md",
        "M  README.md",
        "M  PROJECT_CONTEXT.md",
        "M  docs/setup/stage5-identity-signals-plan.md",
    )
    if porcelain:
        for line in porcelain.splitlines():
            if not any(line.startswith(p) or line == p for p in allowed_prefixes):
                # also allow " M docs/..." variants with leading space
                if line[3:] in {
                    "README.md",
                    "PROJECT_CONTEXT.md",
                    "docs/setup/stage5-identity-signals-plan.md",
                    "docs/setup/stage5d-target-gallery-enrollment-design-and-preflight.md",
                    "scripts/run_reid_target_gallery_preflight.py",
                    "configs/reid/target_gallery_preflight_stage5d.yaml",
                    "tests/test_reid_target_gallery_preflight.py",
                } and line[:2] in {"??", " M", "M ", "A ", "AM"}:
                    continue
                raise PreflightError(
                    "BLOCKED_STAGE5D_A_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != "Close Stage 5C PARSeq holdout validation":
        raise PreflightError("BLOCKED_STAGE5D_A_GIT_CONTRACT_MISMATCH message")
    return head


def validate_stage5c_closure(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = project_root / config["stage5c_closure"]["path"]
    policy = load_json(root / config["stage5c_closure"]["policy_json"])
    required = {
        "stage5c_status": "closed",
        "holdout_validation_decision": "INCONCLUSIVE_SAFE_BUT_LOW_SUPPORT",
        "automated_parseq_jersey_evidence_enabled": False,
        "stage5e_automated_jersey_channel_mode": "diagnostic_only",
        "automated_parseq_identity_assignment_allowed": False,
        "automated_parseq_identity_veto_allowed": False,
        "automated_parseq_gallery_enrollment_allowed": False,
        "threshold_adjustment_using_current_holdout_forbidden": True,
        "discovery_reserve_opened": False,
        "holdout_reserve_opened": False,
        "stage5d_blocked": False,
        "appearance_reid_remains_primary": True,
        "unknown_identity_preserved": True,
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise PreflightError(
                f"BLOCKED_STAGE5D_A_STAGE5C_CLOSURE_MISMATCH {key}={policy.get(key)!r}"
            )
    snap = Path(config["stage5c_closure"]["snapshot_path"])
    if not snap.is_file():
        raise PreflightError("BLOCKED_STAGE5D_A_STAGE5C_CLOSURE_MISMATCH snapshot_missing")
    if sha256_file(snap) != config["stage5c_closure"]["snapshot_sha256"]:
        raise PreflightError("BLOCKED_STAGE5D_A_STAGE5C_CLOSURE_MISMATCH snapshot_sha")
    n, listing = listing_sha(root)
    return {
        "path": str(config["stage5c_closure"]["path"]),
        "listing_file_count": n,
        "listing_sha256": listing,
        "policy": {k: policy[k] for k in required},
        "snapshot_sha256": config["stage5c_closure"]["snapshot_sha256"],
    }


def validate_assets(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    video = project_root / config["source_video"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    for path, label in ((video, "video"), (yolo, "yolo"), (osnet, "osnet")):
        if not path.is_file() or path.is_symlink():
            raise PreflightError(f"BLOCKED_STAGE5D_A_ASSET_INTEGRITY {label}")
    if video.stat().st_size != int(config["source_video"]["expected_bytes"]):
        raise PreflightError("BLOCKED_STAGE5D_A_ASSET_INTEGRITY video_bytes")
    if sha256_file(video) != config["source_video"]["expected_sha256"]:
        raise PreflightError("BLOCKED_STAGE5D_A_ASSET_INTEGRITY video_sha")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise PreflightError("BLOCKED_STAGE5D_A_ASSET_INTEGRITY yolo_bytes")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise PreflightError("BLOCKED_STAGE5D_A_ASSET_INTEGRITY yolo_sha")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise PreflightError("BLOCKED_STAGE5D_A_ASSET_INTEGRITY osnet_sha")

    det = load_json(
        project_root
        / config["upstream"]["rebuild_root"]
        / "detection"
        / "detection_summary.json"
    )
    src = det["source"]
    if int(src["frame_count_source"]) != int(config["source_video"]["expected_frames"]):
        raise PreflightError("BLOCKED_STAGE5D_A_ASSET_INTEGRITY frames")
    if int(src["width"]) != int(config["source_video"]["expected_width"]):
        raise PreflightError("BLOCKED_STAGE5D_A_ASSET_INTEGRITY width")
    if int(src["height"]) != int(config["source_video"]["expected_height"]):
        raise PreflightError("BLOCKED_STAGE5D_A_ASSET_INTEGRITY height")
    if float(src["fps"]) != float(config["source_video"]["expected_fps"]):
        raise PreflightError("BLOCKED_STAGE5D_A_ASSET_INTEGRITY fps")
    duration = float(src["frame_count_source"]) / float(src["fps"])
    # Metadata-only duration (no video decode). Expected ≈34.111979s.
    if abs(duration - 34.111979) > 0.05:
        raise PreflightError(f"BLOCKED_STAGE5D_A_ASSET_INTEGRITY duration={duration}")
    return {
        "source_video": {
            "path": config["source_video"]["path"],
            "bytes": video.stat().st_size,
            "sha256": config["source_video"]["expected_sha256"],
            "frames": int(src["frame_count_source"]),
            "width": int(src["width"]),
            "height": int(src["height"]),
            "fps": float(src["fps"]),
            "duration_sec": duration,
            "metadata_source": "detection_summary.json (no video decode)",
        },
        "yolo_checkpoint": {
            "path": config["yolo_checkpoint"]["path"],
            "bytes": yolo.stat().st_size,
            "sha256": config["yolo_checkpoint"]["expected_sha256"],
            "loaded": False,
            "inference_run": False,
        },
        "osnet_checkpoint": {
            "path": config["osnet_checkpoint"]["path"],
            "sha256": config["osnet_checkpoint"]["expected_sha256"],
            "loaded": False,
            "inference_run": False,
        },
    }


def resolve_upstream(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    up = config["upstream"]
    exp = config["expected_counts"]
    rebuild_root = project_root / up["rebuild_root"]
    overlay = project_root / up["documented_link_overlay"]
    stage5 = project_root / up["stage5_replay"]
    visibility = project_root / up["visibility_universe"]
    split = project_root / up["canonical_split"]
    for path, label in (
        (rebuild_root, "rebuild"),
        (overlay, "overlay"),
        (stage5, "stage5"),
        (visibility, "visibility"),
        (split, "split"),
    ):
        if not path.is_dir():
            raise PreflightError(
                f"BLOCKED_STAGE5D_A_UPSTREAM_REID_CONTRACT_MISMATCH missing_{label}"
            )

    rebuild = load_json(rebuild_root / "rebuild_summary.json")
    if int(rebuild["tracking_observations"]) != exp["tracking_observations"]:
        raise PreflightError("BLOCKED_STAGE5D_A_UPSTREAM_REID_CONTRACT_MISMATCH obs")
    if int(rebuild["tracking_raw_ids"]) != exp["raw_track_ids"]:
        raise PreflightError("BLOCKED_STAGE5D_A_UPSTREAM_REID_CONTRACT_MISMATCH tracks")
    if int(rebuild["crops"]) != exp["initial_crops"]:
        raise PreflightError("BLOCKED_STAGE5D_A_UPSTREAM_REID_CONTRACT_MISMATCH crops")
    if list(rebuild["embeddings_shape"]) != list(exp["initial_embeddings"]):
        raise PreflightError("BLOCKED_STAGE5D_A_UPSTREAM_REID_CONTRACT_MISMATCH emb_shape")
    if int(rebuild["track_embeddings"]) != exp["aggregated_units"]:
        raise PreflightError("BLOCKED_STAGE5D_A_UPSTREAM_REID_CONTRACT_MISMATCH agg")
    if int(rebuild["candidate_pairs"]) != exp["initial_candidate_pairs"]:
        raise PreflightError("BLOCKED_STAGE5D_A_UPSTREAM_REID_CONTRACT_MISMATCH pairs")

    overlay_sum = load_json(overlay / "documented_replay_summary.json")
    hist = overlay_sum["historical_count_comparison"]["measured"]
    if int(hist["accepted_edges"]) != exp["replayed_documented_links"]:
        raise PreflightError("BLOCKED_STAGE5D_A_UPSTREAM_REID_CONTRACT_MISMATCH links")
    if int(hist["global_candidates"]) != exp["replayed_global_ids"]:
        raise PreflightError("BLOCKED_STAGE5D_A_UPSTREAM_REID_CONTRACT_MISMATCH globals")

    s5 = load_json(stage5 / "stage5_rebuild_summary.json")
    seg = s5["segment_replay"]
    reid = s5["segmented_reid"]
    checks = {
        "final_segment_plan_total": int(seg["total_segments"]),
        "manual_segments": int(seg["manual_segments"]),
        "pass_through_segments": int(seg["pass_through"]),
        "assigned_observations": int(seg["assigned_observations"]),
        "unassigned_observations": int(seg["unassigned_observations"]),
        "embedded_segments": int(reid["embedded"]),
        "no_embedding_segments": int(reid["no_embedding"]),
        "recomputed_embedded_segments": int(reid["recomputed"]),
        "reused_embedded_segments": int(reid["reused"]),
        "new_segment_crops": int(reid["new_crops"]),
        "segmented_candidate_pairs": int(reid["ranked_candidates"]),
    }
    for key, value in checks.items():
        if value != int(exp[key]):
            raise PreflightError(
                f"BLOCKED_STAGE5D_A_UPSTREAM_REID_CONTRACT_MISMATCH {key}={value}"
            )

    vis = load_json(visibility / "stage5c_universe_summary.json")
    vis_count = int(vis.get("canonical_item_count") or 0)
    if vis_count != exp["visibility_universe_items"]:
        raise PreflightError("BLOCKED_STAGE5D_A_UPSTREAM_REID_CONTRACT_MISMATCH vis")

    split_sum = load_json(split / "clean_split_summary.json")
    if split_sum.get("canonical_split_generation") != "r2_capacity_balanced":
        raise PreflightError("BLOCKED_STAGE5D_A_UPSTREAM_REID_CONTRACT_MISMATCH split_gen")

    return {
        "source_video": "data/test_clips/sample.mp4",
        "detection": str(Path(up["rebuild_root"]) / "detection"),
        "raw_tracking": str(Path(up["rebuild_root"]) / "tracking"),
        "documented_link_replay": up["documented_link_overlay"],
        "segmented_reid": str(
            Path(up["stage5_replay"]) / up["segmented_reid_subdir"]
        ),
        "segment_view": str(Path(up["stage5_replay"]) / up["segment_view_subdir"]),
        "visibility_universe": up["visibility_universe"],
        "canonical_split": up["canonical_split"],
        "stage5c_closure": config["stage5c_closure"]["path"],
        "verified_counts": {
            "frames": exp["frames"],
            **{k: int(exp[k]) if not isinstance(exp[k], list) else exp[k] for k in exp},
        },
        "manifest_sources": {
            "rebuild_summary": str(Path(up["rebuild_root"]) / "rebuild_summary.json"),
            "documented_replay_summary": str(
                Path(up["documented_link_overlay"]) / "documented_replay_summary.json"
            ),
            "stage5_rebuild_summary": str(
                Path(up["stage5_replay"]) / "stage5_rebuild_summary.json"
            ),
            "stage5c_universe_summary": str(
                Path(up["visibility_universe"]) / "stage5c_universe_summary.json"
            ),
        },
    }


def embedding_preflight(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    seg_root = (
        project_root
        / config["upstream"]["stage5_replay"]
        / config["upstream"]["segmented_reid_subdir"]
    )
    npz_path = seg_root / "segment_embeddings.npz"
    index_path = seg_root / "segment_embedding_index.jsonl"
    if not npz_path.is_file() or not index_path.is_file():
        raise PreflightError("BLOCKED_STAGE5D_A_EMBEDDING_PREFLIGHT missing")
    data = np.load(npz_path)
    if "vectors" not in data.files or "segment_ids" not in data.files:
        raise PreflightError("BLOCKED_STAGE5D_A_EMBEDDING_PREFLIGHT keys")
    vectors = data["vectors"]
    segment_ids = [str(x) for x in data["segment_ids"].tolist()]
    if vectors.shape != (150, 512):
        raise PreflightError(f"BLOCKED_STAGE5D_A_EMBEDDING_PREFLIGHT shape={vectors.shape}")
    if vectors.dtype != np.float32:
        raise PreflightError(f"BLOCKED_STAGE5D_A_EMBEDDING_PREFLIGHT dtype={vectors.dtype}")
    if int(np.isnan(vectors).sum()) != 0 or int(np.isinf(vectors).sum()) != 0:
        raise PreflightError("BLOCKED_STAGE5D_A_EMBEDDING_PREFLIGHT naninf")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if int((norms == 0).sum()) != 0:
        raise PreflightError("BLOCKED_STAGE5D_A_EMBEDDING_PREFLIGHT zero")
    if len(segment_ids) != len(set(segment_ids)):
        raise PreflightError("BLOCKED_STAGE5D_A_EMBEDDING_PREFLIGHT duplicate_id")

    index_rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(index_rows) != 291:
        raise PreflightError("BLOCKED_STAGE5D_A_EMBEDDING_PREFLIGHT index_rows")
    embedded = [r for r in index_rows if r.get("embedding_available")]
    no_emb = [r for r in index_rows if not r.get("embedding_available")]
    if len(embedded) != 150 or len(no_emb) != 141:
        raise PreflightError("BLOCKED_STAGE5D_A_EMBEDDING_PREFLIGHT coverage")
    for row in embedded:
        if row.get("embedding_dimension") != 512:
            raise PreflightError("BLOCKED_STAGE5D_A_EMBEDDING_PREFLIGHT dim")
        if row.get("embedding_row") is None:
            raise PreflightError("BLOCKED_STAGE5D_A_EMBEDDING_PREFLIGHT missing_row")

    hashes = [hashlib.sha256(vectors[i].tobytes()).hexdigest() for i in range(len(vectors))]
    hash_counts = Counter(hashes)
    exact_dup_groups = sum(1 for n in hash_counts.values() if n > 1)

    # Temporary L2 view for diagnostics only (not published as gallery).
    l2 = vectors / norms[:, None]
    # near-identical: cosine >= 0.9999 excluding self
    near = 0
    for i in range(len(l2)):
        sims = l2 @ l2[i]
        near += int(((sims >= 0.9999) & (np.arange(len(l2)) != i)).sum())
    near //= 2

    source_dist = Counter(
        "reused"
        if str(r.get("representation_source", "")).startswith("reused")
        else (
            "recomputed"
            if "recomput" in str(r.get("representation_source", ""))
            or str(r.get("representation_status", "")).startswith("recomput")
            else "other"
        )
        for r in embedded
    )
    # Prefer summary counts for reused/recomputed
    s5 = load_json(
        project_root
        / config["upstream"]["stage5_replay"]
        / "stage5_rebuild_summary.json"
    )
    return {
        "embedding_artifact_path": str(
            Path(config["upstream"]["stage5_replay"])
            / config["upstream"]["segmented_reid_subdir"]
            / "segment_embeddings.npz"
        ),
        "embedding_artifact_sha256": sha256_file(npz_path),
        "metadata_artifact_path": str(
            Path(config["upstream"]["stage5_replay"])
            / config["upstream"]["segmented_reid_subdir"]
            / "segment_embedding_index.jsonl"
        ),
        "metadata_artifact_sha256": sha256_file(index_path),
        "embedded_segment_count": 150,
        "no_embedding_segment_count": 141,
        "dtype": str(vectors.dtype),
        "shape": list(vectors.shape),
        "nan_count": 0,
        "inf_count": 0,
        "zero_vector_count": 0,
        "duplicate_segment_id_count": 0,
        "missing_embedding_lineage_for_embedded": 0,
        "norm_min": float(norms.min()),
        "norm_median": float(statistics.median(norms.tolist())),
        "norm_max": float(norms.max()),
        "exact_duplicate_vector_groups": exact_dup_groups,
        "near_identical_pair_count_cosine_ge_0_9999": near,
        "reused_embedded_segments": int(s5["segmented_reid"]["reused"]),
        "recomputed_embedded_segments": int(s5["segmented_reid"]["recomputed"]),
        "source_distribution_diagnostic": dict(source_dist),
        "overwrite_forbidden": True,
        "gallery_prototype_published": False,
        "temporary_l2_view_only": True,
    }


def load_stage5c_exclusion_keys(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, set[str]]:
    split = project_root / config["upstream"]["canonical_split"]
    keys = {
        "segment_id": set(),
        "raw_track_id": set(),
        "crop_id": set(),
        "source_crop_path": set(),
        "source_crop_sha256": set(),
        "split_item_id": set(),
    }
    for batch in (
        "discovery_primary",
        "discovery_reserve",
        "holdout_primary",
        "holdout_reserve",
    ):
        man = split / batch / f"{batch}_manifest.jsonl"
        if not man.is_file():
            man = split / f"{batch}_manifest.jsonl"
        if not man.is_file():
            raise PreflightError(
                f"BLOCKED_STAGE5D_A_GALLERY_EXCLUSION_CONTRACT missing {batch}"
            )
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            keys["split_item_id"].add(str(row["split_item_id"]))
            keys["segment_id"].add(str(row.get("segment_id") or ""))
            keys["raw_track_id"].add(str(row.get("raw_track_id") or ""))
            keys["crop_id"].add(str(row.get("crop_id") or ""))
            keys["source_crop_path"].add(str(row.get("source_crop_path") or ""))
            keys["source_crop_sha256"].add(str(row.get("source_crop_sha256") or ""))
    keys = {k: {x for x in v if x} for k, v in keys.items()}
    return keys


def build_source_eligibility(
    project_root: Path,
    config: Mapping[str, Any],
    stage5c_keys: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    seg_root = (
        project_root
        / config["upstream"]["stage5_replay"]
        / config["upstream"]["segmented_reid_subdir"]
    )
    index_path = seg_root / "segment_embedding_index.jsonl"
    crop_man_path = seg_root / "segment_crop_manifest.jsonl"
    crops_by_seg: dict[str, list[dict[str, Any]]] = {}
    if crop_man_path.is_file():
        for line in crop_man_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            crops_by_seg.setdefault(str(row["segment_id"]), []).append(row)

    rows_out: list[dict[str, Any]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row["segment_id"])
        raw_tid = str(row.get("raw_track_id"))
        exclusion_reasons: list[str] = []
        in_stage5c = sid in stage5c_keys["segment_id"] or raw_tid in stage5c_keys[
            "raw_track_id"
        ]
        if in_stage5c:
            exclusion_reasons.append("stage5c_membership_exclusion")
        if not row.get("embedding_available"):
            exclusion_reasons.append("no_existing_osnet_embedding")
        crops = crops_by_seg.get(sid, [])
        # Representative crop paths for recomputed segments; reused may rely on baseline
        rep_paths = [
            str(c.get("crop_relative_path") or c.get("crop_id") or "") for c in crops[:5]
        ]
        eligibility = "eligible_for_future_manual_review" if not exclusion_reasons else "excluded_or_incomplete"
        # Stage 5D-A does not decide target-positive; eligibility is preflight only.
        rows_out.append(
            {
                "segment_id": sid,
                "raw_track_id": int(row["raw_track_id"]),
                "global_id": row.get("global_id"),
                "frame_range": [row.get("first_frame"), row.get("last_frame")],
                "crop_count": int(row.get("crop_count") or 0),
                "representative_crop_paths": rep_paths,
                "representative_crop_ids": list(row.get("crop_ids") or [])[:5],
                "embedding_available": bool(row.get("embedding_available")),
                "embedding_dimension": row.get("embedding_dimension"),
                "embedding_row": row.get("embedding_row"),
                "embedding_sha256": row.get("embedding_sha256"),
                "representation_source": row.get("representation_source"),
                "representation_status": row.get("representation_status"),
                "segment_kind": row.get("segment_kind"),
                "manual_segmentation_provenance": row.get("segment_kind"),
                "source_video_lineage": "data/test_clips/sample.mp4",
                "stage5c_membership_or_overlap": in_stage5c,
                "eligibility_status": eligibility,
                "exclusion_reasons": exclusion_reasons,
                "target_positive_decision": None,
                "gallery_member": False,
            }
        )
    if len(rows_out) != 291:
        raise PreflightError("BLOCKED_STAGE5D_A_UPSTREAM_REID_CONTRACT_MISMATCH eligibility")
    return rows_out


def blank_target_template() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_target_definition_template_v1",
        "target_id": "",
        "target_alias": "",
        "target_description": "",
        "identity_basis": "",
        "human_verified_jersey_number": None,
        "jersey_number_provenance": "",
        "source_video": "data/test_clips/sample.mp4",
        "allowed_anchor_source_policy": "full_body_segmented_reid_crops_with_existing_osnet_embedding",
        "forbidden_source_policy": [
            "stage5c_discovery_or_holdout_items",
            "jersey_number_roi_only",
            "automatic_similarity_enrollment",
            "ocr_based_automatic_enrollment",
            "tracker_id_based_automatic_enrollment",
        ],
        "reviewer": "",
        "final_approver": "",
        "reviewed_at": "",
        "target_definition_frozen": False,
        "notes": "",
    }


def design_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_gallery_design_contract_v1",
        "enrollment_mode": "manual_frozen",
        "automatic_gallery_growth": False,
        "self_training": False,
        "pseudo_label_enrollment": False,
        "ocr_based_automatic_enrollment": False,
        "tracker_id_based_automatic_enrollment": False,
        "similarity_based_automatic_enrollment": False,
        "appearance_reid_primary": True,
        "jersey_roi_forbidden_for_gallery": True,
        "unknown_identity_preserved": True,
        "forced_identity_assignment": False,
        "target_identity_selected_in_stage5d_a": False,
        "gallery_membership_created_in_stage5d_a": False,
        "identity_assignment_in_stage5d_a": False,
        "reid_threshold_selected_in_stage5d_a": False,
        "membership_requires": [
            "human_target_positive",
            "crop_valid",
            "football_player_role_verified",
            "target_dominant_in_crop",
            "no_heavy_overlap",
            "no_wrong_person_attribution",
            "complete_segment_crop_lineage",
            "source_crop_sha_verified",
            "existing_osnet_embedding_matched",
            "gallery_evaluation_exclusion_passed",
        ],
        "annotation_classes": {
            "target_positive": "enrollment_eligible_only",
            "target_negative": "not_enrolled",
            "uncertain": "never_enrolled",
            "invalid_crop": "never_enrolled",
            "non_player": "never_enrolled",
            "multi_person_ambiguous": "never_enrolled",
        },
        "gallery_representation_plan": {
            "base_feature": "existing_osnet_512d_embedding",
            "stored_source": "original_frozen_embedding",
            "scoring_view": "l2_normalized_embedding",
            "enrollment_unit": "manually_approved_segment_or_crop",
            "prototype_types_after_enrollment_freeze": [
                "individual_approved_embeddings",
                "target_centroid",
                "target_medoid",
            ],
            "prototype_creation_in_stage5d_a": False,
            "score_threshold_in_stage5d_a": False,
            "fusion_weight_in_stage5d_a": False,
            "classifier_training": False,
        },
        "diversity_rules_planned": [
            "limited_representatives_per_segment",
            "limited_representatives_per_raw_track",
            "prefer_distinct_time_windows",
            "view_diversity_front_back_side",
            "scale_diversity",
            "pose_diversity",
            "acceptable_blur_occlusion_diversity",
            "near_duplicate_suppression",
        ],
        "diversity_limits_enforced_in_stage5d_a": False,
    }


def exclusion_contract(stage5c_keys: Mapping[str, set[str]]) -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_gallery_evaluation_exclusion_contract_v1",
        "overlap_forbidden_keys": [
            "segment_id",
            "raw_track_id",
            "crop_id",
            "source_crop_path",
            "source_crop_sha256",
            "frame_identity",
            "exact_duplicate_group",
            "near_duplicate_component",
            "documented_link_component",
            "temporal_source_window",
        ],
        "required_overlap_count": 0,
        "same_segment_gallery_and_evaluation_forbidden": True,
        "crop_copy_reuse_forbidden": True,
        "near_duplicate_split_across_gallery_eval_forbidden": True,
        "stage5c_batches_excluded_from_gallery_and_evaluation_inputs": [
            "discovery_primary",
            "discovery_reserve",
            "holdout_primary",
            "holdout_reserve",
        ],
        "stage5c_items_readable_for_exclusion_provenance_only": True,
        "holdout_primary_forbidden_for_gallery_enrollment": True,
        "holdout_results_forbidden_for_target_selection": True,
        "discovery_or_holdout_parseq_confidence_forbidden": True,
        "discovery_reserve_open_forbidden": True,
        "holdout_reserve_open_forbidden": True,
        "future_evaluation_set_created_in_stage5d_a": False,
        "stage5c_exclusion_key_counts": {k: len(v) for k, v in stage5c_keys.items()},
    }


def workflow_preregistration() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_workflow_preregistration_v1",
        "gates": {
            "STAGE5D-A": {
                "name": "design_and_asset_preflight",
                "target_selection": False,
                "image_review": False,
                "status_this_gate": "active_completing",
            },
            "STAGE5D-B": {
                "name": "target_definition_and_anchor_review_package",
                "human_target_alias_basis_approval": True,
                "anchor_review_package": True,
                "anchors_are_final_gallery_membership": False,
            },
            "STAGE5D-C": {
                "name": "bounded_candidate_retrieval_and_human_decisions",
                "uses_frozen_human_approved_anchor_embeddings": True,
                "cosine_similarity_ordering_only": True,
                "automatic_enrollment": False,
            },
            "STAGE5D-D": {
                "name": "manual_enrollment_annotation_freeze",
                "gallery_membership_frozen": True,
                "gallery_evaluation_leakage_audit": True,
                "automatic_gallery_growth": False,
            },
            "STAGE5D-E": {
                "name": "frozen_gallery_prototype_generation",
                "identity_assignment": False,
            },
            "STAGE5D-F": {
                "name": "independent_gallery_retrieval_validation",
                "stage5e_readiness_decision": True,
            },
        },
        "automatic_gallery_growth_forbidden_in_all_stage5d_gates": True,
        "anchor_review_options": [
            "target_anchor_yes",
            "target_anchor_no",
            "uncertain",
            "invalid",
            "multi_person_ambiguous",
            "non_player",
        ],
        "candidate_retrieval_plan": {
            "only_frozen_human_approved_anchor_embeddings": True,
            "existing_embeddings_only": True,
            "cosine_similarity": True,
            "deterministic_ranking": True,
            "bounded_top_k": True,
            "duplicate_and_same_segment_collapse": True,
            "source_diversity": True,
            "score_ordering_only_not_enrollment_decision": True,
            "executed_in_stage5d_a": False,
        },
        "exact_next_gate": "STAGE5D-B_TARGET_DEFINITION_AND_ANCHOR_REVIEW_PACKAGE",
    }


def run_pipeline(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    final_dir = project_root / config["output"]["final_dir"]
    if final_dir.exists():
        raise PreflightError("FAILED_STAGE5D_A_ATOMIC_OUTPUT final_exists")

    head = assert_git_contract(project_root, config["project_head_expected"])
    closure = validate_stage5c_closure(project_root, config)
    assets = validate_assets(project_root, config)
    upstream = resolve_upstream(project_root, config)
    emb = embedding_preflight(project_root, config)
    stage5c_keys = load_stage5c_exclusion_keys(project_root, config)
    eligibility = build_source_eligibility(project_root, config, stage5c_keys)

    token = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp = final_dir.parent / (
        f"_tmp_full_stage4b_rebuild_r2_stage5d_target_gallery_preflight_{token}"
    )
    if tmp.exists():
        raise PreflightError("FAILED_STAGE5D_A_ATOMIC_OUTPUT temp_exists")
    tmp.mkdir(parents=False)
    (tmp / "templates").mkdir()

    try:
        design = design_contract()
        write_json(tmp / "stage5d_gallery_design_contract.json", design)

        inventory = {
            "schema_version": "reid_stage5d_asset_inventory_v1",
            "project_head": head,
            "upstream_chain": upstream,
            "assets": assets,
            "stage5c_closure": closure,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        write_json(tmp / "stage5d_asset_inventory.json", inventory)
        write_json(tmp / "stage5d_embedding_preflight.json", emb)

        eligible_count = sum(
            1 for r in eligibility if r["eligibility_status"].startswith("eligible")
        )
        excluded_count = len(eligibility) - eligible_count
        audit = {
            "schema_version": "reid_stage5d_source_eligibility_audit_v1",
            "segment_units": 291,
            "eligible_for_future_manual_review": eligible_count,
            "excluded_or_incomplete": excluded_count,
            "target_positive_decisions": 0,
            "gallery_members": 0,
            "rows": eligibility,
        }
        write_json(tmp / "stage5d_source_eligibility_audit.json", audit)

        excl = exclusion_contract(stage5c_keys)
        write_json(tmp / "stage5d_gallery_evaluation_exclusion_contract.json", excl)

        workflow = workflow_preregistration()
        write_json(tmp / "stage5d_workflow_preregistration.json", workflow)

        template = blank_target_template()
        write_json(tmp / "templates" / "target_definition_template.json", template)
        if any(
            [
                template["target_id"],
                template["target_alias"],
                template["identity_basis"],
                template["human_verified_jersey_number"] is not None,
                template["target_definition_frozen"],
                template["reviewer"],
                template["final_approver"],
            ]
        ):
            raise PreflightError("FAILED_STAGE5D_A_ATOMIC_OUTPUT template_not_blank")

        summary = {
            "schema_version": "reid_stage5d_preflight_summary_v1",
            "final_status": "COMPLETED_STAGE5D_A_TARGET_GALLERY_PREFLIGHT_READY",
            "project_head": head,
            "stage5c_status": "closed",
            "holdout_validation_decision": "INCONCLUSIVE_SAFE_BUT_LOW_SUPPORT",
            "automated_jersey_channel_mode": "diagnostic_only",
            "upstream_verified": True,
            "embedding_preflight_ok": True,
            "embedded_segments": 150,
            "no_embedding_segments": 141,
            "target_definition_frozen": False,
            "target_id": "",
            "target_positive_decisions": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "similarity_ranking_rows": 0,
            "identity_assignments": 0,
            "automatic_gallery_growth": False,
            "new_inference": 0,
            "reserve_reads": 0,
            "png_count": 0,
            "jpeg_count": 0,
            "mp4_count": 0,
            "exact_next_gate": "STAGE5D-B_TARGET_DEFINITION_AND_ANCHOR_REVIEW_PACKAGE",
        }
        write_json(tmp / "stage5d_preflight_summary.json", summary)

        artifact_rels = [
            "stage5d_gallery_design_contract.json",
            "stage5d_asset_inventory.json",
            "stage5d_embedding_preflight.json",
            "stage5d_source_eligibility_audit.json",
            "stage5d_gallery_evaluation_exclusion_contract.json",
            "stage5d_workflow_preregistration.json",
            "stage5d_preflight_summary.json",
            "templates/target_definition_template.json",
        ]
        arts = []
        for rel in artifact_rels:
            path = tmp / rel
            arts.append(
                {
                    "path": rel,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        manifest = {
            "schema_version": "reid_stage5d_preflight_manifest_v1",
            "project_head": head,
            "artifacts": arts,
            "png_count": 0,
            "jpeg_count": 0,
            "mp4_count": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "similarity_ranking_rows": 0,
            "identity_assignments": 0,
            "atomic_finalization": True,
            "network_download": 0,
            "model_inference": 0,
        }
        write_json(tmp / "stage5d_preflight_manifest.json", manifest)

        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.jpg")) or list(tmp.rglob("*.mp4")):
            raise PreflightError("FAILED_STAGE5D_A_ATOMIC_OUTPUT media")
        os.replace(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return {
        "final_status": summary["final_status"],
        "output_root": str(final_dir),
        "gallery_members": 0,
        "target_id": "",
        "exact_next_gate": summary["exact_next_gate"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs/reid/target_gallery_preflight_stage5d.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    args = parser.parse_args()
    result = run_pipeline(args.config.resolve(), args.project_root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
