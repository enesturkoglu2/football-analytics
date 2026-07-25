#!/usr/bin/env python3
"""Stage 5D-B1E-C — freeze human-verified external positive occurrences.

Binds EXT_004 / EXT_183 / EXT_198 to target_001. Does not convert
unreviewed codes to negatives. No crops, embeddings, gallery, OCR,
similarity, detection, or tracking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "reid_external_positive_occurrence_freeze_config_v1"
SELECTED_CODES = ("EXT_004", "EXT_183", "EXT_198")
CSV_FIELDS = (
    "target_id",
    "external_candidate_code",
    "resolved_raw_track_id",
    "first_frame",
    "last_frame",
    "observation_count",
    "representative_frame",
    "source_video_path",
    "source_video_sha256",
    "manual_occurrence_decision",
    "manual_same_target_as_target_001",
    "manual_human_verified_number_seen",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_identity_continuity_observed",
    "reviewer",
    "final_approver",
    "reviewed_at",
    "manual_notes",
)
FINAL_STATUS = "COMPLETED_STAGE5D_B1E_C_TARGET_001_EXTERNAL_OCCURRENCES_FROZEN"
NEXT_GATE = "STAGE5D-B1E-D_TARGET_001_EXTERNAL_TRACKLET_QUALITY_AND_ANCHOR_REVIEW_PACKAGE"


class OccurrenceFreezeError(RuntimeError):
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
        raise OccurrenceFreezeError("unexpected config schema")
    if not config.get("offline_required"):
        raise OccurrenceFreezeError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/"):
        raise OccurrenceFreezeError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise OccurrenceFreezeError("BLOCKED_STAGE5D_B1E_C_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_GIT_CONTRACT_MISMATCH origin"
        )
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    allowed = {
        "scripts/run_reid_external_positive_occurrence_freeze.py",
        "configs/reid/external_positive_occurrence_freeze_stage5d_target_001.yaml",
        "tests/test_reid_external_positive_occurrence_freeze.py",
        "docs/setup/stage5d-target-external-positive-occurrence-freeze.md",
    }
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path not in allowed:
                raise OccurrenceFreezeError(
                    "BLOCKED_STAGE5D_B1E_C_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != "Add external target tracking seed review package":
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_GIT_CONTRACT_MISMATCH message"
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
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_GIT_CONTRACT_MISMATCH external_tracked"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path) -> str:
    if not snapshot_path.is_file():
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_REVIEW_PACKAGE_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not manifest.is_file():
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_REVIEW_PACKAGE_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man = str(load_json(manifest).get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual):
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_REVIEW_PACKAGE_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def validate_assets(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    ext = project_root / config["external_enrollment_source"]["path"]
    sample = project_root / config["evaluation_source"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if not ext.is_file() or ext.is_symlink():
        raise OccurrenceFreezeError("external source must be regular file")
    if ext.stat().st_size != int(config["external_enrollment_source"]["expected_bytes"]):
        raise OccurrenceFreezeError("external bytes mismatch")
    if sha256_file(ext) != config["external_enrollment_source"]["expected_sha256"]:
        raise OccurrenceFreezeError("external sha mismatch")
    if sha256_file(sample) != config["evaluation_source"]["expected_sha256"]:
        raise OccurrenceFreezeError("sample sha mismatch")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise OccurrenceFreezeError("yolo bytes mismatch")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise OccurrenceFreezeError("yolo sha mismatch")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise OccurrenceFreezeError("osnet sha mismatch")
    return {
        "external_path": config["external_enrollment_source"]["path"],
        "external_sha256": config["external_enrollment_source"]["expected_sha256"],
        "external_bytes": int(config["external_enrollment_source"]["expected_bytes"]),
        "sample_sha256": config["evaluation_source"]["expected_sha256"],
        "yolo_sha256": config["yolo_checkpoint"]["expected_sha256"],
        "osnet_sha256": config["osnet_checkpoint"]["expected_sha256"],
    }


def validate_target_definition(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    path = project_root / config["target_definition"]["path"]
    td = load_json(path)
    if td.get("target_id") != "target_001":
        raise OccurrenceFreezeError("target_id mismatch")
    if td.get("target_alias") != "sarı takım 5 numaralı oyuncu":
        raise OccurrenceFreezeError("target_alias mismatch")
    if not td.get("target_definition_frozen"):
        raise OccurrenceFreezeError("target_definition_frozen required")
    if td.get("identity_basis") != "human_visual_verification_from_source_video":
        raise OccurrenceFreezeError("identity_basis mismatch")
    if int(td.get("human_verified_jersey_number")) != 5:
        raise OccurrenceFreezeError("jersey number mismatch")
    if (
        td.get("jersey_number_provenance")
        != "human_verified_by_user_not_automated_ocr"
    ):
        raise OccurrenceFreezeError("jersey provenance mismatch")
    if td.get("automated_jersey_used") is not False:
        raise OccurrenceFreezeError("automated_jersey_used must be false")
    return {
        "path": config["target_definition"]["path"],
        "sha256": sha256_file(path),
        "definition": td,
    }


def validate_b1e_a(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root / config["stage5d_b1e_a_package"]["path"]
    summary = load_json(root / "stage5d_b1e_a_summary.json")
    if summary.get("final_status") != config["stage5d_b1e_a_package"][
        "expected_final_status"
    ]:
        raise OccurrenceFreezeError("B1E-A status mismatch")
    if summary.get("exact_file_duplicate") is not False:
        raise OccurrenceFreezeError("B1E-A exact duplicate must be false")
    if int(summary.get("verified_overlap_interval_count", -1)) != 0:
        raise OccurrenceFreezeError("B1E-A overlap intervals must be 0")
    return summary


def validate_b1e_b(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = project_root / config["stage5d_b1e_b_package"]["path"]
    summary = load_json(root / "stage5d_b1e_b_summary.json")
    contract = load_json(root / "stage5d_b1e_b_contract.json")
    manifest = load_json(root / "stage5d_b1e_b_manifest.json")
    exp = config["stage5d_b1e_b_package"]
    if summary.get("final_status") != exp["expected_final_status"]:
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_REVIEW_PACKAGE_CONTRACT_MISMATCH status"
        )
    if summary.get("target_id") != "target_001":
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_REVIEW_PACKAGE_CONTRACT_MISMATCH target"
        )
    if summary["external_source"]["sha256"] != config["external_enrollment_source"][
        "expected_sha256"
    ]:
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_REVIEW_PACKAGE_CONTRACT_MISMATCH source_sha"
        )
    if int(summary["external_source"]["frames"]) != 784:
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_REVIEW_PACKAGE_CONTRACT_MISMATCH frames"
        )
    checks = [
        ("detection_total", exp["expected_detection_total"]),
        ("detection_frames_with_boxes", exp["expected_detection_frames_with_boxes"]),
        ("tracking_total_observations", exp["expected_tracking_observations"]),
        ("raw_track_count", exp["expected_raw_track_count"]),
        ("ext_candidate_count", exp["expected_ext_candidate_count"]),
        ("review_eligible_candidate_count", exp["expected_review_eligible_count"]),
    ]
    for key, expected in checks:
        if int(summary.get(key)) != int(expected):
            raise OccurrenceFreezeError(
                f"BLOCKED_STAGE5D_B1E_C_REVIEW_PACKAGE_CONTRACT_MISMATCH {key}"
            )
    if summary.get("two_replay_determinism") is not True:
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_REVIEW_PACKAGE_CONTRACT_MISMATCH determinism"
        )
    for key in (
        "manual_selections",
        "embeddings",
        "osnet_inference",
        "ocr",
        "similarity_ranking_rows",
        "gallery_members",
        "identity_assignments",
    ):
        if int(summary.get(key) or 0) != 0:
            raise OccurrenceFreezeError(
                f"BLOCKED_STAGE5D_B1E_C_REVIEW_PACKAGE_CONTRACT_MISMATCH {key}"
            )
    tpl = (
        root
        / "templates"
        / "target_001_external_seed_manual_review_template.csv"
    )
    with tpl.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 248:
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_REVIEW_PACKAGE_CONTRACT_MISMATCH blank_csv"
        )
    if any(r.get("manual_occurrence_decision") for r in rows):
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_REVIEW_PACKAGE_CONTRACT_MISMATCH prefilled"
        )
    snapshot_sha = resolve_snapshot_sha(Path(exp["snapshot_path"]))
    return {
        "root": root,
        "summary": summary,
        "contract": contract,
        "manifest": manifest,
        "snapshot_sha256": snapshot_sha,
    }


def load_mapping(root: Path) -> list[dict[str, Any]]:
    path = root / "inventory" / "target_001_external_track_candidate_mapping.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 248:
        raise OccurrenceFreezeError(
            "BLOCKED_STAGE5D_B1E_C_REVIEW_PACKAGE_CONTRACT_MISMATCH mapping_count"
        )
    codes = [r["external_candidate_code"] for r in rows]
    if len(codes) != len(set(codes)):
        raise OccurrenceFreezeError("duplicate EXT codes in mapping")
    return rows


def validate_selected_codes(
    mapping_rows: Sequence[Mapping[str, Any]],
    selected: Sequence[str],
    expected_source_sha: str,
) -> list[dict[str, Any]]:
    if tuple(selected) != SELECTED_CODES:
        raise OccurrenceFreezeError("exact selected code list mismatch")
    if len(selected) != len(set(selected)):
        raise OccurrenceFreezeError("duplicate selected codes")
    by_code = {str(r["external_candidate_code"]): r for r in mapping_rows}
    resolved: list[dict[str, Any]] = []
    seen_raw: set[int] = set()
    for code in selected:
        if code not in by_code:
            raise OccurrenceFreezeError(
                "BLOCKED_STAGE5D_B1E_C_SELECTED_CODE_NOT_FOUND " + code
            )
        row = by_code[code]
        raw_id = int(row["raw_external_track_id"])
        if raw_id in seen_raw:
            raise OccurrenceFreezeError(
                "BLOCKED_STAGE5D_B1E_C_OCCURRENCE_LINEAGE_AMBIGUOUS raw_reuse"
            )
        seen_raw.add(raw_id)
        # Ensure code binds to exactly this one mapping row (already unique by code).
        matches = [
            r for r in mapping_rows if r["external_candidate_code"] == code
        ]
        if len(matches) != 1:
            raise OccurrenceFreezeError(
                "BLOCKED_STAGE5D_B1E_C_OCCURRENCE_LINEAGE_AMBIGUOUS " + code
            )
        frames = list(row["observation_frames"])
        if not frames:
            raise OccurrenceFreezeError(f"empty observation frames for {code}")
        if frames != sorted(frames):
            raise OccurrenceFreezeError(f"unsorted observation frames for {code}")
        if int(row["first_frame"]) != int(frames[0]):
            raise OccurrenceFreezeError(f"first_frame mismatch {code}")
        if int(row["last_frame"]) != int(frames[-1]):
            raise OccurrenceFreezeError(f"last_frame mismatch {code}")
        if int(row["observation_count"]) != len(frames):
            raise OccurrenceFreezeError(f"observation_count mismatch {code}")
        if int(row["observation_count"]) <= 0:
            raise OccurrenceFreezeError(f"observation_count must be >0 for {code}")
        bboxes = list(row["bbox_per_observation"])
        lineage = list(row["detection_lineage"])
        if len(bboxes) != len(frames) or len(lineage) != len(frames):
            raise OccurrenceFreezeError(f"incomplete lineage for {code}")
        if row.get("source_video_sha256") != expected_source_sha:
            raise OccurrenceFreezeError(f"source sha mismatch for {code}")
        if row.get("representative_frame") is None:
            raise OccurrenceFreezeError(f"missing representative_frame for {code}")
        resolved.append(dict(row))
    return resolved


def compose_notes(shared: str, specific: str) -> str:
    shared = " ".join(shared.split())
    specific = " ".join(specific.split())
    return f"{shared} [{specific}]"


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_b1e_c_occurrence_freeze_contract_v1",
        "target_id": "target_001",
        "review_scope": "positive_occurrence_selection_only",
        "target_occurrence_freeze": True,
        "selected_positive_count": 3,
        "reviewed_negative_count": 0,
        "unreviewed_count": 245,
        "unreviewed_not_converted_to_negative": True,
        "automated_ocr_used": False,
        "similarity_used": False,
        "model_identity_prediction_used": False,
        "external_occurrences_are_gallery_members": False,
        "external_occurrences_are_final_anchors": False,
        "automatic_gallery_growth": False,
        "unknown_identity_preserved": True,
        "new_detection": 0,
        "new_tracking": 0,
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
    for key in (
        "external_enrollment_source",
        "evaluation_source",
        "yolo_checkpoint",
        "stage5d_b1e_b_package",
        "target_definition",
        "output",
    ):
        path_val = config[key].get("path") or config[key].get("final_dir")
        if path_val:
            assert_no_path_traversal(str(path_val))

    head = assert_git_contract(project_root, config["project_head_expected"])
    final_dir = project_root / config["output"]["final_dir"]
    if final_dir.exists():
        raise OccurrenceFreezeError("final_exists")
    assets = validate_assets(project_root, config)
    target_meta = validate_target_definition(project_root, config)
    b1e_a = validate_b1e_a(project_root, config)
    b1e_b = validate_b1e_b(project_root, config)
    mapping_rows = load_mapping(b1e_b["root"])

    human = config["human_positive_occurrence_freeze"]
    selected = list(human["selected_external_candidate_codes"])
    resolved = validate_selected_codes(
        mapping_rows,
        selected,
        expected_source_sha=assets["external_sha256"],
    )

    reviewed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = project_root / (
        "_tmp_full_stage4b_rebuild_r2_stage5d_target_001_external_occurrence_freeze_"
        + uuid.uuid4().hex
    )
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        freeze_dir = tmp / "occurrence_freeze"
        val_dir = tmp / "validation"
        runtime_dir = tmp / "runtime"
        cfg_dir = tmp / "effective_configs"
        freeze_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)
        runtime_dir.mkdir(parents=True)
        cfg_dir.mkdir(parents=True)
        shutil.copy2(config_path, cfg_dir / config_path.name)

        csv_rows: list[dict[str, Any]] = []
        per_notes = human["per_code_notes"]
        for row in resolved:
            code = row["external_candidate_code"]
            csv_rows.append(
                {
                    "target_id": "target_001",
                    "external_candidate_code": code,
                    "resolved_raw_track_id": int(row["raw_external_track_id"]),
                    "first_frame": int(row["first_frame"]),
                    "last_frame": int(row["last_frame"]),
                    "observation_count": int(row["observation_count"]),
                    "representative_frame": int(row["representative_frame"]),
                    "source_video_path": assets["external_path"],
                    "source_video_sha256": assets["external_sha256"],
                    "manual_occurrence_decision": human["manual_occurrence_decision"],
                    "manual_same_target_as_target_001": human[
                        "manual_same_target_as_target_001"
                    ],
                    "manual_human_verified_number_seen": human[
                        "manual_human_verified_number_seen"
                    ],
                    "manual_crop_valid": human["manual_crop_valid"],
                    "manual_target_dominant": human["manual_target_dominant"],
                    "manual_identity_continuity_observed": human[
                        "manual_identity_continuity_observed"
                    ],
                    "reviewer": human["reviewer"],
                    "final_approver": human["final_approver"],
                    "reviewed_at": reviewed_at,
                    "manual_notes": compose_notes(
                        human["shared_manual_notes"], per_notes[code]
                    ),
                }
            )

        csv_path = freeze_dir / "target_001_external_positive_occurrences_frozen.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)

        unreviewed_codes = sorted(
            r["external_candidate_code"]
            for r in mapping_rows
            if r["external_candidate_code"] not in selected
        )
        if len(unreviewed_codes) != 245:
            raise OccurrenceFreezeError("unreviewed count must be 245")

        freeze_payload = {
            "schema_version": "reid_target_external_positive_occurrence_freeze_v1",
            "final_status": FINAL_STATUS,
            "target_id": "target_001",
            "target_definition_path": target_meta["path"],
            "target_definition_sha256": target_meta["sha256"],
            "target_alias": target_meta["definition"]["target_alias"],
            "human_verified_jersey_number": 5,
            "jersey_number_provenance": target_meta["definition"][
                "jersey_number_provenance"
            ],
            "external_source_path": assets["external_path"],
            "external_source_sha256": assets["external_sha256"],
            "enrollment_only_source": True,
            "sample_verified_overlap": 0,
            "future_independent_evaluation_input": False,
            "selected_external_candidate_codes": list(SELECTED_CODES),
            "resolved_raw_track_ids": [
                int(r["resolved_raw_track_id"]) for r in csv_rows
            ],
            "observation_ranges": [
                {
                    "external_candidate_code": r["external_candidate_code"],
                    "resolved_raw_track_id": r["resolved_raw_track_id"],
                    "first_frame": r["first_frame"],
                    "last_frame": r["last_frame"],
                    "observation_count": r["observation_count"],
                    "representative_frame": r["representative_frame"],
                }
                for r in csv_rows
            ],
            "human_decisions": csv_rows,
            "selected_positive_count": 3,
            "reviewed_negative_count": 0,
            "unreviewed_count": 245,
            "unreviewed_external_candidate_codes_count": 245,
            "review_scope": "positive_occurrence_selection_only",
            "target_occurrence_freeze": True,
            "automated_ocr_used": False,
            "similarity_used": False,
            "model_identity_prediction_used": False,
            "external_occurrences_are_gallery_members": False,
            "external_occurrences_are_final_anchors": False,
            "automatic_gallery_growth": False,
            "unknown_identity_preserved": True,
            "reviewer": human["reviewer"],
            "final_approver": human["final_approver"],
            "reviewed_at": reviewed_at,
            "exact_next_gate": NEXT_GATE,
        }
        write_json(
            freeze_dir / "target_001_external_positive_occurrence_freeze.json",
            freeze_payload,
        )

        contract = build_contract()
        write_json(
            freeze_dir / "target_001_external_occurrence_freeze_contract.json",
            contract,
        )

        lineage_validation = {
            "schema_version": "reid_stage5d_b1e_c_occurrence_lineage_validation_v1",
            "selected_codes": list(SELECTED_CODES),
            "duplicate_selected": 0,
            "unknown_selected": 0,
            "unique_raw_track_bindings": True,
            "source_video_sha256": assets["external_sha256"],
            "b1e_b_two_replay_determinism": True,
            "b1e_b_tracking_replay_fingerprint_sha256": b1e_b["summary"][
                "tracking_replay_fingerprint_sha256"
            ],
            "resolved": [
                {
                    "external_candidate_code": r["external_candidate_code"],
                    "resolved_raw_track_id": r["resolved_raw_track_id"],
                    "first_frame": r["first_frame"],
                    "last_frame": r["last_frame"],
                    "observation_count": r["observation_count"],
                    "bbox_lineage_complete": True,
                    "detection_lineage_complete": True,
                    "observation_frames_sorted": True,
                }
                for r in csv_rows
            ],
            "unreviewed_policy": {
                "manual_review_status": "unreviewed",
                "manual_occurrence_decision": "",
                "not_counted_as_target_occurrence_no": True,
                "not_used_as_distractor_or_negative": True,
                "not_added_to_classifier_negative_set": True,
                "not_added_to_gallery": True,
                "no_identity_assignment": True,
                "count": 245,
            },
        }
        write_json(
            val_dir / "target_001_external_occurrence_lineage_validation.json",
            lineage_validation,
        )

        freeze_manifest = {
            "schema_version": "reid_stage5d_b1e_c_occurrence_freeze_manifest_v1",
            "csv": "occurrence_freeze/target_001_external_positive_occurrences_frozen.csv",
            "freeze_json": (
                "occurrence_freeze/target_001_external_positive_occurrence_freeze.json"
            ),
            "contract": (
                "occurrence_freeze/target_001_external_occurrence_freeze_contract.json"
            ),
            "selected_positive_count": 3,
            "reviewed_negative_count": 0,
            "unreviewed_count": 245,
            "png_count": 0,
            "mp4_count": 0,
            "crop_copy_count": 0,
            "embedding_copy_count": 0,
        }
        write_json(
            freeze_dir / "target_001_external_occurrence_freeze_manifest.json",
            freeze_manifest,
        )

        write_json(
            runtime_dir / "runtime.json",
            {
                "schema_version": "reid_stage5d_b1e_c_runtime_v1",
                "started_at": reviewed_at,
                "project_head": head,
                "osnet_loaded": False,
                "yolo_inference": 0,
                "bytetrack_inference": 0,
                "network_download": 0,
            },
        )

        summary = {
            "schema_version": "reid_stage5d_b1e_c_summary_v1",
            "final_status": FINAL_STATUS,
            "project_head": head,
            "target_id": "target_001",
            "target_alias": "sarı takım 5 numaralı oyuncu",
            "human_verified_jersey_number": 5,
            "jersey_number_provenance": "human_verified_by_user_not_automated_ocr",
            "external_source_path": assets["external_path"],
            "external_source_sha256": assets["external_sha256"],
            "selected_external_candidate_codes": list(SELECTED_CODES),
            "resolved_raw_track_ids": [
                int(r["resolved_raw_track_id"]) for r in csv_rows
            ],
            "observation_ranges": freeze_payload["observation_ranges"],
            "selected_positive_count": 3,
            "reviewed_negative_count": 0,
            "unreviewed_count": 245,
            "review_scope": "positive_occurrence_selection_only",
            "target_occurrence_freeze": True,
            "automated_ocr_used": False,
            "similarity_used": False,
            "model_identity_prediction_used": False,
            "approved_anchor_crops": 0,
            "embeddings": 0,
            "osnet_inference": 0,
            "ocr": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
            "new_detection": 0,
            "new_tracking": 0,
            "b1e_b_snapshot_sha256": b1e_b["snapshot_sha256"],
            "b1e_a_final_status": b1e_a["final_status"],
            "exact_next_gate": NEXT_GATE,
        }
        write_json(tmp / "stage5d_b1e_c_summary.json", summary)

        # Manifest after all other files exist except itself.
        n_files, listing = listing_sha(tmp)
        # listing currently excludes summary? summary already written.
        # Write manifest then recompute? Gate expects manifest with listing of package.
        # Write preliminary listing excluding manifest, then include both.
        manifest = {
            "schema_version": "reid_stage5d_b1e_c_manifest_v1",
            "final_status": FINAL_STATUS,
            "project_head": head,
            "listing_file_count_before_manifest": n_files,
            "listing_sha256_before_manifest": listing,
            "selected_positive_count": 3,
            "reviewed_negative_count": 0,
            "unreviewed_count": 245,
            "gallery_members": 0,
            "embeddings": 0,
            "png_count": 0,
            "mp4_count": 0,
        }
        write_json(tmp / "stage5d_b1e_c_manifest.json", manifest)
        n_final, listing_final = listing_sha(tmp)
        manifest["listing_file_count"] = n_final
        manifest["listing_sha256"] = listing_final
        write_json(tmp / "stage5d_b1e_c_manifest.json", manifest)

        # Budget checks.
        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")):
            raise OccurrenceFreezeError("artifact budget violated: media present")
        if list(tmp.rglob("*.pt")) or list(tmp.rglob("*.npy")):
            raise OccurrenceFreezeError("artifact budget violated: weights/embeddings")

        os.replace(str(tmp), str(final_dir))
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return {
        "final_status": FINAL_STATUS,
        "selected_positive_count": 3,
        "reviewed_negative_count": 0,
        "unreviewed_count": 245,
        "resolved_raw_track_ids": [int(r["resolved_raw_track_id"]) for r in csv_rows],
        "next": NEXT_GATE,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 5D-B1E-C external positive occurrence freeze"
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.config.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
