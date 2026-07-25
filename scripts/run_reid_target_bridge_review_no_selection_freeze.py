#!/usr/bin/env python3
"""Stage 5D-B1D — freeze bridge review with no candidate selection.

Records human no-continuation decision and external enrollment handoff.
No detection, tracking, embedding, OCR, similarity, or gallery membership.
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
from typing import Any, Mapping, Optional, Sequence

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "reid_target_bridge_review_no_selection_freeze_config_v1"
FROZEN_SEED_CODE = "SEED_CANDIDATE_07"
EXPECTED_BRIDGE_CODES = (
    "BRIDGE_CANDIDATE_01",
    "BRIDGE_CANDIDATE_02",
    "BRIDGE_CANDIDATE_03",
    "BRIDGE_CANDIDATE_04",
    "BRIDGE_CANDIDATE_05",
)
ALLOWED_CANDIDATE_DECISIONS = (
    "target_anchor_yes",
    "target_anchor_no",
    "uncertain",
    "invalid",
    "multi_person_ambiguous",
    "non_player",
)
DECISION_CSV_FIELDS = (
    "target_id",
    "bridge_candidate_code",
    "segment_id",
    "raw_track_id",
    "first_frame",
    "last_frame",
    "manual_bridge_decision",
    "selected_as_target_continuation",
    "manual_target_confirmed",
    "manual_identity_continuity_observed",
    "manual_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)


class BridgeNoSelectionError(RuntimeError):
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
        raise BridgeNoSelectionError("unexpected config schema")
    if not config.get("offline_required"):
        raise BridgeNoSelectionError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/"):
        raise BridgeNoSelectionError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise BridgeNoSelectionError("BLOCKED_STAGE5D_B1D_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise BridgeNoSelectionError("BLOCKED_STAGE5D_B1D_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    allowed = {
        "scripts/run_reid_target_bridge_review_no_selection_freeze.py",
        "configs/reid/target_bridge_review_no_selection_stage5d_target_001.yaml",
        "tests/test_reid_target_bridge_review_no_selection_freeze.py",
        "docs/setup/stage5d-target-bridge-no-selection-and-external-enrollment.md",
    }
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path not in allowed:
                raise BridgeNoSelectionError(
                    "BLOCKED_STAGE5D_B1D_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    return head


def validate_assets(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    video = project_root / config["source_video"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if video.stat().st_size != int(config["source_video"]["expected_bytes"]):
        raise BridgeNoSelectionError("BLOCKED_STAGE5D_B1D_SOURCE_VIDEO_INTEGRITY bytes")
    if sha256_file(video) != config["source_video"]["expected_sha256"]:
        raise BridgeNoSelectionError("BLOCKED_STAGE5D_B1D_SOURCE_VIDEO_INTEGRITY sha")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise BridgeNoSelectionError("BLOCKED_STAGE5D_B1D_SOURCE_VIDEO_INTEGRITY yolo_b")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise BridgeNoSelectionError("BLOCKED_STAGE5D_B1D_SOURCE_VIDEO_INTEGRITY yolo")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise BridgeNoSelectionError("BLOCKED_STAGE5D_B1D_SOURCE_VIDEO_INTEGRITY osnet")
    return {
        "path": config["source_video"]["path"],
        "sha256": config["source_video"]["expected_sha256"],
        "bytes": int(config["source_video"]["expected_bytes"]),
    }


def validate_b1c2_and_seed(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    b1c2_root = project_root / config["stage5d_b1c2_package"]["path"]
    summary = load_json(b1c2_root / "stage5d_b1c2_summary.json")
    exp = config["stage5d_b1c2_package"]
    if summary.get("final_status") != exp["expected_final_status"]:
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_B1C2_CONTRACT_MISMATCH status"
        )
    if summary.get("target_id") != "target_001":
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_B1C2_CONTRACT_MISMATCH target"
        )
    if summary.get("original_frozen_seed_code") != exp["expected_frozen_seed"]:
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_B1C2_CONTRACT_MISMATCH seed"
        )
    if summary.get("original_frozen_seed_segment_id") != exp["expected_segment_id"]:
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_B1C2_CONTRACT_MISMATCH segment"
        )
    if int(summary.get("original_frozen_seed_raw_track_id")) != int(
        exp["expected_raw_track_id"]
    ):
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_B1C2_CONTRACT_MISMATCH track"
        )
    if int(summary.get("eligible_bridge_candidate_count")) != int(
        exp["expected_eligible_count"]
    ):
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_B1C2_CONTRACT_MISMATCH eligible"
        )
    if int(summary.get("eligible_bridge_manual_selection") or 0) != 0:
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_B1C2_CONTRACT_MISMATCH manual"
        )
    for key in ("derived_anchors", "gallery_members", "prototypes", "identity_assignments"):
        if int(summary.get(key) or 0) != 0:
            raise BridgeNoSelectionError(
                f"BLOCKED_STAGE5D_B1D_B1C2_CONTRACT_MISMATCH {key}"
            )
    if summary.get("frozen_seed_embedding_used") is not False:
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_B1C2_CONTRACT_MISMATCH embedding_used"
        )

    mapping_path = (
        b1c2_root / "inventory" / "target_001_seed_to_eligible_bridge_mapping.jsonl"
    )
    rows = [
        json.loads(line)
        for line in mapping_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 5:
        raise BridgeNoSelectionError(
            f"BLOCKED_STAGE5D_B1D_B1C2_CONTRACT_MISMATCH mapping_count={len(rows)}"
        )
    codes = [str(r["bridge_candidate_code"]) for r in rows]
    if codes != list(EXPECTED_BRIDGE_CODES):
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_B1C2_CONTRACT_MISMATCH codes "
            + json.dumps(codes)
        )
    for row in rows:
        if str(row["segment_id"]) == "raw_222_full" or int(row["raw_track_id"]) == 222:
            raise BridgeNoSelectionError(
                "BLOCKED_STAGE5D_B1D_B1C2_CONTRACT_MISMATCH frozen_in_mapping"
            )
        if row.get("selected_as_target_continuation") not in ("", None):
            raise BridgeNoSelectionError(
                "BLOCKED_STAGE5D_B1D_B1C2_CONTRACT_MISMATCH prefilled_selection"
            )

    b1b_root = project_root / config["stage5d_b1b_package"]["path"]
    freeze = load_json(
        b1b_root / "seed_freeze" / "target_001_manual_seed_selection_frozen.json"
    )
    td = load_json(project_root / config["target_definition"]["path"])
    if freeze.get("selected_neutral_seed_code") != FROZEN_SEED_CODE:
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_FROZEN_SEED_CONTRACT_MISMATCH seed"
        )
    if freeze.get("target_seed_frozen") is not True:
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_FROZEN_SEED_CONTRACT_MISMATCH not_frozen"
        )
    if freeze.get("seed_is_gallery_member") is not False:
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_FROZEN_SEED_CONTRACT_MISMATCH gallery"
        )
    if td.get("target_definition_frozen") is not True:
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_FROZEN_SEED_CONTRACT_MISMATCH td"
        )
    if td.get("target_alias") != "sarı takım 5 numaralı oyuncu":
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_FROZEN_SEED_CONTRACT_MISMATCH alias"
        )

    return {
        "summary": summary,
        "mapping_rows": rows,
        "freeze": freeze,
        "target_definition": td,
        "mapping_sha256": sha256_file(mapping_path),
        "freeze_sha256": sha256_file(
            b1b_root / "seed_freeze" / "target_001_manual_seed_selection_frozen.json"
        ),
    }


def validate_human_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    human = dict(config["human_no_selection_freeze"])
    if human.get("selected_bridge_candidate_code") not in ("", None):
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_FORCE_SELECTION_FORBIDDEN selected_code"
        )
    human["selected_bridge_candidate_code"] = ""
    if human.get("manual_target_continuation_found") != "no":
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_HUMAN_DECISION_MISMATCH continuation"
        )
    if human.get("manual_review_result") != "NO_ELIGIBLE_BRIDGE_CONTINUATION_SELECTED":
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_HUMAN_DECISION_MISMATCH result"
        )
    if human.get("manual_target_confirmed") != "yes":
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_HUMAN_DECISION_MISMATCH confirmed"
        )
    if human.get("manual_human_verified_number_seen") != "yes":
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_HUMAN_DECISION_MISMATCH number"
        )
    if human.get("reviewer") != "Furkan" or human.get("final_approver") != "Furkan":
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_HUMAN_DECISION_MISMATCH reviewer"
        )
    decisions = human.get("candidate_decisions") or {}
    if set(decisions.keys()) != set(EXPECTED_BRIDGE_CODES):
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_HUMAN_DECISION_MISMATCH decision_keys"
        )
    expected = {
        "BRIDGE_CANDIDATE_01": "non_player",
        "BRIDGE_CANDIDATE_02": "target_anchor_no",
        "BRIDGE_CANDIDATE_03": "target_anchor_no",
        "BRIDGE_CANDIDATE_04": "target_anchor_no",
        "BRIDGE_CANDIDATE_05": "target_anchor_no",
    }
    for code, decision in expected.items():
        if decisions.get(code) != decision:
            raise BridgeNoSelectionError(
                f"BLOCKED_STAGE5D_B1D_HUMAN_DECISION_MISMATCH {code}"
            )
        if decision not in ALLOWED_CANDIDATE_DECISIONS:
            raise BridgeNoSelectionError(f"invalid decision vocab {decision}")
    # Force-selection prohibition: no candidate may be target_anchor_yes.
    if any(v == "target_anchor_yes" for v in decisions.values()):
        raise BridgeNoSelectionError(
            "BLOCKED_STAGE5D_B1D_FORCE_SELECTION_FORBIDDEN target_anchor_yes"
        )
    return human


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_b1d_bridge_no_selection_freeze_contract_v1",
        "target_id": "target_001",
        "original_frozen_seed_preserved": True,
        "frozen_seed_enrollment_allowed": False,
        "selected_eligible_bridge_source": None,
        "force_selection_forbidden": True,
        "current_video_eligible_source_search_closed": True,
        "next_source_type": "external_enrollment_clip",
        "derived_anchors": 0,
        "approved_anchors": 0,
        "gallery_members": 0,
        "prototypes": 0,
        "identity_assignments": 0,
        "no_new_detection": True,
        "no_new_tracking": True,
        "no_new_embedding": True,
        "no_ocr": True,
        "no_similarity": True,
        "frozen_seed_embedding_used": False,
        "unknown_identity_preserved": True,
    }


def build_external_requirements(config: Mapping[str, Any]) -> dict[str, Any]:
    req = dict(config["external_enrollment_requirements"])
    return {
        "schema_version": "reid_stage5d_b1d_external_enrollment_requirements_v1",
        "target_id": req["target_id"],
        "human_verified_jersey_number": int(req["human_verified_jersey_number"]),
        "automated_ocr_used": bool(req["automated_ocr_used"]),
        "clip_must_be_enrollment_only": bool(req["clip_must_be_enrollment_only"]),
        "clip_must_not_be_future_evaluation_input": bool(
            req["clip_must_not_be_future_evaluation_input"]
        ),
        "human_seed_selection_required": bool(req["human_seed_selection_required"]),
        "several_full_body_views_preferred": bool(
            req["several_full_body_views_preferred"]
        ),
        "manual_frozen_enrollment": bool(req["manual_frozen_enrollment"]),
        "automatic_gallery_growth": bool(req["automatic_gallery_growth"]),
        "unknown_identity_preserved": bool(req["unknown_identity_preserved"]),
        "source_reason": "NO_ELIGIBLE_BRIDGE_CONTINUATION_SELECTED",
        "current_video_eligible_source_search_closed": True,
    }


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    head = assert_git_contract(project_root, config["project_head_expected"])
    final_rel = config["output"]["final_dir"]
    assert_no_path_traversal(final_rel)
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise BridgeNoSelectionError("FAILED_STAGE5D_B1D_ATOMIC_OUTPUT final_exists")

    assets = validate_assets(project_root, config)
    upstream = validate_b1c2_and_seed(project_root, config)
    human = validate_human_freeze(config)
    reviewed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    tmp = project_root / (
        "outputs/reid/"
        "_tmp_full_stage4b_rebuild_r2_stage5d_target_001_bridge_review_no_selection_freeze_"
        f"{uuid.uuid4().hex}"
    )
    if tmp.exists():
        raise BridgeNoSelectionError("FAILED_STAGE5D_B1D_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=True)

    try:
        freeze_dir = tmp / "bridge_review_freeze"
        freeze_dir.mkdir(parents=True)

        decisions = human["candidate_decisions"]
        csv_path = freeze_dir / "target_001_bridge_review_decisions_frozen.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(DECISION_CSV_FIELDS))
            writer.writeheader()
            for row in upstream["mapping_rows"]:
                code = str(row["bridge_candidate_code"])
                writer.writerow(
                    {
                        "target_id": "target_001",
                        "bridge_candidate_code": code,
                        "segment_id": row["segment_id"],
                        "raw_track_id": row["raw_track_id"],
                        "first_frame": row["first_frame"],
                        "last_frame": row["last_frame"],
                        "manual_bridge_decision": decisions[code],
                        "selected_as_target_continuation": "no",
                        "manual_target_confirmed": human["manual_target_confirmed"],
                        "manual_identity_continuity_observed": "no",
                        "manual_notes": "",
                        "reviewer": human["reviewer"],
                        "final_approver": human["final_approver"],
                        "reviewed_at": reviewed_at,
                    }
                )

        no_selection = {
            "schema_version": "reid_stage5d_b1d_bridge_no_selection_v1",
            "target_id": "target_001",
            "selected_bridge_candidate_code": "",
            "manual_target_continuation_found": human[
                "manual_target_continuation_found"
            ],
            "manual_review_result": human["manual_review_result"],
            "manual_target_confirmed": human["manual_target_confirmed"],
            "manual_human_verified_number_seen": human[
                "manual_human_verified_number_seen"
            ],
            "reviewer": human["reviewer"],
            "final_approver": human["final_approver"],
            "manual_notes": human["manual_notes"],
            "candidate_decisions": dict(decisions),
            "bridge_review_frozen": True,
            "force_selection_forbidden": True,
            "selected_eligible_bridge_source": None,
            "original_frozen_seed_code": FROZEN_SEED_CODE,
            "original_frozen_seed_preserved": True,
            "frozen_seed_enrollment_allowed": False,
            "frozen_seed_embedding_used": False,
            "derived_anchors": 0,
            "approved_anchors": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
            "current_video_eligible_source_search_closed": True,
            "next_source_type": "external_enrollment_clip",
            "b1c2_mapping_sha256": upstream["mapping_sha256"],
            "b1b_freeze_sha256": upstream["freeze_sha256"],
            "approved_at": reviewed_at,
        }
        write_json(
            freeze_dir / "target_001_bridge_review_no_selection.json", no_selection
        )
        write_json(
            freeze_dir / "target_001_bridge_review_freeze_contract.json",
            build_contract(),
        )
        write_json(
            freeze_dir / "target_001_bridge_review_freeze_manifest.json",
            {
                "schema_version": "reid_stage5d_b1d_bridge_freeze_manifest_v1",
                "target_id": "target_001",
                "selected_bridge_candidate_code": "",
                "manual_review_result": human["manual_review_result"],
                "candidate_decision_count": len(decisions),
                "decisions_csv": "bridge_review_freeze/target_001_bridge_review_decisions_frozen.csv",
                "no_selection_json": "bridge_review_freeze/target_001_bridge_review_no_selection.json",
                "gallery_members": 0,
                "derived_anchors": 0,
            },
        )

        handoff_dir = tmp / "external_enrollment_handoff"
        handoff_dir.mkdir(parents=True)
        write_json(
            handoff_dir / "target_001_external_enrollment_requirements.json",
            build_external_requirements(config),
        )

        write_json(
            tmp / "runtime" / "runtime.json",
            {
                "schema_version": "reid_stage5d_b1d_runtime_v1",
                "started_at": reviewed_at,
                "offline_required": True,
                "network_download": 0,
                "new_detection": 0,
                "new_tracking": 0,
                "new_embedding": 0,
                "ocr": 0,
                "similarity_inference": 0,
                "frozen_seed_embedding_used": False,
            },
        )
        eff = tmp / "effective_configs"
        eff.mkdir(parents=True)
        shutil.copy2(config_path, eff / config_path.name)

        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")) or list(tmp.rglob("*.jpg")):
            raise BridgeNoSelectionError(
                "FAILED_STAGE5D_B1D_ATOMIC_OUTPUT unexpected_media"
            )

        final_status = (
            "COMPLETED_STAGE5D_B1D_NO_BRIDGE_SELECTION_EXTERNAL_ENROLLMENT_REQUIRED"
        )
        exact_next = "STAGE5D-B1E_TARGET_001_EXTERNAL_ENROLLMENT_CLIP_DESIGN_AND_INGEST"

        summary = {
            "schema_version": "reid_stage5d_b1d_summary_v1",
            "final_status": final_status,
            "project_head": head,
            "target_id": "target_001",
            "target_alias": upstream["target_definition"]["target_alias"],
            "original_frozen_seed_code": FROZEN_SEED_CODE,
            "original_frozen_seed_segment_id": "raw_222_full",
            "original_frozen_seed_raw_track_id": 222,
            "original_frozen_seed_preserved": True,
            "frozen_seed_enrollment_allowed": False,
            "frozen_seed_embedding_used": False,
            "eligible_bridge_candidate_count": 5,
            "selected_bridge_candidate_code": "",
            "selected_eligible_bridge_source": None,
            "manual_target_continuation_found": human[
                "manual_target_continuation_found"
            ],
            "manual_review_result": human["manual_review_result"],
            "candidate_decisions": dict(decisions),
            "force_selection_forbidden": True,
            "current_video_eligible_source_search_closed": True,
            "next_source_type": "external_enrollment_clip",
            "derived_anchors": 0,
            "approved_anchors": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
            "new_detection": 0,
            "new_tracking": 0,
            "new_embedding": 0,
            "ocr": 0,
            "similarity_ranking_rows": 0,
            "png_count": 0,
            "mp4_count": 0,
            "source_video": assets,
            "exact_next_gate": exact_next,
        }
        write_json(tmp / "stage5d_b1d_summary.json", summary)

        n_files, listing = listing_sha(tmp)
        write_json(
            tmp / "stage5d_b1d_manifest.json",
            {
                "schema_version": "reid_stage5d_b1d_manifest_v1",
                "final_status": final_status,
                "project_head": head,
                "listing_file_count": n_files,
                "listing_sha256": listing,
                "selected_bridge_candidate_code": "",
                "manual_review_result": human["manual_review_result"],
                "gallery_members": 0,
                "derived_anchors": 0,
                "frozen_seed_embedding_used": False,
            },
        )

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_b1d_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/target_bridge_review_no_selection_stage5d_target_001.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        summary = run(args.config.resolve(), args.project_root.resolve())
    except BridgeNoSelectionError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "selected": summary["selected_bridge_candidate_code"],
                "result": summary["manual_review_result"],
                "next": summary["exact_next_gate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
