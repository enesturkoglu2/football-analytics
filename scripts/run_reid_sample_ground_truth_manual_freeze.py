#!/usr/bin/env python3
"""Stage 5D-F2A — Freeze human sample ground-truth decisions for target_001.

Freezes exact human decisions for SAMPLE_EVAL_001...150. Does not compare
gallery↔sample embeddings, compute similarity/rank/metrics, select thresholds,
assign identities, or mutate the gallery.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "reid_sample_ground_truth_manual_freeze_config_v1"
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FINAL_STATUS = "COMPLETED_STAGE5D_F2A_TARGET_001_SAMPLE_GROUND_TRUTH_FROZEN"
NEXT_GATE = (
    "STAGE5D-F3_TARGET_001_INDEPENDENT_SAMPLE_RETRIEVAL_SCORING_AND_EVALUATION"
)
ALLOWED_DIRTY = {
    "scripts/run_reid_sample_ground_truth_manual_freeze.py",
    "configs/reid/sample_ground_truth_manual_freeze_stage5d_target_001.yaml",
    "tests/test_reid_sample_ground_truth_manual_freeze.py",
    "docs/setup/stage5d-target-sample-ground-truth-manual-review-and-freeze.md",
}

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
POSITIVE_NUMBER_YES = {
    "SAMPLE_EVAL_024",
    "SAMPLE_EVAL_069",
    "SAMPLE_EVAL_100",
    "SAMPLE_EVAL_102",
}
POSITIVE_NUMBER_UNCERTAIN = {
    "SAMPLE_EVAL_003",
    "SAMPLE_EVAL_028",
    "SAMPLE_EVAL_042",
    "SAMPLE_EVAL_046",
}
UNCERTAIN_IDS = (
    "SAMPLE_EVAL_067",
    "SAMPLE_EVAL_070",
    "SAMPLE_EVAL_078",
    "SAMPLE_EVAL_081",
    "SAMPLE_EVAL_083",
    "SAMPLE_EVAL_084",
    "SAMPLE_EVAL_086",
    "SAMPLE_EVAL_088",
)
NON_PLAYER_IDS = (
    "SAMPLE_EVAL_005",
    "SAMPLE_EVAL_006",
    "SAMPLE_EVAL_017",
    "SAMPLE_EVAL_018",
    "SAMPLE_EVAL_019",
    "SAMPLE_EVAL_055",
    "SAMPLE_EVAL_057",
)
AMBIGUOUS_IDS = (
    "SAMPLE_EVAL_050",
    "SAMPLE_EVAL_094",
    "SAMPLE_EVAL_096",
    "SAMPLE_EVAL_097",
    "SAMPLE_EVAL_098",
    "SAMPLE_EVAL_103",
    "SAMPLE_EVAL_108",
    "SAMPLE_EVAL_109",
    "SAMPLE_EVAL_112",
    "SAMPLE_EVAL_117",
    "SAMPLE_EVAL_118",
    "SAMPLE_EVAL_122",
    "SAMPLE_EVAL_123",
    "SAMPLE_EVAL_129",
    "SAMPLE_EVAL_133",
    "SAMPLE_EVAL_134",
    "SAMPLE_EVAL_138",
    "SAMPLE_EVAL_141",
    "SAMPLE_EVAL_142",
    "SAMPLE_EVAL_144",
    "SAMPLE_EVAL_146",
    "SAMPLE_EVAL_148",
    "SAMPLE_EVAL_149",
    "SAMPLE_EVAL_150",
)
AMBIGUOUS_TARGET_PRESENT = {
    "SAMPLE_EVAL_108": {
        "manual_same_target_as_target_001": "yes",
        "manual_target_dominant": "uncertain",
        "manual_identity_continuity_observed": "yes",
        "manual_human_verified_number_seen": "uncertain",
        "manual_notes": (
            "Hedef oyuncu crop içinde bulunuyor ancak beyaz oyuncuyla ağır "
            "overlap nedeniyle temiz retrieval evaluation item'ı değildir."
        ),
        "target_present": True,
    },
    "SAMPLE_EVAL_148": {
        "manual_same_target_as_target_001": "yes",
        "manual_target_dominant": "yes",
        "manual_identity_continuity_observed": "yes",
        "manual_human_verified_number_seen": "yes",
        "manual_notes": (
            "Öndeki sarı oyuncu insan tarafından target_001, sarı takım 5 numara "
            "olarak doğrulandı; fakat crop içinde başka oyuncu bulunduğu için "
            "multi-person ambiguous ve metrik dışıdır."
        ),
        "target_present": True,
    },
}
NEGATIVE_IDS = (
    "SAMPLE_EVAL_001",
    "SAMPLE_EVAL_002",
    "SAMPLE_EVAL_004",
    "SAMPLE_EVAL_007",
    "SAMPLE_EVAL_008",
    "SAMPLE_EVAL_009",
    "SAMPLE_EVAL_010",
    "SAMPLE_EVAL_011",
    "SAMPLE_EVAL_012",
    "SAMPLE_EVAL_013",
    "SAMPLE_EVAL_014",
    "SAMPLE_EVAL_015",
    "SAMPLE_EVAL_016",
    "SAMPLE_EVAL_020",
    "SAMPLE_EVAL_021",
    "SAMPLE_EVAL_022",
    "SAMPLE_EVAL_023",
    "SAMPLE_EVAL_025",
    "SAMPLE_EVAL_026",
    "SAMPLE_EVAL_027",
    "SAMPLE_EVAL_029",
    "SAMPLE_EVAL_030",
    "SAMPLE_EVAL_031",
    "SAMPLE_EVAL_032",
    "SAMPLE_EVAL_033",
    "SAMPLE_EVAL_034",
    "SAMPLE_EVAL_035",
    "SAMPLE_EVAL_036",
    "SAMPLE_EVAL_037",
    "SAMPLE_EVAL_038",
    "SAMPLE_EVAL_039",
    "SAMPLE_EVAL_040",
    "SAMPLE_EVAL_041",
    "SAMPLE_EVAL_043",
    "SAMPLE_EVAL_044",
    "SAMPLE_EVAL_045",
    "SAMPLE_EVAL_047",
    "SAMPLE_EVAL_048",
    "SAMPLE_EVAL_049",
    "SAMPLE_EVAL_051",
    "SAMPLE_EVAL_052",
    "SAMPLE_EVAL_053",
    "SAMPLE_EVAL_054",
    "SAMPLE_EVAL_056",
    "SAMPLE_EVAL_058",
    "SAMPLE_EVAL_059",
    "SAMPLE_EVAL_060",
    "SAMPLE_EVAL_061",
    "SAMPLE_EVAL_062",
    "SAMPLE_EVAL_063",
    "SAMPLE_EVAL_064",
    "SAMPLE_EVAL_065",
    "SAMPLE_EVAL_066",
    "SAMPLE_EVAL_068",
    "SAMPLE_EVAL_071",
    "SAMPLE_EVAL_072",
    "SAMPLE_EVAL_073",
    "SAMPLE_EVAL_074",
    "SAMPLE_EVAL_075",
    "SAMPLE_EVAL_076",
    "SAMPLE_EVAL_077",
    "SAMPLE_EVAL_079",
    "SAMPLE_EVAL_080",
    "SAMPLE_EVAL_082",
    "SAMPLE_EVAL_085",
    "SAMPLE_EVAL_087",
    "SAMPLE_EVAL_089",
    "SAMPLE_EVAL_090",
    "SAMPLE_EVAL_091",
    "SAMPLE_EVAL_092",
    "SAMPLE_EVAL_093",
    "SAMPLE_EVAL_095",
    "SAMPLE_EVAL_099",
    "SAMPLE_EVAL_101",
    "SAMPLE_EVAL_104",
    "SAMPLE_EVAL_105",
    "SAMPLE_EVAL_106",
    "SAMPLE_EVAL_107",
    "SAMPLE_EVAL_110",
    "SAMPLE_EVAL_111",
    "SAMPLE_EVAL_113",
    "SAMPLE_EVAL_114",
    "SAMPLE_EVAL_115",
    "SAMPLE_EVAL_116",
    "SAMPLE_EVAL_119",
    "SAMPLE_EVAL_120",
    "SAMPLE_EVAL_121",
    "SAMPLE_EVAL_124",
    "SAMPLE_EVAL_125",
    "SAMPLE_EVAL_126",
    "SAMPLE_EVAL_127",
    "SAMPLE_EVAL_128",
    "SAMPLE_EVAL_130",
    "SAMPLE_EVAL_131",
    "SAMPLE_EVAL_132",
    "SAMPLE_EVAL_135",
    "SAMPLE_EVAL_136",
    "SAMPLE_EVAL_137",
    "SAMPLE_EVAL_139",
    "SAMPLE_EVAL_140",
    "SAMPLE_EVAL_143",
    "SAMPLE_EVAL_145",
    "SAMPLE_EVAL_147",
)

DECISION_CSV_FIELDS = (
    "sample_eval_code",
    "target_id",
    "segment_id",
    "raw_track_id",
    "evaluation_component_id",
    "segment_start_frame",
    "segment_end_frame",
    "representative_frame",
    "representative_crop_path",
    "representative_crop_sha256",
    "manual_occurrence_decision",
    "manual_same_target_as_target_001",
    "manual_identity_continuity_observed",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_single_person",
    "manual_human_verified_number_seen",
    "manual_view_category",
    "manual_notes",
    "clean_positive",
    "clean_negative",
    "retrieval_metric_eligible",
    "metric_exclusion_reason",
    "target_present",
    "gallery_member",
    "enrollment_source",
    "sample_ground_truth_only",
    "component_label",
    "reviewer",
    "final_approver",
    "reviewed_at",
)


class GroundTruthFreezeError(RuntimeError):
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
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


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
        raise GroundTruthFreezeError("unexpected config schema")
    if not config.get("offline_required"):
        raise GroundTruthFreezeError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise GroundTruthFreezeError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise GroundTruthFreezeError("BLOCKED_STAGE5D_F2A_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise GroundTruthFreezeError("BLOCKED_STAGE5D_F2A_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise GroundTruthFreezeError(
                    "BLOCKED_STAGE5D_F2A_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_GIT_CONTRACT_MISMATCH message"
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
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_GIT_CONTRACT_MISMATCH external_tracked"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    if not sidecar.is_file() or not manifest.is_file() or not listing.is_file():
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man = str(load_json(manifest).get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH snapshot_sha"
        )
    if not listing.read_text(encoding="utf-8").strip():
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH listing"
        )
    return actual


def validate_decision_sets() -> None:
    all_codes = {f"SAMPLE_EVAL_{i:03d}" for i in range(1, 151)}
    groups = {
        "positive": set(POSITIVE_IDS),
        "uncertain": set(UNCERTAIN_IDS),
        "non_player": set(NON_PLAYER_IDS),
        "ambiguous": set(AMBIGUOUS_IDS),
        "negative": set(NEGATIVE_IDS),
    }
    expected_lens = {
        "positive": 8,
        "uncertain": 8,
        "non_player": 7,
        "ambiguous": 24,
        "negative": 103,
    }
    for name, ids in groups.items():
        if len(ids) != expected_lens[name]:
            raise GroundTruthFreezeError(f"decision set size {name}")
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = groups[a] & groups[b]
            if inter:
                raise GroundTruthFreezeError(f"overlap {a}/{b}: {sorted(inter)}")
    union = set().union(*groups.values())
    if union != all_codes:
        raise GroundTruthFreezeError(
            f"coverage missing={sorted(all_codes - union)} extra={sorted(union - all_codes)}"
        )
    if POSITIVE_NUMBER_YES | POSITIVE_NUMBER_UNCERTAIN != set(POSITIVE_IDS):
        raise GroundTruthFreezeError("positive number-seen partition")
    if set(AMBIGUOUS_TARGET_PRESENT) - set(AMBIGUOUS_IDS):
        raise GroundTruthFreezeError("ambiguous special not in ambiguous set")


def build_decision_for_code(code: str) -> dict[str, Any]:
    if code in POSITIVE_IDS:
        number = "yes" if code in POSITIVE_NUMBER_YES else "uncertain"
        return {
            "manual_occurrence_decision": "target_occurrence_yes",
            "manual_same_target_as_target_001": "yes",
            "manual_identity_continuity_observed": "yes",
            "manual_crop_valid": "yes",
            "manual_target_dominant": "yes",
            "manual_single_person": "yes",
            "manual_human_verified_number_seen": number,
            "manual_view_category": "",
            "manual_notes": "Human-confirmed clean target_001 positive.",
            "clean_positive": True,
            "clean_negative": False,
            "retrieval_metric_eligible": True,
            "metric_exclusion_reason": "",
            "target_present": True,
            "gallery_member": False,
            "enrollment_source": False,
            "sample_ground_truth_only": True,
        }
    if code in UNCERTAIN_IDS:
        return {
            "manual_occurrence_decision": "uncertain",
            "manual_same_target_as_target_001": "uncertain",
            "manual_identity_continuity_observed": "uncertain",
            "manual_crop_valid": "yes",
            "manual_target_dominant": "yes",
            "manual_single_person": "yes",
            "manual_human_verified_number_seen": "uncertain",
            "manual_view_category": "",
            "manual_notes": "Human uncertain; excluded from retrieval metrics.",
            "clean_positive": False,
            "clean_negative": False,
            "retrieval_metric_eligible": False,
            "metric_exclusion_reason": "uncertain",
            "target_present": None,
            "gallery_member": False,
            "enrollment_source": False,
            "sample_ground_truth_only": True,
        }
    if code in NON_PLAYER_IDS:
        return {
            "manual_occurrence_decision": "non_player",
            "manual_same_target_as_target_001": "no",
            "manual_identity_continuity_observed": "no",
            "manual_crop_valid": "yes",
            "manual_target_dominant": "no",
            "manual_single_person": "yes",
            "manual_human_verified_number_seen": "no",
            "manual_view_category": "",
            "manual_notes": "Human-confirmed non-player / non-target person.",
            "clean_positive": False,
            "clean_negative": True,
            "retrieval_metric_eligible": True,
            "metric_exclusion_reason": "",
            "target_present": False,
            "gallery_member": False,
            "enrollment_source": False,
            "sample_ground_truth_only": True,
        }
    if code in AMBIGUOUS_IDS:
        special = AMBIGUOUS_TARGET_PRESENT.get(code)
        if special:
            return {
                "manual_occurrence_decision": "multi_person_ambiguous",
                "manual_same_target_as_target_001": special[
                    "manual_same_target_as_target_001"
                ],
                "manual_identity_continuity_observed": special[
                    "manual_identity_continuity_observed"
                ],
                "manual_crop_valid": "uncertain",
                "manual_target_dominant": special["manual_target_dominant"],
                "manual_single_person": "no",
                "manual_human_verified_number_seen": special[
                    "manual_human_verified_number_seen"
                ],
                "manual_view_category": "",
                "manual_notes": special["manual_notes"],
                "clean_positive": False,
                "clean_negative": False,
                "retrieval_metric_eligible": False,
                "metric_exclusion_reason": "multi_person_ambiguous_target_present",
                "target_present": True,
                "gallery_member": False,
                "enrollment_source": False,
                "sample_ground_truth_only": True,
            }
        return {
            "manual_occurrence_decision": "multi_person_ambiguous",
            "manual_same_target_as_target_001": "no",
            "manual_identity_continuity_observed": "no",
            "manual_crop_valid": "uncertain",
            "manual_target_dominant": "uncertain",
            "manual_single_person": "no",
            "manual_human_verified_number_seen": "no",
            "manual_view_category": "",
            "manual_notes": "Multi-person / overlap; excluded from retrieval metrics.",
            "clean_positive": False,
            "clean_negative": False,
            "retrieval_metric_eligible": False,
            "metric_exclusion_reason": "multi_person_ambiguous",
            "target_present": False,
            "gallery_member": False,
            "enrollment_source": False,
            "sample_ground_truth_only": True,
        }
    if code in NEGATIVE_IDS:
        return {
            "manual_occurrence_decision": "target_occurrence_no",
            "manual_same_target_as_target_001": "no",
            "manual_identity_continuity_observed": "no",
            "manual_crop_valid": "yes",
            "manual_target_dominant": "yes",
            "manual_single_person": "yes",
            "manual_human_verified_number_seen": "no",
            "manual_view_category": "",
            "manual_notes": "Human-confirmed clean target-negative player crop.",
            "clean_positive": False,
            "clean_negative": True,
            "retrieval_metric_eligible": True,
            "metric_exclusion_reason": "",
            "target_present": False,
            "gallery_member": False,
            "enrollment_source": False,
            "sample_ground_truth_only": True,
        }
    raise GroundTruthFreezeError(f"unknown code {code}")


def label_components(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    by_comp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_comp[row["evaluation_component_id"]].append(row)

    labels: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []
    for cid, members in sorted(by_comp.items()):
        has_pos = any(m["clean_positive"] for m in members)
        has_neg = any(m["clean_negative"] for m in members)
        only_excluded = all(not m["retrieval_metric_eligible"] for m in members)
        if has_pos and has_neg:
            labels[cid] = "conflicting_component"
            conflicts.append(
                {
                    "evaluation_component_id": cid,
                    "member_codes": [m["sample_eval_code"] for m in members],
                    "positive_members": [
                        m["sample_eval_code"] for m in members if m["clean_positive"]
                    ],
                    "negative_members": [
                        m["sample_eval_code"] for m in members if m["clean_negative"]
                    ],
                }
            )
        elif has_pos:
            labels[cid] = "positive_component"
        elif has_neg:
            labels[cid] = "negative_component"
        elif only_excluded:
            labels[cid] = "excluded_component"
        else:
            labels[cid] = "excluded_component"
    dist = Counter(labels.values())
    stats = {
        "positive_component_count": int(dist.get("positive_component", 0)),
        "negative_component_count": int(dist.get("negative_component", 0)),
        "excluded_component_count": int(dist.get("excluded_component", 0)),
        "conflicting_component_count": int(dist.get("conflicting_component", 0)),
        "conflict_details": conflicts,
        "component_count": len(labels),
    }
    return labels, stats


def validate_f2(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    exp = config["stage5d_f2_package"]
    root = project_root / exp["path"]
    summary = load_json(root / "stage5d_f2_summary.json")
    contract = load_json(root / "stage5d_f2_contract.json")
    if summary.get("final_status") != exp["expected_final_status"]:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH status"
        )
    if summary.get("target_id") != TARGET_ID:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH target"
        )
    checks = {
        "scoreable_evaluation_items": exp["expected_scoreable"],
        "unscoreable_no_embedding_items": exp["expected_unscoreable"],
        "evaluation_component_count": exp["expected_components"],
        "contact_sheets": exp["expected_contact_sheets"],
        "manual_ground_truth_decisions": exp["expected_manual_decisions"],
        "similarity_rows": exp["expected_similarity_rows"],
        "gallery_members": exp["expected_gallery_members"],
    }
    for key, expected in checks.items():
        if int(summary.get(key) or 0) != int(expected):
            raise GroundTruthFreezeError(
                f"BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH {key}"
            )
    if summary.get("threshold_selected") is not False:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH threshold"
        )
    if int(summary.get("identity_assignments") or 0) != 0:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH identity"
        )
    if contract.get("scoreable_evaluation_items") != 150:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH contract"
        )
    snapshot_sha = resolve_snapshot_sha(
        Path(exp["snapshot_path"]), exp["expected_snapshot_sha256"]
    )
    mapping = load_jsonl(root / "inventory" / "target_001_sample_evaluation_mapping.jsonl")
    if len(mapping) != 150:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH mapping"
        )
    codes = [r["sample_eval_code"] for r in mapping]
    if codes != [f"SAMPLE_EVAL_{i:03d}" for i in range(1, 151)]:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH codes"
        )
    if len(set(codes)) != 150:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH unique"
        )
    tpl = root / "templates" / "target_001_sample_ground_truth_review_template.csv"
    with tpl.open(encoding="utf-8", newline="") as handle:
        blank_rows = list(csv.DictReader(handle))
    if len(blank_rows) != 150:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH template"
        )
    if any(r.get("manual_occurrence_decision") for r in blank_rows):
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F2A_REVIEW_PACKAGE_CONTRACT_MISMATCH template_filled"
        )
    return {
        "root": root,
        "summary": summary,
        "contract": contract,
        "mapping": mapping,
        "snapshot_sha256": snapshot_sha,
        "package_sha256": sha256_file(root / "stage5d_f2_manifest.json"),
    }


def validate_gallery_meta(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root / config["gallery_v1"]["path"]
    summary = load_json(root / "stage5d_b1e_f_summary.json")
    if int(summary.get("individual_gallery_members")) != int(
        config["gallery_v1"]["expected_members"]
    ):
        raise GroundTruthFreezeError("gallery members mismatch")
    return {
        "summary": summary,
        "sha256": sha256_file(root / "stage5d_b1e_f_manifest.json"),
        "npy_loaded": False,
    }


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f2a_gt_freeze_{final_dir.name}_{token}"
    if tmp.exists():
        raise GroundTruthFreezeError("FAILED_STAGE5D_F2A_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=False, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise GroundTruthFreezeError("FAILED_STAGE5D_F2A_ATOMIC_OUTPUT final_exists")
    os.rename(tmp, final_dir)


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = (project_root or _PROJECT_ROOT).resolve()
    config = load_config(config_path)
    final_rel = config["output"]["final_dir"]
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise GroundTruthFreezeError("FAILED_STAGE5D_F2A_ATOMIC_OUTPUT final_exists")

    validate_decision_sets()
    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )

    sample = project_root / config["evaluation_source"]["path"]
    ext = project_root / config["external_enrollment_source"]["path"]
    if sha256_file(sample) != config["evaluation_source"]["expected_sha256"]:
        raise GroundTruthFreezeError("sample sha mismatch")
    if sha256_file(ext) != config["external_enrollment_source"]["expected_sha256"]:
        raise GroundTruthFreezeError("external sha mismatch")
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise GroundTruthFreezeError("yolo sha mismatch")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise GroundTruthFreezeError("osnet sha mismatch")

    f2 = validate_f2(project_root, config)
    gallery = validate_gallery_meta(project_root, config)
    f1_root = project_root / config["stage5d_f1_package"]["path"]
    f1_summary = load_json(f1_root / "stage5d_f1_summary.json")
    f1_sha = sha256_file(f1_root / "stage5d_f1_manifest.json")
    target_def_path = project_root / config["target_definition"]["path"]
    target_def = load_json(target_def_path)
    if target_def.get("target_id") != TARGET_ID:
        raise GroundTruthFreezeError("target definition mismatch")
    target_def_sha = sha256_file(target_def_path)

    reviewed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    reviewer = config["human_review"]["reviewer"]
    approver = config["human_review"]["final_approver"]

    mapping_by_code = {r["sample_eval_code"]: r for r in f2["mapping"]}
    frozen_rows: list[dict[str, Any]] = []
    for i in range(1, 151):
        code = f"SAMPLE_EVAL_{i:03d}"
        base = mapping_by_code[code]
        assert_no_path_traversal(base["representative_crop_path"])
        decision = build_decision_for_code(code)
        frame_range = list(base["frame_range"])
        frozen_rows.append(
            {
                "sample_eval_code": code,
                "target_id": TARGET_ID,
                "segment_id": base["segment_id"],
                "raw_track_id": int(base["raw_track_id"]),
                "evaluation_component_id": base["evaluation_component_id"],
                "segment_start_frame": int(frame_range[0]),
                "segment_end_frame": int(frame_range[1]),
                "representative_frame": int(base["representative_frame"]),
                "representative_crop_path": base["representative_crop_path"],
                "representative_crop_sha256": base["representative_crop_sha256"],
                **decision,
                "component_label": "",  # filled after
                "reviewer": reviewer,
                "final_approver": approver,
                "reviewed_at": reviewed_at,
            }
        )

    component_labels, component_stats = label_components(frozen_rows)
    for row in frozen_rows:
        row["component_label"] = component_labels[row["evaluation_component_id"]]

    # Distribution checks
    occ = Counter(r["manual_occurrence_decision"] for r in frozen_rows)
    exp = config["expected_distribution"]
    expected_occ = {
        "target_occurrence_yes": exp["target_occurrence_yes"],
        "target_occurrence_no": exp["target_occurrence_no"],
        "non_player": exp["non_player"],
        "uncertain": exp["uncertain"],
        "multi_person_ambiguous": exp["multi_person_ambiguous"],
    }
    for key, count in expected_occ.items():
        if int(occ.get(key, 0)) != int(count):
            raise GroundTruthFreezeError(f"distribution mismatch {key}")
    if occ.get("invalid", 0) != 0:
        raise GroundTruthFreezeError("invalid must be 0")
    if len(frozen_rows) != 150:
        raise GroundTruthFreezeError("reviewed total")

    eligible = [r for r in frozen_rows if r["retrieval_metric_eligible"]]
    positives = [r for r in eligible if r["clean_positive"]]
    negatives = [r for r in eligible if r["clean_negative"]]
    excluded = [r for r in frozen_rows if not r["retrieval_metric_eligible"]]
    if len(positives) != 8 or len(negatives) != 110 or len(excluded) != 32:
        raise GroundTruthFreezeError(
            f"eligibility mismatch pos={len(positives)} neg={len(negatives)} excl={len(excluded)}"
        )
    if len(eligible) != 118:
        raise GroundTruthFreezeError("eligible total")
    if [r["sample_eval_code"] for r in positives] != list(POSITIVE_IDS):
        # positives may not be in POSITIVE_IDS order if sorted by code; compare sets
        if {r["sample_eval_code"] for r in positives} != set(POSITIVE_IDS):
            raise GroundTruthFreezeError("positive ID set mismatch")

    # Special-case assertions
    row108 = next(r for r in frozen_rows if r["sample_eval_code"] == "SAMPLE_EVAL_108")
    row148 = next(r for r in frozen_rows if r["sample_eval_code"] == "SAMPLE_EVAL_148")
    row146 = next(r for r in frozen_rows if r["sample_eval_code"] == "SAMPLE_EVAL_146")
    row150 = next(r for r in frozen_rows if r["sample_eval_code"] == "SAMPLE_EVAL_150")
    if row108["target_present"] is not True or row108["retrieval_metric_eligible"]:
        raise GroundTruthFreezeError("SAMPLE_EVAL_108 special")
    if row148["target_present"] is not True or row148["retrieval_metric_eligible"]:
        raise GroundTruthFreezeError("SAMPLE_EVAL_148 special")
    if row146["target_present"] is not False or row150["target_present"] is not False:
        raise GroundTruthFreezeError("146/150 target-absent")

    tmp = create_temp_root(final_dir)
    try:
        freeze_dir = tmp / "ground_truth_freeze"
        val_dir = tmp / "validation"
        runtime = tmp / "runtime"
        cfg_dir = tmp / "effective_configs"
        for d in (freeze_dir, val_dir, runtime, cfg_dir):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, cfg_dir / Path(config_path).name)

        decisions_csv = freeze_dir / "target_001_sample_ground_truth_decisions_frozen.csv"
        with decisions_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(DECISION_CSV_FIELDS))
            writer.writeheader()
            for row in frozen_rows:
                out = {k: row.get(k, "") for k in DECISION_CSV_FIELDS}
                # JSON-friendly booleans as True/False strings already ok via DictWriter
                for bool_key in (
                    "clean_positive",
                    "clean_negative",
                    "retrieval_metric_eligible",
                    "gallery_member",
                    "enrollment_source",
                    "sample_ground_truth_only",
                ):
                    out[bool_key] = bool(row[bool_key])
                if row["target_present"] is None:
                    out["target_present"] = ""
                else:
                    out["target_present"] = bool(row["target_present"])
                writer.writerow(out)

        eligible_csv = freeze_dir / "target_001_sample_metric_eligible_ground_truth.csv"
        with eligible_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(DECISION_CSV_FIELDS))
            writer.writeheader()
            for row in eligible:
                out = {k: row.get(k, "") for k in DECISION_CSV_FIELDS}
                for bool_key in (
                    "clean_positive",
                    "clean_negative",
                    "retrieval_metric_eligible",
                    "gallery_member",
                    "enrollment_source",
                    "sample_ground_truth_only",
                ):
                    out[bool_key] = bool(row[bool_key])
                out["target_present"] = (
                    "" if row["target_present"] is None else bool(row["target_present"])
                )
                writer.writerow(out)

        freeze_payload = {
            "schema_version": "reid_target_sample_ground_truth_freeze_v1",
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "target_definition_sha256": target_def_sha,
            "gallery_v1_manifest_sha256": gallery["sha256"],
            "gallery_members": 7,
            "f1_design_manifest_sha256": f1_sha,
            "f2_review_package_manifest_sha256": f2["package_sha256"],
            "f2_snapshot_sha256": f2["snapshot_sha256"],
            "sample_source_path": config["evaluation_source"]["path"],
            "sample_source_sha256": config["evaluation_source"]["expected_sha256"],
            "decision_lists": {
                "target_occurrence_yes": list(POSITIVE_IDS),
                "target_occurrence_no": list(NEGATIVE_IDS),
                "non_player": list(NON_PLAYER_IDS),
                "uncertain": list(UNCERTAIN_IDS),
                "multi_person_ambiguous": list(AMBIGUOUS_IDS),
                "invalid": [],
                "ambiguous_target_present": ["SAMPLE_EVAL_108", "SAMPLE_EVAL_148"],
                "ambiguous_target_absent_special": ["SAMPLE_EVAL_146", "SAMPLE_EVAL_150"],
            },
            "decision_distribution": {
                "reviewed_total": 150,
                "target_occurrence_yes": 8,
                "target_occurrence_no": 103,
                "non_player": 7,
                "uncertain": 8,
                "multi_person_ambiguous": 24,
                "invalid": 0,
            },
            "metric_eligibility_distribution": {
                "clean_positive_metric_items": 8,
                "clean_negative_metric_items": 110,
                "excluded_metric_items": 32,
                "eligible_total": 118,
            },
            "component_label_distribution": {
                "positive_component_count": component_stats["positive_component_count"],
                "negative_component_count": component_stats["negative_component_count"],
                "excluded_component_count": component_stats["excluded_component_count"],
                "conflicting_component_count": component_stats[
                    "conflicting_component_count"
                ],
                "conflict_details": component_stats["conflict_details"],
            },
            "human_reviewer": reviewer,
            "final_approver": approver,
            "reviewed_at": reviewed_at,
            "manual_decisions_frozen": True,
            "similarity_observed_before_freeze": False,
            "gallery_vectors_read": False,
            "sample_embedding_vectors_read": False,
            "ocr_used": False,
            "model_identity_used": False,
            "threshold_selected": False,
            "gallery_mutation": False,
            "automatic_gallery_growth": False,
            "similarity_rows": 0,
            "ranking_rows": 0,
            "metric_results": 0,
            "identity_assignments": 0,
            "exact_next_gate": NEXT_GATE,
        }
        write_json(freeze_dir / "target_001_sample_ground_truth_freeze.json", freeze_payload)

        contract = {
            "schema_version": "reid_stage5d_f2a_sample_ground_truth_freeze_contract_v1",
            "target_id": TARGET_ID,
            "reviewed_total": 150,
            "clean_positive_metric_items": 8,
            "clean_negative_metric_items": 110,
            "excluded_metric_items": 32,
            "manual_decisions_frozen": True,
            "similarity_observed_before_freeze": False,
            "gallery_vectors_read": False,
            "sample_embedding_vectors_read": False,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "automatic_gallery_growth": False,
            "uncertain_not_negative": True,
            "ambiguous_not_negative": True,
            "exact_next_gate": NEXT_GATE,
        }
        write_json(
            freeze_dir / "target_001_sample_ground_truth_freeze_contract.json",
            contract,
        )

        validation = {
            "schema_version": "reid_target_001_sample_ground_truth_validation_v1",
            "missing_codes": 0,
            "duplicate_codes": 0,
            "unknown_codes": 0,
            "reviewed_total": 150,
            "decision_distribution": freeze_payload["decision_distribution"],
            "metric_eligibility_distribution": freeze_payload[
                "metric_eligibility_distribution"
            ],
            "positive_ids": list(POSITIVE_IDS),
            "uncertain_ids": list(UNCERTAIN_IDS),
            "non_player_ids": list(NON_PLAYER_IDS),
            "ambiguous_ids": list(AMBIGUOUS_IDS),
            "negative_count": 103,
            "sample_eval_108_target_present_ambiguous": True,
            "sample_eval_148_target_present_ambiguous": True,
            "sample_eval_146_target_absent_ambiguous": True,
            "sample_eval_150_target_absent_ambiguous": True,
            "component_stats": component_stats,
            "all_checks_passed": True,
        }
        write_json(val_dir / "target_001_sample_ground_truth_validation.json", validation)

        write_json(
            freeze_dir / "target_001_sample_ground_truth_freeze_manifest.json",
            {
                "schema_version": "reid_target_001_sample_ground_truth_freeze_manifest_v1",
                "decisions_csv": "ground_truth_freeze/target_001_sample_ground_truth_decisions_frozen.csv",
                "eligible_csv": "ground_truth_freeze/target_001_sample_metric_eligible_ground_truth.csv",
                "reviewed_total": 150,
                "eligible_total": 118,
            },
        )

        write_json(
            runtime / "runtime.json",
            {
                "generated_at": reviewed_at,
                "project_head": head,
                "offline_required": True,
                "network_used": False,
                "osnet_loaded": False,
                "yolo_loaded": False,
                "gallery_vectors_read": False,
                "sample_embedding_vectors_read": False,
                "similarity_rows": 0,
                "ranking_rows": 0,
                "metric_results": 0,
                "threshold_selected": False,
                "identity_assignments": 0,
                "gallery_mutation": False,
                "automatic_gallery_growth": False,
                "png_written": 0,
                "mp4_written": 0,
                "npy_written": 0,
            },
        )

        summary = {
            "schema_version": "reid_stage5d_f2a_sample_ground_truth_freeze_summary_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "reviewed_total": 150,
            "target_occurrence_yes": 8,
            "target_occurrence_no": 103,
            "non_player": 7,
            "uncertain": 8,
            "multi_person_ambiguous": 24,
            "invalid": 0,
            "clean_positive_metric_items": 8,
            "clean_negative_metric_items": 110,
            "excluded_metric_items": 32,
            "eligible_total": 118,
            "positive_exact_ids": list(POSITIVE_IDS),
            "component_label_distribution": freeze_payload["component_label_distribution"],
            "similarity_rows": 0,
            "ranking_rows": 0,
            "metric_results": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_members": 7,
            "gallery_unchanged": True,
            "gallery_vectors_read": False,
            "sample_embedding_vectors_read": False,
            "ocr_used": False,
            "model_identity_used": False,
            "f2_snapshot_sha256": f2["snapshot_sha256"],
            "sample_sha256": config["evaluation_source"]["expected_sha256"],
            "external_source_sha256": config["external_enrollment_source"][
                "expected_sha256"
            ],
            "reviewer": reviewer,
            "final_approver": approver,
            "network_used": False,
            "package_environment_changed": False,
        }
        write_json(tmp / "stage5d_f2a_summary.json", summary)

        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")) or list(tmp.rglob("*.npy")):
            raise GroundTruthFreezeError("artifact budget violated")

        n_before, listing_before = listing_sha(tmp)
        manifest = {
            "schema_version": "reid_stage5d_f2a_sample_ground_truth_freeze_manifest_v1",
            "final_status": FINAL_STATUS,
            "project_head": head,
            "output_root": final_rel,
            "listing_file_count_before_manifest": n_before,
            "listing_sha256_before_manifest": listing_before,
            "f2_snapshot_sha256": f2["snapshot_sha256"],
            "reviewed_total": 150,
            "eligible_total": 118,
            "generated_at": reviewed_at,
        }
        write_json(tmp / "stage5d_f2a_manifest.json", manifest)
        n_final, listing_final = listing_sha(tmp)
        manifest["listing_file_count"] = n_final
        manifest["listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f2a_manifest.json", manifest)
        summary["output_file_count"] = n_final
        summary["output_listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f2a_summary.json", summary)

        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_f2a_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze human sample ground-truth decisions for target_001."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to sample_ground_truth_manual_freeze_stage5d_target_001.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(Path(args.config))
    except GroundTruthFreezeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"status={summary['final_status']} "
        f"reviewed={summary['reviewed_total']} "
        f"pos={summary['clean_positive_metric_items']} "
        f"neg={summary['clean_negative_metric_items']} "
        f"excl={summary['excluded_metric_items']} "
        f"next_gate={summary['exact_next_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
