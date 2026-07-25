#!/usr/bin/env python3
"""Stage 5D-F2 — Target 001 sample ground-truth review package.

Similarity-blind / label-blind human review package for 150 scoreable
sample segments. Does not compare gallery↔sample embeddings, load
embedding vectors, fill ground-truth, score, rank, or mutate the gallery.
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
CONFIG_SCHEMA = "reid_sample_ground_truth_review_config_v1"
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FINAL_STATUS = (
    "COMPLETED_STAGE5D_F2_TARGET_001_SAMPLE_GROUND_TRUTH_REVIEW_READY"
)
NEXT_GATE = "STAGE5D-F2A_TARGET_001_SAMPLE_GROUND_TRUTH_MANUAL_REVIEW_AND_FREEZE"
SCOREABLE_N = 150
UNSCOREABLE_N = 141
SHEET_COUNT = 13
ITEMS_PER_SHEET = 12
LAST_SHEET_ITEMS = 6
ALLOWED_DIRTY = {
    "scripts/run_reid_sample_ground_truth_review_package.py",
    "configs/reid/sample_ground_truth_review_stage5d_target_001.yaml",
    "tests/test_reid_sample_ground_truth_review_package.py",
    "docs/setup/stage5d-target-sample-ground-truth-review-package.md",
}
TEMPLATE_FIELDS = (
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
    "manual_human_verified_number_seen",
    "manual_view_category",
    "manual_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
OCCURRENCE_VOCAB = (
    "target_occurrence_yes",
    "target_occurrence_no",
    "uncertain",
    "invalid",
    "multi_person_ambiguous",
    "non_player",
)
VIEW_VOCAB = (
    "front",
    "rear",
    "left_side",
    "right_side",
    "front_oblique",
    "rear_oblique",
    "unknown",
)


class GroundTruthReviewError(RuntimeError):
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
        raise GroundTruthReviewError("unexpected config schema")
    if not config.get("offline_required"):
        raise GroundTruthReviewError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise GroundTruthReviewError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F2_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F2_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise GroundTruthReviewError(
                    "BLOCKED_STAGE5D_F2_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_GIT_CONTRACT_MISMATCH message"
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
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_GIT_CONTRACT_MISMATCH external_tracked"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_VALIDATION_DESIGN_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    if not sidecar.is_file() or not manifest.is_file() or not listing.is_file():
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_VALIDATION_DESIGN_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man = str(load_json(manifest).get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_VALIDATION_DESIGN_CONTRACT_MISMATCH snapshot_sha"
        )
    if not listing.read_text(encoding="utf-8").strip():
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_VALIDATION_DESIGN_CONTRACT_MISMATCH listing"
        )
    return actual


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def ranges_overlap(a: Sequence[int], b: Sequence[int]) -> bool:
    return int(a[0]) <= int(b[1]) and int(b[0]) <= int(a[1])


def assign_evaluation_components(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    n = len(rows)
    uf = UnionFind(n)
    buckets: dict[tuple[str, Any], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        keys = [
            ("raw_track_id", int(row["raw_track_id"])),
            ("documented_link_component", row.get("documented_link_component_id")),
            ("exact_crop_sha", row.get("exact_crop_sha_group")),
            ("exact_duplicate_group", row.get("exact_duplicate_embedding_group")),
            ("source_observation_component", row.get("source_observation_component")),
        ]
        if row.get("near_duplicate_cluster_id"):
            keys.append(("near_duplicate_component", row["near_duplicate_cluster_id"]))
        if row.get("leakage_group_id"):
            keys.append(("leakage_group", row["leakage_group_id"]))
        for key in keys:
            if key[1] is None or key[1] == "":
                continue
            buckets[key].append(i)
    for idxs in buckets.values():
        for j in range(1, len(idxs)):
            uf.union(idxs[0], idxs[j])

    # Overlapping temporal windows within shared track / documented component.
    by_doc: dict[str, list[int]] = defaultdict(list)
    by_track: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_doc[str(row.get("documented_link_component_id"))].append(i)
        by_track[int(row["raw_track_id"])].append(i)
    for group in list(by_doc.values()) + list(by_track.values()):
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                ia, ib = group[a], group[b]
                if ranges_overlap(rows[ia]["frame_range"], rows[ib]["frame_range"]):
                    uf.union(ia, ib)

    roots = [uf.find(i) for i in range(n)]
    groups: dict[int, list[int]] = defaultdict(list)
    for i, root in enumerate(roots):
        groups[root].append(i)

    def group_sort_key(idxs: list[int]) -> tuple[int, int, str]:
        starts = [int(rows[i]["frame_range"][0]) for i in idxs]
        ends = [int(rows[i]["frame_range"][1]) for i in idxs]
        sids = [str(rows[i]["segment_id"]) for i in idxs]
        return (min(starts), min(ends), min(sids))

    ordered_roots = sorted(groups.keys(), key=lambda r: group_sort_key(groups[r]))
    root_to_code = {
        root: f"SAMPLE_COMPONENT_{i:03d}" for i, root in enumerate(ordered_roots, start=1)
    }
    codes = [root_to_code[uf.find(i)] for i in range(n)]
    sizes = [len(groups[r]) for r in ordered_roots]
    # Membership consistency: each item exactly one component.
    if len(codes) != n or len(set(zip([r["segment_id"] for r in rows], codes))) != n:
        raise GroundTruthReviewError("component membership conflict")
    stats = {
        "component_count": len(ordered_roots),
        "component_size_min": int(min(sizes)),
        "component_size_median": float(sorted(sizes)[len(sizes) // 2]),
        "component_size_max": int(max(sizes)),
        "singleton_component_count": sum(1 for s in sizes if s == 1),
        "multi_item_component_count": sum(1 for s in sizes if s > 1),
        "largest_components": [
            {
                "evaluation_component_id": root_to_code[r],
                "size": len(groups[r]),
                "segment_ids": [rows[i]["segment_id"] for i in sorted(groups[r], key=lambda j: rows[j]["segment_id"])],
            }
            for r in sorted(ordered_roots, key=lambda x: (-len(groups[x]), group_sort_key(groups[x])))[:10]
        ],
        "duplicate_membership_conflict": 0,
        "grouping_keys_applied": [
            "segment_id",
            "raw_track_id",
            "documented_link_component",
            "exact_crop_sha",
            "exact_duplicate_group",
            "near_duplicate_component",
            "overlapping_temporal_source_window",
            "source_observation_component",
        ],
    }
    return codes, stats


def order_and_code_items(
    rows: Sequence[Mapping[str, Any]], component_codes: Sequence[str]
) -> list[dict[str, Any]]:
    indexed = list(range(len(rows)))
    component_order = {}
    for code in component_codes:
        if code not in component_order:
            component_order[code] = len(component_order)

    def sort_key(i: int) -> tuple[int, int, int, str]:
        row = rows[i]
        return (
            component_order[component_codes[i]],
            int(row["frame_range"][0]),
            int(row["frame_range"][1]),
            str(row["segment_id"]),
        )

    ordered = sorted(indexed, key=sort_key)
    out: list[dict[str, Any]] = []
    for rank, i in enumerate(ordered, start=1):
        row = dict(rows[i])
        row["sample_eval_code"] = f"SAMPLE_EVAL_{rank:03d}"
        row["evaluation_component_id"] = component_codes[i]
        row["component_label"] = None
        out.append(row)
    if [r["sample_eval_code"] for r in out] != [
        f"SAMPLE_EVAL_{i:03d}" for i in range(1, len(rows) + 1)
    ]:
        raise GroundTruthReviewError("SAMPLE_EVAL coverage mismatch")
    return out


def select_context_frames(
    obs_frames: Sequence[int],
    *,
    start_frame: int,
    end_frame: int,
    representative_frame: int,
) -> list[tuple[str, int]]:
    if not obs_frames:
        raise GroundTruthReviewError("segment has no observations")
    frames = sorted(set(int(f) for f in obs_frames))

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
    for i, text in enumerate(
        (role, f"frame={frame_index}", f"t={time_sec:.3f}s")
    ):
        y = 28 + i * 26
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA
        )
    return out


def render_item_tile(
    *,
    eval_code: str,
    crop_bgr: np.ndarray,
    contexts: Sequence[tuple[str, np.ndarray]],
    start_frame: int,
    end_frame: int,
    start_time: float,
    end_time: float,
    observation_count: int,
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    tile = np.full((tile_h, tile_w, 3), 28, dtype=np.uint8)
    cv2.putText(
        tile,
        eval_code,
        (16, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    meta = [
        f"frames {start_frame}-{end_frame}",
        f"t {start_time:.2f}s-{end_time:.2f}s",
        f"obs={observation_count}",
    ]
    y = 72
    for text in meta:
        cv2.putText(
            tile, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA
        )
        y += 22

    crop_disp = _fit_bgr(crop_bgr, tile_w - 40, int(tile_h * 0.48))
    ch, cw = crop_disp.shape[:2]
    ox = (tile_w - cw) // 2
    oy = 140
    tile[oy : oy + ch, ox : ox + cw] = crop_disp

    ctx_y = oy + ch + 18
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
    items: Sequence[Mapping[str, Any]],
    *,
    sheet_index: int,
    min_width: int,
    cols: int,
) -> np.ndarray:
    n = len(items)
    if n > ITEMS_PER_SHEET:
        raise GroundTruthReviewError("sheet exceeds 12 items")
    rows_n = int(math.ceil(n / cols)) if n else 1
    tile_w = max(900, int(math.ceil(min_width / cols)))
    tile_h = 780
    header_h = 56
    width = max(min_width, cols * tile_w)
    height = header_h + rows_n * tile_h
    sheet = np.full((height, width, 3), 14, dtype=np.uint8)
    cv2.putText(
        sheet,
        f"target_001 sample ground-truth review — sheet {sheet_index:02d}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    for i, item in enumerate(items):
        r, c = divmod(i, cols)
        tile = render_item_tile(
            eval_code=item["sample_eval_code"],
            crop_bgr=item["crop_bgr"],
            contexts=item["context_panels"],
            start_frame=int(item["frame_range"][0]),
            end_frame=int(item["frame_range"][1]),
            start_time=float(item["start_time"]),
            end_time=float(item["end_time"]),
            observation_count=int(item["observation_count"]),
            tile_w=tile_w,
            tile_h=tile_h,
        )
        y0 = header_h + r * tile_h
        x0 = c * tile_w
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return sheet


def validate_immutable_assets(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    ext = project_root / config["external_enrollment_source"]["path"]
    sample = project_root / config["evaluation_source"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if not ext.is_file() or ext.is_symlink():
        raise GroundTruthReviewError("external must be regular file")
    if ext.stat().st_size != int(config["external_enrollment_source"]["expected_bytes"]):
        raise GroundTruthReviewError("external bytes")
    if sha256_file(ext) != config["external_enrollment_source"]["expected_sha256"]:
        raise GroundTruthReviewError("external sha")
    if sample.stat().st_size != int(config["evaluation_source"]["expected_bytes"]):
        raise GroundTruthReviewError("sample bytes")
    if sha256_file(sample) != config["evaluation_source"]["expected_sha256"]:
        raise GroundTruthReviewError("sample sha")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise GroundTruthReviewError("yolo bytes")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise GroundTruthReviewError("yolo sha")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise GroundTruthReviewError("osnet sha")
    return {
        "external_sha256": config["external_enrollment_source"]["expected_sha256"],
        "sample_sha256": config["evaluation_source"]["expected_sha256"],
        "sample_path": config["evaluation_source"]["path"],
    }


def validate_f1(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    exp = config["stage5d_f1_package"]
    root = project_root / exp["path"]
    summary = load_json(root / "stage5d_f1_summary.json")
    contract = load_json(root / "stage5d_f1_contract.json")
    if summary.get("final_status") != exp["expected_final_status"]:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_VALIDATION_DESIGN_CONTRACT_MISMATCH status"
        )
    checks = {
        "gallery_members": exp["expected_gallery_members"],
        "scoreable_sample_units": exp["expected_scoreable"],
        "no_embedding_sample_units": exp["expected_no_embedding"],
        "sample_similarity_rows": exp["expected_similarity_rows"],
        "retrieval_rankings": exp["expected_ranking_rows"],
        "sample_ground_truth_decisions": exp["expected_manual_decisions"],
        "identity_assignments": exp["expected_identity_assignments"],
    }
    for key, expected in checks.items():
        if int(summary.get(key) or 0) != int(expected):
            raise GroundTruthReviewError(
                f"BLOCKED_STAGE5D_F2_VALIDATION_DESIGN_CONTRACT_MISMATCH {key}"
            )
    if summary.get("threshold_selected") is not False:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_VALIDATION_DESIGN_CONTRACT_MISMATCH threshold"
        )
    if summary.get("automatic_gallery_growth") is not False:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_VALIDATION_DESIGN_CONTRACT_MISMATCH auto_growth"
        )
    if summary.get("primary_retrieval_score") != exp["expected_primary_score"]:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_VALIDATION_DESIGN_CONTRACT_MISMATCH primary"
        )
    if list(summary.get("secondary_scores") or []) != list(exp["expected_secondary_scores"]):
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_VALIDATION_DESIGN_CONTRACT_MISMATCH secondary"
        )
    if int(summary.get("sample_embedding_dimension") or 0) != int(
        exp["expected_embedding_dimension"]
    ):
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_VALIDATION_DESIGN_CONTRACT_MISMATCH dim"
        )
    uni = load_json(root / "sample_universe" / "target_001_sample_universe_summary.json")
    if int(uni.get("total_segment_units")) != int(exp["expected_total_segments"]):
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_VALIDATION_DESIGN_CONTRACT_MISMATCH total"
        )
    snapshot_sha = resolve_snapshot_sha(
        Path(exp["snapshot_path"]), exp["expected_snapshot_sha256"]
    )
    scoreable = load_jsonl(
        root / "sample_universe" / "target_001_sample_scoreable_universe.jsonl"
    )
    no_emb = load_jsonl(
        root / "sample_universe" / "target_001_sample_no_embedding_inventory.jsonl"
    )
    if len(scoreable) != SCOREABLE_N or len(no_emb) != UNSCOREABLE_N:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_VALIDATION_DESIGN_CONTRACT_MISMATCH counts"
        )
    return {
        "root": root,
        "summary": summary,
        "contract": contract,
        "scoreable": scoreable,
        "no_embedding": no_emb,
        "snapshot_sha256": snapshot_sha,
        "design_sha256": sha256_file(root / "stage5d_f1_manifest.json"),
    }


def validate_gallery_no_npy(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = project_root / config["gallery_v1"]["path"]
    summary = load_json(root / "stage5d_b1e_f_summary.json")
    contract = load_json(root / "stage5d_b1e_f_contract.json")
    if summary.get("readiness") != config["gallery_v1"]["expected_readiness"]:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_ENROLLMENT_EVALUATION_LEAKAGE readiness"
        )
    if int(summary.get("individual_gallery_members")) != 7:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_ENROLLMENT_EVALUATION_LEAKAGE members"
        )
    if int(summary.get("centroid_count")) != 1 or int(summary.get("medoid_count")) != 1:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_ENROLLMENT_EVALUATION_LEAKAGE prototypes"
        )
    if contract.get("automatic_gallery_growth") is not False:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_ENROLLMENT_EVALUATION_LEAKAGE auto_growth"
        )
    # Do not load gallery NPY vectors.
    members = load_jsonl(root / "gallery" / "target_001_gallery_members.jsonl")
    if len(members) != 7:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_ENROLLMENT_EVALUATION_LEAKAGE member_rows"
        )
    if any(config["forbidden_sample_seed_segment"] in json.dumps(m) for m in members):
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_ENROLLMENT_EVALUATION_LEAKAGE sample_seed"
        )
    return {
        "summary": summary,
        "members": members,
        "gallery_crop_shas": {m["crop_sha256"] for m in members},
        "npy_loaded": False,
    }


def validate_independence(
    project_root: Path,
    config: Mapping[str, Any],
    gallery_info: Mapping[str, Any],
) -> dict[str, Any]:
    b1ea = project_root / config["stage5d_b1e_a_package"]["path"]
    summary_a = load_json(b1ea / "stage5d_b1e_a_summary.json")
    overlap = load_json(
        b1ea / config["stage5d_b1e_a_package"]["overlap_audit_relpath"]
    )
    if bool(overlap.get("exact_file_duplicate")):
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_ENROLLMENT_EVALUATION_LEAKAGE exact_dup"
        )
    if int(overlap.get("verified_overlapping_pair_count") or 0) != 0:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_ENROLLMENT_EVALUATION_LEAKAGE pairs"
        )
    if int(summary_a.get("verified_overlap_interval_count") or 0) != 0:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_ENROLLMENT_EVALUATION_LEAKAGE intervals"
        )
    # Crop SHA overlap without loading gallery/sample embedding vectors.
    sample_crop_roots = [
        project_root / config["sample_baseline_crops"]["path"],
        project_root / config["sample_segmented_reid"]["path"] / "crops",
    ]
    sample_shas: set[str] = set()
    for root in sample_crop_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.is_file():
                sample_shas.add(sha256_file(path))
    if gallery_info["gallery_crop_shas"] & sample_shas:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_ENROLLMENT_EVALUATION_LEAKAGE crop_sha"
        )
    return {
        "exact_file_duplicate": False,
        "verified_overlapping_frame_pairs": 0,
        "verified_overlap_intervals": 0,
        "gallery_sample_crop_sha_overlap": 0,
        "sample_seed_in_gallery": False,
    }


def resolve_crop_path(
    project_root: Path, config: Mapping[str, Any], rel: str
) -> Path:
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
    raise GroundTruthReviewError(f"crop missing: {rel}")


def parse_rep_frame(crop_id: str) -> int:
    # track_<id>_frame_<n>_rank_<r>
    parts = crop_id.split("_")
    if "frame" not in parts:
        raise GroundTruthReviewError(f"bad crop_id {crop_id}")
    idx = parts.index("frame")
    return int(parts[idx + 1])


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f2_gt_review_{final_dir.name}_{token}"
    if tmp.exists():
        raise GroundTruthReviewError("FAILED_STAGE5D_F2_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=False, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise GroundTruthReviewError("FAILED_STAGE5D_F2_ATOMIC_OUTPUT final_exists")
    os.rename(tmp, final_dir)


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = (project_root or _PROJECT_ROOT).resolve()
    config = load_config(config_path)
    final_rel = config["output"]["final_dir"]
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise GroundTruthReviewError("FAILED_STAGE5D_F2_ATOMIC_OUTPUT final_exists")

    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )
    assets = validate_immutable_assets(project_root, config)
    f1 = validate_f1(project_root, config)
    gallery_info = validate_gallery_no_npy(project_root, config)
    independence = validate_independence(project_root, config, gallery_info)

    # Embedding artifact integrity by SHA only — do not load vectors.
    emb_npz = (
        project_root
        / config["sample_segmented_reid"]["path"]
        / config["sample_segmented_reid"]["embeddings_npz"]
    )
    if not emb_npz.is_file():
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_SAMPLE_UNIVERSE_MISMATCH emb_missing"
        )
    if sha256_file(emb_npz) != config["sample_segmented_reid"][
        "expected_embeddings_sha256"
    ]:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_SAMPLE_UNIVERSE_MISMATCH emb_sha"
        )
    sample_embedding_vectors_read = False
    gallery_vectors_read = False

    scoreable = f1["scoreable"]
    for row in scoreable:
        if list(row.get("embedding_shape") or []) != [512]:
            raise GroundTruthReviewError(
                "BLOCKED_STAGE5D_F2_SAMPLE_UNIVERSE_MISMATCH emb_shape_meta"
            )
        if row.get("embedding_finite") is not True or row.get("embedding_non_zero") is not True:
            raise GroundTruthReviewError(
                "BLOCKED_STAGE5D_F2_SAMPLE_UNIVERSE_MISMATCH emb_flags"
            )

    # Crop manifests for representative bbox/frame.
    baseline_crops = {
        r["crop_id"]: r
        for r in load_jsonl(
            project_root
            / config["sample_baseline_crops"]["path"]
            / config["sample_baseline_crops"]["crop_manifest"]
        )
    }
    segment_crops = {
        r["crop_id"]: r
        for r in load_jsonl(
            project_root
            / config["sample_segmented_reid"]["path"]
            / config["sample_segmented_reid"]["segment_crop_manifest"]
        )
    }
    crop_lookup = {**baseline_crops, **segment_crops}

    track_segments = {
        r["segment_id"]: r
        for r in load_jsonl(
            project_root
            / config["sample_segment_view"]["path"]
            / config["sample_segment_view"]["track_segments"]
        )
    }

    # Enrich + validate representative crops.
    for row in scoreable:
        cid = row["representative_crop_id"]
        crop_meta = crop_lookup.get(cid)
        if crop_meta is None:
            raise GroundTruthReviewError(
                f"BLOCKED_STAGE5D_F2_SAMPLE_UNIVERSE_MISMATCH crop_meta {cid}"
            )
        crop_path = resolve_crop_path(
            project_root, config, row["representative_crop_path"]
        )
        if crop_path.is_symlink() or not crop_path.is_file():
            raise GroundTruthReviewError(
                "BLOCKED_STAGE5D_F2_SAMPLE_UNIVERSE_MISMATCH crop_file"
            )
        actual_sha = sha256_file(crop_path)
        if actual_sha != row["representative_crop_sha256"]:
            raise GroundTruthReviewError(
                "BLOCKED_STAGE5D_F2_SAMPLE_UNIVERSE_MISMATCH crop_sha"
            )
        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise GroundTruthReviewError(
                "BLOCKED_STAGE5D_F2_SAMPLE_UNIVERSE_MISMATCH crop_decode"
            )
        row["_crop_path_abs"] = str(crop_path)
        row["_crop_bgr"] = image
        row["representative_frame"] = int(
            crop_meta.get("frame_index", parse_rep_frame(cid))
        )
        row["representative_bbox_xyxy"] = [float(x) for x in crop_meta["bbox_xyxy"]]
        row["representative_crop_width"] = int(image.shape[1])
        row["representative_crop_height"] = int(image.shape[0])

    component_codes, component_stats = assign_evaluation_components(scoreable)
    ordered_items = order_and_code_items(scoreable, component_codes)

    # Segment observations for context frames.
    obs_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    obs_path = (
        project_root
        / config["sample_segment_view"]["path"]
        / config["sample_segment_view"]["segment_observations"]
    )
    needed = {r["segment_id"] for r in ordered_items}
    for row in load_jsonl(obs_path):
        sid = row.get("segment_id")
        if sid in needed:
            src = row.get("source_observation") or {}
            bbox = src.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                raise GroundTruthReviewError(
                    f"BLOCKED_STAGE5D_F2_SAMPLE_UNIVERSE_MISMATCH bbox {sid}"
                )
            obs_by_segment[sid].append(
                {
                    "frame_index": int(row["frame_index"]),
                    "bbox_xyxy": [float(x) for x in bbox],
                    "timestamp_sec": float(src.get("timestamp_sec", row["frame_index"] / 30.0)),
                }
            )
    for item in ordered_items:
        sid = item["segment_id"]
        obs = sorted(obs_by_segment.get(sid) or [], key=lambda o: o["frame_index"])
        if not obs:
            raise GroundTruthReviewError(
                f"BLOCKED_STAGE5D_F2_SAMPLE_UNIVERSE_MISMATCH no_obs {sid}"
            )
        item["_observations"] = obs
        item["start_time"] = float(obs[0]["timestamp_sec"])
        item["end_time"] = float(obs[-1]["timestamp_sec"])
        # Observation count must match segment observation rows.
        if len(obs) <= 0:
            raise GroundTruthReviewError(
                f"BLOCKED_STAGE5D_F2_SAMPLE_UNIVERSE_MISMATCH empty_obs {sid}"
            )
        item["observation_count"] = len(obs)
        item["_context_spec"] = select_context_frames(
            [o["frame_index"] for o in obs],
            start_frame=int(item["frame_range"][0]),
            end_frame=int(item["frame_range"][1]),
            representative_frame=int(item["representative_frame"]),
        )

    # Decode only required sample frames (sorted single pass).
    needed_frames = sorted(
        {
            fi
            for item in ordered_items
            for _, fi in item["_context_spec"]
        }
    )
    sample_path = project_root / config["evaluation_source"]["path"]
    cap = cv2.VideoCapture(str(sample_path))
    if not cap.isOpened():
        raise GroundTruthReviewError("sample open failed")
    frame_cache: dict[int, np.ndarray] = {}
    try:
        for fi in needed_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise GroundTruthReviewError(f"failed decode frame {fi}")
            frame_cache[fi] = frame
    finally:
        cap.release()

    fps = float(config["evaluation_source"]["expected_fps"])
    for item in ordered_items:
        bbox_by_frame = {
            o["frame_index"]: o["bbox_xyxy"] for o in item["_observations"]
        }
        panels: list[tuple[str, np.ndarray]] = []
        for role, fi in item["_context_spec"]:
            bbox = bbox_by_frame.get(fi)
            if bbox is None:
                raise GroundTruthReviewError(
                    f"context frame {fi} missing bbox for {item['segment_id']}"
                )
            annotated = annotate_context(
                frame_cache[fi],
                bbox,
                role=role,
                frame_index=fi,
                time_sec=fi / fps,
            )
            panels.append((role, annotated))
        item["crop_bgr"] = item["_crop_bgr"]
        item["context_panels"] = panels

    # Unscoreable inventory enrichment.
    unscoreable_rows: list[dict[str, Any]] = []
    for row in f1["no_embedding"]:
        sid = row["segment_id"]
        ts = track_segments.get(sid) or {}
        unscoreable_rows.append(
            {
                "segment_id": sid,
                "raw_track_id": int(row["raw_track_id"]),
                "frame_range": [
                    int(ts.get("first_observation_frame", -1)),
                    int(ts.get("last_observation_frame", -1)),
                ],
                "observation_count": int(ts.get("observation_count") or 0),
                "no_embedding_reason": row.get("no_embedding_reason"),
                "source_component": f"raw_track_{int(row['raw_track_id'])}",
                "scoreable": False,
                "ground_truth_reviewed": False,
                "automatic_negative": False,
                "retrieval_metric_eligible": False,
                "recompute_authorized": False,
                "include_in_contact_sheet": False,
            }
        )
    if len(unscoreable_rows) != UNSCOREABLE_N:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F2_SAMPLE_UNIVERSE_MISMATCH unscoreable"
        )

    target_def_path = project_root / config["target_definition"]["path"]
    target_def = load_json(target_def_path)
    if target_def.get("target_id") != TARGET_ID:
        raise GroundTruthReviewError("target definition mismatch")
    target_def_sha = sha256_file(target_def_path)

    tmp = create_temp_root(final_dir)
    try:
        inv = tmp / "inventory"
        gtr = tmp / "ground_truth_review"
        pkg = tmp / "review_packages" / "target_001_sample_ground_truth_review"
        tpl = tmp / "templates"
        runtime = tmp / "runtime"
        cfg_dir = tmp / "effective_configs"
        for d in (inv, gtr, pkg, tpl, runtime, cfg_dir):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, cfg_dir / Path(config_path).name)

        # Mapping inventory
        mapping_path = inv / "target_001_sample_evaluation_mapping.jsonl"
        with mapping_path.open("w", encoding="utf-8") as handle:
            for item in ordered_items:
                rec = {
                    "sample_eval_code": item["sample_eval_code"],
                    "target_id": TARGET_ID,
                    "segment_id": item["segment_id"],
                    "raw_track_id": int(item["raw_track_id"]),
                    "frame_range": list(item["frame_range"]),
                    "observation_count": int(item["observation_count"]),
                    "representative_frame": int(item["representative_frame"]),
                    "representative_crop_path": item["representative_crop_path"],
                    "representative_crop_sha256": item["representative_crop_sha256"],
                    "representative_bbox_xyxy": item["representative_bbox_xyxy"],
                    "source_type": item["source_type"],
                    "leakage_component_id": item.get("leakage_group_id"),
                    "evaluation_component_id": item["evaluation_component_id"],
                    "component_membership": [item["evaluation_component_id"]],
                    "documented_link_component_id": item.get(
                        "documented_link_component_id"
                    ),
                    "exact_crop_sha_group": item.get("exact_crop_sha_group"),
                    "exact_duplicate_embedding_group": item.get(
                        "exact_duplicate_embedding_group"
                    ),
                    "near_duplicate_cluster_id": item.get("near_duplicate_cluster_id"),
                    "temporal_source_component": item.get("temporal_source_component"),
                    "source_observation_component": item.get(
                        "source_observation_component"
                    ),
                    "existing_embedding_artifact_path": item.get(
                        "embedding_artifact_path"
                    ),
                    "existing_embedding_artifact_sha256": item.get(
                        "embedding_artifact_sha256"
                    ),
                    "existing_embedding_vector_sha256": item.get(
                        "embedding_vector_sha256"
                    ),
                    "embedding_shape_metadata": [512],
                    "scoreable": True,
                    "all_manual_fields_blank": True,
                    "component_label": None,
                    "manual_occurrence_decision": "",
                    "similarity_computed": False,
                    "rank_computed": False,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with (inv / "target_001_sample_unscoreable_no_embedding_inventory.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in unscoreable_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(
            inv / "target_001_sample_unscoreable_summary.json",
            {
                "schema_version": "reid_target_001_sample_unscoreable_summary_v1",
                "unscoreable_count": UNSCOREABLE_N,
                "scoreable": False,
                "automatic_negative": False,
                "contact_sheet_included": False,
                "retrieval_metric_eligible": False,
                "recompute_authorized": False,
                "reason_distribution": dict(
                    Counter(r["no_embedding_reason"] for r in unscoreable_rows)
                ),
            },
        )
        write_json(
            inv / "target_001_sample_evaluation_component_stats.json",
            {
                "schema_version": "reid_target_001_sample_evaluation_component_stats_v1",
                **component_stats,
            },
        )

        # Contact sheets
        sheet_paths: list[str] = []
        sheet_item_counts: list[int] = []
        min_width = int(config["contact_sheets"]["min_width_px"])
        cols = int(config["contact_sheets"]["grid_cols"])
        for sheet_i in range(1, SHEET_COUNT + 1):
            start = (sheet_i - 1) * ITEMS_PER_SHEET
            end = start + (
                LAST_SHEET_ITEMS if sheet_i == SHEET_COUNT else ITEMS_PER_SHEET
            )
            chunk = ordered_items[start:end]
            if sheet_i < SHEET_COUNT and len(chunk) != ITEMS_PER_SHEET:
                raise GroundTruthReviewError(f"sheet {sheet_i} size")
            if sheet_i == SHEET_COUNT and len(chunk) != LAST_SHEET_ITEMS:
                raise GroundTruthReviewError("sheet 13 size")
            sheet = render_contact_sheet(
                chunk, sheet_index=sheet_i, min_width=min_width, cols=cols
            )
            if sheet.shape[1] < min_width:
                raise GroundTruthReviewError("sheet width below minimum")
            name = f"sample_ground_truth_sheet_{sheet_i:02d}.png"
            out = pkg / name
            if not cv2.imwrite(str(out), sheet):
                raise GroundTruthReviewError(f"sheet write failed {name}")
            sheet_paths.append(
                f"review_packages/target_001_sample_ground_truth_review/{name}"
            )
            sheet_item_counts.append(len(chunk))
        if sum(sheet_item_counts) != SCOREABLE_N or len(sheet_paths) != SHEET_COUNT:
            raise GroundTruthReviewError("sheet coverage mismatch")

        # Blank template
        with (tpl / "target_001_sample_ground_truth_review_template.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TEMPLATE_FIELDS))
            writer.writeheader()
            for item in ordered_items:
                writer.writerow(
                    {
                        "sample_eval_code": item["sample_eval_code"],
                        "target_id": TARGET_ID,
                        "segment_id": item["segment_id"],
                        "raw_track_id": item["raw_track_id"],
                        "evaluation_component_id": item["evaluation_component_id"],
                        "segment_start_frame": item["frame_range"][0],
                        "segment_end_frame": item["frame_range"][1],
                        "representative_frame": item["representative_frame"],
                        "representative_crop_path": item["representative_crop_path"],
                        "representative_crop_sha256": item["representative_crop_sha256"],
                        "manual_occurrence_decision": "",
                        "manual_same_target_as_target_001": "",
                        "manual_identity_continuity_observed": "",
                        "manual_crop_valid": "",
                        "manual_target_dominant": "",
                        "manual_human_verified_number_seen": "",
                        "manual_view_category": "",
                        "manual_notes": "",
                        "reviewer": "",
                        "final_approver": "",
                        "reviewed_at": "",
                    }
                )

        review_contract = {
            "schema_version": "reid_target_001_sample_ground_truth_review_contract_v1",
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "target_definition_sha256": target_def_sha,
            "gallery_validation_design_sha256": f1["design_sha256"],
            "sample_source_sha256": assets["sample_sha256"],
            "scoreable_count": SCOREABLE_N,
            "unscoreable_count": UNSCOREABLE_N,
            "similarity_blind": True,
            "label_blind": True,
            "gallery_vectors_read": gallery_vectors_read,
            "sample_embedding_vectors_read": sample_embedding_vectors_read,
            "no_ocr": True,
            "no_rank": True,
            "no_model_identity": True,
            "human_review_required": True,
            "final_approval_required": True,
            "unreviewed_not_negative": True,
            "uncertain_excluded_from_metrics": True,
            "invalid_excluded_from_metrics": True,
            "multi_person_excluded_from_metrics": True,
            "positives_require_target_occurrence_yes": True,
            "negatives_require_target_occurrence_no_or_non_player": True,
            "decisions_require_separate_freeze_gate": True,
            "threshold_selected": False,
            "gallery_mutation": False,
            "identity_assignment": False,
            "automatic_gallery_growth": False,
            "allowed_occurrence_decisions": list(OCCURRENCE_VOCAB),
            "allowed_view_categories": list(VIEW_VOCAB),
            "allowed_tri_state": ["yes", "no", "uncertain"],
            "contact_sheet_count": SHEET_COUNT,
            "exact_next_gate": NEXT_GATE,
        }
        write_json(
            gtr / "target_001_sample_ground_truth_review_contract.json",
            review_contract,
        )
        review_manifest = {
            "schema_version": "reid_target_001_sample_ground_truth_review_manifest_v1",
            "scoreable_count": SCOREABLE_N,
            "unscoreable_count": UNSCOREABLE_N,
            "contact_sheets": sheet_paths,
            "sheet_item_counts": sheet_item_counts,
            "template": "templates/target_001_sample_ground_truth_review_template.csv",
            "mapping": "inventory/target_001_sample_evaluation_mapping.jsonl",
            "manual_decisions": 0,
            "similarity_rows": 0,
            "ranking_rows": 0,
        }
        write_json(
            gtr / "target_001_sample_ground_truth_review_manifest.json",
            review_manifest,
        )

        write_json(
            runtime / "runtime.json",
            {
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "project_head": head,
                "offline_required": True,
                "network_used": False,
                "sample_mp4_context_decode_frames": len(needed_frames),
                "sample_mp4_inference": False,
                "osnet_loaded": False,
                "yolo_loaded": False,
                "gallery_vectors_read": False,
                "sample_embedding_vectors_read": False,
                "similarity_rows": 0,
                "ranking_rows": 0,
                "manual_ground_truth_decisions": 0,
                "threshold_selected": False,
                "identity_assignments": 0,
                "automatic_gallery_growth": False,
                "gallery_members": 7,
                "source_crop_copies": 0,
                "contact_sheet_png_count": SHEET_COUNT,
                "mp4_written": 0,
            },
        )

        contract = {
            "schema_version": "reid_stage5d_f2_sample_ground_truth_review_contract_v1",
            "target_id": TARGET_ID,
            "scoreable_evaluation_items": SCOREABLE_N,
            "unscoreable_no_embedding_items": UNSCOREABLE_N,
            "contact_sheets": SHEET_COUNT,
            "manual_ground_truth_decisions": 0,
            "similarity_rows": 0,
            "retrieval_ranking_rows": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_members": 7,
            "gallery_unchanged": True,
            "automatic_gallery_growth": False,
            "gallery_vectors_read": False,
            "sample_embedding_vectors_read": False,
            "exact_next_gate": NEXT_GATE,
        }
        write_json(tmp / "stage5d_f2_contract.json", contract)

        summary = {
            "schema_version": "reid_stage5d_f2_sample_ground_truth_review_summary_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "scoreable_evaluation_items": SCOREABLE_N,
            "unscoreable_no_embedding_items": UNSCOREABLE_N,
            "contact_sheets": SHEET_COUNT,
            "sheet_item_counts": sheet_item_counts,
            "sample_eval_code_first": "SAMPLE_EVAL_001",
            "sample_eval_code_last": "SAMPLE_EVAL_150",
            "evaluation_component_count": component_stats["component_count"],
            "component_size_min": component_stats["component_size_min"],
            "component_size_median": component_stats["component_size_median"],
            "component_size_max": component_stats["component_size_max"],
            "singleton_component_count": component_stats["singleton_component_count"],
            "multi_item_component_count": component_stats["multi_item_component_count"],
            "manual_ground_truth_decisions": 0,
            "similarity_rows": 0,
            "retrieval_ranking_rows": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_members": 7,
            "gallery_unchanged": True,
            "automatic_gallery_growth": False,
            "gallery_vectors_read": False,
            "sample_embedding_vectors_read": False,
            "ocr_used": False,
            "model_identity_used": False,
            "existing_stage5c_labels_prefilled": False,
            "f1_snapshot_sha256": f1["snapshot_sha256"],
            "sample_sha256": assets["sample_sha256"],
            "external_source_sha256": assets["external_sha256"],
            "independence": independence,
            "network_used": False,
            "package_environment_changed": False,
        }
        write_json(tmp / "stage5d_f2_summary.json", summary)

        pngs = list(tmp.rglob("*.png"))
        if len(pngs) != SHEET_COUNT:
            raise GroundTruthReviewError(f"png budget {len(pngs)}")
        if list(tmp.rglob("*.mp4")) or list(tmp.rglob("*.npy")):
            raise GroundTruthReviewError("mp4/npy forbidden in package")

        n_before, listing_before = listing_sha(tmp)
        manifest = {
            "schema_version": "reid_stage5d_f2_sample_ground_truth_review_manifest_v1",
            "final_status": FINAL_STATUS,
            "project_head": head,
            "output_root": final_rel,
            "listing_file_count_before_manifest": n_before,
            "listing_sha256_before_manifest": listing_before,
            "contact_sheets": sheet_paths,
            "f1_snapshot_sha256": f1["snapshot_sha256"],
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        write_json(tmp / "stage5d_f2_manifest.json", manifest)
        n_final, listing_final = listing_sha(tmp)
        manifest["listing_file_count"] = n_final
        manifest["listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f2_manifest.json", manifest)
        summary["output_file_count"] = n_final
        summary["output_listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f2_summary.json", summary)

        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_f2_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build similarity-blind sample ground-truth review package for target_001."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to sample_ground_truth_review_stage5d_target_001.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(Path(args.config))
    except GroundTruthReviewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"status={summary['final_status']} "
        f"scoreable={summary['scoreable_evaluation_items']} "
        f"sheets={summary['contact_sheets']} "
        f"decisions={summary['manual_ground_truth_decisions']} "
        f"next_gate={summary['exact_next_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
