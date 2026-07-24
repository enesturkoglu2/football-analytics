#!/usr/bin/env python3
"""Stage 5C clean label-blind discovery/holdout split (rebuild r2).

Builds discovery/holdout primary+reserve batches from the 474-item clean
review universe without using manual labels, OCR predictions, historical
C3D membership, or threshold selection. Pre-registers evaluation rules only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

CONFIG_SCHEMA = "reid_jersey_clean_split_config_v1"
PREREG_SCHEMA = "reid_jersey_clean_discovery_holdout_preregistration_v1"
CONTRACT_SCHEMA = "reid_jersey_clean_split_contract_v1"
SUMMARY_SCHEMA = "reid_jersey_clean_split_summary_v1"
MANIFEST_SCHEMA = "reid_jersey_clean_split_manifest_v1"
SOURCE_SCHEMA = "reid_jersey_clean_split_source_universe_v1"
SELECTED_SCHEMA = "reid_jersey_clean_split_selected_item_v1"
LEAKAGE_SCHEMA = "reid_jersey_clean_split_leakage_group_v1"

STRATA = (
    "high_signal_candidate",
    "mid_signal_candidate",
    "safety_candidate",
)
STRATUM_SHORT = {
    "high_signal_candidate": "high",
    "mid_signal_candidate": "mid",
    "safety_candidate": "safety",
}
SOURCE_TYPE_REUSED = "reused_baseline_selected_crop"
SOURCE_TYPE_RECOMPUTED = "recomputed_manual_segment"
SOURCE_SHORT = {
    "reused": SOURCE_TYPE_REUSED,
    "recomputed": SOURCE_TYPE_RECOMPUTED,
}
BATCH_ORDER = (
    "discovery_primary",
    "discovery_reserve",
    "holdout_primary",
    "holdout_reserve",
)

PROHIBITED_SELECTION_PREFIXES = ("manual_",)
PROHIBITED_SELECTION_SUBSTRINGS = (
    "parseq",
    "mmocr",
    "prediction",
    "ocr_",
    "accepted_prediction",
    "exact_match",
    "wrong_number",
    "previous_discovery",
    "previous_holdout",
    "pilot_member",
    "jersey_number",
)

MANUAL_BLANK_FIELDS = (
    "manual_crop_valid",
    "manual_number_visible",
    "manual_number_readable",
    "manual_jersey_number",
    "manual_notes",
    "reviewer",
    "reviewed_at",
)

TEMPLATE_COLUMNS = (
    "split_item_id",
    "batch_order",
    "review_item_id",
    "source_crop_path",
    "source_crop_sha256",
    "contact_sheet_path",
    "contact_sheet_page",
    "tile_index",
    "manual_crop_valid",
    "manual_number_visible",
    "manual_number_readable",
    "manual_jersey_number",
    "manual_notes",
    "reviewer",
    "reviewed_at",
)

ALLOWED_TERNARY = frozenset({"yes", "no", "uncertain"})
JERSEY_RE = re.compile(r"^[0-9]{1,2}$")

QUALITY_FIELDS = (
    "laplacian_variance",
    "tenengrad_mean",
    "local_contrast",
    "entropy",
    "edge_density",
)


class CleanSplitError(RuntimeError):
    """Contract failure for clean label-blind split design."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise CleanSplitError(f"{path}:{line_no} not an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise CleanSplitError("config must be a mapping")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise CleanSplitError("unexpected config schema_version")
    if not config.get("offline_required"):
        raise CleanSplitError("offline_required must be true")
    if not config.get("threshold_selection_forbidden"):
        raise CleanSplitError("threshold_selection_forbidden must be true")
    return config


def _finite_number(value: Any, label: str) -> float:
    if value is None:
        raise CleanSplitError(f"missing {label}")
    number = float(value)
    if not math.isfinite(number):
        raise CleanSplitError(f"non-finite {label}: {value}")
    return number


def assert_no_prohibited_selection_fields(row: Mapping[str, Any]) -> None:
    for key, value in row.items():
        key_l = str(key).lower()
        if any(key_l.startswith(p) for p in PROHIBITED_SELECTION_PREFIXES):
            if value not in (None, "", False):
                raise CleanSplitError(f"prohibited selection field populated: {key}")
        if any(s in key_l for s in PROHIBITED_SELECTION_SUBSTRINGS):
            if value not in (None, "", False) and key_l not in {
                "ocr_predictions_used",
                "old_manual_labels_recovered",
                "new_manual_labels_generated",
            }:
                # Flags that are explicitly false are allowed.
                if not (isinstance(value, bool) and value is False):
                    if key_l.startswith("ocr_") and value is None:
                        continue
                    if "jersey_number" in key_l and value is None:
                        continue
                    if value is not None and value is not False:
                        # Allow boolean false provenance flags only.
                        pass


def _git_output(args: Sequence[str], *, project_root: Path) -> str:
    env = os.environ.copy()
    git_dir = project_root / ".git"
    if git_dir.exists():
        env["GIT_DIR"] = str(git_dir)
        env["GIT_WORK_TREE"] = str(project_root)
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(project_root),
        env=env,
    )
    return result.stdout.strip()


def project_git_head(project_root: Path) -> str:
    return _git_output(["rev-parse", "HEAD"], project_root=project_root)


def project_git_status(project_root: Path) -> str:
    return _git_output(["status", "--porcelain"], project_root=project_root)


# ---------------------------------------------------------------------------
# Feature scoring / strata
# ---------------------------------------------------------------------------


def _field_value(row: Mapping[str, Any], name: str) -> Optional[float]:
    if name == "source_crop_area":
        w = row.get("source_crop_width")
        h = row.get("source_crop_height")
        if w is None or h is None:
            return None
        return float(int(w) * int(h))
    value = row.get(name)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise CleanSplitError(f"NaN/Inf in field {name}")
    return number


def rank_to_unit(
    values: Sequence[Optional[float]],
    *,
    higher_better: bool,
    tie_ids: Sequence[str],
) -> list[Optional[float]]:
    indexed = [
        (i, values[i])
        for i in range(len(values))
        if values[i] is not None and math.isfinite(float(values[i]))
    ]
    out: list[Optional[float]] = [None] * len(values)
    if not indexed:
        return out
    indexed.sort(
        key=lambda t: (
            -float(t[1]) if higher_better else float(t[1]),
            tie_ids[t[0]],
        )
    )
    m = len(indexed)
    for rank0, (i, _) in enumerate(indexed):
        rank = rank0 + 1
        out[i] = 1.0 if m == 1 else float(m - rank) / float(m - 1)
    return out


def score_universe(
    rows: Sequence[Mapping[str, Any]],
    feature_contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(rows)
    ids = [str(r["review_item_id"]) for r in rows]
    pos_specs = list(feature_contract["positive_raw_fields"])
    neg_specs = list(feature_contract["negative_raw_fields"])
    penalties = list(feature_contract.get("missingness_penalties") or [])

    units: dict[str, list[Optional[float]]] = {}
    missing_counts: dict[str, int] = {}
    feature_log: list[dict[str, Any]] = []

    for spec in pos_specs + neg_specs:
        name = str(spec["name"])
        higher = str(spec["direction"]) == "higher_better"
        raw_vals: list[Optional[float]] = []
        for row in rows:
            raw_vals.append(_field_value(row, name))
        miss = sum(1 for v in raw_vals if v is None)
        missing_counts[name] = miss
        if str(spec.get("missing_policy")) == "reject_row" and miss:
            raise CleanSplitError(f"required field missing: {name} count={miss}")
        units[name] = rank_to_unit(raw_vals, higher_better=higher, tie_ids=ids)
        feature_log.append(
            {
                "exact_source_artifact": "clean_review_items.jsonl",
                "exact_field_name": name,
                "semantic_purpose": "composite_visibility_quality_score",
                "missing_count": miss,
                "direction": spec["direction"],
                "normalization": "in_universe_rank_to_unit",
                "weight": float(spec["weight"]),
                "label_independence_justification": feature_contract[
                    "label_independence_justification"
                ].strip()
                if isinstance(feature_contract.get("label_independence_justification"), str)
                else feature_contract.get("label_independence_justification"),
            }
        )

    for pen in penalties:
        src = str(pen["source_field"])
        miss = sum(
            1
            for row in rows
            if (
                (not bool(row.get(src)))
                if pen.get("when_false_or_missing")
                else False
            )
        )
        missing_counts[str(pen["name"])] = miss
        feature_log.append(
            {
                "exact_source_artifact": "clean_review_items.jsonl",
                "exact_field_name": src,
                "semantic_purpose": "explicit_missingness_penalty",
                "missing_count": miss,
                "direction": "missing_is_worse",
                "normalization": "fixed_penalty_subtraction",
                "weight": float(pen["penalty"]),
                "label_independence_justification": pen.get("justification"),
            }
        )

    scored: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        weighted = 0.0
        weight_sum = 0.0
        used: dict[str, Optional[float]] = {}
        for spec in pos_specs + neg_specs:
            name = str(spec["name"])
            unit = units[name][i]
            used[name] = unit
            if unit is None:
                continue
            w = float(spec["weight"])
            weighted += w * float(unit)
            weight_sum += w
        if weight_sum <= 0:
            raise CleanSplitError(
                f"no usable features for {row.get('review_item_id')}"
            )
        score = weighted / weight_sum
        applied_penalties: dict[str, float] = {}
        for pen in penalties:
            src = str(pen["source_field"])
            if pen.get("when_false_or_missing") and not bool(row.get(src)):
                p = float(pen["penalty"])
                score -= p
                applied_penalties[str(pen["name"])] = p
        item = dict(row)
        item["composite_score"] = float(score)
        item["feature_units"] = used
        item["applied_missingness_penalties"] = applied_penalties
        item["source_crop_area"] = int(
            int(row["source_crop_width"]) * int(row["source_crop_height"])
        )
        scored.append(item)

    meta = {
        "normalization": feature_contract.get("normalization"),
        "missing_counts": missing_counts,
        "feature_contract_log": feature_log,
        "expected_missing_quality_kit_joins": 54,
    }
    return scored, meta


def assign_quantile_strata(
    scored_rows: Sequence[Mapping[str, Any]],
    strata_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bands = strata_contract["bands"]
    ordered = sorted(
        scored_rows,
        key=lambda row: (-float(row["composite_score"]), str(row["review_item_id"])),
    )
    n = len(ordered)
    out: list[dict[str, Any]] = []
    for index, row in enumerate(ordered):
        q = (float(index) / float(n)) if n else 0.0
        item = dict(row)
        assigned = None
        for name in STRATA:
            band = bands[name]
            start = float(band["quantile_start"])
            end = float(band["quantile_end"])
            if start <= q < end or (end >= 1.0 and q >= start):
                assigned = name
                break
        if assigned is None:
            raise CleanSplitError(f"failed to assign stratum at quantile {q}")
        item["signal_stratum"] = assigned
        item["stratum"] = assigned
        item["stratum_is_ground_truth"] = False
        item["stratum_quantile"] = q
        out.append(item)
    semantics = strata_contract.get("semantics") or {}
    if semantics.get("high_signal_candidate_is_readable_positive") is not False:
        raise CleanSplitError("strata semantics must deny readable-positive meaning")
    if semantics.get("safety_candidate_is_negative_ground_truth") is not False:
        raise CleanSplitError("strata semantics must deny negative GT meaning")
    return out


def assign_timeline_bins(
    rows: Sequence[Mapping[str, Any]], *, n_bins: int = 4
) -> list[dict[str, Any]]:
    frames = sorted(int(r["frame_index"]) for r in rows)
    if not frames:
        return [dict(r) for r in rows]
    # Quartile edges by rank of frame values.
    unique_sorted = sorted(set(frames))
    out: list[dict[str, Any]] = []
    for row in rows:
        frame = int(row["frame_index"])
        # Rank among unique frames.
        rank = unique_sorted.index(frame) if frame in unique_sorted else 0
        m = len(unique_sorted)
        q = float(rank) / float(m) if m else 0.0
        bin_id = min(n_bins - 1, int(math.floor(q * n_bins)))
        item = dict(row)
        item["timeline_bin"] = bin_id
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Near-duplicate + leakage grouping
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
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


def dhash_bits(image_bgr: np.ndarray, *, resize: tuple[int, int]) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise CleanSplitError("empty image for dHash")
    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    w, h = int(resize[0]), int(resize[1])
    small = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    # Classic dHash: compare adjacent horizontal pixels -> (h*(w-1)) bits.
    if w < 2:
        raise CleanSplitError("dHash resize width must be >= 2")
    diff = small[:, 1:].astype(np.float64) > small[:, :-1].astype(np.float64)
    bits = diff.astype(np.uint8).reshape(-1)
    return bits


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape:
        raise CleanSplitError("hash shape mismatch")
    return int(np.sum(a.astype(np.uint8) != b.astype(np.uint8)))


def extract_hash_region(
    image: np.ndarray, row: Mapping[str, Any]
) -> tuple[np.ndarray, str]:
    if bool(row.get("roi_valid")):
        x0 = int(row["number_roi_x"])
        y0 = int(row["number_roi_y"])
        x1 = int(row["number_roi_x_max"])
        y1 = int(row["number_roi_y_max"])
        if x1 > x0 and y1 > y0:
            region = image[y0:y1, x0:x1]
            if region.size:
                return region, "number_roi"
    return image, "full_crop"


def compute_near_duplicates(
    rows: Sequence[Mapping[str, Any]],
    *,
    nd_contract: Mapping[str, Any],
    project_root: Path,
) -> tuple[dict[str, np.ndarray], list[tuple[str, str, int]], dict[str, Any]]:
    resize = tuple(nd_contract["resize"])
    max_dist = int(nd_contract["near_duplicate_max_distance"])
    hashes: dict[str, np.ndarray] = {}
    for row in rows:
        rid = str(row["review_item_id"])
        path = Path(str(row["source_crop_path"]))
        _validate_crop_path(path, project_root=project_root)
        expected = str(row["source_crop_sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise CleanSplitError(f"SHA mismatch for {rid}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise CleanSplitError(f"missing/unreadable crop: {path}")
        region, _ = extract_hash_region(image, row)
        bits = dhash_bits(region, resize=resize)
        if int(bits.size) != int(nd_contract["bit_length"]):
            raise CleanSplitError(
                f"dHash bit length {bits.size} != {nd_contract['bit_length']}"
            )
        hashes[rid] = bits

    ids = sorted(hashes.keys())
    edges: list[tuple[str, str, int]] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dist = hamming_distance(hashes[ids[i]], hashes[ids[j]])
            if dist <= max_dist:
                edges.append((ids[i], ids[j], dist))
    stats = {
        "algorithm": nd_contract["algorithm"],
        "input_region": nd_contract["input_region"],
        "resize": list(resize),
        "bit_length": int(nd_contract["bit_length"]),
        "distance_metric": nd_contract["distance_metric"],
        "near_duplicate_max_distance": max_dist,
        "near_duplicate_edge_count": len(edges),
    }
    return hashes, edges, stats


def _validate_crop_path(path: Path, *, project_root: Path) -> None:
    text = str(path)
    if ".." in Path(text).parts:
        raise CleanSplitError(f"path traversal rejected: {path}")
    resolved = path.expanduser().resolve()
    root = project_root.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CleanSplitError(f"crop path outside project: {path}") from exc
    if not resolved.is_file():
        raise CleanSplitError(f"missing crop: {path}")


def build_leakage_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    near_dup_edges: Sequence[tuple[str, str, int]],
    documented_components: Sequence[Sequence[int]],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str], dict[str, Any]]:
    uf = UnionFind()
    nd_uf = UnionFind()
    id_to_row = {str(r["review_item_id"]): r for r in rows}
    for rid in id_to_row:
        uf.add(rid)
        nd_uf.add(rid)

    key_buckets: dict[tuple[str, Any], list[str]] = defaultdict(list)
    for row in rows:
        rid = str(row["review_item_id"])
        key_buckets[("review_item_id", rid)].append(rid)
        key_buckets[("crop_id", str(row["crop_id"]))].append(rid)
        key_buckets[("source_crop_sha256", str(row["source_crop_sha256"]))].append(rid)
        key_buckets[("segment_id", str(row["segment_id"]))].append(rid)
        key_buckets[("raw_track_id", int(row["raw_track_id"]))].append(rid)
        key_buckets[
            ("documented_global_candidate_id", int(row["documented_global_candidate_id"]))
        ].append(rid)

    for _key, members in key_buckets.items():
        for other in members[1:]:
            uf.union(members[0], other)

    # Exact SHA also seeds near-dup clusters.
    sha_buckets: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        sha_buckets[str(row["source_crop_sha256"])].append(str(row["review_item_id"]))
    exact_dup_clusters = 0
    for members in sha_buckets.values():
        if len(members) > 1:
            exact_dup_clusters += 1
        for other in members[1:]:
            nd_uf.union(members[0], other)
            uf.union(members[0], other)

    track_to_ids: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        track_to_ids[int(row["raw_track_id"])].append(str(row["review_item_id"]))
    for pair in documented_components:
        a, b = int(pair[0]), int(pair[1])
        if track_to_ids.get(a) and track_to_ids.get(b):
            seed = track_to_ids[a][0]
            for rid in track_to_ids[a] + track_to_ids[b]:
                uf.union(seed, rid)

    cross_track_nd = 0
    for a, b, _dist in near_dup_edges:
        uf.union(a, b)
        nd_uf.union(a, b)
        if int(id_to_row[a]["raw_track_id"]) != int(id_to_row[b]["raw_track_id"]):
            cross_track_nd += 1

    groups_map: dict[str, list[str]] = defaultdict(list)
    for rid in id_to_row:
        groups_map[uf.find(rid)].append(rid)
    nd_map: dict[str, list[str]] = defaultdict(list)
    for rid in id_to_row:
        nd_map[nd_uf.find(rid)].append(rid)

    # Stable leakage_group_id / near_duplicate_cluster_id.
    group_id_by_root: dict[str, str] = {}
    for idx, root in enumerate(sorted(groups_map.keys()), start=1):
        group_id_by_root[root] = f"leakage_{idx:04d}"
    nd_id_by_root: dict[str, str] = {}
    for idx, root in enumerate(sorted(nd_map.keys()), start=1):
        nd_id_by_root[root] = f"nd_cluster_{idx:04d}"

    item_to_group: dict[str, str] = {}
    item_to_nd: dict[str, str] = {}
    group_rows: list[dict[str, Any]] = []
    for root, members in sorted(groups_map.items(), key=lambda kv: group_id_by_root[kv[0]]):
        gid = group_id_by_root[root]
        members_sorted = sorted(members)
        for rid in members_sorted:
            item_to_group[rid] = gid
        tracks = sorted({int(id_to_row[m]["raw_track_id"]) for m in members_sorted})
        segs = sorted({str(id_to_row[m]["segment_id"]) for m in members_sorted})
        globals_ = sorted(
            {
                int(id_to_row[m]["documented_global_candidate_id"])
                for m in members_sorted
            }
        )
        group_rows.append(
            {
                "schema_version": LEAKAGE_SCHEMA,
                "leakage_group_id": gid,
                "member_review_item_ids": members_sorted,
                "member_count": len(members_sorted),
                "raw_track_ids": tracks,
                "segment_ids": segs,
                "documented_global_candidate_ids": globals_,
            }
        )

    for root, members in nd_map.items():
        nid = nd_id_by_root[root]
        for rid in members:
            item_to_nd[rid] = nid

    sizes = [g["member_count"] for g in group_rows]
    nd_sizes = [len(v) for v in nd_map.values()]
    audit = {
        "leakage_group_count": len(group_rows),
        "leakage_max_group_size": max(sizes) if sizes else 0,
        "leakage_singleton_count": sum(1 for s in sizes if s == 1),
        "exact_duplicate_sha_cluster_count": exact_dup_clusters,
        "near_duplicate_edge_count": len(near_dup_edges),
        "near_duplicate_connected_cluster_count": sum(
            1 for s in nd_sizes if s > 1
        ),
        "near_duplicate_maximum_cluster_size": max(nd_sizes) if nd_sizes else 0,
        "cross_raw_track_near_duplicate_count": cross_track_nd,
        "documented_components_enforced": [list(p) for p in documented_components],
        "transitive_closure": True,
    }
    return group_rows, item_to_group, item_to_nd, audit


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


class BatchDiversityState:
    def __init__(
        self,
        *,
        max_per_segment: int,
        max_per_raw_track: int,
        min_frame_gap: int,
    ) -> None:
        self.max_per_segment = max_per_segment
        self.max_per_raw_track = max_per_raw_track
        self.min_frame_gap = min_frame_gap
        self.segment_counts: dict[str, int] = defaultdict(int)
        self.track_counts: dict[int, int] = defaultdict(int)
        self.track_frames: dict[int, list[int]] = defaultdict(list)
        self.source_type_counts: dict[str, int] = defaultdict(int)
        self.timeline_counts: dict[int, int] = defaultdict(int)
        self.selected_ids: set[str] = set()
        self.selected_shas: set[str] = set()
        self.selected_paths: set[str] = set()

    def accepts(self, row: Mapping[str, Any]) -> bool:
        rid = str(row["review_item_id"])
        if rid in self.selected_ids:
            return False
        sha = str(row["source_crop_sha256"])
        if sha in self.selected_shas:
            return False
        path = str(row["source_crop_path"])
        if path in self.selected_paths:
            return False
        segment = str(row["segment_id"])
        if self.segment_counts[segment] >= self.max_per_segment:
            return False
        track = int(row["raw_track_id"])
        if self.track_counts[track] >= self.max_per_raw_track:
            return False
        frame = int(row["frame_index"])
        for prior in self.track_frames[track]:
            if abs(frame - prior) < self.min_frame_gap:
                return False
        return True

    def add(self, row: Mapping[str, Any]) -> None:
        self.selected_ids.add(str(row["review_item_id"]))
        self.selected_shas.add(str(row["source_crop_sha256"]))
        self.selected_paths.add(str(row["source_crop_path"]))
        self.segment_counts[str(row["segment_id"])] += 1
        track = int(row["raw_track_id"])
        self.track_counts[track] += 1
        self.track_frames[track].append(int(row["frame_index"]))
        self.source_type_counts[str(row["crop_source_type"])] += 1
        self.timeline_counts[int(row["timeline_bin"])] += 1


def max_selectable_in_group(
    items: Sequence[Mapping[str, Any]],
    *,
    max_per_segment: int,
    max_per_raw_track: int,
    min_frame_gap: int,
) -> int:
    state = BatchDiversityState(
        max_per_segment=max_per_segment,
        max_per_raw_track=max_per_raw_track,
        min_frame_gap=min_frame_gap,
    )
    got = 0
    for row in sorted(items, key=lambda r: str(r["review_item_id"])):
        if state.accepts(row):
            state.add(row)
            got += 1
    return got


def audit_recomputed_capacity(
    stratified_rows: Sequence[Mapping[str, Any]],
    *,
    item_to_group: Mapping[str, str],
    allocation: Mapping[str, Any],
    source_recomputed_key: str = SOURCE_TYPE_RECOMPUTED,
) -> dict[str, Any]:
    """Dynamic recomputed capacity under leakage + diversity (no labels)."""
    max_seg = int(allocation["max_selected_per_segment"])
    max_track = int(allocation["max_selected_per_raw_track"])
    min_gap = int(allocation["min_frame_distance_same_raw_track"])
    recomp = [
        r
        for r in stratified_rows
        if str(r["crop_source_type"]) == source_recomputed_key
    ]
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in recomp:
        by_group[str(item_to_group[str(row["review_item_id"])])].append(row)

    group_reports = []
    global_max = 0
    for gid in sorted(by_group.keys()):
        items = list(by_group[gid])
        cap = max_selectable_in_group(
            items,
            max_per_segment=max_seg,
            max_per_raw_track=max_track,
            min_frame_gap=min_gap,
        )
        global_max += cap
        group_reports.append(
            {
                "leakage_group_id": gid,
                "item_count": len(items),
                "raw_track_count": len({int(r["raw_track_id"]) for r in items}),
                "strata_availability": dict(
                    Counter(str(r["signal_stratum"]) for r in items)
                ),
                "timeline_availability": dict(
                    Counter(int(r["timeline_bin"]) for r in items)
                ),
                "maximum_selectable_item_count": cap,
            }
        )

    return {
        "recomputed_item_count": len(recomp),
        "recomputed_raw_track_count": len({int(r["raw_track_id"]) for r in recomp}),
        "recomputed_leakage_group_count": len(by_group),
        "recomputed_count_by_stratum": dict(
            Counter(str(r["signal_stratum"]) for r in recomp)
        ),
        "groups": group_reports,
        "global_maximum_selectable_recomputed_count": global_max,
    }


def hamilton_proportional_reference(
    total_recomputed: int,
    *,
    batch_totals: Mapping[str, int],
    batch_order: Sequence[str] = BATCH_ORDER,
) -> dict[str, int]:
    """Largest-remainder (Hamilton) allocation of recomputed across batches."""
    sizes = [int(batch_totals[b]) for b in batch_order]
    grand = sum(sizes)
    if grand <= 0:
        raise CleanSplitError("empty batch totals")
    raw = [total_recomputed * s / grand for s in sizes]
    floors = [int(math.floor(x)) for x in raw]
    rem = total_recomputed - sum(floors)
    order = sorted(
        range(len(batch_order)),
        key=lambda i: (-(raw[i] - floors[i]), i),
    )
    for i in order:
        if rem <= 0:
            break
        floors[i] += 1
        rem -= 1
    return {batch_order[i]: floors[i] for i in range(len(batch_order))}


def objective_sort_key(
    vector: tuple[int, int, int, int],
    *,
    batch_totals: Sequence[int],
    universe_ratio: float,
) -> tuple[Any, ...]:
    dp, dr, hp, hr = vector
    totals = list(batch_totals)
    ratios = [vector[i] / totals[i] for i in range(4)]
    absdev = sum(abs(r - universe_ratio) for r in ratios)
    disc = (dp + dr) / (totals[0] + totals[1])
    hold = (hp + hr) / (totals[2] + totals[3])
    return (
        -sum(vector),
        absdev,
        abs(disc - hold),
        -(dp + hp),
        (-dp, -dr, -hp, -hr),
    )


def enumerate_quota_vectors(
    *,
    total: int,
    hard_minima: Mapping[str, int],
    batch_totals: Mapping[str, int],
    batch_order: Sequence[str] = BATCH_ORDER,
) -> list[tuple[int, int, int, int]]:
    mins = [int(hard_minima[b]) for b in batch_order]
    maxs = [int(batch_totals[b]) for b in batch_order]
    if sum(mins) > total:
        return []
    out: list[tuple[int, int, int, int]] = []
    for dp in range(mins[0], min(maxs[0], total - sum(mins[1:])) + 1):
        for dr in range(mins[1], min(maxs[1], total - dp - sum(mins[2:])) + 1):
            for hp in range(mins[2], min(maxs[2], total - dp - dr - mins[3]) + 1):
                hr = total - dp - dr - hp
                if mins[3] <= hr <= maxs[3]:
                    out.append((dp, dr, hp, hr))
    return out


def allocate_batches(
    stratified_rows: Sequence[Mapping[str, Any]],
    *,
    item_to_group: Mapping[str, str],
    batch_quotas: Mapping[str, Mapping[str, int]],
    allocation: Mapping[str, Any],
    source_type_keys: Optional[Mapping[str, str]] = None,
    prefer_small_leakage_groups: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Capacity-balanced allocation: recomp-first (safety→mid→high), then reused."""
    max_seg = int(allocation["max_selected_per_segment"])
    max_track = int(allocation["max_selected_per_raw_track"])
    min_gap = int(allocation["min_frame_distance_same_raw_track"])
    if not bool(allocation.get("no_silent_stratum_backfill", True)):
        raise CleanSplitError("no_silent_stratum_backfill must be true")
    if not bool(allocation.get("no_silent_source_type_backfill", True)):
        raise CleanSplitError("no_silent_source_type_backfill must be true")
    keys = dict(source_type_keys or SOURCE_SHORT)
    reused_key = str(keys.get("reused", SOURCE_TYPE_REUSED))
    recomp_key = str(keys.get("recomputed", SOURCE_TYPE_RECOMPUTED))
    stratum_pref = (
        "safety_candidate",
        "mid_signal_candidate",
        "high_signal_candidate",
    )

    group_sizes: dict[str, int] = defaultdict(int)
    for row in stratified_rows:
        if str(row["crop_source_type"]) == recomp_key:
            group_sizes[str(item_to_group[str(row["review_item_id"])])] += 1

    group_owner: dict[str, str] = {}
    globally_selected: set[str] = set()
    results: dict[str, list[dict[str, Any]]] = {b: [] for b in BATCH_ORDER}

    for batch in BATCH_ORDER:
        quotas = batch_quotas[batch]
        if "reused" not in quotas or "recomputed" not in quotas:
            raise CleanSplitError(f"missing source-type quotas for {batch}")
        if int(quotas["reused"]) + int(quotas["recomputed"]) != int(quotas["total"]):
            raise CleanSplitError(f"source-type quotas must sum to total for {batch}")
        rem_stratum = {s: int(quotas[STRATUM_SHORT[s]]) for s in STRATA}
        rem_recomp = int(quotas["recomputed"])
        rem_reused = int(quotas["reused"])
        state = BatchDiversityState(
            max_per_segment=max_seg,
            max_per_raw_track=max_track,
            min_frame_gap=min_gap,
        )
        batch_order_n = 0

        while rem_recomp > 0:
            picked = None
            for stratum in stratum_pref:
                if rem_stratum[stratum] <= 0:
                    continue
                cands = [
                    r
                    for r in stratified_rows
                    if str(r["review_item_id"]) not in globally_selected
                    and str(r["signal_stratum"]) == stratum
                    and str(r["crop_source_type"]) == recomp_key
                ]

                def sort_key(r: Mapping[str, Any]) -> tuple[Any, ...]:
                    gid = str(item_to_group[str(r["review_item_id"])])
                    size_key = group_sizes[gid] if prefer_small_leakage_groups else 0
                    return (
                        size_key,
                        state.timeline_counts[int(r["timeline_bin"])],
                        state.track_counts[int(r["raw_track_id"])],
                        str(r["review_item_id"]),
                    )

                cands.sort(key=sort_key)
                for row in cands:
                    gid = str(item_to_group[str(row["review_item_id"])])
                    owner = group_owner.get(gid)
                    if owner is not None and owner != batch:
                        continue
                    if not state.accepts(row):
                        continue
                    picked = row
                    break
                if picked is not None:
                    break
            if picked is None:
                raise CleanSplitError(
                    f"BLOCKED_SOURCE_TYPE_AND_STRATUM_CAPACITY batch={batch} "
                    f"phase=recomputed remaining={rem_recomp} "
                    f"stratum_remaining={rem_stratum}"
                )
            batch_order_n += 1
            item = dict(picked)
            item["batch"] = batch
            item["batch_order"] = batch_order_n
            item["split_item_id"] = f"{batch}_{batch_order_n:03d}"
            item["stratum"] = str(picked["signal_stratum"])
            item["leakage_group_id"] = item_to_group[str(item["review_item_id"])]
            item["schema_version"] = SELECTED_SCHEMA
            item["annotation_status"] = "unreviewed"
            results[batch].append(item)
            state.add(item)
            globally_selected.add(str(item["review_item_id"]))
            group_owner[str(item["leakage_group_id"])] = batch
            rem_recomp -= 1
            rem_stratum[str(picked["signal_stratum"])] -= 1

        while sum(rem_stratum.values()) > 0:
            picked = None
            for stratum in STRATA:
                if rem_stratum[stratum] <= 0:
                    continue
                if rem_reused <= 0:
                    break
                cands = [
                    r
                    for r in stratified_rows
                    if str(r["review_item_id"]) not in globally_selected
                    and str(r["signal_stratum"]) == stratum
                    and str(r["crop_source_type"]) == reused_key
                ]

                def sort_key_r(r: Mapping[str, Any]) -> tuple[Any, ...]:
                    return (
                        state.timeline_counts[int(r["timeline_bin"])],
                        state.track_counts[int(r["raw_track_id"])],
                        str(r["review_item_id"]),
                    )

                cands.sort(key=sort_key_r)
                for row in cands:
                    gid = str(item_to_group[str(row["review_item_id"])])
                    owner = group_owner.get(gid)
                    if owner is not None and owner != batch:
                        continue
                    if not state.accepts(row):
                        continue
                    picked = row
                    break
                if picked is not None:
                    break
            if picked is None:
                raise CleanSplitError(
                    f"BLOCKED_SOURCE_TYPE_AND_STRATUM_CAPACITY batch={batch} "
                    f"phase=reused remaining={rem_reused} "
                    f"stratum_remaining={rem_stratum}"
                )
            batch_order_n += 1
            item = dict(picked)
            item["batch"] = batch
            item["batch_order"] = batch_order_n
            item["split_item_id"] = f"{batch}_{batch_order_n:03d}"
            item["stratum"] = str(picked["signal_stratum"])
            item["leakage_group_id"] = item_to_group[str(item["review_item_id"])]
            item["schema_version"] = SELECTED_SCHEMA
            item["annotation_status"] = "unreviewed"
            results[batch].append(item)
            state.add(item)
            globally_selected.add(str(item["review_item_id"]))
            group_owner[str(item["leakage_group_id"])] = batch
            rem_reused -= 1
            rem_stratum[str(picked["signal_stratum"])] -= 1

        if rem_reused != 0 or rem_recomp != 0:
            raise CleanSplitError(
                f"BLOCKED_SOURCE_TYPE_AND_STRATUM_CAPACITY batch={batch} "
                f"leftover reused={rem_reused} recomp={rem_recomp}"
            )
        if len(results[batch]) != int(quotas["total"]):
            raise CleanSplitError(
                f"batch size mismatch {batch}: {len(results[batch])}"
            )
        for stratum in STRATA:
            got = sum(1 for r in results[batch] if r["stratum"] == stratum)
            need = int(quotas[STRATUM_SHORT[stratum]])
            if got != need:
                raise CleanSplitError(
                    f"BLOCKED_SOURCE_TYPE_AND_STRATUM_CAPACITY batch={batch} "
                    f"stratum={stratum} need={need} got={got}"
                )
        got_reused = sum(
            1 for r in results[batch] if str(r["crop_source_type"]) == reused_key
        )
        got_recomp = sum(
            1 for r in results[batch] if str(r["crop_source_type"]) == recomp_key
        )
        if got_reused != int(quotas["reused"]) or got_recomp != int(quotas["recomputed"]):
            raise CleanSplitError(
                f"BLOCKED_SOURCE_TYPE_AND_STRATUM_CAPACITY batch={batch} "
                f"reused={got_reused}/{quotas['reused']} "
                f"recomputed={got_recomp}/{quotas['recomputed']}"
            )

    return results


def search_capacity_balanced_quotas(
    stratified_rows: Sequence[Mapping[str, Any]],
    *,
    item_to_group: Mapping[str, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Exhaustive descending-total search for feasible recomputed quota vector."""
    import time

    search_cfg = config["capacity_search"]
    allocation = config["allocation"]
    batches_cfg = config["batches"]
    hard_minima = search_cfg["hard_minima"]
    batch_totals = {b: int(batches_cfg[b]["total"]) for b in BATCH_ORDER}
    uni_num, uni_den = search_cfg["universe_recomputed_ratio"]
    universe_ratio = float(uni_num) / float(uni_den)
    timeout = float(search_cfg["search"]["timeout_seconds"])
    max_tried = int(search_cfg["search"]["max_vectors_tried"])
    prefer_small = bool(search_cfg["allocator"].get("prefer_small_leakage_groups", True))

    capacity = audit_recomputed_capacity(
        stratified_rows,
        item_to_group=item_to_group,
        allocation=allocation,
    )
    global_max = int(capacity["global_maximum_selectable_recomputed_count"])
    minima_sum = sum(int(hard_minima[b]) for b in BATCH_ORDER)
    proportional_ref = hamilton_proportional_reference(
        global_max, batch_totals=batch_totals
    )
    proportional_tuple = tuple(proportional_ref[b] for b in BATCH_ORDER)

    started = time.monotonic()
    tried = 0
    failures: list[dict[str, Any]] = []
    selected = None
    selected_batches = None

    for total in range(global_max, minima_sum - 1, -1):
        vectors = enumerate_quota_vectors(
            total=total,
            hard_minima=hard_minima,
            batch_totals=batch_totals,
        )
        vectors.sort(
            key=lambda v: (
                # Exact Hamilton/proportional reference preferred when feasible.
                0 if v == proportional_tuple else 1,
                objective_sort_key(
                    v,
                    batch_totals=[batch_totals[b] for b in BATCH_ORDER],
                    universe_ratio=universe_ratio,
                ),
            )
        )
        for vector in vectors:
            if tried >= max_tried:
                raise CleanSplitError("BLOCKED_CAPACITY_SEARCH_INCONCLUSIVE")
            if time.monotonic() - started > timeout:
                raise CleanSplitError("BLOCKED_CAPACITY_SEARCH_INCONCLUSIVE")
            tried += 1
            batch_quotas = {}
            for i, batch in enumerate(BATCH_ORDER):
                q = dict(batches_cfg[batch])
                q["recomputed"] = int(vector[i])
                q["reused"] = int(batch_totals[batch]) - int(vector[i])
                q["total"] = int(batch_totals[batch])
                batch_quotas[batch] = q
            try:
                batches = allocate_batches(
                    stratified_rows,
                    item_to_group=item_to_group,
                    batch_quotas=batch_quotas,
                    allocation=allocation,
                    source_type_keys=config.get("source_type_keys"),
                    prefer_small_leakage_groups=prefer_small,
                )
                assert_zero_overlap(batches)
            except CleanSplitError as exc:
                failures.append(
                    {
                        "vector": list(vector),
                        "total": total,
                        "error": str(exc)[:240],
                    }
                )
                continue
            selected = vector
            selected_batches = batches
            break
        if selected is not None:
            break

    if selected is None:
        raise CleanSplitError("BLOCKED_CAPACITY_BALANCED_SPLIT_INFEASIBLE")

    scores = objective_sort_key(
        selected,
        batch_totals=[batch_totals[b] for b in BATCH_ORDER],
        universe_ratio=universe_ratio,
    )
    selected_map = {BATCH_ORDER[i]: int(selected[i]) for i in range(4)}
    # Exhaustive descending totals + objective-ordered vectors: first success is
    # the maximum feasible recomputed total under the fixed allocator contract.
    max_feasible_total = sum(selected)
    selected_is_max: bool | str = True
    catalog_incomplete = False

    feasible_at_total: list[list[int]] = [list(selected)]
    catalog_vectors = enumerate_quota_vectors(
        total=max_feasible_total,
        hard_minima=hard_minima,
        batch_totals=batch_totals,
    )
    catalog_vectors.sort(
        key=lambda v: (
            0 if v == proportional_tuple else 1,
            objective_sort_key(
                v,
                batch_totals=[batch_totals[b] for b in BATCH_ORDER],
                universe_ratio=universe_ratio,
            ),
        )
    )
    for vector in catalog_vectors:
        if list(vector) == list(selected):
            continue
        if tried >= max_tried or time.monotonic() - started > timeout:
            catalog_incomplete = True
            break
        tried += 1
        batch_quotas_c = {}
        for i, batch in enumerate(BATCH_ORDER):
            q = dict(batches_cfg[batch])
            q["recomputed"] = int(vector[i])
            q["reused"] = int(batch_totals[batch]) - int(vector[i])
            q["total"] = int(batch_totals[batch])
            batch_quotas_c[batch] = q
        try:
            allocate_batches(
                stratified_rows,
                item_to_group=item_to_group,
                batch_quotas=batch_quotas_c,
                allocation=allocation,
                source_type_keys=config.get("source_type_keys"),
                prefer_small_leakage_groups=prefer_small,
            )
        except CleanSplitError:
            failures.append(
                {
                    "vector": list(vector),
                    "total": max_feasible_total,
                    "error": "catalog_infeasible",
                }
            )
            continue
        feasible_at_total.append(list(vector))

    batch_ranges = {
        BATCH_ORDER[i]: {
            "min": min(v[i] for v in feasible_at_total),
            "max": max(v[i] for v in feasible_at_total),
        }
        for i in range(4)
    }
    discovery_vals = [v[0] + v[1] for v in feasible_at_total]
    holdout_vals = [v[2] + v[3] for v in feasible_at_total]
    primary_vals = [v[0] + v[2] for v in feasible_at_total]
    reserve_vals = [v[1] + v[3] for v in feasible_at_total]
    feasible_ranges = {
        "at_maximum_feasible_total": max_feasible_total,
        "per_batch": batch_ranges,
        "discovery_aggregate": {"min": min(discovery_vals), "max": max(discovery_vals)},
        "holdout_aggregate": {"min": min(holdout_vals), "max": max(holdout_vals)},
        "primary_aggregate": {"min": min(primary_vals), "max": max(primary_vals)},
        "reserve_aggregate": {"min": min(reserve_vals), "max": max(reserve_vals)},
        "feasible_vector_count_at_total": len(feasible_at_total),
        "feasible_vectors_at_total": feasible_at_total,
        "catalog_incomplete": catalog_incomplete,
    }
    if catalog_incomplete:
        selected_is_max = "unresolved"
        feasible_ranges["search_limitation"] = (
            "maximum total proven by descending search; "
            "full vector catalog at that total incomplete"
        )

    return {
        "capacity_audit": capacity,
        "proportional_reference_vector": proportional_ref,
        "proportional_reference_tuple": list(proportional_tuple),
        "selected_quota_vector": selected_map,
        "selected_quota_tuple": list(selected),
        "selected_total_recomputed": max_feasible_total,
        "global_capacity_upper_bound": global_max,
        "maximum_feasible_recomputed_count": max_feasible_total,
        "selected_recomputed_is_maximum_feasible": selected_is_max,
        "feasible_ranges": feasible_ranges,
        "objective_scores": {
            "total_recomputed_negated_for_sort": scores[0],
            "sum_abs_ratio_deviation": scores[1],
            "discovery_holdout_aggregate_gap": scores[2],
            "primary_recomputed_negated_for_sort": scores[3],
            "lexicographic_descending": list(scores[4]),
        },
        "search_stats": {
            "vectors_tried": tried,
            "failures_recorded": len(failures),
            "elapsed_seconds": time.monotonic() - started,
            "timeout_seconds": timeout,
            "search_inconclusive": False,
        },
        "batch_quotas": {
            b: {
                **dict(batches_cfg[b]),
                "recomputed": selected_map[b],
                "reused": batch_totals[b] - selected_map[b],
                "total": batch_totals[b],
            }
            for b in BATCH_ORDER
        },
        "batches": selected_batches,
        "failure_samples": failures[:20],
    }



def assert_zero_overlap(batches: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    keys = (
        "review_item_id",
        "crop_id",
        "source_crop_sha256",
        "source_crop_path",
        "segment_id",
        "raw_track_id",
        "documented_global_candidate_id",
        "leakage_group_id",
        "near_duplicate_cluster_id",
    )
    seen: dict[str, dict[Any, str]] = {k: {} for k in keys}
    overlaps: list[dict[str, Any]] = []
    for batch, rows in batches.items():
        for row in rows:
            for key in keys:
                value = row[key]
                if key == "raw_track_id" or key == "documented_global_candidate_id":
                    value = int(value)
                else:
                    value = str(value)
                prior = seen[key].get(value)
                if prior is not None and prior != batch:
                    overlaps.append(
                        {"key": key, "value": value, "batch_a": prior, "batch_b": batch}
                    )
                else:
                    seen[key][value] = batch
    if overlaps:
        raise CleanSplitError(
            f"BLOCKED_SPLIT_OVERLAP_OR_LEAKAGE count={len(overlaps)} first={overlaps[0]}"
        )
    return {"overlap_count": 0, "checked_keys": list(keys)}


# ---------------------------------------------------------------------------
# Review artifacts
# ---------------------------------------------------------------------------


def _fit_display(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / max(w, 1), box_h / max(h, 1))
    nw = max(1, int(math.floor(w * scale)))
    nh = max(1, int(math.floor(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_NEAREST)


def render_contact_sheet(
    items: Sequence[Mapping[str, Any]],
    images: Mapping[str, np.ndarray],
    *,
    max_items: int,
    columns: int,
) -> np.ndarray:
    if len(items) > max_items:
        raise CleanSplitError("contact sheet exceeds max items")
    rows_n = int(math.ceil(len(items) / columns)) if items else 1
    tile_w, tile_h = 280, 360
    sheet = np.full((rows_n * tile_h, columns * tile_w, 3), 24, dtype=np.uint8)
    for index, item in enumerate(items):
        r, c = divmod(index, columns)
        image = images[str(item["crop_id"])]
        tile = np.full((tile_h, tile_w, 3), 40, dtype=np.uint8)
        meta = [
            str(item["split_item_id"])[:42],
            f"order={item['batch_order']}",
        ]
        y = 16
        for text in meta:
            cv2.putText(
                tile,
                text,
                (6, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            y += 16
        display = _fit_display(image, tile_w - 12, 170)
        dh, dw = display.shape[:2]
        ox = (tile_w - dw) // 2
        oy = 60
        tile[oy : oy + dh, ox : ox + dw] = display
        cw = int(item["source_crop_width"])
        scale = dw / max(cw, 1)
        rx0 = ox + int(round(int(item["number_roi_x"]) * scale))
        ry0 = oy + int(round(int(item["number_roi_y"]) * scale))
        rx1 = ox + int(round(int(item["number_roi_x_max"]) * scale))
        ry1 = oy + int(round(int(item["number_roi_y_max"]) * scale))
        cv2.rectangle(
            tile,
            (rx0, ry0),
            (max(rx0 + 1, rx1 - 1), max(ry0 + 1, ry1 - 1)),
            (0, 220, 255),
            1,
        )
        roi = image[
            int(item["number_roi_y"]) : int(item["number_roi_y_max"]),
            int(item["number_roi_x"]) : int(item["number_roi_x_max"]),
        ]
        if roi.size:
            zoom = _fit_display(roi, tile_w - 12, 100)
            zh, zw = zoom.shape[:2]
            zx = (tile_w - zw) // 2
            zy = 250
            tile[zy : zy + zh, zx : zx + zw] = zoom
        y0 = r * tile_h
        x0 = c * tile_w
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return sheet


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise CleanSplitError(f"failed to write png: {path}")


def build_review_instructions_tr() -> str:
    return """# Stage 5C Clean Split — İnceleme Talimatları (rebuild r2, capacity-balanced)

Bu paket, 474-item label-blind clean universe üzerinden önceden sabitlenmiş
discovery/holdout batch'leridir (`canonical_split_generation=r2_capacity_balanced`).
Model, OCR ve manuel label bilgisi seçimde kullanılmamıştır. Kapasiteye duyarlı
source-type kota seçimi annotation/prediction/threshold öncesinde yapılmıştır.

## İnceleme sırası

1. Önce **discovery primary** incelenir.
2. Discovery reserve yalnız primary annotation minimumlarını karşılamazsa,
   önceden belirlenen `batch_order` ile açılır.
3. Discovery annotation freeze ve (sonraki kapıda) discovery inference
   bittikten sonra candidate gate pre-register edilir.
4. **Holdout primary**, discovery gate seçilmeden açılmaz / incelenmez.
5. Holdout reserve yalnız primary label minimumları yetmezse açılır.

## Contact sheet

- Contact sheet'te model prediction / confidence yoktur.
- Stratum, score, source type, kit, raw track, segment, documented global ID
  ve near-duplicate bilgisi reviewer'a gösterilmez.
- Stratum ground-truth değildir.

## Etiket kuralları

- Numara net değilse tahmin edilmez.
- `manual_number_readable=yes` yalnız tam sayının güvenle okunabildiği durumda.
- `09` ve `9` exact olarak farklıdır (görüntüdeki leading zero korunur).
- `readable=yes` → jersey number zorunlu (`0`–`99`, 1–2 digit exact string).
- `readable=no` veya `uncertain` → jersey number boş.

## Bilimsel sınırlar

- Holdout sonucuna göre gate değiştirilmez.
- Bu çalışma deployment accuracy benchmark değildir.
- Historical-never-seen claim yapılmayacaktır.
- Eski C3E threshold/cut kullanılmayacaktır.
"""


def validate_annotation_values(
    *,
    manual_crop_valid: str,
    manual_number_visible: str,
    manual_number_readable: str,
    manual_jersey_number: str,
) -> list[str]:
    errors: list[str] = []
    for field, value in (
        ("manual_crop_valid", manual_crop_valid),
        ("manual_number_visible", manual_number_visible),
        ("manual_number_readable", manual_number_readable),
    ):
        if value and value not in ALLOWED_TERNARY:
            errors.append(f"{field} invalid")
    jersey = (manual_jersey_number or "").strip()
    readable = manual_number_readable
    if readable == "yes":
        if not jersey:
            errors.append("readable=yes requires jersey_number")
        elif not JERSEY_RE.fullmatch(jersey):
            errors.append("jersey_number must be 1-2 digit exact string 0-99")
        if manual_number_visible != "yes":
            errors.append("readable=yes requires visible=yes")
    elif readable in ("no", "uncertain"):
        if jersey:
            errors.append("readable!=yes requires blank jersey_number")
    elif jersey:
        errors.append("jersey_number requires readable=yes")
    return errors


def build_annotation_template_rows(
    selected: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in selected:
        row = {col: "" for col in TEMPLATE_COLUMNS}
        row.update(
            {
                "split_item_id": str(item["split_item_id"]),
                "batch_order": str(item["batch_order"]),
                "review_item_id": str(item["review_item_id"]),
                "source_crop_path": str(item["source_crop_path"]),
                "source_crop_sha256": str(item["source_crop_sha256"]),
                "contact_sheet_path": str(item.get("contact_sheet_path") or ""),
                "contact_sheet_page": str(item.get("contact_sheet_page") or ""),
                "tile_index": str(item.get("tile_index") or ""),
            }
        )
        rows.append(row)
    return rows


def write_annotation_template(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TEMPLATE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in TEMPLATE_COLUMNS})


def _artifact_meta(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".jsonl":
        meta["row_count"] = sum(1 for line in path.open() if line.strip())
    elif path.suffix == ".csv":
        meta["row_count"] = max(0, sum(1 for _ in path.open()) - 1)
    elif path.suffix == ".png":
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise CleanSplitError(f"cannot read png meta: {path}")
        meta["height"] = int(image.shape[0])
        meta["width"] = int(image.shape[1])
    return meta


def derive_discovery_candidate_gate_rule() -> dict[str, Any]:
    return {
        "threshold_selected": False,
        "deployment_threshold_selected": False,
        "rule_name": "lowest_confidence_cut_zero_error_acceptance",
        "eligible_requires": {
            "accepted_wrong_positive": 0,
            "accepted_negative": 0,
            "accepted_exact_gt": 0,
            "sentinel_type": None,
        },
        "primary_selection": "lowest_confidence_cut_among_eligible",
        "ties": ["higher_accepted_exact", "cut_numeric_ascending"],
        "if_none_eligible": "NO_SAFE_DISCOVERY_CANDIDATE_GATE",
        "candidate_gate_is_deployment_threshold": False,
        "candidate_gate_is_calibrated_probability": False,
        "holdout_labels_predictions_forbidden_during_gate_selection": True,
    }


def classify_holdout_decision_rule() -> dict[str, Any]:
    return {
        "PASS_INDEPENDENT_HIGH_PRECISION_SIGNAL": {
            "holdout_labeled_minimums_met": True,
            "accepted_wrong_positive": 0,
            "accepted_negative": 0,
            "accepted_exact_gte": 2,
        },
        "INCONCLUSIVE_SAFE_BUT_LOW_SUPPORT": {
            "holdout_labeled_minimums_met": True,
            "accepted_wrong_positive": 0,
            "accepted_negative": 0,
            "accepted_exact_lt": 2,
        },
        "FAIL_INDEPENDENT_GATE_SAFETY": {
            "accepted_wrong_positive_gt": 0,
            "or_accepted_negative_gt": 0,
        },
        "BLOCKED_INSUFFICIENT_LABELED_DISCOVERY": {
            "after_primary_plus_reserve": True,
            "readable_lt": 8,
            "or_negative_lt": 20,
        },
        "BLOCKED_INSUFFICIENT_LABELED_HOLDOUT": {
            "after_primary_plus_reserve": True,
            "readable_lt": 10,
            "or_negative_lt": 24,
        },
        "rules_immutable_after_results": True,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _batch_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "stratum": dict(Counter(str(r["stratum"]) for r in rows)),
        "crop_source_type": dict(Counter(str(r["crop_source_type"]) for r in rows)),
        "timeline_bin": {
            str(k): v for k, v in Counter(int(r["timeline_bin"]) for r in rows).items()
        },
        "documented_vs_other": {
            "note": "documented_global_candidate_id present for all; not player identity",
            "unique_documented_global_candidate_ids": len(
                {int(r["documented_global_candidate_id"]) for r in rows}
            ),
        },
    }


def run_clean_split(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    git_before: str,
    git_head: str,
    prior_untracked_hashes: Mapping[str, str],
    ntfs_snapshot: Mapping[str, Any],
    network_audit: Mapping[str, Any],
) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    universe_rel = str(config["source"]["clean_universe_items"])
    universe_path = (project_root / universe_rel).resolve()
    expected_n = int(config["source"]["expected_universe_size"])
    rows = load_jsonl(universe_path)
    if len(rows) != expected_n:
        raise CleanSplitError(
            f"BLOCKED_CLEAN_UNIVERSE_INPUT_CONTRACT_MISMATCH size={len(rows)}"
        )

    for row in rows:
        if str(row.get("annotation_status")) != "unreviewed":
            raise CleanSplitError("annotation_status must be unreviewed")
        if not bool(row.get("roi_valid")):
            raise CleanSplitError(f"invalid ROI: {row.get('review_item_id')}")
        for field in MANUAL_BLANK_FIELDS:
            if field in row and row[field] not in (None, ""):
                raise CleanSplitError(f"manual field populated: {field}")
        for field in (
            "ocr_prediction",
            "ocr_confidence",
            "ocr_exact_match",
            "ocr_wrong",
            "threshold_acceptance",
        ):
            if row.get(field) not in (None, ""):
                raise CleanSplitError(f"OCR/prediction field populated: {field}")

    # Score + strata
    scored, feature_meta = score_universe(rows, config["feature_contract"])
    scored = assign_timeline_bins(
        scored, n_bins=int(config["diversity_targets"]["timeline_quartiles"])
    )
    stratified = assign_quantile_strata(scored, config["strata"])

    # Near-dup + leakage
    _hashes, nd_edges, nd_stats = compute_near_duplicates(
        stratified,
        nd_contract=config["near_duplicate"],
        project_root=project_root,
    )
    group_rows, item_to_group, item_to_nd, leakage_audit = build_leakage_groups(
        stratified,
        near_dup_edges=nd_edges,
        documented_components=config["documented_components"],
    )
    leakage_audit.update(nd_stats)

    for row in stratified:
        rid = str(row["review_item_id"])
        row["leakage_group_id"] = item_to_group[rid]
        row["near_duplicate_cluster_id"] = item_to_nd[rid]

    # Capacity-aware quota search (label/prediction blind).
    search_result = search_capacity_balanced_quotas(
        stratified,
        item_to_group=item_to_group,
        config=config,
    )
    batches = search_result["batches"]
    batch_quotas = search_result["batch_quotas"]

    reused_key = str(
        (config.get("source_type_keys") or SOURCE_SHORT).get("reused", SOURCE_TYPE_REUSED)
    )
    recomp_key = str(
        (config.get("source_type_keys") or SOURCE_SHORT).get(
            "recomputed", SOURCE_TYPE_RECOMPUTED
        )
    )
    selected_all = [r for b in BATCH_ORDER for r in batches[b]]
    sel_reused = sum(1 for r in selected_all if str(r["crop_source_type"]) == reused_key)
    sel_recomp = sum(1 for r in selected_all if str(r["crop_source_type"]) == recomp_key)
    for batch in BATCH_ORDER:
        if sum(
            1
            for r in batches[batch]
            if str(r["crop_source_type"]) == recomp_key
        ) <= 0:
            raise CleanSplitError(
                f"BLOCKED_CAPACITY_BALANCED_SPLIT_INFEASIBLE {batch} recomputed=0"
            )

    overlap = assert_zero_overlap(batches)
    selected_ids = {
        str(r["review_item_id"]) for rows_ in batches.values() for r in rows_
    }
    if len(selected_ids) != 128:
        raise CleanSplitError(f"allocated must be 128, got {len(selected_ids)}")

    # Output dirs
    for sub in (
        "source_universe",
        "leakage_audit",
        "discovery_primary",
        "discovery_reserve",
        "holdout_primary",
        "holdout_reserve",
        "review_packages/discovery_primary",
        "review_packages/discovery_reserve",
        "review_packages/holdout_primary",
        "review_packages/holdout_reserve",
        "runtime",
        "effective_configs",
    ):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    search_export = {k: v for k, v in search_result.items() if k != "batches"}
    write_json(output_dir / "capacity_search_result.json", search_export)
    write_json(output_dir / "runtime" / "capacity_search_result.json", search_export)

    # Contact sheets
    sheet_cfg = config["contact_sheets"]
    max_per = int(sheet_cfg["max_items_per_sheet"])
    cols = int(sheet_cfg["layout_columns"])
    png_paths: list[Path] = []
    for batch in BATCH_ORDER:
        images: dict[str, np.ndarray] = {}
        for item in batches[batch]:
            path = Path(str(item["source_crop_path"]))
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise CleanSplitError(f"unreadable crop for sheet: {path}")
            images[str(item["crop_id"])] = img
        pages = [
            batches[batch][i : i + max_per]
            for i in range(0, len(batches[batch]), max_per)
        ]
        for page_i, page_items in enumerate(pages, start=1):
            rel = (
                f"review_packages/{batch}/contact_sheet_{page_i:02d}.png"
            )
            abs_path = output_dir / rel
            sheet = render_contact_sheet(
                page_items, images, max_items=max_per, columns=cols
            )
            write_png(abs_path, sheet)
            png_paths.append(abs_path)
            for tile_i, item in enumerate(page_items, start=1):
                item["contact_sheet_path"] = rel
                item["contact_sheet_page"] = page_i
                item["tile_index"] = tile_i
        if len(pages) > int(
            {"discovery_primary": 4, "discovery_reserve": 2, "holdout_primary": 4, "holdout_reserve": 2}[
                batch
            ]
        ):
            raise CleanSplitError(f"too many sheets for {batch}")

    if len(png_paths) > int(sheet_cfg["max_total_png"]):
        raise CleanSplitError(f"PNG budget exceeded: {len(png_paths)}")

    # Forbid JPEG copies / videos in output
    # (we never write them)

    # Source universe export
    source_rows = []
    for row in stratified:
        export = {
            "schema_version": SOURCE_SCHEMA,
            "review_item_id": row["review_item_id"],
            "crop_id": row["crop_id"],
            "segment_id": row["segment_id"],
            "raw_track_id": row["raw_track_id"],
            "documented_global_candidate_id": row["documented_global_candidate_id"],
            "frame_index": row["frame_index"],
            "source_crop_path": row["source_crop_path"],
            "source_crop_sha256": row["source_crop_sha256"],
            "crop_source_type": row["crop_source_type"],
            "composite_score": row["composite_score"],
            "signal_stratum": row["signal_stratum"],
            "timeline_bin": row["timeline_bin"],
            "leakage_group_id": row["leakage_group_id"],
            "near_duplicate_cluster_id": row["near_duplicate_cluster_id"],
            "annotation_status": "unreviewed",
        }
        source_rows.append(export)
    write_jsonl(output_dir / "source_universe" / "clean_split_source_universe.jsonl", source_rows)
    # Also top-level aliases required by contract section 17
    write_jsonl(output_dir / "clean_split_source_universe.jsonl", source_rows)

    write_jsonl(output_dir / "leakage_audit" / "leakage_groups.jsonl", group_rows)
    write_jsonl(output_dir / "leakage_groups.jsonl", group_rows)
    write_json(output_dir / "leakage_audit" / "leakage_audit_summary.json", leakage_audit)
    write_json(output_dir / "leakage_audit_summary.json", leakage_audit)

    unselected = [
        r for r in stratified if str(r["review_item_id"]) not in selected_ids
    ]
    unselected_manifest = []
    for row in sorted(unselected, key=lambda r: str(r["review_item_id"])):
        unselected_manifest.append(
            {
                "schema_version": SELECTED_SCHEMA,
                "batch": "unselected_pool",
                "review_item_id": row["review_item_id"],
                "crop_id": row["crop_id"],
                "segment_id": row["segment_id"],
                "raw_track_id": row["raw_track_id"],
                "documented_global_candidate_id": row["documented_global_candidate_id"],
                "leakage_group_id": row["leakage_group_id"],
                "near_duplicate_cluster_id": row["near_duplicate_cluster_id"],
                "frame_index": row["frame_index"],
                "source_crop_path": row["source_crop_path"],
                "source_crop_sha256": row["source_crop_sha256"],
                "crop_source_type": row["crop_source_type"],
                "stratum": row["stratum"],
                "composite_score": row["composite_score"],
                "timeline_bin": row["timeline_bin"],
                "annotation_status": "unreviewed",
            }
        )
    if len(unselected_manifest) != 346:
        raise CleanSplitError(f"unselected must be 346, got {len(unselected_manifest)}")

    def _selected_manifest_row(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": SELECTED_SCHEMA,
            "split_item_id": item["split_item_id"],
            "batch": item["batch"],
            "batch_order": item["batch_order"],
            "review_item_id": item["review_item_id"],
            "crop_id": item["crop_id"],
            "segment_id": item["segment_id"],
            "raw_track_id": item["raw_track_id"],
            "documented_global_candidate_id": item["documented_global_candidate_id"],
            "leakage_group_id": item["leakage_group_id"],
            "near_duplicate_cluster_id": item["near_duplicate_cluster_id"],
            "frame_index": item["frame_index"],
            "source_crop_path": item["source_crop_path"],
            "source_crop_sha256": item["source_crop_sha256"],
            "crop_source_type": item["crop_source_type"],
            "stratum": item["stratum"],
            "composite_score": item["composite_score"],
            "timeline_bin": item["timeline_bin"],
            "contact_sheet_path": item.get("contact_sheet_path"),
            "contact_sheet_page": item.get("contact_sheet_page"),
            "tile_index": item.get("tile_index"),
            "annotation_status": "unreviewed",
            "source_universe_artifact": "clean_split_source_universe.jsonl",
            "source_universe_sha256": None,  # filled after write
        }

    batch_manifest_rows = {
        batch: [_selected_manifest_row(r) for r in batches[batch]] for batch in BATCH_ORDER
    }

    # Annotation templates (no bias fields)
    forbidden_reviewer = set(config["reviewer_facing_bias_fields_forbidden"])
    for batch in BATCH_ORDER:
        for col in TEMPLATE_COLUMNS:
            if col in forbidden_reviewer:
                raise CleanSplitError(f"template column is bias field: {col}")
        tmpl = build_annotation_template_rows(batches[batch])
        if len(tmpl) != len(batches[batch]):
            raise CleanSplitError("annotation row count mismatch")
        write_annotation_template(
            output_dir / f"{batch}_annotation_template.csv", tmpl
        )
        write_annotation_template(
            output_dir / batch / f"{batch}_annotation_template.csv", tmpl
        )
        write_jsonl(
            output_dir / f"{batch}_manifest.jsonl",
            batch_manifest_rows[batch],
        )
        write_jsonl(
            output_dir / batch / f"{batch}_manifest.jsonl",
            batch_manifest_rows[batch],
        )

    write_jsonl(output_dir / "unselected_pool_manifest.jsonl", unselected_manifest)

    instructions = build_review_instructions_tr()
    (output_dir / "review_instructions.md").write_text(instructions, encoding="utf-8")

    # Effective config copy
    cfg_src = project_root / "configs/reid/jersey_clean_split_stage5c_rebuild_r2.yaml"
    shutil.copy2(cfg_src, output_dir / "effective_configs" / cfg_src.name)

    prereg = {
        "schema_version": PREREG_SCHEMA,
        "rebuild_generation": "r2",
        "split_generation": "r2",
        "threshold_selected": False,
        "deployment_threshold_selected": False,
        "discovery_predictions_seen": False,
        "holdout_predictions_seen": False,
        "discovery_labels_seen": False,
        "holdout_labels_seen": False,
        "historical_threshold_reused": False,
        "discovery_annotation_sufficiency": {
            "primary_min_readable_positive": int(
                config["preregistration"]["discovery_primary_min_readable"]
            ),
            "primary_min_non_readable_negative": int(
                config["preregistration"]["discovery_primary_min_negative"]
            ),
            "reserve_open_policy": "batch_order_only_if_primary_minimums_unmet",
            "inference_before_reserve_decision_forbidden": True,
        },
        "discovery_candidate_gate_derivation": derive_discovery_candidate_gate_rule(),
        "holdout_annotation_sufficiency": {
            "primary_min_readable_positive": int(
                config["preregistration"]["holdout_primary_min_readable"]
            ),
            "primary_min_negative_non_readable": int(
                config["preregistration"]["holdout_primary_min_negative"]
            ),
            "reserve_open_policy": "batch_order_only_if_primary_label_minimums_unmet",
            "reserve_open_decision_before_reserve_labels": True,
            "inference_before_reserve_annotation_freeze_forbidden": True,
        },
        "holdout_decision_rules": classify_holdout_decision_rule(),
        "independence": dict(config["independence"]),
        "batch_order": list(BATCH_ORDER),
        "capacity_balancing": dict(config.get("capacity_balancing") or {}),
        "capacity_search": {
            "selected_quota_vector": search_result["selected_quota_vector"],
            "proportional_reference_vector": search_result[
                "proportional_reference_vector"
            ],
            "selected_total_recomputed": search_result["selected_total_recomputed"],
            "maximum_feasible_recomputed_count": search_result[
                "maximum_feasible_recomputed_count"
            ],
            "selected_recomputed_is_maximum_feasible": search_result[
                "selected_recomputed_is_maximum_feasible"
            ],
            "objective_scores": search_result["objective_scores"],
            "labels_used_for_quota_selection": False,
            "predictions_used_for_quota_selection": False,
            "contact_sheets_viewed_for_quota_selection": False,
        },
        "canonical_split_generation": config.get(
            "canonical_split_generation", "r2_capacity_balanced"
        ),
        # True only in successfully published capacity-balanced root artifacts.
        "previous_split_deprecated_for_downstream": True,
        "previous_r4_split_deprecation_reason": (
            (config.get("previous_split") or {}).get(
                "deprecation_reason_if_replaced"
            )
        ),
    }
    write_json(output_dir / "clean_split_preregistration.json", prereg)

    strata_counts = dict(Counter(str(r["signal_stratum"]) for r in stratified))
    video_path = project_root / "data/test_clips/sample.mp4"
    video_sha = sha256_file(video_path) if video_path.is_file() else None

    input_roots = {
        "base": "outputs/reid/full_stage4b_rebuild_r2",
        "overlay": "outputs/reid/full_stage4b_rebuild_r2_documented_link_overlay",
        "stage5": "outputs/reid/full_stage4b_rebuild_r2_stage5_replay",
        "visibility_universe": "outputs/reid/full_stage4b_rebuild_r2_stage5c_visibility_universe",
    }
    input_meta = {}
    for name, rel in input_roots.items():
        p = project_root / rel
        input_meta[name] = {
            "path": str(p),
            "exists": p.is_dir(),
        }

    command = (
        "python scripts/build_reid_jersey_clean_split.py "
        "--config configs/reid/jersey_clean_split_stage5c_rebuild_r2.yaml "
        f"--output-dir {output_dir}"
    )

    selected_source = {
        "reused": sel_reused,
        "recomputed": sel_recomp,
        "total": len(selected_all),
    }
    unselected_for_src = [
        r for r in stratified if str(r["review_item_id"]) not in selected_ids
    ]
    unselected_source = {
        "reused": sum(
            1 for r in unselected_for_src if str(r["crop_source_type"]) == reused_key
        ),
        "recomputed": sum(
            1 for r in unselected_for_src if str(r["crop_source_type"]) == recomp_key
        ),
        "total": len(unselected_for_src),
    }
    # Dynamic totals must match selected quota vector.
    if sel_recomp != int(search_result["selected_total_recomputed"]):
        raise CleanSplitError(
            f"BLOCKED_CAPACITY_BALANCED_SPLIT_INFEASIBLE selected_recomp "
            f"{sel_recomp} != {search_result['selected_total_recomputed']}"
        )
    if unselected_source["recomputed"] != 75 - sel_recomp:
        raise CleanSplitError(
            f"unselected recomputed inconsistency: {unselected_source}"
        )
    if unselected_source["reused"] != 399 - sel_reused:
        raise CleanSplitError(
            f"unselected reused inconsistency: {unselected_source}"
        )

    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "project_head": git_head,
        "source_video_sha256": video_sha,
        "input_roots": input_meta,
        "clean_universe_path": str(universe_path),
        "clean_universe_sha256": sha256_file(universe_path),
        "builder_path": str(project_root / "scripts/build_reid_jersey_clean_split.py"),
        "builder_sha256": sha256_file(
            project_root / "scripts/build_reid_jersey_clean_split.py"
        ),
        "config_path": str(cfg_src),
        "config_sha256": sha256_file(cfg_src),
        "exact_command": command,
        "feature_contract": feature_meta,
        "strata_contract": config["strata"],
        "near_duplicate_contract": config["near_duplicate"],
        "leakage_grouping_contract": config["leakage_grouping"],
        "deterministic_allocation_contract": config["allocation"],
        "batch_quotas": batch_quotas,
        "reviewer_bias_exclusion_contract": config[
            "reviewer_facing_bias_fields_forbidden"
        ],
        "historical_independence_limitation": config["independence"],
        "network_policy": "pass_offline_label_blind_split_design_only",
        "visual_output_budget": {
            "contact_sheet_png_max": 12,
            "selected_crop_jpeg_copies": 0,
            "video": 0,
            "full_frame_images": 0,
            "individual_overviews": 0,
        },
        "provenance": {
            "rebuild_generation": "r2",
            "split_generation": "r2",
            "canonical_split_generation": config.get(
                "canonical_split_generation", "r2_capacity_balanced"
            ),
            "clean_universe_label_blind": True,
            "historical_manual_labels_available": False,
            "historical_c3d_membership_available": False,
            "strict_historical_never_seen_claim": False,
            "player_identity_independence_guaranteed": False,
            "independent_within_rebuild_r2_protocol": True,
            "historical_parseq_threshold_reused": False,
            "model_predictions_used_for_selection": False,
            "capacity_balancing_performed_before_annotation": True,
            "capacity_balancing_performed_before_predictions": True,
            "capacity_balancing_performed_before_threshold_selection": True,
            "quota_vector_selected_by_preregistered_algorithm": True,
            "labels_used_for_quota_selection": False,
            "predictions_used_for_quota_selection": False,
            "contact_sheets_viewed_for_quota_selection": False,
            "leakage_rules_relaxed": False,
            "strata_rules_relaxed": False,
            "diversity_rules_relaxed": False,
        },
        "capacity_balancing": dict(config.get("capacity_balancing") or {}),
        "capacity_search": {
            "selected_quota_vector": search_result["selected_quota_vector"],
            "proportional_reference_vector": search_result[
                "proportional_reference_vector"
            ],
            "objective_scores": search_result["objective_scores"],
            "maximum_feasible_recomputed_count": search_result[
                "maximum_feasible_recomputed_count"
            ],
            "global_capacity_upper_bound": search_result[
                "global_capacity_upper_bound"
            ],
            "selected_recomputed_is_maximum_feasible": search_result[
                "selected_recomputed_is_maximum_feasible"
            ],
            "feasible_ranges": search_result["feasible_ranges"],
        },
        "previous_split": {
            **dict(config.get("previous_split") or {}),
            "deprecated_for_downstream": True,
            "deprecation_reason": (config.get("previous_split") or {}).get(
                "deprecation_reason_if_replaced"
            ),
        },
        "canonical_split_root": config.get("canonical_split_root"),
        "selected_source_totals": selected_source,
        "unselected_source_totals": unselected_source,
    }
    write_json(output_dir / "clean_split_contract.json", contract)

    discovery_source = {
        "reused": sum(
            1
            for b in ("discovery_primary", "discovery_reserve")
            for r in batches[b]
            if str(r["crop_source_type"]) == reused_key
        ),
        "recomputed": sum(
            1
            for b in ("discovery_primary", "discovery_reserve")
            for r in batches[b]
            if str(r["crop_source_type"]) == recomp_key
        ),
    }
    holdout_source = {
        "reused": sum(
            1
            for b in ("holdout_primary", "holdout_reserve")
            for r in batches[b]
            if str(r["crop_source_type"]) == reused_key
        ),
        "recomputed": sum(
            1
            for b in ("holdout_primary", "holdout_reserve")
            for r in batches[b]
            if str(r["crop_source_type"]) == recomp_key
        ),
    }

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "final_status": "COMPLETED_CAPACITY_BALANCED_CLEAN_DISCOVERY_HOLDOUT_DESIGN",
        "started_at": started,
        "canonical_split_generation": config.get(
            "canonical_split_generation", "r2_capacity_balanced"
        ),
        "canonical_split_root": config.get("canonical_split_root"),
        "previous_split_deprecated_for_downstream": True,
        "previous_split_path": (config.get("previous_split") or {}).get("path"),
        "universe_counts": {
            "source_universe": 474,
            "discovery_primary": 40,
            "discovery_reserve": 16,
            "holdout_primary": 48,
            "holdout_reserve": 24,
            "allocated": 128,
            "unselected": 346,
        },
        "source_type_counts": {
            "selected": selected_source,
            "unselected": unselected_source,
            "discovery_total": discovery_source,
            "holdout_total": holdout_source,
            "per_batch": {
                batch: dict(Counter(str(r["crop_source_type"]) for r in batches[batch]))
                for batch in BATCH_ORDER
            },
        },
        "capacity_balancing": dict(config.get("capacity_balancing") or {}),
        "capacity_search": {
            "selected_quota_vector": search_result["selected_quota_vector"],
            "proportional_reference_vector": search_result[
                "proportional_reference_vector"
            ],
            "objective_scores": search_result["objective_scores"],
            "maximum_feasible_recomputed_count": search_result[
                "maximum_feasible_recomputed_count"
            ],
            "global_capacity_upper_bound": search_result[
                "global_capacity_upper_bound"
            ],
            "selected_recomputed_is_maximum_feasible": search_result[
                "selected_recomputed_is_maximum_feasible"
            ],
            "feasible_ranges": search_result["feasible_ranges"],
            "search_stats": search_result["search_stats"],
        },
        "previous_r4_split_deprecation_reason": (
            (config.get("previous_split") or {}).get(
                "deprecation_reason_if_replaced"
            )
        ),
        "feature_missing_counts": feature_meta["missing_counts"],
        "strata_distributions": strata_counts,
        "exact_near_duplicate_statistics": {
            "exact_duplicate_sha_cluster_count": leakage_audit[
                "exact_duplicate_sha_cluster_count"
            ],
            "near_duplicate_edge_count": leakage_audit["near_duplicate_edge_count"],
            "near_duplicate_connected_cluster_count": leakage_audit[
                "near_duplicate_connected_cluster_count"
            ],
            "near_duplicate_maximum_cluster_size": leakage_audit[
                "near_duplicate_maximum_cluster_size"
            ],
            "cross_raw_track_near_duplicate_count": leakage_audit[
                "cross_raw_track_near_duplicate_count"
            ],
        },
        "leakage_group_statistics": {
            "leakage_group_count": leakage_audit["leakage_group_count"],
            "leakage_max_group_size": leakage_audit["leakage_max_group_size"],
            "leakage_singleton_count": leakage_audit["leakage_singleton_count"],
        },
        "batch_distributions": {
            batch: _batch_distribution(batches[batch]) for batch in BATCH_ORDER
        },
        "overlap_checks": overlap,
        "annotation_status": {
            "all_unreviewed": True,
            "manual_fields_blank": True,
            "predictions_confidence_absent": True,
        },
        "preregistration_status": {
            "threshold_selected": False,
            "deployment_threshold_selected": False,
            "file": "clean_split_preregistration.json",
        },
        "historical_limitation": config["independence"],
        "downstream_readiness": True,
        "next_gate": "REBUILD-R5_STAGE5C_DISCOVERY_PRIMARY_ANNOTATION_FREEZE",
        "contact_sheet_png_count": len(png_paths),
        "previous_r4_sheets_deprecated": True,
        "capacity_balanced_sheets_canonical": True,
        "contact_sheets_visually_interpreted_by_agent": False,
    }
    write_json(output_dir / "clean_split_summary.json", summary)

    # Manifest (without self sha)
    artifact_paths = [
        output_dir / "clean_split_source_universe.jsonl",
        output_dir / "leakage_groups.jsonl",
        output_dir / "leakage_audit_summary.json",
        output_dir / "discovery_primary_manifest.jsonl",
        output_dir / "discovery_reserve_manifest.jsonl",
        output_dir / "holdout_primary_manifest.jsonl",
        output_dir / "holdout_reserve_manifest.jsonl",
        output_dir / "unselected_pool_manifest.jsonl",
        output_dir / "discovery_primary_annotation_template.csv",
        output_dir / "discovery_reserve_annotation_template.csv",
        output_dir / "holdout_primary_annotation_template.csv",
        output_dir / "holdout_reserve_annotation_template.csv",
        output_dir / "review_instructions.md",
        output_dir / "clean_split_contract.json",
        output_dir / "clean_split_preregistration.json",
        output_dir / "clean_split_summary.json",
        output_dir / "capacity_search_result.json",
    ] + png_paths

    artifacts = [_artifact_meta(p) for p in artifact_paths]
    jpeg_count = sum(1 for p in output_dir.rglob("*.jpg") if p.is_file())
    mp4_count = sum(1 for p in output_dir.rglob("*.mp4") if p.is_file())
    if jpeg_count or mp4_count:
        raise CleanSplitError(f"unexpected media jpeg={jpeg_count} mp4={mp4_count}")

    git_after = project_git_status(project_root)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": artifacts,
        "input_artifact_paths": {
            "clean_universe": {
                "path": str(universe_path),
                "sha256": sha256_file(universe_path),
            },
            "config": {"path": str(cfg_src), "sha256": sha256_file(cfg_src)},
        },
        "source_crop_references_only": True,
        "git_before": git_before,
        "git_after": git_after,
        "git_head": git_head,
        "prior_untracked_file_hashes": dict(prior_untracked_hashes),
        "ntfs_snapshot": ntfs_snapshot,
        "network_audit": network_audit,
        "atomic_finalization": True,
        "temp_cleanup_policy": "exact_unique_temp_root_only",
        "png_count": len(png_paths),
        "jpeg_count": jpeg_count,
        "mp4_count": mp4_count,
    }
    write_json(output_dir / "clean_split_manifest.json", manifest)

    return summary


def create_unique_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    tmp = (
        parent
        / f"_tmp_full_stage4b_rebuild_r2_stage5c_clean_split_capacity_balanced_{token}"
    )
    if tmp.exists():
        raise CleanSplitError(f"temp root exists: {tmp}")
    tmp.mkdir(parents=False, exist_ok=False)
    return tmp


def atomic_publish(temp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise CleanSplitError(
            f"FAILED_ATOMIC_CAPACITY_BALANCED_SPLIT final exists (no overwrite): {final_dir}"
        )
    os.rename(temp_dir, final_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            _PROJECT_ROOT / "configs/reid/jersey_clean_split_stage5c_rebuild_r2.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            _PROJECT_ROOT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5c_clean_split_capacity_balanced"
        ),
    )
    parser.add_argument("--project-root", default=str(_PROJECT_ROOT))
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_root = Path(args.project_root).resolve()
    final_dir = Path(args.output_dir)
    if not final_dir.is_absolute():
        final_dir = (project_root / final_dir).resolve()
    else:
        final_dir = final_dir.resolve()

    config = load_config(args.config)
    git_head = project_git_head(project_root)
    git_before = project_git_status(project_root)

    prior_files = [
        "scripts/build_reid_jersey_parseq_holdout.py",
        "configs/reid/jersey_parseq_holdout_stage5c_c3f_a.yaml",
        "tests/test_reid_jersey_parseq_holdout.py",
        "scripts/build_reid_jersey_clean_review_universe.py",
        "configs/reid/jersey_clean_review_universe_stage5c_rebuild_r2.yaml",
        "tests/test_reid_jersey_clean_review_universe.py",
    ]
    prior_hashes = {
        rel: sha256_file(project_root / rel) for rel in prior_files
    }

    ntfs_dir = Path(
        "/mnt/c/Users/enest/Documents/football_analytics_recovery/rebuild_snapshots"
    )
    ntfs_name = "REBUILD_R4B_FAILED_CAPACITY_CODE_b386f07.tar.gz"
    ntfs_path = ntfs_dir / ntfs_name
    ntfs_snapshot = {
        "path": str(ntfs_path),
        "exists": ntfs_path.is_file(),
        "sha256": sha256_file(ntfs_path) if ntfs_path.is_file() else None,
        "bytes": ntfs_path.stat().st_size if ntfs_path.is_file() else None,
        "role": "failed_r4b_capacity_code_snapshot",
    }

    network_audit = {
        "policy": "pass_offline_label_blind_split_design_only",
        "downloads": 0,
        "model_checkpoint_load": False,
        "ocr_inference": False,
    }

    temp_dir: Path | None = None
    try:
        if final_dir.exists():
            raise CleanSplitError(
                f"FAILED_ATOMIC_CAPACITY_BALANCED_SPLIT final root exists: {final_dir}"
            )
        temp_dir = create_unique_temp_root(final_dir)
        summary = run_clean_split(
            config,
            project_root=project_root,
            output_dir=temp_dir,
            git_before=git_before,
            git_head=git_head,
            prior_untracked_hashes=prior_hashes,
            ntfs_snapshot=ntfs_snapshot,
            network_audit=network_audit,
        )
        atomic_publish(temp_dir, final_dir)
        temp_dir = None
        print(json.dumps({"status": summary["final_status"], "output": str(final_dir)}))
        return 0
    except Exception as exc:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
