#!/usr/bin/env python3
"""Stage 5D-F3 — Independent sample retrieval scoring and evaluation.

Compares frozen gallery-v1 (7 anchors) with 150 existing sample embeddings.
Uses F2A frozen GT and F2B amended strong-signal contract. No threshold,
identity assignment, gallery mutation, or new embeddings.
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

import numpy as np
import yaml
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.metrics import average_precision_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.candidates import cosine_similarity  # noqa: E402
from football_analytics.reid.segment_regression import (  # noqa: E402
    embedding_vector_sha256,
)

CONFIG_SCHEMA = "reid_target_independent_retrieval_evaluation_config_v1"
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FINAL_STATUS = "COMPLETED_STAGE5D_F3_TARGET_001_INDEPENDENT_RETRIEVAL_EVALUATED"
DIM = 512
GALLERY_IDS = (
    "target_001_ext_anchor_001",
    "target_001_ext_anchor_003",
    "target_001_ext_anchor_004",
    "target_001_ext_anchor_006",
    "target_001_ext_anchor_008",
    "target_001_ext_anchor_011",
    "target_001_ext_anchor_014",
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
ALLOWED_DIRTY = {
    "scripts/run_reid_target_independent_retrieval_evaluation.py",
    "configs/reid/target_independent_retrieval_evaluation_stage5d_target_001.yaml",
    "tests/test_reid_target_independent_retrieval_evaluation.py",
    "docs/setup/stage5d-target-independent-retrieval-scoring-and-evaluation.md",
}
NEXT_BY_OUTCOME = {
    "INDEPENDENT_RETRIEVAL_STRONG_SIGNAL": (
        "STAGE5D-F4_TARGET_001_INDEPENDENT_CALIBRATION_SOURCE_DESIGN"
    ),
    "INDEPENDENT_RETRIEVAL_PROMISING_BUT_OVERLAPPING": (
        "STAGE5D-F3A_TARGET_001_RETRIEVAL_ERROR_ANALYSIS_AND_GALLERY_DIAGNOSTICS"
    ),
    "INDEPENDENT_RETRIEVAL_WEAK": (
        "STAGE5D-F3A_TARGET_001_RETRIEVAL_ERROR_ANALYSIS_AND_GALLERY_DIAGNOSTICS"
    ),
    "INDEPENDENT_RETRIEVAL_INSUFFICIENT_GROUND_TRUTH": (
        "STAGE5D-F2C_TARGET_001_GROUND_TRUTH_SUPPORT_EXPANSION_DESIGN"
    ),
}


class RetrievalEvalError(RuntimeError):
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
        raise RetrievalEvalError("unexpected config schema")
    if not config.get("offline_required"):
        raise RetrievalEvalError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise RetrievalEvalError(f"path traversal rejected: {rel}")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise RetrievalEvalError(f"invalid bool: {value!r}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise RetrievalEvalError(
                    "BLOCKED_STAGE5D_F3_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GIT_CONTRACT_MISMATCH message")
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str, *, blocker: str) -> str:
    if not snapshot_path.is_file():
        raise RetrievalEvalError(f"{blocker} snapshot")
    sidecar = Path(str(snapshot_path) + ".sha256")
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_snapshot_manifest.json")
    )
    if not manifest.is_file():
        alt = snapshot_path.with_name(
            snapshot_path.name.replace(".tar.gz", "_manifest.json")
        )
        if alt.is_file():
            manifest = alt
    if not sidecar.is_file() or not listing.is_file() or not manifest.is_file():
        raise RetrievalEvalError(f"{blocker} snapshot_sidecar")
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(man_payload.get("sha256") or man_payload.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise RetrievalEvalError(f"{blocker} snapshot_sha")
    return actual


def validate_f2b(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["stage5d_f2b_amendment"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_f2b_summary.json")
    if summary.get("final_status") != config["stage5d_f2b_amendment"][
        "expected_final_status"
    ]:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_METRIC_AMENDMENT_CONTRACT_MISMATCH status"
        )
    checks = {
        "segment_positives": 8,
        "segment_negatives": 110,
        "segment_excluded": 32,
        "eligible_segment_total": 118,
        "clean_positive_components": 4,
        "clean_negative_components": 95,
        "metric_component_total": 99,
        "excluded_components": 26,
        "conflicting_components": 4,
    }
    for key, want in checks.items():
        if summary.get(key) != want:
            raise RetrievalEvalError(
                f"BLOCKED_STAGE5D_F3_METRIC_AMENDMENT_CONTRACT_MISMATCH {key}"
            )
    if summary.get("segment_recall_ceilings") != {
        "Recall@1": 0.125,
        "Recall@3": 0.375,
        "Recall@5": 0.625,
        "Recall@10": 1.0,
    }:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_METRIC_AMENDMENT_CONTRACT_MISMATCH ceilings"
        )
    if summary.get("component_recall_ceilings") != {
        "Recall@1": 0.25,
        "Recall@3": 0.75,
        "Recall@5": 1.0,
        "Recall@10": 1.0,
    }:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_METRIC_AMENDMENT_CONTRACT_MISMATCH cceilings"
        )
    if not summary.get("original_scoring_formulas_unchanged"):
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_METRIC_AMENDMENT_CONTRACT_MISMATCH formulas"
        )
    amended = load_json(
        root
        / "metric_feasibility"
        / "target_001_amended_retrieval_outcome_contract.json"
    )
    strong = amended["INDEPENDENT_RETRIEVAL_STRONG_SIGNAL"]
    if strong["ranking"]["segment_Recall@5"] != 0.625:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_METRIC_AMENDMENT_CONTRACT_MISMATCH strong"
        )
    snap = resolve_snapshot_sha(
        Path(config["stage5d_f2b_amendment"]["snapshot_path"]),
        config["stage5d_f2b_amendment"]["expected_snapshot_sha256"],
        blocker="BLOCKED_STAGE5D_F3_METRIC_AMENDMENT_CONTRACT_MISMATCH",
    )
    return {
        "root": root,
        "summary": summary,
        "amended": amended,
        "snapshot_sha256": snap,
        "listing_sha256": listing_sha(root)[1],
        "contract_sha256": sha256_file(root / "stage5d_f2b_contract.json"),
        "amended_sha256": sha256_file(
            root
            / "metric_feasibility"
            / "target_001_amended_retrieval_outcome_contract.json"
        ),
    }


def validate_f2a(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["stage5d_f2a_freeze"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_f2a_summary.json")
    if summary.get("final_status") != config["stage5d_f2a_freeze"][
        "expected_final_status"
    ]:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH status"
        )
    if summary.get("reviewed_total") != 150:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH reviewed"
        )
    if summary.get("clean_positive_metric_items") != 8:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH pos"
        )
    if summary.get("clean_negative_metric_items") != 110:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH neg"
        )
    if summary.get("excluded_metric_items") != 32:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH excl"
        )
    if summary.get("eligible_total") != 118:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH eligible"
        )
    if summary.get("positive_exact_ids") != list(POSITIVE_IDS):
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH positive_ids"
        )
    freeze = load_json(
        root / "ground_truth_freeze" / "target_001_sample_ground_truth_freeze.json"
    )
    if freeze.get("similarity_observed_before_freeze") is not False:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH sim_obs"
        )
    if freeze.get("gallery_vectors_read") is not False:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH gvec"
        )
    if freeze.get("sample_embedding_vectors_read") is not False:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH svec"
        )
    if freeze.get("manual_decisions_frozen") is not True:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH frozen"
        )
    conflicts = {
        row["evaluation_component_id"]
        for row in summary["component_label_distribution"]["conflict_details"]
    }
    if conflicts != set(CONFLICT_COMPONENTS):
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH conflicts"
        )
    snap = resolve_snapshot_sha(
        Path(config["stage5d_f2a_freeze"]["snapshot_path"]),
        config["stage5d_f2a_freeze"]["expected_snapshot_sha256"],
        blocker="BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH",
    )
    decisions_path = (
        root
        / "ground_truth_freeze"
        / "target_001_sample_ground_truth_decisions_frozen.csv"
    )
    with decisions_path.open(encoding="utf-8") as handle:
        decisions = list(csv.DictReader(handle))
    if len(decisions) != 150:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_GROUND_TRUTH_CONTRACT_MISMATCH decisions"
        )
    return {
        "root": root,
        "summary": summary,
        "freeze": freeze,
        "decisions": decisions,
        "snapshot_sha256": snap,
        "listing_sha256": listing_sha(root)[1],
        "gt_sha256": sha256_file(decisions_path),
        "freeze_json_sha256": sha256_file(
            root
            / "ground_truth_freeze"
            / "target_001_sample_ground_truth_freeze.json"
        ),
    }


def validate_gallery(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["gallery_v1"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_b1e_f_summary.json")
    if summary.get("final_status") != config["gallery_v1"]["expected_final_status"]:
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY status")
    if int(summary.get("individual_gallery_members", -1)) != 7:
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY members")
    if int(summary.get("centroid_count", -1)) != 1:
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY centroid")
    if int(summary.get("medoid_count", -1)) != 1:
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY medoid")
    if summary.get("medoid_anchor_candidate_id") != config["gallery_v1"][
        "expected_medoid_anchor"
    ]:
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY medoid_id")
    if int(summary.get("embedding_dimension", -1)) != 512:
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY dim")
    if summary.get("embedding_dtype") != "float32":
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY dtype")
    if int(summary.get("automatic_members", -1)) != 0:
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY automatic")
    members = load_jsonl(root / config["gallery_v1"]["members_jsonl_rel"])
    order = [row["anchor_candidate_id"] for row in sorted(
        members, key=lambda r: int(r["gallery_row_index"])
    )]
    if order != list(GALLERY_IDS):
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY order")
    sha_meta = load_json(root / config["gallery_v1"]["embeddings_sha_rel"])
    if sha_meta.get("row_order") != list(GALLERY_IDS):
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY row_order")
    ind_path = root / config["gallery_v1"]["individual_npy_rel"]
    cen_path = root / config["gallery_v1"]["centroid_npy_rel"]
    med_path = root / config["gallery_v1"]["medoid_npy_rel"]
    ind_sha = sha256_file(ind_path)
    if ind_sha != sha_meta.get("sha256"):
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY npy_sha")
    individual = np.load(ind_path)
    centroid = np.load(cen_path)
    medoid = np.load(med_path)
    if individual.shape != (7, DIM) or individual.dtype != np.float32:
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY ind_shape")
    if centroid.shape not in {(DIM,), (1, DIM)}:
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY cen_shape")
    if medoid.shape not in {(DIM,), (1, DIM)}:
        raise RetrievalEvalError("BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY med_shape")
    centroid = np.asarray(centroid, dtype=np.float32).reshape(DIM)
    medoid = np.asarray(medoid, dtype=np.float32).reshape(DIM)
    for name, arr in (
        ("individual", individual),
        ("centroid", centroid.reshape(1, DIM)),
        ("medoid", medoid.reshape(1, DIM)),
    ):
        if not np.isfinite(arr).all():
            raise RetrievalEvalError(f"BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY {name}_nan")
        norms = np.linalg.norm(arr, axis=1)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise RetrievalEvalError(f"BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY {name}_norm")
        if np.max(np.abs(norms - 1.0)) > float(config["scoring"]["norm_atol"]):
            raise RetrievalEvalError(
                "BLOCKED_STAGE5D_F3_SCORING_NORMALIZATION_CONTRACT_MISMATCH gallery"
            )
    snap = resolve_snapshot_sha(
        Path(config["gallery_v1"]["snapshot_path"]),
        config["gallery_v1"]["expected_snapshot_sha256"],
        blocker="BLOCKED_STAGE5D_F3_GALLERY_INTEGRITY",
    )
    return {
        "root": root,
        "summary": summary,
        "individual": individual,
        "centroid": centroid,
        "medoid": medoid,
        "individual_path": ind_path,
        "centroid_path": cen_path,
        "medoid_path": med_path,
        "individual_sha256": ind_sha,
        "centroid_sha256": sha256_file(cen_path),
        "medoid_sha256": sha256_file(med_path),
        "snapshot_sha256": snap,
        "listing_sha256": listing_sha(root)[1],
        "member_ids": list(GALLERY_IDS),
    }


def validate_sample_universe(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    f2_rel = config["stage5d_f2_package"]["path"]
    f1_rel = config["stage5d_f1_package"]["path"]
    assert_no_path_traversal(f2_rel)
    assert_no_path_traversal(f1_rel)
    mapping = load_jsonl(
        project_root / f2_rel / config["stage5d_f2_package"]["mapping_rel"]
    )
    universe = load_jsonl(
        project_root
        / f1_rel
        / config["stage5d_f1_package"]["scoreable_universe_rel"]
    )
    if len(mapping) != 150 or len(universe) != 150:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH count"
        )
    codes = [row["sample_eval_code"] for row in mapping]
    expected = [f"SAMPLE_EVAL_{i:03d}" for i in range(1, 151)]
    if codes != expected:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH codes"
        )
    by_segment = {row["segment_id"]: row for row in universe}
    artifact_rel = config["sample_embeddings"]["artifact_path"]
    assert_no_path_traversal(artifact_rel)
    artifact = project_root / artifact_rel
    if not artifact.is_file() or artifact.is_symlink():
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH artifact"
        )
    art_sha = sha256_file(artifact)
    if art_sha != config["sample_embeddings"]["expected_artifact_sha256"]:
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH artifact_sha"
        )
    with np.load(artifact, allow_pickle=False) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        segment_ids = [str(x) for x in data["segment_ids"].tolist()]
    if vectors.shape != (150, DIM):
        raise RetrievalEvalError(
            "BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH shape"
        )
    queries: list[dict[str, Any]] = []
    for map_row in mapping:
        seg = map_row["segment_id"]
        uni = by_segment.get(seg)
        if uni is None:
            raise RetrievalEvalError(
                f"BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH missing_{seg}"
            )
        row_idx = int(uni["embedding_row"])
        if segment_ids[row_idx] != seg:
            raise RetrievalEvalError(
                f"BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH row_{seg}"
            )
        if map_row["existing_embedding_artifact_sha256"] != art_sha:
            raise RetrievalEvalError(
                "BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH map_sha"
            )
        if uni["embedding_artifact_sha256"] != art_sha:
            raise RetrievalEvalError(
                "BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH uni_sha"
            )
        vec = vectors[row_idx]
        if vec.shape != (DIM,):
            raise RetrievalEvalError(
                "BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH vec_shape"
            )
        if not np.isfinite(vec).all():
            raise RetrievalEvalError(
                "BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH nonfinite"
            )
        if float(np.linalg.norm(vec)) <= 0:
            raise RetrievalEvalError(
                "BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH zero"
            )
        vsha = embedding_vector_sha256(vec)
        if vsha != map_row["existing_embedding_vector_sha256"]:
            raise RetrievalEvalError(
                "BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH vec_sha_map"
            )
        if vsha != uni["embedding_vector_sha256"]:
            raise RetrievalEvalError(
                "BLOCKED_STAGE5D_F3_SAMPLE_EMBEDDING_UNIVERSE_MISMATCH vec_sha_uni"
            )
        norm = float(np.linalg.norm(vec))
        if abs(norm - 1.0) > float(config["scoring"]["norm_atol"]):
            raise RetrievalEvalError(
                "BLOCKED_STAGE5D_F3_SCORING_NORMALIZATION_CONTRACT_MISMATCH sample"
            )
        queries.append(
            {
                "sample_eval_code": map_row["sample_eval_code"],
                "segment_id": seg,
                "evaluation_component_id": map_row["evaluation_component_id"],
                "embedding_row": row_idx,
                "embedding_vector_sha256": vsha,
                "vector": vec,
            }
        )
    return {
        "queries": queries,
        "artifact_path": artifact,
        "artifact_sha256": art_sha,
        "vectors": vectors,
        "f1_root": project_root / f1_rel,
        "f2_root": project_root / f2_rel,
        "f1_listing_sha256": listing_sha(project_root / f1_rel)[1],
        "f2_listing_sha256": listing_sha(project_root / f2_rel)[1],
    }


def score_queries(
    queries: Sequence[Mapping[str, Any]],
    gallery: np.ndarray,
    centroid: np.ndarray,
    medoid: np.ndarray,
    member_ids: Sequence[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n = len(queries)
    matrix = np.zeros((n, 7), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for i, query in enumerate(queries):
        vec = np.asarray(query["vector"], dtype=np.float32)
        scores = []
        for j in range(7):
            score = cosine_similarity(vec, gallery[j])
            scores.append(score)
            matrix[i, j] = np.float32(score)
        order = sorted(range(7), key=lambda j: (-scores[j], member_ids[j]))
        best_j, second_j = order[0], order[1]
        top3 = sorted(scores, reverse=True)[:3]
        row = {
            "sample_eval_code": query["sample_eval_code"],
            "segment_id": query["segment_id"],
            "evaluation_component_id": query["evaluation_component_id"],
            "individual_cosine_scores": {
                member_ids[j]: float(scores[j]) for j in range(7)
            },
            "individual_cosine_vector": [float(x) for x in scores],
            "best_gallery_anchor_id": member_ids[best_j],
            "best_gallery_anchor_score": float(scores[best_j]),
            "second_best_gallery_anchor_id": member_ids[second_j],
            "second_best_gallery_anchor_score": float(scores[second_j]),
            "max_individual_cosine": float(max(scores)),
            "top3_mean_individual_cosine": float(sum(top3) / 3.0),
            "centroid_cosine": float(cosine_similarity(vec, centroid)),
            "medoid_cosine": float(cosine_similarity(vec, medoid)),
            "mean_individual_cosine": float(sum(scores) / 7.0),
            "embedding_vector_sha256": query["embedding_vector_sha256"],
            "embedding_row": query["embedding_row"],
        }
        rows.append(row)
    return matrix, rows


def recall_at_k(labels: Sequence[int], k: int, n_pos: int) -> float:
    if n_pos <= 0:
        return 0.0
    return float(sum(labels[:k])) / float(n_pos)


def average_precision_from_ranks(positive_ranks_1based: Sequence[int], n_pos: int) -> float:
    if n_pos <= 0:
        return 0.0
    ranks = sorted(int(r) for r in positive_ranks_1based)
    hits = 0
    precision_sum = 0.0
    for rank in ranks:
        hits += 1
        precision_sum += hits / float(rank)
    return float(precision_sum / n_pos)


def ranking_metrics(
    ranked_labels: Sequence[int],
    ranked_scores: Sequence[float],
    *,
    n_pos: int,
) -> dict[str, Any]:
    labels = [int(x) for x in ranked_labels]
    scores = [float(x) for x in ranked_scores]
    pos_ranks = [i + 1 for i, lab in enumerate(labels) if lab == 1]
    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)
    ap_sklearn = float(average_precision_score(y_true, y_score)) if n_pos > 0 else 0.0
    ap_rank = average_precision_from_ranks(pos_ranks, n_pos)
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")
    auprc = ap_sklearn
    mrr = float(1.0 / pos_ranks[0]) if pos_ranks else 0.0
    pos_scores = [scores[i] for i, lab in enumerate(labels) if lab == 1]
    neg_scores = [scores[i] for i, lab in enumerate(labels) if lab == 0]
    margin = (
        float(min(pos_scores) - max(neg_scores))
        if pos_scores and neg_scores
        else float("nan")
    )
    return {
        "positive_count": n_pos,
        "negative_count": len(labels) - n_pos,
        "prevalence": float(n_pos) / float(len(labels)) if labels else 0.0,
        "Recall@1": recall_at_k(labels, 1, n_pos),
        "Recall@3": recall_at_k(labels, 3, n_pos),
        "Recall@5": recall_at_k(labels, 5, n_pos),
        "Recall@10": recall_at_k(labels, 10, n_pos),
        "positives_in_top_1": int(sum(labels[:1])),
        "positives_in_top_3": int(sum(labels[:3])),
        "positives_in_top_5": int(sum(labels[:5])),
        "positives_in_top_10": int(sum(labels[:10])),
        "positives_in_top_20": int(sum(labels[:20])),
        "MRR": mrr,
        "Average_Precision": ap_rank,
        "Average_Precision_sklearn": ap_sklearn,
        "AUROC": auroc,
        "AUPRC": auprc,
        "every_positive_rank": pos_ranks,
        "best_positive_score": float(max(pos_scores)) if pos_scores else None,
        "worst_positive_score": float(min(pos_scores)) if pos_scores else None,
        "mean_positive_score": float(sum(pos_scores) / len(pos_scores))
        if pos_scores
        else None,
        "median_positive_score": float(np.median(pos_scores)) if pos_scores else None,
        "best_negative_score": float(max(neg_scores)) if neg_scores else None,
        "worst_negative_score": float(min(neg_scores)) if neg_scores else None,
        "mean_negative_score": float(sum(neg_scores) / len(neg_scores))
        if neg_scores
        else None,
        "median_negative_score": float(np.median(neg_scores)) if neg_scores else None,
        "separation_margin_min_pos_minus_max_neg": margin,
        "implementation": {
            "library": "scikit-learn",
            "version": SKLEARN_VERSION,
            "average_precision_score": "sklearn.metrics.average_precision_score",
            "roc_auc_score": "sklearn.metrics.roc_auc_score",
            "Recall@K_definition": f"positives_in_top_K / {n_pos}",
            "AP_primary_definition": "rank_precision_interpolation_standard",
        },
    }


def rank_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    code_key: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda r: (-float(r[score_key]), str(r[code_key])),
    )
    out = []
    for idx, row in enumerate(ordered, start=1):
        item = dict(row)
        item["rank"] = idx
        out.append(item)
    return out


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f3_retrieval_eval_{final_dir.name}_{token}"
    if tmp.exists():
        raise RetrievalEvalError(f"temp exists: {tmp}")
    tmp.mkdir(parents=True, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise RetrievalEvalError(f"final root already exists: {final_dir}")
    os.rename(tmp, final_dir)


def select_outcome(
    *,
    segment_metrics: Mapping[str, Any],
    component_metrics: Mapping[str, Any],
    amended_strong: Mapping[str, Any],
) -> dict[str, Any]:
    support = amended_strong["support"]
    support_checks = {
        "clean_positive_segment_count_ge_2": {
            "expected": support["clean_positive_segment_count_ge"],
            "observed": segment_metrics["positive_count"],
            "passed": segment_metrics["positive_count"]
            >= support["clean_positive_segment_count_ge"],
        },
        "clean_negative_segment_count_ge_20": {
            "expected": support["clean_negative_segment_count_ge"],
            "observed": segment_metrics["negative_count"],
            "passed": segment_metrics["negative_count"]
            >= support["clean_negative_segment_count_ge"],
        },
        "clean_positive_component_count_ge_2": {
            "expected": support["clean_positive_component_count_ge"],
            "observed": component_metrics["positive_count"],
            "passed": component_metrics["positive_count"]
            >= support["clean_positive_component_count_ge"],
        },
        "clean_negative_component_count_ge_20": {
            "expected": support["clean_negative_component_count_ge"],
            "observed": component_metrics["negative_count"],
            "passed": component_metrics["negative_count"]
            >= support["clean_negative_component_count_ge"],
        },
    }
    support_ok = all(v["passed"] for v in support_checks.values())
    ranking = amended_strong["ranking"]
    quality = amended_strong["quality"]
    seg_margin = segment_metrics["separation_margin_min_pos_minus_max_neg"]
    comp_margin = component_metrics["separation_margin_min_pos_minus_max_neg"]
    strong_checks = {
        "support_passed": {
            "expected": True,
            "observed": support_ok,
            "passed": support_ok,
        },
        "segment_Recall@5": {
            "expected": ranking["segment_Recall@5"],
            "observed": segment_metrics["Recall@5"],
            "passed": math.isclose(
                segment_metrics["Recall@5"],
                ranking["segment_Recall@5"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
        },
        "segment_Recall@10": {
            "expected": ranking["segment_Recall@10"],
            "observed": segment_metrics["Recall@10"],
            "passed": math.isclose(
                segment_metrics["Recall@10"],
                ranking["segment_Recall@10"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
        },
        "component_Recall@5": {
            "expected": ranking["component_Recall@5"],
            "observed": component_metrics["Recall@5"],
            "passed": math.isclose(
                component_metrics["Recall@5"],
                ranking["component_Recall@5"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
        },
        "segment_AP_ge": {
            "expected": quality["segment_AP_ge"],
            "observed": segment_metrics["Average_Precision"],
            "passed": segment_metrics["Average_Precision"] >= quality["segment_AP_ge"],
        },
        "component_AP_ge": {
            "expected": quality["component_AP_ge"],
            "observed": component_metrics["Average_Precision"],
            "passed": component_metrics["Average_Precision"]
            >= quality["component_AP_ge"],
        },
        "segment_margin_gt_0": {
            "expected": True,
            "observed": seg_margin,
            "passed": bool(seg_margin is not None and seg_margin > 0),
        },
        "component_margin_gt_0": {
            "expected": True,
            "observed": comp_margin,
            "passed": bool(comp_margin is not None and comp_margin > 0),
        },
    }
    if not support_ok:
        outcome = "INDEPENDENT_RETRIEVAL_INSUFFICIENT_GROUND_TRUTH"
        rationale = "Frozen support gates failed."
    elif all(v["passed"] for v in strong_checks.values()):
        outcome = "INDEPENDENT_RETRIEVAL_STRONG_SIGNAL"
        rationale = "All amended strong-signal conditions passed."
    else:
        positives_near_top = (
            segment_metrics["Recall@10"] > 0.0
            or (
                segment_metrics["every_positive_rank"]
                and min(segment_metrics["every_positive_rank"]) <= 10
            )
        )
        overlap = bool(seg_margin is not None and seg_margin <= 0)
        if positives_near_top and overlap:
            outcome = "INDEPENDENT_RETRIEVAL_PROMISING_BUT_OVERLAPPING"
            rationale = (
                "Positives appear near the top of the ranking, but "
                "positive/negative score margin <= 0 (overlap)."
            )
        else:
            outcome = "INDEPENDENT_RETRIEVAL_WEAK"
            rationale = (
                "Target positives are not consistently top-ranked under the "
                "amended strong-signal minima (no new numeric deployment "
                "threshold invented)."
            )
    return {
        "schema_version": "reid_target_001_independent_retrieval_outcome_v1",
        "descriptive_outcome": outcome,
        "exact_next_gate": NEXT_BY_OUTCOME[outcome],
        "support_checks": support_checks,
        "strong_signal_checks": strong_checks,
        "rationale": rationale,
        "limitations": {
            "independent_retrieval_ranking_evaluation": True,
            "deployment_validation": False,
            "threshold_calibration": False,
            "calibrated_probability": False,
            "automatic_identity_assignment": False,
            "gallery_automatic_growth_permission": False,
            "threshold_selected": False,
            "sample_used_for_threshold_selection_forbidden": True,
        },
        "observed_summary": {
            "segment_Recall@5": segment_metrics["Recall@5"],
            "segment_Recall@10": segment_metrics["Recall@10"],
            "segment_AP": segment_metrics["Average_Precision"],
            "segment_margin": seg_margin,
            "component_Recall@5": component_metrics["Recall@5"],
            "component_AP": component_metrics["Average_Precision"],
            "component_margin": comp_margin,
            "every_positive_segment_rank": segment_metrics["every_positive_rank"],
            "every_positive_component_rank": component_metrics["every_positive_rank"],
        },
    }


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
        raise RetrievalEvalError(f"final root already exists: {final_dir}")

    # Validate contracts before loading vectors for sample/gallery scoring paths
    # that mutate nothing; gallery/sample NPYs are loaded only after F2B/F2A ok.
    f2b = validate_f2b(project_root, config)
    f2a = validate_f2a(project_root, config)
    gallery = validate_gallery(project_root, config)
    sample = validate_sample_universe(project_root, config)

    # Source integrity
    sample_mp4 = project_root / config["evaluation_source"]["path"]
    if sha256_file(sample_mp4) != config["evaluation_source"]["expected_sha256"]:
        raise RetrievalEvalError("sample.mp4 sha mismatch")
    if config["evaluation_source"].get("decode_forbidden") is not True:
        raise RetrievalEvalError("decode_forbidden required")

    decisions_by_code = {row["sample_eval_code"]: row for row in f2a["decisions"]}
    generated_at = datetime.now(timezone.utc).isoformat()
    tmp = create_temp_root(final_dir)
    try:
        for sub in (
            "scores",
            "rankings",
            "metrics",
            "evaluation",
            "audit",
            "runtime",
            "effective_configs",
        ):
            (tmp / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, tmp / "effective_configs" / Path(config_path).name)

        pre_score = {
            "schema_version": "reid_target_001_f3_scoring_contract_pre_score_v1",
            "gallery_file_paths": {
                "individual": str(
                    gallery["individual_path"].relative_to(project_root)
                ).replace("\\", "/"),
                "centroid": str(
                    gallery["centroid_path"].relative_to(project_root)
                ).replace("\\", "/"),
                "medoid": str(gallery["medoid_path"].relative_to(project_root)).replace(
                    "\\", "/"
                ),
            },
            "gallery_file_sha256": {
                "individual": gallery["individual_sha256"],
                "centroid": gallery["centroid_sha256"],
                "medoid": gallery["medoid_sha256"],
            },
            "sample_embedding_source_path": str(
                sample["artifact_path"].relative_to(project_root)
            ).replace("\\", "/"),
            "sample_embedding_source_sha256": sample["artifact_sha256"],
            "gallery_count": 7,
            "query_count": 150,
            "dimension": DIM,
            "primary_formula": "max_individual_cosine",
            "secondary_formulas": list(config["scoring"]["secondary"]),
            "metric_universes": {
                "diagnostic_queries": 150,
                "segment_metric_rows": 118,
                "component_metric_rows": 99,
            },
            "ground_truth_sha256": f2a["gt_sha256"],
            "amendment_contract_sha256": f2b["amended_sha256"],
            "scores_seen_at_contract_freeze": False,
            "ranks_seen_at_contract_freeze": False,
            "threshold_selected": False,
            "gallery_member_order": list(GALLERY_IDS),
            "generated_at": generated_at,
        }
        write_json(tmp / "runtime" / "target_001_f3_scoring_contract_pre_score.json", pre_score)

        # Two-pass scoring
        m1, rows1 = score_queries(
            sample["queries"],
            gallery["individual"],
            gallery["centroid"],
            gallery["medoid"],
            gallery["member_ids"],
        )
        m2, rows2 = score_queries(
            sample["queries"],
            gallery["individual"],
            gallery["centroid"],
            gallery["medoid"],
            gallery["member_ids"],
        )
        max_abs = float(np.max(np.abs(m1 - m2)))
        atol = float(config["scoring"]["numeric_atol"])
        if max_abs > atol:
            raise RetrievalEvalError("BLOCKED_STAGE5D_F3_SCORING_NONDETERMINISTIC matrix")
        for a, b in zip(rows1, rows2):
            for key in (
                "max_individual_cosine",
                "top3_mean_individual_cosine",
                "centroid_cosine",
                "medoid_cosine",
                "mean_individual_cosine",
                "best_gallery_anchor_id",
            ):
                if a[key] != b[key] and not (
                    isinstance(a[key], float)
                    and isinstance(b[key], float)
                    and abs(a[key] - b[key]) <= atol
                ):
                    raise RetrievalEvalError(
                        "BLOCKED_STAGE5D_F3_SCORING_NONDETERMINISTIC scores"
                    )
        score_rows = rows1
        matrix = m1

        # Attach GT
        full_rows: list[dict[str, Any]] = []
        for row in score_rows:
            gt = decisions_by_code[row["sample_eval_code"]]
            eligible = parse_bool(gt["retrieval_metric_eligible"])
            clean_pos = parse_bool(gt["clean_positive"])
            clean_neg = parse_bool(gt["clean_negative"])
            item = {
                **row,
                "manual_occurrence_decision": gt["manual_occurrence_decision"],
                "clean_positive": clean_pos,
                "clean_negative": clean_neg,
                "retrieval_metric_eligible": eligible,
                "metric_exclusion_reason": gt.get("metric_exclusion_reason") or None,
                "component_label": gt["component_label"],
                "target_present": parse_bool(gt["target_present"]),
                "gallery_sha256": gallery["individual_sha256"],
                "binary_label": 1 if clean_pos else (0 if clean_neg else None),
            }
            full_rows.append(item)

        # Score artifacts
        np.save(tmp / "scores" / "target_001_sample_individual_cosine.npy", matrix)
        scores_jsonl = tmp / "scores" / "target_001_sample_retrieval_scores.jsonl"
        with scores_jsonl.open("w", encoding="utf-8") as handle:
            for row in full_rows:
                payload = {
                    k: v
                    for k, v in row.items()
                    if k != "individual_cosine_vector" or True
                }
                handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False))
                handle.write("\n")
        write_json(
            tmp / "scores" / "target_001_sample_retrieval_scores_sha256.json",
            {
                "schema_version": "reid_target_001_sample_retrieval_scores_sha256_v1",
                "scores_jsonl_sha256": sha256_file(scores_jsonl),
                "individual_cosine_npy_sha256": sha256_file(
                    tmp / "scores" / "target_001_sample_individual_cosine.npy"
                ),
                "rows": 150,
                "matrix_shape": [150, 7],
            },
        )

        # Segment ranking (118)
        eligible_rows = [r for r in full_rows if r["retrieval_metric_eligible"]]
        if len(eligible_rows) != 118:
            raise RetrievalEvalError(f"eligible rows != 118: {len(eligible_rows)}")
        pos_n = sum(1 for r in eligible_rows if r["clean_positive"])
        neg_n = sum(1 for r in eligible_rows if r["clean_negative"])
        if pos_n != 8 or neg_n != 110:
            raise RetrievalEvalError(f"pos/neg mismatch {pos_n}/{neg_n}")
        seg_ranked = rank_rows(
            eligible_rows, score_key="max_individual_cosine", code_key="sample_eval_code"
        )
        seg_csv = tmp / "rankings" / "target_001_segment_primary_ranking.csv"
        seg_fields = [
            "rank",
            "sample_eval_code",
            "evaluation_component_id",
            "binary_label",
            "manual_occurrence_decision",
            "max_individual_cosine",
            "best_gallery_anchor_id",
            "best_gallery_anchor_score",
            "top3_mean_individual_cosine",
            "centroid_cosine",
            "medoid_cosine",
            "mean_individual_cosine",
            "segment_id",
            "embedding_vector_sha256",
            "component_label",
            "positive_rank_flag",
        ]
        with seg_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=seg_fields)
            writer.writeheader()
            for row in seg_ranked:
                writer.writerow(
                    {
                        "rank": row["rank"],
                        "sample_eval_code": row["sample_eval_code"],
                        "evaluation_component_id": row["evaluation_component_id"],
                        "binary_label": row["binary_label"],
                        "manual_occurrence_decision": row["manual_occurrence_decision"],
                        "max_individual_cosine": f"{row['max_individual_cosine']:.8f}",
                        "best_gallery_anchor_id": row["best_gallery_anchor_id"],
                        "best_gallery_anchor_score": f"{row['best_gallery_anchor_score']:.8f}",
                        "top3_mean_individual_cosine": f"{row['top3_mean_individual_cosine']:.8f}",
                        "centroid_cosine": f"{row['centroid_cosine']:.8f}",
                        "medoid_cosine": f"{row['medoid_cosine']:.8f}",
                        "mean_individual_cosine": f"{row['mean_individual_cosine']:.8f}",
                        "segment_id": row["segment_id"],
                        "embedding_vector_sha256": row["embedding_vector_sha256"],
                        "component_label": row["component_label"],
                        "positive_rank_flag": str(bool(row["clean_positive"])).lower(),
                    }
                )

        positive_segment_ranks = []
        for row in seg_ranked:
            if not row["clean_positive"]:
                continue
            positive_segment_ranks.append(
                {
                    "sample_eval_code": row["sample_eval_code"],
                    "absolute_rank": row["rank"],
                    "primary_score": row["max_individual_cosine"],
                    "best_gallery_anchor": row["best_gallery_anchor_id"],
                    "seven_individual_scores": row["individual_cosine_scores"],
                    "top3_mean_individual_cosine": row["top3_mean_individual_cosine"],
                    "centroid_cosine": row["centroid_cosine"],
                    "medoid_cosine": row["medoid_cosine"],
                    "mean_individual_cosine": row["mean_individual_cosine"],
                    "evaluation_component_id": row["evaluation_component_id"],
                    "conflicting_component_membership": row["evaluation_component_id"]
                    in CONFLICT_COMPONENTS,
                    "segment_level_eligibility": True,
                }
            )
        write_json(
            tmp / "rankings" / "target_001_positive_segment_ranks.json",
            {
                "schema_version": "reid_target_001_positive_segment_ranks_v1",
                "count": 8,
                "positives": positive_segment_ranks,
            },
        )

        seg_labels = [int(r["binary_label"]) for r in seg_ranked]
        seg_scores = [float(r["max_individual_cosine"]) for r in seg_ranked]
        segment_metrics = ranking_metrics(seg_labels, seg_scores, n_pos=8)
        # negatives above each positive
        neg_above = {}
        for prow in positive_segment_ranks:
            code = prow["sample_eval_code"]
            pscore = prow["primary_score"]
            neg_above[code] = sum(
                1
                for r in seg_ranked
                if r["clean_negative"] and r["max_individual_cosine"] > pscore
            )
        segment_metrics["negatives_scoring_above_each_positive"] = neg_above
        segment_metrics["schema_version"] = "reid_target_001_segment_retrieval_metrics_v1"
        segment_metrics["ranking_score"] = "max_individual_cosine"
        segment_metrics["metric_universe_size"] = 118
        write_json(
            tmp / "metrics" / "target_001_segment_retrieval_metrics.json",
            segment_metrics,
        )

        # Component aggregation on clean 99
        by_comp: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in full_rows:
            by_comp[row["evaluation_component_id"]].append(row)
        clean_comp_rows: list[dict[str, Any]] = []
        conflict_audit = []
        for cid, members in sorted(by_comp.items()):
            label = members[0]["component_label"]
            if label == "conflicting_component":
                conflict_audit.append(
                    {
                        "evaluation_component_id": cid,
                        "component_metric_excluded": True,
                        "members": [
                            {
                                "sample_eval_code": m["sample_eval_code"],
                                "manual_occurrence_decision": m[
                                    "manual_occurrence_decision"
                                ],
                                "clean_positive": m["clean_positive"],
                                "clean_negative": m["clean_negative"],
                                "max_individual_cosine": m["max_individual_cosine"],
                                "best_gallery_anchor_id": m["best_gallery_anchor_id"],
                                "individual_cosine_scores": m[
                                    "individual_cosine_scores"
                                ],
                                "segment_rank_if_eligible": next(
                                    (
                                        r["rank"]
                                        for r in seg_ranked
                                        if r["sample_eval_code"] == m["sample_eval_code"]
                                    ),
                                    None,
                                ),
                            }
                            for m in members
                        ],
                    }
                )
                continue
            if label not in {"positive_component", "negative_component"}:
                continue
            eligible_members = [m for m in members if m["retrieval_metric_eligible"]]
            if not eligible_members:
                continue
            scores = [m["max_individual_cosine"] for m in eligible_members]
            winning = max(
                eligible_members,
                key=lambda m: (
                    m["max_individual_cosine"],
                    # deterministic: prefer higher score then code asc for tie on max?
                    # max() with key - for ties Python keeps first; sort explicitly
                ),
            )
            winning = sorted(
                eligible_members,
                key=lambda m: (-m["max_individual_cosine"], m["sample_eval_code"]),
            )[0]
            clean_comp_rows.append(
                {
                    "evaluation_component_id": cid,
                    "component_label": label,
                    "binary_label": 1 if label == "positive_component" else 0,
                    "component_primary_score": float(max(scores)),
                    "component_median_score": float(np.median(scores)),
                    "winning_segment_code": winning["sample_eval_code"],
                    "member_segment_codes": [
                        m["sample_eval_code"] for m in eligible_members
                    ],
                    "all_member_codes": [m["sample_eval_code"] for m in members],
                    "excluded_member_codes": [
                        m["sample_eval_code"]
                        for m in members
                        if not m["retrieval_metric_eligible"]
                    ],
                    "best_gallery_anchor_id": winning["best_gallery_anchor_id"],
                    "top3_mean_max": float(
                        max(m["top3_mean_individual_cosine"] for m in eligible_members)
                    ),
                    "top3_mean_median": float(
                        np.median(
                            [m["top3_mean_individual_cosine"] for m in eligible_members]
                        )
                    ),
                    "centroid_max": float(
                        max(m["centroid_cosine"] for m in eligible_members)
                    ),
                    "centroid_median": float(
                        np.median([m["centroid_cosine"] for m in eligible_members])
                    ),
                    "medoid_max": float(
                        max(m["medoid_cosine"] for m in eligible_members)
                    ),
                    "medoid_median": float(
                        np.median([m["medoid_cosine"] for m in eligible_members])
                    ),
                    "mean_individual_max": float(
                        max(m["mean_individual_cosine"] for m in eligible_members)
                    ),
                    "mean_individual_median": float(
                        np.median(
                            [m["mean_individual_cosine"] for m in eligible_members]
                        )
                    ),
                }
            )
        if len(clean_comp_rows) != 99:
            raise RetrievalEvalError(f"clean components != 99: {len(clean_comp_rows)}")
        cpos = sum(1 for r in clean_comp_rows if r["binary_label"] == 1)
        cneg = sum(1 for r in clean_comp_rows if r["binary_label"] == 0)
        if cpos != 4 or cneg != 95:
            raise RetrievalEvalError(f"component pos/neg {cpos}/{cneg}")

        comp_ranked = rank_rows(
            clean_comp_rows,
            score_key="component_primary_score",
            code_key="evaluation_component_id",
        )
        comp_csv = tmp / "rankings" / "target_001_component_primary_ranking.csv"
        comp_fields = [
            "rank",
            "evaluation_component_id",
            "binary_label",
            "component_primary_score",
            "component_median_score",
            "winning_segment_code",
            "member_segment_codes",
            "best_gallery_anchor_id",
            "top3_mean_max",
            "centroid_max",
            "medoid_max",
            "mean_individual_max",
        ]
        with comp_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=comp_fields)
            writer.writeheader()
            for row in comp_ranked:
                writer.writerow(
                    {
                        "rank": row["rank"],
                        "evaluation_component_id": row["evaluation_component_id"],
                        "binary_label": row["binary_label"],
                        "component_primary_score": f"{row['component_primary_score']:.8f}",
                        "component_median_score": f"{row['component_median_score']:.8f}",
                        "winning_segment_code": row["winning_segment_code"],
                        "member_segment_codes": "|".join(row["member_segment_codes"]),
                        "best_gallery_anchor_id": row["best_gallery_anchor_id"],
                        "top3_mean_max": f"{row['top3_mean_max']:.8f}",
                        "centroid_max": f"{row['centroid_max']:.8f}",
                        "medoid_max": f"{row['medoid_max']:.8f}",
                        "mean_individual_max": f"{row['mean_individual_max']:.8f}",
                    }
                )
        positive_comp_ranks = [
            {
                "evaluation_component_id": row["evaluation_component_id"],
                "rank": row["rank"],
                "max_score": row["component_primary_score"],
                "median_score": row["component_median_score"],
                "winning_segment_code": row["winning_segment_code"],
                "member_segment_codes": row["member_segment_codes"],
                "best_gallery_anchor_id": row["best_gallery_anchor_id"],
                "excluded_members_diagnostic": row["excluded_member_codes"],
            }
            for row in comp_ranked
            if row["binary_label"] == 1
        ]
        write_json(
            tmp / "rankings" / "target_001_positive_component_ranks.json",
            {
                "schema_version": "reid_target_001_positive_component_ranks_v1",
                "count": 4,
                "positives": positive_comp_ranks,
            },
        )
        comp_labels = [int(r["binary_label"]) for r in comp_ranked]
        comp_scores = [float(r["component_primary_score"]) for r in comp_ranked]
        component_metrics = ranking_metrics(comp_labels, comp_scores, n_pos=4)
        component_metrics["schema_version"] = (
            "reid_target_001_component_retrieval_metrics_v1"
        )
        component_metrics["ranking_score"] = "component_primary_score_max_eligible"
        component_metrics["metric_universe_size"] = 99
        component_metrics["excluded_components"] = 26
        component_metrics["conflicting_components"] = 4
        write_json(
            tmp / "metrics" / "target_001_component_retrieval_metrics.json",
            component_metrics,
        )
        write_json(
            tmp / "audit" / "target_001_conflicting_component_score_audit.json",
            {
                "schema_version": "reid_target_001_conflicting_component_score_audit_v1",
                "count": 4,
                "components": conflict_audit,
                "note": "Audit only; does not alter ground truth or component metrics.",
            },
        )

        # Secondary score diagnostics (segment universe)
        secondary_diag = {}
        for score_key in (
            "top3_mean_individual_cosine",
            "centroid_cosine",
            "medoid_cosine",
            "mean_individual_cosine",
        ):
            ranked = rank_rows(
                eligible_rows, score_key=score_key, code_key="sample_eval_code"
            )
            labels = [int(r["binary_label"]) for r in ranked]
            scores = [float(r[score_key]) for r in ranked]
            secondary_diag[score_key] = ranking_metrics(labels, scores, n_pos=8)
            secondary_diag[score_key]["note"] = (
                "Diagnostic only; cannot promote to primary or mutate gallery."
            )
        write_json(
            tmp / "metrics" / "target_001_secondary_score_diagnostics.json",
            {
                "schema_version": "reid_target_001_secondary_score_diagnostics_v1",
                "primary_unchanged": "max_individual_cosine",
                "diagnostics": secondary_diag,
            },
        )

        # Gallery anchor diagnostics
        best_all = Counter(r["best_gallery_anchor_id"] for r in full_rows)
        best_pos = Counter(
            r["best_gallery_anchor_id"] for r in full_rows if r["clean_positive"]
        )
        best_neg = Counter(
            r["best_gallery_anchor_id"] for r in full_rows if r["clean_negative"]
        )
        pos_spreads = []
        for r in full_rows:
            if not r["clean_positive"]:
                continue
            vals = list(r["individual_cosine_scores"].values())
            pos_spreads.append(
                {
                    "sample_eval_code": r["sample_eval_code"],
                    "max_individual": max(vals),
                    "min_individual": min(vals),
                    "spread": max(vals) - min(vals),
                    "best_anchor": r["best_gallery_anchor_id"],
                }
            )
        # Leave-one-anchor-out AP (diagnostic)
        loo = []
        for drop_idx, drop_id in enumerate(GALLERY_IDS):
            alt_rows = []
            for r in eligible_rows:
                scores = [
                    r["individual_cosine_scores"][aid]
                    for j, aid in enumerate(GALLERY_IDS)
                    if j != drop_idx
                ]
                alt_rows.append(
                    {
                        "sample_eval_code": r["sample_eval_code"],
                        "binary_label": r["binary_label"],
                        "loo_score": max(scores),
                    }
                )
            ranked = rank_rows(alt_rows, score_key="loo_score", code_key="sample_eval_code")
            labels = [int(r["binary_label"]) for r in ranked]
            scores = [float(r["loo_score"]) for r in ranked]
            mets = ranking_metrics(labels, scores, n_pos=8)
            loo.append(
                {
                    "dropped_anchor_id": drop_id,
                    "Average_Precision": mets["Average_Precision"],
                    "Recall@5": mets["Recall@5"],
                    "Recall@10": mets["Recall@10"],
                    "MRR": mets["MRR"],
                }
            )
        write_json(
            tmp / "audit" / "target_001_gallery_anchor_scoring_diagnostics.json",
            {
                "schema_version": "reid_target_001_gallery_anchor_scoring_diagnostics_v1",
                "best_anchor_counts_all_queries": dict(best_all),
                "best_anchor_counts_positive_eligible": dict(best_pos),
                "best_anchor_counts_negative_eligible": dict(best_neg),
                "positive_query_gallery_spreads": pos_spreads,
                "anchor_014_behavior": {
                    "best_count_all": best_all.get("target_001_ext_anchor_014", 0),
                    "best_count_positive": best_pos.get("target_001_ext_anchor_014", 0),
                    "best_count_negative": best_neg.get("target_001_ext_anchor_014", 0),
                },
                "medoid_anchor_004_behavior": {
                    "best_count_all": best_all.get("target_001_ext_anchor_004", 0),
                    "best_count_positive": best_pos.get("target_001_ext_anchor_004", 0),
                    "best_count_negative": best_neg.get("target_001_ext_anchor_004", 0),
                },
                "leave_one_anchor_out_diagnostic": {
                    "note": (
                        "Diagnostic only; frozen gallery not mutated; official "
                        "primary unchanged; no automatic anchor removal."
                    ),
                    "runs": loo,
                },
            },
        )

        # Excluded item diagnostics
        excluded = [r for r in full_rows if not r["retrieval_metric_eligible"]]
        uncertain = [
            r for r in excluded if r["manual_occurrence_decision"] == "uncertain"
        ]
        ambiguous = [
            r
            for r in excluded
            if r["manual_occurrence_decision"] == "multi_person_ambiguous"
        ]
        write_json(
            tmp / "audit" / "target_001_excluded_item_score_diagnostics.json",
            {
                "schema_version": "reid_target_001_excluded_item_score_diagnostics_v1",
                "excluded_count": 32,
                "uncertain_count": len(uncertain),
                "ambiguous_count": len(ambiguous),
                "target_present_ambiguous": list(TARGET_PRESENT_AMBIGUOUS),
                "official_metric_inclusion": False,
                "score_summary": {
                    "mean_primary": float(
                        np.mean([r["max_individual_cosine"] for r in excluded])
                    )
                    if excluded
                    else None,
                    "max_primary": float(
                        max(r["max_individual_cosine"] for r in excluded)
                    )
                    if excluded
                    else None,
                    "min_primary": float(
                        min(r["max_individual_cosine"] for r in excluded)
                    )
                    if excluded
                    else None,
                },
                "items": [
                    {
                        "sample_eval_code": r["sample_eval_code"],
                        "manual_occurrence_decision": r["manual_occurrence_decision"],
                        "metric_exclusion_reason": r["metric_exclusion_reason"],
                        "max_individual_cosine": r["max_individual_cosine"],
                        "best_gallery_anchor_id": r["best_gallery_anchor_id"],
                        "target_present": r["target_present"],
                    }
                    for r in excluded
                ],
            },
        )

        outcome = select_outcome(
            segment_metrics=segment_metrics,
            component_metrics=component_metrics,
            amended_strong=f2b["amended"]["INDEPENDENT_RETRIEVAL_STRONG_SIGNAL"],
        )
        write_json(
            tmp / "evaluation" / "target_001_independent_retrieval_outcome.json",
            outcome,
        )

        contract = {
            "schema_version": "reid_stage5d_f3_independent_retrieval_evaluation_contract_v1",
            "final_status": FINAL_STATUS,
            "descriptive_outcome": outcome["descriptive_outcome"],
            "exact_next_gate": outcome["exact_next_gate"],
            "target_id": TARGET_ID,
            "queries_scored": 150,
            "individual_cosine_shape": [150, 7],
            "official_segment_metric_rows": 118,
            "official_component_metric_rows": 99,
            "gallery_members": 7,
            "new_embeddings": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "automatic_gallery_growth": False,
            "two_pass_deterministic": True,
            "two_pass_max_abs_diff": max_abs,
            "sample_video_decoded": False,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3_contract.json", contract)

        summary = {
            "schema_version": "reid_stage5d_f3_independent_retrieval_evaluation_summary_v1",
            "final_status": FINAL_STATUS,
            "descriptive_outcome": outcome["descriptive_outcome"],
            "exact_next_gate": outcome["exact_next_gate"],
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "queries_scored": 150,
            "individual_cosine_shape": [150, 7],
            "official_segment_metric_rows": 118,
            "official_component_metric_rows": 99,
            "segment_metrics": {
                "Recall@1": segment_metrics["Recall@1"],
                "Recall@3": segment_metrics["Recall@3"],
                "Recall@5": segment_metrics["Recall@5"],
                "Recall@10": segment_metrics["Recall@10"],
                "MRR": segment_metrics["MRR"],
                "Average_Precision": segment_metrics["Average_Precision"],
                "AUROC": segment_metrics["AUROC"],
                "AUPRC": segment_metrics["AUPRC"],
                "separation_margin": segment_metrics[
                    "separation_margin_min_pos_minus_max_neg"
                ],
                "every_positive_rank": segment_metrics["every_positive_rank"],
            },
            "component_metrics": {
                "Recall@1": component_metrics["Recall@1"],
                "Recall@3": component_metrics["Recall@3"],
                "Recall@5": component_metrics["Recall@5"],
                "Recall@10": component_metrics["Recall@10"],
                "MRR": component_metrics["MRR"],
                "Average_Precision": component_metrics["Average_Precision"],
                "AUROC": component_metrics["AUROC"],
                "AUPRC": component_metrics["AUPRC"],
                "separation_margin": component_metrics[
                    "separation_margin_min_pos_minus_max_neg"
                ],
                "every_positive_rank": component_metrics["every_positive_rank"],
            },
            "positive_segment_ranks": {
                p["sample_eval_code"]: p["absolute_rank"] for p in positive_segment_ranks
            },
            "positive_component_ranks": {
                p["evaluation_component_id"]: p["rank"] for p in positive_comp_ranks
            },
            "strong_signal_checks": outcome["strong_signal_checks"],
            "two_pass_deterministic": True,
            "two_pass_max_abs_diff": max_abs,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_members": 7,
            "gallery_mutation": False,
            "automatic_gallery_growth": False,
            "new_embeddings": 0,
            "sample_video_decoded": False,
            "network_used": False,
            "package_environment_changed": False,
            "f2b_snapshot_sha256": f2b["snapshot_sha256"],
            "f2a_snapshot_sha256": f2a["snapshot_sha256"],
            "gallery_snapshot_sha256": gallery["snapshot_sha256"],
            "sample_embedding_sha256": sample["artifact_sha256"],
            "sample_sha256": config["evaluation_source"]["expected_sha256"],
            "sklearn_version": SKLEARN_VERSION,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3_summary.json", summary)

        # budget
        npy_files = list(tmp.rglob("*.npy"))
        if len(npy_files) != 1:
            raise RetrievalEvalError(f"npy budget != 1: {len(npy_files)}")
        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")):
            raise RetrievalEvalError("png/mp4 forbidden")
        csv_files = list(tmp.rglob("*.csv"))
        if len(csv_files) != 2:
            raise RetrievalEvalError(f"csv budget != 2: {len(csv_files)}")

        n_before, listing_before = listing_sha(tmp)
        manifest = {
            "schema_version": "reid_stage5d_f3_independent_retrieval_evaluation_manifest_v1",
            "final_status": FINAL_STATUS,
            "descriptive_outcome": outcome["descriptive_outcome"],
            "project_head": head,
            "output_root": final_rel,
            "listing_file_count_before_manifest": n_before,
            "listing_sha256_before_manifest": listing_before,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3_manifest.json", manifest)
        n_final, listing_final = listing_sha(tmp)
        manifest["listing_file_count"] = n_final
        manifest["listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f3_manifest.json", manifest)
        summary["output_file_count"] = n_final
        summary["output_listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f3_summary.json", summary)

        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    # Upstream immutability
    if listing_sha(f2b["root"])[1] != f2b["listing_sha256"]:
        raise RetrievalEvalError("F2B mutated")
    if listing_sha(f2a["root"])[1] != f2a["listing_sha256"]:
        raise RetrievalEvalError("F2A mutated")
    if listing_sha(gallery["root"])[1] != gallery["listing_sha256"]:
        raise RetrievalEvalError("gallery mutated")
    if listing_sha(sample["f1_root"])[1] != sample["f1_listing_sha256"]:
        raise RetrievalEvalError("F1 mutated")
    if sha256_file(sample["artifact_path"]) != sample["artifact_sha256"]:
        raise RetrievalEvalError("sample embeddings mutated")
    if sha256_file(gallery["individual_path"]) != gallery["individual_sha256"]:
        raise RetrievalEvalError("gallery npy mutated")

    return load_json(final_dir / "stage5d_f3_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate target_001 independent sample retrieval."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to target_independent_retrieval_evaluation_stage5d_target_001.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(Path(args.config))
    except RetrievalEvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"status={summary['final_status']} "
        f"outcome={summary['descriptive_outcome']} "
        f"seg_R@5={summary['segment_metrics']['Recall@5']} "
        f"seg_AP={summary['segment_metrics']['Average_Precision']:.4f} "
        f"next_gate={summary['exact_next_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
