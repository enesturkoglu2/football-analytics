#!/usr/bin/env python3
"""ReID-R2D SportsReID SoccerNet single-crop football-domain ablation runner."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from football_analytics.reid.embedding import (  # noqa: E402
    MODEL_ID_SPORTSREID_SOCCERNET,
    embed_image_paths_with_model,
    load_reid_osnet_by_model_id,
)
from football_analytics.reid.model_registry import get_reid_model_spec  # noqa: E402
from football_analytics.reid.r2d_domain_ablation import (  # noqa: E402
    R2DError,
    candidate_review_outcome,
    classify_football_domain_outcome,
    evaluate_joined,
    next_gate_for_outcome,
    outcome_to_final_status,
    rank_primary,
    resolve_crop_file,
    score_queries_against_galleries,
    sha256_bytes,
    sha256_file,
    sha256_json,
    validate_embedding_matrix,
)

F3M = (
    PROJECT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_frozen_target_distractor_scoring_evaluation"
)
F3G = (
    PROJECT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_gallery_v2_and_distractor_gallery_v1"
)
FINAL_OUT = PROJECT / "outputs/reid/target_001_reid_r2d_sportsreid_single_crop_domain_ablation"
# Inference uses ONLY the sanitized registry checkpoint (~9MB). The original
# ~1.02GB SportsReID training checkpoint and Market1501 weights are never opened.
SANITIZED_SPORTSREID_SHA256 = "c61e0da2007f7c7f4d889cb68774dfeecf8c4c433e0bfe3858b48b8655f83e91"
SANITIZED_SPORTSREID_SIZE_BYTES = 8964749

SEARCH_ROOTS = [
    PROJECT / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_anchor_review_package",
    PROJECT / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_refinement_review_package",
    PROJECT / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_refinement_crop_review_package",
    F3G,
    PROJECT / "outputs/reid",
    PROJECT,
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def embed_paths_two_pass(model: Any, paths: list[str]) -> tuple[np.ndarray, float]:
    e1 = embed_image_paths_with_model(model, paths, batch_size=8)
    e2 = embed_image_paths_with_model(model, paths, batch_size=8)
    maxdiff = float(np.max(np.abs(e1 - e2)))
    return e1, maxdiff


def build_visuals(
    *,
    out_dir: Path,
    metrics_a: dict[str, Any],
    metrics_c: dict[str, Any],
    joined_c: list[dict[str, Any]],
) -> list[str]:
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    # 1 metric comparison
    keys = ["AP", "AUROC", "same_team_AUROC", "Recall@5", "Recall@10", "MRR"]
    x = np.arange(len(keys))
    a_vals = [float(metrics_a[k]) for k in keys]
    c_vals = [float(metrics_c[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, a_vals, 0.4, label="A Market1501")
    ax.bar(x + 0.2, c_vals, 0.4, label="C SportsReID")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20)
    ax.set_title("Market1501 vs SportsReID metrics (development holdout)")
    ax.legend()
    p = figs / "market1501_vs_sportsreid_metric_comparison.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    created.append(str(p))

    # 2 positive ranks
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics_a["positive_ranks"], marker="o", label="A")
    ax.plot(metrics_c["positive_ranks"], marker="o", label="C")
    ax.set_xlabel("positive index")
    ax.set_ylabel("rank (lower better)")
    ax.set_title("Positive rank comparison")
    ax.invert_yaxis()
    ax.legend()
    p = figs / "market1501_vs_sportsreid_positive_rank_comparison.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    created.append(str(p))

    # 3 score distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, pred in [
        ("target", lambda r: int(r["binary_clean_player_label"]) == 1),
        ("same_team", lambda r: bool(r.get("same_team_negative_cohort"))),
        ("other_team", lambda r: bool(r.get("other_team_negative_cohort"))),
    ]:
        vals = [float(r["S_primary"]) for r in joined_c if pred(r)]
        ax.hist(vals, bins=20, alpha=0.5, label=label)
    ax.set_title("SportsReID S=T_max-D_max score distribution")
    ax.legend()
    p = figs / "sportsreid_score_distribution.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    created.append(str(p))

    # 4 top20 composition
    comp = metrics_c["top20_composition"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(list(comp.keys()), list(comp.values()))
    ax.set_title("SportsReID top-20 composition")
    p = figs / "sportsreid_top20_composition.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    created.append(str(p))

    return created


def build_contact_sheet_and_video(
    *,
    out_dir: Path,
    joined_c: list[dict[str, Any]],
    query_by_id: dict[str, dict[str, Any]],
    target_crop_by_id: dict[str, Path],
    distractor_crop_by_id: dict[str, Path],
) -> tuple[str, str]:
    figs = out_dir / "figures"
    vids = out_dir / "videos"
    figs.mkdir(exist_ok=True)
    vids.mkdir(exist_ok=True)

    # contact sheet: first 12 by rank
    ranked = sorted(joined_c, key=lambda r: int(r["rank"]))[:12]
    cells = []
    for row in ranked:
        q = query_by_id[row["stable_query_id"]]
        q_img = cv2.imread(q["crop_path"], cv2.IMREAD_COLOR)
        t_img = cv2.imread(str(target_crop_by_id[row["T_max_member_id"]]), cv2.IMREAD_COLOR)
        d_img = cv2.imread(str(distractor_crop_by_id[row["D_max_member_id"]]), cv2.IMREAD_COLOR)
        for img in (q_img, t_img, d_img):
            if img is None:
                raise R2DError("contact sheet decode failed")
        q_r = cv2.resize(q_img, (64, 128))
        t_r = cv2.resize(t_img, (64, 128))
        d_r = cv2.resize(d_img, (64, 128))
        strip = np.concatenate([q_r, t_r, d_r], axis=1)
        label = f"r{row['rank']} {row['stable_query_id']}"
        cv2.putText(strip, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        cells.append(strip)
    while len(cells) < 12:
        cells.append(np.zeros_like(cells[0]))
    rows_img = [np.concatenate(cells[i : i + 3], axis=1) for i in range(0, 12, 3)]
    sheet = np.concatenate(rows_img, axis=0)
    sheet_path = figs / "sportsreid_query_best_target_best_distractor_contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)

    # video all 115
    video_path = vids / "sportsreid_gt_diagnostic_review.mp4"
    writer = None
    for row in sorted(joined_c, key=lambda r: int(r["rank"])):
        q = query_by_id[row["stable_query_id"]]
        q_img = cv2.imread(q["crop_path"], cv2.IMREAD_COLOR)
        t_img = cv2.imread(str(target_crop_by_id[row["T_max_member_id"]]), cv2.IMREAD_COLOR)
        d_img = cv2.imread(str(distractor_crop_by_id[row["D_max_member_id"]]), cv2.IMREAD_COLOR)
        if q_img is None or t_img is None or d_img is None:
            raise R2DError("video decode failed")
        q_r = cv2.resize(q_img, (128, 256))
        t_r = cv2.resize(t_img, (128, 256))
        d_r = cv2.resize(d_img, (128, 256))
        frame = np.concatenate([q_r, t_r, d_r], axis=1)
        canvas = np.zeros((320, frame.shape[1], 3), dtype=np.uint8)
        canvas[40:296, :, :] = frame
        gt = "target" if int(row["binary_clean_player_label"]) == 1 else (
            "same_team" if row.get("same_team_negative_cohort") else "other_team"
        )
        lines = [
            "DEVELOPMENT GT OVERLAY — NOT MODEL INPUT",
            f"id={row['stable_query_id']} rank={row['rank']} gt={gt}",
            f"T={row['T_max']:.3f}({row['T_max_member_id']}) D={row['D_max']:.3f}({row['D_max_member_id']}) S={row['S_primary']:.3f}",
        ]
        y = 14
        for line in lines:
            cv2.putText(canvas, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y += 12
        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                2.0,
                (w, h),
            )
        for _ in range(2):
            writer.write(canvas)
    if writer is None:
        raise R2DError("video writer empty")
    writer.release()
    return str(sheet_path), str(video_path)


def main() -> int:
    if FINAL_OUT.exists():
        print("BLOCKED_R2D_OUTPUT_ROOT_ALREADY_EXISTS")
        return 2

    head = subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != "313ef9ea08667318546fa2541c588e1d939bb5b9":
        print("BLOCKED_R2D_SOURCE_CONTRACT_MISMATCH", head)
        return 2

    token = uuid.uuid4().hex[:10]
    tmp = PROJECT / f"outputs/reid/_tmp_target_001_reid_r2d_sportsreid_single_crop_domain_ablation_{token}"
    tmp.mkdir(parents=True, exist_ok=False)
    (tmp / "figures").mkdir()
    (tmp / "videos").mkdir()
    (tmp / "embeddings").mkdir()
    (tmp / "scoring").mkdir()
    (tmp / "evaluation").mkdir()
    (tmp / "runtime").mkdir()

    started = time.perf_counter()
    try:
        # --- model contract (sanitized SportsReID only; never open full ~1GB ckpt) ---
        spec = get_reid_model_spec(MODEL_ID_SPORTSREID_SOCCERNET)
        ckpt = Path(str(spec["checkpoint_path"]))
        if "sanitized" not in str(ckpt) or "state_dict_only" not in ckpt.name:
            raise R2DError("SportsReID checkpoint must be sanitized state_dict_only asset")
        if "_quarantine_" in str(ckpt):
            raise R2DError("quarantine / full SportsReID checkpoint is forbidden")
        if not ckpt.is_file() or sha256_file(ckpt) != str(spec["sha256"]).lower():
            raise R2DError("SportsReID checkpoint SHA/path mismatch")
        if int(ckpt.stat().st_size) != int(spec["size_bytes"]):
            raise R2DError("SportsReID checkpoint size mismatch")
        if sha256_file(ckpt) != SANITIZED_SPORTSREID_SHA256:
            raise R2DError("SportsReID checkpoint SHA != pre-registered sanitized asset")
        if int(ckpt.stat().st_size) != SANITIZED_SPORTSREID_SIZE_BYTES:
            raise R2DError("SportsReID checkpoint size != pre-registered sanitized asset")
        if int(ckpt.stat().st_size) >= 100_000_000:
            raise R2DError("checkpoint too large; full training checkpoint forbidden")
        rejected_paths = [
            str(p) for p in (spec.get("rejected_checkpoint_paths") or [])
        ]

        # --- frozen inputs ---
        f3m_summary = json.loads((F3M / "stage5d_f3m_summary.json").read_text(encoding="utf-8"))
        if (
            int(f3m_summary["complete_universe"]) != 243
            or int(f3m_summary["scoreable"]) != 115
            or int(f3m_summary["positive"]) != 10
            or int(f3m_summary["same_team_negative"]) != 55
            or int(f3m_summary["other_team_negative"]) != 50
        ):
            raise R2DError("frozen universe count mismatch")

        target_rows = _load_jsonl(F3G / "target_gallery_v2/target_001_gallery_v2_member_inventory.jsonl")
        distractor_rows = _load_jsonl(
            F3G / "distractor_gallery_v1/target_001_same_team_distractor_member_inventory.jsonl"
        )
        query_proj = _load_jsonl(
            F3M / "pre_execution/target_001_holdout_v2_scoreable_query_input_projection.jsonl"
        )
        if len(target_rows) != 13 or len(distractor_rows) != 23 or len(query_proj) != 115:
            raise R2DError("membership count mismatch")

        target_members = []
        target_crop_by_id: dict[str, Path] = {}
        for row in sorted(target_rows, key=lambda r: int(r["gallery_row_index"])):
            path = resolve_crop_file(row["crop_path"], SEARCH_ROOTS, PROJECT)
            digest = sha256_file(path)
            if digest != str(row["crop_sha256"]).lower():
                raise R2DError(f"target crop SHA mismatch: {row['member_id']}")
            target_crop_by_id[row["member_id"]] = path
            target_members.append(
                {
                    "member_id": row["member_id"],
                    "crop_path": str(path),
                    "crop_sha256": digest,
                    "gallery_row_index": int(row["gallery_row_index"]),
                }
            )

        distractor_members = []
        distractor_crop_by_id: dict[str, Path] = {}
        for row in sorted(distractor_rows, key=lambda r: int(r["gallery_row_index"])):
            path = resolve_crop_file(row["crop_path"], SEARCH_ROOTS, PROJECT)
            digest = sha256_file(path)
            if digest != str(row["crop_sha256"]).lower():
                raise R2DError(f"distractor crop SHA mismatch: {row['member_id']}")
            distractor_crop_by_id[row["member_id"]] = path
            distractor_members.append(
                {
                    "member_id": row["member_id"],
                    "crop_path": str(path),
                    "crop_sha256": digest,
                    "gallery_row_index": int(row["gallery_row_index"]),
                }
            )

        if set(m["member_id"] for m in target_members) & set(
            m["member_id"] for m in distractor_members
        ):
            raise R2DError("target/distractor membership overlap")

        query_members = []
        query_by_id: dict[str, dict[str, Any]] = {}
        for row in sorted(query_proj, key=lambda r: int(r["query_order_index"])):
            path = Path(row["representative_crop_path_absolute"]).resolve()
            if not path.is_file():
                raise R2DError(f"query crop missing: {path}")
            digest = sha256_file(path)
            if digest != str(row["representative_crop_sha256"]).lower():
                raise R2DError(f"query crop SHA mismatch: {row['stable_query_id']}")
            item = {
                "stable_query_id": row["stable_query_id"],
                "segment_id": row["segment_id"],
                "component_id": row["component_id"],
                "query_order_index": int(row["query_order_index"]),
                "frame_index": int(row["source_frame_index"]),
                "crop_path": str(path),
                "crop_sha256": digest,
            }
            query_members.append(item)
            query_by_id[item["stable_query_id"]] = item

        if len({q["stable_query_id"] for q in query_members}) != 115:
            raise R2DError("duplicate query IDs")

        # Pre-GT seal (no GT fields)
        pre_gt_seal = {
            "schema_version": "r2d_pre_gt_experiment_seal_v1",
            "query_universe_sha256": sha256_json(
                [{"stable_query_id": q["stable_query_id"], "crop_sha256": q["crop_sha256"]} for q in query_members]
            ),
            "target_gallery_membership_sha256": sha256_json(
                [{"member_id": m["member_id"], "crop_sha256": m["crop_sha256"]} for m in target_members]
            ),
            "distractor_membership_sha256": sha256_json(
                [{"member_id": m["member_id"], "crop_sha256": m["crop_sha256"]} for m in distractor_members]
            ),
            "crop_manifest_sha256": sha256_file(
                F3M / "pre_execution/target_001_holdout_v2_scoreable_query_input_projection.jsonl"
            ),
            "model_id": MODEL_ID_SPORTSREID_SOCCERNET,
            "checkpoint_sha256": str(spec["sha256"]).lower(),
            "preprocessing_contract": "bgr_to_rgb_256x128_imagenet",
            "embedding_dimension": 512,
            "scoring_formula": "S = T_max - D_max",
            "tie_break": [
                "primary_score_descending",
                "T_max_descending",
                "D_max_ascending",
                "query_stable_id_ascending",
            ],
            "metric_set": [
                "Recall@1/3/5/10/20",
                "MRR",
                "AP",
                "AUPRC",
                "AUROC",
                "margin",
                "same_team_AUROC/AP",
                "other_team_AUROC/AP",
            ],
            "outcome_rules": {
                "STRONG": {
                    "AP_delta": 0.20,
                    "AUROC_delta": 0.15,
                    "same_team_AUROC_delta": 0.10,
                    "Recall@10": 0.60,
                    "Recall@5": 0.30,
                }
            },
            "no_threshold": True,
            "no_query_drop": True,
            "no_gallery_change": True,
            "holdout_role": "development_and_error_analysis_only",
            "gt_loaded": False,
        }
        seal_bytes = json.dumps(pre_gt_seal, sort_keys=True, separators=(",", ":")).encode("utf-8")
        pre_gt_seal_sha = sha256_bytes(seal_bytes)
        _write_json(tmp / "r2d_pre_gt_experiment_seal.json", pre_gt_seal)
        _write_json(
            tmp / "input_manifests.json",
            {
                "target_members": target_members,
                "distractor_members": distractor_members,
                "query_members": query_members,
            },
        )

        # Active path: load SportsReID only
        assert "football_analytics.reid.multiframe_r2b" not in sys.modules
        loaded = load_reid_osnet_by_model_id(MODEL_ID_SPORTSREID_SOCCERNET)
        if loaded["model_id"] != MODEL_ID_SPORTSREID_SOCCERNET:
            raise R2DError("model id mismatch")
        if loaded["checkpoint_sha256"] != str(spec["sha256"]).lower():
            raise R2DError("loaded checkpoint SHA mismatch")
        if not loaded["weights_only"]:
            raise R2DError("weights_only required")
        assert "football_analytics.reid.multiframe_r2b" not in sys.modules

        model = loaded["model"]

        # Gallery embed
        t_paths = [m["crop_path"] for m in target_members]
        T, t_diff = embed_paths_two_pass(model, t_paths)
        t_rep = validate_embedding_matrix(T, expected_rows=13, max_abs_diff=t_diff)
        np.save(tmp / "embeddings/target_gallery_embeddings.npy", T)
        _write_jsonl(
            tmp / "embeddings/target_gallery_embedding_manifest.jsonl",
            [
                {
                    **m,
                    "embedding_row": i,
                    "embedding_dim": 512,
                    "model_id": MODEL_ID_SPORTSREID_SOCCERNET,
                    "checkpoint_sha256": loaded["checkpoint_sha256"],
                }
                for i, m in enumerate(target_members)
            ],
        )
        _write_json(tmp / "embeddings/target_gallery_determinism.json", t_rep)

        # Distractor embed
        d_paths = [m["crop_path"] for m in distractor_members]
        D, d_diff = embed_paths_two_pass(model, d_paths)
        d_rep = validate_embedding_matrix(D, expected_rows=23, max_abs_diff=d_diff)
        np.save(tmp / "embeddings/distractor_gallery_embeddings.npy", D)
        _write_jsonl(
            tmp / "embeddings/distractor_gallery_embedding_manifest.jsonl",
            [
                {
                    **m,
                    "embedding_row": i,
                    "embedding_dim": 512,
                    "model_id": MODEL_ID_SPORTSREID_SOCCERNET,
                    "checkpoint_sha256": loaded["checkpoint_sha256"],
                }
                for i, m in enumerate(distractor_members)
            ],
        )
        _write_json(tmp / "embeddings/distractor_gallery_determinism.json", d_rep)

        # Query embed (order frozen by query_order_index; no GT)
        q_paths = [m["crop_path"] for m in query_members]
        Q, q_diff = embed_paths_two_pass(model, q_paths)
        q_rep = validate_embedding_matrix(Q, expected_rows=115, max_abs_diff=q_diff)
        np.save(tmp / "embeddings/query_embeddings.npy", Q)
        _write_jsonl(
            tmp / "embeddings/query_embedding_manifest.jsonl",
            [
                {
                    **m,
                    "embedding_row": i,
                    "embedding_dim": 512,
                    "model_id": MODEL_ID_SPORTSREID_SOCCERNET,
                    "checkpoint_sha256": loaded["checkpoint_sha256"],
                }
                for i, m in enumerate(query_members)
            ],
        )
        _write_json(tmp / "embeddings/query_determinism.json", q_rep)
        if Q.shape[0] != 115:
            raise R2DError("BLOCKED_R2D_QUERY_UNIVERSE_DRIFT")

        # Scoring without GT
        score_rows = score_queries_against_galleries(
            Q=Q,
            T=T,
            D=D,
            target_ids=[m["member_id"] for m in target_members],
            distractor_ids=[m["member_id"] for m in distractor_members],
            query_meta=query_members,
        )
        for row in score_rows:
            row["model_id"] = MODEL_ID_SPORTSREID_SOCCERNET
            row["checkpoint_sha256"] = loaded["checkpoint_sha256"]
        ranked = rank_primary(score_rows)
        _write_jsonl(tmp / "scoring/pre_gt_primary_scores.jsonl", score_rows)
        _write_jsonl(tmp / "scoring/pre_gt_primary_ranking.jsonl", ranked)
        np.save(tmp / "scoring/target_cosine.npy", (Q @ T.T).astype(np.float32))
        np.save(tmp / "scoring/distractor_cosine.npy", (Q @ D.T).astype(np.float32))

        score_seal = {
            "schema_version": "r2d_pre_gt_score_result_seal_v1",
            "n_scores": len(ranked),
            "score_sha256": sha256_json(
                [
                    {
                        "stable_query_id": r["stable_query_id"],
                        "S_primary": r["S_primary"],
                        "T_max": r["T_max"],
                        "D_max": r["D_max"],
                        "rank": r["rank"],
                    }
                    for r in ranked
                ]
            ),
            "gt_loaded": False,
            "model_id": MODEL_ID_SPORTSREID_SOCCERNET,
            "checkpoint_sha256": loaded["checkpoint_sha256"],
            "pre_gt_experiment_seal_sha256": pre_gt_seal_sha,
        }
        score_seal["seal_sha256"] = sha256_json({k: v for k, v in score_seal.items() if k != "seal_sha256"})
        _write_json(tmp / "r2d_pre_gt_score_result_seal.json", score_seal)

        # GT open only now
        gt_join_frozen = _load_jsonl(
            F3M / "evaluation/target_001_holdout_v2_scored_ground_truth_join.jsonl"
        )
        gt_by_id = {r["stable_query_id"]: r for r in gt_join_frozen}
        joined_c = []
        for row in ranked:
            gt = gt_by_id[row["stable_query_id"]]
            joined_c.append(
                {
                    **row,
                    "binary_clean_player_label": int(gt["binary_clean_player_label"]),
                    "manual_ground_truth_decision": gt["manual_ground_truth_decision"],
                    "same_team_negative_cohort": bool(gt["same_team_negative_cohort"]),
                    "other_team_negative_cohort": bool(gt["other_team_negative_cohort"]),
                }
            )
        if len(joined_c) != 115:
            raise R2DError("joined query count drift")
        _write_jsonl(tmp / "evaluation/c_scored_ground_truth_join.jsonl", joined_c)

        metrics_c = evaluate_joined(joined_c)

        # Frozen A metrics from sealed F3M scores (no re-inference/re-scoring of T/D/S)
        metrics_a = evaluate_joined(gt_join_frozen)
        metrics_a["frozen_not_rescored"] = True
        metrics_a["source"] = str(F3M / "stage5d_f3m_summary.json")
        # Cross-check against F3M summary primary numbers
        if not math.isclose(float(metrics_a["AP"]), float(f3m_summary["segment_metrics"]["AP"]), rel_tol=0, abs_tol=1e-9):
            raise R2DError("frozen A AP mismatch vs F3M summary")
        if not math.isclose(float(metrics_a["AUROC"]), float(f3m_summary["segment_metrics"]["AUROC"]), rel_tol=0, abs_tol=1e-9):
            raise R2DError("frozen A AUROC mismatch vs F3M summary")

        deltas = {
            "AP": float(metrics_c["AP"]) - float(metrics_a["AP"]),
            "AUROC": float(metrics_c["AUROC"]) - float(metrics_a["AUROC"]),
            "same_team_AUROC": float(metrics_c["same_team_AUROC"]) - float(metrics_a["same_team_AUROC"]),
            "Recall@10": float(metrics_c["Recall@10"]) - float(metrics_a["Recall@10"]),
            "positive_median_rank": float(metrics_c["positive_median_rank"])
            - float(metrics_a["positive_median_rank"]),
            "margin": float(metrics_c["margin"]) - float(metrics_a["margin"]),
        }
        outcome = classify_football_domain_outcome(
            metrics_a=metrics_a,
            metrics_c=metrics_c,
            query_drop=0,
            deterministic=True,
        )
        cand = candidate_review_outcome(metrics_c)
        final_status = outcome_to_final_status(outcome)
        next_gate = next_gate_for_outcome(outcome)

        comparison = {
            "A": metrics_a,
            "C": metrics_c,
            "deltas": deltas,
            "outcome": outcome,
            "candidate_review_outcome": cand,
            "query_drop": 0,
        }
        _write_json(tmp / "evaluation/a_vs_c_comparison.json", comparison)
        _write_json(tmp / "evaluation/metrics_C.json", metrics_c)
        _write_json(tmp / "evaluation/metrics_A_frozen.json", metrics_a)
        _write_json(
            tmp / "evaluation/candidate_review_usefulness.json",
            {
                "outcome": cand,
                "Recall@1": metrics_c["Recall@1"],
                "Recall@3": metrics_c["Recall@3"],
                "Recall@5": metrics_c["Recall@5"],
                "Recall@10": metrics_c["Recall@10"],
                "positive_ranks": metrics_c["positive_ranks"],
                "positives_in_top_3": sum(1 for r in metrics_c["positive_ranks"] if r <= 3),
                "positives_in_top_5": sum(1 for r in metrics_c["positive_ranks"] if r <= 5),
                "positives_in_top_10": sum(1 for r in metrics_c["positive_ranks"] if r <= 10),
                "automatic_identity_sufficient": False,
                "development_only": True,
            },
        )

        fig_paths = build_visuals(
            out_dir=tmp, metrics_a=metrics_a, metrics_c=metrics_c, joined_c=joined_c
        )
        sheet, video = build_contact_sheet_and_video(
            out_dir=tmp,
            joined_c=joined_c,
            query_by_id=query_by_id,
            target_crop_by_id=target_crop_by_id,
            distractor_crop_by_id=distractor_crop_by_id,
        )

        active_path = {
            "command": "python scripts/run_reid_r2d_sportsreid_single_crop_domain_ablation.py",
            "runner": "scripts/run_reid_r2d_sportsreid_single_crop_domain_ablation.py",
            "active_model_id": MODEL_ID_SPORTSREID_SOCCERNET,
            "safe_loader": "football_analytics.reid.safe_checkpoint / load_reid_osnet_by_model_id",
            "model_builder": "torchreid osnet_x1_0 via embedding.build_osnet_cpu_model",
            "preprocessing": "embedding.load_and_preprocess_crop",
            "checkpoint_path": str(ckpt),
            "checkpoint_sha256": loaded["checkpoint_sha256"],
            "market1501_opened": False,
            "full_sportsreid_opened": False,
            "multiframe_r2b_imported": "football_analytics.reid.multiframe_r2b" in sys.modules,
            "r2b_runner_executed": False,
            "silent_fallback": False,
        }
        if active_path["multiframe_r2b_imported"]:
            raise R2DError("BLOCKED_R2D_ACTIVE_CODE_PATH_AMBIGUOUS")
        _write_json(tmp / "runtime/active_execution_path.json", active_path)
        _write_json(
            tmp / "runtime/network_and_asset_access_audit.json",
            {
                "network": False,
                "dataset_download": False,
                "package_change": False,
                "active_checkpoint_path": str(ckpt),
                "active_checkpoint_sha256": loaded["checkpoint_sha256"],
                "active_checkpoint_size_bytes": int(ckpt.stat().st_size),
                "rejected_checkpoint_paths_not_opened": rejected_paths,
                "market1501_used": False,
                "full_sportsreid_used": False,
                "weights_only": True,
                "allowlist_used": False,
            },
        )

        contract = {
            "gate": "REID_R2D_SOCCERNET_OSNET_SINGLE_CROP_DOMAIN_ABLATION",
            "head": head,
            "model_id": MODEL_ID_SPORTSREID_SOCCERNET,
            "experiment_a_frozen_not_rescored": True,
            "scoring": "S=T_max-D_max",
            "holdout_role": "development_and_error_analysis_only",
            "multi_frame": False,
            "outcome_rules_pre_registered": True,
        }
        (tmp / "effective_r2d_contract.yaml").write_text(
            yaml.safe_dump(contract, sort_keys=False), encoding="utf-8"
        )

        elapsed = time.perf_counter() - started
        manifest = {
            "schema_version": "target_001_reid_r2d_manifest_v1",
            "final_status": final_status,
            "outcome": outcome,
            "exact_next_gate": next_gate,
            "head_before": head,
            "model_id": MODEL_ID_SPORTSREID_SOCCERNET,
            "checkpoint_path": str(ckpt),
            "checkpoint_sha256": loaded["checkpoint_sha256"],
            "checkpoint_size_bytes": int(ckpt.stat().st_size),
            "target_embed_count": 13,
            "distractor_embed_count": 23,
            "query_embed_count": 115,
            "query_drop": 0,
            "deterministic_max_abs_diff": {
                "target": t_diff,
                "distractor": d_diff,
                "query": q_diff,
            },
            "metrics_A": metrics_a,
            "metrics_C": metrics_c,
            "deltas": deltas,
            "candidate_review_outcome": cand,
            "frozen_A_not_rescored": True,
            "multi_frame_used": False,
            "gallery_gt_modified": False,
            "automatic_identity_assignment": False,
            "threshold_selected": False,
            "holdout_role": "development_and_error_analysis_only",
            "elapsed_sec": elapsed,
            "figures": fig_paths + [sheet],
            "video": video,
            "pre_gt_experiment_seal_sha256": pre_gt_seal_sha,
            "pre_gt_score_result_seal_sha256": score_seal["seal_sha256"],
        }
        _write_json(tmp / "target_001_reid_r2d_manifest.json", manifest)

        report = f"""# Target 001 — ReID-R2D SportsReID Single-Crop Domain Ablation Report

## 1. Final status
- **Status:** `{final_status}`
- **Outcome:** `{outcome}`
- **Exact next gate:** `{next_gate}`
- Holdout v2 development-only. SportsReID source metrics bizim proje metriklerimiz değildir.
- A yeniden skorlanmadı (`frozen_not_rescored=true`).
- Gallery membership değişmedi. Multi-frame kullanılmadı.
- Automatic identity assignment false. Threshold seçilmedi.
- Yeni bağımsız holdout hâlâ zorunlu.

## 2. Amaç ve deney sınırı
Football-domain SportsReID OSNet vs frozen Market1501 single-crop baseline; aynı 13/23/115 membership ve `S=T_max-D_max`.

## 3. Git/repo/environment integrity
- HEAD: `{head}`

## 4. Frozen Experiment A
- AP={metrics_a['AP']:.6f} AUROC={metrics_a['AUROC']:.6f} same-team AUROC={metrics_a['same_team_AUROC']:.6f}
- Recall@10={metrics_a['Recall@10']} positive ranks={metrics_a['positive_ranks']}
- median rank={metrics_a['positive_median_rank']}

## 5. SportsReID model/asset contract
- model_id=`{MODEL_ID_SPORTSREID_SOCCERNET}`
- SHA-256=`{loaded['checkpoint_sha256']}`
- weights_only=true; allowlist=false; full checkpoint unused

## 6. Immutable input contract
- target 13/13, distractor 23/23, query 115/115 crop SHA OK

## 7. Pre-GT seal
- experiment seal SHA `{pre_gt_seal_sha}`
- score seal SHA `{score_seal['seal_sha256']}`

## 8–10. Embeddings
- target/distractor/query: 13/23/115; max-abs-diff target={t_diff}, distractor={d_diff}, query={q_diff}

## 11. Frozen scoring
- `S=T_max-D_max`; tie-break frozen F3H/F3M contract

## 12–13. GT evaluation / A-vs-C
- C AP={metrics_c['AP']:.6f} (Δ {deltas['AP']:+.6f})
- C AUROC={metrics_c['AUROC']:.6f} (Δ {deltas['AUROC']:+.6f})
- C same-team AUROC={metrics_c['same_team_AUROC']:.6f} (Δ {deltas['same_team_AUROC']:+.6f})
- C Recall@1/3/5/10={metrics_c['Recall@1']}/{metrics_c['Recall@3']}/{metrics_c['Recall@5']}/{metrics_c['Recall@10']}
- C positive ranks={metrics_c['positive_ranks']}

## 14. Same-team confusion
- A same-team AUROC={metrics_a['same_team_AUROC']:.6f}
- C same-team AUROC={metrics_c['same_team_AUROC']:.6f}

## 15. Positive rank analysis
- A median={metrics_a['positive_median_rank']} C median={metrics_c['positive_median_rank']}

## 16. Candidate-review usefulness
- outcome=`{cand}`
- Automatic identity için yeterli mi? **Hayır** (development; threshold yok; assignment false)
- Human-in-the-loop candidate review için faydalı mı? `{cand}`

## 17. Diagnostic visualizations
- figures + `{Path(video).name}` (DEVELOPMENT GT OVERLAY — NOT MODEL INPUT); 115/115 coverage

## 18. Tests
- filled after suite

## 19. Runtime/network
- elapsed_sec={elapsed:.2f}; network/package/dataset=false

## 20. Limitations
- Development holdout only; embedding spaces not mixed; no deployment claim.

## 21. Exact next gate
`{next_gate}`
"""
        (tmp / "target_001_reid_r2d_sportsreid_single_crop_domain_ablation_report.md").write_text(
            report, encoding="utf-8"
        )

        shutil.move(str(tmp), str(FINAL_OUT))
        print(
            json.dumps(
                {
                    "final_status": final_status,
                    "outcome": outcome,
                    "next_gate": next_gate,
                    "candidate_review": cand,
                    "metrics_C": {
                        "AP": metrics_c["AP"],
                        "AUROC": metrics_c["AUROC"],
                        "Recall@10": metrics_c["Recall@10"],
                        "positive_ranks": metrics_c["positive_ranks"],
                    },
                }
            )
        )
        return 0
    except Exception as exc:
        # leave temp for debugging unless empty contract failure
        print(f"R2D_FAILED: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
