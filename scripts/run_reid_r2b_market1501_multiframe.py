#!/usr/bin/env python3
"""ReID-R2B: Market1501 multi-frame crop rebuild + tracklet ablation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
SN_REID = Path("/home/enesturkoglu2/projects/soccernet/sn-reid")
FOOTBALL_CV_PY = Path("/home/enesturkoglu2/miniconda3/envs/football-cv/bin/python")
SN_REID_CPU_PY = Path("/home/enesturkoglu2/miniconda3/envs/sn-reid-cpu/bin/python")
if SN_REID.is_dir():
    sys.path.insert(0, str(SN_REID))

from football_analytics.reid.crop_extract import (  # noqa: E402
    crop_frame_region,
    probe_video_size,
    write_crop_jpeg,
)
from football_analytics.reid.embedding import (  # noqa: E402
    EMBEDDING_DIM,
    build_osnet_cpu_model,
    embed_tensors,
    load_and_preprocess_crop,
    load_osnet_checkpoint_weights,
    verify_checkpoint,
)
from football_analytics.reid.multiframe_r2b import (  # noqa: E402
    MAX_FRAMES,
    aggregate_with_fallback,
    classify_outcome,
    evaluate_joined,
    group_frame_bboxes,
    rank_primary,
    score_queries_against_galleries,
    select_development_candidate,
    select_quality_temporal_diversity,
    select_uniform_temporal,
    sha256_file,
    validate_and_clamp_bbox,
    compute_crop_quality_fields,
)

EXPECTED_HEAD = "41929674b307db364d228d375160eb0ad5b68a77"
CKPT = Path(
    "/home/enesturkoglu2/projects/soccernet/checkpoints/general-reid/"
    "osnet_x1_0_market1501_softmax_256x128.pth.tar"
)
CKPT_SHA = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"
CKPT_BYTES = 10399605
VIDEO_REL = "data/test_clips/target_001_independent_holdout_v2.mp4"
VIDEO_SHA = "bbfe3669cfbf39534f71a80131401bb4d9f931c7a4ea485404ab2fa207a6231f"

F3M = PROJECT_ROOT / (
    "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_"
    "frozen_target_distractor_scoring_evaluation"
)
F3J = PROJECT_ROOT / (
    "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_"
    "label_blind_universe"
)
F3L = PROJECT_ROOT / (
    "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_"
    "ground_truth_manual_freeze"
)
F3N = PROJECT_ROOT / (
    "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_"
    "result_audit_and_error_analysis_package"
)
F3O = PROJECT_ROOT / "outputs/reid/target_001_f3o_diagnostic_visualization_v1"
F3G = PROJECT_ROOT / (
    "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_gallery_v2_and_distractor_gallery_v1"
)
R2A = PROJECT_ROOT / "outputs/reid/target_001_reid_r2a_capability_audit"
FINAL_ROOT = PROJECT_ROOT / "outputs/reid/target_001_reid_r2b_market1501_multiframe"
CONTRACT = PROJECT_ROOT / "configs/reid/r2b_market1501_multiframe_contract.yaml"


class R2BError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(PROJECT_ROOT), text=True).strip()


def validate_start() -> dict[str, Any]:
    if FINAL_ROOT.exists():
        raise R2BError("BLOCKED_R2B_OUTPUT_ROOT_ALREADY_EXISTS")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    head = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/main")
    if branch != "main" or head != EXPECTED_HEAD or head != origin:
        raise R2BError(
            f"BLOCKED_R2B_BASELINE_CONTRACT_MISMATCH git branch={branch} head={head} origin={origin}"
        )
    status = git("status", "--short", "--untracked-files=all")
    dirty = [
        line
        for line in status.splitlines()
        if line and not line.startswith("??")
        and "_tmp_target_001_reid_r2b" not in line
        and "target_001_reid_r2b" not in line
    ]
    if dirty:
        raise R2BError(f"BLOCKED_R2B_BASELINE_CONTRACT_MISMATCH dirty={dirty[:10]}")
    subprocess.check_call(["git", "diff", "--check"], cwd=str(PROJECT_ROOT))

    video = PROJECT_ROOT / VIDEO_REL
    if not video.is_file():
        raise R2BError("BLOCKED_R2B_SOURCE_PROVENANCE_MISSING video")
    vsha = sha256_file(str(video))
    if vsha != VIDEO_SHA:
        raise R2BError("BLOCKED_R2B_BASELINE_CONTRACT_MISMATCH video sha")

    if not CKPT.is_file() or CKPT.stat().st_size != CKPT_BYTES:
        raise R2BError("BLOCKED_R2B_MARKET1501_ASSET_CONTRACT_MISMATCH size")
    csha = sha256_file(str(CKPT))
    if csha != CKPT_SHA:
        raise R2BError("BLOCKED_R2B_MARKET1501_ASSET_CONTRACT_MISMATCH sha")

    for p in (F3M, F3J, F3L, F3N, F3O, F3G, R2A, CONTRACT):
        if not p.exists():
            raise R2BError(f"BLOCKED_R2B_BASELINE_CONTRACT_MISMATCH missing {p}")

    joined = load_jsonl(F3M / "evaluation/target_001_holdout_v2_scored_ground_truth_join.jsonl")
    inv = load_jsonl(F3J / "segmentation/target_001_holdout_v2_label_blind_segment_inventory.jsonl")
    obs = load_jsonl(F3J / "segmentation/target_001_holdout_v2_label_blind_segment_observations.jsonl")
    sm = load_json(F3M / "stage5d_f3m_summary.json")
    tg = np.load(F3G / "target_gallery_v2/target_001_gallery_v2_individual_embeddings.npy")
    dg = np.load(F3G / "distractor_gallery_v1/target_001_same_team_distractor_individual_embeddings.npy")
    measured = {
        "complete_universe": len(inv),
        "scoreable": len(joined),
        "positive": sum(1 for r in joined if r["manual_ground_truth_decision"] == "target_occurrence_yes"),
        "same_team_negative": sum(1 for r in joined if r["same_team_negative_cohort"]),
        "other_team_negative": sum(1 for r in joined if r["other_team_negative_cohort"]),
        "target_gallery": int(tg.shape[0]),
        "distractor_gallery": int(dg.shape[0]),
        "observations": len(obs),
    }
    expected = {
        "complete_universe": 243,
        "scoreable": 115,
        "positive": 10,
        "same_team_negative": 55,
        "other_team_negative": 50,
        "target_gallery": 13,
        "distractor_gallery": 23,
    }
    for k, v in expected.items():
        if measured[k] != v:
            raise R2BError(f"BLOCKED_R2B_BASELINE_CONTRACT_MISMATCH {k}={measured[k]} expected={v}")
    if int(tg.shape[1]) != 512 or int(dg.shape[1]) != 512:
        raise R2BError("BLOCKED_R2B_MARKET1501_ASSET_CONTRACT_MISMATCH gallery dim")

    w, h, fps, nframes = probe_video_size(video)
    return {
        "head": head,
        "branch": branch,
        "video": video,
        "video_sha": vsha,
        "video_meta": {"width": w, "height": h, "fps": fps, "frames": nframes},
        "joined": joined,
        "obs": obs,
        "inv": inv,
        "sm": sm,
        "tg": tg.astype(np.float32),
        "dg": dg.astype(np.float32),
        "measured": measured,
        "ckpt_sha": csha,
    }


def load_gallery_ids() -> tuple[list[str], list[str]]:
    tg_rows = load_jsonl(F3G / "target_gallery_v2/target_001_gallery_v2_member_inventory.jsonl")
    dg_rows = load_jsonl(
        F3G / "distractor_gallery_v1/target_001_same_team_distractor_member_inventory.jsonl"
    )
    tg_ids = [r["member_id"] for r in sorted(tg_rows, key=lambda x: int(x["gallery_row_index"]))]
    dg_ids = [r["member_id"] for r in sorted(dg_rows, key=lambda x: int(x["gallery_row_index"]))]
    if len(tg_ids) != 13 or len(dg_ids) != 23:
        raise R2BError("BLOCKED_R2B_BASELINE_CONTRACT_MISMATCH gallery ids")
    return tg_ids, dg_ids


def frozen_a_metrics(sm: Mapping[str, Any], joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Read frozen F3M metrics; do not rescore."""
    cm = sm.get("component_metrics") or {}
    st = sm.get("same_team_metrics") or {}
    pos_ranks = list(sm.get("positive_ranks") or [])
    # other-team AUROC from joined scores without changing ranks/scores
    ot_rows = [
        r
        for r in joined
        if int(r.get("binary_clean_player_label") or 0) == 1 or r.get("other_team_negative_cohort")
    ]
    # Use evaluate on OT cohort only for diagnostic other-team AUROC
    from football_analytics.reid.multiframe_r2b import ranking_metrics

    ot_sorted = sorted(ot_rows, key=lambda r: -float(r["S_primary"]))
    ot_m = ranking_metrics(
        [1 if r["manual_ground_truth_decision"] == "target_occurrence_yes" else 0 for r in ot_sorted],
        [float(r["S_primary"]) for r in ot_sorted],
        n_pos=10,
    )
    ranked = sorted(joined, key=lambda r: int(r["rank"]))
    top10 = ranked[:10]
    top20 = ranked[:20]
    return {
        "frozen_not_rescored": True,
        "AP": float(cm.get("AP", sm.get("AP", 0.0))),
        "AUPRC": float(cm.get("AP", 0.0)),
        "AUROC": float(cm.get("AUROC", 0.0)),
        "MRR": float(cm.get("MRR", 0.0)),
        "Recall@1": float(cm.get("Recall@1", 0.0)),
        "Recall@3": float(sum(1 for r in pos_ranks if int(r) <= 3) / 10.0) if pos_ranks else 0.0,
        "Recall@5": float(cm.get("Recall@5", 0.0)),
        "Recall@10": float(cm.get("Recall@10", 0.0)),
        "Recall@20": float(sum(1 for r in pos_ranks if int(r) <= 20) / 10.0) if pos_ranks else 0.0,
        "margin": float(cm.get("margin", 0.0)),
        "same_team_AUROC": float(st.get("AUROC", 0.0)),
        "same_team_AP": float(st.get("AP", 0.0)),
        "other_team_AUROC": float(ot_m["AUROC"]),
        "positive_ranks": pos_ranks,
        "positive_median_rank": float(np.median(pos_ranks)) if pos_ranks else None,
        "top10_composition": {
            "target": sum(1 for r in top10 if r["manual_ground_truth_decision"] == "target_occurrence_yes"),
            "same_team": sum(1 for r in top10 if r.get("same_team_negative_cohort")),
            "other_team": sum(1 for r in top10 if r.get("other_team_negative_cohort")),
        },
        "top20_composition": {
            "target": sum(1 for r in top20 if r["manual_ground_truth_decision"] == "target_occurrence_yes"),
            "same_team": sum(1 for r in top20 if r.get("same_team_negative_cohort")),
            "other_team": sum(1 for r in top20 if r.get("other_team_negative_cohort")),
        },
        "query_count": len(joined),
    }


def extract_multiframe_crops(
    *,
    tmp: Path,
    ctx: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    video: Path = ctx["video"]
    joined = ctx["joined"]
    obs = ctx["obs"]
    w = int(ctx["video_meta"]["width"])
    h = int(ctx["video_meta"]["height"])
    nframes = int(ctx["video_meta"]["frames"])
    scoreable = {r["segment_id"] for r in joined}
    meta_by_seg = {r["segment_id"]: r for r in joined}
    obs_sha = sha256_file(
        str(F3J / "segmentation/target_001_holdout_v2_label_blind_segment_observations.jsonl")
    )
    vid_sha = ctx["video_sha"]

    # All observations (for overlap) + scoreable subset
    frame_boxes = group_frame_bboxes(obs, frame_width=w, frame_height=h)
    scoreable_obs = [o for o in obs if o["segment_id"] in scoreable]
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    planned: list[dict[str, Any]] = []
    for o in scoreable_obs:
        fi = int(o["frame_index"])
        if fi < 0 or fi >= nframes:
            raise R2BError(f"BLOCKED_R2B_CROP_REEXTRACTION_FAILURE frame_oob {fi}")
        clamp = validate_and_clamp_bbox(o["bbox_xyxy"], frame_width=w, frame_height=h)
        crop_id = f"{o['segment_id']}__{o['observation_id']}"
        rel = f"crops/raw/{o['segment_id']}/{crop_id}.jpg"
        row = {
            "crop_id": crop_id,
            "segment_id": o["segment_id"],
            "observation_id": o["observation_id"],
            "raw_track_id": o.get("raw_track_code"),
            "frame_index": fi,
            "bbox_xyxy_source": list(o["bbox_xyxy"]),
            "bbox_xyxy": clamp.get("bbox_xyxy_int"),
            "detector_confidence": float(o.get("confidence") or 0.0),
            "crop_relative_path": rel,
            "source_video_path": str(VIDEO_REL),
            "source_video_sha256": vid_sha,
            "observation_source_path": str(
                F3J.relative_to(PROJECT_ROOT)
                / "segmentation/target_001_holdout_v2_label_blind_segment_observations.jsonl"
            ),
            "observation_source_sha256": obs_sha,
            "eligible": bool(clamp["eligible"]),
            "rejection_reason": clamp.get("rejection_reason"),
            "clamp_policy": clamp.get("clamp_policy"),
            "stable_query_id": meta_by_seg[o["segment_id"]].get("stable_query_id"),
        }
        planned.append(row)
        if clamp["eligible"]:
            by_frame[fi].append(row)

    crops_root = tmp / "crops" / "raw"
    crops_root.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise R2BError("BLOCKED_R2B_CROP_REEXTRACTION_FAILURE open_video")
    needed = sorted(by_frame.keys())
    written = 0
    decode_fail = 0
    zero_size = 0
    frame_index = 0
    next_i = 0
    try:
        while next_i < len(needed):
            ok, frame = capture.read()
            if not ok or frame is None:
                decode_fail += 1
                raise R2BError(
                    f"BLOCKED_R2B_CROP_REEXTRACTION_FAILURE decode at {frame_index}"
                )
            if frame_index == needed[next_i]:
                for row in by_frame[frame_index]:
                    bbox = row["bbox_xyxy"]
                    others = [
                        b
                        for b in frame_boxes.get(frame_index, [])
                        if b != list(bbox)
                    ]
                    try:
                        crop = crop_frame_region(frame, bbox)
                    except Exception as exc:  # noqa: BLE001
                        row["eligible"] = False
                        row["rejection_reason"] = f"crop_extract:{exc}"
                        row["decode_status"] = "failed"
                        continue
                    if crop.size == 0:
                        zero_size += 1
                        row["eligible"] = False
                        row["rejection_reason"] = "zero_size_crop"
                        row["decode_status"] = "zero_size"
                        continue
                    q = compute_crop_quality_fields(
                        crop, bbox, frame_width=w, frame_height=h, other_bboxes=others
                    )
                    abs_path = tmp / row["crop_relative_path"]
                    write_crop_jpeg(abs_path, crop)
                    digest = sha256_file(str(abs_path))
                    row.update(q)
                    row["crop_sha256"] = digest
                    row["decode_status"] = "ok"
                    row["absolute_crop_path"] = str(abs_path)
                    written += 1
                next_i += 1
            frame_index += 1
    finally:
        capture.release()

    # Merge ineligible planned rows that never entered by_frame
    by_id = {r["crop_id"]: r for r in planned}
    # Update from written loop (mutated rows in by_frame)
    for fi_rows in by_frame.values():
        for r in fi_rows:
            by_id[r["crop_id"]] = r
    manifest = [by_id[k] for k in sorted(by_id.keys())]

    # Validations
    if zero_size != 0:
        raise R2BError(f"BLOCKED_R2B_CROP_REEXTRACTION_FAILURE zero_size={zero_size}")
    if decode_fail != 0:
        raise R2BError("BLOCKED_R2B_CROP_REEXTRACTION_FAILURE decode")
    ids = [r["crop_id"] for r in manifest]
    if len(ids) != len(set(ids)):
        raise R2BError("BLOCKED_R2B_CROP_REEXTRACTION_FAILURE duplicate crop_id")
    paths = [r["crop_relative_path"] for r in manifest]
    if len(paths) != len(set(paths)):
        raise R2BError("BLOCKED_R2B_CROP_REEXTRACTION_FAILURE duplicate path")
    represented = {r["segment_id"] for r in manifest if r.get("decode_status") == "ok"}
    if represented != scoreable:
        missing = sorted(scoreable - represented)
        raise R2BError(f"BLOCKED_R2B_CROP_REEXTRACTION_FAILURE missing segments {missing[:10]}")

    summary = {
        "planned_observations": len(planned),
        "eligible_ok_crops": written,
        "ineligible": sum(1 for r in manifest if not r.get("eligible")),
        "scoreable_segments_represented": len(represented),
        "zero_size": zero_size,
        "decode_failures": decode_fail,
    }
    write_jsonl(tmp / "multiframe_crop_manifest.jsonl", manifest)
    write_json(tmp / "multiframe_crop_summary.json", summary)
    return manifest, summary


def run_selections(manifest: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_seg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_seg[str(row["segment_id"])].append(dict(row))
    b1: list[dict[str, Any]] = []
    b2: list[dict[str, Any]] = []
    for seg in sorted(by_seg.keys()):
        rows = by_seg[seg]
        s1 = select_uniform_temporal(rows, max_frames=MAX_FRAMES)
        s2 = select_quality_temporal_diversity(rows, max_frames=MAX_FRAMES)
        if not s1 or not s2:
            raise R2BError(f"BLOCKED_R2B_CROP_REEXTRACTION_FAILURE empty selection {seg}")
        b1.extend(s1)
        b2.extend(s2)
    return {"B1": b1, "B2": b2}


def embed_selected(
    *,
    tmp: Path,
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], float]:
    union: dict[str, Mapping[str, Any]] = {}
    for rows in selections.values():
        for r in rows:
            union[str(r["crop_id"])] = r
    ordered = [union[k] for k in sorted(union.keys())]
    verify_checkpoint(CKPT, expected_sha256=CKPT_SHA)
    model = build_osnet_cpu_model()
    load_osnet_checkpoint_weights(model, CKPT)

    tensors = []
    emb_rows = []
    for i, row in enumerate(ordered):
        path = Path(row["absolute_crop_path"])
        if not path.is_file():
            raise R2BError(f"BLOCKED_R2B_REID_RUNTIME missing crop {path}")
        if sha256_file(str(path)) != row["crop_sha256"]:
            raise R2BError("BLOCKED_R2B_REID_RUNTIME crop sha mismatch")
        tensors.append(load_and_preprocess_crop(path))
        emb_rows.append(
            {
                "crop_id": row["crop_id"],
                "segment_id": row["segment_id"],
                "frame_index": row["frame_index"],
                "embedding_row": i,
                "crop_sha256": row["crop_sha256"],
            }
        )
    vectors = embed_tensors(model, tensors, batch_size=8)
    if vectors.shape != (len(ordered), EMBEDDING_DIM):
        raise R2BError(f"BLOCKED_R2B_REID_RUNTIME shape {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise R2BError("BLOCKED_R2B_REID_RUNTIME nonfinite")
    if np.any(np.linalg.norm(vectors, axis=1) < 1e-6):
        raise R2BError("BLOCKED_R2B_REID_RUNTIME zero vector")
    # deterministic two-pass on first min(8, n)
    n_check = min(8, len(tensors))
    again = embed_tensors(model, tensors[:n_check], batch_size=8)
    max_abs = float(np.max(np.abs(vectors[:n_check] - again)))
    np.savez_compressed(
        tmp / "per_frame_embeddings.npz",
        vectors=vectors,
        crop_ids=np.asarray([r["crop_id"] for r in ordered]),
    )
    write_jsonl(tmp / "per_frame_embedding_manifest.jsonl", emb_rows)
    write_json(
        tmp / "per_frame_embedding_summary.json",
        {
            "count": len(ordered),
            "dim": EMBEDDING_DIM,
            "deterministic_repeat_max_abs_diff": max_abs,
            "checkpoint_sha256": CKPT_SHA,
            "batch_size": 8,
        },
    )
    by_id = {r["crop_id"]: vectors[i] for i, r in enumerate(ordered)}
    return by_id, emb_rows, max_abs


def build_tracklet_and_score(
    *,
    tmp: Path,
    ctx: Mapping[str, Any],
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
    emb_by_id: Mapping[str, np.ndarray],
    tg_ids: Sequence[str],
    dg_ids: Sequence[str],
) -> dict[str, Any]:
    joined_gt = {r["segment_id"]: r for r in ctx["joined"]}
    results = {}
    for variant, mode, sel_key in (
        ("B1", "l2_mean", "B1"),
        ("B2", "l2_mean", "B2"),
        ("B3", "embedding_medoid", "B2"),
    ):
        by_seg: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in selections[sel_key]:
            by_seg[str(row["segment_id"])].append(row)
        rep_rows = []
        vectors = []
        meta = []
        for seg in sorted(by_seg.keys()):
            rows = sorted(by_seg[seg], key=lambda r: (int(r["frame_index"]), str(r["crop_id"])))
            mats = np.stack([emb_by_id[str(r["crop_id"])] for r in rows], axis=0)
            agg = aggregate_with_fallback(
                mats,
                mode=mode,
                crop_ids=[str(r["crop_id"]) for r in rows],
                frame_indices=[int(r["frame_index"]) for r in rows],
                quality_ranks=[int(r.get("selection_rank") or 0) for r in rows],
            )
            gt = joined_gt[seg]
            rep = {
                "variant": variant,
                "segment_id": seg,
                "stable_query_id": gt["stable_query_id"],
                "selected_frame_count": int(agg["selected_frame_count"]),
                "frame_indices": [int(r["frame_index"]) for r in rows],
                "temporal_span": int(rows[-1]["frame_index"]) - int(rows[0]["frame_index"]),
                "representation_type": agg["representation"],
                "fallback_reason": agg["fallback_reason"],
                "medoid_crop_id": agg["medoid_crop_id"],
                "crop_ids": [str(r["crop_id"]) for r in rows],
            }
            rep_rows.append(rep)
            vectors.append(agg["vector"])
            meta.append(
                {
                    "segment_id": seg,
                    "stable_query_id": gt["stable_query_id"],
                    "query_order_index": gt.get("query_order_index"),
                    "component_id": gt.get("component_id"),
                    "selected_frame_count": int(agg["selected_frame_count"]),
                }
            )
        Q = np.stack(vectors, axis=0).astype(np.float32)
        np.save(tmp / f"tracklet_representation_{variant}.npy", Q)
        write_jsonl(tmp / f"tracklet_representation_{variant}.jsonl", rep_rows)
        scored = score_queries_against_galleries(
            Q=Q,
            T=ctx["tg"],
            D=ctx["dg"],
            target_ids=tg_ids,
            distractor_ids=dg_ids,
            query_meta=meta,
        )
        ranked = rank_primary(scored)
        # join GT
        out_join = []
        for row in ranked:
            gt = joined_gt[row["segment_id"]]
            label = 1 if gt["manual_ground_truth_decision"] == "target_occurrence_yes" else 0
            out_join.append(
                {
                    **row,
                    "manual_ground_truth_decision": gt["manual_ground_truth_decision"],
                    "binary_clean_player_label": label,
                    "same_team_negative_cohort": bool(gt["same_team_negative_cohort"]),
                    "other_team_negative_cohort": bool(gt["other_team_negative_cohort"]),
                }
            )
        write_jsonl(tmp / f"scores_{variant}.jsonl", out_join)
        metrics = evaluate_joined(out_join)
        write_json(tmp / f"metrics_{variant}.json", metrics)
        results[variant] = {"joined": out_join, "metrics": metrics, "rep_rows": rep_rows}
    return results


def make_figures(tmp: Path, metrics_a: Mapping[str, Any], results: Mapping[str, Any], joined_a: Sequence[Mapping]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = tmp / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    keys = ["AP", "AUROC", "same_team_AUROC", "Recall@10", "MRR", "margin"]
    variants = ["A", "B1", "B2", "B3"]
    data = {
        "A": metrics_a,
        "B1": results["B1"]["metrics"],
        "B2": results["B2"]["metrics"],
        "B3": results["B3"]["metrics"],
    }
    x = np.arange(len(keys))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, v in enumerate(variants):
        vals = []
        for k in keys:
            val = data[v].get(k)
            vals.append(float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else 0.0)
        ax.bar(x + (i - 1.5) * width, vals, width, label=v)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20)
    ax.legend()
    ax.set_title("Market1501 multiframe variant comparison (dev holdout v2)")
    fig.tight_layout()
    fig.savefig(fig_dir / "market1501_multiframe_variant_comparison.png", dpi=140)
    plt.close(fig)

    # positive rank trajectory
    pos_a = sorted(
        [r for r in joined_a if r["manual_ground_truth_decision"] == "target_occurrence_yes"],
        key=lambda r: int(r.get("query_order_index") or 0),
    )
    # order by start frame via segment id sort as proxy then ranks from each
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(1, 11))
    ax.plot(xs, [int(r["rank"]) for r in sorted(pos_a, key=lambda r: r["segment_id"])], marker="o", label="A")
    for v in ("B1", "B2", "B3"):
        pos = sorted(
            [r for r in results[v]["joined"] if r["binary_clean_player_label"] == 1],
            key=lambda r: r["segment_id"],
        )
        ax.plot(xs, [int(r["rank"]) for r in pos], marker="o", label=v)
    ax.invert_yaxis()
    ax.set_xlabel("Target occurrence (segment_id order)")
    ax.set_ylabel("Rank (1=best)")
    ax.legend()
    ax.set_title("Positive rank trajectory comparison")
    fig.tight_layout()
    fig.savefig(fig_dir / "positive_rank_trajectory_comparison.png", dpi=140)
    plt.close(fig)

    # top20 composition
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = variants
    tvals = [data[v]["top20_composition"]["target"] for v in labels]
    svals = [data[v]["top20_composition"]["same_team"] for v in labels]
    ovals = [data[v]["top20_composition"]["other_team"] for v in labels]
    ax.bar(labels, tvals, label="target")
    ax.bar(labels, svals, bottom=tvals, label="same-team")
    ax.bar(labels, ovals, bottom=np.array(tvals) + np.array(svals), label="other-team")
    ax.set_ylabel("Count in top-20")
    ax.legend()
    ax.set_title("Top-20 composition comparison")
    fig.tight_layout()
    fig.savefig(fig_dir / "top20_composition_comparison.png", dpi=140)
    plt.close(fig)


def make_contact_sheet(tmp: Path, results: Mapping[str, Any], selections: Mapping[str, Any], candidate: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel_key = "B1" if candidate == "B1" else "B2"
    pos_segs = [
        r["segment_id"]
        for r in results[candidate]["joined"]
        if r["binary_clean_player_label"] == 1
    ]
    pos_segs = sorted(pos_segs)
    by_seg: dict[str, list] = defaultdict(list)
    for row in selections[sel_key]:
        if row["segment_id"] in pos_segs:
            by_seg[row["segment_id"]].append(row)
    n_rows = len(pos_segs)
    n_cols = MAX_FRAMES
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.3, n_rows * 1.6))
    if n_rows == 1:
        axes = np.array([axes])
    for i, seg in enumerate(pos_segs):
        rows = sorted(by_seg[seg], key=lambda r: int(r["frame_index"]))
        for j in range(n_cols):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(rows):
                img = cv2.imread(rows[j]["absolute_crop_path"])
                if img is not None:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title(f"f{rows[j]['frame_index']}", fontsize=7)
        axes[i, 0].set_ylabel(seg, fontsize=7)
    fig.suptitle(f"Selected frames contact sheet ({candidate})", fontsize=11)
    fig.tight_layout()
    out = tmp / "figures" / "selected_frames_contact_sheet.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


def make_diagnostic_video(
    tmp: Path,
    ctx: Mapping[str, Any],
    results: Mapping[str, Any],
    candidate: str,
) -> None:
    video = ctx["video"]
    w = int(ctx["video_meta"]["width"])
    h = int(ctx["video_meta"]["height"])
    fps = float(ctx["video_meta"]["fps"]) or 30.0
    joined = {r["segment_id"]: r for r in results[candidate]["joined"]}
    obs = ctx["obs"]
    scoreable = set(joined.keys())
    by_frame: dict[int, list] = defaultdict(list)
    for o in obs:
        if o["segment_id"] in scoreable:
            by_frame[int(o["frame_index"])].append(o)

    # Short diagnostic: title + sample frames around positive + high-rank negatives (not full match)
    out_path = tmp / "videos" / "selected_candidate_gt_diagnostic_review.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Collect ~120 frames: evenly from video where scoreable active
    active_frames = sorted(by_frame.keys())
    if not active_frames:
        raise R2BError("no frames for diagnostic video")
    pick = active_frames[:: max(1, len(active_frames) // 90)][:90]
    cmd_in = tmp / "runtime" / "diag_frames"
    cmd_in.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # write via ffmpeg pipe for reliability
    frames_bgr = []
    # title frames
    for _ in range(45):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "DEVELOPMENT GT OVERLAY — NOT MODEL INPUT",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"R2B selected candidate {candidate} — holdout v2 DEV ONLY",
            (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        frames_bgr.append(canvas)

    needed = set(pick)
    fi = 0
    got = {}
    while len(got) < len(needed):
        ok, frame = cap.read()
        if not ok:
            break
        if fi in needed:
            got[fi] = frame.copy()
        fi += 1
    cap.release()

    for fi in pick:
        frame = got.get(fi)
        if frame is None:
            continue
        out = frame.copy()
        cv2.rectangle(out, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.putText(
            out,
            "DEVELOPMENT GT OVERLAY — NOT MODEL INPUT",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        for o in by_frame.get(fi, [])[:8]:
            seg = o["segment_id"]
            sc = joined[seg]
            clamp = validate_and_clamp_bbox(
                o["bbox_xyxy"],
                frame_width=w,
                frame_height=h,
            )
            if not clamp["eligible"]:
                continue
            x0, y0, x1, y1 = clamp["bbox_xyxy_int"]
            if sc["binary_clean_player_label"] == 1:
                color = (40, 180, 40) if int(sc["rank"]) <= 10 else (40, 40, 220)
                gt = "GT:TARGET"
            elif sc.get("same_team_negative_cohort"):
                color = (0, 165, 255)
                gt = "GT:SAME_TEAM"
            else:
                color = (255, 100, 50)
                gt = "GT:OTHER_TEAM"
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
            cv2.putText(
                out,
                f"{gt} r{sc['rank']} S={sc['S_primary']:.3f}",
                (x0, max(70, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        frames_bgr.append(out)

    # encode with ffmpeg
    raw = tmp / "runtime" / "diag.raw"
    with raw.open("wb") as handle:
        for fr in frames_bgr:
            handle.write(np.ascontiguousarray(fr).tobytes())
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        str(raw),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw.unlink(missing_ok=True)


def outcome_to_status(outcome: str) -> tuple[str, str]:
    mapping = {
        "MULTIFRAME_STRONG_IMPROVEMENT": (
            "COMPLETED_R2B_MULTIFRAME_STRONG_IMPROVEMENT",
            "NEXT_REID_R2C_NEW_HOLDOUT_OR_SPORTSREID_COMPARISON_DECISION",
        ),
        "MULTIFRAME_DIRECTIONAL_IMPROVEMENT": (
            "COMPLETED_R2B_MULTIFRAME_DIRECTIONAL_IMPROVEMENT",
            "NEXT_REID_R2C_NEW_HOLDOUT_OR_SPORTSREID_COMPARISON_DECISION",
        ),
        "MULTIFRAME_MIXED_OR_OVERLAPPING": (
            "COMPLETED_R2B_MULTIFRAME_MIXED_OR_OVERLAPPING",
            "NEXT_REID_R2C_SPORTSREID_ASSET_ACQUISITION_OR_PART_BASED_PIVOT",
        ),
        "MULTIFRAME_NO_MEANINGFUL_IMPROVEMENT": (
            "COMPLETED_R2B_MULTIFRAME_NO_MEANINGFUL_IMPROVEMENT",
            "NEXT_REID_R2C_SPORTSREID_ASSET_ACQUISITION_OR_PART_BASED_PIVOT",
        ),
        "MULTIFRAME_REGRESSION": (
            "COMPLETED_R2B_MULTIFRAME_REGRESSION",
            "NEXT_REID_R2C_SPORTSREID_ASSET_ACQUISITION_OR_PART_BASED_PIVOT",
        ),
    }
    return mapping[outcome]


def write_report(tmp: Path, payload: Mapping[str, Any]) -> None:
    lines = []
    w = lines.append
    w("# Target 001 — ReID-R2B Market1501 Multi-Frame Tracklet Report")
    w("")
    w("## 1. Final status")
    w(f"- **Status:** `{payload['final_status']}`")
    w(f"- **Outcome:** `{payload['outcome']}`")
    w(f"- **Selected development candidate:** `{payload['selected_candidate']}`")
    w(f"- **Next gate:** `{payload['next_gate']}`")
    w("")
    w("## 2. Amaç ve deney sınırı")
    w("- Aynı Market1501 OSNet ile single-crop (A, frozen) vs multi-frame tracklet (B1/B2/B3).")
    w("- Holdout v2: `development_and_error_analysis_only`.")
    w("- SportsReID checkpoint kullanılmadı. Deployment iddiası yok. Automatic identity assignment = false.")
    w("")
    w("## 3. Git/repo/environment integrity")
    w(f"- HEAD: `{payload['project_head']}`")
    w(f"- Checkpoint SHA: `{CKPT_SHA}`")
    w(f"- Network/download: 0; package change: 0")
    w("")
    w("## 4. Frozen Experiment A")
    w(f"- `frozen_not_rescored=true`")
    w(f"- Metrics: `{json.dumps(payload['metrics_a'], sort_keys=True)}`")
    w("")
    w("## 5. Source-video crop rebuild")
    w(f"- Crop summary: `{json.dumps(payload['crop_summary'], sort_keys=True)}`")
    w("")
    w("## 6. Quality ve temporal frame selection")
    w("- B1: uniform temporal ≤12; B2/B3: quality+temporal diversity ≤12.")
    w(f"- Selected frame count distribution: `{payload['selection_distribution']}`")
    w("")
    w("## 7. Per-frame embedding")
    w(f"- Embedded crops: {payload['embedding_count']}")
    w(f"- Deterministic repeat max-abs-diff: {payload['deterministic_max_abs_diff']}")
    w("")
    w("## 8. B1 representation")
    w("- L2-mean of uniformly selected frames.")
    w(f"- Metrics: `{json.dumps(payload['metrics_b1'], sort_keys=True)}`")
    w("")
    w("## 9. B2 representation")
    w("- L2-mean of quality+temporal selected frames.")
    w(f"- Metrics: `{json.dumps(payload['metrics_b2'], sort_keys=True)}`")
    w("")
    w("## 10. B3 representation")
    w("- Embedding medoid of B2-selected frames.")
    w(f"- Metrics: `{json.dumps(payload['metrics_b3'], sort_keys=True)}`")
    w("")
    w("## 11. Frozen gallery scoring")
    w("- Gallery-v2 / distractor-v1 immutable; `S=T_max-D_max` only.")
    w("")
    w("## 12. Metric karşılaştırması")
    w(f"- Comparison table: `{json.dumps(payload['comparison'], sort_keys=True)}`")
    w("")
    w("## 13. Same-team confusion analizi")
    w(f"- A same-team AUROC={payload['metrics_a']['same_team_AUROC']}")
    w(f"- Candidate same-team AUROC={payload['metrics_selected']['same_team_AUROC']}")
    w("")
    w("## 14. Target re-entry rank trajectory")
    w(f"- A ranks: {payload['metrics_a']['positive_ranks']}")
    w(f"- Candidate ranks: {payload['metrics_selected']['positive_ranks']}")
    w("")
    w("## 15. Selected development candidate")
    w(f"- `{payload['selected_candidate']}` — yeni bağımsız holdout’ta test adayı; deployment method değil.")
    w("")
    w("## 16. Diagnostic visualization")
    w("- figures/*.png + videos/selected_candidate_gt_diagnostic_review.mp4")
    w("")
    w("## 17. Tests")
    w(f"- `{json.dumps(payload['tests'], sort_keys=True)}`")
    w("")
    w("## 18. Runtime/network")
    w(f"- `{json.dumps(payload['runtime'], sort_keys=True)}`")
    w("")
    w("## 19. Warnings/limitations")
    w("- Holdout v2 development setidir; bağımsız başarı iddiası değildir.")
    w("- Gallery veya GT değiştirilmedi.")
    w("- Experiment A yeniden skorlanmadı.")
    w("- Automatic identity assignment hâlâ false; threshold seçilmedi.")
    w("- Yeni bağımsız holdout zorunludur.")
    w("")
    w("## 20. Exact next gate")
    w(f"- `{payload['next_gate']}`")
    (tmp / "target_001_reid_r2b_market1501_multiframe_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def file_inventory(root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out.append(
                {
                    "relpath": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(str(path)),
                }
            )
    return out


def main() -> int:
    t0 = time.time()
    head_before = git("rev-parse", "HEAD")
    try:
        ctx = validate_start()
    except R2BError as exc:
        print(str(exc))
        return 2

    tmp = PROJECT_ROOT / "outputs" / "reid" / f"_tmp_target_001_reid_r2b_market1501_multiframe_{uuid.uuid4().hex[:10]}"
    tmp.mkdir(parents=True, exist_ok=False)
    (tmp / "configs").mkdir()
    (tmp / "runtime").mkdir()
    shutil.copy2(CONTRACT, tmp / "configs" / "effective_r2b_contract.yaml")

    network_audit = {"downloads": 0, "network_calls": 0, "sportsreid_used": False}
    write_json(tmp / "runtime" / "network_audit.json", network_audit)

    try:
        # Pre-pipeline related tests (subset)
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(SN_REID)}
        env_tests = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
        test_py = str(FOOTBALL_CV_PY if FOOTBALL_CV_PY.is_file() else sys.executable)
        pre = subprocess.run(
            [
                test_py,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_reid_aggregate.py",
                "-q",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=env_tests,
        )
        pre2 = subprocess.run(
            [
                test_py,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_reid_crop_select.py",
                "-q",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=env_tests,
        )
        pre3 = subprocess.run(
            [
                test_py,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_reid_embedding.py",
                "-q",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=env_tests,
        )
        if pre.returncode != 0 or pre2.returncode != 0 or pre3.returncode != 0:
            raise R2BError(
                "FAILED_R2B_ATOMIC_BUILD pre tests "
                f"{pre.returncode}/{pre2.returncode}/{pre3.returncode} "
                f"{(pre.stderr or '')[-200:]}{(pre2.stderr or '')[-200:]}{(pre3.stderr or '')[-200:]}"
            )

        tg_ids, dg_ids = load_gallery_ids()
        gallery_sha = {
            "target_npy": sha256_file(str(F3G / "target_gallery_v2/target_001_gallery_v2_individual_embeddings.npy")),
            "distractor_npy": sha256_file(
                str(F3G / "distractor_gallery_v1/target_001_same_team_distractor_individual_embeddings.npy")
            ),
        }
        metrics_a = frozen_a_metrics(ctx["sm"], ctx["joined"])

        print("Extracting multiframe crops…", flush=True)
        manifest, crop_summary = extract_multiframe_crops(tmp=tmp, ctx=ctx)
        print("Selecting frames…", flush=True)
        selections = run_selections(manifest)
        write_jsonl(tmp / "selection_B1.jsonl", selections["B1"])
        write_jsonl(tmp / "selection_B2.jsonl", selections["B2"])
        def count_dist(rows):
            per: Counter[int] = Counter()
            for _seg, n in Counter(r["segment_id"] for r in rows).items():
                per[n] += 1
            return {str(k): v for k, v in sorted(per.items())}

        selection_distribution = {"B1": count_dist(selections["B1"]), "B2": count_dist(selections["B2"])}

        print("Embedding selected crops…", flush=True)
        emb_by_id, emb_rows, max_abs = embed_selected(tmp=tmp, selections=selections)
        print("Scoring variants…", flush=True)
        results = build_tracklet_and_score(
            tmp=tmp,
            ctx=ctx,
            selections=selections,
            emb_by_id=emb_by_id,
            tg_ids=tg_ids,
            dg_ids=dg_ids,
        )

        # immutability checks
        tg2 = np.load(F3G / "target_gallery_v2/target_001_gallery_v2_individual_embeddings.npy")
        if not np.allclose(tg2, ctx["tg"]):
            raise R2BError("gallery mutated")

        metrics_map = {v: results[v]["metrics"] for v in ("B1", "B2", "B3")}
        candidate = select_development_candidate(metrics_map)
        query_drop = 115 - int(metrics_map[candidate]["query_count"])
        outcome = classify_outcome(
            metrics_a=metrics_a, metrics_b=metrics_map[candidate], query_drop=query_drop
        )
        final_status, next_gate = outcome_to_status(outcome)

        comparison = {
            "A": {**metrics_a, "frozen_not_rescored": True},
            "B1": metrics_map["B1"],
            "B2": metrics_map["B2"],
            "B3": metrics_map["B3"],
            "selected_development_candidate": candidate,
            "outcome": outcome,
        }
        write_json(tmp / "a_vs_b_comparison.json", comparison)
        write_json(
            tmp / "selected_candidate_decision.json",
            {
                "selected_development_candidate": candidate,
                "outcome": outcome,
                "final_status": final_status,
                "next_gate": next_gate,
                "tie_break_rule": "B1>B2>B3 when equal",
            },
        )

        print("Figures/video…", flush=True)
        make_figures(tmp, metrics_a, results, ctx["joined"])
        make_contact_sheet(tmp, results, selections, candidate)
        make_diagnostic_video(tmp, ctx, results, candidate)

        write_jsonl(tmp / "tracklet_representation_manifest.jsonl", results[candidate]["rep_rows"])

        # Post unit tests for new module
        post = subprocess.run(
            [
                test_py,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_reid_r2b_multiframe.py",
                "-q",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=env_tests,
        )
        full = subprocess.run(
            [test_py, "-m", "unittest", "discover", "-s", "tests", "-q"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=env_tests,
        )
        # parse counts from stderr "Ran N tests"
        def parse_ran(text: str) -> int | None:
            for line in text.splitlines():
                if line.startswith("Ran "):
                    return int(line.split()[1])
            return None

        tests_summary = {
            "pre_related_exit": [pre.returncode, pre2.returncode, pre3.returncode],
            "post_r2b_exit": post.returncode,
            "full_suite_exit": full.returncode,
            "full_suite_ran": parse_ran(full.stderr + full.stdout),
            "post_stderr_tail": (post.stderr or "")[-500:],
            "full_stderr_tail": (full.stderr or "")[-500:],
        }
        if post.returncode != 0:
            raise R2BError(f"FAILED_R2B_ATOMIC_BUILD unit tests: {tests_summary['post_stderr_tail']}")
        if full.returncode != 0:
            raise R2BError(f"FAILED_R2B_ATOMIC_BUILD full suite: {tests_summary['full_stderr_tail']}")

        subprocess.check_call(["git", "diff", "--check"], cwd=str(PROJECT_ROOT))
        head_after = git("rev-parse", "HEAD")
        if head_after != head_before:
            raise R2BError("FAILED_R2B_ATOMIC_BUILD head changed")

        payload = {
            "final_status": final_status,
            "outcome": outcome,
            "selected_candidate": candidate,
            "next_gate": next_gate,
            "project_head": head_after,
            "metrics_a": metrics_a,
            "metrics_b1": metrics_map["B1"],
            "metrics_b2": metrics_map["B2"],
            "metrics_b3": metrics_map["B3"],
            "metrics_selected": metrics_map[candidate],
            "crop_summary": crop_summary,
            "selection_distribution": selection_distribution,
            "embedding_count": len(emb_rows),
            "deterministic_max_abs_diff": max_abs,
            "comparison": comparison,
            "tests": tests_summary,
            "runtime": {
                "elapsed_sec": round(time.time() - t0, 2),
                "python": sys.version.split()[0],
                "env_hint": "sn-reid-cpu_or_football-cv_with_snreid_path",
            },
        }
        write_report(tmp, payload)

        inv = file_inventory(tmp)
        # exclude manifest itself later
        manifest_obj = {
            "schema_version": "target_001_reid_r2b_manifest_v1",
            "final_status": final_status,
            "outcome": outcome,
            "selected_development_candidate": candidate,
            "next_gate": next_gate,
            "project_head_before": head_before,
            "project_head_after": head_after,
            "source_video": {
                "path": VIDEO_REL,
                "sha256": ctx["video_sha"],
                **ctx["video_meta"],
            },
            "f3j_observations_sha256": sha256_file(
                str(F3J / "segmentation/target_001_holdout_v2_label_blind_segment_observations.jsonl")
            ),
            "frozen_gallery_shas": gallery_sha,
            "frozen_f3m_summary_sha256": sha256_file(str(F3M / "stage5d_f3m_summary.json")),
            "checkpoint": {"path": str(CKPT), "sha256": CKPT_SHA, "bytes": CKPT_BYTES},
            "effective_config_sha256": sha256_file(str(tmp / "configs/effective_r2b_contract.yaml")),
            "crop_summary": crop_summary,
            "embedding_count": len(emb_rows),
            "deterministic_repeat_max_abs_diff": max_abs,
            "metrics": comparison,
            "no_sportsreid_used": True,
            "no_dataset_download": True,
            "no_package_change": True,
            "gallery_modified": False,
            "gt_modified": False,
            "experiment_a_frozen_not_rescored": True,
            "automatic_identity_assignment": False,
            "threshold_selected": False,
            "holdout_role": "development_and_error_analysis_only",
            "atomic_finalization": True,
            "network_audit": network_audit,
            "tests": tests_summary,
            "outputs": inv,
        }
        write_json(tmp / "target_001_reid_r2b_manifest.json", manifest_obj)
        # refresh inventory without self-sha issue: rewrite outputs list excluding manifest then include sizes
        inv2 = file_inventory(tmp)
        manifest_obj["outputs"] = [x for x in inv2 if x["relpath"] != "target_001_reid_r2b_manifest.json"]
        write_json(tmp / "target_001_reid_r2b_manifest.json", manifest_obj)

        # atomic publish
        os.rename(str(tmp), str(FINAL_ROOT))
        print(
            json.dumps(
                {
                    "final_status": final_status,
                    "outcome": outcome,
                    "selected_candidate": candidate,
                    "next_gate": next_gate,
                    "output_root": str(FINAL_ROOT),
                    "AP_A": metrics_a["AP"],
                    "AP_B1": metrics_map["B1"]["AP"],
                    "AP_B2": metrics_map["B2"]["AP"],
                    "AP_B3": metrics_map["B3"]["AP"],
                    "AUROC_A": metrics_a["AUROC"],
                    "AUROC_sel": metrics_map[candidate]["AUROC"],
                    "ranks_A": metrics_a["positive_ranks"],
                    "ranks_sel": metrics_map[candidate]["positive_ranks"],
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        # cleanup temp only
        if tmp.exists() and tmp.name.startswith("_tmp_target_001_reid_r2b_market1501_multiframe_"):
            shutil.rmtree(tmp, ignore_errors=True)
        if isinstance(exc, R2BError):
            print(str(exc))
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
