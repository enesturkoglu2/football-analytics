"""ReID-R2B Market1501 multi-frame tracklet helpers (selection, aggregation, metrics)."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from football_analytics.reid.crop_select import CropSelectError, clamp_bbox_xyxy, float_bbox_to_int_crop
from football_analytics.reid.embedding import EMBEDDING_DIM, l2_normalize_rows
from football_analytics.reid.quality import compute_edge_contacts, compute_image_metrics

MAX_FRAMES = 12
MIN_FRAMES = 3
BBOX_CLAMP_POLICY = "floor_ceil_int_crop_after_float_clamp_to_frame"


class MultiframeR2BError(RuntimeError):
    """Raised when R2B multiframe helpers fail."""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_clamp_bbox(
    bbox_xyxy: Sequence[float],
    *,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    """Clamp float bbox then convert to int crop box. Returns eligibility metadata."""
    if frame_width <= 0 or frame_height <= 0:
        raise MultiframeR2BError("frame dimensions must be positive")
    if len(bbox_xyxy) != 4:
        return {
            "eligible": False,
            "rejection_reason": "bbox_arity",
            "bbox_xyxy_int": None,
            "clamp_policy": BBOX_CLAMP_POLICY,
        }
    try:
        vals = [float(v) for v in bbox_xyxy]
    except (TypeError, ValueError):
        return {
            "eligible": False,
            "rejection_reason": "bbox_non_numeric",
            "bbox_xyxy_int": None,
            "clamp_policy": BBOX_CLAMP_POLICY,
        }
    if not all(math.isfinite(v) for v in vals):
        return {
            "eligible": False,
            "rejection_reason": "bbox_non_finite",
            "bbox_xyxy_int": None,
            "clamp_policy": BBOX_CLAMP_POLICY,
        }
    try:
        clamped = clamp_bbox_xyxy(vals, video_width=frame_width, video_height=frame_height)
        int_bbox = float_bbox_to_int_crop(
            clamped, video_width=frame_width, video_height=frame_height
        )
    except CropSelectError as exc:
        return {
            "eligible": False,
            "rejection_reason": f"bbox_clamp_empty:{exc}",
            "bbox_xyxy_int": None,
            "clamp_policy": BBOX_CLAMP_POLICY,
        }
    w = int_bbox[2] - int_bbox[0]
    h = int_bbox[3] - int_bbox[1]
    if w <= 0 or h <= 0:
        return {
            "eligible": False,
            "rejection_reason": "zero_size_bbox",
            "bbox_xyxy_int": int_bbox,
            "clamp_policy": BBOX_CLAMP_POLICY,
        }
    return {
        "eligible": True,
        "rejection_reason": None,
        "bbox_xyxy_int": int_bbox,
        "clamp_policy": BBOX_CLAMP_POLICY,
        "bbox_width": w,
        "bbox_height": h,
        "bbox_area": int(w * h),
    }


def iou_xyxy(a: Sequence[int], b: Sequence[int]) -> float:
    x1 = max(int(a[0]), int(b[0]))
    y1 = max(int(a[1]), int(b[1]))
    x2 = min(int(a[2]), int(b[2]))
    y2 = min(int(a[3]), int(b[3]))
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0, int(a[2]) - int(a[0])) * max(0, int(a[3]) - int(a[1]))
    area_b = max(0, int(b[2]) - int(b[0])) * max(0, int(b[3]) - int(b[1]))
    denom = float(area_a + area_b - inter)
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def max_other_person_overlap(
    target_bbox: Sequence[int],
    other_bboxes: Sequence[Sequence[int]],
) -> float:
    best = 0.0
    for other in other_bboxes:
        best = max(best, iou_xyxy(target_bbox, other))
    return float(best)


def uniform_temporal_indices(n: int, k: int) -> list[int]:
    """Select up to k indices uniformly over [0, n-1] (inclusive endpoints when k>=2)."""
    if n <= 0:
        return []
    if k <= 0:
        return []
    if n <= k:
        return list(range(n))
    if k == 1:
        return [n // 2]
    # Evenly spaced including first and last for coverage (B1 allows either; include both).
    raw = [int(round(i * (n - 1) / float(k - 1))) for i in range(k)]
    out: list[int] = []
    seen: set[int] = set()
    for idx in raw:
        idx = min(max(idx, 0), n - 1)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    # Fill gaps if rounding collisions reduced count.
    cursor = 0
    while len(out) < k and cursor < n:
        if cursor not in seen:
            seen.add(cursor)
            out.append(cursor)
        cursor += 1
    return sorted(out)[:k]


def select_uniform_temporal(
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_frames: int = MAX_FRAMES,
) -> list[dict[str, Any]]:
    eligible = [c for c in candidates if c.get("eligible")]
    eligible = sorted(eligible, key=lambda c: (int(c["frame_index"]), str(c["crop_id"])))
    if not eligible:
        return []
    idxs = uniform_temporal_indices(len(eligible), min(max_frames, len(eligible)))
    selected = []
    for rank, i in enumerate(idxs):
        row = dict(eligible[i])
        row["selection_variant"] = "B1"
        row["selection_rank"] = rank
        row["selection_reason"] = "uniform_temporal"
        selected.append(row)
    return selected


def _size_stratum(area: float) -> int:
    # Coarse strata so sharpness only ranks within similar size.
    if area < 2000:
        return 0
    if area < 6000:
        return 1
    if area < 15000:
        return 2
    return 3


def _quality_sort_key(c: Mapping[str, Any]) -> tuple:
    edge = int(c.get("frame_edge_contact_count") or 0)
    overlap = float(c.get("max_other_person_iou") or 0.0)
    area = float(c.get("bbox_area") or 0.0)
    sharp = float(c.get("laplacian_variance") or 0.0)
    stratum = _size_stratum(area)
    # Lower edge/overlap better; larger area better; sharpness only within stratum.
    return (edge, overlap, -area, stratum, -sharp, int(c["frame_index"]), str(c["crop_id"]))


def select_quality_temporal_diversity(
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_frames: int = MAX_FRAMES,
    min_frames: int = MIN_FRAMES,
) -> list[dict[str, Any]]:
    """Temporal slots + local quality pick + near-duplicate suppression."""
    eligible = [c for c in candidates if c.get("eligible")]
    eligible = sorted(eligible, key=lambda c: (int(c["frame_index"]), str(c["crop_id"])))
    n = len(eligible)
    if n == 0:
        return []
    target = min(max_frames, n)
    if n >= min_frames:
        target = max(target, min(min_frames, n))
        target = min(target, max_frames, n)

    # Slot centers across eligible index space.
    slot_idxs = uniform_temporal_indices(n, target)
    frames = [int(c["frame_index"]) for c in eligible]
    span = max(frames) - min(frames) if frames else 0
    window = max(1, int(math.ceil(span / float(max(target * 2, 1)))))
    # Also use index-window for dense tracks.
    index_window = max(1, int(math.ceil(n / float(max(target * 2, 1)))))

    picked: list[Mapping[str, Any]] = []
    used_ids: set[str] = set()
    for slot in slot_idxs:
        slot_frame = int(eligible[slot]["frame_index"])
        local = []
        for j, c in enumerate(eligible):
            if str(c["crop_id"]) in used_ids:
                continue
            if abs(j - slot) <= index_window or abs(int(c["frame_index"]) - slot_frame) <= window:
                local.append(c)
        if not local:
            local = [c for c in eligible if str(c["crop_id"]) not in used_ids]
        if not local:
            break
        best = sorted(local, key=_quality_sort_key)[0]
        used_ids.add(str(best["crop_id"]))
        picked.append(best)

    # Near-duplicate suppression: if two selected frames closer than min_gap, keep better.
    min_gap = max(1, int(math.floor(span / float(max(target * 3, 1))))) if span > 0 else 1
    picked = sorted(picked, key=lambda c: int(c["frame_index"]))
    filtered: list[Mapping[str, Any]] = []
    for c in picked:
        if not filtered:
            filtered.append(c)
            continue
        prev = filtered[-1]
        if abs(int(c["frame_index"]) - int(prev["frame_index"])) < min_gap:
            # Keep better quality.
            if _quality_sort_key(c) < _quality_sort_key(prev):
                filtered[-1] = c
            continue
        filtered.append(c)

    # If suppression removed too many, backfill by quality with spacing.
    if len(filtered) < min(min_frames, n) or len(filtered) < target:
        remaining = [c for c in eligible if str(c["crop_id"]) not in {str(x["crop_id"]) for x in filtered}]
        remaining = sorted(remaining, key=_quality_sort_key)
        for c in remaining:
            if len(filtered) >= target:
                break
            if any(abs(int(c["frame_index"]) - int(p["frame_index"])) < min_gap for p in filtered):
                continue
            filtered.append(c)
        # Last resort: ignore spacing to reach min_frames.
        if len(filtered) < min(min_frames, n):
            for c in remaining:
                if len(filtered) >= min(min_frames, n):
                    break
                if str(c["crop_id"]) in {str(x["crop_id"]) for x in filtered}:
                    continue
                filtered.append(c)

    filtered = sorted(filtered, key=lambda c: (int(c["frame_index"]), str(c["crop_id"])))[:target]
    # Enforce min_frames when available.
    if n >= min_frames and len(filtered) < min_frames:
        raise MultiframeR2BError(
            f"quality selection produced {len(filtered)} < {min_frames} with n={n}"
        )

    out = []
    for rank, c in enumerate(filtered):
        row = dict(c)
        row["selection_variant"] = "B2"
        row["selection_rank"] = rank
        row["selection_reason"] = (
            f"quality_temporal_diversity;min_gap={min_gap};"
            f"edge={row.get('frame_edge_contact_count')};"
            f"iou={row.get('max_other_person_iou')};"
            f"area={row.get('bbox_area')};"
            f"sharp={row.get('laplacian_variance')}"
        )
        out.append(row)
    return out


def l2_mean_aggregate(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows, arithmetic mean, re-L2-normalize. Shape (D,)."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != EMBEDDING_DIM:
        raise MultiframeR2BError(f"vectors must be (N,{EMBEDDING_DIM}), got {arr.shape}")
    if arr.shape[0] == 0:
        raise MultiframeR2BError("empty vectors for aggregation")
    if arr.shape[0] == 1:
        return l2_normalize_rows(arr)[0]
    normed = l2_normalize_rows(arr)
    mean_vec = np.mean(normed, axis=0).astype(np.float32, copy=False)
    mean_norm = float(np.linalg.norm(mean_vec))
    if not math.isfinite(mean_norm) or mean_norm <= 0:
        raise MultiframeR2BError("non-finite or zero mean embedding")
    out = (mean_vec / mean_norm).astype(np.float32, copy=False)
    return out


def embedding_medoid(
    vectors: np.ndarray,
    *,
    crop_ids: Sequence[str],
    frame_indices: Sequence[int],
    quality_ranks: Sequence[int],
) -> tuple[np.ndarray, str, int]:
    """Return medoid vector, crop_id, index. Cosine distance = 1 - cosine similarity."""
    arr = np.asarray(vectors, dtype=np.float32)
    n = arr.shape[0]
    if n == 0:
        raise MultiframeR2BError("empty vectors for medoid")
    if n == 1:
        return l2_normalize_rows(arr)[0], str(crop_ids[0]), 0
    if n == 2:
        # Short-segment fallback: normalized mean (same as L2 mean of 2).
        return l2_mean_aggregate(arr), "mean_fallback_2", -1
    normed = l2_normalize_rows(arr)
    sims = normed @ normed.T
    mean_dist = 1.0 - sims.mean(axis=1)
    best = None
    for i in range(n):
        key = (
            float(mean_dist[i]),
            -int(quality_ranks[i]),  # higher quality rank better => lower key with negative
            int(frame_indices[i]),
            str(crop_ids[i]),
        )
        # quality_ranks: lower number = better selection_rank; "higher quality rank" means better.
        # Interpret quality_ranks as selection_rank (0 best). Prefer lower selection_rank.
        key = (
            float(mean_dist[i]),
            int(quality_ranks[i]),
            int(frame_indices[i]),
            str(crop_ids[i]),
        )
        if best is None or key < best[0]:
            best = (key, i)
    assert best is not None
    i = best[1]
    return normed[i].astype(np.float32, copy=False), str(crop_ids[i]), int(i)


def aggregate_with_fallback(
    vectors: np.ndarray,
    *,
    mode: str,
    crop_ids: Sequence[str] | None = None,
    frame_indices: Sequence[int] | None = None,
    quality_ranks: Sequence[int] | None = None,
) -> dict[str, Any]:
    arr = np.asarray(vectors, dtype=np.float32)
    n = arr.shape[0]
    fallback = None
    medoid_id = None
    if n == 1:
        vec = l2_normalize_rows(arr)[0]
        fallback = "single_frame"
        representation = "single"
    elif n == 2:
        vec = l2_mean_aggregate(arr)
        fallback = "two_frame_l2_mean"
        representation = "l2_mean"
    elif mode == "l2_mean":
        vec = l2_mean_aggregate(arr)
        representation = "l2_mean"
    elif mode == "embedding_medoid":
        if crop_ids is None or frame_indices is None or quality_ranks is None:
            raise MultiframeR2BError("medoid requires crop_ids/frame_indices/quality_ranks")
        vec, medoid_id, _ = embedding_medoid(
            arr,
            crop_ids=crop_ids,
            frame_indices=frame_indices,
            quality_ranks=quality_ranks,
        )
        representation = "embedding_medoid"
        if medoid_id == "mean_fallback_2":
            fallback = "two_frame_l2_mean"
            medoid_id = None
    else:
        raise MultiframeR2BError(f"unknown aggregation mode {mode!r}")
    return {
        "vector": vec,
        "representation": representation,
        "fallback_reason": fallback,
        "medoid_crop_id": medoid_id,
        "selected_frame_count": n,
    }


def argmax_with_id_tiebreak(scores: np.ndarray, ids: Sequence[str]) -> tuple[float, str]:
    best_val = float("-inf")
    best_id = None
    for val, mid in zip(scores.tolist(), ids):
        fval = float(val)
        if fval > best_val or (math.isclose(fval, best_val) and (best_id is None or mid < best_id)):
            best_val = fval
            best_id = mid
    assert best_id is not None
    return best_val, best_id


def score_queries_against_galleries(
    *,
    Q: np.ndarray,
    T: np.ndarray,
    D: np.ndarray,
    target_ids: Sequence[str],
    distractor_ids: Sequence[str],
    query_meta: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if Q.shape[0] != len(query_meta):
        raise MultiframeR2BError("Q rows != query_meta")
    C_t = (Q @ T.T).astype(np.float32)
    C_d = (Q @ D.T).astype(np.float32)
    rows = []
    for i, meta in enumerate(query_meta):
        t_max, t_id = argmax_with_id_tiebreak(C_t[i], target_ids)
        d_max, d_id = argmax_with_id_tiebreak(C_d[i], distractor_ids)
        rows.append(
            {
                **dict(meta),
                "T_max": float(t_max),
                "T_max_member_id": t_id,
                "D_max": float(d_max),
                "D_max_member_id": d_id,
                "S_primary": float(t_max - d_max),
            }
        )
    return rows


def rank_primary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda r: (
            -float(r["S_primary"]),
            -float(r["T_max"]),
            float(r["D_max"]),
            str(r["stable_query_id"]),
        ),
    )
    out = []
    for idx, row in enumerate(ordered, start=1):
        item = dict(row)
        item["rank"] = idx
        out.append(item)
    return out


def _recall_at_k(labels: Sequence[int], k: int, n_pos: int) -> float:
    if n_pos <= 0:
        return 0.0
    return float(sum(labels[:k])) / float(n_pos)


def _ap_from_ranks(pos_ranks: Sequence[int], n_pos: int) -> float:
    if n_pos <= 0:
        return 0.0
    hits = 0
    s = 0.0
    for rank in sorted(int(r) for r in pos_ranks):
        hits += 1
        s += hits / float(rank)
    return float(s / n_pos)


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
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        ap_sklearn = float(average_precision_score(y_true, y_score)) if n_pos > 0 else 0.0
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        ap_sklearn = _ap_from_ranks(pos_ranks, n_pos)
        # Mann-Whitney fallback
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if len(pos) and len(neg):
            correct = sum(float(p > n) for p in pos for n in neg) + 0.5 * sum(
                float(p == n) for p in pos for n in neg
            )
            auroc = float(correct / (len(pos) * len(neg)))
        else:
            auroc = float("nan")
    ap_rank = _ap_from_ranks(pos_ranks, n_pos)
    mrr = float(1.0 / pos_ranks[0]) if pos_ranks else 0.0
    pos_scores = [scores[i] for i, lab in enumerate(labels) if lab == 1]
    neg_scores = [scores[i] for i, lab in enumerate(labels) if lab == 0]
    margin = (
        float(min(pos_scores) - max(neg_scores)) if pos_scores and neg_scores else float("nan")
    )
    return {
        "Recall@1": _recall_at_k(labels, 1, n_pos),
        "Recall@3": _recall_at_k(labels, 3, n_pos),
        "Recall@5": _recall_at_k(labels, 5, n_pos),
        "Recall@10": _recall_at_k(labels, 10, n_pos),
        "Recall@20": _recall_at_k(labels, 20, n_pos),
        "MRR": mrr,
        "AP": ap_rank,
        "AUPRC": ap_sklearn,
        "AUROC": auroc,
        "margin": margin,
        "positive_ranks": pos_ranks,
        "positive_median_rank": float(np.median(pos_ranks)) if pos_ranks else None,
        "n_pos": n_pos,
        "n_neg": len(labels) - n_pos,
    }


def cohort_auroc_ap(
    joined: Sequence[Mapping[str, Any]],
    *,
    negative_predicate,
) -> dict[str, float]:
    """Binary metrics: positives vs selected negative cohort only (+ all positives)."""
    rows = []
    for r in joined:
        if int(r["binary_clean_player_label"]) == 1:
            rows.append(r)
        elif negative_predicate(r):
            rows.append(r)
    ranked = sorted(rows, key=lambda r: -float(r["S_primary"]))
    labels = [int(r["binary_clean_player_label"]) for r in ranked]
    scores = [float(r["S_primary"]) for r in ranked]
    n_pos = sum(labels)
    m = ranking_metrics(labels, scores, n_pos=n_pos)
    return {"AUROC": float(m["AUROC"]), "AP": float(m["AP"])}


def evaluate_joined(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = sorted(joined, key=lambda r: int(r["rank"]))
    labels = [int(r["binary_clean_player_label"]) for r in ranked]
    scores = [float(r["S_primary"]) for r in ranked]
    base = ranking_metrics(labels, scores, n_pos=sum(labels))
    st = cohort_auroc_ap(joined, negative_predicate=lambda r: bool(r.get("same_team_negative_cohort")))
    ot = cohort_auroc_ap(joined, negative_predicate=lambda r: bool(r.get("other_team_negative_cohort")))
    top10 = ranked[:10]
    top20 = ranked[:20]

    def composition(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            "target": sum(1 for r in rows if int(r["binary_clean_player_label"]) == 1),
            "same_team": sum(1 for r in rows if r.get("same_team_negative_cohort")),
            "other_team": sum(1 for r in rows if r.get("other_team_negative_cohort")),
        }

    return {
        **base,
        "same_team_AUROC": st["AUROC"],
        "same_team_AP": st["AP"],
        "other_team_AUROC": ot["AUROC"],
        "other_team_AP": ot["AP"],
        "top10_composition": composition(top10),
        "top20_composition": composition(top20),
        "query_count": len(ranked),
    }


def classify_outcome(
    *,
    metrics_a: Mapping[str, Any],
    metrics_b: Mapping[str, Any],
    query_drop: int,
) -> str:
    """Pre-registered outcome logic (config thresholds hardcoded to match frozen YAML)."""
    ap_a, ap_b = float(metrics_a["AP"]), float(metrics_b["AP"])
    au_a, au_b = float(metrics_a["AUROC"]), float(metrics_b["AUROC"])
    st_a, st_b = float(metrics_a["same_team_AUROC"]), float(metrics_b["same_team_AUROC"])
    r10_b = float(metrics_b["Recall@10"])
    med_a, med_b = float(metrics_a["positive_median_rank"]), float(metrics_b["positive_median_rank"])
    mar_a, mar_b = float(metrics_a["margin"]), float(metrics_b["margin"])

    strong = (
        query_drop == 0
        and (ap_b - ap_a) >= 0.15
        and (au_b - au_a) >= 0.10
        and (st_b - st_a) >= 0.08
        and r10_b >= 0.50
        and med_b < med_a
        and mar_b > mar_a
    )
    if strong:
        return "MULTIFRAME_STRONG_IMPROVEMENT"

    directional = (
        query_drop == 0
        and ap_b > ap_a
        and au_b > au_a
        and st_b >= st_a
        and med_b < med_a
    )
    if directional:
        return "MULTIFRAME_DIRECTIONAL_IMPROVEMENT"

    improved = (ap_b > ap_a) or (au_b > au_a) or (med_b < med_a)
    worsened = (ap_b < ap_a) or (au_b < au_a) or (med_b > med_a) or (st_b < st_a)
    if improved and worsened:
        return "MULTIFRAME_MIXED_OR_OVERLAPPING"
    if (ap_b < ap_a and au_b < au_a) or (med_b > med_a and ap_b <= ap_a and au_b <= au_a):
        return "MULTIFRAME_REGRESSION"
    if not improved:
        return "MULTIFRAME_NO_MEANINGFUL_IMPROVEMENT"
    return "MULTIFRAME_MIXED_OR_OVERLAPPING"


def score_variant_for_selection(metrics: Mapping[str, Any]) -> tuple:
    """Higher is better for lexicographic max (except ranks inverted already via negatives)."""
    return (
        float(metrics["AP"]),
        float(metrics["AUROC"]),
        float(metrics["same_team_AUROC"]),
        -float(metrics["positive_median_rank"]),
        float(metrics["margin"]),
        float(metrics["Recall@10"]),
        float(metrics["MRR"]),
    )


def select_development_candidate(
    metrics_by_variant: Mapping[str, Mapping[str, Any]],
) -> str:
    """Tie-break simpler first: B1 > B2 > B3 when scores equal."""
    order = ["B1", "B2", "B3"]
    best_v = None
    best_score = None
    for v in order:
        s = score_variant_for_selection(metrics_by_variant[v])
        if best_score is None or s > best_score:
            best_score = s
            best_v = v
    assert best_v is not None
    return best_v


def compute_crop_quality_fields(
    crop_bgr: np.ndarray,
    bbox_xyxy_int: Sequence[int],
    *,
    frame_width: int,
    frame_height: int,
    other_bboxes: Sequence[Sequence[int]],
) -> dict[str, Any]:
    img = compute_image_metrics(crop_bgr)
    edge = compute_edge_contacts(bbox_xyxy_int, frame_width=frame_width, frame_height=frame_height)
    overlap = max_other_person_overlap(bbox_xyxy_int, other_bboxes)
    h, w = crop_bgr.shape[:2]
    rel_h = float(h) / float(frame_height) if frame_height else 0.0
    rel_w = float(w) / float(frame_width) if frame_width else 0.0
    return {
        **img,
        **edge,
        "max_other_person_iou": float(overlap),
        "crop_width": int(w),
        "crop_height": int(h),
        "relative_player_height": rel_h,
        "relative_player_width": rel_w,
        "bbox_area": int((bbox_xyxy_int[2] - bbox_xyxy_int[0]) * (bbox_xyxy_int[3] - bbox_xyxy_int[1])),
    }


def group_frame_bboxes(
    observations: Sequence[Mapping[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
) -> dict[int, list[list[int]]]:
    by_frame: dict[int, list[list[int]]] = defaultdict(list)
    for obs in observations:
        fi = int(obs["frame_index"])
        meta = validate_and_clamp_bbox(
            obs["bbox_xyxy"], frame_width=frame_width, frame_height=frame_height
        )
        if meta["eligible"] and meta["bbox_xyxy_int"] is not None:
            by_frame[fi].append(list(meta["bbox_xyxy_int"]))
    return by_frame
