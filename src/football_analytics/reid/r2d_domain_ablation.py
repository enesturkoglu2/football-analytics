"""ReID-R2D SportsReID single-crop domain ablation helpers (no R2B imports)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


class R2DError(RuntimeError):
    """Raised when R2D domain ablation contract fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    import json

    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(payload.encode("utf-8"))


def resolve_crop_file(rel: str, search_roots: Sequence[Path], project_root: Path) -> Path:
    rel = rel.lstrip("./")
    for base in search_roots:
        candidate = base / rel
        if candidate.is_file():
            return candidate.resolve()
    name = Path(rel).name
    hits = list(project_root.glob(f"outputs/reid/**/{name}"))
    if hits:
        return hits[0].resolve()
    raise R2DError(f"crop not found for relative path: {rel}")


def argmax_with_id_tiebreak(scores: np.ndarray, ids: Sequence[str]) -> tuple[float, str]:
    best_val = float("-inf")
    best_id: str | None = None
    for val, mid in zip(scores.tolist(), ids):
        fval = float(val)
        if fval > best_val or (
            math.isclose(fval, best_val) and (best_id is None or mid < best_id)
        ):
            best_val = fval
            best_id = mid
    if best_id is None:
        raise R2DError("argmax_with_id_tiebreak failed")
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
        raise R2DError("Q rows != query_meta")
    if Q.ndim != 2 or T.ndim != 2 or D.ndim != 2:
        raise R2DError("Q/T/D must be 2-D")
    C_t = (Q @ T.T).astype(np.float32)
    C_d = (Q @ D.T).astype(np.float32)
    rows: list[dict[str, Any]] = []
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
    out: list[dict[str, Any]] = []
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
    total = 0.0
    for rank in sorted(int(r) for r in pos_ranks):
        hits += 1
        total += hits / float(rank)
    return float(total / n_pos)


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
        "positive_mean_rank": float(np.mean(pos_ranks)) if pos_ranks else None,
        "n_pos": n_pos,
        "n_neg": len(labels) - n_pos,
    }


def cohort_auroc_ap(
    joined: Sequence[Mapping[str, Any]],
    *,
    negative_predicate: Callable[[Mapping[str, Any]], bool],
) -> dict[str, float]:
    rows: list[Mapping[str, Any]] = []
    for row in joined:
        if int(row["binary_clean_player_label"]) == 1:
            rows.append(row)
        elif negative_predicate(row):
            rows.append(row)
    ranked = sorted(rows, key=lambda r: -float(r["S_primary"]))
    labels = [int(r["binary_clean_player_label"]) for r in ranked]
    scores = [float(r["S_primary"]) for r in ranked]
    metrics = ranking_metrics(labels, scores, n_pos=sum(labels))
    return {"AUROC": float(metrics["AUROC"]), "AP": float(metrics["AP"])}


def composition(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "target": sum(1 for r in rows if int(r["binary_clean_player_label"]) == 1),
        "same_team": sum(1 for r in rows if r.get("same_team_negative_cohort")),
        "other_team": sum(1 for r in rows if r.get("other_team_negative_cohort")),
    }


def evaluate_joined(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = sorted(joined, key=lambda r: int(r["rank"]))
    labels = [int(r["binary_clean_player_label"]) for r in ranked]
    scores = [float(r["S_primary"]) for r in ranked]
    base = ranking_metrics(labels, scores, n_pos=sum(labels))
    same_team = cohort_auroc_ap(
        joined, negative_predicate=lambda r: bool(r.get("same_team_negative_cohort"))
    )
    other_team = cohort_auroc_ap(
        joined, negative_predicate=lambda r: bool(r.get("other_team_negative_cohort"))
    )
    return {
        **base,
        "same_team_AUROC": same_team["AUROC"],
        "same_team_AP": same_team["AP"],
        "other_team_AUROC": other_team["AUROC"],
        "other_team_AP": other_team["AP"],
        "top5_composition": composition(ranked[:5]),
        "top10_composition": composition(ranked[:10]),
        "top20_composition": composition(ranked[:20]),
        "query_count": len(ranked),
    }


def classify_football_domain_outcome(
    *,
    metrics_a: Mapping[str, Any],
    metrics_c: Mapping[str, Any],
    query_drop: int,
    deterministic: bool,
) -> str:
    ap_a, ap_c = float(metrics_a["AP"]), float(metrics_c["AP"])
    au_a, au_c = float(metrics_a["AUROC"]), float(metrics_c["AUROC"])
    st_a, st_c = float(metrics_a["same_team_AUROC"]), float(metrics_c["same_team_AUROC"])
    r10_c = float(metrics_c["Recall@10"])
    r5_c = float(metrics_c["Recall@5"])
    med_a = float(metrics_a["positive_median_rank"])
    med_c = float(metrics_c["positive_median_rank"])
    mar_a, mar_c = float(metrics_a["margin"]), float(metrics_c["margin"])

    strong = (
        query_drop == 0
        and deterministic
        and (ap_c - ap_a) >= 0.20
        and (au_c - au_a) >= 0.15
        and (st_c - st_a) >= 0.10
        and r10_c >= 0.60
        and r5_c >= 0.30
        and med_c < med_a
        and mar_c > mar_a
    )
    if strong:
        return "FOOTBALL_DOMAIN_STRONG_IMPROVEMENT"

    directional = (
        query_drop == 0
        and ap_c > ap_a
        and au_c > au_a
        and st_c >= st_a
        and med_c < med_a
    )
    if directional:
        return "FOOTBALL_DOMAIN_DIRECTIONAL_IMPROVEMENT"

    improved = (ap_c > ap_a) or (au_c > au_a) or (med_c < med_a)
    worsened = (ap_c < ap_a) or (au_c < au_a) or (med_c > med_a) or (st_c < st_a)
    if improved and worsened:
        return "FOOTBALL_DOMAIN_MIXED_OR_OVERLAPPING"
    if (ap_c < ap_a and au_c < au_a) or (
        med_c > med_a and ap_c <= ap_a and au_c <= au_a
    ):
        return "FOOTBALL_DOMAIN_REGRESSION"
    if not improved:
        return "FOOTBALL_DOMAIN_NO_MEANINGFUL_IMPROVEMENT"
    return "FOOTBALL_DOMAIN_MIXED_OR_OVERLAPPING"


def candidate_review_outcome(metrics_c: Mapping[str, Any]) -> str:
    r5 = float(metrics_c["Recall@5"])
    r10 = float(metrics_c["Recall@10"])
    if r5 >= 0.70:
        return "CANDIDATE_REVIEW_PROMISING"
    if r10 >= 0.70:
        return "CANDIDATE_REVIEW_LIMITED"
    return "CANDIDATE_REVIEW_NOT_USEFUL"


def outcome_to_final_status(outcome: str) -> str:
    return {
        "FOOTBALL_DOMAIN_STRONG_IMPROVEMENT": "COMPLETED_R2D_FOOTBALL_DOMAIN_STRONG_IMPROVEMENT",
        "FOOTBALL_DOMAIN_DIRECTIONAL_IMPROVEMENT": "COMPLETED_R2D_FOOTBALL_DOMAIN_DIRECTIONAL_IMPROVEMENT",
        "FOOTBALL_DOMAIN_MIXED_OR_OVERLAPPING": "COMPLETED_R2D_FOOTBALL_DOMAIN_MIXED_OR_OVERLAPPING",
        "FOOTBALL_DOMAIN_NO_MEANINGFUL_IMPROVEMENT": "COMPLETED_R2D_FOOTBALL_DOMAIN_NO_MEANINGFUL_IMPROVEMENT",
        "FOOTBALL_DOMAIN_REGRESSION": "COMPLETED_R2D_FOOTBALL_DOMAIN_REGRESSION",
    }[outcome]


def next_gate_for_outcome(outcome: str) -> str:
    return {
        "FOOTBALL_DOMAIN_STRONG_IMPROVEMENT": "REID_R2E_NEW_INDEPENDENT_HOLDOUT_PREREGISTRATION",
        "FOOTBALL_DOMAIN_DIRECTIONAL_IMPROVEMENT": "REID_R2E_SPORTSREID_MULTIFRAME_OR_NEW_HOLDOUT_DECISION",
        "FOOTBALL_DOMAIN_MIXED_OR_OVERLAPPING": "REID_R2E_PART_BASED_MODEL_OR_HUMAN_IN_LOOP_PIVOT",
        "FOOTBALL_DOMAIN_NO_MEANINGFUL_IMPROVEMENT": "REID_R2E_PART_BASED_MODEL_OR_HUMAN_IN_LOOP_PIVOT",
        "FOOTBALL_DOMAIN_REGRESSION": "REID_R2E_HUMAN_IN_LOOP_PRIMARY_AND_ALTERNATIVE_MODEL_REVIEW",
    }[outcome]


def validate_embedding_matrix(
    matrix: np.ndarray,
    *,
    expected_rows: int,
    dim: int = 512,
    max_abs_diff: float | None = None,
) -> dict[str, Any]:
    if matrix.shape != (expected_rows, dim):
        raise R2DError(f"unexpected embedding shape {matrix.shape}")
    if matrix.dtype != np.float32:
        raise R2DError("embeddings must be float32")
    if not np.isfinite(matrix).all():
        raise R2DError("non-finite embeddings")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms < 1e-6):
        raise R2DError("zero embedding vector detected")
    report = {
        "rows": int(expected_rows),
        "dim": int(dim),
        "nan_count": int(np.isnan(matrix).sum()),
        "inf_count": int(np.isinf(matrix).sum()),
        "zero_vector_count": int(np.sum(norms < 1e-6)),
        "l2_norm_min": float(np.min(norms)),
        "l2_norm_max": float(np.max(norms)),
        "deterministic_max_abs_diff": max_abs_diff,
    }
    if max_abs_diff is not None and max_abs_diff != 0.0:
        raise R2DError(f"determinism failure: max_abs_diff={max_abs_diff}")
    return report
