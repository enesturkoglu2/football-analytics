# Stage 4B linking policy freeze (4B-5C1)

- **Date:** 2026-07-21
- **Status:** Visual-review hypotheses and conservative linking policy frozen
- **Product commit baseline:** `7098bf1` (aggregation + candidates)
- **Policy config:** `configs/reid/linking_policy_stage4b.yaml`
- **Benchmark source:** `outputs/reid/limited_benchmark_stage4b_12/`
- **Montages:** `outputs/reid/limited_benchmark_stage4b_12/montages/`

No linking code, `global_candidate_id` output, or automatic acceptance was
produced in this gate. Cosine similarity threshold remains **unset**
(`null`).

## Benchmark context (facts only)

| Item | Value |
|---|---|
| Tracks | 12 |
| Crops | 45 |
| Candidate pairs | 66 |
| Exact-frame conflict pairs | 25 |
| Eligible unthresholded pairs | 41 |
| `automatic_linking_performed` | false |
| `accepted_link` rows | none |
| `similarity_threshold` | null |

Tracks were selected by data-quality / length heuristics from
`configs/reid/benchmark_stage4b.yaml`. Same-person membership is **not**
asserted by pool membership.

## Visual review hypotheses (manual only)

These labels are **manual visual hypotheses** from montage review. They are
**not** ground truth, not product decisions, and **not** automatic
`accepted_link` edges.

### A. `likely_same` (manual hypothesis only)

| Pair | Cosine (approx.) | Crops | Observations | Exact-frame conflict |
|---|---:|---:|---:|---|
| **4 ↔ 682** | ~0.870 | 5 / 5 | 252 / 327 | no |

Notes:

- Recorded only as a manual visual hypothesis.
- **Must not** be auto-marked `accepted_link`.
- Future linking may consider this pair only after explicit manual
  acceptance under the frozen policy below.

### B. `likely_different` (manual hypothesis only)

| Pair | Cosine (approx.) | Crops | Observations | Exact-frame conflict |
|---|---:|---:|---:|---|
| **4 ↔ 459** | ~0.868 | 5 / 5 | 252 / 325 | no |

Notes:

- Kit / appearance difference judged visually distinct.
- Cosine is close to the `likely_same` hypothesis above (~0.870 vs ~0.868).
- This demonstrates that **cosine alone cannot authorize acceptance**.

### C. `uncertain`

| Pair | Notes |
|---|---|
| 2 ↔ 463 | Uncertain |
| 2 ↔ 445 | Uncertain |
| 575 ↔ 603 | Uncertain |
| 464 ↔ 575 | Uncertain |
| 464 ↔ 603 | Uncertain |
| 177 ↔ 463 | Uncertain |

Typical uncertain reasons observed in this review set:

- similar team kits / jersey appearance
- single-crop or low crop count
- very short tracks
- dirty / multi-person crops
- low resolution
- viewpoint / pose differences

`uncertain` pairs **must not** be linked under this Stage 4B policy.

### D. Exact-frame hard-ban controls (`rejected_exact_frame_conflict`)

| Pair | Role |
|---|---|
| 445 ↔ 464 | hard ban control |
| 445 ↔ 463 | hard ban control |
| 459 ↔ 463 | hard ban control |
| 463 ↔ 575 | hard ban control |

These pairs remain **rejected** regardless of cosine magnitude. Shared
observed `frame_index` values are a reliable hard-ban signal.

## Frozen linking policy summary

Full machine-readable freeze:
`configs/reid/linking_policy_stage4b.yaml`
(`schema_version: reid_linking_policy_v1`).

Key freezes:

| Principle | Frozen value |
|---|---|
| Automatic linking | **disabled** |
| Cosine threshold | **null** (unset) |
| Cosine usage | **ranking_only** |
| Exact-frame conflict | **hard reject** |
| Span overlap alone | **not** a hard reject |
| Manual acceptance for linking | **required** |
| Single / low-crop auto-link | **forbidden** |
| Strong-review crop floor | ≥ **3** crops / track |
| Strong-review observation floor | ≥ **30** observations / track |
| Uncontrolled Union-Find chaining | **forbidden** |
| Component merge | all cross-member pairs must be exact-frame conflict-free |
| Raw `track_id` | preserved; product identity uses separate `global_candidate_id` |
| Deterministic global id | `min(raw track_id in component)` |

Allowed review / decision labels:

- `likely_same`
- `likely_different`
- `uncertain`
- `rejected_exact_frame_conflict`

Linking permissions under this freeze:

- `uncertain` → linking **not** allowed
- `likely_different` → linking **not** allowed
- exact-frame conflict → linking **not** allowed
- `likely_same` → still requires **manual acceptance**; not automatic

## Domain and model caveats (explicit)

- The Market1501 OSNet checkpoint is **general person ReID**, not
  football-/SoccerNet-trained.
- The sn-reid repository supplies model software; it is **not** a second
  checkpoint.
- Checkpoint weights and sn-reid code were **not** merged into a new model.
- **No new ReID model was trained** in Stage 4B.
- What was built is an **end-to-end football analytics ReID pipeline**
  (crop → embed → aggregate → candidates → review policy).
- Cosine similarity is **not** accuracy and **not** identity proof.
- Exact-frame conflict **is** a trustworthy hard-ban signal.
- Without ground truth, an optimum cosine threshold **cannot** be chosen.
- Under this Stage 4B policy, **automatic acceptance is off**.
- Future linking code may process **only manually accepted edges**.
- Component merges must pass **all cross-member exact-frame checks**.

## Deferred to later gates

- Product linking / controlled component merge implementation
- `track_global_map.jsonl` / `global_candidate_id` emission
- Any non-null cosine acceptance threshold (only if later explicitly
  approved; not chosen here)
- Full-video linking beyond the 12-track review set

## Gate note

This document freezes policy and manual hypotheses only. It does **not**
run linking, invent a threshold, or rewrite raw ByteTrack IDs.
