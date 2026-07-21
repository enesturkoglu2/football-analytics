# Stage 5A — Crop Quality Visual Validation Report

- **Date:** 2026-07-21
- **Gate:** Stage 5A3C — documentation and usage-policy freeze
- **Measurement commit:** `d6122d0` — Implement ReID crop quality analysis
- **Policy config:** `configs/reid/crop_quality_policy_stage5a.yaml`
- **Stage 5A status:** `visually_validated_measurement_baseline`

## A. Scope

Full `sample.mp4` Stage 4B crops were measured and manually reviewed
with diagnostic panels only:

| Item | Value |
|---|---|
| Crops measured | 454 |
| Crop-producing tracks | 135 |
| Frame-edge contact crops | 13 |
| Tracking-bbox overlap crops | 134 |
| Center-inside crops | 26 |
| Union coverage max | 0.938 |
| Unique diagnostic panels | 45 |
| Contact sheets | 8 |
| `quality_threshold` | null |
| `contamination_threshold` | null |
| Automatic exclusion | false |

Review package (Git-ignored):

- `outputs/reid/full_stage4b/quality_stage5a/`
- `outputs/reid/full_stage4b/quality_review_stage5a/`
- `outputs/reid/full_stage4b/reid_quality_review_stage5a.zip`

Important:

- No threshold or exclusion was applied.
- Review outcomes are **manual diagnostic observations**.
- They are **not** ground-truth quality annotations.
- No automatic `good` / `bad` / `blurry` / `clean` / `contaminated`
  labels were produced.

Panel groups reviewed:

- `contamination_high`: 12
- `low_sharpness_stratified`: 10
- `frame_edge_cases`: 13
- selection label `clean_controls` (conceptually
  **zero_tracking_overlap_controls**): 10

## B. Tracking-bbox contamination findings

Manual visual review of the `contamination_high` set (12 crops with
highest union coverage) found that **all 12** showed at least some
visible non-target-person content.

The metric therefore **directionally ranked** high-risk examples well
for manual review.

Limits of the metric:

- Union / max coverage is **not** the fraction of visible foreign-person
  pixels.
- It measures geometric intersection of tracking rectangles.
- Coverage magnitude is **not** calibrated to visual contamination
  severity.

Examples (manual visual observation only):

| Crop | Union | Observation |
|---|---:|---|
| 816@872 | 0.938 | Multiple visible people / strong contamination evidence |
| 891@1006 | 0.912 | Visible overlapping white-kit player |
| 7@232 | 0.745 | Multiple visible non-target people |
| 177@212 | 0.734 | Large geometric bbox overlap but comparatively limited visible non-target content |

## C. Zero-overlap false negatives

The selection group name `clean_controls` is only a selection-pool label.
It is **not** proof of cleanliness.

Correct conceptual name:

`zero_tracking_overlap_controls`

Manual visual observation found visible second-person content despite
`union_other_person_crop_coverage == 0` in:

- 514@446
- 514@454
- 558@517
- 701@937

Manual visual observation also found crops that look more like
off-pitch / sideline people than on-pitch players:

- 13@30
- 13@131
- 13@145
- 16@128

These are **not** identity ground truth.

Conclusions:

- `overlap > 0` may be positive contamination evidence.
- `overlap == 0` is **not** clean proof.
- People missing from tracking cannot be measured by bbox-overlap
  contamination.

## D. Sharpness findings

Within the same short-side bins, low-Laplacian stratified examples
generally looked softer / more motion-smeared than high-Laplacian
controls.

Laplacian is a **directionally useful ranking signal**, but:

- a global threshold is **not** appropriate
- native crop size affects the metric
- larger crops can systematically show lower Laplacian values
- background edges, jersey digits, railings, and second people can
  raise Laplacian
- high Laplacian is **not** identity-usability proof

Visually soft examples (no ground-truth blur label):

- 186@255
- 223@305
- 224@397

Counter-examples:

- 13@131 — high Laplacian, but off-pitch / non-player-like subject
- 514@454 — very high Laplacian, but visible multi-person content

Frozen sharpness policy:

- global Laplacian threshold: **prohibited**
- size-stratified audit/ranking: **allowed**
- binary exclusion: **disabled**
- future task-specific usage required

Task-specific notes:

- jersey-number visibility/OCR may later use sharpness as a strong
  helper
- team/kit color may still use soft crops
- OSNet aggregation weighting may change only after separate validation

## E. Frame-edge findings

Frame-edge contact is **not** proof of body truncation.

Manual visual observations:

- 6@49 and 268@356 largely look usable / full-player
- 112@162 is a clear partial-player example
- 607@596 shows body/bbox truncation **without** video-frame edge
  contact

Conclusion:

`frame_edge_contact != body_completeness`

Frame-edge policy:

- audit only
- not a hard reject
- not automatic exclusion

## F. Missing quality dimensions

Three future signals remain unimplemented:

### 1. `subject_domain_validity`

- on-pitch player
- off-pitch person
- referee / goalkeeper / outlier
- unknown

Not an automatic decision in this baseline.

### 2. `body_completeness`

- head / torso / back / legs visibility
- player occupancy
- bbox-internal truncation
- usable torso/back region

Separate from frame-edge contact.

### 3. `pixel_level_multi_person_contamination`

- second-person content even when tracking has no overlapping bbox
- future options may include detector rerun, segmentation, or
  pose/foreground methods
- **no model is selected or downloaded in this documentation gate**

## G. Region / task-specific quality

Crop-level binary usable/unusable decisions are too coarse for Stage 5
downstream signals:

- kit descriptors can use a torso region
- jersey-number systems can use a torso/back region
- contamination on the lower body does not automatically invalidate a
  clean back number
- signal usage should remain task-specific

## H. Frozen Stage 5A decision

Stage 5A quality status:

**`visually_validated_measurement_baseline`**

Frozen values:

| Field | Value |
|---|---|
| `quality_threshold` | null |
| `contamination_threshold` | null |
| `automatic_exclusion` | false |
| `embedding_weighting` | false |
| `crop_files_modified` | false |
| `embeddings_modified` | false |
| `linking_rerun` | false |

Allowed current usage:

| Signal | Allowed usage |
|---|---|
| Tracking bbox overlap | `audit_and_manual_review_ranking` |
| Native Laplacian | `size_stratified_audit_and_ranking` |
| Frame edge | `audit_only` |

Next implementation gate after this freeze:

**Stage 5B1 — coarse team/kit descriptor** (separate approval).
