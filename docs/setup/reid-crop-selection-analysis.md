# ReID crop selection analysis (Stage 4B-1A)

- **Date:** 2026-07-21
- **Scope:** Read-only analysis of `outputs/tracking/full/tracks.jsonl` + video metadata.
- **No product code, crops, embeddings, linking, or config files were written.**
- **Environment:** `football-cv` only.

## Inputs

| Item | Value |
|---|---|
| Video | `data/test_clips/sample.mp4` (gitignored `*.mp4`) |
| Resolution / FPS / frames | **1336×744**, **30** FPS, **1023** frames |
| Tracks | `outputs/tracking/full/tracks.jsonl` (gitignored `outputs/*`) |
| JSONL fields used | `frame_index`, `timestamp_sec`, `track_id`, `class_id`, `class_name`, `confidence`, `bbox_xyxy` |

## 1. Data integrity

| Check | Result |
|---|---|
| Parsed observation rows | **13309** |
| Bad JSON / missing keys / NaN–Inf skipped | **0 / 0 / 0** |
| Unique `track_id` | **276** |
| `track_id` null | **0** |
| `frame_index` range | **0 … 1022** |
| Class distribution | `class_id=0`, `class_name=person` → 13309 |
| Duplicate `(track_id, frame_index)` keys | **0** |
| Out-of-bounds bbox (vs 1336×744) | **0** |
| Non-positive width/height | **0** |
| Confidence min / max | **0.251 / 0.896** |
| Boxes needing clamp | **0** |
| Edge-touching boxes (`x1≤0` or `y1≤0` or `x2≥W` or `y2≥H`) | **154** |

Integrity is clean enough for crop-selection planning. Edge-touch count is informational (mostly near-boundary players), not geometric invalidity.

## 2. Observation-level percentiles

Valid boxes used: **13309**.  
`quality_score = area × confidence`.  
`aspect_ratio = width / height`.  
`short_side = min(width, height)`.

| Metric | P1 | P5 | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| width | 21.47 | 24.00 | 25.82 | 29.77 | 36.77 | 49.10 | 64.85 | 74.70 | 92.20 |
| height | 48.52 | 56.21 | 60.09 | 71.22 | 87.50 | 109.71 | 130.14 | 144.83 | 183.96 |
| area | 1181.7 | 1452.4 | 1645.5 | 2211.6 | 3224.9 | 5222.6 | 7985.7 | 9708.6 | 16337.0 |
| short_side | 21.47 | 24.00 | 25.82 | 29.77 | 36.77 | 49.10 | 62.13 | 73.07 | 91.24 |
| aspect_ratio | 0.246 | 0.326 | 0.350 | 0.380 | 0.422 | 0.489 | 0.573 | 0.665 | 1.247 |
| confidence | 0.303 | 0.374 | 0.438 | 0.567 | 0.678 | 0.755 | 0.801 | 0.823 | 0.856 |
| quality_score | 510.4 | 736.7 | 911.7 | 1387.5 | 2106.6 | 3480.6 | 5665.0 | 6982.7 | 9263.4 |

## 3. Track-level distribution

| Bucket | Count (of 276) |
|---|---:|
| observation_count = 1 | 72 |
| ≤5 | 124 |
| ≤10 | 148 |
| ≥50 | 86 |
| ≥100 | 50 |

Selected track-level percentiles:

| Metric | P1 | P5 | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| observation_count | 1 | 1 | 1 | 1 | 9 | 67.25 | 166.5 | 230.25 | 322.75 |
| span_frames | 1 | 1 | 1 | 1 | 14 | 77.75 | 169 | 243 | 324 |
| observation_density | 0.118 | 0.444 | 0.667 | 0.897 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| median_confidence | 0.300 | 0.318 | 0.333 | 0.400 | 0.545 | 0.670 | 0.733 | 0.770 | 0.802 |
| max_confidence | 0.306 | 0.321 | 0.341 | 0.434 | 0.662 | 0.794 | 0.845 | 0.873 | 0.885 |
| median_bbox_area | 1077 | 1381 | 1529 | 2003 | 3319 | 5034 | 7796 | 8777 | 11683 |
| max_bbox_area | 1137 | 1446 | 1669 | 2599 | 3988 | 6468 | 9500 | 11996 | 14797 |
| median_short_side | 16.98 | 23.02 | 24.87 | 29.21 | 36.31 | 48.07 | 60.21 | 71.52 | 93.04 |
| max_short_side | 18.41 | 23.82 | 26.40 | 31.94 | 43.53 | 56.67 | 74.64 | 88.37 | 97.79 |
| median_quality_score | 407 | 561 | 705 | 1005 | 1594 | 2919 | 4515 | 5532 | 7836 |
| max_quality_score | 441 | 582 | 741 | 1201 | 2092 | 4236 | 6174 | 8386 | 11105 |

Fragmentation remains high: **148/276** tracks have ≤10 observations. This is why ReID linking is motivated, but also why short-track merges are risky.

## 4. Within-track frame-gap distribution

Consecutive observation deltas on the same track (`n_gaps = 13033`):

| Statistic | Value |
|---|---:|
| delta = 1 ratio | **97.80%** |
| delta ≤ 2 | 98.53% |
| delta ≤ 5 | 99.28% |
| delta ≤ 10 | 99.68% |
| P50 / P75 / P90 / P95 | **1 / 1 / 1 / 1** |
| P99 | **4** |
| max / mean | **31 / 1.115** |

Gaps **> 1** only: **287** deltas → P50=**4**, P75=**7**, P90=**14**.

### Gap candidates for `min_frame_gap_within_track` (not written to config)

| Candidate | Derivation | Diversity role |
|---|---|---|
| 1 | All-gap P90/P95 (mode) | **Not useful for diversity** (allows consecutive frames) |
| 4 | Gaps>1 P50; also near all-gap P99 | Mild temporal spread |
| 7 | Gaps>1 P75 | Moderate spread |
| 14 | Gaps>1 P90 | Stronger spread (~0.47 s at 30 FPS) |

Simulations below use **1, 7, 14** (data-derived set of three). For 4B-1B, prefer choosing among **4 / 7 / 14**; treat **1** as a non-diversity baseline only.

## 5. Temporal pair analysis (critical design freeze)

Unique unordered track pairs: \(276\times275/2 =\) **37950**.

| Category | Count | Share |
|---|---:|---:|
| **A. exact_frame_conflict** (observation frame sets intersect) | **3650** | 9.62% |
| **B. span_interval_overlap only** (`[first,last]` overlaps, but **no** shared exact frame) | **250** | 0.66% |
| **C. no temporal overlap** | **34050** | 89.72% |

### Frozen linking rules (analysis only — no linking run)

1. **Hard temporal conflict:** if two tracks share any exact `frame_index` observation, they **must not** receive the same `global_candidate_id`.
2. **Span overlap ≠ exact conflict:** the 250 span-only pairs are temporally adjacent/overlapping intervals without co-occurrence; they may be ReID candidates subject to similarity + gap rules, but are **not** auto-rejected by the hard exact-frame rule.
3. **Future component merge (not Union-Find alone):** before merging two components, **every cross-member pair** must pass the exact-frame conflict check. If any cross pair conflicts, **reject that component merge**. Candidate edges should be processed in similarity order. Deterministic `global_candidate_id` may use `min(raw track_id)` inside the component.

## 6. Crop filter profiles (candidates — none selected yet)

All thresholds taken from **observation-level percentiles** of area / short_side / confidence.

| Profile | min_bbox_area | min_short_side | min_confidence | Derivation |
|---|---:|---:|---:|---|
| permissive | 1452.4 | 24.00 | 0.374 | obs **P5** each |
| balanced | 2211.6 | 29.77 | 0.567 | obs **P25** each |
| conservative | 3224.9 | 36.77 | 0.678 | obs **P50** each |

### Retention on this video

| Profile | Kept obs | Kept % | Tracks ≥1 crop | Tracks 0 crop | Short tracks (≤10 obs) with ≥1 crop |
|---|---:|---:|---:|---:|---:|
| permissive | 11874 | 89.2% | 214 | 62 | 87 / 148 |
| balanced | 7370 | 55.4% | 135 | 141 | 29 / 148 |
| conservative | 3456 | 26.0% | 87 | 189 | 13 / 148 |

No profile is adopted as config in this gate.

## 7. Crop count × gap simulations

**Filter used for simulation:** balanced profile.  
**Selection rule:** per track, keep best `quality_score` per frame among passers → sort by `quality_score` desc → greedy enforce `min_frame_gap_within_track` → cap at `max_crops_per_track`.  
**No JPEG written.**

| max_crops | min_gap | Total crops | Tracks w/ crops | Mean / median crops | At max | Exactly 1 crop |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 374 | 135 | 2.77 / 3 | 115 | 11 |
| 3 | 7 | 317 | 135 | 2.35 / 3 | 81 | 34 |
| 3 | 14 | 293 | 135 | 2.17 / 3 | 68 | 45 |
| 5 | 1 | 587 | 135 | 4.35 / 5 | 103 | 11 |
| 5 | 7 | 454 | 135 | 3.36 / 4 | 67 | 34 |
| 5 | 14 | 397 | 135 | 2.94 / 3 | 48 | 45 |
| 8 | 1 | 877 | 135 | 6.50 / 8 | 91 | 11 |
| 8 | 7 | 617 | 135 | 4.57 / 4 | 46 | 34 |
| 8 | 14 | 484 | 135 | 3.59 / 3 | 24 | 45 |

Interpretation: gap=1 maximizes crop count but wastes near-duplicates; gap=7/14 trades quantity for temporal diversity. Full-video crop budgets under balanced×(5,7) ≈ **454** JPEGs (order-of-magnitude for disk/CPU planning only).

## 8. Limited benchmark track candidates

**Not identity labels.** Lists are quality/length heuristics only. Selectable crop counts use the **balanced** filter (unique frames passing thresholds).

### A. Long / high-coverage tracks (12)

| track_id | n_obs | span | first–last | median/max area | median/max conf | selectable (bal.) |
|---:|---:|---:|---|---:|---:|---:|
| 463 | 371 | 382 | 384–765 | 4139 / 6583 | 0.769 / 0.877 | 360 |
| 682 | 327 | 327 | 696–1022 | 3400 / 6177 | 0.772 / 0.874 | 308 |
| 459 | 325 | 370 | 383–752 | 4039 / 9843 | 0.782 / 0.885 | 298 |
| 445 | 322 | 323 | 379–701 | 6781 / 10147 | 0.812 / 0.894 | 321 |
| 4 | 252 | 252 | 0–251 | 4203 / 7402 | 0.743 / 0.798 | 166 |
| 13 | 252 | 252 | 0–251 | 4178 / 4877 | 0.621 / 0.820 | 145 |
| 2 | 252 | 252 | 0–251 | 3018 / 4869 | 0.664 / 0.820 | 214 |
| 16 | 249 | 249 | 0–248 | 16370 / 17781 | 0.487 / 0.585 | 16 |
| 703 | 247 | 265 | 758–1022 | 1942 / 3651 | 0.630 / 0.811 | 63 |
| 568 | 246 | 301 | 516–816 | 8000 / 10617 | 0.797 / 0.876 | 242 |
| 7 | 246 | 256 | 0–255 | 3020 / 5540 | 0.652 / 0.785 | 206 |
| 31 | 237 | 250 | 24–273 | 2736 / 7434 | 0.652 / 0.776 | 192 |

Note: track **16** is long but median conf is below balanced `min_confidence` for many frames → only 16 selectable crops; useful stress case, not a “best crop” exemplar.

### B. Short tracks with strong crop quality (10)

| track_id | n_obs | span | first–last | median/max area | median/max conf | selectable (bal.) |
|---:|---:|---:|---|---:|---:|---:|
| 464 | 3 | 3 | 384–386 | 10465 / 10915 | 0.783 / 0.831 | 2 |
| 575 | 4 | 8 | 540–547 | 10266 / 11417 | 0.657 / 0.720 | 3 |
| 177 | 2 | 11 | 212–222 | 8209 / 9530 | 0.687 / 0.741 | 2 |
| 603 | 3 | 3 | 583–585 | 8072 / 9228 | 0.665 / 0.684 | 3 |
| 186 | 5 | 35 | 221–255 | 7307 / 7596 | 0.673 / 0.718 | 4 |
| 524 | 9 | 14 | 467–480 | 8372 / 9029 | 0.476 / 0.715 | 2 |
| 248 | 3 | 3 | 260–262 | 6312 / 6379 | 0.660 / 0.661 | 2 |
| 481 | 10 | 26 | 399–424 | 5737 / 6430 | 0.762 / 0.811 | 6 |
| 252 | 9 | 9 | 262–270 | 6033 / 6387 | 0.652 / 0.749 | 6 |
| 479 | 1 | 1 | 397–397 | 6214 / 6214 | 0.598 / 0.598 | 1 |

These are for **smoke / risk** testing (few observations, large boxes), not claimed identity merges.

## 9. Schema decisions frozen for later gates

### Candidate pair record (minimum fields)

```text
track_id_a
track_id_b
cosine_similarity
temporal_gap_frames
exact_frame_overlap_count
span_interval_overlap
decision
decision_reason
```

### Global map — avoid a single ambiguous `confidence`

Prefer:

```text
track_id
global_candidate_id
linked_track_ids
component_similarity_min    # null if singleton
component_similarity_mean   # null if singleton
accepted_edge_count         # 0 if singleton
evidence_count
temporal_gap_frames         # representative / min gap among accepted edges; null if singleton
decision_reason
model / checkpoint metadata
```

Singletons: similarity fields **null**.

### Component merge safety

- Sort candidate edges by cosine similarity (desc).
- Attempt merge only if **all** cross-member pairs have `exact_frame_overlap_count == 0`.
- Reject merge on any exact-frame conflict.
- Do **not** rely on naive Union-Find without this check.
- Deterministic id: `global_candidate_id = min(raw track_id in component)` (candidate rule).

## 10. Metrics explicitly not produced

No ID-switch, MOTA, HOTA, ReID mAP, or accuracy percentages. This report is descriptive only.

## 11. Awaiting Stage 4B-1B user choices

Do **not** write `configs/reid/*.yaml` until approved:

1. Crop filter profile: **permissive / balanced / conservative**
2. `max_crops_per_track`: **3 / 5 / 8**
3. `min_frame_gap_within_track`: prefer **4 / 7 / 14** (diversity); gap **1** only if near-duplicate crops are intentionally allowed
4. First limited benchmark track set: confirm/adjust lists **A** and **B**
5. Confirm frozen schema + component exact-frame merge rule above

## 12. Suggested default package for 4B-1B discussion (not adopted)

If a single starting point is needed for debate (still requires explicit approval):

- profile **balanced** (P25; retains 55% obs / 135 tracks)
- `max_crops_per_track = 5`
- `min_frame_gap_within_track = 7` (gaps>1 P75)
- limited smoke tracks: first 8 of list A + 6 of list B

This is a **discussion default**, not an implemented config.
