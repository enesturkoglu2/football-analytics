# Stage 4B schema and linking decisions (4B-1B freeze)

- **Date:** 2026-07-21
- **Status:** Decisions frozen for implementation gates 4B-2+
- **Source analysis (unchanged):** `docs/setup/reid-crop-selection-analysis.md`
- **Crop config:** `configs/reid/crop_selection_stage4b.yaml`
- **Benchmark config:** `configs/reid/benchmark_stage4b.yaml`

No product code, crops, embeddings, or similarity runs were produced in this gate.

## Approved crop settings

| Setting | Value | Source |
|---|---|---|
| profile_name | `balanced` | 4B-1B approval |
| min_bbox_area | **2211.6** | analysis report balanced table (obs P25) |
| min_short_side | **29.77** | same |
| min_confidence | **0.567** | same |
| quality_score | `bbox_area * detection_confidence` | approved |
| max_crops_per_track | **5** | approved |
| min_frame_gap_within_track | **7** | approved (gaps>1 P75 diversity candidate) |
| ordering | quality_score desc → confidence desc → frame_index asc | approved |
| selection | greedy frame-gap | approved |
| allow_single_crop_embedding | true | eligibility only; **not** auto-link |
| minimum_crop_count_for_embedding | 1 | same |

Raw ByteTrack `track_id` values are preserved forever; product identity uses separate `global_candidate_id`.

Expected balanced×(5,7) budget on this video (from analysis simulation): ~454 crops across 135 tracks (planning only).

## Approved benchmark pools

Pools are **data-quality heuristics**, not identity labels. Same-person membership is **not** asserted. Linking benchmark subset choice is deferred to **4B-5**.

### smoke_tracks (first crop / embedding smoke)

`463`, `682`, `464`

- `463`, `682` ⊂ `long_quality_pool`
- `464` ⊂ `short_quality_pool`

### long_quality_pool (A)

`463`, `682`, `459`, `445`, `4`, `13`, `2`, `16`, `703`, `568`, `7`, `31`

### short_quality_pool (B)

`464`, `575`, `177`, `603`, `186`, `524`, `248`, `481`, `252`, `479`

---

## Output schemas

All JSON/JSONL values must be finite (no NaN / Infinity).  
Relative paths only in manifests (no machine-specific absolute paths required by schema).

### A. `crop_manifest.jsonl`

One row per selected crop. `schema_version`: `reid_crop_manifest_v1`

| Field | Type / notes |
|---|---|
| crop_id | stable string id (implementation-defined, deterministic) |
| track_id | int (raw ByteTrack id) |
| frame_index | int |
| timestamp_sec | float |
| source_video | relative path string |
| bbox_xyxy | `[x1,y1,x2,y2]` floats |
| bbox_width | float |
| bbox_height | float |
| bbox_area | float |
| short_side | float |
| detection_confidence | float |
| quality_score | float (`area * confidence`) |
| crop_relative_path | relative path under output dir |
| selection_rank | int (1 = highest within track after ordering) |
| schema_version | `reid_crop_manifest_v1` |

### B. `crop_embeddings_index.jsonl`

One row per crop embedding. `schema_version`: `reid_crop_embeddings_index_v1`

| Field | Type / notes |
|---|---|
| crop_id | joins to manifest |
| track_id | int |
| frame_index | int |
| embedding_row | int index into NPZ vectors |
| embedding_shape | e.g. `[512]` |
| embedding_dtype | e.g. `float32` |
| l2_norm | float (~1.0 after normalize) |
| model_name | e.g. `osnet_x1_0` |
| checkpoint_sha256 | hex string |
| preprocessing | short descriptor / object (resize 256×128, ImageNet mean/std) |
| schema_version | `reid_crop_embeddings_index_v1` |

Companion vector file: `crop_embeddings.npz` (aligned by `embedding_row`).

### C. `track_embeddings.jsonl`

One row per track that has ≥1 crop embedding. `schema_version`: `reid_track_embeddings_v1`

| Field | Type / notes |
|---|---|
| track_id | int |
| crop_ids | list of crop_id |
| crop_count | int |
| embedding_row | int into `track_embeddings.npz` |
| aggregation | baseline `l2_mean` |
| embedding_shape | `[512]` |
| l2_norm | float |
| observation_count | int (raw track observations) |
| first_frame | int |
| last_frame | int |
| observed_frame_count | int (unique observed frames) |
| schema_version | `reid_track_embeddings_v1` |

### D. `candidate_pairs.jsonl`

One row per evaluated unordered pair (or directed with canonical `track_id_a < track_id_b`).  
`schema_version`: `reid_candidate_pairs_v1`

| Field | Type / notes |
|---|---|
| track_id_a | int |
| track_id_b | int |
| cosine_similarity | float or null if not computed |
| temporal_gap_frames | int (≥0); definition fixed in 4B-4 |
| exact_frame_overlap_count | int |
| exact_frame_conflict | bool (`exact_frame_overlap_count > 0`) |
| span_interval_overlap | bool |
| decision | e.g. `reject_exact_frame_conflict` / `candidate` / `reject_below_threshold` / … |
| decision_reason | short string |
| schema_version | `reid_candidate_pairs_v1` |

### E. `track_global_map.jsonl`

**Exactly one row per raw `track_id`** present in the run’s track set.  
`schema_version`: `reid_track_global_map_v1`

| Field | Type / notes |
|---|---|
| track_id | int (raw; never rewritten) |
| global_candidate_id | int; deterministic: `min(raw track_id in component)` |
| linked_track_ids | sorted list including self |
| component_similarity_min | float or **null** if singleton / no accepted edges |
| component_similarity_mean | float or **null** if singleton / no accepted edges |
| accepted_edge_count | int (`0` for singleton) |
| decision_reason | string |
| crop_count | int (0 if none selected) |
| has_embedding | bool |
| model_name | string or null if no embedding |
| checkpoint_sha256 | string or null if no embedding |
| aggregation | string or null if no embedding |
| schema_version | `reid_track_global_map_v1` |

Rules:

- Do **not** use a single ambiguous `confidence` field.
- Singleton similarity fields are **null**.
- Tracks with `has_embedding=false` remain singletons (`linked_track_ids=[track_id]`, `accepted_edge_count=0`).
- `component_similarity_*` computed only from **accepted** component edges.
- JSON must not contain NaN/Infinity.

---

## Temporal and component linking decisions

### Hard ban — exact-frame conflict

If two tracks share any common observed `frame_index`, they **must not** share a `global_candidate_id`.  
Record `exact_frame_conflict=true` and reject linking for that pair.

### Span interval overlap — metadata only

`[first_frame, last_frame]` intersection alone is **not** a hard reject.  
Store `span_interval_overlap` on candidate rows. Exact-frame conflict remains the hard temporal rule.

### Component merge (not uncontrolled Union-Find)

1. Process candidate edges in **decreasing cosine similarity** order.
2. Before merging two components, check **all cross-member pairs** for exact-frame conflict.
3. If any cross-member pair conflicts → **reject that component merge**.
4. Count rejected component merges in `reid_summary` diagnostics.
5. Uncontrolled Union-Find chaining without cross-member exact-frame checks is **forbidden**.

### Explicitly deferred

- Cosine similarity **threshold** — not chosen in 4B-1B (4B-4 / 4B-5).
- Team / jersey cues — unavailable; wrong-merge risk remains elevated.
- Ground-truth metrics — **no** ID-switch, MOTA, HOTA, ReID mAP, or accuracy %.

### Diagnostic counts (for later summaries)

Among others: raw track count, embedding coverage, global candidate count, pairs rejected for exact-frame conflict, component merges rejected for cross-member conflict, accepted edge count, similarity histogram summaries (when thresholds exist).

---

## Next gate

**4B-2:** crop select/extract product code + mock tests, using these configs/schemas.  
Separate explicit approval required before implementation.
