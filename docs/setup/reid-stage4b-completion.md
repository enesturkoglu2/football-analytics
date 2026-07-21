# Stage 4B — Track-level ReID Baseline Completion Report

## A. Scope

- Input video: `data/test_clips/sample.mp4`
- Tracking input: `outputs/tracking/full/tracks.jsonl`
- ReID goal: safely associate fragmented ByteTrack tracks under separate
  `global_candidate_id` values when explicit manual approval allows it
- Raw ByteTrack `track_id` values are never rewritten

## B. Architecture

File-based environment boundary:

`football-cv`:

- crop select/extract
- aggregation
- candidate generation
- linking
- unit tests

`sn-reid-cpu`:

- embedding inference only

Pipeline flow:

```text
sample.mp4
→ tracking JSONL
→ crop manifest/JPEG
→ crop embeddings
→ track embeddings
→ candidate pairs
→ manual decisions
→ global ID map
```

## C. Model information

- Model architecture: `osnet_x1_0`
- Checkpoint: `osnet_x1_0_market1501_softmax_256x128.pth.tar`
- Checkpoint SHA-256:
  `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`
- Training dataset: Market1501
- Checkpoint type: general person ReID
- SoccerNet-trained: false
- `pretrained` flag during model construction: false
- FeatureExtractor: not used
- Automatic model download: false
- Device: CPU
- sn-reid repository commit: `621e2b0f2d2a7a3e207b8dd747542b6608bf72db`

Clarifications:

- sn-reid is not a second checkpoint.
- sn-reid provides model architecture and weight-loading infrastructure.
- Market1501 checkpoint weights were not merged with any other sn-reid
  weights.
- No new model was trained in Stage 4B.
- What was delivered is a new end-to-end football ReID *pipeline*
  (crop → embed → aggregate → candidates → manual link), not a
  football-domain fine-tuned ReID network.

## D. Crop and embedding results

Full `sample.mp4` run under `outputs/reid/full_stage4b/`:

- raw observations: 13,309
- raw tracks: 276
- crop count: 454
- crop-producing tracks: 135
- no-crop tracks: 141
- crop embedding shape: `[454, 512]`
- track embedding shape: `[135, 512]`
- aggregation: `l2_mean`
- embedding dtype: `float32`
- L2 normalized: true

Crop histogram (tracks with crops):

- 1 crop: 34 tracks
- 2 crops: 20
- 3 crops: 11
- 4 crops: 3
- 5 crops: 67

Crop selection config: `configs/reid/crop_selection_stage4b.yaml`
(balanced profile; max 5 crops/track; min frame gap 7).

## E. Candidate results

- total pairs: 9,045 (`T * (T - 1) / 2` for `T = 135`)
- exact-frame conflicts: 1,525
- eligible_unthresholded: 7,520
- span overlap: 1,574
- span-only overlap: 49
- temporally disjoint: 7,471
- `similarity_threshold`: null
- automatic linking: false

Evidence classes among embedded tracks (policy minima: ≥3 crops and
≥30 observations for strong review):

- strong: 75
- low_crop: 54
- short: 6

Interpretation limits:

- Cosine similarity is a ranking/audit signal only.
- High cosine is not identity proof.
- Exact-frame conflict remains a hard ban.

## F. Full manual review and linking

- Shortlist reviewed: 30 strong/strong montages
  (`outputs/reid/full_stage4b/review_montages_strong_top30/`)
- Total manual decisions: 42
  (`outputs/reid/full_stage4b/manual_decisions_stage4b.jsonl`)
  - includes the earlier limited-benchmark 12 decisions plus 30
    strong/strong review decisions
- likely_same: 11
- likely_different: 7
- uncertain: 20
- rejected_exact_frame_conflict: 4
- approved edges (`likely_same` + `link_approved=true`): 4

Accepted components:

- `[4, 682]` → `global_candidate_id` 4
- `[231, 635]` → `global_candidate_id` 231
- `[593, 689]` → `global_candidate_id` 593
- `[588, 806]` → `global_candidate_id` 588

Final linking counts:

- linked components: 4
- linked raw tracks: 8
- singleton raw tracks: 268
- embedded `singleton_unlinked`: 127
- `singleton_no_embedding`: 141
- global candidates: 272
- held incomplete components: 0

Arithmetic checks:

- `276 raw = 8 linked + 268 singleton`
- `268 singleton = 127 embedded unlinked + 141 no embedding`
- `272 global candidates = 276 raw - 4` (one ID collapsed per accepted
  pair)

## G. Linking safety

Policy: `configs/reid/linking_policy_stage4b.yaml`

- Manual acceptance required
- Automatic linking disabled
- Similarity threshold null
- Exact-frame hard reject
- Span overlap alone is not a hard reject
- Whole-component cross-member exact-frame checks
- Complete-clique requirement for components of size ≥3
- Uncontrolled transitive chaining disabled
- Deterministic global ID = minimum raw track ID in the component
- Every raw track has exactly one map row
- No-embedding tracks remain singletons

## H. Tests and quality

- unittest count: 165
- `football-cv` pip check: clean
- `sn-reid-cpu` pip check: clean
- Atomic output and overwrite behavior tested
- Strict JSONL/NPZ validation
- NaN/Infinity rejected
- Model/checkpoint unit tests use mocks
- Real `sample.mp4` smoke and full runs were completed separately from
  unit tests (outputs are Git-ignored)

Product HEAD for Stage 4B linking engine: `a4b379c`
(*Implement manual ReID linking pipeline*).

Related Stage 4B commits:

- `5d23c7d` — Define Stage 4B ReID crop and linking schemas
- `10eb356` — Implement ReID crop selection pipeline
- `e14b32f` — Implement ReID embedding extraction pipeline
- `7098bf1` — Implement ReID track aggregation and candidate generation
- `d363a6d` — Define conservative ReID linking policy
- `a4b379c` — Implement manual ReID linking pipeline

## I. Output paths

Git-ignored full Stage 4B results:

- `outputs/reid/full_stage4b/crops`
- `outputs/reid/full_stage4b/embeddings`
- `outputs/reid/full_stage4b/aggregation`
- `outputs/reid/full_stage4b/candidates`
- `outputs/reid/full_stage4b/diagnostics`
- `outputs/reid/full_stage4b/review_montages_strong_top30`
- `outputs/reid/full_stage4b/manual_decisions_stage4b.jsonl`
- `outputs/reid/full_stage4b/linking`

Final map:

- `outputs/reid/full_stage4b/linking/global_id_map.jsonl`

Also present under linking:

- `accepted_edges.jsonl`
- `linking_audit.jsonl`
- `linking_summary.json`

Limited 12-track benchmark outputs remain under
`outputs/reid/limited_benchmark_stage4b_12/` (also Git-ignored).

## J. Limits

- No ground-truth player identity annotation
- Accuracy percentage cannot be computed
- MOTA / HOTA / IDF1 / ReID mAP are not reported
- Market1501 is not football-domain-specific
- Same-team kits can produce false positives
- Small or contaminated crops can degrade embeddings
- Real full run completed only on `sample.mp4`
- Generalization to other matches is not yet validated
- Final global IDs are *candidate* identities, not proven `player_id`
  values

## K. Stage result

Stage 4B status: **completed_baseline**

Completed capabilities:

- crop extraction
- crop embeddings
- track aggregation
- candidate generation
- manual visual review
- conservative audited linking
- complete raw-track → global-candidate mapping

Possible later improvements (not required to close Stage 4B):

- jersey number recognition
- team/kit classification
- crop contamination detection
- temporal/spatial motion consistency
- football-domain ReID fine-tuning
- labeled golden clips and official evaluation metrics

Supporting docs:

- `docs/setup/reid-crop-selection-analysis.md`
- `docs/setup/reid-stage4b-schema-decisions.md`
- `docs/setup/reid-stage4b-linking-policy.md`
- `docs/setup/sn-reid-player-crop-embedding-smoke.md`
