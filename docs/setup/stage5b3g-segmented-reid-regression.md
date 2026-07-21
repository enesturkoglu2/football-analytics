# Stage 5B3G — Segmented ReID Regression Result

- **Date:** 2026-07-21
- **Stage status:** `completed_segmented_reid_regression_baseline`
- **Project HEAD:** `fd48e64` (`Handle exact-conflict ReID regression pairs`)
- **Result directory:**
  `outputs/reid/full_stage4b/segmented_reid_regression_stage5b3`
- **Segment view:**
  `outputs/reid/full_stage4b/segment_view_stage5b3`
- **Raw-track baseline:** `outputs/reid/full_stage4b`

## A. Purpose

Stage 5B3G measures the effect of the non-destructive manual segment
view on the existing Stage 4B ReID representation:

- retire the mixed raw-track embeddings of the 13 manual-split parents
  from the segmented entity universe;
- select new crops and recompute embeddings for their 28 manual
  segments;
- reuse unchanged baseline embeddings for embedded no-split controls
  and pass-through tracks;
- avoid expanding embedding coverage for unaffected tracks that had no
  Stage 4B embedding; and
- compare segmented candidate behavior with the immutable raw-track
  baseline while isolating the segmentation change.

This is a real OSNet regression run, but it is **not** an identity
accuracy test. The manual segment decisions are visual-review
decisions, not identity ground truth.

## B. Runtime and provenance

| Item | Verified value |
|---|---|
| Project HEAD | `fd48e64` |
| sn-reid commit | `621e2b0f2d2a7a3e207b8dd747542b6608bf72db` |
| Checkpoint | `/home/enesturkoglu2/projects/soccernet/checkpoints/general-reid/osnet_x1_0_market1501_softmax_256x128.pth.tar` |
| Checkpoint SHA-256 | `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154` |
| Model | `osnet_x1_0` |
| Training domain | Market1501 general-person ReID; not football-domain validated |
| Embedding dimension | 512 |
| Runtime | `sn-reid-cpu`; CPU inference; CUDA unavailable/false |
| Product `elapsed_sec` | `9.95778945100028` (approximately 9.96 s) |
| Observed wall time | approximately 14.19 s (`/usr/bin/time`, outside the product summary) |
| Video | `data/test_clips/sample.mp4` — development/reference input only |

The required local import boundary was:

```bash
conda run -n sn-reid-cpu \
  env \
  PYTHONPATH=/home/enesturkoglu2/projects/soccernet/sn-reid \
  python scripts/run_segmented_reid_regression.py ...
```

No package, model, checkpoint, or dataset was downloaded.

## C. Segment representation

| Measure | Count |
|---|---:|
| Baseline embedded raw tracks | 135 |
| Manual-split parents with a baseline embedding | 13 / 13 |
| Retired mixed raw-track embeddings | 13 |
| Reused no-split control embeddings | 2 |
| Reused pass-through embeddings | 120 |
| Total reused baseline embeddings | 122 |
| Manual recompute targets | 28 |
| Recomputed manual-segment embeddings | 28 |
| Total embedded segment entities | 150 |
| Derived segments without an embedding | 141 |
| Manual segments without an embedding | 0 |

All 13 parent mixed embeddings remain intact in the Stage 4B baseline,
but none is used as a segmented entity. Manual-parent mixed embedding
reuse is zero.

All 141 no-embedding entities have the explicit reason
`parent_raw_track_absent_from_baseline_embeddings`. This is a deliberate
coverage policy, **not** an inference failure: unaffected tracks without
a baseline embedding were not newly embedded. Baseline coverage
expansion is false.

## D. Crop result

| Measure | Result |
|---|---:|
| New segment crops | 75 |
| Manual segments with crops | 28 / 28 |
| Crops per segment — min / p25 / median / p75 / max | 1 / 1 / 2 / 5 / 5 |
| Crops per segment — mean | 2.6785714286 |
| Selected frame range | 2–584 |
| Selection rank 1 / 2 / 3 / 4 / 5 | 28 / 15 / 12 / 11 / 9 |
| Ambiguous observation crops | 0 |
| Unassigned observation crops | 0 |
| Parent raw-crop fallback | 0 |
| Interpolation | 0 |
| Invalid or empty crops | 0 |
| Bbox clamps in selected rows | 0 |
| Source-observation hash mismatches | 0 |
| Configured-range violations | 0 |

Both single-observation track 177 segments produced one crop and one
embedding:

- `raw_177_s01`: frame 212;
- `raw_177_s02`: frame 222.

Selected crops are a bounded sample of eligible assigned observations;
they are not every observation in each segment.

## E. Embedding result

- Shape: 150 segment vectors × 512 dimensions.
- All values are finite.
- All vectors are unit-normalized within the existing tolerance.
- Recomputed-vector L2 norm range:
  `0.9999999403953552` to `1.0`.
- Reused-vector mismatch count against baseline: 0.
- Duplicate embedding SHA count: 0.
- Manual-parent mixed embedding reuse count: 0.
- Aggregation remains `l2_mean`.
- Quality weighting and kit weighting were not used.

## F. Candidate result

| Measure | Count/value |
|---|---:|
| Embedded segment entities | 150 |
| Possible unordered pairs | 11,175 |
| Exact-overlap hard rejects | 1,668 |
| Ranked candidates | 9,507 |
| Same-parent ranked pairs | 17 |
| Span-overlap-only ranked pairs | 42 |
| Similarity threshold | `null` |
| Automatic link / reject / component | 0 / 0 / 0 |

Ranked cosine distribution:

| Statistic | Cosine |
|---|---:|
| min | 0.3713024259 |
| p5 | 0.7442991138 |
| p25 | 0.8328654766 |
| median | 0.8714325428 |
| p75 | 0.9126812518 |
| p95 | 0.9520472229 |
| max | 0.9914593101 |
| mean | 0.8580985591 |

Exact same-frame overlap remains a hard reject. Hard-rejected rows have
`cosine_similarity=null` and `rank=null`; they were not added to the
ranked candidate count. Span overlap alone remains rank-eligible.

## G. Exact-conflict regression result

| Measure | Count/value |
|---|---:|
| Unaffected baseline pairs | 7,381 |
| Rank-eligible unaffected pairs | 6,177 |
| Unaffected exact-conflict audits | 1,204 |
| Similarity matches | 7,381 |
| Similarity mismatches | 0 |
| Missing non-conflict segmented candidates | 0 |
| Normal-pair maximum absolute cosine difference | 0.0 |
| Exact-conflict audit maximum absolute cosine difference | 0.0 |

For normal unaffected reused/reused pairs, the ranked segmented cosine
exactly matches the baseline cosine. For exact-conflict pairs, the
reused vectors are compared directly and the resulting audit cosine
matches the Stage 4B baseline cosine. The audit cosine is **not** a
ranked segmented similarity and does not remove the hard reject.

Example baseline pair `(1, 2)`:

- common frames: 114;
- baseline and reused-vector audit cosine:
  `0.8663586378097534`;
- segmented rank: `null`;
- hard rejected: true;
- automatic link, automatic reject, and component decisions: `null`.

## H. Affected parent result

All 13 rows have:

- baseline embedding available: true;
- baseline representation:
  `retired_mixed_baseline_embedding`;
- replacement status: `recompute_manual_segment` for every segment;
- `replacement_complete=true`; and
- no automatic merge or identity claim.

Ranks below are global segmented-candidate ranks.

| Raw track | Replacement segments | Complete | Sibling pair cosine / rank |
|---:|---:|---|---|
| 3 | 2 | true | s01↔s02: 0.9180330634 / 2105 |
| 7 | 3 | true | s01↔s03: 0.9747990370 / 55; s01↔s02: 0.9201802611 / 1997; s02↔s03: 0.8974796534 / 3238 |
| 31 | 2 | true | s01↔s02: 0.8458809853 / 6441 |
| 38 | 2 | true | s01↔s02: 0.9433941245 / 827 |
| 138 | 2 | true | s01↔s02: 0.8356609941 / 6976 |
| 177 | 2 | true | s01↔s02: 0.8558201194 / 5804 |
| 213 | 2 | true | s01↔s02: 0.9077724218 / 2660 |
| 231 | 2 | true | s01↔s02: 0.8677652478 / 5004 |
| 354 | 2 | true | s01↔s02: 0.8618113995 / 5392 |
| 375 | 2 | true | s01↔s02: 0.8454438448 / 6468 |
| 414 | 2 | true | s01↔s02: 0.7874542475 / 8582 |
| 418 | 2 | true | s01↔s02: 0.8017207384 / 8292 |
| 514 | 3 | true | s02↔s03: 0.9480118752 / 621; s01↔s03: 0.9089988470 / 2588; s01↔s02: 0.8674327135 / 5031 |

These sibling similarities are ranking/audit observations only. They do
not imply that sibling segments should be merged.

## I. Critical interpretations

### Track 7

- The s01/s03 top-10 neighbor overlap is 6/10.
- s02 has zero top-10 overlap with either s01 or s03.
- s01↔s03 has high OSNet similarity (`0.9747990370`), but no automatic
  merge was performed.
- This supports the visual impurity observation as an audit signal; it
  is not identity ground truth.

### Tracks 31 and 414

The two segment top-neighbor lists separate strongly (zero direct
top-10 overlap for each parent). This demonstrates that the old mixed
embedding had combined distinct appearance neighborhoods. It is not an
accuracy result.

### Track 514

The segment neighborhoods differ, but sibling similarities remain high
in places, especially s02↔s03 (`0.9480118752`). This is consistent with
a known limitation of a Market1501 general-person model on low-resolution
football crops and similar team kits. It supports measuring
jersey-number visibility in Stage 5C; it does not justify an automatic
identity decision.

## J. `[231, 635]` audit

The deterministic baseline rank below is recomputed over baseline
rank-eligible pairs using the frozen cosine ordering.

| Pair | Cosine | Rank | Exact conflict | Span overlap | Delta from baseline raw pair |
|---|---:|---:|---|---|---:|
| raw 231 ↔ raw 635 | 0.9795190096 | 16 | false | false | — |
| `raw_231_s01` ↔ `raw_635_full` | 0.8962208033 | 3296 | false | false | -0.0832982063 |
| `raw_231_s02` ↔ `raw_635_full` | 0.9753487110 | 48 | false | false | -0.0041702986 |
| `raw_231_s01` ↔ `raw_231_s02` | 0.8677652478 | 5004 | false | false | — |

- `raw_231_s01` component inheritance: false.
- `raw_231_s02` automatic link to 635: false.
- Component assignment performed: false.
- Existing `[231, 635]` component unchanged: true.
- Future manual review required: true.

None of these similarities is proof that the entities are the same
player.

## K. Pair-delta result

| Delta class | Count |
|---|---:|
| Normal unaffected reused pairs | 6,177 |
| Unaffected exact-conflict audits | 1,204 |
| Affected baseline pairs | 1,664 |
| Affected replacement segment pairs | 3,313 |
| Same-parent segment pairs | 17 |
| Comparisons missing because of representation absence | 0 |

For 96 affected raw pairs, no ranked replacement similarity is
available because every relevant replacement pair is exact-overlap
hard-rejected. This is not missing representation.

Affected baseline similarities are retained as baseline audit values.
They are never overwritten with a single invented replacement
similarity; each eligible replacement segment pair is preserved
separately.

## L. Safety and interpretation limits

- Identity ground truth is unavailable.
- No accuracy claim is made.
- Ranking changes are not accuracy improvements.
- Manual segments are not proven physical identities.
- Reused pass-through embeddings preserve baseline behavior but do not
  prove track purity.
- A no-embedding entity is not negative identity evidence.
- Selected crops do not cover every observation.
- Quality and kit weighting were disabled.
- No similarity threshold was selected.
- No automatic identity, link, reject, merge, component, global-ID, or
  team assignment was produced.
- Raw tracks, Stage 4B baseline artifacts, and the segment view remained
  immutable.
- The accepted `[231, 635]` component remained unchanged.
- `sample.mp4` is a development/reference input, not a product constant.
- The output remains Git-ignored.

## Artifact hashes and sizes

Hashes and byte sizes were computed directly from the final artifacts.

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| `segment_crop_manifest.jsonl` | `5c8ed13d5fd0b404fc894a4b5f3f7c4bd0c1ea0ac25c6a405ffa5b597d5b675c` | 64,573 |
| `segment_embedding_index.jsonl` | `74ceda4f92baa6f9b2617248e0c887ad8294bc3c62b264fab565b5d15b667920` | 349,435 |
| `segment_embeddings.npz` | `355676d7e017c3b3b5397bf82e9a088686740fbc495df35598fc90b897d28d97` | 314,922 |
| `segment_candidates.jsonl` | `f445c1568cec850edbb808d488dd410eb86d00de6eb8dc7411789a012ce2001c` | 6,921,211 |
| `baseline_to_segment_replacement.jsonl` | `93e2be9f76e091f69d42aa622c7532958ec008cfc421623a6c8331950f84701f` | 74,200 |
| `segmented_reid_pair_deltas.jsonl` | `0b17d17a03db82a4014ee3521e846b8b85608d6abb1340ee6d031a28eef12b96` | 3,865,468 |
| `segmented_reid_regression_summary.json` | `c6eab60f5a24f078fd174f74324647c050b4e41b91db6580700c3857761d056b` | 5,702 |

- Crop JPEG count: 75.
- Crop JPEG total bytes: 261,484.
- Total output file bytes: 11,856,995.
- Filesystem `du -sh`: 12M.

## M. Stage decision

Stage 5B3 final status:

`completed_segmented_reid_regression_baseline`

Decision:

- do not open another purity-panel loop;
- preserve the non-destructive segment view;
- use the segmented representation as the input entity view for later
  identity-signal stages;
- proceed next to **Stage 5C-A jersey-number visibility/readability
  measurement baseline**;
- do not run OCR, a recognizer, or a jersey checkpoint at the start of
  Stage 5C.

Stage 5C-A produces visibility/audit evidence only. Recognition
capability investigation, isolated recognition, aggregation, target
enrollment, evidence fusion, and spatial continuity remain separately
gated.

The retained roadmap is:

- Stage 5C-B: SoccerNet jersey-capability audit;
- Stage 5C-C: isolated recognizer smoke test;
- Stage 5C-D: tracklet-level multi-frame aggregation;
- Stage 5D: target-player enrollment and gallery memory;
- Stage 5E: evidence fusion and golden evaluation; and
- Stage 6: GameState/calibration/pitch coordinates and spatial
  continuity.
