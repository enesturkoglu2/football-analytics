# Stage 5D-B1C — Target 001 Alternate Eligible Anchor Source Review

## Status (this rebuild run)

`COMPLETED_STAGE5D_B1C_NO_ELIGIBLE_SOURCE_IN_EARLY_WINDOW`

Exact next gate:
`STAGE5D-B1C2_TARGET_001_ADDITIONAL_HUMAN_WINDOW_REVIEW`

If eligible candidates had been found, the success status would have been
`COMPLETED_STAGE5D_B1C_ALTERNATE_ELIGIBLE_SOURCE_REVIEW_READY` with next gate
`STAGE5D-B1D_TARGET_001_ALTERNATE_SOURCE_SELECTION_FREEZE_AND_ANCHOR_DERIVATION`.

## Purpose

Preserve frozen target `target_001` and frozen-but-enrollment-ineligible
identity seed `SEED_CANDIDATE_07` (`raw_222_full` / track 222). From an early
human number-visible window (frames 30–75), surface only existing
observation chains that pass Stage 5C / Stage 5D-A gallery exclusion rules
as neutral `ALT_SEED_CANDIDATE_*` codes for later human selection.

This gate does **not**:

- freeze an alternate seed
- derive or approve anchors
- create gallery members / prototypes / identity assignments
- run YOLO, ByteTrack, OSNet, OCR/PARSeq, or similarity
- use `raw_222_full` / track 222 crops or embeddings for scoring/enrollment

## Frozen original seed (immutable)

| Field | Value |
|---|---|
| selected_neutral_seed_code | `SEED_CANDIDATE_07` |
| segment_id | `raw_222_full` |
| raw_track_id | 222 |
| eligibility | `frozen_identity_seed_only_stage5c_excluded` |
| exclusion | Stage 5C `holdout_primary` |
| usable as anchor/gallery/similarity | **no** |

## Early human review window

| Item | Value |
|---|---|
| Source video | `data/test_clips/sample.mp4` |
| Window frames | 30–75 inclusive (~1.00–2.50 s) |
| Preferred human reference | frames 36–60 |
| Detection/tracking | existing observations only |

## Eligibility filters

A chain may receive an alternate neutral code only if **all** hold:

- no Stage 5C membership
- no overlap with frozen seed exclusion component
- no Stage 5D-A exclusion-key overlap (segment, track, crop, SHA, frame,
  exact/near-duplicate, documented-link, temporal window, discovery/holdout
  primary/reserve)
- complete existing segment + crop + OSNet embedding lineage
- embedding shape 512, finite, non-zero
- not frozen as multi-person/ambiguous

Ineligible players may appear naturally in full frames but must not receive
selection boxes or alternate codes.

## Artifacts

Root:

`outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_alternate_eligible_source_review`

| Path | Role |
|---|---|
| `inventory/target_001_alternate_eligible_source_mapping.jsonl` | Neutral alt codes (empty when none eligible) |
| `eligibility/target_001_early_window_source_eligibility_audit.json` | Per-chain exclusion audit |
| `alternate_source_review/target_001_alternate_source_review_contract.json` | Policy contract |
| `alternate_source_review/target_001_alternate_source_review_manifest.json` | Package manifest |
| `review_packages/target_001_alternate_eligible_source_review/` | Sheet + clip when eligible |
| `templates/target_001_alternate_eligible_source_review_template.csv` | Blank manual template |
| `stage5d_b1c_summary.json` / `stage5d_b1c_manifest.json` | Gate summary |

When eligible count is 0: PNG=0, MP4=0 (no review media). When eligible > 0:
exactly one PNG sheet and one MP4 window clip with only eligible boxes labeled
`ALT_SEED_CANDIDATE_*`.

## Manual template

Blank fields only. Allowed tri-state: `yes` / `no` / `uncertain`.
No model suggestions. Alternate selection freeze is a separate later gate.

## Run

```bash
conda run -n football-cv python scripts/run_reid_target_alternate_eligible_source_review.py \
  --config configs/reid/target_alternate_eligible_source_review_stage5d_target_001.yaml
```

## Tests

`tests/test_reid_target_alternate_eligible_source_review.py`
