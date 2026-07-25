# Stage 5D-A — Target gallery enrollment design and preflight

- **Gate:** STAGE5D-A
- **Status:** design / asset preflight ready
- **Exact next gate:**
  `STAGE5D-B_TARGET_DEFINITION_AND_ANCHOR_REVIEW_PACKAGE`

## 1. Stage 5C closure context

Stage 5C is **closed** with holdout decision
`INCONCLUSIVE_SAFE_BUT_LOW_SUPPORT`.

- automated PARSeq jersey evidence = **diagnostic_only**
- may not assign identity, veto appearance, or enroll gallery
- discovery candidate cut retained for provenance only
- appearance ReID remains the **primary** identity channel
- unknown identity is preserved

## 2. Upstream segmented ReID provenance

Canonical rebuild-r2 chain (verified from manifests):

| Stage | Root |
|---|---|
| Source video | `data/test_clips/sample.mp4` |
| Detection + tracking + Stage 4B | `outputs/reid/full_stage4b_rebuild_r2` |
| Documented-link replay | `…_documented_link_overlay` |
| Stage 5A/5B/5B3 + segmented OSNet | `…_stage5_replay` |
| Visibility / clean universe | `…_stage5c_visibility_universe` |
| Canonical clean split | `…_clean_split_capacity_balanced` |
| Stage 5C closure | `…_stage5c_closure` |

Verified structural counts:

- frames **1023**; observations **13309**; raw tracks **276**
- initial crops **454**; embeddings **454×512**; aggregated **135**;
  pairs **9045**
- replayed links **4**; global IDs **272**
- segments **291** (manual **28** / pass-through **261**)
- assigned obs **13301**; unassigned **8**
- embedded **150**; no-embedding **141**
- recomputed **28**; reused **122**; new crops **75**;
  segmented pairs **9507**
- visibility universe **474**

## 3. Appearance gallery principles

- enrollment_mode = **manual_frozen**
- automatic gallery growth / self-training / pseudo-label = **false**
- OCR / tracker-ID / similarity automatic enrollment = **false**
- gallery uses **full-body** (or widest person) crops + existing OSNet
  512-D embeddings — **not** jersey-number ROI
- only human `target_positive` may enroll
- `uncertain` / `invalid_crop` / `non_player` /
  `multi_person_ambiguous` never enroll
- unknown identity preserved; no forced assignment

## 4. Gallery vs evaluation exclusion

Overlap must be **0** on segment/crop/SHA/frame/duplicate/near-dup/
documented-link/temporal keys.

Stage 5C discovery/holdout primary+reserve batches are **excluded** as
gallery/evaluation inputs (readable only for exclusion provenance).

## 5. Existing embedding preflight

- artifact: `segment_embeddings.npz` (150×512 float32)
- NaN/Inf/zero-vector = 0
- duplicate segment IDs = 0
- temporary L2 view allowed for diagnostics only
- no gallery/prototype publish in Stage 5D-A

## 6. Target definition

Blank template:
`templates/target_definition_template.json`

- target_id / alias / basis empty
- `target_definition_frozen=false`
- no automatic jersey number as identity

## 7. Preregistered Stage 5D workflow

| Gate | Purpose |
|---|---|
| 5D-A | design + preflight (this gate) |
| 5D-B | target definition + anchor review package |
| 5D-C | bounded cosine retrieval + human decisions |
| 5D-D | enrollment annotation freeze |
| 5D-E | prototype generation (no identity assignment) |
| 5D-F | independent retrieval validation / Stage 5E readiness |

Human approval is required before any gallery growth at every gate.

## 8. What Stage 5D-A does **not** do

- target player selection
- anchor segment selection
- target-positive decisions
- gallery membership
- identity assignment
- ReID threshold selection
- contact sheets / candidate ranking / prototypes
- model inference (YOLO / OSNet / PARSeq)

## 9. Limitations

- OSNet Market1501 is general person ReID, not football-domain trained
- 141 segments deliberately lack embeddings
- Stage 5C jersey channel is not an enrollment source
- design diversity limits are planned, not yet enforced

## 10. Exact next gate

`STAGE5D-B_TARGET_DEFINITION_AND_ANCHOR_REVIEW_PACKAGE`

Target alias/basis will be human-approved; full-body anchor contact
sheets will be produced without similarity scores, model identity
predictions, or jersey OCR predictions. Anchor choices are **not**
final gallery membership.
