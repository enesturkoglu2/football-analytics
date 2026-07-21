# Stage 5B3D — Full-Observation Raw-Track Purity Visual Validation

- **Date:** 2026-07-21
- **Gate:** Stage 5B3E — documentation and manual segment-plan freeze
- **Purity audit commit:** `f73aef7` — Implement raw track purity audit
- **Localization review package (Git-ignored):**
  `outputs/reid/full_stage4b/purity_localization_review_stage5b3d/`
  and `outputs/reid/full_stage4b/reid_purity_localization_review_stage5b3d.zip`
- **Review index:**
  `outputs/reid/full_stage4b/purity_localization_review_stage5b3d/review_index.json`
- **Segmentation policy:**
  `configs/reid/manual_track_segmentation_policy_stage5b3.yaml`
- **Manual decisions (plan only, not applied):**
  `configs/reid/manual_track_segment_decisions_stage5b3.yaml`
- **Stage 5B3 status:**
  `visually_validated_manual_segment_plan_not_applied`

## A. Scope

Full-observation diagnostic review of visually suspicious raw tracks
from Stage 5B3B/5B3C, using every available raw tracking observation
(not only selected ReID crops).

| Item | Value |
|---|---|
| Focus raw tracks | 15 |
| Raw observations reviewed | 1329 |
| Candidate inspection windows | 22 |
| Diagnostic MP4 clips | 15 |
| Observation contact-sheet pages | 42 |
| Candidate-window sheet pages | 114 |
| Every available raw observation reviewed | true |
| Missing frames interpolated | false |
| Identity-switch ground truth created | false |
| Exact automatic split created | false |
| Automatic split / delete / global-ID rewrite | false |
| Team assignment / link / reject | false |
| Change / split threshold | null |

Important:

- `sample.mp4` is a **development / reference** clip only.
- Manual visual observations are **not** identity-switch ground truth.
- Selected-crop transitions are **inspection windows**, not exact split
  frames.
- Frame ranges below mean **existing raw observations only** inside the
  inclusive bounds. Missing frames create no observations and are not
  interpolated.
- No track was split, deleted, or rewritten in this gate.

## B. Headline outcome (manual visual plan)

| Outcome | Count |
|---|---:|
| Raw tracks — probable multi-player / manual split candidate | 13 |
| Raw tracks — no-split contamination controls | 2 |
| Probable switch events (manual visual) | 15 |
| Gap-bounded events | 11 |
| Overlap-ambiguous events | 2 |
| Adjacent-observation events | 2 |

These counts are **manual diagnostic summaries**. They are **not**
accuracy claims and **not** identity ground truth.

## C. Boundary-type definitions

| Type | Meaning |
|---|---|
| `gap_bounded` | Probable change across an unobserved frame gap; exact real-world switch frame unresolved |
| `overlap_ambiguous` | Existing observations near the change cannot be cleanly assigned to either segment yet |
| `adjacent_observations` | Consecutive existing observations on either side of the proposed boundary (still not GT) |

Exact real-world switch frame remains **unresolved** for all events.

## D. Manual visual findings by track

### Track 3 — `manual_split_candidate` (1 event, `gap_bounded`)

- Segment 1 existing observations: **0–53**
- Unobserved gap: **54–64**
- Segment 2 existing observations: **65–68**
- Manual observation: yellow number-5-like appearance → white
  number-8-like appearance
- Exact real-world switch frame: unresolved

### Track 7 — `manual_split_candidate` (2 events)

- Segment 1 existing observations: **0–154**
- Ambiguous / unassigned existing observations: **155, 156, 157, 158**
- Segment 2 existing observations: **159–164**
- Unobserved gap: **165–169**
- Segment 3 existing observations: **170–255**
- Manual observation: white appearance → short yellow hijack → white
  appearance
- Boundary 1 (`154` / `159` region): `overlap_ambiguous`
- Boundary 2 (gap **165–169**): `gap_bounded`
- Segment 1 and segment 3 **may** be the same physical player, but must
  **not** be automatically merged

### Track 31 — `manual_split_candidate` (1 event, `gap_bounded`)

- Segment 1: **24–251**
- Unobserved gap: **252–255**
- Segment 2: **256–273**
- Manual observation: white → yellow

### Track 38 — `manual_split_candidate` (1 event, `gap_bounded`)

- Segment 1: **32–62**
- Unobserved gap: **63–66**
- Segment 2: **67–169**
- Manual observation: yellow number-5-like → white

### Track 138 — `manual_split_candidate` (1 event, `gap_bounded`)

- Segment 1: **160–232**
- Unobserved gap: **233–248**
- Segment 2: **249–273**
- Manual observation: yellow → white

### Track 177 — `manual_split_candidate` (1 event, `gap_bounded`)

- Segment 1: frame **212** only
- Unobserved gap: **213–221**
- Segment 2: frame **222** only
- Manual observation: white → yellow
- `too_sparse_for_track_embedding: true`

### Track 213 — `manual_split_candidate` (1 event, `gap_bounded`)

- Segment 1: **240–251**
- Unobserved gap: **252–262**
- Segment 2: **263–298**
- Manual observation: white → yellow

### Track 231 — `manual_split_candidate` (1 event, `overlap_ambiguous`)

- Segment 1: **253–280**
- Ambiguous / unassigned existing observations: **281, 282, 283, 284**
- Segment 2: **285–391**
- Manual observation: white number-9-like → yellow number-17-like
- Existing Stage 4B component **`[231, 635]` unchanged**
- Only future segment `raw_231_s02` may be re-reviewed against `635`
- Segment `raw_231_s01` must **not** automatically inherit the component
- No automatic component reversal

### Track 354 — `manual_split_candidate` (1 event, `gap_bounded`)

- Segment 1: **309–342**
- Unobserved gap: **343–371**
- Segment 2: **372–374**
- Manual observation: yellow → white number-8-like

### Track 375 — `manual_split_candidate` (1 event, `gap_bounded`)

- Segment 1: **326–361**
- Unobserved gap: **362–388**
- Segment 2: **389–391**
- Manual observation: white → yellow

### Track 414 — `manual_split_candidate` (1 event, `gap_bounded`)

- Segment 1: **360–379**
- Unobserved gap: **380–386**
- Segment 2: **387–391**
- Manual observation: white number-10-like → yellow number-26-like

### Track 418 — `manual_split_candidate` (1 event, `gap_bounded`)

- Segment 1: **363–364**
- Unobserved gap: **365–386**
- Segment 2: **387–388**
- Manual observation: white → yellow

### Track 514 — `manual_split_candidate` (2 events, `adjacent_observations`)

- Segment 1: **440–454**
- Segment 2: **455–503**
- Segment 3: **504–587**
- Manual observation: white number-8-like → yellow number-30-like →
  white number-3-like
- Boundaries: **454/455** and **503/504** (`adjacent_observations`)
- Visual observations are still **not** identity ground truth

### Track 268 — `no_split_contamination_control`

- Existing observations: **268–358**
- Target bbox remained on the white player
- Nearby yellow player / background explains descriptor difference
- Manual split candidate: **false**
- Raw track preserved as-is in the plan

### Track 590 — `no_split_contamination_control`

- Existing observations: **568–743**
- Target bbox remained on the white player
- Yellow player entered crop / nearby region late in the sequence
- Manual split candidate: **false**
- Raw track preserved as-is in the plan

## E. Policy freeze (this gate)

Documented in:

- `configs/reid/manual_track_segmentation_policy_stage5b3.yaml`
- `configs/reid/manual_track_segment_decisions_stage5b3.yaml`

Frozen rules:

- Raw `tracks.jsonl` remains **immutable**
- No in-place mutation, interpolation, or observation deletion
- Segmentation implementation is **not** available yet
- Non-destructive segment view is required before any real run
- Automatic split / boundary selection / segment merge / segment link =
  **false**
- Global-ID rewrite = **false**
- `[231, 635]` unchanged; `raw_231_s01` does not inherit the component
- Manual visual review is **not** ground truth; accuracy claims
  prohibited
- Mixed raw-track embeddings must **not** be used as final evaluation
  for impure tracks once segments exist

## F. Next gates

| Gate | Scope |
|---|---|
| **Stage 5B3E** | Manual segment plan + policy freeze — **this document** |
| **Stage 5B3F** | Non-destructive manual segment-view implementation (no real mutation) |
| **Stage 5B3G** | Segmented crop / embedding / ReID regression run on `sample.mp4` (dev/reference): old raw-track baseline vs segmented-track result; no raw-track in-place mutation; no automatic segment merging |
| **Stage 5C** | Jersey-number visibility audit |

## G. Explicit non-claims

- No identity-switch ground truth
- No track-purity ground truth
- No exact split decisions applied to data
- No team A/B labels
- No threshold or composite score
- No change to Stage 4B global-ID map or accepted components
- `sample.mp4` remains development / reference only
