# Stage 5B — Torso Kit Descriptor Visual Validation Report

- **Date:** 2026-07-21
- **Gate:** Stage 5B2C — documentation and usage-policy freeze
- **Measurement commit:** `777cc43` — Implement torso kit descriptor analysis
- **Measurement config:** `configs/reid/kit_descriptor_stage5b.yaml`
- **Policy config:** `configs/reid/kit_visual_validation_policy_stage5b.yaml`
- **Stage 5B status:**
  `visually_validated_measurement_baseline_with_track_impurity_findings`

## A. Scope

Full `sample.mp4` Stage 4B crops were measured and manually reviewed
with diagnostic kit panels only:

| Item | Value |
|---|---|
| Crops measured | 454 |
| Crop-producing tracks | 135 |
| Track review panels | 24 |
| Crop review panels | 30 |
| Contact sheets | 10 |
| Crop dominant family (green/yellow/white/gray) | 211 / 106 / 73 / 64 |
| Track dominant family (green/yellow/white/gray) | 65 / 35 / 19 / 16 |
| Same crop-dominant family on all crops | 78 tracks |
| Multiple dominant families | 57 tracks |
| Team assignment | false |
| Clustering | false |
| Kit similarity threshold | null |

Review package (Git-ignored):

- `outputs/reid/full_stage4b/kit_stage5b/`
- `outputs/reid/full_stage4b/kit_review_stage5b/`
- `outputs/reid/full_stage4b/reid_kit_review_stage5b.zip`

Important:

- `sample.mp4` is a **development / reference** clip only.
- Review outcomes are **manual diagnostic observations**.
- They are **not** ground-truth team annotations.
- No team assignment, forced two-team clustering, or kit similarity
  threshold was applied.

## B. Torso ROI findings

The normalized center torso ROI directionally matched jersey/torso
regions on many upright, reasonably full-body crops.

Visually successful yellow-kit examples:

- tracks `6`, `2`, `41`, `445`

Visually successful white/gray-kit examples:

- tracks `111`, `661`, `10`, `59`

Limits:

- ROI is **not** segmentation.
- Player pose, partial crops, off-pitch subjects, occlusion, and bbox
  geometry can break the approximation.

Failed / limited examples (manual visual observation):

- track `13`: off-pitch-like person; ROI measures head/neck/background
  more than kit
- `177@212`: substantial grass/background influence inside torso ROI
- bent / sideways players: fixed ROI does not fully match anatomical
  torso

Frozen conclusion:

- Current ROI remains the **measurement baseline**.
- It is **not** a validated segmentation for team assignment.
- No crop-level binary `ROI-valid` auto-label is applied yet.

## C. Absolute color-family findings

Visually white-kit-like crops that measured dominant `green`:

- tracks `480`, `626`, `680`, `808`

Likely contributing effects:

- broadcast color cast
- grass reflection
- white balance
- low resolution
- background leakage

Frozen conclusion:

- `dominant_color_family` is **not** a team label.
- `green` must **not** be forced to mean pitch/background.
- `white` / `gray` / `yellow` are **not** automatic team labels.
- Sample-specific family→team mapping is **prohibited**.
- Absolute color-family assignment remains **audit only**.

Future candidates (not implemented in this gate):

- camera-segment-relative color normalization
- color constancy
- continuous Lab / HSV / histogram comparison

## D. Track consistency findings

Within-track kit change or multi-player influence was observed
(manual visual observation, **not** ground-truth identity switch):

| Track | Observation |
|---|---|
| `31` | yellow crop followed by white crops |
| `414` | white number 10 followed by yellow number 26 |
| `7` | mostly white with one yellow crop |
| `213` | yellow crops followed by a white crop |
| `38` | yellow/non-target influence followed by white crops |
| `514` | mixed white and yellow numbered-player appearances |
| `231` | white number 9 followed by yellow number 17 |
| `138` | white crop followed by yellow crops |
| `590` | white crops with later yellow/non-target influence |

These may reflect:

- probable raw-track identity switch
- severe multi-person contamination
- crop-selection contamination
- or a combination

They are **not** labeled as confirmed ground-truth identity switches.

Frozen conclusion:

- `raw_track_id` is **not** an atomic player-identity guarantee.
- Within-track kit change may be **positive impurity-risk evidence**.
- Raw-track purification / splitting is a separate problem and must be
  handled in a later gate.

## E. Agreement false reassurance

Track `3` example:

- visually mixed white/yellow player content
- all crop dominant families measured as `green`
- dominant-family agreement = `1.0`

Frozen conclusion:

- `agreement == 1` is **not** track-purity proof.
- Same-family consistency can be false reassurance from color cast or
  coarse bins.
- Zero detected kit-family change is **not** purity proof.
- Continuous descriptor change must be inspected separately.

## F. Region-specific contamination

Example `816@872`:

- crop-level union contamination is high
- torso ROI still largely contains the target yellow jersey

Frozen conclusion:

- crop-level contamination ≠ torso-region contamination
- future torso-specific overlap/coverage measurement is required
- crop-level quality must **not** force the same exclusion decision for
  every downstream task

## G. Existing ReID component audit note

Accepted Stage 4B component:

`[231, 635] → global_candidate_id 231`

Kit visual review of raw track `231` observed:

- white number-9 crop
- yellow number-17 crops

Policy for this gate:

- existing Stage 4B outputs are **not** automatically modified
- the component is **not** automatically deleted
- raw track `231` must **not** be assumed to be a pure single-player track
- component `[231, 635]` requires future track-purity review after any
  purity/segmentation work
- current `global_candidate_id` is **not** proven player identity

## H. Frozen usage policy

Stage 5B status:

`visually_validated_measurement_baseline_with_track_impurity_findings`

Allowed:

- crop/track descriptor audit
- manual review ranking
- within-track kit-change risk ranking
- future continuous descriptor comparison input

Forbidden:

- automatic team assignment
- forced two-team clustering
- dominant-family → team mapping
- same-kit auto link
- different-kit hard reject
- raw-track automatic split
- global-ID automatic rewrite

## I. New intermediate gate — Stage 5B3

**Stage 5B3 — raw-track purity and within-track kit-change audit**

Goals:

- inspect crops in frame order
- measure continuous kit-descriptor change
- produce possible change-point candidates
- attach contamination/quality context
- produce a manual split-review shortlist

First Stage 5B3 version constraints:

- automatic split = false
- automatic track deletion = false
- global ID rewrite = false
- team assignment = false
- threshold = null
- manual review required

Stage **5C** jersey-number visibility comes **after** Stage 5B3.
