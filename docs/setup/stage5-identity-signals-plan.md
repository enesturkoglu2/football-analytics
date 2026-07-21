# Stage 5 — Football ReID auxiliary identity-signals plan (5A1)

- **Date:** 2026-07-21
- **Original gate:** Stage 5A1 — policy and schema planning
- **Stage 5A status:** `visually_validated_measurement_baseline` (completed)
- **Stage 5B status:**
  `visually_validated_measurement_baseline_with_track_impurity_findings`
  (completed measurement + visual validation)
- **Stage 5B3 status:**
  `visually_validated_manual_segment_plan_not_applied`
- **Next technical gate:** Stage 5B3F — non-destructive manual segment-view
  implementation
- **Focused target-player gate:** Stage 5D
- **Pitch-position / GameState:** Stage 6 (separate from Stage 5)
- **Identity-signals policy:** `configs/reid/identity_signals_stage5.yaml`
- **Crop-quality policy:** `configs/reid/crop_quality_policy_stage5a.yaml`
- **Kit measurement config:** `configs/reid/kit_descriptor_stage5b.yaml`
- **Kit visual-validation policy:**
  `configs/reid/kit_visual_validation_policy_stage5b.yaml`
- **Track purity audit config:**
  `configs/reid/track_purity_audit_stage5b3.yaml`
- **Manual segmentation policy:**
  `configs/reid/manual_track_segmentation_policy_stage5b3.yaml`
- **Manual segment decisions (plan only):**
  `configs/reid/manual_track_segment_decisions_stage5b3.yaml`
- **Quality visual validation:** `docs/setup/stage5a-quality-visual-validation.md`
- **Kit visual validation:** [stage5b-kit-visual-validation.md](stage5b-kit-visual-validation.md)
- **Full-observation purity visual validation:**
  [stage5b3d-full-observation-visual-validation.md](stage5b3d-full-observation-visual-validation.md)
- **Focused-player / pitch roadmap:** [focused-player-and-pitch-position-roadmap.md](focused-player-and-pitch-position-roadmap.md)
- **Stage 4B baseline:** `completed_baseline` (commit `76ede91`)
- **Quality measurement commit:** `d6122d0`
- **Kit measurement commit:** `777cc43`
- **Purity audit commit:** `f73aef7`

Stage 5A1 originally froze design/policy only. Stage 5A2/5A3 later
implemented measurement and visual validation without selecting a
threshold or enabling exclusion.

Stage 5B0 later documented the multi-match **focused target-player**
product direction and Stage 6 pitch-position / GameState adapter path.
Product rules are **not** hard-coded to `sample.mp4`; that clip remains
development / reference only. Multi-match generalization is not yet
validated.

Stage 5B1/5B2 later implemented torso/kit measurement and visual
validation. Absolute color families remain audit-only. Within-track kit
change is impurity-risk evidence; raw tracks are **not** atomic identity
guarantees. No team assignment, clustering, auto-split, or global-ID
rewrite was enabled.

Stage 5B3A–5B3D later implemented measurement-only purity audit, real
run, selected-crop transition review, and full-observation localization
review. Stage 5B3E freezes a **manual non-destructive segment plan**
that is **not applied** yet. Raw tracks remain immutable.

## 1. Goal

Stage 5 does **not** train a new ReID model. It adds football-context
auxiliary identity evidence on top of the existing OSNet Market1501
embedding baseline from Stage 4B.

The system is intended for **many matches**. Primary product priority is
reliable follow-up of usually **1–2 pre-selected target players**, while
all other players remain available as anonymous context (team/role /
global candidate) for proximity, pressure, space, and team-shape
analysis. There is **no** forced two-team assignment and **no** forced
target-player assignment.

Stage 4B context (`sample.mp4` development/reference facts — **not**
cross-match constants):

| Item | Value |
|---|---|
| Raw tracks | 276 |
| Crops | 454 |
| Track embeddings | 135 |
| Candidate pairs | 9,045 |
| Manual decisions | 42 |
| Accepted components | 4 |
| Global candidates | 272 |
| Model | `osnet_x1_0` / Market1501 / not football-domain trained |
| Automatic linking | false |
| Similarity threshold | null |

Core Stage 5 flow:

```text
track crops
→ crop quality and contamination signals
→ team/kit appearance signals
→ jersey-number evidence
→ focused target enrollment / gallery design (Stage 5D)
→ optional weak appearance signals
→ pair-level evidence fusion (Stage 5E)
→ manual-review ranking
→ conservative linking input
```

Raw ByteTrack `track_id` / `raw_track_id` values remain preserved.
Stage 4B exact-frame hard-ban and manual-acceptance linking rules remain
in force unless a later gate explicitly revises them.

## 2. Frozen signal priorities (identity evidence)

### A. First priority — team/kit signal

Scope:

- coarse dominant jersey/torso color
- shorts color
- light/dark kit family
- possible team-side grouping
- goalkeeper / referee / outlier kit likelihood

Rules:

- Kit signal is **not** identity proof.
- Same kit does **not** mean same player.
- Clear kit mismatch may be **negative evidence** for a pair.
- Lighting, shadow, and camera color shifts must be considered.
- First version must **not** use kit mismatch as a hard reject.
- Usage: audit and review ranking signal.

### B. Second priority — jersey-number evidence

Scope:

- whether a jersey number is visible
- number readability / confidence
- crop orientation: front / back / side / unknown
- per-crop number candidates
- track-level number consensus
- pair-level same/different number evidence

Rules:

- Readable **different** numbers are strong **negative** evidence.
- Matching visible numbers are supportive **positive** evidence.
- Same number alone is **not** link proof.
- Unreadable numbers stay `unknown`.
- OCR output is **not** ground truth.
- Low-confidence OCR must not auto-link or auto-reject.
- A visibility audit is required before OCR/model adoption.

### C. Third priority — crop quality and contamination

Scope:

- second-person likelihood inside the crop
- severe occlusion
- truncated head/feet
- motion blur
- crop sharpness
- player occupancy ratio
- background dominance
- bbox edge contact
- usable torso/back region

Rules:

- Contamination degrades embedding quality.
- Dirty crops may later be down-weighted in track aggregation.
- Very dirty crops may be excluded from identity-signal extraction.
- Raw crops are **never deleted**; only quality metadata is produced.

### D. Fourth priority — temporal/spatial consistency

Scope:

- exact-frame hard-ban (preserved from Stage 4B)
- track end/start times
- temporal gap
- last/first pitch location
- plausible motion distance
- camera cut / scene change cues

Rules:

- Exact-frame conflict remains a definitive hard reject.
- Spatial rules must not be hard rejects while camera transform is unknown.
- First version usage: audit and ranking only.

### E. Fifth priority — weak appearance signals

Scope:

- hair / head appearance
- shoe color
- socks
- sleeve length
- undershirt color
- body silhouette

Rules:

- These are weak and optional.
- Alone they cannot create an accepted link or hard reject.
- If shoes/hair are not visible → `unknown`.
- Confidence must stay low because the image area is small.
- Sensitive to lighting, blur, and viewpoint.

## 3. Implementation gate order

| Gate | Scope | Status |
|---|---|---|
| **5A1** | identity-signal policy and schema planning | completed |
| **5A2** | crop-quality and contamination baseline — code + mock tests | completed |
| **5A3** | real full run + visual validation on existing `sample.mp4` crops | completed |
| **5B0** | focused-player + pitch-position roadmap (docs only) | completed |
| **5B1** | coarse team/kit descriptor — code + mock tests | completed |
| **5B2** | `sample.mp4` kit descriptor run and visual validation (dev/reference only) | completed |
| **5B3A** | raw-track purity audit implementation (code + mock tests) | completed |
| **5B3B** | real purity measurement run on full Stage 4B/5B outputs | completed |
| **5B3C** | selected-crop transition visual review panels | completed |
| **5B3D** | full-observation visual validation / switch localization review | completed |
| **5B3E** | manual segment plan and policy freeze (docs/config only) | completed |
| **5B3F** | non-destructive manual segment-view implementation | next technical |
| **5B3G** | segmented crop/embedding/ReID regression run | pending |
| **5C1** | jersey-number visibility audit — **no OCR model yet** | pending |
| **5C2** | jersey-number extraction baseline selection — OCR/model decision needs separate approval | pending |
| **5D** | focused target-player enrollment and gallery design (`target_A` / `target_B` / `non_target` / `unknown`) | pending |
| **5E** | pair-level auxiliary evidence fusion and manual-review ranking | pending |
| **6A** | camera calibration / pitch projection adapter investigation | pending (Stage 6) |
| **6B** | isolated GameState or equivalent 2D pitch-position integration | pending (Stage 6) |
| **6C** | appearance + kit + number + pitch-position continuity fusion | pending (Stage 6) |
| **6D** | multi-match golden clips and official evaluation | pending (Stage 6) |

Hair, shoe, and controlled-marker features:

- Optional weak / supporting evidence
- **Not** core identity proof
- **Not required** for core Stage 5 / Stage 6 success

Frozen product-stage summary (see also
[focused-player-and-pitch-position-roadmap.md](focused-player-and-pitch-position-roadmap.md)
and [stage5b-kit-visual-validation.md](stage5b-kit-visual-validation.md)):

- **5A** crop quality — completed
- **5B** coarse team/kit measurement + visual validation — completed
- **5B3A** purity audit implementation — completed
- **5B3B** real measurement run — completed
- **5B3C** selected-crop transition visual review — completed
- **5B3D** full-observation visual validation — completed
- **5B3E** manual segment plan and policy freeze — completed
  (status: `visually_validated_manual_segment_plan_not_applied`)
- **5B3F** non-destructive manual segment-view implementation — next
- **5B3G** segmented crop/embedding/ReID regression run — pending
  (old raw-track baseline vs segmented-track result on `sample.mp4`
  development/reference; no raw-track in-place mutation; no automatic
  segment merging)
- **5C** jersey-number visibility audit
- **5D** focused target-player enrollment / gallery
- **5E** pair-level fusion + manual-review ranking
- **6A–6D** pitch-position / GameState adapter path and multi-match eval

### Stage 5A completion notes

- Stage 5A2 quality measurement implementation: **completed**
- Stage 5A3 real full run and visual validation: **completed**
- Stage 5A status: **`visually_validated_measurement_baseline`**
- No quality/contamination threshold selected (`null`)
- No automatic exclusion selected
- No embedding aggregation weighting enabled
- Tracking bbox overlap is positive-risk evidence only; zero overlap is
  **not** clean proof
- Global Laplacian threshold is **prohibited**; size-stratified
  audit/ranking is allowed
- Frame-edge contact is **audit only** (not body-completeness proof)

### Stage 5B completion notes

- Stage 5B1 measurement implementation: **completed**
- Stage 5B2 real run and visual validation: **completed**
- Stage 5B status:
  **`visually_validated_measurement_baseline_with_track_impurity_findings`**
- No team assignment / forced clustering / kit similarity threshold
- Absolute color-family labels remain **audit only** (not team labels)
- `raw_track_id` is **not** an atomic player-identity guarantee
- Within-track kit change is impurity-risk evidence; agreement == 1 is
  **not** purity proof
- No automatic split, track deletion, or global-ID rewrite
- Accepted Stage 4B component `[231, 635]` requires future purity review;
  not auto-reversed
- `sample.mp4` remains development / reference only

### Stage 5B1 guardrails (retained)

- Use a torso-oriented descriptor
- Preserve off-pitch / outlier / unknown handling
- No forced two-team assignment
- Kit similarity is **not** identity proof
- Must not produce any automatic link or hard reject

### Stage 5B3 completion notes

- Stage 5B3A purity audit implementation: **completed**
- Stage 5B3B real measurement run: **completed**
- Stage 5B3C selected-crop transition visual review: **completed**
- Stage 5B3D full-observation visual validation: **completed**
  (15 focus tracks, 1329 raw observations, 22 windows; no interpolation)
- Stage 5B3E manual segment plan + policy freeze: **completed**
- Stage 5B3 status:
  **`visually_validated_manual_segment_plan_not_applied`**
- Manual plan: 13 split candidates, 2 no-split contamination controls,
  15 probable switch events (11 gap-bounded, 2 overlap-ambiguous,
  2 adjacent-observation) — **not** ground truth
- Raw tracks immutable; plan **not applied**; no exact split decision
  written into tracking outputs
- `[231, 635]` unchanged; only future `raw_231_s02` may be re-reviewed
  against `635`; `raw_231_s01` must not inherit the component
- Next technical gate: **Stage 5B3F non-destructive manual segment-view**
- Later: **Stage 5B3G** segmented crop/embedding/ReID regression test
  on real video (`sample.mp4` development/reference), comparing old
  raw-track baseline vs segmented-track result without mutating raw
  tracks or auto-merging segments
- Stage 5B3 remains a **safety gate before Stage 5C** jersey-number
  visibility

### Stage 5B3 guardrails (retained)

- Automatic split = **false**
- Automatic track deletion = **false**
- Global ID rewrite = **false**
- Team assignment = **false**
- Threshold = **null**
- Manual review required
- No identity-switch / track-purity ground truth
- No accuracy claim from manual visual observations
- Frame-range semantics: existing observations only; no interpolation

### Stage 5D / Stage 6 roadmap notes (preview)

- Stage **5D** is the focused target-player enrollment and gallery design
  gate (`target_A` / `target_B` / `non_target` / `unknown`)
- Automatic target assignment starts **off**
- No automatic gallery expansion from high cosine alone
- Remaining `unknown` is required when evidence is insufficient
- Other players stay available as anonymous context for analysis
- GameState / 2D pitch position is **Stage 6**, as a separate adapter;
  it does **not** replace ReID
- Initial spatial usage is audit/ranking only; spatial hard reject starts
  **false**
- `sample.mp4` remains development / reference only

Note: implementation starts with crop quality (5A2) as a foundation for
downstream kit/number reliability. Identity-evidence priority ranking
above still places team/kit and jersey number ahead of weak appearance
for review decisions.

Config field `signal_priorities` lists pipeline-oriented enablement order
(`crop_quality` → `team_kit` → `jersey_number` → `temporal_spatial` →
`weak_appearance`) and does not override the evidence priority narrative
in section 2.

## 4. Draft future data schemas (not frozen product schemas)

These schemas are **planning drafts only**. They are **not** frozen
product schemas. No product JSONL files are created in 5A1.

### A. `crop_identity_signals.jsonl` (draft)

Suggested fields:

- `crop_id`
- `track_id`
- `frame_index`
- `quality_score`
- `contamination_score`
- `blur_score`
- `occupancy_ratio`
- `torso_region_valid`
- `kit_color_descriptor`
- `jersey_number_visible`
- `jersey_number_candidates`
- `weak_appearance_signals`
- `schema_version`

### B. `track_identity_signals.jsonl` (draft)

Suggested fields:

- `track_id`
- `usable_crop_count`
- `rejected_quality_crop_count`
- `kit_consensus`
- `jersey_number_consensus`
- `jersey_number_confidence`
- `weak_signal_summary`
- `schema_version`

### C. `pair_identity_evidence.jsonl` (draft)

Suggested fields:

- `track_id_a`
- `track_id_b`
- `cosine_similarity`
- `exact_frame_conflict`
- `kit_compatibility`
- `jersey_number_relation`
- `quality_evidence`
- `temporal_spatial_evidence`
- `weak_appearance_evidence`
- `manual_review_priority`
- `automatic_decision`
- `schema_version`

Until a later freeze gate, field names/types may change. Do not treat
these drafts as authoritative product contracts.

## 5. Success criteria

Stage 5 is successful when:

- Dirty crops are measurably flagged
- Team/kit signals are visually validated
- Jersey-number visibility rates are measured
- Auxiliary signals are explainable in audit files
- Exact-frame safety is preserved
- No accuracy claim is made without ground truth
- Anonymous full-scene context tracks remain available for analysis
- Focused target-player design (Stage 5D) can keep `unknown` and avoid
  forced assignment

Stage 5 completion does **not** require:

- Continuously naming all 22 players
- Implementing every hair/shoe/marker feature
- Training a new ReID model
- Enabling automatic linking
- Installing or running GameState inside `football-cv`

## 6. Hard non-goals for early Stage 5 gates

- Selecting a cosine / identity-score threshold
- Enabling automatic linking or automatic identity fusion
- Enabling automatic target assignment or automatic gallery expansion
- Forced two-team or forced target-player assignment
- Automatic team assignment or forced two-team clustering from kit
  descriptors
- Dominant-family → team mapping or sample-specific color mapping
- Automatic raw-track split / deletion or global-ID rewrite
- Claiming MOTA / HOTA / IDF1 / ReID mAP / accuracy %
- Deleting or rewriting raw crops / raw track IDs
- Hard-coding product rules to `sample.mp4` IDs / counts / kit colors
- Treating GameState as a ReID replacement
- Downloading OCR or team-classification models without a later approval
  gate (especially before/without 5C1 visibility audit)
- Bulk-installing sn-gamestate into `football-cv`

## 7. Relationship to Stage 4B linking policy

Stage 5 auxiliary signals feed **manual-review ranking** and future
audited linking inputs. Until fusion is explicitly enabled and approved:

- `automatic_identity_fusion_enabled = false`
- `automatic_linking_enabled = false`
- `identity_score_threshold = null`
- Stage 4B policy remains the linking safety baseline:
  `configs/reid/linking_policy_stage4b.yaml`

## 8. References

- `docs/setup/reid-stage4b-completion.md`
- `docs/setup/reid-stage4b-linking-policy.md`
- `docs/setup/reid-stage4b-schema-decisions.md`
- `docs/setup/stage5a-quality-visual-validation.md`
- [stage5b-kit-visual-validation.md](stage5b-kit-visual-validation.md)
- [stage5b3d-full-observation-visual-validation.md](stage5b3d-full-observation-visual-validation.md)
- [focused-player-and-pitch-position-roadmap.md](focused-player-and-pitch-position-roadmap.md)
- `configs/reid/linking_policy_stage4b.yaml`
- `configs/reid/crop_selection_stage4b.yaml`
- `configs/reid/identity_signals_stage5.yaml`
- `configs/reid/crop_quality_policy_stage5a.yaml`
- `configs/reid/kit_descriptor_stage5b.yaml`
- `configs/reid/kit_visual_validation_policy_stage5b.yaml`
- `configs/reid/track_purity_audit_stage5b3.yaml`
- `configs/reid/manual_track_segmentation_policy_stage5b3.yaml`
- `configs/reid/manual_track_segment_decisions_stage5b3.yaml`
