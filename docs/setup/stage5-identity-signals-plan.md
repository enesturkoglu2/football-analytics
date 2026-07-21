# Stage 5 — Football ReID auxiliary identity-signals plan (5A1)

- **Date:** 2026-07-21
- **Gate:** Stage 5A1 — policy and schema planning only
- **Status:** planning (no product extraction code in this gate)
- **Policy config:** `configs/reid/identity_signals_stage5.yaml`
- **Stage 4B baseline:** `completed_baseline` (commit `76ede91`)

This gate freezes design and policy only. No OCR/team model download,
no inference, no pipeline run, no product code, and no commit/push.

## 1. Goal

Stage 5 does **not** train a new ReID model. It adds football-context
auxiliary identity evidence on top of the existing OSNet Market1501
embedding baseline from Stage 4B.

Stage 4B context (facts):

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
→ optional weak appearance signals
→ pair-level evidence fusion
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

| Gate | Scope |
|---|---|
| **5A1** | identity-signal policy and schema planning *(this gate)* |
| **5A2** | crop-quality and contamination baseline — code + mock tests |
| **5A3** | quality smoke/full run on existing `sample.mp4` crops |
| **5B1** | coarse team/kit descriptor — code + mock tests |
| **5B2** | `sample.mp4` kit descriptor run and visual validation |
| **5C1** | jersey-number visibility audit — **no OCR model yet** |
| **5C2** | jersey-number extraction baseline selection — OCR/model decision needs separate approval |
| **5D** | pair-level evidence fusion and review ranking |
| **5E** | golden clips / ground-truth evaluation preparation |

Hair and shoe features:

- Optional enhancement **after** Stage 5D
- **Not required** for core Stage 5 success

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

Stage 5 completion does **not** require:

- Implementing every hair/shoe feature
- Training a new ReID model
- Enabling automatic linking

## 6. Hard non-goals for early Stage 5 gates

- Selecting a cosine / identity-score threshold
- Enabling automatic linking or automatic identity fusion
- Claiming MOTA / HOTA / IDF1 / ReID mAP / accuracy %
- Deleting or rewriting raw crops / raw track IDs
- Downloading OCR or team-classification models without a later approval
  gate (especially before/without 5C1 visibility audit)

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
- `configs/reid/linking_policy_stage4b.yaml`
- `configs/reid/crop_selection_stage4b.yaml`
- `configs/reid/identity_signals_stage5.yaml`
