# Stage 5C-A2b3 — Jersey Manual Review Annotation Protocol

Gate: **Stage 5C-A2b3a** — protocol, config, CSV validator, docs, synthetic tests.

This document defines how humans fill the blank annotation template produced
by Stage 5C-A2b2 panel generation. It does **not** run OCR, recognition,
identity assignment, team assignment, or gallery updates.

## Artifacts

| Role | Path / name |
|---|---|
| Protocol config | `configs/reid/jersey_manual_review_stage5c.yaml` |
| Validator module | `src/football_analytics/reid/jersey_annotation.py` |
| CLI | `scripts/validate_jersey_review_annotations.py` |
| Real review output (immutable) | `outputs/reid/full_stage4b/jersey_review_stage5c/` |
| Blank template | `jersey_review_annotations_template.csv` |
| Canonical items | `jersey_review_items.jsonl` |

Product code must **not** hard-code observed run sizes (for example
canonical item count). Counts always come from review artifacts.

## Core semantics

### Blank row vs `uncertain`

| State | Meaning |
|---|---|
| **Unreviewed** | Every manual field is blank. No human decision was recorded. Valid under partial-dataset review. |
| **`uncertain`** | An explicit human decision that a field cannot be decided safely. Not the same as leaving the cell blank. |

A row with only `reviewer` or only `reviewed_at` filled is **invalid**.

A reviewed row always requires both `reviewer` and timezone-aware ISO-8601
`reviewed_at`.

### Manual field definitions

#### `manual_crop_valid`

| Value | Meaning |
|---|---|
| `valid` | Tile shows the correct crop; the target player is inspectable; crop is not wholly broken or wrong-source. |
| `invalid` | Wrong/broken crop, target player not distinguishable, or a severe source problem that invalidates review. |
| `uncertain` | Validity cannot be decided safely. |

This field is **not** an image-quality score.

#### `manual_back_facing`

| Value | Meaning |
|---|---|
| `yes` | Player back is sufficiently visible. |
| `no` | Not a back view (clear front/side). |
| `uncertain` | Orientation cannot be decided safely. |

`back_facing=no` does **not** mean the number is invisible.

#### `manual_number_visible`

| Value | Meaning |
|---|---|
| `yes` | At least one digit or meaningful digit fragment belonging to the **target** player's jersey is visible. |
| `no` | No jersey number belonging to the target player is visible. |
| `uncertain` | Whether a mark is a jersey number cannot be decided safely. |

Never label another player's number as the target player's number.

#### `manual_number_readable`

| Value | Meaning |
|---|---|
| `yes` | A full one- or two-digit number can be written without guessing. |
| `no` | A number may be visible, but the full value cannot be read safely. |
| `uncertain` | Two plausible readings, or readability is borderline. |

`visible=yes, readable=no` is a valid combination.

#### `manual_digit_count`

| Value | Meaning |
|---|---|
| `0` | No visible number. |
| `1` | Single digit confidently determined. |
| `2` | Two digits confidently determined. |
| `uncertain` | Digit count is not certain. |

This is **not** a bbox or OCR prediction.

#### `manual_jersey_number`

Fill **only** when:

- `manual_number_readable=yes`
- `manual_digit_count` is `1` or `2`

Store as a **string**. Examples: `"7"`, `"10"`, `"01"`.

Reject guesses, `?`, dashes, spaces, letters, and more than two digits.
Preserve leading zeros.

#### `manual_contamination_affects_number_region`

| Value | Meaning |
|---|---|
| `yes` | Overlap from another person/object meaningfully affects the target player's likely number region. |
| `no` | Contamination does not affect the number region. |
| `uncertain` | Effect cannot be assessed safely. |

Automatic bbox contamination metrics do **not** auto-fill this field.

#### `manual_notes`

Short observational notes only. No identity guesses or player names.

Examples:

- `partial left digit`
- `another player covers upper torso`
- `side view`

#### `reviewer` / `reviewed_at`

Required for any reviewed row. `reviewed_at` must be ISO-8601 **with timezone**
(for example `2026-07-22T14:00:00+03:00` or `...Z`).

## Cross-field rules (validator)

1. **Fully blank manual fields** → `unreviewed`, valid, counts as no decision.
2. **`manual_crop_valid=invalid`** → `back_facing` blank or `uncertain`;
   visible/readable/digit_count/jersey blank; contamination blank or
   `uncertain`; reviewer + reviewed_at required.
3. **`crop_valid` in {valid, uncertain}** reviewed rows → all core fields
   filled (crop_valid, back_facing, visible, readable, digit_count,
   contamination, reviewer, reviewed_at). Notes/jersey conditional.
4. **`visible=no`** → `readable=no`, `digit_count=0`, jersey blank.
5. **`visible=uncertain`** → not `readable=yes`; jersey blank;
   `digit_count` in {`0`, `uncertain`}.
6. **`readable=yes`** → `visible=yes`; `digit_count` in {`1`,`2`}; jersey
   filled; jersey length equals digit_count; ASCII digits only.
7. **`readable=no`** → jersey blank; if `visible=yes` then digit_count in
   {`1`,`2`,`uncertain`}; if `visible=no` then digit_count=`0`.
8. **`readable=uncertain`** → jersey blank; visible in {`yes`,`uncertain`};
   digit_count in {`1`,`2`,`uncertain`}.
9. **Jersey filled** → readable=yes, visible=yes, length matches digit_count,
   one or two ASCII digits.
10. Any filled decision field → reviewer + timezone-aware reviewed_at.
    Reviewer/timestamp alone without decisions → reject.

## Exact template identity

Annotation CSV provenance columns must match canonical review items
**exactly**, in template order:

- `review_item_id`, `review_index`, `segment_id`, `raw_track_id`,
  `crop_id`, `frame_index`, `master_panel_path`, `master_panel_page`,
  `master_panel_tile_index`, `group_memberships`

Rules:

- every expected ID exactly once
- no missing/extra rows
- no row reordering
- provenance fields unchanged
- `group_memberships` byte-identical JSON string
- only manual columns may change

## CLI

```bash
conda run -n football-cv \
  python scripts/validate_jersey_review_annotations.py \
  --review-dir PATH \
  --annotations-csv PATH \
  --config configs/reid/jersey_manual_review_stage5c.yaml \
  --report-path PATH \
  [--overwrite]
```

The CLI opens the source review directory read-only, never modifies the
annotation CSV, and writes the validation report atomically.

Report schema: `reid_jersey_annotation_validation_report_v1`.

Invalid CSV may still produce a report with `status=invalid` and a
nonzero CLI exit code.

## Pilot review plan (next gate: Stage 5C-A2b3b)

This gate does **not** create or fill a pilot CSV.

### Pilot universe construction

1. **All critical-segment crops**
   - Declared critical segments: 14 (from measurement/review planning).
   - Observed in the current real Stage 5C-A2b2 run: **42** critical-segment
     crops. Treat that number as a run observation only; do not hard-code it
     into product code.
2. **Balance controls** (deterministic, duplicate-free from canonical items):
   - ROI-height top: 6
   - ROI-height bottom: 6
   - contamination-high: 6
   - contamination-low: 6
   - local-contrast top: 6
   - local-contrast bottom: 6

Add the critical set first. Then walk each balance group in canonical order
and append items not already selected.

Target: **at most 78 unique** pilot items (fewer if overlap). The true pilot
count is reported in Stage 5C-A2b3b preflight.

### Pilot goals

- check that field definitions are understandable
- check consistency on similar crops
- calibrate visible vs readable
- assess contamination-field usability
- catch protocol errors before full review of all canonical items

Pilot outcomes are **not** model accuracy.

## Hard limits

- Manual jersey labels are **not** physical player identity ground truth.
- The same number can exist on both teams.
- A readable number alone is **not** a global identity.
- Preserve unknown/`uncertain`; do not guess.
- Do not assign another player's visible number to the target.
- Panel ROI zoom is **not** a recognition input.
- Selected crops are **not** all observations.
- This stage does **not** update the gallery.
- Stage 5D target enrollment is separate.
- Stage 5E fusion/evaluation is separate.
- Review items, panels, and source crops stay immutable.

## Safety flags (always false in validation reports)

- `OCR_performed`
- `recognizer_performed`
- `checkpoint_loaded`
- `gallery_updated`
- `identity_assigned`
- `team_assigned`
- `source_review_modified`
- `panel_modified`
- `source_crop_modified`
- `identity_ground_truth_available`
- `accuracy_claimed`
