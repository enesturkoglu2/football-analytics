# Focused Player ReID and Pitch-Position Roadmap

- **Date:** 2026-07-21
- **Gate:** Stage 5B0 — product-target and pitch-position roadmap (docs only)
- **HEAD reference:** `bda1973` — Document Stage 5A crop quality validation
- **Stage 4B:** `completed_baseline`
- **Stage 5A:** `visually_validated_measurement_baseline`
- **Related plan:** [stage5-identity-signals-plan.md](stage5-identity-signals-plan.md)

This gate freezes product-direction documentation only. No product code,
config change, inference, dependency install, or commit/push.

Frozen safety defaults (unchanged by this roadmap):

| Setting | Value |
|---|---|
| Automatic linking | false |
| Similarity threshold | null |
| Quality threshold | null |
| Contamination threshold | null |
| Automatic crop exclusion | false |
| Automatic target assignment | false |
| Automatic gallery expansion | false |
| Spatial hard reject (initial) | false |

## A. Primary product goal

- The system is designed to run on **many different matches**.
- `sample.mp4` is only a **development / reference full-run video**.
- No product rule is hard-coded to `sample.mp4` track IDs, crop counts,
  kit colors, or resolution.
- Counts measured on `sample.mp4` are **not** expected constants for
  other matches.
- Multi-match generalization is **not yet validated**.

Primary usage priority:

- Reliably following a pre-selected **1 or 2 target players** is more
  important than naming all 22 players on every match.
- Other players may remain anonymous global candidates or team/role-level
  entities.
- Other players continue to be tracked for proximity, pressure, space,
  team shape, and contextual analysis.

## B. Two-layer analysis architecture

### Layer 1 — full-scene anonymous context

For all visible players, preserve as much as available:

- `raw_track_id`
- `global_candidate_id`
- team/kit candidate
- role candidate
- anonymous identity state
- temporal track information
- future 2D pitch position

Even without real player names, Layer 1 can support:

- nearest opponent to a target player
- pressure distance
- player density
- empty space
- team width and depth
- team blocks
- team/opponent distribution around a target player

### Layer 2 — focused target-player identity

For usually 1 or 2 pre-selected players, stronger identity evidence is
used:

- manually verified enrollment crops
- target embedding gallery
- team/kit compatibility
- jersey-number evidence
- crop/region quality
- temporal continuity
- future pitch-position continuity
- optional controlled visual markers

Possible states:

- `target_A`
- `target_B`
- `non_target`
- `unknown`

Tracks are **not** forcibly assigned to a target player.

## C. Target enrollment policy

- A target player is introduced with **manually verified** crops or
  tracks.
- The gallery is **not** expanded automatically from every high cosine
  hit.
- Gallery growth is allowed only for manually verified tracks, or later
  for exceptionally strong multi-evidence tracks under a future approved
  gate.
- Incorrect gallery growth can cause **identity drift**.
- Target gallery source track/crop **provenance** is preserved.
- Raw `track_id` / `raw_track_id` values are **never rewritten**.

## D. Focused ReID safety policy

- OSNet cosine alone is **not** target-player proof.
- Same kit alone is **not** player proof.
- Same jersey number alone is **not** player proof.
- Armband / marker alone is **not** player proof.
- Exact-frame conflict remains a **hard reject**.
- Remaining `unknown` is mandatory when evidence is insufficient.
- Automatic target assignment starts **off**.
- Target acceptance requires manual confirmation or multi-evidence
  acceptance under a later approved gate.
- Uncontrolled transitive chaining is **forbidden**.

## E. Controlled visual marker option

In controlled training or special capture settings, optional helpers may
include:

- high-contrast armband
- distinctive sleeve band
- another visual marker permitted by the organization

Rules:

- Broadcast matches are **not** assumed to contain markers.
- The system must work **without** markers.
- When a marker is not visible → `unknown` for that signal.
- A marker alone must **not** create an auto-link.
- Marker color should separate from kits, referee clothing, and
  opponents.
- Small pixel area, viewpoint, occlusion, and motion blur limit this
  signal.
- Official match equipment rules must be checked separately.

This gate does **not** implement a marker detector.

## F. 2D pitch position / GameState direction

GameState or an equivalent camera-calibration / pitch-projection system
will later be used as a **separate adapter**.

Goals:

- per-frame player footpoint estimate
- project footpoint into 2D pitch coordinates
- track start/end pitch locations
- temporal gap
- physically plausible travel distance
- target-player continuity

Planned data boundary:

```text
sn-gamestate or separate calibration environment
→ pitch_position JSONL/Parquet
→ football-cv adapter
→ pair-level spatial evidence
```

Example future fields:

- `frame_index`
- `raw_track_id`
- `pitch_x_m`
- `pitch_y_m`
- `projection_valid`
- `calibration_confidence`
- `position_source`
- `camera_segment_id`
- `schema_version`

Rules:

- sn-gamestate dependencies are **not** bulk-installed into `football-cv`
- use an isolated environment / subprocess / file adapter
- preserve `raw_track_id`
- account for camera cuts and calibration segments
- no spatial hard reject in the first version due to calibration error
  risk
- pitch position first usage: `audit_and_ranking`
- exact-frame hard reject remains separate and preserved

## G. Relationship of GameState to ReID

GameState does **not** replace current ReID.

Combined evidence may include:

- OSNet appearance
- team/kit
- jersey number
- target gallery
- temporal gap
- 2D pitch continuity
- quality evidence

This combination may support:

- down-ranking impossible or weak motion transitions
- ranking recently ended / newly started tracks
- reducing ambiguity among same-kit players
- sustaining a target player's on-pitch trajectory

But:

- camera calibration can be wrong
- bbox footpoint is approximate
- players can be occluded
- after a camera cut, the pitch transform can change
- 2D position alone is **not** identity proof

## H. Multi-match validation strategy

`sample.mp4`:

- development / reference clip only

Future validation set should include:

- different matches
- different teams and kit colors
- day / night
- different camera zoom levels
- different stadiums
- different broadcast quality
- clips with different camera-cut density
- clips with different target-player visibility

Future golden clips should include:

- true target-player track fragments
- same-player / different-player pair labels
- team labels
- jersey-number visibility / readability
- pitch-position validity
- occlusion / camera-cut metadata

No accuracy percentage is reported without ground truth.

## I. Frozen stage order

| Stage | Scope | Notes |
|---|---|---|
| **5A** | crop-quality measurement and visual validation | **completed** |
| **5B** | coarse team/kit descriptor | torso-oriented; no forced two-team assignment; no auto link/reject |
| **5C** | jersey-number visibility audit | recognition/OCR selection needs separate approval |
| **5D** | focused target-player enrollment and gallery design | `target_A` / `target_B` / `non_target` / `unknown`; no automatic gallery expansion |
| **5E** | pair-level auxiliary evidence fusion | manual-review ranking first |
| **6A** | camera calibration / pitch projection adapter investigation | docs/investigation first |
| **6B** | isolated GameState or equivalent 2D pitch-position integration | adapter boundary |
| **6C** | appearance + kit + number + pitch-position continuity fusion | still no uncontrolled auto-link |
| **6D** | multi-match golden clips and official evaluation | ground truth required before metrics |

Hair, shoe, and marker signals:

- optional weak / supporting evidence
- **not** core identity proof
- **not** required to complete the main stages above

## J. Success definition

Project success is **not** only continuously naming all 22 players.

A successful system:

- preserves all players as anonymous context tracks
- can follow designated 1–2 target players with high precision
- can remain `unknown` under ambiguity
- limits incorrect identity merges
- preserves team/position context of other players for analysis
- explains decisions with auditable evidence
