# ReID-R2C2 — Safe SportsReID SoccerNet OSNet adapter

- **Date:** 2026-07-27
- **Status:** controlled production adapter for sanitized SportsReID state-dict only

## Model IDs

| model_id | Domain | Loader |
|---|---|---|
| `osnet_x1_0_market1501` | Market1501 baseline | existing `torchreid` `load_pretrained_weights` (unchanged) |
| `osnet_x1_0_sportsreid_soccernet` | SoccerNet-trained SportsReID | `weights_only=True` sanitized state-dict loader |

Registry: `configs/reid/model_registry.yaml`

Selection is explicit. There is **no** silent fallback from SportsReID to Market1501.

## SportsReID contract

- Checkpoint must be the sanitized state-dict-only file from R2C security review
- SHA-256 is mandatory
- Original ~1.02GB full training checkpoint is rejected
- Allowlist is not used in the production adapter
- `weights_only=False` is forbidden
- Classifier-only mismatches may be ignored; backbone mismatches fail closed

## Not approved for

- Automatic identity assignment
- Deployment thresholding
- Gallery auto-expansion
