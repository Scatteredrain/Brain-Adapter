# Release Config Aliases

This directory provides public-facing aliases for the configs that are currently the best candidates for paper and rebuttal comparisons.

## Why this directory exists

The original configs under `configs/icp/` still use:

- old experiment names
- private absolute dataset paths
- private checkpoint paths

These release aliases keep the original experiment logic but replace the most private or confusing fields with placeholders that are easier to publish and maintain.

## Current alias set

### BiomedCLIP comparison group

- `abmil_biomedclip.yml`
- `abmil_biomedclip_vpt.yml`
- `abmil_biomedclip_clip_adapter.yml`

### RadCLIP comparison group

- `abmil_radclip.yml`
- `abmil_radclip_vpt.yml`
- `abmil_radclip_clip_adapter.yml`

## Notes

- These aliases are not yet a guarantee of exact paper-table reproducibility.
- They are the current best public-facing starting points.
- The `brain_adapter_paper*.yml` configs now run behind explicit paper-core guards, so incompatible regression and retrieval paths will fail fast instead of silently mixing tasks.
- Exact paper-table reproducibility still needs final environment and dataset verification.

## Expected user action

Before running, replace placeholder paths such as:

- `REPLACE_WITH_TRAIN_IMAGE_LIST`
- `REPLACE_WITH_TEST_IMAGE_LIST`
- `REPLACE_WITH_TRAIN_TEXT_LIST`
- `REPLACE_WITH_TEST_TEXT_LIST`
- `REPLACE_WITH_TRAIN_LABEL_LIST`
- `REPLACE_WITH_TEST_LABEL_LIST`
- `REPLACE_WITH_RADCLIP_CHECKPOINT`

with your actual local paths.
