# Pipeline

The pipeline module is the main entry point for agribound. It orchestrates the full workflow: composite building, optional fine-tuning, delineation, post-processing, LULC crop filtering, evaluation, and export.

The LULC crop filtering step (enabled by default) automatically removes non-agricultural polygons using the best available land-use/land-cover dataset for the study area location and year. See the [Configuration Reference](../user-guide/configuration.md#post-processing) for details.

::: agribound.pipeline
