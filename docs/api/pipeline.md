# Pipeline

`agribound.pipeline` runs the end-to-end workflow: seed → output-reuse check →
composite (+ LULC raster prefetch) → fine-tuning → engine → SAM refinement →
study-area selection → post-processing → LULC crop filter → metadata columns →
evaluation → export → provenance record. `build_composite` runs only the first
stage. See the [Quickstart](../user-guide/quickstart.md) and
[Reproducibility](../user-guide/reproducibility.md).

::: agribound.pipeline
    options:
      members:
        - delineate
        - build_composite
        - study_area_in_crs
        - select_in_study_area
