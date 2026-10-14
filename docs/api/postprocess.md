# Post-Processing

Polygonisation, merging, area filtering, smoothing, simplification,
regularisation, the centre-pivot step, the LULC crop filter, label-raster
helpers and the crop history of polygons.

::: agribound.postprocess

## Smoothing and simplification

::: agribound.postprocess.simplify
    options:
      heading_level: 3

## Centre pivots

The opt-in centre-pivot step (`engine_params["pivots"]`: `"snap"` or
`"merge"`), which the pipeline runs after the post-processing and before the
LULC crop filter. See
[Post-processing](../user-guide/configuration.md#post-processing).

::: agribound.postprocess.pivots
    options:
      members:
        - snap_pivots
        - resolve_pivot_mode
        - fit_circle
        - circle_tolerance_m
        - PIVOT_MODES
        - PIVOT_COLUMN
      heading_level: 3

## Label rasters

::: agribound.postprocess.regions
    options:
      members:
        - merge_small_regions
        - polygonize_labels
        - MERGE_RULES
      heading_level: 3

## Crop history

The per-year classes of a crop-type map stack for any polygons
(`agribound.attach_crop_history`). See
[Crop-sequence zones](../user-guide/engines.md#crop-history-of-other-outlines).

::: agribound.postprocess.crop_history
    options:
      members:
        - attach_crop_history
        - DEFAULT_PREFIX
      heading_level: 3

## LULC crop filter

::: agribound.postprocess.lulc_filter
    options:
      members:
        - filter_by_lulc
        - prefetch_lulc_raster
        - select_lulc_dataset
      heading_level: 3
