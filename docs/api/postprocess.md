# Post-Processing

Polygonisation, merging, area filtering, smoothing, simplification,
regularisation and the LULC crop filter.

::: agribound.postprocess

## Smoothing and simplification

::: agribound.postprocess.simplify
    options:
      heading_level: 3

## LULC crop filter

::: agribound.postprocess.lulc_filter
    options:
      members:
        - filter_by_lulc
        - prefetch_lulc_raster
        - select_lulc_dataset
      heading_level: 3
