# Post-Processing

The postprocess module provides utilities for cleaning up delineated field boundary polygons, including polygonization, simplification, regularization, area filtering, cross-tile merging, and LULC-based crop filtering.

The `lulc_filter` submodule automatically removes non-agricultural polygons using NLCD (US), Dynamic World (global, ≥2016), or C3S Land Cover (global, pre-Sentinel). This is integrated into the main pipeline and enabled by default.

::: agribound.postprocess
