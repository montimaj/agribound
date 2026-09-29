# Composites

Composite builders create the stage-A raster: Earth Engine composites,
USGS NAIP Plus exports, local GeoTIFFs and embedding rasters. See
[Satellite sources](../user-guide/satellite-sources.md).

::: agribound.composites

## Earth Engine

::: agribound.composites.gee
    options:
      members:
        - GEECompositeBuilder
        - ExportTaskStartedError
        - apply_composite_method
        - date_window
      heading_level: 3

## USGS NAIP Plus

::: agribound.composites.usgs
    options:
      members:
        - USGSNAIPPlusCompositeBuilder
      heading_level: 3

## Local GeoTIFFs and embeddings

::: agribound.composites.local
    options:
      members:
        - LocalCompositeBuilder
        - EmbeddingCompositeBuilder
        - validate_local_raster
        - tessera_coverage
      heading_level: 3
