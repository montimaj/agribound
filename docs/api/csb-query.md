# Published Crop Sequence Boundaries

Data access for the published USDA NASS Crop Sequence Boundaries (CSB), read
in place from the national release zips. CSB are crop-history zones made from
the Cropland Data Layer, not field outlines. See
[Crop-sequence zones](../user-guide/engines.md#published-crop-sequence-boundaries-query_csb).

::: agribound.csb_query
    options:
      members:
        - query_csb
        - csb_release
        - csb_dataset
        - CSB_YEARS
        - CSB_WINDOW_YEARS
        - CSB_REVISION
        - CSB_BASE_URL
        - CSB_GDAL_OPTIONS
        - CSB_DATA_LABEL
