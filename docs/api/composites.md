# Composites

Composite builders create the stage-A raster: Earth Engine composites,
USGS NAIP Plus exports, Umbra and Capella SAR composites, local GeoTIFFs,
embedding rasters and stacks of annual crop-type maps. See
[Satellite sources](../user-guide/satellite-sources.md).

::: agribound.composites

## Earth Engine

::: agribound.composites.gee
    options:
      members:
        - GEECompositeBuilder
        - ExportTaskStartedError
        - apply_composite_method
        - collection_summary
        - IMAGE_LIST_MAX
        - date_window
        - landsat_pan_missions_for_window
        - naip_year_areas
        - split_at_antimeridian
        - hls_far_points
        - near_antimeridian
        - hls_footprint_filter
        - hls_antimeridian_images
        - near_hls_reversed_footprint_tiles
        - sentinel1_orbit_pass
        - sentinel1_db_to_uint8
        - sentinel1_uint8_to_db
      heading_level: 3

## Scenes and dates

The scene survey (`agribound scenes`) and the date layers
(`agribound composite --date-layers`) of Earth Engine composites. See
[Dates in a composite](../user-guide/satellite-sources.md#dates-in-a-composite).

::: agribound.composites.scenes
    options:
      members:
        - survey_scenes
        - rank_acquisitions
        - survey_scale_m
        - month_windows
        - acquisition_key
        - survey_expression
        - build_date_layers
        - date_layers_image
        - date_layers_path
        - check_scene_survey
        - check_date_layers
        - day_to_date
        - SURVEY_COLUMNS
        - S2_SURVEY_COLUMNS
        - RANK_RULES
        - SCENE_SURVEY_VERSION
        - SURVEY_MIN_SCALE_M
        - SURVEY_MAX_PIXELS
        - SURVEY_MAX_VERTICES
        - DATE_LAYER_BANDS
        - DATE_ORIGIN
        - DATE_LAYERS_VERSION
      heading_level: 3

## USGS NAIP Plus

::: agribound.composites.usgs
    options:
      members:
        - USGSNAIPPlusCompositeBuilder
      heading_level: 3

## Open VHR SAR (Umbra, Capella)

::: agribound.composites.vhr_sar
    options:
      members:
        - VHRSARCompositeBuilder
        - load_index
        - cached_index
        - refresh_index
        - select_collects
        - build_dem_hae
        - warp_product
        - register_adaptive
        - capella_db
        - level_normalise
        - median_composite
        - Provider
        - CollectRecord
        - ArchiveIndex
        - Selection
      heading_level: 3

## Crop-type map stacks (CDL, ACI)

The stacks of the `cdl-sequence` and `aci-sequence` sources: one band of
class codes per year, value scale `categorical`. See
[Crop-type map sequences](../user-guide/satellite-sources.md#crop-type-map-sequences-cdl-sequence-aci-sequence).

::: agribound.composites.crop_sequence
    options:
      members:
        - CropSequenceCompositeBuilder
        - build_crop_sequence
        - window_years
        - band_names
        - pad_grid
        - stack_cache_path
        - recipe_parts
        - collection_request
        - check_window_images
        - stack_image
        - parse_legend
        - pyramid_level_m
        - resampling_tag
        - valid_fraction_by_band
        - CropMapDataset
        - CROP_MAP_DATASETS
        - DATASET_SOURCES
        - CDL_AGRICULTURAL_CLASSES
        - CDL_BARRIER_CLASSES
        - ACI_AGRICULTURAL_CLASSES
        - ACI_BARRIER_CLASSES
        - CROP_SEQUENCE_PAD_PX
        - CROP_SEQUENCE_RESOLUTION_M
        - CROP_SEQUENCE_RECIPE_VERSION
      heading_level: 3

## Local GeoTIFFs and embeddings

::: agribound.composites.local
    options:
      members:
        - LocalCompositeBuilder
        - EmbeddingCompositeBuilder
        - validate_local_raster
        - tessera_coverage
        - check_tessera_raster
        - current_tessera_stamp
        - TesseraRasterCheck
        - tessera_raster_path
        - cached_tessera_tags
      heading_level: 3

## TESSERA store stamp

::: agribound.composites.tessera_stamp
    options:
      members:
        - store_stamp
        - compute_stamp
        - StoreStamp
        - ObjectState
        - StoreCheckError
        - read_object_keys
        - pixel_window
        - chunk_keys
        - geotessera_store
        - disk_cached_key
        - refresh_local_cache
        - record_local_cache
      heading_level: 3
