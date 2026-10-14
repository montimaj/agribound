# Engines

Every engine subclasses `DelineationEngine` and is resolved by name with
`get_engine`. See [Delineation engines](../user-guide/engines.md) for an
overview.

::: agribound.engines

## Delineate-Anything

::: agribound.engines.delineate_anything
    options:
      members:
        - DelineateAnythingEngine
        - DA_MODELS
      heading_level: 3

## Fields of The World

::: agribound.engines.ftw
    options:
      members:
        - FTWEngine
        - list_ftw_models
      heading_level: 3

## GeoAI

::: agribound.engines.geoai_field
    options:
      members:
        - GeoAIEngine
        - merge_window_seams
        - window_edges
      heading_level: 3

## DINOv3

::: agribound.engines.dinov3
    options:
      members:
        - DINOv3Engine
        - segment_raster
        - window_taper
        - resolve_input_normalization
        - DINOV3_INPUT_NORMALIZATIONS
        - half_band_dilate_px
        - resolve_dilate_rule
        - resolve_dilate_px
        - DILATE_HALF_BAND
      heading_level: 3

## Prithvi-EO-2.0

::: agribound.engines.prithvi
    options:
      members:
        - PrithviEngine
      heading_level: 3

## Embedding clustering

::: agribound.engines.embedding
    options:
      members:
        - EmbeddingEngine
      heading_level: 3

## SAM 2.1 automatic masks

Direct delineation with Segment Anything 2.1 automatic masks on large tiles
(opt-in; never an engine default). See
[SAM 2.1 automatic masks](../user-guide/engines.md#sam-21-automatic-masks-sam-auto)
for the method, the per-source settings, the pre-registered evaluation, the
limits and the costs.

::: agribound.engines.sam_auto
    options:
      members:
        - SAMAutoEngine
        - SAMAutoSettings
        - resolve_settings
        - check_params
        - settings_warning
        - resolve_model
        - hpc_tiling_problem
        - largest_whole_object_px
        - tile_origins
        - tile_stride
        - interior_edges
        - cut_by_tile
        - box_nms
        - partition_masks
        - labels_to_polygons
        - pixel_size_m
        - read_tile
        - checkpoint_path
        - load_generator
        - SAM2PointGridGenerator
        - TileCandidates
        - sam2_distribution
        - sam2_files_sha256
        - SOURCE_SETTINGS
        - SOURCE_KEYS
        - DEFAULT_PARAMS
        - COSTLY_SOURCES
        - DID_NOT_TRANSFER
        - NO_DEFAULT_SOURCES
        - DEFAULT_MODEL
        - CHECKPOINT_FILES
        - PINNED_CHECKPOINTS
        - SAM2_META_COMMIT
        - SAM2_1_1_0_FILES_SHA256
      heading_level: 3

## Crop-sequence zones

Zones of uniform multi-year crop history from a crop-type map stack (not field
instances). See
[Crop-sequence zones](../user-guide/engines.md#crop-sequence-zones-crop-sequence).

::: agribound.engines.crop_sequence
    options:
      members:
        - CropSequenceEngine
        - crop_sequence_zones
        - Zones
        - cultivated_mask
        - barrier_mask
        - focal_mode
        - sequence_regions
        - unique_sequence_share
        - zone_attributes
        - merge_max_px
        - read_stack
        - pixel_size
        - DEFAULT_PARAMS
        - METHODS
        - IMPLEMENTATION
      heading_level: 3

## Ensemble

::: agribound.engines.ensemble
    options:
      members:
        - EnsembleEngine
      heading_level: 3

## SAM refinement

!!! warning "SAM 3 backends are untested"
    `sam_backend="sam3"` and `"sam3-hf"` have not been run end to end with
    agribound 1.1.0 (the `facebook/sam3` weights are gated); a WARNING is
    logged when one is loaded. See
    [SAM refinement](../user-guide/sam-refinement.md#sam-3-is-untested).

::: agribound.engines.samgeo_engine
    options:
      members:
        - refine_boundaries
        - crop_window_px
        - is_refinable
        - prefetch
      heading_level: 3

## Fine-tuning

::: agribound.engines.finetune
    options:
      members:
        - fine_tune
      heading_level: 3
