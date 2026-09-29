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

## Ensemble

::: agribound.engines.ensemble
    options:
      members:
        - EnsembleEngine
      heading_level: 3

## SAM refinement

!!! warning "SAM 3 backends are untested"
    `sam_backend="sam3"` and `"sam3-hf"` have not been run end to end with
    agribound 1.0.0 (the `facebook/sam3` weights are gated); a WARNING is
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
