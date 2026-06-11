# Changelog

All notable changes to agribound will be documented in this file.

## [0.1.3.post1] - 2026-06-11

### Changed
- Jeremy Rapp moved to second author across all citations (CITATION.cff, README, docs, manuscript)
- Citations updated to reference the manuscript in preparation for *Remote Sensing of Environment* (previously *Journal of Open Source Software*)

### Removed
- JOSS paper sources (`paper/paper.md`, `paper/paper.bib`) following the decision to target the RSE special issue "Geospatial Foundation Models for Advancing Remote Sensing of Environment"

## [0.1.3] - 2026-05-26

### Added
- Example 17: query helper for published FTW polygons -- access pre-computed Fields of The World field boundaries by AOI without running inference (contributed by Jeremy Rapp, Michigan State University)
- `agribound.ftw_query` module with CLI subcommand for AOI-based queries against published FTW polygons
- PyArrow backend (`agribound.ftw_arrow`) for efficient parquet-based polygon retrieval with area masking
- Jupyter notebook walkthrough for Example 17 and accompanying user-guide page (`docs/user-guide/ftw-query.md`)
- Unit tests for FTW query and PyArrow backend

### Changed
- Pinned `lycheeverse/lychee-action` to v0.18.1 in CI to stabilize link-check workflow
- Applied Ruff formatting across `cli.py`, `ftw_arrow.py`, `ftw_query.py`, and the FTW query notebook

## [0.1.2] - 2026-04-04

### Added
- Example 16: USGS NAIP Plus ImageServer support -- same NAIP data as GEE, acquired directly from the [USGS USGSNAIPPlus ImageServer](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer) -- for non-GEE high-resolution field delineation (contributed by Jeremy Rapp, Michigan State University)
- Jeremy Rapp added to project authors and citations

### Changed
- Updated all citations to include Jeremy Rapp
- Updated example documentation to highlight USGS NAIP Plus workflow

## [0.1.1] - 2026-03-30

### Fixed
- YOLO fine-tuning checkpoint path mismatch: used absolute paths and `exist_ok=True` to prevent Ultralytics from auto-incrementing directory names (e.g., `DA-large2`, `DA-large3`)
- README images now use absolute URLs so they render correctly on PyPI

### Changed
- SPOT 6/7 multispectral resolution corrected from 1.5 m to 6 m throughout documentation and examples (panchromatic remains 1.5 m)
- FTW citation year updated from 2024 to 2025 (AAAI publication)
- GeoAI engine name standardized to "GeoAI Field Boundary" (was "GeoAI Field Delineator" in some places)
- DINOv2/v3 references simplified to DINOv3 throughout
- Embedding engine install command corrected in docs (was showing `agribound[geoai]`)
- Prithvi ViT embed mode: added documentation noting that fine-tuning is recommended (raw embeddings tend to over-merge fields)
- Updated project structure in README
- Added Australia Murray-Darling Basin entry to docs gallery
- Badges updated: Zenodo DOI, Python 3.10+, GitHub Pages docs, release badge, GEE, GitHub stars

## [0.1.0] - 2026-03-29

### Added
- Initial public release
- Seven delineation engines: Delineate-Anything, FTW, GeoAI, DINOv3, Prithvi, Embedding, Ensemble
- SAM2 boundary refinement post-processing
- Multi-satellite support: Landsat, Sentinel-2, HLS, NAIP, SPOT, local GeoTIFF, Google/TESSERA embeddings
- Automatic LULC crop filtering (NLCD, Dynamic World, C3S)
- Google Earth Engine composite generation
- Fine-tuning support for DA (YOLO), GeoAI (Mask R-CNN), DINOv3 (LoRA), Prithvi (terratorch)
- CLI (`agribound delineate`) and Python API (`agribound.delineate()`)
- 16 example scripts and Jupyter notebooks
- MkDocs documentation site
- Pytest suite (unit + integration)
