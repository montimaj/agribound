# Agribound Examples

This directory contains example scripts and Jupyter notebooks demonstrating agribound's capabilities across different continents, satellite sources, and delineation engines.

## Prerequisites

1. Install agribound with the required extras for the example you want to run:

```bash
conda create -n agribound python=3.12 rasterio geopandas fiona shapely pyproj -c conda-forge
conda activate agribound
pip install -e ".[all]"
```

2. For GEE-based examples (all except `10_local_tif_quickstart.py`), authenticate with Google Earth Engine:

```bash
gcloud config set project YOUR_GEE_PROJECT
earthengine authenticate
agribound auth --project YOUR_GEE_PROJECT
```

See the [GEE Setup guide](https://montimaj.github.io/agribound/user-guide/gee-setup/) for details.

## Running an Example

### Python Scripts

All GEE-based examples require a `--gee-project` argument:

```bash
python examples/01_new_mexico_landsat_timeseries.py --gee-project YOUR_GEE_PROJECT
```

The local TIF example does not require GEE:

```bash
python examples/10_local_tif_quickstart.py
```

### Jupyter Notebooks

Interactive notebook versions are available in the [`notebooks/`](notebooks/) directory. Set the `GEE_PROJECT` variable in the first code cell of each notebook before running:

```bash
cd examples/notebooks
jupyter lab
```

Outputs (GeoPackage files and HTML maps) are saved to `outputs/<example_name>/`.

## Scripts

| # | Script | Region | Satellite | Engine | Est. Runtime | Description |
|---|--------|--------|-----------|--------|-------------|-------------|
| 01 | `01_new_mexico_landsat_timeseries.py` | New Mexico, USA | Landsat 5--9 | delineate-anything | ~8--12 h | 40-year annual field boundaries (1985--2025). Fine-tunes on NMOSE reference boundaries and evaluates per-year accuracy. Best run on HPC/cloud with GPU. |
| 02 | `02_india_ganges_sentinel2.py` | Ganges Plain, India | Sentinel-2 | ftw | ~30--60 min | Smallholder field delineation using FTW's country-specific model for India. Runs 2020--2024. |
| 03 | `03_australia_murray_darling_hls.py` | Murray-Darling Basin, Australia | HLS | prithvi | ~45--90 min | Large-scale irrigated agriculture using the Prithvi foundation model in embedding mode. Runs 2022--2024. |
| 04 | `04_france_beauce_sentinel2.py` | Beauce, France | Sentinel-2 | geoai | ~15--30 min | European large-field agriculture using geoai's Mask R-CNN. Single year (2023). |
| 05 | `05_pampas_embeddings.py` | Argentine Pampas (Pergamino) | Google + TESSERA | embedding | ~5--10 min | CPU-only unsupervised clustering from pre-computed satellite embeddings (64-D Google, 128-D TESSERA). ~50 km bbox over the Pampas agricultural heartland (2020). |
| 06 | `06_kenya_smallholder_ftw.py` | Central Kenya | Sentinel-2 | ftw | ~10--20 min | Demonstrates `min_field_area` tuning for smallholder agriculture. Compares results at 100, 500, 1000, and 2500 m2 thresholds. |
| 07 | `07_usa_naip_high_res.py` | Central Valley, California, USA | NAIP | delineate-anything | ~20--40 min | 1 m resolution field extraction from NAIP imagery. Large commercial fields. |
| 08 | `08_china_north_plain_spot.py` | North China Plain | SPOT 6/7 | delineate-anything | ~15--30 min | 6 m resolution SPOT imagery. **Restricted access** -- see note below. |
| 09 | `09_ensemble_comparison.py` | Andalusia, Spain | Sentinel-2 | ensemble | ~30--60 min | Runs delineate-anything, FTW, and geoai on the same AOI, then runs the ensemble engine with vote strategy. Visualizes per-engine and consensus results. |
| 10 | `10_local_tif_quickstart.py` | User-provided | Local GeoTIFF | delineate-anything | ~2--5 min | Minimal 5-line quickstart using a local file. No GEE required. Edit `LOCAL_TIF` and `STUDY_AREA` paths before running. |
| 11 | `11_mississippi_alluvial_plain_spot.py` | Mississippi Alluvial Plain, USA | SPOT 6/7 | delineate-anything | ~15--30 min | SPOT-based delineation of row-crop agriculture (2021--2023). Includes cross-year stability analysis using IoU/F1. **Restricted access** -- see note below. |
| 12 | `12_new_mexico_ensemble_timeseries.py` | Eastern Lea County, NM, USA | All (Sentinel-2, Landsat, HLS, NAIP, SPOT, Google & TESSERA embeddings) | All (ensemble) | ~3--6 h | Multi-source, multi-model ensemble (2020--2022) over ~20 km center pivot area with per-model fine-tuning (DA, GeoAI, DINOv3, Prithvi). Grand ensemble boundaries refined by SAM2 after majority-vote merging. Best run on HPC/cloud with GPU. |
| 13 | `13_sam2_refine_dinov3.py` | Lea County, NM, USA | Sentinel-2 | SAM2 refinement | ~5--15 min | Standalone SAM2 boundary refinement on pre-computed DINOv3 field boundaries (555 fields). Crops each field from the raster and refines with SAM2 box prompts. Compares before/after metrics against NMOSE reference. |
| 14 | `14_dinov3_sam2_ensemble.py` | Eastern Lea County, NM, USA | Sentinel-2, Landsat, HLS, NAIP, SPOT | DINOv3 + SAM2 | ~1--2 h | Runs DINOv3 (SAT-493M) across 5 satellite sources with per-source SAM2 refinement. Compares per-source results against NMOSE reference boundaries. Uses a ~20 km bbox over the center pivot area to keep NAIP/SPOT runtimes practical. |
| 15 | `15_pampas_semi_supervised.py` | Pampas (Pergamino), Argentina | Google + TESSERA embeddings + Dynamic World + Sentinel-2 | Embedding + SAM2 (no training) | ~15--30 min | Fully automated pipeline requiring **no reference boundaries or training**. Clusters both Google (64-D) and TESSERA (128-D) embeddings, LULC-filters to crops, then refines with SAM2 on Sentinel-2. 6-way comparison: 2 embeddings × (raw, LULC, SAM2). GPU recommended. |

## Notebooks

Interactive Jupyter notebook versions of each example are in the [`notebooks/`](notebooks/) directory. These are designed for step-by-step exploration with inline map visualization. Set `GEE_PROJECT` in the first code cell before running.

| # | Notebook | Description | Key Difference from Script |
|---|----------|-------------|---------------------------|
| 01 | [`01_new_mexico_landsat_timeseries.ipynb`](notebooks/01_new_mexico_landsat_timeseries.ipynb) | New Mexico Landsat time series with fine-tuning | Runs 2023--2025 (3 years) instead of the full 40-year range, suitable for interactive use |
| 02 | [`02_india_ganges_sentinel2.ipynb`](notebooks/02_india_ganges_sentinel2.ipynb) | India Ganges Plain smallholder fields (FTW) | Same scope as script |
| 03 | [`03_australia_murray_darling_hls.ipynb`](notebooks/03_australia_murray_darling_hls.ipynb) | Australia Murray-Darling Basin (Prithvi + HLS) | Same scope as script |
| 04 | [`04_france_beauce_sentinel2.ipynb`](notebooks/04_france_beauce_sentinel2.ipynb) | France Beauce region (GeoAI Mask R-CNN) | Same scope as script |
| 05 | [`05_pampas_embeddings.ipynb`](notebooks/05_pampas_embeddings.ipynb) | Pampas embeddings (CPU-only, Google + TESSERA) | Same scope as script |
| 06 | [`06_kenya_smallholder_ftw.ipynb`](notebooks/06_kenya_smallholder_ftw.ipynb) | Kenya smallholder `min_area` tuning (FTW) | Same scope as script |
| 07 | [`07_usa_naip_high_res.ipynb`](notebooks/07_usa_naip_high_res.ipynb) | USA Central Valley NAIP 1 m (Delineate-Anything) | Same scope as script |
| 08 | [`08_china_north_plain_spot.ipynb`](notebooks/08_china_north_plain_spot.ipynb) | China North Plain SPOT 6/7 (**restricted**) | Same scope as script |
| 09 | [`09_ensemble_comparison.ipynb`](notebooks/09_ensemble_comparison.ipynb) | Ensemble multi-engine comparison (Andalusia) | Same scope as script |
| 10 | [`10_local_tif_quickstart.ipynb`](notebooks/10_local_tif_quickstart.ipynb) | Local GeoTIFF quickstart (no GEE) | Same scope as script |
| 11 | [`11_mississippi_alluvial_plain_spot.ipynb`](notebooks/11_mississippi_alluvial_plain_spot.ipynb) | Mississippi Alluvial Plain SPOT 6/7 (**restricted**) | Same scope as script |
| 12 | [`12_new_mexico_ensemble_timeseries.ipynb`](notebooks/12_new_mexico_ensemble_timeseries.ipynb) | Lea County multi-source grand ensemble (2020--2022) | Same scope as script |
| 13 | [`13_sam2_refine_dinov3.ipynb`](notebooks/13_sam2_refine_dinov3.ipynb) | SAM2 boundary refinement on DINOv3 output | Same scope as script |
| 14 | [`14_dinov3_sam2_ensemble.ipynb`](notebooks/14_dinov3_sam2_ensemble.ipynb) | DINOv3 + SAM2 multi-source comparison (Eastern Lea County) | Runs single year (2022) instead of 2020--2022 |
| 15 | [`15_pampas_semi_supervised.ipynb`](notebooks/15_pampas_semi_supervised.ipynb) | Embedding + SAM2 (Pampas, no training required) | Same scope as script |

## Runtime Notes

- Estimated runtimes assume a single NVIDIA GPU (e.g., A100/V100) and moderate internet speed for GEE downloads.
- GEE composite generation adds ~2--5 minutes per year per source.
- CPU-only runs (example 05, embedding engine) are 2--5x slower for inference but have no GPU requirement.
- Fine-tuning (examples 01, 12) takes ~30 minutes per model on an Apple M2 Max (MPS). In example 12, DA (2 variants) and GeoAI/Prithvi are fine-tuned on NMOSE reference boundaries (~1.5 hours total). FTW uses pre-trained weights directly (fine-tuning not yet supported — FTW requires paired temporal windows). Fine-tuned checkpoints are cached and reused across years.
- SAM2 boundary refinement (example 12) runs once on the final grand ensemble output per year. Example 14 runs SAM2 per source using each sensor's native raster for accurate per-field segmentation. With the `large` model and per-field cropping, refinement takes ~2--5 minutes per source per year depending on field count.
- **NAIP and SPOT over large areas:** NAIP (1 m) and SPOT (1.5 m) produce rasters that are 100–900x larger in pixel count than Sentinel-2 (10 m) for the same study area. Inference on these high-resolution sources over county-scale or larger areas can take hours even on GPU. Consider subsetting the study area or using `tile_size` to process in chunks. Fine-tuning on NAIP/SPOT is also significantly slower due to the larger training chips.
- **Apple Silicon (MPS):** The GeoAI engine (Mask R-CNN) crashes on MPS due to Metal command buffer errors. Agribound automatically falls back to CPU for GeoAI training and inference. All other engines (FTW, Delineate-Anything, Prithvi) work correctly on MPS.
- The 40-year New Mexico script (01) is best run as an overnight batch job or on HPC. The notebook version runs only 2023--2025.

## LULC Crop Filtering

Agribound automatically filters detected field boundaries to agricultural areas using land-use/land-cover (LULC) data. This is **enabled by default** (`lulc_filter=True`) and removes non-agricultural polygons (roads, water, forest, urban areas, etc.) from the output.

The appropriate LULC dataset is selected automatically based on the study area location and target year:

| Region | Dataset | Years | Resolution | Crop Classes |
|--------|---------|-------|------------|-------------|
| CONUS | USGS Annual NLCD | 1985–2024 (nearest year) | 30 m | 81 (Pasture/Hay), 82 (Cultivated Crops) |
| Global, ≥2015 | Google Dynamic World | 2015–present (nearest year) | 10 m | `crops` probability band |
| Global, <2015 | Copernicus C3S Land Cover | 1992–2022 (nearest year) | 300 m | 10, 20, 30 (Cropland classes) |

**Configuration:**
- `lulc_filter=True` (default) — enable crop filtering
- `lulc_filter=False` — disable (used for local files without GEE, or unsupervised embedding clusters)
- `lulc_crop_threshold=0.3` (default) — minimum fraction of crop pixels to keep a polygon

**Disabled by default for:**
- Example 05 (unsupervised embedding clusters — no semantic meaning)
- Example 10 (local GeoTIFF — no GEE access)

## SPOT Access

Examples 08 and 11 use SPOT 6/7 imagery, which is restricted to select GEE users under a data-sharing agreement. This source is primarily for internal DRI use. If you receive an access error, contact the agribound author (sayantan.majumdar@dri.edu) to request field boundary processing for your study area.

## Recommended Approach: DINOv3 + SAM2 (Example 14)

Based on testing over Lea County, NM, the **DINOv3 + SAM2 multi-source ensemble** (example 14) produces the best results. Key findings:

- **DINOv3 fine-tuned per source** produces cleaner field boundaries than other engines. The ViT backbone adapts well to each sensor's spectral characteristics with just 10--30 epochs of fine-tuning.
- **FTW over-segments** in this region, picking up too many small polygons (roads, pivot edges, noise). FTW is designed for global generalization across 25 countries but tends to be aggressive in arid/irrigated landscapes like southeastern New Mexico.
- **Per-source SAM2 boundary refinement** produces pixel-accurate edges using each sensor's native raster at its own resolution, avoiding resolution mismatches from refining against a single raster. Pre-SAM outputs are saved for comparison.
- **Multi-source diversity** (Sentinel-2, Landsat, HLS, NAIP, SPOT) provides more meaningful ensemble diversity than running multiple model architectures on the same image.

For new study areas with reference boundaries available for fine-tuning, we recommend starting with example 14 (DINOv3 + SAM2 ensemble) rather than the full multi-model ensemble (example 12).

## NMOSE Reference Data

Examples 01 and 12 use NMOSE (New Mexico Office of the State Engineer) WUCB agricultural polygon boundaries located in `examples/NMOSE Field Boundaries/WUCB ag polys.shp` for fine-tuning and evaluation. Example 12 filters to Lea County (County 25). This shapefile is included in the repository.

## Output Structure

Each example creates an output directory under `outputs/`:

```
outputs/
├── new_mexico_timeseries/
│   ├── fields_landsat_1985.gpkg
│   ├── fields_landsat_1986.gpkg
│   ├── ...
│   ├── map_predicted_vs_reference.html
│   ├── map_timeseries_comparison.html
│   └── map_latest.html
├── india_ganges/
│   ├── fields_s2_2020.gpkg
│   ├── ...
│   └── map_ganges.html
└── ...
```

- `.gpkg` files contain field boundary polygons with area, perimeter, and provenance metadata.
- `.html` files are standalone interactive maps (open in any browser) showing field boundaries overlaid on satellite basemaps.
