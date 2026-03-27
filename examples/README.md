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
| 05 | `05_riodelaplata_embeddings.py` | Rio de la Plata, South America | Google + TESSERA | embedding | ~10--20 min | CPU-only unsupervised clustering from pre-computed satellite embeddings. Uses a GEE vector asset as study area input. Compares Google vs TESSERA embeddings for 2020--2024. |
| 06 | `06_kenya_smallholder_ftw.py` | Central Kenya | Sentinel-2 | ftw | ~10--20 min | Demonstrates `min_field_area` tuning for smallholder agriculture. Compares results at 100, 500, 1000, and 2500 m2 thresholds. |
| 07 | `07_usa_naip_high_res.py` | Central Valley, California, USA | NAIP | delineate-anything | ~20--40 min | 1 m resolution field extraction from NAIP imagery. Large commercial fields. |
| 08 | `08_china_north_plain_spot.py` | North China Plain | SPOT 6/7 | delineate-anything | ~15--30 min | 6 m resolution SPOT imagery. **Restricted access** -- see note below. |
| 09 | `09_ensemble_comparison.py` | Andalusia, Spain | Sentinel-2 | ensemble | ~30--60 min | Runs delineate-anything, FTW, and geoai on the same AOI, then runs the ensemble engine with vote strategy. Visualizes per-engine and consensus results. |
| 10 | `10_local_tif_quickstart.py` | User-provided | Local GeoTIFF | delineate-anything | ~2--5 min | Minimal 5-line quickstart using a local file. No GEE required. Edit `LOCAL_TIF` and `STUDY_AREA` paths before running. |
| 11 | `11_mississippi_alluvial_plain_spot.py` | Mississippi Alluvial Plain, USA | SPOT 6/7 | delineate-anything | ~15--30 min | SPOT-based delineation of row-crop agriculture (2021--2023). Includes cross-year stability analysis using IoU/F1. **Restricted access** -- see note below. |
| 12 | `12_new_mexico_ensemble_timeseries.py` | Lea County, NM, USA | All (Sentinel-2, Landsat, HLS, NAIP, SPOT, Google & TESSERA embeddings) | All (ensemble) | ~3--6 h | Multi-source, multi-model ensemble (2020--2022) with per-model fine-tuning on NMOSE references. Expands FTW into 3 EfficientNet models (B3/B5/B7) and DA into both variants. Each model is independently fine-tuned before inference. Merges via majority vote. Best run on HPC/cloud with GPU. |

## Notebooks

Interactive Jupyter notebook versions of each example are in the [`notebooks/`](notebooks/) directory. These are designed for step-by-step exploration with inline map visualization. Set `GEE_PROJECT` in the first code cell before running.

| # | Notebook | Description | Key Difference from Script |
|---|----------|-------------|---------------------------|
| 01 | [`01_new_mexico_landsat_timeseries.ipynb`](notebooks/01_new_mexico_landsat_timeseries.ipynb) | New Mexico Landsat time series with fine-tuning | Runs 2023--2025 (3 years) instead of the full 40-year range, suitable for interactive use |
| 02 | [`02_india_ganges_sentinel2.ipynb`](notebooks/02_india_ganges_sentinel2.ipynb) | India Ganges Plain smallholder fields (FTW) | Same scope as script |
| 03 | [`03_australia_murray_darling_hls.ipynb`](notebooks/03_australia_murray_darling_hls.ipynb) | Australia Murray-Darling Basin (Prithvi + HLS) | Same scope as script |
| 04 | [`04_france_beauce_sentinel2.ipynb`](notebooks/04_france_beauce_sentinel2.ipynb) | France Beauce region (GeoAI Mask R-CNN) | Same scope as script |
| 05 | [`05_riodelaplata_embeddings.ipynb`](notebooks/05_riodelaplata_embeddings.ipynb) | Rio de la Plata embeddings (CPU-only, Google + TESSERA) | Same scope as script |
| 06 | [`06_kenya_smallholder_ftw.ipynb`](notebooks/06_kenya_smallholder_ftw.ipynb) | Kenya smallholder `min_area` tuning (FTW) | Same scope as script |
| 07 | [`07_usa_naip_high_res.ipynb`](notebooks/07_usa_naip_high_res.ipynb) | USA Central Valley NAIP 1 m (Delineate-Anything) | Same scope as script |
| 08 | [`08_china_north_plain_spot.ipynb`](notebooks/08_china_north_plain_spot.ipynb) | China North Plain SPOT 6/7 (**restricted**) | Same scope as script |
| 09 | [`09_ensemble_comparison.ipynb`](notebooks/09_ensemble_comparison.ipynb) | Ensemble multi-engine comparison (Andalusia) | Same scope as script |
| 10 | [`10_local_tif_quickstart.ipynb`](notebooks/10_local_tif_quickstart.ipynb) | Local GeoTIFF quickstart (no GEE) | Same scope as script |
| 11 | [`11_mississippi_alluvial_plain_spot.ipynb`](notebooks/11_mississippi_alluvial_plain_spot.ipynb) | Mississippi Alluvial Plain SPOT 6/7 (**restricted**) | Same scope as script |
| 12 | [`12_new_mexico_ensemble_timeseries.ipynb`](notebooks/12_new_mexico_ensemble_timeseries.ipynb) | Lea County multi-source grand ensemble (2020--2022) | Same scope as script |

## Runtime Notes

- Estimated runtimes assume a single NVIDIA GPU (e.g., A100/V100) and moderate internet speed for GEE downloads.
- GEE composite generation adds ~2--5 minutes per year per source.
- CPU-only runs (example 05, embedding engine) are 2--5x slower for inference but have no GPU requirement.
- The 40-year New Mexico script (01) is best run as an overnight batch job or on HPC. The notebook version runs only 2023--2025.

## SPOT Access

Examples 08 and 11 use SPOT 6/7 imagery, which is restricted to select GEE users under a data-sharing agreement. This source is primarily for internal DRI use. If you receive an access error, contact the agribound author (sayantan.majumdar@dri.edu) to request field boundary processing for your study area.

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
