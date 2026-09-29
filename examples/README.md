# Agribound Examples

Example scripts (`NN_*.py`) and notebook copies (`notebooks/NN_*.ipynb`, same
code) for agribound 1.0.0, plus the HPC scripts in [`hpc/`](hpc/) and the
region definitions in [`regions/`](regions/). Each script's docstring states
its data, assumptions, caveats and prerequisites; read it before running.

## Prerequisites

1. An environment with the extras the example needs (listed in each script's
   docstring). From a clone of the repository:

    ```bash
    conda env create -f environment.yml        # core: agribound[all,dev] (FTW, no Prithvi)
    conda activate agribound
    # Prithvi (examples 03, 12): a separate environment
    conda env create -f environment-gfm.yml    # agribound[all-gfm,dev] (no FTW)
    ```

2. For Earth Engine examples, authenticate once:

    ```bash
    earthengine authenticate
    agribound auth --project YOUR_GEE_PROJECT
    ```

    Scripts take `--gee-project`; the default is `$GEE_PROJECT`, then the
    `gcloud` project, then the `project_id` of the credentials file
    (`$AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY`, else
    `$GOOGLE_APPLICATION_CREDENTIALS`; see
    [GEE setup](../docs/user-guide/gee-setup.md)). No script or region file
    names a project: use your own. The LULC crop filter (on
    by default) also needs Earth Engine. Examples 10, 16 and 17 need no
    Earth Engine; examples 05 (with `--google-backend source_coop`), 20 (with
    `--predicted`) and 21 do not need it either.

3. Run from the repository root, for example
   `python examples/04_france_beauce_sentinel2.py`. Outputs go to
   `outputs/<example>/`. An existing output made with the same settings is
   loaded instead of recomputed; one made with other settings gives a
   `FileExistsError`. Examples 01, 05, 10 and 13 take `--overwrite` to recompute;
   for the other examples, delete the output (or its `outputs/<example>/`
   directory) first.

Notebooks: open them from `examples/notebooks/`; the setup cell changes to
the repository root. They are generated from the scripts by
`tools/sync_notebooks.py`, so changes belong in the script. Command-line options keep their defaults; edit the
configuration cell and set `GEE_PROJECT` (or `gcloud config set project`).

On macOS, scripts that run FTW must keep their `if __name__ == "__main__":`
guard (ftw-tools' data-loader workers use the `spawn` start method).

## Scripts

| # | Script | Region | Source(s) | Engine(s) | What it shows |
|---|---|---|---|---|---|
| 01 | [01_new_mexico_landsat_timeseries.py](01_new_mexico_landsat_timeseries.py) | New Mexico, USA | Landsat 5/7/8/9 | Delineate-Anything (fine-tuned) | Fine-tuning on NMOSE polygons, reusing the checkpoint from the provenance record, per-year evaluation (default 2025 only; `--years 1985-2025` for the full series) |
| 02 | [02_india_ganges_sentinel2.py](02_india_ganges_sentinel2.py) | Nadia, West Bengal, India | Sentinel-2, Google and TESSERA embeddings, SPOT pan | FTW, embedding, Delineate-Anything | Four label-free approaches on one area (SPOT restricted; uses 2020) |
| 03 | [03_australia_murray_darling_hls.py](03_australia_murray_darling_hls.py) | Narrabri, Murray-Darling Basin, Australia | HLS, SPOT 6/7 (restricted) | Prithvi (`embed`, `pca`), Delineate-Anything | The label-free Prithvi modes (land-cover segments, not field instances), compared with Delineate-Anything v2 on SPOT; GFM environment for `embed` |
| 04 | [04_france_beauce_sentinel2.py](04_france_beauce_sentinel2.py) | Beauce, France | Sentinel-2 | FTW | FTW default model with two crop-calendar season windows |
| 05 | [05_pampas_embeddings.py](05_pampas_embeddings.py) | Pergamino, Argentina | Google and TESSERA embeddings | embedding | CPU-only clustering; `source_coop` backend without Earth Engine |
| 06 | [06_kenya_smallholder_ftw.py](06_kenya_smallholder_ftw.py) | Kakamega, Kenya | Sentinel-2 | FTW | Four minimum-area thresholds on smallholder fields |
| 07 | [07_usa_naip_high_res.py](07_usa_naip_high_res.py) | Fresno County, California, USA | NAIP (1 m) | Delineate-Anything | High-resolution, label-free delineation |
| 08 | [08_china_north_plain_spot.py](08_china_north_plain_spot.py) | Hengshui, North China Plain | SPOT 6/7 (restricted) | Delineate-Anything | SPOT composites (uncalibrated DN) |
| 09 | [09_ensemble_comparison.py](09_ensemble_comparison.py) | Carmona, Andalusia, Spain | Sentinel-2 | Delineate-Anything, FTW, ensemble | Intersection, union and vote merges of two engines |
| 10 | [10_local_tif_quickstart.py](10_local_tif_quickstart.py) | any | local GeoTIFF | Delineate-Anything | Minimal run without Earth Engine (`--tif`) |
| 11 | [11_mississippi_alluvial_plain_spot.py](11_mississippi_alluvial_plain_spot.py) | Greenville, Mississippi, USA | SPOT 6/7 (restricted) | Delineate-Anything | 2021-2023 series and year-to-year agreement |
| 12 | [12_new_mexico_ensemble_timeseries.py](12_new_mexico_ensemble_timeseries.py) | Eastern Lea County, New Mexico, USA | S2, Landsat, HLS, NAIP, SPOT (restricted), embeddings | all engines | Per-source multi-model vote ensembles with SAM 2, evaluated against NMOSE (2022; both environments) |
| 13 | [13_sam2_refine_dinov3.py](13_sam2_refine_dinov3.py) | Eastern Lea County | (example 12 output) | SAM refinement | Stand-alone `refine_boundaries` on a finished layer; SAM backends (`--sam-backend`; the SAM 3 backends are untested) |
| 14 | [14_dinov3_sam2_ensemble.py](14_dinov3_sam2_ensemble.py) | Eastern Lea County | S2, Landsat, HLS, NAIP, SPOT (restricted) | DINOv3 (fine-tuned) ± SAM 2 | The pipeline's SAM stage and its size gate on five sources |
| 15 | [15_pampas_semi_supervised.py](15_pampas_semi_supervised.py) | Pergamino, Argentina | embeddings, Sentinel-2, SPOT 6/7 (restricted) | embedding, LULC filter, SAM 2, Delineate-Anything | A label-free chain by hand: cluster → crop filter → SAM on S2 or embedding pseudo-RGB, compared with Delineate-Anything v2 on S2 and SPOT |
| 16 | [16_usa_usgs_naip_plus.py](16_usa_usgs_naip_plus.py) | Fresno County, California, USA | USGS NAIP Plus | Delineate-Anything | The source that needs no Earth Engine (LULC filter off) |
| 17 | [17_query_published_ftw_polygons.py](17_query_published_ftw_polygons.py) | synthetic | local tile store | - | `query_ftw` with the manifest backend, offline |
| 18 | [18_agent_orchestration.py](18_agent_orchestration.py) | Beauce, France | - | agent tools | Read-only tools and `propose_run` without an LLM; `--llm` for a dry-run agent session; MCP configuration |
| 19 | [19_hpc_tiling.py](19_hpc_tiling.py) | Beauce, France | Sentinel-2 | Delineate-Anything | `agribound.hpc` make → stage → delineate → merge locally |
| 20 | [20_stratified_evaluation.py](20_stratified_evaluation.py) | San Juan Basin, New Mexico, USA | Sentinel-2 (or `--predicted`); Landsat, SPOT 6/7 (restricted), NAIP | Delineate-Anything, FTW | Stratified (sub-basin), size-class and boundary evaluation with bootstrap intervals for Delineate-Anything on Sentinel-2 2019; overall object and boundary metrics (no strata, size classes or intervals) for the same engine on Landsat, Sentinel-2, SPOT and NAIP of 2018, and for FTW and Delineate-Anything on Sentinel-2 2019 with and without the crop filter (`--resolution-year`, `--no-resolution`, `--no-lulc-comparison`; `--predicted` skips both comparisons) |
| 21 | [21_published_ftw_audit.py](21_published_ftw_audit.py) | Belen, New Mexico, USA | published FTW polygons | - | `query_ftw` (confidence coverage) and evaluation against NMOSE |

Estimated runtimes are given in each docstring; most were not measured
for 1.0 (the docstrings say which were).

## Notebooks

| # | Notebook |
|---|---|
| 01 | [01_new_mexico_landsat_timeseries.ipynb](notebooks/01_new_mexico_landsat_timeseries.ipynb) |
| 02 | [02_india_ganges_sentinel2.ipynb](notebooks/02_india_ganges_sentinel2.ipynb) |
| 03 | [03_australia_murray_darling_hls.ipynb](notebooks/03_australia_murray_darling_hls.ipynb) |
| 04 | [04_france_beauce_sentinel2.ipynb](notebooks/04_france_beauce_sentinel2.ipynb) |
| 05 | [05_pampas_embeddings.ipynb](notebooks/05_pampas_embeddings.ipynb) |
| 06 | [06_kenya_smallholder_ftw.ipynb](notebooks/06_kenya_smallholder_ftw.ipynb) |
| 07 | [07_usa_naip_high_res.ipynb](notebooks/07_usa_naip_high_res.ipynb) |
| 08 | [08_china_north_plain_spot.ipynb](notebooks/08_china_north_plain_spot.ipynb) |
| 09 | [09_ensemble_comparison.ipynb](notebooks/09_ensemble_comparison.ipynb) |
| 10 | [10_local_tif_quickstart.ipynb](notebooks/10_local_tif_quickstart.ipynb) |
| 11 | [11_mississippi_alluvial_plain_spot.ipynb](notebooks/11_mississippi_alluvial_plain_spot.ipynb) |
| 12 | [12_new_mexico_ensemble_timeseries.ipynb](notebooks/12_new_mexico_ensemble_timeseries.ipynb) |
| 13 | [13_sam2_refine_dinov3.ipynb](notebooks/13_sam2_refine_dinov3.ipynb) |
| 14 | [14_dinov3_sam2_ensemble.ipynb](notebooks/14_dinov3_sam2_ensemble.ipynb) |
| 15 | [15_pampas_semi_supervised.ipynb](notebooks/15_pampas_semi_supervised.ipynb) |
| 16 | [16_usa_usgs_naip_plus.ipynb](notebooks/16_usa_usgs_naip_plus.ipynb) |
| 17 | [17_query_published_ftw_polygons.ipynb](notebooks/17_query_published_ftw_polygons.ipynb) |
| 18 | [18_agent_orchestration.ipynb](notebooks/18_agent_orchestration.ipynb) |
| 19 | [19_hpc_tiling.ipynb](notebooks/19_hpc_tiling.ipynb) |
| 20 | [20_stratified_evaluation.ipynb](notebooks/20_stratified_evaluation.ipynb) |
| 21 | [21_published_ftw_audit.ipynb](notebooks/21_published_ftw_audit.ipynb) |

## HPC and regions

- [`hpc/README.md`](hpc/README.md): Slurm job scripts, NSF ACCESS system
  profiles, Earth Engine throttling and the two-phase stage/compute workflow
  (`agribound tiles`).
- [`regions/README.md`](regions/README.md): 16 region definitions and the
  region driver [`run_region_delineation.sh`](run_region_delineation.sh).

## Notes

- **LULC crop filter.** On by default; it reads Annual NLCD (where at least
  90 % of the area has NLCD data), Dynamic World (2016 to the last complete
  year) or C3S (before 2016) from Earth Engine and removes polygons with a
  crop fraction below 0.3. It is off in examples 05, 10 and 16 (see their
  docstrings). The Earth Engine catalogue notes that Dynamic World crop
  probabilities can be low in arid regions, so the default threshold may
  remove real fields there.
- **Label-free vs fine-tuned.** Delineate-Anything, FTW, the embedding engine
  and Prithvi's `embed`/`pca` modes run without labels. GeoAI and DINOv3 have
  no published field-boundary weights and need fine-tuning on reference
  boundaries (examples 12, 14). FTW cannot be fine-tuned in agribound.
- **Resolution.** Delineate-Anything was trained on 0.25-10 m imagery; 30 m
  Landsat and HLS are outside that range, and FTW is calibrated on
  Sentinel-2 (both are logged and recorded in `engine_meta`). SAM refines only
  fields whose bounding box is at least about 49 pixels on each side (about
  1.5 km at 30 m, 490 m at 10 m, 49 m at 1 m).
- **Ensembles** combine several engines or models on the same raster
  (examples 09, 12); results from different sensors are compared, not merged
  (example 14).
- **Evaluation against NMOSE** in examples 01, 12, 14 is in-sample for the
  fine-tuned engines (they were trained on the same polygons), and NMOSE may
  not contain every field in an area, so predictions of missing fields count
  as false positives. Example 20 shows a stratified evaluation.
- **SPOT 6/7** (examples 02, 03, 08, 11, 12, 14, 15, 20) is restricted to
  select Earth Engine users (internal DRI use). Without access the SPOT runs
  fail with a message: examples 08 and 11, which use only SPOT, produce no
  fields; the others report the failed SPOT run and continue with their other
  sources. External users who need SPOT-based field boundaries can contact
  the package author (sayantan.majumdar@dri.edu).
- **NMOSE reference data** (examples 01, 12, 13, 14, 20, 21) are not included
  in the repository; the scripts expect
  `examples/NMOSE Field Boundaries/WUCB ag polys.shp`. Contact the author for
  access.
- **Large areas.** High-resolution sources (NAIP, SPOT) over large areas
  produce very large rasters; tile them with `agribound tiles` (example 19,
  `hpc/README.md`) rather than running one composite.
