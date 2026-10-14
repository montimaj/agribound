# Agribound Examples

Example scripts (`NN_*.py`) and notebook copies (`notebooks/NN_*.ipynb`, same
code) for agribound 1.1.0, plus the HPC scripts in [`hpc/`](hpc/) and the
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
   `outputs/<example>/`. An existing output is loaded instead of recomputed
   when its provenance record matches the run's configuration, study area,
   results versions and, for TESSERA, the state of the TESSERA store;
   otherwise it is refused with a `FileExistsError`. That
   includes outputs of earlier code whose results for the same settings
   differ: 1.0.0 outputs of the embedding engine or of SAM refinement;
   outputs of agribound 1.0.1 or earlier made by Delineate-Anything with the
   native backend, the default (examples 02, 03, 07-12, 15, 16, 19, 20, 22 and
   23), by DINOv3 (examples 12, 14 and 23) or by Delineate-Anything
   fine-tuning (examples 01 and 12); Delineate-Anything fine-tunes made
   with recipe version 2 (example 23); and, when the TESSERA store can be
   reached, outputs made from TESSERA embeddings (examples 02, 05, 12, 15,
   23 and 24) before agribound recorded the state of the TESSERA store, or
   from store objects that have changed since (the cached TESSERA rasters
   are read again by themselves; see
   [Reproducibility](../docs/user-guide/reproducibility.md#output-reuse)).
   See the CHANGELOG, [Unreleased]:
   [Results produced with agribound 1.0.1 that are affected](../CHANGELOG.md#results-produced-with-agribound-101-that-are-affected).
   Examples 07, 10, 16 and 19, the fine-tuning run of example 01 and the main
   run of example 20 then stop with the error; the others print that the run
   failed (for most, `<run> failed: FileExistsError: ...`) and go on with
   their other runs (example 09 then makes no merges). Examples 01, 05, 10,
   13, 22, 23, 24, 25, 26, 27 and 28 take `--overwrite` to recompute; for the other
   examples, delete the output (or its `outputs/<example>/` directory) first.

Notebooks: open them from `examples/notebooks/`; the setup cell changes to
the repository root. They are generated from the scripts by
`tools/sync_notebooks.py`, so changes belong in the script. Command-line options keep their defaults; edit the
configuration cell and set `GEE_PROJECT` (or `gcloud config set project`).

On macOS, scripts that run FTW must keep their `if __name__ == "__main__":`
guard (ftw-tools' data-loader workers use the `spawn` start method).

## Scripts

| # | Script | Region | Source(s) | Engine(s) | What it shows |
|---|---|---|---|---|---|
| 01 | [01_new_mexico_landsat_timeseries.py](01_new_mexico_landsat_timeseries.py) | New Mexico, USA | Landsat 5/7/8/9 | Delineate-Anything (fine-tuned) | Fine-tuning on the DRI/NMOSE field boundaries (V18) and the 2024 composite, reusing the checkpoint from the provenance record, per-year evaluation (default 2025 only, a year held out from the fine-tuning; `--years 1985-2025` for the full series) |
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
| 12 | [12_new_mexico_ensemble_timeseries.py](12_new_mexico_ensemble_timeseries.py) | Eastern Lea County, New Mexico, USA | S2, Landsat, HLS, NAIP, SPOT (restricted), embeddings | all engines | Per-source multi-model vote ensembles with SAM 2, evaluated against the DRI/NMOSE field boundaries (V18) (2024; NAIP 2022, SPOT 2023; both environments) |
| 13 | [13_sam2_refine_dinov3.py](13_sam2_refine_dinov3.py) | Eastern Lea County | (example 12 output) | SAM refinement | Stand-alone `refine_boundaries` on a finished layer; SAM backends (`--sam-backend`; the SAM 3 backends are untested) |
| 14 | [14_dinov3_sam2_ensemble.py](14_dinov3_sam2_ensemble.py) | Eastern Lea County | S2, Landsat, HLS, NAIP, SPOT (restricted) | DINOv3 (fine-tuned) ± SAM 2 | The pipeline's SAM stage and its size gate on five sources |
| 15 | [15_pampas_semi_supervised.py](15_pampas_semi_supervised.py) | Pergamino, Argentina | embeddings, Sentinel-2, SPOT 6/7 (restricted) | embedding, LULC filter, SAM 2, Delineate-Anything | A label-free chain by hand: cluster → crop filter → SAM on S2 or embedding pseudo-RGB (each on every polygon, and in a variant that keeps parts over 50 ha unrefined), compared with Delineate-Anything v2 on S2 and SPOT |
| 16 | [16_usa_usgs_naip_plus.py](16_usa_usgs_naip_plus.py) | Fresno County, California, USA | USGS NAIP Plus | Delineate-Anything | The source that needs no Earth Engine (LULC filter off) |
| 17 | [17_query_published_ftw_polygons.py](17_query_published_ftw_polygons.py) | synthetic | local tile store | - | `query_ftw` with the manifest backend, offline |
| 18 | [18_agent_orchestration.py](18_agent_orchestration.py) | Beauce, France | - | agent tools | Read-only tools and `propose_run` without an LLM; `--llm` for a dry-run agent session; MCP configuration |
| 19 | [19_hpc_tiling.py](19_hpc_tiling.py) | Beauce, France | Sentinel-2 | Delineate-Anything | `agribound.hpc` make → stage → delineate → merge locally |
| 20 | [20_stratified_evaluation.py](20_stratified_evaluation.py) | San Juan Basin, New Mexico, USA | Sentinel-2 (or `--predicted`); Landsat, SPOT 6/7 (restricted), NAIP | Delineate-Anything, FTW | Stratified (sub-basin), size-class and boundary evaluation against the DRI/NMOSE field boundaries (V18) with bootstrap intervals for Delineate-Anything on Sentinel-2 2024; overall object and boundary metrics (no strata, size classes or intervals) for the same engine on Landsat, Sentinel-2, SPOT and NAIP of 2022 (one year for all four, so that only the resolution differs), and for FTW and Delineate-Anything on Sentinel-2 2024 with and without the crop filter (`--resolution-year`, `--no-resolution`, `--no-lulc-comparison`; `--predicted` skips both comparisons) |
| 21 | [21_published_ftw_audit.py](21_published_ftw_audit.py) | Belen, New Mexico, USA | published FTW polygons | - | `query_ftw` (confidence coverage) and evaluation of the 2024 polygons against the DRI/NMOSE field boundaries (V18), both layers without polygons under 2,500 m² |
| 22 | [22_global_south_spot_pan.py](22_global_south_spot_pan.py) | India, China, Argentina, Kenya, Egypt, Brazil | SPOT 6/7 panchromatic (restricted) | Delineate-Anything | Six study areas at 1.5 m (3 km squares; 6 km for the centre pivots), from smallholder paddies to centre pivots; crop filter as a separate step |
| 23 | [23_tree_crops.py](23_tree_crops.py) | Ghana, Papua New Guinea, California (USA), Spain | SPOT 6/7 panchromatic (restricted), NAIP, Landsat 8/9 panchromatic, Sentinel-2, Sentinel-1, Google and TESSERA embeddings | Delineate-Anything (released and fine-tuned), DINOv3 (fine-tuned), SAM 2, FTW, embedding | Tree crops (oil palm estate and smallholders, almond and pistachio orchards, olive groves) evaluated against RSPO, DWR / Land IQ and SIGPAC references; Delineate-Anything and DINOv3 fine-tuned on the same labels near each site; the default crop filter and `lulc_tree_crops` as a separate step; `--s2-single-date` (off by default) adds Delineate-Anything on the Sentinel-2 composite of each area's clearest acquisition of the year, chosen by `agribound.survey_scenes` |
| 24 | [24_nevada_field_boundaries.py](24_nevada_field_boundaries.py) | Diamond Valley and Mason Valley, Nevada, USA | NAIP (2022), SPOT 6/7 (2023; restricted), Sentinel-2, Landsat 8/9, Landsat 8/9 panchromatic, Sentinel-1, Google and TESSERA embeddings (2024) | Delineate-Anything (released and fine-tuned), DINOv3 (fine-tuned), SAM 2, opt-in SAM 2.1 automatic masks (`sam-auto`), FTW, embedding | Centre pivots (Diamond Valley) and flood- and sprinkler-irrigated fields (Mason Valley) evaluated against the DRI field boundaries (Nevada, 2026-08-20) in one 5 km square per basin chosen by a fixed rule; Delineate-Anything and DINOv3 fine-tuned on NAIP and SPOT of a separate square at least 5 km away (held out); recall per irrigation system, field size and irrigation status in the imagery year; an irrigated-only score (the fields irrigated in the imagery year, with the predictions on the other fields ignored); the crop filter as a separate step; `--options` adds the "-opt" runs (the gallery runs again with the opt-in options of 1.1.0 that passed the pre-registered test of the Diamond Valley options, run on three pivot squares in other Nevada basins: 3 m grid, half-band growth, pivot snap and crop filter, each where it applies; reported results, not tests, with paired 1 km block intervals of the change); `--sam-auto` adds SAM 2.1 automatic masks on NAIP, SPOT 6/7 and Landsat 8/9; `--dry-run` prints the squares and every configuration |
| 25 | [25_multistate_openet_field_boundaries.py](25_multistate_openet_field_boundaries.py) | 47 US states (the conterminous US except Alabama) | NAIP (each square's latest complete year in Earth Engine, 2021-2023), Sentinel-2 (2024) | Delineate-Anything; opt-in crop-sequence and SAM 2.1 automatic masks (`sam-auto`) | One 5 km square per state, the cell with the most farmed fields (CDL 2024 crop, hay or pasture classes) chosen by a fixed rule (`--select` re-applies it), evaluated against the OpenET agricultural field boundaries v2.1.1 (USDA FSA Common Land Units, state and federal agency fields, and DRI's field delineation in Oregon; local data, not yet public), and against the DRI field boundaries in New Mexico (V18) and Nevada (2026-08-20); a per-state table and `summary.csv` with the years and farmed fields; `--only` takes states or Census regions; `--crop-sequence` adds crop-history zones from CDL 2016-2023 (`cdl-sequence`, `crop-sequence`; CPU, no model), scored against the same references by the same rules; `--sam-auto` adds SAM 2.1 automatic masks (`sam-auto`) on the same NAIP and Sentinel-2 composites, with rows in `summary_sam_auto.csv` (these squares were its pre-registered evaluation: on NAIP F1 0.017 higher than Delineate-Anything's on the 46 squares without California, on Sentinel-2 it did not transfer); `--dry-run` prints the squares and every configuration |
| 26 | [26_sar_field_boundaries.py](26_sar_field_boundaries.py) | UC Davis (California), Texas A&M (Texas), Intel Ohio One (Ohio), Orleans County (Vermont), Centerfield (Utah) and Diamond Valley (Nevada), USA; Sevilla, Spain | Sentinel-1, Umbra and Capella open SAR (2022-2025), NAIP (2021-2023), Sentinel-2; optional UAVSAR L-band quad-pol (2018) | Delineate-Anything | What agribound does with radar today, framed as capability limits: Delineate-Anything v2 as released on Sentinel-1 and on Umbra or Capella composites (read at 3 m without tile merging), beside NAIP with the default and with the same settings and Sentinel-2, on the areas fixed by the Umbra prototype, the Capella pilot and the held-out test (three sites held out) and on example 24's Diamond Valley square, against DWR / Land IQ, OpenET (local data, not yet public), SIGPAC, Utah WRLU and DRI Nevada (local data) references; a per-site table with merge and split rates and `summary.csv` with the collects; `--uavsar` adds a UAVSAR Pauli composite (Earthdata login); `--dry-run` prints the areas, the reference counts, every configuration and the cached collect selection |
| 27 | [27_quebec_crop_sequence.py](27_quebec_crop_sequence.py) | Bas-Saint-Laurent and Montérégie, Quebec, Canada | AAFC Annual Crop Inventory 2016-2023 (`aci-sequence`), Sentinel-2 (2023) | crop-sequence, Delineate-Anything | What eight yearly crop-type maps give where no field map exists: crop-history zones (not field outlines; CPU, no model) scored against the parcels declared to La Financière agricole du Québec for 2023 (BDPPAD, CC BY 4.0, downloaded once by the script), beside Delineate-Anything v2 on Sentinel-2 and the chance baselines of `evaluate(..., baselines=...)`; merge and split rates, the share of adjacent parcels with the same ACI history, and `attach_crop_history` on the Delineate-Anything outlines; `--dry-run` downloads the reference and prints the squares and every configuration |
| 28 | [28_new_york_field_boundaries.py](28_new_york_field_boundaries.py) | Black Dirt (Orange County), Finger Lakes (Cayuga County) and North Country (Jefferson County), New York, USA | NAIP (2022), Sentinel-2 (2024); opt-in SPOT 6/7 (2021; restricted), Sentinel-1, Landsat 8/9 panchromatic, Google and TESSERA embeddings (2024) | Delineate-Anything (released and fine-tuned), DINOv3 (fine-tuned), SAM 2, SAM 2.1 automatic masks (`sam-auto`), FTW, embedding | Muck strips, dairy rotations and hay evaluated against the OpenET agricultural field boundaries v2.1.1 (in New York all USDA FSA Common Land Units; local data, not yet public) in one 5 km square per region chosen by a fixed rule (`--select` re-applies it), with example 24's held-out protocol (Delineate-Anything and DINOv3 fine-tuned on NAIP of a square at least 5 km away; in the Black Dirt also on a square 1 km away, labelled near-sample); recall per land use, strip shape and field size, merge and split rates, chance baselines and paired F1 intervals against the released model (field and 1 km block bootstrap), with the pre-registered claim and gallery-panel rules applied by the script; `sam-auto` runs on NAIP by default (`--no-sam-auto` drops it), and `--sam-auto-sentinel2` adds it on Sentinel-2, where it did not transfer in its pre-registered evaluation (never a gallery panel); the Black Dirt's released runs reproduce example 25's New York scores (`--pipeline-check`); the crop filter as a separate step; `--dry-run` prints the squares and every configuration |

Each docstring gives the example's runtime as measured in the examples' runs
for agribound 1.1 (2026-10-09 to 2026-10-13, on an Apple M2 Max with MPS; the
Hugging Face and torch hub weights already cached, and the composites
downloaded unless the docstring says that they were reused), and names the
run. Example 01 was not run end to end, and of example 12 only the NAIP runs
were. Example 28's runtime is that of its own run of 2026-10-12/13 with
agribound 1.1.0 (3.1 hours, with `--sam-auto-sentinel2`; `--spot` and
`--extended` were not run). Example 24's `--options --sam-auto` was measured
in its run of 2026-10-13 (5.3 minutes with the other outputs in place), and
example 25's `--sam-auto` in the delineate steps of the SAM evaluation's runs
on its squares (about 84 s per NAIP square and 7 s per Sentinel-2 square).

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
| 22 | [22_global_south_spot_pan.ipynb](notebooks/22_global_south_spot_pan.ipynb) |
| 23 | [23_tree_crops.ipynb](notebooks/23_tree_crops.ipynb) |
| 24 | [24_nevada_field_boundaries.ipynb](notebooks/24_nevada_field_boundaries.ipynb) |
| 25 | [25_multistate_openet_field_boundaries.ipynb](notebooks/25_multistate_openet_field_boundaries.ipynb) |
| 26 | [26_sar_field_boundaries.ipynb](notebooks/26_sar_field_boundaries.ipynb) |
| 27 | [27_quebec_crop_sequence.ipynb](notebooks/27_quebec_crop_sequence.ipynb) |
| 28 | [28_new_york_field_boundaries.ipynb](notebooks/28_new_york_field_boundaries.ipynb) |

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
  crop fraction below 0.3. It is off in examples 05, 10, 16, 25, 26 and 27
  (see their docstrings); examples 22, 23, 24 and 28, and example 15 for its
  embedding polygons, apply it as a separate step. Dynamic World
  files plantations and many orchards under `trees`, so where the filter
  uses it (outside the conterminous US) it can remove tree crops;
  `lulc_tree_crops=True` counts `crops` + `trees` instead (example 23). The
  Earth Engine catalogue notes that Dynamic World crop probabilities can be
  low in arid regions, so the default threshold may remove real fields
  there.
- **Label-free vs fine-tuned.** Delineate-Anything, FTW, the embedding engine,
  `sam-auto` (SAM 2.1 automatic masks; opt-in) and Prithvi's `embed`/`pca`
  modes run without labels. GeoAI and DINOv3 have
  no published field-boundary weights and need fine-tuning on reference
  boundaries (examples 12, 14, 23, 24 and 28). FTW cannot be fine-tuned in agribound.
- **Resolution.** Delineate-Anything was trained on 0.25-10 m imagery; 30 m
  Landsat and HLS are outside that range, and FTW is calibrated on
  Sentinel-2 (both are logged and recorded in `engine_meta`). SAM refines only
  fields whose bounding box is at least about 49 pixels on each side (about
  1.5 km at 30 m, 490 m at 10 m, 49 m at 1 m).
- **Ensembles** combine several engines or models on the same raster
  (examples 09, 12); results from different sensors are compared, not merged
  (example 14).
- **Evaluation against the DRI/NMOSE field boundaries (V18)** in examples
  01, 12, 14 is in-sample for the fine-tuned engines (they were trained on the
  same polygons; example 01's default year, 2025, is held out from its 2024
  fine-tuning in time only), and the reference may not contain every field in
  an area, so predictions of missing fields count as false positives. Example
  20 shows a stratified evaluation. Examples 12, 14 and 20 evaluate 2024
  imagery, or the latest year of a source without 2024 imagery there (NAIP
  2022; SPOT 6/7 2023 in Lea County), so NAIP and SPOT are 1-2 years older
  than the other sources. Example 20's resolution comparison uses 2022 for all
  four sources, so that its rows differ in resolution only (the only 2022 SPOT
  6/7 scene has valid data over 76.7 % of its box).
- **Minimum area.** agribound drops predicted polygons smaller than 2,500 m²
  (`min_area`, the default), so the examples that evaluate (01, 12, 13, 14,
  20, 21, 23, 24, 25, 26, 27 and 28) also drop the reference polygons smaller than that, and
  print how many: of the V18 polygons, 1 of 389 in the Lea County box, 350 of
  1,705 in the San Juan box and 152 of 797 near Belen (example 21, which also
  drops the published FTW polygons under 2,500 m²; `--min-area 0` keeps
  both). Example 01 fine-tunes on all 85,279 V18 polygons and evaluates
  against the 68,357 of them that are at least 2,500 m². Example 24 drops
  none of the 74 and 187 DRI fields of its evaluation squares, and example 25
  0 to 37 fields per square (none in Rhode Island and Texas, 37 in Iowa), and
  171 of the 1,000 V18 polygons of its New Mexico square (its docstring lists
  them). Example 28 drops 3, 4 and 2 of the CLUs of its three evaluation
  squares.
- **Evaluation against the DRI field boundaries (Nevada, 2026-08-20)** in
  example 24 is held out: its fine-tuned models learn from a separate square
  of each basin, at least 5 km from the evaluated one. It follows the same
  year rule (2024; NAIP 2022, SPOT 6/7 2023).
- **SPOT 6/7** (examples 02, 03, 08, 11, 12, 14, 15, 20, 22, 23, 24, 28) is restricted to
  select Earth Engine users (internal DRI use). Without access the SPOT runs
  fail with a message: examples 08, 11 and 22, which use only SPOT, produce no
  fields; examples 02, 03, 12, 14, 15 and 20 report the failed SPOT run and
  continue with their other sources. Example 23 checks access first and skips
  its SPOT runs, SAM 2 refinement and fine-tuning (all on SPOT); the rest
  runs. Examples 24 and 28 (with `--spot`) check access first and skip their
  SPOT runs and SPOT fine-tunings. External users who need SPOT-based field boundaries can contact
  the package author (sayantan.majumdar@dri.edu).
- **DRI/NMOSE field boundaries (V18)** (examples 01, 12, 13, 14, 20, 21, 25) are
  not included in the repository; the scripts read
  `examples/NM_Field_Boundaries.zip` in place. The New Mexico region presets
  need the shapefile extracted to `examples/NM_Field_Boundaries/` (see
  [`regions/README.md`](regions/README.md)). Contact the author for access.
- **DRI field boundaries (Nevada, 2026-08-20)** (examples 24, 25 and 26) are
  not included in the repository; the scripts read
  `examples/NV_Field_Boundaries.zip` in place. Figures and documentation may
  show their outlines and where the squares lie, with the attribution in the
  scripts' docstrings. Contact the author for access.
- **OpenET agricultural field boundaries v2.1.1** (examples 25, 26 and 28; Desert
  Research Institute for the OpenET Consortium) are not public yet (they are
  to be released with their data descriptor) and not included in the
  repository; the scripts read a local copy of the per-state zips in place
  (`--openet-dir` or `OFB_DATA_DIR`). Their outlines come from the USDA FSA
  Common Land Units (CLU) in most states, from state and federal agencies
  elsewhere and, in Oregon and Nevada, from DRI; figures may show them, CLU
  fields included, with the attribution in the scripts' docstrings. In New
  Mexico and Nevada example 25 uses the two DRI files above instead (without
  one, it skips the state). The New York file's README says that the licence
  of the release is not final; on 2026-10-12 the author approved showing its
  outlines (all CLUs) in figures, with the attribution in example 28's
  docstring. The file itself is not redistributed.
- **Large areas.** High-resolution sources (NAIP, SPOT) over large areas
  produce very large rasters; tile them with `agribound tiles` (example 19,
  `hpc/README.md`) rather than running one composite.
