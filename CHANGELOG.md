# Changelog

All notable changes to agribound will be documented in this file.

## [Unreleased]

### Results produced with agribound 1.0.1 that are affected

- **Delineate-Anything fine-tuning (`engine="delineate-anything"` with
  `fine_tune=True`).** The training recipe changed (see Fixed and Changed), so
  the fine-tuned weights, and the polygons delineated with them, change. Such
  outputs now raise `FileExistsError` instead of being reused (results
  version `delineate_anything_finetune` 3; outputs made on the main branch
  since 1.0.1 record 2 and are not reused either; pass `overwrite=True`, CLI
  `--overwrite`), and cached fine-tuning runs are trained again (the recipe
  version, now 3, is part of their cache key). Runs that load a fine-tuned
  checkpoint through `engine_params["checkpoint_path"]` are not affected by
  the recipe change, but with the native backend their polygons change with
  the tile stitching (next entry). Agent plans for such runs get a new plan
  directory (it includes the results versions). `engine_params={"yolo_warmup_bias_lr": 0.1, "chip_stride": 512,
  "yolo_nbs": 64, "yolo_sliver_depth_px": 0}` (with `chip_stride` equal to
  the chip size, 512 px below 4 m GSD) restores the 1.0.1 recipe, except on
  rasters with pixels that are 0 in every band, which the chips now treat as
  nodata, and on uint8 rasters with a nodata value other than 0 (see Fixed).
  The GeoAI and Prithvi trainers are unchanged; the DINOv3 trainer changed
  too (below). The Delineate-Anything fine-tunings of the 1.0 gallery were
  made with 1.0.1's recipe (example 12's NAIP runs) and with recipe version
  2, the bias warmup fix only (example 23). The development run of 1.1
  (2026-10-09/10) made them again with recipe 3 and the tile stitching of
  1.1: example 12's NAIP model drew 301 polygons instead of 374, and its
  in-sample F1 rose from 0.37 to 0.48 against V18 (0.44 to 0.56 against the
  NMOSE WUCB polygons); example 23's fine-tuned models scored 0.40 at Twifo
  Praso (0.23 before), 0.45 at Oro (0.21), 0.78 at Madera (0.76) and 0.14 at
  Úbeda (0.008).
- **Delineate-Anything with the native backend (`engine="delineate-anything"`,
  the default backend), including fine-tuned runs, runs that load a
  checkpoint through `engine_params["checkpoint_path"]`, and ensembles with
  such a member (the default members include one).** How the detections of
  overlapping tiles are combined changed (see Changed), so the polygons
  change. Such outputs now raise `FileExistsError` instead of being reused
  (results version `delineate_anything` 2; pass `overwrite=True`, CLI
  `--overwrite`), HPC tiles show as `stale`, and agent plans get a new plan
  directory. Cached native results are recomputed (the backend's
  implementation version, now 4, is part of their cache key).
  `engine_params={"merge_tile_seams": False}` turns off the new seam merge;
  the other two changes cannot be turned off. The `reference` and `ftw`
  backends are unchanged. The outputs of examples 02, 03, 07-12, 15, 16, 19,
  20, 22 and 23 of the 1.0 runs that used the native backend were made
  with the tile stitching of 1.0.0 and 1.0.1. The development run of 1.1
  made them again (of example 12, its NAIP runs; example 20 now runs other
  years): with the released weights on the same input, the numbers of
  polygons fell by 0 to 7 % (example 22's Nile Delta square: 1,465 against
  1,578; example 10: 330 in both), and the gallery and the engines guide
  describe what the stitching still gets wrong.
- **DINOv3 (`engine="dinov3"`), fine-tuned or with a checkpoint given in
  `engine_params["checkpoint_path"]`, and ensembles with a DINOv3 member.**
  agribound now runs DINOv3's sliding window itself and averages the
  predictions of overlapping windows (see Changed), so the polygons change,
  also for a checkpoint fine-tuned by 1.0.1. Such outputs now raise
  `FileExistsError` instead of being reused (results version `dinov3` 2; pass
  `overwrite=True`, CLI `--overwrite`), HPC tiles show as `stale`, and agent
  plans get a new plan directory. Cached segmentations are recomputed (the
  inference version is part of their cache key).
  `engine_params={"window_stitching": "crop"}` gives the 1.0.1 output of a
  1.0.1 checkpoint (at Twifo Praso, example 23, the same segmentation pixel
  for pixel).
- **DINOv3 fine-tuning (`engine="dinov3"` with `fine_tune=True`).** The
  training recipe changed (see Changed), so the fine-tuned weights, and the
  polygons delineated with them, change. Such outputs raise `FileExistsError`
  (results version `dinov3_finetune` 2), and cached fine-tuning runs are
  trained again (the recipe version, now 2, is part of their cache key).
  `engine_params={"class_weights": None, "monitor": "val_loss",
  "input_normalization": "none", "boundary_erosion": 2,
  "window_stitching": "crop", "processing_gsd_m": "native"}` restores the
  1.0.1 recipe and inference (`processing_gsd_m` matters only on rasters
  coarser than 10 m, next entry; with these settings the training loop
  gives the weights of geoai's `train_dinov3_segmentation` exactly, which a
  test checks on a small model). The DINOv3 outputs of the 1.0 gallery
  (examples 12, 14 and 23) were made with 1.0.1's recipe and inference. The
  development run of 1.1 made them again: example 12's NAIP model drew 415
  polygons instead of 195 (in-sample F1 against V18 0.55, against 0.43);
  example 14's models, trained and scored on V18, reached in-sample F1 0.57
  (Sentinel-2), 0.59 (SPOT 6/7) and 0.56 (NAIP) with SAM 2; example 23's
  scored 0.50 at Twifo Praso (0 before), 0.50 at Oro (0.23), 0.91 at Madera
  (0.26) and 0.18 at Úbeda (0.045).
- **DINOv3 fine-tuning on imagery with pixels coarser than 10 m
  (`engine="dinov3"` with `fine_tune=True` and `source="landsat"`, `"hls"`
  or `"landsat-pan"`, `source="naip"` with `naip_resolution_m` above
  10.1 m, or a `local` raster with such pixels).** Fine-tuning
  and inference now read a bilinear copy of the composite with 10 m pixels
  (see Changed), so the chips, the weights and the polygons change. Such
  outputs raise `FileExistsError` instead of being reused (results version
  `dinov3_processing_grid` 2; outputs made on the main branch before this
  change do not record it and are not reused either; pass `overwrite=True`,
  CLI `--overwrite`), HPC tiles show as `stale`, agent plans get a new plan
  directory, and the chips and fine-tuning runs get new cache keys. The
  component applies only where the copy is read: DINOv3 runs on
  Sentinel-1, Sentinel-2, SPOT 6/7, NAIP and other imagery of 10 m or
  finer keep their results versions, configuration hashes and cache keys,
  and so does a run that loads a checkpoint through
  `engine_params["checkpoint_path"]` without `processing_gsd_m` (it reads
  the grid recorded with the checkpoint, none for earlier checkpoints, so
  its output does not change). Setting `processing_gsd_m` in a DINOv3 run
  or ensemble member to a value that gives a grid counts too. Pixel sizes
  are in metres, also in a projected CRS in feet; a `local` raster in a
  geographic CRS counts when a crop of it at any of its latitudes would be
  read on the grid (the pixel width in metres depends on the latitude).
  `engine_params={"processing_gsd_m": "native"}` restores the earlier
  behaviour. Example 14's DINOv3 outputs on Landsat and HLS were made
  without the grid in the development run of 1.1 (in-sample F1 0.014 for
  both, with a 1 px band) and again with it by agribound 1.1.0 on 2026-10-11
  (0.534 and 0.547, with a 4 px band measured on the grid; see Changed).
  Example 12's Landsat and HLS runs have not been run.
- **TESSERA embeddings (`source="tessera-embedding"` with
  `tessera_version="v1"`, the default): an upstream data change.** The TESSERA
  v1 Zarr store was converted again upstream on 2026-10-06/08 with
  geotessera's fix of the tile placement (geotessera issue #429, commit
  ce2d8256fd "zarr: use floor not round when placing tiles", release v0.11.0;
  the fix is in the code that builds the store, and the geotessera 0.10.x that
  agribound requires reads the reconverted store unchanged).
  The earlier conversion had placed about three quarters of the tiles one 10 m
  pixel too far east, too far south, or both: this was the 9-10 m east/south
  offset described for 1.0.0 and 1.0.1. The reconverted store lines up with
  Sentinel-2 to about 1 m in examples 15 and 02
  ([Satellite Sources](https://montimaj.github.io/agribound/user-guide/satellite-sources/#embeddings)).
  The TESSERA results of examples 02, 05 and 15 therefore differ from those of
  1.0.0 and 1.0.1 (measured when they were run again on 2026-10-09). Examples
  23 and 24 read the reconverted store too. The TESSERA rows of example 23 in
  the 1.0 gallery were read on 2026-10-05 (2026-10-06 UTC), before the store
  was rewritten over its areas; the development run of 1.1 read them from
  the rewritten store (Twifo Praso 721 segments before, 703 after; Madera 696
  and 653; Úbeda 618 and 585; 99.60-99.75 % of the pixels moved by one
  pixel), and the run of agribound 1.1.0 on 2026-10-11 made them again,
  because their records had no store state (702, 653 and 585 segments).
  agribound now detects such a rewrite (see Fixed): it records the
  state of the store's objects it reads, and a TESSERA raster cached before
  this change, or read before a rewrite, is read again once (its file is
  replaced) when the store can be reached. An output made from such a raster
  raises `FileExistsError` instead of being reused (pass `overwrite=True`,
  CLI `--overwrite`), HPC tiles show as `stale` (their staged raster and their
  output), and agent proposals made with network access get a new plan
  directory. Without network access, an output is refused when the cached
  raster has been read again since it was made (their store stamps differ);
  otherwise the cached raster and the output are reused with a WARNING.
- **HLS (`source="hls"`): image counts everywhere, and composites within about
  3° of 180°.** HLSS30 images of the MGRS tiles at 180° are no longer counted
  elsewhere, and inside those tiles they are now used (see Fixed). Away from
  180° the composites, and therefore the outputs, are unchanged, but
  `AGRIBOUND_N_IMAGES` falls (example 03: 77 images, not 298; example 14's
  October 2024 composite: 10, not 21; both recorded by agribound 1.1.0 on
  2026-10-11, with example 03's outputs unchanged). Cached HLS
  composites are made again once, and HLS tiles staged on HPC are staged
  again. Outputs are reused (the configuration hash and `RESULTS_VERSIONS`
  are unchanged), except those whose study area lies at the affected tiles:
  they raise `FileExistsError` (pass `overwrite=True`, CLI `--overwrite`).
- **Export grids that reach across 180° (every Earth Engine source).** Such a
  grid (its study area ends at 180° or up to about 10 km short of it) selected
  images all around the globe at its latitude (see Fixed); the image counts
  and years of these composites change, and for `hls` their pixels can too.

### Added

- Example 23 (`examples/23_tree_crops.py`): how agribound does on tree crops,
  in four study areas with reference boundaries: the Twifo Oil Palm
  Plantations estate in Ghana and Higaturu oil palm smallholders in Papua New
  Guinea (RSPO GeoRSPO concession maps, downloaded at run time and pinned by
  SHA-256), almond and pistachio orchards in Madera County, California
  (DWR / Land IQ 2022) and olive groves near Úbeda, Spain (SIGPAC).
  Delineate-Anything v2 on SPOT 6/7 panchromatic (with and without SAM 2), on
  Landsat 8/9 panchromatic, Sentinel-2 and NAIP; Delineate-Anything v2 and
  DINOv3 each fine-tuned on the same labels and SPOT-Pan composite of a
  training area near each study area (another estate for Twifo Praso, other
  parcels of the same Higaturu scheme for Oro, and squares near Madera and
  Úbeda chosen by a fixed rule from the labels and the imagery),
  never on the evaluated squares; FTW; and Google Satellite Embedding and
  TESSERA clusters. The LULC crop filter runs as a separate step, with the
  default rule and with `lulc_tree_crops=True`. The released Delineate-Anything
  v2 has not seen the Ghana and Papua New Guinea fields, but its training data
  (FBIS-73M) cover the Madera square and 87 % of the Úbeda square, with labels
  that match 121 of the 122 DWR / Land IQ reference fields and 166 of the 225
  SIGPAC recintos. The gallery shows the results in five entries. Thanks to
  Jacob Abramowitz (The University of Alabama in Huntsville), who asked about
  tree crops and pointed to the RSPO concession maps and the subdivided Twifo
  estate.
- **Example 23: Sentinel-1 rows.** Each of the four areas gets a run of
  Delineate-Anything v2 as released on its `sentinel1` composite, scored by
  the example's own rules: `max_detections=1000` as for the example's other
  10-15 m sources, no LULC filter. The 1.0.1 outputs have none of these rows;
  the development run of 1.1 (2026-10-10) made them, with the backend of the
  optical rows, and they scored as in the pilot that first scored these runs:
  tree-crop F1 0.80 at Madera (Sentinel-2: 0.80) and 0.06 at Úbeda, and F1_ov
  0.11 at Twifo and 0.00 at Oro. The docstring gives the pilot's caveat:
  Delineate-Anything's training data include Sentinel-1 patches (18 within
  10 km of the Madera square).

  New option `--dry-run`: it prints every run's configuration (validated) and
  runs, downloads and writes nothing.
- **Example 23: `--s2-single-date`.** An opt-in option (off by default; without
  it the runs and their outputs are unchanged) that adds, in each area,
  Delineate-Anything v2 as released (`max_detections=1000`) on the Sentinel-2
  composite of the area's clearest acquisition of the year.
  `agribound.survey_scenes` ranks the acquisitions over the square under the
  area's own Sentinel-2 settings (Cloud Score+ at Twifo Praso and Oro, SCL at
  Madera and Úbeda); the table is saved as `scenes_sentinel2_<year>.csv`, and
  the run, named `sentinel2 da-<date>`, uses a one-day `date_range` on the
  date ranked first and is scored like the other rows. It repeats with the
  released code the pilot in which the single clearest acquisition gave
  Delineate-Anything a lower F1 than the year's median at Twifo Praso (0.097
  against 0.248; IoU ≥ 0.5, precision counted over the predictions that
  overlap a reference polygon, the example's `F1_ov`); on 2026-10-10 the
  survey ranked first the pilot's dates at Twifo Praso, Oro and Madera. Where
  that date has several granules, the run's composite is their median, while
  the pilot used one granule (at Madera, tiles T10SGF and T10SGG against
  T10SGF alone). `--dry-run` prints the run with a placeholder date, because
  the survey is an Earth Engine request. Run with agribound 1.1.0 on
  2026-10-11, the single date scored F1_ov 0.095 against 0.244 for the
  year's median at Twifo Praso and 0 against 0 at Oro, and tree-crop F1
  0.843 against 0.802 at Madera and 0.057 against 0.159 at Úbeda.
- Example 24 (`examples/24_nevada_field_boundaries.py`): Nevada, in two
  contrasting basins of the DRI field boundaries (Nevada, 2026-08-20), the
  field layer of the Nevada Water Inventory (local data, not in this
  repository; read from `examples/NV_Field_Boundaries.zip` in place): centre
  pivots irrigated from groundwater in Diamond Valley and flood- and
  sprinkler-irrigated fields in Mason Valley. Each basin has a 5 km
  evaluation square and a separate training square at least 5 km from it,
  both chosen by a fixed rule from the inventory alone before any model ran
  (the square of UTM zone 11N, with corners on a 1 km grid, holding the most
  DRI fields of the basin by representative point: 74 and 54 fields in
  Diamond Valley's evaluation and training squares, 187 and 161 in Mason
  Valley's); the script records the squares (`BASINS`; corners and centres
  in its docstring) and warns when the inventory gives other squares.
  Delineate-Anything v2 as released on NAIP (2022), SPOT 6/7
  (2023), Sentinel-2, Landsat 8/9, Landsat 8/9 panchromatic and Sentinel-1
  (2024), on NAIP also with SAM 2; Delineate-Anything v2 and DINOv3, each
  fine-tuned on the NAIP and on the SPOT composite of the training square (no
  evaluated field or pixel is used); FTW on Sentinel-2; and Google Satellite
  Embedding and TESSERA clusters (2024). NAIP and SPOT use their latest years
  over the squares, so their imagery is 2 and 1 years older than the other
  sources'; the SPOT, Sentinel-2, Landsat, Landsat PAN and Sentinel-1
  composites cover 1 April to 31 October. Recall is reported per irrigation
  system (`ITYPE_2024`: centre pivot, sprinkler, controlled flood,
  uncontrolled flood or no apparent irrigation equipment, micro-irrigation,
  retired), field-size class and irrigation status in the imagery year
  (IrrMapper), reference fields under 2,500 m² are dropped (none in either
  evaluation square), and the LULC crop filter runs as a separate step.
  `--dry-run` prints the squares, the reference counts and every run's
  configuration without running anything. Figures and documentation may show
  the inventory's outlines and where the squares lie.
- **Example 24: irrigated-only scores.** Every run is also scored against the
  reference fields irrigated in the year of its imagery (IrrMapper share at
  least 0.5: 39, 40 and 34 of Diamond Valley's 74 fields in 2022, 2023 and
  2024; 148, 173 and 159 of Mason Valley's 187), with the other fields of the
  square as `ignore` of `agribound.evaluate` (above): a prediction whose
  representative point lies in one of them is not scored. Every other
  prediction is, so predictions on ground the inventory does not map still
  count as false positives. Matching, tolerance, strata and bootstrap are
  those of the main score. Each run writes `metrics_<output>_irrigated.json`
  beside its metrics file, `summary.csv` gets 13 columns after the existing
  ones (`n_reference_irrigated`, `n_predicted_scored`, `n_predicted_ignored`,
  `precision_irrigated`, `recall_irrigated_score`, `f1_irrigated`, both with
  bootstrap intervals, `iou_mean_irrigated`, `boundary_f1_irrigated`,
  `fp_noref_irrigated`), and the script prints a second table per basin; the
  existing columns, metrics files and tables are unchanged. Scored on
  2026-10-11 on the outputs of the development run of 1.1 (34 of the 36
  outputs reused; the two TESSERA runs made again, with the same polygons,
  because their records predated the TESSERA store stamp; the all-fields
  scores are unchanged): the recall equals that of the
  irrigated stratum in all 28 runs; in Diamond Valley 4 to 158 predictions
  per run are ignored and F1 rises by 0.011 to 0.238 (Delineate-Anything on
  NAIP 0.255 to 0.380, DINOv3 fine-tuned on NAIP 0.380 to 0.565; the 35
  fields not irrigated in 2022 are those that are not centre pivots, and the
  78 and 51 predictions of these two runs on ground the inventory does not
  map stay false positives); in Mason Valley 2 to 49 are ignored and F1
  changes by -0.001 to +0.053. Gallery entries 24c and 24d show four runs of
  each basin with this score.
- Example 25 (`examples/25_multistate_openet_field_boundaries.py`): a simple
  check of Delineate-Anything v2 as released across the conterminous US, in
  one 5 km square in each of the 47 states of the OpenET agricultural field
  boundaries v2.1.1 (Desert Research Institute for the OpenET Consortium;
  every state except Alabama; not public yet, read in place from a local
  copy given by `--openet-dir` or `OFB_DATA_DIR`), against every field of the
  square: USDA Farm Service Agency Common Land Units (CLU; all fields of 35
  squares), the fields of state and federal agencies and, in Oregon, DRI's
  field delineation. In New Mexico and Nevada it uses the DRI field
  boundaries instead (V18, and the Nevada boundaries of 2026-08-20); a state
  whose zip is missing is skipped with a message. Each square was chosen by
  a fixed rule from the field boundaries and the 2024 USDA Cropland Data
  Layer alone, before any model ran (the 5 km UTM grid cell inside the
  state's Census 2023 outline with the most farmed fields: fields of at
  least 2,500 m² whose representative point CDL 2024 maps as a crop, hay or
  pasture class, sampled in Earth Engine; the docstring lists the classes
  and why the OpenET files' own `CROP_2024` is not used), and `--select`
  re-applies it. Delineate-Anything v2 runs on NAIP (each square's latest
  year in Earth Engine with complete cover, 2021-2023) and on Sentinel-2
  (1 April to 31 October 2024), without the LULC crop filter, and is
  evaluated with one-to-one matching at IoU >= 0.5; a per-state table and
  `summary.csv` carry the years, sources and farmed-field counts. `--only`
  takes state codes or the Census regions (`west`, `midwest`, `south`,
  `northeast`), and `--dry-run` prints the squares, the reference counts and
  every configuration without running anything. Delineate-Anything v2's
  training data (FBIS-73M) hold California DWR / Land IQ labels: 128
  labelled training patches of its public lists intersect California's
  square, so its California score is not a held-out test. In Montana and
  North Dakota (NAIP F1 0.01 and 0.04) most reference fields are CLU strips
  about 50 m wide, with the width and orientation of the crop and fallow
  strips of the 2005 NAIP, while by 2023 both squares are farmed in blocks of
  many strips, so few CLU lines show an edge in the image (in Montana 10 % of
  the lines that two fields share, against 5 % by chance; the docstring and
  the gallery give the measurements, made on 2026-10-12 after the scores were
  known; they changed nothing in example 25).
- **Example 25: `--crop-sequence`.** An opt-in option (off by default; the
  default run, its configurations and its outputs are unchanged) adds, in
  every state, crop-history zones from the USDA Cropland Data Layer
  2016-2023 (`source="cdl-sequence"`, `year=2023`, `crop_sequence_years=8`,
  `engine="crop-sequence"`, `lulc_filter=False`;
  `zones_cdl-sequence_2016-2023.gpkg`, named and reported as zones, as in
  example 27), scored against the same reference fields by the same rules,
  with a map layer and a table row (`CDL zones`). A run now replaces the summary rows of the states and runs
  it scores, so a run without the option keeps the zones' rows. Cost: one
  small Earth Engine download and a few CPU seconds per state (4.5 minutes
  for the 47 states, the other runs reused, with agribound 1.1.0 on
  2026-10-11). In that run the zones scored F1 0.004 (Montana) to 0.259
  (Indiana), median 0.080, against a median of 0.314 for Delineate-Anything
  v2 on NAIP.
- Example 26 (`examples/26_sar_field_boundaries.py`): what agribound can do
  with radar today, framed as capability limits. Delineate-Anything v2 as
  released runs on Sentinel-1 (`sentinel1`) and on Umbra or Capella open-data
  composites (`umbra`, `capella`; read with the documented settings
  `processing_gsd_m=3` and `merge_tile_pieces=False`), beside optical controls
  on the same areas (NAIP with Delineate-Anything's defaults and with the same
  settings, and Sentinel-2), at seven sites with field references (public: DWR
  / Land IQ, SIGPAC and Utah WRLU; local data: the OpenET field boundaries
  v2.1.1 until their release, and DRI's Nevada fields): UC Davis (Umbra; DWR /
  Land IQ 2024, provisional), the three held-out sites of the Umbra and
  Capella sources (Texas A&M, Intel Ohio One and Orleans County, Vermont;
  OpenET, read in place from `--openet-dir` or `OFB_DATA_DIR`, and the sites
  are skipped without them), Sevilla (Capella; SIGPAC recintos of the previous
  campaign; Sentinel-2 is the only optical control), Centerfield, Utah (Umbra
  1 m single-look products; Utah WRLU 2025) and example 24's Diamond Valley
  square (Sentinel-1 only; the DRI field boundaries of Nevada). The recipe was
  chosen on the Umbra prototype's four areas (UC Davis, Centerfield, Purdue
  and Mallorca) and the Capella pilot (Sevilla). The areas, windows and
  settings are those fixed by the Umbra prototype, the Capella pilot and the
  held-out test, reused unchanged; Diamond Valley's Sentinel-1, NAIP and
  Sentinel-2 runs repeat example 24's runs on that square, a consistency
  check. The script records the collects that the automatic selection picks
  from the archive index of 2026-10-09 and says when a newer index picks
  others. The docstring lists the caveats (Delineate-Anything v2's training
  data hold Sentinel-1 patches and the DWR and SIGPAC labels; only three sites
  are held out; Centerfield's single-look products; static CLU references;
  references read live from their services, with a warning when SIGPAC's
  previous campaign is no longer 2025; NAIP one to four years from the radar;
  no NAIP at Sevilla) and the earlier scores as context. Scoring is one-to-one
  at IoU >= 0.5 with bootstrap intervals of F1 and merge and split rates,
  against reference fields of at least 2,500 m² (the dropped ones are
  counted); `summary.csv` records the years, windows, images and collect IDs;
  two maps per site. `--uavsar` adds a compact UAVSAR L-band quad-pol Pauli
  composite near UC Davis in 2018 (HTTP range reads with an Earthdata login in
  `~/.netrc`; DWR / Land IQ 2018) as a reference point for full polarimetry.
  `--dry-run` reads the references and prints the areas, the reference counts,
  every configuration and, with a cached archive index, the collects that
  would be selected; it reads that index without rebuilding it and writes
  nothing. Its run of 2026-10-10 with the development code of 1.1 (gallery
  entries 26 and 26b) gave Umbra or Capella F1 0.370, 0.261 and 0.274 at the
  three held-out sites, against 0.406, 0.310 and 0.249 for NAIP read with the
  same settings and 0.282, 0.193 and 0.261 with the defaults, and
  Sentinel-1 0.035 to 0.159 there.
- Example 27 (`examples/27_quebec_crop_sequence.py`): what eight yearly
  crop-type maps give where no field map exists. In two 5 km squares in
  Quebec (Bas-Saint-Laurent, the crop-sequence pilot's square, hay and
  cereals; Montérégie, the next cell by the pilot's rule, soybeans and grain
  corn), crop-history zones from the AAFC Annual Crop Inventory 2016-2023
  (`aci-sequence`, `crop-sequence`) are scored against the parcels with a
  production declared to La Financière agricole du Québec for 2023 (BDPPAD
  v03, CC BY 4.0; the script downloads the 173 MB file once and checks its
  SHA-256), beside Delineate-Anything v2 as released on a Sentinel-2 median
  of April-October 2023 and the chance baselines of
  `evaluate(..., baselines=...)` (a blind grid of each run's median polygon
  size and the shifted reference), with merge and split rates and the share
  of adjacent declared parcels with the same ACI history;
  `attach_crop_history` adds the ACI history to the Delineate-Anything
  outlines. Run with agribound 1.1.0 on 2026-10-11 (1.9 minutes,
  Delineate-Anything on MPS): F1 0.197 and 0.119 for the zones, 0.486 and
  0.472 for Delineate-Anything (0.490 and 0.471 in a run of 2026-10-10 with
  `--device cpu`), 0.007-0.018 for the blind grids; at 60 m every layer had a
  boundary F of 0.85-0.99, the shifted reference included. The docstring
  states the reference's limits (insurance parcels of FADQ clients, not
  exhaustive; parcels of hay-only
  clients not updated since 2010).
- Example 28 (`examples/28_new_york_field_boundaries.py`, notebook and tests):
  three New York landscapes against the OpenET agricultural field boundaries
  v2.1.1 (Desert Research Institute for the OpenET Consortium; not public
  yet, read in place from a local copy given by `--openet-dir` or
  `OFB_DATA_DIR`), whose New York fields are all USDA FSA Common Land Units
  (CLU): the onion, vegetable and sod strips on the drained muck of the Black
  Dirt (Orange County; example 25's New York square), the dairy rotations of
  the Finger Lakes (Cayuga County) and the corn, hay and pasture of the North
  Country (Jefferson County). One 5 km evaluation square per region and one
  training square 5-15 km from it, chosen by fixed rules from the field
  boundaries and the 2024 Cropland Data Layer before any model ran
  (`--select` re-applies them); in the Black Dirt a second training square
  1 km away on the same NAIP flight, reported as near-sample. Delineate-Anything
  v2 as released on NAIP (2022) and Sentinel-2 (2024), on NAIP also with SAM 2
  refinement and with SAM 2.1 automatic masks (`sam-auto`; `--no-sam-auto`
  drops that run); Delineate-Anything v2 and DINOv3 fine-tuned on the NAIP of
  the training square; FTW on Sentinel-2; `--sam-auto-sentinel2`, `--spot`
  and `--extended` add opt-in runs. Recall is reported per land use of the
  imagery year (`CROP_<year>`), strip shape (Polsby-Popper score below 0.3)
  and field size, with merge and split rates, chance baselines and paired F1
  intervals against the released model (2,000 resamples of the reference
  fields and of 1 km blocks). The claim and gallery-panel rules were fixed
  before the run (PREREG-NY, with amendments A1-A3, all before any run: the
  implementation of the paired intervals, the opt-in Sentinel-2 `sam-auto`
  runs, and the author's approval to show the outlines), and the script
  applies them (`better_by_rule`, `above_chance`, `panels.json`; `sample`
  says which rows are near-sample, and is to be quoted beside
  `better_by_rule`). The Black Dirt's two released runs reproduce example
  25's New York scores polygon for polygon (`--pipeline-check`). In the run of
  2026-10-12/13 (agribound 1.1.0; 3.1 hours on an Apple M2 Max with MPS, with
  `--sam-auto-sentinel2`; one recorded deviation, in the driver's check after
  the run, changed no number), Delineate-Anything v2 as released on NAIP scored F1
  0.147 in the Black Dirt, 0.339 in the Finger Lakes and 0.373 in the North
  Country. Fine-tuned on a square 15 and 10 km away, DINOv3 scored 0.433 in
  the Finger Lakes and both models 0.433 and 0.432 in the North Country,
  above the released model by both paired intervals; Delineate-Anything v2
  fine-tuned in the Finger Lakes (0.346) was not clearly different. In the
  Black Dirt, fine-tuned on pasture and forest 5 km away, Delineate-Anything
  v2 scored below the released model (0.085) and DINOv3 no differently
  (0.143); fine-tuned on muck 1 km away (near-sample, not held out) they
  scored 0.305 and 0.280. No Black Dirt run is above chance by the
  pre-registered rule: 305 of its 668 fields are strips, and the reference
  shifted 24.3 m east and north still matches 470 of the 668 (explained after
  the scores were seen). `sam-auto` showed no clear difference from
  Delineate-Anything v2 on NAIP in the three squares (+0.023 to +0.035) and
  scored lower on Sentinel-2 (−0.109 to −0.227); FTW scored lower than
  Delineate-Anything v2 on Sentinel-2 in all three. Figures may show the
  outlines (the author's decision of 2026-10-12). A re-check of FBIS-73M,
  Delineate-Anything v2's training data, on 2026-10-13 (its public lists:
  201,869 US- and Canadian-prefixed patches, located by their GeoTIFF
  headers) found no field label in New York and no patch within 131 km of
  the squares; one Ontario patch reaches across the Niagara River onto 6.5 ha
  of New York shore, where the OpenET file has no field, and none of its
  fields is in New York. Only those patches were located, and the public
  lists were published after training.
- **Example 24: `--options` and `--sam-auto`.** Two opt-in flags (off by
  default; the default run, its configurations and its dry run are unchanged,
  the dry run's output byte for byte). `--options` adds four "-opt" runs per
  basin, each with the opt-in options of 1.1.0 that passed the pre-registered
  test of the Diamond Valley options (PREREG-DV3: three pivot squares in
  other Nevada basins, rules fixed on 2026-10-12, scored on 2026-10-13;
  amended on 2026-10-13 for a defect of its check step that changed no layer
  or score): Delineate-Anything v2 on NAIP with `processing_gsd_m=3` and
  `pivots="snap"`; DINOv3 on NAIP with `dilate_interior_px="half_band"` and
  `pivots="snap"`; on SPOT 6/7, DINOv3 with both (Diamond Valley) or
  Delineate-Anything v2 with `pivots="snap"` (Mason Valley); Sentinel-1 with
  `pivots="snap"` only. The three optical runs are scored on their
  crop-filter layer (the filter is never applied to Sentinel-1), and
  `options.csv` and `metrics_<output>_change.json` give each run's change
  against its base run in F1, irrigated-only F1 and boundary F1, with paired
  1 km-block intervals (`paired_bootstrap`, 2,000 resamples, seed 42). These
  runs combine options that were tested one at a time, and on these squares
  only the pivot snap (and in Mason Valley the 3 m grid) was tested blind, so
  their scores are reported results, not tests. In the run of 2026-10-13
  (agribound 1.1.0, 5.3 minutes, the 36 earlier outputs reused unchanged), in
  Diamond Valley Delineate-Anything v2 on NAIP went from F1 0.255 to 0.422
  (+0.167, 1 km-block interval +0.090 to +0.240), DINOv3 on NAIP from 0.380 to
  0.481 and DINOv3 on SPOT from 0.500 to 0.557 (boundary F1 0.073 to 0.402),
  and Sentinel-1 stayed at 0.346; in Mason Valley the four runs changed F1 by
  +0.110, +0.033, −0.004 and 0. Each "-opt" layer equals the layer the
  pre-registered test made with the same options. `--sam-auto` adds SAM 2.1
  automatic masks on NAIP, SPOT 6/7 and Landsat 8/9, the imagery types its
  pre-registered evaluation kept (not on Sentinel-2, where it did not
  transfer); the six runs repeat that evaluation's Nevada runs exactly, with
  the scores it gave them after its amendment of 2026-10-13 (below). The
  docstring also gives the reference's limits in Diamond Valley: 4 of its 74
  fields cannot be matched by an outline of one piece, so such outlines can
  reach recall 0.95 at most.
- **Example 25: `--sam-auto`.** An opt-in option (off by default; the default
  run and its outputs are unchanged) adds SAM 2.1 automatic masks
  (`engine="sam-auto"`) on the same NAIP and Sentinel-2 composites in every
  state, scored by the same rules; their rows go to `summary_sam_auto.csv`, so
  `summary.csv` keeps one Delineate-Anything row per state and source. These
  squares and composites were the engine's pre-registered evaluation (below):
  on NAIP its F1 was 0.017 higher than Delineate-Anything's on the 46 squares
  without California (95 % interval 0.004 to 0.030), and on Sentinel-2 it did
  not transfer (0.152 lower). Its `delineate` steps there took 3,965 s on NAIP
  and 321 s on Sentinel-2 for the 47 states (about 84 s and 7 s per square;
  Apple M2 Max, MPS). The docstring now gives what FBIS-73M's public lists
  hold near the squares: 128 labelled training patches inside California's
  square, and no patch inside any other.
- **Engine `sam-auto`: SAM 2.1 automatic masks.** Segment Anything 2.1 (large,
  Apache-2.0) automatic mask generation on large overlapping tiles. A score
  partition turns the masks into non-overlapping polygons, with no field
  labels and no fine-tuning. It is opt-in: never an engine default, and the
  agent proposes it only when it is named. Install with the new `sam-auto`
  extra, also in `all` and `all-gfm`: sam2 1.1.0 (a third-party PyPI build
  whose 62 code, configuration and licence files are byte-identical to Meta's
  commit 2b90b9f, pinned because the engine calls its predictor internals),
  OpenCV and the Hugging Face hub client, without segment-geospatial, whose
  patool dependency is GPL-3; CI's engine job installs the extra and checks
  that segment-geospatial is absent. The checkpoint (898 MB) is pinned by
  revision and SHA-256, and `agribound prefetch --engine sam-auto` fetches it.
  - **Settings.** The per-source settings were chosen in-sample on example
    24's two Nevada training squares (by mean F1 among 72 settings per
    source; settings within 0.01 tied, then the fewest SAM passes per km²,
    then boundary F1). They are defaults for NAIP and SPOT 6/7; Sentinel-2
    and Landsat are available with a WARNING that states their measured cost
    and, for Sentinel-2, its result. Sentinel-1 has no default; other sources
    run at native scale with a WARNING.
  - **Evaluation** (PREREG-SAM, fixed on 2026-10-12 before any `sam-auto`
    output existed on its areas; run on 2026-10-13 on an Apple M2 Max with
    MPS). Against Delineate-Anything v2 as released, on the same composites
    and with each example's own scoring, the paired F1 differences were:
    - NAIP: +0.017 (95 % interval +0.004 to +0.030) on 46 of example 25's
      squares (California left out: Delineate-Anything's training data hold
      its DWR labels), so better by the pre-registered rule; both engines
      are near F1 0.3 there.
    - Sentinel-2: −0.152 (−0.178 to −0.127) on the same squares, so on
      Sentinel-2 it **did not transfer**; its setting was not changed or
      tried again.
    - SPOT 6/7: a mean of +0.256 over four Nevada and New Mexico areas, and
      Landsat (8/9; 7/8/9 on the San Juan box) +0.126 over the same areas.
      As pre-registered, these means pool pairs scored without the crop
      filter in Nevada and with it in New Mexico (all four unfiltered: +0.211
      and +0.091, a secondary result). Four areas support no "better" claim.
      Mason Valley's SPOT 1 km-block interval, −0.003 to +0.161, has its
      lower bound at the edge of 0 (seeds 0-19 give −0.003 to +0.004).
    - Amendment of 2026-10-13: Mason Valley's pairs had been scored against a
      crop-filter copy of its reference (176 of its 187 fields); both Nevada
      areas were scored again with the reference after the first scores and
      the endpoints had been read, a recorded deviation from the
      pre-registration. No outcome changed, and Mason Valley's corrected
      values were partly known beforehand.
    - Known and held out: Delineate-Anything's scores on most areas were
      known before the evaluation, from the examples' own runs. The Nevada
      areas lie 5.0 km (Diamond Valley) and 16.0 km (Mason Valley) from the
      squares the settings were chosen on, in the same basins: held out from
      that choice, not from the landscape.
    - Secondary results, which decide nothing: Quebec's declared parcels
      (Sentinel-2, example 27's squares) −0.404 and −0.363; New York (example
      28's own runs, read by an amendment made before any run) NAIP +0.023 to
      +0.035, each 1 km-block interval including 0 (the Black Dirt's lower
      end, −0.0015, at the edge of 0), and Sentinel-2 −0.109 to −0.227;
      against example 24's six NAIP and SPOT "-opt" runs on the same
      composites, −0.119 to +0.083 before their
      crop filter and −0.086 to +0.092 with both layers crop-filtered
      (reported results: both scores of 10 of the 12 pairs had been seen
      before).
  - **Cost.** GPU time on two New Mexico boxes (415 and 496 km²; composite
    grids of about 430 and 526 km²), back to back with Delineate-Anything on
    the same composites: 6.1 times Delineate-Anything's on NAIP, 6.4 on SPOT,
    17.0 on Sentinel-2 and 20.7 on Landsat (3.1 and 2.1 on example 25's 5 km
    squares, against runs of other sessions).
  - **Limits.** SAM outlines non-field land too: on the 46 NAIP squares 21 %
    of its polygons overlap no reference field, against 22 % of
    Delineate-Anything's. It merged neighbouring fields more often than
    Delineate-Anything on SPOT, Sentinel-2 and Landsat in all 12 Nevada and
    New Mexico area pairs, and less often on average on the 46 squares, on
    NAIP (0.41 against 0.53) and on Sentinel-2 (0.46 against 0.53). Objects
    larger than about half a tile are
    dropped. In a pre-registered multi-tile development check (2026-10-12),
    SPOT and Landsat lost 0.09 and 0.11 F1 on one of three squares. Output is
    identical on one device and differs between devices.
  - **Examples.** Example 28 runs it on NAIP (`--no-sam-auto` drops the run)
    and, with `--sam-auto-sentinel2`, on Sentinel-2. Examples 24 and 25 run
    it with `--sam-auto`.
  - **Unchanged.** Other engines' configuration hashes and results versions
    are unchanged (results version `sam_auto` 1). `agribound tiles make`
    refuses it, also as an ensemble member. The configuration checks its
    `engine_params`, also a member's, when it is built, and refuses
    `sam_refine` and the refinement stage's `sam_model` with it. The engines
    guide describes the method, the settings, the evaluation, the limits and
    the costs, and the workflow diagram draws nine engines (the orange
    family's legend now reads "Foundation model": SAM 2.1 is a general
    segmentation model, not a geospatial one).
- **Centre-pivot post-process** (`engine_params["pivots"]`, opt-in;
  `agribound.postprocess.pivots.snap_pivots`). `"snap"` replaces groups of
  polygons whose union is a circle (a centre pivot that an engine cut into
  pieces along crop lines, mosaic seams or its own tiles) by the fitted circle
  and snaps single round polygons; `"merge"` replaces such groups by their
  union; `"off"` (default) does nothing. Squares, rectangles and partial
  pivots are left as they are. The step runs after the post-processing and
  before the LULC crop filter (with `aoi_selection="clip"` its result is
  clipped again), adds an `agribound:pivot` column and the provenance fact
  `pivots`, and takes about 10 ms per polygon. In an ensemble it is set on the
  ensemble and refused in a member's `engine_params`. Outputs without the
  option, their configuration hashes and results versions are unchanged. The
  rule and its constants were fixed in a pre-registered pilot on development
  squares (2026-10-12: F1 +0.000 to +0.011 in Diamond Valley, no change in
  Mason Valley, +0.044 on a Lea County square), whose script the module
  reproduces exactly (114 of 114 layers). In the confirmatory test
  (PREREG-DV3; three fresh Nevada pivot squares and example 24's two
  evaluation squares, the four gallery run types of each; rules fixed on
  2026-10-12, scored on 2026-10-13; amended on 2026-10-13 for a defect of its
  check step that changed no layer or score), F1 fell on none of the 20 runs
  and rose on 6, where the step rejoined pivots cut into pieces
  (Delineate-Anything v2 on NAIP +0.003 to +0.018, Sentinel-1 +0.005 and
  +0.015); boundary F1 changed by −0.015 to +0.030 (lower on inset DINOv3
  outlines). In Mason Valley it changed F1 in none of the four runs (it
  snapped 2 polygons in each Delineate-Anything run; boundary F1 −0.0003 on
  SPOT 6/7).
- **DINOv3 growth from the predicted band**
  (`engine_params["dilate_interior_px"] = "half_band"`, opt-in;
  `agribound.engines.dinov3.half_band_dilate_px`). Each field interior grows
  by g = ceil(T/2) px, T the median thickness of the boundary band the model
  predicts where a field meets background, measured on each prediction,
  instead of the band width the checkpoint was trained on. `engine_meta`
  records `dilate_interior_rule`, `band_thickness_px`, `band_edge_px`,
  `dilate_fallback` and `trained_dilate_px` (only with the option). The
  default growth, outputs, configuration hashes and results versions are
  unchanged; Prithvi refuses the value, and the configuration checks it when
  it is built. A pre-registered development pilot (2026-10-12) chose it
  (boundary F1 up on 8 of 9 held-out runs, F1 −0.003 to +0.033) over a
  width-matched rule by the number of runs it improved, although the
  width-matched rule placed the held-out SPOT outlines closer (boundary F1
  0.27 and 0.28 against 0.24 and 0.25). In the confirmatory test (PREREG-DV3,
  as above: three fresh Nevada squares, with example 24's Diamond Valley
  models applied outside their basin), boundary F1 rose on 5 of 6 runs, the
  fewest its rule allowed: on SPOT 6/7 by +0.20 to +0.25 (g 10–13 px), on
  NAIP by −0.003 to +0.017 (g 5–7 px), with F1 changes of −0.002 to +0.052.
  On example 24's Diamond Valley SPOT raster it gives boundary F1 0.37
  against 0.07 (F1 0.55 against 0.50), a result known before the test from an
  earlier sweep of the growth on that raster. It overshoots where the
  predicted boundary class covers the ground between fields and falls short
  where the band is narrow.
- **Paired comparison of two layers.** `agribound.evaluate.paired_bootstrap(predicted_a,
  predicted_b, reference, ...)` scores two prediction layers against one
  reference with the rules of `evaluate()`, and reports the differences A − B
  of F1, precision and recall (and boundary F1 with `boundary_tolerance_m`).
  Each difference has a 95 % percentile interval from resamples that both
  layers share (`n_resamples`, default 2,000; `seed`, default 42).
  - By default it resamples reference fields, as `evaluate(...,
    bootstrap=n, bootstrap_seed=seed)` does, so each layer's own intervals
    are those `evaluate()` reports. Predictions that overlap no reference
    field are then fixed in every resample, which makes the intervals too
    narrow where many lie on ground the reference does not map.
  - With `block_size_m` (for example 1,000) it resamples square blocks
    instead. Their edges are at multiples of the side in `block_crs`
    (default: the UTM zone of the layers' centre). Each block brings the
    reference fields and the predictions of both layers whose representative
    point lies in it, so those false positives are resampled too.
  - It takes `iou_threshold`, `matching`, `strata`, `boundary_tolerance_m`,
    `equal_area_crs` and `ignore`. Strata are resampled separately with
    fields, and only group the results with blocks.
  - `evaluate()` and every existing output are unchanged; the golden test
    passes. On 11 pairs of layers in three development squares, the per-block
    counts equal those of an independent scorer written for the 1.1 review.
  - Examples 24 and 28 and the SAM evaluation report its 1 km-block
    intervals. An end near 0 depends on the random draws: Mason Valley's SPOT
    pair of the SAM evaluation had a lower bound of −0.003 at seed 42, and
    −0.003 to +0.004 at seeds 0-19 (the evaluation guide says so).
- `tools/fetch_naip_history.py`: exports one square of a year's NAIP from
  Earth Engine, RGB-only years included (agribound's `naip` source reads
  4-band images only). It mosaics the year's images, latest on top, and
  averages them to the square's grid. It writes the GeoTIFF with the tags of
  a composite and a JSON record of the export: asset IDs, dates, bands,
  export settings and SHA-256. Earth Engine's averages can differ by 1 in a
  few pixels between exports (14 and 19 of the 6.25 million pixels of the
  Montana square in three exports), so the record pins the file. Gallery
  entry 25c draws its 2005 image of example 25's Montana square.
- `engine_params["merge_tile_seams"]` (Delineate-Anything, native backend):
  join tile-cut pieces that meet along a tile edge (see Changed). It defaults
  to True when `resolve_overlaps` is True and `tile_step` is 0.5, which it
  needs, and to False otherwise; True with either of them changed raises
  `ValueError`. The native backend's `engine_meta` records
  `merge_tile_seams`, `tile_seam_params` (the thresholds), `n_seam_joins`
  and `tile_edge_strip_px` (4).
- `engine_params["processing_gsd_m"]` and `engine_params["stretch_percentiles"]`
  (Delineate-Anything, native backend), for fields larger than the inference
  tiles. `processing_gsd_m` reads the raster at a coarser pixel size, so a
  512 px tile covers more ground (1536 m at 3 m, against 768 m on 1.5 m
  SPOT-Pan): the raster is resampled once by average resampling and cached
  in the working directory, the polygons stay in the raster's CRS, and values
  outside the 0.25-10 m training range or finer than the raster's pixels
  raise `ValueError`. `stretch_percentiles` sets the percentiles of the scene
  stretch (default `[1, 99]`, upstream's; uint8 rasters are not stretched).
  `engine_meta` records `processing_gsd_m`, `processing` (the resampled copy)
  and `stretch_percentiles`. Delineate-Anything fine-tuning cuts its chips
  from the same resampled raster with the same stretch, and the training
  metadata records the input the chips gave the model: pixel size (`gsd_m`),
  super-resolution factor, `raster_dtype` and stretch percentiles. A run that
  gives a checkpoint another input (pixel sizes more than 1 % apart, for
  example a checkpoint fine-tuned on 1.5 m chips run on a 3 m raster, another
  factor, or another stretch, where a uint8 raster counts as not stretched)
  logs a WARNING and records both in
  `engine_meta["checkpoint_input_mismatch"]`. The options themselves are not
  compared: `processing_gsd_m` equal to the raster's pixel size, or
  `stretch_percentiles` on a uint8 raster, changes nothing the model sees.
  Checkpoints fine-tuned with agribound 1.0.1 are checked too (their metadata
  records the pixel size, the factor and the stretch bounds, from which a
  uint8 input is recognised). When a resampled copy is read, its version is
  part of the cache key of the native results and of the fine-tuning chips
  made from it, so a new version of the copy is not served results or chips
  made from the old one.
  Neither option is set by default, and runs that do not set them keep their
  results, cache keys and configuration hash. With the released weights on
  the cached SPOT-Pan composite of Twifo Praso (example 23), the default and
  both options (`processing_gsd_m=3`, `stretch_percentiles=[0.1, 99.9]`) gave
  122 and 134 polygons, recall at IoU 0.5 0.49 and 0.71 (27 and 39 of the 55
  blocks), precision among the polygons that overlap a block 0.23 and 0.32
  (0.22 and 0.29 over all polygons), F1 0.32 and 0.44, split rate 0.44 and
  0.35, merge rate 0.05 in both, and 12.7 % and 2.7 % of the boundary length
  within 2 m of an edge of the run's own tiles (2.1 % and 1.0 % by chance).
  Each option alone gave recall 0.56 (3 m) and 0.65 (stretch), with merge
  rates of 0.18 and 0.20. The two values were picked while investigating the
  tile-edge lines at this site, so these numbers are not an independent test
  of them.
- `engine_params["yolo_warmup_bias_lr"]` (Delineate-Anything fine-tuning): the
  warmup learning rate of the bias parameters (default 0; see Fixed).
- `engine_params["chip_stride"]` (fine-tuning chips, every fine-tunable
  engine): the number of pixels between the origins of neighbouring training
  chips, from 1 to the chip size. The default is half the chip size for
  Delineate-Anything (see Changed) and the chip size for GeoAI, DINOv3 and
  Prithvi, whose chips do not change. The train/validation split is now made
  on chip-size cells, so chips that overlap never put the same pixel in both
  splits (see Changed); for chips that do not overlap the split is the same
  as before. `chips_meta.json` records `chip_stride`, and for overlapping
  chips the number of chips and cells in each split and of dropped chips.
- `engine_params["yolo_nbs"]` (Delineate-Anything fine-tuning): Ultralytics'
  nominal batch size `nbs`, the number of images per optimizer step. The
  default is the batch size (`yolo_batch`), so gradients are not accumulated;
  Ultralytics' own default is 64 (see Changed).
- `engine_params["yolo_sliver_depth_px"]` (Delineate-Anything fine-tuning):
  label parts that the chip frame cuts to a strip at most this many pixels of
  the model input deep get no label (default 16; 0 keeps them; see Changed).
  The training metadata records it, and the number of dropped slivers per
  split (`dataset.n_train_slivers`, `dataset.n_val_slivers`).
- DINOv3 `engine_params` (see Changed): for fine-tuning, `class_weights`
  (default `"inverse_frequency"`; `None` for an unweighted loss, or three
  numbers), `class_weight_cap` (default 10), `monitor` (default `"val_miou"`;
  or `"val_loss"`) and `input_normalization` (default `"sat493m"`, or
  `"lvd1689m"` when the file name of `weights_path` says so; `"none"`, or a
  dict of `mean` and `std`); for inference, `window_stitching` (default
  `"blend"`; `"crop"` is agribound 1.0.1's rule). The training metadata
  (`<checkpoint>.agribound.json`) records `recipe_version`, `class_weights`,
  `class_weight_rule`, `class_weight_cap`, `class_pixel_counts`, `monitor`,
  `best_epoch`, `epochs_run` and `input_normalization`, and `engine_meta`
  records `input_normalization`, `window_stitching` and `n_windows`.
  `trainer_kwargs` keys that are not Lightning `Trainer` arguments
  (`class_weights`, `loss_fn`, `monitor_metric`, `mode`, `patience`) raise
  `ValueError` naming the engine parameter to use. With agribound 1.0.1,
  `class_weights` and `loss_fn` reached geoai's `train_dinov3_segmentation`,
  and the other three raised `TypeError` (agribound passed them itself).
- Training chips of DINOv3 and Prithvi count the reference fields that have
  no pixel outside the boundary band (`chips_meta.json`:
  `n_reference_fields`, `n_reference_fields_without_interior`): such a field,
  less than 2 × `boundary_erosion` + 1 pixels across, is labelled boundary
  only and cannot be learned as a field. A WARNING is logged when more than
  10 % of the fields have none.
- `agribound.engines.finetune.widest_boundary_erosion(reference,
  raster_path, max_px=4, max_share=0.1)` and `field_interior_depths`: a rule
  on the training labels alone for the boundary band of DINOv3 and Prithvi
  on coarse imagery. Each reference polygon is rasterised on the raster's
  grid as the training masks are, and the function returns the widest
  `boundary_erosion` of at most `max_px` pixels for which at most
  `max_share` of the polygons have no interior pixel (10 %, the share above
  which the training chips log a WARNING; 1 px when no band qualifies), with
  the share for every band. Only the raster's grid is read. With
  `grid_gsd_m` (for example `dinov3_processing_gsd(raster_path)`, also
  exported) the polygons are measured on the grid of DINOv3's processing
  copy instead (see Changed: DINOv3 on rasters coarser than 10 m).
- `landsat-pan`: native 15 m Landsat 7/8/9 B8 TOA imagery through the existing
  source interfaces, with sensor QA masks, scene cloud filtering and RGB
  replication. By default a composite uses one PAN bandpass: Landsat 8/9, or
  Landsat 7 for date windows before 2013-03-18 (see `landsat_pan_missions`)
  (contributed by Jeremy Rapp, The University of Alabama in Huntsville).
- `landsat_pan_missions` (CLI `--landsat-pan-missions`): the missions whose
  PAN band `source="landsat-pan"` uses; the other sources ignore it. `"auto"`
  (default) uses Landsat 8 (`LC08`, from 2013-03-18) and Landsat 9 (`LC09`,
  from 2021-10-31) whenever the date window overlaps their record, else
  Landsat 7 (`LE07`, 1999-05-28 to 2024-01-19), so the two bandpasses are
  never mixed. A list of `LE07`, `LC08` and `LC09` (or a comma-separated
  string) uses exactly those missions; a list with Landsat 7 and Landsat 8 or
  9 mixes the two bandpasses in one median, and a WARNING is logged when
  images of both contribute. For `landsat-pan`, a setting none of whose
  missions has a record overlapping the year or `date_range` (for example
  `LE07` with 2025, or `LC09` with 2020) is rejected when the configuration is
  created (`ValueError`), so `--dry-run`, agent plans and `agribound tiles
  make` refuse it instead of every tile ending as no-data. Agent plans warn
  when it is not `"auto"`.
- `landsat-pan` composite tags `AGRIBOUND_LANDSAT_PAN_MISSIONS` (the setting),
  `AGRIBOUND_MISSIONS_SELECTED` (the missions searched for the window),
  `AGRIBOUND_SENSORS` and `AGRIBOUND_SENSOR_IMAGES` (the missions with images
  after the bounds, date and scene cloud filters, and the number of images of
  each, counted by `SPACECRAFT_ID` in the image-count request) and
  `AGRIBOUND_SPECTRAL_RESPONSE` (the PAN bandpasses of those missions). The
  provenance record keeps them (`facts.composite`), and `tools/make_gallery.py`
  labels a Landsat PAN background from `AGRIBOUND_SENSORS` (for example
  "Landsat 8/9 PAN TOA").
- `sentinel1`: Sentinel-1 C-band SAR composites from the Earth Engine
  collection `COPERNICUS/S1_GRD` (sigma0 in dB), 2015 to present, at 10 m
  (about 20 × 22 m resolution). It uses IW-mode images at 10 m with both VV
  and VH, of one orbit pass: the pass with more acquisitions over the
  study-area extent in the window (`DESCENDING` on a tie). Earth Engine stores
  each acquisition as GRD slices of about 25 s; the slices of one satellite
  and absolute orbit (`platform_number`, `orbitNumber_start`) count as one
  acquisition, so a study area that crosses a slice boundary does not count
  its acquisitions twice. One pass can still combine relative orbits that view
  the area at different incidence angles (there is no incidence-angle
  normalisation). Per image, pixels with VV below -30 dB (border noise and
  invalid edges) are masked, VV − VH (dB) is added as a third band, and VV
  (-25 to 0 dB), VH (-32 to -8 dB) and VV − VH (0 to 15 dB) are mapped to
  1-255; the median, rounded to whole values, is exported as uint8 with nodata
  0 (`value_scale` `"uint8"`; R = VV, G = VH, B = VV_VH). `cloud_cover_max`
  and `s2_cloud_mask` are ignored, and `greenest`/`max_ndvi` are rejected (see
  Changed). Composite tags record the orbit pass and the acquisitions of each
  pass (`AGRIBOUND_ORBIT_PASS`, `AGRIBOUND_ORBIT_PASS_ACQUISITIONS`), the
  satellites with acquisitions and their numbers (`AGRIBOUND_SENSORS`,
  `AGRIBOUND_SENSOR_IMAGES`: `S1A` to `S1D`), the dB ranges
  (`AGRIBOUND_S1_DB_RANGES`, JSON) and the scaling; `AGRIBOUND_N_IMAGES`
  counts acquisitions too, as `AGRIBOUND_IMAGE_COUNTS` notes.
  `agribound.composites.gee.sentinel1_uint8_to_db` converts values back to dB,
  and the reflectance conversions (`agribound.io.raster.to_unit_reflectance`,
  `to_s2_dn`) refuse the source unless `value_scale` is passed. The Sentinel-1
  recipe version is part of the composite cache key, so it can change without
  invalidating other sources' composites. Delineate-Anything, GeoAI and DINOv3
  accept the source, and their source notes say that the effective
  resolution is about 20 m and that fine-tuning is likely needed: for GeoAI
  and DINOv3, SAR backscatter is out of distribution for their optical
  pre-training; the training data of Delineate Anything v2, FBIS-73M, list
  76,448 Sentinel-1 (`*_S1_10m_256_custom`) and 7,852 `*_SYN-S1_256_custom`
  patches among their 1,432,187 (checked 2026-10-08; the published training
  list of FBIS-22M, the data of the v1 models, names none), but how their
  bands were rendered is not documented, and its zero-shot F1 on agribound's
  composites ranged from 0.00 to 0.80 (below). SAM refinement accepts it
  too. FTW and Prithvi do not support it, and the default ensemble members
  are refused. The agent's
  `check_availability(live=True)` counts the acquisitions that pass the
  composite's scene filters, per orbit pass and satellite.
  `recommend_configurations` ranks `sentinel1` candidates after those of every
  optical and embedding source, whatever their resolution. Delineate-Anything v2
  as released (IoU ≥ 0.5, no LULC filter) scored F1 0.18 and 0.39 with default
  settings on Sentinel-1 composites of two 5 km smoke squares (Lea County, New
  Mexico, 2022; near Davis, California, 2024). In a pilot (2026-10-08) on
  example 23's four squares (with `max_detections=1000`), on 20.5 km² near
  Davis that lie inside the Davis smoke square, and on a 415 km² box of
  eastern Lea County that contains the Lea County smoke square, it scored:
  - 0.80 on the tree crops of example 23's Madera square (2022);
  - 0.455 on 20.5 km² near Davis (2024);
  - 0.173 on 415 km² of eastern Lea County (2024);
  - 0.11, 0.06 and 0.00 on example 23's Twifo Praso (2020), Úbeda (2023) and
    Oro (2021) squares.

  The California and Spanish scores are not held out. FBIS-73M uses DWR and
  SIGPAC polygons as labels there. 2 of its Sentinel-1 patches intersect the
  Davis area and about 38 lie within 10 km of it, and 18 lie within 10 km of
  the Madera square. Two settings were also checked:
  - `stretch_percentiles` has no effect on the uint8 composite. A float copy
    stretched at (0.1, 99.9) scored lower in 4 of the 6 pilot areas.
  - `merge_tile_pieces=False` changed F1 by at most 0.005 at 10 m.

  The Delineate-Anything source note, the agent's ranking rule and the agent,
  engines and satellite-sources guides say so. `tools/make_gallery.py` labels
  a Sentinel-1 background with its orbit pass and its number of acquisitions,
  and the workflow diagram lists the source.
- `umbra` and `capella`: open very-high-resolution X-band SAR from the AWS
  Registry of Open Data (the Umbra Open Data Program, `s3://umbra-open-data-catalog`,
  and Capella Space open data, `s3://capella-open-data`; both CC BY 4.0),
  composited locally without Earth Engine by
  `agribound.composites.vhr_sar` (one module with a provider adapter per
  vendor). Amplitude only: Umbra GEC and Capella GEO products; CPHD, SICD,
  SLC, SIDD and CSI are never downloaded. The recipe (`vhr-sar-amplitude-1`)
  was fixed before it was scored, from an Umbra prototype at four sites and a
  Capella pilot (2026-10-08):
  - **Index.** The bucket listing plus the per-collect metadata (Umbra
    `.stac.v2.json` sidecars; the Capella static STAC items plus the head of
    each GEO `_extended.json`), cached with its snapshot time under
    `vhr_sar_cache_dir` (default `~/.cache/agribound/vhr_sar`). Umbra copies are
    merged by collect ID; polarization and geometry come from the sidecar
    (`METADATA.json` disagreed for 36 of 204 Centerfield collects, so folders
    without a sidecar are not used); collects dated after the snapshot (2035,
    2036) are dropped; Capella incidence is read at full precision, with the
    calibration label and scale factor. The index is refreshed when the date
    window ends after an index more than a day old, or with
    `agribound.composites.vhr_sar.refresh_index`. On 2026-10-09: 10,212 Umbra
    collects (35 s to build) and 2,281 Capella GEO collects (17 s).
  - **Selection.** Collects whose footprint covers the study area (98 %; the
    metadata footprints are approximate), grouped by orbit pass, look side and
    polarization (HH and VV are never mixed); the largest subset within 5° of
    incidence and 25° of view azimuth (circular); up to
    `vhr_sar_max_collects` (6) spread over the window; or the collects pinned
    by `vhr_sar_collect_ids`. Deterministic for one index snapshot. On the
    prototype's evaluation areas it picks the prototype's own collects at
    Centerfield and Purdue, the VV pair on Mallorca and the prototype's
    geometry group at UC Davis.
  - **Products and radiometry.** Umbra GEC orthorectified with its RPCs over
    the Copernicus GLO-30 DEM (GLO-90, with a WARNING, for tiles GLO-30 does
    not publish, such as Armenia and Azerbaijan; a tile missing from both is
    sea, with a WARNING) converted to ellipsoidal heights with EGM2008
    (the geotransform, with a WARNING, without RPCs), uncalibrated log-scaled
    DN; Capella GEO (terrain-corrected with a 30 m DSM, which displaces tree
    tops: WARNING) as sigma0 dB + 50, with `vhr_sar_calibration="full"` to
    drop collects whose calibration is limited or partial. Every date is
    level-normalised before the median; no beam flattening.
  - **Co-registration** (`vhr_sar_coregister`, default `"auto"`): dates to a
    reference date, then the stack to an optical base of 1 m or finer
    (`vhr_sar_coregister_base`, else USGS NAIP Plus where it has imagery;
    without a base, dates only, with a WARNING; the base's pixel size is read
    in its CRS unit, so US-survey-foot orthophotos are accepted), with 2 m and
    4 m block fallbacks; each date is warped once with its total shift. A
    registration counts only when a majority of the tile estimates agree
    within a capped radius (at most three tolerances), else the build stops
    with an error: unrelated images are refused at every level. Sentinel-2 is
    never the base. `USGSNAIPPlusCompositeBuilder.build` takes a
    `resolution_m` for this (part of the cache key when given).
  - **Output.** float32 at `vhr_sar_resolution_m` (1 m), 0 = nodata: band
    `MEDIAN`, read as R, G and B (value scales `dn` for Umbra and the new `db`
    for Capella); `vhr_sar_bands="median+dates"` (experimental) adds the mean
    of the closest pair of dates and each date (2 or more collects), and the
    engines then read `{"R": 1, "G": 2, "B": 1}` unless `bands` is given
    (`agribound.config.effective_bands`; derived when read, not stored).
  - **Budget and errors.** `vhr_sar_max_download_gb` (5) caps the downloads
    (cached products do not count); downloads resume only from a partial file
    of the same ETag, are checked against the size and ETag of the index
    (`If-Match`), are cached per collect and copy, and hold a lock, so
    processes that share the cache download a product or DEM once. Clear
    errors when nothing covers the study area or window (with the nearest
    dates), when fewer than 2 consistent dates exist (`vhr_sar_max_collects=1`
    builds a single date, with a WARNING: in the Umbra prototype single dates
    scored F1 0.00-0.08 with Delineate-Anything's default settings; read at
    2 m without tile merging, 0.15-0.40 against 0.22-0.42 for the median at
    three of the four sites, and about 0 at the fourth), when the study area
    exceeds one scene, and when the DEM or the co-registration fails.
  - **Provenance.** `AGRIBOUND_ATTRIBUTION` (Umbra: "SAR data: Umbra Lab,
    Inc., CC BY 4.0"; Capella: "© Capella Space Corp, All Rights Reserved."),
    `AGRIBOUND_LICENSE`, `AGRIBOUND_LICENSE_URL`, `AGRIBOUND_DATA_CITATION`
    (the AWS registry's citation with the access date), `AGRIBOUND_CHANGES`,
    `AGRIBOUND_SAR_COLLECTS` (one JSON entry per collect, with its key, ETag,
    applied shift and registration quality), `AGRIBOUND_INDEX_SNAPSHOT` and
    a JSON manifest; the cache key holds the recipe, the settings (with the
    base file's path, size and modification time), the sorted collect IDs
    with their ETags and the snapshot.

  New configuration fields `vhr_sar_collect_ids`, `vhr_sar_max_collects`,
  `vhr_sar_polarization`, `vhr_sar_resolution_m`, `vhr_sar_coregister`,
  `vhr_sar_coregister_base`, `vhr_sar_bands`, `vhr_sar_calibration`,
  `vhr_sar_max_download_gb` and `vhr_sar_cache_dir` (CLI `--vhr-sar-*`)
  apply only to these sources and enter the configuration hash only for them
  (the download cap and the cache directory never), so the hashes, tile
  manifests and agent plan directories of other configurations are
  unchanged. Delineate-Anything, GeoAI and DINOv3 accept the sources; FTW,
  Prithvi and the default ensemble refuse them. Delineate-Anything reads them
  best with `engine_params={"processing_gsd_m": 3, "merge_tile_pieces": False}`
  (documented; not an engine default): in the Umbra prototype this fixed
  recipe scored F1 0.439, 0.163, 0.257 and 0.360 near UC Davis, at
  Centerfield, near Purdue and on Mallorca (orthophotos 0.426, 0.360, 0.240,
  0.460; one date with default settings 0.008, 0.000, 0.081, 0.065), and
  Capella GEO near Sevilla 0.50-0.54; none of these is held out. In a
  held-out test of the fixed recipe, written down before any build or score
  (OpenET field boundaries v2.1.1 of at least 2,500 m²), F1 was 0.390 at the
  Texas A&M AgriLife farm (Umbra, 6 VV dates of 2024; 136 fields), 0.261 near
  the Intel Ohio site (Umbra, 4 HH dates of 2024; 77 fields) and 0.274 in
  Orleans County, VT (Capella, 2 HH dates of 2022; 112 fields), against 0.284,
  0.193 and 0.261 for NAIP + Delineate-Anything with its defaults and 0.403,
  0.310 and 0.249 with the SAR settings: parity at all three sites by the
  rule fixed in advance, at low absolute scores; Capella's in-sample 0.50-0.54
  near Sevilla did not carry over. `vhr_sar_bands="median+dates"` gave no
  consistent gain there (0.378 and 0.303 against 0.390 and 0.261). The agent
  considers the sources only when they are named in
  `recommend_configurations(sources=...)` and then proposes these two
  settings; `check_availability(live=True)` lists the covering collects;
  `propose_run` reserves `vhr_sar_cache_dir` and adds the licence and credit
  to the plan warnings; `plan_network_services` lists the AWS buckets, the
  DEM and geoid grid, and USGS NAIP Plus. The gallery tool prints the vendor
  credit, the licence (when the credit does not name it) and "modified" under
  maps drawn on these composites. A rebuild of the prototype's UC
  Davis composite agreed with it (date-to-date shifts within 0.1 m;
  r = 0.994 at 3 m once the 2.2 m difference between the two NAIP 2022
  bases is removed), and the Sevilla composite with the pilot's (r = 0.993 at
  1 m, mean difference -0.05 dB). The satellite-sources, configuration,
  engines, agent and reproducibility guides describe the sources, and the
  workflow diagram lists them.
- **The embedding engine clusters local feature rasters** (`engine="embedding"`,
  `source="local"`), for example a stack of SAR backscatter in dB,
  polarimetric entropy and alpha angle, or temporal statistics, one
  real-valued feature per band. `engine_params["standardize"]` scales each
  band before PCA and clustering: `"none"` (default), `"zscore"` (mean and
  standard deviation of the sampled pixels) or `"robust"` (median and
  interquartile range / 1.349). Without it, the band with the largest numeric
  spread decides the clusters. In a synthetic test (25 square fields of 1.44
  ha on a road grid at 10 m; two feature bands in small units carry the
  classes, and a third band in large units is noise), the run without
  standardisation scored F1 0.00 and the z-scored run 0.98: every field, plus
  the road grid as one extra polygon. `"robust"` failed the same test (F1 at
  most 0.13 over four seeds): the two-valued class bands get a larger robust
  scale than their standard deviation, and so less weight than the noise
  band (the engines guide says so). The statistics come from the seeded
  pixel sample that fits PCA and the clusterer, and are recorded in
  `engine_meta["standardize"]`. Inputs are checked:
  - complex pixels (for example SAR SLC) are refused;
  - integer bands are read as float32;
  - a pixel of a feature raster is skipped when any band is NaN or equals the
    nodata value (an embedding pixel, as before, only when every band does);
  - a local raster clustered without standardisation whose band standard
    deviations differ by more than 10 times logs a WARNING.

  Existing configurations are unchanged. `standardize="none"` is kept out of
  the cluster cache key and out of `engine_meta["params"]`, so these stay as
  they were in 1.0.1:
  - the configuration hashes (a test pins those of example 23's k = 20 runs);
  - the cluster cache entries;
  - the `embedding` results version (2).

  Two side effects of accepting `local`:
  - `agribound prefetch --engine embedding` now defaults to `source="local"`.
    It needs no GEE project, and prefetch reads no raster.
  - A region file of `examples/run_region_delineation.sh` that lists both the
    `local` source and the `embedding` engine now runs that pair. No region
    file in `examples/regions/` lists `local`.

  The change starts from the tooling patch of the SAR study (2026-10-08).
- `lulc_tree_crops` (CLI `--lulc-tree-crops/--no-lulc-tree-crops`, default
  False): count tree cover as crop in the LULC filter, for orchards and
  plantations. Dynamic World files plantations under `trees` (Brown et al.
  2022, Table 1: "Plantations such as apples, bananas, citrus, and rubber"),
  so the default filter can remove tree crops: with Dynamic World it kept 0
  (2020) and 1 (2023) of 95 oil-palm blocks at Twifo Praso, Ghana (RSPO
  GeoRSPO concession boundaries). With the option, the Dynamic World value is
  the annual median of the per-image sum of the `crops` and `trees`
  probabilities, and C3S also counts its tree-cover classes (50, 60-62, 70-72,
  80-82, 90); all 95 blocks were kept for both years (C3S 2015 kept all 95
  with and without the option). With Dynamic World or C3S the filter then no
  longer removes forest. NLCD and CDL are unchanged (NLCD class 82 already
  includes "perennial woody crops such as orchards and vineyards"), so with
  them it still removes forest. `lulc_stats["tree_crops"]` records the
  setting. Dynamic World and C3S rasters made with the option are cached
  separately and tagged `AGRIBOUND_LULC_TREE_CROPS=True`; other LULC rasters
  are tagged `False`, including the NLCD and CDL rasters that runs with and
  without the option share; a raster cached by 1.0.0 or 1.0.1, which is still
  reused, has no tag. Agent plans warn when it is set, and, when it is
  not, where the filter uses Dynamic World or C3S (`lulc_dataset`
  `"dynamic_world"` or `"c3s"`, or `"auto"` for a study area outside the
  conterminous-US envelope); `recommend_configurations` notes the same for
  such study areas, and the agent's system prompt lists tree crops among the
  limitations to report. Thanks to Jacob Abramowitz (The University of
  Alabama in Huntsville) for asking about tree crops and pointing to the RSPO
  GeoRSPO concession boundaries.
- **Scene survey and single-date composites.** `agribound scenes` (Python:
  `agribound.survey_scenes(config)`) lists the acquisitions of a composite's
  date window over the study area (the images of one UTC date that pass
  `cloud_cover_max`), ranked by the share of the study area each shows clear
  under the composite's per-pixel mask (Sentinel-2 ties go to the higher mean
  Cloud Score+ `cs_cdf`), with the number of images and their asset IDs, the
  scene cloud cover, the footprint share and, for Sentinel-2, the SCL and
  Cloud Score+ clear shares and the mean aerosol optical thickness. One Earth
  Engine request, or one per month when Earth Engine runs out of memory or
  time on the whole window (the images selected for the whole window, so the
  same rows; example 23's Sentinel-2 surveys at Madera, 2022, 177 images on
  96 dates, and Úbeda, 2023, needed it); nothing is written unless `--csv` is
  given (`--json` prints JSON). Sources:
  `sentinel2`, `landsat`, `landsat-pan` and `hls` (ranked by
  clear share; `hls` with the composite's footprint rules), `spot` and
  `spot-pan` (no pixel mask: the share their scenes cover); the registry
  lists them in `SCENE_SURVEY_SOURCES` and `SCENE_RANKED_SOURCES`. A one-day
  `date_range` (start = end) with the default `median` builds the composite of
  one acquisition. In a pilot at four tree-crop squares, the single clearest
  Sentinel-2 acquisition gave Delineate-Anything a lower F1 at IoU ≥ 0.5 than
  the default median at Twifo Praso, Ghana (0.097 against 0.248) and
  Chenderiang, Malaysia (0.107 against 0.163), with precision counted over
  the predictions that overlap a reference polygon, because the RSPO maps do
  not cover every field; see "Dates in a composite" in the satellite-sources
  guide.
- **Image dates of Earth Engine composites.** Composites built from now on
  list their images in the GeoTIFF tag `AGRIBOUND_IMAGE_LIST` (`[asset ID,
  UTC date, scene cloud %]`, at most 1,000, sorted by date;
  `AGRIBOUND_IMAGE_LIST_TRUNCATED` says how many were left out), with
  `AGRIBOUND_IMAGE_DATES` (distinct dates of all images),
  `AGRIBOUND_IMAGE_DATE_FIRST` and `AGRIBOUND_IMAGE_DATE_LAST`; the
  provenance record copies them. They come from the existing image-count
  request. Pixels, cache keys and configuration hashes do not change; cached
  composites are reused with their old tags.
- **Date layers.** `agribound composite --date-layers` (Python:
  `agribound.build_date_layers(config)`) downloads, next to the cached
  composite, a uint16 GeoTIFF on its grid (`<source>_date_layers_<key>.tif`)
  with the per-pixel number of valid observations and the first and last
  observation date (days since 1970-01-01; for `greenest`, the date of the
  observation used). One image-count request and one download; never a band
  of the composite. Same sources as the survey, with `export_method="local"`.
- **Chance baselines.** `agribound.evaluate.chance_baselines(predicted,
  reference, ...)` scores, with the rules of `evaluate()`, two layers that
  use no image: a blind square grid whose cells have the predictions' median
  area, laid over the study area (or the predictions' bounding box) in its
  UTM zone, as the mean of 36 grids offset by sixths of a cell
  (`footprint=True` cuts the cells to the predictions), and, with `shift_m`,
  the reference moved east and north (`"auto"`: half the median width of its
  polygons). `evaluate(..., baselines=True)`, or a dict of these options,
  adds them under `metrics["baselines"]` with the call's own rules; where the
  layers give none (no evaluated prediction without `cell_side_m` and
  `study_area`, or a shift without reference polygons) it keeps its metrics
  and reports `{"skipped": <reason>}` with a WARNING. Without it,
  `evaluate()` and every existing metric, default and output are unchanged;
  the pipeline computes no baselines, and `agribound evaluate` only with
  `--baselines` (below). New
  constants `BLIND_GRID_OFFSET_STEPS` (6) and `BLIND_GRID_MAX_CELLS`
  (1,000,000). The reason: in two pilots on
  small-field sites, boundary F at loose tolerances gave layers that use no
  image about the scores of real outputs. On ten US squares with median
  fields of 0.9-5.2 ha, a blind grid had a boundary F at 60 m of 0.65-0.90,
  at least that of the published USDA NASS Crop Sequence Boundaries on 6 of
  them, while its object F1 was 0.001-0.048. With the pilots' cell sides the
  function gives the counts and scores of that pilot's blind grid on all ten
  squares and of the SNIC test's 900 blind grids.
- **Ignoring predictions on excluded ground.** `agribound.evaluate.evaluate(...,
  ignore=polygons)` (also `evaluate_frame` and `chance_baselines`) leaves out
  of the score the predictions whose representative point
  (`shapely.point_on_surface` of the repaired prediction in the predictions'
  CRS, tested in the CRS of `ignore`, boundary included) lies in one of the
  polygons, typically reference fields left out of the reference passed,
  such as fields not irrigated in the year of the imagery. They are removed
  before matching, so they are neither true nor false positives, and counted
  in the new key `count_predicted_ignored`; the other keys are computed
  without them (`count_predicted` counts the scored predictions), except
  `count_predicted_dropped` and `count_predicted_repaired`, which describe
  the layer passed, as without `ignore` (a repaired prediction is counted
  whether it was scored or removed). Every other prediction is scored as
  before: one on ground in neither layer is still a false positive. With
  `baselines`, the baseline layers are scored with the same `ignore`.
  Without `ignore`, every default, metric and output is unchanged (a golden
  test, `tests/unit/test_evaluate_golden.py`, compares the full results on
  fixed layouts with those of the code before the change).
- **Crop-sequence zones from annual crop-type maps** (opt-in; sources
  `cdl-sequence` and `aci-sequence`, engine `crop-sequence`). The sources
  download, on Earth Engine, the USDA NASS Cropland Data Layer
  (`USDA/NASS/CDL`, conterminous US, years from 2008) or the AAFC Annual Crop
  Inventory (`AAFC/ACI`, Canada, years from 2011): `crop_sequence_years`
  maps (default 8; CLI `--crop-sequence-years`) ending in `year`, as one band
  of class codes per year (uint8, nodata 0, value scale `categorical`), on
  the export grid at 30 m with nearest-neighbour resampling, padded by 10
  pixels (CDL is 10 m from 2024: Earth Engine reads those maps from the 20 m
  level of their mode pyramid, so each 30 m pixel takes the most frequent
  class of the 2 × 2 10 m cells of the 20 m cell under its centre). One
  request lists the maps and one download fetches the stack; batch exports
  are refused. The engine turns the stack
  into zones of uniform multi-year crop history, as USDA NASS makes its Crop
  Sequence Boundaries: pixels in an agricultural class in more than half of
  the years, a per-year focal mode with a 3 px radius, developed classes in
  any year as barriers, 8-connected regions of equal sequence, and regions
  under 1 ha merged into their largest neighbour. It runs on the CPU without
  model weights (1,375 km² in 9.6 s) and adds each zone's majority class per
  year (`agribound:crop_<year>`). On the raw stacks of the crop-sequence
  pilot's 11 squares, it reproduces the zones of the pilot's prototype
  exactly (0 m² symmetric difference). The zones are not field instances: in
  the 47 squares of example 25 their object F1 against the reference fields
  was 0.004-0.259 (2016-2023; median 0.080, against 0.314 for
  Delineate-Anything v2 on NAIP), and in example 27's two Quebec squares
  0.119-0.197, against 0.472-0.486 for Delineate-Anything v2 on Sentinel-2
  (agribound 1.1.0, 2026-10-11). Thanks to Jacob T.
  Stid (Michigan State University) and Jeremy Rapp, who shared the
  crop-sequence recipe that this route follows. The other engines refuse
  rasters whose `AGRIBOUND_VALUE_SCALE` is `categorical`, and so do
  ensembles, SAM refinement and fine-tuning; the reflectance conversions
  (which take arrays) refuse the two sources and
  `value_scale="categorical"`. `date_range`, `sam_refine`, batch exports
  and an `export_crs` that is geographic or not in metres are refused for
  the two sources. Stacks of other series run with `source="local"` and
  `engine_params` `value_scale="categorical"` and a `dataset` or class
  lists; with a study area, the local builder crops such a stack for this
  engine with a margin of 10 pixels around the study area's bounding box
  (its own cache-key part), as the two sources pad theirs. New field
  `crop_sequence_years`, which enters the configuration hash and the cache
  key only for the two sources, so the hashes, cache keys and outputs of
  other configurations are unchanged; new extra `crop-sequence`
  (`scikit-image>=0.24`, in `all` and `all-gfm`);
  `agribound.postprocess.regions` (`merge_small_regions`,
  `polygonize_labels`). The agent proposes the route only when the source or
  the engine is named. See
  [Crop-sequence zones](https://montimaj.github.io/agribound/user-guide/engines/#crop-sequence-zones-crop-sequence)
  and [Crop-type map sequences](https://montimaj.github.io/agribound/user-guide/satellite-sources/#crop-type-map-sequences-cdl-sequence-aci-sequence).
- **`agribound.attach_crop_history(fields, stack)`** adds to any polygons,
  for example Delineate-Anything outlines from imagery, the most frequent
  class of each year of a crop-type map stack (`agribound:crop_<year>`, ties
  to the lowest code; `agribound:crop_px`; with `shares=True`, the share of
  the majority class), the way NASS attributes its Crop Sequence Boundaries.
  The outlines and the CRS are not changed.
- **`agribound.query_csb(study_area, year)`** reads the published USDA NASS
  Crop Sequence Boundaries (rev23, public domain; the eight-year windows
  2008-2015 to 2018-2025) of a study area from the release zip on the NASS
  web site in place (`/vsizip//vsicurl/`), or from a local copy
  (`source=`), with `clip`, `columns`, `dst_crs` and `output_path` (and a
  provenance record). A query of the remote zip took 195-462 s on
  2026-10-10.
- **Documentation:** [Satellite Sources](https://montimaj.github.io/agribound/user-guide/satellite-sources/#landsat-panchromatic-landsat-pan)
  describes the `landsat-pan` mission rule and composite tags,
  [Sentinel-1](https://montimaj.github.io/agribound/user-guide/satellite-sources/#sentinel-1-sentinel1)
  the `sentinel1` scene filters, scaling, tags and caveats, and
  [Tree crops](https://montimaj.github.io/agribound/user-guide/satellite-sources/#tree-crops)
  the new LULC option, its measurements and its trade-off.
  [Fine-Tuning](https://montimaj.github.io/agribound/user-guide/fine-tuning/)
  describes the Delineate-Anything recipe changes (the bias warmup,
  overlapping chips, their cost and their split by chip-size cells,
  `yolo_nbs`, slivers, the nodata rule of the chips), the tile-frame bias
  they address, and recommends a smaller `yolo_lr0` for small training
  sets. Its
  [engine-specific settings](https://montimaj.github.io/agribound/user-guide/fine-tuning/#engine-specific-settings)
  describe the DINOv3 recipe, how its default boundary width was chosen and
  the Twifo Praso runs, and [DINOv3](https://montimaj.github.io/agribound/user-guide/engines/#dinov3-dinov3)
  in the engines guide the input normalisation and the blended windows.
  [Fields larger than the tiles](https://montimaj.github.io/agribound/user-guide/engines/#fields-larger-than-the-tiles)
  describes `processing_gsd_m` and `stretch_percentiles`, with the Twifo
  Praso runs.
  [Dates in a composite](https://montimaj.github.io/agribound/user-guide/satellite-sources/#dates-in-a-composite)
  describes what a composite records about its dates, single-date
  composites, and what a pilot at four tree-crop squares measured (scores,
  date seams, haze, QA60 and the scale of SNIC in Earth Engine scripts). Its
  last part,
  [Superpixel segmentation (SNIC, SLIC)](https://montimaj.github.io/agribound/user-guide/satellite-sources/#superpixel-segmentation-snic-slic),
  reports a test of SNIC and SLIC on the Sentinel-2 median as an engine for
  oil palm estate blocks. The test ran at four RSPO estates in Ghana, Papua
  New Guinea, Peru and Guatemala, with a rule fixed in advance. No setting
  passed, so agribound has no superpixel engine. Thanks to Jacob Abramowitz
  (The University of Alabama in Huntsville), who raised the question of date
  mixing in tropical composites and shared his SNIC script, and to Jacob T.
  Stid (Michigan State University) and Jeremy Rapp, who shared the
  crop-sequence recipe that the crop-sequence route follows.
  [Chance baselines](https://montimaj.github.io/agribound/user-guide/evaluation/#chance-baselines)
  in the evaluation guide describes the blind grid and the shifted reference,
  with the pilots' numbers and the boundary precision a layer gets by chance
  among square fields.
  [SAM 2.1 automatic masks](https://montimaj.github.io/agribound/user-guide/engines/#sam-21-automatic-masks-sam-auto)
  in the engines guide gives that engine's method, settings, pre-registered
  evaluation, limits and costs, and its DINOv3 section the half-band growth
  and its test; the configuration guide's
  [Post-processing](https://montimaj.github.io/agribound/user-guide/configuration/#post-processing)
  the centre-pivot step and its test; the evaluation guide example 24's
  options, example 28 and how an interval end near 0 moves with the random
  draws; the HPC guide the engine that cannot be tiled; and the installation
  guide the `sam-auto` extra.
- `tools/make_gallery.py`: multi-area entries draw a reference per panel
  (`Layer.reference`), the polygons the crop filter removed (`removed_vs`) and,
  with `crop_on_reference`, the square with the most reference polygons; a
  panel can take its background from another output's provenance
  (`Layer.background_from`, e.g. the Sentinel-2 composite for embedding
  clusters); long panel titles are set smaller to fit, and counts read
  "1 field", "1 image". Panels of one study area (`Layer.area`, e.g. several
  models on one area) share a number, a locator dot and the window of the
  area's first panel. A DINOv3 run that loads a checkpoint fine-tuned in
  another run is credited to that checkpoint in the footer.
- `agribound evaluate --baselines` also scores the chance baselines with the
  evaluation's rules (`evaluate(..., baselines=True)`: a blind square grid
  whose cells have the predictions' median area, over the predictions'
  bounding box, the mean of 36 offset grids) and reports them under
  `baselines`. The study area, the shifted reference and the other options
  are available from Python. The command's other keys are as without the
  flag.

### Fixed

- TESSERA: a rewrite of the store under the same URL went unnoticed. The v1
  store was converted again on 2026-10-06/08 with its metadata documents
  (`zarr.json`) unchanged, so only the shard objects changed; a cached TESSERA
  raster was reused as it was, and so was an output made from it. agribound
  now records the state of the objects a TESSERA raster is read from (a store
  stamp, `agribound.composites.tessera_stamp`):
  - The objects are the shards of the `embeddings` and `scales` arrays that
    hold the box read, per UTM-zone part, and the chunks of the `x` and `y`
    coordinate arrays that hold it (geotessera selects the window and builds
    the returned transform from their values). Their keys are computed from
    the arrays' own Zarr metadata (shape, chunk grid, chunk key encoding,
    dimension names, the zone's `spatial:transform` and `time` values; seven
    small GET requests per UTM zone), and their ETag and Last-Modified come
    from HTTP HEAD (4 to 10 requests for the areas of examples 02, 05, 15 and
    23).
  - The SHA-256 of the zones' `proj:code` and `spatial:transform` and of the
    sorted keys and ETags, the newest Last-Modified and the number of objects
    are written to the raster's tags (`TESSERA_STORE_FINGERPRINT`,
    `TESSERA_STORE_LAST_MODIFIED`, `TESSERA_STORE_OBJECTS`; the box read is in
    `TESSERA_READ_BBOX`) and from there to the provenance record
    (`facts["composite"]`).
  - A cached raster keeps its file name. It is read again, and replaced, when
    its fingerprint differs from the store's current one or it has none; it
    is reused with a WARNING when the store's state cannot be determined (no
    network, an HTTP error, or `AGRIBOUND_TESSERA_STORE_CHECK=0`). The stamp
    written with a new raster is computed right before the read, without
    answers remembered in the process.
  - The output reuse check compares the recorded fingerprint with the
    current one for `source="tessera-embedding"` (a record without one does
    not match). When the store cannot be checked, it compares the recorded
    fingerprint with the one in the cached raster's tags instead: an output
    made before the raster was read again (for example by the HPC composite
    stage with network access) is not reused; otherwise it is reused with a
    WARNING.
  - HPC: `tiles status` checks staged TESSERA rasters against the store and
    shows them as `stale` when the store has changed, so `tiles status --list
    not-done --stage composite` and `submit_region.sh --resume` stage them
    again; the composite stage reads them again. Tiles that cannot be checked
    are named in one WARNING per `tiles status`, and `tiles merge` lists them
    in its summary. `submit_region.sh --offline-gpu` sets
    `AGRIBOUND_TESSERA_STORE_CHECK=0` in the compute jobs (not in the stage
    jobs). An agent proposal made with network access includes the
    fingerprint in its plan directory.
  - With `embedding_cache_dir`, geotessera's own cache keeps every object a
    read covers whole, the shards included (up to 1.7 GB each), and never
    checks them against the store again. agribound now reads through a store
    that keeps only the metadata documents and coordinate chunks on disk (the
    shards are read from the store every time; agribound caches the rasters
    it builds). Before a read it deletes the copies whose ETag has changed or
    was not recorded, and after the read it records the ETags
    (`<embedding_cache_dir>/<store>.agribound-etags.json`).
  - Requests: a redirected HEAD request stays a HEAD request (urllib sent it
    as GET before Python 3.13); a request whose connection cannot be made
    (failed name lookup, refused or timed-out connection, unreachable network
    or host, a connection not permitted, an unverified certificate) fails at
    once; answers are reused for 60 s within a process; a failed request is
    not repeated for 60 s, and only a store whose connection could not be
    made fails every check for that long (an HTTP error in one area no longer
    makes the checks of other areas fail).

  Also: two processes that build the same TESSERA raster at once (both found
  it missing or stale) wrote the same `<file>.part`, so one failed and the
  raster could be left unreadable; each writer now uses a temporary file of
  its own (`<file>.<pid>.<random>.part`, for every embedding raster).

  The configuration hash, the cache keys and `RESULTS_VERSIONS` are
  unchanged. Checked on 2026-10-09: the shard keys computed for the areas of
  examples 02, 05, 15 and 23 are those that geotessera 0.10.2's
  `read_region` requested for the same reads (for example 15 they include a
  row of shards that the 1.1 review had not listed), and the ETags agree with
  the review's HEAD requests.
- HLS (`source="hls"`): images of the MGRS tiles at 180° were selected
  everywhere except over their own tile. In Earth Engine (checked
  2026-10-09), 10,324 `NASA/HLS/HLSS30/v002` images of 31 of the tiles that
  straddle 180° (acquired 2015-12-22 to 2025-05-30) have a
  `system:footprint` ring that runs clockwise, which Earth Engine reads as
  the whole globe except the tile, so `filterBounds` with the export grid's
  outline returned them anywhere else. HLSL30 has no such image; nor had the
  Landsat 7/8/9, Sentinel-2, Sentinel-1 and SPOT 6/7 collections in the years
  checked (2022 and 2024).
  - Away from 180° these images hold no pixel in the export grid, so the
    composites are unchanged (identical pixel for pixel at five
    configurations, including the HLS composites of examples 03 and 14). But
    `AGRIBOUND_N_IMAGES`, `AGRIBOUND_IMAGE_YEARS`, the image-count log line,
    the provenance records and the years listed by the no-image error counted
    them. Example 03 (Murray-Darling, 2022) recorded 298 images; the real
    count is 77 (20 HLSL30 and 57 HLSS30, tile T55JGG). Example 14 recorded
    22 (1.0.0, October 2022; real 2) and 21 (its 1.1 run, October 2024; real
    10); run again with agribound 1.1.0 (2026-10-11), examples 03 and 14
    recorded 77 and 10. A window with no other image (Lea County, 2022-10-13
    to 2022-10-31, cloud cover <= 20 %: seven such images) downloaded a
    composite without
    valid pixels and raised "no valid pixel ... although 7 image(s) matched
    the filters"; it now raises the no-image error.
  - Inside an affected tile the composites lacked the tile's affected HLSS30
    images: in a 5 km square in Fiji (2022, cloud cover <= 80 %), 29 of the
    45 images of tile T01KAA, with which the mean number of observations per
    pixel rises from 90.2 to 106.5 and every pixel changes. No example or
    documented study area lies in these tiles.
  - `hls` composites now exclude the images whose footprint meets one of two
    points on the far side of the Earth (the antipode of the centroid of the
    grid outline, and that point moved 90° east;
    `agribound.composites.gee.hls_footprint_filter`). When the outline comes
    within 5° of 180°, the images of the tiles there that have such a
    footprint and whose footprint ring, read with the even-odd rule, meets
    the outline are listed with one request and loaded by ID
    (`hls_antimeridian_images`; their numbers are in the new tags
    `AGRIBOUND_HLSS30_IMAGES_BY_ID` and `AGRIBOUND_HLSL30_IMAGES_BY_ID`).
    HLSL30, which had no such image, is listed too, because the exclusion
    applies to it as well. The two parts work only together: the exclusion
    alone would drop the images of a study area that straddles the edge of
    an affected tile, which were selected and used before.
  - The agent's `check_availability(live=True)` counted these images too
    (1,254 instead of 160 HLS images for the Murray-Darling box in 2022) and
    now counts by the composite's rules, over the study area's bounding box.
  - `HLS_RECIPE_VERSION` (`"2"`, tag `AGRIBOUND_HLS_RECIPE_VERSION`) is part
    of the cache key of `hls` composites only, so cached HLS composites,
    FTW's window composites included, are made again once. The cache keys of
    the other sources, the configuration hash and `RESULTS_VERSIONS` are
    unchanged, so existing outputs are reused: away from 180° they are
    unchanged, and their provenance records keep the old image count.
  - An HLS output whose provenance record has no
    `AGRIBOUND_HLS_RECIPE_VERSION` among its composite's tags (made before
    this change) is not reused when its study area lies within 5° of 180°
    and within 0.5° of the latitudes of the affected tiles
    (`near_hls_reversed_footprint_tiles`): `delineate` raises
    `FileExistsError` (pass `overwrite=True`, CLI `--overwrite`) and its HPC
    tile shows as `stale`. The study area is read without Earth Engine (a
    GEE asset from its local copy; without one, the output is reused with a
    WARNING). Elsewhere, for example in New Zealand, such outputs are reused.
  - The stage markers of HPC tiles of `hls` (and `sentinel1`) include the
    source's recipe version. A tile whose HLS composite was staged before
    shows as not staged (`pending`), so `submit_region.sh --resume` stages it
    again: the delineation stage reads the composite under its new cache key
    and would otherwise download it (on a node without internet access, the
    tile failed). Tiles recorded as having no data are tried again. The
    markers of the other sources are unchanged.
- Earth Engine sources: the export grid of a study area that ends at 180°, or
  a little short of it (0.2 km for a 0.1°-tall area at 17° S, 3.4 km for a
  2°-tall one, up to about 10 km at 64° N to 66° N), reaches across 180°:
  its edges are rounded outwards to whole pixels, and away from the equator
  the edges of a UTM grid are not meridians. The grid's outline, with which
  images are selected, then ran from −180° to 180° as one polygon, and
  `filterBounds` selected images all around the globe at that latitude: for
  a 5 km square that ends at 180° in Fiji (2022, cloud cover <= 80 %), 20,806
  HLS images of tiles worldwide; now the 76 of tile T01KAB. The outline is now
  split at 180° (`ee_geometry`, `split_at_antimeridian`), and the far points
  of the HLS selection come from its centroid near 180°. Composites of the
  other sources cached before for such a grid are reused with their old tags;
  delete them to make them again. Study areas that cross 180° are still not
  supported.
- The USGS NAIP Plus client (the `usgs-naip-plus` source, and the
  co-registration base of the `umbra` and `capella` sources) gave up after 3
  retries 1, 2 and 4 s apart: an example 16 run failed on HTTP 502, and the
  held-out test of the radar sources met an HTTP 504. It now retries a request
  or download after HTTP 429 or 5xx (also as an ArcGIS error payload), a
  timeout or a dropped connection 5 times, waiting 4, 8, 16, 32 and 60 s plus
  up to 25 % at random, and logs each retry at WARNING; other HTTP errors
  (4xx) fail at once instead of being retried. `usgs_retries` defaults to 5
  (was 3). The exported images are unchanged. `usgs_retries` is not part of
  `config_hash` or of any cache key, and the tile-manifest signatures and
  agent plan directories take it at 1.0.1's default, 3, while it holds the
  new default (`agribound.provenance.DEFAULTS_OF_1_0_1`), so manifests and
  plan directories written by 1.0.x stay current.
- Delineate-Anything fine-tuning names its optimizer (AdamW), so Ultralytics
  applied its default `warmup_bias_lr=0.1`, which it uses only for a named
  optimizer. Ultralytics' own `optimizer="auto"`, whose AdamW choice the
  recipe follows (as documented), sets 0 ("no higher than 0.01 for Adam"),
  and so does agribound now. On 62 training chips of an oil
  palm estate (example 23), where the released weights score a validation
  mask mAP50 of 0.17, a test at `lr0=1e-4` (momentum 0.9) scored 0.46 after 20
  epochs without the bias warmup and 0.32 with it; with `yolo_lr0=1e-4` the
  new recipe scores 0.47 after 20 epochs. At the default `lr0` (0.002) the
  pretrained weights are wrecked with either setting: the run scored 0.00 with
  the 1.0.1 recipe, and with the new one 0.02 after the first epoch (the
  checkpoint Ultralytics keeps) and 0.00 from the third epoch on. The
  fine-tuning guide now recommends a smaller `yolo_lr0` for small training
  sets.
- Delineate-Anything fine-tuning chips treated a pixel that is 0 in every band
  as valid, while the native backend treats it as nodata at inference
  (upstream DelAnyFlow's default `nodata_value: [0, 0, 0]`): the labels were
  not clipped there, and such pixels counted towards `min_valid_fraction`. The
  chips now follow the backend's rule, and their image is 0 at every invalid
  pixel, as in the backend's tiles (a uint8 raster's nodata value, other than
  0, stayed in the chips). No composite of the fine-tunings of the 1.0
  gallery (examples 12 and 23) has such pixels. GeoAI, DINOv3 and Prithvi
  use the same rule at training and inference and are unchanged.
- `tools/make_gallery.py` with geopandas 1.2 or later: geopandas 1.2 labels
  the axes of every plot with the CRS axis names ("Easting [metre]"). The
  labels pushed the locator insets out of the footer, which stopped the render
  of every entry with an inset, and entries without one showed them beside
  each map. The tool now removes them.
- The Delineate-Anything module docstring said that the Cauvery delta
  SPOT-Pan composite of the filter-order test belongs to example 23. It
  belongs to example 22; Twifo Praso and Oro are example 23's. Two tests were
  added:
  - A checkpoint whose pixel size is 1.02 % from the raster's, just outside
    `PROCESSING_GSD_TOLERANCE` (1 %), must log the input-mismatch warning. A
    looser tolerance fails this test.
  - `field_interior_depths` on a field that covers the whole raster: the depth
    is the raster's longer side.
- `show_boundaries(satellite_tif=...)` drew no composite: example 10's
  `map.html` had no "Satellite Composite" layer in the 1.0 and 1.1 runs.
  leafmap's `add_raster` asked localtileserver for the raster's metadata,
  which for NaN nodata, as in every float composite agribound writes, is not
  valid JSON (localtileserver 1.0: HTTP 500), and the layer was skipped with
  a WARNING. A localtileserver layer also points at a tile server that stops
  with the Python process, so a saved `map.html` could not have shown it, and
  it showed bands 1-3 whatever the source. The composite is now embedded in
  the map as an image (layer "Satellite Composite"): warped to EPSG:3857, at
  most 2,048 px on its long side (`OVERLAY_MAX_SIDE`), stretched to 8 bits
  with per-band 1-99 % percentiles (uint8 data are kept), with nodata
  transparent. The new argument `satellite_bands` (`{"R": 4, "G": 3, "B":
  2}`; default bands 1-3, or band 1 in grey for a raster with fewer than 3
  bands) picks the bands. Example 10 passes its `--bands`; its `map.html`
  grows by about 0.66 MB.
- `tools/make_gallery.py`: `gallery_stats.json` now records where a
  single-area entry's squares are: `crop_window` (the `crop_m` square as
  drawn) and `zoom_window` (the zoom panel), each with CRS, bounds, centre
  lon/lat and side (`side_m`). `LOCAL_SQUARES` entries leave them out. Before,
  only the zoom side (`zoom_window_m`, kept) was recorded, so a zoom that
  moved between renders left no trace: entry 20's moved about 11.5 km between
  the 1.0 and 1.1 renders. The PNGs are unchanged.
- Ultralytics (8.4) stops non-maximum suppression after 2 s + 0.05 s per
  image of a batch, logs "NMS time limit ... exceeded" and returns no
  detections for the batch's remaining images. Delineate-Anything predictions
  (the native and `ftw` backends; the `reference` backend runs the upstream
  code in a subprocess, as before) and fine-tuning validation, on which
  `best.pt` is chosen, now run without that limit
  (`agribound.engines.delineate_anything.no_nms_time_limit`). The logs of
  the 1.0 run and the 1.1 re-run (examples 02-26) hold five such lines, all
  in fine-tuning validation passes: example 23's NORPALM fine-tune at epochs
  5, 8, 10 and 11 (3.15 s for its 23 validation chips) and example 12's NAIP
  fine-tune at epoch 4 (3.6 s, batches of 32). The epochs kept as `best.pt`
  (20 and 9) were not cut. No prediction batch was cut: a Delineate-Anything
  batch holds 4 tiles by default (8 in example 12), whose limit, 2.2-2.4 s,
  appears in no log. So the outputs of these runs do not change, and the
  results versions are unchanged. Detections that never reached the limit are
  identical.
- NAIP and USGS NAIP Plus composites (R, G, B, N; uint8) were written with
  band 4 tagged as alpha, GDAL's default for a 4-band uint8 GeoTIFF, while
  the band descriptions said N. GDAL's warper and rio-tiler (used by leafmap
  and localtileserver) then treat NIR as transparency: when a USGS NAIP Plus
  export had more than one tile, the writer merged them into a mosaic tagged
  so and blanked every band of a pixel whose NIR was 0 as it resampled the
  mosaic onto the export grid, and viewers that honour alpha drew low-NIR
  pixels transparent. The composites are now written with
  `ALPHA=UNSPECIFIED` (`gee.assemble_tiles`, and the USGS mosaic and
  output), and band 4 has no colour interpretation. The tiles that the
  service's `exportImage` returns tag no band as alpha (two 64 × 64 px tiles
  of 2022 imagery, in California and New Mexico, checked on 2026-10-11: four
  bands of undefined colour interpretation, no nodata), so a single tile,
  which the writer resamples directly, was not affected; a tile tagged alpha
  now has the tag removed before it is merged or resampled. The same GDAL
  default tagged band 4 of the crop of a 4-band uint8 local raster
  (`source="local"` with a study area) whatever the raster's own tags, and
  of the outputs of `agribound.io.raster.write_raster`,
  `agribound.io.raster.clip_raster_to_geometry` and
  `agribound.io.crs.reproject_raster`: these now keep the source's colour
  interpretation (a band tagged alpha in the source stays so; a palette band
  keeps its colour table), and `write_raster` tags no band as alpha. Read
  through GDAL's warper, as DINOv3's processing copy of a coarse raster is
  (see Changed), such a crop had R, G and B set to 0 wherever band 4 was 0.
  No pixel whose NIR is 0 with valid R, G and B was found in the four 1.1
  NAIP composites checked (example 12's, example 25's AR and AZ squares,
  example 16's), and the pixels do not change: rewritten by both versions,
  a 512 × 512 px window of example 12's Earth Engine NAIP composite and,
  through the USGS writer's Web Mercator to UTM resampling, two
  1,024 × 512 px windows of example 16's composite (one with 13,510 pixels
  of NIR 1-5) gave identical pixels. On a grid shifted by half a pixel in
  the tiles' own CRS, 94 of 523,776 pixels differed by 1 DN. Cache keys are
  unchanged, so a cached composite or local crop keeps the alpha tag (and
  any pixels blanked by it) until it is made again.
- The examples' `metrics_*.json` files and `tools/make_gallery.py`'s
  `gallery_stats.json` were not JSON when a score was undefined or a size
  class open: `agribound.evaluate` reports those as NaN and `inf`, and they
  were written as `NaN` and `Infinity` (37 of the 189 metrics files of the
  1.1 run, in examples 23 and 24; 121 values in
  `assets/gallery_1.0/gallery_stats.json`). Strict JSON parsers, such as
  JavaScript's `JSON.parse`, reject them. Examples 20, 21 and 23-27 and the
  gallery tool now write them with `agribound.provenance.to_jsonable` and
  `allow_nan=False`, as agribound's own JSON outputs are: NaN and infinite
  values are written as `null`. These are the undefined scores and the open
  upper bound of the last size class (`upper_ha`, and the last entry of
  `size_class_edges_ha`; `Infinity` in the 28 metrics files of example 24's
  1.1 run), so a JSON reader can no longer tell that bound from a missing
  value. No other value changes (re-serialised this way, the 152 metrics
  files of the 1.1 run without such values are identical byte for byte).
  Python readers that did arithmetic on these NaN now see `None`.
- NAIP (`source="naip"` without `date_range`): a WARNING said that images
  from the neighbouring years "fill gaps", and `AGRIBOUND_IMAGE_YEARS` listed
  those years, whenever their images intersected the study-area extent, also
  when the `year` images covered all of it. In example 25's 1.1 run that
  happened in all 7 squares with such images (LA, ND, NJ, NY, SD, VT, WV),
  where the `year` images covered every 2 m pixel. One more Earth Engine
  request (`gee.naip_year_areas`) now measures where each neighbouring year
  shows under the `year` images (the newer year on top), by image footprint.
  A year whose images fill more than one export pixel is named in the WARNING
  with its share of the extent and stays in `AGRIBOUND_IMAGE_YEARS`; the
  others are logged at INFO and listed in the new tag
  `AGRIBOUND_IMAGE_YEARS_UNDERNEATH`. When the request fails, every year is
  listed and the WARNING says the share was not measured. Checked on Earth
  Engine on 2026-10-11: example 25's LA square gives `2023`, with `2022`
  underneath; a 4.1 × 5.6 km box across the eastern edge of Oregon's 2022
  images gives `2022,2023` (Idaho's 2023 images fill 65.1 % of it by
  footprint, 65.0 % of its 10 m pixels), with `2021` underneath. The mosaic,
  its pixels and the cache keys are unchanged; a cached composite keeps its
  tags.

### Changed

- **Delineate-Anything, native backend: how the tiles are stitched.** The
  straight north-south and east-west lines in the outputs of examples 22 and
  23 lie on the edges of the 768 m inference tiles (512 px at 1.5 m, every
  384 m). Three changes, each closer to upstream DelAnyFlow. (1) A detection
  is tile-cut when its polygon (the largest part of its mask) reaches within
  4 px of an interior tile edge, the strip DelAnyFlow marks
  (`merging_edge_width`). It had to reach the outermost pixel, although masks
  often stop 2-3 px short of the edge, and a smaller part of the mask that the
  polygon leaves out counted too. (2) Overlap resolution (`resolve_overlaps`)
  gives contested area to complete detections before tile-cut ones, and then
  by confidence and area. A tile-cut detection with the higher confidence took
  the area, so the straight line where its tile cut it was cut into a field
  that another tile saw whole. A union of tile-cut pieces
  (`merge_tile_pieces`) stays tile-cut. `resolve_overlaps()` no longer takes
  the duplicate thresholds, which only set that order. (3) Tile-cut pieces
  that meet along a tile edge without overlapping are joined when they pass
  the seam test of upstream DelAnyFlow's `merge_tile_seams` (commit 488e537),
  with its thresholds: they touch along at least 8 px and 30 % of the longer
  cut edge; the cut edges start and end within max(8 px, a quarter of the
  contact) of each other, unless the smaller piece is at most 16 px deep; the
  tile centred on the edge, composed as DelAnyFlow composes a tile, did not
  see two fields there; the mean model input on the two sides, within 4 px of
  the edge, differs by at most 0.2 (0-1 scale); and each piece is the other's
  longest contact on that edge. Unlike upstream, complete detections are never
  joined. Before, only pieces that overlap were joined (`merge_tile_pieces`).
  Before their union, the pieces of a seam join are snapped to the
  model-input pixel grid. Two pieces from neighbouring tiles get the
  coordinates of their shared edge from different tile transforms, which
  differ by about 1e-10 m when the pixel size or the origin is not exact in
  binary. Without the snap, the union could keep only the larger piece and
  drop the smaller one, while the join was still counted (52 % of a test
  field kept on a 1.2 m raster, on a 2.4 m `processing_gsd_m` copy of a 1.5 m
  raster and at two of eight seam positions on the 0.59994 m Web Mercator
  grid of `usgs-naip-plus`). A piece that the union still cannot connect is
  now kept as it was, and no join is counted. The 1.5 m UTM composites of
  examples 22 and 23 were not affected: the numbers below are unchanged.
  Measured with the released weights on the cached SPOT-Pan composites,
  before and after: at Twifo Praso (example 23), 128 and 122 polygons,
  recall at IoU 0.5 0.45 and 0.49, precision among the predictions that
  overlap a block 0.21 and 0.23, split rate 0.49 and 0.44, merge rate 0.036
  and 0.055, and 14.0 % and 12.7 % of the boundary length within 2 m of a
  tile edge (2.1 % by chance); in the Cauvery delta (example 22, no
  reference), 3,153 and 3,056 polygons and 4.6 % and 2.5 % on the tile grid
  (1.9 % by chance); at Oro the released model still finds one parcel (the
  development run of 1.1 gave the same counts and scores at Twifo Praso and
  in the Cauvery delta). With the Higaturu fine-tuned model at Oro: 392 and
  350 polygons, recall 0.23 and
  0.27, precision among overlapping predictions 0.19 and 0.25, split rate
  0.34 and 0.23, merge rate 0.17 and 0.20, and 20.9 % and 13.7 % on the tile
  grid. Without the seam merge (the first two changes only) the tile-grid
  shares were 13.2 % (Twifo), 3.2 % (Cauvery) and 17.8 % (Oro, fine-tuned).
  Of the seam joins that could be checked against the references (both
  pieces at least half in one reference field), 1 of 4 at Twifo and 6 of 18
  at Oro (with that earlier Higaturu model; 0 of 1 with the model of the
  current recipe) join two fields; the upstream configuration guide warns of
  such joins where two fields meet exactly on a tile edge. In the
  development run of 1.1 the test also joined two orchard blocks at Madera
  (example 23) whose road gap runs up to 21 m off the tile edge, at 2-3° to
  it: 2 of the 3 checkable joins of the SPOT-Pan output. The tree-crop F1
  there is 0.83, against 0.86 with the stitching of 1.0.1 and 0.85 with the
  seam merge off: the seam merge accounts for 4 of the 5 matched orchards
  lost (88 against 93). The test reads a strip 2 px wide along the edge
  (upstream's `seam_split_window_px` of 1; upstream's guide notes that a
  window of 8 joins about half as many neighbouring fields). That run also
  showed tile-cut detections chained by `merge_tile_pieces` across fields
  that the image separates (up to 359.6 ha in example 25's Montana square,
  where with the piece merging off the largest polygon is 78.4 ha), a block
  of strip plots in the Nile Delta (example 22) drawn as one polygon from a
  low-confidence outline that the 4 px rule counts as tile-cut, and outlines
  that end on the edge of the last tile that detected a field (see the
  engines guide). Candidate fixes,
  among them a window of 8 px, were evaluated on the native
  Delineate-Anything outputs of that run against rules set before the
  evaluation; each lowered F1 by more than 0.005 in some evaluation or put
  more boundary near tile edges in some output, so none was adopted and the
  results version stays 2. At Twifo 74 % of the boundary that still lies
  exactly on a tile edge (25 of 34 km) is the
  outer edge of a block wider than a tile, with nothing detected beyond it,
  so there is nothing to join. The minimum area (`min_field_area_m2`) is
  still applied to the raw detections before the merges and again after the
  seam merge, whereas upstream applies its `minimum_area_m2` only after its
  seam merge, so a tile-cut piece smaller than the minimum cannot be joined
  to its neighbour. Keeping the tile-cut detections until the second filter
  was measured on the same composites, before and after: 6 and 8 seam joins
  at Twifo Praso (122 and 123 polygons; 27 of 55 blocks matched in both;
  precision 0.221 and 0.220, the extra polygon a false positive; split and
  merge rates unchanged), 38 and 45 with the Higaturu fine-tuned model at Oro
  (350 polygons in both; recall 0.268 and 0.272; precision among the
  predictions that overlap a parcel 0.253 and 0.256; split rate 0.228 and
  0.232; merge rate 0.202 and 0.199; 13.7 % and 13.2 % of the boundary on the
  tile grid) and 69 in both in the Cauvery delta (3,056 polygons). The order
  was kept.
- **Delineate-Anything fine-tuning recipe** (recipe version 3). A model
  fine-tuned on few chips can learn the frame of its chips: the model that
  example 23 fine-tuned on its Higaturu area (22 training chips) drew
  detections along the frame of every inference tile at Oro. On the same
  pixels, its per-tile detections covered 53 % of the pixels within 16 px of
  an interior tile side and 18 % of those 64-128 px inside (the reference
  parcels: 47 % and 50 %). Three changes: (1) the chips overlap, half a chip
  apart (`chip_stride`), as the inference tiles do, so every field less than
  half a chip across is whole in some chip; the split is made on chip-size
  cells, and a chip that covers cells of both splits is dropped, so no pixel
  is in both a training and a validation chip. Overlapping chips mean about
  four times as many chips, and so several times the training time and disk
  (Higaturu: 110 chips instead of 27), and with a random split the
  validation share of the chips falls below `fine_tune_val_split` (Higaturu
  at 0.2: 9 of the 47 cells are validation cells, but only 6 of the 77 chips
  kept are validation chips; 33 chips covered cells of both splits and were
  dropped). (2) The optimizer steps after
  every batch (`yolo_nbs` defaults to the batch size): with Ultralytics'
  default `nbs` of 64, the 22 chips (two batches of 16 per epoch) got 11
  optimizer steps in 20 epochs (a step every four batches, after a warmup
  of three epochs that raises the number of batches per step from one). (3) A part of a reference polygon that the
  chip frame cuts to a strip at most 16 px deep gets no label
  (`yolo_sliver_depth_px`; 16 px is the depth at which upstream DelAnyFlow's
  tile-seam merge treats a tile-cut piece as a strip along the border,
  `seam_max_strip_depth_px`): 9.8 % of the Higaturu training instances.
  Trained again with example 23's settings (`yolo_lr0=1e-4`, 20 epochs,
  `min_label_fraction` 0.25, random split, seed 42) on the cached Higaturu
  composite and run at Oro with this release's inference, before and after:
  350 and 272 polygons; recall at IoU 0.5 0.27 and 0.42; precision among the
  predictions that overlap a parcel 0.25 and 0.49 (F1 0.26 and 0.45); split
  rate 0.23 and 0.25; merge rate 0.20 and 0.49; 13.7 % and 3.2 % of the
  boundary length within 2 m of a tile edge (1.9 % by chance); per-tile
  coverage near and inside the tile frame 0.53 / 0.18 and 0.51 / 0.54; and
  22 % and 5 % of the 259 parcels detected only in tiles that cut them. The
  new model outlines whole plantings and joins neighbouring parcels that the
  image does not separate, hence the higher merge rate; the earlier one drew
  parts of parcels (median per-tile detection 1.58 ha against 0.65 ha; median
  parcel 1.53 ha). With seed 7, the earlier recipe gave a model that found
  almost nothing (recall 0.03, precision 0.05, 22 % of the boundary on the
  tile grid) and the new one recall 0.52, precision 0.57, split rate 0.21,
  merge rate 0.36 and 2.4 % on the grid. Undoing one change at a time (seed
  42), recall, precision, split and merge rates, share on the grid and
  coverage near / inside the frame were: chips that do not overlap 0.39,
  0.34, 0.37, 0.33, 8.4 % and 0.69 / 0.49; `nbs` 64 0.41, 0.43, 0.30, 0.45,
  4.8 % and 0.64 / 0.64; slivers kept 0.43, 0.48, 0.25, 0.50, 4.1 % and
  0.55 / 0.57. The earlier recipe, set through the
  new parameters (`chip_stride=512`, `yolo_nbs=64`,
  `yolo_sliver_depth_px=0`), gives the Higaturu checkpoint of the 1.0 gallery
  bit for bit. The changes were made after the bias was found at Oro, so these
  numbers are not an independent test of them; their values (half a chip,
  `nbs` equal to the batch, 16 px) were not tuned there. The development
  run of 1.1 gave the same Oro scores (272 polygons, recall 0.42, precision
  among overlapping predictions 0.49, 3.2 % on the tile grid) and made the
  other Delineate-Anything fine-tunings of examples 12 and 23 (NAIP;
  NORPALM, Cressey, Ibros) again with this recipe (see "Results produced
  with agribound 1.0.1 that are affected").
- **DINOv3 fine-tuning recipe** (recipe version 2). The DINOv3 models that
  example 23 fine-tuned with agribound 1.0.1 hardly ever predicted the
  field-boundary class, so touching fields merged: on the validation chips of
  their four training areas the class covered at most 1.8 % of the predicted
  pixels (boundary IoU at most 0.04; mean merge rate 0.77), and at Twifo Praso
  the NORPALM model drew one polygon over 94 % of the square. agribound now
  trains DINOv3 with its own Lightning loop (that of geoai's
  `train_dinov3_segmentation`, which 1.0.1 called) and four changes: (1) the
  cross-entropy is weighted by class, inversely to the class's share of the
  labelled pixels of the training masks, normalised so that a labelled pixel
  weighs 1 on average and capped at 10 (`class_weights`,
  `class_weight_cap`); (2) the checkpoint is the epoch with the best mean IoU
  of the three classes over the validation chips, and early stopping follows
  the same metric (`monitor`; 1.0.1 used the validation loss); (3) the model
  normalises its [0, 1] input with the mean and standard deviation its
  backbone was pre-trained with (SAT-493M: mean 0.430, 0.411, 0.296 and
  standard deviation 0.213, 0.156, 0.143, from the facebookresearch/dinov3
  README) inside its `forward`, so training and inference normalise alike
  (`input_normalization`; 1.0.1 fed the [0, 1] values); (4) the boundary band
  is 4 px wide by default (`boundary_erosion`; 2 px before). The width was
  chosen on the validation chips of example 23's four training areas (SPOT
  6/7 panchromatic at 1.5 m), never on an evaluated square, by a rule fixed
  before the widths could be compared: of 2, 4, 8 and 12 px, the width with
  the highest field F1 (IoU ≥ 0.5) averaged over the four areas or, of those
  within 0.01 of it, the narrowest (mean F1 0.588, 0.641, 0.649 and 0.647;
  the 1.0.1 models 0.421). With 4 px the mean merge rate on those chips was 0.39 and the split
  rate 0.09 (1.0.1: 0.77 and 0.01). Fine-tuned again on NORPALM with example
  23's settings (20 epochs, `min_label_fraction` 0.5, random split, seed 42)
  and run on the cached Twifo Praso composite with this release's inference,
  before and after: 1 and 102 polygons; 0 and 36 of the 55 reference blocks
  matched at IoU 0.5; precision among the predictions that overlap a block 0
  and 0.40 (F1 0 and 0.50); merge rate 1.00 and 0.13; split rate 0 and 0.27.
  Many of the extra boundaries follow tracks inside the reference blocks.
  With seed 7: 107 polygons, 36 matched, precision 0.39, merge rate 0.16,
  split rate 0.29. Undoing one change at a time (seed 42): without class
  weights 22 polygons, none matched, merge rate 0.98; with the checkpoint by
  validation loss (epoch 5 instead of 13) 142 polygons, 29 matched, precision
  0.25, merge rate 0.04, split rate 0.51; without input normalisation 113
  polygons, 33 matched, precision 0.33, merge rate 0.25, split rate 0.38. The
  changes were made after the merging was found at Twifo Praso, so these
  numbers are not an independent test of them; no setting was tuned there,
  although NORPALM models with 2, 4 and 8 px bands were scored at Twifo
  Praso while the widths were compared on the validation chips (the rule
  picked the width). The development run of 1.1 gave the same numbers at
  Twifo Praso (102 polygons, 36 blocks matched). The width is in pixels: a
  field less than 9 px across has no interior pixel and cannot be learned. On the composite grids of the
  eastern Lea County square of examples 12 and 14 that is 5.6 % of the 393
  DRI/NMOSE training polygons at Sentinel-2's 10 m and 41.2 % at 30 m
  (1.0 % and 13.7 % with 2 px, 3.3 % with 1 px at 30 m); the chips count
  such fields, and `widest_boundary_erosion` picks a band from the labels
  (see Added). The checkpoint also holds the size of its training chips
  (`chip_size` hyper-parameter), which inference uses as the default window
  when `<checkpoint>.agribound.json` is missing; before, such a checkpoint
  ran on geoai's 512 px windows although it was trained on 256 px chips. The
  other DINOv3 fine-tunings of examples 12, 14 and 23 change too; the
  development run of 1.1 made them again (see "Results produced with
  agribound 1.0.1 that are affected").
- **DINOv3 on rasters coarser than 10 m (Landsat, HLS, Landsat PAN).** In the
  1.1 run of example 14 the DINOv3 models fine-tuned on the 30 m Landsat and
  HLS 2024 composites found 3 of the 388 DRI/NMOSE V18 fields each (F1 0.014).
  The composite of the square held 4 training chips of 256 px (3 training, 1
  validation), so an epoch was one optimizer step at batch size 8, and the
  checkpoint by `val_miou` of the one validation chip kept the model after 3
  steps: early stopping ended training after 13, while the validation loss
  fell at every epoch. DINOv3 also predicts on its 16 px patch grid, 480 m at
  30 m, more than the shorter side of 47 % of the training fields. DINOv3 now
  reads a raster with pixels more than 1 % larger than 10 m (Landsat and HLS,
  30 m; Landsat PAN, 15 m) on a bilinear copy with 10 m pixels, for its
  training chips and at inference (`engine_params["processing_gsd_m"]`,
  default `"auto"`; `"native"` reads the raster as it is; a number of metres
  finer than the raster's pixels sets that grid, and a coarser one raises
  `ValueError`). Pixel sizes are measured in metres: in a projected CRS in US
  survey feet the pixel width is converted (a 10 m raster in EPSG:2263, 32.8
  ft pixels, is read as it is; a 30 m one on a grid of 32.8 ft), and in a
  geographic CRS it is the width along the parallel through the raster's
  centre. The copy holds the grid pixels that lie inside the raster: where the
  raster's size times the factor is not a whole number (an odd size at Landsat
  PAN's 15 m, factor 1.5), a strip narrower than one 10 m pixel along its last
  column or row is left out, so the copy has no unfilled pixels (the warp
  leaves 0 there, which counts as a valid pixel in a raster without nodata).
  Before, DINOv3 ignored `processing_gsd_m`, a Delineate-Anything parameter.
  On the 10 m grid the same composite gives 52 chips (42 training, 10
  validation). Three remedies were fixed in advance and tested with example
  14's Landsat 2024 settings on the CPU (seed 42), against a control of the
  recipe as it was (F1 0.023, 5 matches): checkpoint and early stopping by the
  validation loss below 4 validation chips, F1 0.107; 80 px chips at 30 m, F1
  0.102; the 10 m grid, F1 0.538 (163 matches; recall 37 of the 74 fields of
  the validation chips, which the model was not trained on). The rule, written
  before the runs, took the highest F1 among the remedies that change nothing
  at 10 m or finer. The validation-loss rule would have changed example 24's
  SPOT 6/7 runs (2 validation chips). With the grid as code, Landsat 2024 with
  seeds 0 and 1 scored 0.528 and 0.545, and HLS 2024 (seed 42) 0.542. Landsat
  PAN was not tested. Rasters of 10 m or finer (Sentinel-1, Sentinel-2, SPOT
  6/7, NAIP) are read as before, with the same chips, cache keys,
  configuration hashes and results versions. The processing grid enters the
  keys of the chips and of the fine-tuning run only when a copy is read, and
  `results_versions` gains `dinov3_processing_grid` (2) only for those runs
  (see "Results produced with agribound 1.0.1 that are affected").
  `<checkpoint>.agribound.json` and the checkpoint's hyper-parameters record
  `processing_gsd_m`, and inference reads the grid the checkpoint was trained
  on (`engine_meta["processing"]`), also when the checkpoint is copied or
  renamed without that file (`inference_processing_gsd` reads the
  hyper-parameter then). On the 10 m grid inference has 9 times as many
  pixels: the delineation step took 55 s against 14-27 s at 30 m (CPU, this
  square). Examples 12 and 14 measure the band on that grid (see "Examples 12
  and 14: DINOv3's boundary band per source") and the fine-tuning guide
  (Processing grid). Example 14's Landsat and HLS runs, made again with
  agribound 1.1.0 on 2026-10-11 (MPS), scored in-sample F1 0.534 and 0.547
  against the 388 V18 fields.
- **DINOv3 inference: agribound's own sliding window, with blended windows.**
  The straight north-south and east-west lines in the DINOv3 outputs of
  example 23 lie on the write lines of geoai's `dinov3_segment_geotiff`, which
  keeps each window's prediction up to the middle of its overlaps (with
  256 px windows half a window apart, every 192 m from 288 m on at 1.5 m), so
  the class changes wherever two neighbouring windows disagree there.
  agribound now runs the windows itself
  (`agribound.engines.dinov3.segment_raster`; the same windows and mirror
  padding) and gives each pixel the class with the highest average
  probability over the windows that cover it, each window weighted by a
  cosine taper that is 1 in its centre and falls over the overlap, so that
  the weights of neighbouring windows sum to 1
  (`engine_params["window_stitching"]="blend"`, the default; `"crop"` is
  geoai's rule). With the checkpoint example 23 fine-tuned on NORPALM (1.0.1)
  at Twifo Praso, before and after: across the write lines the class changed
  between neighbouring columns at 2.59 % and 0.21 % of the rows (elsewhere
  0.11 %), and between neighbouring rows at 2.10 % and 0.08 % of the columns
  (elsewhere 0.08 %); 66 % and 0 % of the long (at least 30 m) straight
  north-south and east-west segments of the outline lay on them (3.0 and
  0.3 km of such segments; about 0.9 % would by chance). That model still
  drew one polygon over most of the square: the merging comes from the
  model (see above). With `"crop"` the segmentation is that of 1.0.1, pixel
  for pixel. With the model of the new recipe (see above), crop and blend:
  across the write lines the class changed between neighbouring columns 6.5
  and 0.9 times, and between neighbouring rows 3.7 and 1.3 times, as often as
  across the other lines of the 16 px patch grid on which the decoder
  predicts (the write lines are lines of that grid); 13.5 % and 0.6 % of the
  long straight segments of the outline lay on them; 37 and 36 of the 55
  blocks were matched (merge rate 0.13 for both, split rate 0.22 and 0.27).
- `landsat-pan` no longer mixes Landsat 7 and Landsat 8/9 PAN by default.
  Landsat 7 ETM+ PAN (0.52-0.90 µm) includes the near infrared and Landsat
  8/9 OLI PAN (0.50-0.68 µm) does not, so their values differ, most over
  vegetation: over the oil-palm blocks of Twifo Praso, Ghana, in 2023, the
  median TOA reflectance was 0.220 (Landsat 7) against 0.096 (Landsat 8) and
  0.081 (Landsat 9). The source merged every
  mission whose record overlapped the window into one median; it now follows
  `landsat_pan_missions`, whose default `"auto"` uses Landsat 8/9 whenever the
  window overlaps their record and Landsat 7 only for earlier windows. For a
  2023 composite of Lost Hills, California, it used 16 Landsat 8 and 16
  Landsat 9 images; the merged median had 52, 20 of them from Landsat 7. When
  a window has no images, the `NoDataError` lists the years that have images
  of the missions searched, labelled with those missions, and names the
  missions; with `"auto"`, when Landsat 7 has images in the window that pass
  the same filters, it also gives their number and says that
  `landsat_pan_missions="LE07"` uses them (one extra Earth Engine request, on
  that failure path only).
- `AGRIBOUND_SENSORS` of a `landsat-pan` composite names only the missions
  that contributed images (it named every mission whose record overlapped the
  window).
- The `landsat-pan` composite cache key includes `landsat_pan_missions`.
- `landsat-pan` follows `landsat` in `SOURCE_REGISTRY` and
  `agribound list-sources`, and its registry coverage text (shown by
  `list-sources` and the agent tools) states the default mission rule. The
  agent's live `check_availability` counts, for `landsat-pan`, only the
  collections of the missions `"auto"` uses for the year, and names the other
  missions that have images, with their counts, in its message; it counted all
  three, including Landsat 7 in 2013-2024.
- Delineate-Anything has a `landsat-pan` source note (`engine_notes`, used by
  the agent tools): 15 m Landsat PAN composites are outside its 0.25-10 m
  training range, and the single band is replicated to grey R, G, B.
- `composite_method="greenest"` or `"max_ndvi"` with a source that has no NIR
  and red bands (`landsat-pan`, `spot-pan`, `sentinel1`) is rejected when the
  configuration is created (`ValueError`), so `--dry-run`, agent plans and
  `agribound tiles make` refuse it. The composite builder raised the
  `ValueError` only after the image count, and a window without images raised
  `NoDataError` instead.
- Fields added after 1.0.1 enter `config_hash` only where they apply
  (`agribound.provenance.HASH_CONDITIONAL_FIELDS`): `lulc_tree_crops` only
  when it is True, `landsat_pan_missions` always for `source="landsat-pan"`
  (even at `"auto"`) and never for the other sources. Other configurations
  keep their hashes, so their outputs are still reused. The signature of HPC
  tile manifests and the agent's plan directories follow the same rule,
  except that a field that does not apply still counts when it is set to a
  non-default value (`agribound.provenance.drop_inapplicable_fields`), so
  `agribound tiles make` keeps a manifest written by 1.0.x for the same
  configuration without `--overwrite` (and `tiles merge` reuses the merged
  output), and an agent plan finds the output of the same earlier proposal.
  Agent plan IDs change, because a plan hashes every field of its
  configuration.
- **If you ran `landsat-pan` before this change** (it was only on the main
  branch): those outputs now raise `FileExistsError` instead of being reused;
  recompute them with `overwrite=True` (CLI `--overwrite`). Their composites
  are rebuilt, because the composite cache key changed. Tiled runs need
  `agribound tiles make --overwrite`, then `tiles run --overwrite` (their
  tiles show as `stale`) and `tiles merge --overwrite`; agent plans get a new
  plan directory.
- New Mexico examples (01, 12, 13, 14, 20 and 21), their notebooks, the Lea
  County and New Mexico statewide region presets and the gallery entries of
  examples 12, 13, 14 and 20 use the DRI/NMOSE field boundaries (V18)
  instead of the NMOSE WUCB polygons, with the same study boxes. The
  examples read `examples/NM_Field_Boundaries.zip` in place (local data, not
  distributed). Lea County is selected by `FIPS_code == "35025"` (WUCB:
  `County == "25"`), and example 20 stratifies by `Sub_Basin`, the HUC-8
  subbasin name (WUCB: `WUCR_SBasi`); V18 has no `Source` attribute, so
  example 20 prints the most common `CDL_Crop_T` classes instead. Examples 12
  and 14 write their reference subset to new files
  (`lea_county_reference_v18.gpkg`, `reference_v18.gpkg`), so an older WUCB
  file is not reused; example 01 copies V18 once to `nm_reference_v18.gpkg`
  (agribound reads `reference_boundaries` from a vector file, not from a zip
  archive) and uses its bounding box as the study area. The region presets
  read the shapefile extracted to `examples/NM_Field_Boundaries/` (`unzip
  examples/NM_Field_Boundaries.zip -d examples/NM_Field_Boundaries`;
  gitignored). Examples 13, 20 and 21 accept `--reference
  zip://ARCHIVE!MEMBER`.
- New Mexico examples: imagery years. Examples 12 and 14 use 2024 for
  Sentinel-2, Landsat, HLS and the embeddings and the latest year with
  imagery over the box for the others, NAIP 2022 and SPOT 6/7 2023
  (`SOURCE_YEARS`; was 2022 for every source), so their NAIP and SPOT
  imagery is 2 and 1 years older. Example 14's `--years` now overrides those
  years for every source (default: none). Example 20 delineates Sentinel-2
  2024 (was 2019) and runs its resolution comparison in 2022 (was 2018) for
  all four sources, so that its rows differ in resolution only (the examples
  otherwise use 2024 wherever a source has it): 2022 is the year closest to
  2024 that Landsat, Sentinel-2, SPOT 6/7 and NAIP all cover there. The only
  2022 SPOT 6/7 scene has valid data over 76.7 % of the San Juan box; 158 of
  the 1,355 evaluated reference fields lie entirely outside the valid data,
  and 5 more partly. Output names, log lines and
  tables carry the year of each source: example 20's metrics, per-field
  table and map are now `metrics_<source>_<engine>_<year>.json`,
  `per_field_evaluation_<source>_<engine>_<year>.csv` and
  `map_predicted_vs_reference_<source>_<engine>_<year>.html` (with
  `--predicted`, named after the predicted file), the maps of examples 12 and
  14 are named after the years they show (`..._2022-2024.html`), and example
  01's maps after their year (`map_predicted_vs_reference_<year>.html`,
  `map_latest_<year>.html`, `map_timeseries_comparison_<first>-<last>.html`).
  Example 13's default input is example 12's
  `fields_sentinel2_dinov3_2024.gpkg`. Outputs of the fine-tuned NAIP 2022
  runs of examples 12 and 14 made with the WUCB reference raise
  `FileExistsError` when the updated examples run in the same output
  directory (their configuration names another reference file): delete them
  or use a new directory. The scores of the 1.0 gallery were computed
  against the WUCB polygons; the development run of 1.1 ran examples 12 (its
  NAIP runs), 13, 14, 20 and 21 against V18 (see the gallery). Example 01
  was not run.
- Examples that evaluate (01, 12, 13, 14, 20, 21, 23, 24, 25, 26, 27 and 28)
  drop the reference polygons smaller than 2,500 m², the predictions' default
  minimum area (`min_area`; areas in EPSG:6933, as agribound's area filter
  computes them, except in examples 23, 24 and 26, which compute them in the
  UTM zone of the square or site), and print how many they drop, so that
  both layers have the same minimum area. Polygons under 1,250 m² can never
  be matched at IoU ≥ 0.5 by a prediction of at least 2,500 m². With the DRI/NMOSE field
  boundaries (V18): 1 of the 389 polygons in the Lea County box (examples 12,
  13 and 14), 350 of the 1,705 in the San Juan box (example 20) and 152 of
  the 797 near Belen (example 21). Example 01 fine-tunes on all V18 polygons
  and evaluates against a copy without the 16,922 smaller ones
  (`nm_reference_v18_min2500.gpkg`, 68,357 polygons). Example 21 filters the
  published FTW polygons at the same area (`--min-area`, default 2,500; 0
  keeps every polygon of both layers). Example 23 already dropped them and
  now prints the counts; example 24 drops none of the 74 and 187 fields of
  its evaluation squares; example 25 drops 0 to 37 fields per square (none
  in Rhode Island and Texas, 37 in Iowa) and 171 of the 1,000 V18 polygons
  of its New Mexico square.
- Example 13 labels its evaluation "partly in-sample" when the input was
  fine-tuned and `--reference` differs from the fine-tuning reference: it
  counts the evaluated polygons that lie (by representative point) in the
  fine-tuning reference, and labels the evaluation in-sample, partly
  in-sample or neither. The docstring example evaluates example 20's output,
  which was not fine-tuned.
- Example 01 states that its default year, 2025, is a held-out-year check of
  the fine-tuning on the 2024 composite (held out in time, not in space: the
  evaluation polygons are the training labels), and labels the 2024 and the
  other years' evaluations accordingly.
- **Examples 12 and 14: DINOv3's boundary band per source.** A field less
  than 2 × `boundary_erosion` + 1 pixels across has no interior pixel and
  cannot be learned; with DINOv3's default 4 px band that is 41 % of the 393
  training polygons of the eastern Lea County box on the 30 m Landsat and
  HLS grids (13.7 % with 2 px, 3.3 % with 1 px), 5.6 % on Sentinel-2's 10 m
  grid, 1.3 % on SPOT 6/7 (6 m) and none on NAIP (1 m). Both examples now
  set `engine_params["boundary_erosion"]` of each DINOv3 run with
  `widest_boundary_erosion` on the run's composite and the training
  polygons, a rule on the labels and the composite grid alone (no model was
  trained or scored to choose it), measured on the grid DINOv3 reads: 4 px
  (the default) on every source, Landsat and HLS on their 10 m processing
  grid (6 % of the polygons without an interior pixel; on their own 30 m
  grid the rule gives 1 px; see DINOv3 on rasters coarser than 10 m). The
  script prints the shares. Prithvi keeps its 2 px default (example 12), so
  its Landsat and HLS chips log the WARNING. The DINOv3 outputs of both
  examples in the 1.0 gallery were made with 1.0.1's recipe. The development
  run of 1.1 made example 14's again, and example 12's NAIP run, with the
  current recipe and bands (Landsat and HLS of example 14 then still at
  30 m, with a 1 px band); agribound 1.1.0 made example 14's Landsat and HLS
  runs again on the 10 m grid with the 4 px band (2026-10-11).
- `tools/make_gallery.py` renders by default from `outputs/examples_1.1`
  into `assets/gallery_1.1`, so the 1.0 gallery is kept (`--run-root
  outputs/examples_1.0 --out-dir assets/gallery_1.0` renders into it again).
  `Entry.reference_min_area_m2` drops the reference polygons smaller than
  that area before they are drawn, counted or used to place the crop square
  (entries 20, 20b, 20c and 13 use 2,500 m², so they draw the reference that
  example 20 evaluates). Reference data whose outlines must not be drawn can
  be listed in `RESTRICTED_REFERENCES`: an entry is refused when any file it
  names (layers, references, overlay, `removed_vs`, backgrounds,
  `model_from`, metrics) names one of them, as written, normalised (`..`,
  `//`) or as the files it names under the run root (globs expanded,
  symlinks resolved). Outputs whose squares must not be placed can be listed
  in `LOCAL_SQUARES`: such an entry takes no explicit `window`,
  `mark_windows` or locator inset, and its `gallery_stats.json` record keeps
  the counts and scores but leaves out the coordinates (the lon/lat of
  `location` and the panels' windows). Both lists are empty: the OpenET
  field boundaries, V18 and the DRI field boundaries of Nevada (examples 24
  and 25) may be drawn and their squares placed. The title of entry 13
  carries its year (Sentinel-2 2024).
- The [Example Gallery](https://montimaj.github.io/agribound/gallery/) shows
  agribound 1.1 (`assets/gallery_1.1`; the 1.0 images stay in
  `assets/gallery_1.0`). Each entry reads its own run root
  (`Entry.run_root`): the development run of 1.1 (`outputs/examples_1.1`),
  the runs of agribound 1.1.0 on 2026-10-11 (examples 03, 14, 23 with
  `--s2-single-date`, 25 with `--crop-sequence` and 27, and example 10,
  which has no image), example 26's run of 2026-10-10, example 24's run of
  2026-10-13 with `--options --sam-auto` (`outputs/examples_1.1_ex24b`, which
  reuses the outputs of its re-scoring run of 2026-10-11), example 28's run
  of 2026-10-12/13 (`outputs/examples_1.1_r3/ny`), the SAM evaluation's runs
  (`outputs/examples_1.1_r3/sam_eval`) and the NAIP 2005 export of
  Montana's square (`outputs/examples_1.1_r3/montana`). New entries:
  24 and 24b (Nevada: Diamond Valley and Mason Valley against the DRI field
  boundaries), 24c and 24d (example 24's irrigated-only scores: the fields
  irrigated in the year of each panel's imagery in cyan, the others
  hatched, and the predictions the score ignored in grey;
  `Entry.scored_part`), 24e (Diamond Valley before and after the opt-in
  options of 1.1.0, with the panels fixed in PREREG-DV3 section 7; reported
  results, not tests), 25 (a map of the 47 squares coloured by F1), 25b
  (four squares, Delineate-Anything beside the crop-history zones), 25c
  (Montana: the CLU reference of example 25's square on the NAIP of 2005 and
  on the 2023 composite, no model output; a new entry kind, `Entry.imagery`
  with `ImagePanel` and `render_imagery`: one reference on several images in
  one window, a panel's own run root, and a check of an exported image
  against its record), 26 and 26b (radar at the held-out and in-sample
  sites), 27 (Quebec), 28, 28b and 28c (New York, with the panels fixed in
  PREREG-NY section 5) and the SAM entry (`sam`: SAM 2.1 automatic masks
  against Delineate-Anything v2 as released, with the panels fixed before the
  evaluation: Diamond Valley's evaluation square on SPOT 6/7 and Tennessee's
  NAIP square, the square at the median of the paired NAIP differences).
  Entries 24 to 24d rank the default and `sam-auto` runs of the run of
  2026-10-13 for their panels (the "-opt" runs have their own entry, 24e), so
  `sam-auto` takes three of their panels, by F1 and by the irrigated-only
  F1: Diamond Valley's NAIP and SPOT 6/7 and Mason Valley's SPOT 6/7. Each
  panel of 24 to 24d is drawn on its own input imagery
  (`Entry.per_layer_background`): NAIP 2022 in the top row, SPOT 6/7 2023 at
  bottom left and Sentinel-1 2024 at bottom right. Panels can give their F1
  (`Layer.metrics`, `Entry.title_f1`), and an entry can add a footer line
  with the Copernicus credit of the Sentinel images drawn or delineated on
  (`Entry.copernicus_credit`; entries 24 to 24d, 26, 26b, 27 and 28 to 28c).
  Multi-area panels can take a given window (`Layer.window`), their own run
  root (`Layer.run_root`) and one record of a score file
  (`Layer.metrics_record`); a footer names the images under a NAIP mosaic
  from other years, and an image with a `sam-auto` panel credits SAM 2.1
  (Meta, Apache-2.0) and the sam2 build it ran with. The footers carry the
  credits of the references drawn or scored: the DRI field boundaries of
  Nevada, the OpenET field boundaries with their source agencies (CLU for
  New York and Tennessee), and the DRI/NMOSE field boundaries (V18) for the
  SAM entry's New Mexico endpoint areas.
- Jacob C. Abramowitz (Earth System Science Center, The University of Alabama
  in Huntsville) added to the project authors and the software citation
  (CITATION.cff, pyproject.toml, README, docs), before Samapriya Roy. Jeremy
  Rapp's affiliation in CITATION.cff is now the Earth System Science Center,
  The University of Alabama in Huntsville.
- `agribound.evaluate.evaluate()` records PROJ's network setting when it
  computes the metrics: `proj_network` (new function `proj_network_enabled`,
  `pyproj.network.is_network_enabled()`). Scores against a reference in
  another datum depend on the transformation grids, which PROJ downloads
  only with the network on: example 20's Landsat 2018 output of agribound
  1.0 had 79 matches against the NMOSE WUCB polygons (NAD83) with the grids
  and 78 without. The key records the setting, not the use of the grids: a
  transformation keeps the pipeline it was built with, and geopandas keeps
  the transformation of each pair of CRSs for the life of the process, so
  after `pyproj.network.set_network_enabled(True)` a pair already
  reprojected in the process is still reprojected without the grids, while
  `proj_network` is `true` (pyproj 3.7.2, PROJ 9.7.1, geopandas 1.1.4: a
  NAD83 point in Lea County moved 0 m, against 0.376 m with a new
  transformation). Setting `PROJ_NETWORK=ON` before the process starts
  avoids this. `tools/make_gallery.py`
  now enables PROJ's network access while it renders (as `PROJ_NETWORK=ON`
  does; the previous setting is restored afterwards, and `--no-proj-network`
  keeps the environment's setting) and records it per entry (`proj_network`
  in `gallery_stats.json`). Without the grids, entry 20's NAD83 outlines of
  V18 were drawn about 0.3 m away, and 124,255 pixels of its PNG differed.
- Provenance records hold `elapsed_utc_s` beside `wall_s`: the seconds from
  `started_utc` to `finished_utc` by the system clock, to the millisecond.
  `wall_s` comes from `time.perf_counter()`, which does not advance while the
  computer sleeps, so a run during which a laptop slept looked shorter than
  its timestamps: in example 12's NAIP run of 2026-10-09, the GeoAI run's
  `wall_s` was 4,495 s and its timestamps 9,175 s apart (DINOv3: 3,929 s and
  16,011 s). Records written before have only the timestamps, to the second.
- `interior_polygons` (`agribound.engines.finetune._data`, used by the DINOv3
  and Prithvi engines) takes an optional `out_dir` for its label raster
  (`<stem>_fields_d<dilate_px>.tif`), which by default it still writes next
  to the prediction. A script that polygonised a cached DINOv3 prediction of
  example 14's 1.1 run had written two such files into that run's cache
  (their content unchanged).
- The example scripts log with the level name (`%(asctime)s %(levelname)s
  [%(name)s] %(message)s`; 26 scripts and their notebooks). Without it,
  agribound's WARNINGs could not be told from INFO lines in the logs: the 1.1
  run's logs hold 37 agribound WARNING lines, which its driver counted as 0.
- `tools/sync_notebooks.py` keeps cell ids. It gave every cell of a rewritten
  notebook a new random id, so a rewrite changed every cell (rewriting
  notebook 20 kept 0 of its 23 ids, although 19 cells were unchanged). A cell
  now keeps the id of the stored cell it matches (the same type and source,
  or an edited cell in the same place); a new cell gets an id derived from
  the notebook name and the cell index, so a notebook written afresh gets the
  same ids every time. Notebook content is unchanged.

## [1.0.1] - 2026-09-30

A bug-fix release. It changes the results of the `embedding` engine and of
SAM refinement for the same configuration, and it makes output reuse check the
study area and the release that made an output. 1.0.0 outputs of embedding or
SAM-refined runs are therefore no longer reused: see
[Results produced with agribound 1.0.0 that are affected](#results-produced-with-agribound-100-that-are-affected).
All other 1.0.0 outputs, caches and provenance records remain valid. The
release also adds example 22, re-runs the affected examples and renumbers the
gallery to 1.0.1.

### Results produced with agribound 1.0.0 that are affected

- **Embedding engine (`engine="embedding"`).** Clusters, and therefore
  polygons, change (see Fixed). Recompute with `overwrite=True` (CLI:
  `--overwrite`).
- **Any run with SAM refinement (`sam_refine=True`,
  `engine_params["sam_refine"]`, the embedding engine's own refinement, or
  `refine_boundaries`).** Polygons whose mask covered less than half of them
  are no longer replaced (see Fixed). Recompute, or pass
  `engine_params={"sam_min_coverage": 0}` to keep the 1.0.0 behaviour.
- Such 1.0.0 outputs now raise `FileExistsError` instead of being reused
  (pass `overwrite=True`); HPC tiles show as `stale` and need
  `agribound tiles run --overwrite`, and `tiles merge` of a finished 1.0.0 run
  that used them now refuses ("N tiles are not done") until they are re-run or
  `--allow-missing` is passed. Agent plans for such runs get a new plan
  directory. Examples 12, 14 and 15 have no `--overwrite`: delete their
  outputs first. Example 13 reuses its refined output by file name: pass
  `--overwrite`.

### Fixed
- **Embedding engine k-means depended on the pixel sample.**
  `clustering_method="kmeans"` now fits scikit-learn `KMeans(n_init=10)` (ten
  complete restarts, the lowest error kept) whatever the raster size, for the
  final fit and for the silhouette choice of k. 1.0.0, like 0.1.x, used
  `MiniBatchKMeans(batch_size=10000, n_init=3)` above 100,000 valid pixels
  and `KMeans(n_init=5)` below; with 31 seeds on example 15's TESSERA rasters
  `MiniBatchKMeans` reached the lower-error k = 5 solution in 30 of 62 runs,
  and not at the default seed 42. The engine metadata records
  `clusterer="KMeans"` and `kmeans_n_init=10`; cached cluster rasters are
  recomputed (cache version `embedding-clusters-v4`). On example 15's seed-42
  fit samples the error fell from 6,687,488 to 6,381,504 (TESSERA) and from
  2,942.7 to 2,733.2 (Google Satellite Embedding). In the 1.0.1 run the share
  of the TESSERA crop-filter area in polygons over 200 ha fell from 38.4 % to
  18.2 %, and the Google crop polygons follow 12 of 29 hand-checked centre
  pivots with a polygon of their own (IoU >= 0.8), against 4. Lower error is
  not better fields everywhere: the TESSERA solution also forms one connected
  21,452 ha green-crop region, which the representative-point study-area rule
  drops, so about 2,900 ha of Delineate-Anything fields there get no polygon.
  Results can still differ slightly with the number of OpenMP threads
  ([Reproducibility](https://montimaj.github.io/agribound/user-guide/reproducibility/#seeds)).
- **SAM refinement replaced a polygon covering several fields by one of them.**
  SAM returns one object per box prompt, and the refined polygon replaced the
  whole input, so the rest of a multi-field polygon (for example a centre
  pivot an embedding cluster merged with its neighbour) was left without a
  polygon. New `engine_params["sam_min_coverage"]` (default 0.5; 0 restores
  the 1.0.0 behaviour): a mask that, after the overlap trim, covers less than
  this share of its input polygon is not used; the polygon keeps its input
  geometry (`agribound:sam_refined` False, `agribound:sam_score` NaN) and is
  counted in the new `sam_stats["n_low_coverage"]`, also in the provenance
  record. The test also applies with `sam_overlaps="keep"`, and the ensemble
  engine accepts the key. Replayed on example 15's 1.0.0 inputs, refining
  every crop polygon on Sentinel-2 left 1 instead of 8 (TESSERA) and 2 instead
  of 14 (Google) of 29 checked pivots less than half covered; in the 1.0.1 run
  (with the new clusters) 2 and 4, with 6.6 % and 4.9 % of the input area
  removed, against 15.7 % and 24.1 % in 1.0.0. Example 13 (Delineate-Anything
  input, every mask covering at least 92 % of its polygon) is unchanged. On
  polygons that hold several fields a rejected mask may have matched one of
  them well: the [SAM Refinement](https://montimaj.github.io/agribound/user-guide/sam-refinement/#masks-that-cover-too-little-of-the-polygon)
  guide gives the measured trade-off, including Lea County (example 14 inputs).
- **Output reuse ignored the study area's contents and the release that made
  an output.** Provenance records gain `facts["aoi_fingerprint"]` and
  `facts["results_versions"]` (new `agribound._results.RESULTS_VERSIONS`;
  `config_hash` is unchanged). An existing output is no longer reused when its
  study-area file's geometry changed at the same path (with no study area:
  when the local raster's path, size or modification time changed), or when a
  component's results changed for the same configuration: 1.0.1 sets the
  `embedding` and `sam_refine` entries to 2. Other 1.0.0 outputs are still
  reused; when their study area is a file, or there is no study area, a
  WARNING says the fingerprint cannot be verified. `agribound tiles status`
  reports such tiles as `stale`, and agent plan directories include the
  results versions in their name.
- **Example 15, step 6** (SAM 2 on TESSERA dimensions, split variant): parts
  kept unrefined now get `agribound:sam_refined = False`. In 1.0.0 they had no
  value, so `fields_tessera_crop_sam2-tessera-split_2024.gpkg` stored the column
  as text (`"True"`/`"False"`/NULL), and `astype(bool)` reads `"False"` as True,
  so every polygon SAM was given counted as refined (1,672 instead of 261).
  `tools/make_gallery.py` now reads text flags as booleans. The step-6 masks
  are now also trimmed against the kept parts.
- **Archived 0.1.x gallery:** the Pampas screenshot shows the 0.1.x split
  variant with SAM 2 on TESSERA dimensions, not SAM 2 on Sentinel-2 as its
  caption (and the 0.1.x README) said. SAM 2 changed little of that layer:
  its unrefined polygons over 50 ha hold 66 % of the area.

### Added
- **Example 22: SPOT 6/7 panchromatic across the Global South.** Delineate-Anything
  v2 as released on 1.5 m SPOT-Pan composites of six study areas (3 km squares;
  6 km for the pivots): Cauvery Delta paddies (India), the Hetao irrigation
  district (Inner Mongolia, China), Mendoza vineyards (Argentina), the Mwea
  rice scheme (Kenya), Nile Delta strip plots (Egypt) and centre pivots in
  western Bahia (Brazil), with the crop filter as a separate step. It is also
  a new gallery image.
- **Example 15 (Pampas): a split variant of SAM 2 on Sentinel-2** for both
  embeddings (`fields_{google,tessera}_crop_sam2-s2-split_2024.gpkg`), the rule
  the example already applied to SAM 2 on TESSERA dimensions: multi-part
  polygons are split into parts (the Pampas crop layers have none) and parts
  over 50 ha are kept unrefined; refined masks are trimmed where they overlap
  the kept parts. It leaves no checked pivot (TESSERA) and 1 (Google, already
  missing after the crop filter) less than half covered. Examples 12-15 print
  `n_low_coverage`.
- **Documentation:** [SAM Refinement](https://montimaj.github.io/agribound/user-guide/sam-refinement/#masks-that-cover-too-little-of-the-polygon)
  describes the coverage test and polygons that cover several fields;
  [Engines](https://montimaj.github.io/agribound/user-guide/engines/#embedding-clustering-embedding)
  the k-means change and its measurements;
  [Reproducibility](https://montimaj.github.io/agribound/user-guide/reproducibility/#output-reuse)
  the new reuse checks and their limits, and the OpenMP thread count;
  [Satellite Sources](https://montimaj.github.io/agribound/user-guide/satellite-sources/#embeddings)
  the offset measured between TESSERA v1 and optical composites: in the
  Pampas, the TESSERA raster agribound builds (the same in 1.0.0 and 1.0.1),
  on the store's grid, sits about 9-10 m east of the Sentinel-2 and SPOT 6/7
  composites, and polygons of TESSERA clusters carry that offset; the 0.1.x
  mosaic of the v1 GeoTIFF tiles placed the data about 9 m west of the store's
  grid there, which cancelled most of it.
- `tools/make_gallery.py`: `Entry.window` (an explicit square window),
  `Entry.mark_windows` (numbered squares drawn on the panels) and
  `Entry.multi_area` (one panel per study area, each in its own CRS and window,
  with a world locator) and `Layer.crop_m` (a panel's own window side there).
- `tools/make_gallery_pampas_0.1x.py`: renders the 0.1.x Pampas README image next
  to the 0.1.x and 1.0.1 layers drawn in the same frame (a new gallery image).

### Changed
- **Gallery:** renumbered to 1.0.1. Examples 02, 05, 13, 14, 15 and 22 were
  re-run with 1.0.1; the other entries show 1.0.0 outputs, which 1.0.1 reuses
  unchanged (their code paths did not change). The Pampas embedding panels
  show the split variant; the Google Satellite Embedding and TESSERA
  comparison has a whole-study-area overview and three zoomed windows (two
  groups of centre pivots and an area of large merged polygons, re-chosen on
  the 1.0.1 layers); a comparison with the 0.1.x README image, in its own
  frame, is added.
- The SAM 3 backends are documented, and warn at load time, as untested in
  1.0.1 (they have still not been run end to end: the weights are gated).

## [1.0.0] - 2026-09-29

agribound 1.0.0 is a major release. It changes defaults, removes silent
fallbacks, and fixes defects that affected results produced with 0.1.x. Read
[Results produced with agribound <= 0.1.3 that are affected](#results-produced-with-agribound--013-that-are-affected)
before reusing earlier outputs, and the
[migration guide](https://montimaj.github.io/agribound/migration-1.0/) before
upgrading scripts.

### Results produced with agribound <= 0.1.3 that are affected

Each item below was confirmed in the 0.1.3 code (and, where stated, in cached
0.1.x outputs) and is fixed in 1.0.0. Outputs produced under the stated
conditions should be regenerated with 1.0.0.

**Inputs to the engines**

- **FTW two-window inputs were copies of the annual composite.** The composite
  cache key (`{source}_{year}_composite`) ignored `date_range`, so both FTW
  season windows were served the cached annual composite (window A and B files
  were byte-identical in the cached example outputs), and the windows were
  fixed to April/October regardless of hemisphere. *Affected:* every FTW run
  with a two-window model (the default `FTW_PRUE_EFNET_*` models). *Now:* each
  window is its own median composite around FTW's crop-calendar start and end
  of season, cached separately; a window without imagery raises unless
  `allow_annual_fallback=True`.
- **Landsat and HLS were on the wrong radiometric scale for FTW and
  Prithvi.** Landsat composites held raw Collection 2 digital numbers (no
  `2.75e-5` scale or `-0.2` offset) and HLS composites held 0-1 reflectance,
  while FTW (which divides by 3000) and Prithvi (Prithvi-EO-2.0 statistics)
  expect reflectance × 10000. *Affected:* FTW and Prithvi runs on `landsat`
  and `hls`, and `greenest`/`max_ndvi` Landsat composites (NDVI computed on
  digital numbers). *Now:* Sentinel-2, Landsat and HLS composites are
  reflectance × 10000, and every source declares its `value_scale`.
- **HLSS30 SWIR slots held red-edge bands.** The HLSS30 bands `B6`/`B7`
  (red-edge 2/3) were mapped onto the HLSL30 SWIR 1/SWIR 2 slots `B6`/`B7`, so
  HLS composites mixed red edge and SWIR in bands 6-7. *Affected:* HLS runs
  that use SWIR (Prithvi on HLS). *Now:* HLSS30 `B8A`, `B11`, `B12` map to
  `B5`, `B6`, `B7`.
- **Composites were exported in EPSG:4326** with metre scales, which gives
  pixels that are square in degrees and anisotropic on the ground (about
  8.6 m × 10 m at 31° S for Sentinel-2). *Affected:* all Earth Engine
  composites. *Now:* `export_crs="utm"` by default.
- **Composites were clipped to the study-area geometry.** When the study area
  was a set of field polygons (for example the reference boundaries, as in
  `examples/run_namoi_delineation.sh`), the engine saw imagery only inside
  those polygons, which imprints their outlines on the input and likely
  inflates accuracy against the same polygons. *Affected:* Earth Engine
  composites for non-rectangular study areas. *Now:* composites cover the
  bounding box; `aoi_selection` removes predictions outside the study area.
- **NAIP mosaics mixed neighbouring years in arbitrary order.** A NAIP
  composite for year Y used every image from Y-1 to Y+1
  (`calendarRange(year - 1, year + 1)`) and mosaicked them in collection
  order, which was never sorted by date, so an image from Y-1 or Y+1 could
  cover year-Y imagery where both existed; `date_range` was ignored for NAIP.
  *Affected:* NAIP runs where a neighbouring year also has imagery over the
  study area. *Now:* only 4-band images are used; without `date_range`,
  year-Y images are on top and Y-1/Y+1 images only fill gaps (newest on top
  within each group, with a WARNING naming the other years); with
  `date_range`, only images in that range are used, newest on top.
- **Google Satellite Embedding downloads through geoai-py were not usable.**
  The 0.1.x geoai download call omitted the CRS and resolution, so geoai
  reprojected to EPSG:4326 at a resolution of 10 degrees (reproduced with
  geoai-py 0.36.0: a 1 × 1 pixel raster for a small AOI), and geoai 0.36 used
  only the first intersecting tile; the Earth Engine fallback path added a
  65th `FILL_MASK` band that the embedding engine clustered. *Affected:*
  `google-embedding` runs. *Now:* Earth Engine export of the 64 bands
  (default) or a direct reader of the Source Cooperative mirror.
- **TESSERA downloads fail with geotessera < 0.10**, because the old data host
  (`dl2.geotessera.org`) returns HTTP 410; mosaics across UTM zones also
  failed. *Affected:* any new `tessera-embedding` run with 0.1.x. *Now:*
  `geotessera>=0.10.2,<0.11`, Zarr streaming per UTM zone, `tessera_version`.

**Engines**

- **Non-Sentinel-2 Delineate-Anything silently used a fallback.** When the
  Delineate-Anything repository could not be imported (it was looked up only
  at `~/VSCode/Delineate-Anything` and `/opt/delineate-anything`, and needs the
  GDAL Python bindings), the engine logged at INFO and ran a simplified YOLO
  path that passed RGB arrays to Ultralytics, which treats NumPy input as BGR
  (red and blue swapped), with no merging of the overlapping tiles' detections
  (duplicate and overlapping polygons) and a fixed confidence of 0.005.
  *Affected:* Delineate-Anything runs on sources other than Sentinel-2, and
  Sentinel-2 runs with a fine-tuned checkpoint, in environments without that
  repository. *Now:* explicit backends (`native` default, `reference`,
  `ftw`) and no fallback; the native backend passes BGR, merges tile pieces
  and suppresses duplicates.
- **GeoAI without a checkpoint loaded a US building-footprint model**
  (geoai's default detector weights), and with a checkpoint it read the band
  after each intended one (1-based indices used as 0-based) and applied a
  Sentinel-2 normalisation that differed from training. *Affected:* all
  0.1.x `geoai` results. *Now:* a checkpoint is required, and inference uses
  the same bands and stretch as the training chips.
- **Prithvi `fine_tune=True` did not fine-tune** (skipped at INFO), and
  `segment` mode could not run, so 0.1.x Prithvi results come from the
  pre-trained encoder (`embed`) or the PCA baseline. *Now:* Prithvi + UPerNet
  fine-tuning and `segment` inference are implemented.
- **Ensemble members inherited the ensemble's `engine_params`**, including a
  fine-tuned `checkpoint_path` (0.1.x fine-tuned a fallback engine for
  `engine="ensemble"` and handed its checkpoint to every member), and members
  of the same engine with different models reused the first member's cached
  output. *Affected:* ensembles with fine-tuning or with repeated engines.
  *Now:* members get only their own parameters and separate caches.

**Caching and reproducibility**

- **Caches ignored the study area and often the year.** Composites were keyed
  by source and year, and engine intermediates (for example
  `dinov3_segmentation_{source}.tif`, `geoai_output_{source}.geojson`,
  `embedding_clusters_{source}.tif`, `prithvi_clusters_{source}.tif`,
  `sam_refine_rgb_{source}.tif`, the Delineate-Anything work directory and the
  FTW prediction) by source only, so multi-year or multi-AOI runs sharing an
  output directory could reuse stale intermediates. In the cached example-14
  outputs, the 2021 and 2022 Landsat DINOv3 results equalled 2020, and SAM
  refinement for 2021 and 2022 used 2020 imagery. *Affected:* any run that
  shared an output directory with an earlier run of a different year, study
  area, model or parameters. *Now:* every intermediate is keyed by
  `agribound._cache.cache_key`.
- **Existing outputs were returned regardless of configuration.** Any
  non-empty file at `output_path` was loaded and returned without comparing
  settings. *Affected:* re-runs to the same path with changed settings. *Now:*
  reuse only when the provenance record's configuration hash matches.
- **Fine-tuned checkpoints were cached by engine (and model or source) only**,
  not by study area, year or reference file, so a second fine-tuning run in the
  same output directory reused the first checkpoint; fresh DINOv3 runs
  returned `last.ckpt` while cached runs picked a different checkpoint.
  *Now:* checkpoints are cached per fine-tuning key, and the best checkpoint on
  the validation chips is returned (validation loss for DINOv3 and Prithvi,
  mask IoU for GeoAI, Ultralytics fitness `best.pt` for Delineate-Anything).
- **Unseeded splits and samples.** The fine-tuning train/validation split, the
  DINOv3 training and the embedding engine's PCA and clustering samples were
  unseeded, so results varied between runs. *Now:* everything is seeded from
  `config.seed`; the split is spatial (`fine_tune_split="block"`) by default.
- **Delineate-Anything fine-tuning labels** turned the boundary-strip regions
  of the masks into additional `field` instances. *Now:* labels are built from
  the reference polygons.

**Post-processing**

- **Outputs held polygons smaller than `min_field_area_m2`.** The area filter
  ran only before smoothing and simplification, which shrink polygons, so
  polygons that these steps took below the minimum were written. In one
  1.0.0 test output made before the fix (`google-embedding`), 20 of 315
  polygons were below 2,500 m². *Affected:* 0.1.x outputs with smoothing or
  simplification on (the defaults). *Now:* the area filter runs again after
  smoothing, simplification and regularisation.

**LULC filter**

- **C3S cropland classes 11 (herbaceous) and 12 (tree/shrub) were ignored**;
  only 10, 20 and 30 counted. *Affected:* LULC-filtered runs before 2015
  outside the conterminous US, where C3S was used (in a 2010 test over Namoi,
  class 11 pixels outnumbered class 10 pixels about two to one). *Now:*
  classes 10, 11, 12, 20, 30.
- **NLCD routing dropped polygons in northern Mexico and southern Canada.** A
  centroid-in-bounding-box test sent those areas to NLCD, which has no data
  there; a missing mean became a crop fraction of 0 and every polygon was
  removed. *Now:* NLCD is used only where at least 90 % of the area has valid
  NLCD pixels, and missing values are NaN, kept and flagged.
- **LULC failures were swallowed.** Any error (no Earth Engine credentials,
  quota, timeout) was logged as a WARNING and the unfiltered polygons were
  written. *Affected:* outputs written while the filter was failing. *Now:*
  `lulc_on_error="raise"` by default.

**Evaluation**

- **Matching was not one-to-one.** Each reference field was matched to its
  best prediction and one prediction could match several reference fields
  (for example one prediction covering two fields at IoU 0.3 gave precision =
  recall = 1 with `iou_threshold=0.3`). *Affected:* 0.1.x evaluation metrics
  with merged predictions or low thresholds. *Now:* one-to-one greedy matching
  by default; `matching="many_to_one"` uses the 0.1.x matching rule. Other
  1.0.0 changes can still change the numbers slightly: invalid geometries are
  repaired (0.1.x never matched them, so they counted as unmatched), edges are densified before reprojection,
  null, empty and zero-area rows are dropped, and `delineate()` evaluates
  against the reference polygons in the study area (selected with the
  `aoi_selection` rule, or by intersection when it is `"none"`).

### Added

- `agribound.registry`: single source of truth for sources and engines
  (`SOURCE_REGISTRY`, `ENGINE_REGISTRY`, value scales, year ranges, label-free
  and fine-tunable flags, supported sources, references, notes).
- Reproducibility: `seed` (default 42) with `agribound._repro.seed_everything`;
  content-addressed cache keys (`agribound._cache`); a provenance record
  `<output>.provenance.json` for every run (`agribound.provenance`), with
  configuration hash, versions, platform, device, step timings, peak memory,
  engine metadata, facts and warnings. Every WARNING logged by an `agribound`
  logger during the run is recorded (identical messages once, at most 200;
  `warnings_not_recorded` counts the rest), as are the composite's
  `AGRIBOUND_*`/`TESSERA_*` tags (`facts.composite`), the image count, valid
  fraction and cloud mask of each FTW season window, and the path and
  SHA-256 of the cached FTW registry checkpoint.
- Configuration fields: `seed`, `export_crs`, `s2_cloud_mask`,
  `cloud_score_threshold`, `naip_resolution_m`, `tessera_version`,
  `tessera_variant`, `embedding_cache_dir`, `google_embedding_backend`,
  `aoi_selection`, `lulc_dataset`, `lulc_mode`, `lulc_on_error`,
  `lulc_nodata_policy`, `sam_refine`, `sam_backend`, `sam_model`,
  `sam_min_crop_px`, `sam_crop_padding`, `fine_tune_split`,
  `fine_tune_block_size_m`, `fine_tune_split_column`,
  `gee_service_account_key`, `gee_high_volume`, `gee_max_requests`,
  `gee_workload_tag`, `cache_dir`, `overwrite`, `provenance`;
  `AgriboundConfig.merged()`; study areas as `bbox:` strings or WKT.
- `agribound.build_composite()` and `agribound composite`: stage A only, so
  downloads and delineation can run on different nodes; `lulc_mode="raster"`
  prefetches the LULC raster.
- `aoi_selection` (`representative_point`, `intersects`, `clip`, `none`) to
  restrict predictions to the study-area geometry.
- Sentinel-2 Cloud Score+ masking; Landsat missions selected by the overlap
  of the date window with each mission's acquisition period (0.1.x selected
  them by year); NAIP exact-year priority and `date_range` support;
  valid-pixel check (`AGRIBOUND_VALID_FRACTION`); batch-export
  task records that prevent duplicate tasks; Google embeddings from the Source
  Cooperative mirror; TESSERA `v1`, `v1.1` and `v2`.
- LULC datasets CDL (`cultivated`) and explicit dataset choice; raster mode for
  offline filtering; `lulc:*` output columns.
- Delineate-Anything v2 (`large_v2`, default) with pinned weights and SHA-256
  checks; `native`, `reference` and `ftw` backends.
- FTW: default model from the ftw-tools registry, crop-calendar season
  windows, `list_ftw_models()` / `agribound list-ftw-models`,
  `FTWEngine.stage_inputs()`.
- Prithvi `segment` mode and Prithvi + UPerNet fine-tuning (terratorch);
  DINOv3 LoRA and decoder-only options.
- SAM refinement stage for every engine except `embedding`, with backends
  `sam2`, `sam2.1`, `sam3` (Meta, CUDA + triton) and `sam3-hf` (transformers;
  both SAM 3 backends are untested: not run end to end because the weights are
  gated, and a WARNING is logged when one is selected);
  `agribound:sam_refined` column and `sam_stats`;
  `engine_params["sam_overlaps"]` (`"trim"`, default, or `"keep"`; see
  Changed).
- `evaluate()`: one-to-one matching, area-weighted metrics, over- and
  under-segmentation means, Hausdorff and mean boundary distances, boundary
  precision/recall/F1 and coverage within a tolerance, per-stratum and
  per-size-class metrics, bootstrap confidence intervals; `evaluate_frame()`,
  `pixels_per_field()`; `agribound evaluate`.
- Engine `prefetch()` and `agribound prefetch` for offline nodes.
- `agribound.hpc` and `agribound tiles` (`make`, `run`, `status`, `merge`,
  `prefetch`, `matrix`, `region`, `gee-project`): tiling with halos, two-phase
  stage/compute runs, no-data tiles, representative-point merge;
  `examples/hpc/` (Slurm scripts and NSF ACCESS profiles) and
  `examples/regions/` (16 regions and a region driver).
- `agribound tiles gee-project` prints the Earth Engine project that runs will
  use (`--project` or the base configuration's `gee_project`, then
  `GEE_PROJECT`, gcloud, and the credentials file's `project_id`) without
  contacting Earth Engine, prints nothing when no run uses Earth Engine, and
  exits with status 1 and instructions when one is needed and none is found.
  `examples/run_region_delineation.sh` calls it before it validates or runs
  anything (also with `--dry-run`) and passes the project to every run;
  `examples/hpc/submit_region.sh` calls it before tiling or the first
  `sbatch`, and exports the project to the jobs as `GEE_PROJECT` when
  `BASE.yaml` has no `gee_project`. The region files and HPC profiles name
  no Earth Engine project: pass your own with `--gee-project`, `GEE_PROJECT`,
  gcloud or a service-account key.
- Optional agent layer (`agribound[agent]`): `agribound.agent()`,
  `agribound agent` and `agribound mcp serve`, with read-only tools, one
  proposed plan, a human confirmation gate bound to the plan hash, at most one
  execution per session, and a JSON transcript. Proposals that contain
  control or format characters, or that set a reserved field (output
  location, caches, `overwrite`, `provenance`, credentials, `gee_project`,
  `embedding_cache_dir`), are refused; values on the review screen are
  escaped; plans that change `usgs_service_url`, `export_method` or
  `gcs_bucket` carry a warning and name the host, bucket or Drive. The
  reviewer is asked on every execution, and unused approvals are revoked
  when a plan fails the hash check. `mcp serve --transport streamable-http`,
  which has no authentication, is refused (exit status 2) with
  `--allow-execute` or a non-loopback `--host` unless
  `--allow-unauthenticated-http` is given (then a WARNING is logged).
- Published FTW polygon query: `by-admin-conf` layout (default),
  `min_confidence`, `keep_null_confidence`, file pruning by row-group
  statistics; the `agribound:clipped` column; a `<output>.provenance.json`
  record for queries written to a file (`provenance=True`: parameters,
  backend, source, duplicate/clipped/returned counts, versions).
- Output columns `agribound:compactness`, `agribound:version`,
  `agribound:run_id`; `id` unique across runs.
- CLI: `--config` combined with explicit flags, `--dry-run`, `--engine-param`,
  and flags for the new configuration fields.
- `environment-gfm.yml` and the `all-gfm`, `dinov3`, `sam3`, `embedding` and
  `agent` extras.
- Examples 18-21 (agent, HPC tiling, stratified evaluation, published FTW
  audit).
- GeoAI fine-tuning logs a WARNING (also in the training metadata and the
  provenance record) when the best validation IoU is below 0.1 or there are
  fewer than 10 training chips.
- `.gitattributes`: `*.sh`, `*.sbatch` and `examples/hpc/profiles/*.env` keep
  LF line endings in every checkout, also on Windows with
  `core.autocrlf=true`.
- Documentation: migration guide, reproducibility, evaluation, HPC, agent and
  SAM refinement pages.
- Documentation gallery regenerated from the 1.0.0 example runs. Examples 01
  and 12 were not run end to end; 12's NAIP runs were, and have a gallery
  entry. `tools/make_gallery.py` draws each output on
  a composite from the run and names it under the map, together with the
  imagery window and the model version. The composite is usually the engine's
  input. The exceptions are FTW, shown on season window A (the first of its
  two inputs); the ensemble entry, shown on Delineate-Anything's input; and
  the Pampas embedding entries, shown on the Sentinel-2 composite SAM 2 read,
  including under the unrefined embedding clusters. The images (3000 px wide)
  and the facts the captions quote (`gallery_stats.json`) are in
  `assets/gallery_1.0/`. New entries for examples 12 (NAIP runs), 13, 16,
  19 and 20.
- The README, the documentation home page, the gallery and the 1.0.0 release
  post embed 1600 px WebP previews (`assets/gallery_1.0/preview/`), each
  linked to its 3000 px PNG. Locator insets that show India draw it from the
  Survey of India State Map outline (via DataMeet); the other boundaries
  are from Natural Earth.
- Workflow diagram `assets/agribound_workflow_1.0.{png,svg}`, rendered by
  `tools/make_workflow_diagram.py`. The script also writes
  `agribound_workflow_1.0.pdf`, which `.gitignore` (`*.pdf`) keeps out of the
  repository.
- Resolution comparisons in the examples: example 20 runs Delineate-Anything
  v2 on Landsat (30 m), Sentinel-2 (10 m), SPOT 6/7 (6 m) and NAIP (1 m) of
  2018 and evaluates each against NMOSE (`--no-resolution` or `--predicted`
  skips it), and compares the pre-trained FTW and Delineate-Anything engines
  with and without the crop filter on Sentinel-2 2019 (`--no-lulc-comparison`
  or `--predicted` skips it); example 15 adds Delineate-Anything v2 on its
  Sentinel-2 composite and on SPOT 6/7 2023; example 03 adds
  Delineate-Anything v2 on SPOT 6/7 2023. SPOT runs need restricted Earth
  Engine access; without it the SPOT run fails, the error is printed and the
  script continues.

### Changed

- GeoAI fine-tuning sizes its chips from the reference fields by default
  (1.25 × the 90th-percentile field bounding-box side, rounded up to a multiple
  of 32 px and clamped to 256–1024 px; `engine_params["chip_size"]` overrides)
  instead of a fixed 256 px. GeoAI infers on windows of the chip size, so at
  NAIP 1 m the fixed 256 px chip split every centre pivot into window-sized
  pieces (in-sample F1 0.01 in example 12's NAIP run). A
  WARNING is logged when more than 10 % of the reference fields are larger than
  the chip.
- The GeoAI engine joins instances that a field was split into at the edges of
  the overlapping inference windows (geoai keeps partial detections from
  neighbouring windows) and fills gaps of up to 2 px along those edges
  (`engine_params["merge_window_seams"]`, default on; `seam_min_px`,
  `seam_max_gap_px`); counts are recorded in `engine_meta`. In example 12's
  NAIP run, in-sample F1 rose from 0.25 to 0.48.
- Python >= 3.12; `Development Status :: 4 - Beta`.
- Packaging: licence metadata in PEP 639 form (`license = "Apache-2.0"`,
  `license-files = ["LICENSE"]`, no licence classifier), so building needs
  `setuptools>=77`; Ruff `target-version = "py312"`.
- `ftw` extra: `ftw-tools>=2.0.0b5,<3` (pre-release on PyPI); `all` no longer
  includes `prithvi` (conflicting `lightning` pins); `tessera` extra pins
  `geotessera>=0.10.2,<0.11`; `shapely>=2.1`.
- Default composite CRS is the UTM zone of the study area (`export_crs="utm"`),
  not EPSG:4326; composites cover the study-area bounding box without polygon
  masking.
- Landsat and HLS composites are reflectance × 10000.
- `delineate()` reuses an existing output only when its provenance record
  matches the configuration; otherwise it raises `FileExistsError`
  (`overwrite=True` re-runs).
- Named arguments and keyword arguments of `delineate(config=...)` are applied
  on top of the configuration.
- Configuration validation: engine/source compatibility (including the
  ensemble's default members), year ranges, enums, unknown keys and
  non-finite sizes (`inf`, `NaN`) fail when the configuration is created.
- Delineate-Anything: confidence parameter `conf_threshold` (the old
  `confidence` raises); default model `large_v2`; no automatic routing of
  Sentinel-2 through FTW.
- GeoAI and DINOv3 require a checkpoint.
- Ensemble: members receive only their own `engine_params`; `vote_count` is
  the maximum agreement inside each polygon (the `min_votes` constant is in
  its own column); members that returned no polygons are still left out of
  the vote, now with a WARNING (0.1.x logged it at INFO); the vote rule
  `max(min(2, n), ceil(vote_threshold × n))` is unchanged.
- SAM refinement: `sam_batch_size` is a decoder batch size (default 32; it was
  a logging interval); fields are encoded at about native scale in shared
  windows (`sam_window_px`, default 1024) instead of per-field upsampled crops;
  fields outside the raster are not refined. Refined masks no longer take
  area from other polygons (`engine_params["sam_overlaps"]="trim"`, default;
  where two refined masks grew over the same area, the higher SAM score keeps
  it); `"keep"` keeps the masks as SAM returned them, as 0.1.x did. This is a
  trade-off: on a Namoi test area (Delineate-Anything + SAM 2, 16 of 230
  polygons refined) overlap fell from 3.59 ha (`"keep"`) to 0.20 ha (0.18 ha
  without SAM), but the best IoU of the one reference field that SAM had
  improved fell from 0.773 to 0.676 (0.636 without SAM).
- LULC filter: errors raise by default; missing values are NaN and kept;
  Dynamic World from 2016; NLCD routing by NLCD coverage; C3S classes 10, 11,
  12, 20, 30; Annual NLCD 1985-2025.
- `evaluate()` default matching is one-to-one.
- `metrics:perimeter` is geodesic; `determination:datetime` is the last day
  of the imagery the engine read (for FTW two-window models the end of the
  later season window, which can be in the next year; also for FTW ensemble
  members); `id` is `"<run_id>-<n>"`.
- `min_field_area_m2` is applied again after smoothing, simplification and
  regularisation (`facts.postprocess.min_field_area_applied`), so outputs can
  have fewer polygons. Output reuse compares only the configuration hash,
  which this does not change: pass `overwrite=True` to recompute an output
  written before this change.
- DINOv3 fine-tuning deletes geoai-py's unused `last.ckpt` once the best
  checkpoint is known; a ViT-L/16 full fine-tuning checkpoint is about
  3.7 GB, and training needs about twice that in free space.
- `delineate`, `prefetch` and `composite` print a one-line error naming the
  extra to install when an optional dependency is missing (`-v` logs the
  traceback).
- Earth Engine authentication: service-account key, then
  `AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY`, then stored credentials, then
  Application Default Credentials; never interactive in batch jobs.
- Earth Engine project: without `gee_project`, `GEE_PROJECT` or a `gcloud`
  project, the `project_id` of the credentials file is used
  (`gee_service_account_key`, else `AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY`, else
  `GOOGLE_APPLICATION_CREDENTIALS`).
- `query_ftw` default layout `by-admin-conf` (years 2024-2025).
- `query_ftw(deduplicate=True)` (the default) removes repeated polygons per
  (prediction year, normalized geometry); the published `id` is not unique
  per polygon in `by-admin-conf`. 0.1.x keyed on the `field_id` or
  `geometry_hash` column when present, else on a geometry hash without the
  year.
- `agribound.engines.finetune` is a package (import path unchanged).
- The manuscript citation is now "Measuring what geospatial AI delivers for
  policy-grade agricultural field boundaries" (in preparation for *Remote
  Sensing of Environment*).
- The 0.1.x gallery screenshots moved to `assets/gallery_0.1x/`; the 0.1.x
  gallery is kept as the archived page `docs/gallery-0.1x.md`, with the same
  images and some captions corrected (FTW model names, years that did not
  match the 0.1.x example scripts, the Prithvi remark).

### Fixed

- The fine-tuning cache key now includes the engine's default chip-size rule,
  so a changed default no longer reuses a checkpoint trained on chips of
  another size.
- All defects listed in
  [Results produced with agribound <= 0.1.3 that are affected](#results-produced-with-agribound--013-that-are-affected).
- `agribound delineate --config file.yaml` required `--study-area` and ignored
  every other flag.
- The FTW extra resolved to ftw-tools 1.4.3, whose API is incompatible with
  the FTW engine.
- `export_method="gdrive"/"gcs"` passed a pseudo-path to the engines; it now
  stops with `ExportTaskStartedError` and records the task.
- Earth Engine authentication could block batch jobs on an interactive prompt.
- NAIP composites failed for years with RGB-only images.
- `is_gee_source()` did not include `spot-pan`.
- The output file was written in `output_format` even when the extension of
  `output_path` implied another format; conflicting settings now raise.
- `merge_polygons` did not merge frames with a non-default index.
- `query_ftw(clip=True)` kept the published whole-polygon `metrics:area` and
  `metrics:perimeter` on the polygons it clipped (0.1.x did the same when the
  source had these columns), so area sums overstated the area in the AOI (in
  a Namoi query for 2024, 1869.6 ha against 1516.9 ha of clipped geometry).
  They are now recomputed from the clipped geometry (EPSG:6933 area,
  geodesic perimeter), and a clip that returns a geometry collection keeps
  its polygonal part instead of dropping the polygon.
- Polygon regularisation failed silently with geoai-py 0.36 (the input was
  returned unchanged); it now uses the geoai-py 0.43.1 API.
- DINOv3 `small`/`base` backbones could not load the ViT-L SAT-493M weights;
  they now need `weights_path`, and the error says so.
- Documentation: many statements that did not match the code (for example
  "Dask-based parallelism", "LoRA" DINOv3 fine-tuning, `--simplify` "in
  pixels", NLCD and C3S year ranges, a SAM "batch size", the FTW country
  count). The Sentinel-2 coverage text now says that the Earth Engine
  collection holds L2A images from 2015-07-04 (agribound still accepts 2017
  onwards).

### Deprecated

These spellings are still accepted in 1.0.0; prefer the new ones:

- `engine_params["sam_refine"]` → `sam_refine`;
  `engine_params["sam_model"]` → `sam_model`.
- Delineate-Anything `engine_params["model_size"]` → `da_model`.
- GeoAI `engine_params["model_path"]` → `checkpoint_path`.
- Prithvi `engine_params["patch_size"]` → `tile_size`.

### Removed

- Python 3.10 and 3.11 support.
- The `dask[distributed]` and `fiona` dependencies (neither was imported).
- Silent fallbacks: the Delineate-Anything YOLO fallback, GeoAI's default
  (building) weights, fine-tuning of a substitute engine, the annual-composite
  fallback for FTW windows (now opt-in), and swallowed LULC errors (now
  opt-in with `lulc_on_error="warn"`).
- The hard-coded Delineate-Anything repository paths (use
  `engine_params["da_repo"]` or `AGRIBOUND_DA_REPO` with
  `backend="reference"`).
- The Delineate-Anything parameters `confidence` and `minimal_confidence`
  (use `conf_threshold`).
- The 0.1.x cache layout (old cache files are not read).
- The 0.1.x workflow figure `assets/agribound_workflow.png` (replaced by
  `assets/agribound_workflow_1.0.*`).

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
- Fine-tuning support for DA (YOLO), GeoAI (Mask R-CNN), DINOv3, Prithvi (terratorch); the DINOv3 trainer ran full fine-tuning (the 0.1.0 notes said LoRA), and the Prithvi fine-tuning path was not reachable (see 1.0.0)
- CLI (`agribound delineate`) and Python API (`agribound.delineate()`)
- 16 example scripts and Jupyter notebooks
- MkDocs documentation site
- Pytest suite (unit + integration)
