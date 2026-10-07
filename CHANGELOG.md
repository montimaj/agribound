# Changelog

All notable changes to agribound will be documented in this file.

## [Unreleased]

### Results produced with agribound 1.0.1 that are affected

- **Delineate-Anything fine-tuning (`engine="delineate-anything"` with
  `fine_tune=True`).** The training recipe changed (see Fixed), so the
  fine-tuned weights, and the polygons delineated with them, change. Such
  outputs now raise `FileExistsError` instead of being reused (results
  version `delineate_anything_finetune` 2; pass `overwrite=True`, CLI
  `--overwrite`), and cached fine-tuning runs are trained again (the recipe
  version is part of their cache key). Runs that load a fine-tuned checkpoint
  through `engine_params["checkpoint_path"]` are unaffected. Agent plans for
  such runs get a new plan directory (it includes the results versions).
  `engine_params={"yolo_warmup_bias_lr": 0.1}` restores the 1.0.1 recipe.
  The other trainers (GeoAI, DINOv3, Prithvi) are unchanged.

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
  SIGPAC recintos. Gallery entries. Thanks to Jacob
  Abramowitz (The University of Alabama in Huntsville), who asked about tree
  crops and pointed to the RSPO concession maps and the subdivided Twifo
  estate.
- `engine_params["yolo_warmup_bias_lr"]` (Delineate-Anything fine-tuning): the
  warmup learning rate of the bias parameters (default 0; see Fixed).
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
  without the option share. Agent plans warn when it is set, and, when it is
  not, where the filter uses Dynamic World or C3S (`lulc_dataset`
  `"dynamic_world"` or `"c3s"`, or `"auto"` for a study area outside the
  conterminous-US envelope); `recommend_configurations` notes the same for
  such study areas, and the agent's system prompt lists tree crops among the
  limitations to report. Thanks to Jacob Abramowitz (The University of
  Alabama in Huntsville) for asking about tree crops and pointing to the RSPO
  GeoRSPO concession boundaries.
- **Documentation:** [Satellite Sources](https://montimaj.github.io/agribound/user-guide/satellite-sources/#landsat-panchromatic-landsat-pan)
  describes the `landsat-pan` mission rule and composite tags, and
  [Tree crops](https://montimaj.github.io/agribound/user-guide/satellite-sources/#tree-crops)
  the new LULC option, its measurements and its trade-off.
  [Fine-Tuning](https://montimaj.github.io/agribound/user-guide/fine-tuning/)
  describes the Delineate-Anything recipe change and recommends a smaller
  `yolo_lr0` for small training sets.
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

### Fixed

- Delineate-Anything fine-tuning used Ultralytics' default
  `warmup_bias_lr=0.1`, which is meant for SGD, with AdamW. Ultralytics'
  `optimizer="auto"`, whose AdamW choice the recipe follows (as documented),
  sets 0 for AdamW, and so does agribound now. On 62 training chips of an oil
  palm estate (example 23), where the released weights score a validation
  mask mAP50 of 0.17, a test at `lr0=1e-4` (momentum 0.9) scored 0.46 after 20
  epochs without the bias warmup and 0.32 with it; with `yolo_lr0=1e-4` the
  new recipe scores 0.47 after 20 epochs. At the default `lr0` (0.002) the
  pretrained weights are wrecked with either setting: the run scored 0.00 with
  the 1.0.1 recipe, and with the new one 0.02 after the first epoch (the
  checkpoint Ultralytics keeps) and 0.00 from the third epoch on. The
  fine-tuning guide now recommends a smaller `yolo_lr0` for small training
  sets.

### Changed

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
  missions' image counts in its message; it counted all three, including
  Landsat 7 in 2013-2024.
- Delineate-Anything has a `landsat-pan` source note (`engine_notes`, used by
  the agent tools): 15 m Landsat PAN composites are outside its 0.25-10 m
  training range, and the single band is replicated to grey R, G, B.
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
