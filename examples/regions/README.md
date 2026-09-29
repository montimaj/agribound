# Region definitions

One YAML file per region, read by `examples/run_region_delineation.sh` (through
`agribound tiles region`) and `agribound.hpc.regions.load_region`. Each file
has a `run` block with the driver's defaults (years, sources, engines, tile size,
halo, reference, TESSERA version). The rest of the file records how the region
was checked: WorldCover cropland fraction, image counts per source, NAIP and
USGS NAIP Plus vintages, SPOT scene years, AlphaEarth years, TESSERA tile
coverage, reference data and FTW notes. `run_<name>.sh` runs the driver with
that region's defaults, and passes extra arguments on:

```bash
examples/regions/run_beauce_fr.sh --test --gee-project <project>     # small test box, here
examples/regions/run_beauce_fr.sh --mode slurm --profile delta --gee-project <project> \
    --gee-service-account-key /path/ee.json --dry-run                 # print the Slurm jobs
```

The region files name no Earth Engine project: use your own. The driver takes
it from `--gee-project`, else `$AGRIBOUND_GEE_PROJECT` or `$GEE_PROJECT`, else
the gcloud configuration, else the `project_id` of the service-account key. If
a run needs Earth Engine (every run with the LULC filter, which is on by
default) and none is found, the driver stops before it runs or submits
anything. The image counts in the files were queried with the author's project;
the public collections give the same counts with any project.

| Region | Title | bbox area (km2) | WorldCover cropland | Tiles | Default years | FTW country | TESSERA >= 90 % tiles | Reference |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| `andalusia_guadalquivir_es` | Guadalquivir valley, Andalusia, Spain | 44,132 | 0.20 | 135 (20 km, halo 1000 m) | 2023, 2024 | yes | v1: 2017-2025 | - |
| `beauce_fr` | Beauce plain, France | 6,936 | 0.72 | 20 (20 km, halo 1000 m) | 2023, 2024 | yes | v1: 2017-2025 | - |
| `central_valley_ca_us` | Central Valley, California, USA | 222,621 | 0.09 | 609 (20 km, halo 1500 m) | 2022, 2024 | no | v1: 2017-2025 | - |
| `iowa_corn_belt_us` | Iowa (Corn Belt), USA | 187,046 | 0.67 | 510 (20 km, halo 1000 m) | 2023, 2024 | no | v1: 2017-2025 | - |
| `lea_county_nm_us` | Lea County, New Mexico, USA | 12,432 | 0.02 | 144 (10 km, halo 1200 m) | 2022, 2024 | no | v1: 2017-2025 | local |
| `mato_grosso_br` | Central Mato Grosso soybean belt, Brazil | 216,083 | 0.29 | 575 (20 km, halo 3000 m) | 2023, 2024 | yes | v1.1: 2017-2025 | - |
| `mekong_delta_vn` | Mekong Delta, Vietnam | 74,326 | 0.35 | 196 (20 km, halo 1000 m) | 2023, 2024 | yes | not a default source | - |
| `mississippi_alluvial_plain_us` | Mississippi Alluvial Plain (Delta), USA | 85,660 | 0.44 | 262 (20 km, halo 1500 m) | 2023, 2024 | no | v1: 2017-2025 | - |
| `murray_darling_riverina_au` | Riverina (Murray-Darling Basin), New South Wales, Australia | 50,635 | 0.36 | 144 (20 km, halo 2000 m) | 2023, 2024 | no | v1.1: 2017-2025 | - |
| `namoi_catchment_au` | Namoi catchment, New South Wales, Australia | 60,761 | 0.30 | 178 (20 km, halo 2000 m) | 2023, 2024 | no | v1.1: 2017-2025 | local |
| `new_mexico_statewide_us` | New Mexico statewide, USA | 350,775 | 0.02 | 950 (20 km, halo 1500 m) | 2022, 2024 | no | v1: 2017-2025 | local |
| `north_china_plain_cn` | North China Plain, China | 246,865 | 0.56 | 673 (20 km, halo 1000 m) | 2024 | no | v1: 2024 | - |
| `pampas_ar` | Pampas, Argentina | 229,850 | 0.65 | 642 (20 km, halo 2000 m) | 2024 | no | v1: 2024 | - |
| `punjab_in` | Punjab, India | 96,547 | 0.70 | 255 (20 km, halo 1000 m) | 2024 | yes | v1: 2024 | - |
| `vinnytsia_ua` | Vinnytsia oblast, Ukraine | 41,246 | 0.64 | 121 (20 km, halo 2000 m) | 2023, 2024 | no | v1: 2017-2025 | - |
| `western_kenya_ke` | Western Kenya | 31,511 | 0.24 | 90 (20 km, halo 1000 m) | 2023, 2024 | yes | v1: 2017-2025 | - |

Notes on the columns:

- **WorldCover cropland**: ESA WorldCover v200 (2021) class 40, as the share
  of 5,000 seeded random 10 m points in the bbox. The GEE check ran on
  2026-09-26/27. Class 40 is annual cropland only; orchards and vineyards map
  to tree or shrub cover.
- **Tiles**: `agribound.hpc.tiles.make_tiles` on a UTM grid, with cores clipped
  to the bbox. The halo is an agribound choice: fields longer than the halo
  that cross a tile core boundary can be truncated. With the default halos,
  the New Mexico, Lea County and Namoi reference data each contain fields
  longer than the halo (see each file's `halo_note` and notes); check
  `n_reaching_halo_edge` in the merge summary.
- **FTW country**: whether the country is in `ftw_tools.settings.ALL_COUNTRIES`.
- **TESSERA**: the years in which the region's `tessera_version` has at least
  90 % of the tiles intersecting the bbox (manifests of 2026-09-26; the
  expected count includes water, so coastal boxes stay below 100 %).
- **Reference**: `local` means the region uses a local, undistributed file
  (examples/namoi_polygons.geojson, the NMOSE shapefile).

Admin-boundary boxes came from TIGER/2018 (US states and counties) and FAO GAUL
2015 level 1 (Punjab, Vinnytsia), rounded outwards to 1e-4 degrees; the New
Mexico box is the union of the TIGER state box and the extent of the NMOSE
polygons, and the Namoi box is the extent of the Namoi reference polygons.
The other boxes are hand-drawn; see each file's `bbox_source`. Boxes can cover more
than the region they are named after, and the WorldCover fraction is computed
over the whole box: the New Mexico box is about 7 % Texas and 3 % Mexico, the Punjab box
18 % Pakistan, the western Kenya box 15 % Uganda, the Vinnytsia box 8 %
Moldova and the Mekong Delta box 8 % Cambodia (see each file's notes). Clip
results (and evaluations against a reference that stops at the border) to the
region itself.

Default years stay within the TESSERA coverage of the region's default
`tessera_version`: Punjab, Pampas and the North China Plain default to 2024
only, and the Mekong Delta has no TESSERA source by default. Tiles of a bbox
that have no input data (open water; outside a source's coverage) are
recorded as `no-data` and merged as empty; see `examples/hpc/README.md`.

`namoi_catchment_au.yaml` and `run_namoi_catchment_au.sh` are distributed:
`.gitignore` excludes them from its Namoi rules (`Namoi*` and `*namoi*.sh`,
which keep the user's own Namoi data and scripts local). The Namoi reference
polygons the YAML points to (`examples/namoi_polygons.geojson`) are not
distributed; without them `run_region_delineation.sh` logs a WARNING and runs
without a reference.
