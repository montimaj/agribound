"""
15 — Label-Free Pipeline from Embeddings to SAM 2 (Pampas, Argentina)

Chains label-free steps by hand, without reference boundaries or model
training, near Pergamino (Buenos Aires Province):

    1. Cluster Google Satellite Embedding (64-D) and TESSERA v1 (128-D)
       embeddings of 2024 with the ``embedding`` engine (no LULC filter).
    2. Keep the crop polygons with the LULC filter
       (``agribound.postprocess.lulc_filter.filter_by_lulc``; Dynamic World,
       the mean ``crops`` probability of the 2024 annual median, >= 0.3).
    3. Build an October 2024 Sentinel-2 composite (``agribound.build_composite``).
    4. Refine the crop polygons of each embedding with SAM 2 on that composite.
    5. Refine the TESSERA crop polygons with SAM 2 on three TESSERA embedding
       dimensions used as a pseudo-RGB image (``engine_params["sam_rgb_bands"]``;
       experimental: embedding dimensions are not colours).
    6. A variant of 5: multi-part polygons are split into parts, so each part
       gets its own box prompt, and polygons larger than 50 ha are kept
       unrefined (a heuristic: one box prompt around several fields asks SAM
       for one object).
    7. For comparison, the label-free Delineate-Anything v2 engine
       (``large_v2``) on the Sentinel-2 composite of step 3 and on SPOT 6/7
       (6 m, 2023), with the same minimum area and LULC crop filter as the
       embedding polygons. SPOT 6/7 is restricted to select Earth Engine users;
       without access that run fails and the script continues.

The pipeline can do steps 1+5 in one call (``engine="embedding"`` with
``sam_refine=True`` and ``engine_params["sam_rgb_bands"]``: SAM on the
embedding raster's pseudo-RGB), but it refines before the LULC filter; this
script filters first. Refinement on a Sentinel-2 composite (step 4) has no
single pipeline call: it needs the separate ``refine_boundaries`` call shown
here. Nothing here is evaluated
against reference boundaries: the table compares polygon counts and areas
only.

Years: 2024 for all steps except the SPOT run. TESSERA v1 is near-global only
for 2024; every tile of this study area exists for 2024 (v1 manifest,
2026-09-27). The embeddings are annual, the Sentinel-2 composite covers
October only. AIRBUS/SPOT6_7 ends on 2023-11-15, so the SPOT run uses 2023
(6 scenes cover the whole study area; queried 2026-09-28).

Study area: a pentagon with a bounding box of about 29 x 32 km. The TESSERA
builder assembles the 128-band float32 mosaic of that box in memory (~4.8 GB).

Estimated runtime: steps 1-6 took 22 minutes on an Apple M2 Max (MPS; SAM 2
on the CPU) on 2026-09-28; step 7 adds the SPOT download and two
Delineate-Anything runs.

Prerequisites:
    pip install "agribound[gee,samgeo,tessera,delineate-anything]"
    agribound auth --project YOUR_GEE_PROJECT
    Run from the repository root: python examples/15_pampas_semi_supervised.py
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

import agribound

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

warnings.filterwarnings("ignore", category=FutureWarning, module=r"geedim\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"geedim\..*")
warnings.filterwarnings("ignore", message=".*organizePolygons.*")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("geedim").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/pampas_semi_supervised")
STUDY_AREA = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-60.552058, -33.789222],
                        [-60.274083, -33.734778],
                        [-60.242050, -33.903368],
                        [-60.363950, -33.995060],
                        [-60.505737, -34.015376],
                        [-60.552058, -33.789222],
                    ]
                ],
            },
            "properties": {"name": "Pampas (Pergamino, Buenos Aires)"},
        }
    ],
}
YEAR = 2024
SPOT_YEAR = 2023  # AIRBUS/SPOT6_7 ends on 2023-11-15
S2_DATE_RANGE = (f"{YEAR}-10-01", f"{YEAR}-10-31")
MIN_AREA = 5000  # m^2
CROP_THRESHOLD = 0.3  # LULC crop value threshold
SAM_MODEL = "large"  # SAM 2 size alias (facebook/sam2-hiera-large)
TESSERA_PSEUDO_RGB = [1, 2, 3]  # 1-based embedding dimensions shown to SAM
MAX_REFINE_AREA_HA = 50.0
EMBEDDINGS = {
    "google": {"source": "google-embedding"},
    "tessera": {"source": "tessera-embedding", "tessera_version": "v1"},
}


def area_ha(gdf):
    """Total polygon area in hectares (equal-area EPSG:6933)."""
    from agribound.io.crs import get_equal_area_crs

    if len(gdf) == 0:
        return 0.0
    return float(gdf.geometry.to_crs(get_equal_area_crs()).area.sum() / 10000)


def refine(gdf, raster_path, config):
    """SAM 2 refinement plus the pipeline's area filter, smoothing and simplification."""
    from agribound.engines.samgeo_engine import refine_boundaries
    from agribound.postprocess import filter_polygons
    from agribound.postprocess.simplify import simplify_polygons, smooth_polygons

    refined = refine_boundaries(gdf, raster_path, config)
    stats = refined.attrs.get("sam_stats", {})
    print(
        f"    SAM 2: refined {stats.get('n_refined')} of {stats.get('n_total')} "
        f"(too small: {stats.get('n_skipped_small')}, failed: {stats.get('n_failed')})"
    )
    refined = filter_polygons(refined, min_area_m2=MIN_AREA)
    refined = smooth_polygons(refined, iterations=3)
    return simplify_polygons(refined, tolerance=2.0)


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="Label-free embeddings -> LULC -> SAM 2.")
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project ID (default: $GEE_PROJECT, then the gcloud project, then "
            "the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def main():
    args = parse_args()
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study_area = str(OUTPUT_DIR / "study_area.geojson")
    Path(study_area).write_text(json.dumps(STUDY_AREA))

    from agribound.config import AgriboundConfig
    from agribound.postprocess.lulc_filter import filter_by_lulc

    configs, clusters, crops = {}, {}, {}

    # --- 1. Embedding clustering (no LULC filter) -------------------------------------------
    print(f"{'=' * 70}\n1. Embedding clustering ({YEAR})\n{'=' * 70}")
    for name, source_kwargs in EMBEDDINGS.items():
        output_path = OUTPUT_DIR / f"fields_{source_kwargs['source']}_embedding_{YEAR}_all.gpkg"
        configs[name] = AgriboundConfig(
            study_area=study_area,
            year=YEAR,
            engine="embedding",
            output_path=str(output_path),
            gee_project=args.gee_project,
            device="cpu",
            min_field_area_m2=MIN_AREA,
            lulc_filter=False,
            lulc_crop_threshold=CROP_THRESHOLD,
            **source_kwargs,
        )
        try:
            clusters[name] = agribound.delineate(config=configs[name])
        except Exception as exc:
            print(f"  {name} failed: {type(exc).__name__}: {exc}")
            continue
        print(f"  {name}: {len(clusters[name])} cluster polygons -> {output_path}")

    # --- 2. LULC crop filter -------------------------------------------------------------------
    print(f"\n{'=' * 70}\n2. LULC crop filter (threshold {CROP_THRESHOLD})\n{'=' * 70}")
    for name, gdf in clusters.items():
        crops[name] = filter_by_lulc(gdf, configs[name])
        stats = crops[name].attrs.get("lulc_stats", {})
        crops[name].to_file(OUTPUT_DIR / f"fields_{name}_crop_{YEAR}.gpkg", layer="fields")
        print(
            f"  {name}: kept {len(crops[name])} of {len(gdf)} polygons "
            f"({stats.get('dataset')} {stats.get('year_used')})"
        )

    # --- 3. Sentinel-2 composite ------------------------------------------------------------------
    window = f"{S2_DATE_RANGE[0]} to {S2_DATE_RANGE[1]}"
    print(f"\n{'=' * 70}\n3. Sentinel-2 composite {window}\n{'=' * 70}")
    s2_config = AgriboundConfig(
        study_area=study_area,
        source="sentinel2",
        year=YEAR,
        date_range=S2_DATE_RANGE,
        output_path=str(OUTPUT_DIR / "s2_refinement.gpkg"),  # names the cache dir only
        gee_project=args.gee_project,
        composite_method="median",
        cloud_cover_max=20,
        sam_backend="sam2",
        sam_model=SAM_MODEL,
    )
    s2_raster = agribound.build_composite(s2_config)
    print(f"  {s2_raster}")

    results = []  # (label, gdf)
    for name in EMBEDDINGS:
        if name in clusters:
            results.append((f"{name} clusters", clusters[name]))
            results.append((f"{name} crops (LULC)", crops[name]))

    # --- 4. SAM 2 on the Sentinel-2 composite -----------------------------------------------
    print(f"\n{'=' * 70}\n4. SAM 2 on Sentinel-2\n{'=' * 70}")
    for name, crop_gdf in crops.items():
        print(f"  {name}: {len(crop_gdf)} crop polygons")
        refined = refine(crop_gdf, s2_raster, s2_config)
        refined.to_file(OUTPUT_DIR / f"fields_{name}_crop_sam2-s2_{YEAR}.gpkg", layer="fields")
        results.append((f"{name} + SAM 2 (S2)", refined))

    # --- 5 and 6. SAM 2 on TESSERA pseudo-RGB -----------------------------------------------------
    if "tessera" in crops:
        print(f"\n{'=' * 70}\n5. SAM 2 on TESSERA dimensions {TESSERA_PSEUDO_RGB}\n{'=' * 70}")
        tessera_config = configs["tessera"].merged(
            sam_backend="sam2",
            sam_model=SAM_MODEL,
            engine_params={"sam_rgb_bands": TESSERA_PSEUDO_RGB},
        )
        tessera_raster = agribound.build_composite(tessera_config)  # cached in step 1
        refined = refine(crops["tessera"], tessera_raster, tessera_config)
        refined.to_file(
            OUTPUT_DIR / f"fields_tessera_crop_sam2-tessera_{YEAR}.gpkg", layer="fields"
        )
        results.append(("tessera + SAM 2 (TESSERA)", refined))

        print(f"\n{'=' * 70}\n6. Variant: split multi-part polygons, skip > 50 ha\n{'=' * 70}")
        from agribound.io.crs import get_equal_area_crs

        parts = crops["tessera"].explode(index_parts=False).reset_index(drop=True)
        parts = parts[parts.geometry.geom_type == "Polygon"]
        large = parts.geometry.to_crs(get_equal_area_crs()).area / 10000 > MAX_REFINE_AREA_HA
        print(f"  {len(parts)} parts; {int(large.sum())} larger than {MAX_REFINE_AREA_HA} ha")
        refined_small = refine(parts[~large], tessera_raster, tessera_config)
        variant = gpd.GeoDataFrame(
            pd.concat([refined_small, parts[large].to_crs(refined_small.crs)], ignore_index=True),
            geometry="geometry",
            crs=refined_small.crs,
        )
        variant.to_file(
            OUTPUT_DIR / f"fields_tessera_crop_sam2-tessera-split_{YEAR}.gpkg", layer="fields"
        )
        results.append(("tessera + SAM 2 (split)", variant))

    # --- 7. Delineate-Anything v2 on Sentinel-2 and on SPOT -------------------------------------
    print(f"\n{'=' * 70}\n7. Delineate-Anything v2 on Sentinel-2 and SPOT 6/7\n{'=' * 70}")
    da_runs = [
        # The step-3 composite settings, so the cached composite is reused.
        (
            "sentinel2",
            YEAR,
            dict(date_range=S2_DATE_RANGE, composite_method="median", cloud_cover_max=20),
        ),
        ("spot", SPOT_YEAR, dict(composite_method="median", cloud_cover_max=15)),
    ]
    for source, year, source_kwargs in da_runs:
        output_path = OUTPUT_DIR / f"fields_{source}_delineate-anything_{year}.gpkg"
        try:
            gdf = agribound.delineate(
                study_area=study_area,
                source=source,
                year=year,
                engine="delineate-anything",
                output_path=str(output_path),
                gee_project=args.gee_project,
                min_field_area_m2=MIN_AREA,
                lulc_filter=True,
                lulc_crop_threshold=CROP_THRESHOLD,
                **source_kwargs,
            )
        except Exception as exc:  # e.g. no SPOT access
            print(f"  {source} {year} failed: {type(exc).__name__}: {exc}")
            continue
        print(f"  {source} {year}: {len(gdf)} polygons -> {output_path}")
        results.append((f"Delineate-Anything ({source} {year})", gdf))

    # --- Comparison (counts and areas only; no reference data) --------------------------------
    print(f"\n{'=' * 70}\nComparison\n{'=' * 70}")
    print(f"  {'Method':<30} {'Polygons':>9} {'Area (ha)':>12}")
    for label, gdf in results:
        print(f"  {label:<30} {len(gdf):>9} {area_ha(gdf):>12,.1f}")

    if results:
        from agribound.visualize import show_comparison

        web_map = show_comparison(
            [gdf for _, gdf in results],
            labels=[label for label, _ in results],
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_comparison.html"),
        )
        show_in_notebook(web_map)
        print(f"\n  Map: {OUTPUT_DIR / 'map_comparison.html'}")
    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
