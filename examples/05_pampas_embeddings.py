"""
05 — Argentine Pampas: Google and TESSERA Embeddings (CPU only)

Clusters pre-computed satellite embeddings near Pergamino (Buenos Aires
Province) with the ``embedding`` engine: K-means on PCA-reduced embeddings,
then every connected region of a cluster becomes a polygon. No labels, model
weights or GPU are needed. Clusters are land-cover segments, not field
instances.

Sources (2024):
    - Google Satellite Embedding V1 (AlphaEarth Foundations), 64-D, 10 m,
      annual 2017-2025. Read from Earth Engine (``google_embedding_backend=
      "gee"``, default) or from the public Source Cooperative mirror
      (``--google-backend source_coop``, no Earth Engine needed).
    - TESSERA v1 (``tessera_version="v1"``), 128-D, 10 m. TESSERA v1 is
      near-global only for 2024; the v1 manifest has every tile of this box
      for 2024 but only a few for 2017-2023, so this example uses 2024.

The LULC crop filter is off (``lulc_filter=False``): the clusters are not
screened against a crop map, and no Earth Engine access is needed for it.

Study area: a 0.2 x 0.2 degree box (about 19 x 23 km at 10 m, ~4.3 M pixels;
ESA WorldCover 2021 classifies ~92 % of it as cropland). The TESSERA builder
assembles the 128-band float32 mosaic in memory (~2.2 GB for this box).
``--full-box`` uses the original 0.5 x 0.5 degree box (-60.8, -34.0, -60.3,
-33.5): about 6.9 GB of Google and 13.8 GB of TESSERA float32 embeddings.

Outputs: ``fields_google-embedding-<backend>_embedding_<sub|full>_2024.gpkg``
and ``fields_tessera-embedding-v1_embedding_<sub|full>_2024.gpkg``. An
existing output made with the same settings is loaded instead of
recomputed; ``--overwrite`` recomputes it.

Estimated runtime (not measured for 1.0): ~5-15 minutes for the default box
(download-bound, CPU).

Prerequisites:
    pip install "agribound[gee,tessera]"
    agribound auth --project YOUR_GEE_PROJECT   # not needed with --google-backend source_coop
    Run from the repository root: python examples/05_pampas_embeddings.py
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import agribound

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

warnings.filterwarnings("ignore", category=FutureWarning, module=r"geedim\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"geedim\..*")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("geedim").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/pampas")
YEAR = 2024
DEFAULT_BBOX = (-60.65, -33.85, -60.45, -33.65)  # 0.2 x 0.2 degrees near Pergamino
FULL_BBOX = (-60.8, -34.0, -60.3, -33.5)  # original 0.5 x 0.5 degree box


def write_bbox(bbox, path, name):
    """Write a (minx, miny, maxx, maxy) EPSG:4326 box as a GeoJSON study area."""
    minx, miny, maxx, maxy = bbox
    ring = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
    feature = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}}
    feature["properties"] = {"name": name}
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    return str(path)


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="Pampas: Google vs TESSERA embeddings.")
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project ID (default: $GEE_PROJECT, then the gcloud project, then "
            "the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    parser.add_argument(
        "--google-backend",
        default="gee",
        choices=["gee", "source_coop"],
        help="Where the Google Satellite Embeddings are read from.",
    )
    parser.add_argument("--full-box", action="store_true", help="Use the 0.5-degree box.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute outputs that already exist."
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
    bbox = FULL_BBOX if args.full_box else DEFAULT_BBOX
    tag = "full" if args.full_box else "sub"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study_area = write_bbox(bbox, OUTPUT_DIR / f"study_area_{tag}.geojson", "Pampas (Pergamino)")

    # label: (output name tag, source keyword arguments). The Google output name
    # includes the backend, so runs with either backend keep separate outputs.
    runs = {
        f"Google {YEAR}": (
            f"google-embedding-{args.google_backend}",
            dict(
                source="google-embedding",
                google_embedding_backend=args.google_backend,
                gee_project=args.gee_project,
            ),
        ),
        f"TESSERA v1 {YEAR}": (
            "tessera-embedding-v1",
            dict(source="tessera-embedding", tessera_version="v1"),
        ),
    }
    results = {}
    for label, (name, source_kwargs) in runs.items():
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        output_path = OUTPUT_DIR / f"fields_{name}_embedding_{tag}_{YEAR}.gpkg"
        try:
            gdf = agribound.delineate(
                study_area=study_area,
                year=YEAR,
                engine="embedding",
                output_path=str(output_path),
                device="cpu",
                min_area=5000,
                lulc_filter=False,  # clusters are not screened against a crop map
                overwrite=args.overwrite,
                **source_kwargs,
            )
        except Exception as exc:
            print(f"  {label} failed: {type(exc).__name__}: {exc}")
            continue
        results[label] = gdf
        print(f"  {len(gdf)} polygons -> {output_path}")

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    print(f"  {'Run':<20} {'Polygons':>9} {'Area (ha)':>12}")
    for label, gdf in results.items():
        area = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0.0
        print(f"  {label:<20} {len(gdf):>9} {area:>12,.1f}")

    if not results:
        print("\nNo run succeeded; no map written.")
        return
    from agribound.visualize import show_comparison

    web_map = show_comparison(
        list(results.values()),
        labels=list(results.keys()),
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / f"map_comparison_{tag}.html"),
    )
    show_in_notebook(web_map)
    print(f"\nComparison map: {OUTPUT_DIR / f'map_comparison_{tag}.html'}")


if __name__ == "__main__":
    main()
