"""Landsat versus published FTW: reference evaluation and peer-product agreement.

Run this tutorial offline from a fresh checkout. The small bundle contains real
Camargue 2024 predictions and six preselected public-reference site summaries
across France, Netherlands, Canada and Vietnam. Vietnam has 2021 reference labels
versus 2024 imagery; source reuse/training overlap is disclosed in the inventory.

Published FTW Global polygons are model predictions from Sentinel-2, not benchmark
labels or independent ground truth. Boundary placement and one-to-one individual
field detection can rank products differently against declaration units.

Run: python examples/30_landsat_ftw_product_comparison.py
Live: add --live --gee-project YOUR_PROJECT --checkpoint PATH_TO_large_v2.pt
The default local GeoParquet query uses the published 2024 snapshot with complete
polygons, clip=False, deduplication and no max_features. Confidence >=69 is a
predeclared sensitivity, retaining unknown confidence separately. It is not the
same confidence score as the Landsat model. A second sensitivity applies 1000 m2.
Live queries are pinned once; use a new output directory to refresh the product.

Accuracy CSVs use shared source-derived reference coverage; separate agreement
CSVs describe correspondence between peer predictions. Neither demonstrates
verified independent validation or an isolated sensor-resolution effect.
See docs/user-guide/landsat-tutorials.md for exact commands and data attribution.
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

ROOT = (
    next(
        p
        for p in [Path.cwd(), *Path.cwd().parents]
        if (p / "pyproject.toml").is_file() and (p / "agribound").is_dir()
    )
    if "__file__" not in globals()
    else Path(__file__).resolve().parents[1]
)
sys.path.insert(0, str(ROOT / "examples"))
SUPPORT = import_module("landsat_tutorial_support")

BUNDLE = ROOT / "examples/data/landsat_tutorial"
OUTPUT = ROOT / "outputs/landsat_tutorial/ftw"
NOTEBOOK_ARGS = []
DISPLAY_METHODS = ["pan", "sr", "pan_nir_red", "hybrid"]


def retrieve(args):
    """Query complete cached polygons for a supported year, then derive sensitivities."""
    products = SUPPORT.query_product(
        BUNDLE, args.output_dir, live=args.live, query_if_missing=args.stage in ("all", "query")
    )
    print("Whole-AOI FTW counts:", {name: len(frame) for name, frame in products.items()})
    return products


def compare(args, landsat, ftw):
    """Keep reference accuracy and direct Landsat/FTW agreement in different tables."""
    accuracy = SUPPORT.evaluate_products(BUNDLE, args.output_dir, {**landsat, **ftw})
    agreement = SUPPORT.agreement_products(args.output_dir, landsat, ftw["ftw"])
    headline = accuracy[(accuracy.boundary_tolerance_m == 15) & (accuracy.size_class_ha == "all")]
    print("Reference-coverage evaluation (declarations, not verified physical truth):")
    print(headline[["product", "boundary_f1", "detection_f1"]].to_string(index=False))
    print("Separate peer agreement:")
    print(
        agreement[agreement.boundary_tolerance_m == 15][
            ["method", "symmetric_boundary_agreement_f1", "correspondence_f1"]
        ].to_string(index=False)
    )


def render(args, landsat, ftw):
    """Show one common imagery background and six landscape-selected measured summaries."""
    products = {"ftw": ftw["ftw"], **{m: landsat[m] for m in DISPLAY_METHODS}}
    SUPPORT.maps(BUNDLE, args.output_dir, products, stem="ftw_comparison")
    SUPPORT.summary_figure(BUNDLE, args.output_dir)
    if "ipykernel" in sys.modules:
        from IPython.display import Image, display

        for name in ["ftw_comparison_shared_edges.png", "six_public_locations.png"]:
            display(Image(filename=str(args.output_dir / name), width=1100))


def main(argv=None):
    """Resume evaluation/rendering independently; live work is an explicit opt-in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=["all", "query", "evaluate", "figures"], default="all")
    parser.add_argument(
        "--live", action="store_true", help="Explicitly allow downloads and pinned-model inference"
    )
    parser.add_argument(
        "--gee-project",
        help=(
            "Earth Engine project; fallback: $GEE_PROJECT, gcloud, project_id in "
            "$AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or $GOOGLE_APPLICATION_CREDENTIALS"
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(SUPPORT.load_bundle(BUNDLE)["limitations"])
    ftw = retrieve(args)
    if args.stage == "query":
        return
    if args.live and args.stage == "all":
        paths, _ = SUPPORT.prepare_inputs(
            BUNDLE, args.output_dir, live=True, project=args.gee_project
        )
        paths = SUPPORT.run_landsat(
            BUNDLE,
            args.output_dir,
            paths,
            live=True,
            checkpoint=args.checkpoint,
            project=args.gee_project,
        )
    else:
        paths = SUPPORT.cached_landsat_paths(BUNDLE, args.output_dir, live=args.live)
    landsat = SUPPORT.load_products(BUNDLE, paths)
    if args.stage in ("all", "evaluate"):
        compare(args, landsat, ftw)
    if args.stage in ("all", "figures"):
        render(args, landsat, ftw)
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main(NOTEBOOK_ARGS if "ipykernel" in sys.modules else None)
