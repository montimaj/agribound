"""Native Landsat PAN and controlled multispectral field delineation.

This fast offline tutorial uses real Camargue 2024 inputs and measured polygons.
It prepares all eight input controls, checks unchanged NIR, reevaluates delivered
predictions, and renders identical common-background maps. It does not run a
model offline. RPG references are crop declarations, not verified independent
physical-field truth. Climate/crop labels come from the bundled source inventory.

Install: python -m pip install -e . matplotlib nbformat nbclient ipykernel
Run: python examples/29_landsat_pan_field_delineation.py
Live: add --live --gee-project YOUR_PROJECT --checkpoint PATH_TO_large_v2.pt
See docs/user-guide/landsat-tutorials.md for authentication and resume commands.

Native PAN is 15 m B8 TOA; SR RGB is 30 m surface reflectance. The established
visible fusion injects zero-mean PAN detail into RGB while retaining 30 m means.
The PAN/NIR/Red stack delivers PAN in logical channel 1. Hybrid NIR is resampled,
never sharpened. E controls grid/context effects; F-G tests native PAN detail.
False color and stacking remain experiments with an RGB-trained checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

# Resolve paths in a script or a notebook without changing the working directory.
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
OUTPUT = ROOT / "outputs/landsat_tutorial/pan"
# Notebook users can edit these explicit arguments; [] always selects offline.
NOTEBOOK_ARGS = []
DISPLAY_METHODS = ["pan", "sr", "combined", "pan_nir_red", "hybrid"]


def prepare(args):
    """Check scene/grid support and prepare controls with their channel identities."""
    paths, integrity = SUPPORT.prepare_inputs(
        BUNDLE, args.output_dir, live=args.live, project=args.gee_project
    )
    print("Logical model channels:", integrity["logical_channels"])
    print("PAN/SR common support:", integrity["integrity"]["valid_fraction"])
    print(
        "NIR change after documented resampling:",
        integrity["integrity"]["nir_max_abs_change_after_resampling"],
    )
    return paths


def delineate(args, paths):
    """Reuse measured polygons offline, or run the pinned model after explicit opt-in."""
    paths = SUPPORT.run_landsat(
        BUNDLE,
        args.output_dir,
        paths,
        live=args.live,
        checkpoint=args.checkpoint,
        project=args.gee_project,
    )
    return SUPPORT.load_products(BUNDLE, paths)


def compare(args, products):
    """Evaluate common reference coverage at identical metre tolerances."""
    table = SUPPORT.evaluate_products(BUNDLE, args.output_dir, products)
    headline = table[(table.boundary_tolerance_m == 15) & (table.size_class_ha == "all")]
    print(
        headline[["product", "n_reference", "boundary_f1", "detection_f1"]].to_string(index=False)
    )
    return table


def render(args, products):
    """Render whole-AOI and frozen small-field/shared-edge views with one SR background."""
    SUPPORT.maps(
        BUNDLE, args.output_dir, {m: products[m] for m in DISPLAY_METHODS}, stem="pan_comparison"
    )
    if "ipykernel" in sys.modules:
        from IPython.display import Image, display

        display(
            Image(filename=str(args.output_dir / "pan_comparison_small_fields.png"), width=1100)
        )


def main(argv=None):
    """Choose a resumable stage; downloads and inference require --live."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=["all", "prepare", "evaluate", "figures"], default="all")
    parser.add_argument(
        "--live", action="store_true", help="Explicitly allow EE download and model inference"
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
    manifest = SUPPORT.load_bundle(BUNDLE)
    print(manifest["limitations"])
    if args.stage in ("all", "prepare"):
        paths = prepare(args)
        if args.stage == "prepare":
            return
        products = delineate(args, paths)
    else:
        paths = SUPPORT.cached_landsat_paths(BUNDLE, args.output_dir, live=args.live)
        products = SUPPORT.load_products(BUNDLE, paths)
    if args.stage in ("all", "evaluate"):
        compare(args, products)
    if args.stage in ("all", "figures"):
        render(args, products)
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main(NOTEBOOK_ARGS if "ipykernel" in sys.modules else None)
