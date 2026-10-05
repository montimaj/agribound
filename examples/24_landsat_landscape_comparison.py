"""Run the matched PAN/SR/combined comparison in three additional landscapes.

All sites use public IGN RPG 2023, May-August 2023 Landsat 8/9, and the same
released model/settings as example 23. Sites are fixed before observing scores.
Run: python examples/24_landsat_landscape_comparison.py --gee-project ee-rappjer
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

CASES = {
    "brittany": {
        "bbox": [-2.88, 48.16, -2.82, 48.20],
        "description": "Near Saint-Caradec: candidate bocage/irregular-field landscape",
    },
    "landes": {
        "bbox": [-0.96, 44.30, -0.90, 44.34],
        "description": "Near Ychoux: candidate large-field/forest-edge landscape",
    },
    "alsace": {
        "bbox": [7.49, 48.35, 7.55, 48.39],
        "description": "West of Benfeld: candidate elongated-parcel landscape",
    },
}


def reference_profile(path):
    """Describe parcel size and geometry without using prediction scores."""
    ref = gpd.read_file(path)
    ref = ref.to_crs(ref.estimate_utm_crs())
    area = ref.geometry.area.to_numpy()
    compactness = 4 * np.pi * area / ref.geometry.length.to_numpy() ** 2
    elongation = []
    for rectangle in ref.geometry.minimum_rotated_rectangle():
        points = np.asarray(rectangle.exterior.coords)
        sides = np.linalg.norm(np.diff(points, axis=0), axis=1)
        elongation.append(float(sides.max() / sides.min()) if sides.min() > 0 else None)
    return {
        "n_reference": len(ref),
        "median_area_ha": float(np.median(area) / 10000),
        "fraction_under_1ha": float(np.mean(area < 10000)),
        "median_compactness": float(np.median(compactness)),
        "median_elongation": float(np.median([v for v in elongation if v is not None])),
        "note": "Geometry of declared reference parcels; management splits may be invisible",
    }


def plot_summary(
    headline, output_dir, title="Matched 2023 Landsat 8/9; same model and public RPG references"
):
    """Compare sites separately with common axes and a fixed boundary tolerance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = list(headline.site.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    x = np.arange(len(sites))
    for ax, metric, title in zip(
        axes,
        ["f1", "boundary_f1"],
        ["Field detection F1 (IoU >=0.5)", "Boundary F1 (15 m tolerance)"],
        strict=True,
    ):
        for i, (experiment, color) in enumerate(
            [("pan", "#4c78a8"), ("sr", "#f58518"), ("combined", "#54a24b")]
        ):
            values = [
                float(
                    headline[(headline.site == site) & (headline.experiment == experiment)][
                        metric
                    ].iloc[0]
                )
                for site in sites
            ]
            bars = ax.bar(x + (i - 1) * 0.24, values, 0.24, label=experiment, color=color)
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_xticks(x, [site.title() for site in sites])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("F1")
    axes[1].legend(title="Input")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / "landscape_comparison.png", dpi=180)
    plt.close(fig)


def summarize(output_dir, baseline):
    """Retain site and size strata; do not average incomparable field counts."""
    tables, manifest = [], {}
    for name, folder in [("beauce", baseline), *[(n, output_dir / n) for n in CASES]]:
        status_path = folder / "run_status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        manifest[name] = {"status": status["status"], "output_dir": str(folder)}
        if status["status"] != "complete":
            continue
        reference = json.loads((folder / "reference_evaluation.json").read_text(encoding="utf-8"))
        scenes = json.loads((folder / "inputs" / "scene_manifest.json").read_text(encoding="utf-8"))
        manifest[name].update(
            {
                "reference": reference,
                "matched_scenes": len(scenes["pairs"]),
                "unmatched_pan": len(scenes["unmatched_pan"]),
                "unmatched_sr": len(scenes["unmatched_sr"]),
            }
        )
        manifest[name]["parcel_profile"] = reference_profile(folder / "reference_evaluation.gpkg")
        table = pd.read_csv(folder / "comparison.csv")
        table.insert(0, "site", name)
        tables.append(table)
    if tables:
        combined = pd.concat(tables, ignore_index=True)
        combined.to_csv(output_dir / "landscape_comparison.csv", index=False)
        headline = combined[
            (combined.size_class_ha == "all") & (combined.boundary_tolerance_m == 15)
        ]
        headline.to_csv(output_dir / "headline_15m.csv", index=False)
        plot_summary(headline, output_dir)
        print(
            headline[
                [
                    "site",
                    "experiment",
                    "n_reference",
                    "n_predicted",
                    "f1",
                    "boundary_f1",
                    "inference_s",
                ]
            ].to_string(index=False)
        )
    (output_dir / "suite_status.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project; defaults to $GEE_PROJECT, then gcloud, then the "
            "credentials project_id from $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/landsat_landscapes"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/landsat_pan_sr_comparison"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--sites", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sites.json").write_text(json.dumps(CASES, indent=2), encoding="utf-8")
    script = Path(__file__).with_name("23_landsat_pan_sr_comparison.py")
    failures = []
    if not args.summarize_only:
        for name in args.sites:
            folder = args.output_dir / name
            folder.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(script),
                "--output-dir",
                str(folder),
                "--device",
                args.device,
                "--bbox",
                *map(str, CASES[name]["bbox"]),
            ]
            if args.gee_project:
                command.extend(["--gee-project", args.gee_project])
            if args.checkpoint:
                command.extend(["--checkpoint", str(args.checkpoint)])
            print(f"Running {name}: {CASES[name]['description']}", flush=True)
            with (folder / "run.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                failures.append(name)
                print(f"Failed {name}; see {folder / 'run.log'}", flush=True)
            else:
                print(f"Completed {name}", flush=True)
            summarize(args.output_dir, args.baseline)
    else:
        summarize(args.output_dir, args.baseline)
    if failures:
        raise SystemExit(f"Comparisons failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
