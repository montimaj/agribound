"""Run frozen North American sites with matched PAN/SR and known reference coverage.

See docs/user-guide/landsat-north-america-comparison.md. Example 23 supplies all
imagery, fusion, inference and processing. Previous cohorts are read, never run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from agribound.evaluate import evaluate, evaluate_frame


def load_file(filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_file("23_landsat_pan_sr_comparison.py")
SUITE = load_file("25_landsat_international_comparison.py")
ADAPTERS = load_file("landsat_reference_adapters.py")


def prepare_reference(source, site, cached, folder):
    frame = ADAPTERS.read_reference(cached, source, site["bbox"])
    n_read = len(frame)
    frame = ADAPTERS.filter_reference(frame, site["filters"], year=source["reference_year"])
    selected, n_repaired = SUITE.select_reference(frame, site)
    column = source.get("id_column")
    selected["source_id"] = selected[column].astype(str) if column else selected.index.astype(str)
    if selected.source_id.duplicated().any():
        raise ValueError("Duplicate selected provider identifiers")
    ref_path = folder / "reference.gpkg"
    selected.to_file(ref_path, layer="reference", driver="GPKG")
    # Store the fixed union's components, preserving every provider vertex.
    # Dissolving in a geographic CRS can create invalid slivers/shared edges.
    coverage = selected[["source_id", "geometry"]].copy()
    mask_path = folder / "evaluation_coverage.gpkg"
    coverage.to_file(mask_path, layer="coverage", driver="GPKG")
    crop = source.get("crop_column")
    counts = selected[crop].value_counts(dropna=False).to_dict() if crop else {}
    meta = {
        **source,
        "site": site,
        "n_read_in_bbox": n_read,
        "n_selected": len(selected),
        "n_invalid_repaired": n_repaired,
        "source_file_sha256": ADAPTERS.file_sha256(cached),
        "crop_or_use_class_counts": {str(k): int(v) for k, v in counts.items()},
        "crop_class_interpretation": "Provider crop/use attributes; no visual species inference",
        "evaluation_coverage_path": str(mask_path),
        "evaluation_coverage_sha256": ADAPTERS.file_sha256(mask_path),
        "coverage_scope": "Union of selected mapped footprints; conditional precision",
        "physical_gold_standard": False,
        "selection": "Whole polygons with representative points inside the frozen bbox",
        "training_overlap": "Training geographic overlap not documented; held-out status unknown",
    }
    metadata_path = folder / "reference_source.json"
    BASE.save_json(metadata_path, meta)
    return ref_path, metadata_path


def aerial_audit(site, folder, reference, *, offline=False, project="ee-rappjer"):
    """Reproduce the dated, prediction-blind reference inspection views."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    metadata = site["aerial_audit"]
    path = folder / "reference_aerial.jpg"
    bounds = site["bbox"]
    if not path.exists():
        if offline:
            raise ValueError("Offline audit needs reference_aerial.jpg")
        if metadata.get("naip_ids"):
            import ee

            from agribound.auth import setup_gee

            setup_gee(project=project, interactive=False)
            images = ee.ImageCollection.fromImages([ee.Image(i) for i in metadata["naip_ids"]])
            url = (
                images.mosaic()
                .select(["R", "G", "B"])
                .getThumbURL(
                    {
                        "region": ee.Geometry.Rectangle(bounds),
                        "dimensions": "2000x1600",
                        "crs": "EPSG:4326",
                        "min": 0,
                        "max": 255,
                        "format": "jpg",
                    }
                )
            )
        else:
            url = metadata["url"]
        with urlopen(url, timeout=180) as response:
            path.write_bytes(response.read())
    array = np.asarray(Image.open(path))
    frame = reference.to_crs(4326).reset_index(drop=True)
    areas = reference.to_crs(6933).area.reset_index(drop=True)
    views = [bounds]
    for position in [int(areas.idxmin()), int(areas.sort_values().index[len(frame) // 2])]:
        x, y = frame.geometry.iloc[position].representative_point().coords[0]
        views.append([x - 0.004, y - 0.003, x + 0.004, y + 0.003])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, view in zip(axes, views, strict=True):
        ax.imshow(array, extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
        frame.boundary.plot(ax=ax, color="cyan", linewidth=0.6)
        ax.set_xlim(view[0], view[2])
        ax.set_ylim(view[1], view[3])
        ax.set_aspect("auto")
    fig.suptitle(f"{site['id']}: provider reference on dated aerial imagery; no predictions")
    fig.tight_layout()
    fig.savefig(folder / "reference_audit.png", dpi=160)
    plt.close(fig)
    BASE.save_json(
        folder / "reference_audit.json",
        {
            **metadata,
            "reference_aerial_sha256": ADAPTERS.file_sha256(path),
            "review": site["audit_review"],
            "notes": site["audit_notes"],
            "zoom_selection": "Smallest and median-area reference; no predictions",
            "verified_all_physical_edges": False,
        },
    )


def evaluate_coverage(folder, site):
    """Retain raw AOI scores, adding explicitly conditional coverage scores."""
    prior = folder / "comparison_coverage.csv"
    previous_sha256 = ADAPTERS.file_sha256(prior) if prior.exists() else None
    ref = gpd.read_file(folder / "reference_evaluation.gpkg")
    mask_path = folder / "evaluation_coverage.gpkg"
    coverage = gpd.read_file(mask_path)
    previous_mask_sha256 = ADAPTERS.file_sha256(mask_path)
    if len(coverage) == 1 and len(ref) > 1:
        # Correct the early dissolved representation, without changing the
        # frozen set of reference footprints or any inference geometry.
        coverage = ref[["geometry"]].copy()
        coverage.to_file(mask_path, layer="coverage", driver="GPKG")
    for filename in ("reference_source.json", "reference_evaluation.json"):
        metadata_path = folder / filename
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous = metadata.get("evaluation_coverage_sha256")
        current = ADAPTERS.file_sha256(mask_path)
        if previous and previous != current:
            history = metadata.setdefault("previous_coverage_sha256", [])
            if previous not in history:
                history.append(previous)
        metadata["evaluation_coverage_sha256"] = current
        metadata["coverage_representation"] = "Components of the frozen footprint union"
        BASE.save_json(metadata_path, metadata)
    original = pd.read_csv(folder / "comparison.csv")
    rows, predictions = [], {}
    for name in ("pan", "sr", "combined"):
        raw_path = folder / f"fields_{name}.gpkg"
        predicted, selection = ADAPTERS.select_coverage(gpd.read_file(raw_path), coverage)
        predictions[name] = predicted
        output = folder / f"fields_{name}_evaluated.gpkg"
        predicted.to_file(output, layer="fields", driver="GPKG")
        timing = original[original.experiment == name].iloc[0]
        for tolerance in (10, 15, 30):
            metrics = evaluate(
                predicted,
                ref,
                size_bins=BASE.SIZE_BINS,
                boundary_tolerance_m=tolerance,
                boundary_mask=coverage,
                bootstrap=200,
            )
            BASE.save_json(folder / f"metrics_coverage_{name}_{tolerance}m.json", metrics)
            for size, values in [("all", metrics), *metrics["per_size_class"].items()]:
                rows.append(
                    {
                        "experiment": name,
                        "resolution_m": timing.resolution_m,
                        "boundary_tolerance_m": tolerance,
                        "size_class_ha": size,
                        "n_reference": values.get("n", metrics["count_reference"]),
                        "n_predicted": values["count_predicted"],
                        "precision": values["precision"],
                        "recall": values["recall"],
                        "f1": values["f1"],
                        "matched_iou": values["iou_mean"],
                        "boundary_precision": values["boundary_precision"],
                        "boundary_recall": values["boundary_recall"],
                        "boundary_f1": values["boundary_f1"],
                        "oversegmentation": values["oversegmentation_mean"],
                        "undersegmentation": values["undersegmentation_mean"],
                        "inference_s": timing.inference_s,
                        **selection,
                    }
                )
        evaluate_frame(
            predicted,
            ref,
            size_bins=BASE.SIZE_BINS,
            boundary_tolerance_m=15,
            boundary_mask=coverage,
        ).to_csv(folder / f"per_field_coverage_{name}.csv", index_label="reference_index")
        BASE.save_json(folder / f"coverage_selection_{name}.json", selection)
        parent = json.loads(
            raw_path.with_suffix(".gpkg.provenance.json").read_text(encoding="utf-8")
        )
        parent["coverage_evaluation"] = {
            **selection,
            "parent_output": str(raw_path),
            "parent_output_sha256": ADAPTERS.file_sha256(raw_path),
            "mask_sha256": ADAPTERS.file_sha256(folder / "evaluation_coverage.gpkg"),
            "boundary_mask": "Original lines intersect fixed mask; no synthetic clipping edges",
            "polygon_overlap": "Whole polygons, greedy one-to-one IoU >=0.5",
            "reference_kind": site["reference_kind"],
            "physical_gold_standard": False,
            "mask_reprojection_guard_m": 0.001,
            "mask_components": "Fixed reference footprints unioned in boundary UTM",
        }
        BASE.save_json(output.with_suffix(".gpkg.provenance.json"), parent)
    pd.DataFrame(rows).to_csv(folder / "comparison_coverage.csv", index=False)
    BASE.save_json(
        folder / "coverage_evaluation.json",
        {
            "previous_comparison_sha256": previous_sha256,
            "previous_mask_sha256": previous_mask_sha256,
            "mask_sha256": ADAPTERS.file_sha256(mask_path),
            "comparison_sha256": ADAPTERS.file_sha256(prior),
            "evaluator_sha256": ADAPTERS.file_sha256(
                Path(__file__).resolve().parents[1] / "agribound" / "evaluate.py"
            ),
            "mask_reprojection_guard_m": 0.001,
            "guard_reason": "Retain coincident boundary segments after floating-point reprojection",
            "mask_projection": "Original vertices directly to boundary UTM; no area densification",
            "mask_representation": "Individual frozen footprint polygons; no geographic dissolve",
            "inference_repeated": False,
            "polygon_geometry_changed": False,
        },
    )
    BASE.save_json(
        folder / "imagery_coverage.json",
        imagery_coverage(
            folder / "inputs" / "landsat.tif",
            {
                "aoi": gpd.GeoDataFrame(geometry=[box(*site["bbox"])], crs=4326),
                "known_footprints": coverage,
            },
        ),
    )
    map_dir = folder / "coverage"
    map_dir.mkdir(exist_ok=True)
    BASE.comparison_figure(
        folder / "inputs" / "landsat.tif",
        predictions,
        ref,
        box(*site["bbox"]),
        map_dir,
        reference_label=f"{site['reference_kind']} reference; known coverage only",
        date_label=f"{site['date_start']} through {site['date_end']}",
    )


def imagery_coverage(raster_path, scopes):
    """Valid common support at 30 m cell centres, separate from vector coverage."""
    import rasterio
    from rasterio.features import geometry_mask

    result = {"resolution_m": 30, "sampling": "30 m cell centres; all six SR bands valid"}
    with rasterio.open(raster_path) as raster:
        valid = np.isfinite(raster.read(masked=True).filled(np.nan)).all(axis=0)
        extent = gpd.GeoSeries([box(*raster.bounds)], crs=raster.crs).to_crs(6933).iloc[0]
        for scope, frame in scopes.items():
            if frame.crs is None or frame.empty:
                raise ValueError("Imagery coverage requires nonempty geometry with CRS")
            projected = frame.to_crs(raster.crs)
            included = geometry_mask(
                projected.geometry,
                out_shape=valid.shape,
                transform=raster.transform,
                invert=True,
                all_touched=False,
            )
            n_pixels = int(included.sum())
            n_valid = int((included & valid).sum())
            area = frame.to_crs(6933).geometry.union_all()
            result[scope] = {
                "sampled_cells": n_pixels,
                "valid_cells": n_valid,
                "valid_fraction": n_valid / n_pixels if n_pixels else None,
                "geometry_area_within_export_fraction": area.intersection(extent).area / area.area,
                "interpretation": "Pixel fraction within exported area; not observation frequency",
            }
    return result


def aggregate_metrics(table):
    """Site means within reference type and coverage scope, never pooled matches."""
    metrics = [
        "f1",
        "matched_iou",
        "boundary_f1",
        "precision",
        "recall",
        "boundary_precision",
        "boundary_recall",
        "oversegmentation",
        "undersegmentation",
        "inference_s",
    ]
    rows = []
    headline = table[table.size_class_ha == "all"]
    for keys, group in headline.groupby(
        ["cohort", "reference_kind", "evaluation_scope", "experiment", "boundary_tolerance_m"]
    ):
        for weighting in ("equal_site", "reference_count"):
            row = dict(
                zip(
                    [
                        "cohort",
                        "reference_kind",
                        "evaluation_scope",
                        "experiment",
                        "boundary_tolerance_m",
                    ],
                    keys,
                    strict=True,
                )
            )
            row.update(
                weighting=weighting, n_sites=len(group), n_reference=int(group.n_reference.sum())
            )
            weights = (
                np.ones(len(group)) if weighting == "equal_site" else group.n_reference.to_numpy()
            )
            for metric in metrics:
                finite = np.isfinite(group[metric].to_numpy())
                row[metric] = (
                    float(np.average(group[metric].to_numpy()[finite], weights=weights[finite]))
                    if finite.any()
                    else np.nan
                )
                row[f"{metric}_n_sites"] = int(finite.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def cross_site_figure(headline, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = headline.site.unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(sites) * 0.55)), sharey=True)
    y = np.arange(len(sites))
    for ax, metric, label in zip(
        axes, ["f1", "boundary_f1"], ["Detection F1 (IoU >=0.5)", "Boundary F1 (15 m)"], strict=True
    ):
        for j, (experiment, color) in enumerate(
            [("pan", "#4c78a8"), ("sr", "#f58518"), ("combined", "#54a24b")]
        ):
            values = [
                headline[(headline.site == name) & (headline.experiment == experiment)][
                    metric
                ].iloc[0]
                for name in sites
            ]
            ax.barh(y + (j - 1) * 0.24, values, 0.24, label=experiment, color=color)
        ax.set_xlim(0, 1)
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    labels = []
    for name in sites:
        row = headline[headline.site == name].iloc[0]
        labels.append(f"{name} [{row.reference_kind}; {row.evaluation_scope}]")
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[1].legend()
    fig.suptitle(
        "Matched Landsat 8/9; fixed checkpoint. Reference types and coverage scopes differ."
    )
    fig.tight_layout()
    fig.savefig(output_dir / "north_america_comparison.png", dpi=180)
    fig.savefig(output_dir / "north_america_comparison.pdf")
    plt.close(fig)


def summarize(config, digest, output_dir, existing_output_root):
    tables, statuses = [], {}
    for site in config["sites"]:
        folder = output_dir / site["id"]
        path = folder / "run_status.json"
        status = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"status": "unmeasured"}
        )
        if status["status"] == "complete" and (folder / "comparison_coverage.csv").exists():
            for scope, filename in [
                ("known_footprints", "comparison_coverage.csv"),
                ("aoi", "comparison.csv"),
            ]:
                table = pd.read_csv(folder / filename)
                table.insert(0, "site", site["id"])
                table.insert(1, "cohort", "north_america")
                table["reference_kind"] = config["sources"][site["reference_source"]][
                    "reference_kind"
                ]
                table["evaluation_scope"] = scope
                manifest = json.loads(
                    (folder / "inputs" / "scene_manifest.json").read_text(encoding="utf-8")
                )
                fusion = json.loads((folder / "fusion_validation.json").read_text(encoding="utf-8"))
                prep = json.loads((folder / "preparation_timing.json").read_text(encoding="utf-8"))
                table["matched_scenes"] = len(manifest["pairs"])
                table["unmatched_pan"] = len(manifest["unmatched_pan"])
                table["unmatched_sr"] = len(manifest["unmatched_sr"])
                table["valid_export_fraction"] = fusion["sr_valid_fraction"]
                table["paired_acquisition_s"] = prep["paired_acquisition_s"]
                table["fusion_s"] = prep["fusion_s"]
                coverage_path = folder / "imagery_coverage.json"
                if coverage_path.exists():
                    imagery = json.loads(coverage_path.read_text(encoding="utf-8"))[scope]
                    for metric in (
                        "sampled_cells",
                        "valid_cells",
                        "valid_fraction",
                        "geometry_area_within_export_fraction",
                    ):
                        table[f"imagery_{metric}"] = imagery[metric]
                tables.append(table)
            status["reference_kind"] = table.reference_kind.iloc[0]
            status["parcel_profile"] = SUITE.load_example(24).reference_profile(
                folder / "reference_evaluation.gpkg"
            )
        elif status["status"] == "failed":
            status["execution_status"] = "failed"
            status["status"] = "unmeasured"
        statuses[site["id"]] = status
    previous = existing_output_root / "landsat_international" / "international_comparison.csv"
    if previous.exists():
        table = pd.read_csv(previous)
        table["reference_kind"] = "legacy_mixed_reference_units"
        table["evaluation_scope"] = "aoi"
        tables.append(table)
    if tables:
        combined = pd.concat(tables, ignore_index=True)
        combined.to_csv(output_dir / "north_america_comparison.csv", index=False)
        headline = combined[
            (combined.size_class_ha == "all") & (combined.boundary_tolerance_m == 15)
        ]
        headline.to_csv(output_dir / "headline_15m_all_scopes.csv", index=False)
        primary = headline[
            (headline.cohort != "north_america") | (headline.evaluation_scope == "known_footprints")
        ]
        primary.to_csv(output_dir / "headline_15m.csv", index=False)
        aggregate_metrics(combined).to_csv(output_dir / "aggregate_metrics.csv", index=False)
        cross_site_figure(primary, output_dir)
        print(
            primary[
                [
                    "site",
                    "experiment",
                    "reference_kind",
                    "evaluation_scope",
                    "n_reference",
                    "n_predicted",
                    "f1",
                    "boundary_f1",
                    "inference_s",
                ]
            ].to_string(index=False),
            flush=True,
        )
    BASE.save_json(
        output_dir / "suite_status.json", {"configuration_sha256": digest, "sites": statuses}
    )
    inventory = []
    for site in config["sites"]:
        source = config["sources"][site["reference_source"]]
        profile = statuses[site["id"]].get("parcel_profile", {})
        inventory.append(
            {
                "site": site["id"],
                "country": site["country"],
                "region": site["region"],
                "bbox_wgs84": json.dumps(site["bbox"]),
                "imagery_start": site["date_start"],
                "imagery_end": site["date_end"],
                "provider": source["provider"],
                "reference_kind": source["reference_kind"],
                "reference_year": source["reference_year"],
                "geometry_vintage": source["geometry_vintage"],
                "vintage_verified": source["vintage_verified"],
                "license": source["license"],
                "redistribution": source["redistribution"],
                "download_url": source["url"],
                "documentation": source["documentation"],
                "source_sha256": source["sha256"],
                "crop_context": site["crop_context"],
                "climate_context": site["climate_context"],
                "climate_evidence": site["climate_evidence"],
                "crop_evidence": source.get("crop_evidence", source["documentation"]),
                "reference_filters": json.dumps(site["filters"], ensure_ascii=False),
                "audit": json.dumps(site["aerial_audit"], ensure_ascii=False),
                "audit_notes": site["audit_notes"],
                "limitations": source["limitations"],
                "selection_rationale": site["selection_rationale"],
                "physical_gold_standard": False,
                "training_overlap": "Unknown",
                "n_reference": profile.get("n_reference"),
            }
        )
    pd.DataFrame(inventory).to_csv(output_dir / "reference_inventory.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("landsat_north_america_sites.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/landsat_north_america"))
    parser.add_argument("--existing-output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--gee-project",
        default="ee-rappjer",
        help=(
            "Earth Engine project; fallback: $GEE_PROJECT, gcloud, project_id in "
            "$AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or $GOOGLE_APPLICATION_CREDENTIALS"
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--sites", nargs="+")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Prepare references and dated aerial views without inference",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--reevaluate-only",
        action="store_true",
        help="Recompute conditional coverage metrics/maps from completed outputs; no inference",
    )
    args = parser.parse_args(argv)
    config, digest = SUITE.freeze_config(args.config, args.output_dir)
    known = {s["id"] for s in config["sites"]}
    if args.sites and not set(args.sites).issubset(known):
        parser.error(f"Unknown site; choose from {sorted(known)}")
    failures = []
    if not args.summarize_only:
        for site in config["sites"]:
            if args.sites and site["id"] not in args.sites:
                continue
            folder = args.output_dir / site["id"]
            folder.mkdir(parents=True, exist_ok=True)
            status_path = folder / "run_status.json"
            run_key = folder / "suite_run_key.json"
            key = {
                "configuration_sha256": digest,
                "device": args.device,
                "gee_project": args.gee_project,
            }
            if (
                status_path.exists()
                and json.loads(status_path.read_text(encoding="utf-8"))["status"] == "complete"
            ):
                if not run_key.exists() or json.loads(run_key.read_text(encoding="utf-8")) != key:
                    raise ValueError(
                        "Completed site differs from frozen configuration; "
                        "use a new output directory"
                    )
                if args.reevaluate_only or not (folder / "comparison_coverage.csv").exists():
                    evaluate_coverage(
                        folder,
                        {
                            **site,
                            "reference_kind": config["sources"][site["reference_source"]][
                                "reference_kind"
                            ],
                        },
                    )
                print(f"Reusing completed {site['id']}", flush=True)
                continue
            if args.reevaluate_only:
                failures.append(site["id"])
                print(f"Unmeasured {site['id']}: no completed output to evaluate", flush=True)
                continue
            BASE.save_json(run_key, key)
            try:
                source = config["sources"][site["reference_source"]]
                cached = ADAPTERS.fetch_source(
                    source, args.output_dir / "references", offline=args.offline
                )
                ref, metadata = prepare_reference(source, site, cached, folder)
                aerial_audit(
                    site, folder, gpd.read_file(ref), offline=args.offline, project=args.gee_project
                )
                if args.audit_only:
                    BASE.save_json(status_path, {"status": "reference_prepared", "experiments": []})
                    continue
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("23_landsat_pan_sr_comparison.py")),
                    "--output-dir",
                    str(folder),
                    "--bbox",
                    *map(str, site["bbox"]),
                    "--date-start",
                    site["date_start"],
                    "--date-end",
                    site["date_end"],
                    "--reference",
                    str(ref),
                    "--reference-metadata",
                    str(metadata),
                    "--reference-year",
                    str(source["reference_year"]),
                    "--reference-label",
                    f"{source['reference_kind']} {source['reference_year']}",
                    "--device",
                    args.device,
                    "--gee-project",
                    args.gee_project,
                ]
                if args.checkpoint:
                    command.extend(["--checkpoint", str(args.checkpoint)])
                if args.offline:
                    command.append("--offline")
                if args.prepare_only:
                    command.append("--prepare-only")
                print(f"Running {site['id']}: {site['region']}", flush=True)
                start = time.perf_counter()
                with (folder / "run.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=False
                    )
                if result.returncode:
                    raise RuntimeError(
                        f"Comparison exited {result.returncode}; see {folder / 'run.log'}"
                    )
                if not args.prepare_only:
                    evaluate_coverage(folder, {**site, "reference_kind": source["reference_kind"]})
                BASE.save_json(
                    folder / "suite_timing.json",
                    {
                        "comparison_and_coverage_s": time.perf_counter() - start,
                        "includes_audit_and_reference_preparation": False,
                    },
                )
                print(f"Completed {site['id']}", flush=True)
            except Exception as exc:
                failures.append(site["id"])
                BASE.save_json(
                    status_path, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"Unmeasured {site['id']}: {exc}", flush=True)
            summarize(config, digest, args.output_dir, args.existing_output_root)
    summarize(config, digest, args.output_dir, args.existing_output_root)
    if failures:
        raise SystemExit(f"Sites still unmeasured: {', '.join(failures)}")


if __name__ == "__main__":
    main()
