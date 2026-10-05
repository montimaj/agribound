"""Run six frozen international sites through example 23's PAN/SR/fusion workflow.

Public reference sources, licenses, SHA-256 pins and selection rationale are in
landsat_international_sites.json. Existing completed sites are verified and reused.
No French inference is repeated; only its existing metrics are read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

# Provider-documentation clarifications do not alter the frozen site selection,
# imagery dates, reference geometries or scores. The conversion's nominal year
# is retained, alongside the original boundary and survey dates.
REFERENCE_NOTES = {
    "boundaries_south_africa_2018.parquet": {
        "vintage_verified": False,
        "nominal_reference_year": 2018,
        "boundary_imagery_years": [2016, 2018],
        "digitization_period": "May-August 2017; exceptional edits from January 2018 imagery",
        "crop_survey_period": "May 2017-March 2018",
        "vintage_note": (
            "Conversion says 2018, but provider describes mainly 2016 aerial geometry "
            "and 2017-2018 surveys; contemporaneous physical boundaries in "
            "July-October 2018 are not verified"
        ),
        "additional_terms": (
            "Provider document also restricts further distribution and commercial use "
            "and describes competition/academic research scope. Do not redistribute "
            "extracted reference vectors; confirm permission before uses beyond this scope."
        ),
        "provider_documentation": "https://radiantearth.blob.core.windows.net/mlhub/esa-food-security-challenge/Crops_GT_Western_Cape_Doc.pdf",
    }
}


def load_example(number):
    path = next(Path(__file__).parent.glob(f"{number}_*.py"))
    spec = importlib.util.spec_from_file_location(f"comparison_{number}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_config(config_path, output_dir):
    """Refuse to mix changed selection criteria into an existing experiment."""
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if data["schema_version"] != 1:
        raise ValueError("Unknown site configuration schema")
    from agribound.engines.delineate_anything import DA_MODELS

    comparison = data["comparison"]
    if (
        comparison["cloud_cover_max"] != 20
        or comparison["engine"] != "delineate-anything"
        or comparison["checkpoint"] != "large_v2"
        or comparison["checkpoint_sha256"] != DA_MODELS["large_v2"].sha256
        or comparison["boundary_tolerances_m"] != [10, 15, 30]
        or comparison["size_bins_ha"] != [0, 1, 5, 20, 100, 100000]
    ):
        raise ValueError(
            "Suite configuration disagrees with the fixed example 23 comparison settings"
        )
    ids = [s["id"] for s in data["sites"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate site identifiers")
    for site in data["sites"]:
        bounds = site["bbox"]
        if len(bounds) != 4 or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("Invalid site bbox")
        if site["reference_source"] not in data["sources"]:
            raise ValueError("Site has an unknown reference source")
        if site["date_start"] > site["date_end"]:
            raise ValueError("Invalid site acquisition window")
    canonical = json.dumps(data, sort_keys=True, indent=2) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = output_dir / "sites_frozen.json"
    if frozen.exists() and frozen.read_text(encoding="utf-8") != canonical:
        raise ValueError("Frozen site configuration changed; use another output directory")
    frozen.write_text(canonical, encoding="utf-8")
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return data, digest


def fetch_reference(source, cache_dir, offline=False):
    """Download anonymously with TLS, atomic replacement and a pinned checksum."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / source["filename"]
    if target.exists():
        if sha256(target) != source["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {target}")
        return target
    if offline:
        raise ValueError(f"Offline reference missing: {target}")
    temporary = target.with_suffix(".part")
    with urlopen(source["url"], timeout=180) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    if sha256(temporary) != source["sha256"]:
        raise ValueError(f"Downloaded reference checksum mismatch: {target}")
    temporary.replace(target)
    return target


def select_reference(frame, site):
    """Preserve whole declared/digitized parcels, selecting by representative point."""
    if frame.crs is None:
        raise ValueError("Reference lacks a CRS; do not guess its coordinate system")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise ValueError("Reference contains missing/empty geometries")
    if not frame.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Reference requires polygon geometries")
    repaired = frame.copy()
    n_invalid = int((~repaired.geometry.is_valid).sum())
    repaired.geometry = repaired.geometry.make_valid(method="structure", keep_collapsed=False)
    if not repaired.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Repair produced non-polygon reference geometries")
    projected = repaired.to_crs(4326)
    selected = repaired.loc[projected.geometry.representative_point().within(box(*site["bbox"]))]
    if len(selected) < 2:
        raise ValueError("Site needs at least two selected reference polygons")
    if "id" in selected and selected["id"].duplicated().any():
        raise ValueError("Duplicate reference identifiers")
    return selected.reset_index(drop=True), n_invalid


def prepare_reference(source, site, cached_file, folder):
    """Export the actual site reference and retain source classes without relabeling."""
    frame = gpd.read_parquet(cached_file)
    selected, n_invalid = select_reference(frame, site)
    if "determination_datetime" in selected:
        dates = pd.to_datetime(selected.determination_datetime, utc=True)
        years = sorted(dates.dt.year.dropna().unique().tolist())
        if years != [source["reference_year"]]:
            raise ValueError(f"Reference dates {years} disagree with declared vintage")
    path = folder / "reference.gpkg"
    selected.to_file(path, driver="GPKG", layer="reference")
    column = source["crop_column"]
    counts = selected[column].fillna("unknown").value_counts().to_dict() if column else {}
    metadata = {
        **source,
        **REFERENCE_NOTES.get(source.get("filename"), {}),
        "site": site,
        "n_downloaded": len(frame),
        "n_selected": len(selected),
        "n_invalid_repaired_in_source": n_invalid,
        "selection": "Whole polygons with representative points inside the evaluation bbox",
        "source_file_sha256": sha256(cached_file),
        "crop_or_use_class_counts": {str(k): int(v) for k, v in counts.items()},
        "crop_class_interpretation": "Provider attributes, not image-derived species predictions",
    }
    meta_path = folder / "reference_source.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path, meta_path


def aggregate_metrics(table):
    """Means of site metrics, explicitly distinct from pooling matches or lengths."""
    metrics = [
        "precision",
        "recall",
        "f1",
        "matched_iou",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "oversegmentation",
        "undersegmentation",
        "inference_s",
    ]
    rows = []
    headline = table[table.size_class_ha == "all"]
    cohorts = {
        "international": headline[headline.cohort == "international"],
        "france_baseline": headline[headline.cohort == "france_baseline"],
        "all_sites": headline,
    }
    for cohort, values in cohorts.items():
        for (experiment, tolerance), group in values.groupby(
            ["experiment", "boundary_tolerance_m"]
        ):
            for weighting in ("equal_site", "reference_count"):
                row = {
                    "cohort": cohort,
                    "experiment": experiment,
                    "boundary_tolerance_m": tolerance,
                    "weighting": weighting,
                    "n_sites": len(group),
                    "n_reference": int(group.n_reference.sum()),
                }
                weights = (
                    np.ones(len(group))
                    if weighting == "equal_site"
                    else group.n_reference.to_numpy()
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


def clarify_reference_metadata(folder):
    """Correct documentation-only vintage facts, preserving geometry and metrics.

    This also updates already-completed runs made before the provider's PDF was
    inspected. The audit explicitly records that inference was not repeated.
    """
    source_path = folder / "reference_source.json"
    if not source_path.exists():
        return
    source = json.loads(source_path.read_text(encoding="utf-8"))
    notes = REFERENCE_NOTES.get(source.get("filename"))
    if not notes:
        return
    changed = False
    for path in [source_path, folder / "reference_evaluation.json"]:
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            if (
                any(meta.get(k) != v for k, v in notes.items())
                or meta.get("vintage_matches") is True
            ):
                meta.update(notes)
                if "vintage_matches" in meta:
                    meta["vintage_matches"] = False
                path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                changed = True
    status_path = folder / "run_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if "reference" in status:
            status["reference"].update(notes)
            status["reference"]["vintage_matches"] = False
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    for path in folder.glob("fields_*.gpkg.provenance.json"):
        provenance = json.loads(path.read_text(encoding="utf-8"))
        reference = provenance.get("facts", {}).get("reference")
        if isinstance(reference, dict):
            reference.update(notes)
            reference["vintage_matches"] = False
            provenance["reference_metadata_clarification"] = (
                "Original provider documentation checked; boundary/survey dates clarified. "
                "Reference geometry, imagery, inference and scores unchanged."
            )
            path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if changed:
        (folder / "reference_metadata_clarification.json").write_text(
            json.dumps(
                {"notes": notes, "scope": "Documentation only; no inference or metric changes"},
                indent=2,
            ),
            encoding="utf-8",
        )


def summary_figure(headline, folder, unmeasured=()):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = headline.site.unique()
    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(sites) * 0.65)), sharey=True)
    y = np.arange(len(sites))
    for ax, metric, title in zip(
        axes, ["f1", "boundary_f1"], ["Detection F1 (IoU >=0.5)", "Boundary F1 (15 m)"], strict=True
    ):
        for i, (experiment, color) in enumerate(
            [("pan", "#4c78a8"), ("sr", "#f58518"), ("combined", "#54a24b")]
        ):
            values = [
                headline[(headline.site == site) & (headline.experiment == experiment)][
                    metric
                ].iloc[0]
                for site in sites
            ]
            bars = ax.barh(y + (i - 1) * 0.24, values, 0.24, label=experiment, color=color)
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.set_yticks(y, [s.replace("_", " ") for s in sites])
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[1].legend(title="Input", loc="lower right")
    fig.suptitle("Matched Landsat 8/9; fixed checkpoint; site-specific years and seasons")
    if unmeasured:
        fig.text(
            0.5,
            0.005,
            f"Unmeasured sites (no score plotted): {', '.join(unmeasured)}",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(folder / "international_comparison.png", dpi=180)
    fig.savefig(folder / "international_comparison.pdf")
    plt.close(fig)


def summarize(config, config_hash, output_dir, french_dir):
    example = load_example(24)
    tables, statuses = [], {}
    folders = [(s["id"], output_dir / s["id"], "international") for s in config["sites"]]
    folders += [("beauce", french_dir / "landsat_pan_sr_comparison", "france_baseline")]
    folders += [
        (s, french_dir / "landsat_landscapes" / s, "france_baseline") for s in example.CASES
    ]
    for name, folder, cohort in folders:
        clarify_reference_metadata(folder)
        status_path = folder / "run_status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.exists()
            else {"status": "unmeasured"}
        )
        statuses[name] = {"status": status["status"], "output_dir": str(folder)}
        if status["status"] != "complete":
            statuses[name]["error"] = status.get("error")
            failure_path = folder / "inputs" / "scene_selection_failure.json"
            if failure_path.exists():
                statuses[name]["scene_selection_failure"] = json.loads(
                    failure_path.read_text(encoding="utf-8")
                )
                statuses[name]["execution_status"] = statuses[name]["status"]
                statuses[name]["status"] = "unmeasured"
            if (folder / "reference_evaluation.json").exists():
                statuses[name]["reference"] = json.loads(
                    (folder / "reference_evaluation.json").read_text(encoding="utf-8")
                )
                statuses[name]["parcel_profile"] = example.reference_profile(
                    folder / "reference_evaluation.gpkg"
                )
            continue
        reference = json.loads((folder / "reference_evaluation.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (folder / "inputs" / "scene_manifest.json").read_text(encoding="utf-8")
        )
        fusion = json.loads((folder / "fusion_validation.json").read_text(encoding="utf-8"))
        statuses[name].update(
            {
                "reference": reference,
                "parcel_profile": example.reference_profile(folder / "reference_evaluation.gpkg"),
                "matched_scenes": len(manifest["pairs"]),
                "unmatched_pan": len(manifest["unmatched_pan"]),
                "unmatched_sr": len(manifest["unmatched_sr"]),
                "valid_fraction": fusion["sr_valid_fraction"],
                "date_window_end_exclusive": manifest["date_window_end_exclusive"],
                "missions": manifest["missions"],
            }
        )
        table = pd.read_csv(folder / "comparison.csv")
        table.insert(0, "site", name)
        table.insert(1, "cohort", cohort)
        tables.append(table)
    if tables:
        combined = pd.concat(tables, ignore_index=True)
        combined.to_csv(output_dir / "international_comparison.csv", index=False)
        headline = combined[
            (combined.size_class_ha == "all") & (combined.boundary_tolerance_m == 15)
        ]
        headline.to_csv(output_dir / "headline_15m.csv", index=False)
        aggregate_metrics(combined).to_csv(output_dir / "aggregate_metrics.csv", index=False)
        summary_figure(
            headline,
            output_dir,
            [s["id"] for s in config["sites"] if statuses[s["id"]]["status"] != "complete"],
        )
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
            ].to_string(index=False),
            flush=True,
        )
    (output_dir / "suite_status.json").write_text(
        json.dumps({"configuration_sha256": config_hash, "sites": statuses}, indent=2),
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("landsat_international_sites.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/landsat_international"))
    parser.add_argument("--french-output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project; defaults to $GEE_PROJECT, then gcloud, then the "
            "credentials project_id from $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS."
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--sites", nargs="+")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    config, digest = freeze_config(args.config, args.output_dir)
    known_sites = {s["id"] for s in config["sites"]}
    if args.sites and not set(args.sites).issubset(known_sites):
        parser.error(f"Unknown site; choose from {sorted(known_sites)}")
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
            if status_path.exists() and json.loads(status_path.read_text())["status"] == "complete":
                if not run_key.exists() or json.loads(run_key.read_text()) != key:
                    raise ValueError(
                        "Completed site has a different configuration; use a new output directory"
                    )
                print(f"Reusing completed {site['id']}", flush=True)
                continue
            run_key.write_text(json.dumps(key, indent=2), encoding="utf-8")
            try:
                source = config["sources"][site["reference_source"]]
                cached = fetch_reference(
                    source, args.output_dir / "references", offline=args.offline
                )
                ref, metadata = prepare_reference(source, site, cached, folder)
                reference_meta = json.loads(metadata.read_text(encoding="utf-8"))
                vintage_label = (
                    "nominal " if reference_meta.get("vintage_verified") is False else ""
                )
                script = Path(__file__).with_name("23_landsat_pan_sr_comparison.py")
                command = [
                    sys.executable,
                    str(script),
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
                    f"{site['country']} {vintage_label}{source['reference_year']} reference",
                    "--device",
                    args.device,
                ]
                if args.gee_project:
                    command.extend(["--gee-project", args.gee_project])
                if args.checkpoint:
                    command.extend(["--checkpoint", str(args.checkpoint)])
                if args.offline:
                    command.append("--offline")
                if args.prepare_only:
                    command.append("--prepare-only")
                print(f"Running {site['id']}: {site['region']}", flush=True)
                with (folder / "run.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=False
                    )
                if result.returncode:
                    raise RuntimeError(
                        f"Comparison exited {result.returncode}; see {folder / 'run.log'}"
                    )
                print(f"Completed {site['id']}", flush=True)
            except Exception as exc:
                failures.append(site["id"])
                status_path.write_text(
                    json.dumps(
                        {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, indent=2
                    ),
                    encoding="utf-8",
                )
                print(f"Unmeasured {site['id']}: {exc}", flush=True)
            summarize(config, digest, args.output_dir, args.french_output_root)
    summarize(config, digest, args.output_dir, args.french_output_root)
    if failures:
        raise SystemExit(f"Sites still unmeasured: {', '.join(failures)}")


if __name__ == "__main__":
    main()
