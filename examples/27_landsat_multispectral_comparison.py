"""Run frozen Landsat NIR/stack/hybrid comparisons, reusing verified baselines.

See docs/user-guide/landsat-multispectral-comparison.md. Cached inputs permit
offline execution; only missing imagery needs Earth Engine project ee-rappjer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from agribound.composites.landsat_matched import validate_matched_inputs
from agribound.composites.landsat_multispectral import (
    CHANNELS,
    build_inputs,
    hybrid_reduced_validation,
    validate_grids,
)
from agribound.composites.pan_fusion import inject_pan_detail
from agribound.config import AgriboundConfig
from agribound.engines.delineate_anything import DA_MODELS, file_sha256
from agribound.evaluate import evaluate, evaluate_frame
from agribound.io.raster import write_raster

ROOT = Path(__file__).resolve().parents[1]


def load_helper(filename):
    spec = importlib.util.spec_from_file_location(
        filename.replace(".", "_"), ROOT / "examples" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_helper("23_landsat_pan_sr_comparison.py")
ADAPTERS = load_helper("landsat_reference_adapters.py")
NA = load_helper("26_landsat_north_america_comparison.py")
save_json = BASE.save_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def validate_prepared_raster(path, method_id, prepared):
    """Never adopt an existing input whose recorded checksum is unavailable."""
    expected = prepared.get("generated_sha256", {}).get(method_id)
    if expected is None:
        raise ValueError(
            "Prepared raster lacks matching input provenance; use another output directory"
        )
    if file_sha256(path) != expected:
        raise ValueError("Prepared raster checksum mismatch")


def freeze(config_path, out):
    config = read_json(config_path)
    if config["schema_version"] != 1 or len(config["methods"]) != 8:
        raise ValueError("Expected version 1 with eight methods")
    ids = [s["id"] for s in config["sites"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate site IDs")
    if config["comparison"]["checkpoint_sha256"] != DA_MODELS["large_v2"].sha256:
        raise ValueError("Checkpoint pin differs from registry")
    for parent in config["inherited_configs"]:
        if file_sha256(ROOT / parent["path"]) != parent["sha256"]:
            raise ValueError("Inherited site configuration changed")
    for method in config["methods"]:
        if not method["baseline"] and method["channels"] != CHANNELS[method["id"]]:
            raise ValueError("Frozen channel order differs from implementation")
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "experiments_frozen.json"
    if frozen.exists() and read_json(frozen) != config:
        raise ValueError("Frozen configuration changed; use another output directory")
    save_json(frozen, config)
    return config, digest(config)


def window_configuration(reference, sr_path, site):
    """Select 900 m windows from reference geometry only, in fixed input order."""
    with rasterio.open(sr_path) as src:
        crs = src.crs
    ref = reference.to_crs(crs)
    areas = ref.to_crs(6933).area
    eligible = areas[areas >= 1000]
    smallest = ref.loc[
        (eligible if len(eligible) else areas).idxmin()
    ].geometry.representative_point()
    centre = None
    for i, geom in enumerate(ref.geometry):
        for j in sorted(ref.sindex.query(geom, predicate="intersects")):
            if j > i:
                edge = geom.boundary.intersection(ref.geometry.iloc[j].boundary)
                if edge.length > 30:
                    centre = edge.centroid
                    break
        if centre is not None:
            break
    shared_found = centre is not None
    if centre is None:
        centre = ref.loc[areas.sort_values().index[len(ref) // 2]].geometry.representative_point()

    def zoom(p):
        return [p.x - 450, p.y - 450, p.x + 450, p.y + 450]

    return {
        "crs": str(crs),
        "overview": list(gpd.GeoSeries([box(*site["bbox"])], crs=4326).to_crs(crs).iloc[0].bounds),
        "small_fields": zoom(smallest),
        "shared_edges": zoom(centre),
        "shared_edge_found": shared_found,
        "small_field_area_ha": float(areas.min() / 10000),
        "selection": "Smallest >=1000 m2 reference; first >30 m shared edge; median-area fallback",
        "small_field_note": (
            "Smallest model-area-eligible field; smaller references remain evaluated"
        ),
        "shared_edge_note": "Actual reference shared edge"
        if shared_found
        else "No exact >30 m shared edge; median-area reference fallback",
    }


def prepare_site(site, folder, args):
    old = ROOT / site["existing_dir"]
    prerequisites = [old / "configuration.json", old / "reference_evaluation.gpkg"]
    if not all(path.exists() for path in prerequisites):
        if not args.bootstrap_existing or args.offline:
            raise RuntimeError(
                "Existing reference/configuration missing; prepare prior suites or use "
                "--bootstrap-existing online"
            )
        if site["cohort"] == "france":
            command = [
                sys.executable,
                str(ROOT / "examples/23_landsat_pan_sr_comparison.py"),
                "--output-dir",
                str(old),
                "--bbox",
                *map(str, site["bbox"]),
            ]
        else:
            number = "25" if site["cohort"] == "international" else "26"
            script = next((ROOT / "examples").glob(f"{number}_*.py"))
            command = [
                sys.executable,
                str(script),
                "--output-dir",
                str(old.parent),
                "--sites",
                site["id"],
            ]
        command += ["--prepare-only", "--gee-project", args.gee_project]
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode or not all(path.exists() for path in prerequisites):
            raise RuntimeError("Prior reference/imagery preparation failed; see child output")
    inputs = old / "inputs"
    if not (inputs / "landsat.tif").exists():
        if args.offline:
            raise RuntimeError(
                "No cached paired imagery; fixed-window scene selection previously failed"
            )
        from agribound.auth import setup_gee
        from agribound.composites.landsat_matched import export_matched

        cfg = AgriboundConfig(**read_json(old / "configuration.json")).merged(
            gee_project=args.gee_project
        )
        setup_gee(project=args.gee_project, interactive=False)
        inputs = folder / "acquisition"
        export_geometry, _ = BASE.buffered_study_area(box(*site["bbox"]))
        pan_path, sr_path, manifest = export_matched(cfg, export_geometry, inputs)
        fused_path = inputs / "pan_sr_rgb.tif"
        BASE.make_fused(sr_path, pan_path, fused_path)
    else:
        pan_path, sr_path = inputs / "landsat-pan.tif", inputs / "landsat.tif"
        fused_path = inputs / "pan_sr_rgb.tif"
        manifest = read_json(inputs / "scene_manifest.json")
    expected_end = (pd.Timestamp(site["date_end"]) + pd.Timedelta(days=1)).date().isoformat()
    if manifest["date_window_end_exclusive"] != [site["date_start"], expected_end]:
        raise ValueError("Cached scene window differs from frozen site")
    if manifest["cloud_cover_max"] != 20 or not set(manifest["missions"]).issubset(
        {"LC08", "LC09"}
    ):
        raise ValueError("Expected <=20% paired Landsat 8/9")
    export_geometry, _ = BASE.buffered_study_area(box(*site["bbox"]))
    validate_matched_inputs(pan_path, sr_path, manifest, export_geometry)
    reference_path = old / "reference_evaluation.gpkg"
    reference = gpd.read_file(reference_path)
    metadata = read_json(old / "reference_evaluation.json")
    source_paths = [sr_path, pan_path, fused_path, reference_path, inputs / "scene_manifest.json"]
    coverage_path = old / "evaluation_coverage.gpkg"
    if coverage_path.exists():
        source_paths.append(coverage_path)
    source_hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in source_paths}
    signature = digest(
        {
            "sources": source_hashes,
            "implementation": file_sha256(ROOT / "agribound/composites/landsat_multispectral.py"),
        }
    )
    prepared_path = folder / "input_provenance.json"
    prepared = read_json(prepared_path) if prepared_path.exists() else {}
    if prepared and prepared["signature"] != signature:
        raise ValueError("Input parents/implementation changed; use a new output directory")
    raster_paths = {"pan": pan_path, "sr": sr_path, "combined": fused_path}
    start = time.perf_counter()
    with (
        rasterio.open(sr_path) as sr_src,
        rasterio.open(pan_path) as pan_src,
        rasterio.open(fused_path) as f_src,
    ):
        validate_grids(sr_src, pan_src, f_src)
        if sr_src.descriptions != tuple(f"SR_B{i}" for i in range(2, 8)):
            raise ValueError("SR raster band descriptions must explicitly identify B2--B7 order")
        if sr_src.tags().get("AGRIBOUND_VALUE_SCALE") != "reflectance_x10000":
            raise ValueError("Expected the established SR reflectance_x10000 scaling")
        sr = sr_src.read(masked=True).filled(np.nan).astype(np.float32) / 10000.0
        pan = pan_src.read(1, masked=True).filled(np.nan)
        fused = f_src.read(masked=True).filled(np.nan)
        arrays, integrity = build_inputs(sr, pan, fused)
        # Preserve original inference only if every existing input has common support.
        if not all(
            integrity[k]
            for k in (
                "baseline_sr_support_identical",
                "baseline_pan_support_identical",
                "baseline_fused_support_identical",
            )
        ):
            raise ValueError("Baseline support differs; requires explicit preserved baseline rerun")
        regenerated, diagnostic = inject_pan_detail(sr[[2, 1, 0]], pan)
        if not np.allclose(regenerated, fused, atol=1e-7, equal_nan=True):
            raise ValueError("Cached visible fusion differs from established implementation")
        for name, array in arrays.items():
            path = folder / "inputs" / f"{name}.tif"
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                transform = sr_src.transform if name == "false_color" else pan_src.transform
                write_raster(path, array, sr_src.crs, transform, nodata=np.nan, dtype="float32")
                with rasterio.open(path, "r+") as dst:
                    for i, label in enumerate(CHANNELS[name], 1):
                        dst.set_band_description(i, label)
                    dst.update_tags(
                        AGRIBOUND_VALUE_SCALE="unit; per-channel TOA/SR/fused identities",
                        AGRIBOUND_CHANNELS=json.dumps(CHANNELS[name]),
                        AGRIBOUND_PARENT_SIGNATURE=signature,
                        AGRIBOUND_MASK="All six SR bands and all four PAN subpixels",
                    )
            else:
                validate_prepared_raster(path, name, prepared)
            raster_paths[name] = path
    preparation_s = prepared.get("preparation_s", time.perf_counter() - start)
    existing_fusion = (
        read_json(old / "fusion_validation.json")
        if (old / "fusion_validation.json").exists()
        else diagnostic
    )
    save_json(
        folder / "fusion_validation.json",
        {
            "existing_rgb": existing_fusion,
            "hybrid_visible": hybrid_reduced_validation(sr[[2, 1, 0]], pan),
            "integrity": integrity,
        },
    )
    save_json(
        prepared_path,
        {
            "signature": signature,
            "source_sha256": source_hashes,
            "generated_sha256": {
                k: file_sha256(v) for k, v in raster_paths.items() if k in CHANNELS
            },
            "preparation_s": preparation_s,
            "cached_on_this_call": bool(prepared),
            "radiometry": integrity["channel_radiometry"],
            "integrity": integrity,
        },
    )
    save_json(folder / "scene_manifest.json", manifest)
    windows_path = folder / "figure_windows.json"
    windows = window_configuration(reference, sr_path, site)
    if windows_path.exists() and read_json(windows_path) != windows:
        raise ValueError("Reference-selected map windows changed")
    save_json(windows_path, windows)
    scopes = {"aoi": gpd.GeoDataFrame(geometry=[box(*site["bbox"])], crs=4326)}
    if coverage_path.exists():
        scopes["known_footprints"] = gpd.read_file(coverage_path)
    save_json(folder / "imagery_coverage.json", NA.imagery_coverage(sr_path, scopes))
    save_json(
        folder / "reference_inventory.json",
        {
            "site": site,
            "reference": metadata,
            "source_sha256": source_hashes,
            "training_geographic_overlap": "Not documented; held-out status unknown",
        },
    )
    return raster_paths, reference, metadata, manifest, scopes, signature


def validate_baseline(provenance, method, manifest, checkpoint_hash):
    if provenance["status"] != "success":
        raise ValueError("Baseline was not successful")
    meta, cfg = provenance["engine_meta"], provenance["config"]
    expected = {
        "backend": "native",
        "model_key": "large_v2",
        "conf_threshold": 0.15,
        "super_resolution": 1,
        "tile_step": 0.5,
        "batch_size": 1,
        "precision": "fp32",
    }
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("Baseline inference settings differ")
    if meta.get("checkpoint_sha256", meta.get("weights_sha256")) != checkpoint_hash:
        raise ValueError("Baseline checkpoint differs")
    if any(
        cfg.get(k) != v
        for k, v in {
            "min_field_area_m2": 1000,
            "simplify_tolerance": 2,
            "seed": 42,
            "lulc_filter": False,
            "sam_refine": False,
        }.items()
    ):
        raise ValueError("Baseline postprocessing differs")
    if provenance["facts"]["scene_manifest"] != manifest:
        raise ValueError("Baseline scene selection differs")
    if meta["gsd_m"] != method["resolution_m"]:
        raise ValueError("Baseline grid resolution differs")
    expected_bgr = {"pan": [1, 1, 1], "sr": [1, 2, 3], "combined": [3, 2, 1]}
    if meta.get("band_indices_bgr") != expected_bgr[method["id"]]:
        raise ValueError("Baseline logical channel order differs")


def method_signature(config_hash, input_signature, raster, method, checkpoint_hash, device):
    return digest(
        {
            "configuration": config_hash,
            "inputs": input_signature,
            "raster_sha256": file_sha256(raster),
            "method": method,
            "checkpoint": checkpoint_hash,
            "device": device,
            "engine_sha256": file_sha256(ROOT / "agribound/engines/delineate_anything.py"),
        }
    )


def reuse_baseline(site, method, folder, signature, manifest, checkpoint_hash):
    name = method["id"]
    source = ROOT / site["existing_dir"] / f"fields_{name}.gpkg"
    parent = read_json(source.with_suffix(".gpkg.provenance.json"))
    validate_baseline(parent, method, manifest, checkpoint_hash)
    target = folder / source.name
    shutil.copy2(source, target)
    parent["multispectral_comparison"] = {
        "signature": signature,
        "inference_reused": True,
        "parent_output": str(source),
        "parent_sha256": file_sha256(source),
        "logical_channels": method["channels"],
    }
    save_json(target.with_suffix(".gpkg.provenance.json"), parent)
    return gpd.read_file(target), parent


def evaluate_method(site, method, predicted, parent, reference, scopes, folder):
    """Whole polygon matching; boundary mask intersects original lines only."""
    rows = []
    inference_s = next(s["wall_s"] for s in parent["steps"] if s["name"] == "delineate")
    postprocess_s = next(s["wall_s"] for s in parent["steps"] if s["name"] == "postprocess")
    for scope, mask in scopes.items():
        if scope == "aoi":
            selected, selection = predicted, {}
            boundary_mask = None
        else:
            selected, selection = ADAPTERS.select_coverage(predicted, mask)
            boundary_mask = mask
            output = folder / f"fields_{method['id']}_evaluated.gpkg"
            selected.to_file(output, driver="GPKG", layer="fields")
            save_json(
                output.with_suffix(".gpkg.provenance.json"),
                {
                    **parent,
                    "coverage_evaluation": selection,
                    "coverage": "Frozen footprints; whole polygons; original lines inside mask",
                },
            )
        for tolerance in (10, 15, 30):
            metric = evaluate(
                selected,
                reference,
                size_bins=BASE.SIZE_BINS,
                boundary_tolerance_m=tolerance,
                boundary_mask=boundary_mask,
                bootstrap=200,
            )
            save_json(folder / f"metrics_{scope}_{method['id']}_{tolerance}m.json", metric)
            for size, values in [("all", metric), *metric["per_size_class"].items()]:
                rows.append(
                    dict(
                        site=site["id"],
                        country=site["country"],
                        cohort=site["cohort"],
                        reference_kind=site["reference_kind"],
                        evaluation_scope=scope,
                        experiment=method["id"],
                        resolution_m=method["resolution_m"],
                        boundary_tolerance_m=tolerance,
                        size_class_ha=size,
                        n_reference=values.get("n", metric["count_reference"]),
                        n_predicted=values["count_predicted"],
                        precision=values["precision"],
                        recall=values["recall"],
                        f1=values["f1"],
                        matched_iou=values["iou_mean"],
                        boundary_precision=values["boundary_precision"],
                        boundary_recall=values["boundary_recall"],
                        boundary_f1=values["boundary_f1"],
                        oversegmentation=values["oversegmentation_mean"],
                        undersegmentation=values["undersegmentation_mean"],
                        inference_s=inference_s,
                        postprocess_s=postprocess_s,
                        timing_device=parent["device"],
                        timing_machine=parent["machine"],
                        timing_platform=parent["platform"],
                        timing_torch=parent["versions"].get("torch"),
                        inference_reused=parent["multispectral_comparison"]["inference_reused"],
                    )
                )
        evaluate_frame(
            selected,
            reference,
            size_bins=BASE.SIZE_BINS,
            boundary_tolerance_m=15,
            boundary_mask=boundary_mask,
        ).to_csv(folder / f"per_field_{scope}_{method['id']}.csv", index_label="reference_index")
    table = pd.DataFrame(rows)
    table.to_csv(folder / f"comparison_{method['id']}.csv", index=False)
    return table


def run_site(site, methods, config_hash, out, args, checkpoint, checkpoint_hash):
    folder = out / site["id"]
    folder.mkdir(exist_ok=True)
    rasters, reference, metadata, manifest, scopes, input_signature = prepare_site(
        site, folder, args
    )
    base = AgriboundConfig(**read_json(ROOT / site["existing_dir"] / "configuration.json"))
    base = base.merged(
        device=args.device,
        gee_project=args.gee_project,
        engine_params={**base.engine_params, "checkpoint_path": str(checkpoint)},
    )
    statuses = (
        read_json(folder / "run_status.json").get("methods", {})
        if (folder / "run_status.json").exists()
        else {}
    )
    for method in methods:
        name = method["id"]
        if args.prepare_only:
            statuses.setdefault(name, {"status": "prepared"})
            continue
        signature = method_signature(
            config_hash, input_signature, rasters[name], method, checkpoint_hash, args.device
        )
        target = folder / f"fields_{name}.gpkg"
        provenance_path = target.with_suffix(".gpkg.provenance.json")
        try:
            if target.exists() and provenance_path.exists():
                parent = read_json(provenance_path)
                if parent.get("multispectral_comparison", {}).get("signature") != signature:
                    raise ValueError(
                        "Completed output signature differs; use another output directory"
                    )
                if file_sha256(target) != parent["multispectral_comparison"]["output_sha256"]:
                    raise ValueError("Completed GeoPackage checksum mismatch")
                predicted = gpd.read_file(target)
            elif args.evaluate_only or args.figures_only:
                raise RuntimeError("Requested output has not been delineated")
            elif method["baseline"] and (ROOT / site["existing_dir"] / target.name).exists():
                predicted, parent = reuse_baseline(
                    site, method, folder, signature, manifest, checkpoint_hash
                )
            else:
                print(f"Delineating {site['id']} {method['name']} {method['channels']}", flush=True)
                cfg = base.merged(
                    source="local",
                    local_tif_path=str(rasters[name]),
                    bands={"R": 1, "G": 2, "B": 3},
                )
                if method["baseline"]:
                    cfg = (
                        base.merged(source="landsat-pan" if name == "pan" else "landsat")
                        if name != "combined"
                        else cfg
                    )
                predicted, _ = BASE.run_experiment(
                    name,
                    rasters[name],
                    cfg,
                    reference,
                    box(*site["bbox"]),
                    manifest,
                    folder,
                    reference_meta=metadata,
                )
                parent = read_json(provenance_path)
                parent["multispectral_comparison"] = {
                    "signature": signature,
                    "inference_reused": False,
                    "logical_channels": method["channels"],
                }
            parent["multispectral_comparison"]["output_sha256"] = file_sha256(target)
            save_json(provenance_path, parent)
            evaluator_signature = digest(
                {
                    "signature": signature,
                    "evaluator": file_sha256(ROOT / "agribound/evaluate.py"),
                    "runner": file_sha256(Path(__file__)),
                }
            )
            old_status = statuses.get(name, {})
            if args.figures_only and not (folder / f"comparison_{name}.csv").exists():
                raise RuntimeError("Figure-only execution requires existing evaluation tables")
            if not args.figures_only and (
                args.evaluate_only
                or old_status.get("evaluation_signature") != evaluator_signature
                or not (folder / f"comparison_{name}.csv").exists()
            ):
                evaluate_method(site, method, predicted, parent, reference, scopes, folder)
            statuses[name] = {
                "status": "complete",
                "signature": signature,
                "evaluation_signature": (
                    old_status.get("evaluation_signature")
                    if args.figures_only
                    else evaluator_signature
                ),
                "inference_reused": parent["multispectral_comparison"]["inference_reused"],
            }
            print(
                f"Complete {site['id']} {name}; reused={statuses[name]['inference_reused']}",
                flush=True,
            )
        except Exception as error:
            statuses[name] = {"status": "unmeasured", "error": f"{type(error).__name__}: {error}"}
            print(f"UNMEASURED {site['id']} {name}: {error}", flush=True)
        save_json(folder / "run_status.json", {"methods": statuses})
    save_json(folder / "run_status.json", {"methods": statuses})


def summarize(config, out):
    tables, statuses = [], []
    for site in config["sites"]:
        folder = out / site["id"]
        current = (
            read_json(folder / "run_status.json").get("methods", {})
            if (folder / "run_status.json").exists()
            else {}
        )
        for method in config["methods"]:
            status = current.get(
                method["id"], {"status": "unmeasured", "error": "Not yet executed"}
            )
            statuses.append({"site": site["id"], "experiment": method["id"], **status})
            path = folder / f"comparison_{method['id']}.csv"
            if status["status"] == "complete" and path.exists():
                table = pd.read_csv(path)
                imagery = read_json(folder / "imagery_coverage.json")
                manifest = read_json(folder / "scene_manifest.json")
                table["matched_scenes"] = len(manifest["pairs"])
                table["unmatched_pan"] = len(manifest["unmatched_pan"])
                table["unmatched_sr"] = len(manifest["unmatched_sr"])
                table["preparation_s"] = read_json(folder / "input_provenance.json")[
                    "preparation_s"
                ]
                fractions = {
                    s: imagery[s]["valid_fraction"] for s in table.evaluation_scope.unique()
                }
                table["valid_imagery_fraction"] = table.evaluation_scope.map(fractions)
                table["primary_scope"] = table.evaluation_scope == (
                    "known_footprints" if site["cohort"] == "north_america" else "aoi"
                )
                tables.append(table)
    pd.DataFrame(statuses).to_csv(out / "suite_status.csv", index=False)
    if not tables:
        return
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "comparison.csv", index=False)
    headline = table[
        (table.size_class_ha == "all") & (table.boundary_tolerance_m == 15) & table.primary_scope
    ]
    headline.to_csv(out / "headline_15m.csv", index=False)
    aggregates = []
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
    ]
    for category, values in [
        ("all_primary_scopes", headline),
        *[
            (kind, headline[headline.reference_kind == kind])
            for kind in headline.reference_kind.unique()
        ],
    ]:
        for method, group in values.groupby("experiment"):
            for weighting in ("equal_site", "reference_count"):
                weights = (
                    np.ones(len(group))
                    if weighting == "equal_site"
                    else group.n_reference.to_numpy()
                )
                row = dict(
                    reference_category=category,
                    experiment=method,
                    weighting=weighting,
                    n_sites=len(group),
                    n_reference=int(group.n_reference.sum()),
                )
                for metric in metrics:
                    valid = np.isfinite(group[metric])
                    row[metric] = (
                        np.average(group.loc[valid, metric], weights=weights[valid])
                        if valid.any()
                        else np.nan
                    )
                    row[f"{metric}_n_sites"] = int(valid.sum())
                aggregates.append(row)
    pd.DataFrame(aggregates).to_csv(out / "aggregate_metrics.csv", index=False)
    differences = []
    for left, right in config["planned_contrasts"]:
        pair = headline[headline.experiment == left].merge(
            headline[headline.experiment == right],
            on=["site", "evaluation_scope"],
            suffixes=("_left", "_right"),
        )
        for _, row in pair.iterrows():
            result = {
                "site": row.site,
                "contrast": f"{left} minus {right}",
                "n_reference": row.n_reference_left,
            }
            for metric in metrics:
                result[metric] = row[f"{metric}_left"] - row[f"{metric}_right"]
            differences.append(result)
    pd.DataFrame(differences).to_csv(out / "paired_differences.csv", index=False)
    save_json(
        out / "suite_status.json",
        {
            "sites": statuses,
            "measured_site_count": headline.site.nunique(),
            "completed_methods": sum(s["status"] == "complete" for s in statuses),
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "examples/landsat_multispectral_sites.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/landsat_multispectral")
    parser.add_argument(
        "--gee-project",
        default="ee-rappjer",
        help=(
            "Earth Engine project; fallback: $GEE_PROJECT, gcloud, project_id in "
            "$AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or $GOOGLE_APPLICATION_CREDENTIALS"
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--sites", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--bootstrap-existing",
        action="store_true",
        help="Prepare missing prior reference/configuration using examples 23/25/26; online only",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    args.output_dir = args.output_dir.resolve()
    config, config_hash = freeze(args.config, args.output_dir)
    sites = [s for s in config["sites"] if not args.sites or s["id"] in args.sites]
    methods = [m for m in config["methods"] if not args.methods or m["id"] in args.methods]
    if args.sites and set(args.sites) - {s["id"] for s in sites}:
        parser.error("Unknown site")
    if args.methods and set(args.methods) - {m["id"] for m in methods}:
        parser.error("Unknown method")
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = next(
            (
                path
                for cache in (
                    ROOT / "outputs/landsat_pan_sr_comparison/model_cache",
                    args.output_dir / "model_cache",
                )
                for path in cache.glob("**/DelineateAnythingv2.pt")
            ),
            None,
        )
    checkpoint_hash = config["comparison"]["checkpoint_sha256"]
    needs_weights = not (
        args.summarize_only or args.figures_only or args.evaluate_only or args.prepare_only
    )
    if needs_weights and checkpoint is None and not args.offline:
        from huggingface_hub import hf_hub_download

        from agribound.engines.delineate_anything import DA_HF_REPO

        spec = DA_MODELS["large_v2"]
        checkpoint = Path(
            hf_hub_download(
                repo_id=DA_HF_REPO,
                filename=spec.filename,
                revision=spec.revision,
                cache_dir=str(args.output_dir / "model_cache"),
            )
        )
    if needs_weights and (checkpoint is None or file_sha256(checkpoint) != checkpoint_hash):
        raise ValueError("Provide --checkpoint matching the pinned large_v2 SHA-256")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output_dir / "ultralytics"))
    Path(os.environ["YOLO_CONFIG_DIR"]).joinpath("Ultralytics").mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        # All reference-only windows and representative membership are frozen before inference.
        prepared_sites = []
        for site in sites:
            folder = args.output_dir / site["id"]
            folder.mkdir(exist_ok=True)
            try:
                prepare_site(site, folder, args)
                prepared_sites.append(site)
            except Exception as error:
                save_json(
                    folder / "run_status.json",
                    {
                        "methods": {
                            m["id"]: {
                                "status": "unmeasured",
                                "error": f"{type(error).__name__}: {error}",
                            }
                            for m in config["methods"]
                        }
                    },
                )
                print(f"UNMEASURED {site['id']}: {error}", flush=True)
        if not args.prepare_only:
            for site in prepared_sites:
                run_site(
                    site, methods, config_hash, args.output_dir, args, checkpoint, checkpoint_hash
                )
    summarize(config, args.output_dir)
    save_json(
        args.output_dir / "model_pin.json",
        {
            "model": "large_v2",
            "revision": DA_MODELS["large_v2"].revision,
            "sha256": checkpoint_hash,
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "training_inputs": "512x512 RGB patches; see author paper",
            "source": "https://arxiv.org/html/2607.19069v1",
        },
    )
    if not args.skip_figures and not args.prepare_only:
        plots = load_helper("landsat_multispectral_figures.py")
        plots.generate(config, args.output_dir)
    report = load_helper("landsat_multispectral_report.py")
    report.build(config, args.output_dir)
    if not args.prepare_only and not args.summarize_only:
        status = pd.read_csv(args.output_dir / "suite_status.csv")
        requested = status[
            status.site.isin([s["id"] for s in sites])
            & status.experiment.isin([m["id"] for m in methods])
        ]
        if (requested.status != "complete").any():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
