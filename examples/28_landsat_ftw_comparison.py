"""Frozen Landsat A-H versus published FTW products; accuracy and agreement separate.

See docs/user-guide/landsat-ftw-comparison.md. Prior suites are read only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box

from agribound.comparison_ftw import (
    ftw_variant,
    inside_aoi,
    product_agreement,
    select_product_coverage,
    validate_ftw_snapshot,
)
from agribound.config import AgriboundConfig
from agribound.engines.delineate_anything import DA_MODELS, file_sha256
from agribound.evaluate import evaluate, evaluate_frame
from agribound.ftw_arrow import FTW_VECTOR_LAYOUTS, PUBLISHED_YEARS
from agribound.ftw_query import query_ftw

ROOT = Path(__file__).resolve().parents[1]


def helper(filename):
    spec = importlib.util.spec_from_file_location(
        filename.replace(".", "_"), ROOT / "examples" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MS = helper("27_landsat_multispectral_comparison.py")
BASE, ADAPTERS, NA = MS.BASE, MS.ADAPTERS, MS.NA
INTERNATIONAL = helper("25_landsat_international_comparison.py")
save_json, read_json, digest = MS.save_json, MS.read_json, MS.digest
METRICS = [
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


def freeze(path, out):
    config = read_json(path)
    if config["schema_version"] != 1:
        raise ValueError("Unknown comparison configuration")
    parent = config["inherited_config"]
    if file_sha256(ROOT / parent["path"]) != parent["sha256"]:
        raise ValueError("Inherited frozen Landsat configuration changed")
    if config["comparison"]["checkpoint_sha256"] != DA_MODELS["large_v2"].sha256:
        raise ValueError("Checkpoint pin differs")
    if len({s["id"] for s in config["sites"]}) != len(config["sites"]):
        raise ValueError("Duplicate site IDs")
    for site in config["sites"]:
        if site["ftw_year"] not in PUBLISHED_YEARS:
            raise ValueError("Unsupported FTW year; do not score as empty")
        if site["bbox"][0] >= site["bbox"][2] or site["bbox"][1] >= site["bbox"][3]:
            raise ValueError("Invalid bbox")
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "experiments_frozen.json"
    if frozen.exists() and read_json(frozen) != config:
        raise ValueError("Frozen comparison changed; use another output directory")
    save_json(frozen, config)
    return config, digest(config)


def write_gpkg(frame, path, layer="fields"):
    copy = frame.copy()
    for column in copy.columns:
        if column != copy.geometry.name and copy[column].dtype == object:
            copy[column] = copy[column].map(
                lambda value: (
                    json.dumps(value, default=str)
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else value
                )
            )
    copy.to_file(path, driver="GPKG", layer=layer)


def checked_download(url, path, *, offline):
    pin = path.with_suffix(path.suffix + ".pin.json")
    if path.exists():
        if not pin.exists() or read_json(pin)["sha256"] != file_sha256(path):
            raise ValueError("Unverified or modified downloaded source")
        return read_json(pin)
    if offline:
        raise RuntimeError(f"Offline source missing: {path}")
    temporary = path.with_suffix(".part")
    with urlopen(url, timeout=180) as response, temporary.open("wb") as stream:
        headers = {k: response.headers.get(k) for k in ("ETag", "Last-Modified", "Content-Length")}
        shutil.copyfileobj(response, stream)
    temporary.replace(path)
    record = {"url": url, "headers": headers, "sha256": file_sha256(path)}
    save_json(pin, record)
    return record


def dutch_reference(source, site, folder, *, offline):
    target = folder / "provider_snapshot.geojson"
    pin = target.with_suffix(".pin.json")
    if target.exists():
        if not pin.exists() or read_json(pin)["sha256"] != file_sha256(target):
            raise ValueError("Dutch reference snapshot checksum mismatch")
    else:
        if offline:
            raise RuntimeError("Dutch WFS snapshot not cached")
        bounds = Transformer.from_crs(4326, source["crs"], always_xy=True).transform_bounds(
            *site["bbox"]
        )
        features, urls, seen = [], [], set()
        for start in range(0, 20000, 1000):
            params = dict(
                service="WFS",
                version="2.0.0",
                request="GetFeature",
                typeNames=source["type_name"],
                outputFormat="application/json",
                srsName=f"EPSG:{source['crs']}",
                count=1000,
                startIndex=start,
                bbox=",".join(map(str, bounds)) + f",urn:ogc:def:crs:EPSG::{source['crs']}",
            )
            url = source["url"] + "?" + urlencode(params)
            with urlopen(url, timeout=90) as response:
                page = json.load(response)
            rows = page["features"]
            ids = [row.get("id") for row in rows]
            if any(i is None or i in seen for i in ids) or len(set(ids)) != len(ids):
                raise ValueError("WFS pagination duplicated or omitted identifiers")
            seen.update(ids)
            features.extend(rows)
            urls.append(url)
            if len(rows) < 1000:
                break
        else:
            raise ValueError("WFS reference exceeds fixed pagination limit")
        target.write_text(
            json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
        )
        save_json(
            pin,
            {
                "sha256": file_sha256(target),
                "urls": urls,
                "n_features": len(features),
                "crs": source["crs"],
            },
        )
    data = read_json(target)
    frame = gpd.GeoDataFrame.from_features(data["features"], crs=source["crs"])
    if not frame["jaar"].eq(site["reference_year"]).all():
        raise ValueError("Live Dutch layer changed year; frozen year must not be replaced")
    frame = frame[frame.category.isin(["Bouwland", "Grasland", "Blijvende teelten"])]
    return frame, {
        **source,
        **read_json(pin),
        "crop_column": "gewas",
        "exclusion": "Landscape elements (ditches etc.) excluded by provider category",
    }


def reference_source(site, config, out, *, offline):
    """Materialize a new reference once, pin its geometry, preserve old suites."""
    old = ROOT / site["existing_dir"] if site["historical"] else out / "preparation" / site["id"]
    if site["historical"]:
        return old
    old.mkdir(parents=True, exist_ok=True)
    pin_path = old / "reference_pin.json"
    if pin_path.exists():
        if read_json(pin_path)["site_sha256"] != digest(site):
            raise ValueError("Prepared site configuration changed")
        for name, expected in read_json(pin_path)["files"].items():
            if file_sha256(old / name) != expected:
                raise ValueError("Prepared reference/configuration checksum mismatch")
        return old
    mode = site["reference_mode"]
    source = config["reference_sources"][mode]
    if mode == "rpg":
        if offline and not (old / "reference.gpkg").exists():
            raise RuntimeError("RPG reference not cached; run --prepare online")
        frame, metadata = BASE.download_reference(old, box(*site["bbox"]), site["reference_year"])
        metadata.update(source, crop_column="code_cultu")
    elif mode == "nl_wfs":
        frame, metadata = dutch_reference(source, site, old, offline=offline)
    elif mode == "qc_zip":
        cache = out / "reference_cache"
        cache.mkdir(exist_ok=True)
        path = cache / "qc2024.zip"
        downloaded = checked_download(source["url"], path, offline=offline)
        spec = {"adapter": "zip-shapefile", "encoding": "cp1252"}
        frame = ADAPTERS.read_reference(path, spec, site["bbox"])
        rules = [
            {"column": "NBPRO", "op": "ge", "value": 1},
            {"column": "TYPPAR", "op": "in", "value": ["PAC"]},
            {
                "column": "DESCODPR1",
                "op": "not-in",
                "value": ["Partie non cultivée", "Terre en friche"],
            },
        ]
        frame = ADAPTERS.filter_reference(frame, rules, year=site["reference_year"])
        metadata = {**source, **downloaded, "crop_column": "DESCODPR1", "filters": rules}
    elif mode == "vn_cached":
        path = ROOT / source["path"]
        if not path.exists():
            original = read_json(ROOT / "examples/landsat_international_sites.json")["sources"][
                "vietnam"
            ]
            path = INTERNATIONAL.fetch_reference(original, out / "reference_cache", offline=offline)
        if file_sha256(path) != source["sha256"]:
            raise ValueError("Vietnam reference changed")
        frame = gpd.read_parquet(path)
        metadata = {
            **source,
            "crop_column": None,
            "geometry_vintage": "Digitized from August 2021 imagery; not contemporary with 2024",
            "training_overlap": "FTW benchmark source; extracted patch split unknown",
        }
    else:
        raise ValueError("Unknown reference adapter")
    frame, repaired = INTERNATIONAL.select_reference(frame, site)
    write_gpkg(frame, old / "reference_evaluation.gpkg", "reference")
    write_gpkg(frame[[frame.geometry.name]], old / "evaluation_coverage.gpkg", "coverage")
    metadata.update(
        reference_year=site["reference_year"],
        n_reference=len(frame),
        n_repaired=repaired,
        physical_gold_standard=False,
        positional_uncertainty=site["position_uncertainty"],
        coverage="Known whole reference footprints; not exhaustive non-field truth",
        geometry_vintage=metadata.get(
            "geometry_vintage", "Declaration year; original physical geometry date uncertain"
        ),
        site=site,
    )
    crop = metadata.get("crop_column")
    metadata["crop_counts"] = (
        frame[crop].fillna("unknown").value_counts().to_dict()
        if crop in frame
        else {"unknown": len(frame)}
    )
    save_json(old / "reference_evaluation.json", metadata)
    geometry, crs = BASE.buffered_study_area(box(*site["bbox"]))
    cfg = AgriboundConfig(
        source="landsat-pan",
        engine="delineate-anything",
        year=int(site["date_start"][:4]),
        study_area=box(*site["bbox"]).wkt,
        date_range=(site["date_start"], site["date_end"]),
        gee_project="ee-rappjer",
        export_crs=crs,
        cloud_cover_max=20,
        lulc_filter=False,
        sam_refine=False,
        min_field_area_m2=1000,
        simplify_tolerance=2,
        seed=42,
        device="cpu",
        n_workers=0,
        cache_dir=str(old / ".agribound_cache"),
        engine_params={
            "backend": "native",
            "da_model": "large_v2",
            "super_resolution": 1,
            "half": False,
            "batch_size": 1,
            "conf_threshold": 0.15,
            "tile_step": 0.5,
        },
    )
    save_json(old / "configuration.json", cfg.to_dict())
    names = [
        "reference_evaluation.gpkg",
        "evaluation_coverage.gpkg",
        "reference_evaluation.json",
        "configuration.json",
    ]
    save_json(
        pin_path,
        {"site_sha256": digest(site), "files": {name: file_sha256(old / name) for name in names}},
    )
    return old


def prepare_site(site, config, out, args):
    folder = out / site["id"]
    folder.mkdir(exist_ok=True)
    old = reference_source(site, config, out, offline=args.offline)
    reference_path = old / "reference_evaluation.gpkg"
    reference = gpd.read_file(reference_path)
    metadata = read_json(old / "reference_evaluation.json")
    if site["coverage_policy"] == "aoi":
        coverage = gpd.GeoDataFrame(geometry=[box(*site["bbox"])], crs=4326)
    else:
        path = old / "evaluation_coverage.gpkg"
        coverage = (
            gpd.read_file(path) if path.exists() else reference[[reference.geometry.name]].copy()
        )
    pins = {
        "configuration": digest(config),
        "reference_sha256": file_sha256(reference_path),
        "coverage_geometry_sha256": digest(
            [g.hex() for g in coverage.to_crs(reference.crs).geometry.to_wkb()]
        ),
        "checkpoint_sha256": config["comparison"]["checkpoint_sha256"],
    }
    pin_path = folder / "reference_support.json"
    if pin_path.exists() and read_json(pin_path)["pins"] != pins:
        raise ValueError("Reference/configuration support changed")
    if not (folder / "reference.gpkg").exists():
        shutil.copy2(reference_path, folder / "reference.gpkg")
        write_gpkg(coverage, folder / "evaluation_coverage.gpkg", "coverage")
    save_json(
        pin_path,
        {
            "pins": pins,
            "metadata": metadata,
            "site": site,
            "coverage_policy": site["coverage_policy"],
            "reference_reused": site["historical"],
        },
    )
    sr = old / "inputs/landsat.tif"
    windows_path = folder / "figure_windows.json"
    if not windows_path.exists():
        if sr.exists():
            windows = MS.window_configuration(reference, sr, site)
        else:
            # The existing selector only reads the raster CRS. A one-cell CRS
            # fixture lets us freeze its reference-only windows before imagery.
            import rasterio
            from rasterio.transform import from_origin

            grid = folder / "window_crs_fixture.tif"
            with rasterio.open(
                grid,
                "w",
                driver="GTiff",
                height=1,
                width=1,
                count=1,
                dtype="uint8",
                crs=reference.estimate_utm_crs(),
                transform=from_origin(0, 1, 1, 1),
            ) as stream:
                stream.write(np.zeros((1, 1, 1), dtype="uint8"))
            windows = MS.window_configuration(reference, grid, site)
            windows["crs_fixture_is_imagery"] = False
        save_json(windows_path, windows)
    selection_pin = folder / "selection_integrity.json"
    selection = {
        "windows_sha256": file_sha256(windows_path),
        "site_sha256": digest(site),
        "reference_sha256": file_sha256(reference_path),
    }
    if selection_pin.exists() and read_json(selection_pin) != selection:
        raise ValueError("Frozen reference-only map selection changed")
    save_json(selection_pin, selection)
    crop = metadata.get("crop_column") or next(
        (
            c
            for c in (
                "code_cultu",
                "crop_type",
                "crop_name",
                "MAIN_CROP",
                "Description",
                "DESCODPR1",
            )
            if c in reference
        ),
        None,
    )
    crop_path = folder / "crop_strata.json"
    if crop in reference and not crop_path.exists():
        counts = reference[crop].fillna("unknown").astype(str).value_counts()
        mapping = {label: label if n >= 5 else "other_sparse_labels" for label, n in counts.items()}
        mapping["unknown"] = "unknown"
        save_json(
            crop_path,
            {
                "source_column": crop,
                "mapping": mapping,
                "counts": counts.to_dict(),
                "minimum_reference_count": 5,
                "selection": "Reference counts only; frozen before predictions",
            },
        )
    return old


def prepare_new_imagery(site, old, args):
    from agribound.composites.landsat_matched import export_matched, validate_matched_inputs

    cfg = AgriboundConfig(**read_json(old / "configuration.json"))
    geometry, _ = BASE.buffered_study_area(box(*site["bbox"]))
    paths = [old / "inputs/landsat-pan.tif", old / "inputs/landsat.tif"]
    start = time.perf_counter()
    if all(p.exists() for p in paths):
        manifest = read_json(old / "inputs/scene_manifest.json")
    else:
        if args.offline:
            raise RuntimeError("Matched new-site imagery not cached; run --prepare online")
        from agribound.auth import setup_gee

        setup_gee(project=args.gee_project, interactive=False)
        pan, sr, manifest = export_matched(cfg, geometry, old / "inputs")
        paths = [pan, sr]
    validate_matched_inputs(paths[0], paths[1], manifest, geometry)
    fused = old / "inputs/pan_sr_rgb.tif"
    if not fused.exists():
        diagnostic = BASE.make_fused(paths[1], paths[0], fused)
        save_json(old / "fusion_validation.json", diagnostic)
        save_json(
            old / "preparation_timing.json",
            {"acquisition_and_fusion_s": time.perf_counter() - start, "cached_imagery": False},
        )


def ftw_query_signature(site, config):
    return digest(
        {
            "site": site,
            "ftw": config["ftw"],
            "query_helper": file_sha256(ROOT / "agribound/ftw_query.py"),
            "arrow_helper": file_sha256(ROOT / "agribound/ftw_arrow.py"),
        }
    )


def retrieve_ftw(site, config, out, args):
    folder = out / site["id"]
    target, pin_path = folder / "ftw_raw.parquet", folder / "ftw_snapshot.json"
    signature = ftw_query_signature(site, config)
    if target.exists():
        if not pin_path.exists():
            raise ValueError("FTW cache has no provenance pin")
        record = read_json(pin_path)
        if record["signature"] != signature or record["sha256"] != file_sha256(target):
            raise ValueError(
                "FTW snapshot/configuration checksum mismatch; use new output directory"
            )
        frame = gpd.read_parquet(target)
        validate_ftw_snapshot(frame, site["ftw_year"], record["query"])
        return record
    if args.offline:
        raise RuntimeError("FTW snapshot missing in offline mode; run --query-ftw online")
    geometry, _ = BASE.buffered_study_area(box(*site["bbox"]))
    source = (
        FTW_VECTOR_LAYOUTS[config["ftw"]["layout"]].rstrip("/")
        + "/admin:country_code="
        + site["country_code"]
    )
    start = time.perf_counter()
    frame = query_ftw(
        study_area=geometry,
        year=site["ftw_year"],
        clip=False,
        deduplicate=True,
        min_confidence=None,
        keep_null_confidence=True,
        source_url=source,
        cache_dir=folder / "ftw_index",
        output_path=target,
    )
    info = frame.attrs["ftw_query"]
    validate_ftw_snapshot(frame, site["ftw_year"], info)
    record = {
        "signature": signature,
        "sha256": file_sha256(target),
        "query": info,
        "query_s": time.perf_counter() - start,
        "license": "CC-BY-4.0",
        "nominal_prediction_year": site["ftw_year"],
        "source_metadata_url": config["ftw"]["url"],
        "product_resolution_m": 10,
        "ftw_inference_s": None,
        "remote_version": "alpha results-by-admin-conf; local snapshot SHA-256 pin",
        "remote_file_metadata": {
            p.name: read_json(p) for p in (folder / "ftw_index").glob("partition_index*.json")
        },
    }
    save_json(pin_path, record)
    return record


def copy_landsat(site, method, config, out):
    source_folder = (
        ROOT / site["landsat_run_dir"] if site["historical"] else out / "landsat_runs" / site["id"]
    )
    name = method["id"]
    source = source_folder / f"fields_{name}.gpkg"
    status_path = source_folder / "run_status.json"
    if (
        not status_path.exists()
        or read_json(status_path).get("methods", {}).get(name, {}).get("status") != "complete"
    ):
        detail = (
            read_json(status_path).get("methods", {}).get(name, {}).get("error")
            if status_path.exists()
            else "Source run status missing"
        )
        raise RuntimeError(f"Source Landsat experiment unmeasured: {detail}")
    provenance = read_json(source.with_suffix(".gpkg.provenance.json"))
    source_hash = file_sha256(source)
    ms = provenance["multispectral_comparison"]
    if source_hash != ms["output_sha256"] or ms["logical_channels"] != method["channels"]:
        raise ValueError("Completed Landsat output/channel checksum mismatch")
    meta = provenance["engine_meta"]
    expected_meta = {
        "backend": "native",
        "model_key": "large_v2",
        "conf_threshold": 0.15,
        "super_resolution": 1,
        "tile_step": 0.5,
        "batch_size": 1,
        "precision": "fp32",
        "gsd_m": method["resolution_m"],
    }
    expected_config = {
        "min_field_area_m2": 1000,
        "simplify_tolerance": 2,
        "seed": 42,
        "lulc_filter": False,
        "sam_refine": False,
    }
    expected_bgr = {"pan": [1, 1, 1], "sr": [1, 2, 3]}.get(name, [3, 2, 1])
    if (
        any(meta.get(k) != v for k, v in expected_meta.items())
        or any(provenance["config"].get(k) != v for k, v in expected_config.items())
        or meta.get("band_indices_bgr") != expected_bgr
    ):
        raise ValueError("Completed Landsat inference/preprocessing settings differ")
    if (
        meta.get("checkpoint_sha256", meta.get("weights_sha256"))
        != config["comparison"]["checkpoint_sha256"]
    ):
        raise ValueError("Completed Landsat checkpoint differs")
    prep = read_json(source_folder / "input_provenance.json")
    for path, expected in prep["source_sha256"].items():
        if file_sha256(ROOT / path) != expected:
            raise ValueError("Completed Landsat input/reference source changed")
    for channel, expected in prep["generated_sha256"].items():
        if file_sha256(source_folder / "inputs" / f"{channel}.tif") != expected:
            raise ValueError("Completed multispectral input changed")
    folder = out / site["id"]
    signature = digest(
        {
            "configuration": digest(config),
            "source_output": source_hash,
            "inputs": prep["signature"],
            "reference": read_json(folder / "reference_support.json")["pins"],
        }
    )
    target = folder / source.name
    pin = target.with_suffix(".gpkg.provenance.json")
    if target.exists():
        if (
            not pin.exists()
            or read_json(pin).get("ftw_comparison", {}).get("signature") != signature
            or file_sha256(target) != source_hash
        ):
            raise ValueError("Local reused Landsat output is unverified or modified")
    else:
        shutil.copy2(source, target)
        provenance["ftw_comparison"] = {
            "signature": signature,
            "parent_output": str(source),
            "parent_sha256": source_hash,
            "inference_repeated": False,
            "historical": site["historical"],
        }
        save_json(pin, provenance)
    for filename in (
        "scene_manifest.json",
        "fusion_validation.json",
        "imagery_coverage.json",
        "input_provenance.json",
    ):
        if (source_folder / filename).exists() and not (folder / filename).exists():
            shutil.copy2(source_folder / filename, folder / filename)
    return {
        "status": "complete",
        "signature": signature,
        "inference_reused": site["historical"],
        "output_sha256": source_hash,
    }


def metric_rows(metric, site, product, scope, tolerance, stratum="all_crops"):
    rows = []
    for size, values in [("all", metric), *metric["per_size_class"].items()]:
        rows.append(
            dict(
                track="reference_evaluation",
                site=site["id"],
                country=site["country"],
                product=product,
                evaluation_scope=scope,
                primary_scope=scope == "reference_coverage",
                temporal_category=site["temporal_category"],
                reference_kind=site["reference_kind"],
                training_overlap=site["training_overlap"],
                boundary_tolerance_m=tolerance,
                size_class_ha=size,
                crop_stratum=stratum,
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
            )
        )
    return rows


def select_final_aoi(frame, site, folder, product):
    """Standardize final inclusion while preserving delivered vectors.

    Historical exports selected polygons before simplification and in their
    native CRS. Reuse inference, then select whole final polygons in geographic
    CRS for every system, retaining the original vectors unchanged.
    """
    selected = inside_aoi(frame, site["bbox"])
    target = folder / f"fields_{product}_aoi.gpkg"
    write_gpkg(selected, target)
    save_json(
        target.with_suffix(".gpkg.provenance.json"),
        {
            "input_sha256": file_sha256(folder / f"fields_{product}.gpkg"),
            "criterion": "Whole final polygons with representative points inside EPSG:4326 AOI",
            "bbox": site["bbox"],
            "n_delivered": len(frame),
            "n_retained": len(selected),
            "n_excluded": len(frame) - len(selected),
            "geometry_modified": False,
            "inference_repeated": False,
        },
    )
    return selected


def evaluate_site(site, config, out, *, force=False):
    start = time.perf_counter()
    folder = out / site["id"]
    if not (folder / "reference.gpkg").exists():
        return evaluate_reference_free(site, config, out)
    reference = gpd.read_file(folder / "reference.gpkg")
    coverage = gpd.read_file(folder / "evaluation_coverage.gpkg")
    if reference.empty or coverage.empty:
        raise ValueError("Unknown/empty reference coverage; reference accuracy unmeasured")
    # Preserve source vectors. Repair only the working mask used for geometric
    # operations; evaluate already documents its own reference/prediction repairs.
    invalid_coverage = int((~coverage.is_valid).sum())
    coverage.geometry = coverage.geometry.make_valid(method="structure")
    save_json(
        folder / "geometry_integrity.json",
        {
            "coverage_polygons_repaired_in_memory": invalid_coverage,
            "reference_invalid_input": int((~reference.is_valid).sum()),
            "raw_geometry_preserved": True,
        },
    )
    products, hashes = {}, {}
    statuses = read_json(folder / "product_status.json")["products"]
    for name, status in statuses.items():
        path = folder / f"fields_{name}.gpkg"
        if status["status"] == "complete" and path.exists():
            if file_sha256(path) != status["output_sha256"]:
                raise ValueError("Evaluation input GeoPackage changed")
            products[name] = gpd.read_file(path)
            hashes[name] = status["output_sha256"]
    signature = digest(
        {
            "products": hashes,
            "support": read_json(folder / "reference_support.json")["pins"],
            "evaluator": file_sha256(ROOT / "agribound/evaluate.py"),
            "agreement": file_sha256(ROOT / "agribound/comparison_ftw.py"),
            "runner": file_sha256(Path(__file__)),
        }
    )
    cache = folder / "evaluation_signature.json"
    if not force and cache.exists() and read_json(cache)["signature"] == signature:
        for name, expected in read_json(cache)["tables"].items():
            if file_sha256(folder / name) != expected:
                raise ValueError("Evaluation table changed")
        (folder / "evaluation_failure.json").unlink(missing_ok=True)
        return
    rows, agreement_rows = [], []
    selected = {}
    products = {
        name: select_final_aoi(frame, site, folder, name) for name, frame in products.items()
    }
    for product, frame in products.items():
        selected[product], selection = select_product_coverage(
            frame, coverage, ADAPTERS.select_coverage
        )
        write_gpkg(selected[product], folder / f"fields_{product}_evaluated.gpkg")
        save_json(
            folder / f"fields_{product}_evaluated.gpkg.provenance.json",
            {
                "input_sha256": hashes[product],
                "coverage": read_json(folder / "reference_support.json")["pins"],
                "selection": selection,
                "geometry_modified": False,
            },
        )
        for scope, pred, boundary_mask in [
            ("reference_coverage", selected[product], coverage),
            ("aoi_descriptive", frame, None),
        ]:
            for tolerance in config["comparison"]["boundary_tolerances_m"]:
                metric = evaluate(
                    pred,
                    reference,
                    size_bins=BASE.SIZE_BINS,
                    boundary_tolerance_m=tolerance,
                    boundary_mask=boundary_mask,
                    bootstrap=200,
                )
                save_json(folder / f"reference_metrics_{product}_{scope}_{tolerance}m.json", metric)
                rows.extend(metric_rows(metric, site, product, scope, tolerance))
        evaluate_frame(
            selected[product],
            reference,
            size_bins=BASE.SIZE_BINS,
            boundary_tolerance_m=15,
            boundary_mask=coverage,
        ).to_csv(folder / f"per_field_{product}.csv", index_label="reference_index")
    # Crop strata use reference attributes and their fixed footprints, not predicted crops.
    metadata = read_json(folder / "reference_support.json")["metadata"]
    crop = metadata.get("crop_column")
    if crop is None:
        crop = next(
            (
                c
                for c in (
                    "code_cultu",
                    "crop_type",
                    "crop_name",
                    "MAIN_CROP",
                    "Description",
                    "DESCODPR1",
                )
                if c in reference
            ),
            None,
        )
    if crop in reference:
        labels = reference[crop].fillna("unknown").astype(str)
        grouping = read_json(folder / "crop_strata.json")["mapping"]
        groups = labels.map(grouping)
        for label in sorted(groups.unique()):
            ref = reference[groups == label]
            if ref.empty:
                continue
            mask = ref[[ref.geometry.name]]
            for product, frame in products.items():
                pred, _ = select_product_coverage(frame, mask, ADAPTERS.select_coverage)
                metric = evaluate(
                    pred,
                    ref,
                    size_bins=BASE.SIZE_BINS,
                    boundary_tolerance_m=15,
                    boundary_mask=mask,
                    bootstrap=0,
                )
                rows.extend(metric_rows(metric, site, product, "reference_coverage", 15, label))
    if "ftw" in products:
        for method in config["methods"]:
            name = method["id"]
            if name not in products:
                continue
            for scope, preds, mask in [
                ("whole_aoi", products, None),
                ("reference_coverage", selected, coverage),
            ]:
                for tolerance in config["comparison"]["boundary_tolerances_m"]:
                    result = product_agreement(
                        preds[name],
                        preds["ftw"],
                        tolerance_m=tolerance,
                        size_bins=BASE.SIZE_BINS,
                        boundary_mask=mask,
                    )
                    agreement_rows.append(
                        {
                            "site": site["id"],
                            "country": site["country"],
                            "product": name,
                            "ftw_year": site["ftw_year"],
                            "landsat_year": int(site["date_start"][:4]),
                            "temporal_category": site["temporal_category"],
                            "evaluation_scope": scope,
                            **result,
                        }
                    )
    pd.DataFrame(rows).to_csv(folder / "reference_accuracy.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(folder / "prediction_agreement.csv", index=False)
    tables = {
        p.name: file_sha256(p)
        for p in folder.glob("*.csv")
        if p.name.startswith(("reference_accuracy", "prediction_agreement", "per_field"))
    }
    save_json(
        cache,
        {
            "signature": signature,
            "tables": tables,
            "evaluation_s": time.perf_counter() - start,
            "inference_repeated": False,
        },
    )
    (folder / "evaluation_failure.json").unlink(missing_ok=True)


def evaluate_reference_free(site, config, out):
    """A missing annotation source never prevents product agreement."""
    folder = out / site["id"]
    statuses = read_json(folder / "product_status.json")["products"]
    if statuses.get("ftw", {}).get("status") != "complete":
        return
    ftw = select_final_aoi(gpd.read_file(folder / "fields_ftw.gpkg"), site, folder, "ftw")
    rows = []
    for method in config["methods"]:
        name = method["id"]
        if statuses.get(name, {}).get("status") != "complete":
            continue
        left = select_final_aoi(gpd.read_file(folder / f"fields_{name}.gpkg"), site, folder, name)
        for tolerance in config["comparison"]["boundary_tolerances_m"]:
            result = product_agreement(left, ftw, tolerance_m=tolerance, size_bins=BASE.SIZE_BINS)
            rows.append(
                {
                    "site": site["id"],
                    "country": site["country"],
                    "product": name,
                    "ftw_year": site["ftw_year"],
                    "landsat_year": int(site["date_start"][:4]),
                    "temporal_category": site["temporal_category"],
                    "evaluation_scope": "whole_aoi",
                    **result,
                }
            )
    pd.DataFrame(rows).to_csv(folder / "prediction_agreement.csv", index=False)
    save_json(
        folder / "reference_evaluation_status.json",
        {
            "status": "unmeasured",
            "reason": "Suitable reference missing; agreement remains available",
        },
    )


def summarize(config, out):
    reference_tables, agreement_tables, status_rows, inventory = [], [], [], []
    audit_path = out / "benchmark_split_overlap.csv"
    audit = (
        pd.read_csv(audit_path).set_index("site").to_dict("index") if audit_path.exists() else {}
    )
    audit_helper = helper("landsat_ftw_training_audit.py")
    categories = {name: audit_helper.spatial_category(record) for name, record in audit.items()}
    for site in config["sites"]:
        folder = out / site["id"]
        statuses = (
            read_json(folder / "product_status.json")["products"]
            if (folder / "product_status.json").exists()
            else {}
        )
        for name in [m["id"] for m in config["methods"]] + config["ftw"]["variants"]:
            status_rows.append(
                {
                    "site": site["id"],
                    "product": name,
                    **statuses.get(name, {"status": "unmeasured", "error": "Not executed"}),
                }
            )
        for name, collection in [
            ("reference_accuracy.csv", reference_tables),
            ("prediction_agreement.csv", agreement_tables),
        ]:
            path = folder / name
            if (
                path.exists()
                and path.stat().st_size > 2
                and not (folder / "evaluation_failure.json").exists()
            ):
                table = pd.read_csv(path)
                # Never reintroduce stale tables after a product failure.
                table = table[
                    table["product"].isin(
                        [k for k, v in statuses.items() if v["status"] == "complete"]
                    )
                ]
                table["benchmark_spatial_category"] = table.site.map(categories).fillna("unknown")
                if (
                    name == "prediction_agreement.csv"
                    and statuses.get("ftw", {}).get("status") != "complete"
                ):
                    table = table.iloc[:0]
                collection.append(table)
        if (folder / "reference_support.json").exists():
            support = read_json(folder / "reference_support.json")
            meta = support["metadata"]
            inventory.append(
                {
                    "site": site["id"],
                    "country": site["country"],
                    "region": site["region"],
                    "bbox": json.dumps(site["bbox"]),
                    "landsat_window": json.dumps([site["date_start"], site["date_end"]]),
                    "ftw_year": site["ftw_year"],
                    "reference_year": site["reference_year"],
                    "reference_kind": site["reference_kind"],
                    "temporal_category": site["temporal_category"],
                    "training_overlap": site["training_overlap"],
                    "benchmark_split": site["benchmark_split"],
                    "climate": site["climate"],
                    "climate_evidence": site.get("climate_evidence"),
                    "crop_context": site["crop_context"],
                    "crop_evidence": json.dumps(site.get("crop_evidence")),
                    "license": meta.get("license", meta.get("attribution")),
                    "provider_links": json.dumps(
                        meta.get("sources", meta.get("documentation", meta.get("urls", [])))
                    ),
                    "geometry_vintage": meta.get(
                        "geometry_vintage",
                        meta.get(
                            "vintage_note",
                            "Declaration/reference year; physical edge date uncertain",
                        ),
                    ),
                    "position_uncertainty": meta.get(
                        "position_uncertainty",
                        meta.get("positional_uncertainty", "No numerical error bound documented"),
                    ),
                    "coverage_policy": site["coverage_policy"],
                    "reference_sha256": support["pins"]["reference_sha256"],
                    "redistribution": site["redistribution"],
                    "reference_dataset_reuse": "Source reuse; not new independent evidence"
                    if site["historical"] or site["reference_mode"] == "vn_cached"
                    else "New declaration-year snapshot; training spatial overlap unknown",
                }
            )
    pd.DataFrame(status_rows).to_csv(out / "product_status.csv", index=False)
    pd.DataFrame(inventory).to_csv(out / "reference_inventory.csv", index=False)
    agreement = (
        pd.concat(agreement_tables, ignore_index=True) if agreement_tables else pd.DataFrame()
    )
    agreement.to_csv(out / "prediction_agreement.csv", index=False)
    if not reference_tables:
        return
    reference = pd.concat(reference_tables, ignore_index=True)
    reference.to_csv(out / "reference_accuracy.csv", index=False)
    headline = reference[
        (reference.size_class_ha == "all")
        & (reference.crop_stratum == "all_crops")
        & (reference.boundary_tolerance_m == 15)
        & reference.primary_scope
    ]
    headline.to_csv(out / "reference_headline_15m.csv", index=False)
    rows = []
    keys = [
        "temporal_category",
        "reference_kind",
        "training_overlap",
        "benchmark_spatial_category",
        "product",
    ]
    for values, group in headline.groupby(keys):
        for weighting in ("equal_site", "reference_count"):
            weights = (
                np.ones(len(group)) if weighting == "equal_site" else group.n_reference.to_numpy()
            )
            row = dict(
                zip(keys, values, strict=True),
                weighting=weighting,
                n_sites=len(group),
                n_reference=int(group.n_reference.sum()),
            )
            for metric in METRICS:
                valid = np.isfinite(group[metric])
                row[metric] = (
                    np.average(group.loc[valid, metric], weights=weights[valid])
                    if valid.any()
                    else np.nan
                )
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "reference_aggregate.csv", index=False)
    pairs = []
    for left, right in [(m["id"], "ftw") for m in config["methods"]] + [
        tuple(p) for p in config["planned_landsat_contrasts"]
    ]:
        paired = headline[headline["product"] == left].merge(
            headline[headline["product"] == right], on="site", suffixes=("_left", "_right")
        )
        for _, r in paired.iterrows():
            pairs.append(
                {
                    "site": r.site,
                    "contrast": f"{left} minus {right}",
                    "n_reference": r.n_reference_left,
                    "temporal_category": r.temporal_category_left,
                    "reference_kind": r.reference_kind_left,
                    "benchmark_spatial_category": r.benchmark_spatial_category_left,
                    **{metric: r[f"{metric}_left"] - r[f"{metric}_right"] for metric in METRICS},
                }
            )
    pd.DataFrame(pairs).to_csv(out / "reference_paired_differences.csv", index=False)


def checkpoint_path(args, config):
    path = args.checkpoint or next(
        (ROOT / "outputs/landsat_pan_sr_comparison/model_cache").glob("**/DelineateAnythingv2.pt"),
        None,
    )
    if path is None or file_sha256(path) != config["comparison"]["checkpoint_sha256"]:
        raise ValueError("Provide cached --checkpoint with the pinned SHA-256")
    return path


def evaluation_job(item):
    site, config, out, force = item
    try:
        evaluate_site(site, config, out, force=force)
        print(f"Evaluation complete {site['id']}", flush=True)
    except Exception as error:
        save_json(
            out / site["id"] / "evaluation_failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        print(f"UNMEASURED evaluation {site['id']}: {error}", flush=True)


def figure_job(item):
    site, out = item
    try:
        module = helper("landsat_ftw_figures.py")
        data = module.load_site(site, out)
        if data is None:
            raise ValueError("Frozen views not prepared")
        module.site_maps(site, out, data)
        print(f"Maps complete {site['id']}", flush=True)
        return None
    except Exception as error:
        return {"site": site["id"], "error": f"{type(error).__name__}: {error}"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "examples/landsat_ftw_comparison_sites.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/landsat_ftw_comparison")
    parser.add_argument("--sites", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--gee-project",
        default="ee-rappjer",
        help=(
            "Earth Engine project; fallback: $GEE_PROJECT, gcloud, project_id in "
            "$AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or $GOOGLE_APPLICATION_CREDENTIALS"
        ),
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--query-workers", type=int, default=2)
    parser.add_argument("--evaluation-workers", type=int, default=4)
    parser.add_argument("--figure-workers", type=int, default=2)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--skip-site-maps",
        action="store_true",
        help="Regenerate representative/quantitative/input figures only",
    )
    stages = parser.add_mutually_exclusive_group()
    for flag in (
        "prepare",
        "query-ftw",
        "run-landsat",
        "evaluate-only",
        "figures-only",
        "summarize-only",
    ):
        stages.add_argument("--" + flag, action="store_true")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    out = args.output_dir.resolve()
    config, signature = freeze(args.config, out)
    sites = [s for s in config["sites"] if not args.sites or s["id"] in args.sites]
    methods = [m for m in config["methods"] if not args.methods or m["id"] in args.methods]
    if args.sites and set(args.sites) - {s["id"] for s in sites}:
        parser.error("Unknown site")
    if args.methods and set(args.methods) - {m["id"] for m in methods}:
        parser.error("Unknown method")
    if args.query_workers < 1 or args.query_workers > 4:
        parser.error("query-workers must be 1-4")
    if not 1 <= args.evaluation_workers <= 8:
        parser.error("evaluation-workers must be 1-8")
    if not 1 <= args.figure_workers <= 3:
        parser.error("figure-workers must be 1-3")
    all_stages = not any(
        (
            args.prepare,
            args.query_ftw,
            args.run_landsat,
            args.evaluate_only,
            args.figures_only,
            args.summarize_only,
        )
    )
    ready = []
    if not args.summarize_only and not args.figures_only:
        for site in sites:
            try:
                old = prepare_site(site, config, out, args)
                ready.append((site, old))
            except Exception as error:
                print(f"UNMEASURED reference {site['id']}: {error}", flush=True)
                folder = out / site["id"]
                folder.mkdir(exist_ok=True)
                save_json(
                    folder / "reference_evaluation_status.json",
                    {"status": "unmeasured", "error": str(error)},
                )
                # A missing reference must not prevent an independent FTW query.
                centre = box(*site["bbox"]).centroid
                zone = int((centre.x + 180) // 6) + 1
                crs = f"EPSG:{(32600 if centre.y >= 0 else 32700) + zone}"
                bounds = list(
                    gpd.GeoSeries([box(*site["bbox"])], crs=4326).to_crs(crs).iloc[0].bounds
                )
                save_json(
                    folder / "figure_windows.json",
                    {
                        "crs": crs,
                        "overview": bounds,
                        "small_fields": bounds,
                        "shared_edges": bounds,
                        "shared_edge_found": False,
                        "selection": "Reference-free fixed AOI grid",
                    },
                )
                ready.append((site, None))
    if args.prepare:
        for site, old in ready:
            if not site["historical"] and old is not None:
                try:
                    prepare_new_imagery(site, old, args)
                except Exception as error:
                    print(f"UNMEASURED imagery {site['id']}: {error}", flush=True)
    if args.query_ftw or all_stages:

        def query(item):
            site, old = item
            folder = out / site["id"]
            current = (
                read_json(folder / "product_status.json").get("products", {})
                if (folder / "product_status.json").exists()
                else {}
            )
            try:
                record = retrieve_ftw(site, config, out, args)
                buffered_raw = gpd.read_parquet(folder / "ftw_raw.parquet")
                raw_gpkg = folder / "ftw_raw.gpkg"
                raw_pin = raw_gpkg.with_suffix(".gpkg.provenance.json")
                if raw_gpkg.exists():
                    if (
                        not raw_pin.exists()
                        or read_json(raw_pin)["source_snapshot_sha256"] != record["sha256"]
                        or read_json(raw_pin)["output_sha256"] != file_sha256(raw_gpkg)
                    ):
                        raise ValueError("Buffered raw FTW GeoPackage changed")
                else:
                    write_gpkg(buffered_raw, raw_gpkg)
                    save_json(
                        raw_pin,
                        {
                            "source_snapshot_sha256": record["sha256"],
                            "output_sha256": file_sha256(raw_gpkg),
                            "query": record["query"],
                            "geometry_modified": False,
                            "query_buffer_m": 600,
                        },
                    )
                raw = inside_aoi(buffered_raw, site["bbox"])
                for variant in config["ftw"]["variants"]:
                    frame, diagnostic = ftw_variant(raw, variant)
                    diagnostic.update(
                        n_buffered_query_polygons=len(buffered_raw),
                        n_outside_representative_aoi=len(buffered_raw) - len(raw),
                    )
                    path = folder / f"fields_{variant}.gpkg"
                    if not path.exists():
                        write_gpkg(frame, path)
                    else:
                        pin = path.with_suffix(".gpkg.provenance.json")
                        if (
                            not pin.exists()
                            or read_json(pin)["output_sha256"] != file_sha256(path)
                            or read_json(pin)["source_snapshot_sha256"] != record["sha256"]
                        ):
                            raise ValueError("FTW derived output checksum mismatch")
                    save_json(
                        path.with_suffix(".gpkg.provenance.json"),
                        {
                            "product_kind": "published_prediction",
                            "source_snapshot_sha256": record["sha256"],
                            "diagnostic": diagnostic,
                            "output_sha256": file_sha256(path),
                            "query_inference_separation": "Query time; upstream inference unknown",
                        },
                    )
                    current[variant] = {
                        "status": "complete",
                        "output_sha256": file_sha256(path),
                        "n_polygons": len(frame),
                        "query_s": record["query_s"],
                    }
                save_json(
                    folder / "confidence_diagnostics.json",
                    {
                        variant: ftw_variant(raw, variant)[1]
                        for variant in config["ftw"]["variants"]
                    },
                )
                print(f"FTW complete {site['id']}: {len(raw)} polygons", flush=True)
            except Exception as error:
                for name in config["ftw"]["variants"]:
                    current[name] = {
                        "status": "unmeasured",
                        "error": f"{type(error).__name__}: {error}",
                    }
                print(f"UNMEASURED FTW {site['id']}: {error}", flush=True)
            save_json(folder / "product_status.json", {"products": current})

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.query_workers) as pool:
            list(pool.map(query, ready))
    if args.run_landsat or all_stages:
        try:
            checkpoint = checkpoint_path(args, config)
            checkpoint_error = None
        except (ValueError, OSError) as error:
            checkpoint, checkpoint_error = None, str(error)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(out / "ultralytics"))
        Path(os.environ["YOLO_CONFIG_DIR"]).joinpath("Ultralytics").mkdir(
            parents=True, exist_ok=True
        )
        for site, old in ready:
            folder = out / site["id"]
            current = (
                read_json(folder / "product_status.json").get("products", {})
                if (folder / "product_status.json").exists()
                else {}
            )
            try:
                if old is None or checkpoint is None:
                    raise RuntimeError(
                        checkpoint_error or "Reference/configuration preparation blocked"
                    )
                if not site["historical"]:
                    prepare_new_imagery(site, old, args)
                    run_args = SimpleNamespace(
                        device=args.device,
                        gee_project=args.gee_project,
                        offline=args.offline,
                        bootstrap_existing=False,
                        prepare_only=False,
                        evaluate_only=False,
                        figures_only=False,
                    )
                    dynamic = {**site, "existing_dir": str(old.relative_to(ROOT))}
                    run_dir = out / "landsat_runs"
                    run_dir.mkdir(exist_ok=True)
                    MS.run_site(
                        dynamic,
                        methods,
                        signature,
                        run_dir,
                        run_args,
                        checkpoint,
                        config["comparison"]["checkpoint_sha256"],
                    )
                for method in methods:
                    try:
                        current[method["id"]] = copy_landsat(site, method, config, out)
                        print(f"Landsat ready {site['id']} {method['id']}", flush=True)
                    except Exception as error:
                        current[method["id"]] = {
                            "status": "unmeasured",
                            "error": f"{type(error).__name__}: {error}",
                        }
            except Exception as error:
                for method in methods:
                    current[method["id"]] = {
                        "status": "unmeasured",
                        "error": f"{type(error).__name__}: {error}",
                    }
                print(f"UNMEASURED Landsat {site['id']}: {error}", flush=True)
            save_json(folder / "product_status.json", {"products": current})
    if args.evaluate_only or all_stages:
        jobs = [(site, config, out, args.evaluate_only) for site, _old in ready]
        if args.evaluation_workers == 1:
            for job in jobs:
                evaluation_job(job)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.evaluation_workers
            ) as pool:
                list(pool.map(evaluation_job, jobs))
    summarize(config, out)
    if not args.skip_figures and not args.prepare and not args.query_ftw and not args.run_landsat:
        jobs = [] if args.skip_site_maps else [(site, out) for site in config["sites"]]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.figure_workers) as pool:
            failures = [r for r in pool.map(figure_job, jobs) if r is not None]
        helper("landsat_ftw_figures.py").generate(config, out, render_maps=False, failures=failures)
    helper("landsat_ftw_report.py").build(config, out)
    if all_stages:
        status = pd.read_csv(out / "product_status.csv")
        if (status[status.site.isin([s["id"] for s in sites])].status != "complete").any():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
