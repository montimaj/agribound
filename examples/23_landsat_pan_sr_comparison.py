"""Compare native Landsat PAN, SR RGB and conservative PAN+SR image fusion.

Run from the repository root; see docs/user-guide/landsat-pan-comparison.md.
Reference: public IGN RPG 2023 agricultural parcels near Beauce, France.
Matched Landsat 8/9 scenes: May-August 2023, <=20% scene cloud cover.
All experiments use the same released Delineate-Anything large_v2 checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from agribound._repro import new_run_id, seed_everything
from agribound.composites.landsat_matched import export_matched, validate_matched_inputs
from agribound.composites.pan_fusion import inject_pan_detail, reduced_resolution_validation
from agribound.config import AgriboundConfig
from agribound.engines import get_engine
from agribound.evaluate import evaluate, evaluate_frame
from agribound.io.raster import write_raster
from agribound.io.vector import write_vector
from agribound.pipeline import select_in_study_area
from agribound.postprocess import filter_polygons, simplify_polygons
from agribound.provenance import RunRecorder, write_provenance

DEFAULT_BBOX = (1.62, 48.13, 1.68, 48.17)
SIZE_BINS = [0, 1, 5, 20, 100, 100000]
WFS_URL = "https://data.geopf.fr/wfs/ows"


def save_json(path, data):
    """Write strict JSON, representing undefined numerical values as null."""

    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    Path(path).write_text(json.dumps(clean(data), indent=2, allow_nan=False), encoding="utf-8")


def download_reference(out_dir, geometry, year=2023):
    """Download all intersecting parcels from the year-specific public IGN WFS."""
    from pyproj import Transformer

    path = out_dir / "reference.gpkg"
    bounds = Transformer.from_crs(4326, 2154, always_xy=True).transform_bounds(*geometry.bounds)
    request_key = {"bbox_2154": list(bounds), "year": year, "service": WFS_URL}
    metadata_path = out_dir / "reference_source.json"
    if path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta["request"] != request_key:
            raise ValueError(
                "Reference cache belongs to another area/year; use a new output directory"
            )
        return gpd.read_file(path), meta
    features, urls = [], []
    for offset in range(0, 20000, 1000):
        url = (
            WFS_URL
            + "?"
            + urlencode(
                {
                    "SERVICE": "WFS",
                    "VERSION": "2.0.0",
                    "REQUEST": "GetFeature",
                    "TYPENAMES": f"RPG.{year}:parcelles_graphiques",
                    "OUTPUTFORMAT": "application/json",
                    "SRSNAME": "EPSG:2154",
                    "BBOX": ",".join(map(str, bounds)) + ",urn:ogc:def:crs:EPSG::2154",
                    "COUNT": 1000,
                    "STARTINDEX": offset,
                    "SORTBY": "id_parcel",
                }
            )
        )
        with urlopen(url, timeout=90) as response:
            page = json.load(response)
        urls.append(url)
        rows = page["features"]
        features.extend(rows)
        if len(rows) < 1000:
            break
    else:
        raise ValueError("Reference query exceeded 20,000 rows; choose a smaller area")
    if not features:
        raise ValueError("No RPG reference parcels intersect the study area")
    reference = gpd.GeoDataFrame.from_features(features, crs=2154)
    if "id_parcel" in reference and reference["id_parcel"].duplicated().any():
        raise ValueError("WFS returned duplicate parcels across pages")
    write_vector(reference, path)
    meta = {
        "request": request_key,
        "urls": urls,
        "reference_year": year,
        "sha256": hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest(),
        "n_downloaded": len(reference),
        "attribution": "IGN / ASP, Registre Parcellaire Graphique, Licence Ouverte 2.0",
        "limitations": (
            "Declared agricultural parcels, not exhaustive physical field truth; "
            "declaration splits may be invisible"
        ),
    }
    save_json(metadata_path, meta)
    return reference, meta


def make_fused(sr_path, pan_path, output):
    """Fuse aligned temporal composites and validate their degraded RGB means."""
    with rasterio.open(sr_path) as src, rasterio.open(pan_path) as pan_src:
        if src.crs != pan_src.crs or pan_src.transform != src.transform * rasterio.Affine.scale(
            0.5
        ):
            raise ValueError("PAN and SR must share CRS, extent and an exact aligned 2:1 grid")
        rgb = src.read([3, 2, 1], masked=True).filled(np.nan) / 10000.0
        pan = pan_src.read(1, masked=True).filled(np.nan)
        transform, crs, tags = pan_src.transform, pan_src.crs, pan_src.tags()
    fused, diagnostic = inject_pan_detail(rgb, pan)
    diagnostic["reduced_resolution"] = reduced_resolution_validation(rgb, pan)
    diagnostic["sr_valid_fraction"] = float(np.isfinite(rgb).all(axis=0).mean())
    diagnostic["pan_valid_fraction"] = float(np.isfinite(pan).mean())
    if diagnostic["coarse_max_abs_error"] > 1e-6:
        raise ValueError("Fusion failed coarse-scale SR mean-preservation validation")
    write_raster(output, fused, crs, transform, nodata=np.nan, dtype="float32")
    tags.update(
        {
            "AGRIBOUND_SOURCE": "local PAN+SR experimental RGB fusion",
            "AGRIBOUND_VALUE_SCALE": "unit",
            "AGRIBOUND_FUSION": diagnostic["method"],
            "AGRIBOUND_PARENT_SR": str(sr_path),
            "AGRIBOUND_PARENT_PAN": str(pan_path),
            "AGRIBOUND_FUSION_DIAGNOSTICS": json.dumps(diagnostic),
        }
    )
    with rasterio.open(output, "r+") as dst:
        dst.set_band_description(1, "R_fused")
        dst.set_band_description(2, "G_fused")
        dst.set_band_description(3, "B_fused")
        dst.update_tags(**tags)
    return diagnostic


def run_experiment(name, raster, config, reference, aoi, manifest, output_dir, reference_meta=None):
    """Run Agribound's engine, common polygon processing and provenance recorder.

    Inputs are already exported under controlled pairing, so the normal
    automatic composite stage is deliberately bypassed for all three runs.
    Config.source still selects native band mapping and engine source metadata.
    """
    output = output_dir / f"fields_{name}.gpkg"
    run_id = new_run_id()
    config = config.merged(
        output_path=str(output), cache_dir=str(output_dir / ".agribound_cache" / run_id)
    )
    recorder = RunRecorder(config, run_id=run_id)
    try:
        with recorder:
            seed_everything(config.seed)
            recorder.set(
                "workflow", "matched prepared-input comparison; automatic composites bypassed"
            )
            recorder.set("scene_manifest", manifest)
            recorder.set("reference", reference_meta)
            recorder.set("raster_path", str(raster))
            with rasterio.open(raster) as src:
                recorder.set("composite", src.tags())
                resolution = src.res[0]
            recorder.set("lulc_status", "disabled for all experiments")
            recorder.set(
                "postprocess",
                {
                    "min_field_area_m2": config.min_field_area_m2,
                    "simplify_tolerance_m": config.simplify_tolerance,
                    "smooth_iterations": 0,
                },
            )
            with recorder.step("delineate"):
                start = time.perf_counter()
                predicted = get_engine(config.engine).delineate(str(raster), config)
                inference_s = time.perf_counter() - start
            recorder.record_engine_meta(predicted.attrs.get("engine_meta", {}))
            with recorder.step("postprocess"):
                outline = gpd.GeoSeries([aoi], crs=4326).to_crs(predicted.crs).iloc[0]
                predicted, selection = select_in_study_area(
                    predicted, outline, "representative_point"
                )
                recorder.set("aoi_selection", selection)
                predicted = filter_polygons(predicted, min_area_m2=config.min_field_area_m2)
                predicted = simplify_polygons(predicted, tolerance=config.simplify_tolerance)
                predicted = filter_polygons(predicted, min_area_m2=config.min_field_area_m2)
            rows = []
            with recorder.step("evaluate"):
                for tolerance in (10, 15, 30):
                    metrics = evaluate(
                        predicted,
                        reference,
                        size_bins=SIZE_BINS,
                        boundary_tolerance_m=tolerance,
                        bootstrap=200,
                    )
                    save_json(output_dir / f"metrics_{name}_{tolerance}m.json", metrics)
                    for size, values in [("all", metrics), *metrics["per_size_class"].items()]:
                        rows.append(
                            {
                                "experiment": name,
                                "resolution_m": resolution,
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
                                "inference_s": inference_s,
                            }
                        )
                frame = evaluate_frame(
                    predicted, reference, size_bins=SIZE_BINS, boundary_tolerance_m=15
                )
                frame.to_csv(output_dir / f"per_field_{name}.csv", index_label="reference_index")
                recorder.set(
                    "evaluation_metrics_15m",
                    json.loads(
                        (output_dir / f"metrics_{name}_15m.json").read_text(encoding="utf-8")
                    ),
                )
            with recorder.step("export"):
                predicted["experiment"] = name
                predicted["input_resolution_m"] = resolution
                write_vector(predicted, output)
            recorder.set("n_output", len(predicted))
            recorder.set("output_path", str(output))
    finally:
        write_provenance(output, recorder.to_dict())
    return predicted, rows


def comparison_figure(
    sr_path,
    predictions,
    reference,
    aoi,
    output_dir,
    reference_label="RPG 2023 reference",
    date_label="May-August 2023",
):
    """Identical SR background and reference-selected zooms in every column."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    with rasterio.open(sr_path) as src:
        rgb = src.read([3, 2, 1], masked=True).filled(np.nan)
        crs, bounds = src.crs, src.bounds
    rendered = np.zeros_like(rgb)
    for i, band in enumerate(rgb):
        valid = band[np.isfinite(band)]
        low, high = np.percentile(valid, [1, 99])
        rendered[i] = np.clip((band - low) / max(high - low, 1), 0, 1)
    background = np.moveaxis(np.nan_to_num(rendered), 0, -1)
    ref = reference.to_crs(crs)
    ref.geometry = ref.geometry.make_valid(method="structure")
    extent = gpd.GeoSeries([aoi], crs=4326).to_crs(crs).iloc[0].bounds
    areas = ref.to_crs(6933).area
    smallest = ref.loc[areas[areas >= 1000].idxmin()].geometry.representative_point()
    middle = ref.loc[areas.sort_values().index[len(ref) // 2]].geometry.representative_point()
    shared = None
    for i, geom in enumerate(ref.geometry):
        for j in ref.sindex.query(geom, predicate="intersects"):
            if j > i:
                edge = geom.boundary.intersection(ref.geometry.iloc[j].boundary)
                if edge.length > 30:
                    shared = edge.centroid
                    break
        if shared is not None:
            break
    centre = shared if shared is not None else middle
    zooms = [
        extent,
        (smallest.x - 450, smallest.y - 450, smallest.x + 450, smallest.y + 450),
        (centre.x - 450, centre.y - 450, centre.x + 450, centre.y + 450),
    ]
    save_json(
        output_dir / "figure_windows.json",
        {
            "crs": str(crs),
            "bounds": zooms,
            "selection": "reference geometry only, before comparing predictions",
        },
    )
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for column, (name, layer) in enumerate(predictions.items()):
        pred = layer.to_crs(crs)
        for row, window in enumerate(zooms):
            ax = axes[row, column]
            ax.imshow(
                background,
                extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                interpolation="nearest",
            )
            if len(ref):
                ref.boundary.plot(ax=ax, color="cyan", linewidth=0.8)
            if len(pred):
                pred.boundary.plot(ax=ax, color="magenta", linewidth=0.8)
            ax.set_xlim(window[0], window[2])
            ax.set_ylim(window[1], window[3])
            ax.set_aspect("equal")
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(
                    {
                        "pan": "PAN only (15 m)",
                        "sr": "SR only (30 m)",
                        "combined": "PAN + SR (15 m)",
                    }[name]
                )
    axes[0, 0].set_ylabel("Study area")
    axes[1, 0].set_ylabel("Small field (900 m view)")
    axes[2, 0].set_ylabel("Shared edges (900 m view)")
    fig.legend(
        handles=[
            Line2D([0], [0], color="cyan", label=reference_label),
            Line2D([0], [0], color="magenta", label="Prediction"),
        ],
        loc="lower center",
        ncol=2,
    )
    fig.suptitle(f"Matched Landsat 8/9, {date_label}; identical 30 m SR background")
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.savefig(output_dir / "comparison.png", dpi=180)
    fig.savefig(output_dir / "comparison.pdf")
    plt.close(fig)


def parse_args(argv=None):
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
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/landsat_pan_sr_comparison")
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=DEFAULT_BBOX,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Download paired inputs and reference, then stop",
    )
    parser.add_argument("--reference", type=Path, help="Use a local reference instead of IGN WFS")
    parser.add_argument("--reference-year", type=int, default=2023)
    parser.add_argument(
        "--reference-metadata", type=Path, help="JSON provenance for local references"
    )
    parser.add_argument("--reference-label", default="RPG 2023 reference")
    parser.add_argument("--date-start", default="2023-05-01")
    parser.add_argument("--date-end", default="2023-08-31", help="Inclusive final acquisition date")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Reuse existing reference and inputs without Earth Engine/network access",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Local released large_v2 checkpoint (SHA-256 checked)"
    )
    return parser.parse_args(argv)


def buffered_study_area(aoi):
    """Choose the site's UTM zone and provide 600 m of inference context."""
    study_area = gpd.GeoSeries([aoi], crs=4326)
    export_crs = study_area.estimate_utm_crs()
    export_geometry = study_area.to_crs(export_crs).buffer(600).to_crs(4326).iloc[0]
    return export_geometry, str(export_crs)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "run_status.json", {"status": "running"})
    aoi = box(*args.bbox)
    if not aoi.is_valid or aoi.area <= 0:
        raise ValueError("bbox must have west<east and south<north")
    export_geometry, export_crs = buffered_study_area(aoi)
    base = AgriboundConfig(
        source="landsat-pan",
        year=int(args.date_start[:4]),
        date_range=(args.date_start, args.date_end),
        gee_project=args.gee_project,
        study_area=aoi.wkt,
        engine="delineate-anything",
        cloud_cover_max=20,
        export_crs=str(export_crs),
        lulc_filter=False,
        sam_refine=False,
        min_field_area_m2=1000,
        simplify_tolerance=2,
        device=args.device,
        n_workers=0,
        cache_dir=str(out / ".agribound_cache"),
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
    save_json(out / "configuration.json", base.to_dict())
    if not args.offline:
        from agribound.auth import setup_gee

        setup_gee(project=base.gee_project, interactive=False)
    if args.reference:
        reference = gpd.read_file(args.reference)
        reference_meta = (
            json.loads(args.reference_metadata.read_text(encoding="utf-8"))
            if args.reference_metadata
            else {}
        )
        reference_meta.update(
            {
                "path": str(args.reference),
                "reference_year": args.reference_year,
                "local_file_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            }
        )
    else:
        if args.offline and not (out / "reference.gpkg").exists():
            raise ValueError("Offline mode needs an existing reference.gpkg or --reference")
        reference, reference_meta = download_reference(out, export_geometry, args.reference_year)
    reference, _ = select_in_study_area(
        reference,
        gpd.GeoSeries([aoi], crs=4326).to_crs(reference.crs).iloc[0],
        "representative_point",
    )
    if len(reference) < 2:
        raise ValueError("Study area needs multiple reference fields")
    write_vector(reference, out / "reference_evaluation.gpkg")
    reference_meta.update(
        {
            "n_evaluated": len(reference),
            "imagery_year": base.year,
            "vintage_matches": all(int(d[:4]) == args.reference_year for d in base.date_range)
            and reference_meta.get("vintage_verified", True),
            "area_ha_quantiles": reference.to_crs(6933)
            .area.div(10000)
            .quantile([0, 0.25, 0.5, 0.75, 1])
            .to_dict(),
        }
    )
    save_json(out / "reference_evaluation.json", reference_meta)
    start = time.perf_counter()
    reused_inputs = all(
        (out / "inputs" / f"{source}.tif").exists() for source in ("landsat-pan", "landsat")
    )
    if args.offline:
        from agribound.composites.gee import date_window

        if not reused_inputs:
            raise ValueError("Offline mode needs both prepared Landsat input rasters")
        manifest = json.loads((out / "inputs" / "scene_manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["date_window_end_exclusive"] != list(date_window(base))
            or manifest["cloud_cover_max"] != base.cloud_cover_max
        ):
            raise ValueError("Offline input manifest does not match this experiment")
        pan, sr = out / "inputs" / "landsat-pan.tif", out / "inputs" / "landsat.tif"
    else:
        pan, sr, manifest = export_matched(base, export_geometry, out / "inputs")
    validate_matched_inputs(pan, sr, manifest, export_geometry)
    acquisition_s = time.perf_counter() - start
    fused = out / "inputs" / "pan_sr_rgb.tif"
    start = time.perf_counter()
    fusion_diagnostic = make_fused(sr, pan, fused)
    fusion_s = time.perf_counter() - start
    save_json(out / "fusion_validation.json", fusion_diagnostic)
    save_json(
        out / "preparation_timing.json",
        {
            "paired_acquisition_s": acquisition_s,
            "fusion_s": fusion_s,
            "inputs_already_present": reused_inputs,
            "note": "Shared acquisition; may reuse inputs. Excludes model download.",
        },
    )
    if args.prepare_only:
        save_json(out / "run_status.json", {"status": "prepared", "experiments": []})
        print(f"Prepared matched imagery and {len(reference)} reference parcels in {out}")
        return
    from agribound.engines.delineate_anything import DA_MODELS, download_da_weights

    # Download and verify once, outside inference timings; every run gets the same file.
    if args.checkpoint:
        from agribound.engines.delineate_anything import file_sha256

        if file_sha256(args.checkpoint) != DA_MODELS["large_v2"].sha256:
            raise ValueError("--checkpoint must match the pinned published large_v2 weights")
        checkpoint = str(args.checkpoint)
    elif args.offline:
        raise ValueError(
            "Offline delineation needs --checkpoint with the released large_v2 weights"
        )
    else:
        checkpoint = download_da_weights("large_v2")
    os.environ.setdefault("YOLO_CONFIG_DIR", str((out / "ultralytics").resolve()))
    Path(os.environ["YOLO_CONFIG_DIR"]).joinpath("Ultralytics").mkdir(parents=True, exist_ok=True)
    base = base.merged(engine_params={**base.engine_params, "checkpoint_path": checkpoint})
    rows, predictions = [], {}
    for name, raster, cfg in (
        ("pan", pan, base),
        ("sr", sr, base.merged(source="landsat")),
        (
            "combined",
            fused,
            base.merged(source="local", local_tif_path=str(fused), bands={"R": 1, "G": 2, "B": 3}),
        ),
    ):
        print(f"Delineating {name} from {raster}", flush=True)
        predictions[name], values = run_experiment(
            name, raster, cfg, reference, aoi, manifest, out, reference_meta=reference_meta
        )
        rows.extend(values)
        pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    comparison_figure(
        sr,
        predictions,
        reference,
        aoi,
        out,
        reference_label=args.reference_label,
        date_label=f"{args.date_start} through {args.date_end}",
    )
    table = pd.DataFrame(rows)
    headline = table[(table.size_class_ha == "all") & (table.boundary_tolerance_m == 15)]
    print(headline.to_string(index=False))
    save_json(
        out / "run_status.json",
        {
            "status": "complete",
            "experiments": list(predictions),
            "reference": reference_meta,
            "fusion": fusion_diagnostic,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save_json(
            parse_args().output_dir / "run_status.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
