"""Thin tutorial orchestration over shared Landsat, FTW and map helpers.

No downloads, inference or plotting imports occur at import time. The bundle
contains measured products; offline execution prepares controls and reevaluates
those products, rather than pretending to rerun a model without weights.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from agribound.comparison_ftw import (
    ftw_variant,
    inside_aoi,
    product_agreement,
    select_product_coverage,
    validate_ftw_snapshot,
)
from agribound.composites.landsat_multispectral import (
    CHANNELS,
    build_inputs,
    hybrid_reduced_validation,
    validate_grids,
)
from agribound.composites.pan_fusion import inject_pan_detail, reduced_resolution_validation
from agribound.evaluate import evaluate
from agribound.ftw_query import query_ftw
from agribound.io.raster import write_raster
from agribound.io.vector import write_vector

METHODS = {
    "pan": ["PAN B8 TOA"] * 3,
    "sr": ["SR_B4", "SR_B3", "SR_B2"],
    "combined": ["R_fused", "G_fused", "B_fused"],
    **CHANNELS,
}
SIZE_BINS = [0, 1, 5, 20, 100, 100000]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_code_hashes():
    """Pin implementations used by live inference, not only its configuration."""
    import agribound

    package = Path(agribound.__file__).parent
    names = [
        "pipeline.py",
        "config.py",
        "registry.py",
        "engines/delineate_anything.py",
        "composites/landsat_matched.py",
        "composites/pan_fusion.py",
        "composites/landsat_multispectral.py",
        "evaluate.py",
        "io/raster.py",
    ]
    paths = [
        *[(name, package / name) for name in names],
        *[(str(p.relative_to(package)), p) for p in (package / "postprocess").glob("*.py")],
    ]
    return {name: sha256(path) for name, path in paths}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_bundle(bundle):
    """Fail before reuse if a curated input, result or reference has changed."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported tutorial bundle schema")
    for entry in manifest["files"]:
        path = bundle / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Tutorial bundle hash mismatch: {entry['path']}")
    return manifest


def prepare_inputs(bundle, out, *, live=False, project=None):
    """Prepare D-H and validate the established visible fusion without NIR injection."""
    manifest = load_bundle(bundle)
    out.mkdir(parents=True, exist_ok=True)
    inputs = bundle / "inputs"
    site = manifest["site"]
    export_geometry = (
        gpd.GeoSeries([box(*site["bbox"])], crs=4326).to_crs(32631).buffer(600).to_crs(4326).iloc[0]
    )
    if live:
        from agribound.auth import setup_gee
        from agribound.composites.landsat_matched import export_matched, validate_matched_inputs
        from agribound.config import AgriboundConfig

        inputs = out / "live_inputs"
        config = AgriboundConfig(
            source="landsat-pan",
            year=site["ftw_year"],
            gee_project=project,
            date_range=(site["date_start"], site["date_end"]),
            cloud_cover_max=manifest["comparison"]["cloud_cover_max"],
            export_crs="EPSG:32631",
            study_area=box(*site["bbox"]).wkt,
            cache_dir=str(out / "cache"),
        )
        setup_gee(project=config.gee_project, interactive=False)
        inputs.mkdir(parents=True, exist_ok=True)
        pan_path, sr_path, scene_manifest = export_matched(config, export_geometry, inputs)
        validate_matched_inputs(pan_path, sr_path, scene_manifest, export_geometry)
        save_json(out / "live_scene_manifest.json", scene_manifest)
    with (
        rasterio.open(inputs / "landsat.tif") as src,
        rasterio.open(inputs / "landsat-pan.tif") as pan_src,
    ):
        validate_grids(src, pan_src)
        sr = src.read(masked=True).filled(np.nan).astype(np.float32) / 10000
        pan = pan_src.read(1, masked=True).filled(np.nan)
        crs, high_transform, low_transform = src.crs, pan_src.transform, src.transform
    fused, fusion = inject_pan_detail(sr[[2, 1, 0]], pan)
    fused_path = inputs / "pan_sr_rgb.tif"
    if live and not fused_path.exists():
        write_raster(fused_path, fused, crs, high_transform, nodata=np.nan)
    with (
        rasterio.open(fused_path) as src,
        rasterio.open(inputs / "landsat.tif") as low,
        rasterio.open(inputs / "landsat-pan.tif") as high,
    ):
        validate_grids(low, high, src)
        cached = src.read(masked=True).filled(np.nan)
    if not np.allclose(fused, cached, rtol=0, atol=1e-7, equal_nan=True):
        raise ValueError("Established fusion behavior changed; do not reuse old predictions")
    fused = cached
    data, integrity = build_inputs(sr, pan, fused)
    if not all(
        integrity[k]
        for k in (
            "baseline_sr_support_identical",
            "baseline_pan_support_identical",
            "baseline_fused_support_identical",
        )
    ):
        raise ValueError("A-H do not have identical observation support")
    paths = {
        "pan": inputs / "landsat-pan.tif",
        "sr": inputs / "landsat.tif",
        "combined": fused_path,
    }
    for method, array in data.items():
        path = out / f"input_{method}.tif"
        write_raster(
            path,
            array,
            crs,
            low_transform if method == "false_color" else high_transform,
            nodata=np.nan,
        )
        paths[method] = path
    diagnostic = {
        "integrity": integrity,
        "visible_fusion": fusion,
        "reduced_rgb": reduced_resolution_validation(sr[[2, 1, 0]], pan),
        "reduced_hybrid": hybrid_reduced_validation(sr[[2, 1, 0]], pan),
        "logical_channels": METHODS,
        "mode": "live" if live else "offline cached measurements",
        "bundle_sha256": sha256(bundle / "manifest.json"),
        "inputs_sha256": {k: sha256(v) for k, v in paths.items()},
        "scientific_code_sha256": scientific_code_hashes(),
        "export_buffer_m": 600,
    }
    save_json(out / "input_integrity.json", diagnostic)
    return paths, diagnostic


def run_landsat(bundle, out, paths, *, live=False, checkpoint=None, project=None):
    """Use measured cached products by default; inference requires explicit opt-in."""
    if not live:
        return {method: bundle / f"predictions/fields_{method}.gpkg" for method in METHODS}
    from agribound._repro import new_run_id, seed_everything
    from agribound.engines import get_engine
    from agribound.engines.delineate_anything import DA_MODELS
    from agribound.pipeline import select_in_study_area
    from agribound.postprocess import filter_polygons, simplify_polygons
    from agribound.provenance import RunRecorder, write_provenance

    manifest = load_bundle(bundle)
    spec = DA_MODELS["large_v2"]
    if not checkpoint or sha256(checkpoint) != manifest["comparison"]["checkpoint_sha256"]:
        raise ValueError("Live inference requires --checkpoint with the pinned large_v2 SHA-256")
    site = manifest["site"]
    result = {}
    for method, path in paths.items():
        target = out / f"fields_{method}_raw.gpkg"
        config = inference_config(manifest, method, path, target, out, checkpoint, project)
        pin = {
            "config": config.to_dict(),
            "input": sha256(path),
            "bundle": sha256(bundle / "manifest.json"),
            "checkpoint": sha256(checkpoint),
            "revision": spec.revision,
            "tutorial_code": sha256(Path(__file__)),
            "scientific_code_sha256": scientific_code_hashes(),
        }
        record = target.with_suffix(".resume.json")
        if target.exists():
            old = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}
            if old.get("pin") != pin or old.get("output_sha256") != sha256(target):
                raise ValueError("Live cache mismatch; choose a separate output directory")
        else:
            recorder = RunRecorder(config, run_id=new_run_id())
            with recorder:
                seed_everything(config.seed)
                recorder.set("workflow", "tutorial matched prepared-input comparison")
                recorder.set("input_pin", pin)
                with recorder.step("delineate"):
                    start = time.perf_counter()
                    frame = get_engine(config.engine).delineate(str(path), config)
                    inference_s = time.perf_counter() - start
                recorder.record_engine_meta(frame.attrs.get("engine_meta", {}))
                with recorder.step("postprocess"):
                    start = time.perf_counter()
                    outline = (
                        gpd.GeoSeries([box(*site["bbox"])], crs=4326).to_crs(frame.crs).iloc[0]
                    )
                    frame, selection = select_in_study_area(frame, outline, "representative_point")
                    frame = filter_polygons(frame, min_area_m2=1000)
                    frame = simplify_polygons(frame, tolerance=2)
                    frame = filter_polygons(frame, min_area_m2=1000)
                    postprocess_s = time.perf_counter() - start
                    recorder.set("aoi_selection", selection)
                    recorder.set(
                        "postprocess",
                        {
                            "min_area_m2": 1000,
                            "simplify_m": 2,
                            "smooth_iterations": 0,
                            "hole_removal": False,
                        },
                    )
                write_vector(frame, target)
            write_provenance(target, recorder.to_dict())
            save_json(
                record,
                {
                    "pin": pin,
                    "output_sha256": sha256(target),
                    "inference_s": inference_s,
                    "postprocess_s": postprocess_s,
                },
            )
        result[method] = target
    return result


def inference_config(manifest, method, path, target, out, checkpoint, project):
    """Keep inference and cached replay tied to one explicit scientific recipe."""
    from agribound.config import AgriboundConfig

    if method not in METHODS:
        raise ValueError(f"Unknown Landsat tutorial method: {method}")
    bands = (
        {"R": 1, "G": 1, "B": 1}
        if method == "pan"
        else ({"R": 3, "G": 2, "B": 1} if method == "sr" else {"R": 1, "G": 2, "B": 3})
    )
    site = manifest["site"]
    return AgriboundConfig(
        source="landsat-pan" if method == "pan" else "landsat" if method == "sr" else "local",
        gee_project=project,
        local_tif_path=str(Path(path).resolve()),
        study_area=box(*site["bbox"]).wkt,
        bands=bands,
        year=site["ftw_year"],
        engine="delineate-anything",
        device="cpu",
        seed=42,
        min_field_area_m2=1000,
        simplify_tolerance=2,
        lulc_filter=False,
        output_path=str(Path(target).resolve()),
        cache_dir=str((out / "cache").resolve()),
        engine_params={
            "backend": "native",
            "da_model": "large_v2",
            "checkpoint_path": str(Path(checkpoint).resolve()),
            "super_resolution": 1,
            "half": False,
            "batch_size": 1,
            "conf_threshold": 0.15,
            "tile_step": 0.5,
        },
    )


def cached_landsat_paths(bundle, out, *, live=False):
    """Load all eight products; validate live replay without weights or downloads.

    Recorded checkpoint/project paths are provenance, not requests to access
    those resources. Input/output bytes and the frozen recipe/code must still
    match, including when only evaluation or figures are requested.
    """
    manifest = load_bundle(bundle)
    if not live:
        return {method: bundle / f"predictions/fields_{method}.gpkg" for method in METHODS}
    from agribound.engines.delineate_anything import DA_MODELS

    baselines = {"pan": "landsat-pan.tif", "sr": "landsat.tif", "combined": "pan_sr_rgb.tif"}
    paths = {}
    for method in METHODS:
        target = out / f"fields_{method}_raw.gpkg"
        record = target.with_suffix(".resume.json")
        raster = (
            out / "live_inputs" / baselines[method]
            if method in baselines
            else out / f"input_{method}.tif"
        )
        try:
            old = json.loads(record.read_text(encoding="utf-8"))
            pin = old["pin"]
            config = pin["config"]
            # Rebuild settings from the frozen recipe; retain recorded resource
            # locations only after checking their expected local destinations.
            if (
                Path(config["local_tif_path"]).resolve() != raster.resolve()
                or Path(config["output_path"]).resolve() != target.resolve()
                or Path(config["cache_dir"]).resolve() != (out / "cache").resolve()
            ):
                raise ValueError("Recorded artifact paths changed")
            expected_config = inference_config(
                manifest,
                method,
                config["local_tif_path"],
                config["output_path"],
                Path(config["cache_dir"]).parent,
                config["engine_params"]["checkpoint_path"],
                config["gee_project"],
            ).to_dict()
            expected = {
                "config": expected_config,
                "input": sha256(raster),
                "bundle": sha256(bundle / "manifest.json"),
                "checkpoint": manifest["comparison"]["checkpoint_sha256"],
                "revision": DA_MODELS["large_v2"].revision,
                "tutorial_code": sha256(Path(__file__)),
                "scientific_code_sha256": scientific_code_hashes(),
            }
            if pin != expected or old["output_sha256"] != sha256(target):
                raise ValueError("Recorded recipe or artifact hash changed")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Live cache mismatch for {method}; verify the recorded run or use a "
                "separate output directory with --live --stage all"
            ) from exc
        paths[method] = target
    return paths


def load_products(bundle, paths):
    """Apply the frozen final AOI rule after postprocessing; preserve raw geometry."""
    manifest = load_bundle(bundle)
    return {
        name: inside_aoi(gpd.read_file(path), manifest["site"]["bbox"])
        for name, path in paths.items()
    }


def evaluate_products(bundle, out, products):
    """Evaluate original boundary lines within the same known reference coverage."""
    from landsat_reference_adapters import select_coverage

    reference = gpd.read_file(bundle / "reference.gpkg")
    coverage = gpd.read_file(bundle / "evaluation_coverage.gpkg")
    rows = []
    for name, frame in products.items():
        selected, selection = select_product_coverage(frame, coverage, select_coverage)
        write_vector(selected, out / f"fields_{name}_evaluated.gpkg")
        save_json(out / f"selection_{name}.json", selection)
        for tolerance in (10, 15, 30):
            score = evaluate(
                selected,
                reference,
                size_bins=SIZE_BINS,
                boundary_tolerance_m=tolerance,
                boundary_mask=coverage,
                bootstrap=0,
            )
            for size, values in [("all", score), *score["per_size_class"].items()]:
                rows.append(
                    {
                        "track": "reference_evaluation",
                        "product": name,
                        "boundary_tolerance_m": tolerance,
                        "size_class_ha": size,
                        "n_reference": values.get("n", score["count_reference"]),
                        "n_predicted": values["count_predicted"],
                        "boundary_f1": values["boundary_f1"],
                        "detection_f1": values["f1"],
                        "matched_iou": values["iou_mean"],
                        "boundary_precision": values["boundary_precision"],
                        "boundary_recall": values["boundary_recall"],
                        "oversegmentation": values["oversegmentation_mean"],
                        "undersegmentation": values["undersegmentation_mean"],
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(out / "reference_accuracy.csv", index=False)
    return table


def query_product(bundle, out, *, live=False, query_if_missing=True):
    """Read a frozen buffered snapshot locally, or explicitly query supported 2024 FTW."""
    manifest = load_bundle(bundle)
    site = manifest["site"]
    snapshot = out / "ftw_live.parquet" if live else bundle / "ftw_raw.parquet"
    if live and not snapshot.exists() and not query_if_missing:
        raise ValueError("FTW snapshot missing; explicitly run --live --stage query first")
    if live and snapshot.exists():
        record = json.loads((out / "ftw_live_pin.json").read_text(encoding="utf-8"))
        if record["sha256"] != sha256(snapshot) or record["bundle"] != sha256(
            bundle / "manifest.json"
        ):
            raise ValueError("FTW snapshot pin changed; use a separate output directory")
        raw = gpd.read_parquet(snapshot)
        info = record["query"]
    else:
        area = gpd.GeoSeries([box(*site["bbox"])], crs=4326).to_crs(32631).buffer(600).to_crs(4326)
        if not live:
            # Add only a covering-bbox index to a working copy for Arrow pruning.
            # Preserve the raw provider snapshot and full polygon geometry.
            indexed = out / "ftw_indexed.parquet"
            gpd.read_parquet(snapshot).to_parquet(indexed, write_covering_bbox=True)
            snapshot = indexed
        kwargs = {} if live else {"source_url": str(snapshot), "source_backend": "pyarrow"}
        raw = query_ftw(
            area,
            year=site["ftw_year"],
            clip=False,
            deduplicate=True,
            max_features=None,
            cache_dir=out / "ftw_cache",
            **kwargs,
        )
        info = raw.attrs.get("ftw_query", {})
        if live:
            raw.to_parquet(snapshot)
            save_json(
                out / "ftw_live_pin.json",
                {
                    "sha256": sha256(snapshot),
                    "query": info,
                    "bundle": sha256(bundle / "manifest.json"),
                },
            )
    validate_ftw_snapshot(raw, site["ftw_year"], info)
    products, diagnostics = {}, []
    for name in ("ftw", "ftw_conf69", "ftw_conf69_known", "ftw_area1000"):
        variant, diagnostic = ftw_variant(raw, name)
        products[name] = inside_aoi(variant, site["bbox"])
        diagnostics.append(diagnostic)
    pd.DataFrame(diagnostics).to_csv(out / "confidence_sensitivity.csv", index=False)
    save_json(out / "ftw_query.json", info)
    return products


def agreement_products(out, landsat, ftw):
    """Write peer-product agreement separately from reference accuracy."""
    rows = []
    for method, frame in landsat.items():
        for tolerance in (10, 15, 30):
            rows.append(
                {
                    "method": method,
                    **product_agreement(frame, ftw, tolerance_m=tolerance, size_bins=SIZE_BINS),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "prediction_agreement.csv", index=False)
    return table


def maps(bundle, out, products, *, stem):
    """Use one unsharpened RGB background and frozen reference-selected zooms."""
    import matplotlib.pyplot as plt
    from landsat_multispectral_figures import background, draw_map, legend, save

    imagery = out / "live_inputs/landsat.tif"
    image, extent, crs = background(imagery if imagery.exists() else bundle / "inputs/landsat.tif")
    reference = gpd.read_file(bundle / "reference.gpkg").to_crs(crs)
    coverage = gpd.read_file(bundle / "evaluation_coverage.gpkg").to_crs(crs)
    windows = json.loads((bundle / "figure_windows.json").read_text(encoding="utf-8"))
    for view in ("overview", "small_fields", "shared_edges"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 9))
        panels = [("Reference only", None), *products.items()]
        for ax, (name, prediction) in zip(axes.flat, panels, strict=True):
            draw_map(
                ax,
                image,
                extent,
                reference,
                prediction.to_crs(crs) if prediction is not None else None,
                windows[view],
                coverage,
            )
            grid = 30 if name in ("sr", "false_color") else 15
            title = (
                "FTW published predictions; nominal 2024; 10 m sensor"
                if name == "ftw"
                else (f"{name}: {grid} m grid\n" + " / ".join(METHODS.get(name, [])))
            )
            ax.set_title(title if prediction is not None else name, fontsize=9)
        fig.suptitle(
            f"Camargue 2024 — {view.replace('_', ' ')}\n"
            "RPG declarations; training independence unknown"
        )
        legend(fig)
        fig.subplots_adjust(top=0.89, bottom=0.08, hspace=0.24)
        save(fig, out / f"{stem}_{view}")


def summary_figure(bundle, out):
    """Show every preselected site, including the old-reference Vietnamese case."""
    import matplotlib.pyplot as plt
    from landsat_multispectral_figures import save

    table = pd.read_csv(bundle / "reference_headline_15m.csv")
    table = table[(table.primary_scope) & (table["product"].isin(["ftw", *METHODS]))]
    sites = load_bundle(bundle)["selected_summary_sites"]
    labels = [
        "FR Camargue · Mediterranean\nRice/annual declarations",
        "FR Bordeaux · oceanic context\nVine declarations",
        "NL Flevoland · maritime\nArable/grass declarations",
        "NL Bollenstreek · maritime\nFlower/grass declarations",
        "CA Yamaska · continental\nMaize/soy crop units",
        "VN Mekong · tropical monsoon\nRice region; species unknown; 2021 labels",
    ]
    counts = table.groupby("site").n_reference.first().reindex(sites)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric, title in zip(
        axes,
        ["boundary_f1", "f1"],
        ["Boundary F1 at 15 m", "One-to-one detection F1 at IoU >= 0.5"],
        strict=True,
    ):
        grid = table.pivot(index="site", columns="product", values=metric).reindex(
            index=sites, columns=["ftw", *METHODS]
        )
        image = ax.imshow(grid, vmin=0, vmax=1, cmap="cividis", aspect="auto")
        ax.set_xticks(range(len(grid.columns)), grid.columns, rotation=55, ha="right")
        ax.set_yticks(
            range(len(sites)),
            [f"{label} · n={n} labels" for label, n in zip(labels, counts, strict=True)],
            fontsize=8,
        )
        ax.set_title(title)
        for row in range(len(sites)):
            for col in range(len(grid.columns)):
                value = grid.iloc[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.2f}" if np.isfinite(value) else "NM",
                    ha="center",
                    va="center",
                    color="white" if value < 0.55 else "black",
                    fontsize=7,
                )
        fig.colorbar(image, ax=ax, fraction=0.03)
    fig.suptitle(
        "Measured products, shared references; not verified independent validation\n"
        "France / Netherlands / Canada / Vietnam (2021 labels vs 2024 imagery)",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, out / "six_public_locations")
