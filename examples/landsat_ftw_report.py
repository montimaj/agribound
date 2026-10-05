"""Measured summaries, training/temporal audits and checksummed artifact inventory."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ftw_training_audit_reused", ROOT / "examples/landsat_ftw_training_audit.py"
)
TRAINING_AUDIT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TRAINING_AUDIT)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(frame):
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        rows.append(
            "| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in row) + " |"
        )
    return "\n".join(rows)


def aoi_edge_counts(frame, bbox):
    """Describe whole-polygon edge handling without changing the source geometry."""
    if frame.crs is None:
        raise ValueError("AOI audit requires a documented CRS")
    if frame.empty:
        return {
            "n_source_polygons": 0,
            "n_selected_by_representative_point": 0,
            "n_selected_crossing_aoi_edge": 0,
            "n_intersecting_excluded_by_policy": 0,
            "n_outside_aoi": 0,
        }
    geographic = frame.to_crs(4326)
    aoi = box(*bbox)
    keep = geographic.geometry.representative_point().within(aoi)
    intersects = geographic.geometry.intersects(aoi)
    contained = geographic.geometry.map(aoi.covers)
    return {
        "n_source_polygons": len(frame),
        "n_selected_by_representative_point": int(keep.sum()),
        "n_selected_crossing_aoi_edge": int((keep & ~contained).sum()),
        "n_intersecting_excluded_by_policy": int((intersects & ~keep).sum()),
        "n_outside_aoi": int((~intersects).sum()),
    }


def inventories(config, out):
    timing, confidence, audit, snapshots, support_rows = [], [], [], [], []
    view_crops, source_families, edge_rows = [], [], []
    benchmark_path = out / "benchmark_split_overlap.csv"
    benchmark = (
        pd.read_csv(benchmark_path).set_index("site").to_dict("index")
        if benchmark_path.exists()
        else {}
    )
    for site in config["sites"]:
        folder = out / site["id"]
        source = (
            ROOT / site["landsat_run_dir"]
            if site["historical"]
            else out / "landsat_runs" / site["id"]
        )
        statuses = (
            read(folder / "product_status.json")["products"]
            if (folder / "product_status.json").exists()
            else {}
        )
        refpath = folder / "reference_support.json"
        meta = read(refpath)["metadata"] if refpath.exists() else {}
        family = (
            "RPG"
            if site["country"] == "France"
            else "BRP"
            if site["country"] == "Netherlands"
            else "AI4SmallFarms Vietnam"
            if site["country"] == "Vietnam"
            else site["reference_mode"]
            if not site["historical"]
            else site["reference_kind"]
        )
        source_families.append(
            {
                "site": site["id"],
                "source_family": family,
                "reference_year": site["reference_year"],
                "interpretation": "Shared sources are not new independent datasets",
            }
        )
        if (folder / "reference.gpkg").exists() and (folder / "figure_windows.json").exists():
            reference = gpd.read_file(folder / "reference.gpkg")
            windows = read(folder / "figure_windows.json")
            crop_meta = (
                read(folder / "crop_strata.json") if (folder / "crop_strata.json").exists() else {}
            )
            column = crop_meta.get("source_column")
            projected = reference.to_crs(windows["crs"])
            for name in ("overview", "small_fields", "shared_edges"):
                selected = reference.loc[projected.intersects(box(*windows[name])).to_numpy()]
                counts = (
                    selected[column].fillna("unknown").astype(str).value_counts().to_dict()
                    if column in selected
                    else {"unknown": len(selected)}
                )
                view_crops.append(
                    {
                        "site": site["id"],
                        "view": name,
                        "n_reference_intersecting": len(selected),
                        "source_crop_column": column,
                        "provider_labels_inside_view": json.dumps(counts),
                        "regional_context": site["crop_context"],
                        "crop_label_year": site["reference_year"],
                        "unknown_species_not_inferred": column is None,
                    }
                )
        audit.append(
            {
                "site": site["id"],
                "reference_kind": site["reference_kind"],
                "ftw_benchmark_source_reuse": site["cohort"] == "international"
                or site["reference_mode"] == "vn_cached",
                "benchmark_patch_split": site["benchmark_split"],
                "benchmark_spatial_category": TRAINING_AUDIT.spatial_category(
                    benchmark.get(site["id"], {})
                ),
                **{
                    key: value
                    for key, value in benchmark.get(site["id"], {}).items()
                    if key.startswith("n_")
                },
                "ftw_training_overlap": "Unknown parcel membership; benchmark-source reuse flagged",
                "landsat_training_overlap": "Unknown parcel membership in FBIS-73M",
                "verified_independent_holdout": False,
                "reference_year": site["reference_year"],
                "ftw_year": site["ftw_year"],
                "temporal_category": site["temporal_category"],
                "geometry_vintage": meta.get("geometry_vintage", "Physical edge date uncertain"),
                "evidence": json.dumps(config["evidence_links"]),
            }
        )
        for product, path, preselection in [
            ("reference", folder / "reference.gpkg", False),
            ("ftw_raw", folder / "ftw_raw.parquet", True),
            *[
                (method["id"], folder / f"fields_{method['id']}.gpkg", False)
                for method in config["methods"]
            ],
        ]:
            if not path.exists():
                continue
            frame = gpd.read_parquet(path) if path.suffix == ".parquet" else gpd.read_file(path)
            counts = aoi_edge_counts(frame, site["bbox"])
            edge_rows.append(
                {
                    "site": site["id"],
                    "product": product,
                    **counts,
                    "pre_aoi_selection_source_available": preselection,
                    "n_preselection_edge_fields_excluded": (
                        counts["n_intersecting_excluded_by_policy"] if preselection else np.nan
                    ),
                    "limitation": (
                        "Buffered raw FTW snapshot; complete geometries preserved"
                        if preselection
                        else "Delivered selected polygons; original excluded-field count unknown"
                    ),
                }
            )
        if (folder / "confidence_diagnostics.json").exists():
            for _variant, d in read(folder / "confidence_diagnostics.json").items():
                confidence.append(
                    {
                        "site": site["id"],
                        **d,
                        "null_rate_input": d["n_null_confidence_input"] / d["n_input"]
                        if d["n_input"]
                        else np.nan,
                    }
                )
        if (folder / "ftw_snapshot.json").exists():
            snap = read(folder / "ftw_snapshot.json")
            indexes = snap.get(
                "remote_file_metadata",
                {p.name: read(p) for p in (folder / "ftw_index").glob("partition_index*.json")},
            )
            snapshots.append(
                {
                    "site": site["id"],
                    "snapshot_sha256": snap["sha256"],
                    "nominal_year": snap["nominal_prediction_year"],
                    "captured_partition_size_mtime_bbox": json.dumps(indexes),
                    "etag": "Not supplied by PyArrow; captured size/mtime and local SHA pin",
                    "actual_feature_dates": "Planting/harvest calendar; polygon-level dates absent",
                }
            )
            timing.append(
                {
                    "site": site["id"],
                    "product": "ftw",
                    "stage": "query_download",
                    "seconds": snap["query_s"],
                    "inference_seconds": np.nan,
                    "definition": "Retrieval, not upstream model inference",
                }
            )
        for method in config["methods"]:
            name = method["id"]
            path = folder / f"fields_{name}.gpkg.provenance.json"
            if statuses.get(name, {}).get("status") != "complete" or not path.exists():
                continue
            p = read(path)
            engine = p["engine_meta"]
            common = {
                "site": site["id"],
                "product": name,
                "reused": site["historical"],
                "hostname": p["hostname"],
                "device": engine["device"],
                "precision": engine["precision"],
                "python": p["python"],
                "torch": p["versions"].get("torch"),
                "output_grid_m": engine["gsd_m"],
                "tile_footprint_m": engine["tile_size_native_px"] * engine["gsd_m"],
            }
            timing.append({**common, "stage": "model_inference", "seconds": engine["inference_s"]})
            for step in p["steps"]:
                if step["name"] in ("delineate", "postprocess"):
                    timing.append({**common, "stage": step["name"], "seconds": step["wall_s"]})
        for file, key, stage in [
            ("evaluation_signature.json", "evaluation_s", "evaluation"),
            ("rendering_timing.json", "site_maps_render_s", "rendering"),
        ]:
            if (folder / file).exists():
                timing.append(
                    {
                        "site": site["id"],
                        "product": "all",
                        "stage": stage,
                        "seconds": read(folder / file)[key],
                    }
                )
        old = (
            ROOT / site["existing_dir"] if site["historical"] else out / "preparation" / site["id"]
        )
        for directory in (old, source):
            path = directory / "preparation_timing.json"
            if path.exists():
                for key, value in read(path).items():
                    if key.endswith("_s"):
                        timing.append(
                            {
                                "site": site["id"],
                                "product": "shared_inputs",
                                "stage": key,
                                "seconds": value,
                                "reused": site["historical"],
                            }
                        )
        manifest = folder / "scene_manifest.json"
        imagery = folder / "imagery_coverage.json"
        if manifest.exists():
            scenes = read(manifest)
            support_rows.append(
                {
                    "site": site["id"],
                    "matched_scenes": len(scenes["pairs"]),
                    "unmatched_pan": len(scenes["unmatched_pan"]),
                    "unmatched_sr": len(scenes["unmatched_sr"]),
                    "window": json.dumps(scenes["date_window_end_exclusive"]),
                    "coverage": json.dumps(read(imagery)) if imagery.exists() else "unknown",
                    "crop_strata": json.dumps(read(folder / "crop_strata.json"))
                    if (folder / "crop_strata.json").exists()
                    else "unknown",
                }
            )
        input_pin = folder / "input_provenance.json"
        if input_pin.exists():
            prepared = read(input_pin)
            timing.append(
                {
                    "site": site["id"],
                    "product": "shared_inputs",
                    "stage": "multispectral_input_preparation",
                    "seconds": prepared["preparation_s"],
                    "cached_preparation": prepared["cached_on_this_call"],
                    "definition": "Recorded preparation call; cached calls are not cold downloads",
                }
            )
    for name, rows in [
        ("runtime_stages", timing),
        ("confidence_sensitivity", confidence),
        ("training_temporal_audit", audit),
        ("ftw_snapshot_inventory", snapshots),
        ("imagery_reference_support", support_rows),
        ("view_crop_inventory", view_crops),
        ("reference_source_reuse", source_families),
        ("aoi_edge_audit", edge_rows),
    ]:
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)


def paired_summary(out):
    path = out / "reference_paired_differences.csv"
    if not path.exists() or path.stat().st_size < 3:
        return
    pairs = pd.read_csv(path)
    rows = []
    rng = np.random.default_rng(42)
    for keys, group in pairs.groupby(
        ["temporal_category", "reference_kind", "benchmark_spatial_category", "contrast"]
    ):
        for metric in ("boundary_f1", "f1"):
            data = group[np.isfinite(group[metric])]
            n = len(data)
            if not n:
                continue
            for weighting in ("equal_site", "reference_count"):
                w = np.ones(n) if weighting == "equal_site" else data.n_reference.to_numpy()
                values = data[metric].to_numpy()
                indexes = rng.integers(0, n, size=(2000, n))
                means = (values[indexes] * w[indexes]).sum(1) / w[indexes].sum(1)
                rows.append(
                    {
                        "temporal_category": keys[0],
                        "reference_kind": keys[1],
                        "benchmark_spatial_category": keys[2],
                        "contrast": keys[3],
                        "metric": metric,
                        "weighting": weighting,
                        "n_sites": n,
                        "mean_difference": np.average(values, weights=w),
                        "site_bootstrap_95_low": np.quantile(means, 0.025),
                        "site_bootstrap_95_high": np.quantile(means, 0.975),
                        "improved_sites": int((values > 1e-9).sum()),
                        "worsened_sites": int((values < -1e-9).sum()),
                    }
                )
    pd.DataFrame(rows).to_csv(out / "reference_paired_summary.csv", index=False)


def build(config, out):
    out = Path(out)
    inventories(config, out)
    paired_summary(out)
    status = pd.read_csv(out / "product_status.csv")
    lines = [
        "# Landsat and published FTW comparison",
        "",
        "Published FTW polygons are predictions. Reference evaluation and product agreement "
        "are separate tracks.",
        "",
        f"Frozen inventory: {len(config['sites'])} sites. "
        "Six additional AOIs were selected before scores. Completed products:",
        "",
    ]
    for name, group in status.groupby("product"):
        lines.append(f"- {name}: {group.status.eq('complete').sum()} sites")
    path = out / "reference_headline_15m.csv"
    if path.exists():
        headline = pd.read_csv(path)
        lines += [
            "",
            "## Per-site reference evaluation",
            "",
            "Known-coverage conditional scores. Declaration/mapping-unit splits are "
            "not necessarily physical boundaries. No training-independent holdouts "
            "are verified. Contemporary declarations have uncertain physical-edge vintage.",
            "",
        ]
        for metric in ("boundary_f1", "f1"):
            values = headline[
                headline["product"].isin(["ftw", *[m["id"] for m in config["methods"]]])
            ]
            lines += [
                f"### {metric} at headline thresholds",
                "",
                table(values.pivot(index="site", columns="product", values=metric).reset_index()),
                "",
            ]
        paired = pd.read_csv(out / "reference_paired_differences.csv")
        new = {
            s["id"]
            for s in config["sites"]
            if not s["historical"]
            and s["temporal_category"] == "same_nominal_year_geometry_uncertain"
        }
        selected = paired[paired.site.isin(new)]
        lines += [
            "## Input tradeoffs",
            "",
            "These descriptive paired differences cover the "
            "five new sites with matching nominal declaration/product years, separately "
            "from historical sites and the 2021 Vietnam survey.",
            "",
        ]
        for contrast in (
            "false_color minus sr",
            "false_color_15m minus false_color",
            "pan_nir_red minus coarse_pan_nir_red",
            "hybrid minus false_color_15m",
        ):
            data = selected[selected.contrast == contrast]
            if len(data):
                lines.append(
                    f"- {contrast}: mean boundary F1 difference {data.boundary_f1.mean():+.3f}; "
                    f"mean detection F1 difference {data.f1.mean():+.3f}, {len(data)} sites. "
                    f"Boundary F1 improves at {(data.boundary_f1 > 0).sum()} sites and "
                    f"declines at {(data.boundary_f1 < 0).sum()}."
                )
        lines += [
            "",
            "False color and stacking are experiments with RGB-trained weights, "
            "not optimized multispectral models. F–G measures native PAN detail under "
            "the fixed stack. E–D changes both grid and physical tile context. H–E "
            "adds visible detail with unchanged NIR. Interpret improvements per site; "
            "neither a universal PAN advantage nor climate/crop causation follows.",
            "",
        ]
        for contrast in ("pan_nir_red minus pan", "hybrid minus pan", "hybrid minus combined"):
            data = selected[selected.contrast == contrast]
            if len(data):
                lines.append(
                    f"- {contrast}, matching nominal-year cohort: "
                    f"boundary F1 {data.boundary_f1.mean():+.3f}, "
                    f"detection F1 {data.f1.mean():+.3f} (equal-site mean)."
                )
        extra = selected[selected.contrast.isin(["pan_nir_red minus pan", "hybrid minus pan"])]
        means = extra.groupby("contrast")[["boundary_f1", "f1"]].mean()
        if len(means) == 2 and (means <= 0).all().all():
            lines += [
                "",
                "Stacking and the hybrid do not improve average headline "
                "scores over PAN alone in the measured nominal-year cohort. "
                "Their complexity does not justify a general replacement here. "
                "Individual detection objectives can differ; inspect both measures "
                "and reference suitability.",
                "",
            ]
        mekong = headline[headline.site == "vn_mekong"].set_index("product")
        if {"sr", "pan", "hybrid"}.issubset(mekong.index) and (
            mekong.loc["sr", "boundary_f1"] > mekong.loc["pan", "boundary_f1"]
            and mekong.loc["hybrid", "boundary_f1"] > mekong.loc["sr", "boundary_f1"]
        ):
            lines += [
                "SR RGB exceeds PAN boundary F1 at the historical Mekong site, "
                "and the hybrid performs better on its 2021 reference labels. "
                "This is a site-specific descriptive result; comparisons with "
                "2024 FTW remain cross-year and do not validate 2024 accuracy.",
                "",
            ]
        if len(selected):
            measured = headline[
                headline.site.isin(new)
                & headline["product"].isin(["ftw", *[m["id"] for m in config["methods"]]])
            ]
            wide = measured.pivot(index="site", columns="product", values="boundary_f1")
            complete = wide.dropna()
            leading = (complete["ftw"] > complete.drop(columns="ftw").max(axis=1)).sum()
            lines += [
                f"FTW has higher boundary F1 than every measured Landsat method at "
                f"{leading}/{len(complete)} complete new nominal-year sites. "
                "Detection F1 can rank products differently. Declaration splits and "
                "different delivered-system definitions can affect these measures; "
                "the comparison does not isolate sensor resolution or training.",
                "",
            ]
    lines += [
        "## Coverage, independence and time",
        "",
        "Raw geometries are retained. Invalid working geometries are repaired in "
        "memory for geometric operations with repair counts recorded. The "
        "reference-footprint mask is independent of predictions; unmapped land is "
        "excluded from primary precision. Whole-AOI scores are descriptive. "
        "Representative-point inclusion preserves whole polygons. Boundary metrics "
        "use original lines inside the mask, not clipping-created edges.",
        "The common final AOI rule is applied after delivered postprocessing in "
        "EPSG:4326. Separate fields_*_aoi.gpkg artifacts preserve raw inference "
        "vectors and record exclusions; the edge audit documents rescoring "
        "without repeating inference.",
        "",
        "International FTW-converted references and the adjacent Vietnam patch "
        "reuse source datasets. Extracted benchmark splits and parcel membership "
        "in both training corpora are distinct questions. The cached v1 chip-split "
        "audit records spatial overlaps; exact model parcel membership remains unknown. "
        "No independent holdout is claimed.",
        "",
        "Historical Landsat dates are frozen; 2024 FTW disagreement may reflect "
        "land changes. Weighted means average site scores by reference count; "
        "they are not pooled precision/recall. Equal-site means and descriptive "
        "site-bootstrap intervals retain temporal/reference categories. Spatial "
        "dependence, small samples and uncertain boundary vintages limit inference.",
        "",
        "FTW download time is separate from Landsat inference. Native 10/15/30 m "
        "information remains distinct. Resampling and vector coordinates do not "
        "establish finer spatial accuracy.",
        "",
        "## Unmeasured products",
        "",
    ]
    for row in status[status.status != "complete"].itertuples():
        stage = "query-ftw" if row.product.startswith("ftw") else "run-landsat"
        lines.append(
            f"- {row.site} / {row.product}: {getattr(row, 'error', 'not executed')}. "
            f"Resume: `python examples/28_landsat_ftw_comparison.py --{stage} "
            f"--sites {row.site}`."
        )
    lines += [
        "",
        "The frozen historical Red River window has no matched Landsat scenes "
        "under its cloud threshold. FTW can complete independently. A changed "
        "window requires a separate site-year. Restricted vectors/overlay maps "
        "stay local. Regenerate maps with `--figures-only --offline`.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = []
    for path in sorted(out.rglob("*")):
        if (
            not path.is_file()
            or path.name == "artifact_manifest.json"
            or any(
                p in ("research", "ftw_index", ".agribound_cache", "ultralytics")
                for p in path.parts
            )
            or path.suffix in (".log", ".part")
        ):
            continue
        relative = path.relative_to(out)
        site = next((s for s in config["sites"] if s["id"] in relative.parts), None)
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "redistribution": site["redistribution"]
                if site
                else "Local; consult component licenses",
            }
        )
    paths = [
        ROOT / "examples/28_landsat_ftw_comparison.py",
        ROOT / "agribound/comparison_ftw.py",
        ROOT / "examples/landsat_ftw_figures.py",
        Path(__file__),
        ROOT / config["inherited_config"]["path"],
        ROOT / "examples/landsat_ftw_training_audit.py",
        ROOT / "examples/landsat_ftw_comparison_sites.json",
        ROOT / "docs/user-guide/landsat-ftw-comparison.md",
        ROOT / "tests/unit/test_landsat_ftw_comparison.py",
    ]
    paths += [
        ROOT / p
        for p in (
            "examples/23_landsat_pan_sr_comparison.py",
            "examples/25_landsat_international_comparison.py",
            "examples/26_landsat_north_america_comparison.py",
            "examples/27_landsat_multispectral_comparison.py",
            "examples/landsat_reference_adapters.py",
            "examples/landsat_multispectral_figures.py",
            "agribound/ftw_query.py",
            "agribound/ftw_arrow.py",
            "agribound/evaluate.py",
            "agribound/composites/landsat_matched.py",
            "agribound/composites/landsat_multispectral.py",
            "agribound/composites/pan_fusion.py",
            "agribound/engines/delineate_anything.py",
            "examples/landsat_comparison_constraints.txt",
        )
    ]
    manifest = {
        "files": files,
        "code_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "checkpoint_revision": "369d0b4c44cf9bec2bd3a27bc81810cadd2c963e",
        "checkpoint_sha256": config["comparison"]["checkpoint_sha256"],
        "restricted_reference_policy": "Keep restricted vectors and overlay maps local",
    }
    (out / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
