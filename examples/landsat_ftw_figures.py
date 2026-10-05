"""Common-background FTW/Landsat maps and separately labelled quantitative plots."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import textwrap
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

from agribound.comparison_ftw import inside_aoi

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "multispectral_maps_reused", ROOT / "examples/landsat_multispectral_figures.py"
)
MAPS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MAPS)
SHORT = {"ftw": "Published FTW · native 10 m\nSentinel-2 / PRUE", **MAPS.SHORT}
MAIN = ["ftw", "pan", "sr", "false_color", "pan_nir_red", "hybrid"]
CONTROLS = ["ftw", "combined", "false_color_15m", "coarse_pan_nir_red"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_site(site, out):
    folder = out / site["id"]
    if not (folder / "figure_windows.json").exists():
        return None
    windows = read(folder / "figure_windows.json")
    old = ROOT / site["existing_dir"] if site["historical"] else out / "preparation" / site["id"]
    path = old / "inputs/landsat.tif"
    if path.exists():
        image, extent, crs = MAPS.background(path)
        # Frozen map CRS must equal the imagery CRS; never distort an overlay.
        from pyproj import CRS

        if CRS(crs) != CRS(windows["crs"]):
            raise ValueError("Frozen views and background CRS differ")
        background_note = "Common unsharpened SR RGB"
    else:
        crs = windows["crs"]
        image = np.full((2, 2, 3), 0.65)
        b = windows["overview"]
        extent = (b[0], b[2], b[1], b[3])
        background_note = "No Landsat composite; common neutral background"
    reference_path = folder / "reference.gpkg"
    reference = (
        gpd.read_file(reference_path).to_crs(crs)
        if reference_path.exists()
        else gpd.GeoDataFrame(geometry=[], crs=crs)
    )
    reference.geometry = reference.geometry.make_valid(method="structure")
    coverage_path = folder / "evaluation_coverage.gpkg"
    coverage = gpd.read_file(coverage_path).to_crs(crs) if coverage_path.exists() else None
    if coverage is not None:
        coverage.geometry = coverage.geometry.make_valid(method="structure")
    statuses = (
        read(folder / "product_status.json")["products"]
        if (folder / "product_status.json").exists()
        else {}
    )
    predictions = {
        name: inside_aoi(gpd.read_file(folder / f"fields_{name}.gpkg"), site["bbox"]).to_crs(crs)
        for name in SHORT
        if statuses.get(name, {}).get("status") == "complete"
        and (folder / f"fields_{name}.gpkg").exists()
    }
    return image, extent, reference, predictions, windows, coverage, background_note


def panel(ax, data, method, view):
    image, extent, reference, predictions, windows, coverage, _ = data
    MAPS.draw_map(ax, image, extent, reference, predictions.get(method), windows[view], coverage)
    x0, y0, x1, y1 = windows["overview"]
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="white", linewidth=0.6, linestyle=":")
    if method != "reference" and method not in predictions:
        ax.text(
            0.5,
            0.5,
            "UNMEASURED",
            transform=ax.transAxes,
            ha="center",
            bbox={"facecolor": "white", "alpha": 0.9},
            fontsize=10,
        )


def legend(fig):
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=MAPS.REF_COLOR,
                linestyle="--",
                label="Source reference (quality varies)",
            ),
            Line2D([0], [0], color=MAPS.PRED_COLOR, label="Prediction product"),
            Line2D([0], [0], color="#666666", label="Same unsharpened background"),
        ],
        loc="lower center",
        ncol=3,
        fontsize=10,
    )


def site_maps(site, out, data):
    start = time.perf_counter()
    folder = out / site["id"]
    files = [
        folder / "reference.gpkg",
        folder / "evaluation_coverage.gpkg",
        folder / "figure_windows.json",
        folder / "product_status.json",
        Path(__file__),
        ROOT / "examples/landsat_multispectral_figures.py",
    ]
    old = ROOT / site["existing_dir"] if site["historical"] else out / "preparation" / site["id"]
    files += [old / "inputs/landsat.tif", *folder.glob("fields_*.gpkg")]
    signature = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
        if p.exists() and "_evaluated" not in p.name
    }
    cache = folder / "map_signature.json"
    if (
        cache.exists()
        and read(cache)["inputs"] == signature
        and all(
            hashlib.sha256((folder / name).read_bytes()).hexdigest() == sha
            for name, sha in read(cache)["figures"].items()
        )
    ):
        return
    groups = [
        ["ftw", "pan", "sr", "combined"],
        ["ftw", "false_color", "false_color_15m", "pan_nir_red"],
        ["ftw", "coarse_pan_nir_red", "hybrid"],
    ]
    for number, methods in enumerate(groups, 1):
        fig, axes = plt.subplots(
            3, len(methods) + 1, figsize=(4 * (len(methods) + 1), 12), squeeze=False
        )
        for row, view in enumerate(("overview", "small_fields", "shared_edges")):
            panel(axes[row, 0], data, "reference", view)
            axes[row, 0].set_ylabel(
                {
                    "overview": "Whole AOI",
                    "small_fields": "Small fields · 900 m",
                    "shared_edges": "Shared edges · 900 m",
                }[view],
                fontsize=11,
            )
            if row == 0:
                axes[row, 0].set_title("Reference only\nWhole source polygons", fontsize=10)
            for column, method in enumerate(methods, 1):
                panel(axes[row, column], data, method, view)
                if row == 0:
                    axes[row, column].set_title(SHORT[method], fontsize=10)
        fig.suptitle(
            f"{site['id']} · {site['country']} · FTW {site['ftw_year']}\n"
            f"Landsat {site['date_start']}–{site['date_end']} · "
            f"reference {site['reference_year']} · {site['reference_kind']}\n"
            f"{site['temporal_category']}; {site['coverage_policy']} conditional evaluation; "
            "training overlap unknown",
            fontsize=12,
        )
        ref = data[2]
        median = ref.to_crs(6933).area.median() / 10000 if len(ref) else np.nan
        note = data[-1] + "; whole polygons shown, unknown mapped coverage is not negative truth."
        note += f" {len(ref)} reference units; median {median:.2f} ha."
        if not data[4]["shared_edge_found"]:
            note += " Shared-edge view: median-reference fallback."
        fig.text(0.5, 0.044, note, ha="center", fontsize=9)
        legend(fig)
        fig.subplots_adjust(
            left=0.035, right=0.995, bottom=0.08, top=0.86, wspace=0.035, hspace=0.16
        )
        MAPS.save(fig, out / site["id"] / f"boundary_comparison_page_{number}")
    # Difference maps show geometric coverage disagreement; neither product is truth.
    for method in ("pan_nir_red", "hybrid"):
        products = data[3]
        if "ftw" not in products or method not in products:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        left, right = products[method], products["ftw"]
        left, right = left.copy(), right.copy()
        left.geometry = left.geometry.make_valid(method="structure")
        right.geometry = right.geometry.make_valid(method="structure")
        diffs = [
            (
                right.geometry.union_all().difference(left.geometry.union_all()),
                "FTW-only footprint",
            ),
            (
                left.geometry.union_all().difference(right.geometry.union_all()),
                "Landsat-only footprint",
            ),
            (
                left.geometry.union_all().symmetric_difference(right.geometry.union_all()),
                "Symmetric footprint difference",
            ),
        ]
        for ax, (geom, title) in zip(axes, diffs, strict=True):
            panel(ax, data, "reference", "overview")
            if not geom.is_empty:
                gpd.GeoSeries([geom], crs=left.crs).plot(ax=ax, color="#ffb000", alpha=0.55)
            ax.set_title(title)
            window = data[4]["overview"]
            ax.set_xlim(window[0], window[2])
            ax.set_ylim(window[1], window[3])
        fig.suptitle(f"{site['id']} · {method} versus FTW · disagreement, not error", fontsize=13)
        fig.subplots_adjust(bottom=0.10, top=0.86, wspace=0.05)
        MAPS.save(fig, out / site["id"] / f"disagreement_{method}")
    (out / site["id"] / "rendering_timing.json").write_text(
        json.dumps({"site_maps_render_s": time.perf_counter() - start}) + "\n"
    )
    rendered = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.iterdir()
        if p.suffix in (".png", ".pdf", ".svg")
    }
    cache.write_text(json.dumps({"inputs": signature, "figures": rendered}, indent=2) + "\n")


def representative(config, out, loaded):
    sites = [
        next(s for s in config["sites"] if s["id"] == name)
        for name in config["representative"]["sites"]
    ]
    for group, methods in (("main", MAIN), ("controls", CONTROLS)):
        for view in ("small_fields", "overview") if group == "main" else ("shared_edges",):
            fig, axes = plt.subplots(
                len(sites),
                len(methods),
                figsize=(3.2 * len(methods), 3.0 * len(sites)),
                squeeze=False,
            )
            for row, site in enumerate(sites):
                data = loaded.get(site["id"])
                for column, method in enumerate(methods):
                    ax = axes[row, column]
                    if data is None:
                        ax.text(0.5, 0.5, "UNMEASURED", ha="center", transform=ax.transAxes)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        panel(ax, data, method, view)
                    if row == 0:
                        ax.set_title(SHORT[method], fontsize=9)
                    if column == 0:
                        label = (
                            f"{site['id']} ({site['country']})\n"
                            + textwrap.fill(site["climate"][:65], width=40)
                            + "\n"
                            + textwrap.fill(
                                textwrap.shorten(site["crop_context"], width=80, placeholder="..."),
                                width=40,
                            )
                            + "\n"
                            + f"Landsat {site['date_start'][:4]} / FTW {site['ftw_year']}\n"
                            + textwrap.fill(
                                f"Ref {site['reference_year']}: "
                                + site["reference_kind"].replace("_", " "),
                                width=40,
                            )
                        )
                        if data is not None and len(data[2]):
                            ref = data[2]
                            median = ref.to_crs(6933).area.median() / 10000
                            label += f"\nn={len(ref)}, median {median:.2f} ha"
                        ax.set_ylabel(
                            label, fontsize=8, rotation=0, ha="right", va="center", labelpad=12
                        )
            fig.suptitle(
                "Frozen landscape selection · published FTW and Landsat systems\n"
                "Dashed references; solid predictions. Native 10/15/30 m; NIR remains 30 m.\n"
                "Zooms share a 900 m scale; overview scales differ. "
                "Regional crop context is not a verified crop inside each view.",
                fontsize=12,
            )
            legend(fig)
            fig.subplots_adjust(
                left=0.18, right=0.995, bottom=0.03, top=0.93, wspace=0.025, hspace=0.08
            )
            MAPS.save(fig, out / f"representative_{group}_{view}")


def heatmap(frame, sites, products, value, title, stem, column="product"):
    pivot = frame.pivot_table(index="site", columns=column, values=value, aggfunc="first").reindex(
        index=sites, columns=products
    )
    pivot.to_csv(stem.with_suffix(".csv"))
    fig, ax = plt.subplots(figsize=(1.25 * len(products) + 5, 0.43 * len(sites) + 2))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(products)), products, rotation=40, ha="right")
    ax.set_yticks(range(len(sites)), sites)
    for r in range(len(sites)):
        for c in range(len(products)):
            score = pivot.iloc[r, c]
            ax.text(
                c,
                r,
                f"{score:.2f}" if np.isfinite(score) else "—",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if np.isfinite(score) and score < 0.6 else "black",
            )
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="F1", shrink=0.7)
    fig.tight_layout()
    MAPS.save(fig, stem)


def quantitative(config, out):
    sites = [s["id"] for s in config["sites"]]
    methods = [m["id"] for m in config["methods"]]
    path = out / "reference_headline_15m.csv"
    if not path.exists():
        return
    headline = pd.read_csv(path)
    for value, label in (
        ("boundary_f1", "Boundary F1 at 15 m"),
        ("f1", "One-to-one detection F1, IoU ≥0.5"),
    ):
        heatmap(
            headline,
            sites,
            ["ftw", *methods],
            value,
            f"Reference evaluation · {label}\n"
            "Coverage and year categories vary; not all references independent",
            out / f"reference_{value}_heatmap",
        )
    path = out / "prediction_agreement.csv"
    if path.exists() and path.stat().st_size > 2:
        agreement = pd.read_csv(path)
        group = agreement[
            (agreement.boundary_tolerance_m == 15) & (agreement.evaluation_scope == "whole_aoi")
        ]
        for value in ("symmetric_boundary_agreement_f1", "correspondence_f1"):
            heatmap(
                group,
                sites,
                methods,
                value,
                f"Prediction agreement with FTW · {value}\n"
                "Historical cohorts are cross-year; disagreement is not measured error",
                out / f"agreement_{value}_heatmap",
            )
    pairs = pd.read_csv(out / "reference_paired_differences.csv")
    contrasts = [f"{m} minus ftw" for m in methods] + [
        f"{a} minus {b}" for a, b in config["planned_landsat_contrasts"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    for ax, metric in zip(axes, ("boundary_f1", "f1"), strict=True):
        pivot = pairs.pivot_table(index="site", columns="contrast", values=metric).reindex(
            index=sites, columns=contrasts
        )
        pivot.to_csv(out / f"reference_paired_{metric}_source.csv")
        cmap = plt.get_cmap("RdBu").copy()
        cmap.set_bad("#dddddd")
        im = ax.imshow(pivot.to_numpy(), vmin=-0.5, vmax=0.5, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(contrasts)), contrasts, rotation=75, ha="right", fontsize=8)
        ax.set_yticks(range(len(sites)), sites, fontsize=8)
        ax.set_title(metric)
        fig.colorbar(im, ax=ax, shrink=0.7, label="Left minus right")
    fig.suptitle(
        "Paired differences against the same source references · categories in source table"
    )
    fig.tight_layout()
    MAPS.save(fig, out / "reference_paired_differences")
    accuracy = pd.read_csv(out / "reference_accuracy.csv")
    size = accuracy[
        (accuracy.boundary_tolerance_m == 15)
        & accuracy.primary_scope
        & (accuracy.crop_stratum == "all_crops")
        & (accuracy.size_class_ha != "all")
    ]
    size.to_csv(out / "field_size_figure_source.csv", index=False)
    classes = list(dict.fromkeys(size.size_class_ha))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ("boundary_f1", "f1"), strict=True):
        for product, group in size.groupby("product"):
            if product not in ["ftw", *methods]:
                continue
            means = (
                group[group.n_reference > 0]
                .groupby("size_class_ha")[metric]
                .mean()
                .reindex(classes)
            )
            ax.plot(range(len(classes)), means, label=product, marker="o", markersize=3)
        ax.set_xticks(range(len(classes)), classes, rotation=25)
        ax.set_ylim(0, 1)
        ax.set_title(metric + " · descriptive equal-site mean")
        ax.set_xlabel("Reference field-size bin (ha); counts in source CSV")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    MAPS.save(fig, out / "field_size_performance")
    crop = accuracy[(accuracy.crop_stratum != "all_crops") & (accuracy.size_class_ha == "all")]
    if len(crop):
        crop.to_csv(out / "crop_strata_figure_source.csv", index=False)
        labels = (
            crop[["site", "crop_stratum", "n_reference"]]
            .drop_duplicates()
            .sort_values(["site", "crop_stratum"])
        )
        fig, ax = plt.subplots(figsize=(12, max(6, 0.25 * len(labels))))
        x = np.arange(len(labels))
        ax.barh(x, labels.n_reference, color="#3377aa")
        ax.set_yticks(x, [f"{r.site} · {r.crop_stratum}" for r in labels.itertuples()], fontsize=7)
        ax.set_xlabel("Reference polygons (unknown and sparse groups retained)")
        ax.set_title("Crop strata available from provider labels · counts, not inferred species")
        fig.tight_layout()
        MAPS.save(fig, out / "crop_strata_counts")
        plot = crop[crop["product"].isin(["ftw", *methods])].copy()
        plot["site_crop"] = plot.site + " / " + plot.crop_stratum
        pivot = plot.pivot_table(index="site_crop", columns="product", values="boundary_f1")
        pivot.to_csv(out / "crop_strata_performance_source.csv")
        fig, ax = plt.subplots(figsize=(12, max(6, 0.25 * len(pivot))))
        im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(pivot)), pivot.index, fontsize=7)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        ax.set_title("Provider crop strata · boundary F1 at 15 m; counts in companion figure")
        fig.colorbar(im, ax=ax, label="Reference boundary F1")
        fig.tight_layout()
        MAPS.save(fig, out / "crop_strata_performance")


def runtime_figures(out):
    path = out / "runtime_stages.csv"
    if not path.exists():
        return
    timing = pd.read_csv(path)
    inference = timing[timing.stage == "model_inference"]
    scores_path = out / "reference_headline_15m.csv"
    if len(inference) and scores_path.exists():
        scores = pd.read_csv(scores_path)
        merged = inference.merge(scores, on=["site", "product"])
        merged.to_csv(out / "runtime_accuracy_source.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, group in merged.groupby("product"):
            ax.scatter(group.seconds, group.boundary_f1, label=name, s=30, alpha=0.7)
        ax.set_xlabel("Landsat model inference seconds (historical hardware in source CSV)")
        ax.set_ylabel("Reference boundary F1 at 15 m")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, ncol=2)
        ax.set_title("Landsat runtime versus conditional reference scores · FTW inference unknown")
        fig.tight_layout()
        MAPS.save(fig, out / "landsat_runtime_accuracy")
    query = timing[timing.stage == "query_download"]
    if len(query):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(query.site, query.seconds, color="#3377aa")
        ax.set_xlabel("FTW query/download seconds; not upstream inference time")
        ax.set_title("Published-product retrieval timing")
        fig.tight_layout()
        MAPS.save(fig, out / "ftw_retrieval_runtime")


def input_illustration(site, out, data):
    """Explain Landsat channels on a separate page, never boundary backgrounds."""
    import rasterio

    old = ROOT / site["existing_dir"] if site["historical"] else out / "preparation" / site["id"]
    sr, pan = old / "inputs/landsat.tif", old / "inputs/landsat-pan.tif"
    if not sr.exists() or not pan.exists():
        return
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, path, indexes, title in zip(
        axes[:3],
        [sr, sr, pan],
        [(3, 2, 1), (4, 3, 2), (1, 1, 1)],
        ["SR RGB · native 30 m", "SR NIR/Red/Green · native 30 m", "PAN B8 TOA · native 15 m"],
        strict=True,
    ):
        image, extent, _ = MAPS.background(path, indexes, muted=False)
        window = data[4]["overview"]
        ax.imshow(image, extent=extent, interpolation="nearest")
        ax.set_xlim(window[0], window[2])
        ax.set_ylim(window[1], window[3])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        MAPS.scalebar(ax, window)
    with rasterio.open(pan) as stream:
        values = stream.read(1, masked=True).astype(float).filled(np.nan)
        bounds = stream.bounds
    h, w = values.shape
    if h % 2 or w % 2:
        raise ValueError("PAN illustration requires aligned 2:1 grid")
    coarse = values.reshape(h // 2, 2, w // 2, 2).mean((1, 3)).repeat(2, 0).repeat(2, 1)
    detail = values - coarse
    limit = max(float(np.nanpercentile(np.abs(detail), 98)), 1e-8)
    image = axes[3].imshow(
        detail,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        vmin=-limit,
        vmax=limit,
        cmap="RdBu_r",
        interpolation="nearest",
    )
    window = data[4]["overview"]
    axes[3].set_xlim(window[0], window[2])
    axes[3].set_ylim(window[1], window[3])
    axes[3].set_title("PAN minus aligned 2×2 mean\nNative detail in F–G", fontsize=10)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    MAPS.scalebar(axes[3], window)
    fig.colorbar(image, ax=axes[3], shrink=0.65, label="TOA reflectance difference")
    fig.suptitle(
        f"{site['id']} · explanatory inputs; separate from all boundary-comparison backgrounds"
    )
    fig.text(
        0.5,
        0.03,
        "Display stretches differ by channel. PAN TOA and SR are distinct quantities; "
        "NIR is never PAN-sharpened.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.1, top=0.85, wspace=0.06)
    MAPS.save(fig, out / site["id"] / "input_channels")


def generate(config, out, *, render_maps=True, failures=None):
    out = Path(out)
    start = time.perf_counter()
    loaded = {}
    failures = list(failures or [])
    for site in config["sites"]:
        try:
            data = load_site(site, out)
            if data is not None:
                loaded[site["id"]] = data
                if render_maps:
                    site_maps(site, out, data)
        except Exception as error:
            failures.append({"site": site["id"], "error": f"{type(error).__name__}: {error}"})
    representative(config, out, loaded)
    for site in config["sites"]:
        if site["id"] in loaded:
            input_illustration(site, out, loaded[site["id"]])
    quantitative(config, out)
    runtime_figures(out)
    (out / "figure_status.json").write_text(
        json.dumps(
            {
                "n_sites_rendered": len(loaded) - len(failures),
                "failures": failures,
                "render_s": time.perf_counter() - start,
            },
            indent=2,
        )
        + "\n"
    )
