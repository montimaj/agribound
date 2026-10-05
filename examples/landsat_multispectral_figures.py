"""Regenerable common-background maps and quantitative Landsat suite figures."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
REF_COLOR, PRED_COLOR = "#00e5ff", "#ffb000"
SHORT = {
    "pan": "A PAN · 15 m\nPAN / PAN / PAN",
    "sr": "B SR RGB · 30 m\nRed / Green / Blue",
    "combined": "C Fused RGB · 15 m\nR fused / G fused / B fused",
    "false_color": "D SR false color · 30 m\nNIR / Red / Green",
    "false_color_15m": "E Resampled SR · 15 m grid\nNIR / Red / Green",
    "pan_nir_red": "F Direct stack · 15 m grid\nPAN / NIR / Red",
    "coarse_pan_nir_red": "G Coarse PAN · 15 m grid\nPAN 30 m / NIR / Red",
    "hybrid": "H Hybrid · 15 m grid\nNIR / R fused / G fused",
}
REPRESENTATIVE_LABELS = {
    "beauce": ("Temperate regional context", "Wheat/maize declarations", "Declaration parcels"),
    "nl_meierij": ("Temperate maritime", "Maize/potato declarations", "Declaration parcels"),
    "es_olite": ("Dry-summer Mediterranean", "Annual/perennial land use", "Land-use enclosures"),
    "za_hessequa": (
        "Southern Cape winter crops",
        "Grain/forage labels",
        "Survey/cadastral reference",
    ),
    "vn_mekong": (
        "Tropical monsoonal",
        "Rice region; species unknown",
        "Digitized physical fields",
    ),
    "ca_colusa": ("Dry-summer Central Valley", "Rice mapping units", "Crop mapping units"),
    "ca_orchards": ("Dry-summer Central Valley", "Orchard/vineyard units", "Crop mapping units"),
    "ut_cache": ("Interior mountain valley", "Hay/alfalfa attributes", "Irrigation mapping units"),
}


def save(fig, stem, *, svg=True):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    if svg:
        fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def background(path, indexes=(3, 2, 1), *, muted=True):
    with rasterio.open(path) as src:
        bands = src.read(list(indexes), masked=True).filled(np.nan)
        bounds, crs = src.bounds, src.crs
    rendered = np.zeros_like(bands, dtype=float)
    valid = np.isfinite(bands).all(0)
    for i, band in enumerate(bands):
        sample = band[valid]
        low, high = np.percentile(sample, [1, 99])
        rendered[i] = np.clip((band - low) / max(high - low, 1e-8), 0, 1)
    rgb = np.moveaxis(np.nan_to_num(rendered), 0, -1)
    if muted:
        grey = rgb.mean(-1, keepdims=True)
        rgb = (0.45 * rgb + 0.55 * grey) * 0.7
    rgba = np.concatenate([rgb, valid[..., None].astype(float)], axis=-1)
    return rgba, (bounds.left, bounds.right, bounds.bottom, bounds.top), crs


def layer_in_window(frame, window):
    if frame.empty:
        return frame
    indexes = frame.sindex.query(box(*window), predicate="intersects")
    return frame.iloc[indexes]


def scalebar(ax, window):
    width, height = window[2] - window[0], window[3] - window[1]
    target = width / 4
    base = 10 ** np.floor(np.log10(target))
    length = max(x for x in (base, 2 * base, 5 * base) if x <= target)
    x, y = window[0] + 0.06 * width, window[1] + 0.07 * height
    ax.plot([x, x + length], [y, y], color="black", linewidth=4)
    ax.plot([x, x + length], [y, y], color="white", linewidth=2)
    ax.text(
        x + length / 2,
        y + height * 0.025,
        f"{length:g} m",
        color="white",
        ha="center",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
    )


def draw_map(ax, image, extent, reference, predicted, window, coverage=None):
    ax.set_facecolor("#eeeeee")
    ax.imshow(image, extent=extent, interpolation="nearest")
    ref = layer_in_window(reference, window)
    pred = layer_in_window(predicted, window) if predicted is not None else None
    if len(ref):
        ref.geometry.boundary.plot(ax=ax, color=REF_COLOR, linewidth=0.85, linestyle="--")
    if pred is not None and len(pred):
        pred.geometry.boundary.plot(ax=ax, color=PRED_COLOR, linewidth=0.9)
    if coverage is not None:
        # This outlines known coverage only; it does not clip field polygons.
        outline = gpd.GeoSeries([coverage.geometry.union_all()], crs=coverage.crs)
        outline.boundary.plot(ax=ax, color="white", linewidth=0.4, alpha=0.5)
    ax.set_xlim(window[0], window[2])
    ax.set_ylim(window[1], window[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    scalebar(ax, window)


def load_site(site, out):
    folder = out / site["id"]
    old = ROOT / site["existing_dir"]
    inputs = old / "inputs" if (old / "inputs/landsat.tif").exists() else folder / "acquisition"
    image, extent, crs = background(inputs / "landsat.tif")
    reference = gpd.read_file(old / "reference_evaluation.gpkg").to_crs(crs)
    reference.geometry = reference.geometry.make_valid(method="structure")
    windows = json.loads((folder / "figure_windows.json").read_text())
    coverage = None
    if (old / "evaluation_coverage.gpkg").exists():
        coverage = gpd.read_file(old / "evaluation_coverage.gpkg").to_crs(crs)
    predictions = {
        method: gpd.read_file(path).to_crs(crs)
        for method, path in completed_prediction_paths(folder, coverage is not None).items()
    }
    return image, extent, reference, predictions, windows, coverage


def completed_prediction_paths(folder, has_coverage):
    """Exclude leftover vectors from failed or unverified experiments."""
    status_path = folder / "run_status.json"
    statuses = json.loads(status_path.read_text())["methods"] if status_path.exists() else {}
    suffix = "_evaluated" if has_coverage else ""
    paths = {}
    for method in SHORT:
        path = folder / f"fields_{method}{suffix}.gpkg"
        if statuses.get(method, {}).get("status") == "complete" and path.exists():
            paths[method] = path
    return paths


def legend(fig):
    fig.legend(
        handles=[
            Line2D([0], [0], color=REF_COLOR, linestyle="--", label="Reference"),
            Line2D([0], [0], color=PRED_COLOR, label="Prediction"),
            Line2D([0], [0], color="#777777", label="Common unsharpened SR RGB background"),
        ],
        loc="lower center",
        ncol=3,
        fontsize=10,
    )


def per_site(site, out):
    image, extent, reference, predictions, windows, coverage = load_site(site, out)
    headline = pd.read_csv(out / "headline_15m.csv")
    scores = headline[headline.site == site["id"]].set_index("experiment")
    groups = [
        ["pan", "sr", "combined", "false_color"],
        ["false_color_15m", "pan_nir_red", "coarse_pan_nir_red", "hybrid"],
    ]
    for number, methods in enumerate(groups, 1):
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        for row, key in enumerate(("overview", "small_fields", "shared_edges")):
            window = windows[key]
            draw_map(axes[row, 0], image, extent, reference, None, window, coverage)
            if row == 0:
                axes[row, 0].set_title("Reference only\nCommon SR background", fontsize=11)
            axes[row, 0].set_ylabel(
                {
                    "overview": "Whole study area",
                    "small_fields": "Small fields · 900 m",
                    "shared_edges": "Shared edges · 900 m",
                }[key],
                fontsize=11,
            )
            for column, method in enumerate(methods, 1):
                ax = axes[row, column]
                draw_map(ax, image, extent, reference, predictions.get(method), window, coverage)
                if method not in predictions:
                    ax.text(
                        0.5,
                        0.5,
                        "UNMEASURED",
                        transform=ax.transAxes,
                        ha="center",
                        bbox={"facecolor": "white", "alpha": 0.8},
                    )
                if row == 0:
                    score = ""
                    if method in predictions and method in scores.index:
                        item = scores.loc[method]
                        score = f"\nBoundary F1 {item.boundary_f1:.3f} · detection F1 {item.f1:.3f}"
                    ax.set_title(SHORT[method] + score, fontsize=10)
        scope = (
            "Known-footprint conditional scores"
            if coverage is not None
            else "Whole-AOI reference agreement"
        )
        fig.suptitle(
            f"{site['id']} · {site['country']} · {site['date_start']} to {site['date_end']}\n"
            f"{scope}; reference type: {site['reference_kind'][:110]}",
            fontsize=13,
        )
        note = (
            "Identical extents in each row. NIR/SR on a 15 m grid remain native 30 m information."
        )
        if not windows["shared_edge_found"]:
            note += " Shared-edge row uses median-area fallback."
        fig.text(0.5, 0.042, note, ha="center", fontsize=9)
        legend(fig)
        fig.subplots_adjust(
            left=0.045, right=0.995, bottom=0.075, top=0.87, wspace=0.04, hspace=0.14
        )
        save(fig, out / site["id"] / f"methods_page_{number}")
    # Input differences are shown separately, never used as method-specific map backgrounds.
    old = ROOT / site["existing_dir"]
    inputs = (
        old / "inputs"
        if (old / "inputs/landsat.tif").exists()
        else out / site["id"] / "acquisition"
    )
    with rasterio.open(inputs / "landsat-pan.tif") as src:
        pan = src.read(1, masked=True).filled(np.nan)
        pan_extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    from agribound.composites.landsat_multispectral import repeat_2x
    from agribound.composites.pan_fusion import block_mean

    detail = pan - repeat_2x(block_mean(pan))
    fc, _, _ = background(inputs / "landsat.tif", (4, 3, 2), muted=False)
    rgb, _, _ = background(inputs / "landsat.tif", muted=False)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, data, title in zip(
        axes,
        [rgb, fc, pan, detail],
        ["SR RGB", "SR NIR/Red/Green", "PAN TOA", "PAN minus 30 m block mean"],
        strict=True,
    ):
        if title.startswith("PAN minus"):
            limit = np.nanpercentile(np.abs(detail), 99)
            rendered = ax.imshow(data, extent=pan_extent, cmap="RdBu_r", vmin=-limit, vmax=limit)
            fig.colorbar(rendered, ax=ax, shrink=0.7, label="TOA reflectance residual")
        elif title == "PAN TOA":
            ax.imshow(
                data,
                extent=pan_extent,
                cmap="gray",
                vmin=np.nanpercentile(pan, 1),
                vmax=np.nanpercentile(pan, 99),
            )
        else:
            ax.imshow(data, extent=extent)
        win = windows["small_fields"]
        ax.set_xlim(win[0], win[2])
        ax.set_ylim(win[1], win[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        scalebar(ax, win)
    fig.suptitle(f"{site['id']}: input explanation only; independent per-band display stretches")
    fig.tight_layout()
    save(fig, out / site["id"] / "input_explanation")


def representative_methods(config, *, controls=False):
    primary = config["representative"]["methods"]
    if controls:
        return [None, *[m["id"] for m in config["methods"] if m["id"] not in primary]]
    return primary


def representative(config, out, *, overview=False, controls=False):
    methods = representative_methods(config, controls=controls)
    sites = [
        next(s for s in config["sites"] if s["id"] == name)
        for name in config["representative"]["sites"]
    ]
    fig, axes = plt.subplots(
        len(sites), len(methods), figsize=(3.6 * len(methods), 3.2 * len(sites)), squeeze=False
    )
    metadata = []
    for row, site in enumerate(sites):
        folder = out / site["id"]
        if not (folder / "figure_windows.json").exists():
            for ax in axes[row]:
                ax.text(0.5, 0.5, "UNMEASURED", ha="center")
                ax.set_axis_off()
            continue
        image, extent, ref, predictions, windows, coverage = load_site(site, out)
        window_key = "overview" if overview else config["representative"]["window"]
        window = windows[window_key]
        for column, method in enumerate(methods):
            ax = axes[row, column]
            draw_map(ax, image, extent, ref, predictions.get(method), window, coverage)
            if row == 0:
                ax.set_title(SHORT[method] if method else "Reference only", fontsize=11)
            if method is not None and method not in predictions:
                ax.text(0.5, 0.5, "UNMEASURED", transform=ax.transAxes, ha="center")
            if column == 0:
                climate, crop, quality = REPRESENTATIVE_LABELS[site["id"]]
                median = ref.to_crs(6933).area.median() / 10000
                label = (
                    f"{site['id']} · {site['country']}\n{climate}\n"
                    f"{crop} [site context]\nMedian reference {median:.2f} ha\n{quality}"
                )
                ax.text(
                    -0.05,
                    0.5,
                    textwrap.fill(label, 23, replace_whitespace=False),
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                )
        metadata.append(
            {
                "site": site["id"],
                "methods": methods,
                "window": window,
                "crs": windows["crs"],
                "crop_context": site["crop_context"],
                "climate": site["climate"],
                "reference_kind": site["reference_kind"],
                "climate_evidence": site.get("climate_evidence"),
                "crop_evidence": site.get("crop_evidence"),
                "display_label": REPRESENTATIVE_LABELS[site["id"]],
                "median_reference_area_ha": float(ref.to_crs(6933).area.median() / 10000),
                "view_crop_attributes": {
                    column: {
                        str(k): int(v)
                        for k, v in layer_in_window(ref, window)[column].value_counts().items()
                    }
                    for column in (
                        "code_cultu",
                        "crop_name",
                        "crop_type",
                        "MAIN_CROP",
                        "Description",
                    )
                    if column in ref
                },
                "scale": (
                    "Frozen AOI extent; different physical scales by row, explicit scale bars"
                    if overview
                    else "900 m square, same physical scale in all rows"
                ),
                "selection": config["representative"]["selection"],
            }
        )
    subtitle = (
        "Frozen study-area extents; physical scale varies by row (see scale bars)"
        if overview
        else "Reference-selected 900 m views"
    )
    fig.suptitle(
        (
            "Landsat comparison: fused RGB and spatial controls\n"
            if controls
            else "Landsat field delineation across contrasting agricultural landscapes\n"
        )
        + subtitle
        + " · common unsharpened SR RGB within each row",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.032,
        "Sites selected from geography/reference metadata before new inference. "
        "Reference quality differs; regional crop/climate labels are descriptive.",
        ha="center",
        fontsize=10,
    )
    legend(fig)
    fig.subplots_adjust(left=0.19, right=0.995, top=0.945, bottom=0.05, wspace=0.035, hspace=0.08)
    stem = (
        "representative_controls"
        if controls
        else "representative_overviews"
        if overview
        else "representative_landscapes"
    )
    save(fig, out / stem)
    data_name = (
        "representative_controls_figure_data.json"
        if controls
        else "representative_overview_figure_data.json"
        if overview
        else "representative_figure_data.json"
    )
    (out / data_name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def quantitative(config, out):
    table = pd.read_csv(out / "comparison.csv")
    headline = pd.read_csv(out / "headline_15m.csv")
    sites, methods = [s["id"] for s in config["sites"]], [m["id"] for m in config["methods"]]
    labels = []
    countries = {
        "France": "FR",
        "Netherlands": "NL",
        "Spain": "ES",
        "South Africa": "ZA",
        "Vietnam": "VN",
        "United States": "US",
        "Canada": "CA",
    }
    for site in config["sites"]:
        kind = site["reference_kind"]
        if site["cohort"] == "france" or site["id"] in ("nl_meierij", "qc2023"):
            kind = "declarations"
        elif site["id"].startswith("es_"):
            kind = "land use"
        elif site["id"].startswith("vn_"):
            kind = "digitized fields"
        elif site["id"] == "za_hessequa":
            kind = "survey/cadastral"
        else:
            kind = kind.replace("_mapping_unit", " units").replace("_", " ")
        labels.append(f"{site['id']} [{countries[site['country']]}; {kind}]")
    fig, axes = plt.subplots(1, 2, figsize=(17, 10))
    for ax, metric, title in zip(
        axes,
        ["boundary_f1", "f1"],
        ["Boundary F1 · 15 m", "Field detection F1 · IoU ≥0.5"],
        strict=True,
    ):
        grid = headline.pivot(index="site", columns="experiment", values=metric).reindex(
            index=sites, columns=methods
        )
        grid.to_csv(out / f"figure_data_{metric}.csv")
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")
        image = ax.imshow(grid.to_numpy(), vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(
            range(len(methods)), [m["name"] for m in config["methods"]], rotation=45, ha="right"
        )
        ax.set_yticks(range(len(sites)), labels, fontsize=8)
        ax.set_title(title)
        for i in range(len(sites)):
            for j in range(len(methods)):
                value = grid.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}" if np.isfinite(value) else "—",
                    ha="center",
                    va="center",
                    color="black" if not np.isfinite(value) or value > 0.55 else "white",
                    fontsize=9,
                )
        fig.colorbar(image, ax=ax, fraction=0.025)
    fig.suptitle(
        "All frozen sites and methods; blocked results remain missing\n"
        "North America: known-footprint conditional evaluation; "
        "other sites: original AOI evaluation",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out / "all_site_performance")
    differences = pd.read_csv(out / "paired_differences.csv")
    contrasts = [f"{a} minus {b}" for a, b in config["planned_contrasts"]]
    fig, axes = plt.subplots(1, 2, figsize=(17, 10))
    for ax, metric in zip(axes, ["boundary_f1", "f1"], strict=True):
        grid = differences.pivot(index="site", columns="contrast", values=metric).reindex(
            index=sites, columns=contrasts
        )
        grid.to_csv(out / f"figure_data_delta_{metric}.csv")
        limit = max(0.05, float(np.nanmax(np.abs(grid.to_numpy()))))
        cmap = plt.get_cmap("RdBu").copy()
        cmap.set_bad("#eeeeee")
        image = ax.imshow(grid.to_numpy(), vmin=-limit, vmax=limit, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(contrasts)), [f"{chr(68 + i)}" for i in range(len(contrasts))])
        ax.set_xticklabels(["D−B", "E−D", "F−G", "H−E", "F−A", "F−C", "H−A", "H−C"])
        ax.set_yticks(range(len(sites)), labels, fontsize=8)
        ax.set_title(
            "Paired Δ " + ("boundary F1 · 15 m" if metric == "boundary_f1" else "detection F1")
        )
        for i in range(len(sites)):
            for j in range(len(contrasts)):
                value = grid.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:+.2f}" if np.isfinite(value) else "—",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if np.isfinite(value) and abs(value) > limit * 0.6 else "black",
                )
        fig.colorbar(image, ax=ax, fraction=0.025)
    fig.suptitle("Predeclared paired contrasts; positive values favor the first method")
    fig.text(
        0.5,
        0.015,
        "D−B: false color vs RGB · E−D: resampling/context · F−G: native PAN detail · "
        "H−E: visible fusion · A: PAN · C: fused RGB",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    save(fig, out / "paired_contrasts")
    strata = table[
        (table.boundary_tolerance_m == 15) & table.primary_scope & (table.size_class_ha != "all")
    ]
    strata.to_csv(out / "figure_data_size_classes.csv", index=False)
    bins = list(strata.size_class_ha.unique())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for method in methods:
        sub = strata[strata.experiment == method]
        for ax, metric in zip(axes[:2], ["boundary_f1", "f1"], strict=True):
            values = [
                sub[sub.size_class_ha == b].loc[lambda f: f.n_reference > 0, metric].mean()
                for b in bins
            ]
            ax.plot(range(len(bins)), values, marker="o", label=SHORT[method].split("\n")[0])
            ax.set_xticks(range(len(bins)), bins, rotation=25)
            ax.set_ylim(0, 1)
    counts = (
        strata[strata.experiment == "pan"].groupby("size_class_ha").n_reference.sum().reindex(bins)
    )
    axes[2].bar(range(len(bins)), counts)
    axes[2].set_xticks(range(len(bins)), bins, rotation=25)
    axes[2].bar_label(axes[2].containers[0], fmt="%d")
    axes[2].set_title("Reference counts across measured sites")
    axes[0].set_title("Equal-site boundary F1 · 15 m")
    axes[1].set_title("Equal-site detection F1")
    axes[1].legend(fontsize=7)
    fig.suptitle("Field-size strata; empty bins excluded from means; differing reference quality")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out / "field_size_performance")
    headline.to_csv(out / "figure_data_runtime.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method in methods:
        sub = headline[headline.experiment == method]
        for ax, metric in zip(axes, ["boundary_f1", "f1"], strict=True):
            ax.scatter(sub.inference_s, sub[metric], label=SHORT[method].split("\n")[0], alpha=0.75)
            ax.set_xlabel("Delineation wall time (s), including model load")
            ax.set_ylim(0, 1)
    axes[0].set_ylabel("Boundary F1 · 15 m")
    axes[1].set_ylabel("Detection F1 · IoU ≥0.5")
    axes[1].legend(fontsize=7)
    fig.suptitle(
        "Runtime versus accuracy; single runs and historical baselines\n"
        "Device/host/software compatibility recorded in timing_environment.csv"
    )
    fig.tight_layout()
    save(fig, out / "runtime_accuracy")


def blocked_reference_preview(site, out):
    """Show available reference geometry without implying imagery or predictions exist."""
    path = ROOT / site["existing_dir"] / "reference_evaluation.gpkg"
    if not path.exists():
        return
    reference = gpd.read_file(path)
    crs = reference.estimate_utm_crs()
    reference = reference.to_crs(crs)
    overview = gpd.GeoSeries([box(*site["bbox"])], crs=4326).to_crs(crs).iloc[0].bounds
    point = reference.loc[reference.area.idxmin()].geometry.representative_point()
    small = (point.x - 450, point.y - 450, point.x + 450, point.y + 450)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, window, title in zip(
        axes[:2],
        [overview, small],
        ["Reference overview", "Small-reference view · 900 m"],
        strict=True,
    ):
        grey = np.full((1, 1, 4), [0.85, 0.85, 0.85, 1.0])
        draw_map(ax, grey, (window[0], window[2], window[1], window[3]), reference, None, window)
        ax.set_title(title)
    axes[2].set_axis_off()
    axes[2].text(
        0,
        0.95,
        "A–H: UNMEASURED\n\nNo matched Landsat 8/9 scenes\n"
        "under the frozen date window\nand ≤20% cloud threshold.\n\n"
        "Grey is a plain background.\nNo imagery or predictions shown.\n\n"
        f"{site['date_start']} to {site['date_end']}\nReference: {site['reference_year']}\n"
        f"{len(reference)} reference polygons",
        va="top",
        fontsize=12,
    )
    fig.suptitle(f"{site['id']} · {site['country']} · available reference only", fontsize=14)
    fig.tight_layout()
    save(fig, out / site["id"] / "reference_preview_unmeasured")


def generate(config, out):
    out = Path(out)
    if not (out / "headline_15m.csv").exists():
        return
    for site in config["sites"]:
        if not (out / site["id"] / "figure_windows.json").exists():
            blocked_reference_preview(site, out)
            continue
        print(f"Rendering maps {site['id']}", flush=True)
        per_site(site, out)
    representative(config, out)
    representative(config, out, overview=True)
    representative(config, out, controls=True)
    quantitative(config, out)
