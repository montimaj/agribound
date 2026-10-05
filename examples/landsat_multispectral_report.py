"""Measured summaries, reference inventory and checksummed local artifact manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown_table(frame):
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def build(config, out):
    out = Path(out)
    if not (out / "headline_15m.csv").exists():
        return
    headline = pd.read_csv(out / "headline_15m.csv")
    status = pd.read_csv(out / "suite_status.csv")
    differences = pd.read_csv(out / "paired_differences.csv")
    rows = []
    rng = np.random.default_rng(42)
    for contrast, group in differences.groupby("contrast", sort=False):
        for metric in ("boundary_f1", "f1"):
            finite = group[np.isfinite(group[metric])]
            values = finite[metric].to_numpy()
            weights = finite.n_reference.to_numpy()
            if not len(values):
                continue
            # Paired site bootstrap is descriptive: sites are not a random population sample.
            indexes = rng.integers(0, len(values), size=(2000, len(values)))
            for weighting in ("equal_site", "reference_count"):
                w = np.ones(len(values)) if weighting == "equal_site" else weights
                bootstrap = np.sum(values[indexes] * w[indexes], axis=1) / np.sum(
                    w[indexes], axis=1
                )
                rows.append(
                    dict(
                        contrast=contrast,
                        metric=metric,
                        weighting=weighting,
                        n_sites=len(values),
                        mean_difference=float(np.average(values, weights=w)),
                        site_bootstrap_95_low=float(np.quantile(bootstrap, 0.025)),
                        site_bootstrap_95_high=float(np.quantile(bootstrap, 0.975)),
                        improved_sites=int((values > 1e-9).sum()),
                        worsened_sites=int((values < -1e-9).sum()),
                        tied_sites=int((np.abs(values) <= 1e-9).sum()),
                    )
                )
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(out / "paired_difference_summary.csv", index=False)
    inventory, timing = [], []
    for site in config["sites"]:
        old, folder = ROOT / site["existing_dir"], out / site["id"]
        meta_path = old / "reference_evaluation.json"
        meta = read(meta_path) if meta_path.exists() else read(old / "reference_source.json")
        reference = gpd.read_file(old / "reference_evaluation.gpkg")
        source_links = meta.get("sources", meta.get("documentation", meta.get("urls", [])))
        if isinstance(source_links, str):
            source_links = [source_links]
        inventory.append(
            dict(
                site=site["id"],
                country=site["country"],
                region=site["region"],
                bbox=json.dumps(site["bbox"]),
                date_start=site["date_start"],
                date_end=site["date_end"],
                reference_kind=site["reference_kind"],
                n_reference=len(reference),
                median_area_ha=reference.to_crs(6933).area.median() / 10000,
                reference_year=site["reference_year"],
                vintage_matches=meta.get("vintage_matches"),
                geometry_vintage=meta.get(
                    "geometry_vintage",
                    meta.get(
                        "vintage_note",
                        "Declaration/reference year; physical edge date not verified",
                    ),
                ),
                position_uncertainty=meta.get(
                    "position_uncertainty", "No numerical positional accuracy documented"
                ),
                license=meta.get("license", meta.get("attribution")),
                redistribution=site["redistribution"],
                source_links=json.dumps(source_links),
                limitations=meta.get(
                    "limitations", meta.get("coverage", "See inherited reference metadata")
                ),
                climate=site["climate"],
                climate_evidence=site.get("climate_evidence"),
                crop_context=site["crop_context"],
                crop_evidence=json.dumps(site.get("crop_evidence")),
                site_selection=site.get("rationale", site.get("selection_rationale")),
                reference_sha256=sha256(old / "reference_evaluation.gpkg"),
                per_field_crop_labels="Regional only"
                if site["id"].startswith("vn_")
                else "Provider attributes; species specificity varies",
            )
        )
        for method in config["methods"]:
            path = folder / f"fields_{method['id']}.gpkg.provenance.json"
            if not path.exists():
                continue
            p = read(path)
            timing.append(
                dict(
                    site=site["id"],
                    experiment=method["id"],
                    hostname=p["hostname"],
                    platform=p["platform"],
                    machine=p["machine"],
                    device=p["device"],
                    precision=p["engine_meta"]["precision"],
                    torch=p["versions"].get("torch"),
                    python=p["python"],
                    reused=p["multispectral_comparison"]["inference_reused"],
                )
            )
    pd.DataFrame(inventory).to_csv(out / "reference_inventory.csv", index=False)
    timing = pd.DataFrame(timing)
    timing.to_csv(out / "timing_environment.csv", index=False)
    environment_columns = [
        "hostname",
        "platform",
        "machine",
        "device",
        "precision",
        "torch",
        "python",
    ]
    timing_compatible = len(timing[environment_columns].drop_duplicates()) == 1
    summary = pd.read_csv(out / "aggregate_metrics.csv")
    overall = summary[summary.reference_category == "all_primary_scopes"]
    quantitative = []
    for method in config["methods"]:
        group = overall[overall.experiment == method["id"]].set_index("weighting")
        if group.empty:
            continue
        quantitative.append(
            {
                "Method": method["name"],
                "Measured sites": int(group.loc["equal_site", "n_sites"]),
                "Boundary F1, equal site": f"{group.loc['equal_site', 'boundary_f1']:.3f}",
                "Detection F1, equal site": f"{group.loc['equal_site', 'f1']:.3f}",
                "Boundary F1, field weighted": f"{group.loc['reference_count', 'boundary_f1']:.3f}",
                "Detection F1, field weighted": f"{group.loc['reference_count', 'f1']:.3f}",
            }
        )
    site_rows = []
    for site in config["sites"]:
        row = {"Site": site["id"]}
        for method in config["methods"]:
            group = headline[(headline.site == site["id"]) & (headline.experiment == method["id"])]
            row[method["name"].split()[0]] = (
                f"{group.iloc[0].boundary_f1:.3f}/{group.iloc[0].f1:.3f}"
                if len(group)
                else "Unmeasured"
            )
        site_rows.append(row)
    site_table = pd.DataFrame(site_rows)
    site_table.to_csv(out / "per_site_headline_wide.csv", index=False)
    completed = status[status.status == "complete"]
    reused = int(completed.inference_reused.fillna(False).astype(bool).sum())
    notes = []
    for left, right in config["planned_contrasts"]:
        contrast = f"{left} minus {right}"
        for metric in ("boundary_f1", "f1"):
            result = contrasts[
                (contrasts.contrast == contrast)
                & (contrasts.metric == metric)
                & (contrasts.weighting == "equal_site")
            ]
            if result.empty:
                continue
            r = result.iloc[0]
            notes.append(
                f"- {contrast}, {metric}: mean change {r.mean_difference:+.3f}; "
                f"improved/worsened/tied {r.improved_sites}/{r.worsened_sites}/{r.tied_sites}; "
                "descriptive paired-site bootstrap interval "
                f"[{r.site_bootstrap_95_low:+.3f}, {r.site_bootstrap_95_high:+.3f}]."
            )
    unmeasured = status[status.status != "complete"]
    interpretation = []
    equal = overall[overall.weighting == "equal_site"].set_index("experiment")
    weighted = overall[overall.weighting == "reference_count"].set_index("experiment")
    if set(m["id"] for m in config["methods"]).issubset(equal.index):
        boundary_best = equal.boundary_f1.idxmax()
        detection_best = equal.f1.idxmax()
        interpretation.append(
            f"The highest equal-site boundary average is {boundary_best} "
            f"({equal.loc[boundary_best, 'boundary_f1']:.3f}); the highest detection "
            f"average is {detection_best} ({equal.loc[detection_best, 'f1']:.3f}). "
            "These averages mix reference types and evaluation scopes."
        )
        nir = differences[differences.contrast == "false_color minus sr"]
        interpretation.append(
            f"False-color SR improves detection at {int((nir.f1 > 0).sum())}/{len(nir)} "
            f"sites. Its equal-site detection change is "
            f"{equal.loc['false_color', 'f1'] - equal.loc['sr', 'f1']:+.3f}, "
            "whereas the reference-count-weighted change is "
            f"{weighted.loc['false_color', 'f1'] - weighted.loc['sr', 'f1']:+.3f}. "
            "The weighting disagreement is a reason to inspect individual sites rather "
            "than describe NIR as generally better for this checkpoint."
        )
        control = differences[differences.contrast == "pan_nir_red minus coarse_pan_nir_red"]
        interpretation.append(
            f"Native PAN detail improves the stack's boundary score at "
            f"{int((control.boundary_f1 > 0).sum())}/{len(control)} sites and detection at "
            f"{int((control.f1 > 0).sum())}/{len(control)}. "
            "That supports the utility of native detail within this input design; "
            "it does not establish that stacking is preferable to PAN alone."
        )
        for site_id, first, second in [
            ("ca_colusa", "false_color", "pan"),
            ("vn_mekong", "pan_nir_red", "pan"),
            ("mt_beaverhead", "pan_nir_red", "pan"),
        ]:
            sub = headline[headline.site == site_id].set_index("experiment")
            if first not in sub.index or second not in sub.index:
                continue
            interpretation.append(
                f"At {site_id}, {first} has boundary/detection F1 "
                f"{sub.loc[first, 'boundary_f1']:.3f}/{sub.loc[first, 'f1']:.3f}, "
                f"compared with {second} "
                f"{sub.loc[second, 'boundary_f1']:.3f}/{sub.loc[second, 'f1']:.3f}. "
                "Interpret this using the site's documented reference type; "
                "a boundary improvement can coexist with low field detection."
            )
        for method in ("pan_nir_red", "hybrid"):
            interpretation.append(
                f"For {method}, equal-site boundary/detection changes versus PAN are "
                f"{equal.loc[method, 'boundary_f1'] - equal.loc['pan', 'boundary_f1']:+.3f}/"
                f"{equal.loc[method, 'f1'] - equal.loc['pan', 'f1']:+.3f}; versus existing "
                f"fused RGB they are "
                f"{equal.loc[method, 'boundary_f1'] - equal.loc['combined', 'boundary_f1']:+.3f}/"
                f"{equal.loc[method, 'f1'] - equal.loc['combined', 'f1']:+.3f}. "
                "Its extra construction should be justified with the intended site's "
                "reference and metric, rather than a universal default."
            )
        if equal.n_sites.nunique() > 1:
            interpretation.append(
                "Method averages currently have unequal site coverage; use paired contrasts first."
            )
    lines = [
        "# Landsat multispectral comparison — measured report",
        "",
        f"Completed {len(completed)} method/site combinations at {headline.site.nunique()} sites: "
        f"{len(completed) - reused} new delineations and {reused} reused baseline outputs. "
        f"{len(unmeasured)} combinations remain unmeasured. "
        "No original inference outputs were overwritten.",
        "",
        "## Per-site results",
        "",
        "Each entry is boundary F1 at 15 m / one-to-one detection F1 at IoU ≥0.5. "
        "A–H follow the frozen method order. North American scores are conditional on known "
        "mapped footprints; other sites retain their original AOI evaluation. "
        "These are not uniformly physical-field references.",
        "",
        markdown_table(site_table),
        "",
        "## Aggregate results",
        "",
        markdown_table(pd.DataFrame(quantitative)),
        "",
        "Equal-site summaries give each measured site equal weight. Field-weighted summaries "
        "weight each site score by its reference count; neither is pooled precision or recall. "
        "Reference-quality categories are also tabulated separately in aggregate_metrics.csv. "
        "The mixed all-site summary is descriptive, not a global agricultural accuracy estimate.",
        "",
        "## Planned comparisons",
        "",
        *notes,
        "",
        "D−B changes spectral channels with the same 30 m grid. E−D changes the input grid and "
        "physical inference context without new SR information. F−G keeps NIR, Red, grid and "
        "settings identical while changing only PAN within-block detail, although standard "
        "per-channel normalization is refitted. H−E keeps NIR unchanged and injects visible-band "
        "detail. F−D changes both composition and sampling and cannot isolate PAN detail.",
        "",
        "## Interpretation and limitations",
        "",
        *[paragraph + "\n" for paragraph in interpretation],
        "Assess NIR and combined inputs using the paired contrasts and per-site results above. "
        "Different boundary and detection outcomes reflect missed fields, merges and splits; "
        "the best boundary score need not have the best detection score. Extra fusion preparation "
        "is justified only when the measured improvement matters for the intended landscape "
        "and reference interpretation; the suite establishes no universal winning input.",
        "",
        "The published checkpoint was trained on 512×512 RGB patches "
        "([author paper](https://arxiv.org/html/2607.19069v1)). D–H change the training channel "
        "semantics and are pretrained-input experiments, not optimized multispectral models. "
        "Its documented training GSD ends at 10 m, "
        "so both Landsat resolutions are outside that range.",
        "",
        "PAN B8 is TOA reflectance; NIR and visible SR channels are surface reflectance. "
        "The hybrid visible channels are experimental fused values, not calibrated 15 m SR. "
        "Landsat PAN spans approximately 0.50–0.68 μm and NIR B5 0.85–0.88 μm "
        "([USGS bands](https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites)). "
        "NIR is never sharpened here. "
        "All 15 m NIR/Red/Green channels retain native 30 m information.",
        "",
        "At super-resolution=1, a 512-pixel model tile covers 7.68 km at 15 m and 15.36 km "
        "at 30 m; 0.5 tile step also changes its ground distance. The same checkpoint and "
        "settings do not make physical context identical. Every input uses the existing "
        "scene-level per-channel 1–99% positive-value stretch, uint8 conversion and standard "
        "Ultralytics tensor normalization. Internal BGR reads are reversed by Ultralytics, "
        "restoring the requested logical RGB-slot order.",
        "",
        "Historical baselines and new runs have identical recorded timing environments: "
        f"{timing_compatible}. Delineation wall time includes model load; preparation_s is shared "
        "preparation of all five new inputs, not a per-method acquisition cost. Cached imagery "
        "and model download are excluded. Single timings have no measured runtime uncertainty; "
        "thermal state and operating-system load may differ.",
        "",
        "Sites are purposively selected and spatially dependent fields are not independent "
        "replicates. The paired-site bootstrap describes variability among these sites, not "
        "population confidence or causal crop/climate effects. Existing field-bootstrap "
        "intervals in JSON can be optimistic under spatial dependence. Crop labels are provider "
        "attributes where available; Vietnamese rice descriptions are regional context only. "
        "Positional uncertainty is not numerically documented. Reference vintage, geographic "
        "training overlap and physical-versus-administrative differences remain limitations.",
        "The Mekong comparison has one matched scene and approximately 98.7% common "
        "valid imagery coverage. The smallest fields, the 1000 m² output minimum and "
        "the absence of calibrated positional-error bounds limit small-field conclusions.",
        "",
        "South African geometry mostly predates 2018; Vietnamese reference digitization in "
        "August 2021 postdates the tested seasonal windows. Several North American geometry "
        "dates are not independently established. Full details and original source links are "
        "retained in reference_inventory.csv and per-site reference_inventory.json.",
        "",
        "## Figures and artifacts",
        "",
        "Each measured site has two map pages covering A–H plus a reference-only column, "
        "overview/small-field/shared-edge rows, identical imagery, and metric labels. "
        "The representative plate uses eight preselected sites in six countries and identical "
        "900 m reference-selected windows. It includes site crop context, climate, reference "
        "type and median field size; view-level provider attributes are saved separately. "
        "Climate/crop evidence and selection are in representative_figure_data.json. "
        "A companion representative overview plate uses the same frozen AOIs with explicit "
        "varying scale bars. The detail plate keeps a common 900 m scale. "
        "representative_controls adds reference-only, fused RGB, resampled false color "
        "and coarse-PAN panels at the same frozen sites and 900 m windows; its source "
        "data are representative_controls_figure_data.json. "
        "No Mexico or independent Corn Belt benchmark exists in this frozen cohort.",
        "",
        "All map outputs remain local. The representative plate includes a restricted South "
        "African reference and must not be redistributed without resolving provider terms. "
        "Washington and Montana reference-overlay maps also remain local. See the inventory "
        "for dataset-specific licenses. Numeric CSV/summary figures contain no reference geometry.",
        "",
        "## Unmeasured results",
        "",
    ]
    for site_id, group in unmeasured.groupby("site"):
        error = group.error.dropna().iloc[0] if group.error.notna().any() else "Not executed"
        lines += [
            f"- {site_id}: {error}",
            "  Resume: `python examples/27_landsat_multispectral_comparison.py "
            f"--sites {site_id} --gee-project ee-rappjer --skip-figures`.",
        ]
    lines += [
        "",
        "The Red River live retry preserved the original February–May 2021 window and "
        "20% cloud threshold. No matched observations were found; changing those would "
        "require a separate experiment, not a replacement in this cohort.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    asset_dir = ROOT / "assets/landsat_pan_comparison/multispectral"
    asset_dir.mkdir(parents=True, exist_ok=True)
    release_files = [
        "comparison.csv",
        "headline_15m.csv",
        "aggregate_metrics.csv",
        "paired_differences.csv",
        "paired_difference_summary.csv",
        "per_site_headline_wide.csv",
        "reference_inventory.csv",
        "suite_status.csv",
        "baseline_preservation_validation.csv",
        "engine_validity_check.csv",
        "verification.json",
        "all_site_performance.png",
        "paired_contrasts.png",
        "field_size_performance.png",
        "runtime_accuracy.png",
        "REPORT.md",
    ]
    for name in release_files:
        if (out / name).exists():
            shutil.copy2(out / name, asset_dir / name)
    docs_asset = ROOT / "docs/assets/landsat_multispectral/all_site_performance.png"
    if (out / "all_site_performance.png").exists():
        docs_asset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "all_site_performance.png", docs_asset)
    (asset_dir / "README.md").write_text(
        "# Landsat multispectral results\n\n"
        "Measured numeric summaries and quantitative figures from example 27. "
        "Source licenses, years, geometry limitations and direct evidence links are "
        "retained in reference_inventory.csv. Raw vectors and reference-overlay maps "
        "remain in local outputs, including the restricted representative plate.\n\n"
        "Regenerate: `python examples/27_landsat_multispectral_comparison.py --summarize-only`.\n",
        encoding="utf-8",
    )
    artifacts = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or any(p.startswith(".") for p in path.relative_to(out).parts):
            continue
        if (
            path.name == "artifact_manifest.json"
            or "ultralytics" in path.parts
            or path.suffix == ".log"
        ):
            continue
        restricted = any(
            s["id"] in path.parts
            for s in config["sites"]
            if s["id"] in ("za_hessequa", "wa_grant", "mt_beaverhead")
        )
        if path.name.startswith(
            ("representative_landscapes", "representative_overviews", "representative_controls")
        ):
            restricted = True
        artifacts.append(
            {
                "path": path.relative_to(out).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "redistribution": "Local; inspect dataset terms"
                if restricted
                else "Inspect source attribution before redistribution",
            }
        )
    (out / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "configuration_sha256": sha256(out / "experiments_frozen.json"),
                "source_code_sha256": {
                    str(path.relative_to(ROOT)): sha256(path)
                    for path in [
                        ROOT / "examples/27_landsat_multispectral_comparison.py",
                        ROOT / "examples/landsat_multispectral_figures.py",
                        ROOT / "examples/landsat_multispectral_report.py",
                        ROOT / "agribound/composites/landsat_multispectral.py",
                    ]
                },
                "numeric_asset_sha256": {
                    name: sha256(asset_dir / name)
                    for name in release_files
                    if (asset_dir / name).exists()
                },
                "artifacts": artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    output = ROOT / "outputs/landsat_multispectral"
    build(read(output / "experiments_frozen.json"), output)
