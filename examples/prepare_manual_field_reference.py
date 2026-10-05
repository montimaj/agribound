"""Create a blank, prediction-blind field annotation kit; never labels imagery.

See docs/user-guide/north-america-reference-workflow.md for independent review.
The evaluation area is a search box, not certified reference coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def prepare_template(
    output,
    bbox,
    *,
    reference_year,
    imagery_source,
    imagery_date,
    imagery_license,
    positional_accuracy,
    imagery_path=None,
):
    if output.exists():
        raise ValueError(f"Preserve existing annotation kit: {output}")
    imagery_sha256 = None
    if imagery_path:
        with imagery_path.open("rb") as stream:
            imagery_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    x0, y0, x1, y1 = bbox
    if not (-180 <= x0 < x1 <= 180 and -90 <= y0 < y1 <= 90):
        raise ValueError("bbox must be west south east north in WGS84")
    area = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
    crs = area.estimate_utm_crs()
    columns = [
        "field_id",
        "crop_label",
        "crop_evidence",
        "edge_basis",
        "confidence",
        "annotator",
        "reviewer",
        "review_status",
        "imagery_date",
        "notes",
    ]
    fields = gpd.GeoDataFrame(
        {name: pd.Series(dtype="str") for name in columns},
        geometry=gpd.GeoSeries([], crs=crs),
    )
    edges = gpd.GeoDataFrame(
        {name: pd.Series(dtype="str") for name in ("edge_id", "reason", "reviewer", "notes")},
        geometry=gpd.GeoSeries([], crs=crs),
    )
    coverage = gpd.GeoDataFrame(
        {name: pd.Series(dtype="str") for name in ("reviewer", "review_date", "notes")},
        geometry=gpd.GeoSeries([], crs=crs),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    area.to_crs(crs).to_file(output, layer="evaluation_area", driver="GPKG")
    fields.to_file(output, layer="fields", driver="GPKG", geometry_type="Polygon")
    edges.to_file(output, layer="ambiguous_edges", driver="GPKG", geometry_type="LineString")
    coverage.to_file(output, layer="reviewed_coverage", driver="GPKG", geometry_type="Polygon")
    metadata = {
        "status": "unannotated_unreviewed",
        "physical_gold_standard": False,
        "bbox": bbox,
        "crs": str(crs),
        "reference_year": reference_year,
        "imagery_source": imagery_source,
        "imagery_date": imagery_date,
        "imagery_license": imagery_license,
        "positional_accuracy": positional_accuracy,
        "imagery_file_sha256": imagery_sha256,
        "blindness": "Do not load Agribound predictions or synthetic field products",
        "coverage": "Empty until all eligible fields and omissions are independently reviewed",
        "definition": "Whole visible cultivation units; exclude invisible declaration/plot splits",
        "crop_evidence": (
            "Record producer/provider observations or unknown; no visual species inference"
        ),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bbox", type=float, nargs=4, required=True)
    parser.add_argument("--reference-year", type=int, required=True)
    parser.add_argument("--imagery-source", required=True, help="Dated provider URL or item ID")
    parser.add_argument(
        "--imagery-date", required=True, help="Actual acquisition date or date range"
    )
    parser.add_argument("--imagery-license", required=True)
    parser.add_argument(
        "--positional-accuracy", required=True, help="Documented accuracy, or unknown"
    )
    parser.add_argument("--imagery-path", type=Path)
    args = parser.parse_args()
    prepare_template(
        args.output,
        args.bbox,
        reference_year=args.reference_year,
        imagery_source=args.imagery_source,
        imagery_date=args.imagery_date,
        imagery_license=args.imagery_license,
        positional_accuracy=args.positional_accuracy,
        imagery_path=args.imagery_path,
    )
    print(f"Unannotated kit: {args.output}. Independent annotation/review remains required.")


if __name__ == "__main__":
    main()
