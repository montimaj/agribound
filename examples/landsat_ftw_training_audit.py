"""Audit published FTW v1 chip splits without claiming model-independent validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = {
    "France": "france",
    "Netherlands": "netherlands",
    "Spain": "spain",
    "South Africa": "south_africa",
    "Vietnam": "vietnam",
}


def chip_overlap(reference, chips, bbox):
    """Count reference geometries intersecting published split footprints."""
    if reference.crs is None or chips.crs is None or "split" not in chips:
        raise ValueError("Chip audit needs CRS and documented split labels")
    ref = reference.to_crs(chips.crs)
    aoi = gpd.GeoSeries([box(*bbox)], crs=4326).to_crs(chips.crs).iloc[0]
    subset = chips[chips.intersects(aoi)]
    result = {"n_reference": len(ref), "n_chips_intersecting_aoi": len(subset)}
    for label in ("train", "val", "test", "none"):
        group = subset[subset["split"].fillna("none").eq(label)]
        result[f"n_{label}_chips"] = len(group)
        result[f"n_reference_intersecting_{label}_footprints"] = (
            int(
                (
                    ref.to_crs(6933)
                    .geometry.intersection(group.to_crs(6933).geometry.union_all())
                    .area
                    > 0
                ).sum()
            )
            if len(group)
            else 0
        )
    result["interpretation"] = (
        "Benchmark chip-footprint overlap, not verified model parcel membership"
    )
    return result


def spatial_category(record):
    if record.get("status") != "complete":
        return record.get("status", "unknown")
    if record.get("n_reference_intersecting_train_footprints", 0) > 0:
        return "overlap_with_published_v1_training_footprints"
    return "no_reference_overlap_with_published_v1_training_footprints"


def run(config, out, offline):
    cache = out / "benchmark_split_cache"
    cache.mkdir(exist_ok=True)
    rows = []
    for site in config["sites"]:
        slug = COUNTRIES.get(site["country"])
        record = {
            "site": site["id"],
            "country": site["country"],
            "published_ftw_exact_training_membership": "unknown",
            "delineate_anything_training_membership": "unknown",
            "independent_holdout_established": False,
        }
        if slug is None:
            rows.append(
                {
                    **record,
                    "status": "not_in_published_v1_country_inventory",
                    "limitation": "An unlisted country does not prove model training independence",
                }
            )
            continue
        path = cache / f"chips_{slug}.parquet"
        pin = path.with_suffix(".pin.json")
        url = f"https://data.source.coop/kerner-lab/fields-of-the-world/{slug}/chips_{slug}.parquet"
        try:
            if path.exists():
                stored = json.loads(pin.read_text())
                if hashlib.sha256(path.read_bytes()).hexdigest() != stored["sha256"]:
                    raise ValueError("Changed chip snapshot")
            else:
                if offline:
                    raise RuntimeError("Chip snapshot unavailable offline")
                try:
                    with urlopen(url, timeout=60) as response:
                        payload = response.read()
                        headers = {
                            k: response.headers.get(k)
                            for k in ("ETag", "Last-Modified", "Content-Length")
                        }
                except Exception as http_error:
                    # The provider also documents this anonymous S3 endpoint.
                    from pyarrow.fs import S3FileSystem

                    filesystem = S3FileSystem(anonymous=True, region="us-west-2")
                    key = (
                        "us-west-2.opendata.source.coop/kerner-lab/fields-of-the-world/"
                        f"{slug}/chips_{slug}.parquet"
                    )
                    with filesystem.open_input_file(key) as stream:
                        payload = stream.read()
                    info = filesystem.get_file_info(key)
                    headers = {
                        "http_error": str(http_error),
                        "s3_path": key,
                        "size": info.size,
                        "mtime": str(info.mtime),
                    }
                path.write_bytes(payload)
                pin.write_text(
                    json.dumps(
                        {
                            "url": url,
                            "headers": headers,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            chips = gpd.read_parquet(path)
            reference = gpd.read_file(out / site["id"] / "reference.gpkg")
            record.update(chip_overlap(reference, chips, site["bbox"]))
            record.update(
                status="complete",
                snapshot_sha256=json.loads(pin.read_text())["sha256"],
                source_url=url,
                benchmark_version="Published v1 chip inventory",
            )
        except Exception as error:
            record.update(
                status="unmeasured", source_url=url, error=f"{type(error).__name__}: {error}"
            )
        rows.append(record)
    pd.DataFrame(rows).to_csv(out / "benchmark_split_overlap.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "examples/landsat_ftw_comparison_sites.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/landsat_ftw_comparison")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output_dir, args.offline)
