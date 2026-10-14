#!/usr/bin/env python
"""
Export one square of a year's NAIP from Earth Engine, RGB-only years included.

agribound's ``naip`` source mosaics only the 4-band images (R, G, B, N) of
``USDA/NAIP/DOQQ``, and some early years have three bands (R, G, B; Montana's
flights of 2004-2006 among them). This tool exports one square of
such a year as the background of a figure: the documentation gallery's
Montana image of example 25 (entry 25c of ``tools/make_gallery.py``) draws the
2005 NAIP of example 25's Montana square beside the 2023 composite that
Delineate-Anything read there. That image was made with::

    python tools/fetch_naip_history.py --year 2005 --crs EPSG:32612 \\
        --corner 375000 5385000 --side 5000 --resolution 2 \\
        --out-dir outputs/examples_1.1_r3/montana

Method
------
- Images: those of ``USDA/NAIP/DOQQ`` acquired in the calendar year (from
  ``<year>-01-01`` to ``<year + 1>-01-01``) whose footprint meets the square,
  sorted by acquisition time.
- Bands: R, G and B, with N only when every image has it.
- Mosaic: the latest image on top (``ee.ImageCollection.mosaic`` of the sorted
  images), in the native projection of the first image.
- Pixels: each output pixel is the mean of the native pixels in it
  (``ee.Image.reduceResolution`` with ``ee.Reducer.mean()``, at most
  ``MAX_NATIVE_PIXELS`` native pixels per output pixel), rounded to whole
  numbers; uint8 with nodata 0. agribound's own NAIP composites keep Earth
  Engine's nearest-neighbour resampling instead.
- Grid: the square in ``--crs``, with its origin at the square's upper-left
  corner and ``--resolution`` pixels; the side must be a whole number of pixels.
- Download: agribound's Earth Engine download helper
  (:func:`agribound.composites.gee.export_ee_image`, geedim).

Outputs, in ``--out-dir`` (named ``naip_<year>_<bands>_<resolution>m``, e.g.
``naip_2005_rgb_2m``):

- ``.tif``: the image, with its band names and the ``AGRIBOUND_*`` tags that
  agribound's composites use to describe their contents (collection, mosaic
  order, window, number and list of images, CRS, pixel size, value scale and
  valid fraction), so that ``tools/make_gallery.py`` describes it as it
  describes a composite; ``NAIP_HISTORY_*`` tags name this tool, the pixel
  aggregation and the record;
- ``.json``: the record of the export: the asset ID, date, bands, native CRS
  and pixel size of each image, the export settings and grid, the file's
  SHA-256 and valid fraction, the software versions and the credit.

Earth Engine's averages are not bit-identical between requests: two exports of
the Montana square on 2026-10-12 differed by 1 in 14 of its 6.25 million
pixels. The record's SHA-256 therefore names the file that a figure draws
(``tools/make_gallery.py`` checks it).

``--dry-run`` lists the images and the grid and writes nothing. An existing
image is replaced only with ``--overwrite``. The tool needs ``earthengine-api``
and ``geedim`` (``pip install "agribound[gee]"``) and an Earth Engine project
(``--gee-project``, else the project ``agribound.auth.setup_gee`` finds).

NAIP: USDA Farm Service Agency (public domain), via Google Earth Engine.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agribound.composites.gee import (  # noqa: E402
    ExportGrid,
    _image_list_tags,
    _utc_date,
    ee_geometry,
    ee_warning_monitor,
    export_ee_image,
    grid_footprint_4326,
    raster_valid_fraction,
)

COLLECTION = "USDA/NAIP/DOQQ"
#: Bands every image must have; N is added when every image of the year has it.
RGB = ("R", "G", "B")
#: Most native pixels averaged into one output pixel (``ee.Image.reduceResolution``).
MAX_NATIVE_PIXELS = 1024
AGGREGATION = (
    "mean of the native pixels in each output pixel (ee.Image.reduceResolution, "
    f"ee.Reducer.mean, maxPixels {MAX_NATIVE_PIXELS}), rounded to whole numbers"
)
MOSAIC = "images sorted by acquisition time, the latest on top (ee.ImageCollection.mosaic)"
CREDIT = "NAIP: USDA Farm Service Agency (public domain), via Google Earth Engine"


def square_grid(crs: str, corner: tuple[float, float], side_m: float, resolution_m: float):
    """The pixel grid of the square with lower-left *corner* and side *side_m* in *crs*.

    The origin is the square's upper-left corner. Raises ValueError unless the side and
    the pixel size are positive and the side is a whole number of pixels.
    """
    from rasterio.transform import Affine

    if not side_m > 0 or not resolution_m > 0:
        raise ValueError(f"side and resolution must be positive, got {side_m} and {resolution_m}")
    n = side_m / resolution_m
    if abs(n - round(n)) > 1e-9:
        raise ValueError(
            f"a side of {side_m:g} m is not a whole number of {resolution_m:g} m pixels"
        )
    x0, y0 = (float(v) for v in corner)
    transform = Affine(resolution_m, 0.0, x0, 0.0, -resolution_m, y0 + side_m)
    return ExportGrid(str(crs), transform, int(round(n)), int(round(n)))


def shared_bands(images: list[dict]) -> list[str]:
    """R, G and B, then N when every image has it; ValueError without images or with an image
    that lacks R, G or B."""
    if not images:
        raise ValueError("no image")
    for image in images:
        missing = [b for b in RGB if b not in image["bands"]]
        if missing:
            raise ValueError(f"{image['asset_id']} has no band {', '.join(missing)}")
    return [*RGB, *(["N"] if all("N" in image["bands"] for image in images) else [])]


def output_stem(year: int, bands: list[str], resolution_m: float) -> str:
    """``naip_2005_rgb_2m``: the name of the image and its record, without the suffix."""
    return f"naip_{int(year)}_{''.join(bands).lower()}_{resolution_m:g}m"


def _year_collection(grid: ExportGrid, year: int):
    import ee

    region = ee_geometry(grid_footprint_4326(grid))
    return (
        ee.ImageCollection(COLLECTION)
        .filterBounds(region)
        .filterDate(f"{int(year)}-01-01", f"{int(year) + 1}-01-01")
        .sort("system:time_start")
    )


def list_images(grid: ExportGrid, year: int) -> list[dict]:
    """The images of *year* whose footprint meets *grid* (one Earth Engine request), sorted
    by acquisition time: asset ID (``system:id``, else ``system:index``), UTC date,
    ``system:time_start`` (ms), bands, native CRS and pixel size."""
    import ee

    def row(image):
        native = image.select(0).projection()  # the first band's
        return ee.Feature(
            None,
            {
                "asset_id": ee.Algorithms.If(
                    image.get("system:id"), image.get("system:id"), image.get("system:index")
                ),
                "time_start_ms": image.get("system:time_start"),
                "bands": image.bandNames(),
                "crs": native.crs(),
                "native_resolution_m": native.nominalScale(),
            },
        )

    with ee_warning_monitor(f"NAIP {year} image list"):
        features = _year_collection(grid, year).map(row).getInfo()["features"]
    rows = []
    for feature in features:
        p = feature["properties"]
        asset = str(p["asset_id"])
        rows.append(
            {
                "asset_id": asset if "/" in asset else f"{COLLECTION}/{asset}",
                "date": _utc_date(p["time_start_ms"]),
                "time_start_ms": int(p["time_start_ms"]),
                "bands": list(p["bands"]),
                "crs": p["crs"],
                "native_resolution_m": float(p["native_resolution_m"]),
            }
        )
    return sorted(rows, key=lambda r: (r["time_start_ms"], r["asset_id"]))


def naip_image(grid: ExportGrid, year: int, bands: list[str]):
    """The Earth Engine image this tool exports (module docstring, "Method")."""
    import ee

    images = _year_collection(grid, year)
    native = ee.Image(images.first()).select(0).projection()
    return (
        images.select(list(bands))
        .mosaic()
        .setDefaultProjection(native)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=MAX_NATIVE_PIXELS)
        .reproject(crs=grid.crs, crsTransform=grid.crs_transform)
        .round()
        .uint8()
    )


def download(grid: ExportGrid, year: int, bands: list[str], path: Path, tags: dict) -> None:
    """Download the image of :func:`naip_image` on *grid* to *path* (uint8, nodata 0)."""
    export_ee_image(
        naip_image(grid, year, bands),
        path,
        grid=grid,
        dtype="uint8",
        band_names=bands,
        tags=tags,
        label=f"NAIP {year}",
    )


def image_tags(images: list[dict], grid: ExportGrid, year: int, record_name: str) -> dict:
    """GeoTIFF tags of the image: agribound's composite tags that describe its contents,
    and the ``NAIP_HISTORY_*`` tags of this tool."""
    rows = [[r["asset_id"], r["time_start_ms"], None] for r in images]
    return {
        "AGRIBOUND_COLLECTIONS": COLLECTION,
        "AGRIBOUND_COMPOSITE_METHOD": "mosaic",
        "AGRIBOUND_MOSAIC_ORDER": "newest on top",
        "AGRIBOUND_BAND_FILTER": "none: every image of the year, the bands they all have",
        "AGRIBOUND_DATE_START": f"{int(year)}-01-01",
        "AGRIBOUND_DATE_END_EXCLUSIVE": f"{int(year) + 1}-01-01",
        "AGRIBOUND_IMAGE_YEARS": str(int(year)),
        "AGRIBOUND_N_IMAGES": len(images),
        **_image_list_tags(rows, len(images)),
        "AGRIBOUND_EXPORT_CRS": grid.crs,
        "AGRIBOUND_RESOLUTION_M": float(grid.transform.a),
        "AGRIBOUND_VALUE_SCALE": "uint8",
        "NAIP_HISTORY_TOOL": "tools/fetch_naip_history.py",
        "NAIP_HISTORY_AGGREGATION": AGGREGATION,
        "NAIP_HISTORY_RECORD": record_name,
    }


def _versions() -> dict:
    """agribound's version (this checkout's) and the installed versions of the others."""
    from importlib.metadata import PackageNotFoundError, version

    from agribound._version import __version__

    found = {"agribound": __version__}
    for name in ("earthengine-api", "geedim", "rasterio"):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def build_record(
    *,
    images: list[dict],
    bands: list[str],
    grid: ExportGrid,
    year: int,
    corner: tuple[float, float],
    side_m: float,
    gee_project: str | None,
    out: Path,
    valid_fraction: float,
) -> dict:
    """The JSON record of an export (module docstring, "Outputs")."""
    data = out.read_bytes()
    return {
        "tool": "tools/fetch_naip_history.py",
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": _versions(),
        "gee_project": gee_project,
        "collection": COLLECTION,
        "year": int(year),
        "date_window": [f"{int(year)}-01-01", f"{int(year) + 1}-01-01"],
        "square": {
            "crs": grid.crs,
            "corner": [float(corner[0]), float(corner[1])],
            "side_m": float(side_m),
        },
        "images": [{k: v for k, v in r.items() if k != "time_start_ms"} for r in images],
        "bands": list(bands),
        "export": {
            "crs": grid.crs,
            "crs_transform": grid.crs_transform,
            "width": grid.width,
            "height": grid.height,
            "resolution_m": float(grid.transform.a),
            "dtype": "uint8",
            "nodata": 0,
            "mosaic": MOSAIC,
            "aggregation": AGGREGATION,
            "download": "agribound.composites.gee.export_ee_image (geedim)",
        },
        "output": {
            "file": out.name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "valid_fraction": round(float(valid_fraction), 6),
        },
        "credit": CREDIT,
    }


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--year", type=int, required=True, help="NAIP year (calendar year)")
    ap.add_argument("--crs", required=True, help='CRS of the square, e.g. "EPSG:32612"')
    ap.add_argument(
        "--corner",
        type=float,
        nargs=2,
        required=True,
        metavar=("X", "Y"),
        help="lower-left corner of the square, in the CRS",
    )
    ap.add_argument("--side", type=float, default=5000.0, help="side of the square, m (5000)")
    ap.add_argument("--resolution", type=float, default=2.0, help="pixel size, m (2)")
    ap.add_argument("--out-dir", type=Path, required=True, help="folder of the image and record")
    ap.add_argument("--gee-project", default=None, help="Earth Engine project ID")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing image")
    ap.add_argument(
        "--dry-run", action="store_true", help="list the images and the grid; write nothing"
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    from agribound.auth import setup_gee

    args = _parser().parse_args(argv)
    grid = square_grid(args.crs, tuple(args.corner), args.side, args.resolution)
    setup_gee(project=args.gee_project)
    images = list_images(grid, args.year)
    if not images:
        raise SystemExit(f"No {COLLECTION} image of {args.year} meets the square.")
    bands = shared_bands(images)
    stem = output_stem(args.year, bands, args.resolution)
    out = args.out_dir / f"{stem}.tif"
    record_path = out.with_suffix(".json")
    print(f"{len(images)} images of {args.year}, bands {' '.join(bands)}:")
    for r in images:
        print(
            f"  {r['asset_id']}  {r['date']}  {''.join(r['bands'])}  {r['crs']} "
            f"{r['native_resolution_m']:g} m"
        )
    t = grid.transform
    print(
        f"grid: {grid.crs}, {grid.width} x {grid.height} px of {t.a:g} m, upper-left corner "
        f"({t.c:.12g}, {t.f:.12g}) -> {out}"
    )
    if args.dry_run:
        return 0
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    download(grid, args.year, bands, out, image_tags(images, grid, args.year, record_path.name))
    # The share of the square's pixels with data in any band (a dark pixel can be 0 in one).
    valid = raster_valid_fraction(out, grid_footprint_4326(grid), require="any")
    if not math.isfinite(valid) or valid <= 0:
        raise SystemExit(f"{out} holds no valid pixel")
    import rasterio

    with rasterio.open(out, "r+") as dst:
        dst.update_tags(AGRIBOUND_VALID_FRACTION=str(round(valid, 6)))  # as the builders do
    record = build_record(
        images=images,
        bands=bands,
        grid=grid,
        year=args.year,
        corner=tuple(args.corner),
        side_m=args.side,
        gee_project=args.gee_project,
        out=out,
        valid_fraction=valid,
    )
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out} (valid {valid:.4f}) and {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
