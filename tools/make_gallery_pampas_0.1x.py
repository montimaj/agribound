#!/usr/bin/env python
"""
Render the gallery image that compares the 0.1.x Pampas README image with current outputs.

Usage::

    python tools/make_gallery_pampas_0.1x.py  # writes assets/gallery_1.1/Pampas_0.1x_comparison_*
    python tools/make_gallery_pampas_0.1x.py --old-layer PATH  # the 0.1.x layer elsewhere

The agribound 0.1.x README showed example 15 as a screenshot
(``assets/gallery_0.1x/Pampas_example.png``): a wide, rotated view with thick
red outlines on a basemap of unknown date. This script draws the example's
outputs in the same frame, with lines of the same width, so the two versions
can be compared as they were first seen:

1. the 0.1.x screenshot as published;
2. the 0.1.x layer that the screenshot shows (SAM 2 on three TESSERA
   dimensions, split variant; ``outputs/pampas_semi_supervised``), redrawn;
3. the current layer of the gallery (TESSERA + SAM 2 on Sentinel-2, split
   variant);
4. the current layer made the same way as the 0.1.x one (SAM 2 on TESSERA
   dimensions, split variant).

Panels 2-4 are drawn on the October 2024 Sentinel-2 composite of the current run,
resampled (bilinear) into the screenshot's frame and stretched between the 1st
and 99.5th percentiles pooled over the three bands, as in
``tools/make_gallery.py``; the notes give its window and image count from its
tags. All outlines are red, as in the screenshot.

Frame: :data:`FRAME` maps EPSG:32720 coordinates to screenshot pixels. It was
fitted on 2026-09-29 by matching the 0.1.x crop-filter outlines to the
screenshot's red lines (Nelder-Mead on the mean distance to the nearest red
pixel, truncated at 15, 8 and 4 px in turn): 12.55 m per pixel, rotated
32.8°, so the screenshot covers about 23.4 x 18.3 km. With it, the 0.1.x layer
of panel 2 covers the screenshot's red lines to within 2 px (F1 0.999).

Inputs are the 0.1.x layer of the screenshot (``--old-layer``, by default
``outputs/pampas_semi_supervised/fields_sam2_tessera_improved_2024.gpkg``, an
output of 0.1.x that is not in the repository) and the run root of the current
layers (``--run-root``). As in ``tools/make_gallery.py``, the defaults are the
1.1 run root and gallery (:data:`make_gallery.RUN_ROOT`, ``outputs/examples_1.1``,
and :data:`make_gallery.OUT_DIR`, ``assets/gallery_1.1``); ``--run-root
outputs/examples_1.0 --out-dir assets/gallery_1.0`` renders into the 1.0 gallery
again. Titles and notes name the installed agribound version
(:data:`make_gallery.__version__`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_gallery as mg  # noqa: E402  (style constants, note wrapping, previews)

SCREENSHOT = REPO / "assets" / "gallery_0.1x" / "Pampas_example.png"
OLD_OUTPUTS = REPO / "outputs" / "pampas_semi_supervised"
OLD_LAYER = OLD_OUTPUTS / "fields_sam2_tessera_improved_2024.gpkg"  # default of --old-layer
NEW_OUTPUTS = "outputs/pampas_semi_supervised"  # relative to --run-root
CRS = "EPSG:32720"
#: (a, b, tx, ty): screenshot column = a*x - b*y + tx, row = -b*x - a*y + ty (x, y in EPSG:32720).
FRAME = (0.06696856079718885, -0.043147936621919326, -318235.7152140557, 387253.20521954086)
LINE_PX = 5  # the screenshot's outline width, in screenshot pixels
LINE_RGB = (230, 20, 20)
NAME = "Pampas_0.1x_comparison_example"
V = mg.__version__  # the release the gallery documents


def frame_affine():
    """The affine transform from EPSG:32720 to screenshot pixels."""
    from rasterio.transform import Affine

    a, b, tx, ty = FRAME
    return Affine(a, -b, tx, -b, -a, ty)


def frame_polygon(width: int, height: int):
    """The screenshot's footprint in EPSG:32720."""
    import shapely

    inv = ~frame_affine()
    return shapely.Polygon(
        [inv * (0, 0), inv * (width, 0), inv * (width, height), inv * (0, height)]
    )


def draw_outlines(path: Path, background, width: int, height: int):
    """Draw every polygon ring of *path* on a copy of *background* (a PIL image)."""
    import geopandas as gpd
    from PIL import ImageDraw
    from shapely import affinity

    m = frame_affine()
    gdf = gpd.read_file(path).to_crs(CRS)
    gdf = gdf[gdf.intersects(frame_polygon(width, height).buffer(500))]
    img = background.copy()
    draw = ImageDraw.Draw(img)
    for geom in gdf.geometry:
        g = affinity.affine_transform(geom, [m.a, m.b, m.d, m.e, m.c, m.f])
        for poly in [g] if g.geom_type == "Polygon" else list(g.geoms):
            for ring in [poly.exterior, *poly.interiors]:
                draw.line(list(ring.coords), fill=LINE_RGB, width=LINE_PX, joint="curve")
    return img, len(gpd.read_file(path))


def composite_background(tif: Path, width: int, height: int):
    """The composite's R, G, B bands resampled into the screenshot frame, as a PIL image."""
    import rasterio
    from PIL import Image
    from rasterio.warp import Resampling, reproject

    with rasterio.open(tif) as src:
        idx, _grey = mg._rgb_indices(src, "sentinel2")
        out = np.full((3, height, width), np.nan, np.float32)
        for i, band in enumerate(idx):
            reproject(
                rasterio.band(src, band),
                out[i],
                dst_transform=~frame_affine(),
                dst_crs=src.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    valid = np.isfinite(out).all(axis=0)
    lo, hi = np.percentile(out[:, valid], [1, 99.5])
    rgb = np.clip((out - lo) / (hi - lo), 0, 1)
    rgb[:, ~valid] = 1.0  # outside the composite: white
    return Image.fromarray((np.moveaxis(rgb, 0, -1) * 255).astype(np.uint8))


def composite_note(tif: Path) -> str:
    """The composite as the notes name it, from its tags: "Sentinel-2 L2A 10 m median
    composite of 2024-10-01 to 2024-11-01 (end exclusive), 8 images"."""
    comp = mg._composite_facts({}, tif)
    label = mg.SOURCE_LABELS["sentinel2"]
    if comp.get("resolution_m"):
        label += f" {float(comp['resolution_m']):g} m"
    method = f" {comp['composite_method']}" if comp.get("composite_method") else ""
    window = ""
    if comp.get("date_start") and comp.get("date_end_exclusive"):
        window = f" of {comp['date_start']} to {comp['date_end_exclusive']} (end exclusive)"
    n_img = f", {mg._count(comp['n_images'], 'image')}" if comp.get("n_images") else ""
    return f"{label}{method} composite{window}{n_img}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--run-root",
        default=str(mg.RUN_ROOT),
        help="run root of the current layers (default: make_gallery's, outputs/examples_1.1)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(mg.OUT_DIR),
        help="gallery folder (default: make_gallery's, assets/gallery_1.1)",
    )
    ap.add_argument(
        "--old-layer",
        default=str(OLD_LAYER),
        help="the 0.1.x layer of the screenshot (default: "
        f"{OLD_LAYER.relative_to(REPO)}, not in the repository)",
    )
    args = ap.parse_args(argv)

    import matplotlib.pyplot as plt
    from PIL import Image

    root, out_dir, old_layer = Path(args.run_root), Path(args.out_dir), Path(args.old_layer)
    if not old_layer.exists():
        ap.error(f"{old_layer}: no such file (the 0.1.x layer; pass --old-layer)")
    new = root / NEW_OUTPUTS
    shot = Image.open(SCREENSHOT).convert("RGB")
    w, h = shot.size
    tif = mg._one(f"{NEW_OUTPUTS}/.agribound_cache/sentinel2_composite_*.tif", root)
    bg = composite_background(tif, w, h)
    panels = [("0.1.x README image (as published)", shot)]
    for label, path in [
        ("0.1.x, the layer in that image", old_layer),
        (f"{V}, the gallery layer", new / "fields_tessera_crop_sam2-s2-split_2024.gpkg"),
        (
            f"{V}, made as the 0.1.x layer",
            new / "fields_tessera_crop_sam2-tessera-split_2024.gpkg",
        ),
    ]:
        img, n = draw_outlines(path, bg, w, h)
        panels.append((f"{label} ({n:,} fields)", img))

    notes = [
        "Top left: the image the agribound 0.1.x README showed for example 15 (TESSERA "
        "clusters, Dynamic World crop filter and SAM 2), on a basemap of unknown date. "
        "Top right: the 0.1.x layer it shows, SAM 2 on three TESSERA dimensions after splitting "
        f"multi-part polygons, polygons over 50 ha unrefined. Bottom left: {V}, TESSERA clusters "
        "with SAM 2 on Sentinel-2, parts over 50 ha unrefined (the gallery layer). Bottom right: "
        f"{V}, SAM 2 on three TESSERA dimensions with the same rule (made as the 0.1.x layer).",
        "Panels 2-4: the same frame (about 23.4 x 18.3 km, rotated 32.8°, 12.55 m per pixel; "
        f"outlines 5 px wide, as in the screenshot) on the {composite_note(tif)}, from the {V} "
        "run. "
        "0.1.x polygons cover the whole bounding box of the study area and reach up to about "
        f"650 m beyond it; {V} keeps those whose representative point lies inside the "
        f"study-area pentagon (whole, not clipped). White (top left): outside the {V} "
        "composite, which covers that bounding box. All outlines are red, as in the screenshot.",
        "Frame: fitted by matching the 0.1.x crop-filter outlines to the screenshot's red "
        "lines (2026-09-29).",
    ]
    note_w = mg.FIG_WIDTH_IN - 2 * mg.NOTE_X_IN
    lines = [line for n in notes for line in mg._wrap(n, note_w)]
    ncols, nrows = 2, 2
    panel_w = mg.FIG_WIDTH_IN * (mg.MAPS_RIGHT - mg.MAPS_LEFT) / (ncols + mg.MAPS_WSPACE)
    panel_h = panel_w * h / w
    footer = mg.FOOT_TOP_IN + mg._notes_height_in(len(lines)) + mg.FOOT_BOTTOM_IN
    fig_h = mg.TITLE_IN + nrows * panel_h + mg.ROW_GAP_IN + footer
    fig, axes = plt.subplots(nrows, ncols, figsize=(mg.FIG_WIDTH_IN, fig_h), dpi=mg.DPI)
    for ax, (title, img) in zip(axes.ravel(), panels, strict=True):
        ax.imshow(np.asarray(img), interpolation="antialiased")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6)
        ax.set_title(title, fontsize=mg.TITLE_FS, pad=4)
    first = footer - mg.FOOT_TOP_IN - mg.NOTE_FS / 72
    for i, line in enumerate(lines):
        fig.text(
            mg.NOTE_X_IN / mg.FIG_WIDTH_IN,
            (first - i * mg.NOTE_LINE_IN) / fig_h,
            line,
            ha="left",
            va="baseline",
            fontsize=mg.NOTE_FS,
            color=mg.NOTE_COLOR,
        )
    fig.subplots_adjust(
        left=mg.MAPS_LEFT,
        right=mg.MAPS_RIGHT,
        top=1 - mg.TITLE_IN / fig_h,
        bottom=footer / fig_h,
        wspace=mg.MAPS_WSPACE,
        hspace=mg.ROW_GAP_IN / panel_h,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{NAME}.png"
    fig.savefig(png, dpi=mg.DPI, metadata={"Software": None})
    plt.close(fig)
    preview = mg._write_preview(png, out_dir)
    print(png)
    print(preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
