#!/usr/bin/env python
"""
Render the documentation gallery images from the agribound 1.0 example runs.

Usage::

    python tools/make_gallery.py                       # all entries -> assets/gallery_1.0/
    python tools/make_gallery.py --only 04,06          # some entries
    python tools/make_gallery.py --list                # show the entries and their inputs
    python tools/make_gallery.py --basemap esri        # Esri World Imagery instead of the composite

Inputs are the outputs of ``outputs/examples_1.0/run_all.sh`` (``--run-root``).
Every entry names the output file(s) of one example; a file may be a glob that
must match exactly one file.

Background
----------
By default each map is drawn on a composite from the run, found through the
output's provenance sidecar (``facts.raster_path``): usually the engine's input;
for FTW, its season window A (the first of its two inputs); for the SAM entries
whose refined files have no sidecar, the composite SAM read (``background=``).
The picture therefore shows the acquisition window of that input, which the
0.1.x screenshots (web basemaps of an unknown date) could not. The RGB bands
come from the composite's band descriptions and the source's canonical bands
in :data:`agribound.registry.SOURCE_REGISTRY`; SPOT-Pan is shown in grey.
The Landsat label names the missions of the collections the composite
recorded (``AGRIBOUND_COLLECTIONS``: LE07 and LC08 give "Landsat 7/8 C2 L2").
The composite is read at each panel's own pixel size (averaged down from the
native resolution, never upsampled), with a read window that covers the whole
panel (start rounded down, end rounded up), so rounding leaves no white strip
at the panel edges. Zoom panels are read separately at their own panel size,
so they show native detail unless the zoom window has more native pixels than
the panel: the NAIP zooms of entries 07 (2 km at 1 m) and 16 (1.5 km at
0.6 m) are averaged down to the 1448 px panel width. Imagery is stretched
between the 1st and 99.5th percentiles of the main panel, pooled over the
three bands so colours keep their balance, except SPOT multispectral DN, which
is stretched per band. The zoom panel reuses the main panel's stretch, so both
show the same colours. ``--basemap esri`` draws Esri World Imagery through
contextily instead and prints the Esri attribution on the image.

Locator inset
-------------
Each single-area image has a locator inset at the bottom right, unless the entry sets
``inset=False``. It shows the country in white, the boundaries of its states
or provinces, the containing state or province in pink, the neighbouring
countries in grey and the major lakes, with a red dot at the centre of the
mapped extent. The view is set by the part of the country nearest the dot
and the small parts close to it (``INSET_PART_MAX_AREA``,
``INSET_PART_MAX_DIST``), so Tasmania, Corsica, the Balearics and Tierra del
Fuego are in the view and white, while Alaska, Hawaii and overseas France are
not in it. Boundaries come from Natural Earth (``NATURAL_EARTH``: 1:50m
countries and lakes, 1:10m states and provinces). India is never drawn with
Natural Earth's de facto borders: an India inset shows only the Survey of
India outline (``INDIA_OUTLINE``), with no neighbouring countries, and any
other inset whose view overlaps India draws India from that outline, above
the neighbours' fills and lines. Lakes are drawn above every boundary layer,
so no border runs through them. The files are downloaded over the network
on first use, checked against pinned SHA-256 hashes (a mismatch raises
ValueError) and cached: Natural Earth in ``~/.cache/agribound/naturalearth``,
the India outline in ``~/.cache/agribound/india``. The source credit
(``INSET_NOTE`` or ``INSET_NOTE_INDIA``) is printed under the image.

Multi-area entries (``Entry.multi_area``, rendered by :func:`render_areas`) draw
one panel per layer, each in the CRS of its own background composite, on the
square of side ``Layer.crop_m`` (else ``Entry.crop_m``) that holds the most
polygon representative points among candidates half a side apart. Their inset
is a world locator (``WORLD_INSET_W_IN``, ``WORLD_INSET_PAD_DEG``) with one
numbered red dot per panel; India is drawn from the Survey of India outline as
above, and the credit line starts "Inset: study areas 1-N; ".

Footer
------
Under the maps are the legend and the notes (imagery window, model and
version, inset credit) at ``NOTE_FS`` pt, one line every ``NOTE_LINE_IN``
inches. The notes are wrapped by their measured width so that they stay clear
of the inset and its title, with balanced lines (the fewest lines, then the
narrowest width that keeps that number, so no line is left with one word).
The footer grows with the number of lines; after drawing, the tool checks
that the legend, the notes and the inset do not overlap.

Outputs (``--out-dir``, default ``assets/gallery_1.0``):

- ``<name>.png``: one image per entry, 3000 px wide (10 in at 300 dpi), with
  the imagery window and the model version printed under the panels;
- ``preview/<name>.webp``: a 1600 px wide preview of each PNG for the web
  pages, which embed the preview and link it to the full PNG (Pillow, RGB,
  Lanczos resampling, WebP quality 85, method 6, no metadata);
- ``gallery_stats.json``: per entry, the facts the captions quote (polygon
  counts, area quantiles, source, year, imagery window, engine metadata, the
  inset location (``location``: lon/lat, country, admin1; for a multi-area
  entry, one record per panel and each layer's ``window``) and, when the
  example wrote one, its metrics file). A full run writes the file afresh;
  ``--only`` replaces the rendered entries and keeps the others.

The rendering is deterministic (no random numbers; no dates or software
versions in the PNG metadata; the previews are encoded from the PNG with fixed
settings), so a second run gives byte-identical files.
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import shapely  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.merge import merge as rio_merge  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agribound._version import __version__  # noqa: E402
from agribound.io.raster import percentile_stretch_uint8  # noqa: E402
from agribound.registry import SOURCE_REGISTRY  # noqa: E402

PRED_COLOR = "#ff2d2d"
PRED_LABEL = f"agribound {__version__} fields"  # the release the gallery documents
REF_COLOR = "#00e5ff"
ZOOM_COLOR = "#ffd400"
HIGHLIGHT_COLOR = "#ff9d00"
HIGHLIGHT_LABEL = "Refined by SAM"
REMOVED_COLOR = "#ff3df2"
REMOVED_LABEL = "Removed by the crop filter"
FIG_WIDTH_IN = 10.0
# 3000 px wide full images, for click-through; the docs embed PREVIEW_W px previews
# at 800 px, i.e. 2 px per CSS pixel.
DPI = 300
PREVIEW_W = 1600
PREVIEW_WEBP = {"quality": 85, "method": 6}
MAX_PANEL_READ_SIDE = 6000  # cap on the pixels read per panel side (memory)
TITLE_IN = 0.34
TITLE_FS = 10.5
MAPS_LEFT, MAPS_RIGHT, MAPS_WSPACE = 0.01, 0.99, 0.03  # map grid (figure fractions)
ROW_GAP_IN = TITLE_IN * 1.1  # between map rows, for the titles of the lower rows
# Display stretch: one pair of percentiles pooled over R, G and B (per band for SPOT
# multispectral DN, whose bands have different ranges), so the colour balance of
# reflectance and 8-bit aerial imagery is kept and bright fields are not clipped at
# full resolution.
STRETCH_LOW, STRETCH_HIGH = 1.0, 99.5
LEGEND_FS = 9
NOTE_FS = 8.5
NOTE_LINESPACING = 1.3
NOTE_LINE_IN = NOTE_FS * NOTE_LINESPACING / 72  # baseline to baseline; lines drawn one by one
NOTE_COLOR = "#444444"
SCALE_FS = 9
# Footer layout (inches): legend row at the top, notes under it, inset at the bottom right.
FOOT_TOP_IN = 0.12  # between the maps and the legend
FOOT_BOTTOM_IN = 0.08  # below the notes and the inset
LEGEND_X_IN = 0.05
LEGEND_NOTE_GAP_IN = 0.02  # between the legend row and the first line of notes
NOTE_X_IN = 0.12
NOTE_INSET_GAP_IN = 0.2  # clear space between the notes (or the legend) and the inset
ESRI_ATTRIBUTION = "Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community"

# Locator insets: Natural Earth (public domain), downloaded once and checked by SHA-256.
NATURAL_EARTH = {
    "countries": (
        "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip",
        "5fed433373581fa648920435f937d95f2d3c0200e067409c6478dcdf1b853139",
    ),
    "states": (
        "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip",
        "efc59726337323058f9446210adc96673179cd344e053666ee3d28cb58ba2b05",
    ),
    # Major lakes: the country polygons include them (e.g. the Great Lakes, Lake Victoria).
    "lakes": (
        "https://naciscdn.org/naturalearth/50m/physical/ne_50m_lakes.zip",
        "f28d42c286d96b57a17aac2cbeb432f8c65532c20063495711fbc64e24666df3",
    ),
}
WATER_COLOR = "#dde9f3"
NATURAL_EARTH_DIR = Path.home() / ".cache" / "agribound" / "naturalearth"
# India is drawn from a Survey of India outline (Jammu and Kashmir, including Aksai Chin,
# and Arunachal Pradesh as Indian territory), never from Natural Earth's de facto borders,
# as in the author's grace-grb figures. Pinned to the file's commit in india-geodata; the
# file's only attribute is Source = "Survey of India State Map, Datameet".
INDIA_OUTLINE = (
    "https://raw.githubusercontent.com/yashveeeeeeer/india-geodata/"
    "08ecb3fdf3122b1eb1b1bc149cd7bd6418223588/data/administrative/country/india-soi.geojson",
    "e5321e2010d060c0bffa3b51a1e691af8fbf6063b848c02c123d61d4084cf820",
)
INSET_NOTE = "Inset: Natural Earth country, state/province and lake boundaries"
INSET_NOTE_INDIA = (
    "Inset: India outline from the Survey of India State Map via DataMeet; "
    "other boundaries from Natural Earth"
)
INSET_W_IN, INSET_H_IN = 2.1, 1.5  # largest inset box, outside the maps, at the bottom right
INSET_RIGHT_IN = 0.12  # from the right edge of the image
INSET_TITLE_FS = 8.5
INSET_TITLE_PAD_PT = 3
WORLD_INSET_W_IN = 3.0  # widest world locator (multi-area entries)
WORLD_INSET_PAD_DEG = (12.0, 10.0)  # lon, lat margin around the dots (keeps labels off the frame)
# Inset layers, bottom to top. Lakes are above every boundary layer, so no border runs
# through them; India's Survey of India outline is above the neighbours' fills and the
# subject country's province lines, so no de facto line shows inside it.
Z_NEIGHBOURS, Z_COUNTRY, Z_ADMIN1_LINES, Z_ADMIN1 = 1.0, 1.1, 1.2, 1.3
Z_INDIA, Z_LAKES, Z_MARKER = 1.4, 1.5, 5
# Other parts of the country that set the inset view with the part nearest the dot:
# smaller than this share of that part's area and closer than this share of its larger
# bounding-box side (Tasmania, Corsica, Mallorca, Tierra del Fuego; not Alaska or Hawaii).
INSET_PART_MAX_AREA = 0.10
INSET_PART_MAX_DIST = 0.15


def _sam2_note(versions: dict, resmoothed: bool | str = False) -> str:
    """SAM 2 refinement line (refined files of examples 13 and 15 have no sidecar).

    *resmoothed*: True when the example smooths and simplifies every polygon again after
    SAM (example 13), so the fields SAM did not prompt are not byte-identical; a string
    names the polygons it smooths again (example 15's split variant).
    """
    which = resmoothed if isinstance(resmoothed, str) else "all polygons"
    after = (
        "fields whose padded box is under 64 px on a side are not changed by SAM; the example "
        f"then smooths and simplifies {which} again"
        if resmoothed
        else "fields whose padded box is under 64 px on a side are left unchanged"
    )
    return (
        f"SAM 2: facebook/sam2-hiera-large (sam2 {versions.get('sam2', '?')} via "
        f"segment-geospatial {versions.get('segment-geospatial', '?')}), one box prompt per "
        f"field; {after}"
    )


SOURCE_LABELS = {
    "sentinel2": "Sentinel-2 L2A",
    "landsat": "Landsat C2 L2",  # with the recorded missions: see _landsat_label
    "hls": "HLS",
    "naip": "NAIP",
    "usgs-naip-plus": "USGS NAIP Plus",
    "spot": "SPOT 6/7",
    "spot-pan": "SPOT 6/7 panchromatic",
    "local": "local GeoTIFF",
}
LANDSAT_MISSIONS = {"LT04": 4, "LT05": 5, "LE07": 7, "LC08": 8, "LC09": 9}


def _landsat_label(collections: str | None) -> str:
    """Landsat label from a composite's ``AGRIBOUND_COLLECTIONS`` tag.

    ``"LANDSAT/LE07/C02/T1_L2,LANDSAT/LC08/C02/T1_L2"`` gives ``"Landsat 7/8 C2 L2"``;
    without recognisable Landsat collections the label is ``"Landsat C2 L2"``.
    """
    ids = [c.strip().split("/")[1] for c in (collections or "").split(",") if c.count("/") >= 2]
    missions = sorted({LANDSAT_MISSIONS[i] for i in ids if i in LANDSAT_MISSIONS})
    if not missions:
        return SOURCE_LABELS["landsat"]
    return f"Landsat {'/'.join(str(m) for m in missions)} C2 L2"


@dataclass
class Layer:
    """One panel: an output file and its label."""

    path: str
    label: str
    highlight: str | None = None  # boolean column; True rows are outlined in HIGHLIGHT_COLOR
    # Output of the same run with the crop filter: this layer's polygons missing from it
    # (matched by representative point) are outlined in REMOVED_COLOR.
    removed_vs: str | None = None
    model_from: str | None = None  # output whose sidecar names the model (for files without one)
    background: str | None = None  # per-panel background GeoTIFF glob (per_layer_background)
    background_source: str | None = None
    bg_role: str | None = None
    crop_m: float | None = None  # multi-area entries: this panel's window side (else the entry's)


@dataclass
class Entry:
    """One gallery image."""

    key: str
    name: str
    title: str
    layers: list[Layer]
    reference: str | None = None  # drawn in cyan on every panel
    reference_label: str = "Reference"
    background: str | None = None  # glob of GeoTIFFs; default: the layer's provenance raster
    background_source: str | None = None  # source name when the background is not a layer's raster
    background_from: str | None = None  # take the background raster from this output's provenance
    zoom_m: float | None = None  # add a zoom panel of this width (metres) on the densest area
    crop_m: float | None = None  # crop every panel to the densest square of this width (metres)
    crop_layer: int = 0  # layer whose polygons define the densest square
    crop_on_reference: bool = False  # the densest square of the reference polygons instead
    # Crop every panel to this square instead: (centre x, centre y, side in metres), in the
    # CRS of the background raster (the map CRS).
    window: tuple[float, float, float] | None = None
    # Squares outlined and numbered on every layer panel (not a zoom panel): (label, centre x,
    # centre y, side in metres), in the map CRS, e.g. the windows of the zoomed entries that go
    # with an overview.
    mark_windows: list[tuple[str, float, float, float]] = field(default_factory=list)
    ftw_window: str | None = "a"  # FTW outputs: draw this season window ("a" or "b")
    ncols: int | None = None  # panels per row (default: all in one row)
    bg_role: str | None = None  # how the background relates to the engine input (for the note)
    per_layer_background: bool = False  # each panel on its own layer's imagery
    # Each layer is its own study area: its own CRS, imagery and densest crop_m window, and a
    # world locator with numbered dots instead of the country inset (render_areas).
    multi_area: bool = False
    crop_on: str | None = None  # boolean column, or "circular": count only those rows of that layer
    metrics: str | None = None  # glob of a metrics JSON to copy into gallery_stats.json
    overlay: str | None = (
        None  # extra outline layer, "path" or "path::layer", drawn dashed in yellow
    )
    overlay_label: str = ""
    model_notes: list[str] = field(default_factory=list)  # extra footer lines
    sam2_note: bool = False  # add the SAM 2 model line (refined files have no sidecar)
    sam2_resmoothed: bool | str = (
        False  # True: re-smooths all polygons after SAM (13); str: which (15)
    )
    halo: bool = False  # white halo under the outlines, for small fields on busy imagery
    inset: bool = True  # locator inset (country, state/province, study-area marker)
    line_width: float = 0.7
    notes: dict = field(default_factory=dict)


# Example 15 zoom windows (label, centre x, centre y, side in m; EPSG:32720, the map CRS):
# 1 = the 4 km square with the most centre pivots (the densest circular-polygon window of the
# 1.0.0 gallery; 14 of 29 hand-checked pivots), 2 = a 4 km square around the south-east pivot
# group (8 pivots), 3 = the 4 km square (inside the study area, clear of 1 and 2; 200 m grid) with
# the largest summed TESSERA and Google share of its area in crop-filter polygons over 200 ha, on
# the 1.0.1 layers (TESSERA 53 % in 4 polygons of 246-414 ha holding 6-12 Delineate-Anything
# fields each; Google 77 %; 2026-09-29). The 1.0.0 window, chosen by the same share over 500 ha,
# was 734400, 6249600.
PAMPAS_WINDOWS: list[tuple[str, float, float, float]] = [
    ("1", 740072.0, 6246718.0, 4000.0),
    ("2", 748600.0, 6242300.0, 4000.0),
    ("3", 747200.0, 6246400.0, 4000.0),
]
PAMPAS_SPLIT_NOTE = (
    "SAM 2 variant: polygons over 50 ha kept as the crop filter left them (one box prompt returns "
    "one object, so a polygon covering several fields would lose the rest), and refined masks "
    "trimmed where they overlap them; these crop layers have no multi-part polygons to split"
)
PAMPAS_RESMOOTHED = "the polygons of 50 ha or less"
PAMPAS_SAM2_LAYERS = [
    Layer(
        "outputs/pampas_semi_supervised/fields_google_crop_2024.gpkg",
        "Google embedding clusters, crop filter",
        model_from="outputs/pampas_semi_supervised/fields_google-embedding_embedding_2024_all.gpkg",
    ),
    Layer(
        "outputs/pampas_semi_supervised/fields_google_crop_sam2-s2-split_2024.gpkg",
        "Google + SAM 2 on Sentinel-2",
        highlight="agribound:sam_refined",
    ),
    Layer(
        "outputs/pampas_semi_supervised/fields_tessera_crop_2024.gpkg",
        "TESSERA clusters, crop filter",
        model_from="outputs/pampas_semi_supervised/fields_tessera-embedding_embedding_2024_all.gpkg",
    ),
    Layer(
        "outputs/pampas_semi_supervised/fields_tessera_crop_sam2-s2-split_2024.gpkg",
        "TESSERA + SAM 2 on Sentinel-2",
        highlight="agribound:sam_refined",
    ),
]

# Paths are relative to the run root (outputs/examples_1.0). See its PLAN.md.
ENTRIES: list[Entry] = [
    Entry(
        "14",
        "NM_example",
        "Lea County, NM: DINOv3 (fine-tuned) + SAM 2 from 30 m to 1 m",
        [
            Layer(
                "outputs/lea_county_dinov3_sam2/fields_landsat_dinov3-sam2_2022.gpkg",
                "Landsat 30 m",
                highlight="agribound:sam_refined",
            ),
            Layer(
                "outputs/lea_county_dinov3_sam2/fields_sentinel2_dinov3-sam2_2022.gpkg",
                "Sentinel-2 10 m",
                highlight="agribound:sam_refined",
            ),
            Layer(
                "outputs/lea_county_dinov3_sam2/fields_spot_dinov3-sam2_2022.gpkg",
                "SPOT 6/7 6 m",
                highlight="agribound:sam_refined",
            ),
            Layer(
                "outputs/lea_county_dinov3_sam2/fields_naip_dinov3-sam2_2022.gpkg",
                "NAIP 1 m",
                highlight="agribound:sam_refined",
            ),
        ],
        reference="outputs/lea_county_dinov3_sam2/reference.gpkg",
        reference_label="NMOSE reference (the training labels)",
        per_layer_background=True,
        crop_m=6000,
        crop_on_reference=True,  # the window with the most reference fields
        ncols=2,
        sam2_note=True,
    ),
    Entry(
        "12n",
        "NM_NAIP_models_example",
        "Lea County, NM: Delineate-Anything v2, GeoAI and DINOv3 on NAIP 1 m",
        [
            Layer(
                "outputs/lea_county_ensemble/fields_naip_delineate-anything_large_v2_pretrained_2022.gpkg",
                "DA v2 as released",
            ),
            Layer(
                "outputs/lea_county_ensemble/fields_naip_delineate-anything_large_v2_2022.gpkg",
                "DA v2 fine-tuned",
            ),
            Layer("outputs/lea_county_ensemble/fields_naip_geoai_2022.gpkg", "GeoAI fine-tuned"),
            Layer("outputs/lea_county_ensemble/fields_naip_dinov3_2022.gpkg", "DINOv3 fine-tuned"),
        ],
        reference="outputs/lea_county_ensemble/lea_county_reference.gpkg",
        reference_label="NMOSE reference (the fine-tuning labels)",
        crop_m=6000,
        crop_on_reference=True,  # the window with the most reference fields (as example 14)
        ncols=2,
    ),
    Entry(
        "15",
        "Pampas_example",
        "Pampas: embeddings + SAM 2 vs Delineate-Anything v2 on Sentinel-2 and SPOT",
        [
            Layer(
                "outputs/pampas_semi_supervised/fields_google_crop_sam2-s2-split_2024.gpkg",
                "Google embedding + SAM 2",
                highlight="agribound:sam_refined",
                model_from="outputs/pampas_semi_supervised/fields_google-embedding_embedding_2024_all.gpkg",
                background="outputs/pampas_semi_supervised/.agribound_cache/sentinel2_composite_*.tif",
                background_source="sentinel2",
                bg_role="the composite SAM 2 read (fields from embedding clusters)",
            ),
            Layer(
                "outputs/pampas_semi_supervised/fields_tessera_crop_sam2-s2-split_2024.gpkg",
                "TESSERA + SAM 2",
                highlight="agribound:sam_refined",
                model_from="outputs/pampas_semi_supervised/fields_tessera-embedding_embedding_2024_all.gpkg",
                background="outputs/pampas_semi_supervised/.agribound_cache/sentinel2_composite_*.tif",
                background_source="sentinel2",
                bg_role="the composite SAM 2 read (fields from embedding clusters)",
            ),
            Layer(
                "outputs/pampas_semi_supervised/fields_sentinel2_delineate-anything_2024.gpkg",
                "Delineate-Anything v2, Sentinel-2 10 m",
            ),
            Layer(
                "outputs/pampas_semi_supervised/fields_spot_delineate-anything_2023.gpkg",
                "Delineate-Anything v2, SPOT 6 m (2023)",
            ),
        ],
        per_layer_background=True,
        window=PAMPAS_WINDOWS[0][1:],  # the pivot window of 15c
        mark_windows=[PAMPAS_WINDOWS[0]],  # numbered as its square on 15b
        ncols=2,
        model_notes=[PAMPAS_SPLIT_NOTE],
        sam2_note=True,
        sam2_resmoothed=PAMPAS_RESMOOTHED,
    ),
    Entry(
        "15b",
        "Pampas_SAM2_example",
        "Pampas: Google Satellite Embedding and TESSERA, whole study area",
        PAMPAS_SAM2_LAYERS,
        # The crop and SAM files have no provenance sidecar; SAM 2 read this composite.
        background="outputs/pampas_semi_supervised/.agribound_cache/sentinel2_composite_*.tif",
        background_source="sentinel2",
        bg_role="the composite SAM 2 read (the fields come from embedding clusters)",
        mark_windows=PAMPAS_WINDOWS,
        ncols=2,
        line_width=0.45,
        model_notes=[PAMPAS_SPLIT_NOTE],
        sam2_note=True,
        sam2_resmoothed=PAMPAS_RESMOOTHED,
    ),
    *[
        Entry(
            key,
            name,
            f"Pampas: Google Satellite Embedding and TESSERA, zoom {label} ({title})",
            PAMPAS_SAM2_LAYERS,
            background="outputs/pampas_semi_supervised/.agribound_cache/sentinel2_composite_*.tif",
            background_source="sentinel2",
            bg_role="the composite SAM 2 read (the fields come from embedding clusters)",
            window=(x, y, side),
            mark_windows=[(label, x, y, side)],  # numbered as its square on 15b
            ncols=2,
            model_notes=[PAMPAS_SPLIT_NOTE],
            sam2_note=True,
            sam2_resmoothed=PAMPAS_RESMOOTHED,
        )
        for (label, x, y, side), (key, name, title) in zip(
            PAMPAS_WINDOWS,
            [
                ("15c", "Pampas_SAM2_zoom1_example", "centre pivots"),
                ("15d", "Pampas_SAM2_zoom2_example", "centre pivots, south-east"),
                ("15e", "Pampas_SAM2_zoom3_example", "large merged polygons"),
            ],
            strict=True,
        )
    ],
    Entry(
        "22",
        "Global_South_SPOT_Pan_example",
        "Global South: Delineate-Anything v2 on SPOT 6/7 panchromatic 1.5 m, six landscapes",
        [
            Layer(
                f"outputs/global_south_spot_pan/{slug}/fields_spot-pan_delineate-anything_{year}.gpkg",
                label,
                crop_m=crop,
            )
            for slug, year, label, crop in [
                ("cauvery_delta", 2018, "Cauvery Delta, India", None),
                ("hetao", 2021, "Hetao, China", None),
                ("mendoza", 2019, "Mendoza, Argentina", None),
                ("mwea", 2020, "Mwea, Kenya", None),
                ("nile_delta", 2020, "Nile Delta, Egypt", None),
                (
                    "western_bahia",
                    2018,
                    "Western Bahia, Brazil",
                    6000,
                ),  # pivots ~1 km: the whole area
            ]
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=2000,  # per panel: the densest of the 2 km squares on a 1 km grid
        ncols=3,
    ),
    Entry(
        "02",
        "India_example",
        "West Bengal: FTW on Sentinel-2 vs Delineate-Anything on SPOT-Pan",
        [
            Layer(
                "outputs/india_nadia/fields_sentinel2_ftw_2024.gpkg", "FTW, Sentinel-2 10 m, 2024"
            ),
            Layer(
                "outputs/india_nadia/fields_spot-pan_delineate-anything_2020.gpkg",
                "Delineate-Anything, SPOT-Pan 1.5 m, 2020",
            ),
        ],
        per_layer_background=True,
        crop_m=1000,
        crop_layer=1,
        line_width=0.6,
    ),
    Entry(
        "04",
        "France_example",
        "Beauce: FTW on Sentinel-2",
        [Layer("outputs/france_beauce/fields_sentinel2_ftw_2023.gpkg", "FTW")],
        zoom_m=3000,
    ),
    Entry(
        "06",
        "Kenya_example",
        "Kakamega: FTW on Sentinel-2",
        [
            Layer(f"outputs/kenya_smallholder/fields_sentinel2_ftw_2023_minarea{a}.gpkg", lab)
            for a, lab in (
                (100, "min area 100 m²"),
                (500, "min area 500 m²"),
                (1000, "min area 1,000 m²"),
                (2500, "min area 2,500 m²"),
            )
        ],
        crop_m=1000,
        ncols=2,
        line_width=1.1,
        halo=True,
    ),
    Entry(
        "07",
        "Central_Valley_example",
        "Central Valley: Delineate-Anything on NAIP",
        [
            Layer(
                "outputs/usa_central_valley/fields_naip_delineate-anything_2022.gpkg",
                "Delineate-Anything",
            )
        ],
        zoom_m=2000,
    ),
    Entry(
        "03",
        "Australia_example",
        "Murray-Darling: Prithvi on HLS vs Delineate-Anything v2 on SPOT",
        [
            Layer(
                "outputs/australia_murray_darling/fields_hls_prithvi-embed_2022.gpkg",
                "Prithvi embed, HLS 30 m",
            ),
            Layer(
                "outputs/australia_murray_darling/fields_hls_prithvi-pca_2022.gpkg",
                "PCA baseline, HLS 30 m",
            ),
            Layer(
                "outputs/australia_murray_darling/fields_spot_delineate-anything_2023.gpkg",
                "DA v2, SPOT 6 m (2023)",
            ),
        ],
        per_layer_background=True,
    ),
    Entry(
        "08",
        "China_example",
        "North China Plain: Delineate-Anything on SPOT",
        [
            Layer(
                "outputs/china_north_plain/fields_spot_delineate-anything_2023.gpkg",
                "Delineate-Anything",
            )
        ],
        crop_m=4000,  # the 4 km square with the most polygons, so the outlines separate
        zoom_m=1500,
        line_width=0.5,
    ),
    Entry(
        "09",
        "Spain_example",
        "Andalusia: ensemble vote merge",
        [
            Layer(
                "outputs/ensemble_comparison/fields_sentinel2_delineate-anything_2024.gpkg",
                "Delineate-Anything",
            ),
            Layer("outputs/ensemble_comparison/fields_sentinel2_ftw_2024.gpkg", "FTW"),
            Layer(
                "outputs/ensemble_comparison/fields_sentinel2_merge-vote_2024.gpkg",
                "Vote merge, 2 of 2",
            ),
        ],
        background_from="outputs/ensemble_comparison/fields_sentinel2_delineate-anything_2024.gpkg",
        bg_role="Delineate-Anything's input (FTW read two season windows)",
        crop_m=3000,
    ),
    Entry(
        "11",
        "MAP_example",
        "Mississippi Alluvial Plain: Delineate-Anything on SPOT",
        [
            Layer(
                "outputs/mississippi_alluvial_plain/fields_spot_delineate-anything_2021.gpkg",
                "2021",
            ),
            Layer(
                "outputs/mississippi_alluvial_plain/fields_spot_delineate-anything_2022.gpkg",
                "2022",
            ),
            Layer(
                "outputs/mississippi_alluvial_plain/fields_spot_delineate-anything_2023.gpkg",
                "2023",
            ),
        ],
        per_layer_background=True,
        crop_m=4000,
        crop_layer=2,
    ),
    Entry(
        "20",
        "San_Juan_evaluation_example",
        "San Juan County, NM: Delineate-Anything vs NMOSE",
        [
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2019.gpkg",
                "Delineate-Anything",
            )
        ],
        reference="examples/NMOSE Field Boundaries/WUCB ag polys.shp",
        reference_label="NMOSE reference (not used for training in these runs)",
        zoom_m=2500,
        metrics="outputs/stratified_evaluation/metrics.json",
    ),
    Entry(
        "20b",
        "San_Juan_resolution_example",
        "San Juan County, NM: Delineate-Anything v2 from 30 m to 1 m (2018)",
        [
            Layer(
                "outputs/stratified_evaluation/fields_landsat_delineate-anything_2018.gpkg",
                "Landsat 30 m",
            ),
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2018.gpkg",
                "Sentinel-2 10 m",
            ),
            Layer(
                "outputs/stratified_evaluation/fields_spot_delineate-anything_2018.gpkg",
                "SPOT 6/7 6 m",
            ),
            Layer(
                "outputs/stratified_evaluation/fields_naip_delineate-anything_2018.gpkg",
                "NAIP 1 m",
            ),
        ],
        reference="examples/NMOSE Field Boundaries/WUCB ag polys.shp",
        reference_label="NMOSE reference (not used for training in these runs)",
        per_layer_background=True,
        crop_m=2500,
        crop_on_reference=True,  # the window with the most reference fields
        ncols=2,
        metrics="outputs/stratified_evaluation/metrics_naip_2018.json",
    ),
    Entry(
        "20c",
        "San_Juan_crop_filter_example",
        "San Juan County, NM: FTW and Delineate-Anything v2 with and without the crop filter",
        [
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_ftw_2019.gpkg",
                "FTW, crop filter on",
            ),
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_ftw_2019_nolulc.gpkg",
                "FTW, crop filter off",
                removed_vs="outputs/stratified_evaluation/fields_sentinel2_ftw_2019.gpkg",
            ),
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2019.gpkg",
                "DA v2, crop filter on",
            ),
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2019_nolulc.gpkg",
                "DA v2, crop filter off",
                removed_vs="outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2019.gpkg",
            ),
        ],
        reference="examples/NMOSE Field Boundaries/WUCB ag polys.shp",
        reference_label="NMOSE reference (not used for training in these runs)",
        per_layer_background=True,
        crop_m=3000,
        crop_on_reference=True,  # the square with the most reference fields
        ncols=2,
    ),
    Entry(
        "13",
        "SAM2_refinement_example",
        "SAM 2 refinement of example 20's output",
        [
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2019.gpkg",
                "Delineate-Anything",
            ),
            Layer(
                "outputs/sam2_refinement/fields_sentinel2_delineate-anything_2019_sam2.gpkg",
                "After SAM 2",
                highlight="agribound:sam_refined",
            ),
        ],
        reference="examples/NMOSE Field Boundaries/WUCB ag polys.shp",
        reference_label="NMOSE reference (not used for training in these runs)",
        background_from="outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2019.gpkg",
        crop_m=3000,
        crop_layer=1,
        crop_on="agribound:sam_refined",
        sam2_note=True,
        sam2_resmoothed=True,
    ),
    Entry(
        "16",
        "USGS_NAIP_Plus_example",
        "Central Valley: Delineate-Anything on USGS NAIP Plus",
        [
            Layer(
                "outputs/usa_central_valley_usgs_naip_plus/fields_*_2022.gpkg", "Delineate-Anything"
            )
        ],
        zoom_m=1500,
    ),
    Entry(
        "19",
        "HPC_tiling_example",
        "HPC tiling: four tiles merged",
        [
            Layer(
                "outputs/hpc_tiling_demo/sentinel2_delineate-anything_2024_lulc/fields_merged.gpkg",
                "Merged",
            )
        ],
        background="outputs/hpc_tiling_demo/sentinel2_delineate-anything_2024_lulc/tiles/*/cache/sentinel2_composite_*.tif",
        background_source="sentinel2",
        overlay="outputs/hpc_tiling_demo/sentinel2_delineate-anything_2024_lulc/tiles.gpkg::tiles",
        overlay_label="Tile cores",
    ),
]


# --------------------------------------------------------------------------- helpers


def _one(pattern: str, root: Path) -> Path:
    matches = sorted(glob.glob(str(root / pattern)))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{pattern!r}: expected exactly one match under {root}, found {len(matches)}"
        )
    return Path(matches[0])


def _sidecar(path: Path) -> dict:
    side = path.with_name(path.name + ".provenance.json")
    if not side.exists():
        return {}
    return json.loads(side.read_text())


def _rgb_indices(src: rasterio.DatasetReader, source: str | None) -> tuple[list[int], bool]:
    """1-based band indices for R, G, B and whether the image is grey."""
    desc = list(src.descriptions)
    canon = (SOURCE_REGISTRY.get(source or "", {}) or {}).get("canonical_bands") or {}
    if canon and all(canon.get(k) in desc for k in ("R", "G", "B")):
        idx = [desc.index(canon[k]) + 1 for k in ("R", "G", "B")]
        return idx, len(set(idx)) == 1
    if src.count >= 3:
        return [1, 2, 3], False
    return [1, 1, 1], True


def _stretch_with(arr: np.ndarray, lows, highs) -> np.ndarray:
    """Apply percentile bounds from another read (same mapping as percentile_stretch_uint8)."""
    out = np.zeros(arr.shape, dtype=np.uint8)
    valid = np.all(np.isfinite(arr), axis=0)
    for b in range(arr.shape[0]):
        lo, hi = float(lows[b]), float(highs[b])
        v = np.clip(255.0 * (arr[b] - lo) / max(hi - lo, 1e-12), 0, 255)
        out[b] = np.where(valid, v, 0).astype(np.uint8)
    return out


def _read_background(tifs: list[Path], source: str | None, bounds, crs, out_px, stretch=None):
    """Read an RGB uint8 image of *tifs* inside *bounds* (in *crs*).

    The window is read at the panel's pixel size *out_px* ``(width, height)``,
    averaged down from the native resolution, and never upsampled (matplotlib
    enlarges coarse imagery with nearest-neighbour, so pixels stay visible
    rather than blurred). *stretch* ``(lows, highs, grey)`` reuses the
    percentile bounds of another read, so a zoom panel keeps the colours of its
    parent panel. Returns ``(rgb, extent, stretch)``.
    """
    srcs = [rasterio.open(p) for p in tifs]
    memfile = None
    try:
        if len(srcs) > 1:
            mosaic, transform = rio_merge(
                srcs, bounds=None, nodata=np.nan if srcs[0].dtypes[0].startswith("float") else None
            )
            profile = srcs[0].profile.copy()
            profile.update(
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=transform,
                count=mosaic.shape[0],
            )
            memfile = rasterio.io.MemoryFile()
            ds = memfile.open(**profile)
            ds.write(mosaic)
            ds.descriptions = srcs[0].descriptions
            src = ds
        else:
            src = srcs[0]
        if src.crs != crs:
            raise ValueError(f"background CRS {src.crs} differs from plot CRS {crs}")
        idx, grey = _rgb_indices(src, source)
        x0, y0, x1, y1 = bounds
        sx0, sy0, sx1, sy1 = src.bounds
        x0, y0, x1, y1 = max(x0, sx0), max(y0, sy0), min(x1, sx1), min(y1, sy1)
        # A window that covers the bounds (start rounded down, end rounded up): rounding
        # the length instead could stop a pixel short and leave a white strip in the panel.
        # The overhang of less than one pixel is cut off by the axis limits.
        w = from_bounds(x0, y0, x1, y1, src.transform)
        c0, r0 = math.floor(w.col_off + 1e-6), math.floor(w.row_off + 1e-6)
        c1 = math.ceil(w.col_off + w.width - 1e-6)
        r1 = math.ceil(w.row_off + w.height - 1e-6)
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        want_w = min(max(1, int(out_px[0])), MAX_PANEL_READ_SIDE)
        want_h = min(max(1, int(out_px[1])), MAX_PANEL_READ_SIDE)
        scale = max(win.width / want_w, win.height / want_h, 1.0)
        out_h = max(1, int(round(win.height / scale)))
        out_w = max(1, int(round(win.width / scale)))
        arr = src.read(
            idx,
            window=win,
            out_shape=(3, out_h, out_w),
            resampling=Resampling.average if scale > 1 else Resampling.nearest,
            masked=False,
        )
        wb = rasterio.windows.bounds(win, src.transform)
        arr = arr.astype("float32")
        if src.nodata is not None and not (
            isinstance(src.nodata, float) and math.isnan(src.nodata)
        ):
            arr[:, np.all(arr == src.nodata, axis=0)] = np.nan
        if stretch is None:
            # arr is float32 here, so 8-bit imagery (NAIP, SPOT DN) is stretched as well.
            # SPOT composites are raw DN with band-specific ranges (no surface reflectance),
            # so their bands are stretched separately; reflectance and 8-bit aerial imagery
            # share one stretch to keep their colour balance.
            per_band = (source or "").startswith("spot") and not grey
            _, lows, highs = percentile_stretch_uint8(
                arr, low=STRETCH_LOW, high=STRETCH_HIGH, per_band=per_band, return_bounds=True
            )
            stretch = (lows, highs, grey)
        rgb = _stretch_with(arr, stretch[0], stretch[1])
        return np.moveaxis(rgb, 0, -1), (wb[0], wb[2], wb[1], wb[3]), stretch
    finally:
        for s in srcs:
            s.close()
        if memfile is not None:
            memfile.close()


def _model_note(prov: dict) -> str | None:
    """One line naming the model and its version, from a provenance sidecar."""
    note = _model_note_base(prov)
    cfg = prov.get("config") or prov.get("base_config") or {}
    if note and cfg.get("fine_tune"):
        em = prov.get("engine_meta") or {}
        how = ""
        if "use_lora" in em:
            how = "LoRA, " if em.get("use_lora") else "full, "
        note += (
            f", fine-tuned on the reference polygons ({how}up to "
            f"{cfg.get('fine_tune_epochs')} epochs, {cfg.get('fine_tune_split', 'block')} split)"
        )
    return note


def _model_note_base(prov: dict) -> str | None:
    cfg = prov.get("config") or prov.get("base_config") or {}
    em = prov.get("engine_meta") or {}
    ver = prov.get("versions") or {}
    engine = cfg.get("engine")
    if engine == "delineate-anything" and em.get("weights_filename"):
        # Generation from the weights file name (DelineateAnythingv<N>.pt), not hard-coded.
        m = re.search(r"v(\d+)\.pt$", str(em.get("weights_filename")))
        gen = f" v{m.group(1)}" if m else ""
        rev = str(em.get("weights_revision") or "")[:7]
        return (
            f"Delineate-Anything{gen}, {em.get('model_key')} ({em.get('model_architecture')}, "
            f"{em.get('weights_repo')}/{em.get('weights_filename')} @ {rev}; "
            f"ultralytics {em.get('ultralytics_version') or ver.get('ultralytics')})"
        )
    if engine == "delineate-anything" and em.get("weights") == "checkpoint":
        sha = str(em.get("checkpoint_sha256") or "")[:7]
        return (
            f"Delineate-Anything {em.get('model_key')} checkpoint (sha256 {sha}; "
            f"ultralytics {em.get('ultralytics_version') or ver.get('ultralytics')})"
        )
    if engine == "ftw" and em.get("model"):
        return (
            f"FTW model {em.get('model')} (ftw-baselines {em.get('model_version')} checkpoint; "
            f"ftw-tools {em.get('ftw_tools_version') or ver.get('ftw-tools')})"
        )
    if engine == "prithvi":
        if em.get("weights_repo"):
            rev = str(em.get("weights_revision") or "")[:7]
            return (
                f"Prithvi {em.get('mode')} mode: {em.get('weights_repo')} @ {rev} "
                f"(terratorch {em.get('terratorch_version') or ver.get('terratorch')})"
            )
        return f"Prithvi {em.get('mode')} mode: K-means on PCA of the bands (no ViT weights)"
    if engine == "embedding":
        src = em.get("source") or cfg.get("source")
        if src == "tessera-embedding":
            return (
                f"TESSERA {em.get('tessera_version')} embeddings (geotessera "
                f"{ver.get('geotessera')}), {em.get('clusterer')} with "
                f"{em.get('n_clusters')} clusters"
            )
        coll = ((prov.get("facts") or {}).get("composite") or {}).get("AGRIBOUND_COLLECTIONS")
        if src == "google-embedding":
            return (
                f"Google Satellite Embedding ({coll or 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'}, "
                f"{em.get('n_bands')}-D), {em.get('clusterer')} with "
                f"{em.get('n_clusters')} clusters"
            )
        return f"{src} embeddings, {em.get('clusterer')} with {em.get('n_clusters')} clusters"
    if engine == "dinov3":
        rev = str(em.get("weights_revision") or "")[:7]
        return (
            f"DINOv3 {em.get('model_name')} with {em.get('weights')} @ {rev} "
            f"(geoai-py {em.get('geoai-py_version') or ver.get('geoai-py')})"
        )
    if engine == "geoai":
        win = (
            f"; {em['window_size']} px windows, overlap {em.get('overlap')} px"
            if em.get("window_size")
            else ""
        )
        return (
            "GeoAI Mask R-CNN ResNet50-FPN "
            f"(geoai-py {em.get('geoai-py_version') or ver.get('geoai-py')}{win})"
        )
    if engine and em.get("model"):
        return f"{engine}: {em.get('model')}"
    return None


def _nice_length(width_m: float) -> float:
    target = width_m / 5
    exp = 10 ** math.floor(math.log10(target))
    for m in (5, 2, 1):
        if m * exp <= target:
            return m * exp
    return exp


def _scalebar(ax, extent, crs):
    x0, x1, y0, y1 = extent
    if not crs.is_projected:
        return
    length = _nice_length(x1 - x0)
    pad_x = (x1 - x0) * 0.04
    pad_y = (y1 - y0) * 0.05
    bx, by = x0 + pad_x, y0 + pad_y
    h = (y1 - y0) * 0.008
    ax.add_patch(
        Rectangle((bx, by), length, h, facecolor="white", edgecolor="black", lw=0.6, zorder=6)
    )
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    ax.text(
        bx + length / 2,
        by + h * 2.2,
        label,
        ha="center",
        va="bottom",
        fontsize=SCALE_FS,
        color="white",
        zorder=6,
        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
    )


def _densest_window(gdf: gpd.GeoDataFrame, size: float, extent):
    """Square window of side *size* with the most polygon centroids (deterministic)."""
    x0, x1, y0, y1 = extent
    c = gdf.geometry.representative_point()
    xs, ys = c.x.to_numpy(), c.y.to_numpy()
    best, best_n = None, -1
    step = size / 2
    for cx in np.arange(x0 + size / 2, x1 - size / 2 + 1e-6, step):
        for cy in np.arange(y0 + size / 2, y1 - size / 2 + 1e-6, step):
            n = int(np.count_nonzero((np.abs(xs - cx) <= size / 2) & (np.abs(ys - cy) <= size / 2)))
            if n > best_n:
                best, best_n = (cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2), n
    return best


def _is_circular(gdf: gpd.GeoDataFrame, min_fill: float = 0.8, min_ha: float = 20.0):
    """Centre-pivot-like polygons: >= *min_fill* of their minimum bounding circle, >= *min_ha*."""
    import shapely

    g = gdf if gdf.crs and gdf.crs.is_projected else gdf.to_crs(gdf.estimate_utm_crs())
    area = g.geometry.area.to_numpy()
    radius = shapely.minimum_bounding_radius(g.geometry.to_numpy())
    fill = np.divide(area, np.pi * radius**2, out=np.zeros_like(area), where=radius > 0)
    return (fill >= min_fill) & (area >= min_ha * 1e4)


def _bool_column(values) -> np.ndarray:
    """A flag column as a boolean array; missing values are False.

    GeoPackage has no boolean type for a column that also holds missing values, so a
    flag written through pandas' object dtype comes back as the text "True"/"False";
    ``astype(bool)`` would make both True.
    """
    import pandas as pd

    def one(v) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return False if pd.isna(v) else bool(v)

    return np.array([one(v) for v in values], dtype=bool)


def _is_local_path(value) -> bool:
    """True for absolute filesystem paths (kept out of the committed stats file)."""
    return isinstance(value, str) and (value.startswith(("/", "~")) or value[1:3] == ":\\")


def _area_stats(gdf: gpd.GeoDataFrame) -> dict:
    g = gdf if gdf.crs and gdf.crs.is_projected else gdf.to_crs(gdf.estimate_utm_crs())
    a = g.geometry.area.to_numpy() / 1e4
    if a.size == 0:
        return {"n_polygons": 0}
    q = np.quantile(a, [0.1, 0.5, 0.9])
    return {
        "n_polygons": int(a.size),
        "area_ha_total": round(float(a.sum()), 1),
        "area_ha_p10": round(float(q[0]), 2),
        "area_ha_median": round(float(q[1]), 2),
        "area_ha_p90": round(float(q[2]), 2),
    }


def _composite_facts(prov: dict, tif: Path | None = None) -> dict:
    comp = (prov.get("facts") or {}).get("composite") or {}
    if not comp and tif is not None and tif.exists():
        with rasterio.open(tif) as s:
            comp = s.tags()
    keep = {
        "AGRIBOUND_COLLECTIONS": "collections",
        "AGRIBOUND_COMPOSITE_METHOD": "composite_method",
        "AGRIBOUND_DATE_START": "date_start",
        "AGRIBOUND_DATE_END_EXCLUSIVE": "date_end_exclusive",
        "AGRIBOUND_N_IMAGES": "n_images",
        "AGRIBOUND_EXPORT_CRS": "crs",
        "AGRIBOUND_RESOLUTION_M": "resolution_m",
        "AGRIBOUND_IMAGE_YEARS": "image_years",
        "AGRIBOUND_LOCK_RASTER_IDS": "lock_raster_ids",
    }
    return {v: comp[k] for k, v in keep.items() if k in comp}


def _background_for(entry: Entry, root: Path, prov: dict, basemap: str, layer: Layer | None = None):
    """Background GeoTIFF(s), source, provenance for its dates and role note of one layer."""
    if layer is not None and layer.background:
        tifs = [Path(t) for t in sorted(glob.glob(str(root / layer.background)))]
        if not tifs:
            raise FileNotFoundError(f"{layer.background!r}: no background GeoTIFF under {root}")
        role = layer.bg_role or entry.bg_role or "the engine's input"
        return tifs, layer.background_source or entry.background_source, {}, role
    cfg = prov.get("config") or prov.get("base_config") or {}
    source = entry.background_source or cfg.get("source")
    if entry.background_from and not entry.per_layer_background:
        bprov = _sidecar(_one(entry.background_from, root))
        source = entry.background_source or (bprov.get("config") or {}).get("source") or source
        raster = (bprov.get("facts") or {}).get("raster_path")
        tifs = [root / raster] if raster else []
        prov = {**prov, "facts": {**(prov.get("facts") or {}), **(bprov.get("facts") or {})}}
    elif entry.background:
        tifs = [Path(t) for t in sorted(glob.glob(str(root / entry.background)))]
        if not tifs:
            raise FileNotFoundError(f"{entry.background!r}: no background GeoTIFF under {root}")
    else:
        raster = (prov.get("facts") or {}).get("raster_path")
        tifs = [root / raster] if raster else []
    bg_role = entry.bg_role or "the engine's input"
    comp_prov = prov
    wins = (prov.get("engine_meta") or {}).get("windows") or {}
    win = wins.get(entry.ftw_window) if entry.ftw_window else None
    if not entry.background and isinstance(win, dict) and win.get("raster"):
        # FTW reads two season-window composites, not the annual raster_path.
        tifs = [root / win["raster"]]
        comp_prov = {}  # the window's dates come from its own GeoTIFF tags
        bg_role = f"FTW window {entry.ftw_window.upper()}, one of FTW's two season inputs"
    if basemap == "composite" and not tifs:
        if entry.multi_area:  # render_areas has no Esri mode
            raise FileNotFoundError(
                f"entry {entry.key}: no composite recorded (no facts.raster_path in the layer's "
                "provenance sidecar); set Layer.background to that panel's GeoTIFF"
            )
        raise FileNotFoundError(f"entry {entry.key}: no composite recorded; pass --basemap esri")
    return tifs, source, comp_prov, bg_role


def _panel_names(n: int, ncols: int | None = None) -> list[str]:
    if n == 4 and ncols == 2:
        return ["Top left", "Top right", "Bottom left", "Bottom right"]
    if n == 2:
        return ["Left", "Right"]
    if n == 3:
        return ["Left", "Middle", "Right"]
    return [f"Panel {i + 1}" for i in range(n)]


def _bg_note(bg) -> str:
    tifs, source, comp_prov, bg_role = bg
    comp = _composite_facts(comp_prov, tifs[0] if tifs else None)
    label = SOURCE_LABELS.get(source or "", source or "imagery")
    if source == "landsat":
        label = _landsat_label(comp.get("collections"))
    if comp.get("resolution_m"):
        label += f" {float(comp['resolution_m']):g} m"
    window = ""
    if comp.get("date_start") and comp.get("date_end_exclusive"):
        window = (
            f" {comp.get('composite_method', '')} composite, "
            f"{comp['date_start']} to {comp['date_end_exclusive']} (end exclusive)"
        )
    elif comp.get("image_years"):
        window = f", {comp['image_years']} imagery"
    n_img = f", {comp['n_images']} images" if comp.get("n_images") else ""
    if not n_img and comp.get("lock_raster_ids"):
        n_img = f", {len(comp['lock_raster_ids'].split(','))} source rasters"
    return f"{label}{window}{n_img}; {bg_role}"


_NE_CACHE: dict = {}


def _natural_earth(layer: str) -> gpd.GeoDataFrame:
    """A Natural Earth layer ("countries", "states" or "lakes").

    Downloaded once into ``NATURAL_EARTH_DIR`` (``~/.cache/agribound/naturalearth``) and
    checked against the pinned SHA-256 on every load (a mismatch raises ValueError).
    """
    import hashlib
    import urllib.request

    if layer in _NE_CACHE:
        return _NE_CACHE[layer]
    url, sha = NATURAL_EARTH[layer]
    path = NATURAL_EARTH_DIR / Path(url).name
    if not path.exists():
        NATURAL_EARTH_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (fixed https URL)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha:
        raise ValueError(f"{path}: SHA-256 {digest} does not match the pinned {sha}")
    _NE_CACHE[layer] = gpd.read_file(f"zip://{path}")
    return _NE_CACHE[layer]


def _india_outline():
    """Survey of India outline of India (one MultiPolygon, EPSG:4326).

    Downloaded once into ``~/.cache/agribound/india`` and checked against the pinned
    SHA-256 of ``INDIA_OUTLINE`` on every load (a mismatch raises ValueError).
    """
    import hashlib
    import urllib.request

    if "india" in _NE_CACHE:
        return _NE_CACHE["india"]
    url, sha = INDIA_OUTLINE
    path = NATURAL_EARTH_DIR.parent / "india" / "india-soi.geojson"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (fixed https URL)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha:
        raise ValueError(f"{path}: SHA-256 {digest} does not match the pinned {sha}")
    _NE_CACHE["india"] = gpd.read_file(path).to_crs(4326).union_all()
    return _NE_CACHE["india"]


def _locate(lon: float, lat: float) -> dict:
    """Country and state/province (Natural Earth) at a point."""
    from shapely.geometry import Point

    pt = Point(lon, lat)
    countries, states = _natural_earth("countries"), _natural_earth("states")
    c = countries[countries.contains(pt)]
    if not len(c):  # on a coastline or border: nearest country
        c = countries.iloc[[int(countries.distance(pt).to_numpy().argmin())]]
    st = states[states.contains(pt)]
    return {
        "lon": round(lon, 4),
        "lat": round(lat, 4),
        "country": str(c.iloc[0]["ADMIN"]),
        "country_label": (
            str(c.iloc[0]["NAME"])
            if len(str(c.iloc[0]["NAME"])) <= 14
            else str(c.iloc[0]["ADM0_A3"])
        ),
        "country_a3": str(c.iloc[0]["ADM0_A3"]),
        "admin1": str(st.iloc[0]["name"]) if len(st) else None,
        "_country_geom": (
            _india_outline() if str(c.iloc[0]["ADM0_A3"]) == "IND" else c.iloc[0].geometry
        ),
        "_state_geom": st.iloc[0].geometry if len(st) else None,
    }


def _inset_parts(loc: dict) -> list:
    """Parts of the country that set the inset view.

    The part nearest the study-area dot, plus the other parts of the same country
    that are both small (area under ``INSET_PART_MAX_AREA`` of that part) and near
    (closer than ``INSET_PART_MAX_DIST`` of its larger bounding-box side), such as
    Tasmania, Corsica, Mallorca or Tierra del Fuego; far or large parts (Alaska,
    Hawaii, overseas France) stay out. India is always its whole outline.
    """
    from shapely.geometry import Point

    geom = loc["_country_geom"]
    if loc["country_a3"] == "IND":
        return [geom]
    pt = Point(loc["lon"], loc["lat"])
    parts = list(getattr(geom, "geoms", [geom]))
    main = min(parts, key=lambda g: g.distance(pt))
    x0, y0, x1, y1 = main.bounds
    side = max(x1 - x0, y1 - y0)
    near = [
        p
        for p in parts
        if p is not main
        and p.area < INSET_PART_MAX_AREA * main.area
        and p.distance(main) < INSET_PART_MAX_DIST * side
    ]
    return [main, *near]


def _inset_view(loc: dict):
    """(the whole country clipped to the view, view box) of the locator inset."""
    from shapely.geometry import box

    b = np.array([p.bounds for p in _inset_parts(loc)])
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    pad = 0.08 * max(x1 - x0, y1 - y0)
    view = box(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    geom = loc["_country_geom"]
    return (geom if loc["country_a3"] == "IND" else geom.intersection(view)), view


def _inset_shows_india(loc: dict) -> bool:
    return loc["country_a3"] == "IND" or _india_outline().intersects(_inset_view(loc)[1])


def _inset_aspect(loc: dict) -> float:
    """y/x scale of the inset map (degrees), as for a local equirectangular view."""
    return 1 / max(math.cos(math.radians(loc["lat"])), 0.2)


def _inset_box(loc: dict, view) -> tuple[float, float]:
    """(width, height) in inches of the inset map: the largest box shrunk to the view's aspect."""
    x0, y0, x1, y1 = view.bounds
    ratio = (y1 - y0) * _inset_aspect(loc) / (x1 - x0)
    if ratio > INSET_H_IN / INSET_W_IN:
        return INSET_H_IN / ratio, INSET_H_IN
    return INSET_W_IN, INSET_W_IN * ratio


def _inset_title(loc: dict, width_in: float) -> str:
    """Inset title: state and country on one line when it fits over the map, else two lines."""
    names = [x for x in (loc.get("admin1"), loc["country_label"]) if x]
    one = ", ".join(names)
    # The title may overhang the map by a little on each side, not into the image edge.
    if _text_width_in(one, INSET_TITLE_FS) <= width_in + 2 * (INSET_RIGHT_IN - 0.04):
        return one
    return ",\n".join(names)


def _draw_inset(fig, rect, loc: dict, title: str) -> tuple:
    """Locator inset in figure-fraction *rect*: country, states, the state, a marker.

    Returns ``(axes, india_drawn)``; *india_drawn* is True when India is drawn (so the
    footer credits the Survey of India outline). Every layer has an explicit zorder
    (``Z_*``): matplotlib draws by zorder, not by call order.
    """
    country, view = _inset_view(loc)  # e.g. metropolitan France with Corsica, the CONUS
    x0, y0, x1, y1 = view.bounds
    india = _india_outline()
    india_drawn = False

    iax = fig.add_axes(rect)
    countries, states = _natural_earth("countries"), _natural_earth("states")
    if loc["country_a3"] == "IND":
        # As in the grace-grb figures: no neighbouring country is drawn as a political entity.
        iax.set_facecolor("#f1f1f1")
        gpd.GeoSeries([india], crs=4326).plot(
            ax=iax, color="#fbfbfb", edgecolor="#555555", lw=0.5, zorder=Z_COUNTRY
        )
        india_drawn = True
    else:
        iax.set_facecolor(WATER_COLOR)  # sea
        others = countries[~countries["ADM0_A3"].isin(["IND", loc["country_a3"]])].clip(view)
        if len(others):
            others.plot(
                ax=iax, color="#eeeeee", edgecolor="#a9a9a9", linewidth=0.3, zorder=Z_NEIGHBOURS
            )
        gpd.GeoSeries([country], crs=4326).plot(
            ax=iax, color="#fbfbfb", edgecolor="#555555", lw=0.5, zorder=Z_COUNTRY
        )
        in_country = states[states["adm0_a3"] == loc["country_a3"]].clip(view)
        if len(in_country):
            in_country.boundary.plot(ax=iax, color="#c4c4c4", linewidth=0.25, zorder=Z_ADMIN1_LINES)
    if loc["_state_geom"] is not None:
        gpd.GeoSeries([loc["_state_geom"]], crs=4326).plot(
            ax=iax, color="#ffd9d9", edgecolor="#c0392b", linewidth=0.5, zorder=Z_ADMIN1
        )
    if loc["country_a3"] != "IND" and india.intersects(view):
        # India above the neighbours and the province lines, in its Survey of India
        # outline, so no inset shows the de facto line (e.g. Aksai Chin in the China view).
        gpd.GeoSeries([india.intersection(view)], crs=4326).plot(
            ax=iax, color="#eeeeee", edgecolor="#a9a9a9", linewidth=0.3, zorder=Z_INDIA
        )
        india_drawn = True
    # Lakes above every boundary layer (physical features, no boundaries), so the Great
    # Lakes and Lake Victoria read as water with no border through them; lakes smaller
    # than 0.02 % of the view would only be specks at this size.
    lakes = _natural_earth("lakes").clip(view)
    with warnings.catch_warnings():  # both areas are in square degrees: a relative test
        warnings.simplefilter("ignore", UserWarning)
        lakes = lakes[lakes.geometry.area >= view.area * 2e-4]
    if len(lakes):
        lakes.plot(ax=iax, color=WATER_COLOR, edgecolor="#8fa9bf", linewidth=0.3, zorder=Z_LAKES)
    iax.plot(
        loc["lon"],
        loc["lat"],
        marker="o",
        ms=4,
        mfc=PRED_COLOR,
        mec="black",
        mew=0.6,
        zorder=Z_MARKER,
    )
    iax.set_xlim(x0, x1)
    iax.set_ylim(y0, y1)
    # Shrink the inset box to the map's aspect, keeping it in the bottom-right corner.
    iax.set_aspect(_inset_aspect(loc), adjustable="box")
    iax.set_anchor("SE")
    iax.set_xticks([])
    iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_edgecolor("#333333")
        sp.set_linewidth(0.7)
    iax.set_title(title, fontsize=INSET_TITLE_FS, pad=INSET_TITLE_PAD_PT)
    return iax, india_drawn


# --------------------------------------------------------------------------- footer


@functools.cache
def _text_width_in(s: str, fontsize: float) -> float:
    """Width in inches of one line of text as the Agg renderer lays it out."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties

    if "agg" not in _TEXT_RENDERER:
        _TEXT_RENDERER["agg"] = RendererAgg(8, 8, DPI)
    renderer = _TEXT_RENDERER["agg"]
    w, _h, _d = renderer.get_text_width_height_descent(
        s, FontProperties(size=fontsize), ismath=False
    )
    return w / DPI


_TEXT_RENDERER: dict = {}


def _greedy_wrap(words: list[str], width_in: float, fontsize: float) -> list[str]:
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}" if cur else w
        if cur and _text_width_in(cand, fontsize) > width_in:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def _wrap(text: str, width_in: float, fontsize: float = NOTE_FS) -> list[str]:
    """Balanced wrap of *text* to *width_in* inches (measured, not counted in characters).

    First the fewest lines that fit, then the narrowest width that still gives that
    number of lines, so the lines have similar lengths and none is left with one word.
    A single word wider than *width_in* keeps a line of its own.
    """
    words = text.split()
    lines = _greedy_wrap(words, width_in, fontsize)
    if len(lines) <= 1:
        return lines
    lo, hi = 0.0, width_in
    for _ in range(30):  # bisection on the width; deterministic
        mid = (lo + hi) / 2
        if len(_greedy_wrap(words, mid, fontsize)) <= len(lines):
            hi = mid
        else:
            lo = mid
    return _greedy_wrap(words, hi, fontsize)


def _measure_legend(handles: list, ncol: int) -> tuple[float, float]:
    """(right edge, depth below its anchor, border pad included) in inches of the legend."""
    fig = plt.figure(figsize=(FIG_WIDTH_IN, 3), dpi=DPI)
    try:
        leg = fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(LEGEND_X_IN / FIG_WIDTH_IN, 1.0),
            ncol=ncol,
            fontsize=LEGEND_FS,
            frameon=False,
        )
        fig.canvas.draw()
        bb = leg.get_window_extent()
        return bb.x1 / DPI, 3 - bb.y0 / DPI
    finally:
        plt.close(fig)


def _measure_text_height(text: str, fontsize: float) -> float:
    """Height in inches of a (possibly multi-line) text block, default line spacing."""
    fig = plt.figure(figsize=(FIG_WIDTH_IN, 3), dpi=DPI)
    try:
        t = fig.text(0.5, 0.5, text, fontsize=fontsize)
        fig.canvas.draw()
        return t.get_window_extent().height / DPI
    finally:
        plt.close(fig)


def _notes_height_in(n_lines: int) -> float:
    """Height of *n_lines* note lines: one em to the first baseline, a descent below the last."""
    if n_lines == 0:
        return 0.0
    return NOTE_FS / 72 * 1.25 + (n_lines - 1) * NOTE_LINE_IN


def _check_footer(fig, footer_in: float, legend, notes: list, iax) -> None:
    """Raise if the legend, the notes and the inset overlap or leave the footer."""
    renderer = fig.canvas.get_renderer()
    boxes = {"legend": legend.get_window_extent(renderer)}
    if notes:
        from matplotlib.transforms import Bbox

        boxes["notes"] = Bbox.union([t.get_window_extent(renderer) for t in notes])
    if iax is not None:
        boxes["inset"] = iax.get_tightbbox(renderer)
    top = footer_in * fig.dpi + 1
    width = fig.get_figwidth() * fig.dpi + 1
    for name, b in boxes.items():
        if b.y1 > top or b.y0 < -1 or b.x0 < -1 or b.x1 > width:
            raise RuntimeError(f"footer layout: the {name} leaves the footer ({b.bounds})")
    names = list(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            p, q = boxes[a], boxes[b]
            if p.x0 < q.x1 and q.x0 < p.x1 and p.y0 < q.y1 and q.y0 < p.y1:
                raise RuntimeError(f"footer layout: the {a} overlaps the {b}")


def _write_preview(png: Path, out_dir: Path) -> Path:
    """``preview/<name>.webp``, PREVIEW_W px wide, from the saved PNG (deterministic)."""
    from PIL import Image

    prev = out_dir / "preview" / f"{png.stem}.webp"
    prev.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(png) as im:
        rgb = im.convert("RGB")
    rgb.info = {}  # no metadata in the preview
    size = (PREVIEW_W, round(rgb.height * PREVIEW_W / rgb.width))
    small = rgb.resize(size, Image.Resampling.LANCZOS)
    small.save(prev, format="WEBP", **PREVIEW_WEBP)
    return prev


# --------------------------------------------------------------------------- render


def render(entry: Entry, root: Path, out_dir: Path, basemap: str) -> dict:
    layers = []
    for lay in entry.layers:
        p = _one(lay.path, root)
        layers.append((lay, p, gpd.read_file(p), _sidecar(p)))
    prov0 = layers[0][3]
    if entry.per_layer_background:
        bgs = [_background_for(entry, root, pv, basemap, lay) for lay, _p, _g, pv in layers]
    else:
        bgs = [_background_for(entry, root, prov0, basemap)] * len(layers)
    tifs, source, comp_prov, bg_role = bgs[0]
    # Footer notes, one paragraph each (wrapped below, once the inset size is known).
    if basemap == "composite":
        if entry.per_layer_background:
            pos = _panel_names(len(layers), entry.ncols)
            notes = [f"{pos[i]}: {_bg_note(bg)}" for i, bg in enumerate(bgs)]
        else:
            notes = [_bg_note(bgs[0])]
        paragraphs = [("Background: " if k == 0 else "") + n for k, n in enumerate(notes)]
    else:
        paragraphs = [ESRI_ATTRIBUTION]

    if tifs:
        with rasterio.open(tifs[0]) as s:
            crs = s.crs
    else:
        crs = layers[0][2].estimate_utm_crs()
    frames = [g.to_crs(crs) for _, _, g, _ in layers]
    removed = []  # per layer: boolean mask of polygons the crop filter removed, or None
    for (lay, *_), gdf in zip(layers, frames, strict=True):
        if not lay.removed_vs:
            removed.append(None)
            continue
        kept = gpd.read_file(_one(lay.removed_vs, root)).to_crs(crs)
        pts = gpd.GeoDataFrame(geometry=gdf.geometry.representative_point(), crs=crs)
        hit = gpd.sjoin(pts, kept[["geometry"]], how="left", predicate="within")
        in_kept = hit.groupby(level=0)["index_right"].apply(lambda s: s.notna().any())
        removed.append(~in_kept.reindex(gdf.index, fill_value=False).to_numpy())
    overlay = overlay_lines = None
    if entry.overlay:
        opath, _, olayer = entry.overlay.partition("::")
        overlay = gpd.read_file(_one(opath, root), layer=olayer or None).to_crs(crs)
        # Each shared edge once (line_merge of the union of the boundaries): drawing every
        # polygon's boundary would draw shared edges twice, with different dash phases.
        overlay_lines = gpd.GeoSeries(
            [shapely.line_merge(overlay.boundary.union_all())], crs=overlay.crs
        )
    ref = None
    if entry.reference:
        ref = gpd.read_file(_one(entry.reference, root)).to_crs(crs)
        ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty]

    tb = np.array([f.total_bounds for f in frames if len(f)])
    minx, miny = tb[:, 0].min(), tb[:, 1].min()
    maxx, maxy = tb[:, 2].max(), tb[:, 3].max()
    padx, pady = (maxx - minx) * 0.02, (maxy - miny) * 0.02
    bounds = (minx - padx, miny - pady, maxx + padx, maxy + pady)

    if entry.window:
        wx, wy, side = entry.window
        bounds = (wx - side / 2, wy - side / 2, wx + side / 2, wy + side / 2)
    elif entry.crop_m:
        full = (bounds[0], bounds[2], bounds[1], bounds[3])
        cf = ref if (entry.crop_on_reference and ref is not None) else frames[entry.crop_layer]
        if entry.crop_on == "circular":
            cf = cf[_is_circular(cf)]
        elif entry.crop_on == "removed" and removed[entry.crop_layer] is not None:
            cf = cf[removed[entry.crop_layer]]
        elif entry.crop_on:
            cf = cf[_bool_column(cf[entry.crop_on])]
        z = _densest_window(cf if len(cf) else frames[0], entry.crop_m, full)
        if z is not None:
            bounds = (z[0], z[2], z[1], z[3])
    if ref is not None:
        ref = ref.cx[bounds[0] : bounds[2], bounds[1] : bounds[3]]
    if basemap == "composite":
        # Keep the map inside the imagery: the 2 % padding must not add white margins.
        # Union of the tiles within a panel, intersection across panels (every panel full).
        per_panel = []
        for b_tifs, *_ in bgs:
            tb_ = []
            for t in b_tifs:
                with rasterio.open(t) as src:
                    tb_.append(tuple(src.bounds))
            tb_ = np.array(tb_)
            per_panel.append((tb_[:, 0].min(), tb_[:, 1].min(), tb_[:, 2].max(), tb_[:, 3].max()))
        rb = np.array(per_panel)
        if entry.window and not (
            rb[:, 0].max() <= bounds[0]
            and rb[:, 1].max() <= bounds[1]
            and bounds[2] <= rb[:, 2].min()
            and bounds[3] <= rb[:, 3].min()
        ):
            raise ValueError(f"entry {entry.key}: window {entry.window} is not inside the imagery")
        bounds = (
            max(bounds[0], rb[:, 0].max()),
            max(bounds[1], rb[:, 1].max()),
            min(bounds[2], rb[:, 2].min()),
            min(bounds[3], rb[:, 3].min()),
        )
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    zoom = None
    if entry.zoom_m and len(frames) == 1:
        span = min(extent[1] - extent[0], extent[3] - extent[2])
        if entry.zoom_m < span * 0.6:
            zoom = _densest_window(frames[0], entry.zoom_m, extent)

    loc = None
    if entry.inset:
        from pyproj import Transformer

        to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = to_ll.transform((extent[0] + extent[1]) / 2, (extent[2] + extent[3]) / 2)
        loc = _locate(lon, lat)

    model_lines = []
    for lay, _p, _g, pv in layers:
        if lay.model_from:
            pv = _sidecar(_one(lay.model_from, root))
        m = _model_note(pv)
        if m and m not in model_lines:
            model_lines.append(m)
    extra = list(entry.model_notes)
    if entry.sam2_note:
        sidecars = [pv for *_, pv in layers] + [
            _sidecar(_one(lay.model_from, root)) for lay, *_ in layers if lay.model_from
        ]
        versions = next((pv["versions"] for pv in sidecars if pv.get("versions")), {})
        extra.append(_sam2_note(versions, resmoothed=entry.sam2_resmoothed))
    for m in extra:
        if m not in model_lines:
            model_lines.append(m)
    paragraphs += [("Model: " if k == 0 else "") + m for k, m in enumerate(model_lines)]
    if loc is not None:
        paragraphs.append(INSET_NOTE_INDIA if _inset_shows_india(loc) else INSET_NOTE)

    handles = [Line2D([], [], color=PRED_COLOR, lw=1.5, label=PRED_LABEL)]
    if any(lay.highlight for lay, *_ in layers):
        handles.append(Line2D([], [], color=HIGHLIGHT_COLOR, lw=1.5, label=HIGHLIGHT_LABEL))
    if any(r is not None for r in removed):
        handles.append(Line2D([], [], color=REMOVED_COLOR, lw=1.5, label=REMOVED_LABEL))
    if ref is not None:
        handles.append(Line2D([], [], color=REF_COLOR, lw=1.5, label=entry.reference_label))
    if overlay is not None:
        handles.append(
            Line2D([], [], color=ZOOM_COLOR, lw=1.2, ls=(0, (4, 3)), label=entry.overlay_label)
        )
    if entry.mark_windows:
        labels = ", ".join(label for label, *_ in entry.mark_windows)
        name = "Zoom window" if len(entry.mark_windows) == 1 else "Zoom windows"
        handles.append(
            Rectangle(
                (0, 0), 1, 1, fill=False, edgecolor=ZOOM_COLOR, lw=1.1, label=f"{name} {labels}"
            )
        )

    # Footer layout (inches from the bottom of the image): the legend row at the top,
    # the notes under it, the inset at the bottom right; the height grows with the notes.
    legend_ncol = len(handles)
    legend_right, legend_h = _measure_legend(handles, legend_ncol)
    if legend_right > FIG_WIDTH_IN - NOTE_X_IN:  # too wide for one row: two rows
        legend_ncol = math.ceil(len(handles) / 2)
        legend_right, legend_h = _measure_legend(handles, legend_ncol)
    inset = None
    if loc is not None:
        _country, view = _inset_view(loc)
        inset_w, inset_h = _inset_box(loc, view)
        inset_title = _inset_title(loc, inset_w)
        title_w = max(_text_width_in(t, INSET_TITLE_FS) for t in inset_title.split("\n"))
        map_right = FIG_WIDTH_IN - INSET_RIGHT_IN
        inset = {
            "title": inset_title,
            "left": min(map_right - inset_w, map_right - (inset_w + title_w) / 2),
            "top": FOOT_BOTTOM_IN
            + inset_h
            + INSET_TITLE_PAD_PT / 72
            + _measure_text_height(inset_title, INSET_TITLE_FS),
        }
    note_right = inset["left"] - NOTE_INSET_GAP_IN if inset else FIG_WIDTH_IN - NOTE_X_IN
    note_lines = [line for par in paragraphs for line in _wrap(par, note_right - NOTE_X_IN)]
    bg_note = "\n".join(note_lines)
    notes_top = FOOT_TOP_IN + legend_h + LEGEND_NOTE_GAP_IN  # below the top of the footer
    footer = notes_top + _notes_height_in(len(note_lines)) + FOOT_BOTTOM_IN
    if inset:
        # The inset rises beside the legend when the legend ends clear of it.
        beside_legend = legend_right + NOTE_INSET_GAP_IN <= inset["left"]
        footer = max(footer, inset["top"] + (FOOT_TOP_IN if beside_legend else notes_top))

    n_panels = len(frames) + (1 if zoom else 0)
    ncols = min(entry.ncols or n_panels, n_panels)
    nrows = math.ceil(n_panels / ncols)
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    # The drawn width of one map; each grid cell is exactly one map high, so no white band
    # is left between the maps and the footer.
    panel_w = FIG_WIDTH_IN * (MAPS_RIGHT - MAPS_LEFT) / (ncols + MAPS_WSPACE * (ncols - 1))
    panel_h = panel_w * (aspect if not zoom else max(aspect, 1.0))
    row_gap = ROW_GAP_IN if nrows > 1 else 0.0  # room for the titles of the lower rows
    fig_h = TITLE_IN + nrows * panel_h + (nrows - 1) * row_gap + footer
    main_px = (panel_w * DPI, panel_w * aspect * DPI)

    if basemap == "composite":
        imgs = []
        for i, (b_tifs, b_source, _cp, _role) in enumerate(bgs):
            if i and bgs[i] is bgs[0]:
                imgs.append(imgs[0])
                continue
            with rasterio.open(b_tifs[0]) as s:
                if s.crs != crs:
                    raise ValueError(f"entry {entry.key}: panel {i} raster CRS {s.crs} != {crs}")
            imgs.append(_read_background(b_tifs, b_source, bounds, crs, main_px))
        zoom_img = None
        if zoom:
            z = zoom
            zoom_img = _read_background(
                bgs[0][0],
                bgs[0][1],
                (z[0], z[2], z[1], z[3]),
                crs,
                (panel_w * DPI, panel_w * DPI),
                stretch=imgs[0][2],
            )
    else:
        imgs = [None] * len(layers)
        zoom_img = None

    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH_IN, fig_h), dpi=DPI, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    def draw(ax, gdf, ext, lw, title, hl=None, bg=None, rm=None):
        if bg is not None:
            # "auto": anti-aliased when downsampling, nearest when enlarging by >= 3x,
            # so coarse imagery shows its real pixels instead of a bilinear blur.
            ax.imshow(bg[0], extent=bg[1], interpolation="auto", zorder=1)
        else:
            import contextily as cx

            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            cx.add_basemap(
                ax, crs=crs, source=cx.providers.Esri.WorldImagery, attribution=False, zorder=1
            )
        if ref is not None and len(ref):
            ref.boundary.plot(ax=ax, color=REF_COLOR, linewidth=lw * 0.9, zorder=3)
        if len(gdf):
            n_before = len(ax.collections)
            gdf.boundary.plot(ax=ax, color=PRED_COLOR, linewidth=lw, zorder=4)
            if entry.halo:
                halo = [pe.Stroke(linewidth=lw + 1.6, foreground="white", alpha=0.85), pe.Normal()]
                for coll in ax.collections[n_before:]:
                    coll.set_path_effects(halo)
            if hl and hl in gdf.columns:
                sel = gdf[_bool_column(gdf[hl])]
                if len(sel):
                    sel.boundary.plot(ax=ax, color=HIGHLIGHT_COLOR, linewidth=lw * 1.3, zorder=4.5)
            if rm is not None and rm.any():
                gdf[rm].boundary.plot(ax=ax, color=REMOVED_COLOR, linewidth=lw * 1.3, zorder=4.6)
        if overlay_lines is not None and len(overlay):
            overlay_lines.plot(
                ax=ax, color=ZOOM_COLOR, linewidth=1.1, linestyle=(0, (4, 3)), zorder=5
            )
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6)
        _scalebar(ax, ext, crs)
        ax.set_title(title, fontsize=TITLE_FS, pad=4)

    for i, (ax, (lay, _p, _g, _pv), gdf) in enumerate(zip(axes, layers, frames, strict=False)):
        draw(
            ax,
            gdf,
            extent,
            entry.line_width * (1.0 if n_panels == 1 else 0.8),
            f"{lay.label} ({len(gdf):,} fields)",
            hl=lay.highlight,
            bg=imgs[i],
            rm=removed[i],
        )
    for ax in axes[: len(layers)]:
        for label, wx, wy, side in entry.mark_windows:
            ax.add_patch(
                Rectangle(
                    (wx - side / 2, wy - side / 2),
                    side,
                    side,
                    fill=False,
                    edgecolor=ZOOM_COLOR,
                    lw=1.1,
                    zorder=5,
                )
            )
            ax.text(
                wx - side / 2 + side * 0.04,
                wy + side / 2 - side * 0.04,
                label,
                color=ZOOM_COLOR,
                fontsize=TITLE_FS,
                fontweight="bold",
                ha="left",
                va="top",
                zorder=6,
                clip_on=True,  # like the square: no stray label for a window off the panel
                path_effects=[pe.Stroke(linewidth=2, foreground="black"), pe.Normal()],
            )
    if zoom:
        z = zoom
        axes[0].add_patch(
            Rectangle(
                (z[0], z[2]),
                z[1] - z[0],
                z[3] - z[2],
                fill=False,
                edgecolor=ZOOM_COLOR,
                lw=1.2,
                zorder=5,
            )
        )
        zoom_title = f"Zoom ({entry.zoom_m / 1000:g} km)"
        draw(
            axes[-1],
            frames[0],
            z,
            entry.line_width * 1.3,
            zoom_title,
            hl=layers[0][0].highlight,
            bg=zoom_img,
        )
        for s in axes[-1].spines.values():
            s.set_edgecolor(ZOOM_COLOR)
            s.set_linewidth(1.4)

    iax = None
    if loc is not None:
        rect = [
            1 - (INSET_W_IN + INSET_RIGHT_IN) / FIG_WIDTH_IN,
            FOOT_BOTTOM_IN / fig_h,
            INSET_W_IN / FIG_WIDTH_IN,
            INSET_H_IN / fig_h,
        ]
        iax, _ = _draw_inset(fig, rect, loc, inset["title"])

    # Legend and notes hang from the top of the footer, right under the maps; the notes
    # are drawn line by line, one every NOTE_LINE_IN inches.
    legend = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(LEGEND_X_IN / FIG_WIDTH_IN, (footer - FOOT_TOP_IN) / fig_h),
        ncol=legend_ncol,
        fontsize=LEGEND_FS,
        frameon=False,
    )
    first_baseline = footer - notes_top - NOTE_FS / 72
    note_texts = [
        fig.text(
            NOTE_X_IN / FIG_WIDTH_IN,
            (first_baseline - i * NOTE_LINE_IN) / fig_h,
            line,
            ha="left",
            va="baseline",
            fontsize=NOTE_FS,
            color=NOTE_COLOR,
        )
        for i, line in enumerate(note_lines)
    ]
    fig.subplots_adjust(
        left=MAPS_LEFT,
        right=MAPS_RIGHT,
        top=1 - TITLE_IN / fig_h,
        bottom=footer / fig_h,
        wspace=MAPS_WSPACE,
        hspace=row_gap / panel_h,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{entry.name}.png"
    fig.savefig(png, dpi=DPI, metadata={"Software": None})
    try:
        _check_footer(fig, footer, legend, note_texts, iax)
    except RuntimeError:
        png.unlink()
        raise
    finally:
        plt.close(fig)
    preview = _write_preview(png, out_dir)

    stats = {
        "example": entry.key,
        # POSIX form, so gallery_stats.json reads the same whichever OS rendered it.
        "image": (png.relative_to(REPO) if png.is_relative_to(REPO) else png).as_posix(),
        "preview": (
            preview.relative_to(REPO) if preview.is_relative_to(REPO) else preview
        ).as_posix(),
        "title": entry.title,
        "background": "composite" if basemap == "composite" else "esri",
        "background_note": bg_note,
        "model_notes": model_lines,
        "location": {k: v for k, v in (loc or {}).items() if not k.startswith("_")},
        "layers": [],
    }
    for k, (lay, p, g, pv) in enumerate(layers):
        cfg = pv.get("config") or pv.get("base_config") or {}
        em = pv.get("engine_meta") or {}
        stats["layers"].append(
            {
                "label": lay.label,
                "file": str(p.relative_to(root)),
                **_area_stats(g),
                "source": cfg.get("source"),
                "engine": cfg.get("engine"),
                "year": cfg.get("year"),
                "lulc_filter": cfg.get("lulc_filter"),
                "lulc_dataset": cfg.get("lulc_dataset"),
                "sam_refine": cfg.get("sam_refine"),
                "min_field_area_m2": cfg.get("min_field_area_m2"),
                "composite": _composite_facts(pv, tifs[0] if tifs and len(layers) == 1 else None),
                "ftw_windows": {
                    k: {kk: w.get(kk) for kk in ("start", "end", "n_images", "status")}
                    for k, w in ((em.get("windows") or {}).items())
                    if k in ("a", "b") and isinstance(w, dict)
                },
                "engine_meta": {
                    k: em[k]
                    for k in sorted(em)
                    if isinstance(em[k], (str, int, float, bool)) and not _is_local_path(em[k])
                },
                "n_removed_by_crop_filter": (
                    int(removed[k].sum()) if removed[k] is not None else None
                ),
                "agribound_version": pv.get("agribound_version"),
                "run_status": pv.get("status"),
            }
        )
    if ref is not None:
        stats["reference"] = {"label": entry.reference_label, **_area_stats(ref)}
    if entry.metrics:
        try:
            stats["metrics"] = json.loads(_one(entry.metrics, root).read_text())
        except FileNotFoundError as exc:
            stats["metrics_error"] = str(exc)
    if zoom:
        stats["zoom_window_m"] = entry.zoom_m
    if entry.window:
        stats["window"] = {
            "crs": str(crs),
            "centre": list(entry.window[:2]),
            "side_m": entry.window[2],
        }
    if entry.mark_windows:
        stats["marked_windows"] = [
            {"label": lab, "centre": [x, y], "side_m": s} for lab, x, y, s in entry.mark_windows
        ]
    return stats


def _world_view(points: list[tuple[str, float, float]]):
    """View box (EPSG:4326) around the dots of a multi-area entry."""
    from shapely.geometry import box

    lons = [p[1] for p in points]
    lats = [p[2] for p in points]
    px, py = WORLD_INSET_PAD_DEG
    return box(
        max(min(lons) - px, -180.0),
        max(min(lats) - py, -60.0),
        min(max(lons) + px, 180.0),
        min(max(lats) + py, 80.0),
    )


def _draw_world_inset(fig, rect, points: list[tuple[str, float, float]], view) -> tuple:
    """Locator for multi-area entries: countries of *view* and a numbered dot per study area.

    India is drawn from the Survey of India outline above the neighbouring countries, as in
    :func:`_draw_inset`. Returns ``(axes, india_drawn)``.
    """
    x0, y0, x1, y1 = view.bounds
    iax = fig.add_axes(rect)
    iax.set_facecolor(WATER_COLOR)
    countries = _natural_earth("countries")
    others = countries[countries["ADM0_A3"] != "IND"].clip(view)
    if len(others):
        others.plot(
            ax=iax, color="#f4f4f4", edgecolor="#a9a9a9", linewidth=0.25, zorder=Z_NEIGHBOURS
        )
    india = _india_outline()
    india_drawn = bool(india.intersects(view))
    if india_drawn:
        gpd.GeoSeries([india.intersection(view)], crs=4326).plot(
            ax=iax, color="#f4f4f4", edgecolor="#a9a9a9", linewidth=0.25, zorder=Z_INDIA
        )
    lakes = _natural_earth("lakes").clip(view)
    with warnings.catch_warnings():  # both areas are in square degrees: a relative test
        warnings.simplefilter("ignore", UserWarning)
        lakes = lakes[lakes.geometry.area >= view.area * 2e-4]
    if len(lakes):
        lakes.plot(ax=iax, color=WATER_COLOR, edgecolor="#8fa9bf", linewidth=0.25, zorder=Z_LAKES)
    for label, lon, lat in points:
        iax.plot(lon, lat, marker="o", ms=4, mfc=PRED_COLOR, mec="black", mew=0.6, zorder=Z_MARKER)
        iax.annotate(
            label,
            (lon, lat),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=INSET_TITLE_FS - 1,
            fontweight="bold",
            zorder=Z_MARKER + 1,
            path_effects=[pe.Stroke(linewidth=2, foreground="white"), pe.Normal()],
        )
    iax.set_xlim(x0, x1)
    iax.set_ylim(y0, y1)
    iax.set_aspect(1 / max(math.cos(math.radians((y0 + y1) / 2)), 0.2), adjustable="box")
    iax.set_anchor("SE")
    iax.set_xticks([])
    iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_edgecolor("#333333")
        sp.set_linewidth(0.7)
    return iax, india_drawn


def _areas_bg_note(panels: list[dict]) -> str:
    """Background note of a multi-area entry: one line when every panel has the same source."""
    facts = []
    for p in panels:
        tifs, source, comp_prov, role = p["bg"]
        facts.append((source, role, _composite_facts(comp_prov, tifs[0] if tifs else None)))
    heads = {
        (src, role, c.get("resolution_m"), c.get("composite_method")) for src, role, c in facts
    }
    if len(heads) != 1:
        return "Background: " + "; ".join(f"{p['n']}: {_bg_note(p['bg'])}" for p in panels)
    source, role, res, method = heads.pop()
    label = SOURCE_LABELS.get(source or "", source or "imagery")
    if res:
        label += f" {float(res):g} m"
    per = []
    for p, (_src, _role, c) in zip(panels, facts, strict=True):
        dates = (
            f"{c['date_start']} to {c['date_end_exclusive']}"
            if c.get("date_start") and c.get("date_end_exclusive")
            else "dates not recorded"
        )
        n_img = f", {c['n_images']} images" if c.get("n_images") else ""
        per.append(f"{p['n']}: {dates}{n_img}")
    kind = f"{method} composites" if method else "composites"
    return f"Background: {label} {kind} ({role}); end dates exclusive: " + "; ".join(per)


def render_areas(entry: Entry, root: Path, out_dir: Path, basemap: str) -> dict:
    """Render a multi-area entry: one panel per layer, each on its own study area.

    Each panel is drawn in the CRS of its own background raster, on the square of side
    ``lay.crop_m`` (else ``entry.crop_m``) holding the most polygon representative points
    of that layer, among candidate squares half a side apart inside its imagery, with its
    own scale bar. A world locator with numbered dots replaces the country inset. Options
    that only :func:`render` implements are refused.
    """
    from pyproj import Transformer

    if basemap != "composite":
        raise ValueError(f"entry {entry.key}: multi-area entries need --basemap composite")
    if not entry.crop_m:
        raise ValueError(f"entry {entry.key}: multi-area entries need crop_m")
    unsupported = [
        n
        for n in (
            "reference",
            "crop_on_reference",
            "window",
            "zoom_m",
            "overlay",
            "crop_on",
            "crop_layer",
            "mark_windows",
            "sam2_note",
            "metrics",
            "background",
            "background_from",
        )
        if getattr(entry, n)
    ] + [
        f"layers[{k}].{n}"
        for k, lay in enumerate(entry.layers)
        for n in ("highlight", "removed_vs", "model_from")
        if getattr(lay, n)
    ]
    if unsupported:
        raise ValueError(
            f"entry {entry.key}: multi-area entries do not support {', '.join(unsupported)}"
        )
    panels = []
    for i, lay in enumerate(entry.layers):
        path = _one(lay.path, root)
        gdf, prov = gpd.read_file(path), _sidecar(path)
        bg = _background_for(entry, root, prov, basemap, lay)
        with rasterio.open(bg[0][0]) as src:
            crs, rb = src.crs, src.bounds
        g = gdf.to_crs(crs)
        # The square is placed inside the imagery (which covers the study area), so it fits
        # even where the polygons span less than crop_m in one direction.
        full = (rb.left, rb.right, rb.bottom, rb.top)
        side = lay.crop_m or entry.crop_m
        ext = _densest_window(g, side, full) if len(g) else None
        ext = ext or full
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
            (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2
        )
        panels.append(
            {
                "n": str(i + 1),
                "lay": lay,
                "path": path,
                "gdf": g,
                "prov": prov,
                "bg": bg,
                "crs": crs,
                "extent": ext,
                "lon": lon,
                "lat": lat,
            }
        )

    paragraphs = [_areas_bg_note(panels)]
    model_lines = []
    for p in panels:
        m = _model_note(p["prov"])
        if m and m not in model_lines:
            model_lines.append(m)
    for m in entry.model_notes:
        if m not in model_lines:
            model_lines.append(m)
    paragraphs += [("Model: " if k == 0 else "") + m for k, m in enumerate(model_lines)]
    points = [(p["n"], p["lon"], p["lat"]) for p in panels]
    view = _world_view(points) if entry.inset else None
    if view is not None:
        dots = f"Inset: study areas 1-{len(panels)}; "
        if _india_outline().intersects(view):
            paragraphs.append(dots + INSET_NOTE_INDIA.removeprefix("Inset: "))
        else:
            paragraphs.append(dots + "Natural Earth country and lake boundaries")

    handles = [Line2D([], [], color=PRED_COLOR, lw=1.5, label=PRED_LABEL)]
    legend_ncol = len(handles)
    legend_right, legend_h = _measure_legend(handles, legend_ncol)
    inset = None
    if view is not None:
        vx0, vy0, vx1, vy1 = view.bounds
        ratio = (vy1 - vy0) / max(math.cos(math.radians((vy0 + vy1) / 2)), 0.2) / (vx1 - vx0)
        w_in = min(WORLD_INSET_W_IN, INSET_H_IN / ratio)
        inset = {"w": w_in, "h": w_in * ratio}
        inset["left"] = FIG_WIDTH_IN - INSET_RIGHT_IN - w_in
        inset["top"] = FOOT_BOTTOM_IN + inset["h"]
    note_right = inset["left"] - NOTE_INSET_GAP_IN if inset else FIG_WIDTH_IN - NOTE_X_IN
    note_lines = [line for par in paragraphs for line in _wrap(par, note_right - NOTE_X_IN)]
    notes_top = FOOT_TOP_IN + legend_h + LEGEND_NOTE_GAP_IN
    footer = notes_top + _notes_height_in(len(note_lines)) + FOOT_BOTTOM_IN
    if inset:
        beside_legend = legend_right + NOTE_INSET_GAP_IN <= inset["left"]
        footer = max(footer, inset["top"] + (FOOT_TOP_IN if beside_legend else notes_top))

    n = len(panels)
    ncols = min(entry.ncols or 3, n)
    nrows = math.ceil(n / ncols)
    panel_w = FIG_WIDTH_IN * (MAPS_RIGHT - MAPS_LEFT) / (ncols + MAPS_WSPACE * (ncols - 1))
    panel_h = panel_w  # square windows
    row_gap = ROW_GAP_IN if nrows > 1 else 0.0
    fig_h = TITLE_IN + nrows * panel_h + (nrows - 1) * row_gap + footer
    px = (panel_w * DPI, panel_h * DPI)

    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH_IN, fig_h), dpi=DPI, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    lw = entry.line_width * 0.8
    for ax, p in zip(axes, panels, strict=False):
        e = p["extent"]
        tifs, source, *_ = p["bg"]
        img, img_ext, _stretch = _read_background(
            tifs, source, (e[0], e[2], e[1], e[3]), p["crs"], px
        )
        ax.imshow(img, extent=img_ext, interpolation="auto", zorder=1)
        g = p["gdf"].cx[e[0] : e[1], e[2] : e[3]]
        if len(g):
            n_before = len(ax.collections)
            g.boundary.plot(ax=ax, color=PRED_COLOR, linewidth=lw, zorder=4)
            if entry.halo:
                halo = [pe.Stroke(linewidth=lw + 1.6, foreground="white", alpha=0.85), pe.Normal()]
                for coll in ax.collections[n_before:]:
                    coll.set_path_effects(halo)
        ax.set_xlim(e[0], e[1])
        ax.set_ylim(e[2], e[3])
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        _scalebar(ax, e, p["crs"])
        title = f"{p['n']}. {p['lay'].label} ({len(p['gdf']):,} fields)"
        fs = TITLE_FS
        while fs > 7 and _text_width_in(title, fs) > panel_w * 0.98:
            fs -= 0.5  # a long title is set smaller, not over the next panel
        ax.set_title(title, fontsize=fs, pad=4)

    iax = None
    if inset:
        rect = [
            inset["left"] / FIG_WIDTH_IN,
            FOOT_BOTTOM_IN / fig_h,
            inset["w"] / FIG_WIDTH_IN,
            inset["h"] / fig_h,
        ]
        iax, _india = _draw_world_inset(fig, rect, points, view)
    legend = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(LEGEND_X_IN / FIG_WIDTH_IN, (footer - FOOT_TOP_IN) / fig_h),
        ncol=legend_ncol,
        fontsize=LEGEND_FS,
        frameon=False,
    )
    first_baseline = footer - notes_top - NOTE_FS / 72
    note_texts = [
        fig.text(
            NOTE_X_IN / FIG_WIDTH_IN,
            (first_baseline - i * NOTE_LINE_IN) / fig_h,
            line,
            ha="left",
            va="baseline",
            fontsize=NOTE_FS,
            color=NOTE_COLOR,
        )
        for i, line in enumerate(note_lines)
    ]
    fig.subplots_adjust(
        left=MAPS_LEFT,
        right=MAPS_RIGHT,
        top=1 - TITLE_IN / fig_h,
        bottom=footer / fig_h,
        wspace=MAPS_WSPACE,
        hspace=row_gap / panel_h,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{entry.name}.png"
    fig.savefig(png, dpi=DPI, metadata={"Software": None})
    try:
        _check_footer(fig, footer, legend, note_texts, iax)
    except RuntimeError:
        png.unlink()
        raise
    finally:
        plt.close(fig)
    preview = _write_preview(png, out_dir)

    stats = {
        "example": entry.key,
        "image": (png.relative_to(REPO) if png.is_relative_to(REPO) else png).as_posix(),
        "preview": (
            preview.relative_to(REPO) if preview.is_relative_to(REPO) else preview
        ).as_posix(),
        "title": entry.title,
        "background": "composite",
        "background_note": "\n".join(note_lines),
        "model_notes": model_lines,
        "location": (
            {
                "areas": [
                    {
                        "panel": p["n"],
                        **{k: v for k, v in _locate(p["lon"], p["lat"]).items() if k[0] != "_"},
                    }
                    for p in panels
                ]
            }
            if entry.inset
            else {}
        ),
        "layers": [],
    }
    for p in panels:
        cfg = p["prov"].get("config") or {}
        em = p["prov"].get("engine_meta") or {}
        tifs = p["bg"][0]
        stats["layers"].append(
            {
                "panel": p["n"],
                "label": p["lay"].label,
                "file": str(p["path"].relative_to(root)),
                **_area_stats(p["gdf"]),
                "source": cfg.get("source"),
                "engine": cfg.get("engine"),
                "year": cfg.get("year"),
                "lulc_filter": cfg.get("lulc_filter"),
                "lulc_dataset": cfg.get("lulc_dataset"),
                "sam_refine": cfg.get("sam_refine"),
                "min_field_area_m2": cfg.get("min_field_area_m2"),
                "composite": _composite_facts(p["prov"], tifs[0] if tifs else None),
                "engine_meta": {
                    k: em[k]
                    for k in sorted(em)
                    if isinstance(em[k], (str, int, float, bool)) and not _is_local_path(em[k])
                },
                "window": {
                    "crs": str(p["crs"]),
                    "bounds": [
                        float(v)
                        for v in (p["extent"][0], p["extent"][2], p["extent"][1], p["extent"][3])
                    ],
                    "centre_lonlat": [round(p["lon"], 4), round(p["lat"], 4)],
                },
                "agribound_version": p["prov"].get("agribound_version"),
                "run_status": p["prov"].get("status"),
            }
        )
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-root", default=str(REPO / "outputs" / "examples_1.0"))
    ap.add_argument("--out-dir", default=str(REPO / "assets" / "gallery_1.0"))
    ap.add_argument("--only", default="", help="comma-separated example numbers")
    ap.add_argument("--basemap", choices=("composite", "esri"), default="composite")
    ap.add_argument("--list", action="store_true", help="list the entries and exit")
    args = ap.parse_args(argv)

    root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    entries = [e for e in ENTRIES if not only or e.key in only]
    if args.list:
        for e in entries:
            print(f"{e.key}  {e.name:24s} {e.title}")
            for lay in e.layers:
                print(f"      {lay.path}")
        return 0

    stats_path = out_dir / "gallery_stats.json"
    # A full run writes the file afresh (no entry from an older version of this script
    # survives); --only replaces the rendered entries and keeps the others.
    all_stats = json.loads(stats_path.read_text()) if (only and stats_path.exists()) else {}
    failed = []
    for e in entries:
        try:
            renderer = render_areas if e.multi_area else render
            all_stats[e.key] = renderer(e, root, out_dir, args.basemap)
            print(f"[{e.key}] {out_dir / (e.name + '.png')}")
        except Exception as exc:  # report and continue with the other entries
            failed.append(e.key)
            print(f"[{e.key}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(dict(sorted(all_stats.items())), indent=2) + "\n")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
