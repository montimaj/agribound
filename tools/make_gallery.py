#!/usr/bin/env python
"""
Render the documentation gallery images from the agribound example runs.

Usage::

    python tools/make_gallery.py                       # all entries -> assets/gallery_1.1/
    python tools/make_gallery.py --only 04,06          # some entries
    python tools/make_gallery.py --list                # show the entries and their inputs
    python tools/make_gallery.py --basemap esri        # Esri World Imagery instead of the composite
    python tools/make_gallery.py --run-root outputs/examples_1.0 --out-dir assets/gallery_1.0

Inputs are the outputs of the example runs under each entry's run root, by
default ``outputs/examples_1.1``, the re-run of the examples that follows
1.0.1 (see "Run roots" below), and the images go to ``--out-dir``, by default
``assets/gallery_1.1``. The 1.0
gallery (``assets/gallery_1.0``, rendered from ``outputs/examples_1.0`` and
embedded by the documentation until the 1.1 images replace it) is therefore
not overwritten. The last command above renders into it again from the 1.0
runs; the New Mexico entries (12n, 13, 14, 20, 20b and 20c) now name the V18
and 2024 outputs of the 1.1 runs, so their 1.0 images need the tool of the
v1.0.1 tag (``git show v1.0.1:tools/make_gallery.py``).
Every entry names the output file(s) of one example; a file may be a glob that
must match exactly one file.

Run roots
---------
An entry reads its files under its own run root, ``Entry.run_root`` (relative
to the repository), else ``RUN_ROOT`` (``outputs/examples_1.1``), so one command
renders every entry from the run that holds its current outputs:

- ``outputs/examples_1.1_final`` (``FINAL_RUN_ROOT``), the targeted re-runs with
  the final code: entries 03 (HLS image counts), 14 (the Landsat run with
  DINOv3's 10 m processing grid), 23 and 23b-23e (example 23 with
  ``--s2-single-date``; the TESSERA runs made again), 25b (the crop-history
  zones of example 25) and 27 (new). The outputs those re-runs reused are
  copies of the 1.1 run's (example 25 wrote its reference and metrics files
  again: the same polygons and scores, the metrics now with ``proj_network``);
- ``outputs/examples_1.1_sar`` (``SAR_RUN_ROOT``), the radar run: entries 26
  and 26b;
- ``outputs/examples_1.1_ex24b`` (``EX24_RUN_ROOT``), example 24 with
  ``--options`` and ``--sam-auto``: entries 24, 24b, 24c, 24d and 24e, so that
  the five Nevada figures come from one run root. Its 28 other runs are those
  of the re-scoring run ``outputs/examples_1.1_ex24`` (example 24 scored again
  with its irrigated-only score; copies of the 1.1 run's outputs, no model run
  again), cloned unchanged: the outputs, provenance records, composites,
  references and metrics files byte-identical. It adds the six ``sam-auto``
  runs (identical to the SAM evaluation's E1 runs) and the eight "-opt" runs;
  24c and 24d read its ``metrics_<stem>_irrigated.json``;
- ``outputs/examples_1.1_r3/ny`` (``NY_RUN_ROOT``), example 28's run: entries
  28, 28b and 28c;
- ``outputs/examples_1.1_r3/sam_eval`` (``SAM_EVAL_RUN_ROOT``), the
  pre-registered SAM evaluation: the SAM entry (key ``sam``). Its comparators,
  Delineate-Anything v2 as released, are read where the evaluation read them,
  each under its own run root (``Layer.run_root``: ``outputs/examples_1.1_ex24``
  and ``outputs/examples_1.1_final``);
- ``outputs/examples_1.1_r3`` (``R3_RUN_ROOT``): the left panel of entry 25c,
  the NAIP 2005 image of example 25's Montana square, exported with its record
  by ``tools/fetch_naip_history.py`` (the panel's own run root,
  ``ImagePanel.run_root``; the rest of the entry is read in
  ``outputs/examples_1.1``);
- ``outputs/examples_1.1``: every other entry.

``--run-root`` replaces the run root of every entry, and of every layer and
image panel with its own (as for the 1.0 gallery); ``--list`` prints each
entry's run root.

Examples 24 to 28 and the SAM evaluation
----------------------------------------
Their panels are chosen by rules applied to the examples' summary files, or
fixed before the runs, not by appearance (each rule is written above its
entries, and the tests re-apply it to the run's files): entries 24 and 24b,
the two basins of example 24, draw NAIP Delineate-Anything v2 as released and
the highest-F1 NAIP, SPOT 6/7 and Sentinel-1 runs in the evaluation square
(the ``sam-auto`` runs compete, the "-opt" runs do not), each on its own input
image (NAIP 2022 in the top row, SPOT 6/7 2023 at bottom left, Sentinel-1 2024
at bottom right); entries 24c and 24d draw the same runs on the same images
with example 24's irrigated-only score (``Entry.scored_part``, below; ranked by
that score, the rule picks the same runs); entry 24e draws Diamond Valley's
runs before and after the opt-in options of 1.1.0, with the panels that
PREREG-DV3 fixed before the options were tested (the "-opt" results are the
crop-filter layers that example 24 scores); entries 28, 28b and 28c draw the
three regions of example 28 by the panel rule of PREREG-NY, fixed before the
run; the SAM entry draws SAM 2.1 automatic masks (``sam-auto``) beside
Delineate-Anything v2 as released on the two squares that round-2 PLAN 1.5
fixed before the SAM evaluation, each in its whole 5 km square
(``Layer.window``); entry 25
is an overview map (below) and 25b the four squares at evenly spaced ranks of
its F1; entries 26 (held-out sites) and 26b (in-sample sites) draw, per site,
Sentinel-1, the Umbra or Capella composite and NAIP read with the same
settings (Sentinel-2 at Sevilla), each on its own input, in the square with the
most reference fields among the candidate squares (half a side apart) that lie
inside the site's evaluation area (``Layer.window_within``); entry 27 draws the
crop-history zones and Delineate-Anything of example 27 on the Sentinel-2
composite. With ``MULTISTATE_ZONES = True`` (set since the final run of example
25 with ``--crop-sequence`` was checked), entry 25b draws the CDL zones beside
Delineate-Anything in each square and reads ``outputs/examples_1.1_final``.
Entry 25c draws no model output: example 25's Montana square with its
reference on the NAIP of 2005 and of 2023 ("Imagery entries", below).

Their panels give their scores: ``Layer.metrics`` names the metrics file the
example wrote beside the output (``metrics_<stem>.json``); its summary row and
scalar scores go into the layer's ``gallery_stats.json`` record (``metrics``),
and with ``Entry.title_f1`` the panel title gives the F1 ("(142 fields; F1
0.38)"). The count is of the whole output file and the F1 of the example's
evaluation area, which can be larger than the panel's window; a footer line
says so where it is (25b, 26, 26b, 27). A score file that holds several
records (the SAM evaluation's ``scores.json``) gives one of them with
``Layer.metrics_record``, the keys of the record. ``Layer.noun`` names what a
title counts ("zone" for crop-history zones) and ``Entry.pred_label`` the red
outlines in the legend. With ``Entry.model_positions`` (14, 23e, 24-24e, 28-28c
and the SAM entry) each model line names the panels it drew (in a multi-area
entry, its study areas: "(study area 2)"), so that fine-tuned checkpoints of
one model, and the per-source settings of ``sam-auto``, are told apart. A
DINOv3 line names the processing grid of a run that read a resampled copy of a
raster coarser than 10 m (", read on a 10 m bilinear copy of the 30 m raster":
entry 14's Landsat panel). A ``sam-auto`` line names the checkpoint and the
setting it ran with; under an image with a ``sam-auto`` panel a credit line
follows the model lines: SAM 2.1's maker and licence (Meta, Apache-2.0) and the
sam2 build that ran it, a third-party PyPI build whose files equal a commit of
Meta's code (both from the run's ``engine_meta``), and the device.

Scores on part of the reference
-------------------------------
An entry with ``Entry.scored_part`` (a :class:`ScoredPart`; entries 24c and
24d, example 24's irrigated-only score) draws panels whose metrics files hold a
score made with ``ignore`` of ``agribound.evaluate``. Per panel, the year of
its imagery comes from the metrics file's summary row (``year``): the
reference fields whose column (``irrigated_<year>``) holds the scored value
(``irrigated``) are drawn in cyan, the others, left out of the score, with a
light outline and a white hatch on a thin grey casing (lines
``LEFT_OUT_HATCH_IN`` inches apart, on one lattice of the map). The
predictions not scored are the ones ``ignore`` leaves out, found with
``agribound.evaluate``'s own code on the output taken in the reference's CRS,
as example 24 evaluates it: the representative point of the repaired polygon
lies in a left-out field (repaired too) or on its boundary. They are drawn in
dark grey (``IGNORED_COLOR``), outlined and filled at ``IGNORED_FILL_ALPHA``
under the hatch and every outline, so that one whose outline lies under the
red outline of a scored neighbour still shows; their outline lies on a lighter
grey casing (``IGNORED_CASING``), so that it shows on dark ground too (the
Sentinel-1 images). The others are outlined in red.
The renderer refuses the entry (ValueError) unless the number of scored fields
and of predictions not scored equal the metrics file's ``count_reference`` and
``count_predicted_ignored``, and the predictions scored, dropped and not
scored add up to the output's polygons. Each panel title has a second line,
"145 of 216 fields scored; F1 0.38, irrigated only" (the predictions scored,
the polygons of the output, the F1 of that score); a title of two lines adds
``TITLE_LINE_IN`` above the maps and between their rows. The legend adds the
predictions not scored and the left-out fields (with the panels' years), the
footer a line defining the score with the scored fields per year
(``ScoredPart.note``), and each layer's ``gallery_stats.json`` record a
``scored_part`` record (year, column, counts, F1). A legend too wide for two
rows takes one column.

Credits: a background's credit line is the composite's attribution, the
licence's name and "modified" (Umbra), or the author's wording in
``GALLERY_CREDITS`` (Capella: "© Capella Space Corp, CC BY 4.0, modified"); the
composites' own tags are not changed. With ``Entry.copernicus_credit`` (26,
26b and 27, whose examples' data terms give this credit for figures, 24 to
24d, which draw a Sentinel-1 image, and 28 to 28c, which draw a Sentinel-2
image) a footer line reads "Contains modified Copernicus Sentinel data
<years>", with the calendar years of the Sentinel-1 and Sentinel-2 composites
drawn or delineated on (``_copernicus_note``). ``gallery_stats.json`` holds the
credits as printed (``background_credit`` of a layer, ``copernicus_credit`` of
an entry) for the captions; a layer's ``composite.attribution`` is the
composite's own tag. A NAIP mosaic of agribound 1.1 that fills gaps from images
of other years (its ``AGRIBOUND_IMAGE_YEARS_UNDERNEATH`` tag; example 28's)
names them by date in its background line ("4 images: 2 of 2022-07-04 on top,
gaps filled from 2 of 2021-11-06"; ``_mosaic_images``).

Overview map
------------
An entry with ``Entry.overview`` (and no layers) is a map of the conterminous
US (:func:`render_overview`; EPSG:5070, Natural Earth state, country and lake
boundaries): one marker per study square of the example's summary CSV, at the
square's centre, coloured by a score of one run on the sequential blue ramp
``OVERVIEW_RAMP`` (fixed 0-1 scale, light = low) and shaped by the kind of
reference (``OVERVIEW_KINDS``: CLU, a state or federal agency inventory, DRI
boundaries). Markers that would overlap are moved apart (``_spread_markers``;
the notes give the largest shift), and ``Overview.marks`` are ringed and
numbered as the panels of a companion entry. States without a square are grey.
The map's title is a panel title (``Overview.panel_label``, the number of
squares and the median score; ``map_title`` in the stats), so that the entry's
title is the documentation's caption only. Its ``gallery_stats.json`` record
lists every square with its score (the table behind the map).

Imagery entries
---------------
An entry with ``Entry.imagery`` (an :class:`Imagery`, and no layers) draws its
reference on several images in one window, with no model output
(:func:`render_imagery`). Entry 25c draws example 25's Montana square
(Glacier County) with its reference fields, all USDA FSA Common Land Units, on
the NAIP of 2005 and on the 2023 composite that Delineate-Anything read there,
in the 5 km square. Each panel (:class:`ImagePanel`) names its image as a
layer names its background: a GeoTIFF (``ImagePanel.background``) or the
raster that an output's provenance names (``ImagePanel.background_from``),
under the panel's own run root (``ImagePanel.run_root``, relative to the
repository) or else the entry's; an explicit ``--run-root`` replaces both.
Every image must cover the entry's window (``Entry.window``), and all must be
in one CRS. The legend counts the reference polygons drawn. Under the maps
are each image (its source, pixel size, window and number of images from its
tags, as for the other entries, then ``ImagePanel.bg_role``) and the entry's
own lines (``Imagery.notes``: what the images show and the data credits). An
image exported by ``tools/fetch_naip_history.py`` comes with the record of
that export (``ImagePanel.record``): the renderer refuses the panel unless the
record describes the file drawn (its name and SHA-256), adds the dates of the
record's images to the panel's line, and copies the record's images (asset
IDs and dates), bands and pixel aggregation into the panel's
``gallery_stats.json`` record (``panels``).

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
representative points of the layer's polygons among candidates half a side
apart; with ``Entry.crop_on_reference``, of the panel's reference polygons
(``Layer.reference``, drawn in cyan) when it has any. Their inset
is a world locator (``WORLD_INSET_W_IN``, ``WORLD_INSET_PAD_DEG``, or
``Entry.inset_pad_deg``) with one
numbered red dot per study area; India is drawn from the Survey of India outline as
above, and the credit line starts "Inset: study areas 1-N; ". Panels with the same
``Layer.area`` (several models on one study area) share a number, a dot and the
window of the area's first panel, and a panel of an area that names no
``Layer.reference`` draws the area's reference. ``Layer.window`` gives the
area's window instead of choosing it (the SAM entry: each evaluation square in
full); it must lie inside the imagery of every panel of the area. A layer read
under another run root than the entry's names it (``Layer.run_root``, relative
to the repository, for all the layer's files; its ``gallery_stats.json`` record
gives it as ``run_root``). Each layer's ``window`` record in
``gallery_stats.json`` says how the window was chosen: its side (``side_m``),
the number of candidate squares (``candidates``), what was counted in them
(``most_of``) and, with ``Layer.window_within``, the polygon file the
candidates had to lie in (``inside``; if none does, all the candidates are
used, with a warning), or that it was given (``given``). All the panel titles
of a figure have one size, the largest at which every title fits its panel; a
title of two lines adds ``TITLE_LINE_IN`` above the maps and between their rows.

Footer
------
Under the maps are the legend and the notes (imagery window, model and
version, inset credit) at ``NOTE_FS`` pt, one line every ``NOTE_LINE_IN``
inches. The notes are wrapped by their measured width so that they stay clear
of the inset and its title, with balanced lines (the fewest lines, then the
narrowest width that keeps that number, so no line is left with one word).
The footer grows with the number of lines; after drawing, the tool checks
that the legend, the notes and the inset do not overlap.

Outputs (``--out-dir``, default ``assets/gallery_1.1``):

- ``<name>.png``: one image per entry, 3000 px wide (10 in at 300 dpi), with
  the imagery window and the model version printed under the panels;
- ``preview/<name>.webp``: a 1600 px wide preview of each PNG for the web
  pages, which embed the preview and link it to the full PNG (Pillow, RGB,
  Lanczos resampling, WebP quality 85, method 6, no metadata);
- ``gallery_stats.json``: per entry, the facts the captions quote (polygon
  counts, area quantiles, source, year, imagery window, engine metadata, the
  inset location (``location``: lon/lat, country, admin1; for a multi-area
  entry, one record per study area and each layer's ``window``), the squares a
  single-area entry draws (``crop_window`` for the ``crop_m`` square,
  ``zoom_window`` for the zoom panel: CRS, bounds, centre lon/lat and side;
  ``zoom_window_m`` is the zoom side alone), whether PROJ could download
  transformation grids while the entry was rendered (``proj_network``; see
  below) and, when the example wrote one, its metrics file). The layer counts
  are of the whole
  output file. ``reference`` counts the reference polygons that intersect the
  map (the crop square, when the entry is cropped); in a multi-area entry,
  each layer's ``reference`` describes the panel's whole reference file, and
  ``n_polygons_in_window`` counts those that intersect the panel's square. A
  full run writes the file afresh; ``--only`` replaces the rendered entries
  and keeps the others.

PROJ network access: :func:`main` enables it while it renders
(``pyproj.network.set_network_enabled``, as ``PROJ_NETWORK=ON`` does; the
previous setting is restored afterwards), because reprojections between
datums use PROJ's downloadable grids only with it. Without them the NAD83
outlines of V18 in entry 20 were drawn about 0.3 m away (124,255 pixels of
the PNG differed). ``--no-proj-network`` keeps the setting of the
environment.

Restricted references
---------------------
Reference data whose terms keep their outlines out of published images can
be listed in ``RESTRICTED_REFERENCES`` (names matched case-insensitively
anywhere in a path): :func:`render` and :func:`render_areas` refuse an entry
any of whose files (layers, references, overlay, ``removed_vs``,
backgrounds, ``model_from``, metrics) names one of them. Outputs whose study
squares must not be placed can be listed in ``LOCAL_SQUARES``: such an entry
is refused with an explicit ``window`` or ``mark_windows`` or with the
locator inset (``inset=True``), all of which would place a square, and its
``gallery_stats.json`` record leaves out the coordinates (the lon/lat of
``location``, each panel's ``window``, ``crop_window`` and ``zoom_window``)
and keeps the counts and scores
(``n_polygons``, ``metrics``). Each path is checked as written, normalised
(``..`` and ``//``) and as the files it names under the run root (a layer's
own run root too), with globs expanded and symlinks resolved, as the renderer
reads them. Both lists are empty at present: the OpenET field boundaries (CLU
fields included: New York in examples 28, Tennessee in the SAM entry), the
DRI/NMOSE field boundaries (V18) and the DRI field boundaries of Nevada
(examples 24 and 25) may all be drawn, and their squares placed.

Reference minimum area
----------------------
With ``Entry.reference_min_area_m2``, the reference polygons smaller than that
area (EPSG:6933, agribound's ``filter_polygons``) are dropped before they are
drawn, counted or used to place a crop square, as the examples drop them before
they evaluate, so that an image shows the reference that was evaluated. The
New Mexico entries that draw the whole V18 layer (20, 20b, 20c and 13) use
2,500 m², the predictions' minimum area; ``gallery_stats.json`` records the
value as ``reference.min_area_m2``.

The rendering is deterministic (no random numbers; no dates or software
versions in the PNG metadata; the previews are encoded from the PNG with fixed
settings), so a second run with the same library versions gives byte-identical
files. Other versions can move pixels: geopandas 1.2 draws each multipart
country as one shape, which changes the coastlines of the locator insets
slightly (the maps are unchanged).
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import math
import posixpath
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
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.legend_handler import HandlerBase  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.merge import merge as rio_merge  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agribound._version import __version__  # noqa: E402
from agribound.evaluate import proj_network_enabled  # noqa: E402
from agribound.io.raster import percentile_stretch_uint8  # noqa: E402
from agribound.provenance import to_jsonable  # noqa: E402
from agribound.registry import SOURCE_REGISTRY  # noqa: E402

PRED_COLOR = "#ff2d2d"
PRED_LABEL = f"agribound {__version__} fields"  # the release the gallery documents
REF_COLOR = "#00e5ff"
ZOOM_COLOR = "#ffd400"
HIGHLIGHT_COLOR = "#ff9d00"
HIGHLIGHT_LABEL = "Refined by SAM"
REMOVED_COLOR = "#ff3df2"
REMOVED_LABEL = "Removed by the crop filter"
# Entry.scored_part (entries 24c and 24d): the predictions a score did not count are drawn in
# dark grey, outlined and filled at IGNORED_FILL_ALPHA under the hatch and every outline, so
# that one whose outline lies under the red outline of a scored neighbour still shows; the
# dark grey outline lies on a lighter grey casing (IGNORED_CASING, IGNORED_CASING_LW points
# wider; above the image, the hatch and the left-out outlines, under the cyan and red
# outlines), so that it also shows on dark ground, such as the Sentinel-1 images and the
# darkest NAIP fields (the legend sample draws it too). The reference fields it left out get a
# light outline and a white hatch, with lines LEFT_OUT_HATCH_IN inches apart on the image,
# each on a thin grey casing so that it shows on bright ground too (the legend sample draws
# them on grey).
IGNORED_COLOR = "#454545"
IGNORED_FILL_ALPHA = 0.35
IGNORED_CASING = "#9a9a9a"
IGNORED_CASING_ALPHA = 0.8
IGNORED_CASING_LW = 1.0  # points added to the outline's width; 0: no casing
LEFT_OUT_EDGE = "#e6e6e6"
LEFT_OUT_HATCH = "white"
LEFT_OUT_HATCH_ALPHA = 0.7
LEFT_OUT_HATCH_LW = 0.45  # points
LEFT_OUT_CASING = "#5a5a5a"
LEFT_OUT_CASING_ALPHA = 0.45
LEFT_OUT_CASING_LW = 1.2  # points
LEFT_OUT_HATCH_IN = 0.07
LEFT_OUT_SWATCH = "#9a9a9a"
FIG_WIDTH_IN = 10.0
# 3000 px wide full images, for click-through; the docs embed PREVIEW_W px previews
# at 800 px, i.e. 2 px per CSS pixel.
DPI = 300
PREVIEW_W = 1600
PREVIEW_WEBP = {"quality": 85, "method": 6}
MAX_PANEL_READ_SIDE = 6000  # cap on the pixels read per panel side (memory)
TITLE_IN = 0.34
TITLE_FS = 10.5
# Each further line of a panel title (Entry.scored_part titles have two) adds this height
# above the maps and between the map rows (matplotlib's line spacing, 1.2).
TITLE_LINE_IN = TITLE_FS * 1.2 / 72
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
# Reference data that published images must not draw: an entry is refused when any of its
# files names one of these (case-insensitive; module docstring). None at present.
RESTRICTED_REFERENCES: tuple[str, ...] = ()
# Outputs whose study squares must not be placed: their entries take no explicit window and
# no locator inset, and their gallery_stats.json records leave out every coordinate. None at
# present.
LOCAL_SQUARES: tuple[str, ...] = ()
# Default run root and gallery folder: the re-run of the examples that follows 1.0.1. The
# 1.0 gallery (assets/gallery_1.0, from outputs/examples_1.0) is kept (module docstring).
RUN_ROOT = REPO / "outputs" / "examples_1.1"
OUT_DIR = REPO / "assets" / "gallery_1.1"

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
    "landsat-pan": "Landsat PAN TOA",  # with the recorded sensors: see _landsat_pan_label
    "hls": "HLS",
    "naip": "NAIP",
    "usgs-naip-plus": "USGS NAIP Plus",
    "spot": "SPOT 6/7",
    "spot-pan": "SPOT 6/7 panchromatic",
    "sentinel1": "Sentinel-1 GRD (VV, VH, VV-VH)",  # with the orbit pass: see _sentinel1_label
    "umbra": "Umbra GEC X-band SAR",  # with the credit line (_credit); _bg_note adds the method
    "capella": "Capella GEO X-band SAR",
    "local": "local GeoTIFF",
}
# Credit lines printed for a source's background in place of its composite's attribution,
# licence and "modified" (_credit): the author's wording (2026-10-11). The composite tags keep
# the provider's own credit ("© Capella Space Corp, All Rights Reserved.").
GALLERY_CREDITS = {
    "capella": "© Capella Space Corp, CC BY 4.0, modified",
}
# Entry.copernicus_credit: the sources whose composites contain Copernicus Sentinel data.
COPERNICUS_SOURCES = ("sentinel1", "sentinel2")
LANDSAT_MISSIONS = {"LT04": 4, "LT05": 5, "LE07": 7, "LC08": 8, "LC09": 9}
# Crop-type maps of the crop-sequence engine (engine_meta "dataset"), for its model note.
CROP_SEQUENCE_DATASETS = {
    "aci": "AAFC Annual Crop Inventory",
    "cdl": "USDA Cropland Data Layer",
}


def _count(n: int, noun: str) -> str:
    """``"1 field"``, ``"1,185 fields"``: a count with its noun in the singular or plural."""
    n = int(n)
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


def _and_join(items) -> str:
    """``["a"]`` gives ``"a"``, ``["a", "b", "c"]`` gives ``"a, b and c"``."""
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _panel_names(n: int, ncols: int | None = None) -> list[str]:
    if n == 4 and ncols == 2:
        return ["Top left", "Top right", "Bottom left", "Bottom right"]
    if n == 2:
        return ["Left", "Right"]
    if n == 3:
        return ["Left", "Middle", "Right"]
    return [f"Panel {i + 1}" for i in range(n)]


def _or_join(items) -> str:
    """``["a"]`` gives ``"a"``, ``["a", "b", "c"]`` gives ``"a, b or c"``."""
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " or " + items[-1]


def _year_list(years) -> str:
    """Years (or other whole numbers), consecutive ones as a range: {2022, 2024} gives "2022
    and 2024", {2023, 2024, 2025} gives "2023-2025", [1, 2, 3, 4] gives "1-4"."""
    runs: list[list[int]] = []
    for y in sorted({int(y) for y in years}):
        if runs and y == runs[-1][1] + 1:
            runs[-1][1] = y
        else:
            runs.append([y, y])
    return _and_join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def _composite_years(comp: dict) -> set[int]:
    """Calendar years of a composite's window (``_composite_facts``), from its first day to
    the day before its exclusive end; else the years of its ``image_years``."""
    import datetime as dt

    start, end = comp.get("date_start"), comp.get("date_end_exclusive")
    if start and end:
        try:
            first = dt.date.fromisoformat(str(start)[:10])
            last = dt.date.fromisoformat(str(end)[:10]) - dt.timedelta(days=1)
            return set(range(first.year, last.year + 1))
        except ValueError:
            pass
    return {int(y) for y in re.findall(r"\b(?:19|20)\d\d\b", str(comp.get("image_years") or ""))}


def _copernicus_note(items) -> str | None:
    """Copernicus credit line for ``(source, composite facts)`` pairs (the backgrounds drawn
    and the panels' own inputs): "Contains modified Copernicus Sentinel data <years>" when one
    of them is a Sentinel-1 or Sentinel-2 composite (``COPERNICUS_SOURCES``), else None."""
    comps = [comp for source, comp in items if source in COPERNICUS_SOURCES]
    if not comps:
        return None
    years = set().union(*(_composite_years(c) for c in comps))
    return "Contains modified Copernicus Sentinel data" + (f" {_year_list(years)}" if years else "")


def _credit(source: str | None, comp: dict) -> str:
    """Credit line of a background (empty when it needs none).

    ``GALLERY_CREDITS`` when it holds the source; else the composite's attribution
    (``AGRIBOUND_ATTRIBUTION``: Umbra), the licence's name when the attribution does not hold
    it (CC BY 4.0 section 3(a)(1)(C); the licence URL, ``AGRIBOUND_LICENSE_URL``, is kept in the
    facts for the page text) and the note that the data were modified.
    """
    if source in GALLERY_CREDITS:
        return GALLERY_CREDITS[source]
    if not comp.get("attribution"):
        return ""
    parts = [comp["attribution"]]
    if comp.get("license") and comp["license"] not in comp["attribution"]:
        parts.append(comp["license"])
    return "; ".join([*parts, "modified"])


def _image_noun(source: str | None) -> str:
    """What a composite's ``AGRIBOUND_N_IMAGES`` counts: acquisitions for ``sentinel1``.

    Earth Engine stores a Sentinel-1 acquisition as several GRD slices; the
    builder counts each acquisition once. The Umbra and Capella composites
    count collects.
    """
    if source == "sentinel1":
        return "acquisition"
    return "collect" if source in ("umbra", "capella") else "image"


def _fit_title(title: str, width_in: float) -> float:
    """Font size at which *title* fits *width_in* inches (TITLE_FS, smaller down to 7 pt).

    A title of several lines fits when its widest line does.
    """
    fs = TITLE_FS
    lines = title.split("\n")
    while fs > 7 and max(_text_width_in(line, fs) for line in lines) > width_in * 0.98:
        fs -= 0.5  # a long title is set smaller, not over the next panel
    return fs


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


def _landsat_pan_label(sensors: str | None) -> str:
    """Landsat PAN label from a composite's ``AGRIBOUND_SENSORS`` tag.

    ``"LC08,LC09"`` gives ``"Landsat 8/9 PAN TOA"``; without the tag (composites made
    before it existed) the label is ``"Landsat PAN TOA"``.
    """
    ids = [s.strip() for s in (sensors or "").split(",")]
    missions = sorted({LANDSAT_MISSIONS[i] for i in ids if i in LANDSAT_MISSIONS})
    if not missions:
        return SOURCE_LABELS["landsat-pan"]
    return f"Landsat {'/'.join(str(m) for m in missions)} PAN TOA"


def _sentinel1_label(orbit_pass: str | None) -> str:
    """Sentinel-1 label from a composite's ``AGRIBOUND_ORBIT_PASS`` tag.

    ``"DESCENDING"`` gives ``"Sentinel-1 GRD descending (VV, VH, VV-VH)"``; without
    a recognisable pass the label is ``"Sentinel-1 GRD (VV, VH, VV-VH)"``.
    """
    orbit = (orbit_pass or "").strip().upper()
    if orbit not in ("ASCENDING", "DESCENDING"):
        return SOURCE_LABELS["sentinel1"]
    return f"Sentinel-1 GRD {orbit.lower()} (VV, VH, VV-VH)"


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
    # Per-panel background from another output's provenance (facts.raster_path), e.g. the
    # Sentinel-2 composite of the same year for a panel of embedding clusters.
    background_from: str | None = None
    background_source: str | None = None
    bg_role: str | None = None
    crop_m: float | None = None  # multi-area entries: this panel's window side (else the entry's)
    # Multi-area entries: this panel's reference polygons, drawn in cyan (Entry.reference_label).
    reference: str | None = None
    # Multi-area entries: panels with the same area share a number and one locator dot (e.g.
    # several models on one study area); default: every panel is its own area.
    area: str | None = None
    # The example's metrics file of this output (examples 24-27: metrics_<stem>.json): its
    # summary row and scalar scores go into the layer's gallery_stats.json record, and with
    # Entry.title_f1 the panel title gives its F1.
    metrics: str | None = None
    noun: str = "field"  # what the panel title counts ("zone" for crop-history zones)
    # Multi-area entries: a polygon file (e.g. the evaluation area); the area's window is the
    # square with the most polygons among the candidates inside it (all candidates, with a
    # warning, if none is).
    window_within: str | None = None
    # Multi-area entries: the area's window as given, (centre x, centre y, side in metres) in the
    # CRS of the layer's background (e.g. an evaluation square), instead of the densest square;
    # it must lie inside the imagery of every panel of the area.
    window: tuple[float, float, float] | None = None
    # Multi-area entries: the run root of this layer's files (path, reference, metrics, ...),
    # relative to the repository, when it differs from the entry's (e.g. a comparator from an
    # example's run beside an evaluation's own output); an explicit --run-root replaces it.
    run_root: str | None = None
    # The keys of the record to read in a metrics file that holds several records (e.g. the
    # SAM evaluation's scores.json: ("E1/DV-E/spot", "sam-auto", "own")); the record's scalar
    # scores give the F1 (default: the examples' {"row": ..., "metrics": ...} files).
    metrics_record: tuple[str, ...] | None = None


def _metrics_of(path: str) -> str:
    """The metrics file that examples 24-27 write beside an output: ``metrics_<stem>.json``."""
    folder, _, name = path.rpartition("/")
    metrics = f"metrics_{name.removesuffix('.gpkg')}.json"
    return f"{folder}/{metrics}" if folder else metrics


@dataclass
class Overview:
    """A map of the conterminous US with one marker per study square (:func:`render_overview`).

    The markers are coloured by a score of one run in the example's summary CSV (one row per
    square and run, with ``state``, ``centre_lon``, ``centre_lat``, ``sources`` and
    ``status``) and shaped by the kind of reference (``OVERVIEW_KINDS``).
    """

    summary: str  # summary CSV under the run root
    run: str  # the rows whose run_column holds this value colour the markers
    metric_label: str  # colour bar label
    run_column: str = "imagery"
    metric: str = "f1"
    marks: list[str] = field(default_factory=list)  # states numbered 1..n (a companion entry)
    notes: list[str] = field(default_factory=list)  # footer lines: definitions and credits
    # The map's title as a panel title: this label, then "(<n> squares; median F1 0.31)"; the
    # entry's title is the documentation's caption. None: the entry's title.
    panel_label: str | None = None


@dataclass
class ImagePanel:
    """One panel of an imagery entry (``Entry.imagery``; :func:`render_imagery`): an image,
    named as a layer names its background, with the entry's reference drawn on it."""

    label: str  # the panel title
    background: str | None = None  # a GeoTIFF (a glob that matches one file)
    background_from: str | None = None  # or the raster that this output's provenance names
    background_source: str | None = None  # the image's source; default: the output's
    bg_role: str | None = None  # what the image is: its footer line ends with it
    # The record of tools/fetch_naip_history.py beside the image: the panel is refused unless
    # it describes the file drawn (name and SHA-256).
    record: str | None = None
    # Run root of the panel's files, relative to the repository (default: the entry's); an
    # explicit --run-root replaces it.
    run_root: str | None = None


@dataclass
class Imagery:
    """Panels of one reference on several images in one window, with no model output
    (``Entry.imagery``; :func:`render_imagery`)."""

    panels: list[ImagePanel]
    notes: list[str] = field(default_factory=list)  # footer lines: what is shown, credits


@dataclass
class ScoredPart:
    """Panels scored against part of their reference (``Entry.scored_part``; :func:`render`).

    Each panel's metrics file (``Layer.metrics``) holds a score made with ``ignore`` of
    ``agribound.evaluate``, and its summary row the year of the panel's imagery (``year``).
    The reference fields whose *column* (``{year}`` replaced by that year) holds *value* are
    the scored fields, drawn in cyan (``Entry.reference_label``); the others were left out
    of the score and get a light outline and a white hatch. The predictions ``ignore`` left
    out (found with evaluate's own code on the output in the reference's CRS, as example 24
    evaluates it; :func:`_not_scored`) were not scored and are outlined and filled in grey;
    the others are outlined in red. The numbers of scored fields and of predictions not
    scored must equal the metrics file's ``count_reference`` and ``count_predicted_ignored``.
    """

    column: str  # reference column of the scored fields; "{year}": the panel's year
    value: str  # the column's value of the scored fields
    left_out_label: str  # legend label of the left-out fields; "{years}": the panels' years
    ignored_label: str  # legend label of the predictions not scored
    title: str  # after the F1 in the panel titles, e.g. "irrigated only"
    # Footer line defining the score; "{counts}": the scored fields per year, e.g. "39 of 74
    # in 2022, 40 in 2023 and 34 in 2024" (_scored_counts).
    note: str = ""


@dataclass
class Entry:
    """One gallery image."""

    key: str
    name: str
    title: str
    layers: list[Layer]
    reference: str | None = None  # drawn in cyan on every panel
    reference_label: str = "Reference"
    # Drop the reference polygons (Entry.reference and Layer.reference) smaller than this
    # (m², EPSG:6933), as the example's evaluation does; 0 keeps them all.
    reference_min_area_m2: float = 0.0
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
    sam2_note: bool = False  # add the SAM 2 model line (SAM layer notes name the engine only)
    sam2_resmoothed: bool | str = (
        False  # True: re-smooths all polygons after SAM (13); str: which (15)
    )
    halo: bool = False  # white halo under the outlines, for small fields on busy imagery
    inset: bool = True  # locator inset (country, state/province, study-area marker)
    # Multi-area entries: (lon, lat) margin of the world locator around the dots, when the
    # default WORLD_INSET_PAD_DEG leaves a number on the frame (a wide, short view).
    inset_pad_deg: tuple[float, float] | None = None
    line_width: float = 0.7
    notes: dict = field(default_factory=dict)
    title_f1: bool = False  # panel titles give the F1 of each layer's metrics file
    pred_label: str | None = None  # legend label of the red outlines (default PRED_LABEL)
    # With one background for every panel: a footer line names the input image of each panel
    # that was delineated on another composite (its source, window and image count). No entry
    # uses it at present (24 and 24b did while their panels were all drawn on NAIP).
    input_notes: bool = False
    # A footer line "Contains modified Copernicus Sentinel data <years>", the years of the
    # Sentinel-1 and Sentinel-2 composites drawn or delineated on (the credit that the data
    # terms of examples 26 and 27 give for figures; entries 24-24d draw a Sentinel-1 image).
    copernicus_credit: bool = False
    # Entries whose panels come from different models: each model line names its panels, e.g.
    # "(top right)", or in a multi-area entry its study areas, e.g. "(study area 2)" (two
    # fine-tuned checkpoints of one model read alike but for the hash).
    model_positions: bool = False
    # Run root of the entry's outputs, relative to the repository (default RUN_ROOT); an
    # explicit --run-root replaces it for every entry (module docstring, "Run roots").
    run_root: str | None = None
    # A map of the US with one marker per study square instead of map panels (no layers).
    overview: Overview | None = None
    # Panels scored against part of the reference (ScoredPart): the scored and the left-out
    # fields, and the predictions scored and not scored, are drawn differently; each panel
    # title has a second line with the counts and the F1 of that score.
    scored_part: ScoredPart | None = None
    # The reference on several images in Entry.window, with no model output (no layers).
    imagery: Imagery | None = None


# Output directory of example 23 (tree crops). Its entries (23, 23b-23e) read the final run
# (FINAL_RUN_ROOT): example 23 with --s2-single-date, whose three TESSERA runs were made again
# (Twifo's is drawn in 23b); every other output they draw is a copy of the 1.1 run's.
TC = "outputs/tree_crops"

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

# ---------------------------------------------------------------- examples 24 to 27
# Their panels are chosen by rules applied to the examples' summary files, not by
# appearance; tests/unit/test_make_gallery.py re-applies each rule to the run's summary.

# Example 24 (Nevada), entries 24 (Diamond Valley) and 24b (Mason Valley). Panels per basin,
# from the rows of outputs/nevada_field_boundaries/summary.csv with status "ok" (F1: one-to-one
# at IoU >= 0.5 against the DRI-mapped fields of the 5 km evaluation square; _nevada_panels):
#   1. NAIP, Delineate-Anything v2 as released (the baseline, as in example 25);
#   2. the other NAIP run with the highest F1 (DA or DINOv3 fine-tuned, DA + SAM 2);
#   3. the SPOT 6/7 run with the highest F1;
#   4. the Sentinel-1 run (Delineate-Anything v2 as released, its only run);
# ties go to the earlier row. The "-opt" runs of example 24's --options (labels ending in
# NEVADA_OPT_SUFFIX) never compete for these panels: they are the opt-in options before and
# after, shown in their own entry (24e), with panels fixed before they ran (PREREG-DV3 section
# 7; round-2 PLAN 2.5: "Entry 24, text only (no re-render)" for the options). The direct SAM
# runs of --sam-auto ("<source> sam-auto") compete like the other runs of their source (round-2
# PLAN 1.5: "its existing panel rule ... decides whether a SAM panel appears in entries 24-24d").
# The entries read example 24's run with --options and --sam-auto (EX24_RUN_ROOT,
# outputs/examples_1.1_ex24b; 28 runs reused from outputs/examples_1.1_ex24 unchanged, 6
# sam-auto runs and 8 "-opt" runs). There the rule picks: Diamond Valley naip da 0.255, naip
# sam-auto 0.387 (naip dinov3-ft 0.380), spot sam-auto 0.589 (spot dinov3-ft 0.500), sentinel1
# da 0.346; Mason Valley naip da 0.567, naip dinov3-ft 0.738 (naip sam-auto 0.618), spot
# sam-auto 0.596 (spot da 0.513), sentinel1 da 0.453. Ranking the "-opt" rows too would give
# naip dinov3-ft-opt both NAIP panels (0.481 and 0.771). In the 1.1 run (outputs/examples_1.1,
# no sam-auto or "-opt" rows; the 1.1 gallery) it picked naip dinov3-ft and spot dinov3-ft
# (Diamond Valley), naip dinov3-ft and spot da (Mason Valley). Each panel is drawn on the
# composite it was delineated on (Entry.per_layer_background, as in entries 26 and 23-23e):
# NAIP 1 m 2022 in the top row, SPOT 6/7 6 m 2023 at bottom left, Sentinel-1 10 m 2024 at
# bottom right; the footer gives each panel's composite and the Copernicus line of the
# Sentinel-1 image.
NV = "outputs/nevada_field_boundaries"
NEVADA_RUNS = {  # summary.csv run: (output file stem, panel label)
    "naip da": ("fields_naip_da_2022", "NAIP 1 m, 2022: DA v2 as released"),
    "naip da-sam2": ("fields_naip_da-sam2_2022", "NAIP 1 m, 2022: DA v2 + SAM 2"),
    "naip da-ft": ("fields_naip_da-ft_2022", "NAIP 1 m, 2022: DA v2 fine-tuned"),
    "naip dinov3-ft": ("fields_naip_dinov3-ft_2022", "NAIP 1 m, 2022: DINOv3 fine-tuned"),
    "naip sam-auto": ("fields_naip_sam-auto_2022", "NAIP 1 m, 2022: sam-auto"),
    "spot da": ("fields_spot_da_2023", "SPOT 6/7 6 m, 2023: DA v2 as released"),
    "spot da-ft": ("fields_spot_da-ft_2023", "SPOT 6/7 6 m, 2023: DA v2 fine-tuned"),
    "spot dinov3-ft": ("fields_spot_dinov3-ft_2023", "SPOT 6/7 6 m, 2023: DINOv3 fine-tuned"),
    "spot sam-auto": ("fields_spot_sam-auto_2023", "SPOT 6/7 6 m, 2023: sam-auto"),
    "sentinel1 da": ("fields_sentinel1_da_2024", "Sentinel-1 10 m, 2024: DA v2 as released"),
    # Never a panel (the rule has no Landsat slot); listed so that every run of --sam-auto
    # (NAIP, SPOT 6/7 and Landsat 8/9) is named.
    "landsat sam-auto": ("fields_landsat_sam-auto_2024", "Landsat 30 m, 2024: sam-auto"),
}
NEVADA_OPT_SUFFIX = "-opt"  # example 24's --options runs: left out of the panel rule
NEVADA_PANELS = {  # the rule's runs in EX24_RUN_ROOT
    "diamond_valley": ["naip da", "naip sam-auto", "spot sam-auto", "sentinel1 da"],
    "mason_valley": ["naip da", "naip dinov3-ft", "spot sam-auto", "sentinel1 da"],
}
NEVADA_PANEL_NOTE = (
    "Panels: Delineate-Anything v2 as released on NAIP (top left), then the highest F1 "
    "(one-to-one at IoU >= 0.5 against the DRI-mapped fields of the square) among example 24's "
    "other NAIP runs, its SPOT 6/7 runs and its Sentinel-1 run, with the sam-auto runs of "
    '--sam-auto and without the "-opt" runs of --options (entry 24e)'
)
NEVADA_FINE_TUNE_NOTE = (
    "Fine-tuned: DINOv3 ViT-L/16 (SAT-493M weights), full fine-tuning for up to 20 epochs on "
    "the {sources} composite of a 5 km training square of the basin {where}, never on the "
    "evaluated fields"
)
NEVADA_FINE_TUNE_WHERE = {  # basin: where its training square lies (example 24's BASINS)
    "diamond_valley": "5.0 km south of the evaluated one (54 DRI fields)",
    "mason_valley": "16.0 km north of the evaluated one (179 DRI fields reach into it)",
}
NEVADA_SOURCE_NAMES = {"naip": "NAIP", "spot": "SPOT"}  # in the fine-tuning line
# The sam-auto runs: no field labels and no fine-tuning (SAM 2.1 as released); the engine's
# per-source settings were chosen on example 24's two training squares by mean F1 among 72
# settings per source, by the rule of round-2 sam-direct (DESIGN.md 5.1 and 6.1,
# runs/selection.json): settings within 0.01 of the highest mean F1 tie, then the fewest SAM
# forward passes per km2, then the higher mean boundary F1. The NAIP setting has the highest
# mean F1 of its 72 (0.446); for SPOT 6/7 ten settings tied at the same cost and boundary F1
# (0.605 against 0.603) chose one of mean F1 0.578, the highest being 0.587. So the
# evaluation squares are held out from that choice but lie in the same basins (PREREG-SAM
# section 7; critique F5). The runs repeat the SAM evaluation's E1 runs (identical layers and
# scores; round-5 ex24b NOTES 3.4).
NEVADA_SAM_AUTO_NOTE = (
    "sam-auto ({panels}): no field labels and no fine-tuning (SAM 2.1 as released); the "
    "engine's settings were chosen on example 24's two training squares (by mean F1 among 72 "
    "settings per source; settings within 0.01 of the highest tied, then the fewest SAM passes "
    "per km², then the higher boundary\u00a0F1), one of them {where}: held out from that "
    "choice, not from the landscape"
)
NEVADA_SAM_AUTO_WHERE = {  # basin: where the square the settings were chosen on lies
    "diamond_valley": "5.0 km south of this square",
    "mason_valley": "16.0 km north of this square",
}
NEVADA_CREDIT = "DRI field boundaries (Nevada, 2026-08-20), Desert Research Institute"


def _nevada_notes(basin: str, runs: list[str]) -> list[str]:
    """The lines on the fine-tuned and the sam-auto panels of a 2 x 2 figure of *basin* that
    draws *runs* (summary.csv run labels, in panel order): the DINOv3 fine-tuning line, by
    source and panel, and the sam-auto line, each only when such a panel is drawn."""
    names = [n.lower() for n in _panel_names(len(runs), 2)]
    tuned: dict[str, list[str]] = {}  # source -> the panels of its fine-tuned checkpoint
    sam = []
    for name, run in zip(names, runs, strict=True):
        source, label = run.split(" ", 1)
        if label.startswith("sam-auto"):
            sam.append(name)
        elif "-ft" in label:
            if not label.startswith("dinov3-ft"):  # the line describes DINOv3 only
                raise ValueError(f"no fine-tuning line for the run {run!r}")
            tuned.setdefault(source, []).append(name)
    notes = []
    if tuned:
        sources = " or ".join(
            f"{NEVADA_SOURCE_NAMES[s]} ({_and_join(where)})" for s, where in tuned.items()
        )
        notes.append(
            NEVADA_FINE_TUNE_NOTE.format(sources=sources, where=NEVADA_FINE_TUNE_WHERE[basin])
        )
    if sam:
        notes.append(
            NEVADA_SAM_AUTO_NOTE.format(panels=_and_join(sam), where=NEVADA_SAM_AUTO_WHERE[basin])
        )
    return notes


# Entries 24c (Diamond Valley) and 24d (Mason Valley): the runs of entries 24 and 24b, each on
# its own input image as there, with example 24's irrigated-only score
# (metrics_<stem>_irrigated.json): the reference is the fields irrigated in the year of the
# panel's imagery (IrrMapper share >= 0.5; NAIP 2022, SPOT 2023, Sentinel-1 2024), and a
# prediction whose representative point lies in another field of the square was not scored
# (ignore of agribound.evaluate). These files were first written by the re-scoring run of
# example 24 (outputs/examples_1.1_ex24: the outputs of outputs/examples_1.1, copied; no model
# run again); the run with --options and --sam-auto (outputs/examples_1.1_ex24b, a clone of it
# with byte-identical metrics files for the 28 reused runs) adds those of its new runs, and the
# five Nevada entries (24-24e) read it.
# Ranked by the irrigated-only F1 (summary column f1_irrigated), the rule above picks the same
# runs in both basins (the tests check it): Diamond Valley naip da 0.380, naip sam-auto 0.576
# (naip dinov3-ft 0.565), spot sam-auto 0.817 (spot dinov3-ft 0.707), sentinel1 da 0.584; Mason
# Valley naip da 0.578, naip dinov3-ft 0.770, spot sam-auto 0.615 (spot da 0.532), sentinel1 da
# 0.468.
EX24_RUN_ROOT = "outputs/examples_1.1_ex24b"
NEVADA_IRRIGATED = ScoredPart(
    column="irrigated_{year}",
    value="irrigated",
    left_out_label="Not irrigated in the panel's year ({years}): left out of the score",
    ignored_label=f"{PRED_LABEL} not scored: representative point in a left-out field",
    title="irrigated only",
    note=(
        "Irrigated only: the reference is the DRI-mapped fields irrigated in the year of the "
        "panel's imagery (IrrMapper share of the field at least 0.5; of the square's fields, "
        "{counts}); a prediction whose representative point lies in another field of the "
        "square is not scored, and every other prediction is, so predictions on ground the "
        "inventory does not map count as false positives. F1: one-to-one at IoU >= 0.5; "
        "counts: the predictions scored and the polygons of the output. Red outlines are "
        "drawn over grey ones where they coincide"
    ),
)
NEVADA_IRRIGATED_PANEL_NOTE = (
    "Panels: the runs of the all-fields figure (Delineate-Anything v2 as released on NAIP at "
    "top left, then the highest F1 against all the DRI-mapped fields of the square among "
    "example 24's other NAIP runs, its SPOT 6/7 runs and its Sentinel-1 run, with the sam-auto "
    'runs and without the "-opt" runs); the irrigated-only F1 picks the same runs'
)


def _nevada_panels(rows: list[dict], basin: str, score: str = "f1") -> list[str]:
    """The runs of the panels of *basin* (example 24) by the rule above, from summary rows.

    *score* is the summary column ranked: ``f1`` (the rule of the entries), or
    ``f1_irrigated`` (the irrigated-only F1). The "-opt" runs (labels ending in
    :data:`NEVADA_OPT_SUFFIX`) are left out of the ranking.
    """
    ok = [
        r
        for r in rows
        if r["basin"] == basin and r["status"] == "ok" and not r["run"].endswith(NEVADA_OPT_SUFFIX)
    ]

    def best(keep) -> str | None:
        runs = [r for r in ok if keep(r)]
        return max(runs, key=lambda r: float(r[score]))["run"] if runs else None

    picks = [
        "naip da",
        best(lambda r: r["source"] == "naip" and r["run"] != "naip da"),
        best(lambda r: r["source"] == "spot"),
        best(lambda r: r["source"] == "sentinel1"),
    ]
    return [p for p in picks if p]


def _nevada_entry(
    key: str, name: str, basin: str, title: str, square, irrigated: bool = False
) -> Entry:
    """Entry of one basin of example 24: four panels in the evaluation *square*, each on the
    composite it was delineated on, read from ``EX24_RUN_ROOT``.

    With *irrigated*, the same panels with the irrigated-only score (``NEVADA_IRRIGATED``).
    """
    layers = []
    for run in NEVADA_PANELS[basin]:
        stem, label = NEVADA_RUNS[run]
        path = f"{NV}/{basin}/{stem}.gpkg"
        metrics = _metrics_of(path)
        if irrigated:  # metrics_<stem>_irrigated.json
            metrics = f"{metrics.removesuffix('.json')}_irrigated.json"
        layers.append(Layer(path, label, metrics=metrics))
    x, y = square  # lower-left corner of the 5 km evaluation square (example 24's BASINS)
    scored = {}
    if irrigated:
        scored = {"scored_part": NEVADA_IRRIGATED, "pred_label": f"{PRED_LABEL}, scored"}
    return Entry(
        key,
        name,
        title,
        layers,
        # The evaluated reference: the DRI-mapped fields whose representative point lies in
        # the square, none smaller than 2,500 m2 in this release (example 24).
        reference=f"{NV}/references/{basin}_reference_{'?' * 12}.gpkg",
        reference_label=(
            f"Irrigated in the panel's year: {NEVADA_CREDIT}" if irrigated else NEVADA_CREDIT
        ),
        per_layer_background=True,  # NAIP 2022, SPOT 6/7 2023, Sentinel-1 2024
        window=(x + 2500.0, y + 2500.0, 5000.0),  # the evaluation square
        ncols=2,
        title_f1=True,
        # Each model line names its panels: DINOv3 checkpoints and the sam-auto settings of
        # NAIP and SPOT 6/7 read alike but for the hash or the setting.
        model_positions=True,
        copernicus_credit=True,  # the Sentinel-1 image at bottom right
        run_root=EX24_RUN_ROOT,  # all five Nevada figures from one run of example 24
        model_notes=[
            NEVADA_IRRIGATED_PANEL_NOTE if irrigated else NEVADA_PANEL_NOTE,
            *_nevada_notes(basin, NEVADA_PANELS[basin]),
        ],
        **scored,
    )


# Entry 24e (Diamond Valley): the opt-in options of agribound 1.1.0 before and after, with the
# panels that PREREG-DV3 section 7 fixed before the test (round 3, 2026-10-12): the evaluation
# square (DV-E) in full, each panel on its own input, the DRI outlines as in entry 24:
#   top left: NAIP 2022, Delineate-Anything v2 as released (naip da);
#   top right: naip da-opt, i.e. the 3 m grid, then A-snap (pivots="snap"), then the crop
#     filter (E3, E1 and E4 carried);
#   bottom left: SPOT 6/7 2023, DINOv3 fine-tuned as in 1.1 (spot dinov3-ft; entry 24's bottom
#     left before the sam-auto runs took that slot);
#   bottom right: spot dinov3-ft-opt, i.e. B-half (dilate_interior_px="half_band"), then
#     A-snap, then the crop filter (E2, E1 and E4 carried).
# The right panels are the crop-filter layers that example 24 scores for an optical "-opt" run
# (<output>_crop_t0.3.gpkg, no provenance record: the model and image come from the output's);
# their metrics files are named after the output. They equal round 4's option layers
# (outputs/examples_1.1_r3/dv_eval) coordinate for coordinate, and every score and paired
# interval equals dv_eval/state/opt.json (round-5 ex24b NOTES 3.2-3.3). The footer gives the
# right-minus-left changes of the run's options.csv, the labels PREREG-DV3 asks for (reported
# results, not tests; the effects on this square "partly known before the test": example 26's
# 3 m run differed by merge_tile_pieces=False, +0.046 against the option's +0.056) and Mason
# Valley's "-opt" rows (PREREG-DV3 section 7: "in the text"; naip da, naip dinov3-ft, spot da
# and sentinel1 da); the tests check its numbers against options.csv.
NEVADA_OPTIONS_BASIN = "diamond_valley"
NEVADA_OPTIONS_RUNS = ["naip da", "naip da-opt", "spot dinov3-ft", "spot dinov3-ft-opt"]
NEVADA_OPTIONS_LABELS = {  # "-opt" run: (output file stem, panel title of two lines)
    "naip da-opt": (
        "fields_naip_da-opt_2022",
        "NAIP 1 m, 2022: DA v2 with the options:\n3 m grid, pivots snapped, crop filter",
    ),
    "spot dinov3-ft-opt": (
        "fields_spot_dinov3-ft-opt_2023",
        "SPOT 6/7 6 m, 2023: DINOv3 fine-tuned with the options:\nhalf-band growth, pivots "
        "snapped, crop filter",
    ),
}
NEVADA_OPTIONS_NOTE = (
    "Panels fixed before the test (PREREG-DV3): left, Delineate-Anything v2 as released on NAIP "
    "and DINOv3 fine-tuned on SPOT 6/7, as in agribound 1.1; right, the same runs with the "
    "opt-in options of 1.1.0 that passed the test (example 24's --options), in this order: top "
    "right, the NAIP read on a 3 m grid (processing_gsd_m=3), round polygons and centre pivots "
    'cut into pieces snapped to circles (pivots="snap"; here 25 pivots, 5 of them rejoined '
    "from 11 pieces), then the crop filter (Annual NLCD 2022, at least 30 % crops or pasture); "
    "bottom right, the field interiors grown by half the boundary band the model predicts "
    '(dilate_interior_px="half_band": 12 px here, against the 4 px it was trained with), '
    "pivots snapped (17, none rejoined), then the crop filter (Annual NLCD 2023)"
)
NEVADA_OPTIONS_CHANGES = (
    "Right minus left, with paired 95 % intervals over 2,000 resamples of the square's 25 1 km "
    "blocks: NAIP F1 +0.167 [+0.090, +0.240], irrigated-only F1 +0.232 [+0.130, +0.345], "
    "boundary F1 (10 m) +0.035 [−0.012, +0.080]; SPOT F1 +0.057 [−0.022, +0.128], "
    "irrigated-only F1 +0.083 [−0.032, +0.180], boundary F1 +0.330 [+0.284, +0.376]"
)
NEVADA_OPTIONS_TESTED = (
    "Reported results, not tests: each option was tested alone, by rules fixed before the test "
    "(PREREG-DV3, 2026-10-12; amended 2026-10-13 for a defect of its check step that changed no "
    "layer or score), blind on three 5 km squares of other Nevada basins (the pivots also on "
    "this square and Mason Valley's, the 3 m grid also on Mason Valley's). On this square only "
    "the pivot snapping was tested blind; the effects here of the 3 m grid (example 26's run at "
    "3 m, made with merge_tile_pieces=False), the growth (an earlier sweep) and the crop filter "
    "(example 24's summary) were partly known before the test"
)
NEVADA_OPTIONS_MASON = (
    'Mason Valley (no figure; example 24\'s "-opt" runs on its other evaluation square, 187 '
    "fields): F1 0.567 to 0.677 for Delineate-Anything v2 on NAIP (+0.110 [+0.056, +0.166]), "
    "0.738 to 0.771 for DINOv3 fine-tuned on NAIP (+0.033 [+0.012, +0.060]; the growth "
    "measured 4 px, the trained width, and the pivot step found no pivot, so the change is the "
    "crop filter's), 0.513 to 0.510 for Delineate-Anything v2 on SPOT 6/7 (pivots snapped, "
    "crop filter; −0.004 [−0.018, +0.010]) and 0.453 unchanged for Sentinel-1 (pivot option; "
    "no pivot found)"
)


def _nevada_options_entry() -> Entry:
    """Entry 24e: Diamond Valley's runs of entry 24's top left and of 1.1's bottom left, and the
    same runs with the opt-in options (PREREG-DV3 section 7), read from ``EX24_RUN_ROOT``."""
    basin = NEVADA_OPTIONS_BASIN
    folder = f"{NV}/{basin}"
    layers = []
    for run in NEVADA_OPTIONS_RUNS:
        if run in NEVADA_OPTIONS_LABELS:  # the crop-filter layer of an optical "-opt" run
            stem, label = NEVADA_OPTIONS_LABELS[run]
            output = f"{folder}/{stem}.gpkg"
            layers.append(
                Layer(
                    f"{folder}/{stem}_crop_t0.3.gpkg",
                    label,
                    metrics=_metrics_of(output),  # named after the output, scores this layer
                    model_from=output,
                    background_from=output,
                )
            )
        else:
            stem, label = NEVADA_RUNS[run]
            path = f"{folder}/{stem}.gpkg"
            layers.append(Layer(path, label, metrics=_metrics_of(path)))
    return Entry(
        "24e",
        "Nevada_Diamond_Valley_options_example",
        "Diamond Valley, NV (centre pivots): the opt-in options of 1.1.0 before and after, on "
        "NAIP and SPOT",
        layers,
        reference=f"{NV}/references/{basin}_reference_{'?' * 12}.gpkg",
        reference_label=NEVADA_CREDIT,
        per_layer_background=True,  # NAIP 2022 in the top row, SPOT 6/7 2023 in the bottom row
        window=(586500.0, 4391500.0, 5000.0),  # the evaluation square, as in entry 24
        ncols=2,
        title_f1=True,
        model_positions=True,
        run_root=EX24_RUN_ROOT,
        model_notes=[
            NEVADA_OPTIONS_NOTE,
            NEVADA_OPTIONS_CHANGES,
            NEVADA_OPTIONS_TESTED,
            NEVADA_OPTIONS_MASON,
            *_nevada_notes(basin, NEVADA_OPTIONS_RUNS),
        ],
    )


# Example 25 (47 states), entries 25 (overview map) and 25b (four squares). Panels of 25b,
# numbered 1-4 on the overview: the squares at evenly spaced ranks of the NAIP F1
# (Delineate-Anything v2 as released; outputs/multistate_openet/summary.csv rows "naip" with
# status "ok"; _rank_picks): ranks 1, 16, 32 and 47 of 47 in increasing order, i.e. the
# lowest, the two that cut the ranking into thirds and the highest; ties keep the summary's
# order. In the 1.1 run: MT 0.013, MA 0.268, ID 0.371, CA 0.697. The NAIP year of each square
# is the example's (a one-character glob in the file names). Each 25b panel shows, of the nine
# candidate 2.5 km squares half a side apart in the 5 km study square, the one with the most
# reference fields (by representative point; render_areas).
MS = "outputs/multistate_openet"
MULTISTATE_N_SQUARES = 47  # NAIP rows with status ok in the 1.1 run (the tests check it)
MULTISTATE_PANELS = [  # state, place (Census 2023 county), reference source of its fields
    ("MT", "Glacier County, MT", "USDA FSA CLU"),
    ("MA", "Hampshire County, MA", "USDA FSA CLU"),
    ("ID", "Twin Falls County, ID", "IDWRSP and CLU"),
    ("CA", "Fresno County, CA", "California DWR"),
]
OPENET_CREDIT = (
    "OpenET agricultural field boundaries, version 2.1.1, Desert Research Institute for the "
    "OpenET Consortium (2026), with the field boundaries of their source agencies"
)
# Example 25's crop-history zones (--crop-sequence) are made by the final run, in
# outputs/examples_1.1_final (its NAIP and Sentinel-2 outputs are copies of the 1.1 run's; its
# references and NAIP scores are unchanged, and summary.csv has the same NAIP rows). With
# MULTISTATE_ZONES = True, entry 25b draws them beside Delineate-Anything in each square (on
# the same NAIP image) and reads that run root; set since that run finished and was checked
# (2026-10-11: 25 ok, 47 zone runs).
MULTISTATE_ZONES = True
# The targeted re-runs with the final code (module docstring, "Run roots"): entries 03, 14, 23,
# 23b-23e, 25b (with the zones) and 27 read their outputs here.
FINAL_RUN_ROOT = "outputs/examples_1.1_final"


def _rank_positions(n: int, k: int) -> list[int]:
    """The *k* evenly spaced ranks (0-based) of *n* scores in increasing order.

    Ranks ``round(j (n - 1) / (k - 1))``, halves rounded up, for ``j = 0 .. k - 1``: the
    lowest and the highest score and those between them at equal steps of rank.
    """
    if k < 2 or n < k:
        raise ValueError(f"need k >= 2 and at least k scores, got k={k} and {n}")
    return [(2 * j * (n - 1) + k - 1) // (2 * (k - 1)) for j in range(k)]


def _rank_picks(scores: list[tuple[str, float]], k: int) -> list[str]:
    """Names at *k* evenly spaced ranks of *scores* (increasing; ties keep the given order;
    :func:`_rank_positions`)."""
    order = sorted(range(len(scores)), key=lambda i: (scores[i][1], i))
    return [scores[order[r]][0] for r in _rank_positions(len(scores), k)]


def _multistate_order(score: str) -> str:
    """How the four squares are numbered: "1-4 in increasing <score>: the lowest, the two that
    cut the ranking into thirds, the highest (ranks 1, 16, 32 and 47 of 47)"."""
    ranks = _rank_positions(MULTISTATE_N_SQUARES, len(MULTISTATE_PANELS))
    if len(ranks) != 4:  # the words below describe four ranks
        raise ValueError(f"the note describes four squares, not {len(ranks)}")
    return (
        f"1-4 in increasing {score}: the lowest, the two that cut the ranking into thirds, the "
        f"highest (ranks {_and_join(r + 1 for r in ranks)} of {MULTISTATE_N_SQUARES})"
    )


# The panel titles of entries 25b, 26, 26b and 27 give counts of the whole output file and the
# F1 of the example's evaluation area, which is larger than the panel's window.
MULTISTATE_SCORES_NOTE = (
    "Counts: the whole output; F1: the whole 5 km study square, one-to-one at IoU >= 0.5"
)


def _multistate_squares(zones: bool = MULTISTATE_ZONES) -> Entry:
    """Entry 25b: the squares of MULTISTATE_PANELS; with *zones*, the CDL zones beside them."""
    layers = []
    for state, place, source in MULTISTATE_PANELS:
        folder = f"{MS}/{state.lower()}"
        naip = f"{folder}/fields_naip_da_20??.gpkg"
        common = {"reference": f"{folder}/reference.gpkg", "area": state}
        label = f"{place}, {source}" if not zones else f"{place}: DA v2, NAIP"
        layers.append(Layer(naip, label, metrics=_metrics_of(naip), **common))
        if zones:
            path = f"{folder}/zones_cdl-sequence_2016-2023.gpkg"
            layers.append(
                Layer(
                    path,
                    f"{place}: CDL zones 2016-2023",
                    metrics=_metrics_of(path),
                    noun="zone",
                    background_from=naip,  # the zones' own raster is the CDL stack
                    **common,
                )
            )
    notes = [
        f"Squares {_multistate_order('NAIP F1 of example 25')}; each panel shows the 2.5 km "
        "square with the most reference fields among nine candidates half a side apart in the "
        "5 km study square",
        MULTISTATE_SCORES_NOTE,
        # No vintage for California: the delivery's CADWR_2023prov code is an internal marker,
        # and the delivered vintage cannot be established (openet-field-boundaries,
        # docs/delivery-defects.md, item 4). Names as in example 25's attribution list.
        f"Reference: {OPENET_CREDIT}: 1-2 USDA Farm Service Agency Common Land Units; 3 Idaho "
        "Department of Water Resources, Eastern Snake River Plain irrigated lands (IDWRSP; 8 % "
        "of the fields CLU); 4 California Department of Water Resources, Statewide Crop Mapping "
        "(with Land IQ)",
    ]
    if zones:
        notes.insert(2, "Zones: crop-history zones of the USDA Cropland Data Layer, not fields")
    return Entry(
        "25b",
        "Multistate_OpenET_squares_example",
        "47 states: four squares from the lowest to the highest F1 of Delineate-Anything v2 "
        "on NAIP",
        layers,
        multi_area=True,
        per_layer_background=True,
        crop_m=2500,
        crop_on_reference=True,  # the square with the most reference fields
        reference_label="OpenET field boundaries v2.1.1 (source agencies' outlines)",
        pred_label=(f"agribound {__version__} fields or zones" if zones else None),
        # One role for both panels of a square, so that the note lists its NAIP image once.
        bg_role="the input of Delineate-Anything, also under the zones" if zones else None,
        ncols=2,
        title_f1=True,
        run_root=FINAL_RUN_ROOT if zones else None,
        model_notes=notes,
    )


# Entry 25c (no model output): example 25's Montana square (Glacier County, square 1 of 25 and
# 25b) with its reference fields, all USDA FSA Common Land Units, on the NAIP of 2005 (left) and
# on the 2023 composite that Delineate-Anything read (right), in the 5 km square. agribound's
# naip source reads 4-band images only, and Montana's NAIP of 2004-2006 has R, G and B, so the
# 2005 image was exported with its record by tools/fetch_naip_history.py into R3_RUN_ROOT; the
# reference and the 2023 composite are the 1.1 run's (outputs/examples_1.1, read only).
R3_RUN_ROOT = "outputs/examples_1.1_r3"
MONTANA_SQUARE = (375000, 5385000)  # lower-left corner, EPSG:32612 (example 25's STATES)
MONTANA_NAIP_2005 = "montana/naip_2005_rgb_2m"  # under R3_RUN_ROOT: the .tif and its .json
MONTANA_NOTES = [
    "No model output. The CLU strips have the width and orientation of the crop and fallow "
    "strips of the 2005 image; in 2023 the square is farmed in blocks of many strips",
    "NAIP: USDA Farm Service Agency, via Google Earth Engine (USDA/NAIP/DOQQ)",
    f"Reference: {OPENET_CREDIT}: here the Common Land Units (CLU) of the USDA Farm Service "
    "Agency, the square's fields of at least 2,500 m² by representative point, as example 25 "
    "evaluates them",
]


def _montana_years() -> Entry:
    """Entry 25c: the Montana square of example 25 on NAIP 2005 and 2023 (no model output)."""
    x, y = MONTANA_SQUARE
    return Entry(
        "25c",
        "Multistate_OpenET_Montana_example",
        "Montana: the CLU on NAIP 2005 and 2023",
        [],
        # Example 25's reference of the square: its OpenET fields of at least 2,500 m² whose
        # representative point lies in it.
        reference=f"{MS}/mt/reference.gpkg",
        reference_label="OpenET field boundaries v2.1.1: USDA FSA Common Land Units",
        window=(x + 2500.0, y + 2500.0, 5000.0),  # the 5 km square
        ncols=2,
        line_width=0.45,  # thin outlines: the image shows between the CLU's 50 m strips
        imagery=Imagery(
            panels=[
                ImagePanel(
                    "NAIP 2005, 2 m",
                    background=f"{MONTANA_NAIP_2005}.tif",
                    background_source="naip",
                    bg_role=(
                        "R, G and B, each pixel the mean of the native 1 m pixels, exported by "
                        "tools/fetch_naip_history.py (agribound's naip source reads 4-band "
                        "images only)"
                    ),
                    record=f"{MONTANA_NAIP_2005}.json",
                    run_root=R3_RUN_ROOT,
                ),
                ImagePanel(
                    "NAIP 2023, 1 m: example 25's input",
                    background_from=f"{MS}/mt/fields_naip_da_2023.gpkg",
                    # The images' date: example 25's docstring (STATES), as in Earth Engine.
                    bg_role="the input of Delineate-Anything v2 in example 25, flown 2023-07-22",
                ),
            ],
            notes=MONTANA_NOTES,
        ),
    )


# Example 26 (radar), entries 26 (the held-out sites) and 26b (the in-sample sites). Per site
# (example 26's SITES, the six with Umbra or Capella collects; Diamond Valley has Sentinel-1
# only and is in entry 24, the UAVSAR site is optional): Sentinel-1, the Umbra or Capella
# composite and the optical control read with the radar's Delineate-Anything settings
# (naip-sar-settings: the two 1 m panels differ in imagery only; Sevilla has no NAIP, so
# Sentinel-2), each on its own input image, against the site's reference. Each area shows,
# of the candidate squares of crop_m half a side apart that lie inside the site's evaluation
# area (4 to 17 per site in the 1.1 run), the one with the most reference fields.
SAR = "outputs/sar_field_boundaries"
SAR_RUN_ROOT = "outputs/examples_1.1_sar"
SAR_SITES = {  # slug: (name, radar year, VHR source, optical output stem, optical label)
    "tamu": ("Texas A&M", 2024, "umbra", "fields_naip_sar_settings_2022", "NAIP, SAR settings"),
    "intel_oh": (
        "Intel Ohio",
        2024,
        "umbra",
        "fields_naip_sar_settings_2023",
        "NAIP, SAR settings",
    ),
    "orleans_vt": (
        "Orleans County",
        2022,
        "capella",
        "fields_naip_sar_settings_2023",
        "NAIP, SAR settings",
    ),
    "ucd": ("UC Davis", 2024, "umbra", "fields_naip_sar_settings_2022", "NAIP, SAR settings"),
    "sevilla": ("Sevilla", 2023, "capella", "fields_sentinel2_2023", "Sentinel-2"),
    "centerfield": (
        "Centerfield",
        2025,
        "umbra",
        "fields_naip_sar_settings_2021",
        "NAIP, SAR settings",
    ),
}
SAR_SETTINGS_NOTE = (
    "Delineate-Anything v2 as released; Umbra and Capella, and NAIP with SAR settings: "
    "processing_gsd_m=3 and merge_tile_pieces=False (the settings agribound documents for the "
    "Umbra and Capella composites); Sentinel-1 and Sentinel-2: max_detections=1000"
)
SAR_SCORES_NOTE = (
    "Counts: the whole output; F1: the site's evaluation area, one-to-one at IoU >= 0.5"
)


def _sar_entry(key: str, name: str, title: str, sites: list[str], **kw) -> Entry:
    """Entry of example 26: three panels per site (Sentinel-1, Umbra or Capella, optical)."""
    layers = []
    for slug in sites:
        site, year, vhr, optical, optical_label = SAR_SITES[slug]
        crop = 1500.0 if slug == "sevilla" else None  # recintos of 0.8 ha (median)
        for stem, label in (
            (f"fields_sentinel1_{year}", f"{site}: Sentinel-1"),  # the row's first panel
            (f"fields_{vhr}_{year}", vhr.capitalize()),
            (optical, optical_label),
        ):
            path = f"{SAR}/{slug}/{stem}.gpkg"
            layers.append(
                Layer(
                    path,
                    label,
                    reference=f"{SAR}/{slug}/reference.gpkg",
                    area=slug,
                    crop_m=crop,
                    metrics=_metrics_of(path),
                    # The scored area: inside the valid radar pixels at the held-out sites.
                    window_within=f"{SAR}/{slug}/evaluation_area.geojson",
                )
            )
    return Entry(
        key,
        name,
        title,
        layers,
        multi_area=True,
        per_layer_background=True,
        crop_m=2000,
        crop_on_reference=True,
        ncols=3,
        title_f1=True,
        copernicus_credit=True,  # Sentinel-1 at every site, Sentinel-2 at Sevilla
        run_root=SAR_RUN_ROOT,
        **kw,
    )


# Example 27 (Quebec): per square, the crop-history zones (ACI 2016-2023) and
# Delineate-Anything v2 on Sentinel-2 2023, both on the Sentinel-2 composite (the zones' own
# raster is the ACI stack), against the declared parcels (BDPPAD 2023). Made by the final run.
QC = "outputs/quebec_crop_sequence"
QUEBEC_SQUARES = {"bas-saint-laurent": "Bas-Saint-Laurent", "monteregie": "Montérégie"}

# Example 28 (New York), entries 28 (R1, the Black Dirt), 28b (R2, the Finger Lakes) and 28c
# (R3, the North Country): one 2 x 2 per region on its 5 km evaluation square, each panel on its
# own input, by the panel rule of PREREG-NY section 5, fixed before the run (_new_york_panels;
# the example writes its choice to panels.json, and the tests re-apply the rule to its
# summary.csv):
#   top left: naip da (Delineate-Anything v2 as released);
#   top right: the highest F1 among naip da-sam2, naip da-ft and naip dinov3-ft (the primary,
#     held-out arm; ties: the higher boundary F1, then this order);
#   bottom left: the higher F1 of sentinel2 da and sentinel2 ftw (ties: the higher boundary
#     F1, then sentinel2 da);
#   bottom right: naip sam-auto if it ran; else the best SPOT run (spot da, spot da-ft, spot
#     dinov3-ft) if --spot ran; else the Sentinel-2 run of that pair not shown bottom left.
# Never a panel: R1's near-sample A1 arm, sentinel2 sam-auto and the --extended runs. In the run
# (outputs/examples_1.1_r3/ny, round 4, 2026-10-12/13; 26 of 26 runs ok, neither --spot nor
# --extended): R1 naip da 0.147, naip da-sam2 0.159 (da-ft 0.085, dinov3-ft 0.143), sentinel2 da
# 0.111 (ftw 0.044), naip sam-auto 0.182; R2 0.339, naip dinov3-ft 0.433 (da-sam2 0.332, da-ft
# 0.346), sentinel2 da 0.346 (ftw 0.260), naip sam-auto 0.362; R3 0.373, naip da-ft 0.4333
# (dinov3-ft 0.4322, da-sam2 0.361), sentinel2 da 0.455 (ftw 0.361), naip sam-auto 0.404.
# The reference is the region's evaluation reference (references/<region>_reference_<12 hex
# digits>.gpkg; never the _2024_crop_t0.3 copy): the CLU fields of at least 2,500 m2 whose
# representative point lies in the square (668, 482 and 281). The author approved showing
# the New York CLU outlines on 2026-10-12 (PREREG-NY amendment A3); the file itself is not
# redistributed.
NY = "outputs/new_york_field_boundaries"
NY_RUN_ROOT = "outputs/examples_1.1_r3/ny"
NY_RUNS = {  # summary.csv run: (output file stem, panel label)
    "naip da": ("fields_naip_da_2022", "NAIP 1 m, 2022: DA v2 as released"),
    "naip da-sam2": ("fields_naip_da-sam2_2022", "NAIP 1 m, 2022: DA v2 + SAM 2"),
    "naip da-ft": ("fields_naip_da-ft_2022", "NAIP 1 m, 2022: DA v2 fine-tuned"),
    "naip dinov3-ft": ("fields_naip_dinov3-ft_2022", "NAIP 1 m, 2022: DINOv3 fine-tuned"),
    "naip sam-auto": ("fields_naip_sam-auto_2022", "NAIP 1 m, 2022: sam-auto"),
    "sentinel2 da": ("fields_sentinel2_da_2024", "Sentinel-2 10 m, 2024: DA v2 as released"),
    "sentinel2 ftw": ("fields_sentinel2_ftw_2024", "Sentinel-2 10 m, 2024: FTW"),
    "spot da": ("fields_spot_da_2021", "SPOT 6/7 6 m, 2021: DA v2 as released"),
    "spot da-ft": ("fields_spot_da-ft_2021", "SPOT 6/7 6 m, 2021: DA v2 fine-tuned"),
    "spot dinov3-ft": ("fields_spot_dinov3-ft_2021", "SPOT 6/7 6 m, 2021: DINOv3 fine-tuned"),
}
NY_PANELS = {  # the rule's runs in NY_RUN_ROOT (panels.json)
    "black_dirt": ["naip da", "naip da-sam2", "sentinel2 da", "naip sam-auto"],
    "finger_lakes": ["naip da", "naip dinov3-ft", "sentinel2 da", "naip sam-auto"],
    "north_country": ["naip da", "naip da-ft", "sentinel2 da", "naip sam-auto"],
}
NY_REGIONS = {  # region: (key, image name, title, lower-left corner in EPSG:32618; PREREG-NY 1)
    "black_dirt": (
        "28",
        "New_York_Black_Dirt_example",
        "Black Dirt, Orange County, NY (vegetable strips on muck): released, SAM 2 and sam-auto "
        "runs on NAIP and Sentinel-2 against the Common Land Units",
        (545000, 4575000),
    ),
    "finger_lakes": (
        "28b",
        "New_York_Finger_Lakes_example",
        "Finger Lakes, Cayuga County, NY (dairy rotations): released, fine-tuned and sam-auto "
        "runs on NAIP and Sentinel-2 against the Common Land Units",
        (365000, 4745000),
    ),
    "north_country": (
        "28c",
        "New_York_North_Country_example",
        "North Country, Jefferson County, NY (corn, hay and pasture): released, fine-tuned and "
        "sam-auto runs on NAIP and Sentinel-2 against the Common Land Units",
        (410000, 4845000),
    ),
}
# Where each region's training square lies and how many CLU fields reach into it (example
# 28's docstring and PREREG-NY section 1: edge to edge; labels of any size). R1's top right is
# SAM 2 refinement, so its fine-tuned models (A0 and the near-sample A1) are not drawn.
NY_TRAINING = {
    "black_dirt": "5.0 km west of the evaluated one (310 CLU fields reach into it)",
    "finger_lakes": "15.0 km south of the evaluated one (505 CLU fields reach into it)",
    "north_country": "10.0 km north of the evaluated one (293 CLU fields reach into it)",
}
NY_FINE_TUNE_NOTE = {  # the fine-tuned run's engine: its fine-tuning line
    "dinov3-ft": (
        "Fine-tuned (held out): DINOv3 ViT-L/16 (SAT-493M weights), full fine-tuning for up to "
        "20 epochs on the NAIP ({panels}) of a 5 km training square of the county {where}, "
        "never on the evaluated fields"
    ),
    "da-ft": (
        "Fine-tuned (held out): Delineate-Anything v2 (large_v2), 20 epochs with "
        "yolo_lr0=1e-4, on the NAIP ({panels}) of a 5 km training square of the county "
        "{where}, never on the evaluated fields"
    ),
}
NY_PANEL_NOTE = (
    "Panels by the rule fixed before the run (PREREG-NY): Delineate-Anything v2 as released on "
    "NAIP (top left); the highest F1 of SAM 2 refinement and of Delineate-Anything v2 and "
    "DINOv3 fine-tuned on the region's training square, on NAIP (top right); the higher F1 of "
    "Delineate-Anything v2 and FTW on Sentinel-2 (bottom left); sam-auto on NAIP (bottom right)"
)
NY_SAM_AUTO_NOTE = (
    "sam-auto ({panels}): no field labels and no fine-tuning (SAM 2.1 as released); the "
    "engine's NAIP setting was chosen on two Nevada squares (example 24's training squares), "
    "not on any New York square"
)
NY_SCORES_NOTE = (
    "F1: one-to-one at IoU >= 0.5 against the square's {n:,} CLU fields of at least 2,500 m² "
    "(by representative point); it measures agreement with CLUs (farm-programme units), not "
    "with visible fields, and one square per region is not an accuracy estimate for the county"
)
NY_CHANCE_NOTE = {  # region: the line on the chance baselines (PREREG-NY rule C2)
    "black_dirt": (
        "No run here is above chance by the rule fixed before the run (C2: the lower bound of "
        "F1 above both image-blind baselines): the reference moved 24.3 m east and north still "
        "matches 470 of its 668 fields (F1 0.70), because its strips repeat side by side "
        "(explained after the scores were seen)"
    ),
    "finger_lakes": (
        "Every run here is above chance by the rule fixed before the run (C2: the lower bound "
        "of F1 above both image-blind baselines; the reference moved 59.3 m east and north "
        "matches 42 of its 482 fields, F1 0.087)"
    ),
    "north_country": (
        "Every run here is above chance by the rule fixed before the run (C2: the lower bound "
        "of F1 above both image-blind baselines; the reference moved 82.4 m east and north "
        "matches 10 of its 281 fields, F1 0.036)"
    ),
}
NY_N_REFERENCE = {"black_dirt": 668, "finger_lakes": 482, "north_country": 281}
NY_CREDIT = (
    "OpenET agricultural field boundaries, version 2.1.1, Desert Research Institute for the "
    "OpenET Consortium (2026), with the field boundaries of their source agency: Common Land "
    "Unit (CLU), USDA Farm Service Agency"
)


def _new_york_panels(rows: list[dict], region: str) -> list[str | None]:
    """The runs of the four panels of *region* (example 28) by PREREG-NY section 5, from its
    summary rows (top left, top right, bottom left, bottom right; None: no candidate ran).

    Only rows with status "ok" count. Ties go to the higher boundary F1, then to the order the
    rule lists the runs in.
    """
    ok = {r["run"]: r for r in rows if r["region"] == region and r["status"] == "ok"}

    def best(candidates: list[str]) -> str | None:
        have = [(i, ok[c]) for i, c in enumerate(candidates) if c in ok]
        if not have:
            return None
        pick = max(have, key=lambda t: (float(t[1]["f1"]), float(t[1]["boundary_f1"]), -t[0]))
        return pick[1]["run"]

    top_left = "naip da" if "naip da" in ok else None
    top_right = best(["naip da-sam2", "naip da-ft", "naip dinov3-ft"])
    bottom_left = best(["sentinel2 da", "sentinel2 ftw"])
    if "naip sam-auto" in ok:
        bottom_right = "naip sam-auto"
    else:
        bottom_right = best(["spot da", "spot da-ft", "spot dinov3-ft"])
        if bottom_right is None:
            bottom_right = next(
                (r for r in ("sentinel2 da", "sentinel2 ftw") if r in ok and r != bottom_left),
                None,
            )
    return [top_left, top_right, bottom_left, bottom_right]


def _new_york_entry(region: str) -> Entry:
    """Entry of one region of example 28: the four panels of ``NY_PANELS`` in the 5 km
    evaluation square, each on the composite it was delineated on, read from ``NY_RUN_ROOT``."""
    key, name, title, (x, y) = NY_REGIONS[region]
    runs = NY_PANELS[region]
    names = [n.lower() for n in _panel_names(len(runs), 2)]
    layers = []
    for run in runs:
        stem, label = NY_RUNS[run]
        path = f"{NY}/{region}/{stem}.gpkg"
        refined = "agribound:sam_refined" if run.endswith("-sam2") else None
        layers.append(Layer(path, label, metrics=_metrics_of(path), highlight=refined))
    notes = [NY_PANEL_NOTE]
    for engine, line in NY_FINE_TUNE_NOTE.items():
        tuned = [n for n, r in zip(names, runs, strict=True) if r.split(" ", 1)[1] == engine]
        if tuned:
            notes.append(line.format(panels=_and_join(tuned), where=NY_TRAINING[region]))
    sam = [n for n, r in zip(names, runs, strict=True) if r.endswith(" sam-auto")]
    if sam:
        notes.append(NY_SAM_AUTO_NOTE.format(panels=_and_join(sam)))
    notes += [
        NY_SCORES_NOTE.format(n=NY_N_REFERENCE[region]),
        NY_CHANCE_NOTE[region],
        f"Reference: {NY_CREDIT}",
    ]
    return Entry(
        key,
        name,
        title,
        layers,
        reference=f"{NY}/references/{region}_reference_{'?' * 12}.gpkg",
        reference_label="OpenET field boundaries v2.1.1: USDA FSA Common Land Units",
        per_layer_background=True,  # NAIP 2022 in the top row and at bottom right, S2 2024
        window=(x + 2500.0, y + 2500.0, 5000.0),  # the evaluation square
        ncols=2,
        title_f1=True,
        model_positions=True,
        copernicus_credit=True,  # the Sentinel-2 panel at bottom left
        sam2_note=any(r.endswith("-sam2") for r in runs),
        run_root=NY_RUN_ROOT,
        model_notes=notes,
    )


# The SAM entry (key "sam"): SAM 2.1 automatic masks (sam-auto) against Delineate-Anything v2 as
# released, with the panels that round-2 PLAN.md section 1.5 fixed before the pre-registered
# evaluation (PREREG-SAM) ran (PREREG-SAM itself has no panel rule):
#   study area 1 (top row): example 24's Diamond Valley evaluation square (DV-E) on SPOT 6/7
#     2023, Delineate-Anything v2 as released (left) and sam-auto (right);
#   study area 2 (bottom row): NAIP, the same two engines, on the square of example 25 at the
#     median of the paired NAIP F1 differences of step E4; if two squares tie, the one with
#     the lower Delineate-Anything F1, then the first alphabetically (_sam_median_square):
#     Tennessee (Crockett County, NAIP of 2023-10-18), d = +0.0195, under either reading (the
#     47 squares; or the 46 of the primary endpoint, where Tennessee and Pennsylvania are the
#     middle two and the tie rule picks Tennessee; analyze-sam RESULTS-SAM 8).
# Each row in its whole 5 km square (Layer.window), both arms on byte-identical composites,
# without the crop filter. The sam-auto layers and their scores are the evaluation's own
# (SAM_EVAL_RUN_ROOT; scores.json records, each example's own scoring call); the comparators,
# their references and metrics files are read where PREREG-SAM section 2 names them, under
# their own run roots (Layer.run_root); the sam-auto panel of an area draws its reference.
SAM_EVAL_RUN_ROOT = "outputs/examples_1.1_r3/sam_eval"
# Rows: (area, place, imagery, comparator, its run root, sam-auto output, (scores, record)).
SAM_ENTRY_ROWS = [
    (
        "dv-e",
        "Diamond Valley, NV",
        "SPOT 6/7 6 m, 2023",
        f"{NV}/diamond_valley/fields_spot_da_2023.gpkg",
        "outputs/examples_1.1_ex24",  # E1's comparator and reference (PREREG-SAM section 2)
        "E1/DV-E/out/fields_spot_sam-auto_2023.gpkg",
        ("E1/scores.json", ("E1/DV-E/spot", "sam-auto", "own")),
    ),
    (
        "tn",
        "Crockett County, TN",
        "NAIP 1 m, 2023",
        f"{MS}/tn/fields_naip_da_2023.gpkg",
        FINAL_RUN_ROOT,  # E4's comparator and reference (PREREG-SAM section 2)
        "E4/TN/out/fields_naip_sam-auto_2023.gpkg",
        ("E4/scores.json", ("E4/TN/naip", "sam-auto", "own")),
    ),
]
SAM_ENTRY_ARMS = ("DA v2 as released", "sam-auto")  # the second line of each row's titles
SAM_ENTRY_REFERENCES = {  # area: the comparator's evaluation reference, under its run root
    "dv-e": f"{NV}/references/diamond_valley_reference_{'?' * 12}.gpkg",
    "tn": f"{MS}/tn/reference.gpkg",
}
SAM_ENTRY_SQUARES = {  # area: lower-left corner of its 5 km square (examples 24 and 25)
    "dv-e": (584000, 4389000),  # EPSG:32611
    "tn": (300000, 3960000),  # EPSG:32616
}
SAM_ENTRY_NOTES = [
    "Panels fixed before the evaluation (round-2 plan, section 1.5): 1, example 24's Diamond "
    "Valley evaluation square on SPOT 6/7 2023; 2, example 25's square at the median of the 47 "
    "paired NAIP F1 differences (Tennessee, +0.019); left, Delineate-Anything v2 as released, "
    "right, sam-auto, each pair on the same composite, without the crop filter",
    "F1: each example's own score, one-to-one at IoU >= 0.5 (1: the square's 74 DRI-mapped "
    "fields; 2: its 515 reference fields of at least 2,500 m², all CLU). Paired F1 difference, "
    "sam-auto minus Delineate-Anything v2, with its 95 % interval over 2,000 resamples of the "
    "square's 25 1 km blocks: 1, +0.464 [+0.346, +0.584]; 2, +0.019 [−0.011, +0.053]",
    "Pre-registered endpoints (PREREG-SAM): NAIP +0.017 [+0.004, +0.030], the mean over 46 "
    "squares of example 25 without California; Sentinel-2 −0.152 [−0.178, −0.127], so it did "
    "not transfer; SPOT\u00a06/7 +0.256 and Landsat +0.126 (Landsat\u00a08/9; 7/8/9 on the "
    "San Juan box), means over four Nevada and New Mexico areas with no interval, of pairs "
    "scored without the crop filter in Nevada and with it in New Mexico, as pre-registered "
    "(with all four pairs unfiltered: +0.211 and +0.091, a secondary result)",
    "Amendment of 2026-10-13 (PREREG-SAM): Mason Valley's pairs had been scored against a "
    "crop-filter copy of its reference (176 of its 187 fields); both Nevada areas were scored "
    "again with the reference after their first scores and the endpoints had been read, a "
    "recorded deviation from PREREG-SAM section 6. Mason Valley's scores changed, square 1's "
    "did not",
    "sam-auto: no field labels and no fine-tuning (SAM 2.1 as released); its settings were "
    "chosen on example 24's two training squares (in-sample there), one of them 5.0 km south "
    "of square\u00a01, so square\u00a01 is held out from that choice, not from the landscape. "
    "Delineate-Anything v2's scores on both squares were known before the evaluation "
    "(examples 24 and 25's own runs)",
    f"Reference: 1 {NEVADA_CREDIT}; 2 {NY_CREDIT}. The endpoints' other areas: OpenET v2.1.1, "
    "with the field boundaries of their source agencies; in Nevada, the DRI field boundaries "
    "(1); in New Mexico, the DRI/NMOSE field boundaries (V18), Desert Research Institute and "
    "New Mexico Office of the State Engineer",
]


def _sam_median_square(diffs: list[tuple[str, float, float]]) -> str:
    """The E4 square at the median of the paired NAIP F1 differences (round-2 PLAN 1.5).

    *diffs* holds ``(state, d_i, Delineate-Anything F1)`` per square. With an odd number of
    squares the middle one; with an even number the two middle ones tie: the one with the
    lower Delineate-Anything F1, then the first alphabetically.
    """
    order = sorted(diffs, key=lambda t: t[1])
    n = len(order)
    if n == 0:
        raise ValueError("no squares")
    if n % 2:
        return order[n // 2][0]
    return min(order[n // 2 - 1 : n // 2 + 1], key=lambda t: (t[2], t[0]))[0]


def _sam_auto_entry() -> Entry:
    """The SAM entry: per study area, Delineate-Anything v2 as released (left) and sam-auto
    (right) in the area's 5 km square (``SAM_ENTRY_ROWS``)."""
    layers = []
    for area, place, imagery, comparator, root, sam, (scores, record) in SAM_ENTRY_ROWS:
        x, y = SAM_ENTRY_SQUARES[area]
        layers.append(
            Layer(
                comparator,
                f"{place}: {imagery}\n{SAM_ENTRY_ARMS[0]}",  # a title of two lines
                reference=SAM_ENTRY_REFERENCES[area],
                area=area,
                window=(x + 2500.0, y + 2500.0, 5000.0),  # the whole evaluation square
                metrics=_metrics_of(comparator),
                run_root=root,
            )
        )
        layers.append(
            Layer(
                sam,
                f"{place}: {imagery}\n{SAM_ENTRY_ARMS[1]}",
                area=area,  # the window and the reference of the area's first panel
                metrics=scores,
                metrics_record=record,
            )
        )
    return Entry(
        "sam",
        "SAM2_automatic_masks_example",
        "SAM 2.1 automatic masks against Delineate-Anything v2",
        layers,
        multi_area=True,
        per_layer_background=True,
        crop_m=5000,  # each area's window is given (Layer.window): its evaluation square
        reference_label="Reference (1 DRI field boundaries; 2 OpenET v2.1.1, USDA FSA CLU)",
        ncols=2,
        title_f1=True,
        model_positions=True,  # the NAIP and SPOT 6/7 settings of sam-auto read alike
        run_root=SAM_EVAL_RUN_ROOT,
        model_notes=SAM_ENTRY_NOTES,
    )


# Paths are relative to the run root (default: outputs/examples_1.1; the 1.0 runs in
# outputs/examples_1.0 are described in its PLAN.md).
ENTRIES: list[Entry] = [
    Entry(
        "14",
        "NM_example",
        "Lea County, NM: DINOv3 (fine-tuned) + SAM 2 from 30 m to 1 m",
        # 2024 imagery, or the latest year of a source without 2024 imagery (example 14).
        [
            Layer(
                "outputs/lea_county_dinov3_sam2/fields_landsat_dinov3-sam2_2024.gpkg",
                "Landsat 30 m, 2024",
                highlight="agribound:sam_refined",
            ),
            Layer(
                "outputs/lea_county_dinov3_sam2/fields_sentinel2_dinov3-sam2_2024.gpkg",
                "Sentinel-2 10 m, 2024",
                highlight="agribound:sam_refined",
            ),
            Layer(
                "outputs/lea_county_dinov3_sam2/fields_spot_dinov3-sam2_2023.gpkg",
                "SPOT 6/7 6 m, 2023",
                highlight="agribound:sam_refined",
            ),
            Layer(
                "outputs/lea_county_dinov3_sam2/fields_naip_dinov3-sam2_2022.gpkg",
                "NAIP 1 m, 2022",
                highlight="agribound:sam_refined",
            ),
        ],
        reference="outputs/lea_county_dinov3_sam2/reference_v18.gpkg",
        reference_label="DRI/NMOSE field boundaries (V18; the training labels)",
        per_layer_background=True,
        crop_m=6000,
        crop_on_reference=True,  # the window with the most reference fields
        ncols=2,
        sam2_note=True,
        # The Landsat model line names DINOv3's 10 m processing grid; the lines say which
        # panels they describe.
        model_positions=True,
        # The final run: the Landsat run made again with DINOv3's 10 m processing grid; the
        # other three panels are copies of the 1.1 run's outputs.
        run_root=FINAL_RUN_ROOT,
    ),
    Entry(
        "12n",
        "NM_NAIP_models_example",
        "Lea County, NM: Delineate-Anything v2, GeoAI and DINOv3 on NAIP 1 m (2022)",
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
        reference="outputs/lea_county_ensemble/lea_county_reference_v18.gpkg",
        reference_label="DRI/NMOSE field boundaries (V18; the fine-tuning labels)",
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
        "23",
        "Tree_Crops_SPOT_Pan_example",
        "Tree crops: Delineate-Anything v2 on SPOT 6/7 panchromatic 1.5 m, four landscapes",
        [
            Layer(
                f"outputs/tree_crops/{slug}/fields_spot-pan_da_{year}.gpkg",
                label,
                crop_m=crop,
                reference=f"outputs/tree_crops/references/{slug}_reference_{'?' * 12}.gpkg",
            )
            for slug, year, label, crop in [
                ("twifo", 2020, "Twifo Praso, Ghana: oil palm estate", 2000),
                ("oro", 2021, "Oro, Papua New Guinea: oil palm smallholders", 1000),
                ("madera", 2022, "Madera, California: almonds, pistachios", 2000),
                ("jaen", 2023, "Úbeda, Spain: olive groves", 1000),
            ]
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=2000,
        crop_on_reference=True,  # per panel: the square with the most reference polygons
        reference_label="Reference (RSPO GeoRSPO, member-declared; DWR / Land IQ; SIGPAC)",
        ncols=2,
        inset_pad_deg=(16.0, 16.0),  # a wide, short view: room above 3 and 4, right of 2
        run_root=FINAL_RUN_ROOT,  # example 23's final run (see TC)
    ),
    Entry(
        "23b",
        "Tree_Crops_Twifo_example",
        "Twifo Praso, Ghana: oil palm estate blocks, Delineate-Anything v2, FTW and embeddings",
        [
            Layer(f"{TC}/twifo/fields_spot-pan_da_2020.gpkg", "SPOT-Pan 1.5 m"),
            Layer(
                f"{TC}/twifo/fields_spot-pan_da-sam2_2020.gpkg",
                "SPOT-Pan 1.5 m + SAM 2",
                highlight="agribound:sam_refined",
            ),
            Layer(f"{TC}/twifo/fields_spot-pan_da-ft_2020.gpkg", "SPOT-Pan 1.5 m, fine-tuned"),
            Layer(f"{TC}/twifo/fields_sentinel2_da_2020.gpkg", "Sentinel-2 10 m"),
            Layer(f"{TC}/twifo/fields_landsat-pan_da_2020.gpkg", "Landsat PAN 15 m"),
            Layer(f"{TC}/twifo/fields_sentinel2_ftw_2020.gpkg", "FTW, Sentinel-2"),
            Layer(
                f"{TC}/twifo/fields_google-embedding_k20_2020.gpkg",
                "Google embedding, k = 20",
                background_from=f"{TC}/twifo/fields_sentinel2_da_2020.gpkg",
                bg_role="the Sentinel-2 composite of the year (the engine reads the embeddings)",
            ),
            Layer(
                f"{TC}/twifo/fields_tessera-embedding_k20_2020.gpkg",
                "TESSERA, k = 20",
                background_from=f"{TC}/twifo/fields_sentinel2_da_2020.gpkg",
                bg_role="the Sentinel-2 composite of the year (the engine reads the embeddings)",
            ),
        ],
        reference=f"{TC}/references/twifo_reference_{'?' * 12}.gpkg",
        reference_label="RSPO GeoRSPO estate blocks (member-declared)",
        per_layer_background=True,
        crop_on_reference=True,  # the 2 km square with the most reference blocks
        crop_m=2000,
        ncols=4,
        model_notes=[
            # 185 training chips (and 23 validation chips): "Prepared 208 training chips ...
            # 185 train, 23 val" in the 1.1 run's logs/23.log (the fine-tuning the final run
            # reused).
            "Panels 1-5: Delineate-Anything v2 (large_v2); panel 3 fine-tuned on 109 blocks of the "
            "NORPALM estate (69 km away; SPOT-Pan 2020, 185 training chips, 20 epochs, "
            "yolo_lr0=1e-4)"
        ],
        sam2_note=True,
        run_root=FINAL_RUN_ROOT,  # example 23's final run (see TC): TESSERA made again
    ),
    Entry(
        "23c",
        "Tree_Crops_Oro_example",
        "Oro, Papua New Guinea: oil palm smallholder parcels, released and fine-tuned models",
        [
            Layer(f"{TC}/oro/fields_spot-pan_da_2021.gpkg", "SPOT-Pan 1.5 m, released"),
            Layer(f"{TC}/oro/fields_spot-pan_da-ft_2021.gpkg", "SPOT-Pan 1.5 m, fine-tuned"),
            Layer(
                f"{TC}/oro/fields_google-embedding_k20_2021.gpkg",
                "Google embedding, k = 20",
                background_from=f"{TC}/oro/fields_sentinel2_da_2021.gpkg",
                bg_role="the Sentinel-2 composite of the year (the engine reads the embeddings)",
            ),
        ],
        reference=f"{TC}/references/oro_reference_{'?' * 12}.gpkg",
        reference_label="RSPO GeoRSPO smallholder parcels (member-declared)",
        per_layer_background=True,
        crop_on_reference=True,  # the 1.5 km square with the most reference parcels
        crop_m=1500,
        ncols=3,
        model_notes=[
            # 71 training chips (and 6 validation chips; logs/23.log of the 1.1 run); the
            # parcels counted are those that intersect a training chip with a positive area.
            "Panels 1-2: Delineate-Anything v2 (large_v2); panel 2 fine-tuned on Higaturu scheme "
            "parcels in a 6 km square 13.6 km away (SPOT-Pan March 2021, 71 training chips "
            "holding parts of 437 of the square's 564 parcels, 20 epochs, yolo_lr0=1e-4)"
        ],
        run_root=FINAL_RUN_ROOT,  # example 23's final run (see TC)
    ),
    Entry(
        "23d",
        "Tree_Crops_Madera_example",
        "Madera County, California: almond and pistachio orchards, NAIP 1 m to Landsat PAN 15 m",
        [
            Layer(f"{TC}/madera/fields_naip_da_2022.gpkg", "NAIP 1 m"),
            Layer(f"{TC}/madera/fields_spot-pan_da_2022.gpkg", "SPOT-Pan 1.5 m"),
            Layer(f"{TC}/madera/fields_sentinel2_da_2022.gpkg", "Sentinel-2 10 m"),
            Layer(f"{TC}/madera/fields_landsat-pan_da_2022.gpkg", "Landsat PAN 15 m"),
            Layer(f"{TC}/madera/fields_sentinel2_ftw_2022.gpkg", "FTW, Sentinel-2"),
            Layer(
                f"{TC}/madera/fields_google-embedding_k20_2022.gpkg",
                "Google embedding, k = 20",
                background_from=f"{TC}/madera/fields_sentinel2_da_2022.gpkg",
                bg_role="the Sentinel-2 composite of the year (the engine reads the embeddings)",
            ),
        ],
        reference=f"{TC}/references/madera_reference_{'?' * 12}.gpkg",
        reference_label="DWR / Land IQ 2022 fields (all crops)",
        per_layer_background=True,
        crop_on_reference=True,  # the 2.5 km square with the most reference fields
        crop_m=2500,
        ncols=3,
        model_notes=["Panels 1-4: Delineate-Anything v2 (large_v2), as released"],
        run_root=FINAL_RUN_ROOT,  # example 23's final run (see TC)
    ),
    Entry(
        "23e",
        "Tree_Crops_DINOv3_example",
        "Tree crops: Delineate-Anything v2 (released, fine-tuned) and DINOv3 (fine-tuned)",
        [
            Layer(
                f"{TC}/{slug}/fields_spot-pan_{run}_{year}.gpkg",
                f"{name}: {model}",
                crop_m=crop,
                reference=f"{TC}/references/{slug}_reference_{'?' * 12}.gpkg",
                area=slug,  # the three models of a study area share its window and dot
            )
            for slug, year, name, crop in [
                ("twifo", 2020, "Twifo Praso", 2000),
                ("oro", 2021, "Oro", 1000),
                ("madera", 2022, "Madera", 2000),
                ("jaen", 2023, "Úbeda", 1000),
            ]
            for run, model in [
                ("da", "DA v2 released"),
                ("da-ft", "DA v2 fine-tuned"),
                ("dinov3-ft", "DINOv3 fine-tuned"),
            ]
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=2000,
        crop_on_reference=True,  # per area: the square with the most reference polygons
        reference_label="Reference (RSPO GeoRSPO, member-declared; DWR / Land IQ; SIGPAC)",
        ncols=3,
        inset_pad_deg=(16.0, 16.0),
        model_positions=True,  # each area's two checkpoints: "(study area 2)"
        model_notes=[
            "Fine-tuned on the SPOT-Pan chips of a training area near each study area, never the "
            "evaluated square: 1 NORPALM estate, 2 Higaturu scheme parcels, 3 Cressey (DWR), "
            "4 Ibros (SIGPAC); Delineate-Anything 20 epochs, yolo_lr0=1e-4; DINOv3 ViT-L/16 "
            "(SAT-493M), full fine-tuning, at most 20 epochs"
        ],
        run_root=FINAL_RUN_ROOT,  # example 23's final run (see TC)
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
        # The final run, made afresh: the HLS composite counts its images without the
        # antimeridian footprints (HLS recipe 2).
        run_root=FINAL_RUN_ROOT,
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
        "San Juan County, NM: Delineate-Anything (2024) vs DRI/NMOSE V18",
        [
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2024.gpkg",
                "Delineate-Anything",
            )
        ],
        # The zip archive is read in place (one shapefile in it).
        reference="examples/NM_Field_Boundaries.zip",
        reference_label="DRI/NMOSE field boundaries (V18; not used for training in these runs)",
        reference_min_area_m2=2500,  # the evaluated reference (the predictions' minimum area)
        zoom_m=2500,
        metrics="outputs/stratified_evaluation/metrics_sentinel2_delineate-anything_2024.json",
    ),
    Entry(
        "20b",
        "San_Juan_resolution_example",
        "San Juan County, NM: Delineate-Anything v2 from 30 m to 1 m (2022)",
        [
            Layer(
                "outputs/stratified_evaluation/fields_landsat_delineate-anything_2022.gpkg",
                "Landsat 30 m",
            ),
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2022.gpkg",
                "Sentinel-2 10 m",
            ),
            Layer(
                "outputs/stratified_evaluation/fields_spot_delineate-anything_2022.gpkg",
                "SPOT 6/7 6 m",
            ),
            Layer(
                "outputs/stratified_evaluation/fields_naip_delineate-anything_2022.gpkg",
                "NAIP 1 m",
            ),
        ],
        reference="examples/NM_Field_Boundaries.zip",
        reference_label="DRI/NMOSE field boundaries (V18; not used for training in these runs)",
        reference_min_area_m2=2500,  # the evaluated reference (the predictions' minimum area)
        per_layer_background=True,
        crop_m=2500,
        crop_on_reference=True,  # the window with the most reference fields
        ncols=2,
        metrics="outputs/stratified_evaluation/metrics_naip_2022.json",
    ),
    Entry(
        "20c",
        "San_Juan_crop_filter_example",
        "San Juan County, NM: FTW and Delineate-Anything v2 with and without the crop filter "
        "(2024)",
        [
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_ftw_2024.gpkg",
                "FTW, crop filter on",
            ),
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_ftw_2024_nolulc.gpkg",
                "FTW, crop filter off",
                removed_vs="outputs/stratified_evaluation/fields_sentinel2_ftw_2024.gpkg",
            ),
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2024.gpkg",
                "DA v2, crop filter on",
            ),
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2024_nolulc.gpkg",
                "DA v2, crop filter off",
                removed_vs="outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2024.gpkg",
            ),
        ],
        reference="examples/NM_Field_Boundaries.zip",
        reference_label="DRI/NMOSE field boundaries (V18; not used for training in these runs)",
        reference_min_area_m2=2500,  # the evaluated reference (the predictions' minimum area)
        per_layer_background=True,
        crop_m=3000,
        crop_on_reference=True,  # the square with the most reference fields
        ncols=2,
    ),
    Entry(
        "13",
        "SAM2_refinement_example",
        "SAM 2 refinement of example 20's output (Sentinel-2 2024)",
        [
            Layer(
                "outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2024.gpkg",
                "Delineate-Anything",
            ),
            Layer(
                "outputs/sam2_refinement/fields_sentinel2_delineate-anything_2024_sam2.gpkg",
                "After SAM 2",
                highlight="agribound:sam_refined",
            ),
        ],
        reference="examples/NM_Field_Boundaries.zip",
        reference_label="DRI/NMOSE field boundaries (V18; not used for training in these runs)",
        reference_min_area_m2=2500,  # the evaluated reference (the predictions' minimum area)
        background_from="outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2024.gpkg",
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
    _nevada_entry(
        "24",
        "Nevada_Diamond_Valley_example",
        "diamond_valley",
        "Diamond Valley, NV (centre pivots): released and sam-auto runs on NAIP, SPOT and "
        "Sentinel-1",
        (584000, 4389000),
    ),
    _nevada_entry(
        "24b",
        "Nevada_Mason_Valley_example",
        "mason_valley",
        "Mason Valley, NV (flood and sprinkler irrigation): released, fine-tuned and sam-auto "
        "runs on NAIP, SPOT and Sentinel-1",
        (310000, 4306000),
    ),
    _nevada_entry(
        "24c",
        "Nevada_Diamond_Valley_irrigated_example",
        "diamond_valley",
        "Diamond Valley, NV (centre pivots), irrigated fields only: released and sam-auto runs "
        "on NAIP, SPOT and Sentinel-1",
        (584000, 4389000),
        irrigated=True,
    ),
    _nevada_entry(
        "24d",
        "Nevada_Mason_Valley_irrigated_example",
        "mason_valley",
        "Mason Valley, NV (flood and sprinkler irrigation), irrigated fields only: released, "
        "fine-tuned and sam-auto runs on NAIP, SPOT and Sentinel-1",
        (310000, 4306000),
        irrigated=True,
    ),
    _nevada_options_entry(),
    Entry(
        "25",
        "Multistate_OpenET_overview_example",
        "47 states: F1 of Delineate-Anything v2 as released on NAIP, one 5 km square per state",
        [],
        overview=Overview(
            summary=f"{MS}/summary.csv",
            run="naip",
            metric_label="Object F1 (IoU >= 0.5)",
            marks=[state for state, *_ in MULTISTATE_PANELS],
            # The entry's title is the documentation's caption, not printed on the map.
            panel_label="NAIP 1 m: Delineate-Anything v2 as released",
            notes=[
                "Markers: the centre of each state's 5 km study square (example 25), drawn "
                f"far larger than the square; numbers {_multistate_order('F1')}",
                "Colour: object F1 of Delineate-Anything v2 (large_v2) as released on NAIP "
                "1 m (the latest year whose images cover the square), one-to-one at "
                "IoU >= 0.5 against the square's reference fields of at least 2,500 m²",
                f"Reference: {OPENET_CREDIT}; New Mexico: DRI/NMOSE field boundaries (V18), "
                "Desert Research Institute and New Mexico Office of the State Engineer; "
                "Nevada: DRI field boundaries (Nevada, 2026-08-20), Desert Research Institute",
            ],
        ),
        inset=False,  # the map is its own locator
    ),
    _multistate_squares(),
    _montana_years(),
    _sar_entry(
        "26",
        "SAR_held_out_example",
        "Radar, held-out sites: Sentinel-1, Umbra and Capella against NAIP with the same settings",
        ["tamu", "intel_oh", "orleans_vt"],
        reference_label="OpenET field boundaries v2.1.1 (USDA FSA Common Land Units)",
        model_notes=[
            SAR_SETTINGS_NOTE,
            "Held-out sites: the Umbra and Capella recipe and these settings were fixed "
            "before the sites were scored (example 26)",
            SAR_SCORES_NOTE,
            f"Reference: {OPENET_CREDIT} (here the Common Land Units of the USDA Farm Service "
            "Agency)",
        ],
    ),
    _sar_entry(
        "26b",
        "SAR_in_sample_example",
        "Radar, in-sample sites: Sentinel-1, Umbra and Capella against NAIP or Sentinel-2",
        ["ucd", "sevilla", "centerfield"],
        reference_label="Reference (1 DWR / Land IQ 2024; 2 SIGPAC 2025; 3 Utah WRLU 2025)",
        model_notes=[
            SAR_SETTINGS_NOTE,
            "In-sample sites: the Umbra and Capella recipe and these settings were chosen on "
            "these areas (example 26)",
            SAR_SCORES_NOTE,
            # SIGPAC (CC BY 4.0) is drawn as example 26 evaluates it: the recintos of field
            # land uses (SIGPAC_NON_FIELD left out) of at least 2,500 m2, hence "modified".
            "Reference: 1 California Department of Water Resources, Statewide Crop Mapping "
            "2024 (provisional, with Land IQ); 2 SIGPAC recintos of 2025, Fondo Español de "
            "Garantía Agraria (FEGA), CC BY 4.0, modified (field land uses only, at least "
            "2,500 m²); 3 Utah Division of Water Resources, Water Related Land Use 2025",
        ],
    ),
    Entry(
        "27",
        "Quebec_crop_sequence_example",
        "Quebec: crop-history zones (ACI 2016-2023) and Delineate-Anything v2 on Sentinel-2 "
        "against the declared parcels",
        [
            Layer(
                f"{QC}/{slug}/{stem}.gpkg",
                f"{place}: {label}",
                reference=f"{QC}/{slug}/reference.gpkg",
                area=slug,
                metrics=_metrics_of(f"{QC}/{slug}/{stem}.gpkg"),
                noun=noun,
                # The zones' own raster is the ACI stack: draw them on the Sentinel-2 image.
                background_from=(
                    f"{QC}/{slug}/fields_sentinel2_da_2023.gpkg" if noun == "zone" else None
                ),
            )
            for slug, place in QUEBEC_SQUARES.items()
            for stem, label, noun in (
                ("zones_aci-sequence_2016-2023", "zones, ACI 2016-2023", "zone"),
                ("fields_sentinel2_da_2023", "DA v2, Sentinel-2 2023", "field"),
            )
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=2500,
        crop_on_reference=True,  # the square with the most declared parcels
        reference_label="BDPPAD 2023 declared parcels, La Financière agricole du Québec",
        pred_label=f"agribound {__version__} zones or fields",
        bg_role="the Sentinel-2 composite of 2023, the input of Delineate-Anything",
        ncols=2,
        title_f1=True,
        copernicus_credit=True,  # the Sentinel-2 composites (example 27's attribution)
        run_root=FINAL_RUN_ROOT,
        model_notes=[
            "Zones: crop-history zones of the AAFC Annual Crop Inventory, not field outlines: "
            "neighbouring parcels with the same crop sequence share a zone",
            "Counts: the whole output; F1: the whole 5 km square, one-to-one at IoU >= 0.5",
            # The evaluated parcels (example 27's reference.gpkg): a subset, hence "modified".
            "Reference: BDPPAD (Base de données des parcelles et productions agricoles "
            "déclarées) 2023, La Financière agricole du Québec, CC BY 4.0 (Données Québec), "
            "modified (declared parcels only: a production declared for 2023, at least "
            "2,500 m²)",
            "ACI: Agriculture and Agri-Food Canada, Annual Crop Inventory 2016-2023; contains "
            "information licensed under the Open Government Licence – Canada (2.0)",
        ],
    ),
    *[_new_york_entry(region) for region in NY_REGIONS],
    _sam_auto_entry(),
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
    em = prov.get("engine_meta") or {}
    if note and cfg.get("fine_tune"):
        how = ""
        if "use_lora" in em:
            how = "LoRA, " if em.get("use_lora") else "full, "
        note += (
            f", fine-tuned on the reference polygons ({how}up to "
            f"{cfg.get('fine_tune_epochs')} epochs, {cfg.get('fine_tune_split', 'block')} split)"
        )
    if note and cfg.get("engine") == "dinov3":
        note += _dinov3_grid_note(em)
    return note


def _dinov3_grid_note(em: dict) -> str:
    """DINOv3's processing grid (engine_meta ``processing``): ", read on a 10 m bilinear copy
    of the 30 m raster" when the run read a resampled copy (rasters coarser than 10 m), for
    fine-tuning and inference alike; empty when it read the raster as it is.

    Delineate-Anything records its own ``processing`` (e.g. ``processing_gsd_m=3`` for the
    Umbra and Capella composites); the entries that use it say so in their notes.
    """
    proc = em.get("processing")
    if not isinstance(proc, dict) or not proc.get("gsd_m"):
        return ""
    gsd = float(proc["gsd_m"])
    how = f" {proc['resampling']}" if proc.get("resampling") else ""
    native = f" of the {gsd * float(proc['scale']):g} m raster" if proc.get("scale") else ""
    return f", read on a {gsd:g} m{how} copy{native}"


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
        geoai = f"geoai-py {em.get('geoai-py_version') or ver.get('geoai-py')}"
        if not cfg.get("fine_tune") and (cfg.get("engine_params") or {}).get("checkpoint_path"):
            # A checkpoint fine-tuned in another run (e.g. on a training area): name it, as
            # for Delineate-Anything; the backbone weights are only its starting point.
            sha = str(em.get("checkpoint_sha256") or "")[:7]
            return (
                f"DINOv3 {em.get('model_name')} fine-tuned checkpoint (sha256 {sha}; from "
                f"{em.get('weights')} @ {rev}; {geoai})"
            )
        return f"DINOv3 {em.get('model_name')} with {em.get('weights')} @ {rev} ({geoai})"
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
    if engine == "crop-sequence":
        years = em.get("years") or []
        dataset = em.get("dataset") or em.get("source")
        name = CROP_SEQUENCE_DATASETS.get(str(dataset), str(dataset))
        span = f" {years[0]}-{years[-1]}" if years else ""
        size = f" at {float(em['pixel_size_m']):g} m" if em.get("pixel_size_m") else ""
        region = (em.get("params") or {}).get("min_region_m2")
        merged = f", regions under {float(region) / 1e4:g} ha merged" if region else ""
        crop = (
            f", cultivated in at least {em['min_crop_years']} of {len(years)} years"
            if em.get("min_crop_years") and years
            else ""
        )
        return (
            f"Crop-history zones ({em.get('implementation')}): {name}{span}{size}{crop}, "
            f"focal mode of radius {em.get('smooth_radius_px')} px{merged}; zones of equal "
            "crop sequence"
        )
    if engine == "sam-auto" and em.get("model"):
        return _sam_auto_note(cfg, em)
    if engine and em.get("model"):
        return f"{engine}: {em.get('model')}"
    return None


# A no-break space: footer text that _wrap keeps on one line (printed as a space); the
# gallery_stats.json copies of the footer have plain spaces (_plain).
NBSP = "\u00a0"


def _plain(text: str) -> str:
    """*text* with its no-break spaces (``NBSP``) as plain spaces, for gallery_stats.json."""
    return text.replace(NBSP, " ")


# SAM 2.1's maker and licence (the checkpoint's model card and Meta's repository), printed under
# every image with a sam-auto panel together with the sam2 build that ran it (_sam_auto_credit).
SAM_AUTO_LICENCE = "Meta, Apache-2.0"


def _sam_auto_note(cfg: dict, em: dict) -> str:
    """Model line of a ``sam-auto`` run, from its engine_meta: the checkpoint and its
    revision, and the setting it ran with ("the engine's NAIP default: 2,048 m tiles, 32
    points per side, predicted IoU > 0.8, stability >= 0.95"). The line differs between
    sources, so each names its panels in an entry with ``model_positions``."""
    rev = str((em.get("checkpoint") or {}).get("revision") or "")[:7]
    s = em.get("settings") or {}
    source = cfg.get("source")
    parts = []
    if s.get("sam_tile_m"):
        parts.append(f"{float(s['sam_tile_m']):,.0f} m tiles")
    if s.get("points_per_side"):
        parts.append(f"{s['points_per_side']} points per side")
    if s.get("pred_iou_thresh") is not None:
        parts.append(f"predicted IoU > {float(s['pred_iou_thresh']):g}")
    if s.get("stability_score_thresh") is not None:
        parts.append(f"stability >= {float(s['stability_score_thresh']):g}")
    basis = em.get("settings_basis")
    # The source's name held on one line of the footer (_wrap breaks at ASCII spaces only).
    label = SOURCE_LABELS.get(source or "", source or "the source").replace(" ", NBSP)
    how = f"the engine's {label} default" if basis == "default" else f"settings ({basis})"
    model = f"{em.get('model')} @ {rev}" if rev else str(em.get("model"))
    return f"SAM 2.1 automatic masks (sam-auto): {model}, {how}: {', '.join(parts)}"


def _sam_auto_credit(provs) -> str | None:
    """The credit line of the ``sam-auto`` panels (provenance records *provs*): SAM 2.1's
    maker and licence (``SAM_AUTO_LICENCE``), the sam2 build that ran it (the PyPI sam2 is a
    third-party build; engine_meta records the commit of Meta's code its files equal) and
    the device (outputs differ between devices). None when no panel is a sam-auto run."""
    builds = []
    for pv in provs:
        cfg = pv.get("config") or pv.get("base_config") or {}
        if cfg.get("engine") != "sam-auto":
            continue
        em = pv.get("engine_meta") or {}
        sam2 = em.get("sam2") or {}
        build = f"sam2 {sam2.get('version', '?')}"
        if sam2.get("meta_commit"):
            n = f"{sam2['n_files']} " if sam2.get("n_files") else ""
            build += (
                f", a third-party PyPI build whose {n}code, configuration and licence files "
                f"equal Meta's commit {str(sam2['meta_commit'])[:7]}"
            )
        if em.get("device"):
            build += f", run on device {em['device']}"
        builds.append(build)
    if not builds:
        return None
    return f"SAM 2.1: {SAM_AUTO_LICENCE}; " + "; ".join(dict.fromkeys(builds))


def _bare_axes(ax) -> None:
    """No ticks and no axis labels on a map (call after the geopandas plots).

    geopandas >= 1.2 labels the axes of every ``.plot()`` with the CRS axis names
    ("Easting [metre]"); ``add_labels=False`` would break geopandas 1.0 and 1.1.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


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


def _densest_window(gdf: gpd.GeoDataFrame, size: float, extent, within=None, record=None):
    """Square window of side *size* with the most polygon representative points.

    The candidates are half a side apart inside *extent* (the first wins a tie, so the choice
    is deterministic); with *within* (a shapely geometry in the same CRS), only those that lie
    inside it. If none does, the densest of all the candidates is returned with a warning.
    *record* (a dict), when given, receives ``candidates``: the number of squares the window
    was chosen from.
    """
    x0, x1, y0, y1 = extent
    c = gdf.geometry.representative_point()
    xs, ys = c.x.to_numpy(), c.y.to_numpy()
    best, best_n, n_all = None, -1, 0
    inside, inside_n, n_inside = None, -1, 0
    step = size / 2
    for cx in np.arange(x0 + size / 2, x1 - size / 2 + 1e-6, step):
        for cy in np.arange(y0 + size / 2, y1 - size / 2 + 1e-6, step):
            n = int(np.count_nonzero((np.abs(xs - cx) <= size / 2) & (np.abs(ys - cy) <= size / 2)))
            square = (cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2)
            n_all += 1
            if n > best_n:
                best, best_n = square, n
            if within is not None and within.contains(
                shapely.box(square[0], square[2], square[1], square[3])
            ):
                n_inside += 1
                if n > inside_n:
                    inside, inside_n = square, n
    if within is not None and inside is None:
        warnings.warn(
            f"none of the {n_all} candidate squares of {size:g} m lies inside the given "
            "polygon: the window is chosen among all of them",
            UserWarning,
            stacklevel=2,
        )
    if record is not None:
        record["candidates"] = n_inside if inside is not None else n_all
    return inside if inside is not None else best


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
        "AGRIBOUND_SENSORS": "sensors",
        "AGRIBOUND_ORBIT_PASS": "orbit_pass",
        "AGRIBOUND_ATTRIBUTION": "attribution",
        "AGRIBOUND_LICENSE": "license",
        "AGRIBOUND_LICENSE_URL": "license_url",
    }
    facts = {v: comp[k] for k, v in keep.items() if k in comp}
    if comp.get("AGRIBOUND_IMAGE_YEARS_UNDERNEATH"):
        # A NAIP mosaic with images of other years under those of its year (agribound 1.1
        # composites record them): the image list gives each image's date (_mosaic_images).
        facts["image_years_underneath"] = comp["AGRIBOUND_IMAGE_YEARS_UNDERNEATH"]
        if comp.get("AGRIBOUND_IMAGE_LIST"):
            facts["image_list"] = comp["AGRIBOUND_IMAGE_LIST"]
    return facts


def _mosaic_images(comp: dict) -> str | None:
    """The images of a mosaic with images of other years underneath, by date (composite
    facts with ``image_years_underneath`` and ``image_list``): "2 of 2022-07-04 on top, gaps
    filled from 2 of 2021-11-06". None for other composites or an unreadable list."""
    if not comp.get("image_years_underneath") or not comp.get("image_list"):
        return None
    try:
        images = json.loads(comp["image_list"])
        dates = [str(im[1])[:10] for im in images]
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    top_years = {y.strip() for y in str(comp.get("image_years") or "").split(",") if y.strip()}
    counts: dict[str, int] = {}
    for d in sorted(dates, reverse=True):  # newest first, as the mosaic stacks them
        counts[d] = counts.get(d, 0) + 1
    top = [f"{n} of {d}" for d, n in counts.items() if d[:4] in top_years]
    under = [f"{n} of {d}" for d, n in counts.items() if d[:4] not in top_years]
    if not top or not under:
        return None
    return f"{_and_join(top)} on top, gaps filled from {_and_join(under)}"


def _background_for(entry: Entry, root: Path, prov: dict, basemap: str, layer: Layer | None = None):
    """Background GeoTIFF(s), source, provenance for its dates and role note of one layer."""
    if layer is not None and layer.background_from:
        bprov = _sidecar(_one(layer.background_from, root))
        raster = (bprov.get("facts") or {}).get("raster_path")
        if not raster:
            raise FileNotFoundError(
                f"{layer.background_from!r}: its provenance records no facts.raster_path"
            )
        source = layer.background_source or (bprov.get("config") or {}).get("source")
        role = layer.bg_role or entry.bg_role or "the engine's input"
        return [root / raster], source, bprov, role
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


def _bg_credit(bg) -> str:
    """The credit line printed with a background ``(tifs, source, provenance, role)``."""
    tifs, source, comp_prov, _role = bg
    return _credit(source, _composite_facts(comp_prov, tifs[0] if tifs else None))


def _copernicus_for(entry: Entry, bgs, provs) -> str | None:
    """With ``Entry.copernicus_credit``: the Copernicus line of the backgrounds drawn (*bgs*)
    and of the panels' own inputs (*provs*, their provenance records); else None."""
    if not entry.copernicus_credit:
        return None
    drawn = [(bg[1], _composite_facts(bg[2], bg[0][0] if bg[0] else None)) for bg in bgs]
    inputs = [((pv.get("config") or {}).get("source"), _composite_facts(pv)) for pv in provs]
    return _copernicus_note(drawn + inputs)


def _bg_note(bg, role: bool = True, exclusive: bool = True) -> str:
    """Note on one background: source, credit, window and image count, then its role.

    Without *role* the role is left out; without *exclusive*, the "(end exclusive)" after
    the window (for a list that says it once).
    """
    tifs, source, comp_prov, bg_role = bg
    comp = _composite_facts(comp_prov, tifs[0] if tifs else None)
    label = SOURCE_LABELS.get(source or "", source or "imagery")
    if source == "landsat":
        label = _landsat_label(comp.get("collections"))
    elif source == "landsat-pan":
        label = _landsat_pan_label(comp.get("sensors"))
    elif source == "sentinel1":
        label = _sentinel1_label(comp.get("orbit_pass"))
    if comp.get("resolution_m"):
        label += f" {float(comp['resolution_m']):g} m"
    window = ""
    if comp.get("date_start") and comp.get("date_end_exclusive"):
        window = (
            f" {comp.get('composite_method', '')} composite, "
            f"{comp['date_start']} to {comp['date_end_exclusive']}"
            + (" (end exclusive)" if exclusive else "")
        )
    elif comp.get("image_years"):
        window = f", {comp['image_years']} imagery"
    n_img = f", {_count(comp['n_images'], _image_noun(source))}" if comp.get("n_images") else ""
    if n_img and (mosaic := _mosaic_images(comp)):  # images of other years underneath
        n_img += f": {mosaic}"
    if not n_img and comp.get("lock_raster_ids"):
        n_img = f", {len(comp['lock_raster_ids'].split(','))} source rasters"
    # The credit line a licence requires (Umbra, Capella; _credit).
    credit = _credit(source, comp)
    credit = f" ({credit})" if credit else ""
    return f"{label}{credit}{window}{n_img}" + (f"; {bg_role}" if role else "")


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
    _bare_axes(iax)
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


# A token after which the notes never break a line: a number ("5", "0.70", "2,500"), a list
# number ("3:") or a copyright sign ("(©" of "(© Capella Space Corp").
_NO_BREAK_AFTER = re.compile(r"^(\d[\d,.]*:?|\(?©)$")


def _wrap(text: str, width_in: float, fontsize: float = NOTE_FS) -> list[str]:
    """Balanced wrap of *text* to *width_in* inches (measured, not counted in characters).

    First the fewest lines that fit, then the narrowest width that still gives that
    number of lines, so the lines have similar lengths and none is left with one word.
    A single word wider than *width_in* keeps a line of its own. A number, a list number
    such as "3:" or a copyright sign stays on the line of the word after it ("5 km",
    "0.70 (CA)", "3: 2023-01-01", "© Capella"; ``_NO_BREAK_AFTER``), and lines break at
    ASCII spaces only, so a no-break space (U+00A0) holds its words together too.
    """
    tokens = [w for w in re.split(r"[ \t\r\n]+", text) if w]
    words, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and _NO_BREAK_AFTER.match(tokens[i]):
            words.append(f"{tokens[i]} {tokens[i + 1]}")
            i += 2
        else:
            words.append(tokens[i])
            i += 1
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


class _NotScoredSwatch(Rectangle):
    """Legend handle of the predictions a score did not count (``Entry.scored_part``): its
    face and edge are those of the sample (:class:`_NotScoredHandler`)."""


class _NotScoredHandler(HandlerBase):
    """Legend sample of the predictions not scored, drawn as on the maps: the translucent grey
    fill, then the lighter grey casing (``IGNORED_CASING``, ``IGNORED_CASING_LW`` points wider
    than the edge) and the dark grey edge on it."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        box = {"xy": (-xdescent, -ydescent), "width": width, "height": height, "transform": trans}
        lw = orig_handle.get_linewidth()
        artists = [Rectangle(**box, facecolor=orig_handle.get_facecolor(), edgecolor="none", lw=0)]
        if IGNORED_CASING_LW > 0:
            casing = to_rgba(IGNORED_CASING, IGNORED_CASING_ALPHA)
            artists.append(
                Rectangle(**box, fill=False, edgecolor=casing, lw=lw + IGNORED_CASING_LW)
            )
        artists.append(Rectangle(**box, fill=False, edgecolor=orig_handle.get_edgecolor(), lw=lw))
        return artists


class _LeftOutSwatch(Rectangle):
    """Legend handle of the reference fields a score left out (``Entry.scored_part``)."""


class _LeftOutHandler(HandlerBase):
    """Legend sample of the left-out fields: their outline and white hatch on a grey ground.

    The hatch lines are drawn as on the maps (:func:`_hatch_lines`): at 45 degrees,
    ``LEFT_OUT_HATCH_IN`` inches apart, with the maps' widths, opacities and casing.
    """

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        x0, y0 = -xdescent, -ydescent
        ground = Rectangle(
            (x0, y0),
            width,
            height,
            facecolor=LEFT_OUT_SWATCH,
            edgecolor=LEFT_OUT_EDGE,
            linewidth=0.8,
            transform=trans,
        )
        artists = [ground]
        step = LEFT_OUT_HATCH_IN * 72 * math.sqrt(2.0)  # along x, in points (the box's units)
        for c in np.arange(-math.floor(height / step) * step, width, step):
            # The line x - y = c of the box, from its lower or left edge to its upper or right.
            xs, xe = max(0.0, c), min(width, height + c)
            if xs >= xe:
                continue
            for colour, alpha, lw in (
                (LEFT_OUT_CASING, LEFT_OUT_CASING_ALPHA, LEFT_OUT_CASING_LW),
                (LEFT_OUT_HATCH, LEFT_OUT_HATCH_ALPHA, LEFT_OUT_HATCH_LW),
            ):
                line = Line2D(
                    [x0 + xs, x0 + xe],
                    [y0 + xs - c, y0 + xe - c],
                    color=colour,
                    alpha=alpha,
                    linewidth=lw,
                    solid_capstyle="butt",
                    transform=trans,
                )
                line.set_clip_path(ground)  # the wide casing ends inside the sample
                artists.append(line)
        return artists


# Legend handlers of the gallery's own handles (passed to every legend; unused by the others).
LEGEND_HANDLERS = {_LeftOutSwatch: _LeftOutHandler(), _NotScoredSwatch: _NotScoredHandler()}


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
            handler_map=LEGEND_HANDLERS,
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


def _removed_mask(gdf: gpd.GeoDataFrame, kept_path: str, root: Path, crs) -> np.ndarray:
    """Polygons of *gdf* missing from the crop-filtered output *kept_path*.

    A polygon counts as kept when its representative point lies within a polygon of
    the filtered output (the filter drops polygons, it does not reshape them).
    """
    kept = gpd.read_file(_one(kept_path, root)).to_crs(crs)
    pts = gpd.GeoDataFrame(geometry=gdf.geometry.representative_point(), crs=crs)
    hit = gpd.sjoin(pts, kept[["geometry"]], how="left", predicate="within")
    in_kept = hit.groupby(level=0)["index_right"].apply(lambda s: s.notna().any())
    return ~in_kept.reindex(gdf.index, fill_value=False).to_numpy()


def _read_reference(pattern: str, root: Path, crs, min_area_m2: float = 0.0):
    """Reference polygons of *pattern* in *crs* (None: the file's own CRS), without empty
    geometries.

    With *min_area_m2*, the polygons smaller than that are dropped first, with
    agribound's area filter (areas in EPSG:6933), as the examples' evaluations do.
    """
    ref = gpd.read_file(_one(pattern, root))
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty]
    if min_area_m2 > 0:
        from agribound.postprocess import filter_polygons

        ref = filter_polygons(ref, min_area_m2=min_area_m2)
    return ref if crs is None else ref.to_crs(crs)


def _hatch_lines(polys: gpd.GeoSeries, spacing: float) -> gpd.GeoSeries:
    """Lines at 45 degrees, *spacing* apart (map units, measured across them), clipped to the
    union of *polys*.

    The lines lie on one lattice of the map (``x - y`` a multiple of ``spacing * sqrt(2)``),
    so a polygon is hatched alike in every panel of a figure. Empty without polygons.
    """
    geoms = [g for g in polys if g is not None and not g.is_empty]
    if not geoms:
        return gpd.GeoSeries([], crs=polys.crs)
    union = shapely.union_all(geoms)
    minx, miny, maxx, maxy = union.bounds
    step = spacing * math.sqrt(2.0)
    lines = [  # x - y = c, from the bottom (y = miny) to the top (y = maxy) of the union
        shapely.LineString([(miny + c, miny), (maxy + c, maxy)])
        for c in np.arange(math.floor((minx - maxy) / step) * step, maxx - miny + step, step)
    ]
    hatch = shapely.intersection(shapely.MultiLineString(lines), union)
    return gpd.GeoSeries([hatch], crs=polys.crs)


def _not_scored(pred: gpd.GeoDataFrame, left_out: gpd.GeoDataFrame, equal_area_crs) -> np.ndarray:
    """Mask of the polygons of *pred* that ``ignore`` of agribound.evaluate leaves out when
    *left_out* is passed as ``ignore``, computed with evaluate's own code.

    The predictions are taken in the CRS of *left_out*, the reference's, as example 24
    passes them to evaluate (it reprojects its outputs to EPSG:32611, the CRS of its
    reference). As in evaluate, the invalid polygons of both layers are repaired and the
    null, empty, non-polygonal and zero-area ones dropped, and a prediction is left out when
    the representative point of its repaired polygon lies in a left-out field or on its
    boundary. *equal_area_crs* is the metrics file's (evaluate drops a polygon whose area in
    it is zero); None: evaluate's default.
    """
    # evaluate's private helpers: the rule itself, not a copy of it
    from agribound.evaluate import (
        _drop_ignored,
        _prepare_ignore,
        _prepare_layer,
        _resolve_equal_area_crs,
    )

    mask = np.zeros(len(pred), dtype=bool)
    if not len(pred) or not len(left_out):
        return mask
    ea_crs = _resolve_equal_area_crs(equal_area_crs)
    layer = _prepare_layer(pred.to_crs(left_out.crs), "predicted", ea_crs)
    scored = _drop_ignored(layer, _prepare_ignore(left_out, ea_crs))
    mask[np.setdiff1d(layer.keep, scored.keep)] = True
    return mask


def _scored_panel(part: ScoredPart, metrics_path: Path, ref, pred) -> dict:
    """The scored part of one panel (``Entry.scored_part``), checked against its metrics file.

    *ref* is the whole reference layer in its own CRS, *pred* the panel's output. The year
    comes from the metrics file's summary row. Returns the year, the reference column, the
    mask of the scored reference fields (*column* equal to *value*), the mask of the
    predictions not scored (those ``ignore`` of agribound.evaluate leaves out with the other
    fields as ``ignore``, computed with its own code: :func:`_not_scored`) and the counts.
    Raises ValueError when the counts differ from the metrics file's, so that a figure never
    draws a split other than the one scored.
    """
    data = json.loads(metrics_path.read_text())
    row = data.get("row") if isinstance(data.get("row"), dict) else {}
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    if row.get("year") is None:
        raise ValueError(f"{metrics_path.name}: the summary row gives no year")
    year = int(row["year"])
    column = part.column.format(year=year)
    if column not in ref.columns:
        raise ValueError(f"{metrics_path.name}: the reference has no column {column!r}")
    scored = (ref[column] == part.value).to_numpy(dtype=bool)
    ignored = _not_scored(pred, ref[~scored], metrics.get("equal_area_crs"))
    record = {
        "year": year,
        "column": column,
        "value": part.value,
        "reference_scored": int(scored.sum()),
        "reference_left_out": int((~scored).sum()),
        "predicted": len(pred),
        "predicted_scored": metrics.get("count_predicted"),
        "predicted_ignored": int(ignored.sum()),
        "f1": metrics.get("f1"),
    }
    checks = (
        ("count_reference", record["reference_scored"]),
        ("count_predicted_ignored", record["predicted_ignored"]),
    )
    for key, drawn in checks:
        if metrics.get(key) != drawn:
            raise ValueError(
                f"{metrics_path.name}: {key} is {metrics.get(key)!r}, but the figure would "
                f"draw {drawn}"
            )
    counted = sum(int(metrics.get(k) or 0) for k in ("count_predicted", "count_predicted_dropped"))
    if counted + record["predicted_ignored"] != len(pred):
        raise ValueError(
            f"{metrics_path.name}: {counted} predictions scored or dropped and "
            f"{record['predicted_ignored']} not scored, but the output has {len(pred)}"
        )
    return {**record, "scored_mask": scored, "ignored_mask": ignored}


def _scored_counts(parts: list[dict]) -> str:
    """The scored reference fields per year of the panels (:func:`_scored_panel` records),
    the first year with the size of the reference: "39 of 74 in 2022, 40 in 2023 and 34 in
    2024"."""
    by_year: dict[int, dict] = {}
    for p in parts:
        by_year.setdefault(p["year"], p)
    items = []
    for k, (year, p) in enumerate(sorted(by_year.items())):
        total = p["reference_scored"] + p["reference_left_out"]
        of = f" of {total:,}" if k == 0 else ""
        items.append(f"{p['reference_scored']:,}{of} in {year}")
    return _and_join(items)


def _reference_stats(entry: Entry, ref: gpd.GeoDataFrame) -> dict:
    """``gallery_stats.json`` record of a drawn reference layer."""
    stats = {"label": entry.reference_label, **_area_stats(ref)}
    if entry.reference_min_area_m2 > 0:
        stats["min_area_m2"] = entry.reference_min_area_m2
    return stats


# Scores of agribound.evaluate copied from a panel's metrics file (with the summary row).
METRICS_SCORES = (
    "f1",
    "precision",
    "recall",
    "iou_mean",
    "count_tp",
    "count_fp",
    "count_fn",
    "count_reference",
    "count_predicted",
    "boundary_f1",
    "boundary_tolerance_m",
    "area_weighted_f1",
    "iou_threshold",
    "matching",
)


def _metrics_record(path: Path, root: Path, record: tuple[str, ...] | None = None) -> dict:
    """``gallery_stats.json`` record of a panel's metrics file (``Layer.metrics``).

    Examples 24-28 write ``{"row": <the run's summary row>, "metrics": <its scores>}``. The
    record keeps the file, ``f1`` (the row's, else the scores'), the row's scalars and the
    ``METRICS_SCORES`` of the scores, also one level down (example 27 holds one set of
    scores per boundary tolerance); the per-class tables are left out.

    With *record* (``Layer.metrics_record``), the file holds several records (e.g. the SAM
    evaluation's ``scores.json``) and the one at those keys is read as the scores; the stats
    name the keys (``record``). A missing key raises KeyError.
    """

    def scalars(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is None or isinstance(v, (str, int, float, bool))}

    def scores_of(d: dict) -> dict:
        return {k: d[k] for k in METRICS_SCORES if k in d and k in scalars(d)}

    data = json.loads(path.read_text())
    if record:
        picked = data
        for key in record:
            if not isinstance(picked, dict) or key not in picked:
                raise KeyError(f"{path.name}: no record {'/'.join(record)!r} (missing {key!r})")
            picked = picked[key]
        data = {"metrics": picked if isinstance(picked, dict) else {}}
    raw = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    scores = scores_of(raw)
    for k, v in raw.items():
        if isinstance(v, dict) and (inner := scores_of(v)):
            scores[k] = inner
    row = scalars(data.get("row") if isinstance(data.get("row"), dict) else {})
    return {
        "file": (path.relative_to(root) if path.is_relative_to(root) else path).as_posix(),
        **({"record": list(record)} if record else {}),
        "f1": row.get("f1", scores.get("f1")),
        "row": row,
        "scores": scores,
    }


def _panel_title(
    entry: Entry, lay: Layer, n: int, metrics: dict | None, scored: dict | None = None
) -> str:
    """``"<label> (<n> fields)"``; with ``Entry.title_f1``, ``"(<n> fields; F1 0.38)"``.

    With *scored* (the panel's :func:`_scored_panel` record; ``Entry.scored_part``), two
    lines: the label, then ``"145 of 216 fields scored; F1 0.38, irrigated only"`` (the
    predictions scored, the polygons of the output, and the F1 of that score).
    """
    count = _count(n, lay.noun)
    f1 = (scored or metrics or {}).get("f1")
    has_f1 = entry.title_f1 and isinstance(f1, (int, float)) and math.isfinite(f1)
    if scored is not None:
        what = entry.scored_part.title if entry.scored_part else ""
        line = f"{scored['predicted_scored']:,} of {count} scored"
        line += f"; F1 {f1:.2f}, {what}" if has_f1 else f", {what}"
        return f"{lay.label}\n{line.rstrip(', ')}"
    if has_f1:
        return f"{lay.label} ({count}; F1 {f1:.2f})"
    return f"{lay.label} ({count})"


def _layer_paths(lay: Layer) -> list[str]:
    """Every file a layer names (globs as written, ``/`` separators)."""
    paths = [lay.path, lay.reference, lay.removed_vs, lay.model_from]
    paths += [lay.background, lay.background_from, lay.metrics, lay.window_within]
    return [p.replace("\\", "/") for p in paths if p]


def _entry_paths(entry: Entry) -> list[str]:
    """Every file an entry names (globs as written, ``/`` separators)."""
    paths = [entry.reference, entry.overlay, entry.background, entry.background_from]
    paths.append(entry.metrics)
    if entry.overview is not None:
        paths.append(entry.overview.summary)
    for panel in entry.imagery.panels if entry.imagery is not None else []:
        paths += [panel.background, panel.background_from, panel.record]
    paths = [p.replace("\\", "/") for p in paths if p]
    for lay in entry.layers:
        paths += _layer_paths(lay)
    return paths


def _layer_root(entry: Entry, lay: Layer, root: Path | None) -> Path | None:
    """Run root of a layer's files: its own (``Layer.run_root``, relative to the repository)
    when the entry is read from its own run root, else *root* (an explicit ``--run-root``
    replaces every run root), as :func:`_panel_root` for imagery panels."""
    if lay.run_root and root is not None and root == _entry_root(entry):
        return REPO / lay.run_root
    return root


def _own_root_paths(entry: Entry, root: Path | None) -> list[tuple[str, Path | None]]:
    """``(path, run root)`` of the files of the layers with their own run root, so that the
    guards also check them as the files they name there."""
    return [
        (p, _layer_root(entry, lay, root))
        for lay in entry.layers
        if lay.run_root
        for p in _layer_paths(lay)
    ]


def _path_forms(path: str, root: Path | None) -> list[str]:
    """*path* as written and normalised, and the files it names under *root*, resolved.

    The resolved forms are what the renderer reads: the glob expanded as :func:`_one` and
    the backgrounds expand it, and ``..``, ``//`` and symlinks resolved (the file part of
    an overlay's ``path::layer`` and of ``zip://ARCHIVE!MEMBER``). All use ``/``.
    """
    forms = [path, posixpath.normpath(path)]
    if root is not None:
        name = path.split("::", 1)[0]
        if name.lower().startswith("zip://"):
            name = name[len("zip://") :].split("!", 1)[0]
        forms += [Path(m).resolve().as_posix() for m in sorted(glob.glob(str(root / name)))]
    return forms


def _names_any(path: str, parts: tuple[str, ...], root: Path | None = None) -> bool:
    """True when a form of *path* (:func:`_path_forms`) contains one of *parts*.

    Case-insensitive; with *root*, the files the path names under it are checked too.
    """
    if not parts:
        return False
    forms = [form.lower() for form in _path_forms(path, root)]
    return any(part.lower() in form for part in parts for form in forms)


def _hides_squares(entry: Entry, root: Path | None = None) -> bool:
    """True for an entry of an example whose study squares stay local (``LOCAL_SQUARES``)."""
    if any(_names_any(p, LOCAL_SQUARES, root) for p in _entry_paths(entry)):
        return True
    return any(_names_any(p, LOCAL_SQUARES, where) for p, where in _own_root_paths(entry, root))


def _location_record(loc: dict | None, hide_coordinates: bool = False) -> dict:
    """``gallery_stats.json`` record of an inset location; no lon/lat if *hide_coordinates*."""
    hidden = ("lon", "lat") if hide_coordinates else ()
    return {k: v for k, v in (loc or {}).items() if k[0] != "_" and k not in hidden}


def _check_publishable(entry: Entry, root: Path | None = None) -> None:
    """Refuse an entry that would draw a restricted reference or place a local square.

    With *root* (the run root), the files each path names under it are checked as well
    (:func:`_path_forms`), so a glob, ``..``, ``//`` or a symlink cannot hide a name.
    """
    checks = [(p, root) for p in _entry_paths(entry)] + _own_root_paths(entry, root)
    for path, where in checks:
        if _names_any(path, RESTRICTED_REFERENCES, where):
            raise ValueError(
                f"entry {entry.key}: {path} is a restricted reference (not for public "
                "distribution); draw the predictions only"
            )
    if _hides_squares(entry, root):
        for name in ("window", "mark_windows", "inset", "overview"):
            if getattr(entry, name):
                raise ValueError(
                    f"entry {entry.key}: {name} would publish where a study square that "
                    "stays local lies; use crop_m or zoom_m, and inset=False"
                )


def render(entry: Entry, root: Path, out_dir: Path, basemap: str) -> dict:
    _check_publishable(entry, root)
    if any(lay.reference for lay in entry.layers):
        raise ValueError(
            f"entry {entry.key}: Layer.reference is for multi-area entries; use Entry.reference"
        )
    for name, used in (
        ("Layer.area", any(lay.area for lay in entry.layers)),
        ("Layer.crop_m", any(lay.crop_m for lay in entry.layers)),
        ("Layer.window_within", any(lay.window_within for lay in entry.layers)),
        ("Layer.window", any(lay.window is not None for lay in entry.layers)),
        ("Layer.run_root", any(lay.run_root for lay in entry.layers)),
        ("Entry.inset_pad_deg", entry.inset_pad_deg is not None),
    ):
        if used:
            raise ValueError(f"entry {entry.key}: {name} is for multi-area entries")
    if entry.input_notes and entry.per_layer_background:
        raise ValueError(
            f"entry {entry.key}: input_notes names the inputs of panels drawn on one common "
            "background; with per_layer_background each panel is drawn on its own input"
        )
    if entry.overview is not None or entry.imagery is not None or not entry.layers:
        raise ValueError(
            f"entry {entry.key}: map panels need layers (overviews: render_overview; imagery "
            "without model output: render_imagery)"
        )
    part = entry.scored_part
    if part is not None and (
        not entry.reference
        or entry.zoom_m
        or not all(lay.metrics and not lay.metrics_record for lay in entry.layers)
    ):
        raise ValueError(
            f"entry {entry.key}: scored_part needs Entry.reference and Layer.metrics on every "
            "panel (an example's metrics file, not a Layer.metrics_record), and no zoom panel"
        )
    layers = []
    for lay in entry.layers:
        p = _one(lay.path, root)
        layers.append((lay, p, gpd.read_file(p), _sidecar(p)))
    scores = [
        _metrics_record(_one(lay.metrics, root), root, lay.metrics_record) if lay.metrics else None
        for lay in entry.layers
    ]
    parts = [None] * len(layers)  # per panel: its _scored_panel record (Entry.scored_part)
    if part is not None:
        whole = _read_reference(entry.reference, root, None, entry.reference_min_area_m2)
        parts = [
            _scored_panel(part, _one(lay.metrics, root), whole, g) for lay, _p, g, _pv in layers
        ]
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
    if entry.input_notes and not entry.per_layer_background:
        # The panels drawn on the common background but delineated on another composite.
        pos = _panel_names(len(layers), entry.ncols)
        drawn_on = {Path(t).resolve() for t in tifs}
        items = []
        for i, (_lay, _p, _g, pv) in enumerate(layers):
            raster = (pv.get("facts") or {}).get("raster_path")
            if not raster or (root / raster).resolve() in drawn_on:
                continue
            own = ([root / raster], (pv.get("config") or {}).get("source"), pv, "")
            items.append(f"{pos[i].lower()}, {_bg_note(own, role=False, exclusive=False)}")
        if items:
            paragraphs.append("Delineated on: " + "; ".join(items) + " (end dates exclusive)")
    copernicus = _copernicus_for(entry, bgs, [pv for *_, pv in layers])
    if copernicus:
        paragraphs.append(copernicus)

    if tifs:
        with rasterio.open(tifs[0]) as s:
            crs = s.crs
    else:
        crs = layers[0][2].estimate_utm_crs()
    frames = [g.to_crs(crs) for _, _, g, _ in layers]
    removed = [  # per layer: boolean mask of polygons the crop filter removed, or None
        _removed_mask(gdf, lay.removed_vs, root, crs) if lay.removed_vs else None
        for (lay, *_), gdf in zip(layers, frames, strict=True)
    ]
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
        ref = _read_reference(entry.reference, root, crs, entry.reference_min_area_m2)

    tb = np.array([f.total_bounds for f in frames if len(f)])
    minx, miny = tb[:, 0].min(), tb[:, 1].min()
    maxx, maxy = tb[:, 2].max(), tb[:, 3].max()
    padx, pady = (maxx - minx) * 0.02, (maxy - miny) * 0.02
    bounds = (minx - padx, miny - pady, maxx + padx, maxy + pady)

    cropped = False  # True when crop_m placed the map's square
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
            cropped = True
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

    positions = _panel_names(len(layers), entry.ncols)
    panels_of: dict[str, list[str]] = {}  # model line -> the panels it drew, in order
    model_provs = []  # per panel: the record that names its model
    for i, (lay, _p, _g, pv) in enumerate(layers):
        if lay.model_from:
            pv = _sidecar(_one(lay.model_from, root))
        model_provs.append(pv)
        m = _model_note(pv)
        if m:
            panels_of.setdefault(m, []).append(positions[i].lower())
    model_lines = list(panels_of)
    if entry.model_positions and len(panels_of) > 1:
        model_lines = [f"{m} ({_and_join(where)})" for m, where in panels_of.items()]
    sam_credit = _sam_auto_credit(model_provs)  # sam-auto panels only
    if sam_credit:
        model_lines.append(sam_credit)
    extra = list(entry.model_notes)
    if part is not None and part.note:
        extra.append(part.note.format(counts=_scored_counts(parts)))
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

    handles = [Line2D([], [], color=PRED_COLOR, lw=1.5, label=entry.pred_label or PRED_LABEL)]
    if part is not None:  # outlined, cased and filled as on the maps (_NotScoredHandler)
        handles.append(
            _NotScoredSwatch(
                (0, 0),
                1,
                1,
                facecolor=to_rgba(IGNORED_COLOR, IGNORED_FILL_ALPHA),
                edgecolor=IGNORED_COLOR,
                lw=1.2,
                label=part.ignored_label,
            )
        )
    if any(lay.highlight for lay, *_ in layers):
        handles.append(Line2D([], [], color=HIGHLIGHT_COLOR, lw=1.5, label=HIGHLIGHT_LABEL))
    if any(r is not None for r in removed):
        handles.append(Line2D([], [], color=REMOVED_COLOR, lw=1.5, label=REMOVED_LABEL))
    if ref is not None:
        handles.append(Line2D([], [], color=REF_COLOR, lw=1.5, label=entry.reference_label))
    left_out_label = None
    if part is not None:
        years = _or_join(sorted({p["year"] for p in parts}))
        left_out_label = part.left_out_label.format(years=years)
        handles.append(_LeftOutSwatch((0, 0), 1, 1, label=left_out_label))
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
        if legend_right > FIG_WIDTH_IN - NOTE_X_IN and legend_ncol > 1:  # still: one column
            legend_ncol = 1
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
    titles = [
        _panel_title(entry, lay, len(gdf), scores[i], parts[i])
        for i, ((lay, *_), gdf) in enumerate(zip(layers, frames, strict=True))
    ]
    more_lines = max(t.count("\n") for t in titles)  # each further title line adds its height
    title_in = TITLE_IN + more_lines * TITLE_LINE_IN
    # Room for the titles of the lower rows.
    row_gap = ROW_GAP_IN + more_lines * TITLE_LINE_IN if nrows > 1 else 0.0
    fig_h = title_in + nrows * panel_h + (nrows - 1) * row_gap + footer
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

    def draw(ax, gdf, ext, lw, title, hl=None, bg=None, rm=None, split=None, ign=None):
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
        if split is not None:  # Entry.scored_part: the scored and the left-out fields
            scored_ref, left_ref = split
            if len(left_ref):
                spacing = LEFT_OUT_HATCH_IN * (ext[1] - ext[0]) / panel_w
                hatch = _hatch_lines(left_ref.geometry, spacing)
                if not hatch.is_empty.all():
                    for colour, alpha, width, z in (
                        (LEFT_OUT_CASING, LEFT_OUT_CASING_ALPHA, LEFT_OUT_CASING_LW, 2.1),
                        (LEFT_OUT_HATCH, LEFT_OUT_HATCH_ALPHA, LEFT_OUT_HATCH_LW, 2.2),
                    ):
                        hatch.plot(ax=ax, color=colour, alpha=alpha, linewidth=width, zorder=z)
                left_ref.boundary.plot(ax=ax, color=LEFT_OUT_EDGE, linewidth=lw * 0.9, zorder=2.5)
            if len(scored_ref):
                scored_ref.boundary.plot(ax=ax, color=REF_COLOR, linewidth=lw * 0.9, zorder=3)
        elif ref is not None and len(ref):
            ref.boundary.plot(ax=ax, color=REF_COLOR, linewidth=lw * 0.9, zorder=3)
        if len(gdf):
            shown = gdf
            if ign is not None:  # Entry.scored_part: the predictions not scored, in grey
                if ign.any():
                    # Filled under the hatch and the outlines (above the image), so that a
                    # polygon whose outline a red one covers still shows.
                    gdf[ign].plot(
                        ax=ax,
                        facecolor=to_rgba(IGNORED_COLOR, IGNORED_FILL_ALPHA),
                        edgecolor="none",
                        linewidth=0,
                        zorder=2.0,
                    )
                    if IGNORED_CASING_LW > 0:  # a lighter grey casing, for dark ground
                        gdf[ign].boundary.plot(
                            ax=ax,
                            color=IGNORED_CASING,
                            alpha=IGNORED_CASING_ALPHA,
                            linewidth=lw + IGNORED_CASING_LW,
                            zorder=2.6,
                        )
                    gdf[ign].boundary.plot(ax=ax, color=IGNORED_COLOR, linewidth=lw, zorder=3.9)
                shown = gdf[~ign]
            n_before = len(ax.collections)
            if len(shown):
                shown.boundary.plot(ax=ax, color=PRED_COLOR, linewidth=lw, zorder=4)
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
        _bare_axes(ax)
        for s in ax.spines.values():
            s.set_linewidth(0.6)
        _scalebar(ax, ext, crs)
        ax.set_title(title, fontsize=_fit_title(title, panel_w), pad=4)

    for i, (ax, (lay, _p, _g, _pv), gdf) in enumerate(zip(axes, layers, frames, strict=False)):
        split = ign = None
        if parts[i] is not None and ref is not None:  # this panel's year and its split
            hit = (ref[parts[i]["column"]] == part.value).to_numpy(dtype=bool)
            split, ign = (ref[hit], ref[~hit]), parts[i]["ignored_mask"]
        draw(
            ax,
            gdf,
            extent,
            entry.line_width * (1.0 if n_panels == 1 else 0.8),
            titles[i],
            hl=lay.highlight,
            bg=imgs[i],
            rm=removed[i],
            split=split,
            ign=ign,
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
        handler_map=LEGEND_HANDLERS,
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
        top=1 - title_in / fig_h,
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

    hide = _hides_squares(entry, root)  # LOCAL_SQUARES: no coordinates in the stats
    stats = {
        "example": entry.key,
        # POSIX form, so gallery_stats.json reads the same whichever OS rendered it.
        "image": (png.relative_to(REPO) if png.is_relative_to(REPO) else png).as_posix(),
        "preview": (
            preview.relative_to(REPO) if preview.is_relative_to(REPO) else preview
        ).as_posix(),
        "title": entry.title,
        "background": "composite" if basemap == "composite" else "esri",
        "background_note": _plain(bg_note),
        "model_notes": [_plain(m) for m in model_lines],
        # The credit lines as printed, for the captions (the composites' tags may differ).
        **({"copernicus_credit": copernicus} if entry.copernicus_credit else {}),
        "proj_network": proj_network_enabled(),
        "location": _location_record(loc, hide),
        "layers": [],
    }
    for k, (lay, p, g, pv) in enumerate(layers):
        cfg = pv.get("config") or pv.get("base_config") or {}
        em = pv.get("engine_meta") or {}
        credit = _bg_credit(bgs[k]) if basemap == "composite" else ""
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
                **({"background_credit": credit} if credit else {}),
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
                **({"metrics": scores[k]} if scores[k] is not None else {}),
                # Entry.scored_part: the panel's year, its split and its counts (as drawn).
                **(
                    {"scored_part": {n: v for n, v in parts[k].items() if "_mask" not in n}}
                    if parts[k] is not None
                    else {}
                ),
            }
        )
    if ref is not None:
        stats["reference"] = _reference_stats(entry, ref)
    if part is not None:
        stats["scored_part"] = {
            "column": part.column,
            "value": part.value,
            "title": part.title,
            "ignored_label": part.ignored_label,
            "left_out_label": left_out_label,
        }
    if entry.metrics:
        try:
            stats["metrics"] = json.loads(_one(entry.metrics, root).read_text())
        except FileNotFoundError as exc:
            stats["metrics_error"] = str(exc)
    if zoom:
        stats["zoom_window_m"] = entry.zoom_m
        if not hide:
            stats["zoom_window"] = _square_record(zoom, crs, entry.zoom_m)
    if cropped and not hide:
        # As drawn: the crop_m square, cut to the imagery where it reaches past it.
        stats["crop_window"] = _square_record(extent, crs, entry.crop_m)
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


def _world_view(points: list[tuple[str, float, float]], pad_deg=None):
    """View box (EPSG:4326) around the dots of a multi-area entry (margin *pad_deg*)."""
    from shapely.geometry import box

    lons = [p[1] for p in points]
    lats = [p[2] for p in points]
    px, py = pad_deg or WORLD_INSET_PAD_DEG
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
    _bare_axes(iax)
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
        roles = {role for _src, role, _c in facts}
        if len(roles) == 1:  # one role, said once (e.g. each panel on its engine's input)
            items = [f"{p['n']}: {_bg_note(p['bg'], role=False, exclusive=False)}" for p in panels]
            head = f"Background ({roles.pop()}; end dates exclusive): "
            return head + "; ".join(dict.fromkeys(items))
        items = [f"{p['n']}: {_bg_note(p['bg'])}" for p in panels]
        return "Background: " + "; ".join(dict.fromkeys(items))
    source, role, res, method = heads.pop()
    label = SOURCE_LABELS.get(source or "", source or "imagery")
    if res:
        label += f" {float(res):g} m"
    credits = [c for c in dict.fromkeys(_credit(source, c) for _s, _r, c in facts) if c]
    if credits:  # e.g. every panel on an Umbra or Capella composite
        label += f" ({'; '.join(credits)})"
    per = []
    for p, (_src, _role, c) in zip(panels, facts, strict=True):
        dates = (
            f"{c['date_start']} to {c['date_end_exclusive']}"
            if c.get("date_start") and c.get("date_end_exclusive")
            else "dates not recorded"
        )
        n_img = f", {_count(c['n_images'], _image_noun(source))}" if c.get("n_images") else ""
        per.append(f"{p['n']}: {dates}{n_img}")
    kind = f"{method} composites" if method else "composites"
    # Panels of one area (Layer.area) with the same imagery give one item.
    per = list(dict.fromkeys(per))
    return f"Background: {label} {kind} ({role}); end dates exclusive: " + "; ".join(per)


def _area_points(panels: list[dict]) -> list[tuple[str, float, float]]:
    """One ``(number, lon, lat)`` locator dot per area, at the area's first panel's window."""
    first: dict[str, tuple[str, float, float]] = {}
    for p in panels:
        first.setdefault(p["n"], (p["n"], p["lon"], p["lat"]))
    return list(first.values())


def _window_record(panel: dict) -> dict:
    """``gallery_stats.json`` record of a panel's window: CRS, bounds and centre lon/lat."""
    x0, x1, y0, y1 = panel["extent"]
    return {
        "crs": str(panel["crs"]),
        "bounds": [float(v) for v in (x0, y0, x1, y1)],
        "centre_lonlat": [round(panel["lon"], 4), round(panel["lat"], 4)],
    }


def _given_window(window: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """``(x0, x1, y0, y1)`` of a window given as ``(centre x, centre y, side)``
    (``Layer.window``)."""
    cx, cy, side = (float(v) for v in window)
    return (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)


def _square_record(extent, crs, side_m: float) -> dict:
    """``gallery_stats.json`` record of a square drawn by :func:`render` (crop or zoom).

    *extent* is ``(x0, x1, y0, y1)`` in *crs*; the record is :func:`_window_record`'s
    (CRS, bounds, centre lon/lat) plus the requested side ``side_m``.
    """
    from pyproj import Transformer

    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
        (extent[0] + extent[1]) / 2, (extent[2] + extent[3]) / 2
    )
    record = _window_record({"extent": extent, "crs": crs, "lon": lon, "lat": lat})
    return {**record, "side_m": side_m}


def render_areas(entry: Entry, root: Path, out_dir: Path, basemap: str) -> dict:
    """Render a multi-area entry: one panel per layer, each on its own study area.

    Each panel is drawn in the CRS of its own background raster, on the square of side
    ``lay.crop_m`` (else ``entry.crop_m``) holding the most polygon representative points
    of that layer (with ``entry.crop_on_reference``, of the panel's ``lay.reference``
    polygons when it has any), among candidate squares half a side apart inside its
    imagery, with its own scale bar. A panel's ``lay.reference`` is drawn in cyan and the
    polygons missing from ``lay.removed_vs`` in magenta. A world locator with numbered dots
    replaces the country inset. Panels with the same ``lay.area`` (e.g. several models on
    one study area) share a number, a dot and the window of the area's first panel, which
    their imagery must cover in the same CRS. Options that only :func:`render` implements
    are refused.
    """
    from pyproj import Transformer

    _check_publishable(entry, root)
    if basemap != "composite":
        raise ValueError(f"entry {entry.key}: multi-area entries need --basemap composite")
    if not entry.crop_m:
        raise ValueError(f"entry {entry.key}: multi-area entries need crop_m")
    unsupported = [
        n
        for n in (
            "reference",
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
            "input_notes",
            "overview",
            "scored_part",
            "imagery",
        )
        if getattr(entry, n)
    ] + [
        f"layers[{k}].{n}"
        for k, lay in enumerate(entry.layers)
        for n in ("highlight", "model_from")
        if getattr(lay, n)
    ]
    if unsupported:
        raise ValueError(
            f"entry {entry.key}: multi-area entries do not support {', '.join(unsupported)}"
        )
    # Panel numbers: one per Layer.area, and one per panel without an area.
    keys = [("area", lay.area) if lay.area else ("panel", i) for i, lay in enumerate(entry.layers)]
    numbers: dict[tuple, str] = {}
    for key in keys:
        numbers.setdefault(key, str(len(numbers) + 1))
    # Layer.area -> (CRS, window, how it was chosen) of its first panel
    area_windows: dict[str, tuple] = {}
    loaded = []  # per panel: path, provenance, background, CRS, imagery bounds, pixel size
    roots = [_layer_root(entry, lay, root) for lay in entry.layers]  # Layer.run_root, else root
    for lay, lroot in zip(entry.layers, roots, strict=True):
        path = _one(lay.path, lroot)
        prov = _sidecar(path)
        bg = _background_for(entry, lroot, prov, basemap, lay)
        with rasterio.open(bg[0][0]) as src:
            res = max(abs(src.res[0]), abs(src.res[1]))
            loaded.append((path, prov, bg, src.crs, src.bounds, res))
    # The window of an area is placed inside the imagery that all its panels cover (e.g. a
    # 10 m Sentinel-1 grid reaches a few metres past a 1 m grid of the same study area).
    area_imagery: dict[str, tuple] = {}  # Layer.area -> (CRS, (x0, x1, y0, y1))
    for lay, (_path, _prov, _bg, crs, rb, _res) in zip(entry.layers, loaded, strict=True):
        if not lay.area:
            continue
        crs0, b = area_imagery.get(lay.area, (crs, (rb.left, rb.right, rb.bottom, rb.top)))
        if crs0 == crs:  # a panel in another CRS is refused below
            b = (max(b[0], rb.left), min(b[1], rb.right), max(b[2], rb.bottom), min(b[3], rb.top))
        area_imagery[lay.area] = (crs0, b)
    panels = []
    area_refs: dict[str, gpd.GeoDataFrame] = {}  # Layer.area -> the reference of its first panel
    for i, lay in enumerate(entry.layers):
        path, prov, bg, crs, rb, res = loaded[i]
        lroot = roots[i]
        g = gpd.read_file(path).to_crs(crs)
        ref = None
        if lay.reference:
            ref = _read_reference(lay.reference, lroot, crs, entry.reference_min_area_m2)
        elif lay.area in area_refs:
            # A panel of an area that names no reference draws the area's (e.g. an output read
            # under another run root than the reference; the CRS is checked below).
            ref = area_refs[lay.area]
        if lay.area and ref is not None:
            area_refs.setdefault(lay.area, ref)
        removed = _removed_mask(g, lay.removed_vs, lroot, crs) if lay.removed_vs else None
        metrics = (
            _metrics_record(_one(lay.metrics, lroot), lroot, lay.metrics_record)
            if lay.metrics
            else None
        )
        # The square is placed inside the imagery (which covers the study area), so it fits
        # even where the polygons span less than crop_m in one direction. With
        # crop_on_reference it is the square with the most reference polygons.
        full = (rb.left, rb.right, rb.bottom, rb.top)
        if lay.area in area_windows:
            # Every panel of an area shows the window of the area's first panel.
            crs0, ext, choice = area_windows[lay.area]
            inside = (
                ext[0] >= full[0] - res
                and ext[1] <= full[1] + res
                and ext[2] >= full[2] - res
                and ext[3] <= full[3] + res
            )
            if crs0 != crs or not inside:
                raise ValueError(
                    f"entry {entry.key}: panel {i + 1} ({lay.label}) cannot show the window of "
                    f"area {lay.area!r}: its imagery is in another CRS or does not cover it"
                )
            if lay.window is not None and _given_window(lay.window) != ext:
                raise ValueError(
                    f"entry {entry.key}: panel {i + 1} ({lay.label}) gives another window than "
                    f"the first panel of area {lay.area!r}"
                )
        elif lay.window is not None:
            # A window given for the area (e.g. its evaluation square), not chosen from
            # candidates; it must lie inside the imagery of every panel of the area.
            if lay.area:
                full = area_imagery[lay.area][1]
            ext = _given_window(lay.window)
            if not (
                ext[0] >= full[0] and ext[1] <= full[1] and ext[2] >= full[2] and ext[3] <= full[3]
            ):
                raise ValueError(
                    f"entry {entry.key}: the window {lay.window} of panel {i + 1} "
                    f"({lay.label}) is not inside the imagery"
                )
            choice = {"side_m": float(lay.window[2]), "given": True}
            if lay.area:
                area_windows[lay.area] = (crs, ext, choice)
        else:
            if lay.area:
                full = area_imagery[lay.area][1]
            side = lay.crop_m or entry.crop_m
            dense = ref if (entry.crop_on_reference and ref is not None and len(ref)) else g
            within = None
            if lay.window_within:
                area = gpd.read_file(_one(lay.window_within, lroot)).to_crs(crs)
                within = area.geometry.union_all().buffer(1e-6)
            # How the window was chosen, for the stats: the side, the number of candidate
            # squares (half a side apart) and what was counted in them.
            choice = {
                "side_m": float(side),
                "candidates": 0,
                "most_of": "reference polygons" if dense is ref else "panel polygons",
                **({"inside": lay.window_within} if lay.window_within else {}),
            }
            ext = _densest_window(dense, side, full, within, choice) if len(dense) else None
            ext = ext or full
            if lay.area:
                area_windows[lay.area] = (crs, ext, choice)
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
            (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2
        )
        panels.append(
            {
                "n": numbers[keys[i]],
                "lay": lay,
                "root": lroot,
                "path": path,
                "gdf": g,
                "ref": ref,
                "removed": removed,
                "metrics": metrics,
                "prov": prov,
                "bg": bg,
                "crs": crs,
                "extent": ext,
                "choice": choice,
                "lon": lon,
                "lat": lat,
            }
        )

    paragraphs = [_areas_bg_note(panels)]
    copernicus = _copernicus_for(entry, [p["bg"] for p in panels], [p["prov"] for p in panels])
    if copernicus:
        paragraphs.append(copernicus)
    areas_of: dict[str, list[int]] = {}  # model line -> the numbers of the areas it drew
    for p in panels:
        m = _model_note(p["prov"])
        if m and int(p["n"]) not in areas_of.setdefault(m, []):
            areas_of[m].append(int(p["n"]))
    model_lines = list(areas_of)
    if entry.model_positions and len(areas_of) > 1:
        # E.g. the fine-tuned checkpoints of each study area, which read alike but for the
        # hash: "(study area 2)"; a model of every area: "(study areas 1-4)".
        model_lines = [
            f"{m} (study area{'s' if len(ns) > 1 else ''} {_year_list(ns)})"
            for m, ns in areas_of.items()
        ]
    sam_credit = _sam_auto_credit([p["prov"] for p in panels])  # sam-auto panels only
    if sam_credit:
        model_lines.append(sam_credit)
    for m in entry.model_notes:
        if m not in model_lines:
            model_lines.append(m)
    paragraphs += [("Model: " if k == 0 else "") + m for k, m in enumerate(model_lines)]
    points = _area_points(panels)
    view = _world_view(points, entry.inset_pad_deg) if entry.inset else None
    if view is not None:
        dots = f"Inset: study areas 1-{len(points)}; "
        if _india_outline().intersects(view):
            paragraphs.append(dots + INSET_NOTE_INDIA.removeprefix("Inset: "))
        else:
            paragraphs.append(dots + "Natural Earth country and lake boundaries")

    handles = [Line2D([], [], color=PRED_COLOR, lw=1.5, label=entry.pred_label or PRED_LABEL)]
    if any(p["removed"] is not None for p in panels):
        handles.append(Line2D([], [], color=REMOVED_COLOR, lw=1.5, label=REMOVED_LABEL))
    if any(p["ref"] is not None for p in panels):
        handles.append(Line2D([], [], color=REF_COLOR, lw=1.5, label=entry.reference_label))
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
    titles = [
        f"{p['n']}. {_panel_title(entry, p['lay'], len(p['gdf']), p['metrics'])}" for p in panels
    ]
    # A label of several lines (the SAM entry) adds TITLE_LINE_IN per further line above the
    # maps and between their rows, as in render.
    more_lines = max(t.count("\n") for t in titles)
    title_in = TITLE_IN + more_lines * TITLE_LINE_IN
    row_gap = ROW_GAP_IN + more_lines * TITLE_LINE_IN if nrows > 1 else 0.0
    fig_h = title_in + nrows * panel_h + (nrows - 1) * row_gap + footer
    px = (panel_w * DPI, panel_h * DPI)

    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH_IN, fig_h), dpi=DPI, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    lw = entry.line_width * 0.8
    # One title size for the figure: the largest at which every title fits its panel.
    title_fs = min(_fit_title(t, panel_w) for t in titles)
    for ax, p, title in zip(axes, panels, titles, strict=False):
        e = p["extent"]
        tifs, source, *_ = p["bg"]
        img, img_ext, _stretch = _read_background(
            tifs, source, (e[0], e[2], e[1], e[3]), p["crs"], px
        )
        ax.imshow(img, extent=img_ext, interpolation="auto", zorder=1)
        if p["ref"] is not None and len(p["ref"]):
            p["ref"].cx[e[0] : e[1], e[2] : e[3]].boundary.plot(
                ax=ax, color=REF_COLOR, linewidth=lw * 0.9, zorder=3
            )
        in_window = p["gdf"].index.isin(p["gdf"].cx[e[0] : e[1], e[2] : e[3]].index)
        g = p["gdf"][in_window]
        if len(g):
            n_before = len(ax.collections)
            g.boundary.plot(ax=ax, color=PRED_COLOR, linewidth=lw, zorder=4)
            if entry.halo:
                halo = [pe.Stroke(linewidth=lw + 1.6, foreground="white", alpha=0.85), pe.Normal()]
                for coll in ax.collections[n_before:]:
                    coll.set_path_effects(halo)
            rm = p["removed"][in_window] if p["removed"] is not None else None
            if rm is not None and rm.any():
                g[rm].boundary.plot(ax=ax, color=REMOVED_COLOR, linewidth=lw * 1.3, zorder=4.6)
        ax.set_xlim(e[0], e[1])
        ax.set_ylim(e[2], e[3])
        ax.set_aspect("equal")
        ax.set_anchor("N")
        _bare_axes(ax)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        _scalebar(ax, e, p["crs"])
        ax.set_title(title, fontsize=title_fs, pad=4)

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
        top=1 - title_in / fig_h,
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

    hide = _hides_squares(entry, root)  # LOCAL_SQUARES: no coordinates in the stats
    stats = {
        "example": entry.key,
        "image": (png.relative_to(REPO) if png.is_relative_to(REPO) else png).as_posix(),
        "preview": (
            preview.relative_to(REPO) if preview.is_relative_to(REPO) else preview
        ).as_posix(),
        "title": entry.title,
        "background": "composite",
        "background_note": _plain("\n".join(note_lines)),
        "model_notes": [_plain(m) for m in model_lines],
        # The credit lines as printed, for the captions (the composites' tags may differ).
        **({"copernicus_credit": copernicus} if entry.copernicus_credit else {}),
        "proj_network": proj_network_enabled(),
        "location": (
            {
                "areas": [
                    {"panel": n, **_location_record(_locate(lon, lat), hide)}
                    for n, lon, lat in points
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
        credit = _bg_credit(p["bg"])
        stats["layers"].append(
            {
                "panel": p["n"],
                "label": p["lay"].label,
                "file": str(p["path"].relative_to(p["root"])),
                # A layer read under its own run root (Layer.run_root) names it.
                **(
                    {"run_root": p["root"].relative_to(REPO).as_posix()}
                    if p["root"] != root and p["root"].is_relative_to(REPO)
                    else {}
                ),
                **_area_stats(p["gdf"]),
                "source": cfg.get("source"),
                "engine": cfg.get("engine"),
                "year": cfg.get("year"),
                "lulc_filter": cfg.get("lulc_filter"),
                "lulc_dataset": cfg.get("lulc_dataset"),
                "sam_refine": cfg.get("sam_refine"),
                "min_field_area_m2": cfg.get("min_field_area_m2"),
                "composite": _composite_facts(p["prov"], tifs[0] if tifs else None),
                **({"background_credit": credit} if credit else {}),
                "engine_meta": {
                    k: em[k]
                    for k in sorted(em)
                    if isinstance(em[k], (str, int, float, bool)) and not _is_local_path(em[k])
                },
                **({} if hide else {"window": {**_window_record(p), **p["choice"]}}),
                "agribound_version": p["prov"].get("agribound_version"),
                "run_status": p["prov"].get("status"),
                "n_removed_by_crop_filter": (
                    int(p["removed"].sum()) if p["removed"] is not None else None
                ),
                # The whole reference of the study area; n_polygons_in_window counts the
                # reference polygons that intersect the panel's square.
                "reference": (
                    {
                        **_reference_stats(entry, p["ref"]),
                        "n_polygons_in_window": len(
                            p["ref"].cx[
                                p["extent"][0] : p["extent"][1], p["extent"][2] : p["extent"][3]
                            ]
                        ),
                    }
                    if p["ref"] is not None
                    else None
                ),
                **({"metrics": p["metrics"]} if p["metrics"] is not None else {}),
            }
        )
    return stats


# --------------------------------------------------------------------------- overview map

# Overview maps (render_overview): the conterminous US in Albers equal-area, one marker per
# study square, coloured by a score on the sequential blue ramp of the reference palette
# (steps 100-700, light = low, on a fixed 0-1 scale so that a score's colour never depends on
# the other scores; the markers' dark outline keeps the lightest visible) and shaped by the
# kind of reference of the square (OVERVIEW_KINDS, from the source of most of its fields:
# the summary's "sources", e.g. "CLU 80 %, NEDNR 20 %").
OVERVIEW_CRS = "EPSG:5070"
OVERVIEW_NOT_CONUS = ("AK", "HI")
OVERVIEW_RAMP = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
OVERVIEW_KINDS = {  # kind: (marker, legend label)
    "CLU": ("s", "USDA FSA Common Land Units (CLU)"),
    "agency": ("o", "State or federal agency inventory"),
    "DRI": ("^", "DRI field boundaries (NM: DRI/NMOSE V18)"),
}
OVERVIEW_DRI_CODES = ("DRI", "V18")
OVERVIEW_MARKER_PT = 7.0
OVERVIEW_GAP_PT = 1.5  # markers closer than their size plus this are moved apart
OVERVIEW_MARK_EDGE_PT = 1.4  # the ring of the numbered squares
OVERVIEW_NO_SQUARE = "#e2e2de"  # states without a square
OVERVIEW_CBAR_W_IN, OVERVIEW_CBAR_H_IN = 2.6, 0.12


def _reference_kind(sources) -> str:
    """Kind of reference of a square ("CLU", "DRI" or "agency") from its summary ``sources``.

    The first source listed holds most of the square's fields (``"CLU 80 %, NEDNR 20 %"``
    is CLU); V18 (New Mexico) and DRI (Nevada, Oregon) are DRI boundaries.
    """
    code = str(sources or "").split(",")[0].strip().split(" ")[0]
    if code == "CLU":
        return "CLU"
    return "DRI" if code in OVERVIEW_DRI_CODES else "agency"


def _marks_label(n: int) -> str:
    """Legend label of the numbered markers of an overview, whatever the companion image's
    number of panels (e.g. two per square with the crop-history zones)."""
    squares = "Square 1" if n == 1 else f"Squares 1-{n}"
    return f"{squares}, drawn in the companion image"


def _overview_title(ov: Overview, n: int, median: float) -> str | None:
    """The map's own title, as a panel title: ``Overview.panel_label`` with the number of
    squares and the median score (None without a panel label: the entry's title is used)."""
    if not ov.panel_label:
        return None
    metric = "F1" if ov.metric == "f1" else ov.metric
    return f"{ov.panel_label} ({_count(n, 'square')}; median {metric} {median:.2f})"


def _spread_markers(xy, min_sep: float, iterations: int = 200) -> np.ndarray:
    """Marker positions moved apart until no two lie within *min_sep* in both x and y.

    The markers are about square, so two are apart when they are *min_sep* apart on one
    axis. A pair that is too close is pushed apart on the axis on which they are further
    apart (x on a tie), each by half the shortfall, pair after pair in a fixed order
    (deterministic); markers that are apart are not moved.
    """
    pos = np.asarray(xy, dtype=float).copy()
    for _ in range(iterations):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = pos[j] - pos[i]
                if max(abs(d[0]), abs(d[1])) >= min_sep:
                    continue
                axis = 0 if abs(d[0]) >= abs(d[1]) else 1
                sign = 1.0 if d[axis] >= 0 else -1.0
                step = (min_sep - abs(d[axis])) / 2 + min_sep * 1e-6
                pos[i, axis] -= sign * step
                pos[j, axis] += sign * step
                moved = True
        if not moved:
            break
    return pos


def render_overview(entry: Entry, root: Path, out_dir: Path, basemap: str) -> dict:
    """Render an overview entry: the conterminous US with one marker per study square.

    The markers sit at the squares' centres (the summary's ``centre_lon``, ``centre_lat``),
    coloured by ``Overview.metric`` of the ``Overview.run`` rows with status "ok" and shaped
    by the kind of reference; markers that would overlap are moved apart (a line ties a
    marker to its square when the square is no longer under it). ``Overview.marks`` are
    ringed and numbered 1..n. States without a square are grey. Boundaries: Natural Earth
    (the inset layers). *basemap* is not used (no imagery).
    """
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Patch
    from pyproj import Transformer
    from shapely.geometry import box

    ov = entry.overview
    if ov is None or entry.layers or entry.imagery is not None:
        raise ValueError(f"entry {entry.key}: an overview entry has Entry.overview and no layers")
    if entry.scored_part is not None:
        raise ValueError(f"entry {entry.key}: scored_part is for map panels (render)")
    if entry.multi_area or entry.inset:
        raise ValueError(f"entry {entry.key}: an overview is its own locator (inset=False)")
    _check_publishable(entry, root)
    summary_path = _one(ov.summary, root)
    table = pd.read_csv(summary_path)
    keep = (table[ov.run_column] == ov.run) & (table["status"] == "ok")
    rows = table[keep].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"entry {entry.key}: no {ov.run!r} rows with status ok in {summary_path}")
    missing = [s for s in ov.marks if s not in set(rows["state"])]
    if missing:
        raise ValueError(f"entry {entry.key}: marked states without a row: {missing}")
    values = rows[ov.metric].astype(float).to_numpy()
    kinds = np.array([_reference_kind(s) for s in rows["sources"]])
    metric = "F1" if ov.metric == "f1" else ov.metric

    states = _natural_earth("states")
    conus = (states["adm0_a3"] == "USA") & ~states["postal"].isin(OVERVIEW_NOT_CONUS)
    us = states[conus].to_crs(OVERVIEW_CRS)
    bx0, by0, bx1, by1 = us.total_bounds
    pad = 0.015 * (bx1 - bx0)
    view = box(bx0 - pad, by0 - pad, bx1 + pad, by1 + pad)
    vx0, vy0, vx1, vy1 = view.bounds
    # States of the conterminous US without a square (DC is not a state).
    no_square = us[~us["postal"].isin(set(rows["state"])) & (us["postal"] != "DC")]

    # Marker positions: at the squares' centres, moved apart where two would overlap.
    map_w = FIG_WIDTH_IN * (MAPS_RIGHT - MAPS_LEFT)
    map_h = map_w * (vy1 - vy0) / (vx1 - vx0)
    to_map = Transformer.from_crs("EPSG:4326", OVERVIEW_CRS, always_xy=True)
    x, y = to_map.transform(
        rows["centre_lon"].to_numpy(dtype=float), rows["centre_lat"].to_numpy(dtype=float)
    )
    xy = np.column_stack([x, y])
    m_per_pt = (vx1 - vx0) / (map_w * 72)
    spread = _spread_markers(xy, (OVERVIEW_MARKER_PT + OVERVIEW_GAP_PT) * m_per_pt)
    shift = np.hypot(*(spread - xy).T)
    moved = shift > 1.0  # metres
    # A line ties a moved marker to its square when the square is no longer under it.
    tied = np.abs(spread - xy).max(axis=1) > OVERVIEW_MARKER_PT / 2 * m_per_pt

    # Footer notes: the numbers, then the entry's definitions and credits, model, map.
    lo, hi = int(np.argmin(values)), int(np.argmax(values))
    years = rows["year"].astype(int)
    span = f"{years.min()}" if years.min() == years.max() else f"{years.min()}-{years.max()}"
    none = ", ".join(sorted(no_square["name"])) or "none"
    spread_note = ""
    if moved.any():
        spread_note = (
            f"; {_count(int(moved.sum()), 'marker')} moved apart so that none overlap, by at "
            f"most {shift.max() / 1000:.0f} km"
            + (
                "; a line ties a marker to its square when the square is outside it"
                if tied.any()
                else ""
            )
        )
    paragraphs = [
        f"{len(rows)} squares, one per state (no square: {none}); {metric} median "
        f"{np.median(values):.2f}, from {values[lo]:.2f} ({rows['state'][lo]}) to "
        f"{values[hi]:.2f} ({rows['state'][hi]}); imagery of {span}{spread_note}",
        *ov.notes,
    ]
    model_lines = []
    for out in rows["output"]:
        m = _model_note(_sidecar(root / str(out)))
        if m and m not in model_lines:
            model_lines.append(m)
    paragraphs += [("Model: " if k == 0 else "") + m for k, m in enumerate(model_lines)]
    paragraphs.append(
        "Map: Natural Earth state, country and lake boundaries; Albers equal-area projection "
        f"of the conterminous US ({OVERVIEW_CRS})"
    )

    handles = [
        Line2D(
            [],
            [],
            marker=OVERVIEW_KINDS[k][0],
            ls="",
            ms=OVERVIEW_MARKER_PT,
            mfc="#a7a7a3",
            mec="#333333",
            mew=0.5,
            label=OVERVIEW_KINDS[k][1],
        )
        for k in OVERVIEW_KINDS
        if k in set(kinds)
    ]
    if ov.marks:
        handles.append(
            Line2D(
                [],
                [],
                marker="s",
                ls="",
                ms=OVERVIEW_MARKER_PT,
                mfc="#a7a7a3",
                mec="black",
                mew=OVERVIEW_MARK_EDGE_PT,
                label=_marks_label(len(ov.marks)),
            )
        )
    if len(no_square):
        handles.append(
            Patch(facecolor=OVERVIEW_NO_SQUARE, edgecolor="#a9a9a9", lw=0.4, label="No square")
        )
    # Footer layout: the legend at the top left, the colour bar (its label above it) at the
    # top right, the notes under both across the image.
    cbar_left = FIG_WIDTH_IN - INSET_RIGHT_IN - OVERVIEW_CBAR_W_IN
    for legend_ncol in range(len(handles), 0, -1):  # the fewest rows that end left of the bar
        legend_right, legend_h = _measure_legend(handles, legend_ncol)
        if legend_right + NOTE_INSET_GAP_IN <= cbar_left:
            break
    label_h = _measure_text_height(ov.metric_label, LEGEND_FS)
    tick_h = _measure_text_height("0.5", LEGEND_FS - 1)
    cbar_block = label_h + 3 / 72 + OVERVIEW_CBAR_H_IN + 5.5 / 72 + tick_h
    note_lines = [line for par in paragraphs for line in _wrap(par, FIG_WIDTH_IN - 2 * NOTE_X_IN)]
    notes_top = FOOT_TOP_IN + max(legend_h, cbar_block) + LEGEND_NOTE_GAP_IN
    footer = notes_top + _notes_height_in(len(note_lines)) + FOOT_BOTTOM_IN
    fig_h = TITLE_IN + map_h + footer

    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h), dpi=DPI)
    ax = fig.add_axes([MAPS_LEFT, footer / fig_h, map_w / FIG_WIDTH_IN, map_h / fig_h])
    ax.set_facecolor(WATER_COLOR)  # sea
    countries = _natural_earth("countries")
    others = countries[countries["ADM0_A3"].isin(("CAN", "MEX"))].to_crs(OVERVIEW_CRS)
    others = others.clip(view)
    if len(others):
        others.plot(ax=ax, color="#eeeeee", edgecolor="#a9a9a9", linewidth=0.3, zorder=Z_NEIGHBOURS)
    us.plot(ax=ax, color="#fbfbfb", edgecolor="#b4b4b4", linewidth=0.4, zorder=Z_COUNTRY)
    if len(no_square):
        no_square.plot(
            ax=ax, color=OVERVIEW_NO_SQUARE, edgecolor="#b4b4b4", linewidth=0.4, zorder=Z_ADMIN1
        )
    lakes = _natural_earth("lakes").to_crs(OVERVIEW_CRS).clip(view)
    lakes = lakes[lakes.geometry.area >= view.area * 2e-4]  # both in m2 here
    if len(lakes):
        lakes.plot(ax=ax, color=WATER_COLOR, edgecolor="#8fa9bf", linewidth=0.3, zorder=Z_LAKES)

    for i in np.flatnonzero(tied):  # a line from the square to its moved marker
        ax.plot(
            [xy[i, 0], spread[i, 0]], [xy[i, 1], spread[i, 1]], color="#555555", lw=0.5, zorder=4
        )
        ax.plot(xy[i, 0], xy[i, 1], marker="o", ms=1.8, mfc="#333333", mec="none", zorder=4.1)
    cmap = LinearSegmentedColormap.from_list("overview", OVERVIEW_RAMP)
    norm = Normalize(0.0, 1.0)
    index = {state: i for i, state in enumerate(rows["state"])}
    marked = np.zeros(len(rows), dtype=bool)
    marked[[index[s] for s in ov.marks]] = True
    for kind, (marker, _label) in OVERVIEW_KINDS.items():
        for ring in (False, True):  # the numbered squares carry a black ring
            sel = (kinds == kind) & (marked == ring)
            if sel.any():
                ax.scatter(
                    spread[sel, 0],
                    spread[sel, 1],
                    s=OVERVIEW_MARKER_PT**2,
                    marker=marker,
                    c=cmap(norm(values[sel])),
                    edgecolors="black" if ring else "#333333",
                    linewidths=OVERVIEW_MARK_EDGE_PT if ring else 0.5,
                    zorder=Z_MARKER + (0.1 if ring else 0),
                )
    for k, state in enumerate(ov.marks, 1):
        # The number sits on the side away from the nearest other marker.
        i = index[state]
        rest = np.delete(np.arange(len(rows)), i)
        if len(rest):
            j = rest[int(np.argmin(np.hypot(*(spread[rest] - spread[i]).T)))]
            d = spread[i] - spread[j]
        else:
            d = np.array([1.0, 1.0])  # a single square: up and to the right
        u = d / max(math.hypot(d[0], d[1]), 1e-9)
        off = OVERVIEW_MARKER_PT / 2 + 2.5
        ax.annotate(
            str(k),
            (spread[i, 0], spread[i, 1]),
            xytext=(u[0] * off, u[1] * off),
            textcoords="offset points",
            ha="left" if u[0] >= 0 else "right",
            va="bottom" if u[1] >= 0 else "top",
            fontsize=TITLE_FS - 1,
            fontweight="bold",
            zorder=Z_MARKER + 1,
            path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()],
        )
    ax.set_xlim(vx0, vx1)
    ax.set_ylim(vy0, vy1)
    ax.set_aspect("equal")
    _bare_axes(ax)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    map_title = _overview_title(ov, len(rows), float(np.median(values))) or entry.title
    ax.set_title(map_title, fontsize=_fit_title(map_title, map_w), pad=4)

    cbar_bottom = footer - FOOT_TOP_IN - label_h - 3 / 72 - OVERVIEW_CBAR_H_IN
    cax = fig.add_axes(
        [
            cbar_left / FIG_WIDTH_IN,
            cbar_bottom / fig_h,
            OVERVIEW_CBAR_W_IN / FIG_WIDTH_IN,
            OVERVIEW_CBAR_H_IN / fig_h,
        ]
    )
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal"
    )
    cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.tick_params(labelsize=LEGEND_FS - 1, length=2, width=0.5, colors=NOTE_COLOR)
    cb.outline.set_linewidth(0.5)
    cax.set_title(ov.metric_label, fontsize=LEGEND_FS, pad=3)
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
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{entry.name}.png"
    fig.savefig(png, dpi=DPI, metadata={"Software": None})
    try:
        _check_footer(fig, footer, legend, note_texts, cax)
    except RuntimeError:
        png.unlink()
        raise
    finally:
        plt.close(fig)
    preview = _write_preview(png, out_dir)

    squares = []
    for i, r in rows.iterrows():
        record = {
            "state": r["state"],
            "kind": str(kinds[i]),
            "sources": r["sources"],
            "reference": r["reference"],
            "year": int(r["year"]),
            ov.metric: float(r[ov.metric]),
            "centre_lonlat": [round(float(r["centre_lon"]), 4), round(float(r["centre_lat"]), 4)],
            "moved_on_map_km": round(float(shift[i]) / 1000, 1),
        }
        for k in ("precision", "recall", "n_reference", "n_predicted", "output"):
            if k in r and pd.notna(r[k]):
                record[k] = r[k].item() if hasattr(r[k], "item") else r[k]
        squares.append(record)
    return {
        "example": entry.key,
        "image": (png.relative_to(REPO) if png.is_relative_to(REPO) else png).as_posix(),
        "preview": (
            preview.relative_to(REPO) if preview.is_relative_to(REPO) else preview
        ).as_posix(),
        "title": entry.title,
        "map_title": map_title,
        "background": "Natural Earth",
        "background_note": "\n".join(note_lines),
        "model_notes": model_lines,
        "proj_network": proj_network_enabled(),
        "summary": summary_path.relative_to(root).as_posix(),
        "run": ov.run,
        "metric": ov.metric,
        "n_squares": len(rows),
        "median": float(np.median(values)),
        "lowest": {"state": rows["state"][lo], ov.metric: float(values[lo])},
        "highest": {"state": rows["state"][hi], ov.metric: float(values[hi])},
        "no_square": sorted(no_square["name"]),
        "marks": [
            {"n": k, "state": s, ov.metric: float(values[index[s]])}
            for k, s in enumerate(ov.marks, 1)
        ],
        "squares": squares,
    }


# --------------------------------------------------------------------------- imagery entries


def _panel_root(entry: Entry, panel: ImagePanel, root: Path) -> Path:
    """Run root of an imagery panel's files: its own (``ImagePanel.run_root``, relative to the
    repository) when the entry is read from its own run root, else *root* (an explicit
    ``--run-root`` replaces every run root)."""
    if panel.run_root and root == _entry_root(entry):
        return REPO / panel.run_root
    return root


def _image_record(panel: ImagePanel, proot: Path, tif: Path) -> dict | None:
    """The record of ``tools/fetch_naip_history.py`` for a panel's image (``ImagePanel.record``):
    the facts the stats keep. Raises ValueError unless it describes the file drawn (its name
    and SHA-256); None without a record."""
    import hashlib

    if not panel.record:
        return None
    path = _one(panel.record, proot)
    data = json.loads(path.read_text())
    output = data.get("output") if isinstance(data.get("output"), dict) else {}
    sha = hashlib.sha256(tif.read_bytes()).hexdigest()
    if (output.get("file"), output.get("sha256")) != (tif.name, sha):
        raise ValueError(
            f"{path.name} records {output.get('file')!r} (sha256 {str(output.get('sha256'))[:12]}),"
            f" but the panel draws {tif.name!r} (sha256 {sha[:12]})"
        )
    images = data.get("images") or []
    return {
        "file": (path.relative_to(proot) if path.is_relative_to(proot) else path).as_posix(),
        "collection": data.get("collection"),
        "year": data.get("year"),
        "images": [[im.get("asset_id"), im.get("date")] for im in images],
        "native_resolution_m": sorted({im.get("native_resolution_m") for im in images}),
        "bands": data.get("bands"),
        "aggregation": (data.get("export") or {}).get("aggregation"),
        "sha256": sha,
        "credit": data.get("credit"),
    }


def render_imagery(entry: Entry, root: Path, out_dir: Path, basemap: str) -> dict:
    """Render an imagery entry: the entry's reference on each image of ``Entry.imagery``, in
    the entry's window, with no model output.

    Each panel's image is found as a layer's background (:func:`_background_for`) under the
    panel's run root (:func:`_panel_root`) and drawn in ``Entry.window`` (every image must
    cover it, in one CRS), with the reference polygons (``Entry.reference``, in the images'
    CRS) in cyan. The legend counts the reference polygons drawn. The footer gives each image
    (:func:`_bg_note`, with the dates of its record's images and ``ImagePanel.bg_role``), then
    ``Imagery.notes`` and the inset credit. A panel with ``ImagePanel.record`` is refused
    unless the record describes the file drawn (:func:`_image_record`). Options that only the
    other renderers implement are refused.
    """
    im = entry.imagery
    if im is None or not im.panels or entry.layers:
        raise ValueError(f"entry {entry.key}: an imagery entry has Entry.imagery and no layers")
    if basemap != "composite":
        raise ValueError(f"entry {entry.key}: imagery entries draw their own images (composite)")
    if not entry.window or not entry.reference:
        raise ValueError(f"entry {entry.key}: imagery entries need Entry.window and a reference")
    unsupported = [
        n
        for n in (
            "background",  # each panel names its image (ImagePanel)
            "background_from",
            "background_source",
            "bg_role",
            "zoom_m",
            "crop_m",
            "crop_on_reference",
            "crop_on",
            "mark_windows",
            "per_layer_background",
            "multi_area",
            "inset_pad_deg",
            "metrics",
            "overlay",
            "model_notes",  # the footer lines are Imagery.notes
            "sam2_note",
            "halo",
            "title_f1",
            "pred_label",
            "input_notes",
            "copernicus_credit",
            "model_positions",
            "overview",
            "scored_part",
        )
        if getattr(entry, n)
    ]
    if unsupported:
        raise ValueError(
            f"entry {entry.key}: imagery entries do not support {', '.join(unsupported)}"
        )
    _check_publishable(entry, root)
    panels = []
    for i, p in enumerate(im.panels):
        proot = _panel_root(entry, p, root)
        for path in (p.background, p.background_from, p.record):  # as the files they name
            if path and _names_any(path, RESTRICTED_REFERENCES, proot):
                raise ValueError(
                    f"entry {entry.key}: {path} is a restricted reference (not for public "
                    "distribution)"
                )
            if path and _names_any(path, LOCAL_SQUARES, proot):
                raise ValueError(
                    f"entry {entry.key}: window and inset would publish where a study square "
                    "that stays local lies"
                )
        if bool(p.background) == bool(p.background_from):
            raise ValueError(
                f"entry {entry.key}: panel {i + 1} needs one of background and background_from"
            )
        bg = _background_for(entry, proot, {}, basemap, p)
        if len(bg[0]) != 1:
            raise ValueError(f"entry {entry.key}: panel {i + 1} needs one GeoTIFF, not {bg[0]}")
        tif = bg[0][0]
        with rasterio.open(tif) as src:
            crs, rb = src.crs, src.bounds
        record = _image_record(p, proot, tif)
        panels.append(
            {"p": p, "root": proot, "bg": bg, "tif": tif, "crs": crs, "rb": rb, "record": record}
        )
    crs = panels[0]["crs"]
    if any(q["crs"] != crs for q in panels):
        raise ValueError(f"entry {entry.key}: the panels' images are not in one CRS")
    wx, wy, side = entry.window
    bounds = (wx - side / 2, wy - side / 2, wx + side / 2, wy + side / 2)
    for i, q in enumerate(panels):
        rb = q["rb"]
        inside = rb.left <= bounds[0] and rb.bottom <= bounds[1]
        inside = inside and bounds[2] <= rb.right and bounds[3] <= rb.top
        if not inside:
            raise ValueError(
                f"entry {entry.key}: window {entry.window} is not inside the image of panel {i + 1}"
            )
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    ref = _read_reference(entry.reference, root, crs, entry.reference_min_area_m2)
    ref = ref.cx[bounds[0] : bounds[2], bounds[1] : bounds[3]]

    # Footer notes: each image, then the entry's lines and the inset credit.
    pos = _panel_names(len(panels), entry.ncols)
    notes = []
    for i, q in enumerate(panels):
        note = _bg_note(q["bg"], role=False)  # ends with the image count, when it is known
        n_images = _composite_facts(q["bg"][2], q["tif"]).get("n_images")
        if q["record"] and n_images:
            dates = sorted({d for _id, d in q["record"]["images"] if d})
            note += f" of {_and_join(dates)}" if dates else ""
        if q["p"].bg_role:
            note += f"; {q['p'].bg_role}"
        notes.append(f"{pos[i]}: {note}")
    paragraphs = [("Background: " if k == 0 else "") + n for k, n in enumerate(notes)]
    paragraphs += list(im.notes)
    loc = None
    if entry.inset:
        from pyproj import Transformer

        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(wx, wy)
        loc = _locate(lon, lat)
        paragraphs.append(INSET_NOTE_INDIA if _inset_shows_india(loc) else INSET_NOTE)
    ref_label = f"{entry.reference_label} ({_count(len(ref), 'field')})"
    handles = [Line2D([], [], color=REF_COLOR, lw=1.5, label=ref_label)]

    # Footer layout (as in render): the legend at the top, the notes under it, the inset at
    # the bottom right; the height grows with the notes.
    legend_ncol = len(handles)
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
    notes_top = FOOT_TOP_IN + legend_h + LEGEND_NOTE_GAP_IN
    footer = notes_top + _notes_height_in(len(note_lines)) + FOOT_BOTTOM_IN
    if inset:
        beside_legend = legend_right + NOTE_INSET_GAP_IN <= inset["left"]
        footer = max(footer, inset["top"] + (FOOT_TOP_IN if beside_legend else notes_top))

    n = len(panels)
    ncols = min(entry.ncols or n, n)
    nrows = math.ceil(n / ncols)
    panel_w = FIG_WIDTH_IN * (MAPS_RIGHT - MAPS_LEFT) / (ncols + MAPS_WSPACE * (ncols - 1))
    panel_h = panel_w  # a square window
    row_gap = ROW_GAP_IN if nrows > 1 else 0.0
    fig_h = TITLE_IN + nrows * panel_h + (nrows - 1) * row_gap + footer
    px = (panel_w * DPI, panel_h * DPI)
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH_IN, fig_h), dpi=DPI, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    lw = entry.line_width * (1.0 if n == 1 else 0.8)
    titles = [q["p"].label for q in panels]
    # One title size for the figure: the largest at which every title fits its panel.
    title_fs = min(_fit_title(t, panel_w) for t in titles)
    for ax, q, title in zip(axes, panels, titles, strict=False):
        tifs, source, *_ = q["bg"]
        img, img_ext, _stretch = _read_background(tifs, source, bounds, crs, px)
        ax.imshow(img, extent=img_ext, interpolation="auto", zorder=1)
        if len(ref):
            ref.boundary.plot(ax=ax, color=REF_COLOR, linewidth=lw * 0.9, zorder=3)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_anchor("N")
        _bare_axes(ax)
        for s in ax.spines.values():
            s.set_linewidth(0.6)
        _scalebar(ax, extent, crs)
        ax.set_title(title, fontsize=title_fs, pad=4)

    iax = None
    if loc is not None:
        rect = [
            1 - (INSET_W_IN + INSET_RIGHT_IN) / FIG_WIDTH_IN,
            FOOT_BOTTOM_IN / fig_h,
            INSET_W_IN / FIG_WIDTH_IN,
            INSET_H_IN / fig_h,
        ]
        iax, _ = _draw_inset(fig, rect, loc, inset["title"])
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

    records = []
    for i, q in enumerate(panels):
        tifs, source, comp_prov, _role = q["bg"]
        proot, tif = q["root"], q["tif"]
        credit = _bg_credit(q["bg"])
        records.append(
            {
                "panel": pos[i],
                "label": q["p"].label,
                "file": (tif.relative_to(proot) if tif.is_relative_to(proot) else tif).as_posix(),
                "run_root": (
                    proot.relative_to(REPO) if proot.is_relative_to(REPO) else proot
                ).as_posix(),
                **({"from": q["p"].background_from} if q["p"].background_from else {}),
                "source": source,
                "composite": _composite_facts(comp_prov, tif),
                **({"background_credit": credit} if credit else {}),
                **({"record": q["record"]} if q["record"] else {}),
            }
        )
    return {
        "example": entry.key,
        "image": (png.relative_to(REPO) if png.is_relative_to(REPO) else png).as_posix(),
        "preview": (
            preview.relative_to(REPO) if preview.is_relative_to(REPO) else preview
        ).as_posix(),
        "title": entry.title,
        "background": "composite",
        "background_note": "\n".join(note_lines),
        "notes": list(im.notes),
        "proj_network": proj_network_enabled(),
        "location": _location_record(loc),
        "window": {"crs": str(crs), "centre": [wx, wy], "side_m": side},
        "reference": {**_reference_stats(entry, ref), "legend": ref_label},
        "panels": records,
    }


def _entry_root(entry: Entry, run_root=None) -> Path:
    """Run root of *entry*: *run_root* (``--run-root``) when given, else the entry's own
    ``run_root`` (relative to the repository), else RUN_ROOT."""
    if run_root is not None:
        return Path(run_root)
    return REPO / entry.run_root if entry.run_root else RUN_ROOT


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--run-root",
        default=None,
        help="run root of the example outputs, for every entry (default: each entry's own "
        f"run root, else {RUN_ROOT.relative_to(REPO)}; outputs/examples_1.0 for the 1.0 "
        "gallery)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help=f"gallery folder (default {OUT_DIR.relative_to(REPO)}; assets/gallery_1.0 for "
        "the 1.0 gallery)",
    )
    ap.add_argument("--only", default="", help="comma-separated example numbers")
    ap.add_argument("--basemap", choices=("composite", "esri"), default="composite")
    ap.add_argument("--list", action="store_true", help="list the entries and exit")
    ap.add_argument(
        "--no-proj-network",
        action="store_true",
        help="keep PROJ's network setting of the environment (PROJ_NETWORK) instead of "
        "enabling it while rendering",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    import pyproj.network

    args = _parser().parse_args(argv)

    out_dir = Path(args.out_dir)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    entries = [e for e in ENTRIES if not only or e.key in only]
    if args.list:
        for e in entries:
            root = _entry_root(e, args.run_root)
            shown = root.relative_to(REPO) if root.is_relative_to(REPO) else root
            print(f"{e.key}  {e.name:24s} {e.title}")
            print(f"      run root: {shown}")
            if e.overview:
                paths = [e.overview.summary]
            elif e.imagery:
                paths = [
                    (p.background or p.background_from)
                    + (f" (run root: {p.run_root})" if p.run_root and not args.run_root else "")
                    for p in e.imagery.panels
                ]
            else:
                paths = [
                    lay.path
                    + (f" (run root: {lay.run_root})" if lay.run_root and not args.run_root else "")
                    for lay in e.layers
                ]
            for path in paths:
                print(f"      {path}")
        return 0

    stats_path = out_dir / "gallery_stats.json"
    # A full run writes the file afresh (no entry from an older version of this script
    # survives); --only replaces the rendered entries and keeps the others.
    all_stats = json.loads(stats_path.read_text()) if (only and stats_path.exists()) else {}
    failed = []
    network = pyproj.network.is_network_enabled()
    if not args.no_proj_network:
        pyproj.network.set_network_enabled(True)  # datum grids, as PROJ_NETWORK=ON
    try:
        for e in entries:
            try:
                renderer = (
                    render_overview
                    if e.overview
                    else render_imagery
                    if e.imagery
                    else render_areas
                    if e.multi_area
                    else render
                )
                all_stats[e.key] = renderer(e, _entry_root(e, args.run_root), out_dir, args.basemap)
                print(f"[{e.key}] {out_dir / (e.name + '.png')}")
            except Exception as exc:  # report and continue with the other entries
                failed.append(e.key)
                print(f"[{e.key}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        pyproj.network.set_network_enabled(network)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Strict JSON: copied metrics hold NaN where a score is undefined (RFC 8259 has no NaN).
    stats = to_jsonable(dict(sorted(all_stats.items())))
    stats_path.write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
