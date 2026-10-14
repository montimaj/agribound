"""
24 — Nevada: Centre Pivots and Flood Irrigation against the DRI Field Boundaries (2022-2024)

How agribound does in two contrasting Nevada basins, evaluated against the DRI
field boundaries (Nevada, 2026-08-20), the field layer of the Nevada Water
Inventory (Desert Research Institute). It is the Nevada counterpart of
examples 12 and 20 (New Mexico), with the held-out protocol of example 23:
the fine-tuned models learn from a separate square of the same basin and
never see the evaluated fields. Like example 23, it has no HLS, GeoAI or
Prithvi runs and no vote ensembles (example 12 has them).

    1. Diamond Valley (hydrographic basin 153_DiamondValley, Eureka County,
       about 1,800 m): large fields irrigated from groundwater, many of them
       under centre pivots or linear moves.
    2. Mason Valley (basin 108_MasonValley, Lyon County, about 1,340 m):
       smaller fields under flood irrigation or wheel lines and other
       sprinklers, irrigated from surface water and groundwater.

Data: the inventory is not in this repository (contact the author for
access). The script reads examples/NV_Field_Boundaries.zip in place
(``--reference`` takes another copy). Figures and documentation may show its
outlines, where the squares lie, and the counts and scores, with the
attribution "DRI field boundaries (Nevada, 2026-08-20), Desert Research
Institute".

Study squares, chosen by a fixed rule from the inventory alone, before any
model ran on them (``rule_square``):

    - candidates: the 5 km squares of UTM zone 11N (EPSG:32611, the
      inventory's CRS) whose corners lie on a 1 km grid;
    - the evaluation square of a basin holds the most DRI fields of the
      basin (``SOURCECODE`` "DRI", at least 2,500 m2) by representative
      point; ties go to the southernmost, then the westernmost square;
    - its training square is chosen the same way among the squares at
      least 5 km from the evaluation square, edge to edge.

The squares (``BASINS``; lower-left corners in UTM zone 11N, centres in
degrees; the script re-applies the rule and warns when the inventory gives
other squares, for example in another release):

    Diamond Valley, evaluation: corner (584000, 4389000), centre -115.9915,
        39.6689. 74 fields, median 48.2 ha: 39 pivot, 16 sprinkler, 1
        controlled flood, 18 uncontrolled flood or no apparent equipment;
        4, 11, 43 and 16 fields under 2 ha, 2-10 ha, 10-50 ha and 50 ha or
        more; irrigated (IrrMapper) 39, 40 and 34 in 2022, 2023 and 2024.
    Diamond Valley, training: corner (582000, 4379000), centre -116.0161,
        39.5791, 5.0 km south of the evaluation square (centres 10.2 km
        apart). 54 fields.
    Mason Valley, evaluation: corner (310000, 4306000), centre -119.1623,
        38.9054. 187 fields, median 7.7 ha: 105 sprinkler, 39 controlled
        flood, 30 uncontrolled flood or no apparent equipment, 9 retired, 4
        pivot; 27, 83, 77 and 0 by size class; irrigated 148, 173 and 159 in
        2022, 2023 and 2024.
    Mason Valley, training: corner (308000, 4327000), centre -119.1912,
        39.0941, 16.0 km north of the evaluation square (centres 21.1 km
        apart). 161 fields.

The evaluation reference holds DRI-mapped fields only; the training labels
are every field that reaches into the training square (see Fine-tuning; the
script prints their providers and how many were split from oversized
polygons, ``NWI_ID_PARENT``: in this release all were mapped by DRI and none
was split). Where fields are large, a 5 km square holds few of them: with 74
fields, Diamond Valley's scores have wide intervals.

Approaches (each a separate agribound run with ``lulc_filter=False``):

    - Delineate-Anything v2 (``large_v2``), used as released, on NAIP (1 m,
      2022), SPOT 6/7 (6 m, 2023; restricted), Sentinel-2 (10 m), Landsat
      8/9 (30 m), Landsat 8/9 panchromatic (``landsat-pan``, 15 m) and
      Sentinel-1 (VV, VH and VV - VH backscatter as R, G and B; 10 m pixels,
      about 20 m resolution), the last four of 2024; on NAIP also with SAM 2
      refinement. Landsat and Landsat PAN are outside the model's 0.25-10 m
      training range (agribound logs a WARNING);
    - Delineate-Anything v2 and DINOv3, each fine-tuned on the NAIP and on
      the SPOT composite of the basin's training square and run on the same
      source of the evaluation square (four fine-tuned models per basin);
    - FTW (default model, ``FTW_PRUE_EFNET_B5``: two Sentinel-2 windows from
      FTW's crop calendar) on Sentinel-2 2024;
    - unsupervised k-means (k = 20) of the Google Satellite Embedding and of
      TESSERA, 2024. These are land-cover segments, not field instances;
    - with ``--sam-auto`` (opt-in, off by default): the direct SAM engine
      (``engine="sam-auto"``, SAM 2.1 large automatic masks) with its own
      per-source defaults on NAIP, SPOT 6/7 and Landsat 8/9, the imagery
      types that its pre-registered evaluation (2026-10-13) kept. On
      Sentinel-2 it did not transfer in that evaluation (mean F1 0.15 below
      Delineate-Anything's on 46 squares of example 25; on this example's
      two evaluation squares, a secondary result, it scored higher), so it
      does not run on Sentinel-2 here. Those defaults were chosen on this
      example's two training squares, so its scores on the evaluation
      squares are held out from that choice but come from the same basins,
      5.0 km (Diamond Valley) and 16.0 km (Mason Valley) away;
    - with ``--options`` (opt-in, off by default): the "-opt" runs, four
      runs per basin made again with the opt-in options of agribound 1.1.0
      that passed a pre-registered test (see Options below).

Years (``SOURCE_YEARS``): 2024 imagery, or the latest year of a source without
2024 imagery: NAIP 2022 (Earth Engine's USDA/NAIP/DOQQ has nothing later over
the squares; USGS NAIP Plus also stops at 2023) and SPOT 6/7 2023 (it ends on
2023-11-15). NAIP and SPOT are therefore 2 and 1 years older than the other
sources, and fields that changed in between count as errors in their scores.
Output names, log lines and the summary carry the year of every run.

Composites: SPOT, Sentinel-2, Landsat, Landsat PAN and Sentinel-1 are medians
over the growing season, 1 April to 31 October of their year (``SEASON``). The
calendar year would add winter images, and the Sentinel-2 cloud mask keeps
snow: of the 85 Sentinel-2 images of 2024 over the Diamond Valley square with
at most 20 % cloud, 26 report more than 5 % snow or ice in their tile
metadata, 20 of them from November to March. FTW builds its own two windows;
NAIP and the embeddings are one image per year. Queried in Earth Engine on
2026-10-08: NAIP 2022 covers every square, flown on 30 June (Mason Valley
evaluation), 10 June (Mason Valley training) and, in Diamond Valley, on 29
June and 19-20 September (both squares join quarter-quads of both dates).
SPOT 6/7 has one 2023 scene over the Diamond Valley squares (2023-07-14) and,
in the window, six over the Mason Valley evaluation square and five over its
training square, all without cloud. TESSERA v1 has 2024 tiles over both
evaluation squares.

Fine-tuning. The labels are every field of the inventory that reaches into
the training square, whatever its size (the chips label every other pixel as
background): 54 fields in Diamond Valley and 179 in Mason Valley. None of
them reaches into the evaluation square (the script checks), and the
training composites cover the training square only, so no evaluated field or
pixel is used. Both engines get the same labels, composite and rules: their
default chips and minimum labelled share, a random 20 % for validation and
20 epochs; Delineate-Anything with ``yolo_lr0=1e-4``, as in example 23 (the
default learning rate can wreck the pretrained weights on few chips), DINOv3
(ViT-L/16 backbone pre-trained on SAT-493M satellite imagery, DPT head; no
published field-boundary weights) with agribound's defaults. DINOv3's chips
do not overlap, so 20 % of them are validation chips. Delineate-Anything's
chips overlap, half a chip apart, so the 20 % applies to chip-size cells: a
chip takes the split of its cells, and the chips that cover cells of both
splits are dropped (no pixel is in both splits). At 6 m the square is about
834 px wide, so the SPOT models learn from few chips: with the chip rules of
2026-10-08, at most 9 DINOv3 chips (256 px) and 25 overlapping
Delineate-Anything chips (256 px every 128 px, over 9 cells) before the
validation split, against 361 and 324 on NAIP at 1 m. On the SPOT square a
random 20 % of the 9 cells gives Delineate-Anything about 2 validation cells,
so only a chip that lies entirely in them (2 or 3 of the 25 positions) can be
a validation chip, and the chips that straddle both splits are dropped. If
no chip with enough labelled pixels lies there, that fine-tuning stops with
``ValueError: No validation chip is left ...``; the script reports it as
failed (``failed: ValueError`` in ``summary.csv`` for the run that needs its
model) and goes on with the other runs. In the development run of agribound
1.1 (2026-10-10) every fine-tuning ran, on these training and validation
chips (Diamond Valley; Mason Valley): Delineate-Anything on SPOT 17 and 2 in
each basin (6 dropped), DINOv3 on SPOT 7 and 2 in each basin,
Delineate-Anything on NAIP 154 and 23 (75 dropped; 222 and 27, 74 dropped)
and DINOv3 on NAIP 201 and 50 (274 and 69). The fine-tuning
outputs (in-sample, in ``<basin>_train/``) are never evaluated. The squares
and all settings were fixed before this example ran any model. FBIS-73M is
the training set of the released Delineate-Anything v2 weights. In its
public lists (revision 9e4cdbd; the US- and Canadian-prefixed patches,
located by their own GeoTIFF headers on 2026-10-13), no patch lies within
10 km of either evaluation square: the nearest lies 285.8 km from Diamond
Valley's (the nearest labelled one 288.0 km) and 38.0 km from Mason Valley's
(labelled), both in California, where all its labelled US training patches
lie. Only the US- and Canadian-prefixed patches were located, and the public
lists were published after training and changed on the day of the release,
so they may not be exactly what the released weights saw. Its training list
(checked on 2026-10-08) names 76,448 Sentinel-1 patches
(``*_S1_10m_256_custom``) and 7,852 ``*_SYN-S1_256_custom`` patches among its
1,432,187 but not how their bands were rendered, so radar input is not new to
the model, although agribound's composite may differ from those patches.

Crop filter. Every run delineates without the LULC crop filter, which is then
applied as a separate step, as in example 23, with the default rule: in the
conterminous US, Annual NLCD of the run's year, and a polygon is kept when at
least 30 % of it is cultivated crops or pasture/hay (classes 82 and 81). The
table reports how many polygons it keeps, ``summary.csv`` also the share of
the matched predictions (IoU >= 0.5) it keeps (``matched_kept_crop``), and
the script prints how many reference fields it keeps (NLCD 2024). A field
that was not irrigated in a year may be mapped as shrubland or grassland that
year, so the filter can remove it.

Evaluation (``agribound.evaluate``): one-to-one matching at IoU >= 0.5,
boundary metrics at a 10 m tolerance, and 95 % bootstrap intervals (200
resamples of the reference fields, stratified by irrigation system; they
treat fields as independent, so they are too narrow when errors are
spatially clustered; ``summary.csv`` gives them for recall and F1). The
reference is the DRI fields whose representative point lies inside the
evaluation square, without those smaller than 2,500 m2 (the predictions'
minimum area; the script prints how many it drops: none in either evaluation
square of the 2026-08-20 release, so the reference is the 74 and 187 fields
above), and the predictions are selected by the same representative-point
rule. Recall is also reported per stratum:

    - irrigation system (``ITYPE_2024``, after the layer metadata's
      descriptions of the codes): pivot (1: centre pivot or linear move),
      sprinkler (2: wheel line, hand line, solid set, ...), controlled flood
      (4: levelled land, borders, basins, furrows), uncontrolled flood or no
      apparent irrigation equipment (3), retired (0: developed or no longer
      irrigated) and micro (5: micro-sprinklers, drip). Code 3 is a stratum
      of its own: a field with no apparent irrigation equipment need not be
      irrigated at all, so read its recall with the irrigated / not
      irrigated strata. ``IRR_SRC`` is the water source, not the system. The
      script prints how many evaluated fields have another ITYPE in 2022 or
      2023;
    - field size: under 2 ha, 2-10 ha, 10-50 ha, 50 ha or more;
    - irrigated or not in the year of the run's imagery: the inventory's
      IrrMapper share of the field (``irrmapper_<year>``) is at least 0.5 or
      not. Fields that were not irrigated in that year stay in the reference.

The irrigated / not irrigated strata depend on the imagery year, so the
per-stratum table prints the number of reference fields of every stratum
once per imagery year (2022, 2023 and 2024), and ``summary.csv`` gives the
irrigated and not irrigated counts of each run (``n_irrigated``,
``n_not_irrigated``).

Irrigated-only score. Every run is also scored against the reference fields
that were irrigated in the year of its imagery (IrrMapper share at least
0.5, the rule of the strata above): in Diamond Valley 39, 40 and 34 fields
in 2022, 2023 and 2024, in Mason Valley 148, 173 and 159. The other fields
of the square (35, 34 and 40; 39, 14 and 28) are passed to
``agribound.evaluate`` as ``ignore``: a prediction whose representative
point lies in one of them is not scored, whether or not it outlines that
field (``n_predicted_ignored``). Every other prediction is scored as in the
main score, with the same matching, tolerance, strata and bootstrap. The
score thus drops the fields not irrigated that year from the reference,
and the predictions on them, but it is not a score on irrigated land
alone: predictions on ground the inventory does not map (corners between
pivots, rangeland, roads) still count as false positives, and a prediction
that covers an irrigated field and an ignored one is scored or ignored by
where its representative point lies. Its recall can differ from that of the
irrigated stratum (``recall_irrigated``), which comes from one matching of
all fields and all predictions. ``summary.csv`` adds
``n_reference_irrigated``, ``n_predicted_scored``, ``n_predicted_ignored``,
``precision_irrigated``, ``recall_irrigated_score`` (bootstrap interval
``recall_irrigated_score_ci_low`` and ``_ci_high``), ``f1_irrigated``
(``f1_irrigated_ci_low`` and ``_ci_high``), ``iou_mean_irrigated``,
``boundary_f1_irrigated`` and ``fp_noref_irrigated`` (scored predictions
that overlap no irrigated reference field); each run writes
``metrics_<output>_irrigated.json`` beside its ``metrics_<output>.json``
(whose ``metrics_irrigated`` key holds the irrigated / not irrigated strata,
not this score), and the script prints a second table per basin.

Merge and split rates as in example 23: a reference field is merged when its
main prediction also covers at least 25 % of another reference field, and
split when at least two predictions each cover 10 % of it. The inventory maps
irrigated fields and fields retired from irrigation, not all land, so
predictions elsewhere count as false positives (FP-noref: predictions that
overlap no reference field). It records no imagery date per field, so fields
that changed between its mapping and the imagery count as errors. The annual
ITYPE series are reconstructed, not observed (layer metadata), so the system
of a field in a given year is an estimate. The DRI-mapped fields are DRI's
field delineation, which the OpenET field boundaries also use in Nevada and
Oregon (example 25).

The reference also limits what any outline can score in Diamond Valley (from
the reference alone): 38 of its 74 fields are circles (intersection over
union at least 0.9 with the circle of equal area) and 19 are the land around
a pivot mapped as one field, 13 of them in several pieces. For 4 of these 19
(3 or 4 pieces each) the largest piece is 27-36 % of the field, so no outline
of a single piece reaches the IoU of 0.5 that a match needs: outlines that do
not join separate pieces can match at most 70 of the 74 fields (recall 0.95;
F1 0.97 with no false positive). In Mason Valley 1 of the 187 fields lies
around a pivot (2 pieces, the larger 81 % of the field), and every field is
within reach of an outline of one piece.

Options (``--options``: the "-opt" runs). Four runs per basin are made again
with the opt-in options of agribound 1.1.0 that passed a pre-registered test
(decided on 2026-10-13): each option was tested on its own, by rules fixed
before the test, on three 5 km pivot squares in other Nevada basins, chosen
before any model ran on them, with this example's settings and models (so
the DINOv3 models ran outside the basin they learned from), the pivot snap
also on the four runs below of both evaluation squares and the 3 m grid also
in Mason Valley. Each "-opt" run has every option that applies to it, in the
pre-registered order: inside the engine, Delineate-Anything reads NAIP on a
3 m grid (``engine_params["processing_gsd_m"] = 3``) and DINOv3 grows the
field interiors by half the boundary band measured on its own prediction
(``"dilate_interior_px": "half_band"``); then the pipeline rejoins centre
pivots cut into pieces and snaps them to circles (``"pivots": "snap"``, its
last geometric step); last, this script's crop-filter step, whose layer is
the result of the optical runs (the filter is never applied to Sentinel-1):

    - Diamond Valley: ``naip da-opt`` (3 m grid, pivots, crop filter),
      ``naip dinov3-ft-opt`` and ``spot dinov3-ft-opt`` (half band, pivots,
      crop filter) and ``sentinel1 da-opt`` (pivots);
    - Mason Valley: ``naip da-opt``, ``naip dinov3-ft-opt``, ``spot da-opt``
      (Delineate-Anything as released: pivots, crop filter) and
      ``sentinel1 da-opt``.

Everything else is the base run's (``naip da``, ``naip dinov3-ft``, ...,
with the same models). An "-opt" run is scored like the others: its rows in
the tables and ``summary.csv`` score its result (for an optical run the
crop-filter layer; ``kept_crop`` and ``matched_kept_crop`` then describe the
crop-filter step on the run's output), and ``options.csv`` gives its change
against the base run in F1, irrigated-only F1 and boundary F1, with 95 %
intervals from a paired bootstrap of the square's 1 km blocks (2,000
resamples, seed 42; ``agribound.evaluate.paired_bootstrap``), which also
resamples the predictions on ground the inventory does not map
(``metrics_<output>_change.json`` holds the records). The "-opt" runs combine
options that were tested one at a time, so their scores are reported
results, not tests. On this example's evaluation squares only the pivot snap
was tested blind, and the 3 m grid in Mason Valley; the other effects there
were known or partly known before the test: the half-band growth from an
earlier sweep of the growth on these rasters, the 3 m grid in Diamond Valley
from example 26 (which scored it with ``merge_tile_pieces=False``), and the
crop filter from this example's ``kept_crop`` and ``matched_kept_crop``
columns.

SPOT 6/7 (AIRBUS/SPOT6_7) is restricted to select Earth Engine users. Without
access the SPOT runs and the SPOT fine-tunings are skipped and the rest runs.
``--no-dinov3`` skips DINOv3 (full fine-tuning of a ViT-L; each checkpoint is
about 3.7 GB, in the training square's ``.agribound_cache``), ``--no-fine-tune``
all fine-tuning and ``--no-sam`` the SAM 2 run; ``--sam-auto`` adds the direct
SAM runs and ``--options`` the "-opt" runs (an "-opt" run of a model that is
skipped is skipped too); ``--only mason_valley`` (or ``diamond_valley``) runs
one basin. ``--dry-run`` reads the inventory, prints the squares, the
reference counts and every run's configuration, and runs and writes nothing.

Runtime: 116 minutes (6,976 s; the script printed 115.8 minutes) in the
development run of agribound 1.1 (2026-10-10, Apple M2 Max with MPS;
composites and FTW's checkpoint downloaded, the other model weights already
cached). The eight fine-tunings took 102 minutes, each with its in-sample
inference and the first of each source with the download of its training
composite (Diamond Valley; Mason Valley): Delineate-Anything on NAIP 11 and
20 minutes, DINOv3 on NAIP 23 and 27, Delineate-Anything on SPOT 3 and 3,
DINOv3 on SPOT 8 and 7. The 28 evaluated runs and their scores took 14
minutes, the largest on the NAIP composites (5,001-5,002 px square). A run
that finds its outputs scores them again: on 2026-10-11, when the
irrigated-only score was added, it took 71 s, reusing 34 of the 36 outputs
and making the two TESSERA runs again (with the same polygons; their records
predated the TESSERA store stamp). On 2026-10-13, ``--options --sam-auto``
with those 36 outputs in place took 5 minutes (322 s; the script printed 5.3
minutes): in the pipeline (provenance records), each direct SAM run took
83-84 s on a NAIP square and 7 s on a SPOT 6/7 or Landsat square, model
loading included, and each "-opt" run 4-11 s (DINOv3 reused its cached
predictions; the 3 m Delineate-Anything runs took the longest).

Prerequisites:
    pip install "agribound[gee,delineate-anything,dinov3,ftw,embedding,tessera,samgeo]"
        (or use environment.yml; --sam-auto also needs sam-auto)
    agribound auth --project YOUR_GEE_PROJECT
    DRI field boundaries (Nevada, 2026-08-20) at "examples/NV_Field_Boundaries.zip"
        (local data, not in this repository; read in place, no need to unzip)
    Run from the repository root: python examples/24_nevada_field_boundaries.py
"""

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path

import agribound
from agribound.provenance import to_jsonable

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

warnings.filterwarnings("ignore", category=FutureWarning, module=r"geedim\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"geedim\..*")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
REFERENCE_ZIP = "examples/NV_Field_Boundaries.zip"
REFERENCE = f"zip://{REFERENCE_ZIP}!nevada_agricultural_field_boundaries_20260820.gpkg"
REFERENCE_LABEL = "DRI field boundaries (Nevada, 2026-08-20)"
REFERENCE_COLUMNS = [
    "NWI_ID", "NWI_ID_PARENT", "SOURCECODE", "Basin", "IRR_SRC",
    "ITYPE_2022", "ITYPE_2023", "ITYPE_2024", "irrmapper_2022", "irrmapper_2023", "irrmapper_2024",
]  # fmt: skip
OUTPUT_DIR = Path("outputs/nevada_field_boundaries")
REFERENCE_DIR = OUTPUT_DIR / "references"
UTM_CRS = "EPSG:32611"  # UTM zone 11N: the inventory's CRS and the zone of both basins
SIDE_M = 5000  # side of every square (evaluation and training)
GRID_M = 1000  # the squares' corners lie on this grid of UTM zone 11N
MIN_GAP_M = 5000  # a training square lies at least this far from its evaluation square
MIN_AREA = 2500  # m^2; the default minimum field area (also applied to the reference)
BOUNDARY_TOLERANCE_M = 10.0
N_BOOTSTRAP = 200
CROP_THRESHOLD = 0.3
N_CLUSTERS = 20  # embedding k-means, fixed in advance as in example 23
MERGE_SHARE = 0.25  # merged: the main prediction also covers this share of another field
SPLIT_SHARE = 0.10  # split: at least two predictions each cover this share of the field
#: Imagery year of each source: 2024, or the latest year with imagery over the squares.
SOURCE_YEARS = {
    "naip": 2022,  # Earth Engine NAIP (USDA/NAIP/DOQQ): 2022 is the latest year here
    "spot": 2023,  # SPOT 6/7 ends on 2023-11-15
    "sentinel2": 2024,
    "landsat": 2024,
    "landsat-pan": 2024,
    "sentinel1": 2024,
    "google-embedding": 2024,
    "tessera-embedding": 2024,
}
#: Composite window (month-day, inclusive) of these sources: the growing season.
SEASON = ("04-01", "10-31")
SEASONAL_SOURCES = ("spot", "sentinel2", "landsat", "landsat-pan", "sentinel1")
SPOT_CLOUD_COVER_MAX = 15
#: ITYPE_2024 (irrigation system) codes of the inventory and their strata, after the layer
#: metadata's descriptions of the codes. Code 3 is a stratum of its own: a field with no
#: apparent irrigation equipment need not be irrigated at all.
IRRIGATION_SYSTEMS = {
    0: "retired",  # developed or no longer irrigated
    1: "pivot",  # centre pivot or linear move
    2: "sprinkler",  # wheel line, hand line, solid set, big gun, traveling gun, pods
    3: "uncontrolled flood or no apparent irrigation equipment",
    4: "controlled flood",  # levelled land, borders, basins, furrows
    5: "micro",  # micro-sprinklers, drip lines, subsurface drip
}
#: Short name of each stratum: the table columns and the summary.csv keys (recall_<name>).
STRATA = {
    "pivot": "pivot",
    "sprinkler": "sprinkler",
    "controlled flood": "flood_ctl",
    "uncontrolled flood or no apparent irrigation equipment": "flood_unc_none",
    "retired": "retired",
    "micro": "micro",
}
IRRIGATED_SHARE = 0.5  # irrigated in a year: IrrMapper classes at least half the field irrigated
SIZE_BINS_HA = (0, 2, 10, 50, float("inf"))
FINE_TUNE_EPOCHS = 20
FINE_TUNE_SOURCES = ("naip", "spot")
#: Sources of the direct SAM runs (--sam-auto): the imagery types that the engine's
#: pre-registered evaluation kept (2026-10-13; not Sentinel-2, where it did not transfer),
#: each with its default setting (agribound.engines.sam_auto.SOURCE_SETTINGS); SPOT only
#: when SPOT runs.
SAM_AUTO_SOURCES = ("naip", "spot", "landsat")
#: Engine parameters of the fine-tunings (Delineate-Anything: as in example 23).
FINE_TUNE_PARAMS = {"delineate-anything": {"yolo_lr0": 1e-4}, "dinov3": {}}
#: The "-opt" runs (--options): the base runs ("<source> <label>") made again with the
#: options that passed the pre-registered test of the Diamond Valley options (three pivot
#: squares in other Nevada basins; docstring, "Options"), each "<source> <label>-opt".
OPTION_BASES = {
    "diamond_valley": ("naip da", "naip dinov3-ft", "spot dinov3-ft", "sentinel1 da"),
    "mason_valley": ("naip da", "naip dinov3-ft", "spot da", "sentinel1 da"),
}
OPTION_GSD_M = 3.0  # Delineate-Anything on NAIP reads the composite on a 3 m grid
OPTION_PIVOTS = "snap"  # every "-opt" run: pivots rejoined and snapped to circles
OPTION_GROWTH = "half_band"  # DINOv3: interiors grown by half the measured boundary band
OPTION_NO_CROP = ("sentinel1",)  # the crop filter is never applied to Sentinel-1
N_PAIRED = 2000  # resamples of the paired bootstrap of an "-opt" run's change
BLOCK_M = 1000.0  # its blocks: 1 km squares of UTM zone 11N

# The basins and their squares (lower-left corners in UTM zone 11N, m), chosen by the
# rule in the docstring from the inventory alone, before any model ran on them. The
# script re-applies the rule and warns when the inventory gives other squares.
BASINS = {
    "diamond_valley": {
        "name": "Diamond Valley (basin 153), Eureka County: centre pivots",
        "basin": "153_DiamondValley",
        "evaluation": (584000, 4389000),
        "training": (582000, 4379000),
    },
    "mason_valley": {
        "name": "Mason Valley (basin 108), Lyon County: flood and sprinkler irrigation",
        "basin": "108_MasonValley",
        "evaluation": (310000, 4306000),
        "training": (308000, 4327000),
    },
}


def reference_exists(path):
    """True if *path* exists; for ``zip://ARCHIVE!MEMBER``, if the archive exists."""
    if str(path).startswith("zip://"):
        return Path(str(path)[len("zip://") :].split("!", 1)[0]).exists()
    return Path(path).exists()


def load_inventory(path):
    """The inventory's fields in UTM zone 11N, with valid polygonal geometries."""
    import geopandas as gpd
    import shapely

    gdf = gpd.read_file(path, columns=REFERENCE_COLUMNS).to_crs(UTM_CRS)
    polys = []
    for geom in shapely.make_valid(gdf.geometry.to_numpy()):
        parts = shapely.get_parts(geom)
        polys.append(shapely.union_all(parts[shapely.area(parts) > 0]))
    gdf = gpd.GeoDataFrame(gdf.drop(columns="geometry"), geometry=polys, crs=UTM_CRS)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)


def square(corner):
    """The SIDE_M square with lower-left *corner* (UTM zone 11N) as a shapely box."""
    from shapely.geometry import box

    x, y = corner
    return box(x, y, x + SIDE_M, y + SIDE_M)


def square_study_area(area, name):
    """A GeoJSON FeatureCollection of *area* (a polygon in UTM zone 11N) in EPSG:4326."""
    import geopandas as gpd
    from shapely.geometry import mapping

    geometry = gpd.GeoSeries([area], crs=UTM_CRS).to_crs(4326).iloc[0]
    feature = {"type": "Feature", "geometry": mapping(geometry), "properties": {"name": name}}
    return {"type": "FeatureCollection", "features": [feature]}


def lon_lat(area):
    """Longitude and latitude of the centre of *area* (UTM zone 11N)."""
    import geopandas as gpd

    centre = gpd.GeoSeries([area.centroid], crs=UTM_CRS).to_crs(4326).iloc[0]
    return centre.x, centre.y


def rep_point_inside(gdf, area):
    """Rows whose representative point lies inside *area* (same CRS)."""
    return gdf[gdf.representative_point().within(area).to_numpy()].copy()


def rule_fields(fields, basin):
    """The fields the square rule counts: DRI fields of *basin* of at least MIN_AREA."""
    keep = (fields["Basin"] == basin) & (fields["SOURCECODE"] == "DRI") & (fields.area >= MIN_AREA)
    return fields[keep.to_numpy()]


def rule_square(fields, basin, exclude=None):
    """Lower-left corner of the square the rule picks, and its number of fields.

    The candidates are the SIDE_M squares whose corners lie on the GRID_M grid of
    UTM zone 11N; with *exclude* (a square), only those at least MIN_GAP_M from it
    edge to edge. The rule picks the candidate holding the most ``rule_fields`` by
    representative point; ties go to the southernmost, then the westernmost.
    """
    import numpy as np

    points = rule_fields(fields, basin).representative_point()
    x, y = points.x.to_numpy(), points.y.to_numpy()

    def grid(values):
        first = np.floor(values.min() / GRID_M) * GRID_M - SIDE_M
        return np.arange(first, np.ceil(values.max() / GRID_M) * GRID_M + 1, GRID_M)

    best_n, best = 0, None
    for y0 in grid(y):  # south to north, then west to east: a tie keeps the first
        in_row = (y > y0) & (y < y0 + SIDE_M)
        if not in_row.any():
            continue
        for x0 in grid(x):
            corner = (int(x0), int(y0))
            if exclude is not None and square(corner).distance(exclude) < MIN_GAP_M:
                continue
            n = int((in_row & (x > x0) & (x < x0 + SIDE_M)).sum())
            if n > best_n:
                best_n, best = n, corner
    return best, best_n


def add_strata(reference):
    """Add the strata columns: ``irrigation`` and ``irrigated_<year>`` per imagery year."""
    reference["irrigation"] = reference["ITYPE_2024"].map(IRRIGATION_SYSTEMS)
    for year in sorted(set(SOURCE_YEARS.values())):
        share = reference[f"irrmapper_{year}"]
        reference[f"irrigated_{year}"] = [
            None if math.isnan(v) else ("irrigated" if v >= IRRIGATED_SHARE else "not irrigated")
            for v in share.astype(float)
        ]
    return reference


def prepare_basin(slug, fields):
    """Squares, evaluation reference and training labels of one basin.

    Re-applies the square rule and warns when the inventory gives other squares than
    the recorded ones (``BASINS``), which are used.
    """
    spec = BASINS[slug]
    corners, counts, exclude = {}, {}, None
    for kind in ("evaluation", "training"):
        corner, n = rule_square(fields, spec["basin"], exclude)
        counts[kind] = n
        if corner != spec[kind]:
            print(
                f"  WARNING: the rule now gives the {kind} square at {corner} ({n} fields), "
                f"not the recorded {spec[kind]}; the recorded square is used."
            )
            corner = spec[kind]
        corners[kind] = corner
        exclude = square(corner)  # the training square keeps its distance from this one
    evaluation, training = square(corners["evaluation"]), square(corners["training"])
    dri = rep_point_inside(fields[(fields["SOURCECODE"] == "DRI").to_numpy()], evaluation)
    # The predictions' minimum area applies to the reference as well.
    small = (dri.area < MIN_AREA).to_numpy()
    reference = add_strata(dri[~small]).reset_index(drop=True)
    # Training labels: every field of the inventory that reaches into the training square,
    # whatever its size, because the chips label every other pixel as background.
    labels = fields[fields.intersects(training).to_numpy()].reset_index(drop=True)
    if labels.intersects(evaluation).any():
        raise ValueError(f"{slug}: a training field reaches into the evaluation square")
    return {
        "corners": corners,
        "evaluation": evaluation,
        "training": training,
        "reference": reference,
        "n_small": int(small.sum()),
        "labels": labels,
        "training_fields": len(rep_point_inside(rule_fields(fields, spec["basin"]), training)),
        "rule_counts": counts,
    }


def describe_basin(slug, basin):
    """Print a basin's squares, reference and training labels."""
    spec = BASINS[slug]
    ref, labels = basin["reference"], basin["labels"]
    print(f"\n{spec['name']}")
    for kind in ("evaluation", "training"):
        area = basin[kind]
        lon, lat = lon_lat(area)
        x, y = basin["corners"][kind]
        print(
            f"  {kind:<10} square: corner ({x}, {y}) UTM 11N, centre {lon:.5f}, {lat:.5f}; "
            f"rule count {basin['rule_counts'][kind]}"
        )
    gap = basin["training"].distance(basin["evaluation"])
    centres = basin["training"].centroid.distance(basin["evaluation"].centroid)
    print(
        f"  training square {gap / 1000:.1f} km from the evaluation square edge to edge "
        f"(centres {centres / 1000:.1f} km apart)"
    )
    area_ha = ref.area / 1e4
    bins = list(zip(SIZE_BINS_HA[:-1], SIZE_BINS_HA[1:], strict=True))
    sizes = {f"{lo:g}-{hi:g}": int(((area_ha >= lo) & (area_ha < hi)).sum()) for lo, hi in bins}
    systems = ref["irrigation"].fillna("(none)").value_counts().to_dict()
    print(
        f"  reference: {len(ref)} fields (median {area_ha.median():.1f} ha; basins "
        f"{ref['Basin'].value_counts().to_dict()}); {basin['n_small']} smaller DRI fields "
        f"(under {MIN_AREA:,} m2, the predictions' minimum area) dropped"
    )
    print(f"  irrigation systems (ITYPE_2024): {systems}; size classes (ha) {sizes}")
    irrigated = {
        year: int((ref[f"irrigated_{year}"] == "irrigated").sum())
        for year in sorted(set(SOURCE_YEARS.values()))
    }
    print(f"  irrigated (IrrMapper >= {IRRIGATED_SHARE:g}) by year: {irrigated} of {len(ref)}")
    differs = {
        year: int((ref[f"ITYPE_{year}"].fillna(-1) != ref["ITYPE_2024"].fillna(-1)).sum())
        for year in (2022, 2023)
    }
    print(f"  fields whose ITYPE differs from ITYPE_2024: {differs}")
    n_split = int(ref["NWI_ID_PARENT"].notna().sum())
    print(f"  fields split from oversized polygons (NWI_ID_PARENT set): {n_split}")
    training = rep_point_inside(labels[(labels.area >= MIN_AREA).to_numpy()], basin["training"])
    print(
        f"  training labels: {len(labels)} fields reach into the training square "
        f"({basin['training_fields']} counted by the rule; median "
        f"{training.area.median() / 1e4:.1f} ha by representative point); providers "
        f"{labels['SOURCECODE'].value_counts().to_dict()}; split from oversized polygons "
        f"{int(labels['NWI_ID_PARENT'].notna().sum())}"
    )


def reference_path(gdf, slug, label):
    """File of a reference layer, named with a hash of its geometry and strata columns.

    Outputs that were evaluated against, or fine-tuned on, a reference are reused only
    when the reference path is unchanged, so a changed reference gets a new path.
    """
    import shapely

    digest = hashlib.sha256(
        b"".join(shapely.to_wkb(gdf.geometry.to_numpy(), hex=False, output_dimension=2))
    )
    for column in ["irrigation", *(c for c in gdf.columns if c.startswith("irrigated_"))]:
        if column in gdf.columns:
            digest.update("|".join(gdf[column].astype(str)).encode())
    return REFERENCE_DIR / f"{slug}_{label}_{digest.hexdigest()[:12]}.gpkg"


def save_reference(gdf, slug, label):
    """Write a reference layer once (its modification time is part of fine-tuning cache keys)."""
    path = reference_path(gdf, slug, label)
    if not path.exists():
        gdf.to_file(path, layer="reference")
    return path


def merge_split_rates(pred, ref):
    """Shares of reference fields that are merged with another or split (see the docstring)."""
    import numpy as np

    if len(pred) == 0 or len(ref) == 0:
        return float("nan"), float("nan")
    ref = ref.reset_index(drop=True)
    pred = pred.to_crs(ref.crs).reset_index(drop=True)
    r_idx, p_idx = ref.sindex.query(pred.geometry, predicate="intersects")[::-1]
    shared = ref.geometry.iloc[r_idx].intersection(pred.geometry.iloc[p_idx], align=False).area
    shared = np.asarray(shared)
    r_area = ref.area.to_numpy()
    merged = np.zeros(len(ref), bool)
    split = np.zeros(len(ref), bool)
    for r in range(len(ref)):
        mine = np.flatnonzero(r_idx == r)
        if mine.size == 0:
            continue
        split[r] = int((shared[mine] >= SPLIT_SHARE * r_area[r]).sum()) >= 2
        main = p_idx[mine[np.argmax(shared[mine])]]
        others = np.flatnonzero((p_idx == main) & (r_idx != r))
        merged[r] = bool((shared[others] >= MERGE_SHARE * r_area[r_idx[others]]).any())
    return float(merged.mean()), float(split.mean())


def evaluate_run(pred, ref, year, args):
    """Headline and per-stratum metrics of one layer (a dict) and the full metrics."""
    from agribound.evaluate import evaluate

    m = evaluate(
        pred,
        ref,
        iou_threshold=0.5,
        strata="irrigation",
        size_bins=SIZE_BINS_HA,
        bootstrap=args.bootstrap,
        bootstrap_seed=42,
        boundary_tolerance_m=BOUNDARY_TOLERANCE_M,
    )
    # Irrigated or not in the imagery year (IrrMapper): no intervals, no boundary metrics.
    status = evaluate(pred, ref, iou_threshold=0.5, strata=f"irrigated_{year}")
    merged, split = merge_split_rates(pred, ref)
    ci = (m.get("bootstrap") or {}).get("ci", {})
    row = {
        "fields": len(pred),
        "precision": m["precision"],
        "recall": m["recall"],
        "recall_ci_low": (ci.get("recall") or [None, None])[0],
        "recall_ci_high": (ci.get("recall") or [None, None])[1],
        "f1": m["f1"],
        "f1_ci_low": (ci.get("f1") or [None, None])[0],
        "f1_ci_high": (ci.get("f1") or [None, None])[1],
        "iou_mean": m["iou_mean"],
        "best_iou_mean": m["best_iou_mean"],
        "boundary_f1": m.get("boundary_f1"),
        "fp_noref": m["count_fp_unassigned"],
        "merged": merged,
        "split": split,
    }
    for name, key in STRATA.items():
        row[f"recall_{key}"] = (m["per_stratum"].get(name) or {}).get("recall")
    for name, part in m["per_size_class"].items():
        row[f"recall_{name}_ha"] = part["recall"]
    # The irrigated / not irrigated strata depend on the imagery year: their sizes per run.
    for name in ("irrigated", "not irrigated"):
        part = status["per_stratum"].get(name) or {}
        row[f"recall_{name.replace(' ', '_')}"] = part.get("recall")
        row[f"n_{name.replace(' ', '_')}"] = part.get("n", 0)
    return row, {"metrics": m, "metrics_irrigated": status}


def irrigated_only(reference, year):
    """The reference fields irrigated in *year* (``irrigated_<year>``) and all the others."""
    irrigated = (reference[f"irrigated_{year}"] == "irrigated").to_numpy()
    return reference[irrigated], reference[~irrigated]


def evaluate_irrigated(pred, ref, year, args):
    """The irrigated-only score of one layer: summary columns (a dict) and its record.

    The reference is the fields of *ref* irrigated in *year*; the others are passed as
    ``ignore``, so a prediction whose representative point lies in one of them is not
    scored. Matching, tolerance, strata and bootstrap as in ``evaluate_run``.
    """
    from agribound.evaluate import evaluate

    irrigated, other = irrigated_only(ref, year)
    m = evaluate(
        pred,
        irrigated,
        iou_threshold=0.5,
        strata="irrigation",
        size_bins=SIZE_BINS_HA,
        bootstrap=args.bootstrap,
        bootstrap_seed=42,
        boundary_tolerance_m=BOUNDARY_TOLERANCE_M,
        ignore=other,
    )
    ci = (m.get("bootstrap") or {}).get("ci", {})
    recall_ci = ci.get("recall") or [None, None]
    f1_ci = ci.get("f1") or [None, None]
    row = {
        "n_reference_irrigated": m["count_reference"],
        "n_predicted_scored": m["count_predicted"],
        "n_predicted_ignored": m["count_predicted_ignored"],
        "precision_irrigated": m["precision"],
        "recall_irrigated_score": m["recall"],
        "recall_irrigated_score_ci_low": recall_ci[0],
        "recall_irrigated_score_ci_high": recall_ci[1],
        "f1_irrigated": m["f1"],
        "f1_irrigated_ci_low": f1_ci[0],
        "f1_irrigated_ci_high": f1_ci[1],
        "iou_mean_irrigated": m["iou_mean"],
        "boundary_f1_irrigated": m.get("boundary_f1"),
        "fp_noref_irrigated": m["count_fp_unassigned"],
    }
    score = (
        f"irrigated only: reference = the {len(irrigated)} reference fields irrigated in {year} "
        f"(irrmapper_{year} >= {IRRIGATED_SHARE:g}); ignore = the other {len(other)} reference "
        "fields (a prediction whose representative point lies in one is not scored); every "
        "other prediction is scored"
    )
    return row, {"score": score, "metrics": m}


def crop_filter_layer(fields, out_path, config, args, year=None):
    """The crop-filter step: the default rule (Annual NLCD in the US) on one layer.

    The filtered layer is named after *out_path*, with *year* (the land-cover year)
    added when the name of *out_path* does not carry it.
    """
    import geopandas as gpd

    from agribound.postprocess.lulc_filter import filter_by_lulc

    # The threshold is in the name: a layer filtered at another threshold is not reused.
    stem = out_path.stem if year is None else f"{out_path.stem}_{year}"
    path = out_path.with_name(f"{stem}_crop_t{CROP_THRESHOLD:g}.gpkg")
    stale = path.exists() and path.stat().st_mtime < out_path.stat().st_mtime
    if path.exists() and not stale and not args.overwrite:
        return gpd.read_file(path)
    cfg = config.merged(lulc_filter=True, lulc_crop_threshold=CROP_THRESHOLD)
    kept = filter_by_lulc(fields, cfg) if len(fields) else fields
    kept.to_file(path, layer="fields")
    return kept


def matched_kept(fields, kept, frame):
    """Share of the matched predictions (IoU >= 0.5) that a filtered layer keeps."""
    ids = set(frame.loc[frame["matched"], "pred_index"].dropna())
    if not ids:
        return float("nan")
    kept_ids = set(kept["id"]) if "id" in kept.columns else set()
    return len({fields.loc[i, "id"] for i in ids} & kept_ids) / len(ids)


def run_config(out_dir, source, engine, label, args, **overrides):
    """AgriboundConfig of one run: the source's year and window, an output named after both."""
    from agribound.config import AgriboundConfig

    year = SOURCE_YEARS[source]
    settings = {}
    if source in SEASONAL_SOURCES and engine != "ftw":  # FTW builds its own two windows
        settings["date_range"] = (f"{year}-{SEASON[0]}", f"{year}-{SEASON[1]}")
    if source == "spot":
        settings["cloud_cover_max"] = SPOT_CLOUD_COVER_MAX
    return AgriboundConfig(
        study_area=str(out_dir / "study_area.geojson"),
        source=source,
        year=year,
        engine=engine,
        output_path=str(out_dir / f"fields_{source}_{label}_{year}.gpkg"),
        gee_project=args.gee_project,
        lulc_filter=False,
        overwrite=args.overwrite,
        **{**settings, **overrides},
    )


def fine_tune_config(slug, source, engine, labels_path, args):
    """AgriboundConfig that fine-tunes *engine* on *source* of a basin's training square."""
    label = "da" if engine == "delineate-anything" else engine
    return run_config(
        OUTPUT_DIR / f"{slug}_train",
        source,
        engine,
        f"{label}-finetune",
        args,
        reference_boundaries=str(labels_path),
        fine_tune=True,
        fine_tune_epochs=FINE_TUNE_EPOCHS,
        fine_tune_split="random",
        fine_tune_val_split=0.2,
        engine_params=dict(FINE_TUNE_PARAMS[engine]),
    )


def fine_tune_on(slug, source, engine, labels_path, args):
    """Fine-tune *engine* on *source* of a basin's training square; return the checkpoint.

    Both engines get the same labels, composite, epochs and validation share, each on
    its own default chips; the ``yolo_*`` settings apply to Delineate-Anything only.
    """
    from agribound.provenance import read_provenance

    config = fine_tune_config(slug, source, engine, labels_path, args)
    print(
        f"\n{'=' * 70}\n{BASINS[slug]['name']}, training square: fine-tuning {engine} on "
        f"{source} {SOURCE_YEARS[source]}\n{'=' * 70}"
    )
    agribound.delineate(config=config)  # in-sample output: never evaluated
    record = read_provenance(config.output_path) or {}
    return (record.get("facts") or {}).get("fine_tuned_checkpoint")


def basin_runs(slug, args, checkpoints):
    """(label, source, engine, config overrides) of every evaluated run of one basin."""
    da, many = "delineate-anything", {"engine_params": {"max_detections": 1000}}
    runs = [("da", "naip", da, {})]
    if not args.no_sam:
        runs.append(("da-sam2", "naip", da, {"sam_refine": True}))
    for source in FINE_TUNE_SOURCES:
        if source == "spot":
            if not args.spot:
                continue
            runs.append(("da", "spot", da, {}))
        for label, engine in (("da-ft", da), ("dinov3-ft", "dinov3")):
            checkpoint = checkpoints.get((slug, source, engine))
            if checkpoint:
                runs.append(
                    (label, source, engine, {"engine_params": {"checkpoint_path": checkpoint}})
                )
    # At 10-30 m one 256 px tile covers much of a square: up to 1,000 detections per tile.
    for source in ("sentinel2", "landsat", "landsat-pan", "sentinel1"):
        runs.append(("da", source, da, many))
        if source == "sentinel2":
            runs.append(("ftw", "sentinel2", "ftw", {}))
    k = {"engine_params": {"n_clusters": N_CLUSTERS}}
    runs.append((f"k{N_CLUSTERS}", "google-embedding", "embedding", k))
    runs.append((f"k{N_CLUSTERS}", "tessera-embedding", "embedding", k))
    if args.sam_auto:  # opt-in: the engine's own per-source defaults, no engine_params
        for source in SAM_AUTO_SOURCES:
            if source != "spot" or args.spot:
                runs.append(("sam-auto", source, "sam-auto", {}))
    if args.options:  # opt-in: the "-opt" runs, each with its base run's settings and models
        bases = {f"{run[1]} {run[0]}": run for run in runs}
        for name in OPTION_BASES[slug]:
            if name not in bases:  # its base run is skipped (no SPOT access, no checkpoint)
                continue
            label, source, engine, over = bases[name]
            params = {**over.get("engine_params", {}), **option_params(source, engine)}
            runs.append((f"{label}-opt", source, engine, {**over, "engine_params": params}))
    return runs


def option_params(source, engine):
    """The engine_params an "-opt" run adds to those of its base run.

    In the pre-registered order: inside the engine, the 3 m grid of Delineate-Anything on
    NAIP and DINOv3's half-band growth; then the pivot snap, the pipeline's last geometric
    step. The crop filter, this script's separate step, is not an engine parameter
    (``option_crop``).
    """
    params = {}
    if engine == "delineate-anything" and source == "naip":
        params["processing_gsd_m"] = OPTION_GSD_M
    if engine == "dinov3":
        params["dilate_interior_px"] = OPTION_GROWTH
    params["pivots"] = OPTION_PIVOTS
    return params


def option_crop(source):
    """True when the result of an "-opt" run on *source* is its crop-filter layer."""
    return source not in OPTION_NO_CROP


def is_option_run(label):
    """True for the label of an "-opt" run (``"da-opt"``)."""
    return label.endswith("-opt")


def option_steps(source, engine):
    """The options of an "-opt" run, in their order (for the record and options.csv)."""
    steps = [f"{key}={value}" for key, value in option_params(source, engine).items()]
    if option_crop(source):
        steps.append(f"crop filter (threshold {CROP_THRESHOLD:g})")
    return steps


def option_change(result, base, reference, year):
    """The paired change of an "-opt" run's result against its base run's output.

    ``agribound.evaluate.paired_bootstrap`` of the two layers (result minus base), with the
    main score's matching and boundary tolerance, over N_PAIRED resamples (seed 42) of the
    square's BLOCK_M blocks of UTM zone 11N: ``"all"`` against every reference field and
    ``"irrigated"`` against the fields irrigated in *year*, the others ignored, as in the
    irrigated-only score.
    """
    from agribound.evaluate import paired_bootstrap

    common = {
        "boundary_tolerance_m": BOUNDARY_TOLERANCE_M,
        "n_resamples": N_PAIRED,
        "seed": 42,
        "block_size_m": BLOCK_M,
        "block_crs": UTM_CRS,
    }
    irrigated, other = irrigated_only(reference, year)
    out = {"all": paired_bootstrap(result, base, reference, **common)}
    if len(irrigated):
        ignore = other if len(other) else None
        out["irrigated"] = paired_bootstrap(result, base, irrigated, ignore=ignore, **common)
    return out


def option_row(row, base_row, change, record):
    """The options.csv row of an "-opt" run: its result, its base run's and the changes.

    *row* and *base_row* are the summary rows of the run and of its base run, *change* the
    output of ``option_change`` (or None) and *record* the run's provenance record.
    """
    facts = (record or {}).get("facts") or {}
    meta = (record or {}).get("engine_meta") or {}
    pivots = facts.get("pivots") or {}
    out = {
        "basin": row["basin"],
        "run": row["run"],
        "base": base_row["run"],
        "year": row["year"],
        "options": "; ".join(option_steps(row["source"], row["engine"])),
        "result": "crop-filter layer" if option_crop(row["source"]) else "output",
        "n_output": facts.get("n_output"),
        "n_result": row["fields"],
        "growth_px": meta.get("dilate_interior_px") if meta.get("dilate_interior_rule") else None,
        "band_px": meta.get("band_thickness_px"),
        "pivot_groups": pivots.get("n_groups"),
        "pivot_groups_multi": pivots.get("n_groups_multi"),
        "pivot_pieces_merged": pivots.get("n_pieces_merged"),
    }
    for key, part in (("f1", "all"), ("f1_irrigated", "irrigated"), ("boundary_f1", "all")):
        metric = "f1" if key == "f1_irrigated" else key
        paired = (change or {}).get(part) or {}
        low, high = ((paired.get("ci") or {}).get(metric) or [None, None])[:2]
        out[f"{key}_base"] = base_row.get(key)
        out[key] = row.get(key)
        out[f"d_{key}"] = (paired.get("difference") or {}).get(metric)
        out[f"d_{key}_ci_low"], out[f"d_{key}_ci_high"] = low, high
    out["n_blocks"] = ((change or {}).get("all") or {}).get("n_blocks")
    return out


def fine_tune_plan(args):
    """(source, engine) pairs fine-tuned on each training square."""
    if args.no_fine_tune:
        return []
    engines = ["delineate-anything"] + ([] if args.no_dinov3 else ["dinov3"])
    sources = [s for s in FINE_TUNE_SOURCES if s != "spot" or args.spot]
    return [(source, engine) for source in sources for engine in engines]


def dry_run(slugs, basins, args):
    """Print every run's configuration (validated, nothing run or written)."""
    print(f"\n{'=' * 70}\nDry run: configurations (nothing is run or written)\n{'=' * 70}")

    def show(config, crop=False):
        if config.date_range:
            window = " to ".join(config.date_range)
        else:
            window = "FTW's two windows" if config.engine == "ftw" else "calendar year"
        notes = [f"fine-tuning on {config.reference_boundaries}"] if config.fine_tune else []
        notes += ["SAM 2"] if config.sam_refine else []
        notes += [str(config.engine_params.get("checkpoint_path", ""))]
        # The options of an "-opt" run (no other run sets these keys).
        notes += [
            f"{key}={config.engine_params[key]}"
            for key in ("processing_gsd_m", "dilate_interior_px", "pivots")
            if key in config.engine_params
        ]
        notes += ["then the crop filter, whose layer is scored"] if crop else []
        note = "; ".join(n for n in notes if n)
        print(
            f"  {config.source:<17} {config.engine:<18} {config.year} {window:<24} "
            f"{config.output_path}" + (f"\n{'':<64}({note})" if note else "")
        )

    for slug in slugs:
        print(f"\n{BASINS[slug]['name']}")
        labels_path = reference_path(basins[slug]["labels"], f"{slug}_train", "labels")
        checkpoints = {}
        for source, engine in fine_tune_plan(args):
            show(fine_tune_config(slug, source, engine, labels_path, args))
            checkpoints[slug, source, engine] = f"checkpoint fine-tuned on {source} of {slug}_train"
        for label, source, engine, overrides in basin_runs(slug, args, checkpoints):
            config = run_config(OUTPUT_DIR / slug, source, engine, label, args, **overrides)
            show(config, crop=is_option_run(label) and option_crop(source))


def spot_available(args):
    """True when the Earth Engine project can read AIRBUS/SPOT6_7."""
    try:
        import ee

        from agribound.auth import setup_gee

        setup_gee(project=args.gee_project)
        ee.ImageCollection("AIRBUS/SPOT6_7").limit(1).size().getInfo()
        return True
    except Exception as exc:
        print(f"No SPOT 6/7 access ({type(exc).__name__}): SPOT runs are skipped.")
        return False


def fmt(value, digits=2):
    """Format a metric (NaN-safe)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if math.isnan(value) else f"{value:.{digits}f}"


def print_tables(slug, rows, reference):
    """The basin's headline table and its recall per stratum."""
    done = [row for row in rows if row["basin"] == slug and row["status"] == "ok"]
    print(
        f"\n  {'Run':<22} {'Year':>4} {'Fields':>6} {'P':>5} {'R':>5} {'F1':>5} {'IoU':>5} "
        f"{'bF1':>5} {'Merge':>5} {'Split':>5} {'FP-noref':>8} {'Kept':>5}"
    )
    for row in done:
        print(
            f"  {row['run']:<22} {row['year']:>4} {row['fields']:>6} {fmt(row['precision']):>5} "
            f"{fmt(row['recall']):>5} {fmt(row['f1']):>5} {fmt(row['iou_mean']):>5} "
            f"{fmt(row['boundary_f1']):>5} {fmt(row['merged']):>5} {fmt(row['split']):>5} "
            f"{row['fp_noref']:>8} {row['kept_crop']:>5}"
        )
    for row in rows:
        if row["basin"] == slug and row["status"] != "ok":
            print(f"  {row['run']:<22} {row['year']:>4} {row['status']}")

    # Recall per stratum. The irrigated / not irrigated strata depend on the imagery year,
    # so the number of reference fields of every stratum is printed once per imagery year.
    area_ha = reference.area / 1e4
    present = set(reference["irrigation"].dropna())
    systems = [(name, key) for name, key in STRATA.items() if name in present]
    bins = list(zip(SIZE_BINS_HA[:-1], SIZE_BINS_HA[1:], strict=True))
    headers = [key for _, key in systems]
    headers += [f"{lo:g}-{hi:g}".replace("-inf", "+") for lo, hi in bins]
    headers += ["irrigated", "not irr."]
    keys = [f"recall_{key}" for _, key in systems]
    keys += [f"recall_{lo:g}-{hi:g}_ha" for lo, hi in bins]
    keys += ["recall_irrigated", "recall_not_irrigated"]
    fixed = [int((reference["irrigation"] == name).sum()) for name, _ in systems]
    fixed += [int(((area_ha >= lo) & (area_ha < hi)).sum()) for lo, hi in bins]
    widths = [max(len(header), 5) for header in headers]

    def line(first, year, cells):
        return f"  {first:<22} {year:>4}" + "".join(
            f" {cell:>{width}}" for cell, width in zip(cells, widths, strict=True)
        )

    print(
        "\n  Recall per stratum: irrigation system (ITYPE_2024), field size (ha) and "
        "irrigated or not in the year of the imagery (IrrMapper)\n  flood_ctl: controlled "
        "flood; flood_unc_none: uncontrolled flood or no apparent irrigation equipment "
        f"(ITYPE 3); irrigated: IrrMapper share >= {IRRIGATED_SHARE:g} in the imagery year"
    )
    print(line("Run", "Year", headers))
    for year in sorted({row["year"] for row in done}):
        status = [
            int((reference[f"irrigated_{year}"] == name).sum())
            for name in ("irrigated", "not irrigated")
        ]
        print(line("(reference fields)", year, fixed + status))
    for row in done:
        print(line(row["run"], row["year"], [fmt(row.get(key)) for key in keys]))


def print_irrigated_table(slug, rows):
    """The basin's irrigated-only scores (see the docstring)."""
    done = [row for row in rows if row["basin"] == slug and row["status"] == "ok"]
    print(
        "\n  Irrigated-only score: reference = the fields irrigated in the year of the imagery "
        f"(IrrMapper share >= {IRRIGATED_SHARE:g}); a prediction\n  whose representative point "
        "lies in another reference field is ignored (Ignored), every other one is scored "
        "(Scored)"
    )
    print(
        f"  {'Run':<22} {'Year':>4} {'Ref':>4} {'Scored':>6} {'Ignored':>7} {'P':>5} {'R':>5} "
        f"{'F1':>5} {'IoU':>5} {'bF1':>5} {'FP-noref':>8}"
    )
    for row in done:
        print(
            f"  {row['run']:<22} {row['year']:>4} {row['n_reference_irrigated']:>4} "
            f"{row['n_predicted_scored']:>6} {row['n_predicted_ignored']:>7} "
            f"{fmt(row['precision_irrigated']):>5} {fmt(row['recall_irrigated_score']):>5} "
            f"{fmt(row['f1_irrigated']):>5} {fmt(row['iou_mean_irrigated']):>5} "
            f"{fmt(row['boundary_f1_irrigated']):>5} {row['fp_noref_irrigated']:>8}"
        )


def print_option_table(slug, option_rows):
    """The basin's "-opt" runs: each one's change against its base run (see the docstring)."""
    done = [row for row in option_rows if row["basin"] == slug]
    if not done:
        return

    def change(row, key):
        low, high = row[f"d_{key}_ci_low"], row[f"d_{key}_ci_high"]
        if row[f"d_{key}"] is None or low is None or high is None:
            return "-"
        return f"{row[f'd_{key}']:+.3f} [{low:+.3f}, {high:+.3f}]"

    print(
        f'\n  "-opt" runs: change against the base run (paired bootstrap of 1 km blocks, '
        f"{N_PAIRED:,} resamples; 95 % intervals); reported results, not tests. Result: the "
        "crop-filter layer\n  of an optical run, the output of a Sentinel-1 run; F1: base run "
        "-> result"
    )
    print(
        f"  {'Run':<22} {'Year':>4} {'Fields':>6} {'F1':>14} {'Change in F1':>23} "
        f"{'Change, irrigated only':>23} {'Change in bF1':>23}"
    )
    for row in done:
        f1 = f"{fmt(row['f1_base'], 3)} -> {fmt(row['f1'], 3)}"
        print(
            f"  {row['run']:<22} {row['year']:>4} {row['n_result']:>6} {f1:>14} "
            f"{change(row, 'f1'):>23} {change(row, 'f1_irrigated'):>23} "
            f"{change(row, 'boundary_f1'):>23}"
        )


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(
        description=f"Nevada: Diamond Valley and Mason Valley against the {REFERENCE_LABEL}."
    )
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project ID (default: $GEE_PROJECT, then the gcloud project, then "
            "the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    parser.add_argument(
        "--reference",
        default=REFERENCE,
        help="The DRI field boundaries (a file in a zip archive as zip://ARCHIVE!MEMBER).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="BASIN",
        choices=list(BASINS),
        help=f"Run only these basins ({', '.join(BASINS)}).",
    )
    parser.add_argument("--no-spot", action="store_true", help="Skip the SPOT 6/7 runs.")
    parser.add_argument("--no-sam", action="store_true", help="Skip SAM 2 refinement.")
    parser.add_argument(
        "--sam-auto",
        action="store_true",
        help="Also run the direct SAM engine (engine sam-auto, SAM 2.1 automatic masks) with "
        "its per-source defaults on NAIP, SPOT and Landsat, the imagery types its "
        "pre-registered evaluation kept (needs the sam-auto extra).",
    )
    parser.add_argument(
        "--options",
        action="store_true",
        help="Also make the '-opt' runs: four runs per basin again with the opt-in options of "
        "agribound 1.1.0 that passed the pre-registered test of the Diamond Valley options (3 m "
        "grid, half-band growth, pivot snap, crop filter, each where it applies), run on three "
        "pivot squares in other Nevada basins (the pivot snap also on both evaluation squares, "
        "the 3 m grid also in Mason Valley; see the docstring).",
    )
    parser.add_argument(
        "--no-fine-tune",
        action="store_true",
        help="Skip the fine-tuning (Delineate-Anything and DINOv3).",
    )
    parser.add_argument(
        "--no-dinov3", action="store_true", help="Skip the fine-tuned DINOv3 (the slowest step)."
    )
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP, help="Bootstrap resamples.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute outputs that already exist."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the squares, the reference counts and every run's configuration; run "
        "and write nothing (no Earth Engine).",
    )
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def main():
    import geopandas as gpd
    import pandas as pd

    from agribound.evaluate import evaluate_frame
    from agribound.provenance import read_provenance

    args = parse_args()
    start_time = time.time()
    if not reference_exists(args.reference):
        raise SystemExit(
            f"Reference layer not found: {args.reference}\nThis example needs the "
            f"{REFERENCE_LABEL} at {REFERENCE_ZIP} (local data, not in this repository)."
        )
    slugs = args.only or list(BASINS)
    print(f"Reading the {REFERENCE_LABEL}: {args.reference}")
    fields = load_inventory(args.reference)
    basins = {slug: prepare_basin(slug, fields) for slug in slugs}
    for slug in slugs:
        describe_basin(slug, basins[slug])
    print("\nYears: " + ", ".join(f"{source} {year}" for source, year in SOURCE_YEARS.items()))
    if args.dry_run:
        args.spot = not args.no_spot  # access is not checked in a dry run
        dry_run(slugs, basins, args)
        return

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    args.spot = not args.no_spot and spot_available(args)
    label_paths = {}
    for slug in slugs:
        basin, name = basins[slug], BASINS[slug]["name"]
        for out_dir, area, title in (
            (OUTPUT_DIR / slug, basin["evaluation"], name),
            (OUTPUT_DIR / f"{slug}_train", basin["training"], f"{name} (training only)"),
        ):
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "study_area.geojson").write_text(json.dumps(square_study_area(area, title)))
        label_paths[slug] = save_reference(basin["labels"], f"{slug}_train", "labels")

    checkpoints = {}  # (basin, source, engine) -> fine-tuned checkpoint
    failed_fine_tunes = {}  # (basin, source, engine) -> exception name
    for slug in slugs:
        for source, engine in fine_tune_plan(args):
            key = (slug, source, engine)
            what = f"Fine-tuning {engine} on {source} of {slug}_train"
            try:
                checkpoint = fine_tune_on(slug, source, engine, label_paths[slug], args)
            except Exception as exc:
                failed_fine_tunes[key] = type(exc).__name__
                print(f"  {what} failed: {type(exc).__name__}: {exc}")
                continue
            if checkpoint:
                checkpoints[key] = checkpoint
            else:
                failed_fine_tunes[key] = "no checkpoint recorded"
                print(f"  {what} recorded no checkpoint.")

    rows = []
    option_rows = []  # the "-opt" runs' changes against their base runs (options.csv)
    for slug in slugs:
        basin = basins[slug]
        reference = basin["reference"]
        out_dir = OUTPUT_DIR / slug
        print(f"\n{'=' * 70}\n{BASINS[slug]['name']}\n{'=' * 70}")
        ref_path = save_reference(reference, slug, "reference")
        print(
            f"  Reference: {len(reference)} fields ({basin['n_small']} under {MIN_AREA:,} m2 "
            f"dropped) -> {ref_path}"
        )
        # How many reference fields the default crop filter would keep (NLCD of 2024).
        ref_cfg = run_config(out_dir, "sentinel2", "delineate-anything", "reference", args)
        ref_kept = crop_filter_layer(reference, ref_path, ref_cfg, args, year=ref_cfg.year)

        for (key_slug, source, engine), failure in failed_fine_tunes.items():
            if key_slug == slug:  # the run is missing because its fine-tuning failed
                label = "da-ft" if engine == "delineate-anything" else "dinov3-ft"
                missing = [label]
                if args.options and f"{source} {label}" in OPTION_BASES[slug]:
                    missing.append(f"{label}-opt")  # its "-opt" run needs the same model
                for run_label in missing:
                    rows.append(
                        {
                            "basin": slug,
                            "run": f"{source} {run_label}",
                            "source": source,
                            "engine": engine,
                            "year": SOURCE_YEARS[source],
                            "status": f"failed: {failure}",
                        }
                    )
        outputs = {}  # run name -> (its summary row, its output): the "-opt" runs' bases
        for label, source, engine, overrides in basin_runs(slug, args, checkpoints):
            name, year = f"{source} {label}", SOURCE_YEARS[source]
            info = {"basin": slug, "run": name, "source": source, "engine": engine, "year": year}
            try:
                config = run_config(out_dir, source, engine, label, args, **overrides)
                pred = agribound.delineate(config=config).to_crs(UTM_CRS)
            except Exception as exc:  # e.g. no imagery, an engine's package missing
                print(f"  {name} {year} failed: {type(exc).__name__}: {exc}")
                rows.append({**info, "status": f"failed: {type(exc).__name__}"})
                continue
            out_path = Path(config.output_path)
            kept = crop_filter_layer(pred, out_path, config, args)
            option = is_option_run(label)
            # The result of an optical "-opt" run is its crop-filter layer; every other
            # run is scored on its output.
            result = kept if option and option_crop(source) else pred
            row, metrics = evaluate_run(result, reference, year, args)
            frame = evaluate_frame(pred, reference, iou_threshold=0.5)
            row = {
                **info,
                "status": "ok",
                **row,
                "kept_crop": len(kept),
                "matched_kept_crop": matched_kept(pred, kept, frame),
            }
            irrigated_row, irrigated = evaluate_irrigated(result, reference, year, args)
            rows.append({**row, **irrigated_row})  # the irrigated-only columns come last
            if option and result is kept:
                print(
                    f"  {name} {year}: {len(result)} fields (the crop-filter layer of "
                    f"{len(pred)}), F1 {fmt(row['f1'])}"
                )
                crop_layer = out_path.with_name(f"{out_path.stem}_crop_t{CROP_THRESHOLD:g}.gpkg")
                metrics["scored_layer"] = str(crop_layer)
            else:
                print(f"  {name} {year}: {len(pred)} fields, F1 {fmt(row['f1'])}")
            metrics_path = out_dir / f"metrics_{out_path.stem}.json"
            metrics_path.write_text(
                json.dumps(to_jsonable({"row": row, **metrics}), indent=2, allow_nan=False)
            )
            irrigated_path = out_dir / f"metrics_{out_path.stem}_irrigated.json"
            record = {"row": {**info, "status": "ok", **irrigated_row}, **irrigated}
            irrigated_path.write_text(json.dumps(to_jsonable(record), indent=2, allow_nan=False))
            if args.options and name in OPTION_BASES[slug]:
                outputs[name] = ({**row, **irrigated_row}, pred)
            if not option:
                continue
            # The "-opt" run's change against its base run (computed earlier in this loop).
            base_name = f"{source} {label[: -len('-opt')]}"
            base_row, base = outputs.get(base_name, ({"run": base_name}, None))
            change = None
            if base is None:
                print(f"  {name} {year}: no change against {base_name} (the base run failed)")
            else:
                try:
                    change = option_change(result, base, reference, year)
                except Exception as exc:  # reported; the run's own scores stand
                    print(
                        f"  {name} {year}: change against {base_name} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            option_rows.append(
                option_row({**row, **irrigated_row}, base_row, change, read_provenance(out_path))
            )
            change_path = out_dir / f"metrics_{out_path.stem}_change.json"
            change_record = {
                "run": name,
                "base": base_name,
                "result": str(metrics.get("scored_layer", out_path)),
                "options": option_steps(source, engine),
                "rules": (
                    f"agribound.evaluate.paired_bootstrap(result, base, reference): result minus "
                    f"base; {N_PAIRED} resamples of {BLOCK_M:g} m blocks of {UTM_CRS}, seed 42, "
                    f"IoU >= 0.5, boundary tolerance {BOUNDARY_TOLERANCE_M:g} m; 'irrigated': "
                    f"the fields irrigated in {year}, the others ignored"
                ),
                **(change or {}),
            }
            change_path.write_text(
                json.dumps(to_jsonable(change_record), indent=2, allow_nan=False)
            )

        print(
            f"\n  {REFERENCE_LABEL}: {len(reference)} fields in the evaluation square; the "
            f"default crop filter (NLCD {ref_cfg.year}) keeps {len(ref_kept)} of them."
        )
        print_tables(slug, rows, reference)
        print_irrigated_table(slug, rows)
        print_option_table(slug, option_rows)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    if option_rows:
        pd.DataFrame(option_rows).to_csv(OUTPUT_DIR / "options.csv", index=False)
        print(f'"-opt" runs: {OUTPUT_DIR / "options.csv"}')
    print(
        "\nP, R, F1: one-to-one matching at IoU >= 0.5; bF1: boundary F1 within "
        f"{BOUNDARY_TOLERANCE_M:g} m; FP-noref: predictions that overlap no reference field; "
        f"Kept: polygons the default crop filter keeps (threshold {CROP_THRESHOLD:g})."
    )
    print(f"Summary: {OUTPUT_DIR / 'summary.csv'}")

    # Each basin on an interactive map: NAIP, Sentinel-2 and FTW layers and the reference.
    from agribound.visualize import show_comparison

    for slug in slugs:
        out_dir = OUTPUT_DIR / slug
        layers, years = [], set()
        for source, label, text in (
            ("naip", "da", "Delineate-Anything, NAIP"),
            ("naip", "da-ft", "Delineate-Anything fine-tuned, NAIP"),
            ("sentinel2", "da", "Delineate-Anything, Sentinel-2"),
            ("sentinel2", "ftw", "FTW, Sentinel-2"),
        ):
            path = out_dir / f"fields_{source}_{label}_{SOURCE_YEARS[source]}.gpkg"
            if path.exists():
                layers.append((gpd.read_file(path), f"{text} {SOURCE_YEARS[source]}"))
                years.add(SOURCE_YEARS[source])
        if not layers:
            continue
        span = f"{min(years)}-{max(years)}" if len(years) > 1 else str(min(years))
        map_path = out_dir / f"comparison_{span}.html"
        web_map = show_comparison(
            [frame for frame, _ in layers] + [basins[slug]["reference"]],
            labels=[text for _, text in layers] + [REFERENCE_LABEL],
            output_html=str(map_path),
        )
        show_in_notebook(web_map)
        print(f"Map: {map_path}")
    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
