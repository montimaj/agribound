"""
28 — New York: Muck Strips, Dairy Rotations and Hay against the OpenET Field Boundaries (2021-2024)

How agribound does in three New York farming landscapes, evaluated against
the OpenET agricultural field boundaries, version 2.1.1 (Desert Research
Institute for the OpenET Consortium), whose New York fields are all Common
Land Units (CLU) of the USDA Farm Service Agency. It follows the held-out
protocol of example 24: one 5 km evaluation square per region, chosen by a
fixed rule, and fine-tuned models that learn from a separate square of the
same county and never see the evaluated fields.

    1. Black Dirt (R1, Orange County): onions, other vegetables and sod on
       drained organic muck, in strips 30-40 m wide between ditches. Its
       evaluation square is example 25's New York square.
    2. Finger Lakes (R2, Cayuga County): a dairy rotation of corn, alfalfa
       and soybeans on till plain and low hills, with hedgerows, tree lines
       and some contour strips.
    3. North Country (R3, Jefferson County): corn, hay and pasture on the
       Lake Ontario plain, with the largest fields of the three.

What it shows: how far the released models (Delineate-Anything v2 and FTW),
Delineate-Anything with SAM 2 refinement, the direct SAM engine (sam-auto)
and Delineate-Anything and DINOv3 fine-tuned on one square of the county
agree with the CLUs of these squares, beside two image-blind chance
baselines, with merge and split rates and recall by land use, strip shape
and field size. What it does not show: an accuracy for the counties or the
state (one square per region); agreement with the fields that the imagery
shows (a CLU is a farm-programme unit, see below); and anything about Long
Island (dropped by a rule fixed before any model ran, below). No setting was
chosen on a New York square.

Data. The OpenET field boundaries v2.1.1 are not public yet (they are to be
released with their data descriptor), so they are not in this repository and
the script reads a local copy, as example 25 does: ``--openet-dir``, the
``OFB_DATA_DIR`` environment variable or ``OPENET_DIR`` (in the notebook)
names the folder that holds ``NY.zip`` (EPSG:4326; ``NY_README.md``
describes its attributes). The file is read in place with GDAL's
``/vsizip/``, one bounding box per region; nothing is extracted or copied.
Data terms: the file's README says that the licence of this release is not
final ("contact the data steward before redistributing") and that the source
agency's data remain under its terms. On 2026-10-12 the author approved
showing these New York outlines (all CLUs) in figures, with the attribution
below; the file itself is not redistributed. The script writes the outlines
it evaluates and trains on only under ``outputs/``, which git does not track.
Attribution: OpenET agricultural field boundaries, version 2.1.1, Desert
Research Institute for the OpenET Consortium (2026), with the field
boundaries of their source agency: Common Land Unit (CLU), USDA Farm Service
Agency.

What a CLU is. FSA defines a CLU as the smallest unit of land that has a
permanent, contiguous boundary, a common land cover and land management, a
common owner and a common producer; its boundaries follow permanent features
such as fences, roads, waterways and tree lines. In a pilot on the training
squares only (CLU lines against the NAIP 2022 edges, 2026-10-12), the lines
followed tree lines, hedgerows, roads, streams and the ditches of the muck
(1.3-1.7 times the edge strength of the same lines shifted by 30 m, with
offsets of 1-4 m); strips were mapped strip by strip (30-37 m wide); some
uniform fields were cut by ownership or tract lines; CLUs covered house lots,
lawns, paddocks, a solar array, a motocross track and old fields grown into
shrub and young forest; and some hay fields and pastures had none. Of the
farmed area that CDL 2024 maps in the evaluation squares, 83.5-89.3 % lies
in a CLU. So expect low object scores where strips or ownership lines split a
visible field, and false positives on farmland that the file does not map,
and read the merge rates, the strip-like stratum and the chance baselines
with F1. The geometry of every evaluated field is that of the file's
delivery of 2021-04-21 (``MOD_DATE``), so fields that changed before the
imagery year count as errors.

Study squares, chosen by a fixed rule from the field boundaries and the 2024
USDA Cropland Data Layer (CDL) alone, before any model ran on them
(``rule_square``):

    - candidates: the cells of the 5 km grid of UTM zone 18N (corners at
      multiples of 5,000 m, example 25's grid) whose centre lies in the zone
      and which lie entirely inside the county (U.S. Census Bureau 2023
      TIGER/Line outline, which includes the county's water);
    - the evaluation square is the candidate with the most farmed fields:
      fields of the file of at least 2,500 m2 (EPSG:6933) whose
      representative point lies in the cell and on a pixel that CDL 2024
      maps as a crop, hay or pasture class (example 25's ``FARMED_CDL``);
      ties go to the southernmost, then the westernmost cell centre;
    - the training square is chosen the same way among the candidates at
      least 5 km from the evaluation square, edge to edge (example 24's
      rule);
    - a region is kept only if at least half of the farmed CDL 2024 pixels
      (10 m) of its evaluation square lie in a field of the file (rule 1d).
      Long Island (Suffolk County) failed it (0.42) and was dropped: correct
      outlines of its vineyards, sod and nurseries would count as false
      positives. Steuben County passed and is left out (its 2022 NAIP is
      from November).

R1's rule square (A0) is pasture and young forest, not muck: the three
densest cells of Orange County are the evaluation square and two of its
neighbours, and every cell at least 5 km away holds 230 or fewer farmed
fields. R1 therefore has two fine-tuning arms (the author's decision of
2026-10-12). A0, the rule's square, is the primary, held-out arm. A1, the
muck around Pine Island, is a second arm, chosen by a rule fixed before it
was computed: the 5 km square on the 1 km grid inside Orange County with the
most farmed fields, at least 1 km from the evaluation square. A1 lies 1 km
from the evaluation square and was flown on the same day (NAIP, 4 July
2022), so its results are labelled near-sample: no field or pixel is shared
(the script checks), but they are not a held-out result.

The squares (``REGIONS``; lower-left corners in UTM zone 18N, EPSG:32618;
Fields: the reference fields of at least 2,500 m2 by representative point,
with those on a farmed CDL 2024 pixel in brackets; Strips: fields with
``PP_SCORE`` < 0.3; Mapped: the share of rule 1d; ``--select`` re-applies
the rules and reports whether they give these squares, for example with
another release of the file or of CDL):

    Square         Corner (m)       Centre (lon, lat)  Fields     Median   Strips      Mapped
    R1 evaluation  545000, 4575000  -74.4322, 41.3476  668 (617)  1.70 ha  305 (46 %)  0.835
    R1 A0          535000, 4575000  -74.5517, 41.3481  284 (230)  1.69 ha   27 (10 %)  0.652
    R1 A1          539000, 4571000  -74.5042, 41.3119  626 (554)  1.64 ha  177 (28 %)  0.839
    R2 evaluation  365000, 4745000  -76.6221, 42.8686  482 (448)  2.49 ha  105 (22 %)  0.893
    R2 training    370000, 4725000  -76.5564, 42.6894  449 (432)  2.45 ha  144 (32 %)  0.839
    R3 evaluation  410000, 4845000  -76.0873, 43.7754  281 (259)  3.98 ha   29 (10 %)  0.887
    R3 training    415000, 4860000  -76.0275, 43.9110  257 (242)  3.99 ha   30 (12 %)  0.802

NAIP 2022 was flown on 4 July over R1 (A0 also on 14 and 22 October, A1
also on 22 October), on 15 July over R2 (its training square also on 21
October) and on 14 October over R3. The training squares lie 5.0 km (A0,
west), 1.0 km (A1, south-west), 15.0 km (R2, south) and 10.0 km (R3, north)
from their evaluation squares, edge to edge. Their labels are every field
that reaches into them, whatever its size: 310 (A0), 680 (A1), 505 (R2) and
293 (R3). The evaluation squares' land use (the groups of ``CROP_2024``,
below): R1 vegetables and sod 320, row crops 214, hay and pasture 86, not
farmed 46, fallow 2; R2 row crops 235, hay and pasture 222, not farmed 25;
R3 row crops 137, hay and pasture 132, not farmed 12. Their fields by size
(0-1, 1-2, 2-5, 5-10 and 10 ha or more): R1 107, 283, 249, 20 and 9; R2 83,
113, 170, 86 and 30; R3 40, 45, 85, 68 and 43. The strip-like fields are
about 37, 31 and 33 m wide (2 x area / perimeter; medians).

Approaches (each a separate agribound run with ``lulc_filter=False``):

    - Delineate-Anything v2 (``large_v2``), used as released, on NAIP (1 m,
      2022) and on Sentinel-2 (10 m, 2024, up to 1,000 detections per tile,
      as in examples 23-25); on NAIP also with SAM 2 refinement;
    - Delineate-Anything v2 and DINOv3, each fine-tuned on the NAIP of the
      region's training square (in R1 also of A1) and run on the NAIP of the
      evaluation square;
    - the direct SAM engine (``engine="sam-auto"``) on NAIP: SAM 2.1 (large)
      automatic masks on tiles of the composite, made into non-overlapping
      field polygons, with the engine's NAIP default (2,048 m tiles, 32
      points per side, predicted IoU > 0.8, stability >= 0.95, masks claim
      pixels in the order of their scores). That default was chosen on two
      Nevada development squares (example 24's training squares) and is used
      here unchanged; in the engine's pre-registered evaluation it took 6.1
      times Delineate-Anything's GPU time per km2, measured back to back on
      two large New Mexico boxes (3.1 times on example 25's 5 km squares).
      Without the engine or its packages the run is skipped with a message
      (``skipped: ...`` in ``summary.csv``);
    - FTW (default model, ``FTW_PRUE_EFNET_B5``: two Sentinel-2 windows from
      FTW's crop calendar) on Sentinel-2 2024;
    - with ``--spot`` (opt-in): Delineate-Anything v2 on SPOT 6/7 (6 m, 2021;
      restricted), and both models fine-tuned on the SPOT of the training
      squares;
    - with ``--extended`` (opt-in): Delineate-Anything v2 on Landsat 8/9
      panchromatic (15 m) and on Sentinel-1 (VV, VH and VV - VH backscatter
      as R, G and B), and unsupervised k-means (k = 20) of the Google
      Satellite Embedding and of TESSERA, all 2024 (land-cover segments, not
      field instances; TESSERA's 2024 cover of the squares was not checked);
    - with ``--sam-auto-sentinel2`` (opt-in): the direct SAM engine on
      Sentinel-2 2024, for the evaluation of that engine; never a gallery
      panel. On Sentinel-2 the engine did not transfer in that evaluation
      (its F1 was 0.15 below Delineate-Anything's on 46 of example 25's
      squares), and its Sentinel-2 default took 17.0 times
      Delineate-Anything's GPU time per km2, measured back to back on two
      large New Mexico boxes.

Years (``SOURCE_YEARS``): 2024 imagery, or the latest year of a source
without 2024 imagery over the squares: NAIP 2022 (Earth Engine's 2023 images
of New York lie along its borders only, and it has none of 2024 or 2025;
queried on 2026-10-12) and SPOT 6/7 2021 (no growing-season scene over the
evaluation squares in 2022 or 2023; in 2021 R1 has 21 May, R2 24 April only,
bare soil, and R3 3 April and 25 August). NAIP is two years and SPOT three
years older than the other sources. R3's NAIP is from 14 October: on the
training square's image of that day the corn is harvested, the hay green and
the trees in autumn colour. Composites: SPOT, Sentinel-2, Landsat PAN and
Sentinel-1 are medians over the growing season, 1 April to 31 October of
their year (``SEASON``; Sentinel-2 2024 has 19-27 dates with at most 20 %
cloud there, but no clear one in July at R1); FTW builds its own two
windows; NAIP and the embeddings are one image per year. Output names, log
lines and the summary carry the year of every run.

Fine-tuning, as in example 24. The labels are every field of the file that
reaches into the training square, whatever its size (the chips label every
other pixel as background). None of them reaches into the evaluation square
(the script refuses otherwise), and the training composites cover the
training square only, so no evaluated field or pixel is used. Both engines
get the same labels, composite and rules: their default chips and minimum
labelled share, a random 20 % for validation and 20 epochs;
Delineate-Anything with ``yolo_lr0=1e-4``, as in examples 23 and 24 (its
chips overlap, half a chip apart, so the 20 % applies to chip-size cells and
the chips that cover cells of both splits are dropped); DINOv3 (ViT-L/16
backbone pre-trained on SAT-493M satellite imagery, DPT head; no published
field-boundary weights) with agribound's defaults, including its 4 px
boundary band, so a 30 m SPOT strip (5 pixels) keeps no interior pixel. A
fine-tuning that stops with ``ValueError: No validation chip is left ...``
is reported as failed (``failed: ValueError`` in ``summary.csv`` for the run
that needs its model), and the other runs go on. The fine-tuning outputs
(in-sample, in ``<region>_train/`` and ``black_dirt_train_a1/``) are never
evaluated.

Which results are held out (``sample`` in ``summary.csv``):

    - unseen: Delineate-Anything v2 as released (with or without SAM 2) and
      FTW. The US labels of FBIS-73M, the training set of the released
      weights, are the California DWR / Land IQ crop maps; a re-check of
      its public lists on 2026-10-13 (its 201,869 US- and Canadian-prefixed
      patches, each located by its own GeoTIFF header) found no field label
      in New York and no patch within 131 km of the squares (one Ontario
      patch reaches across the Niagara River onto 6.5 ha of New York shore
      without a CLU, and none of its fields is in New York). Only the US-
      and Canadian-prefixed patches were located, and the public lists were
      published after training and changed on the day of the release, so
      they may not be exactly what the released weights saw. FTW's training
      data hold no US fields;
    - held out: the models fine-tuned on the primary training square (A0 in
      R1), 5-15 km from the evaluation square;
    - near-sample: R1's A1 models (1 km away, the same NAIP flight);
    - label-free: sam-auto, whose settings come from Nevada, and the
      embedding clusters. DINOv3's SAT-493M pre-training is self-supervised.

Crop filter, as in example 24. Every run delineates without the LULC crop
filter, which is then applied as a separate step with the default rule: in
the conterminous US, Annual NLCD of the run's year, and a polygon is kept
when at least 30 % of it is cultivated crops or pasture/hay. The tables
report how many polygons it keeps, ``summary.csv`` also the share of the
matched predictions it keeps (``matched_kept_crop``), and the script prints
how many reference fields it keeps (NLCD 2024).

Evaluation (``agribound.evaluate``): one-to-one matching at IoU >= 0.5,
boundary F1 at 10 m (the pilot's offsets were 1-4 m), and 95 % bootstrap
intervals (200 resamples of the reference fields, stratified by the land-use
group of ``CROP_2024``; they treat fields as independent, so they are too
narrow when errors are spatially clustered; ``summary.csv`` gives them for
recall and F1). The reference is the fields of the file whose representative
point lies inside the evaluation square, without those smaller than 2,500 m2
(the predictions' minimum area; the script prints how many it drops: 3, 4
and 2), and the predictions are selected by the same rule. Recall is also
reported per stratum, with the number of reference fields of each:

    - land use in the year of the imagery: the group of the file's
      ``CROP_<year>`` (the majority CDL class of the field): row crops, hay
      and pasture, vegetables and sod, fruit and vines, fallow, other farmed
      (``FARMED_CDL`` classes in no group) and not farmed (``LAND_USE``); a
      field without a class (0) is in no group;
    - shape: strip-like (``PP_SCORE``, the file's Polsby-Popper
      compactness, below 0.3) or compact;
    - field size: 0-1, 1-2, 2-5, 5-10 and 10 ha or more (example 24's bins
      would put 58 % of R1's fields in one class).

Merge and split rates as in example 24: a reference field is merged when the
prediction that covers most of it also covers at least 25 % of another
reference field, and split when at least two predictions each cover 10 % of
it. Strips and ownership lines show up as merges. FP-noref counts the
predictions that overlap no reference field; some of them are farmland the
file does not map.

Chance baselines (``evaluate(..., baselines=...)``, as in example 27, at
10 m): a blind square grid with cells of the run's median polygon area over
the square (the mean of 36 offsets), and the reference shifted east and
north by half its median width. They show what a layer that uses no image
reaches against these CLUs.

Paired comparisons (``agribound.evaluate.paired_bootstrap``). Every run is
compared with Delineate-Anything v2 as released on the same source (``naip
da``, ``spot da`` or ``sentinel2 da``): the F1 difference, with two paired 95 %
intervals (2,000 resamples, seed 42, the same draws for both runs): a field
bootstrap (reference fields drawn within their ``CROP_2024`` group; each
prediction goes with the field it matches, else the one it overlaps most, and
the predictions that overlap no field are kept in every draw) and a block
bootstrap of the square's 1 km blocks that hold a field or a prediction (at
most 25; fields and predictions by representative point), which also draws
the predictions on ground that the file does not map. The metrics file of
each run holds both records (``paired``).

Pre-registered rules (``outputs/v11_features/round3/impl-ny/PREREG-NY.md``,
written on 2026-10-12 before this script, with amendment A1 on the paired
intervals before any run; the script applies the first two):

    - ``better_by_rule``: a run is "better" ("worse") than the released
      model only if both paired intervals lie above (below) 0; otherwise "no
      clear difference";
    - ``above_chance``: the lower bound of the run's F1 interval exceeds the
      higher of the two baselines' F1;
    - statements about strips quote the strip-like recall with its counts;
    - the A1 arm is reported as near-sample, never as a held-out result;
    - ``panels.json``: each region's gallery panels (entries 28, 28b, 28c):
      ``naip da``; the best F1 of ``naip da-sam2``, ``naip da-ft`` and
      ``naip dinov3-ft``; the better of ``sentinel2 da`` and
      ``sentinel2 ftw``; ``naip sam-auto``, else the best SPOT run, else the
      other Sentinel-2 run (ties: the higher boundary F1, then that order).

Pipeline check. R1's two released Delineate-Anything runs must reproduce
example 25's New York outputs (agribound 1.1, 2026-10-10): 585 polygons, 92
matches, F1 0.147 on NAIP 2022 and 266, 52, F1 0.111 on Sentinel-2 2024.
R1's study area is written as example 25 writes it (the square's edges
densified every 100 m), so its two composites are example 25's Earth Engine
requests, with the same cache keys (a copy of example 25's composites in
``black_dirt/.agribound_cache`` is used as it is). The script compares the
counts and F1, and the polygons when example 25's layers are in
``outputs/multistate_openet/ny/``, prints PASSED or FAILED and writes
``pipeline_check.json``; ``--pipeline-check`` runs only these two runs. If
the check fails, the other results are not used until the cause is found.

Outputs (``outputs/new_york_field_boundaries/``): per region (``<region>/``)
the square (``study_area.geojson``), the prediction layers, their
crop-filtered copies, ``metrics_<output>.json`` and a map; the fine-tuning
runs in ``<region>_train/`` (and ``black_dirt_train_a1/``); in
``references/`` the evaluated reference and the training labels, named by a
hash of their content (CLU outlines: local only); ``summary.csv`` with one
row per region and run (a run replaces the rows of the regions it runs and
keeps the others), ``panels.json`` and ``pipeline_check.json``.

SPOT 6/7 (AIRBUS/SPOT6_7) is restricted to select Earth Engine users.
Without access ``--spot`` skips the SPOT runs and fine-tunings and the rest
runs. ``--no-dinov3`` skips DINOv3 (full fine-tuning of a ViT-L; each
checkpoint is about 3.7 GB, in the training square's ``.agribound_cache``),
``--no-fine-tune`` all fine-tuning, ``--no-near-sample`` R1's A1 arm,
``--no-sam`` the SAM 2 refinement and ``--no-sam-auto`` the direct SAM run;
``--only finger_lakes`` (or ``black_dirt``, ``north_country``) runs some
regions. ``--dry-run`` reads the squares' fields and prints the squares, the
reference counts and every run's configuration, and runs and writes nothing
(no Earth Engine). ``--select`` re-applies the square rules and rule 1d and
runs nothing else (it downloads the Census county outlines into
``outputs/new_york_field_boundaries/`` and samples CDL 2024 in Earth
Engine).

Runtime: measured in the run of 2026-10-12/13 with agribound 1.1.0 on an
Apple M2 Max with MPS (Hugging Face and torch hub weights cached, composites
downloaded), with ``--sam-auto-sentinel2``: 3.1 hours (2.1 minutes for
``--pipeline-check``, whose two runs the full run reused, then 184.3
minutes). The eight NAIP fine-tunings took 2.5 hours (Delineate-Anything
12-18 and DINOv3 17-25 minutes each, with their inference on the training
square); the other 24 evaluated runs, the scores, chance baselines and
paired intervals of all 26, the crop filter and the maps about 30 minutes
(sam-auto 2.5-2.8 minutes per square on NAIP and 15-19 seconds on
Sentinel-2, the delineate step with model loading). ``--spot`` and
``--extended`` were not run; the estimates for them (about 45 and 10
minutes) are from examples 24 and 25.

Prerequisites:
    pip install "agribound[gee,delineate-anything,dinov3,ftw,samgeo,sam-auto]"
        (or use environment.yml; sam-auto: the direct SAM engine, SAM 2.1
        through the sam2 package; --extended also needs embedding and
        tessera)
    agribound auth --project YOUR_GEE_PROJECT
    The OpenET field boundaries v2.1.1 (local copy, not public yet):
        --openet-dir DIR or OFB_DATA_DIR=DIR, where DIR holds NY.zip
    Run from the repository root: python examples/28_new_york_field_boundaries.py
"""

import argparse
import hashlib
import importlib.util
import io
import json
import logging
import math
import os
import sys
import time
import urllib.request
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
# GDAL corrects the ring order of some OpenET polygons as it reads them, and says so.
warnings.filterwarnings("ignore", message=".*invalid winding order.*", category=RuntimeWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
DATA_ENV = "OFB_DATA_DIR"  # environment variable: the folder of the <ST>.zip files
#: The folder of NY.zip when neither --openet-dir nor $OFB_DATA_DIR gives it (for example
#: in the notebook); None: one of them must.
OPENET_DIR = None
DATA_LABEL = "OpenET agricultural field boundaries v2.1.1"
REFERENCE_LABEL = "OpenET field boundaries v2.1.1, New York (CLU)"
STATE = "NY"
OUTPUT_DIR = Path("outputs/new_york_field_boundaries")
REFERENCE_DIR = OUTPUT_DIR / "references"
UTM_CRS = "EPSG:32618"  # UTM zone 18N: the zone of every square
UTM_ZONE = 18
SIDE_M = 5000  # side of every square (evaluation and training)
GRID_M = 5000  # the rule's candidates: the cells of the zone's 5 km grid (example 25's)
MIN_GAP_M = 5000  # a training square lies at least this far from its evaluation square
NEAR_GRID_M = 1000  # R1's A1 rule: 5 km squares with corners on the 1 km grid ...
NEAR_GAP_M = 1000  # ... at least 1 km from the evaluation square, edge to edge
MIN_AREA = 2500  # m^2 (EPSG:6933); the predictions' default minimum area, also for the reference
COMPLETENESS_MIN = 0.5  # rule 1d: the share of farmed CDL pixels a kept region's square maps
IOU_THRESHOLD = 0.5
BOUNDARY_TOLERANCE_M = 10.0
N_BOOTSTRAP = 200
SEED = 42  # the bootstrap seed, also of the paired intervals
PAIRED_RESAMPLES = 2000  # resamples of the paired F1 differences
BLOCK_M = 1000  # blocks of the paired block bootstrap
CROP_THRESHOLD = 0.3
N_CLUSTERS = 20  # embedding k-means, fixed in advance as in example 23
MAX_DETECTIONS = 1000  # per tile, at 10-15 m (Sentinel-1, Sentinel-2, Landsat PAN)
MERGE_SHARE = 0.25  # merged: the main prediction also covers this share of another field
SPLIT_SHARE = 0.10  # split: at least two predictions each cover this share of the field
STRIP_PP = 0.3  # strip-like: the file's Polsby-Popper compactness (PP_SCORE) below this
SIZE_BINS_HA = (0, 1, 2, 5, 10, float("inf"))
#: Imagery year of each source: 2024, or the latest year with imagery over the squares.
SOURCE_YEARS = {
    "naip": 2022,  # Earth Engine NAIP (USDA/NAIP/DOQQ): 2022 is the latest year here
    "spot": 2021,  # SPOT 6/7: no growing-season scene over the squares in 2022 or 2023
    "sentinel2": 2024,
    "landsat-pan": 2024,
    "sentinel1": 2024,
    "google-embedding": 2024,
    "tessera-embedding": 2024,
}
#: Composite window (month-day, inclusive) of these sources: the growing season.
SEASON = ("04-01", "10-31")
SEASONAL_SOURCES = ("spot", "sentinel2", "landsat-pan", "sentinel1")
SPOT_CLOUD_COVER_MAX = 15
OPENET_COLUMNS = [
    "Unique_ID", "SOURCECODE", "MOD_DATE", "FIPS", "PP_SCORE",
    *(f"CROP_{year}" for year in sorted(set(SOURCE_YEARS.values()))),
]  # fmt: skip
FINE_TUNE_EPOCHS = 20
FINE_TUNE_SOURCES = ("naip", "spot")
#: Engine parameters of the fine-tunings (Delineate-Anything: as in examples 23 and 24).
FINE_TUNE_PARAMS = {"delineate-anything": {"yolo_lr0": 1e-4}, "dinov3": {}}
#: The direct SAM engine, run with its own per-source defaults (no engine_params here).
SAM_AUTO = "sam-auto"
#: Modules the direct SAM engine needs (SAM 2.1 through the sam2 package, on torch; OpenCV,
#: which sam2 does not install, resamples its tiles).
SAM_AUTO_MODULES = ("torch", "sam2", "cv2")
#: CDL classes that count as farmed: crop, hay and pasture classes (example 25's FARMED_CDL).
FARMED_CDL = frozenset([*range(1, 62), *range(66, 78), *range(204, 255), 176])
#: Land-use groups of the CDL classes in the file's CROP_<year> (the majority CDL class of a
#: field), the groups of the design's strata. A FARMED_CDL class of no group is "other
#: farmed", any other class "not farmed"; 0 (no class assigned) is in no group.
LAND_USE = {
    "row crops": frozenset([
        1, 4, 5, 6, 12, 13, 21, 22, 23, 24, 25, 26, 27, 28, 29, 42,
        205, 225, 226, 228, 236, 237, 238, 239, 240, 241, 254,
    ]),
    "hay and pasture": frozenset([36, 37, 58, 176]),
    "vegetables and sod": frozenset([
        43, 44, 46, 47, 48, 49, 50, 53, 54, 55, 57, 59, 206, 207, 208, 209,
        213, 214, 216, 219, 221, 222, 227, 229, 243, 244, 245, 246, 247, 248, 249,
    ]),
    "fruit and vines": frozenset([
        66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77,
        204, 210, 211, 212, 215, 217, 218, 220, 223, 242, 250,
    ]),
    "fallow": frozenset([61]),
}  # fmt: skip
#: Short name of each land-use group: the table columns and the summary.csv keys
#: (recall_<name>, n_<name>).
LAND_USE_KEYS = {
    "row crops": "row",
    "hay and pasture": "hay",
    "vegetables and sod": "veg",
    "fruit and vines": "fruit",
    "fallow": "fallow",
    "other farmed": "other",
    "not farmed": "not_farmed",
}
#: The 2024 USDA Cropland Data Layer in Earth Engine (band "cropland"), for --select.
CDL_ASSET = "USDA/NASS/CDL/2024"
CDL_BATCH = 5000  # points per Earth Engine request
CDL_PIXEL_M = 10  # grid of rule 1d
#: County outlines for the square rule: U.S. Census Bureau 2023 TIGER/Line, pinned by SHA-256.
TIGER_COUNTIES_URL = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
TIGER_COUNTIES_SHA256 = "692e12c30c83adcaabdbac0d3954fafa55e1c89a24b36d95e72e02dff938652e"

# The regions and their squares (lower-left corners in UTM zone 18N, m), chosen by the rules
# in the docstring from the field boundaries and CDL 2024 alone, before any model ran on
# them. --select re-applies the rules and reports whether they give these squares.
REGIONS = {
    "black_dirt": {
        "name": "Black Dirt (R1, Orange County): onions, other vegetables and sod on muck",
        "fips": "36071",
        "evaluation": (545000, 4575000),  # example 25's New York square
        "training": (535000, 4575000),  # A0, the rule's square: the primary, held-out arm
        "near_sample": (539000, 4571000),  # A1, Pine Island muck, 1 km away: the second arm
    },
    "finger_lakes": {
        "name": "Finger Lakes (R2, Cayuga County): corn, alfalfa and soybeans, dairy",
        "fips": "36011",
        "evaluation": (365000, 4745000),
        "training": (370000, 4725000),
    },
    "north_country": {
        "name": "North Country (R3, Jefferson County): corn, hay and pasture",
        "fips": "36045",
        "evaluation": (410000, 4845000),
        "training": (415000, 4860000),
    },
}
#: The fine-tuning arms: the key of the training square in REGIONS, and the suffix of the
#: arm's run labels (the training folder's suffix is the same with "_").
ARMS = {"training": "", "near_sample": "-a1"}
#: The counts the design recorded (2026-10-12): the reference fields of each evaluation
#: square and the training labels of each training square. A run warns when it reads others.
RECORDED_COUNTS = {
    ("black_dirt", "evaluation"): 668,
    ("black_dirt", "training"): 310,
    ("black_dirt", "near_sample"): 680,
    ("finger_lakes", "evaluation"): 482,
    ("finger_lakes", "training"): 505,
    ("north_country", "evaluation"): 281,
    ("north_country", "training"): 293,
}
#: The released run each run is compared with (paired F1 differences; claim rule C1).
RELEASED = {"naip": "naip da", "spot": "spot da", "sentinel2": "sentinel2 da"}
#: The pipeline check (rule 1 of the pre-registration): R1's two released
#: Delineate-Anything runs must reproduce example 25's New York outputs (agribound 1.1, run
#: of 2026-10-10): polygons, matches and F1 to 4 decimals; and the same polygons when
#: example 25's layers are in EXAMPLE_25_DIR (relative to the repository root).
PIPELINE_REGION = "black_dirt"
PIPELINE_CHECK = {"naip da": (585, 92, 0.1468), "sentinel2 da": (266, 52, 0.1113)}
EXAMPLE_25_DIR = Path("outputs/multistate_openet/ny")
#: Gallery panels of each region (entries 28, 28b, 28c), by the pre-registered rule.
PANEL_CANDIDATES = {
    "top_right": ("naip da-sam2", "naip da-ft", "naip dinov3-ft"),
    "bottom_left": ("sentinel2 da", "sentinel2 ftw"),
    "spot": ("spot da", "spot da-ft", "spot dinov3-ft"),
}


def openet_dir(args):
    """The folder of NY.zip: ``--openet-dir``, else $OFB_DATA_DIR, else OPENET_DIR."""
    folder = args.openet_dir or os.environ.get(DATA_ENV) or OPENET_DIR
    if not folder:
        raise SystemExit(
            f"The {DATA_LABEL} are not public yet; this example reads a local copy. Pass "
            f"--openet-dir DIR or set {DATA_ENV}=DIR, where DIR holds {STATE}.zip."
        )
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise SystemExit(f"Folder of the {DATA_LABEL} not found: {folder}")
    return folder


def openet_path(folder):
    """GDAL path of the New York field boundaries inside their zip (read in place)."""
    archive = Path(folder) / f"{STATE}.zip"
    if not archive.is_file():
        raise SystemExit(f"{archive} not found: the {DATA_LABEL} hold one <ST>.zip per state.")
    return f"/vsizip/{archive}/{STATE}.shp"


def square(corner):
    """The SIDE_M square with lower-left *corner* (UTM zone 18N) as a shapely box."""
    from shapely.geometry import box

    x, y = corner
    return box(x, y, x + SIDE_M, y + SIDE_M)


def reproject(area, crs, to_crs):
    """*area* (a polygon in *crs*) in *to_crs*, with its edges densified every 100 m."""
    import geopandas as gpd

    return gpd.GeoSeries([area.segmentize(100)], crs=crs).to_crs(to_crs).iloc[0]


def square_study_area(area, name):
    """A GeoJSON FeatureCollection of *area* (a square in UTM zone 18N) in EPSG:4326.

    Written as example 25 writes its squares (edges densified every 100 m), so R1's study
    area is example 25's New York study area, and its composites have example 25's cache
    keys.
    """
    from shapely.geometry import mapping

    feature = {
        "type": "Feature",
        "geometry": mapping(reproject(area, UTM_CRS, 4326)),
        "properties": {"name": name},
    }
    return {"type": "FeatureCollection", "features": [feature]}


def lon_lat(area):
    """Longitude and latitude of the centre of *area* (UTM zone 18N)."""
    import geopandas as gpd

    centre = gpd.GeoSeries([area.centroid], crs=UTM_CRS).to_crs(4326).iloc[0]
    return centre.x, centre.y


def rep_point_inside(gdf, area):
    """Rows whose representative point lies inside *area* (same CRS)."""
    return gdf[gdf.representative_point().within(area).to_numpy()].copy()


def read_fields(path, areas):
    """The fields of the file that reach into the bounding box of *areas* (polygons in UTM
    zone 18N), in UTM zone 18N. Reads only that bounding box."""
    import pyogrio
    import shapely

    crs = pyogrio.read_info(path)["crs"]
    bbox = reproject(shapely.box(*shapely.union_all(areas).bounds), UTM_CRS, crs).bounds
    return pyogrio.read_dataframe(path, bbox=bbox, columns=OPENET_COLUMNS).to_crs(UTM_CRS)


def land_use(code):
    """The land-use group of a CDL class (``LAND_USE``); None for 0 (no class assigned)."""
    code = int(code)
    if code == 0:
        return None
    for group, codes in LAND_USE.items():
        if code in codes:
            return group
    return "other farmed" if code in FARMED_CDL else "not farmed"


def add_strata(reference):
    """Add the strata columns: ``land_use_<year>`` per imagery year and ``shape``."""
    for year in sorted(set(SOURCE_YEARS.values())):
        reference[f"land_use_{year}"] = [land_use(c) for c in reference[f"CROP_{year}"].fillna(0)]
    reference["shape"] = [
        None if math.isnan(v) else ("strip-like" if v < STRIP_PP else "compact")
        for v in reference["PP_SCORE"].astype(float)
    ]
    return reference


def train_name(slug, arm):
    """Name of the training folder of a region's fine-tuning arm."""
    return f"{slug}_train{ARMS[arm].replace('-', '_')}"


def prepare_region(slug, path):
    """Squares, evaluation reference and training labels of one region.

    Reads the fields that reach into the bounding box of the region's squares, once.
    Refuses the region when a training label reaches into the evaluation square, and warns
    when a count differs from the recorded one (``RECORDED_COUNTS``).
    """
    from agribound.postprocess import filter_polygons

    spec = REGIONS[slug]
    areas = {kind: square(spec[kind]) for kind in ("evaluation", *ARMS) if kind in spec}
    fields = read_fields(path, list(areas.values()))
    evaluation = areas["evaluation"]
    inside = rep_point_inside(fields, evaluation)
    # The predictions' minimum area applies to the reference as well.
    reference = filter_polygons(inside, min_area_m2=MIN_AREA).reset_index(drop=True)
    region = {
        "areas": areas,
        "reference": add_strata(reference),
        "n_small": len(inside) - len(reference),
        "labels": {},
        "training_fields": {},
    }
    counts = {"evaluation": len(reference)}
    for arm in ARMS:
        if arm not in spec:
            continue
        # Training labels: every field that reaches into the training square, whatever its
        # size, because the chips label every other pixel as background.
        labels = fields[fields.intersects(areas[arm]).to_numpy()].reset_index(drop=True)
        if labels.intersects(evaluation).any():
            raise ValueError(f"{slug}: a training field of the {arm} square reaches into the "
                             "evaluation square")  # fmt: skip
        region["labels"][arm] = labels
        region["training_fields"][arm] = len(
            filter_polygons(rep_point_inside(labels, areas[arm]), min_area_m2=MIN_AREA)
        )
        counts[arm] = len(labels)
    for kind, n in counts.items():
        recorded = RECORDED_COUNTS.get((slug, kind))
        if n != recorded:
            print(
                f"  WARNING: {slug} {kind} square: {n} fields, not the recorded {recorded} "
                "(another release of the file?)"
            )
    return region


def describe_region(slug, region):
    """Print a region's squares, reference and training labels."""
    import pandas as pd

    ref, areas = region["reference"], region["areas"]
    print(f"\n{REGIONS[slug]['name']}")
    for kind, area in areas.items():
        lon, lat = lon_lat(area)
        x, y = REGIONS[slug][kind]
        line = f"  {kind:<11} square: corner ({x}, {y}) UTM 18N, centre {lon:.5f}, {lat:.5f}"
        if kind != "evaluation":
            gap = area.distance(areas["evaluation"]) / 1000
            centres = area.centroid.distance(areas["evaluation"].centroid) / 1000
            arm = "near-sample arm" if kind == "near_sample" else "primary arm (held out)"
            line += f"; {gap:.1f} km from the evaluation square (centres {centres:.1f} km); {arm}"
        print(line)
    area_ha = ref.to_crs(6933).area / 1e4
    bins = list(zip(SIZE_BINS_HA[:-1], SIZE_BINS_HA[1:], strict=True))
    sizes = {f"{lo:g}-{hi:g}": int(((area_ha >= lo) & (area_ha < hi)).sum()) for lo, hi in bins}
    print(
        f"  reference: {len(ref)} fields (median {area_ha.median():.2f} ha; sources "
        f"{ref['SOURCECODE'].value_counts().to_dict()}); {region['n_small']} smaller fields "
        f"(under {MIN_AREA:,} m2, the predictions' minimum area) dropped"
    )
    dates = pd.to_datetime(ref["MOD_DATE"].dropna())
    if len(dates):
        print(f"  MOD_DATE {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}")
    print(f"  size classes (ha): {sizes}")
    strips = ref[(ref["shape"] == "strip-like").to_numpy()]
    width = (2 * strips.area / strips.length).median() if len(strips) else math.nan
    print(
        f"  strip-like (PP_SCORE < {STRIP_PP:g}): {len(strips)} of {len(ref)} "
        f"(median width {fmt(width, 1)} m, 2 x area / perimeter)"
    )
    for year in sorted(set(SOURCE_YEARS.values())):
        groups = ref[f"land_use_{year}"].fillna("(no class)").value_counts().to_dict()
        print(f"  land use (CROP_{year}): {groups}")
    for arm, labels in region["labels"].items():
        print(
            f"  training labels ({arm}): {len(labels)} fields reach into the square "
            f"({region['training_fields'][arm]} of at least {MIN_AREA:,} m2 by representative "
            "point); none reaches into the evaluation square"
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
    for column in [c for c in gdf.columns if c.startswith("land_use_")] + ["shape"]:
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
        if mine.size == 0 or shared[mine].max() <= 0:  # no prediction covers any of it
            continue
        split[r] = int((shared[mine] >= SPLIT_SHARE * r_area[r]).sum()) >= 2
        main = p_idx[mine[np.argmax(shared[mine])]]
        others = np.flatnonzero((p_idx == main) & (r_idx != r))
        merged[r] = bool((shared[others] >= MERGE_SHARE * r_area[r_idx[others]]).any())
    return float(merged.mean()), float(split.mean())


def above_chance(row):
    """Claim rule C2: the lower bound of the run's F1 interval exceeds the higher of the two
    baselines' F1 (None without an interval or a baseline)."""
    values = [row.get(key) for key in ("f1_ci_low", "grid_f1", "shifted_f1")]
    if any(v is None or math.isnan(float(v)) for v in values):
        return None
    return bool(values[0] > max(values[1:]))


def evaluate_run(pred, ref, year, area, args):
    """Headline, per-stratum and chance-baseline metrics of one layer (a dict), and the
    records of the three evaluations behind them."""
    import geopandas as gpd

    from agribound.evaluate import evaluate

    study = gpd.GeoDataFrame(geometry=[area], crs=UTM_CRS)
    m = evaluate(
        pred,
        ref,
        iou_threshold=IOU_THRESHOLD,
        strata="land_use_2024",  # the bootstrap's strata, the same for every run
        size_bins=SIZE_BINS_HA,
        bootstrap=args.bootstrap,
        bootstrap_seed=SEED,
        boundary_tolerance_m=BOUNDARY_TOLERANCE_M,
        baselines={"study_area": study, "shift_m": "auto"},
    )
    # Land use in the imagery year, and strip shape: no intervals, no boundary metrics.
    land = evaluate(pred, ref, iou_threshold=IOU_THRESHOLD, strata=f"land_use_{year}")
    shape = evaluate(pred, ref, iou_threshold=IOU_THRESHOLD, strata="shape")
    merged, split = merge_split_rates(pred, ref)
    ci = (m.get("bootstrap") or {}).get("ci", {})
    baselines = m.get("baselines") or {}
    grid = baselines.get("blind_grid") or {}
    shifted = baselines.get("shifted_reference") or {}
    row = {
        "fields": len(pred),
        "tp": m["count_tp"],
        "precision": m["precision"],
        "recall": m["recall"],
        "recall_ci_low": (ci.get("recall") or [None, None])[0],
        "recall_ci_high": (ci.get("recall") or [None, None])[1],
        "f1": m["f1"],
        "f1_ci_low": (ci.get("f1") or [None, None])[0],
        "f1_ci_high": (ci.get("f1") or [None, None])[1],
        "iou_mean": m["iou_mean"],
        "best_iou_mean": m["best_iou_mean"],
        "undersegmentation_mean": m["undersegmentation_mean"],
        "boundary_f1": m.get("boundary_f1"),
        "fp_noref": m["count_fp_unassigned"],
        "merged": merged,
        "split": split,
    }
    for name, key in LAND_USE_KEYS.items():
        part = land["per_stratum"].get(name) or {}
        row[f"recall_{key}"] = part.get("recall")
        row[f"n_{key}"] = part.get("n", 0)
    for name, key in (("strip-like", "strip"), ("compact", "compact")):
        part = shape["per_stratum"].get(name) or {}
        row[f"recall_{key}"] = part.get("recall")
        row[f"tp_{key}"] = part.get("count_tp", 0)
        row[f"n_{key}"] = part.get("n", 0)
    for name, part in m["per_size_class"].items():
        row[f"recall_{name}_ha"] = part["recall"]
    row.update(
        {
            "grid_cell_m": grid.get("cell_side_m", math.nan),
            "grid_f1": grid.get("f1", math.nan),
            "grid_f1_max": grid.get("f1_max", math.nan),
            "grid_bf1": grid.get("boundary_f1", math.nan),
            "shift_m": shifted.get("shift_m", math.nan),
            "shifted_f1": shifted.get("f1", math.nan),
            "shifted_bf1": shifted.get("boundary_f1", math.nan),
        }
    )
    row["above_chance"] = above_chance(row)
    return row, {"metrics": m, "metrics_land_use": land, "metrics_shape": shape}


def better_by_rule(diff):
    """Claim rule C1: "better" when both paired intervals lie above 0, "worse" when both lie
    below 0, otherwise (an undefined interval included) "no clear difference"."""
    lows = [diff[f"f1_diff_{kind}_ci_low"] for kind in ("field", "block")]
    highs = [diff[f"f1_diff_{kind}_ci_high"] for kind in ("field", "block")]
    if any(v is None or math.isnan(v) for v in lows + highs):
        return "no clear difference"
    if min(lows) > 0:
        return "better"
    if max(highs) < 0:
        return "worse"
    return "no clear difference"


def paired_difference(pred, released, reference):
    """F1(*pred*) - F1(*released*) with its two paired 95 % intervals and claim rule C1.

    Both from :func:`agribound.evaluate.paired_bootstrap` (PAIRED_RESAMPLES resamples, seed
    SEED, the same draws for both layers): reference fields drawn within their CROP_2024
    land-use group, and the BLOCK_M blocks of UTM zone 18N (the squares' grid). Returns the
    summary columns (a dict) and the two records.
    """
    from agribound.evaluate import paired_bootstrap

    common = {"iou_threshold": IOU_THRESHOLD, "n_resamples": PAIRED_RESAMPLES, "seed": SEED}
    records = {
        "field": paired_bootstrap(pred, released, reference, strata="land_use_2024", **common),
        "block": paired_bootstrap(
            pred, released, reference, block_size_m=BLOCK_M, block_crs=UTM_CRS, **common
        ),
    }
    diff = {"f1_diff": records["field"]["difference"]["f1"]}
    for kind, record in records.items():
        low, high = record["ci"]["f1"]
        diff[f"f1_diff_{kind}_ci_low"], diff[f"f1_diff_{kind}_ci_high"] = low, high
    diff["better_by_rule"] = better_by_rule(diff)
    return diff, records


def same_polygons(a, b):
    """True when two layers hold the same polygons (in any order)."""
    import shapely

    if len(a) != len(b) or a.crs != b.crs:
        return False

    def keys(gdf):
        return sorted(shapely.to_wkb(shapely.normalize(gdf.geometry.to_numpy())))

    return keys(a) == keys(b)


def pipeline_check(row, output_path):
    """Rule 1 of the pre-registration for one of R1's released runs: "passed" or
    "FAILED: ...". Compares the polygons, matches and F1 with example 25's (PIPELINE_CHECK)
    and, when example 25's layer is present, the polygons of the two layers."""
    import geopandas as gpd

    expected = PIPELINE_CHECK[row["run"]]
    found = (row["fields"], row["tp"], round(float(row["f1"]), 4))
    problems = (
        [] if found == expected else [f"(polygons, matches, F1) {found}, expected {expected}"]
    )
    theirs = EXAMPLE_25_DIR / Path(output_path).name
    if theirs.exists():
        if not same_polygons(gpd.read_file(output_path), gpd.read_file(theirs)):
            problems.append(f"the polygons differ from {theirs}")
        compared = f"polygons compared with {theirs}"
    else:
        compared = f"{theirs} not found, polygons not compared"
    status = "passed" if not problems else "FAILED: " + "; ".join(problems)
    return f"{status} ({compared})"


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


def fine_tune_config(slug, arm, source, engine, labels_path, args):
    """AgriboundConfig that fine-tunes *engine* on *source* of a region's training square."""
    label = "da" if engine == "delineate-anything" else engine
    return run_config(
        OUTPUT_DIR / train_name(slug, arm),
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


def fine_tune_on(slug, arm, source, engine, labels_path, args):
    """Fine-tune *engine* on *source* of a region's training square; return the checkpoint.

    Both engines get the same labels, composite, epochs and validation share, each on
    its own default chips; the ``yolo_*`` settings apply to Delineate-Anything only.
    """
    from agribound.provenance import read_provenance

    config = fine_tune_config(slug, arm, source, engine, labels_path, args)
    print(
        f"\n{'=' * 70}\n{REGIONS[slug]['name']}, {train_name(slug, arm)}: fine-tuning {engine} "
        f"on {source} {SOURCE_YEARS[source]}\n{'=' * 70}"
    )
    agribound.delineate(config=config)  # in-sample output: never evaluated
    record = read_provenance(config.output_path) or {}
    return (record.get("facts") or {}).get("fine_tuned_checkpoint")


def region_arms(slug, args):
    """The fine-tuning arms of a region: its training square, and R1's A1 unless switched off."""
    arms = [arm for arm in ARMS if arm in REGIONS[slug]]
    return [arm for arm in arms if arm != "near_sample" or not args.no_near_sample]


def fine_tune_plan(args):
    """(source, engine) pairs fine-tuned on each training square."""
    if args.no_fine_tune or args.pipeline_check:
        return []
    engines = ["delineate-anything"] + ([] if args.no_dinov3 else ["dinov3"])
    sources = [s for s in FINE_TUNE_SOURCES if s != "spot" or args.spot]
    return [(source, engine) for source in sources for engine in engines]


def region_runs(slug, args, checkpoints):
    """(label, source, engine, config overrides) of every evaluated run of one region."""
    da, many = "delineate-anything", {"engine_params": {"max_detections": MAX_DETECTIONS}}
    if args.pipeline_check:  # rule 1 of the pre-registration: R1's two released runs only
        return [("da", "naip", da, {}), ("da", "sentinel2", da, many)]
    runs = [("da", "naip", da, {})]
    if not args.no_sam:
        runs.append(("da-sam2", "naip", da, {"sam_refine": True}))
    sam_auto = not args.sam_auto_skipped
    for source in FINE_TUNE_SOURCES:
        if source == "spot":
            if not args.spot:
                continue
            runs.append(("da", "spot", da, {}))
        for arm in region_arms(slug, args):
            for label, engine in (("da-ft", da), ("dinov3-ft", "dinov3")):
                checkpoint = checkpoints.get((slug, arm, source, engine))
                if checkpoint:
                    params = {"engine_params": {"checkpoint_path": checkpoint}}
                    runs.append((label + ARMS[arm], source, engine, params))
        if source == "naip" and sam_auto and not args.no_sam_auto:
            runs.append(("sam-auto", "naip", SAM_AUTO, {}))  # the engine's NAIP default
    # At 10-15 m one 256 px tile covers much of a square: up to 1,000 detections per tile.
    runs.append(("da", "sentinel2", da, many))
    runs.append(("ftw", "sentinel2", "ftw", {}))
    if sam_auto and args.sam_auto_sentinel2:
        runs.append(("sam-auto", "sentinel2", SAM_AUTO, {}))
    if args.extended:
        for source in ("landsat-pan", "sentinel1"):
            runs.append(("da", source, da, many))
        k = {"engine_params": {"n_clusters": N_CLUSTERS}}
        runs.append((f"k{N_CLUSTERS}", "google-embedding", "embedding", k))
        runs.append((f"k{N_CLUSTERS}", "tessera-embedding", "embedding", k))
    return runs


def skipped_runs(args):
    """(run, source, reason) of the sam-auto runs that cannot run here."""
    if args.pipeline_check or not args.sam_auto_skipped:
        return []
    sources = ([] if args.no_sam_auto else ["naip"]) + (
        ["sentinel2"] if args.sam_auto_sentinel2 else []
    )
    return [(f"{source} {SAM_AUTO}", source, args.sam_auto_skipped) for source in sources]


def sample_of(label, engine):
    """How a run relates to the evaluated fields (claim rule C5; ``sample`` in summary.csv)."""
    if label.endswith(ARMS["near_sample"]):
        return "near-sample"
    if label.endswith("-ft"):
        return "held out"
    if engine in (SAM_AUTO, "embedding"):
        return "label-free"
    return "unseen"


def run_info(slug, label, source, engine):
    """The first columns of a run's row in summary.csv."""
    return {
        "region": slug,
        "run": f"{source} {label}",
        "source": source,
        "engine": engine,
        "year": SOURCE_YEARS[source],
        "sample": sample_of(label, engine),
    }


def sam_auto_missing():
    """Why the direct SAM engine (SAM_AUTO) cannot run here, or None when it can."""
    from agribound.registry import ENGINE_REGISTRY

    if SAM_AUTO not in ENGINE_REGISTRY:
        return f"agribound {agribound.__version__} has no {SAM_AUTO!r} engine"
    missing = [name for name in SAM_AUTO_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        extra = ENGINE_REGISTRY[SAM_AUTO].get("install_extra") or SAM_AUTO
        names = f"{', '.join(missing[:-1])} and {missing[-1]}" if len(missing) > 1 else missing[0]
        return f'{names} not installed (pip install "agribound[{extra}]")'
    return None


def dry_run(slugs, regions, args):
    """Print every run's configuration (validated, nothing run or written)."""
    print(f"\n{'=' * 70}\nDry run: configurations (nothing is run or written)\n{'=' * 70}")

    def show(config):
        if config.date_range:
            window = " to ".join(config.date_range)
        else:
            window = "FTW's two windows" if config.engine == "ftw" else "calendar year"
        notes = [f"fine-tuning on {config.reference_boundaries}"] if config.fine_tune else []
        notes += ["SAM 2"] if config.sam_refine else []
        notes += [str(config.engine_params.get("checkpoint_path", ""))]
        note = "; ".join(n for n in notes if n)
        print(
            f"  {config.source:<17} {config.engine:<18} {config.year} {window:<24} "
            f"{config.output_path}" + (f"\n{'':<64}({note})" if note else "")
        )

    for slug in slugs:
        print(f"\n{REGIONS[slug]['name']}")
        checkpoints = {}
        for arm in region_arms(slug, args):
            labels_path = reference_path(
                regions[slug]["labels"][arm], train_name(slug, arm), "labels"
            )
            for source, engine in fine_tune_plan(args):
                show(fine_tune_config(slug, arm, source, engine, labels_path, args))
                checkpoints[slug, arm, source, engine] = (
                    f"checkpoint fine-tuned on {source} of {train_name(slug, arm)}"
                )
        for label, source, engine, overrides in region_runs(slug, args, checkpoints):
            show(run_config(OUTPUT_DIR / slug, source, engine, label, args, **overrides))
        for name, _, reason in skipped_runs(args):
            print(f"  {name}: skipped: {reason}")
        if slug == PIPELINE_REGION:
            for name, (n, tp, f1) in PIPELINE_CHECK.items():
                print(
                    f"  pipeline check: {name} must give {n} polygons, {tp} matches and F1 "
                    f"{f1:.4f} (example 25)"
                )


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


def gallery_panels(rows, slug):
    """The four panels of a region's gallery entry by the pre-registered rule.

    A dict of run names (None where no candidate ran): ``naip da`` top left; the highest
    F1 of PANEL_CANDIDATES["top_right"] top right and of ["bottom_left"] bottom left (ties:
    the higher boundary F1, then the order listed); bottom right ``naip sam-auto``, else the
    highest F1 of the SPOT runs, else the Sentinel-2 run that is not bottom left.
    """
    done = {row["run"]: row for row in rows if row["region"] == slug and row["status"] == "ok"}

    def score(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return -math.inf
        return -math.inf if math.isnan(value) else value

    def best(candidates):
        ran = [name for name in candidates if name in done]
        if not ran:
            return None
        return max(
            ran,
            key=lambda name: (
                score(done[name].get("f1")),
                score(done[name].get("boundary_f1")),
                -candidates.index(name),
            ),
        )

    bottom_left = best(PANEL_CANDIDATES["bottom_left"])
    if f"naip {SAM_AUTO}" in done:
        bottom_right = f"naip {SAM_AUTO}"
    else:
        bottom_right = best(PANEL_CANDIDATES["spot"]) or best(
            [name for name in PANEL_CANDIDATES["bottom_left"] if name != bottom_left]
        )
    return {
        "top_left": "naip da" if "naip da" in done else None,
        "top_right": best(PANEL_CANDIDATES["top_right"]),
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
    }


def merged_rows(path, rows, slugs):
    """This run's *rows*, and the rows of the earlier file at *path* (if any) of the regions
    that this run did not run, in the order of REGIONS."""
    import pandas as pd

    summary = pd.DataFrame(rows)
    if Path(path).exists():
        earlier = pd.read_csv(path)
        summary = pd.concat([earlier[~earlier["region"].isin(slugs)], summary], ignore_index=True)
    order = {slug: i for i, slug in enumerate(REGIONS)}
    summary = summary.sort_values("region", key=lambda col: col.map(order), kind="stable")
    return summary.reset_index(drop=True)


def fmt(value, digits=2):
    """Format a metric (NaN-safe)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if math.isnan(value) else f"{value:.{digits}f}"


def print_tables(slug, rows, reference):
    """The region's headline table and its recall per stratum."""
    done = [row for row in rows if row["region"] == slug and row["status"] == "ok"]
    print(
        f"\n  {'Run':<22} {'Year':>4} {'Sample':<11} {'Fields':>6} {'P':>5} {'R':>5} {'F1':>5} "
        f"{'F1 95 % CI':>11} {'IoU':>5} {'bF1':>5} {'Merge':>5} {'Split':>5} {'FP-noref':>8} "
        f"{'Kept':>5}"
    )
    for row in done:
        interval = f"{fmt(row['f1_ci_low'])}-{fmt(row['f1_ci_high'])}"
        print(
            f"  {row['run']:<22} {row['year']:>4} {row['sample']:<11} {row['fields']:>6} "
            f"{fmt(row['precision']):>5} {fmt(row['recall']):>5} {fmt(row['f1']):>5} "
            f"{interval:>11} {fmt(row['iou_mean']):>5} {fmt(row['boundary_f1']):>5} "
            f"{fmt(row['merged']):>5} {fmt(row['split']):>5} {row['fp_noref']:>8} "
            f"{row['kept_crop']:>5}"
        )
    for row in rows:
        if row["region"] == slug and row["status"] != "ok":
            print(f"  {row['run']:<22} {row['year']:>4} {row['status']}")

    # Recall per stratum. Land use depends on the imagery year, so the number of reference
    # fields of every stratum is printed once per imagery year.
    area_ha = reference.to_crs(6933).area / 1e4
    bins = list(zip(SIZE_BINS_HA[:-1], SIZE_BINS_HA[1:], strict=True))
    groups = [(name, key) for name, key in LAND_USE_KEYS.items()]
    headers = [key for _, key in groups] + ["strip", "compact"]
    headers += [f"{lo:g}-{hi:g}".replace("-inf", "+") for lo, hi in bins]
    keys = [f"recall_{key}" for _, key in groups] + ["recall_strip", "recall_compact"]
    keys += [f"recall_{lo:g}-{hi:g}_ha" for lo, hi in bins]
    fixed = [int((reference["shape"] == name).sum()) for name in ("strip-like", "compact")]
    fixed += [int(((area_ha >= lo) & (area_ha < hi)).sum()) for lo, hi in bins]
    widths = [max(len(header), 5) for header in headers]

    def line(first, year, cells):
        return f"  {first:<22} {year:>4}" + "".join(
            f" {cell:>{width}}" for cell, width in zip(cells, widths, strict=True)
        )

    print(
        "\n  Recall per stratum: land use in the year of the imagery (CROP_<year>: row crops, "
        "hay and pasture, vegetables and sod,\n  fruit and vines, fallow, other farmed, not "
        f"farmed), shape (strip-like: PP_SCORE < {STRIP_PP:g}) and field size (ha)"
    )
    print(line("Run", "Year", headers))
    for year in sorted({row["year"] for row in done}):
        land = [int((reference[f"land_use_{year}"] == name).sum()) for name, _ in groups]
        print(line("(reference fields)", year, land + fixed))
    for row in done:
        print(line(row["run"], row["year"], [fmt(row.get(key)) for key in keys]))


def print_chance_and_pairs(slug, rows):
    """The region's chance baselines (claim rule C2) and paired differences (rule C1)."""
    done = [row for row in rows if row["region"] == slug and row["status"] == "ok"]
    print(
        f"\n  Chance baselines (no image; at {BOUNDARY_TOLERANCE_M:g} m) and claim rule C2 "
        "(above chance: the lower bound of F1 above both baselines' F1)"
    )
    print(
        f"  {'Run':<22} {'Year':>4} {'F1 low':>6} {'Cell':>5} {'Grid F1':>7} {'Grid bF1':>8} "
        f"{'Above chance':>12}"
    )
    for row in done:
        print(
            f"  {row['run']:<22} {row['year']:>4} {fmt(row['f1_ci_low']):>6} "
            f"{fmt(row['grid_cell_m'], 0):>5} {fmt(row['grid_f1'], 3):>7} "
            f"{fmt(row['grid_bf1']):>8} {str(row['above_chance']):>12}"
        )
    if done:
        row = done[0]
        print(
            f"  reference shifted {fmt(row['shift_m'], 0)} m east and north: F1 "
            f"{fmt(row['shifted_f1'], 3)}, bF1 {fmt(row['shifted_bf1'])}"
        )
    paired = [row for row in done if row.get("compared_with")]
    if not paired:
        return
    print(
        f"\n  Paired F1 differences against the released model on the same source (95 % "
        f"intervals, {PAIRED_RESAMPLES:,} resamples: field and 1 km block bootstrap) and claim "
        "rule C1"
    )
    print(f"  {'Run':<22} {'Against':<14} {'dF1':>6} {'Field CI':>13} {'Block CI':>13}  Rule")
    for row in paired:
        field = f"{fmt(row['f1_diff_field_ci_low'], 3)}/{fmt(row['f1_diff_field_ci_high'], 3)}"
        block = f"{fmt(row['f1_diff_block_ci_low'], 3)}/{fmt(row['f1_diff_block_ci_high'], 3)}"
        print(
            f"  {row['run']:<22} {row['compared_with']:<14} {fmt(row['f1_diff'], 3):>6} "
            f"{field:>13} {block:>13}  {row['better_by_rule']} ({row['sample']})"
        )


def county_outlines():
    """New York's county outlines (Census TIGER/Line 2023, EPSG:4269) by FIPS code (GEOID),
    checked by SHA-256 and downloaded once into OUTPUT_DIR."""
    import geopandas as gpd

    path = OUTPUT_DIR / Path(TIGER_COUNTIES_URL).name
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(TIGER_COUNTIES_URL) as response:
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != TIGER_COUNTIES_SHA256:
        raise SystemExit(
            f"{TIGER_COUNTIES_URL}: SHA-256 {digest}, expected {TIGER_COUNTIES_SHA256}"
        )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    counties = gpd.read_file(io.BytesIO(data))
    return counties[counties["STATEFP"] == "36"].set_index("GEOID")


def cdl_classes(points):
    """The CDL 2024 class under each point of *points* (a GeoSeries), 0 where CDL has none.

    Samples Earth Engine's CDL_ASSET on its own grid, CDL_BATCH points per request; the
    classes come back in the order of the points (example 25's sampling).
    """
    import ee
    import numpy as np

    if not len(points):
        return np.zeros(0, dtype=int)
    lonlat = points.to_crs(4326).get_coordinates().to_numpy()
    image = ee.Image(CDL_ASSET).select("cropland").unmask(0)
    classes = []
    for start in range(0, len(lonlat), CDL_BATCH):
        batch = ee.List(
            [[float(lon), float(lat)] for lon, lat in lonlat[start : start + CDL_BATCH]]
        )
        sites = ee.FeatureCollection(batch.map(lambda xy: ee.Feature(ee.Geometry.Point(xy))))
        sampled = image.reduceRegions(sites, ee.Reducer.first(), crs=image.projection())
        classes += sampled.aggregate_array("first").getInfo()
    if len(classes) != len(lonlat):
        raise RuntimeError(f"{CDL_ASSET}: {len(classes)} classes for {len(lonlat)} points")
    return np.asarray(classes, dtype=int)


def farmed(classes):
    """True where a CDL class is in FARMED_CDL."""
    import numpy as np

    return np.isin(classes, sorted(FARMED_CDL))


def rule_square(x, y, is_farmed, inside, grid_m=GRID_M, exclude=None, min_gap_m=0.0):
    """Lower-left corner of the square the rule picks, its farmed fields and its fields.

    Returns ``((x0, y0), n_farmed, n_fields)``, or ``(None, 0, 0)`` when no candidate holds
    a farmed field. *x*, *y* are the representative points of the fields (at least
    MIN_AREA; UTM zone 18N), *is_farmed* whether CDL 2024 maps a FARMED_CDL class under
    each, and *inside* the county's outline in UTM zone 18N. The candidates are the SIDE_M
    squares whose corners lie on the *grid_m* grid, whose centre lies in UTM zone 18 and
    which lie entirely inside *inside*; with *exclude* (a square), only those at least
    *min_gap_m* from it edge to edge. A point counts in the square [x0, x0 + SIDE_M) x
    [y0, y0 + SIDE_M). The rule picks the candidate with the most farmed fields; ties go
    to the southernmost, then the westernmost square centre (latitude, longitude).
    """
    import geopandas as gpd
    import numpy as np

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    is_farmed = np.asarray(is_farmed, dtype=bool)
    if not is_farmed.any():
        return None, 0, 0
    fx, fy = x[is_farmed], y[is_farmed]

    def axis(values):
        first = np.floor((values.min() - SIDE_M) / grid_m) * grid_m
        return np.arange(first, values.max() + 1, grid_m)

    corners = [(x0, y0) for y0 in axis(fy) for x0 in axis(fx)]
    counts = np.array(
        [
            np.count_nonzero((fx >= x0) & (fx < x0 + SIDE_M) & (fy >= y0) & (fy < y0 + SIDE_M))
            for x0, y0 in corners
        ]
    )
    centres = gpd.GeoSeries(
        gpd.points_from_xy(
            [c[0] + SIDE_M / 2 for c in corners], [c[1] + SIDE_M / 2 for c in corners]
        ),
        crs=UTM_CRS,
    ).to_crs(4326)
    lon, lat = centres.x.to_numpy(), centres.y.to_numpy()
    in_zone = np.floor((lon + 180) / 6) + 1 == UTM_ZONE
    for i in np.lexsort((lon, lat, -counts)):  # most farmed first; ties: south, then west
        if counts[i] == 0:
            break
        candidate = square(corners[i])
        if not in_zone[i] or (exclude is not None and candidate.distance(exclude) < min_gap_m):
            continue
        if inside.contains(candidate):
            x0, y0 = corners[i]
            n_fields = np.count_nonzero(
                (x >= x0) & (x < x0 + SIDE_M) & (y >= y0) & (y < y0 + SIDE_M)
            )
            return (int(x0), int(y0)), int(counts[i]), int(n_fields)
    return None, 0, 0


def cdl_grid(area):
    """CDL 2024 on the CDL_PIXEL_M grid of a square (UTM zone 18N), rows from north to south."""
    import ee
    import numpy as np

    x0, _, _, y1 = area.bounds
    n = int(round(SIDE_M / CDL_PIXEL_M))
    request = {
        "expression": ee.Image(CDL_ASSET).select("cropland").unmask(0).toUint8(),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": n, "height": n},
            "affineTransform": {
                "scaleX": CDL_PIXEL_M,
                "shearX": 0,
                "translateX": x0,
                "shearY": 0,
                "scaleY": -CDL_PIXEL_M,
                "translateY": y1,
            },
            "crsCode": UTM_CRS,
        },
    }
    return np.asarray(ee.data.computePixels(request)["cropland"]).astype(int)


def completeness(classes, fields, area):
    """Rule 1d: the share of a square's farmed CDL 2024 pixels that lie in a field.

    *classes* is CDL 2024 on the CDL_PIXEL_M grid of the square *area* (rows from north to
    south, as :func:`cdl_grid` gives it) and *fields* the fields of the file that reach
    into the square, of any size (UTM zone 18N). NaN when the square has no farmed pixel.
    """
    import numpy as np
    from rasterio import features
    from rasterio.transform import from_origin

    x0, _, _, y1 = area.bounds
    shapes = [(geom, 1) for geom in fields.geometry.intersection(area) if not geom.is_empty]
    mapped = np.zeros(classes.shape, dtype=bool)
    if shapes:
        transform = from_origin(x0, y1, CDL_PIXEL_M, CDL_PIXEL_M)
        mapped = features.rasterize(shapes, out_shape=classes.shape, transform=transform)
        mapped = mapped.astype(bool)
    farmed_pixels = farmed(classes)
    if not farmed_pixels.any():
        return math.nan
    return float((farmed_pixels & mapped).sum() / farmed_pixels.sum())


def select_squares(path, slugs):
    """Re-apply the square rules, and rule 1d to the recorded squares; print each square and
    whether it is the recorded one."""
    import geopandas as gpd
    import shapely

    from agribound.postprocess import filter_polygons

    counties = county_outlines()
    start = time.time()
    print(
        f"\nSquare rules ({SIDE_M / 1000:g} km squares; farmed fields: fields of at least "
        f"{MIN_AREA:,} m2 on a farmed CDL 2024 pixel; rule 1d: at least {COMPLETENESS_MIN:g})"
    )
    for slug in slugs:
        spec = REGIONS[slug]
        outline = counties.loc[spec["fips"], "geometry"]
        # Densified first (vertices at most 0.001 degrees, about 100 m, apart).
        inside = gpd.GeoSeries([outline.segmentize(0.001)], crs=counties.crs).to_crs(UTM_CRS)
        inside = inside.iloc[0]
        shapely.prepare(inside)
        every = read_fields(path, [inside])
        fields = filter_polygons(every, min_area_m2=MIN_AREA)
        points = fields.representative_point()
        points = points[shapely.contains_xy(inside, points.x.to_numpy(), points.y.to_numpy())]
        x, y = points.x.to_numpy(), points.y.to_numpy()
        print(f"\n{spec['name']}: {len(points):,} fields with a representative point in it")
        is_farmed = farmed(cdl_classes(points))
        found = {"evaluation": rule_square(x, y, is_farmed, inside)}
        chosen = found["evaluation"][0]
        if chosen is not None:
            found["training"] = rule_square(
                x, y, is_farmed, inside, exclude=square(chosen), min_gap_m=MIN_GAP_M
            )
            if "near_sample" in spec:
                found["near_sample"] = rule_square(
                    x, y, is_farmed, inside, NEAR_GRID_M, square(chosen), NEAR_GAP_M
                )
        for kind, (corner, n_farmed, n) in found.items():
            status = "the recorded square" if corner == spec[kind] else f"recorded: {spec[kind]}"
            print(f"  {kind:<11} {corner}: {n_farmed:,} farmed of {n:,} fields; {status}")
        for kind in ("evaluation", *ARMS):
            if kind in spec:
                area = square(spec[kind])
                share = completeness(cdl_grid(area), every[every.intersects(area).to_numpy()], area)
                print(f"  rule 1d, recorded {kind} square {spec[kind]}: {fmt(share, 3)}")
    print(f"  ({(time.time() - start) / 60:.1f} minutes)")


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(
        description="New York: Black Dirt, Finger Lakes and North Country against the "
        f"{DATA_LABEL}."
    )
    parser.add_argument(
        "--openet-dir",
        default=None,
        help=f"Folder of the {DATA_LABEL}, which holds {STATE}.zip (default: ${DATA_ENV}, then "
        "OPENET_DIR in the script).",
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
        "--only",
        nargs="+",
        default=None,
        metavar="REGION",
        choices=list(REGIONS),
        help=f"Run only these regions ({', '.join(REGIONS)}).",
    )
    parser.add_argument(
        "--spot",
        action="store_true",
        help="Also run SPOT 6/7 2021 (restricted): Delineate-Anything and both fine-tunings.",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Also run Landsat 8/9 PAN and Sentinel-1 (Delineate-Anything) and the Google and "
        "TESSERA embeddings (k-means), 2024.",
    )
    parser.add_argument("--no-sam", action="store_true", help="Skip SAM 2 refinement.")
    parser.add_argument(
        "--no-sam-auto", action="store_true", help="Skip the direct SAM engine on NAIP."
    )
    parser.add_argument(
        "--sam-auto-sentinel2",
        action="store_true",
        help="Also run the direct SAM engine on Sentinel-2 2024 (for the evaluation of that "
        "engine; never a gallery panel). In that evaluation it did not transfer to Sentinel-2 "
        "(F1 0.15 below Delineate-Anything's on 46 of example 25's squares).",
    )
    parser.add_argument(
        "--no-fine-tune",
        action="store_true",
        help="Skip the fine-tuning (Delineate-Anything and DINOv3).",
    )
    parser.add_argument(
        "--no-dinov3", action="store_true", help="Skip the fine-tuned DINOv3 (the slowest step)."
    )
    parser.add_argument(
        "--no-near-sample",
        action="store_true",
        help="Skip R1's second fine-tuning arm (A1, 1 km from the evaluation square).",
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
    parser.add_argument(
        "--pipeline-check",
        action="store_true",
        help="Run only R1's two released Delineate-Anything runs and compare them with "
        "example 25 (the pre-registered pipeline check).",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Re-apply the square rules and rule 1d and report whether they give the "
        "recorded squares; run nothing else (downloads the Census county outlines and "
        "samples CDL 2024 in Earth Engine).",
    )
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def main():
    import geopandas as gpd

    from agribound.evaluate import evaluate_frame

    args = parse_args()
    start_time = time.time()
    openet = openet_path(openet_dir(args))
    slugs = [PIPELINE_REGION] if args.pipeline_check else (args.only or list(REGIONS))
    if args.select:
        from agribound.auth import setup_gee

        setup_gee(project=args.gee_project)
        select_squares(openet, slugs)
        return
    print(f"Reading the {DATA_LABEL}: {openet}")
    regions = {slug: prepare_region(slug, openet) for slug in slugs}
    for slug in slugs:
        describe_region(slug, regions[slug])
    print("\nYears: " + ", ".join(f"{source} {year}" for source, year in SOURCE_YEARS.items()))
    sam_auto_wanted = not args.no_sam_auto or args.sam_auto_sentinel2
    args.sam_auto_skipped = sam_auto_missing() if sam_auto_wanted else None
    if args.sam_auto_skipped:
        print(f"The direct SAM engine cannot run here: {args.sam_auto_skipped}.")
    if args.dry_run:
        dry_run(slugs, regions, args)  # SPOT access is not checked in a dry run
        return

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    args.spot = args.spot and not args.pipeline_check and spot_available(args)
    label_paths = {}
    for slug in slugs:
        region, name = regions[slug], REGIONS[slug]["name"]
        squares = [(OUTPUT_DIR / slug, region["areas"]["evaluation"], name)]
        for arm in region_arms(slug, args) if fine_tune_plan(args) else []:
            title = f"{name} ({train_name(slug, arm)}, training only)"
            squares.append((OUTPUT_DIR / train_name(slug, arm), region["areas"][arm], title))
            label_paths[slug, arm] = save_reference(
                region["labels"][arm], train_name(slug, arm), "labels"
            )
        for out_dir, area, title in squares:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "study_area.geojson").write_text(json.dumps(square_study_area(area, title)))

    checkpoints = {}  # (region, arm, source, engine) -> fine-tuned checkpoint
    failed_fine_tunes = {}  # (region, arm, source, engine) -> exception name
    for slug in slugs:
        for arm in region_arms(slug, args):
            for source, engine in fine_tune_plan(args):
                key = (slug, arm, source, engine)
                what = f"Fine-tuning {engine} on {source} of {train_name(slug, arm)}"
                try:
                    checkpoint = fine_tune_on(
                        slug, arm, source, engine, label_paths[slug, arm], args
                    )
                except Exception as exc:
                    failed_fine_tunes[key] = type(exc).__name__
                    print(f"  {what} failed: {type(exc).__name__}: {exc}")
                    continue
                if checkpoint:
                    checkpoints[key] = checkpoint
                else:
                    failed_fine_tunes[key] = "no checkpoint recorded"
                    print(f"  {what} recorded no checkpoint.")

    rows, checks = [], {}
    for slug in slugs:
        region = regions[slug]
        reference, evaluation = region["reference"], region["areas"]["evaluation"]
        out_dir = OUTPUT_DIR / slug
        print(f"\n{'=' * 70}\n{REGIONS[slug]['name']}\n{'=' * 70}")
        ref_path = save_reference(reference, slug, "reference")
        print(
            f"  Reference: {len(reference)} fields ({region['n_small']} under {MIN_AREA:,} m2 "
            f"dropped) -> {ref_path}"
        )
        # How many reference fields the default crop filter would keep (NLCD of 2024).
        ref_cfg = run_config(out_dir, "sentinel2", "delineate-anything", "reference", args)
        ref_kept = crop_filter_layer(reference, ref_path, ref_cfg, args, year=ref_cfg.year)

        for (key_slug, arm, source, engine), failure in failed_fine_tunes.items():
            if key_slug == slug:  # the run is missing because its fine-tuning failed
                label = ("da-ft" if engine == "delineate-anything" else "dinov3-ft") + ARMS[arm]
                info = run_info(slug, label, source, engine)
                rows.append({**info, "status": f"failed: {failure}"})
        for _, source, reason in skipped_runs(args):
            info = run_info(slug, SAM_AUTO, source, SAM_AUTO)
            rows.append({**info, "status": f"skipped: {reason}"})
        predictions, records = {}, {}  # run -> its polygons; run -> (output stem, metrics)
        for label, source, engine, overrides in region_runs(slug, args, checkpoints):
            info = run_info(slug, label, source, engine)
            name, year = info["run"], info["year"]
            try:
                config = run_config(out_dir, source, engine, label, args, **overrides)
                pred = agribound.delineate(config=config).to_crs(UTM_CRS)
            except Exception as exc:  # e.g. no imagery, an engine's package missing
                print(f"  {name} {year} failed: {type(exc).__name__}: {exc}")
                rows.append({**info, "status": f"failed: {type(exc).__name__}"})
                continue
            out_path = Path(config.output_path)
            kept = crop_filter_layer(pred, out_path, config, args)
            row, metrics = evaluate_run(pred, reference, year, evaluation, args)
            frame = evaluate_frame(pred, reference, iou_threshold=IOU_THRESHOLD)
            predictions[name] = pred
            row = {
                **info,
                "status": "ok",
                **row,
                "kept_crop": len(kept),
                "matched_kept_crop": matched_kept(pred, kept, frame),
            }
            if slug == PIPELINE_REGION and name in PIPELINE_CHECK:
                row["pipeline_check"] = checks[name] = pipeline_check(row, out_path)
                print(f"  Pipeline check, {name} {year}: {row['pipeline_check']}")
            rows.append(row)
            print(f"  {name} {year}: {len(pred)} fields, F1 {fmt(row['f1'])}")
            records[name] = (out_path.stem, metrics)

        # Every run against the released Delineate-Anything run on the same source.
        for row in rows:
            released = RELEASED.get(row["source"])
            if row["region"] != slug or row["status"] != "ok" or row["run"] == released:
                continue
            if released in predictions:
                diff, paired = paired_difference(
                    predictions[row["run"]], predictions[released], reference
                )
                row.update({"compared_with": released, **diff})
                records[row["run"]][1]["paired"] = paired
        for row in rows:
            if row["region"] == slug and row["run"] in records:
                stem, metrics = records[row["run"]]
                metrics_path = out_dir / f"metrics_{stem}.json"
                metrics_path.write_text(
                    json.dumps(to_jsonable({"row": row, **metrics}), indent=2, allow_nan=False)
                )

        print(
            f"\n  {REFERENCE_LABEL}: {len(reference)} fields in the evaluation square; the "
            f"default crop filter (NLCD {ref_cfg.year}) keeps {len(ref_kept)} of them."
        )
        print_tables(slug, rows, reference)
        print_chance_and_pairs(slug, rows)

    if checks:
        record = {"expected (example 25)": PIPELINE_CHECK, "result": checks}
        (OUTPUT_DIR / "pipeline_check.json").write_text(json.dumps(record, indent=2))
        failed = [name for name, result in checks.items() if not result.startswith("passed")]
        verdict = f"FAILED for {', '.join(failed)}" if failed else "PASSED"
        print(f"\nPipeline check (pre-registered): {verdict}")
    if args.pipeline_check:
        print(f"\nTotal runtime: {(time.time() - start_time) / 60:.1f} minutes")
        return
    summary_path = OUTPUT_DIR / "summary.csv"
    # convert_dtypes keeps the counts integers when a failed run leaves them empty
    merged_rows(summary_path, rows, slugs).convert_dtypes().to_csv(summary_path, index=False)
    panels_path = OUTPUT_DIR / "panels.json"
    panels = json.loads(panels_path.read_text()) if panels_path.exists() else {}
    panels.update({slug: gallery_panels(rows, slug) for slug in slugs})
    panels = {slug: panels[slug] for slug in REGIONS if slug in panels}
    panels_path.write_text(json.dumps(panels, indent=2))
    print(
        "\nP, R, F1: one-to-one matching at IoU >= 0.5 (95 % intervals: "
        f"{args.bootstrap} resamples of the reference fields); bF1: boundary F1 within "
        f"{BOUNDARY_TOLERANCE_M:g} m; FP-noref: "
        "predictions that overlap no reference field; Kept: polygons the default crop filter "
        f"keeps (threshold {CROP_THRESHOLD:g}). Scores measure agreement with CLUs, not with "
        "visible fields; one square per region is not an accuracy estimate for the county."
    )
    print(f"Summary: {summary_path}; gallery panels: {panels_path}")

    # Each region on an interactive map: NAIP, Sentinel-2 and FTW layers and the reference.
    from agribound.visualize import show_comparison

    for slug in slugs:
        out_dir = OUTPUT_DIR / slug
        layers, years = [], set()
        for source, label, text in (
            ("naip", "da", "Delineate-Anything, NAIP"),
            ("naip", "da-ft", "Delineate-Anything fine-tuned, NAIP"),
            ("naip", SAM_AUTO, "SAM 2.1 automatic masks (sam-auto), NAIP"),
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
            [frame for frame, _ in layers] + [regions[slug]["reference"]],
            labels=[text for _, text in layers] + [REFERENCE_LABEL],
            output_html=str(map_path),
        )
        show_in_notebook(web_map)
        print(f"Map: {map_path}")
    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
