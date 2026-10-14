"""
26 — Radar: Sentinel-1, Umbra and Capella with Delineate-Anything v2 at Seven Sites (2022-2025)

What agribound can do with synthetic aperture radar (SAR) today, and where it
stops. Delineate-Anything v2, as released (no fine-tuning), delineates fields
on radar composites at seven sites with field references (public: DWR / Land
IQ, SIGPAC and Utah WRLU; local data: the OpenET field boundaries v2.1.1
until their release, and DRI's Nevada fields), beside optical controls on the
same areas:

    - Sentinel-1 (``sentinel1``; C-band, Earth Engine): free and global; a
      median of the window's acquisitions of one orbit pass, 10 m pixels and
      about 20 m resolution;
    - Umbra and Capella open data (``umbra``, ``capella``; X-band, AWS, CC BY
      4.0, no Earth Engine): a co-registered median of 2-6 collects at 1 m,
      only where the vendors have imaged and published, which is few farmland
      sites and fewer still with consistent collects in one season;
    - optionally (``--uavsar``), NASA's airborne UAVSAR L-band quad-pol data
      near UC Davis in 2018, as a reference point for what full polarimetry
      gives.

It measures limits; it does not show that radar can replace optical imagery.
In the earlier runs that it repeats (below), Umbra and Capella composites
scored F1 0.16 to 0.51 (one-to-one, IoU >= 0.5); at the three held-out sites
they came within 0.05 of NAIP read with the same Delineate-Anything settings,
and their lead over NAIP with the default settings came mostly from those
settings. The areas are small (17-25 km2, 74-1,128 reference fields), the
open archives hold little farmland, and Sentinel-1's 20 m resolution blurs
fields of 1-2 ha.

Fixed rules. Nothing below was chosen after seeing a result of this script:
the areas, windows, collects and settings are those of the Umbra prototype and
the Capella pilot (2026-10-08) and of the held-out test of the Umbra and
Capella sources (2026-10-09, whose sites, rules and verdict rule were written
down before any build or score), reused unchanged, and example 24's Diamond
Valley evaluation square. The study areas are stored in ``AREA_RINGS`` (UTM
coordinates rounded to 0.1 m; every reference field selected by the original
outlines is selected by the rounded ones).

The sites (``SITES``; areas in km2: study area / evaluation area; collects:
the automatic selection from the archive indexes of 2026-10-09, with the
download size; reference fields: as counted by ``--dry-run`` on 2026-10-09,
with their median size and the smaller outlines dropped):

    - ``ucd``: UC Davis, Yolo and Solano counties, California (in-sample). The
      Umbra prototype's evaluation area (20.52). Umbra, 2024-06-01 to 09-15: 6 HH
      collects, ascending, right-looking, 2024-07-11 to 09-08, 0.5 m, 27.1-28.5
      deg incidence (1.65 GB). NAIP 2022 (2022-05-20). Reference: DWR / Land
      IQ Statewide Crop Mapping 2024 (provisional), classes other than urban
      and native land: 235 fields, median 1.5 ha (27 dropped).
    - ``tamu``: Texas A&M AgriLife farm, Brazos Bottom, Burleson County, Texas
      (held-out). 17.52 / 17.49. Umbra, 2024-04-01 to 08-15: 6 VV, ascending,
      left-looking, 2024-04-12 to 06-24, 28.9-29.4 deg (3.54 GB). NAIP 2022
      (2022-06-12). Reference: OpenET field boundaries v2.1.1, Texas: 136
      fields, median 4.9 ha (none dropped).
    - ``intel_oh``: Intel Ohio One site, Licking County, Ohio (held-out).
      17.79 / 17.65. Umbra, 2024-06-01 to 09-15: 4 HH, ascending,
      left-looking, 2024-06-11 to 09-06, 24.6-26.6 deg (2.28 GB). NAIP 2023
      (2023-05-21 and 05-26). Reference: OpenET, Ohio: 77 fields, median 5.7
      ha (1 dropped).
    - ``orleans_vt``: Orleans County, Vermont (held-out). 18.19 / 18.06.
      Capella GEO, 2022-06-01 to 09-15: 2 HH spotlight collects, ascending,
      right-looking, 2022-06-18 and 06-26, 41.8-41.9 deg, fully calibrated
      (0.82 GB). NAIP 2023 (2023-06-22). Reference: OpenET, Vermont: 112
      fields, median 3.5 ha (none dropped).
    - ``sevilla``: Sevilla, Andalusia, Spain (in-sample). The Capella pilot's
      evaluation area (23.70). Capella GEO, 2023-01-01 to 01-31: 2 VV,
      descending, left-looking, 2023-01-06 and 01-12, 36.4-36.6 deg (0.83
      GB). No NAIP: PNOA is not an agribound source, so the optical control is
      Sentinel-2 alone. Reference: SIGPAC recintos of the previous campaign,
      2025 when the service was read on 2026-10-09 (the script reads the
      campaign from the service and warns when it is not 2025), without the
      non-field uses, as in example 23: 1,128 fields, median 0.8 ha (1,519
      dropped).
    - ``centerfield``: Centerfield, Sanpete County, Utah (in-sample). The
      Umbra prototype's evaluation area (21.58). Umbra, 2025-05-15 to 09-15: 4
      VV, ascending, left-looking, 2025-05-25 to 07-05, 1 m single-look
      products, 42.8-44.8 deg (0.29 GB). NAIP 2021 (2021-11-05, after the
      harvest). Reference: Utah Water Related Land Use (WRLU) 2025,
      agricultural polygons: 262 fields, median 2.1 ha (29 dropped).
    - ``diamond_valley``: Diamond Valley, Eureka County, Nevada (Sentinel-1
      only: no Umbra or Capella collects). Example 24's evaluation square, 5
      km, lower-left corner (584000, 4389000) in UTM zone 11N (25.00): large
      fields (many under centre pivots), the case most
      favourable to Sentinel-1's resolution. Window 2024-04-01 to 10-31
      (example 24's season). NAIP 2022 (2022-06-29, 09-19 and 09-20).
      Reference: DRI field boundaries (Nevada, 2026-08-20), the DRI-mapped
      fields (``SOURCECODE`` "DRI"), as in example 24: 74 fields, median
      48.2 ha (none dropped). The Sentinel-1, NAIP (defaults) and Sentinel-2
      runs have the settings of example 24's runs on this square (study
      area, year, window, ``max_detections``) and the same reference fields,
      so example 24's ``summary.csv`` should give them the same F1 (its
      intervals use 200 resamples): a consistency check.

Held-out: the recipe of the Umbra and Capella sources and its settings were
fixed before these sites were scored. In-sample: the recipe was chosen on the
Umbra prototype's four areas (UC Davis and Centerfield, both here, and Purdue
and Mallorca) and on the Capella pilot's (Sevilla). The NAIP year follows the
held-out test's rule: the Earth Engine NAIP year (``USDA/NAIP/DOQQ``) closest
to the radar year, a tie going to the acquisition month closest to the
collects (Vermont: 2021 or 2023, 2023 by month). At the six US sites this is
also the latest year that covers the study area (queried on 2026-10-09; Earth
Engine has no 2024 or 2025 NAIP here); at ``ucd_2018`` it is the UAVSAR
year.

Runs (each a separate ``agribound.delineate`` with ``lulc_filter=False`` and
Delineate-Anything v2, ``large_v2``, as released):

    - ``sentinel1``: Sentinel-1 over the site's window, with up to 1,000
      detections per tile (``max_detections``; at 10 m one 256 px tile covers
      2.56 km), as in examples 23-25;
    - ``umbra`` or ``capella``: the composite with every source setting at its
      default (automatic selection of up to 6 collects, 1 m median,
      co-registration to USGS NAIP Plus in the US and date to date elsewhere),
      read by Delineate-Anything with the settings documented for these
      sources, ``processing_gsd_m=3`` and ``merge_tile_pieces=False``
      (``SAR_SETTINGS``; agribound.composites.vhr_sar.DELINEATE_ANYTHING_PARAMS);
    - ``naip``: NAIP (1 m) with Delineate-Anything's defaults, the optical
      control of the prototype and the held-out test;
    - ``naip-sar-settings``: the same NAIP composite read with
      ``SAR_SETTINGS``: in the held-out test most of the radar's lead over
      ``naip`` came from these settings, not from the radar;
    - ``sentinel2``: Sentinel-2 over the site's window, ``max_detections``
      1,000.

Sentinel-1 and Sentinel-2 use the site's radar window (the window in which
the Umbra or Capella collects were searched; Diamond Valley: example 24's
season), so all the satellite images of a site date from one season; NAIP is
one to four years off.

Evaluation (``agribound.evaluate``): one-to-one matching at IoU >= 0.5, with
95 % bootstrap intervals of F1 (1,000 resamples, seed 42), on the site's
evaluation area: the study area, except at the held-out sites, whose
evaluation area is the study area within the valid pixels of the held-out
test's composite, eroded 30 m, and at ``ucd_2018``, whose evaluation area is
the UAVSAR pilot's 3 km square (8.98 km2) inside its processing area (14.44
km2). Every run of a site is scored on the same area. The reference outlines
are made valid (a multi-part outline stays one field, as in examples 23 and
24); those whose representative point lies in the evaluation area and that
cover at least 2,500 m2 (in the site's UTM zone; the predictions' minimum
area) are the reference fields, and the script prints how many smaller ones it
drops. OpenET outlines count when their crop code of the season year
(``CROP_<year>``) is not a non-agricultural class of the Cropland Data Layer
(``NON_AGRICULTURAL_CDL``), as in the held-out test. Predictions are selected
by representative point too. Merged and split rates (example 23's
definitions): a reference field is merged when its main prediction also covers
at least 25 % of another reference field, and split when at least two
predictions each cover at least 10 % of it.

Caveats. Read every score with these:

    - Delineate-Anything v2's training data (FBIS-73M, the training list of
      the Hugging Face dataset ``MykolaL/FBIS-73M``, checked on 2026-10-08)
      include 76,448 Sentinel-1 patches and 7,852 synthetic Sentinel-1
      patches, 2 of them inside the UC Davis area and about 38 within 10 km of
      it, so radar input is not new to the model; and its labels are the DWR /
      Land IQ crop maps in California and SIGPAC in Spain, so the UC Davis and
      Sevilla scores are not held-out geography. Whether FBIS-73M holds X-band
      patches was not checked. In its public lists (revision 9e4cdbd; the US-
      and Canadian-prefixed patches, located by their own GeoTIFF headers on
      2026-10-13), no patch lies within 10 km of the other US sites: the
      nearest lies 235.6 km from Centerfield (in Arizona, unlabelled; the
      nearest labelled one 494.2 km), 285.8 km from Diamond Valley (288.0
      km), 177.2 km from Intel Ohio and 251.0 km from Orleans County (both
      labelled, in Ontario) and about 1,220 km from Texas A&M (in Arizona,
      unlabelled). Sevilla was not traced (only US- and Canadian-prefixed
      patches were located), and the public lists were published after
      training and changed on the day of the release, so they may not be
      exactly what the released weights saw;
    - the recipe and ``SAR_SETTINGS`` were chosen on the Umbra prototype's
      four areas (UC Davis, Centerfield, Purdue and Mallorca) and the Capella
      pilot's (Sevilla): only Texas A&M, Intel Ohio and Orleans County are
      held-out tests (three sites, 77-136 reference fields, F1 intervals of
      about +/- 0.06);
    - Centerfield's 2025 Umbra products are single-look at 1 m, so its
      composite stays speckled: the weak case (F1 0.16 against 0.36 on NAIP in
      the prototype);
    - the OpenET outlines in these three areas are USDA Farm Service Agency
      Common Land Units (CLU; the script prints the sources): farm-programme
      units of the agency's delivery (the files' ``MOD_DATE``, the date of a
      field's last geometry change, is 2020-08-17 in Texas and 2021-04-21 in
      Ohio and Vermont), not a map of the imagery season, so fields changed
      since count as errors; the Texas A&M area holds research plots and the
      Intel Ohio area a construction site;
    - NAIP and the radar differ in year: 2022 against 2024 at UC Davis and
      Texas A&M, 2023 against 2024 in Ohio and against 2022 in Vermont, 2021
      (after the harvest) against 2025 at Centerfield, 2022 against 2024 in
      Diamond Valley;
    - the references are of other years than the imagery at Sevilla (SIGPAC
      2025, Capella January 2023, winter) and in Diamond Valley (the DRI
      fields of 2026); DWR 2024 is provisional;
    - Umbra amplitudes are uncalibrated and the sites differ in polarization
      (HH or VV), incidence and number of collects (2 in Vermont and near
      Sevilla); outside the US the dates are registered to each other only
      (USGS NAIP Plus, the optical base, is US-only);
    - at Sevilla the only optical control is Sentinel-2 at 10 m, against
      recintos with a median of 0.8 ha (1,519 smaller than 2,500 m2 dropped),
      so radar and optical are not compared at a matching resolution there;
    - the automatic selection depends on the archive index: a newer index can
      select other collects (the script says so), and then the scores need not
      match those below;
    - the references are read live from their services: SIGPAC serves the
      previous campaign, which moves on every year (the script warns when it
      is not 2025), DWR 2024 is provisional and Utah WRLU 2025 can be revised,
      so a later run can read other outlines (``summary.csv`` records a hash
      of them, ``reference_sha256``). How well two references agree with each
      other (the ceiling of these scores) was not measured.

What earlier runs measured (context; one-to-one at IoU >= 0.5, reference fields
of at least 2,500 m2):

    - held-out test (2026-10-09; Umbra or Capella / NAIP, defaults / NAIP,
      SAR settings): Texas A&M 0.390 [0.324-0.461] / 0.284 [0.234-0.343] /
      0.403 [0.338-0.471] (136 fields); Intel Ohio 0.261 [0.208-0.313] / 0.193
      [0.137-0.248] / 0.310 [0.240-0.375] (77); Orleans County 0.274
      [0.207-0.350] / 0.261 [0.196-0.337] / 0.249 [0.188-0.319] (112). By the
      rule fixed in advance (radar at most 0.05 below NAIP with defaults) the
      recipe held at all three, but against NAIP read the same way the
      differences were -0.013, -0.048 and +0.025;
    - in-sample: UC Davis, the agribound source with the selection above
      0.485 (235 fields; with the prototype's 4 collects 0.460, the prototype's
      own composite 0.439, NAIP 2022 0.426); Sevilla 0.513 [0.486-0.541]
      (1,128 recintos); Centerfield, the prototype's composite, 0.163 against
      0.360 for NAIP 2021 (262 fields);
    - one Umbra date with Delineate-Anything's defaults: F1 0.00-0.08 at four
      prototype sites;
    - Sentinel-1 (calendar year 2024, defaults): 0.455 on the UC Davis area,
      where Sentinel-2 scored 0.49-0.50; 0.173 in eastern Lea County, New
      Mexico.

This script's run of 2026-10-10 with the development code of agribound 1.1
(the gallery shows it; one-to-one at IoU >= 0.5; Umbra or Capella / NAIP
with the defaults / NAIP with the SAR settings / Sentinel-1 / Sentinel-2):
held out, Texas A&M 0.370 / 0.282 / 0.406 / 0.159 / 0.350,
Intel Ohio 0.261 / 0.193 / 0.310 / 0.132 / 0.394 and
Orleans County 0.274 / 0.261 / 0.249 / 0.035 / 0.281; in-sample,
UC Davis 0.485 / 0.426 / 0.480 / 0.315 / 0.483,
Sevilla 0.513 / - / - / 0.005 / 0.404 and
Centerfield 0.150 / 0.348 / 0.432 / 0.286 / 0.287;
Diamond Valley - / 0.255 / 0.301 / 0.346 / 0.276 (the Sentinel-1,
NAIP-defaults and Sentinel-2 values equal example 24's); with ``--uavsar``,
the UAVSAR Pauli composite 0.432 and NAIP 2018 0.489 (146 fields).

UAVSAR (``--uavsar``; compact; one date). NASA/JPL's UAVSAR flew line
SDelta_15503 over UC Davis on 2018-07-20 (granule ``UAVSAR_GRANULE``). The
script reads its ground-projected covariance (HHHH, HVHV, VVVV and HHVV,
about 5 m) over the UAVSAR pilot's evaluation square (``ucd_2018``: +/-1,500 m
in degrees around 38.5323 N, 121.7848 W, 8.98 km2; DWR / Land IQ 2018
reference) with a 380 m margin, by HTTP range requests (about 10 MB), forms
the Pauli powers |HH - VV|^2 / 2, 2 |HV|^2 and |HH + VV|^2 / 2 as R, G and B
(dB, 2-98 % stretch, bilinear to 5 m UTM, no speckle filter: the pilot's
single-date composite) and runs Delineate-Anything with its defaults on it
(``source="local"``), beside NAIP 2018 (2018-07-15) with the defaults. In the
pilot, which chose its methods on another square (UC Davis was one of its two
test areas, in DWR's labels of FBIS-73M), this composite scored F1 0.417 (146
fields), the Pauli of the line's 6 dates (speckle-filtered, read at 10 m)
0.435, and NAIP 2018 with the defaults, the run repeated here, 0.487 (read at
5 m 0.515, at 10 m 0.521). The script's composite matches the pilot's
(checked on 2026-10-09 from the pilot's data: 18 of 1.7 million values differ,
by 1). It needs an Earthdata login
(https://urs.earthdata.nasa.gov) in ``~/.netrc``, which ``requests`` applies;
the script never reads it.

Data and terms (figures and documentation may show the reference outlines,
the OpenET CLU fields and the Nevada DRI fields included, with these credits):

    - Umbra: Umbra Open Data Program, AWS Registry of Open Data, CC BY 4.0
      (https://umbra.space/open-license). Credit: "SAR data: Umbra Lab, Inc.,
      CC BY 4.0" (agribound's wording; Umbra publishes none), with the note
      of changes (orthorectified, averaged, co-registered, level-normalised
      and median-composited by agribound); every composite carries both;
    - Capella: Capella Space open data, AWS Registry of Open Data, CC BY 4.0
      (https://creativecommons.org/licenses/by/4.0/). Credit: "© Capella
      Space Corp, All Rights Reserved.", with the note of changes (sigma0 in
      dB, averaged, co-registered, level-normalised and median-composited by
      agribound); users must follow US sanctions and export rules;
    - Sentinel-1 and Sentinel-2 (through Earth Engine): "Contains modified
      Copernicus Sentinel data [2022-2025]";
    - NAIP: USDA Farm Production and Conservation Business Center (public
      domain);
    - OpenET agricultural field boundaries, version 2.1.1, Desert Research
      Institute for the OpenET Consortium (2026), with the field boundaries of
      their source agencies (here the CLU of the USDA Farm Service Agency).
      Not public yet: the script reads a local copy in place (``--openet-dir``,
      ``OFB_DATA_DIR`` or ``OPENET_DIR``: the folder of ``TX.zip``, ``OH.zip``
      and ``VT.zip``) and skips these three sites without it;
    - DRI field boundaries (Nevada, 2026-08-20), Desert Research Institute:
      ``examples/NV_Field_Boundaries.zip`` (local data, not in this
      repository; read in place; Diamond Valley is skipped without it);
    - DWR: California Department of Water Resources (2018, 2024). Statewide
      Crop Mapping (with Land IQ), California Natural Resources Agency Open
      Data, https://data.cnra.ca.gov/dataset/statewide-crop-mapping (licence
      not specified); read from DWR's map services;
    - SIGPAC: Fondo Español de Garantía Agraria (FEGA),
      https://sigpac-hubcloud.es/, CC BY 4.0; read from its tile service,
      which serves the previous campaign (2025 on 2026-10-09);
    - Utah WRLU: Utah Division of Water Resources, Water Related Land Use 2025
      (licence not specified); read from its feature service;
    - UAVSAR: NASA/JPL-Caltech, distributed by the Alaska Satellite Facility
      DAAC.

Outputs (``outputs/sar_field_boundaries/<site>/``): the study and evaluation
areas (GeoJSON), the reference fields (``reference.gpkg``), one prediction
layer per run with its provenance record and metrics, and two maps (the radar
runs, and the optical runs, each with the reference and the evaluation area);
``summary.csv`` with one row per run: years, windows, the collects, images or
acquisitions of each composite, reference counts and scores (a run replaces
the rows of the sites it scores and keeps the others, also those of a site
whose reference could not be read this time). agribound
reuses an output whose provenance record matches the run's settings, so a
second run only re-evaluates; ``--overwrite`` recomputes.

Options: ``--only ucd sevilla`` runs some sites; ``--uavsar`` adds the UAVSAR
site (``--only ucd_2018`` runs it alone); ``--dry-run`` reads the references
(map services and local files) and prints the areas, the reference
counts, every configuration and, when the archive index is already in the
cache, the collects the automatic selection picks (it reads the cached index
as it is, never lists the archive); it runs, downloads and writes nothing.
``--vhr-sar-cache-dir`` shares a cache of Umbra and Capella
products between runs.

Runtime: about 51 minutes for the 34 runs with ``--uavsar`` in this script's
run of 2026-10-10 with the development code of agribound 1.1 (Apple M2 Max
with MPS; Earth Engine composites downloaded, the model weights already
cached), in two passes. The first (25 minutes, 1,505 s) made every run but the
Umbra runs at UC Davis, Texas A&M and Intel Ohio, which stopped on a cached
DEM whose record pointed at a cache folder that had been moved (fixed before
agribound 1.1.0); the second (26 minutes, 1,576 s) made those three and reused
the rest. The Umbra and Capella composites took most of the time: 8, 11 and 6
minutes for the three Umbra composites of the second pass, 6 minutes each for
Centerfield's and Orleans County's and under a minute for Sevilla's. Their
products had been downloaded before, by the Umbra prototype, the Capella pilot
and the held-out test, except Centerfield's four (0.29 GB), so a first run
also downloads up to 9.4 GB. The NAIP runs with the defaults took 52 to 69 s
each, download included, those with the SAR settings 6 to 9 s, and the
Sentinel runs 3 to 30 s. Measured on 2026-10-09: ``--dry-run`` of all eight
sites, 34 s.

Prerequisites:
    pip install "agribound[gee,delineate-anything]"
    agribound auth --project YOUR_GEE_PROJECT
    The OpenET field boundaries v2.1.1 (local copy, not public yet):
        --openet-dir DIR or OFB_DATA_DIR=DIR
    Nevada (local data, not in this repository): examples/NV_Field_Boundaries.zip
    --uavsar: an Earthdata login in ~/.netrc
    Run from the repository root:
        python examples/26_sar_field_boundaries.py
"""

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
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
OUTPUT_DIR = Path("outputs/sar_field_boundaries")
USER_AGENT = f"agribound/{agribound.__version__} (+https://github.com/montimaj/agribound)"
MIN_AREA = 2500  # m^2; the predictions' default minimum area, also for the references
IOU_THRESHOLD = 0.5
N_BOOTSTRAP = 1000  # resamples for the F1 intervals, as in the held-out test
BOOTSTRAP_SEED = 42
MERGE_SHARE = 0.25  # merged: the main prediction also covers this share of another field
SPLIT_SHARE = 0.10  # split: at least two predictions each cover this share of the field
#: Delineate-Anything on Umbra and Capella composites, and on NAIP in "naip-sar-settings":
#: the settings documented for these sources (vhr_sar.DELINEATE_ANYTHING_PARAMS).
SAR_SETTINGS = {"processing_gsd_m": 3, "merge_tile_pieces": False}
#: Delineate-Anything on the 10 m composites (Sentinel-1, Sentinel-2), as in examples 23-25.
TEN_M_SETTINGS = {"max_detections": 1000}
IMAGERY = {
    "sentinel1": "Sentinel-1",
    "umbra": "Umbra",
    "capella": "Capella",
    "naip": "NAIP",
    "naip-sar-settings": "NAIP, SAR settings",
    "sentinel2": "Sentinel-2",
    "uavsar": "UAVSAR Pauli",
}
#: Short credit of each imagery source, in the map layer names.
CREDITS = {
    "sentinel1": "contains modified Copernicus Sentinel data",
    "umbra": "SAR data: Umbra Lab, Inc., CC BY 4.0; modified",
    "capella": "© Capella Space Corp, All Rights Reserved; CC BY 4.0; modified",
    "sentinel2": "contains modified Copernicus Sentinel data",
    "uavsar": "UAVSAR: NASA/JPL-Caltech; modified",
}
DATA_ENV = "OFB_DATA_DIR"  # environment variable: the folder of the <ST>.zip files
#: The folder of the OpenET <ST>.zip files when neither --openet-dir nor $OFB_DATA_DIR
#: gives it (for example in the notebook); None: the three OpenET sites are skipped.
OPENET_DIR = None
OPENET_LABEL = "OpenET agricultural field boundaries v2.1.1"
#: CDL classes that are not agricultural: an OpenET field with one of these as its crop
#: code of the season year (CROP_<year>) is left out, as in the held-out test.
NON_AGRICULTURAL_CDL = frozenset(
    [63, 64, 65, 81, 82, 83, 87, 88, 92, 111, 112, 121, 122, 123, 124, 131, 141, 142, 143]
    + [152, 190, 195]
)
NV_ZIP = "examples/NV_Field_Boundaries.zip"
NV_MEMBER = "nevada_agricultural_field_boundaries_20260820.gpkg"
NV_LABEL = "DRI field boundaries (Nevada, 2026-08-20)"
#: DWR / Land IQ Statewide Crop Mapping map services (layer 0) by year.
DWR_LAYERS = {
    2018: "https://utility.arcgis.com/usrsvcs/servers/83f2f37783a64323acc4c5b68e88c0c8/rest/"
    "services/Planning/i15_Crop_Mapping_2018/MapServer/0",
    2024: "https://utility.arcgis.com/usrsvcs/servers/39b63601dfb34274899d15a13465644e/rest/"
    "services/Planning/i15_Crop_Mapping_2024_Provisional/MapServer/0",
}
DWR_NON_FIELD = {"U", "UL", "UR", "UC", "UI", "UV", "NC", "NR", "NV", "NW", "NB", "NS"}
WRLU_2025_LAYER = (
    "https://services.arcgis.com/ZzrwjTRez6FJiOq4/arcgis/rest/services/"
    "WaterRelatedLandUse_2025/FeatureServer/27"
)
#: SIGPAC recintos of the previous campaign ("anterior"), which moves on every year.
SIGPAC_TILE = "https://sigpac-hubcloud.es/mvt/anterior/recinto@3857@geojson/15/{x}/{y}.geojson"
#: The campaign of those tiles on 2026-10-09 (the reference counts and scores above). The
#: service generates a campaign's tiles when it opens, so the year of their Last-Modified
#: date is the campaign (2025-02-10 for the previous campaign, 2026-01-19 for the current).
SIGPAC_CAMPAIGN = 2025
SIGPAC_KEY = ["provincia", "municipio", "agregado", "zona", "poligono", "parcela", "recinto"]
#: SIGPAC land uses that are not fields (roads, buildings, water, urban, forest, scrub,
#: pasture, ...), left out of the reference as in example 23.
SIGPAC_NON_FIELD = {
    "CA", "IM", "ED", "AG", "ZU", "FO", "MT", "ZV", "ZC", "EP", "IS", "PS", "PR", "PA",
}  # fmt: skip
UAVSAR_URL = "https://cumulus.asf.earthdatacloud.nasa.gov/PRODUCTS/UAVSAR/UAVSAR_POLSAR"
UAVSAR_GRANULE = "SDelta_15503_18034_011_180720_L090_CX_01"  # 2018-07-20, line SDelta_15503
UAVSAR_SITE = "ucd_2018"
UAVSAR_CENTRE = (-121.7848, 38.5323)  # lon, lat of the UAVSAR pilot's square
UAVSAR_HALF_SIDE_M = 1500  # in degrees, at 111,320 m per degree of latitude (the pilot's)
UAVSAR_MARGIN_M = 380  # the pilot's processing margin around the square
UAVSAR_RES_M = 5.0
UAVSAR_EPSG = 32610

# The sites: the radar window (inclusive), the Umbra or Capella source and the collects
# that its automatic selection picked from the index of 2026-10-09 (ID prefixes), the
# NAIP year (None: no NAIP), the reference and the UTM zone of the areas in AREA_RINGS.
SITES = {
    "ucd": {
        "name": "UC Davis, Yolo and Solano counties, California",
        "role": "in-sample",
        "epsg": 32610,
        "window": ("2024-06-01", "2024-09-15"),
        "vhr": "umbra",
        "collects": ("85d4116b", "34466d35", "2b41c551", "4504e595", "a11ead0b", "8b5e8679"),
        "naip": 2022,
        "reference": ("dwr", 2024),
    },
    "tamu": {
        "name": "Texas A&M AgriLife farm, Burleson County, Texas",
        "role": "held-out",
        "epsg": 32614,
        "window": ("2024-04-01", "2024-08-15"),
        "vhr": "umbra",
        "collects": ("0cf6b66c", "1d5c7920", "2f7ff598", "2421a2ca", "ea47cc82", "1ead5424"),
        "naip": 2022,
        "reference": ("openet", "TX"),
    },
    "intel_oh": {
        "name": "Intel Ohio One site, Licking County, Ohio",
        "role": "held-out",
        "epsg": 32617,
        "window": ("2024-06-01", "2024-09-15"),
        "vhr": "umbra",
        "collects": ("033cf428", "8f9d6d5f", "e3932541", "ec6984d5"),
        "naip": 2023,
        "reference": ("openet", "OH"),
    },
    "orleans_vt": {
        "name": "Orleans County, Vermont",
        "role": "held-out",
        "epsg": 32618,
        "window": ("2022-06-01", "2022-09-15"),
        "vhr": "capella",
        "collects": ("4d76ec79", "bacbd224"),
        "naip": 2023,
        "reference": ("openet", "VT"),
    },
    "sevilla": {
        "name": "Sevilla, Andalusia, Spain",
        "role": "in-sample",
        "epsg": 32629,
        "window": ("2023-01-01", "2023-01-31"),
        "vhr": "capella",
        "collects": ("8486abef", "cadc5c74"),
        "naip": None,
        "reference": ("sigpac", None),
    },
    "centerfield": {
        "name": "Centerfield, Sanpete County, Utah",
        "role": "in-sample",
        "epsg": 32612,
        "window": ("2025-05-15", "2025-09-15"),
        "vhr": "umbra",
        "collects": ("13852260", "584f996f", "272a73f9", "2120d962"),
        "naip": 2021,
        "reference": ("wrlu", 2025),
    },
    "diamond_valley": {
        "name": "Diamond Valley, Eureka County, Nevada",
        "role": "radar: Sentinel-1 only",
        "epsg": 32611,
        "window": ("2024-04-01", "2024-10-31"),
        "square": (584000, 4389000),  # example 24's evaluation square (lower-left corner)
        "vhr": None,
        "naip": 2022,
        "reference": ("dri", None),
    },
    UAVSAR_SITE: {
        "name": "UC Davis, Yolo and Solano counties, California (UAVSAR, 2018)",
        "role": "UAVSAR pilot test area",
        "epsg": UAVSAR_EPSG,
        "window": ("2018-07-20", "2018-07-20"),
        "vhr": None,
        "naip": 2018,
        "reference": ("dwr", 2018),
    },
}
MAIN_SITES = [slug for slug in SITES if slug != UAVSAR_SITE]

# The study areas (and, at the held-out sites, the evaluation areas "<site>_eval"):
# exterior rings in the site's UTM zone (SITES[...]["epsg"]), metres rounded to 0.1 m.
AREA_RINGS = {
    "ucd": (
        "603140.9 4265800.4, 603188.2 4265879.0, 603240.9 4265956.4, 603312.2 4266075.0, "
        "603476.9 4266328.5, 603508.2 4266383.1, 603692.2 4266675.2, 603768.9 4266784.7, "
        "603804.2 4266851.2, 603932.9 4267052.7, 603948.2 4267083.3, 604008.2 4267167.3, "
        "604028.9 4267188.8, 604120.2 4267323.4, 604152.9 4267364.8, 604196.2 4267431.4, "
        "604228.9 4267468.9, 604256.2 4267515.4, 604280.2 4267551.4, 604292.9 4267564.9, "
        "604320.2 4267611.4, 604343.7 4267642.0, 606145.0 4268356.7, 606299.3 4268256.8, "
        "606417.9 4268185.5, 606463.3 4268152.8, 606517.9 4268121.5, 606611.3 4268052.8, "
        "606661.9 4268025.5, 606739.3 4267972.7, 606957.9 4267841.4, 607011.3 4267800.7, "
        "607077.9 4267761.4, 607143.3 4267716.7, 607225.9 4267669.4, 607317.9 4267605.4, "
        "607347.3 4267576.6, 607425.9 4267541.4, 607459.3 4267516.6, 607489.9 4267501.3, "
        "607520.9 4267474.4, 607577.0 4267414.3, 607679.3 4267340.6, 607733.9 4267309.3, "
        "607801.0 4267254.3, 607865.9 4267213.3, 607907.3 4267180.5, 607933.9 4267163.0, "
        "607961.9 4267149.2, 607991.0 4267128.0, 608700.3 4265340.3, 608636.8 4265263.9, "
        "608595.1 4265205.6, 608507.8 4265063.0, 608475.1 4265017.5, 608443.8 4264963.0, "
        "608415.1 4264921.5, 608373.5 4264850.9, 608351.8 4264806.9, 608275.1 4264697.4, "
        "608243.8 4264642.9, 608195.1 4264573.4, 608167.8 4264526.8, 608107.1 4264437.3, "
        "608051.8 4264346.8, 607987.1 4264249.3, 607907.8 4264118.7, 607875.1 4264077.2, "
        "607867.8 4264062.7, 607855.1 4264049.2, 607831.8 4264010.7, 607783.1 4263961.2, "
        "607755.8 4263914.7, 607691.1 4263829.2, 607465.6 4263502.7, 605681.6 4262794.9, "
        "605166.1 4263124.5, 604896.7 4263301.3, 604862.1 4263320.6, 604488.7 4263561.4, "
        "604310.1 4263672.7, 604280.7 4263697.4, 604238.1 4263724.7, 603856.7 4263993.5"
    ),
    "tamu": (
        "748439.3 3380523.8, 747898.6 3378765.4, 746140.6 3378866.4, 745945.4 3378887.9, "
        "745765.2 3378925.6, 745667.4 3378954.5, 744006.9 3379466.4, 744107.9 3381224.6, "
        "744129.4 3381419.7, 744167.1 3381600.0, 744196.0 3381697.8, 744707.7 3383358.0, "
        "746466.0 3383257.1, 746661.2 3383235.6, 746841.4 3383197.9, 746939.2 3383169.0, "
        "748599.5 3382657.2, 748498.6 3380899.2, 748477.1 3380704.0"
    ),
    "tamu_eval": (
        "745213.2 3383329.0, 746466.0 3383257.1, 746661.2 3383235.6, 746841.4 3383197.9, "
        "746939.2 3383169.0, 748570.0 3382666.3, 748570.0 3382143.4, 748498.6 3380899.2, "
        "748477.1 3380704.0, 748439.3 3380523.8, 747907.7 3378795.0, 747384.0 3378795.0, "
        "746140.6 3378866.4, 745945.4 3378887.9, 745765.2 3378925.6, 745667.4 3378954.5, "
        "744036.0 3379457.4, 744036.0 3379972.6, 744107.9 3381224.6, 744129.4 3381419.7, "
        "744167.1 3381600.0, 744196.0 3381697.8, 744698.8 3383329.0"
    ),
    "intel_oh": (
        "355681.4 4443848.5, 356684.2 4442895.5, 356684.2 4441715.7, 355787.0 4440730.2, "
        "354842.8 4439736.9, 353663.1 4439736.9, 352677.4 4440634.3, 351684.2 4441578.3, "
        "351684.2 4442758.1, 352581.6 4443743.7, 353525.7 4444736.9, 354705.4 4444736.9"
    ),
    "intel_oh_eval": (
        "351714.0 4442790.8, 352581.6 4443743.7, 353497.3 4444707.0, 354738.3 4444707.0, "
        "355681.4 4443848.5, 356655.0 4442923.2, 356655.0 4441683.6, 355787.0 4440730.2, "
        "354870.5 4439766.0, 353631.1 4439766.0, 352677.4 4440634.3, 351714.0 4441550.0"
    ),
    "orleans_vt": (
        "716238.6 4974410.7, 716238.6 4975464.7, 718791.4 4976836.5, 719947.7 4976836.5, "
        "721238.6 4974230.3, 721238.6 4973170.7, 718637.0 4971836.5, 717513.7 4971836.5"
    ),
    "orleans_vt_eval": (
        "719962.3 4976807.0, 721209.0 4974290.1, 721209.0 4973155.5, 718694.4 4971866.0, "
        "717499.1 4971866.0, 716268.0 4974351.3, 716268.0 4975480.5, 718736.5 4976807.0"
    ),
    "sevilla": (
        "760220.5 4085913.7, 760310.8 4086360.0, 760402.7 4086754.1, 760466.2 4087043.2, "
        "760621.6 4087706.9, 760683.4 4087958.1, 760742.2 4088165.3, 760785.3 4088295.9, "
        "760821.3 4088424.5, 760868.9 4088675.1, 760893.2 4088866.2, 760960.4 4089172.8, "
        "760994.2 4089310.6, 761040.4 4089474.4, 761084.9 4089602.3, 761110.1 4089686.3, "
        "761131.4 4089776.9, 761179.7 4089954.8, 761281.9 4090421.0, 761324.6 4090652.6, "
        "766034.0 4089576.3, 765983.8 4089337.9, 765809.4 4088560.1, 765677.8 4087949.6, "
        "765637.2 4087759.2, 765594.6 4087531.7, 765537.2 4087253.1, 765437.0 4086812.7, "
        "765372.0 4086499.9, 765253.0 4085986.0, 765198.5 4085796.0, 765176.6 4085706.6, "
        "765090.6 4085294.4, 765015.0 4084951.3, 764983.4 4084824.9"
    ),
    "centerfield": (
        "426513.7 4324330.9, 426516.0 4324387.5, 426519.9 4324399.5, 426520.0 4324451.5, "
        "426523.9 4324463.5, 426524.0 4324523.6, 426527.9 4324535.5, 426532.0 4324659.6, "
        "426535.9 4324671.6, 426536.0 4324727.6, 426539.9 4324739.6, 426540.0 4324791.7, "
        "426543.9 4324803.6, 426544.0 4324859.7, 426547.9 4324871.7, 426548.0 4324923.7, "
        "426551.9 4324935.7, 426552.0 4324991.7, 426555.9 4325003.7, 426556.0 4325059.8, "
        "426559.9 4325071.7, 426560.0 4325131.8, 426563.9 4325143.8, 426568.0 4325267.8, "
        "426571.9 4325279.8, 426572.0 4325335.9, 426575.9 4325347.8, 426576.0 4325399.9, "
        "426579.9 4325411.9, 426580.0 4325467.9, 426583.9 4325479.9, 426584.0 4325535.9, "
        "426587.9 4325547.9, 426588.0 4325600.0, 426591.9 4325611.9, 426592.0 4325664.0, "
        "426596.0 4325678.9, 426596.0 4325732.0, 426599.9 4325744.0, 426600.0 4325796.0, "
        "426603.9 4325808.0, 426608.0 4325932.1, 426611.9 4325944.0, 426612.0 4325984.1, "
        "426615.9 4325996.1, 426616.0 4326032.1, 426619.9 4326044.1, 426616.5 4326065.0, "
        "426619.9 4326076.1, 426620.0 4326124.2, 426623.9 4326136.1, 426624.0 4326196.2, "
        "426627.9 4326208.1, 426628.0 4326264.2, 426631.9 4326276.2, 426632.0 4326332.2, "
        "426635.9 4326344.2, 426640.0 4326460.3, 426643.9 4326472.2, 426644.0 4326532.3, "
        "426647.9 4326544.3, 426652.0 4326655.6, 427435.4 4328550.7, 427563.1 4328546.0, "
        "427578.0 4328541.9, 427631.1 4328541.9, 427646.0 4328537.9, 427699.1 4328537.9, "
        "427711.1 4328534.1, 427831.1 4328529.9, 427843.1 4328526.1, 427899.1 4328525.9, "
        "427911.1 4328522.1, 428031.1 4328517.9, 428043.1 4328514.1, 428095.1 4328513.9, "
        "428107.1 4328510.1, 428163.1 4328509.9, 428175.1 4328506.1, 428227.1 4328505.9, "
        "428239.1 4328502.1, 428291.1 4328501.9, 428306.0 4328497.9, 428359.1 4328497.9, "
        "428371.1 4328494.1, 428423.1 4328493.9, 428435.1 4328490.1, 428487.1 4328489.9, "
        "428499.1 4328486.1, 428555.1 4328485.9, 428567.1 4328482.1, 428619.1 4328481.9, "
        "428631.1 4328478.1, 428683.1 4328477.9, 428695.1 4328474.1, 428747.1 4328473.9, "
        "428762.0 4328469.9, 428811.1 4328469.9, 428823.1 4328466.1, 428875.1 4328465.9, "
        "428887.1 4328462.1, 428943.1 4328461.9, 428955.1 4328458.1, 429039.1 4328453.9, "
        "429051.1 4328450.1, 429087.1 4328449.9, 429099.1 4328446.1, 429199.1 4328445.9, "
        "429211.1 4328442.1, 429259.1 4328441.9, 429271.1 4328438.1, 429327.1 4328437.9, "
        "429339.1 4328434.1, 429387.1 4328433.9, 429399.1 4328430.1, 429455.1 4328429.9, "
        "429467.1 4328426.1, 429515.1 4328425.9, 429527.1 4328422.0, 429579.1 4328421.9, "
        "429591.1 4328418.0, 429643.1 4328417.9, 429658.0 4328413.9, 429707.1 4328413.9, "
        "429723.5 4328409.9, 431596.0 4327635.9, 431596.0 4327586.5, 431592.0 4327571.6, "
        "431592.0 4327506.5, 431588.0 4327491.6, 431588.0 4327438.4, 431584.1 4327426.5, "
        "431584.0 4327382.4, 431580.0 4327367.5, 431580.0 4327250.4, 431576.1 4327238.4, "
        "431576.0 4327174.3, 431572.1 4327162.4, 431572.0 4327130.3, 431568.6 4327121.3, "
        "431568.0 4327090.3, 431564.1 4327078.3, 431564.0 4327046.3, 431560.1 4327034.3, "
        "431560.0 4326994.3, 431556.1 4326982.3, 431556.0 4326950.3, 431552.1 4326938.3, "
        "431552.0 4326898.2, 431548.1 4326886.3, 431548.0 4326834.2, 431544.0 4326819.3, "
        "431544.0 4326754.2, 431540.1 4326742.2, 431540.0 4326702.2, 431536.0 4326687.3, "
        "431536.0 4326566.1, 431532.1 4326554.2, 431528.0 4326450.1, 431524.1 4326438.1, "
        "431520.0 4326346.0, 431516.1 4326334.1, 431516.0 4326298.0, 431512.1 4326286.1, "
        "431512.0 4326242.0, 431508.1 4326230.0, 431508.0 4326182.0, 431504.1 4326170.0, "
        "431504.0 4326130.0, 431500.1 4326118.0, 431500.0 4326057.9, 431496.1 4326046.0, "
        "431496.0 4325997.9, 431492.1 4325985.9, 431488.0 4325885.9, 431484.1 4325873.9, "
        "431480.0 4325761.8, 431476.1 4325749.9, 431472.0 4325665.8, 431468.1 4325653.8, "
        "431468.0 4325593.8, 431464.1 4325581.8, 431464.0 4325529.7, 431460.0 4325514.8, "
        "431460.0 4325433.7, 431456.1 4325421.7, 431456.0 4325365.7, 431452.0 4325350.8, "
        "431452.0 4325289.6, 431448.1 4325277.7, 431448.0 4325241.6, 431444.1 4325229.7, "
        "431444.0 4325166.5, 430713.8 4323400.0, 430620.9 4323404.0, 430608.9 4323407.9, "
        "430560.9 4323408.1, 430548.9 4323411.9, 430496.9 4323412.1, 430484.9 4323415.9, "
        "430428.9 4323416.1, 430416.9 4323419.9, 430364.9 4323420.1, 430352.9 4323423.9, "
        "430304.9 4323424.1, 430292.9 4323427.9, 430240.9 4323428.1, 430228.9 4323431.9, "
        "430176.9 4323432.1, 430164.9 4323435.9, 430108.9 4323436.1, 430096.9 4323439.9, "
        "430048.9 4323440.1, 430036.9 4323443.9, 429980.9 4323444.1, 429968.9 4323447.9, "
        "429848.9 4323452.1, 429836.9 4323455.9, 429732.9 4323460.1, 429720.9 4323463.9, "
        "429604.9 4323468.1, 429592.9 4323471.9, 429536.9 4323472.1, 429524.9 4323475.9, "
        "429472.9 4323476.1, 429460.9 4323479.9, 429404.9 4323480.1, 429392.9 4323483.9, "
        "429336.9 4323484.1, 429324.9 4323487.9, 429272.9 4323488.1, 429260.9 4323491.9, "
        "429204.9 4323492.1, 429192.9 4323495.9, 429136.9 4323496.1, 429124.9 4323499.9, "
        "429068.9 4323500.1, 429056.9 4323503.9, 429004.9 4323504.1, 428992.9 4323507.9, "
        "428936.9 4323508.1, 428924.9 4323511.9, 428872.9 4323512.1, 428860.9 4323515.9, "
        "428808.9 4323516.1, 428796.9 4323519.9, 428744.9 4323520.1, 428732.9 4323523.9, "
        "428676.9 4323524.1, 428664.9 4323528.0, 428612.9 4323528.1, 428600.9 4323532.0, "
        "428544.9 4323532.1, 428532.9 4323536.0, 428422.7 4323540.1"
    ),
}


def ring_polygon(text):
    """A shapely Polygon from an ``"x y, x y, ..."`` ring."""
    from shapely.geometry import Polygon

    return Polygon([tuple(map(float, point.split())) for point in text.split(",")])


def site_areas(slug):
    """The site's study area and evaluation area (shapely, in its UTM zone)."""
    from shapely.geometry import box

    if slug == UAVSAR_SITE:
        return uavsar_areas()
    if "square" in SITES[slug]:
        x, y = SITES[slug]["square"]
        square = box(x, y, x + 5000, y + 5000)
        return square, square
    study = ring_polygon(AREA_RINGS[slug])
    evaluation = AREA_RINGS.get(f"{slug}_eval")
    return study, ring_polygon(evaluation) if evaluation else study


def to_4326(geometry, epsg):
    """*geometry* (in *epsg*) in EPSG:4326."""
    import geopandas as gpd

    return gpd.GeoSeries([geometry], crs=epsg).to_crs(4326).iloc[0]


def area_geojson(geometry, epsg, name):
    """A GeoJSON FeatureCollection of *geometry* in EPSG:4326."""
    from shapely.geometry import mapping

    geometry = mapping(to_4326(geometry, epsg))
    feature = {"type": "Feature", "geometry": geometry, "properties": {"name": name}}
    return {"type": "FeatureCollection", "features": [feature]}


def rep_point_inside(gdf, area):
    """Rows whose representative point lies inside *area* (same CRS)."""
    return gdf[gdf.representative_point().within(area).to_numpy()].copy()


def http_response(url, data=None, timeout=120, retries=3):
    """GET (or POST *data*) *url* with a descriptive User-Agent: ``(body, headers)``."""
    request = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), response.headers
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def http_get(url, data=None, timeout=120, retries=3):
    """The body (bytes) of :func:`http_response`."""
    return http_response(url, data, timeout, retries)[0]


def arcgis_query(layer, area, epsg, where, fields):
    """Features of an ArcGIS map or feature service *layer* near *area*, in *epsg*."""
    import geopandas as gpd
    import shapely

    west, south, east, north = to_4326(area.buffer(200), epsg).bounds
    features = []
    for _ in range(100):  # pages of up to 1,000 features
        query = {
            "where": where,
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": fields,
            "outSR": 4326,
            "f": "geojson",
            "resultOffset": len(features),
            "resultRecordCount": 1000,
        }
        page = json.loads(http_get(f"{layer}/query", data=urllib.parse.urlencode(query).encode()))
        if "error" in page:
            raise RuntimeError(f"{layer}: {page['error']}")
        features += page.get("features", [])
        exceeded = page.get("exceededTransferLimit") or page.get("properties", {}).get(
            "exceededTransferLimit"
        )
        if not exceeded or not page.get("features"):
            break
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=epsg)
    gdf = gpd.GeoDataFrame.from_features(features, crs=4326)
    gdf["geometry"] = shapely.force_2d(gdf.geometry.to_numpy())
    return gdf.to_crs(epsg)


def sigpac_campaigns(last_modified):
    """The campaigns (years) of SIGPAC tiles from their Last-Modified dates (see
    SIGPAC_CAMPAIGN); missing dates are skipped."""
    from email.utils import parsedate_to_datetime

    return sorted({parsedate_to_datetime(date).year for date in last_modified if date})


def load_sigpac(area, epsg):
    """SIGPAC recintos near *area* without the non-field land uses, in *epsg*, and the
    campaign of the tiles (a label such as "2025"); warns when it is not SIGPAC_CAMPAIGN."""
    import geopandas as gpd
    from shapely.geometry import shape

    west, south, east, north = to_4326(area, epsg).bounds
    n = 2**15

    def tile_x(lon):
        return int((lon + 180) / 360 * n)

    def tile_y(lat):
        rad = math.radians(lat)
        return int((1 - math.log(math.tan(rad) + 1 / math.cos(rad)) / math.pi) / 2 * n)

    # One extra tile on every side: a tile also returns parcels outside its footprint.
    tiles = [
        (x, y)
        for x in range(tile_x(west) - 1, tile_x(east) + 2)
        for y in range(tile_y(north) - 1, tile_y(south) + 2)
    ]
    with ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(lambda t: http_response(SIGPAC_TILE.format(x=t[0], y=t[1])), tiles))
    campaigns = sigpac_campaigns([headers.get("Last-Modified") for _, headers in replies])
    campaign = "/".join(map(str, campaigns)) or "unknown"
    if campaigns != [SIGPAC_CAMPAIGN]:
        print(
            f"  WARNING: SIGPAC now serves the {campaign} campaign as the previous one, not "
            f"{SIGPAC_CAMPAIGN} (2026-10-09): the Sevilla reference and scores are not those "
            "of the docstring"
        )
    rows = [
        {**feature["properties"], "geometry": shape(feature["geometry"])}
        for page, _ in replies
        for feature in json.loads(page).get("features", [])
    ]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=3857).drop_duplicates(SIGPAC_KEY)
    return gdf[~gdf["uso_sigpac"].isin(SIGPAC_NON_FIELD)].to_crs(epsg), campaign


def openet_dir(args):
    """The folder of the OpenET <ST>.zip files, or None (--openet-dir, $OFB_DATA_DIR)."""
    folder = args.openet_dir or os.environ.get(DATA_ENV) or OPENET_DIR
    return Path(folder).expanduser() if folder else None


def read_in_place(path, area, epsg, columns):
    """The features of the GDAL *path* that reach into *area*'s bounding box, in *epsg*."""
    import geopandas as gpd
    import pyogrio

    file_crs = pyogrio.read_info(path)["crs"]
    bbox = tuple(gpd.GeoSeries([area], crs=epsg).to_crs(file_crs).total_bounds)
    return pyogrio.read_dataframe(path, bbox=bbox, columns=columns).to_crs(epsg)


def load_reference(slug, area, epsg, args):
    """The reference fields of a site: ``(fields, n_small, label)`` (see the docstring).

    Raises FileNotFoundError when local reference data are missing.
    """
    import geopandas as gpd
    import shapely

    kind, key = SITES[slug]["reference"]
    year = int(SITES[slug]["window"][0][:4])
    if kind == "dwr":
        gdf = arcgis_query(DWR_LAYERS[key], area, epsg, "1=1", "UniqueID,CLASS2,CROPTYP2")
        gdf = gdf[~gdf["CLASS2"].astype(str).str.strip().isin(DWR_NON_FIELD)]
        label = f"DWR / Land IQ crop mapping {key}" + (" (provisional)" if key == 2024 else "")
    elif kind == "wrlu":
        where = "Landuse = 'Agricultural'"
        gdf = arcgis_query(WRLU_2025_LAYER, area, epsg, where, "OBJECTID,Landuse,Description")
        label = f"Utah WRLU {key}, agricultural"
    elif kind == "sigpac":
        gdf, campaign = load_sigpac(area, epsg)
        label = f"SIGPAC recintos ({campaign} campaign)"
    elif kind == "openet":
        folder = openet_dir(args)
        archive = folder / f"{key}.zip" if folder else None
        if archive is None or not archive.is_file():
            raise FileNotFoundError(
                f"{archive or f'{key}.zip'} not found: the {OPENET_LABEL} are not public yet; "
                f"pass --openet-dir DIR or set {DATA_ENV}=DIR (DIR holds <ST>.zip per state)"
            )
        columns = ["Unique_ID", "SOURCECODE", "MOD_DATE", f"CROP_{year}"]
        gdf = read_in_place(f"/vsizip/{archive}/{key}.shp", area, epsg, columns)
        gdf = gdf[~gdf[f"CROP_{year}"].isin(NON_AGRICULTURAL_CDL)]
        label = f"{OPENET_LABEL} ({key}), agricultural in {year}"
    else:  # the Nevada DRI fields
        if not Path(NV_ZIP).is_file():
            raise FileNotFoundError(
                f"{NV_ZIP} not found: it holds the {NV_LABEL} (local data, not in this "
                "repository; contact the author for access)"
            )
        gdf = read_in_place(f"/vsizip/{NV_ZIP}/{NV_MEMBER}", area, epsg, ["NWI_ID", "SOURCECODE"])
        gdf, label = gdf[gdf["SOURCECODE"] == "DRI"], f"{NV_LABEL}, DRI-mapped"
    # Valid outlines (the polygonal parts of each, joined: a multi-part outline stays one
    # field), selected by representative point; those under MIN_AREA are dropped and counted.
    polys = []
    for geom in shapely.make_valid(gdf.geometry.to_numpy()):
        parts = shapely.get_parts(geom)
        parts = parts[shapely.area(parts) > 0]
        polys.append(shapely.union_all(parts) if len(parts) else None)
    gdf = gpd.GeoDataFrame(gdf.drop(columns="geometry"), geometry=polys, crs=epsg)
    gdf = rep_point_inside(gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty], area)
    keep = (gdf.area >= MIN_AREA).to_numpy()
    return gdf[keep].reset_index(drop=True), int((~keep).sum()), label


def reference_note(ref):
    """The OpenET sources and MOD_DATE range of the reference fields ("" otherwise)."""
    import pandas as pd

    if "SOURCECODE" not in ref or "MOD_DATE" not in ref or not len(ref):
        return ""
    shares = ref["SOURCECODE"].fillna("(none)").value_counts(normalize=True)
    note = ", ".join(f"{code} {round(100 * share)} %" for code, share in shares.items())
    dates = pd.to_datetime(ref["MOD_DATE"].dropna())
    if len(dates):
        note += f"; MOD_DATE {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"
    return note


def uavsar_areas():
    """The UAVSAR pilot's processing area (study area) and square (evaluation area)."""
    import geopandas as gpd
    from shapely.geometry import box

    lon, lat = UAVSAR_CENTRE
    dlat = UAVSAR_HALF_SIDE_M / 111320.0
    dlon = dlat / math.cos(math.radians(lat))
    square = box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    square = gpd.GeoSeries([square], crs=4326).to_crs(UAVSAR_EPSG).iloc[0]
    step, margin = UAVSAR_RES_M, UAVSAR_MARGIN_M
    x0, y0, x1, y1 = square.bounds
    x0, y0 = (math.floor((v - margin) / step) * step for v in (x0, y0))
    x1, y1 = (math.ceil((v + margin) / step) * step for v in (x1, y1))
    return box(x0, y0, x1, y1), square


def uavsar_window(bounds):
    """HHHH, HVHV, VVVV and Re(HHVV) of UAVSAR_GRANULE over lon/lat *bounds*, with their
    affine transform (EPSG:4326), read from the flat GRD files by HTTP range requests.

    The Earthdata login comes from ~/.netrc: requests applies it on the redirect to
    urs.earthdata.nasa.gov; this function never reads it.
    """
    import re

    import numpy as np
    import requests
    from rasterio.transform import Affine

    login = requests.Session()  # trust_env: ~/.netrc for the login redirect
    signed = requests.Session()
    signed.trust_env = False  # the signed download URLs need no credentials
    for session in (login, signed):
        session.headers["User-Agent"] = USER_AGENT
    base = f"{UAVSAR_URL}/{UAVSAR_GRANULE}/"
    response = login.get(f"{base}{UAVSAR_GRANULE}.ann", timeout=120)
    response.raise_for_status()
    ann = {}
    for line in response.text.splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith(";"):
            ann[re.sub(r"\(.*\)", "", key).strip().lower()] = value.split(";")[0].strip()
    rows, cols = (int(float(ann[f"grd_pwr.set_{k}"])) for k in ("rows", "cols"))
    lat0, lon0, dlat, dlon = (
        float(ann[f"grd_pwr.{k}"]) for k in ("row_addr", "col_addr", "row_mult", "col_mult")
    )
    west, south, east, north = bounds  # the addresses are pixel centres; dlat < 0
    r0 = max(math.floor((north - lat0) / dlat), 0)
    r1 = min(math.ceil((south - lat0) / dlat) + 1, rows)
    c0 = max(math.floor((west - lon0) / dlon), 0)
    c1 = min(math.ceil((east - lon0) / dlon) + 1, cols)

    def read(pol, dtype):
        name = UAVSAR_GRANULE.replace("_L090_", f"_L090{pol}_") + ".grd"
        resolved = login.get(base + name, headers={"Range": "bytes=0-0"}, timeout=120)
        resolved.raise_for_status()  # its URL is the signed download URL, after the login
        size = np.dtype(dtype).itemsize

        def row(r):
            start = (r * cols + c0) * size
            span = {"Range": f"bytes={start}-{start + (c1 - c0) * size - 1}"}
            for attempt in range(3):
                try:
                    reply = signed.get(resolved.url, headers=span, timeout=120)
                    reply.raise_for_status()
                    return np.frombuffer(reply.content, dtype)
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))

        with ThreadPoolExecutor(16) as pool:
            return np.stack(list(pool.map(row, range(r0, r1))))

    bands = [read(pol, "<f4") for pol in ("HHHH", "HVHV", "VVVV")] + [read("HHVV", "<c8").real]
    transform = Affine(dlon, 0, lon0 + (c0 - 0.5) * dlon, 0, dlat, lat0 + (r0 - 0.5) * dlat)
    return np.stack(bands).astype(np.float64), transform


def pauli_composite(path, bands, transform, area):
    """Write the Pauli RGB (uint8, 0 = nodata) of *bands* on the 5 m grid of *area*.

    *bands*: HHHH, HVHV, VVVV and Re(HHVV) on the lon/lat grid of *transform*. R, G and B
    are |HH - VV|^2 / 2, 2 |HV|^2 and |HH + VV|^2 / 2, bilinear to UTM, in dB, stretched
    from their 2nd to their 98th percentile to 1-255.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    hh, hv, vv, hhvv = bands
    pauli = np.stack([(hh + vv - 2 * hhvv) / 2, 2 * hv, (hh + vv + 2 * hhvv) / 2])
    pauli[:, ~np.isfinite(bands).all(axis=0) | (hh == 0)] = np.nan  # outside the swath
    x0, y0, x1, y1 = area.bounds
    shape = (round((y1 - y0) / UAVSAR_RES_M), round((x1 - x0) / UAVSAR_RES_M))
    grid = from_origin(x0, y1, UAVSAR_RES_M, UAVSAR_RES_M)
    rgb = np.zeros((3, *shape), np.uint8)
    warp = {"src_transform": transform, "src_crs": "EPSG:4326", "src_nodata": np.nan}
    warp.update(dst_transform=grid, dst_crs=f"EPSG:{UAVSAR_EPSG}", dst_nodata=np.nan)
    for i in range(3):
        power = np.full(shape, np.nan, np.float32)
        reproject(pauli[i].astype(np.float32), power, resampling=Resampling.bilinear, **warp)
        db = 10 * np.log10(np.maximum(power, 1e-10))
        valid = np.isfinite(power)
        lo, hi = np.percentile(db[valid], [2, 98])
        scaled = np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1) * 254 + 1
        rgb[i] = np.where(valid, scaled, 0).astype(np.uint8)
    profile = {"driver": "GTiff", "width": shape[1], "height": shape[0], "count": 3}
    profile.update(dtype="uint8", crs=f"EPSG:{UAVSAR_EPSG}", transform=grid, nodata=0)
    with rasterio.open(path, "w", compress="deflate", **profile) as dst:
        dst.write(rgb)
        dst.update_tags(
            AGRIBOUND_ATTRIBUTION="UAVSAR data: NASA/JPL-Caltech, via the ASF DAAC",
            UAVSAR_GRANULE=UAVSAR_GRANULE,
            PAULI="R=|HH-VV|^2/2, G=2|HV|^2, B=|HH+VV|^2/2; dB, 2-98 % stretch",
        )


def run_configs(slug, args):
    """``(label, AgriboundConfig)`` of every run of one site (see the docstring)."""
    from agribound.config import AgriboundConfig

    site, out_dir = SITES[slug], OUTPUT_DIR / slug
    start, end = site["window"]
    season = {"year": int(start[:4]), "date_range": (start, end)}
    naip = {"source": "naip", "year": site["naip"]}
    if slug == UAVSAR_SITE:  # the Pauli composite (written by pauli_composite) and NAIP
        tif = str(out_dir / f"uavsar_pauli_{start}.tif")
        local = {"local_tif_path": tif, "bands": {"R": 1, "G": 2, "B": 3}}
        runs = [("uavsar", {"source": "local", "year": season["year"], **local}), ("naip", naip)]
    else:
        ten_m = {**season, "engine_params": TEN_M_SETTINGS}
        runs = [("sentinel1", {"source": "sentinel1", **ten_m})]
        if site["vhr"]:
            cache = {"vhr_sar_cache_dir": args.vhr_sar_cache_dir}
            vhr = {"source": site["vhr"], **season, "engine_params": SAR_SETTINGS, **cache}
            runs.append((site["vhr"], vhr))
        if site["naip"]:
            runs += [("naip", naip), ("naip-sar-settings", {**naip, "engine_params": SAR_SETTINGS})]
        runs.append(("sentinel2", {"source": "sentinel2", **ten_m}))
    configs = []
    for label, settings in runs:
        stem = f"fields_{label.replace('-', '_')}_{settings['year']}"
        settings = {**settings, "engine_params": dict(settings.get("engine_params", {}))}
        config = AgriboundConfig(
            study_area=str(out_dir / "study_area.geojson"),
            engine="delineate-anything",
            output_path=str(out_dir / f"{stem}.gpkg"),
            gee_project=args.gee_project,
            lulc_filter=False,
            overwrite=args.overwrite,
            **settings,
        )
        configs.append((label, config))
    return configs


def composite_facts(config):
    """The composite facts of a run's provenance record (images, collects, orbit pass)."""
    from agribound.provenance import read_provenance

    record = read_provenance(config.output_path) or {}
    tags = (record.get("facts") or {}).get("composite") or {}
    collects = json.loads(tags.get("AGRIBOUND_SAR_COLLECTS") or "[]")
    return {
        "n_images": tags.get("AGRIBOUND_N_IMAGES", ""),
        "image_years": tags.get("AGRIBOUND_IMAGE_YEARS", ""),
        "orbit_pass": tags.get("AGRIBOUND_ORBIT_PASS") or tags.get("AGRIBOUND_SAR_ORBIT_STATE", ""),
        "polarization": tags.get("AGRIBOUND_SAR_POLARIZATION", ""),
        "collects": " ".join(f"{c['id']}@{str(c['time'])[:10]}" for c in collects),
        "attribution": tags.get("AGRIBOUND_ATTRIBUTION", ""),
    }


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


def score(pred, ref, area):
    """Scores of *pred* against the reference fields *ref* in the evaluation *area*."""
    from agribound.evaluate import evaluate

    pred = rep_point_inside(pred.to_crs(ref.crs), area)
    bootstrap = {"bootstrap": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED}
    m = evaluate(pred, ref, iou_threshold=IOU_THRESHOLD, **bootstrap)
    f1_lo, f1_hi = ((m.get("bootstrap") or {}).get("ci") or {}).get("f1") or (math.nan, math.nan)
    merged, split = merge_split_rates(pred, ref)
    scores = {
        "n_predicted": len(pred),
        "tp": m["count_tp"],
        "fp": m["count_fp"],
        "fn": m["count_fn"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "f1_lo": f1_lo,
        "f1_hi": f1_hi,
        "iou_mean": m["iou_mean"],
        "merge_rate": merged,
        "split_rate": split,
    }
    return pred, m, scores


def describe(slug, study, evaluation, reference):
    """Print a site's areas and reference fields."""
    site = SITES[slug]
    epsg, (start, end) = site["epsg"], site["window"]
    centre = to_4326(study.centroid, epsg)
    print(f"\n{slug}: {site['name']} ({site['role']})")
    print(
        f"  areas: study {study.area / 1e6:.2f} km2, evaluation {evaluation.area / 1e6:.2f} km2; "
        f"EPSG:{epsg}, centre {centre.x:.4f}, {centre.y:.4f}; "
        + (f"date {start}" if start == end else f"window {start} to {end}")
    )
    if isinstance(reference, Exception):
        print(f"  reference: not read ({type(reference).__name__}: {reference})")
        return
    ref, n_small, label = reference
    size_ha = ref.area / 1e4
    median = f", median {size_ha.median():.1f} ha" if len(ref) else ""
    print(
        f"  reference: {label}: {len(ref):,} fields{median}; "
        f"{n_small:,} smaller than {MIN_AREA:,} m2 dropped"
    )
    if note := reference_note(ref):
        print(f"  sources: {note}")


def show_selection(slug, config, args):
    """Print the collects the automatic selection picks from the cached archive index
    (read as it is: never refreshed, rebuilt or written)."""
    from agribound.composites import vhr_sar
    from agribound.composites.gee import date_window

    cache_dir = vhr_sar.resolve_cache_dir(args.vhr_sar_cache_dir)
    index = vhr_sar.cached_index(config.source, cache_dir)
    if index is None:
        state = (
            "unreadable or of an older format"
            if vhr_sar.index_path(config.source, cache_dir).exists()
            else "not cached"
        )
        print(f"    {config.source} index {state} in {cache_dir}: a run lists the archive")
        return
    study, _ = site_areas(slug)
    try:
        selection = vhr_sar.select_collects(
            index.records,
            to_4326(study, SITES[slug]["epsg"]),
            date_window(config),
            source=config.source,
            snapshot=index.snapshot,
        )
    except Exception as exc:  # e.g. no collect covers the area in a newer index
        print(f"    selection ({index.snapshot}): {type(exc).__name__}: {exc}")
        return
    group, chosen = selection.group, selection.chosen
    size_gb = sum(r.product_bytes for r in chosen) / 1e9
    print(
        f"    selection ({index.snapshot}): {len(chosen)} {group['polarization']} collects, "
        f"{group['orbit_state']}, {group['look_side']}-looking, {size_gb:.2f} GB:"
    )
    print(f"      {', '.join(f'{r.collect_id[:8]} {r.date}' for r in chosen)}")
    if sorted(r.collect_id[:8] for r in chosen) != sorted(SITES[slug]["collects"]):
        print("      NOTE: not the collects recorded on 2026-10-09 (SITES); scores may differ")


def dry_run(slugs, args):
    """Print every run's configuration (validated; nothing is run, downloaded or written)."""
    print(f"\n{'=' * 70}\nDry run: configurations (nothing is run or written)\n{'=' * 70}")
    logging.getLogger("agribound.auth").setLevel(logging.WARNING)  # one GEE-project line each
    for slug in slugs:
        print(f"\n{slug}: {SITES[slug]['name']}")
        for label, config in run_configs(slug, args):
            if config.source == "naip":
                window = f"mosaic, {config.year} on top"
            else:
                window = " to ".join(config.date_range) if config.date_range else "-"
            params = config.engine_params or "defaults"
            print(f"  {label:<18} {config.year} {window:<24} {params} -> {config.output_path}")
            if config.source in ("umbra", "capella"):
                show_selection(slug, config, args)
        if slug == UAVSAR_SITE:
            print(f"  UAVSAR: {UAVSAR_GRANULE} (Earthdata login; not read in a dry run)")


def fmt(value, digits=2):
    """Format a metric (NaN-safe)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if math.isnan(value) else f"{value:.{digits}f}"


def print_table(rows):
    """The per-site table: fields, P, R, F1 with its interval, IoU, merge and split rates."""
    print(
        f"\n{'Site':<15} {'Run':<18} {'Year':>4} {'Ref':>5} {'Pred':>5} {'TP':>4} {'P':>5} "
        f"{'R':>5} {'F1 [95 %]':<17} {'IoU':>5} {'Merge':>5} {'Split':>5}"
    )
    for row in rows:
        start = f"{row['site']:<15} {row['run']:<18} {row['year']:>4} {row['n_reference']:>5}"
        if row["status"] != "ok":
            print(f"{start} {row['status']}")
            continue
        interval = f"{fmt(row['f1'], 3)} [{fmt(row['f1_lo'])}-{fmt(row['f1_hi'])}]"
        print(
            f"{start} {row['n_predicted']:>5} {row['tp']:>4} {fmt(row['precision']):>5} "
            f"{fmt(row['recall']):>5} {interval:<17} {fmt(row['iou_mean']):>5} "
            f"{fmt(row['merge_rate']):>5} {fmt(row['split_rate']):>5}"
        )
    print(
        f"\nOne-to-one matching at IoU >= {IOU_THRESHOLD:g} on each site's evaluation area; "
        f"[95 %]: bootstrap interval of F1 ({N_BOOTSTRAP} resamples); IoU: mean IoU of the "
        "matched pairs; Merge, Split: shares of the reference fields merged with another or "
        f"split. Reference: fields of at least {MIN_AREA:,} m2 by representative point."
    )


def reference_digest(ref):
    """A short hash of the reference outlines, whatever their order (``reference_sha256``)."""
    return hashlib.sha256(b"".join(sorted(ref.geometry.to_wkb()))).hexdigest()[:16]


def merged_summary(path, rows):
    """The summary table for *path*: this run's *rows*, and the rows of the earlier file at
    *path* (if any) of every site that this run gave no row (not run, or its reference not
    read this time), in the order of SITES."""
    import pandas as pd

    summary = pd.DataFrame(rows)
    if Path(path).exists():
        previous = pd.read_csv(path, dtype={"reference_sha256": str})
        scored = set(summary["site"]) if len(summary) else set()
        summary = pd.concat([previous[~previous["site"].isin(scored)], summary], ignore_index=True)
    if len(summary):
        order = {slug: i for i, slug in enumerate(SITES)}
        summary = summary.sort_values("site", key=lambda col: col.map(order), kind="stable")
    return summary.reset_index(drop=True)


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(
        description="Radar: Sentinel-1, Umbra and Capella with Delineate-Anything v2 at "
        "seven sites, beside NAIP and Sentinel-2."
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
        metavar="SITE",
        choices=list(SITES),
        help=f"Run only these sites ({', '.join(SITES)}).",
    )
    parser.add_argument(
        "--uavsar",
        action="store_true",
        help=f"Add the UAVSAR site ({UAVSAR_SITE}; needs an Earthdata login in ~/.netrc).",
    )
    parser.add_argument(
        "--openet-dir",
        default=None,
        help=f"Folder of the {OPENET_LABEL}, one <ST>.zip per state (default: ${DATA_ENV}, "
        "then OPENET_DIR in the script); without it the tamu, intel_oh and orleans_vt sites "
        "are skipped.",
    )
    parser.add_argument(
        "--vhr-sar-cache-dir",
        default=None,
        help="Cache of the Umbra and Capella index and products (default: agribound's, "
        "~/.cache/agribound/vhr_sar).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute outputs that already exist."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read the references and print the areas, the reference counts, every run's "
        "configuration and the cached collect selection; run, download and write nothing.",
    )
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def main():
    import geopandas as gpd

    from agribound.visualize import show_comparison

    args = parse_args()
    start_time = time.time()
    slugs = list(dict.fromkeys(args.only or MAIN_SITES))
    if args.uavsar and UAVSAR_SITE not in slugs:
        slugs.append(UAVSAR_SITE)
    print(f"Reading the references of {', '.join(slugs)}")
    sites = {}
    for slug in slugs:
        study, evaluation = site_areas(slug)
        try:
            reference = load_reference(slug, evaluation, SITES[slug]["epsg"], args)
        except FileNotFoundError as exc:
            print(f"\n{slug} skipped: {exc}")
            continue
        except Exception as exc:  # a map service down, for example
            reference = exc
        describe(slug, study, evaluation, reference)
        sites[slug] = (study, evaluation, reference)
    if args.dry_run:
        dry_run(list(sites), args)
        return
    if not sites:
        raise SystemExit("No site to run.")
    from agribound.auth import setup_gee

    setup_gee(project=args.gee_project)

    rows = []
    for slug, (study, evaluation, reference) in sites.items():
        site, epsg = SITES[slug], SITES[slug]["epsg"]
        if isinstance(reference, Exception):
            print(f"\n{slug} skipped: its reference could not be read ({reference})")
            continue
        ref, n_small, ref_label = reference
        out_dir = OUTPUT_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "study_area.geojson").write_text(
            json.dumps(area_geojson(study, epsg, f"{slug} study area"))
        )
        (out_dir / "evaluation_area.geojson").write_text(
            json.dumps(area_geojson(evaluation, epsg, f"{slug} evaluation area"))
        )
        if len(ref):  # the evaluated reference, for maps such as the gallery's
            ref.to_file(out_dir / "reference.gpkg", layer="reference")
        digest = reference_digest(ref)
        layers = {}
        for label, config in run_configs(slug, args):
            row = {
                "site": slug,
                "role": site["role"],
                "reference": ref_label,
                "reference_sha256": digest,
                "run": label,
                "source": config.source,
                "year": config.year,
                "date_range": " to ".join(config.date_range) if config.date_range else "",
                "engine_params": json.dumps(config.engine_params, sort_keys=True),
                "utm_epsg": epsg,
                "evaluation_km2": round(evaluation.area / 1e6, 3),
                "n_reference": len(ref),
                "n_reference_small_dropped": n_small,
            }
            print(f"\n{'=' * 70}\n{slug}: {IMAGERY[label]} {config.year}\n{'=' * 70}")
            try:
                tif = config.local_tif_path
                if label == "uavsar" and (args.overwrite or not Path(tif).exists()):
                    # the window: the study area and 20 m more (pixels for the bilinear)
                    bounds = to_4326(study.buffer(20, join_style="mitre"), epsg).bounds
                    pauli_composite(tif, *uavsar_window(bounds), study)
                pred = agribound.delineate(config=config)
            except Exception as exc:  # e.g. no imagery, no collect, no Earthdata login
                print(f"  {slug} {label} failed: {type(exc).__name__}: {exc}")
                rows.append({**row, "status": f"failed: {type(exc).__name__}"})
                continue
            facts = composite_facts(config)
            pred, metrics, scores = score(pred, ref, evaluation)
            rows.append({**row, **facts, "status": "ok", **scores, "output": config.output_path})
            stem = Path(config.output_path).stem
            (out_dir / f"metrics_{stem}.json").write_text(
                json.dumps(
                    to_jsonable({"row": rows[-1], "metrics": metrics}), indent=2, allow_nan=False
                )
            )
            chosen_ids = sorted(c[:8] for c in facts["collects"].split())
            if label == site["vhr"] and chosen_ids != sorted(site["collects"]):
                print("  NOTE: not the collects recorded on 2026-10-09 (SITES); scores may differ")
            print(f"  {slug} {label}: {len(pred)} fields, F1 {fmt(scores['f1'], 3)}")
            credit = f" ({CREDITS[label]})" if label in CREDITS else ""
            layers[label] = (pred, f"Delineate-Anything, {IMAGERY[label]} {config.year}{credit}")

        # Two maps per site: the radar runs and the optical runs, each with the reference.
        area_layer = gpd.GeoDataFrame(geometry=[evaluation], crs=epsg)
        for kind, labels in (
            ("radar", ("sentinel1", "umbra", "capella", "uavsar")),
            ("optical", ("naip", "naip-sar-settings", "sentinel2")),
        ):
            chosen = [layers[label] for label in labels if label in layers]
            if not chosen:
                continue
            map_path = out_dir / f"map_{kind}.html"
            names = ["Evaluation area", f"Reference: {ref_label}"] + [name for _, name in chosen]
            web_map = show_comparison(
                [area_layer, ref] + [frame for frame, _ in chosen],
                labels=names,
                output_html=str(map_path),
            )
            show_in_notebook(web_map)
            print(f"Map: {map_path}")

    # This run's rows replace those of the sites it scored; the others are kept, also the
    # rows of a site whose reference could not be read this time.
    path = OUTPUT_DIR / "summary.csv"
    if rows:
        # convert_dtypes keeps the counts integers when a failed run leaves them empty
        merged_summary(path, rows).convert_dtypes().to_csv(path, index=False)
        print_table(rows)
        print(f"Summary: {path}")
    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
