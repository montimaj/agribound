"""
23 — Tree Crops: Oil Palm Estates, Smallholder Oil Palm, Orchards and Olive Groves

How agribound does on tree crops, in four study areas with reference boundaries:

    1. Twifo Praso, Ghana (2020): the Twifo Oil Palm Plantations nucleus estate,
       rectangular planting blocks of about 40 ha separated in the reference by
       gaps of about 4 m along the estate roads. Reference: the estate's blocks
       in the RSPO GeoRSPO concession maps (``MemberNum`` 1-0157-14-000-00).
       5 km square.
    2. Oro Province, Papua New Guinea (2021): oil palm smallholder parcels of
       about 1.5 ha in the Higaturu scheme. Reference: the scheme's parcels in
       GeoRSPO (``SupplyBase`` HOP_SH), which cover about half of the square.
       3 km square.
    3. Madera County, California, USA (2022): almond and pistachio orchards in
       blocks of about 17 ha. Reference: the California Department of Water
       Resources (DWR) / Land IQ Statewide Crop Mapping of water year 2022,
       which maps every field, so the evaluation is split into tree crops and
       other fields. 5 km square.
    4. Ubeda, Jaen, Spain (2023): olive groves of a few hectares with trees
       about 10 m apart. Reference: SIGPAC land parcels ("recintos") of the
       previous campaign (2025), which map all land; recintos split some
       groves along cadastral lines, so the groves are also evaluated after
       joining touching recintos of the same land use (recintos under 2,500 m²
       are dropped first, so they do not connect groves). 4 km square.

Approaches (each a separate agribound run with ``lulc_filter=False``):

    - Delineate-Anything v2 (``large_v2``, as released) on SPOT 6/7
      panchromatic 1.5 m (``spot-pan``; restricted), with and without SAM 2
      refinement;
    - Delineate-Anything v2 and DINOv3, each fine-tuned on the SPOT-Pan
      chips of a training area near the study area (``TRAINING_AREAS``); no
      evaluated square or reference polygon is used in fine-tuning:
        - Twifo: the blocks of the NORPALM estate (Western Region, Ghana;
          2020), whose bounding box is 69 km from the Twifo square (centres
          81 km apart);
        - Oro: Higaturu scheme parcels in a 6 km square 13.6 km from the Oro
          square (centres 19.5 km apart; a March 2021 scene);
        - Madera: DWR / Land IQ fields in a 5 km square 41.9 km from the
          Madera square (centres 48.3 km apart; 2022);
        - Ubeda: SIGPAC recintos in a 4 km square 10.8 km from the Ubeda
          square (centres 16.1 km apart; July 2023).
      The squares near Madera and Ubeda were chosen by a fixed rule from the
      reference labels and the SPOT coverage alone, before any model ran on
      them (see ``TRAINING_AREAS``); their training labels are every mapped
      field that reaches into the square, whatever its size (the chips label
      every other pixel as background). The DINOv3 settings and these two
      squares were fixed before any DINOv3 model, or any model trained on these
      squares, was scored. The Higaturu
      fine-tuning of Delineate-Anything was added after the released model
      found one parcel at Oro, and kept after its own Oro score was seen; none
      of its settings was changed after that score. Both models get the same
      labels and SPOT-Pan composite and the same rules: chips with at least
      ``min_label_fraction`` of their pixels labelled, a random 20 % of the
      chips for validation (a parcel cut by a chip edge can be in both) and 20
      epochs. Each uses its own chips and default recipe. Delineate-Anything:
      512 px chips, Ultralytics' augmentation (flips, brightness, scale,
      translation) and ``yolo_lr0=1e-4``, set on the NORPALM validation chips
      because with so few chips the default learning rate wrecks the
      pretrained weights (see the fine-tuning guide). DINOv3 (ViT-L/16
      backbone pre-trained on SAT-493M satellite imagery, DPT head; it has no
      published field-boundary weights, so it has no released variant):
      agribound's defaults, i.e. 256 px chips, no augmentation, full
      fine-tuning, learning rate 1e-4, batch 4, and early stopping after 10
      epochs without improvement of the validation loss (so at most 20
      epochs); the panchromatic band is copied into its three RGB inputs.
      DINOv3 predicts background, field interior and field boundary pixels
      over the whole square and the fields are its interior regions, while
      Delineate-Anything predicts field instances per 512 px tile. At Madera
      and Ubeda one SPOT scene covers both the training and the evaluated
      square, so the two composites share that acquisition (no labels or
      pixels are shared). At Madera the training scene (8 May 2022, leaf-on)
      differs in season from the evaluation composite, a median of 11 scenes
      of which 9 are from January to March; the released and fine-tuned
      SPOT-Pan models are therefore also run on the 8 May scene of the Madera
      square (runs ``da-may``, ``da-ft-may`` and ``dinov3-ft-may``). The
      released Delineate-Anything weights (and so its fine-tuned models) were
      trained on FBIS-73M, whose training patches cover the Madera square and
      87 % of the Ubeda square, with labels that match the DWR / Land IQ and
      SIGPAC references (121 of the 122 Madera reference fields and 166 of the
      225 Ubeda recintos), while it has no patches in Ghana or Papua New
      Guinea (checked against the public patch list, images and labels on
      2026-10-06). So at Madera and Ubeda Delineate-Anything has seen the
      evaluated fields and DINOv3 has not;
    - Delineate-Anything v2 on Landsat 8/9 panchromatic 15 m (``landsat-pan``;
      outside the model's 0.25-10 m training range) and on Sentinel-2 10 m;
      at Madera also on NAIP (1 m);
    - FTW (``FTW_PRUE_EFNET_B5``, two Sentinel-2 windows). Oil palm has no
      crop season, so at Twifo the two windows are set to clear months;
    - unsupervised k-means (k = 20) of the Google Satellite Embedding and of
      TESSERA (TESSERA has no tiles for Oro in 2021). These are land-cover
      segments, not field instances.

Crop filter. The default LULC crop filter would delete most of these fields:
outside the conterminous US it uses Dynamic World, which counts plantations
and orchards as ``trees`` ("Plantations such as apples, bananas, citrus, and
rubber", Brown et al. 2022, Table 1), not ``crops``. So every run delineates
without the filter, and the filter is then applied as a separate step twice:
with the defaults, and with ``lulc_tree_crops=True``, which counts Dynamic
World ``crops`` + ``trees`` (and the C3S tree-cover classes; NLCD, used in
the US, already counts orchards as cultivated crops). The printed table
reports how many polygons each variant keeps; ``summary.csv`` also gives the
share of the matched predictions (IoU >= 0.5) that each variant keeps
(``matched_kept_default``, ``matched_kept_tree_crops``; at Madera and Jaen,
those matched to tree-crop fields).

Evaluation (``agribound.evaluate``): one-to-one matching at IoU >= 0.5,
boundary metrics at a 10 m tolerance, bootstrap intervals; references and
predictions are selected by their representative point inside the square.
Precision needs care because the RSPO references do not map every field in
their squares:

    - Twifo and Oro (RSPO): ``P_ov`` = precision among the predictions that
      overlap a reference polygon (``count_tp / (count_tp + count_fp -
      count_fp_unassigned)``); at Oro also the precision of the predictions
      with at least half of their area within 20 m of a reference parcel.
    - Madera and Jaen: the references map all agricultural fields, so the
      tree-crop scores are the ``per_stratum["tree"]`` entries of one
      evaluation against all fields.
    - Merge and split rates: a reference field is merged when its main
      prediction also covers at least 25 % of another reference field (of any
      crop), and split when at least two predictions each cover 10 % of it;
      at Madera and Jaen the rates are averaged over the tree-crop fields.
    - Twifo also gets the areal recall and spill of the predictions against
      the estate outline (the blocks joined, clipped to the square), the
      protocol for concession outlines that are not subdivided into blocks:
      areal recall is the share of the estate covered by the predictions,
      spill the share of the area of the predictions whose representative
      point is in the estate that lies outside it.

References and their terms (nothing is stored in this repository; the script
downloads the data into ``outputs/tree_crops/references/``):

    - RSPO: "RSPO Concessions Versions 18 – September 2026" (GeoRSPO,
      https://rspo.org/resources/?category=georspo), pinned by SHA-256. RSPO
      grants no licence; the maps "can now be downloaded and used for
      independent use or analysis" (RSPO, 2020) and are provided for
      "informational and illustrative communication purposes only" (RSPO
      Disclaimer for Map Publication). The boundaries are member-declared. Do
      not redistribute them. If the download fails, download the zip from
      that page and pass ``--rspo-zip``.
    - DWR: California Department of Water Resources (2022). Statewide Crop
      Mapping, California Natural Resources Agency Open Data,
      https://data.cnra.ca.gov/dataset/statewide-crop-mapping (licence not
      specified).
    - SIGPAC: Fondo Español de Garantía Agraria (FEGA), SIGPAC,
      https://sigpac-hubcloud.es/, CC BY 4.0. The service serves the previous
      campaign, which changes each year; the script warns when the parcels
      (identifiers, land uses or outlines) differ from those of an earlier run.

Imagery and reference dates differ: the RSPO maps were published in
September 2026 (no mapping date) and SIGPAC is the 2025 campaign, while SPOT
6/7 ends in November 2023. Each area uses one year for every source, set by
the clearest SPOT coverage; SPOT is filtered by scene cloud cover but not
cloud-masked.

Thanks to Jacob Abramowitz (The University of Alabama in Huntsville) for
asking about tree crops and pointing to the RSPO concession maps and the
subdivided Twifo estate; see Abramowitz et al. (2023), Remote Sensing
Applications: Society and Environment 30, 100968,
doi:10.1016/j.rsase.2023.100968, who used RSPO polygons as reference data in
Ghana.

SPOT 6/7 (AIRBUS/SPOT6_7) is restricted to select Earth Engine users. Without
access the SPOT runs, SAM 2 and the fine-tuning are skipped and the rest runs.
``--no-dinov3`` skips DINOv3, the slowest step (full fine-tuning of a ViT-L;
each checkpoint is about 3.7 GB, in the training area's ``.agribound_cache``).

Estimated runtime: about 2 hours on an Apple M2 Max (MPS), downloads, the
fine-tunings and the crop-filter steps included: about 27 minutes for
everything but DINOv3 and the Madera and Ubeda fine-tunings (2026-10-05), and
94 minutes for those in a run that reused the rest (2026-10-06), of which the
four DINOv3 fine-tunings took 14 to 21 minutes each.

Prerequisites:
    pip install "agribound[gee,delineate-anything,dinov3,ftw,embedding,tessera,samgeo]"
        (or use environment.yml)
    agribound auth --project YOUR_GEE_PROJECT
    Run from the repository root: python examples/23_tree_crops.py
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

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

warnings.filterwarnings("ignore", category=FutureWarning, module=r"geedim\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"geedim\..*")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/tree_crops")
REFERENCE_DIR = OUTPUT_DIR / "references"
USER_AGENT = f"agribound/{agribound.__version__} (+https://github.com/montimaj/agribound)"
MIN_AREA = 2500  # m^2; the default minimum field area (also applied to the references)
BOUNDARY_TOLERANCE_M = 10.0
N_BOOTSTRAP = 200
CROP_THRESHOLD = 0.3
N_CLUSTERS = 20  # embedding k-means; fixed in advance (the automatic choice picks 5)
MERGE_SHARE = 0.25  # merged: the main prediction also covers this share of another field
SPLIT_SHARE = 0.10  # split: at least two predictions each cover this share of the field

RSPO_URL = (
    "https://rspo.org/wp-content/uploads/RSPO-Concessions-Versions-18-%E2%80%93-September-2026.zip"
)
RSPO_SHA256 = "c932dd0b59f14852b0ebfc404d8ed7b6c49d234a056157889f67bc24245aed34"
RSPO_ZIP_NAME = "RSPO-Concessions-Versions-18-September-2026.zip"
DWR_2022_QUERY = (
    "https://utility.arcgis.com/usrsvcs/servers/8b0555ad7cb14dcab66901925427228a/rest/"
    "services/Planning/i15_Crop_Mapping_2022/MapServer/0/query"
)
SIGPAC_TILE = "https://sigpac-hubcloud.es/mvt/anterior/recinto@3857@geojson/15/{x}/{y}.geojson"
SIGPAC_KEY = ["provincia", "municipio", "agregado", "zona", "poligono", "parcela", "recinto"]
# SIGPAC land uses: tree crops (olive, fruit, nut and citrus groves and their mixes) and
# land that is not a field (roads, unproductive land, buildings, water, urban, forest,
# scrub, pasture, ...), which is left out of the reference.
SIGPAC_TREE_USES = {
    "OV", "OF", "OC", "CO", "FY", "FS", "FL", "FF", "FV", "CI", "CF", "CS", "CV", "VO", "VF",
}  # fmt: skip
SIGPAC_NON_FIELD = {
    "CA", "IM", "ED", "AG", "ZU", "FO", "MT", "ZV", "ZC", "EP", "IS", "PS", "PR", "PA",
}  # fmt: skip
DWR_TREE_CLASSES = {"D", "C", "YP"}  # deciduous fruits and nuts, citrus/subtropical, young
DWR_NON_FIELD = {"U", "UL", "UR", "UC", "UI", "UV", "NC", "NR", "NV", "NW", "NB", "NS"}

# The study areas: centre, side of the UTM square, year, reference and per-source settings.
SITES = {
    "twifo": {
        "name": "Twifo Praso, Ghana: oil palm estate",
        "lon": -1.51801,
        "lat": 5.53834,
        "side_m": 5000,
        "year": 2020,
        "reference": ("rspo", {"MemberNum": "1-0157-14-000-00"}),
        "strata": "SupplyBase",  # the estate's two divisions
        "spot": {"cloud_cover_max": 15},  # one scene, 2020-01-03, 0 % cloud
        "landsat_pan": {"cloud_cover_max": 20},
        # The default SCL mask at 20 % keeps 6 hazy images; Cloud Score+ at 80 % keeps 52.
        "s2": {"cloud_cover_max": 80, "s2_cloud_mask": "cloud_score_plus"},
        # FTW's crop-calendar windows have no clear image in 2020; oil palm has no season.
        "ftw_window_dates": ["2020-01-15", "2020-08-15"],
        "tessera": True,
        "naip": False,
        "fine_tuned": "norpalm",  # the training area of its fine-tuned model
    },
    "oro": {
        "name": "Oro Province, Papua New Guinea: oil palm smallholders",
        "lon": 148.12507,
        "lat": -8.80015,
        "side_m": 3000,
        "year": 2021,
        "reference": ("rspo", {"MemberNum": "1-0008-04-000-00", "SupplyBase": "HOP_SH"}),
        "strata": None,
        "spot": {"cloud_cover_max": 15},  # one scene, 2021-06-15, 5.1 % cloud
        "landsat_pan": {"cloud_cover_max": 20},
        "s2": {"cloud_cover_max": 80, "s2_cloud_mask": "cloud_score_plus"},
        "ftw_window_dates": None,  # FTW's crop calendar
        "tessera": False,  # no TESSERA tiles in 2021
        "naip": False,
        "fine_tuned": "higaturu",
    },
    "madera": {
        "name": "Madera County, California: almond and pistachio orchards",
        "lon": -120.46685,
        "lat": 36.989886,
        "side_m": 5000,
        "year": 2022,
        "reference": ("dwr", {}),
        "strata": "stratum",
        "spot": {"cloud_cover_max": 15},  # 11 scenes, mostly January-March (leaf-off)
        "landsat_pan": {"cloud_cover_max": 20},
        "s2": {},  # defaults: SCL mask, 20 %
        "ftw_window_dates": None,
        "tessera": True,
        "naip": True,
        "fine_tuned": "cressey",
        # The Cressey training square has valid SPOT pixels in 2022 almost only on 8 May
        # (leaf-on), while this composite is a median dominated by January-March scenes. The
        # released and fine-tuned SPOT-Pan models are therefore also run on May 2022 alone
        # (the 8 May scene), as runs labelled "<run>-may".
        "spot_season": ("may", ("2022-05-01", "2022-05-31")),
    },
    "jaen": {
        "name": "Ubeda, Jaen, Spain: olive groves",
        "lon": -3.347141,
        "lat": 37.947077,
        "side_m": 4000,
        "year": 2023,
        "reference": ("sigpac", {}),
        "strata": "stratum",
        # Two July scenes; the calendar year adds a scene covering 45 % of the square (a seam).
        "spot": {"cloud_cover_max": 15, "date_range": ("2023-07-01", "2023-07-31")},
        "landsat_pan": {"cloud_cover_max": 20},
        "s2": {},
        "ftw_window_dates": None,
        "tessera": True,
        "naip": False,
        "fine_tuned": "ibros",
    },
}
# Training areas of the fine-tuned models (never evaluated). Delineate-Anything and DINOv3
# are both fine-tuned on every area, with the same labels and chip rules. With few
# training chips Delineate-Anything's default learning rate (0.002) wrecks the pretrained
# weights in so few optimizer steps (see agribound.engines.finetune._yolo); 1e-4 does not.
# The squares near Madera and Ubeda were chosen by a rule fixed before the search, from the
# labels and the SPOT coverage alone (no model had run on them): the side of the evaluation
# square, at least 10 km from it edge to edge (centres at most 60 km apart at Madera, 40 km
# at Ubeda), at least 70 % of the label fields tree crops, and at least 99 % SPOT coverage at 15 %
# cloud in the evaluation year (at Ubeda within one month); among those, the square with
# the most tree-crop fields on a grid of 2.5 km (Madera) or 2 km (Ubeda; the 704 nearest
# of its 1,100 grid points, a limit on SIGPAC downloads). The rule favours many small
# fields, so the training fields are smaller than the evaluated ones.
FINE_TUNE_EPOCHS = 20
TRAINING_AREAS = {
    # NORPALM estate, Western Region, Ghana: the bounding box of its planting blocks.
    "norpalm": {
        "name": "NORPALM estate, Ghana (training only)",
        "reference": (
            "rspo",
            {"MemberNum": "1-0162-14-000-00", "SupplyBase": "NGL ESTATE (NUCLEUS)"},
        ),
        "bbox_utm": ((618154, 535298, 628440, 549513), "EPSG:32630"),  # the blocks
        "year": 2020,
        "spot": {"cloud_cover_max": 15},  # 2020-01-04 and 2020-01-09
        "engine_params": {"min_label_fraction": 0.5, "yolo_lr0": 1e-4},
    },
    # Higaturu scheme parcels near Sorovi, Oro Province: a 6 km square. The parcels cover
    # about a quarter of it, so a training chip needs only 25 % of its pixels labelled.
    "higaturu": {
        "name": "Higaturu smallholders near Sorovi, Papua New Guinea (training only)",
        "reference": ("rspo", {"MemberNum": "1-0008-04-000-00", "SupplyBase": "HOP_SH"}),
        "square": (148.28232, -8.71928, 6000),  # centre lon, lat and side (m)
        "year": 2021,
        # One scene covers 98 % of the square (2021-03-09); the other 2021 scenes 23-37 %.
        "spot": {"cloud_cover_max": 15, "date_range": ("2021-03-01", "2021-03-31")},
        "engine_params": {"min_label_fraction": 0.25, "yolo_lr0": 1e-4},
    },
    # Orchards near Cressey, Merced County, California: DWR / Land IQ 2022 fields in a 5 km
    # square 41.9 km from the Madera square (centres 48.3 km apart), chosen by the rule
    # above: 352 tree-crop fields of 414 with their representative point in the square and at
    # least 2,500 m2, as in the evaluation (median orchard 3.7 ha, smaller than Madera's
    # 16.6 ha); the training labels are all 451 fields that reach into the square. Only the
    # 2022-05-08 scene covers the whole square (87 % of it with no other scene); it is also
    # one of the 11 scenes of the Madera composite.
    "cressey": {
        "name": "Orchards near Cressey, Merced County, California (training only)",
        "reference": ("dwr", {}),
        "square": (-120.6508, 37.39921, 5000),
        "year": 2022,
        "spot": {"cloud_cover_max": 15},
        "engine_params": {"yolo_lr0": 1e-4},
    },
    # Olive groves around Ibros, Jaen: SIGPAC recintos in a 4 km square 10.8 km from the
    # Ubeda square (centres 16.1 km apart), chosen by the rule above: 2,063 tree-crop
    # recintos of 2,088 with their representative point in the square and at least 2,500 m2
    # (median 0.48 ha, smaller than Ubeda's 3.2 ha); the training labels are all 2,983
    # recintos that reach into the square. Its one July scene (2023-07-08) is also one of the
    # two scenes of the Ubeda composite.
    "ibros": {
        "name": "Olive groves around Ibros, Jaen, Spain (training only)",
        "reference": ("sigpac", {}),
        "square": (-3.50698, 38.0186, 4000),
        "year": 2023,
        "spot": {"cloud_cover_max": 15, "date_range": ("2023-07-01", "2023-07-31")},
        "engine_params": {"yolo_lr0": 1e-4},
    },
}


def square_study_area(lon, lat, side_m, name):
    """A GeoJSON FeatureCollection: a square of *side_m* metres in the local UTM zone."""
    import pyproj
    from shapely.geometry import box, mapping
    from shapely.ops import transform

    from agribound.io import get_utm_crs

    utm = get_utm_crs(lon, lat)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    to_ll = pyproj.Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    x, y = to_utm.transform(lon, lat)
    half = side_m / 2
    square = transform(to_ll.transform, box(x - half, y - half, x + half, y + half))
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": mapping(square), "properties": {"name": name}}
        ],
    }


def utm_square(site):
    """The site's square as a shapely box in its UTM zone, and that CRS."""
    import pyproj
    from shapely.geometry import box

    from agribound.io import get_utm_crs

    utm = get_utm_crs(site["lon"], site["lat"])
    x, y = pyproj.Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform(
        site["lon"], site["lat"]
    )
    half = site["side_m"] / 2
    return box(x - half, y - half, x + half, y + half), utm


def rep_point_inside(gdf, area):
    """Rows whose representative point lies inside *area* (same CRS)."""
    return gdf[gdf.representative_point().within(area).to_numpy()].copy()


def http_get(url, path=None, timeout=120, retries=3):
    """GET *url* with a descriptive User-Agent (bytes, or written to *path*)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if path is None:
                    return response.read()
                tmp = Path(f"{path}.part")
                with open(tmp, "wb") as f:
                    while chunk := response.read(1 << 20):
                        f.write(chunk)
                tmp.replace(path)
                return path
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def clean_polygons(gdf, crs, min_area=MIN_AREA):
    """Valid polygons in *crs*, at least *min_area*, without near-duplicates (> 90 % covered)."""
    import geopandas as gpd
    import shapely

    geoms = shapely.make_valid(gdf.geometry.to_numpy())
    polys = [
        shapely.union_all(shapely.get_parts(g)[shapely.area(shapely.get_parts(g)) > 0])
        for g in geoms
    ]
    gdf = gpd.GeoDataFrame(gdf.drop(columns="geometry"), geometry=polys, crs=gdf.crs).to_crs(crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & (gdf.area >= min_area)]
    gdf = gdf.reset_index(drop=True)
    drop = set()
    for i, j in zip(*gdf.sindex.query(gdf.geometry, predicate="intersects"), strict=True):
        if i >= j or i in drop or j in drop:
            continue
        a, b = gdf.geometry.iloc[i], gdf.geometry.iloc[j]
        shared = a.intersection(b).area
        if shared > 0.9 * min(a.area, b.area):
            drop.add(j if b.area <= a.area else i)
    return gdf.drop(index=sorted(drop)).reset_index(drop=True)


def rspo_zip(args):
    """The pinned GeoRSPO zip (downloaded once into REFERENCE_DIR, or --rspo-zip)."""
    if args.rspo_zip:
        path = Path(args.rspo_zip)
        if sha256_of(path) != RSPO_SHA256:
            print(f"  WARNING: {path} is not the pinned GeoRSPO version 18 file (SHA-256 differs)")
        return path
    path = REFERENCE_DIR / RSPO_ZIP_NAME
    if not path.exists():
        print(f"  Downloading the GeoRSPO concession maps (49 MB) from {RSPO_URL}")
        try:
            http_get(RSPO_URL, path, timeout=600)
        except Exception as exc:
            raise SystemExit(
                f"GeoRSPO download failed ({type(exc).__name__}: {exc}). Download "
                "'RSPO Concessions Versions 18 - September 2026' from "
                "https://rspo.org/resources/?category=georspo and pass --rspo-zip PATH."
            ) from exc
    if sha256_of(path) != RSPO_SHA256:
        raise SystemExit(f"{path}: SHA-256 differs from the pinned GeoRSPO version 18 file.")
    return path


def load_rspo(args, area, crs, filters):
    """RSPO polygons matching *filters* (column -> value) near *area* (a UTM polygon)."""
    import geopandas as gpd
    import pyogrio

    box_3395 = gpd.GeoSeries([area.buffer(2000)], crs=crs).to_crs(3395).total_bounds
    gdf = pyogrio.read_dataframe(f"/vsizip/{rspo_zip(args)}", layer=0, bbox=tuple(box_3395))
    for column, value in filters.items():
        gdf = gdf[gdf[column].astype(str).str.strip() == value]
    return clean_polygons(gdf, crs)


def load_dwr(area, crs, min_area=MIN_AREA):
    """DWR / Land IQ 2022 fields near *area*, with ``stratum`` "tree" or "other"."""
    import geopandas as gpd
    import shapely

    west, south, east, north = gpd.GeoSeries([area.buffer(200)], crs=crs).to_crs(4326).total_bounds
    features, offset = [], 0
    while True:
        query = {
            "where": "1=1",
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "UniqueID,CLASS2,CROPTYP2,SPECOND2,YR_PLANTED",
            "outSR": 4326,
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": 1000,
        }
        page = json.loads(http_get(f"{DWR_2022_QUERY}?{urllib.parse.urlencode(query)}"))
        features += page.get("features", [])
        if not page.get("exceededTransferLimit") and not page.get("properties", {}).get(
            "exceededTransferLimit"
        ):
            break
        offset += 1000
    gdf = gpd.GeoDataFrame.from_features(features, crs=4326)
    gdf["geometry"] = shapely.force_2d(gdf.geometry.to_numpy())
    for column in ("CLASS2", "CROPTYP2"):
        gdf[column] = gdf[column].astype(str).str.strip()
    gdf = gdf[~gdf["CLASS2"].isin(DWR_NON_FIELD)]
    tree = gdf["CLASS2"].isin(DWR_TREE_CLASSES) | gdf["CROPTYP2"].isin(DWR_TREE_CLASSES)
    gdf["stratum"] = ["tree" if t else "other" for t in tree]
    return clean_polygons(gdf, crs, min_area)


def load_sigpac(area, crs, name, min_area=MIN_AREA):
    """SIGPAC agricultural recintos near *area*, with ``stratum`` "tree" or "other".

    *name* (the study or training area) names the record of the parcels' digest.
    """
    import geopandas as gpd
    import shapely
    from shapely.geometry import shape

    west, south, east, north = gpd.GeoSeries([area], crs=crs).to_crs(4326).total_bounds
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
        pages = list(pool.map(lambda t: http_get(SIGPAC_TILE.format(x=t[0], y=t[1])), tiles))
    rows = []
    for page in pages:
        for feature in json.loads(page).get("features", []):
            rows.append({**feature["properties"], "geometry": shape(feature["geometry"])})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=3857).drop_duplicates(SIGPAC_KEY)
    # The service serves the previous campaign, which changes each year: warn when the
    # parcels (identifiers, land uses or outlines) differ from those of an earlier run.
    keys = gdf[SIGPAC_KEY + ["uso_sigpac"]].astype(str).agg("|".join, axis=1).tolist()
    outlines = shapely.to_wkb(gdf.geometry.to_numpy(), hex=True, output_dimension=2).tolist()
    digest = hashlib.sha256(json.dumps(sorted(zip(keys, outlines, strict=True))).encode())
    digest = digest.hexdigest()
    record = REFERENCE_DIR / f"sigpac_{name}.sha256"
    if record.exists() and record.read_text().strip() != digest:
        print("  WARNING: the SIGPAC parcels differ from those of an earlier run (new campaign?)")
    record.write_text(digest)
    gdf = gdf[~gdf["uso_sigpac"].isin(SIGPAC_NON_FIELD)]
    gdf["stratum"] = ["tree" if u in SIGPAC_TREE_USES else "other" for u in gdf["uso_sigpac"]]
    return clean_polygons(gdf, crs, min_area)


def dissolve_by_use(gdf):
    """Join touching recintos of the same land use (one grove split by cadastral lines).

    *gdf* has no recintos under MIN_AREA (``clean_polygons``), so these do not connect
    groves.
    """
    parts = gdf.dissolve(by="uso_sigpac", as_index=False).explode(index_parts=False)
    parts["stratum"] = ["tree" if u in SIGPAC_TREE_USES else "other" for u in parts["uso_sigpac"]]
    return parts[parts.area >= MIN_AREA].reset_index(drop=True)


def save_reference(gdf, slug, label):
    """Write a reference layer to a file whose name holds a hash of its contents.

    The hash covers the geometry and, where present, the ``stratum`` column that the
    evaluation reads. Outputs that were evaluated against a reference are reused only
    when the reference path is unchanged, so a changed reference gets a new path.
    """
    import shapely

    digest = hashlib.sha256(
        b"".join(shapely.to_wkb(gdf.geometry.to_numpy(), hex=False, output_dimension=2))
    )
    if "stratum" in gdf.columns:
        digest.update("|".join(gdf["stratum"].astype(str)).encode())
    digest = digest.hexdigest()[:12]
    path = REFERENCE_DIR / f"{slug}_{label}_{digest}.gpkg"
    if not path.exists():
        gdf.to_file(path, layer="reference")
    return path


def merge_split_rates(pred, ref, mask=None):
    """Shares of reference fields that are merged with another or split (see the docstring).

    A merge counts any other field of *ref*; *mask* (one boolean per row of *ref*)
    selects the fields the rates are averaged over (default: all).
    """
    import numpy as np

    keep = np.ones(len(ref), bool) if mask is None else np.asarray(mask, bool)
    if len(pred) == 0 or not keep.any():
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
    return float(merged[keep].mean()), float(split[keep].mean())


def overlap_precision(m):
    """Precision among the predictions that overlap a reference field (``P_ov``).

    A ``per_stratum`` entry already counts only the predictions whose home field
    is in that stratum, so it has no unassigned predictions.
    """
    denominator = m["count_tp"] + m["count_fp"] - m.get("count_fp_unassigned", 0)
    return m["count_tp"] / denominator if denominator else float("nan")


def f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def evaluate_run(pred, ref, site, args, estate=None):
    """Headline metrics of one layer against one reference (dict) and the full metrics."""
    from agribound.evaluate import evaluate

    strata = site["strata"] if site["strata"] in ref.columns else None
    m = evaluate(
        pred,
        ref,
        iou_threshold=0.5,
        strata=strata,
        bootstrap=args.bootstrap,
        bootstrap_seed=42,
        boundary_tolerance_m=BOUNDARY_TOLERANCE_M,
    )
    part = m["per_stratum"]["tree"] if strata == "stratum" else m
    p_ov = overlap_precision(part)
    merged, split = merge_split_rates(
        pred, ref, mask=(ref["stratum"] == "tree").to_numpy() if strata == "stratum" else None
    )
    row = {
        "fields": len(pred),
        "recall": part["recall"],
        "precision": part["precision"],
        "precision_ov": p_ov,
        "f1_ov": f1(p_ov, part["recall"]),
        "iou_mean": part["iou_mean"],
        "best_iou_mean": part["best_iou_mean"],
        "boundary_f1": part.get("boundary_f1"),
        "merged": merged,
        "split": split,
    }
    if site.get("restrict_precision"):
        # Oro: only predictions with at least half of their area within 20 m of a parcel.
        domain = ref.geometry.buffer(20).union_all()
        inside = pred.geometry.intersection(domain).area / pred.area
        m_d = evaluate(pred[(inside >= 0.5).to_numpy()], ref, iou_threshold=0.5)
        row["precision_domain"] = m_d["precision"]
    if estate is not None and len(pred):
        # Twifo: the estate outline as a concession that is not subdivided.
        union = pred.geometry.union_all()
        in_estate = pred[pred.representative_point().within(estate).to_numpy()]
        u_estate = in_estate.geometry.union_all() if len(in_estate) else None
        row["areal_recall"] = union.intersection(estate).area / estate.area
        row["spill"] = (
            u_estate.difference(estate).area / u_estate.area if u_estate else float("nan")
        )
    return row, m


def crop_filter_layers(fields, out_path, config, args):
    """The crop-filter step: default rule and ``lulc_tree_crops=True`` (layers + counts)."""
    import geopandas as gpd

    from agribound.postprocess.lulc_filter import filter_by_lulc

    layers = {}
    for variant, tree_crops in (("crop", False), ("treecrop", True)):
        # The threshold is in the name: a layer filtered at another threshold is not reused.
        path = out_path.with_name(f"{out_path.stem}_{variant}_t{CROP_THRESHOLD:g}.gpkg")
        stale = path.exists() and path.stat().st_mtime < out_path.stat().st_mtime
        if path.exists() and not stale and not args.overwrite:
            layers[variant] = gpd.read_file(path)
            continue
        cfg = config.merged(
            lulc_filter=True, lulc_crop_threshold=CROP_THRESHOLD, lulc_tree_crops=tree_crops
        )
        layers[variant] = filter_by_lulc(fields, cfg) if len(fields) else fields
        layers[variant].to_file(path, layer="fields")
    return layers


def matched_kept(fields, kept, frame):
    """Share of the matched predictions (IoU >= 0.5) that a filtered layer keeps."""
    ids = set(frame.loc[frame["matched"], "pred_index"].dropna())
    if not ids:
        return float("nan")
    kept_ids = set(kept["id"]) if "id" in kept.columns else set()
    return len({fields.loc[i, "id"] for i in ids} & kept_ids) / len(ids)


def base_config(site, slug, source, engine, label, args, **overrides):
    """AgriboundConfig of one run (explicit output path per source, engine and variant)."""
    from agribound.config import AgriboundConfig

    out_dir = OUTPUT_DIR / slug
    settings = {
        "spot-pan": site["spot"],
        "landsat-pan": site["landsat_pan"],
        "sentinel2": site["s2"],
    }.get(source, {})
    return AgriboundConfig(
        study_area=str(out_dir / "study_area.geojson"),
        source=source,
        year=site["year"],
        engine=engine,
        output_path=str(out_dir / f"fields_{source}_{label}_{site['year']}.gpkg"),
        gee_project=args.gee_project,
        lulc_filter=False,
        overwrite=args.overwrite,
        **{**settings, **overrides},
    )


def site_runs(slug, site, args, checkpoints):
    """(label, source, engine, config overrides) of every run of one study area."""
    da, many = "delineate-anything", {"engine_params": {"max_detections": 1000}}
    runs = []
    if args.spot:
        runs.append(("da", "spot-pan", da, {}))
        if not args.no_sam:
            runs.append(("da-sam2", "spot-pan", da, {"sam_refine": True}))
        tuned = []
        for label, engine in (("da-ft", da), ("dinov3-ft", "dinov3")):
            checkpoint = checkpoints.get((site["fine_tuned"], engine))
            if checkpoint:
                tuned.append((label, engine, {"engine_params": {"checkpoint_path": checkpoint}}))
        runs += [(label, "spot-pan", engine, kw) for label, engine, kw in tuned]
        if site.get("spot_season"):
            season, date_range = site["spot_season"]
            for label, engine, kw in [("da", da, {})] + tuned:
                runs.append(
                    (f"{label}-{season}", "spot-pan", engine, {**kw, "date_range": date_range})
                )
    if site["naip"]:
        runs.append(("da", "naip", da, {}))
    runs.append(("da", "landsat-pan", da, many))
    runs.append(("da", "sentinel2", da, many))
    ftw = (
        {"engine_params": {"window_dates": site["ftw_window_dates"]}}
        if site["ftw_window_dates"]
        else {}
    )
    runs.append(("ftw", "sentinel2", "ftw", ftw))
    k = {"engine_params": {"n_clusters": N_CLUSTERS}}
    runs.append((f"k{N_CLUSTERS}", "google-embedding", "embedding", k))
    if site["tessera"]:
        runs.append((f"k{N_CLUSTERS}", "tessera-embedding", "embedding", k))
    return runs


def fine_tune_on(key, engine, args):
    """Fine-tune *engine* on the training area *key*; return the checkpoint path.

    Both engines get the same labels, SPOT-Pan composite, ``min_label_fraction``,
    epochs and validation share, each on its own default chips; the ``yolo_*``
    settings apply to Delineate-Anything only.
    """
    from shapely.geometry import box

    from agribound.config import AgriboundConfig
    from agribound.provenance import read_provenance

    spec = TRAINING_AREAS[key]
    out_dir = OUTPUT_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    kind, _ = spec["reference"]
    if "square" in spec:
        lon, lat, side = spec["square"]
        area, crs = utm_square({"lon": lon, "lat": lat, "side_m": side})
        if kind == "rspo":
            labels = rep_point_inside(load_reference(key, spec, args, area, crs), area)
        else:
            # A complete map (DWR, SIGPAC): every mapped field that reaches into the square,
            # whatever its size, because the chips label every other pixel as background.
            labels = load_reference(key, spec, args, area, crs, min_area=0)
            labels = labels[labels.intersects(area).to_numpy()].reset_index(drop=True)
        study_area = square_study_area(lon, lat, side, spec["name"])
    else:  # the bounding box of the estate's blocks
        bbox, crs = spec["bbox_utm"]
        labels = load_reference(key, spec, args, box(*bbox), crs)
        west, south, east, north = labels.to_crs(4326).total_bounds
        geometry = box(west, south, east, north).__geo_interface__
        study_area = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": geometry}],
        }
    reference = save_reference(labels, key, kind)
    (out_dir / "study_area.geojson").write_text(json.dumps(study_area))
    label = "da" if engine == "delineate-anything" else engine
    output = out_dir / f"fields_spot-pan_{label}-finetune_{spec['year']}.gpkg"
    params = {
        k: v
        for k, v in spec["engine_params"].items()
        if engine == "delineate-anything" or not k.startswith("yolo_")
    }
    print(
        f"\n{'=' * 70}\n{spec['name']}: {len(labels)} reference polygons, "
        f"fine-tuning {engine}\n{'=' * 70}"
    )
    config = AgriboundConfig(
        study_area=str(out_dir / "study_area.geojson"),
        source="spot-pan",
        year=spec["year"],
        engine=engine,
        output_path=str(output),
        gee_project=args.gee_project,
        lulc_filter=False,
        reference_boundaries=str(reference),
        fine_tune=True,
        fine_tune_epochs=FINE_TUNE_EPOCHS,
        fine_tune_split="random",
        fine_tune_val_split=0.2,
        engine_params=params,
        overwrite=args.overwrite,
        **spec["spot"],
    )
    agribound.delineate(config=config)  # in-sample output: never evaluated
    record = read_provenance(output) or {}
    return (record.get("facts") or {}).get("fine_tuned_checkpoint")


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


def load_reference(slug, spec, args, area, crs, min_area=MIN_AREA):
    """Reference polygons of a study or training area (*spec*: its SITES/TRAINING_AREAS entry).

    *min_area* applies to the DWR and SIGPAC maps (the RSPO polygons always use MIN_AREA).
    """
    kind, filters = spec["reference"]
    if kind == "rspo":
        return load_rspo(args, area, crs, filters)
    if kind == "dwr":
        return load_dwr(area, crs, min_area)
    return load_sigpac(area, crs, slug, min_area)


def fmt(value, digits=2):
    """Format a metric (NaN-safe)."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="Tree crops: oil palm, orchards, olives.")
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
        metavar="SLUG",
        choices=list(SITES),
        help=f"Run only these study areas ({', '.join(SITES)}).",
    )
    parser.add_argument("--rspo-zip", default=None, help="A downloaded GeoRSPO zip.")
    parser.add_argument("--no-spot", action="store_true", help="Skip the SPOT 6/7 runs.")
    parser.add_argument("--no-sam", action="store_true", help="Skip SAM 2 refinement.")
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
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def main():
    import geopandas as gpd
    import pandas as pd

    from agribound.evaluate import evaluate_frame

    args = parse_args()
    start_time = time.time()
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    args.spot = not args.no_spot and spot_available(args)
    slugs = args.only or list(SITES)

    checkpoints = {}  # (training area, engine) -> fine-tuned checkpoint
    failed_fine_tunes = {}  # (training area, engine) -> exception name
    engines = ["delineate-anything"] + ([] if args.no_dinov3 else ["dinov3"])
    for key in sorted({SITES[s]["fine_tuned"] for s in slugs} - {None}):
        if not args.spot or args.no_fine_tune:
            break
        for engine in engines:
            try:
                checkpoint = fine_tune_on(key, engine, args)
            except Exception as exc:
                failed_fine_tunes[key, engine] = type(exc).__name__
                print(f"  Fine-tuning {engine} on {key} failed: {type(exc).__name__}: {exc}")
                continue
            if checkpoint:
                checkpoints[key, engine] = checkpoint
            else:
                failed_fine_tunes[key, engine] = "no checkpoint recorded"
                print(f"  Fine-tuning {engine} on {key} recorded no checkpoint.")

    rows = []
    for slug in slugs:
        site = {**SITES[slug], "restrict_precision": slug == "oro"}
        print(f"\n{'=' * 70}\n{site['name']} ({site['year']})\n{'=' * 70}")
        out_dir = OUTPUT_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "study_area.geojson").write_text(
            json.dumps(square_study_area(site["lon"], site["lat"], site["side_m"], site["name"]))
        )
        area, crs = utm_square(site)
        try:
            reference_all = load_reference(slug, site, args, area, crs)
        except Exception as exc:  # e.g. the reference service is down
            print(f"  Reference download failed: {type(exc).__name__}: {exc}; area skipped.")
            continue
        reference = rep_point_inside(reference_all, area)
        references = {"reference": reference}
        if slug == "jaen":
            references["dissolved"] = rep_point_inside(dissolve_by_use(reference_all), area)
        estate = None
        if slug == "twifo":
            estate = reference_all.geometry.union_all().buffer(25).buffer(-25).intersection(area)
        for label, ref in references.items():
            path = save_reference(ref, slug, label)
            n_tree = int((ref["stratum"] == "tree").sum()) if "stratum" in ref else len(ref)
            print(f"  {label}: {len(ref)} polygons ({n_tree} tree crop) -> {path}")
        # How many reference fields each crop-filter variant would keep.
        ref_cfg = base_config(site, slug, "sentinel2", "delineate-anything", "reference", args)
        ref_kept = crop_filter_layers(
            reference, save_reference(reference, slug, "reference"), ref_cfg, args
        )

        for label, engine in (("da-ft", "delineate-anything"), ("dinov3-ft", "dinov3")):
            failure = failed_fine_tunes.get((site["fine_tuned"], engine))
            if failure:  # the run is missing because its fine-tuning failed
                rows.append(
                    {"site": slug, "run": f"spot-pan {label}", "status": f"failed: {failure}"}
                )
        for label, source, engine, overrides in site_runs(slug, site, args, checkpoints):
            name = f"{source} {label}"
            try:
                config = base_config(site, slug, source, engine, label, args, **overrides)
                fields = agribound.delineate(config=config)
            except Exception as exc:  # e.g. no imagery, no TESSERA tiles
                print(f"  {name} failed: {type(exc).__name__}: {exc}")
                rows.append({"site": slug, "run": name, "status": f"failed: {type(exc).__name__}"})
                continue
            out_path = Path(config.output_path)
            fields = fields.to_crs(crs)
            layers = crop_filter_layers(fields, out_path, config, args)
            for ref_label, ref in references.items():
                row, metrics = evaluate_run(
                    fields, ref, site, args, estate=estate if ref_label == "reference" else None
                )
                # Madera and Jaen: the tree-crop fields only, like R and P_ov.
                tree_only = site["strata"] == "stratum"
                frame = evaluate_frame(
                    fields, ref, iou_threshold=0.5, strata="stratum" if tree_only else None
                )
                if tree_only:
                    frame = frame[frame["stratum"] == "tree"]
                row.update(
                    {
                        "site": slug,
                        "run": name if ref_label == "reference" else f"{name} ({ref_label})",
                        "status": "ok",
                        "kept_default": len(layers["crop"]),
                        "kept_tree_crops": len(layers["treecrop"]),
                        "matched_kept_default": matched_kept(fields, layers["crop"], frame),
                        "matched_kept_tree_crops": matched_kept(fields, layers["treecrop"], frame),
                    }
                )
                rows.append(row)
                metrics_path = out_dir / f"metrics_{out_path.stem}_{ref_label}.json"
                metrics_path.write_text(
                    json.dumps({"row": row, "metrics": metrics}, indent=2, default=str)
                )

        n_ref = len(reference)
        print(
            f"\n  Crop filter on the reference: the default rule keeps "
            f"{len(ref_kept['crop'])} of {n_ref}, lulc_tree_crops keeps "
            f"{len(ref_kept['treecrop'])} of {n_ref}."
        )
        header = (
            f"  {'Run':<34} {'Fields':>6} {'R':>5} {'P_ov':>5} {'F1_ov':>5} {'IoU':>5} "
            f"{'bF1':>5} {'Merge':>5} {'Split':>5} {'Kept':>9}"
        )
        print(header)
        for row in rows:
            if row["site"] != slug or row["status"] != "ok":
                continue
            kept = f"{row['kept_default']}/{row['kept_tree_crops']}"
            print(
                f"  {row['run']:<34} {row['fields']:>6} {fmt(row['recall']):>5} "
                f"{fmt(row['precision_ov']):>5} {fmt(row['f1_ov']):>5} {fmt(row['iou_mean']):>5} "
                f"{fmt(row['boundary_f1']):>5} {fmt(row['merged']):>5} {fmt(row['split']):>5} "
                f"{kept:>9}"
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print("\nKept = polygons kept by the default crop filter / with lulc_tree_crops=True.")
    print(f"Summary: {OUTPUT_DIR / 'summary.csv'}")

    # Twifo on an interactive map: SPOT-Pan and Sentinel-2 Delineate-Anything, FTW, reference.
    twifo = OUTPUT_DIR / "twifo"
    layers = [
        (twifo / "fields_spot-pan_da_2020.gpkg", "Delineate-Anything, SPOT-Pan"),
        (twifo / "fields_sentinel2_da_2020.gpkg", "Delineate-Anything, Sentinel-2"),
        (twifo / "fields_sentinel2_ftw_2020.gpkg", "FTW, Sentinel-2"),
    ]
    layers = [(p, label) for p, label in layers if p.exists()]
    # The reference itself (not its _crop/_treecrop copies), newest version last.
    refs = sorted(
        REFERENCE_DIR.glob(f"twifo_reference_{'?' * 12}.gpkg"), key=lambda p: p.stat().st_mtime
    )
    if layers and refs:
        from agribound.visualize import show_comparison

        frames = [gpd.read_file(p) for p, _ in layers] + [gpd.read_file(refs[-1])]
        show_comparison(
            frames,
            labels=[label for _, label in layers] + ["RSPO blocks"],
            output_html=str(twifo / "comparison.html"),
        )
        print(f"Map: {twifo / 'comparison.html'}")
    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
