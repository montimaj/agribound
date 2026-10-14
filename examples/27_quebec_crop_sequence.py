"""
27 — Quebec: Crop-History Zones from AAFC Crop Maps against Declared Parcels (2016-2023)

What eight yearly crop-type maps can give where no field map exists, and where
they stop. The crop-sequence route (source ``aci-sequence``, engine
``crop-sequence``) turns the AAFC Annual Crop Inventory (ACI) maps of
2016-2023 into zones of uniform crop history, the way USDA NASS makes its Crop
Sequence Boundaries, in two 5 km squares in Quebec. The zones are scored
against the parcels that farmers declared to La Financière agricole du Québec
(FADQ) for 2023, beside Delineate-Anything v2 as released on a Sentinel-2
composite of 2023 and beside two image-blind baselines. A third step adds the
ACI history to the Delineate-Anything outlines
(``agribound.attach_crop_history``): outlines from imagery, crop history from
the maps.

The zones are not field outlines. Neighbouring parcels with the same crop
sequence fall into one zone, and parcels narrower than about 60-90 m or
smaller than 1 ha are absorbed or lost: the per-year focal mode has a radius
of 3 pixels (90 m) and regions under 1 ha are merged into a neighbour.

The squares (UTM zone and lower-left corner of the 5 km square; Parcels: the
2023 parcels of at least 2,500 m2 whose representative point lies in the
square, with those that have a production declared for 2023 in brackets;
Crops: the commonest crop groups of the declared parcels, as recorded in the
pilot's plan on 2026-10-09):

    Square             Zone  Corner (m)       Centre (lon, lat)  Parcels    Crops
    bas-saint-laurent  19N   480000, 5320000  -69.2349, 48.0555  526 (490)  hay 49 %, cereals 34 %
    monteregie         18N   610000, 5010000  -73.5662, 45.2570  498 (443)  soybeans 42 %, corn 27 %

Bas-Saint-Laurent is the square of the crop-sequence pilot of 2026-10-09: the
cell of the 5 km grid with the most declared parcels of at least 2,500 m2,
inside Quebec and with ACI pixels over all of it in every year 2016-2023.
Montérégie is the next cell by the same count, which the pilot recorded but
did not run; it is a soybean and grain-corn landscape (corn: grain corn),
where the parcels are narrower (35 % have a short side under 60 m, against
24 % in Bas-Saint-Laurent; in Bas-Saint-Laurent silage corn is 8 % of the
declared parcels). Neither square was chosen after a result of this script.

Reference: BDPPAD (Base de données des parcelles et productions agricoles
déclarées), version 03, year 2023 (``BDPPAD_V03_2023.zip``, 172,750,925 bytes,
SHA-256 pinned in ``BDPPAD_SHA256``), downloaded once from FADQ's open-data
page into ``outputs/quebec_crop_sequence/`` and read in place. The evaluated
parcels are those of at least 2,500 m2 (EPSG:6933, the predictions' minimum
area) whose representative point lies in the square and that have a
production declared for 2023 (``NBPRO`` >= 1). Read the scores with what a
BDPPAD parcel is (FADQ user guide, March 2025):

    - an agricultural parcel tied to the file of a FADQ client, digitised by
      FADQ staff on aerial images finer than 50 cm; the crops are the
      clients' declarations, not observations;
    - the file is not exhaustive: it holds only producers insured in
      programmes that measure areas, and since 2010 the parcels of clients
      insured only for hay or silage corn on a feed-needs basis are no longer
      measured or updated, which matters in the hay-dominated
      Bas-Saint-Laurent square;
    - the year of the file is neither the year of the images nor the year an
      outline was drawn, and parcels without a 2023 declaration (left out
      here) can carry outlines that were not checked again.

Runs (each a separate agribound run with ``lulc_filter=False``):

    - ``zones``: ``source="aci-sequence"``, ``year=2023``,
      ``crop_sequence_years=8`` (ACI 2016-2023, read at 30 m on the square's
      UTM grid; one Earth Engine request for the image list and one
      download), ``engine="crop-sequence"`` with its defaults (the ACI
      agricultural classes 120-199 in more than half of the years, Urban and
      Developed (34) as a barrier, a per-year focal mode of radius 3 px,
      regions of equal sequence, regions under 1 ha merged into their
      largest neighbour); CPU only, no model;
    - ``sentinel2``: Delineate-Anything v2 (``large_v2``), used as released,
      on a Sentinel-2 median of 1 April to 31 October 2023 with up to 1,000
      detections per tile, the settings of examples 23-25 (the window keeps
      winter snow out of the median);
    - then ``agribound.attach_crop_history`` adds to the Delineate-Anything
      outlines, and to the reference parcels, the most frequent ACI class of
      each year (2016-2023) from the zones run's stack.

Evaluation (``agribound.evaluate``): one-to-one matching at IoU >= 0.5,
boundary F at 30 and 60 m, merge and split rates (the shares of reference
parcels whose main prediction also covers 25 % of another parcel, and of
parcels split between predictions that each cover 10 % of them), and the
chance baselines of ``evaluate(..., baselines=...)`` at 60 m: a blind square
grid with cells of the run's median polygon area over the square (mean of 36
offsets) and the reference shifted east and north by half its median width.
On small-field squares a boundary F at a loose tolerance is high for any
dense set of lines, so read F1 and the baselines with it. The script also
reports how many adjacent declared parcels (sharing at least 60 m of
boundary within 15 m) have the same 2016-2023 ACI history, which no method
that reads only these maps can separate.

Pilot numbers (the crop-sequence pilot of 2026-10-09, research code, the
Bas-Saint-Laurent square, all 526 parcels; the 490 declared ones in
brackets): the zones, made by the pilot's prototype of this engine from ACI
2016-2023, F1 0.182 (0.183) and boundary F at 60 m 0.944 (0.927);
Delineate-Anything v2 on a Sentinel-2 2024 composite with agribound's
defaults (a calendar-year median, 300 detections per tile and the Dynamic
World crop filter, which kept 223 of its 525 polygons) F1 0.315 (0.320),
boundary F at 60 m 0.714 (0.718); 13 % of the 691 adjacent parcel pairs had
the same eight-year ACI sequence. On the pilot's ten US squares (CDL) the
zones scored F1 0.006-0.084 and Delineate-Anything on Sentinel-2 0.011-0.649,
and the boundary precision expected of lines drawn without regard to the
fields was 0.967 or more at 60 m on seven of them. These are pilot numbers;
this script measures its own (below). Nothing here is measured on
large-field landscapes such as the Prairies.

This script's results (agribound 1.1.0, 2026-10-11, Delineate-Anything on
MPS; against the declared parcels; Polygons: those scored, by representative
point in the square; P, R: precision and recall; BF60: boundary F at 60 m;
Merge, Split: see Evaluation):

    Square             Run                          Polygons  P     R     F1     BF60  Merge  Split
    bas-saint-laurent  zones, ACI 2016-2023            486    0.20  0.20  0.197  0.92  0.66   0.52
    bas-saint-laurent  Delineate-Anything, S2 2023     456    0.50  0.47  0.486  0.93  0.42   0.16
    monteregie         zones, ACI 2016-2023            397    0.13  0.11  0.119  0.88  0.78   0.49
    monteregie         Delineate-Anything, S2 2023     535    0.43  0.52  0.472  0.91  0.36   0.20

A run of 2026-10-10 with ``--device cpu`` (the development code before
1.1.0) gave the zones the same F1 and Delineate-Anything 457 and 534
polygons with F1 0.490 and 0.471.

The blind grids scored F1 0.007-0.018 and BF60 0.85-0.89, and the reference
shifted by half its median width (54 and 40 m) F1 0.092 and 0.124 and BF60
0.92 and 0.99. So at 60 m every layer, the shifted reference included, has
about the same boundary F, which does not separate them; the object scores
do: the zones match one declared parcel in five (Bas-Saint-Laurent) or nine
(Montérégie), and merge or split most of the others; Delineate-Anything
matches about one in two. 82 of the 638 adjacent declared parcel pairs in
Bas-Saint-Laurent (13 %) and 128 of 546 in Montérégie (23 %) have the same
ACI class in every year 2016-2023. The Sentinel-2 settings and year differ
from the pilot's (above), which may explain most of the difference between
Delineate-Anything's F1 here and in the pilot. The zones differ from the
pilot's too: the pilot built them on an EPSG:5070 grid without
post-processing, this script on the square's UTM grid with a 10-pixel pad
and agribound's default post-processing. Built that way on the pilot's 11
squares (2026-10-10), the engine's zones differed from the pilot's by -0.006
to +0.017 in F1 and by -0.035 to -0.004 in boundary F at 60 m; in
Bas-Saint-Laurent, on the declared parcels, F1 0.183 became 0.197 and
boundary F at 60 m 0.927 became 0.924.

Outputs (``outputs/quebec_crop_sequence/``): the BDPPAD zip; per square
(``<square>/``) the square (``study_area.geojson``), the evaluated parcels
with their ACI history (``reference.gpkg``), the zones
(``zones_aci-sequence_2016-2023.gpkg``), the Delineate-Anything outlines
(``fields_sentinel2_da_2023.gpkg``) and the same outlines with their ACI
history (``fields_sentinel2_da_2023_aci_history.gpkg``), the metrics of each
run and a map; ``summary.csv`` with one row per square and run (a run with
``--only`` replaces the rows of its squares and keeps the others). agribound
reuses an output whose provenance record matches the run's settings, so a
second run only re-evaluates; ``--overwrite`` recomputes.

``--only monteregie`` runs one square. ``--dry-run`` downloads the BDPPAD
file if it is not there yet, reads the squares' parcels and prints every
run's configuration; it makes no Earth Engine request and writes nothing
else. ``--device cpu`` runs Delineate-Anything on the CPU.

Runtime: 1.9 minutes (112 s) for both squares with agribound 1.1.0 on
2026-10-11 (Apple M2 Max, Delineate-Anything on MPS, its weights already
cached), with the BDPPAD file already in place: the zones runs took 2-4 s
each and the Delineate-Anything runs 20 and 23 s, their Sentinel-2 downloads
included. With ``--device cpu`` on 2026-10-10 both squares took 1.9 minutes
(Delineate-Anything about 30 s each), and the download of the BDPPAD file
with the dry run 22 s.

Attribution: Base de données des parcelles et productions agricoles
déclarées (BDPPAD), La Financière agricole du Québec, CC BY 4.0 (Données
Québec); Annual Crop Inventory, Agriculture and Agri-Food Canada, Open
Government Licence - Canada; contains modified Copernicus Sentinel data
(2023). Thanks to Jacob T. Stid (Michigan State University) and Jeremy Rapp,
who shared the crop-sequence recipe that the crop-sequence route follows.

Prerequisites:
    pip install "agribound[gee,crop-sequence,delineate-anything]"
    agribound auth --project YOUR_GEE_PROJECT
    Run from the repository root:
        python examples/27_quebec_crop_sequence.py
"""

import argparse
import hashlib
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/quebec_crop_sequence")
USER_AGENT = f"agribound/{agribound.__version__} (+https://github.com/montimaj/agribound)"
SIDE_M = 5000  # side of the squares
MIN_AREA = 2500  # m^2 (EPSG:6933); the predictions' default minimum area, also for the reference
IOU_THRESHOLD = 0.5
TOLERANCES_M = (30, 60)  # boundary F tolerances; the chance baselines use the larger
YEAR = 2023  # the BDPPAD year, the last ACI year and the Sentinel-2 year
CROP_SEQUENCE_YEARS = 8  # ACI 2016-2023
S2_SEASON = ("04-01", "10-31")  # Sentinel-2 composite window (month-day, inclusive)
S2_MAX_DETECTIONS = 1000  # per tile; at 10 m one 256 px tile covers 2.56 km
MERGE_SHARE = 0.25  # merged: the main prediction also covers this share of another parcel
SPLIT_SHARE = 0.10  # split: at least two predictions each cover this share of the parcel
#: Adjacent parcels: at least this length of one boundary within ADJACENCY_TOLERANCE_M of the
#: other parcel (two 30 m pixels of shared edge, within half a pixel).
ADJACENCY_MIN_SHARED_M = 60.0
ADJACENCY_TOLERANCE_M = 15.0
#: BDPPAD 2023 (version 03): FADQ's open-data file, CC BY 4.0, pinned by size and SHA-256
#: (last modified 2025-01-20, checked on 2026-10-10).
BDPPAD_URL = "https://www.fadq.qc.ca/fileadmin/geo4tb/BDPPAD/BDPPAD_V03_2023.zip"
BDPPAD_BYTES = 172_750_925
BDPPAD_SHA256 = "9258447ac941b23efc78bf582c8555f66d1eb0a732c5949fa2f169697c139f82"
BDPPAD_MEMBER = "BDPPAD_v03_AN_2023_s_20250120.shp"  # EPSG:4269
BDPPAD_COLUMNS = ["IDPAR", "SUPHEC", "GROPRO", "DESGROPRO", "NBPRO", "CODPRO1", "DESCODPR1"]
BDPPAD_LABEL = "BDPPAD 2023, declared parcels (La Financière agricole du Québec, CC BY 4.0)"
#: The squares: EPSG code of the UTM zone and lower-left corner (see the docstring).
SQUARES = {
    "bas-saint-laurent": {"epsg": 32619, "corner": (480000, 5320000)},
    "monteregie": {"epsg": 32618, "corner": (610000, 5010000)},
}
RUNS = {
    "zones": f"Crop-history zones, ACI {YEAR - CROP_SEQUENCE_YEARS + 1}-{YEAR}",
    "sentinel2": f"Delineate-Anything v2, Sentinel-2 {YEAR}",
}
#: Short credit of each run's input, in the map layer names.
CREDITS = {
    "zones": "AAFC Annual Crop Inventory, OGL-Canada",
    "sentinel2": "contains modified Copernicus Sentinel data",
}


def square(name):
    """``(epsg, polygon)``: the 5 km square *name* in its UTM zone."""
    from shapely.geometry import box

    spec = SQUARES[name]
    x, y = spec["corner"]
    return spec["epsg"], box(x, y, x + SIDE_M, y + SIDE_M)


def reproject(area, epsg, crs):
    """*area* (a polygon in *epsg*) in *crs*, with its edges densified every 100 m."""
    import geopandas as gpd

    return gpd.GeoSeries([area.segmentize(100)], crs=epsg).to_crs(crs).iloc[0]


def study_area(name):
    """A GeoJSON FeatureCollection of the square *name* in EPSG:4326."""
    from shapely.geometry import mapping

    epsg, area = square(name)
    feature = {
        "type": "Feature",
        "geometry": mapping(reproject(area, epsg, 4326)),
        "properties": {"name": f"{name} square ({BDPPAD_LABEL})"},
    }
    return {"type": "FeatureCollection", "features": [feature]}


def bdppad_zip():
    """The BDPPAD 2023 zip in OUTPUT_DIR, downloaded first if needed; checked by SHA-256."""
    path = OUTPUT_DIR / Path(BDPPAD_URL).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        print(f"Downloading {BDPPAD_URL} ({BDPPAD_BYTES / 1e6:.0f} MB) ...")
        request = urllib.request.Request(BDPPAD_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        partial.replace(path)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    if path.stat().st_size != BDPPAD_BYTES or digest.hexdigest() != BDPPAD_SHA256:
        raise SystemExit(
            f"{path}: {path.stat().st_size:,} bytes, SHA-256 {digest.hexdigest()}; expected "
            f"{BDPPAD_BYTES:,} bytes and {BDPPAD_SHA256}. Delete it to download it again."
        )
    return path


def rep_point_inside(gdf, area):
    """Rows whose representative point lies inside *area* (same CRS)."""
    return gdf[gdf.representative_point().within(area).to_numpy()].copy()


def load_reference(archive, name):
    """The declared parcels of square *name*, and the counts of those left out.

    Returns ``(reference, counts)``: the parcels of at least MIN_AREA whose
    representative point lies in the square and that have a production declared
    for 2023, in the square's UTM zone; ``counts`` holds ``n_parcels`` (all such
    parcels), ``n_small`` (smaller ones dropped) and ``n_undeclared``. Reads only
    the parcels that reach into the square's bounding box.
    """
    import pyogrio

    from agribound.postprocess import filter_polygons

    epsg, area = square(name)
    path = f"/vsizip/{archive}/{BDPPAD_MEMBER}"
    bbox = reproject(area, epsg, pyogrio.read_info(path)["crs"]).bounds
    parcels = pyogrio.read_dataframe(path, bbox=bbox, columns=BDPPAD_COLUMNS)
    inside = rep_point_inside(parcels.to_crs(epsg), area)
    sized = filter_polygons(inside, min_area_m2=MIN_AREA).reset_index(drop=True)
    declared = sized["NBPRO"].fillna(0).astype(int) >= 1
    counts = {
        "n_parcels": len(sized),
        "n_small": len(inside) - len(sized),
        "n_undeclared": int((~declared).sum()),
    }
    return sized[declared.to_numpy()].reset_index(drop=True), counts


def crop_groups(reference, top=3):
    """The commonest declared crop groups (``DESGROPRO``, FADQ's names) and their shares of
    the parcels that have one."""
    shares = reference["DESGROPRO"].value_counts(normalize=True).head(top)
    return ", ".join(f"{group} {100 * share:.0f} %" for group, share in shares.items())


def run_configs(name, args):
    """``[(run, AgriboundConfig)]`` of square *name*: the zones and Delineate-Anything."""
    from agribound.config import AgriboundConfig

    out_dir = OUTPUT_DIR / name
    common = {
        "study_area": str(out_dir / "study_area.geojson"),
        "year": YEAR,
        "gee_project": args.gee_project,
        "lulc_filter": False,
        "overwrite": args.overwrite,
    }
    first = YEAR - CROP_SEQUENCE_YEARS + 1
    zones = AgriboundConfig(
        source="aci-sequence",
        engine="crop-sequence",
        crop_sequence_years=CROP_SEQUENCE_YEARS,
        output_path=str(out_dir / f"zones_aci-sequence_{first}-{YEAR}.gpkg"),
        **common,
    )
    imagery = AgriboundConfig(
        source="sentinel2",
        engine="delineate-anything",
        date_range=(f"{YEAR}-{S2_SEASON[0]}", f"{YEAR}-{S2_SEASON[1]}"),
        engine_params={"max_detections": S2_MAX_DETECTIONS},
        device=args.device,
        output_path=str(out_dir / f"fields_sentinel2_da_{YEAR}.gpkg"),
        **common,
    )
    return [("zones", zones), ("sentinel2", imagery)]


def merge_split_rates(pred, ref):
    """Shares of reference parcels that are merged with another or split (see the docstring)."""
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


def adjacent_pairs(ref):
    """Pairs ``(i, j)``, ``i < j``, of adjacent parcels: at least ADJACENCY_MIN_SHARED_M of
    one boundary lies within ADJACENCY_TOLERANCE_M of the other parcel."""
    import numpy as np
    import shapely

    geoms = shapely.make_valid(np.asarray(ref.geometry.values, dtype=object))
    buffers = shapely.buffer(geoms, ADJACENCY_TOLERANCE_M)
    i, j = shapely.STRtree(geoms).query(buffers, predicate="intersects")
    keep = i < j
    i, j = i[keep], j[keep]
    shared = np.maximum(
        shapely.length(shapely.intersection(shapely.boundary(geoms[i]), buffers[j])),
        shapely.length(shapely.intersection(shapely.boundary(geoms[j]), buffers[i])),
    )
    adjacent = shared >= ADJACENCY_MIN_SHARED_M
    return i[adjacent], j[adjacent]


def same_history(ref_history, years):
    """Adjacent parcel pairs and those with the same ACI class in every year (none missing)."""
    import numpy as np

    columns = [f"agribound:crop_{year}" for year in years]
    codes = ref_history[columns].to_numpy(dtype=int)
    i, j = adjacent_pairs(ref_history)
    known = (codes[i] > 0).all(axis=1) & (codes[j] > 0).all(axis=1)
    same = known & (codes[i] == codes[j]).all(axis=1)
    with_twin = np.zeros(len(codes), dtype=bool)
    with_twin[i[same]] = with_twin[j[same]] = True
    return {
        "n_adjacent_pairs": int(len(i)),
        "n_pairs_same_history": int(same.sum()),
        "share_pairs_same_history": float(same.sum() / len(i)) if len(i) else float("nan"),
        "share_parcels_with_same_history_neighbour": float(with_twin.mean())
        if len(codes)
        else float("nan"),
    }


def score(pred, ref, area, epsg):
    """Scores of *pred* against the declared parcels *ref* in the square *area*."""
    import geopandas as gpd

    from agribound.evaluate import evaluate

    pred = rep_point_inside(pred.to_crs(ref.crs), area)
    study = gpd.GeoDataFrame(geometry=[area], crs=epsg)
    loose = max(TOLERANCES_M)
    metrics = {}
    for tolerance in TOLERANCES_M:
        options = {"study_area": study, "shift_m": "auto"} if tolerance == loose else False
        metrics[tolerance] = evaluate(
            pred, ref, iou_threshold=IOU_THRESHOLD, boundary_tolerance_m=tolerance,
            baselines=options,
        )  # fmt: skip
    m = metrics[loose]
    baselines = m.get("baselines") or {}
    grid = baselines.get("blind_grid") or {}
    shifted = baselines.get("shifted_reference") or {}
    merged, split = merge_split_rates(pred, ref)
    scores = {
        "n_predicted": len(pred),
        "tp": m["count_tp"],
        "fp": m["count_fp"],
        "fn": m["count_fn"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "iou_mean": m["iou_mean"],
        "area_weighted_f1": m["area_weighted_f1"],
        **{f"bf{t}": metrics[t]["boundary_f1"] for t in TOLERANCES_M},
        "merge_rate": merged,
        "split_rate": split,
        "grid_cell_m": grid.get("cell_side_m", math.nan),
        "grid_f1": grid.get("f1", math.nan),
        f"grid_bf{loose}": grid.get("boundary_f1", math.nan),
        "shift_m": shifted.get("shift_m", math.nan),
        "shifted_f1": shifted.get("f1", math.nan),
        f"shifted_bf{loose}": shifted.get("boundary_f1", math.nan),
    }
    return pred, metrics, scores


def describe(name, reference, counts):
    """Print a square and its reference parcels."""
    import geopandas as gpd

    epsg, area = square(name)
    centre = gpd.GeoSeries([area.centroid], crs=epsg).to_crs(4326).iloc[0]
    print(f"\n{name}: EPSG:{epsg} corner {SQUARES[name]['corner']}, centre {centre.x:.4f}, "
          f"{centre.y:.4f}")  # fmt: skip
    size_ha = reference.to_crs(6933).area / 1e4
    print(
        f"  reference: {len(reference):,} declared parcels of the {counts['n_parcels']:,} of at "
        f"least {MIN_AREA:,} m2 ({counts['n_undeclared']:,} without a 2023 declaration and "
        f"{counts['n_small']:,} smaller ones left out), median {size_ha.median():.2f} ha"
    )
    print(f"  declared crop groups: {crop_groups(reference)}")


def dry_run(names, args):
    """Print every run's configuration (validated; nothing is run or written)."""
    print(f"\n{'=' * 70}\nDry run: configurations (nothing is run or written)\n{'=' * 70}")
    for name in names:
        for run, config in run_configs(name, args):
            window = (
                " to ".join(config.date_range)
                if config.date_range
                else f"{config.year - config.crop_sequence_years + 1}-{config.year}"
            )
            print(
                f"  {name:<18} {run:<9} {config.source:<12} {config.engine:<18} {window:<24} "
                f"{config.engine_params or ''} -> {config.output_path}"
            )


def fmt(value, digits=3):
    """Format a metric (NaN-safe)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if math.isnan(value) else f"{value:.{digits}f}"


def print_table(rows):
    """The per-square table and the chance baselines."""
    loose = max(TOLERANCES_M)
    print(
        f"\n{'Square':<18} {'Run':<10} {'Ref':>4} {'Pred':>5} {'TP':>4} {'P':>5} {'R':>5} "
        f"{'F1':>6} {'IoU':>5} {'BF30':>5} {'BF60':>5} {'Merge':>5} {'Split':>5}"
    )
    for row in rows:
        start = f"{row['square']:<18} {row['run']:<10} {row['n_reference']:>4}"
        if row["status"] != "ok":
            print(f"{start} {row['status']}")
            continue
        print(
            f"{start} {row['n_predicted']:>5} {row['tp']:>4} {fmt(row['precision'], 2):>5} "
            f"{fmt(row['recall'], 2):>5} {fmt(row['f1']):>6} {fmt(row['iou_mean'], 2):>5} "
            f"{fmt(row['bf30'], 2):>5} {fmt(row['bf60'], 2):>5} {fmt(row['merge_rate'], 2):>5} "
            f"{fmt(row['split_rate'], 2):>5}"
        )
    print(f"\nChance baselines (no image; boundary F at {loose} m):")
    shown = set()
    for row in rows:
        if row["status"] != "ok":
            continue
        print(
            f"  {row['square']:<18} blind grid of the {row['run']} run's median size "
            f"({fmt(row['grid_cell_m'], 0)} m cells, mean of 36 offsets): F1 "
            f"{fmt(row['grid_f1'])}, BF{loose} {fmt(row[f'grid_bf{loose}'], 2)}"
        )
        if row["square"] not in shown:
            shown.add(row["square"])
            print(
                f"  {row['square']:<18} reference shifted {fmt(row['shift_m'], 0)} m east and "
                f"north: F1 {fmt(row['shifted_f1'])}, BF{loose} "
                f"{fmt(row[f'shifted_bf{loose}'], 2)}"
            )
    print(
        f"\nOne-to-one matching at IoU >= {IOU_THRESHOLD:g}; IoU: mean IoU of the matched pairs; "
        "BF30, BF60: boundary F at 30 and 60 m; Merge, Split: shares of the parcels merged with "
        f"another or split. Reference: declared parcels of at least {MIN_AREA:,} m2 by "
        "representative point."
    )


def merged_summary(path, rows):
    """This run's *rows*, and the rows of the earlier file at *path* (if any) of every square
    that this run gave no row, in the order of SQUARES."""
    import pandas as pd

    summary = pd.DataFrame(rows)
    if Path(path).exists():
        previous = pd.read_csv(path)
        scored = set(summary["square"]) if len(summary) else set()
        earlier = previous[~previous["square"].isin(scored)]
        summary = pd.concat([earlier, summary], ignore_index=True)
    if len(summary):
        order = {name: i for i, name in enumerate(SQUARES)}
        summary = summary.sort_values("square", key=lambda col: col.map(order), kind="stable")
    return summary.reset_index(drop=True)


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(
        description="Quebec: crop-history zones from the AAFC Annual Crop Inventory against "
        "the 2023 declared parcels (BDPPAD), beside Delineate-Anything v2 on Sentinel-2."
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
        metavar="SQUARE",
        choices=list(SQUARES),
        help=f"Run only these squares ({', '.join(SQUARES)}).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device of the Delineate-Anything runs (default: auto).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute outputs that already exist."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download the BDPPAD file if needed, print the squares, the reference counts and "
        "every run's configuration; run nothing (no Earth Engine) and write nothing else.",
    )
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def main():
    from agribound.visualize import show_comparison

    args = parse_args()
    start_time = time.time()
    names = list(dict.fromkeys(args.only or SQUARES))
    archive = bdppad_zip()
    print(f"Reading the {BDPPAD_LABEL} of {', '.join(names)} ({archive})")
    squares = {}
    for name in names:
        reference, counts = load_reference(archive, name)
        describe(name, reference, counts)
        squares[name] = (reference, counts)
    if args.dry_run:
        dry_run(names, args)
        return

    from agribound.auth import setup_gee

    setup_gee(project=args.gee_project)
    years = list(range(YEAR - CROP_SEQUENCE_YEARS + 1, YEAR + 1))
    rows = []
    for name, (reference, counts) in squares.items():
        epsg, area = square(name)
        out_dir = OUTPUT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "study_area.geojson").write_text(json.dumps(study_area(name)))
        layers, history = {}, {}
        for run, config in run_configs(name, args):
            row = {
                "square": name,
                "run": run,
                "source": config.source,
                "engine": config.engine,
                "year": config.year,
                "window": " to ".join(config.date_range)
                if config.date_range
                else f"{years[0]}-{years[-1]}",
                "engine_params": json.dumps(config.engine_params, sort_keys=True),
                "utm_epsg": epsg,
                "n_reference": len(reference),
                "n_parcels": counts["n_parcels"],
                "n_undeclared": counts["n_undeclared"],
            }
            print(f"\n{'=' * 70}\n{name}: {RUNS[run]}\n{'=' * 70}")
            try:
                pred = agribound.delineate(config=config)
            except Exception as exc:  # e.g. no imagery, a package missing
                print(f"  {name} {run} failed: {type(exc).__name__}: {exc}")
                rows.append({**row, "status": f"failed: {type(exc).__name__}"})
                continue
            pred, metrics, scores = score(pred, reference, area, epsg)
            rows.append({**row, "status": "ok", **scores, "output": config.output_path})
            stem = Path(config.output_path).stem
            (out_dir / f"metrics_{stem}.json").write_text(
                json.dumps(
                    to_jsonable({"row": rows[-1], "metrics": metrics}), indent=2, allow_nan=False
                )
            )
            print(f"  {name} {run}: {len(pred)} polygons, F1 {fmt(scores['f1'])}")
            layers[run] = (pred, f"{RUNS[run]} ({CREDITS[run]})")
            if run == "zones":  # the zones run's ACI stack (cached; no Earth Engine request)
                history["stack"] = agribound.build_composite(config)

        # Outlines from imagery, crop history from the maps; and the reference's history.
        if "stack" in history:
            stack = history["stack"]
            reference = agribound.attach_crop_history(reference, stack)
            twins = same_history(reference, years)
            print(
                f"\n{name}: {twins['n_pairs_same_history']} of {twins['n_adjacent_pairs']} "
                f"adjacent declared parcel pairs ({fmt(twins['share_pairs_same_history'], 2)}) "
                f"have the same ACI class in every year {years[0]}-{years[-1]}"
            )
            for row in rows:
                if row["square"] == name:
                    row.update(twins)
            if "sentinel2" in layers:
                outlines = agribound.attach_crop_history(layers["sentinel2"][0], stack)
                legend = outlines.attrs["crop_history"]["legend"]
                target = out_dir / f"fields_sentinel2_da_{YEAR}_aci_history.gpkg"
                outlines.to_file(target, layer="fields")
                common = outlines[f"agribound:crop_{YEAR}"].value_counts().head(3)
                named = ", ".join(f"{legend.get(int(c), c)} ({n})" for c, n in common.items())
                print(f"  Delineate-Anything outlines with their ACI history: {target}")
                print(f"  commonest ACI {YEAR} classes among them: {named}")
        if len(reference):  # the evaluated reference, for maps such as the gallery's
            reference.to_file(out_dir / "reference.gpkg", layer="reference")

        # The square's map: the reference and both runs.
        if layers:
            map_path = out_dir / f"comparison_{YEAR}.html"
            web_map = show_comparison(
                [reference] + [frame for frame, _ in layers.values()],
                labels=[f"{name}: {BDPPAD_LABEL}"] + [label for _, label in layers.values()],
                output_html=str(map_path),
            )
            show_in_notebook(web_map)
            print(f"Map: {map_path}")

    # This run's rows replace those of its squares; the others are kept.
    path = OUTPUT_DIR / "summary.csv"
    if rows:
        # convert_dtypes keeps the counts integers when a failed run leaves them empty
        merged_summary(path, rows).convert_dtypes().to_csv(path, index=False)
        print_table(rows)
        print(f"Summary: {path}")
    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
