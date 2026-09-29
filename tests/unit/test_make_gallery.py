"""tools/make_gallery.py renders an entry from a synthetic run root.

A small Sentinel-2-like composite, two predicted fields, a reference layer and
the provenance sidecar that points at the composite are written to a temporary
run root; the tool must find the composite through the sidecar, draw it and
report the facts the gallery captions quote. The tool is not shipped in the
wheel, so the tests are skipped when the source checkout is not available.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "make_gallery.py"

if not TOOL.exists():  # e.g. running from an installed wheel
    pytest.skip("tools/make_gallery.py is not available", allow_module_level=True)

pytest.importorskip("matplotlib")
rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import box  # noqa: E402

MODULE_NAME = "make_gallery"
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module  # dataclasses look the module up while the class bodies run
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(MODULE_NAME, None)


def _run_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    out = root / "outputs" / "demo"
    out.mkdir(parents=True)
    tif = out / "composite.tif"
    rng = np.random.default_rng(0)
    data = rng.uniform(200, 3000, size=(len(S2_BANDS), 120, 100)).astype("float32")
    transform = rasterio.transform.from_origin(500000, 4000000, 10, 10)
    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        width=100,
        height=120,
        count=len(S2_BANDS),
        dtype="float32",
        crs="EPSG:32613",
        transform=transform,
        nodata=float("nan"),
    ) as dst:
        dst.write(data)
        dst.descriptions = tuple(S2_BANDS)
        dst.update_tags(
            AGRIBOUND_DATE_START="2024-01-01",
            AGRIBOUND_DATE_END_EXCLUSIVE="2025-01-01",
            AGRIBOUND_COMPOSITE_METHOD="median",
            AGRIBOUND_N_IMAGES="7",
        )
    fields = gpd.GeoDataFrame(
        {"agribound:sam_refined": [True, False]},
        geometry=[box(500100, 3999100, 500400, 3999400), box(500500, 3999500, 500900, 3999900)],
        crs="EPSG:32613",
    )
    gpkg = out / "fields.gpkg"
    fields.to_file(gpkg, driver="GPKG")
    sidecar = {
        "agribound_version": "1.0.0",
        "status": "success",
        "config": {"source": "sentinel2", "engine": "delineate-anything", "year": 2024},
        "facts": {"raster_path": "outputs/demo/composite.tif"},
        "engine_meta": {"model_key": "large_v2"},
    }
    (out / "fields.gpkg.provenance.json").write_text(json.dumps(sidecar))
    ref = gpd.GeoDataFrame(geometry=[box(500090, 3999090, 500410, 3999410)], crs="EPSG:32613")
    ref.to_file(out / "reference.gpkg", driver="GPKG")
    return root


def test_render_uses_the_recorded_composite(tool, tmp_path):
    root = _run_root(tmp_path)
    entry = tool.Entry(
        "99",
        "Demo_example",
        "Demo",
        [tool.Layer("outputs/demo/fields.gpkg", "Demo", highlight="agribound:sam_refined")],
        reference="outputs/demo/reference.gpkg",
        zoom_m=400,
        inset=False,  # the locator inset downloads Natural Earth; tested separately below
    )
    stats = tool.render(entry, root, tmp_path / "gallery", "composite")

    png = tmp_path / "gallery" / "Demo_example.png"
    assert png.exists() and png.stat().st_size > 0
    from PIL import Image

    preview = tmp_path / "gallery" / "preview" / "Demo_example.webp"
    with Image.open(png) as full, Image.open(preview) as small:
        assert small.format == "WEBP" and small.mode == "RGB"
        assert small.width == tool.PREVIEW_W
        assert small.height == round(full.height * tool.PREVIEW_W / full.width)
    assert stats["preview"].endswith("preview/Demo_example.webp")
    assert stats["background"] == "composite"
    layer = stats["layers"][0]
    assert layer["n_polygons"] == 2
    assert layer["area_ha_total"] == pytest.approx(9.0 + 16.0)
    assert layer["source"] == "sentinel2"
    assert layer["composite"]["date_start"] == "2024-01-01"
    assert layer["composite"]["n_images"] == "7"
    assert stats["reference"]["n_polygons"] == 1
    assert stats["zoom_window_m"] == 400


def test_rgb_bands_come_from_the_band_descriptions(tool, tmp_path):
    root = _run_root(tmp_path)
    with rasterio.open(root / "outputs" / "demo" / "composite.tif") as src:
        idx, grey = tool._rgb_indices(src, "sentinel2")
    assert idx == [4, 3, 2]  # B4, B3, B2 (1-based)
    assert grey is False


def test_densest_window_is_deterministic(tool):
    pts = gpd.GeoDataFrame(
        geometry=[box(x, 0, x + 1, 1) for x in (0, 1, 2, 50)] + [box(90, 90, 91, 91)],
        crs="EPSG:32613",
    )
    w1 = tool._densest_window(pts, 10, (0, 100, 0, 100))
    w2 = tool._densest_window(pts, 10, (0, 100, 0, 100))
    assert w1 == w2
    assert w1[0] <= 0.5 and w1[1] >= 2.5  # covers the three clustered fields


def test_main_reports_missing_outputs(tool, tmp_path, capsys):
    rc = tool.main(["--run-root", str(tmp_path), "--out-dir", str(tmp_path / "g"), "--only", "04"])
    assert rc == 1
    assert "[04] FAILED: FileNotFoundError" in capsys.readouterr().err


def test_a_full_run_writes_the_stats_afresh(tool, tmp_path):
    out = tmp_path / "g"
    out.mkdir()
    stats = out / "gallery_stats.json"
    stats.write_text(json.dumps({"zz": {"title": "from an older script"}}))
    args = ["--run-root", str(tmp_path / "empty"), "--out-dir", str(out)]
    assert tool.main([*args, "--only", "04"]) == 1  # --only keeps the other entries
    assert "zz" in json.loads(stats.read_text())
    assert tool.main(args) == 1  # every entry fails here, and nothing old survives
    assert json.loads(stats.read_text()) == {}


def test_the_read_window_covers_the_panel(tool, tmp_path):
    root = _run_root(tmp_path)
    tif = root / "outputs" / "demo" / "composite.tif"
    with rasterio.open(tif) as src:
        crs = src.crs
    bounds = (500003.7, 3998803.3, 500596.2, 3999996.1)  # not on the 10 m pixel grid
    rgb, ext, _ = tool._read_background([tif], "sentinel2", bounds, crs, (300, 600))
    # imshow extent (x0, x1, y0, y1): no strip of the panel is left without imagery
    assert ext[0] <= bounds[0] and ext[1] >= bounds[2]
    assert ext[2] <= bounds[1] and ext[3] >= bounds[3]
    assert ext[1] - ext[0] < bounds[2] - bounds[0] + 2 * 10  # less than a pixel over per side
    assert rgb.shape[2] == 3


@pytest.mark.parametrize(
    ("collections", "label"),
    [
        ("LANDSAT/LE07/C02/T1_L2,LANDSAT/LC08/C02/T1_L2", "Landsat 7/8 C2 L2"),
        (
            "LANDSAT/LC09/C02/T1_L2,LANDSAT/LE07/C02/T1_L2,LANDSAT/LC08/C02/T1_L2",
            "Landsat 7/8/9 C2 L2",
        ),
        ("LANDSAT/LC08/C02/T1_L2", "Landsat 8 C2 L2"),
        ("", "Landsat C2 L2"),
        (None, "Landsat C2 L2"),
        ("COPERNICUS/S2_SR_HARMONIZED", "Landsat C2 L2"),
    ],
)
def test_landsat_label_names_the_recorded_missions(tool, collections, label):
    assert tool._landsat_label(collections) == label


def test_background_note_takes_the_landsat_missions_from_the_composite(tool):
    comp = {
        "AGRIBOUND_COLLECTIONS": "LANDSAT/LE07/C02/T1_L2,LANDSAT/LC08/C02/T1_L2",
        "AGRIBOUND_COMPOSITE_METHOD": "median",
        "AGRIBOUND_DATE_START": "2018-01-01",
        "AGRIBOUND_DATE_END_EXCLUSIVE": "2019-01-01",
        "AGRIBOUND_N_IMAGES": "75",
        "AGRIBOUND_RESOLUTION_M": "30",
    }
    note = tool._bg_note(([], "landsat", {"facts": {"composite": comp}}, "the engine's input"))
    assert note.startswith("Landsat 7/8 C2 L2 30 m median composite, 2018-01-01 to 2019-01-01")
    assert "8/9" not in note


def _fake_boundaries(tool, monkeypatch):
    """Tiny stand-ins for Natural Earth and the Survey of India outline (no network)."""
    countries = gpd.GeoDataFrame(
        {
            "ADMIN": ["India", "China", "Mongolia"],
            "NAME": ["India", "China", "Mongolia"],
            "ADM0_A3": ["IND", "CHN", "MNG"],
        },
        # India and China overlap: a disputed area
        geometry=[box(68, 8, 97, 35), box(73, 18, 135, 53), box(88, 53, 120, 60)],
        crs=4326,
    )
    states = gpd.GeoDataFrame(
        {"name": ["West Bengal", "Hebei"], "adm0_a3": ["IND", "CHN"]},
        geometry=[box(86, 21.5, 89.9, 27.2), box(113, 36, 120, 42.6)],
        crs=4326,
    )
    lakes = gpd.GeoDataFrame(geometry=[box(100, 40, 104, 44)], crs=4326)
    soi = box(68, 6.7, 97.4, 37.1)  # extends further north than the stand-in "de facto" India
    layers = {"countries": countries, "states": states, "lakes": lakes}
    monkeypatch.setattr(tool, "_natural_earth", lambda layer: layers[layer])
    monkeypatch.setattr(tool, "_india_outline", lambda: soi)
    return soi


def test_india_is_drawn_from_the_survey_of_india_outline(tool, monkeypatch):
    soi = _fake_boundaries(tool, monkeypatch)
    loc = tool._locate(88.42, 23.39)
    assert loc["country"] == "India" and loc["admin1"] == "West Bengal"
    assert loc["_country_geom"].equals(soi)  # never the Natural Earth polygon
    assert tool._inset_shows_india(loc)


def test_insets_near_india_credit_the_survey_of_india_outline(tool, monkeypatch):
    _fake_boundaries(tool, monkeypatch)
    loc = tool._locate(115.5, 37.65)  # Hebei, China
    assert loc["country"] == "China" and loc["admin1"] == "Hebei"
    assert tool._inset_shows_india(loc)  # the China view includes India, drawn on top


def test_india_outline_is_pinned():
    import re

    tool_src = TOOL.read_text()
    m = re.search(r"INDIA_OUTLINE = \((.*?)\n\)", tool_src, re.S)
    assert m
    literal = re.sub(r'"\s*"', "", m.group(1))  # join implicitly concatenated strings
    assert re.search(r"india-geodata/[0-9a-f]{40}/", literal)  # a commit, not a branch
    assert re.search(r'"[0-9a-f]{64}"', literal)  # and a SHA-256


def test_polygons_removed_by_the_crop_filter_are_counted(tool, tmp_path):
    root = _run_root(tmp_path)
    out = root / "outputs" / "demo"
    fields = gpd.read_file(out / "fields.gpkg")
    fields.iloc[[0]].to_file(out / "fields_kept.gpkg", driver="GPKG")  # the filter kept one
    entry = tool.Entry(
        "98",
        "Removed_example",
        "Removed",
        [
            tool.Layer(
                "outputs/demo/fields.gpkg", "Filter off", removed_vs="outputs/demo/fields_kept.gpkg"
            )
        ],
        inset=False,
    )
    stats = tool.render(entry, root, tmp_path / "gallery", "composite")
    assert stats["layers"][0]["n_removed_by_crop_filter"] == 1


def test_inset_layers_are_stacked_by_explicit_zorder(tool, monkeypatch):
    import matplotlib.pyplot as plt

    _fake_boundaries(tool, monkeypatch)
    loc = tool._locate(115.5, 37.65)  # Hebei, China: the view includes India
    fig = plt.figure(figsize=(4, 3))
    try:
        iax, india_drawn = tool._draw_inset(fig, [0.1, 0.1, 0.8, 0.8], loc, "Hebei, China")
        assert india_drawn
        zorders = [a.get_zorder() for a in [*iax.collections, *iax.lines]]
    finally:
        plt.close(fig)
    # neighbours, the country, its province lines, the province, India, the lakes, the marker:
    # India above the province lines (no de facto line inside it), lakes above every boundary
    assert zorders == [
        tool.Z_NEIGHBOURS,
        tool.Z_COUNTRY,
        tool.Z_ADMIN1_LINES,
        tool.Z_ADMIN1,
        tool.Z_INDIA,
        tool.Z_LAKES,
        tool.Z_MARKER,
    ]
    assert zorders == sorted(zorders)


def _country(parts):
    from shapely.geometry import MultiPolygon

    return {"lon": 5.0, "lat": 5.0, "country_a3": "XXX", "_country_geom": MultiPolygon(parts)}


def test_inset_view_keeps_small_near_parts_of_the_country(tool):
    main = box(0, 0, 10, 10)  # the part with the study area; larger side 10
    island = box(11, 0, 12, 1)  # 1 % of the area, 1 away (< 15 % of 10): e.g. Tasmania
    far = box(40, 0, 41, 1)  # small but 30 away: e.g. Hawaii
    big = box(0, 11, 10, 14)  # 1 away but 30 % of the area: e.g. Alaska's size
    loc = _country([main, island, far, big])
    parts = tool._inset_parts(loc)
    assert [p.bounds for p in parts] == [main.bounds, island.bounds]
    country, view = tool._inset_view(loc)
    assert view.contains(island) and not view.intersects(far)
    assert country.contains(island)  # drawn white with the rest of the country
    assert not country.intersects(far)


def test_inset_view_is_the_part_nearest_the_study_area(tool):
    loc = _country([box(0, 0, 10, 10), box(30, 0, 34, 4)])
    loc["lon"], loc["lat"] = 32.0, 2.0  # on the smaller part
    parts = tool._inset_parts(loc)
    assert [p.bounds for p in parts] == [(30.0, 0.0, 34.0, 4.0)]


def test_india_insets_keep_the_whole_outline(tool, monkeypatch):
    from shapely.geometry import MultiPolygon

    soi = MultiPolygon([box(68, 8, 97, 35), box(92, 6, 94, 14)])  # with the Andaman Islands
    loc = {"lon": 88.4, "lat": 23.4, "country_a3": "IND", "_country_geom": soi}
    assert tool._inset_parts(loc) == [soi]
    country, view = tool._inset_view(loc)
    assert country.equals(soi) and view.contains(soi)


def test_notes_wrap_to_the_measured_width_with_balanced_lines(tool):
    text = (
        "Model: Delineate-Anything v2, large_v2 (YOLO11x-seg, "
        "MykolaL/DelineateAnything/DelineateAnythingv2.pt @ 369d0b4; ultralytics 8.4.163)"
    )
    width = 5.0
    lines = tool._wrap(text, width)
    greedy = tool._greedy_wrap(text.split(), width, tool.NOTE_FS)
    assert " ".join(lines) == text
    assert len(lines) == len(greedy) == 2
    assert all(tool._text_width_in(line, tool.NOTE_FS) <= width for line in lines)
    assert all(len(line.split()) > 1 for line in lines)  # no line with one word
    widths = [tool._text_width_in(line, tool.NOTE_FS) for line in lines]
    assert max(widths) <= max(tool._text_width_in(g, tool.NOTE_FS) for g in greedy)
    assert tool._wrap(text, width) == lines  # deterministic
