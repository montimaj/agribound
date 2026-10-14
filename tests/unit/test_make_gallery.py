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
import re
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


def test_an_explicit_window_crops_every_panel_and_marked_windows_are_recorded(
    tool, tmp_path, monkeypatch
):
    root = _run_root(tmp_path)
    marks = [("1", 500250.0, 3999250.0, 300.0), ("2", 500700.0, 3999700.0, 400.0)]
    overview = tool.Entry(
        "98",
        "Demo_overview",
        "Demo overview",
        [tool.Layer("outputs/demo/fields.gpkg", "Demo")],
        mark_windows=marks,
        inset=False,
    )
    stats = tool.render(overview, root, tmp_path / "gallery", "composite")
    assert [m["label"] for m in stats["marked_windows"]] == ["1", "2"]
    assert stats["marked_windows"][1] == {
        "label": "2",
        "centre": [500700.0, 3999700.0],
        "side_m": 400.0,
    }
    assert "window" not in stats

    captured = {}
    real = tool._read_background

    def spy(tifs, source, bounds, crs, out_px, stretch=None):
        captured["bounds"] = bounds
        return real(tifs, source, bounds, crs, out_px, stretch)

    monkeypatch.setattr(tool, "_read_background", spy)
    zoomed = tool.Entry(
        "97",
        "Demo_zoom",
        "Demo zoom",
        [tool.Layer("outputs/demo/fields.gpkg", "Demo")],
        window=(500250.0, 3999250.0, 300.0),
        inset=False,
    )
    stats = tool.render(zoomed, root, tmp_path / "gallery", "composite")
    assert captured["bounds"] == pytest.approx((500100.0, 3999100.0, 500400.0, 3999400.0))
    assert stats["window"] == {
        "crs": "EPSG:32613",
        "centre": [500250.0, 3999250.0],
        "side_m": 300.0,
    }


def test_a_window_outside_the_imagery_is_an_error(tool, tmp_path):
    root = _run_root(tmp_path)  # the composite covers x 500000-501000, y 3998800-4000000
    entry = tool.Entry(
        "96",
        "Demo_off",
        "Demo off",
        [tool.Layer("outputs/demo/fields.gpkg", "Demo")],
        window=(500950.0, 3999900.0, 400.0),
        inset=False,
    )
    with pytest.raises(ValueError, match="not inside the imagery"):
        tool.render(entry, root, tmp_path / "gallery", "composite")


def test_the_sam2_note_names_the_polygons_that_are_smoothed_again(tool):
    assert tool._sam2_note({}).endswith("are left unchanged")
    assert "smooths and simplifies all polygons again" in tool._sam2_note({}, resmoothed=True)
    note = tool._sam2_note({}, resmoothed="the polygons of 50 ha or less")
    assert "smooths and simplifies the polygons of 50 ha or less again" in note
    assert "all polygons" not in note


def test_text_flags_from_a_geopackage_are_read_as_booleans(tool, tmp_path):
    # A boolean flag missing from some of the concatenated frames becomes an object column
    # (True/False/NaN), which the GeoPackage driver writes as text ("True"/"False");
    # astype(bool) would make "False" True.
    import pandas as pd

    a = gpd.GeoDataFrame(
        {"flag": [True, False]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:32613"
    )
    b = gpd.GeoDataFrame({"x": [1]}, geometry=[box(2, 0, 3, 1)], crs="EPSG:32613")
    g = gpd.GeoDataFrame(pd.concat([a, b], ignore_index=True), geometry="geometry", crs=a.crs)
    path = tmp_path / "flags.gpkg"
    g.to_file(path, driver="GPKG")
    back = gpd.read_file(path)
    assert tool._bool_column(back["flag"]).tolist() == [True, False, False]
    assert tool._bool_column(["True", "False", None, "true", "0"]).tolist() == [
        True,
        False,
        False,
        True,
        False,
    ]
    assert tool._bool_column([True, False, float("nan"), None]).tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert tool._bool_column(pd.array([True, pd.NA], dtype="boolean")).tolist() == [True, False]


def _second_area(root: Path) -> None:
    """A second study area in another UTM zone (EPSG:32643), for multi-area entries."""
    out = root / "outputs" / "demo2"
    out.mkdir(parents=True)
    rng = np.random.default_rng(1)
    transform = rasterio.transform.from_origin(300000, 2000000, 10, 10)
    with rasterio.open(
        out / "composite.tif",
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=len(S2_BANDS),
        dtype="float32",
        crs="EPSG:32643",
        transform=transform,
        nodata=float("nan"),
    ) as dst:
        dst.write(rng.uniform(200, 3000, size=(len(S2_BANDS), 100, 100)).astype("float32"))
        dst.descriptions = tuple(S2_BANDS)
    gpd.GeoDataFrame(
        geometry=[box(300100 + 150 * i, 1999100, 300200 + 150 * i, 1999200) for i in range(5)],
        crs="EPSG:32643",
    ).to_file(out / "fields.gpkg", driver="GPKG")
    sidecar = {
        "status": "success",
        "config": {"source": "sentinel2", "engine": "delineate-anything", "year": 2022},
        "facts": {"raster_path": "outputs/demo2/composite.tif"},
    }
    (out / "fields.gpkg.provenance.json").write_text(json.dumps(sidecar))


def test_multi_area_entries_draw_each_area_in_its_own_crs_and_window(tool, tmp_path):
    root = _run_root(tmp_path)
    _second_area(root)
    entry = tool.Entry(
        "95",
        "Demo_areas",
        "Demo areas",
        [
            tool.Layer("outputs/demo/fields.gpkg", "Area A"),
            tool.Layer("outputs/demo2/fields.gpkg", "Area B", crop_m=600),
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
        ncols=2,
        inset=False,
    )
    stats = tool.render_areas(entry, root, tmp_path / "gallery", "composite")
    assert (tmp_path / "gallery" / "Demo_areas.png").exists()
    a, b = stats["layers"]
    assert (a["panel"], a["n_polygons"], a["window"]["crs"]) == ("1", 2, "EPSG:32613")
    assert (b["panel"], b["n_polygons"], b["window"]["crs"]) == ("2", 5, "EPSG:32643")
    x0, y0, x1, y1 = b["window"]["bounds"]
    assert (x1 - x0, y1 - y0) == (600.0, 600.0)  # the layer's own window side
    assert x0 >= 300000 and x1 <= 301000 and y0 >= 1999000 and y1 <= 2000000
    # The composites differ (one records its window): one item per area, the role said once.
    note = stats["background_note"].replace("\n", " ")
    assert note.startswith("Background (the engine's input; end dates exclusive): 1: ")
    assert "; 2: Sentinel-2 L2A" in note and "(end exclusive)" not in note


def test_multi_area_panels_draw_their_own_reference_and_crop_filter_removals(tool, tmp_path):
    root = _run_root(tmp_path)
    _second_area(root)
    demo = root / "outputs" / "demo"
    gpd.read_file(demo / "fields.gpkg").iloc[[0]].to_file(demo / "kept.gpkg", driver="GPKG")
    # One reference polygon in the empty south-east corner of area A, out of reach of any
    # 400 m square that holds a predicted field: only crop_on_reference puts the window there.
    gpd.GeoDataFrame(geometry=[box(500750, 3998850, 500950, 3999050)], crs="EPSG:32613").to_file(
        demo / "reference_se.gpkg", driver="GPKG"
    )
    entry = tool.Entry(
        "93",
        "Demo_refs",
        "Demo references",
        [
            tool.Layer(
                "outputs/demo/fields.gpkg",
                "Area A",
                reference="outputs/demo/reference_se.gpkg",
                removed_vs="outputs/demo/kept.gpkg",
            ),
            tool.Layer("outputs/demo2/fields.gpkg", "Area B"),
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
        crop_on_reference=True,
        reference_label="Demo reference",
        ncols=2,
        inset=False,
    )
    stats = tool.render_areas(entry, root, tmp_path / "gallery", "composite")
    a, b = stats["layers"]
    assert a["reference"]["label"] == "Demo reference" and a["reference"]["n_polygons"] == 1
    assert a["n_removed_by_crop_filter"] == 1
    assert b["reference"] is None and b["n_removed_by_crop_filter"] is None
    assert a["reference"]["n_polygons_in_window"] == 1
    # crop_on_reference: the window of panel A is the square with the most reference
    # polygons, around (500850, 3998950), not one with the most predicted fields.
    x0, y0, x1, y1 = a["window"]["bounds"]
    assert x0 <= 500850 <= x1 and y0 <= 3998950 <= y1
    for x, y in ((500250, 3999250), (500700, 3999700)):
        assert not (x0 <= x <= x1 and y0 <= y <= y1)
    # How each window was chosen: 400 m squares half a side apart in the imagery (1,000 m by
    # 1,200 m in area A: 4 by 5 candidates; 1,000 m square in area B: 4 by 4).
    assert {k: a["window"][k] for k in ("side_m", "candidates", "most_of")} == {
        "side_m": 400.0,
        "candidates": 20,
        "most_of": "reference polygons",
    }
    assert (b["window"]["candidates"], b["window"]["most_of"]) == (16, "panel polygons")
    assert "inside" not in a["window"]


def test_a_panel_can_take_its_background_from_another_output(tool, tmp_path):
    root = _run_root(tmp_path)
    _second_area(root)
    entry = tool.Entry(
        "91",
        "Demo_background_from",
        "Demo",
        [
            tool.Layer(
                "outputs/demo/fields.gpkg",
                "A",
                background_from="outputs/demo2/fields.gpkg",
                bg_role="the other output's composite",
            )
        ],
        per_layer_background=True,
        inset=False,
    )
    tifs, source, prov, role = tool._background_for(entry, root, {}, "composite", entry.layers[0])
    assert [t.name for t in tifs] == ["composite.tif"] and "demo2" in str(tifs[0])
    assert source == "sentinel2" and role == "the other output's composite"
    assert prov["facts"]["raster_path"] == "outputs/demo2/composite.tif"


def test_counts_read_in_the_singular_for_one(tool):
    assert tool._count(1, "field") == "1 field"
    assert tool._count(0, "field") == "0 fields"
    assert tool._count(1185, "field") == "1,185 fields"
    assert tool._count("7", "image") == "7 images"


def test_long_panel_titles_are_set_smaller_to_fit(tool):
    short, long = "Area A (2 fields)", "Delineate-Anything v2, SPOT-Pan 1.5 m (1,185 fields)" * 2
    assert tool._fit_title(short, 3.0) == tool.TITLE_FS
    assert 7 <= tool._fit_title(long, 2.0) < tool.TITLE_FS


def test_layer_references_are_for_multi_area_entries(tool, tmp_path):
    root = _run_root(tmp_path)
    entry = tool.Entry(
        "92",
        "Demo_layer_ref",
        "Demo",
        [tool.Layer("outputs/demo/fields.gpkg", "A", reference="outputs/demo/reference.gpkg")],
        inset=False,
    )
    with pytest.raises(ValueError, match="Layer.reference is for multi-area entries"):
        tool.render(entry, root, tmp_path / "gallery", "composite")


NV_ZIP = "examples/NV_Field_Boundaries.zip"
NV_LAYER = "nevada_agricultural_field_boundaries_20260820.gpkg"
NV_OUT = "outputs/nevada_field_boundaries"  # example 24's outputs
NV_SUBSET = f"{NV_OUT}/references/diamond_valley_reference_0123456789ab.gpkg"


@pytest.mark.parametrize(
    "path",
    [
        NV_ZIP,
        f"zip://{NV_ZIP}!{NV_LAYER}",
        NV_SUBSET,
        "outputs/multistate_openet/nv/reference.gpkg",
        "outputs/multistate_openet/ca/reference.gpkg",
        "RESERVOIR - DRI_OpenET_v211_2026_FieldBoundaries/CA.zip",
        "examples/NM_Field_Boundaries.zip",
    ],
)
def test_the_field_boundaries_of_examples_24_and_25_may_be_drawn_and_placed(tool, path):
    """The guard's lists are empty: the DRI Nevada, V18 and OpenET outlines may be shown."""
    assert tool.RESTRICTED_REFERENCES == () and tool.LOCAL_SQUARES == ()
    layer = f"{NV_OUT}/diamond_valley/fields_naip_da_2022.gpkg"
    single = tool.Entry(
        "24", "Demo_nv", "Demo", [tool.Layer(layer, "A")], reference=path, window=(0.0, 0.0, 1.0)
    )
    marked = tool.Entry(
        "24", "Demo_nv_marks", "Demo", [tool.Layer(layer, "A")], mark_windows=[("1", 0, 0, 1)]
    )
    areas = tool.Entry(
        "25",
        "Demo_openet_areas",
        "Demo",
        [tool.Layer("outputs/multistate_openet/nv/fields_naip_da_2022.gpkg", "NV", reference=path)],
        multi_area=True,
        crop_m=2000,
    )
    for entry in (single, marked, areas):
        tool._check_publishable(entry)
        assert not tool._hides_squares(entry)


def test_no_gallery_entry_draws_a_restricted_reference(tool):
    for entry in tool.ENTRIES:
        tool._check_publishable(entry, tool._entry_root(entry))


# The guard itself (module docstring, "Restricted references"), with stand-in lists: names of
# reference data whose outlines must not be drawn, and outputs whose squares must not be placed.
R_ZIP = "examples/Demo_Restricted_Fields.zip"
R_LAYER = "demo_restricted_fields_20260101.gpkg"
R_OUT = "outputs/demo_restricted"
R_SUBSET = f"{R_OUT}/references/a_reference_0123456789ab.gpkg"
L_OUT = "outputs/demo_local"


@pytest.fixture
def guarded(tool, monkeypatch):
    """The tool with stand-in RESTRICTED_REFERENCES and LOCAL_SQUARES."""
    monkeypatch.setattr(
        tool, "RESTRICTED_REFERENCES", ("Demo_Restricted_Fields", "demo_restricted/references")
    )
    monkeypatch.setattr(tool, "LOCAL_SQUARES", ("demo_local/",))
    return tool


@pytest.mark.parametrize(
    ("entry_kw", "layer_kw"),
    [
        # The data: the zip, its layer, extracted copies and an example's subsets.
        ({"reference": R_ZIP}, {}),
        ({"reference": f"zip://{R_ZIP}!{R_LAYER}"}, {}),
        ({"reference": f"examples/Demo_Restricted_Fields/{R_LAYER}"}, {}),
        ({"reference": f"data/{R_LAYER}"}, {}),
        ({"reference": R_SUBSET}, {}),
        ({"reference": "examples/demo_restricted_fields.zip"}, {}),  # any case
        ({"reference": R_SUBSET.replace("/", "\\")}, {}),  # Windows separators
        # Every other file an entry names.
        ({"overlay": f"{R_ZIP}::fields"}, {}),
        ({"overlay": R_SUBSET}, {}),
        ({"background": "examples/Demo_Restricted_Fields/*.tif"}, {}),
        ({"background_from": R_SUBSET}, {}),
        ({"metrics": f"{R_OUT}/references/*.json"}, {}),
        # Every file a layer names: a subset drawn as a panel (also a crop-filtered copy),
        # removed_vs, a panel's reference (multi-area), model_from, backgrounds.
        ({}, {"path": R_SUBSET}),
        ({}, {"path": R_SUBSET.replace(".gpkg", "_2024_crop_t0.3.gpkg")}),
        ({}, {"path": f"examples/{R_LAYER}"}),
        ({}, {"removed_vs": R_SUBSET}),
        ({}, {"removed_vs": R_ZIP}),
        ({"multi_area": True, "crop_m": 400}, {"reference": R_SUBSET}),
        ({"multi_area": True, "crop_m": 400}, {"reference": R_ZIP}),
        ({}, {"model_from": R_SUBSET}),
        ({}, {"background": "examples/Demo_Restricted_Fields/*.tif"}),
        ({}, {"background_from": R_SUBSET}),
        # A panel's metrics file and the polygon its window is chosen in.
        ({}, {"metrics": f"{R_OUT}/references/metrics_a_reference.json"}),
        ({"multi_area": True, "crop_m": 400}, {"window_within": R_SUBSET}),
    ],
)
def test_entries_never_draw_a_restricted_reference(guarded, tmp_path, entry_kw, layer_kw):
    root = _run_root(tmp_path)
    layer = {"path": "outputs/demo/fields.gpkg", "label": "A", **layer_kw}
    entry = guarded.Entry(
        "95", "Demo_restricted", "Demo", [guarded.Layer(**layer)], inset=False, **entry_kw
    )
    renderer = guarded.render_areas if entry.multi_area else guarded.render
    with pytest.raises(ValueError, match="restricted reference"):
        renderer(entry, root, tmp_path / "gallery", "composite")
    assert not (tmp_path / "gallery").exists()


@pytest.mark.parametrize(
    "path",
    [
        f"{R_OUT}/*/a_reference.gpkg",  # a glob, expanded as the renderer expands it
        "outputs/*/references/a_reference.gpkg",
        "outputs/demo_rest*/references/a_reference.gpkg",
        f"{R_OUT}/x/../references/a_reference.gpkg",  # '..'
        f"{R_OUT}//references/a_reference.gpkg",  # '//'
        f"{R_OUT}/./references/a_reference.gpkg",  # './'
        "outputs/innocent.gpkg",  # a symlink to the subset
        "outputs/linked/a_reference.gpkg",  # in a symlinked folder
    ],
)
def test_the_guard_sees_through_globs_dots_and_symlinks(guarded, tmp_path, path):
    """A path is checked as the files it names under the run root, not only as written."""
    import shutil

    root = _run_root(tmp_path)
    subset = root / R_OUT / "references"
    subset.mkdir(parents=True)
    shutil.copy(root / "outputs" / "demo" / "reference.gpkg", subset / "a_reference.gpkg")
    (root / "outputs" / "innocent.gpkg").symlink_to(subset / "a_reference.gpkg")
    (root / "outputs" / "linked").symlink_to(subset, target_is_directory=True)
    entry = guarded.Entry(
        "95",
        "Demo_bypass",
        "Demo",
        [guarded.Layer("outputs/demo/fields.gpkg", "A")],
        reference=path,
        inset=False,
    )
    with pytest.raises(ValueError, match="restricted reference"):
        guarded.render(entry, root, tmp_path / "gallery", "composite")
    assert not (tmp_path / "gallery").exists()
    # The same file under a name that the lists do not hold is drawn.
    import dataclasses

    plain = dataclasses.replace(entry, reference="outputs/demo/reference.gpkg")
    guarded._check_publishable(plain, root)


def _copy_area(root: Path, src: str, dst: str) -> None:
    """Copy the synthetic outputs/<src> (composite, fields, sidecar) to outputs/<dst>."""
    import shutil

    a, b = root / "outputs" / src, root / "outputs" / dst
    b.mkdir(parents=True)
    for name in ("composite.tif", "fields.gpkg"):
        shutil.copy(a / name, b / name)
    sidecar = json.loads((a / "fields.gpkg.provenance.json").read_text())
    sidecar["facts"]["raster_path"] = f"outputs/{dst}/composite.tif"
    (b / "fields.gpkg.provenance.json").write_text(json.dumps(sidecar))


def _fake_us_boundaries(tool, monkeypatch):
    """Stand-ins for Natural Earth around the synthetic area (New Mexico; no network)."""
    countries = gpd.GeoDataFrame(
        {
            "ADMIN": ["United States of America", "Mexico"],
            "NAME": ["United States of America", "Mexico"],
            "ADM0_A3": ["USA", "MEX"],
        },
        geometry=[box(-125, 31.3, -66, 49), box(-117, 15, -87, 31.3)],
        crs=4326,
    )
    states = gpd.GeoDataFrame(
        {"name": ["New Mexico"], "adm0_a3": ["USA"]},
        geometry=[box(-109.05, 31.33, -103.0, 37.0)],
        crs=4326,
    )
    lakes = gpd.GeoDataFrame(geometry=[box(-113, 40.7, -112, 41.5)], crs=4326)
    layers = {"countries": countries, "states": states, "lakes": lakes}
    monkeypatch.setattr(tool, "_natural_earth", lambda layer: layers[layer])
    monkeypatch.setattr(tool, "_india_outline", lambda: box(68, 6.7, 97.4, 37.1))


def test_local_square_stats_keep_the_counts_and_leave_out_the_coordinates(
    guarded, tmp_path, monkeypatch
):
    """A LOCAL_SQUARES entry is drawn without a locator, and its stats hold no lon/lat."""
    _fake_us_boundaries(guarded, monkeypatch)
    root = _run_root(tmp_path)
    _copy_area(root, "demo", "demo_local/a")
    stats = {}
    for key, folder, inset in (("99", "demo", True), ("98", "demo_local/a", False)):
        entry = guarded.Entry(
            key,
            f"Demo_{key}",
            "Demo",
            [guarded.Layer(f"outputs/{folder}/fields.gpkg", "A")],
            inset=inset,
        )
        stats[key] = guarded.render(entry, root, tmp_path / "gallery", "composite")
    shown, hidden = stats["99"], stats["98"]
    assert {"lon", "lat", "admin1"} <= set(shown["location"])
    assert not {"lon", "lat"} & set(hidden["location"])
    assert hidden["layers"][0]["n_polygons"] == shown["layers"][0]["n_polygons"] == 2
    assert "window" not in hidden
    # The record leaves out the coordinates and keeps the rest.
    loc = {"lon": 1.0, "lat": 2.0, "country": "X", "admin1": "Y", "_dot": 3}
    assert guarded._location_record(loc, True) == {"country": "X", "admin1": "Y"}


def test_local_square_panels_leave_out_their_windows(guarded, tmp_path, monkeypatch):
    _fake_boundaries(guarded, monkeypatch)
    root = _run_root(tmp_path)
    _second_area(root)
    _copy_area(root, "demo", "demo_local/a")
    _copy_area(root, "demo2", "demo_local/b")
    entry = guarded.Entry(
        "98",
        "Demo_local_areas",
        "Demo",
        [
            guarded.Layer(f"{L_OUT}/a/fields.gpkg", "A"),
            guarded.Layer(f"{L_OUT}/b/fields.gpkg", "B"),
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
        ncols=2,
        inset=False,
    )
    stats = guarded.render_areas(entry, root, tmp_path / "gallery", "composite")
    assert [lay["n_polygons"] for lay in stats["layers"]] == [2, 5]
    assert all("window" not in lay for lay in stats["layers"])
    text = json.dumps(stats)
    assert "centre_lonlat" not in text and "bounds" not in text
    assert '"lon"' not in text and '"lat"' not in text


@pytest.mark.parametrize(
    ("path", "entry_kw"),
    [
        (f"{L_OUT}/a/fields.gpkg", {"window": (500250.0, 3999250.0, 300.0), "inset": False}),
        (
            f"{L_OUT}/a/fields.gpkg",
            {"mark_windows": [("1", 500250.0, 3999250.0, 300.0)], "inset": False},
        ),
        (f"{L_OUT}/a/fields.gpkg", {}),  # the locator inset is on by default
        ("outputs/demo_lo*/a/fields.gpkg", {}),  # a glob that names the local outputs
        ("outputs/demo/../demo_local/a/fields.gpkg", {}),
    ],
)
def test_local_squares_are_never_placed(guarded, tmp_path, path, entry_kw):
    """An explicit window, window marks or the locator inset would place a local square."""
    root = _run_root(tmp_path)
    _copy_area(root, "demo", "demo_local/a")
    entry = guarded.Entry("98", "Demo_local_window", "Demo", [guarded.Layer(path, "A")], **entry_kw)
    with pytest.raises(ValueError, match="stays local"):
        guarded.render(entry, root, tmp_path / "gallery", "composite")
    assert not (tmp_path / "gallery").exists()


def test_the_defaults_render_the_1_1_runs_into_their_own_gallery(tool):
    """The re-run that follows 1.0.1 goes to assets/gallery_1.1; the 1.0 gallery is kept."""
    args = tool._parser().parse_args([])
    assert args.run_root is None  # each entry's own run root
    assert Path(args.out_dir) == tool.REPO / "assets" / "gallery_1.1"
    plain = next(e for e in tool.ENTRIES if e.key == "04")
    assert tool._entry_root(plain) == tool.RUN_ROOT == tool.REPO / "outputs" / "examples_1.1"
    old = ["--run-root", "outputs/examples_1.0", "--out-dir", "assets/gallery_1.0"]
    args = tool._parser().parse_args(old)
    assert (args.run_root, args.out_dir) == ("outputs/examples_1.0", "assets/gallery_1.0")
    # --run-root replaces the run root of every entry, an entry's own included.
    for entry in tool.ENTRIES:
        assert tool._entry_root(entry, args.run_root) == Path("outputs/examples_1.0")


def test_every_entry_reads_the_run_root_that_holds_its_current_outputs(tool):
    """The final run holds the re-runs of 03, 14 and 23, the zones of 25 and example 27; the
    radar run example 26; example 24's run with --options and --sam-auto its sam-auto and
    "-opt" runs, and the five Nevada figures read it (its other runs are those of the
    re-scoring run, unchanged); the New York run example 28; the SAM evaluation's run root the
    SAM entry (whose comparators name their own run roots); the 1.1 run every other example
    (module docstring, "Run roots")."""
    final = {"03", "14", "23", "23b", "23c", "23d", "23e", "25b", "27"}
    sar = {"26", "26b"}
    nevada = {"24", "24b", "24c", "24d", "24e"}
    new_york = {"28", "28b", "28c"}
    sam = {"sam"}
    for entry in tool.ENTRIES:
        root = tool._entry_root(entry).relative_to(tool.REPO).as_posix()
        if entry.key in final:
            expected = tool.FINAL_RUN_ROOT
        elif entry.key in sar:
            expected = tool.SAR_RUN_ROOT
        elif entry.key in nevada:
            expected = tool.EX24_RUN_ROOT
        elif entry.key in new_york:
            expected = tool.NY_RUN_ROOT
        elif entry.key in sam:
            expected = tool.SAM_EVAL_RUN_ROOT
        else:
            expected = "outputs/examples_1.1"
        assert root == expected, entry.key
        if entry.key not in sam:  # only the SAM entry has layers with their own run root
            assert not any(lay.run_root for lay in entry.layers), entry.key
    assert tool.FINAL_RUN_ROOT == "outputs/examples_1.1_final"
    assert tool.SAR_RUN_ROOT == "outputs/examples_1.1_sar"
    assert tool.EX24_RUN_ROOT == "outputs/examples_1.1_ex24b"
    assert tool.NY_RUN_ROOT == "outputs/examples_1.1_r3/ny"
    assert tool.SAM_EVAL_RUN_ROOT == "outputs/examples_1.1_r3/sam_eval"
    assert final | sar | nevada | new_york | sam <= {e.key for e in tool.ENTRIES}


@pytest.mark.parametrize(("min_area", "n_drawn"), [(0.0, 2), (2500.0, 1)])
def test_reference_min_area_drops_the_small_reference_polygons(tool, tmp_path, min_area, n_drawn):
    """As the examples' evaluations: reference polygons under the minimum area are not drawn."""
    root = _run_root(tmp_path)
    ref = gpd.GeoDataFrame(
        geometry=[box(500090, 3999090, 500410, 3999410), box(500600, 3999600, 500640, 3999640)],
        crs="EPSG:32613",
    )  # 10.24 ha and 1,600 m2
    ref.to_file(root / "outputs" / "demo" / "reference.gpkg", driver="GPKG")
    entry = tool.Entry(
        "94",
        "Demo_min_area",
        "Demo",
        [tool.Layer("outputs/demo/fields.gpkg", "Demo")],
        reference="outputs/demo/reference.gpkg",
        reference_min_area_m2=min_area,
        inset=False,
    )
    stats = tool.render(entry, root, tmp_path / "gallery", "composite")
    assert stats["reference"]["n_polygons"] == n_drawn
    assert stats["reference"].get("min_area_m2") == (min_area or None)


def test_v18_entries_draw_the_evaluated_reference(tool):
    """Entries that draw the whole V18 layer drop its polygons under 2,500 m2, as example 20."""
    v18 = [e for e in tool.ENTRIES if e.reference == "examples/NM_Field_Boundaries.zip"]
    assert {e.key for e in v18} == {"20", "20b", "20c", "13"}
    assert all(e.reference_min_area_m2 == 2500 for e in v18)


def test_multi_area_panels_of_one_area_share_a_number(tool, tmp_path):
    root = _run_root(tmp_path)
    _second_area(root)
    entry = tool.Entry(
        "91",
        "Demo_area_groups",
        "Demo area groups",
        [
            tool.Layer("outputs/demo/fields.gpkg", "Area A, model 1", area="A"),
            tool.Layer("outputs/demo/fields.gpkg", "Area A, model 2", area="A"),
            tool.Layer("outputs/demo2/fields.gpkg", "Area B", area="B"),
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
        ncols=3,
        inset=False,
    )
    stats = tool.render_areas(entry, root, tmp_path / "gallery", "composite")
    assert [lay["panel"] for lay in stats["layers"]] == ["1", "1", "2"]
    # One background item per area, not per panel.
    assert re.findall(r"(?<![\d-])(\d):\s", stats["background_note"]) == ["1", "2"]


def test_areas_get_one_locator_dot_each(tool):
    panels = [
        {"n": "1", "lon": 10.0, "lat": 5.0},
        {"n": "1", "lon": 10.5, "lat": 5.5},
        {"n": "2", "lon": -3.0, "lat": 38.0},
    ]
    assert tool._area_points(panels) == [("1", 10.0, 5.0), ("2", -3.0, 38.0)]


def test_layer_areas_are_for_multi_area_entries(tool, tmp_path):
    root = _run_root(tmp_path)
    entry = tool.Entry(
        "90",
        "Demo_layer_area",
        "Demo",
        [tool.Layer("outputs/demo/fields.gpkg", "A", area="A")],
        inset=False,
    )
    with pytest.raises(ValueError, match="Layer.area is for multi-area entries"):
        tool.render(entry, root, tmp_path / "gallery", "composite")


def test_panels_of_one_area_share_the_window_of_its_first_panel(tool, tmp_path):
    root = _run_root(tmp_path)
    demo = root / "outputs" / "demo"
    # A second "model" on area A whose polygons sit in the opposite corner.
    gpd.GeoDataFrame(geometry=[box(500700, 3998900, 500950, 3999100)], crs="EPSG:32613").to_file(
        demo / "fields_other.gpkg", driver="GPKG"
    )
    (demo / "fields_other.gpkg.provenance.json").write_text(
        (demo / "fields.gpkg.provenance.json").read_text()
    )
    entry = tool.Entry(
        "89",
        "Demo_area_window",
        "Demo",
        [
            tool.Layer("outputs/demo/fields.gpkg", "A, model 1", area="A"),
            tool.Layer("outputs/demo/fields_other.gpkg", "A, model 2", area="A"),
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
        ncols=2,
        inset=False,
    )
    stats = tool.render_areas(entry, root, tmp_path / "gallery", "composite")
    first, second = (lay["window"]["bounds"] for lay in stats["layers"])
    assert first == second


def test_area_names_do_not_collide_with_panels_without_an_area(tool, tmp_path):
    root = _run_root(tmp_path)
    _second_area(root)
    entry = tool.Entry(
        "88",
        "Demo_area_keys",
        "Demo",
        [
            tool.Layer("outputs/demo/fields.gpkg", "Area A", area="#1"),
            tool.Layer("outputs/demo2/fields.gpkg", "Area B"),
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
        ncols=2,
        inset=False,
    )
    stats = tool.render_areas(entry, root, tmp_path / "gallery", "composite")
    assert [lay["panel"] for lay in stats["layers"]] == ["1", "2"]


@pytest.mark.parametrize(
    ("layer_kw", "entry_kw", "name"),
    [
        ({"crop_m": 200}, {}, "Layer.crop_m"),
        ({"window_within": "outputs/demo/reference.gpkg"}, {}, "Layer.window_within"),
        ({}, {"inset_pad_deg": (5.0, 5.0)}, "Entry.inset_pad_deg"),
    ],
)
def test_single_area_entries_refuse_multi_area_fields(tool, tmp_path, layer_kw, entry_kw, name):
    root = _run_root(tmp_path)
    entry = tool.Entry(
        "87",
        "Demo_multi_only",
        "Demo",
        [tool.Layer("outputs/demo/fields.gpkg", "A", **layer_kw)],
        inset=False,
        **entry_kw,
    )
    with pytest.raises(ValueError, match=f"{name} is for multi-area entries"):
        tool.render(entry, root, tmp_path / "gallery", "composite")


def test_input_notes_need_one_common_background(tool, tmp_path):
    """Each panel on its own input leaves nothing for input_notes to name: refused, not
    silently ignored."""
    root = _run_root(tmp_path)
    entry = tool.Entry(
        "87",
        "Demo_input_notes",
        "Demo",
        [tool.Layer("outputs/demo/fields.gpkg", "A")],
        per_layer_background=True,
        input_notes=True,
        inset=False,
    )
    with pytest.raises(ValueError, match="input_notes"):
        tool.render(entry, root, tmp_path / "gallery", "composite")
    assert not (tmp_path / "gallery").exists()


def test_one_locator_dot_and_location_record_per_area(tool, tmp_path, monkeypatch):
    _fake_boundaries(tool, monkeypatch)
    root = _run_root(tmp_path)
    _second_area(root)
    drawn = []
    draw = tool._draw_world_inset

    def spy(fig, rect, points, view):
        drawn.append(list(points))
        return draw(fig, rect, points, view)

    monkeypatch.setattr(tool, "_draw_world_inset", spy)
    entry = tool.Entry(
        "86",
        "Demo_area_inset",
        "Demo",
        [
            tool.Layer("outputs/demo/fields.gpkg", "A, model 1", area="A"),
            tool.Layer("outputs/demo/fields.gpkg", "A, model 2", area="A"),
            tool.Layer("outputs/demo2/fields.gpkg", "B", area="B"),
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
        ncols=3,
    )
    stats = tool.render_areas(entry, root, tmp_path / "gallery", "composite")
    assert [n for n, _lon, _lat in drawn[0]] == ["1", "2"]
    assert "Inset: study areas 1-2;" in stats["background_note"].replace("\n", " ")
    assert [a["panel"] for a in stats["location"]["areas"]] == ["1", "2"]


def test_maps_carry_no_axis_labels(tool, tmp_path, monkeypatch):
    # geopandas >= 1.2 labels the axes of every .plot() with the CRS axis names: text under
    # every panel, and a world locator pushed out of the footer. Stand in for it here.
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    add, labelled = Axes.add_collection, []

    def add_and_label(self, collection, *args, **kwargs):
        labelled.append(self)
        self.set_xlabel(self.get_xlabel() or "Easting [metre]")
        self.set_ylabel(self.get_ylabel() or "Northing [metre]")
        return add(self, collection, *args, **kwargs)

    monkeypatch.setattr(Axes, "add_collection", add_and_label)
    _fake_boundaries(tool, monkeypatch)
    root = _run_root(tmp_path)
    _second_area(root)
    labels = []
    check = tool._check_footer

    def spy(fig, *args):
        labels.extend((ax.get_xlabel(), ax.get_ylabel()) for ax in fig.axes if ax.get_visible())
        return check(fig, *args)

    monkeypatch.setattr(tool, "_check_footer", spy)
    one = tool.Entry(
        "99",
        "Demo_example",
        "Demo",
        [tool.Layer("outputs/demo/fields.gpkg", "Demo")],
        reference="outputs/demo/reference.gpkg",
        inset=False,
    )
    tool.render(one, root, tmp_path / "gallery", "composite")
    two = tool.Entry(
        "86",
        "Demo_area_inset",
        "Demo",
        [
            tool.Layer("outputs/demo/fields.gpkg", "A", area="A"),
            tool.Layer("outputs/demo2/fields.gpkg", "B", area="B"),
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
    )
    tool.render_areas(two, root, tmp_path / "gallery", "composite")  # 2 panels, world locator
    fig = plt.figure(figsize=(4, 3))
    try:
        loc = tool._locate(115.5, 37.65)
        iax, _india = tool._draw_inset(fig, [0.1, 0.1, 0.8, 0.8], loc, "Hebei, China")
        labels.append((iax.get_xlabel(), iax.get_ylabel()))
    finally:
        plt.close(fig)
    assert labelled  # the stand-in ran
    assert labels == [("", "")] * 5


def test_the_background_note_lists_each_area_once_when_all_share_a_source(tool):
    comp = {
        "AGRIBOUND_RESOLUTION_M": "1.5",
        "AGRIBOUND_COMPOSITE_METHOD": "median",
        "AGRIBOUND_N_IMAGES": "2",
        "AGRIBOUND_DATE_START": "2021-01-01",
        "AGRIBOUND_DATE_END_EXCLUSIVE": "2022-01-01",
    }
    panels = [
        {"n": n, "bg": ([], "spot-pan", {"facts": {"composite": comp}}, "the engine's input")}
        for n in ("1", "1", "2")
    ]
    note = tool._areas_bg_note(panels)
    assert note.startswith("Background: SPOT 6/7 panchromatic 1.5 m median composites")
    assert note.endswith(
        "1: 2021-01-01 to 2022-01-01, 2 images; 2: 2021-01-01 to 2022-01-01, 2 images"
    )


def test_dinov3_runs_that_load_a_checkpoint_name_it(tool):
    em = {
        "model_name": "dinov3_vitl16",
        "weights": "giswqs/geoai/dinov3_vitl16_sat493m.pth",
        "weights_revision": "aa2b25d0",
        "geoai-py_version": "0.43.1",
        "checkpoint_sha256": "9600e99c1234",
        "use_lora": False,
    }
    loaded = {
        "config": {"engine": "dinov3", "engine_params": {"checkpoint_path": "/x/best.ckpt"}},
        "engine_meta": em,
    }
    assert tool._model_note(loaded) == (
        "DINOv3 dinov3_vitl16 fine-tuned checkpoint (sha256 9600e99; from "
        "giswqs/geoai/dinov3_vitl16_sat493m.pth @ aa2b25d; geoai-py 0.43.1)"
    )
    # A run that fine-tunes and predicts in one go keeps its earlier note.
    trained = {
        "config": {
            "engine": "dinov3",
            "fine_tune": True,
            "fine_tune_epochs": 30,
            "fine_tune_split": "block",
        },
        "engine_meta": em,
    }
    assert tool._model_note(trained) == (
        "DINOv3 dinov3_vitl16 with giswqs/geoai/dinov3_vitl16_sat493m.pth @ aa2b25d "
        "(geoai-py 0.43.1), fine-tuned on the reference polygons (full, up to 30 epochs, "
        "block split)"
    )


def test_the_background_note_of_areas_with_one_source_is_one_line(tool):
    def panel(n, start, end, k):
        comp = {
            "AGRIBOUND_RESOLUTION_M": "1.5",
            "AGRIBOUND_COMPOSITE_METHOD": "median",
            "AGRIBOUND_N_IMAGES": str(k),
            "AGRIBOUND_DATE_START": start,
            "AGRIBOUND_DATE_END_EXCLUSIVE": end,
        }
        prov = {"facts": {"composite": comp}}  # as in real sidecars; no raster needed
        return {"n": n, "bg": ([], "spot-pan", prov, "the engine's input")}

    note = tool._areas_bg_note(
        [panel("1", "2018-01-01", "2019-01-01", 7), panel("2", "2021-01-01", "2022-01-01", 2)]
    )
    assert note.startswith("Background: SPOT 6/7 panchromatic 1.5 m median composites")
    assert "1: 2018-01-01 to 2019-01-01, 7 images; 2: 2021-01-01 to 2022-01-01, 2 images" in note


def test_multi_area_entries_refuse_options_only_render_implements(tool, tmp_path):
    root = _run_root(tmp_path)
    entry = tool.Entry(
        "94",
        "Demo_bad",
        "Demo bad",
        [tool.Layer("outputs/demo/fields.gpkg", "Area A", highlight="agribound:sam_refined")],
        multi_area=True,
        crop_m=400,
        sam2_note=True,
        inset=False,
    )
    with pytest.raises(ValueError, match="do not support sam2_note, layers.0..highlight"):
        tool.render_areas(entry, root, tmp_path / "gallery", "composite")


def test_the_world_view_pads_the_study_area_dots(tool):
    view = tool._world_view([("1", 80.0, 10.0), ("2", -50.0, -20.0)])
    px, py = tool.WORLD_INSET_PAD_DEG
    assert view.bounds == pytest.approx((-50.0 - px, -20.0 - py, 80.0 + px, 10.0 + py))
    clipped = tool._world_view([("1", 175.0, 75.0)])
    assert clipped.bounds[2] == 180.0 and clipped.bounds[3] == 80.0


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


def _strict_json(text):
    """Parse *text* as RFC 8259 JSON: NaN, Infinity and -Infinity are errors."""

    def refuse(constant):
        raise ValueError(f"not JSON: {constant}")

    return json.loads(text, parse_constant=refuse)


def test_gallery_stats_are_strict_json_when_a_metrics_file_has_nan(tool, tmp_path, monkeypatch):
    """A metrics file with undefined scores (NaN) and an open size class (Infinity) is copied
    into gallery_stats.json; the stats file is JSON, with null for those values."""
    root = _run_root(tmp_path)
    metrics = {"f1": float("nan"), "n": 3, "per_size_class": {"50-inf": {"upper_ha": float("inf")}}}
    (root / "outputs" / "demo" / "metrics.json").write_text(json.dumps(metrics))
    entry = tool.Entry(
        "99",
        "Demo_metrics",
        "Demo",
        [tool.Layer("outputs/demo/fields.gpkg", "Demo")],
        inset=False,
        metrics="outputs/demo/metrics.json",
    )
    monkeypatch.setattr(tool, "ENTRIES", [entry])
    out = tmp_path / "gallery"
    assert tool.main(["--run-root", str(root), "--out-dir", str(out)]) == 0
    stats = _strict_json((out / "gallery_stats.json").read_text())
    copied = stats["99"]["metrics"]
    assert copied["f1"] is None and copied["n"] == 3
    assert copied["per_size_class"]["50-inf"]["upper_ha"] is None


def test_main_renders_with_proj_network_access_and_records_it(tool, tmp_path, monkeypatch):
    """PNGs depended on PROJ_NETWORK (entry 20's NAD83 outlines moved about 0.3 m without
    PROJ's grids), and nothing recorded which way an entry was rendered."""
    import pyproj.network

    root = _run_root(tmp_path)
    entry = tool.Entry(
        "99", "Demo_proj", "Demo", [tool.Layer("outputs/demo/fields.gpkg", "Demo")], inset=False
    )
    monkeypatch.setattr(tool, "ENTRIES", [entry])
    seen = []
    real = tool.render

    def spy(*args, **kwargs):
        seen.append(pyproj.network.is_network_enabled())
        return real(*args, **kwargs)

    monkeypatch.setattr(tool, "render", spy)
    before = pyproj.network.is_network_enabled()
    out = tmp_path / "gallery"
    args = ["--run-root", str(root), "--out-dir", str(out)]
    try:
        pyproj.network.set_network_enabled(False)  # as with PROJ_NETWORK unset
        assert tool.main(args) == 0
        assert (
            seen == [True]
            and json.loads((out / "gallery_stats.json").read_text())["99"]["proj_network"]
        )
        assert pyproj.network.is_network_enabled() is False  # restored
        assert tool.main([*args, "--no-proj-network"]) == 0
        assert seen == [True, False]
        assert json.loads((out / "gallery_stats.json").read_text())["99"]["proj_network"] is False
    finally:
        pyproj.network.set_network_enabled(before)


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


@pytest.mark.parametrize(
    ("sensors", "label"),
    [
        ("LC08,LC09", "Landsat 8/9 PAN TOA"),
        ("LC09, LC08", "Landsat 8/9 PAN TOA"),
        ("LE07", "Landsat 7 PAN TOA"),
        ("LE07,LC08", "Landsat 7/8 PAN TOA"),
        ("", "Landsat PAN TOA"),
        (None, "Landsat PAN TOA"),  # composites made before the tag existed
        ("LANDSAT_8", "Landsat PAN TOA"),
    ],
)
def test_landsat_pan_label_names_the_sensors_with_images(tool, sensors, label):
    assert tool.SOURCE_LABELS["landsat-pan"] == "Landsat PAN TOA"
    assert tool._landsat_pan_label(sensors) == label


def test_background_note_takes_the_landsat_pan_sensors_from_the_composite(tool):
    comp = {
        "AGRIBOUND_COLLECTIONS": "LANDSAT/LE07/C02/T1_TOA,LANDSAT/LC08/C02/T1_TOA",
        "AGRIBOUND_SENSORS": "LC08",  # Landsat 7 was selected but contributed no image
        "AGRIBOUND_COMPOSITE_METHOD": "median",
        "AGRIBOUND_DATE_START": "2015-01-01",
        "AGRIBOUND_DATE_END_EXCLUSIVE": "2016-01-01",
        "AGRIBOUND_RESOLUTION_M": "15",
    }
    prov = {"facts": {"composite": comp}}
    assert tool._composite_facts(prov)["sensors"] == "LC08"
    note = tool._bg_note(([], "landsat-pan", prov, "the engine's input"))
    assert note.startswith("Landsat 8 PAN TOA 15 m median composite, 2015-01-01 to 2016-01-01")
    assert "Landsat 7" not in note


@pytest.mark.parametrize(
    ("orbit_pass", "label"),
    [
        ("DESCENDING", "Sentinel-1 GRD descending (VV, VH, VV-VH)"),
        ("ascending", "Sentinel-1 GRD ascending (VV, VH, VV-VH)"),
        ("", "Sentinel-1 GRD (VV, VH, VV-VH)"),
        (None, "Sentinel-1 GRD (VV, VH, VV-VH)"),
        ("BOTH", "Sentinel-1 GRD (VV, VH, VV-VH)"),
    ],
)
def test_sentinel1_label_names_the_orbit_pass(tool, orbit_pass, label):
    assert tool.SOURCE_LABELS["sentinel1"] == "Sentinel-1 GRD (VV, VH, VV-VH)"
    assert tool._sentinel1_label(orbit_pass) == label


def test_background_note_takes_the_sentinel1_orbit_pass_from_the_composite(tool):
    comp = {
        "AGRIBOUND_COLLECTIONS": "COPERNICUS/S1_GRD",
        "AGRIBOUND_ORBIT_PASS": "DESCENDING",
        "AGRIBOUND_SENSORS": "S1A",
        "AGRIBOUND_COMPOSITE_METHOD": "median",
        "AGRIBOUND_DATE_START": "2023-04-01",
        "AGRIBOUND_DATE_END_EXCLUSIVE": "2023-10-01",
        "AGRIBOUND_N_IMAGES": "15",
        "AGRIBOUND_RESOLUTION_M": "10",
    }
    prov = {"facts": {"composite": comp}}
    assert tool._composite_facts(prov)["orbit_pass"] == "DESCENDING"
    note = tool._bg_note(([], "sentinel1", prov, "the engine's input"))
    assert note.startswith(
        "Sentinel-1 GRD descending (VV, VH, VV-VH) 10 m median composite, 2023-04-01 to 2023-10-01"
    )
    # AGRIBOUND_N_IMAGES of a sentinel1 composite counts acquisitions, not GRD slices.
    assert note.endswith(", 15 acquisitions; the engine's input")
    assert "images" not in note


def test_image_counts_are_named_by_what_they_count(tool):
    assert tool._image_noun("sentinel1") == "acquisition"
    for source in ("sentinel2", "landsat-pan", None):
        assert tool._image_noun(source) == "image"
    prov = {"facts": {"composite": {"AGRIBOUND_N_IMAGES": "1", "AGRIBOUND_RESOLUTION_M": "10"}}}
    assert tool._bg_note(([], "sentinel1", prov, "input")).endswith(", 1 acquisition; input")
    assert tool._bg_note(([], "sentinel2", prov, "input")).endswith(", 1 image; input")
    comp = {
        "AGRIBOUND_COMPOSITE_METHOD": "median",
        "AGRIBOUND_DATE_START": "2023-01-01",
        "AGRIBOUND_DATE_END_EXCLUSIVE": "2024-01-01",
        "AGRIBOUND_N_IMAGES": "31",
        "AGRIBOUND_RESOLUTION_M": "10",
    }
    panels = [
        {"n": n, "bg": ([], "sentinel1", {"facts": {"composite": comp}}, "the engine's input")}
        for n in ("1", "2")
    ]
    assert tool._areas_bg_note(panels).endswith(
        "1: 2023-01-01 to 2024-01-01, 31 acquisitions; 2: 2023-01-01 to 2024-01-01, 31 acquisitions"
    )


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


def test_the_world_locator_draws_india_from_the_survey_of_india_outline(tool, monkeypatch):
    import matplotlib.pyplot as plt

    soi = _fake_boundaries(tool, monkeypatch)
    pts = [("1", 79.07, 10.78), ("2", 107.33, 40.83)]  # Tamil Nadu and Inner Mongolia
    view = tool._world_view(pts)
    fig = plt.figure(figsize=(4, 3))
    try:
        iax, india_drawn = tool._draw_world_inset(fig, [0.1, 0.1, 0.8, 0.8], pts, view)
        colls = list(iax.collections)
        zorders = [c.get_zorder() for c in colls]
        india_top = colls[1].get_datalim(iax.transData).y1
        n_neighbours = len(colls[0].get_paths())
    finally:
        plt.close(fig)
    assert india_drawn
    assert zorders == [tool.Z_NEIGHBOURS, tool.Z_INDIA, tool.Z_LAKES]
    assert india_top == pytest.approx(soi.bounds[3])  # the Survey of India stand-in, not NE's
    assert n_neighbours == 1  # China (Mongolia is north of the view); NE India never drawn


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


def test_vhr_sar_background_note_carries_the_credit(tool):
    comp = {
        "AGRIBOUND_COMPOSITE_METHOD": "median",
        "AGRIBOUND_DATE_START": "2023-01-01",
        "AGRIBOUND_DATE_END_EXCLUSIVE": "2023-02-01",
        "AGRIBOUND_N_IMAGES": "2",
        "AGRIBOUND_RESOLUTION_M": "1.0",
        "AGRIBOUND_ATTRIBUTION": "© Capella Space Corp, All Rights Reserved.",
        "AGRIBOUND_LICENSE": "CC BY 4.0",
        "AGRIBOUND_LICENSE_URL": "https://creativecommons.org/licenses/by/4.0/",
    }
    # Capella: the author's credit (GALLERY_CREDITS) in place of the composite's tags, and the
    # method said once.
    note = tool._bg_note(([], "capella", {"facts": {"composite": comp}}, "the engine's input"))
    assert note.startswith(
        "Capella GEO X-band SAR 1 m (© Capella Space Corp, CC BY 4.0, modified) median "
        "composite, 2023-01-01 to 2023-02-01"
    )
    assert "All Rights Reserved" not in note and note.count("median") == 1
    # The credit does not depend on the tags: a composite without them is credited too.
    bare = {k: v for k, v in comp.items() if "ATTRIBUTION" not in k and "LICENSE" not in k}
    note = tool._bg_note(([], "capella", {"facts": {"composite": bare}}, "the engine's input"))
    assert "(© Capella Space Corp, CC BY 4.0, modified)" in note
    facts = tool._composite_facts({"facts": {"composite": comp}})
    assert facts["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert facts["attribution"] == "© Capella Space Corp, All Rights Reserved."  # tags kept
    # The printed credit, as gallery_stats.json records it (background_credit).
    capella = ([], "capella", {"facts": {"composite": comp}}, "the engine's input")
    assert tool._bg_credit(capella) == "© Capella Space Corp, CC BY 4.0, modified"
    # Umbra's credit already names the licence: it is not repeated.
    umbra = {**comp, "AGRIBOUND_ATTRIBUTION": "SAR data: Umbra Lab, Inc., CC BY 4.0"}
    note = tool._bg_note(([], "umbra", {"facts": {"composite": umbra}}, "the engine's input"))
    assert note.startswith(
        "Umbra GEC X-band SAR 1 m (SAR data: Umbra Lab, Inc., CC BY 4.0; modified) median "
        "composite, 2023-01-01"
    )
    assert note.count("median") == 1
    assert note.endswith(", 2 collects; the engine's input")
    assert tool._image_noun("umbra") == "collect"
    assert set(tool.SOURCE_LABELS) >= {"umbra", "capella"}
    # Without a credit (no tags, no GALLERY_CREDITS entry) nothing is added.
    assert tool._credit("sentinel1", {}) == ""


def test_areas_on_one_credited_source_carry_the_credit_once(tool):
    comp = {
        "AGRIBOUND_COMPOSITE_METHOD": "median",
        "AGRIBOUND_RESOLUTION_M": "1.0",
        "AGRIBOUND_DATE_START": "2023-01-01",
        "AGRIBOUND_DATE_END_EXCLUSIVE": "2023-02-01",
    }
    panels = [
        {"n": n, "bg": ([], "capella", {"facts": {"composite": comp}}, "the engine's input")}
        for n in ("1", "2")
    ]
    note = tool._areas_bg_note(panels)
    assert note.startswith(
        "Background: Capella GEO X-band SAR 1 m (© Capella Space Corp, CC BY 4.0, modified) "
        "median composites"
    )
    assert note.count("Capella Space Corp") == 1


def test_single_area_stats_record_where_the_crop_and_zoom_squares_were_drawn(
    tool, tmp_path, monkeypatch
):
    """``crop_window`` and ``zoom_window`` hold the squares as drawn (CRS, bounds, centre).

    Only ``zoom_window_m`` (the side) used to be recorded: entry 20's zoom moved about
    11.5 km between the 1.0 and 1.1 renders and the stats did not show it.
    """
    from pyproj import Transformer

    root = _run_root(tmp_path)  # composite x 500000-501000, y 3998800-4000000 (EPSG:32613)
    drawn = []
    real = tool._read_background

    def spy(tifs, source, bounds, crs, out_px, stretch=None):
        drawn.append(tuple(bounds))  # (x0, y0, x1, y1): the map panel, then the zoom panel
        return real(tifs, source, bounds, crs, out_px, stretch)

    monkeypatch.setattr(tool, "_read_background", spy)
    entry = tool.Entry(
        "99",
        "Demo_crop_zoom",
        "Demo",
        [tool.Layer("outputs/demo/fields.gpkg", "Demo")],
        crop_m=600,
        zoom_m=300,
        inset=False,
    )
    stats = tool.render(entry, root, tmp_path / "gallery", "composite")
    main, zoom = drawn
    to_ll = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    for key, side, box_ in (("crop_window", 600, main), ("zoom_window", 300, zoom)):
        record = stats[key]
        assert record["crs"] == "EPSG:32613" and record["side_m"] == side
        assert record["bounds"] == pytest.approx(list(box_))
        lon, lat = to_ll.transform((box_[0] + box_[2]) / 2, (box_[1] + box_[3]) / 2)
        assert record["centre_lonlat"] == [round(lon, 4), round(lat, 4)]
    assert main[2] - main[0] == pytest.approx(600) and zoom[2] - zoom[0] == pytest.approx(300)
    assert stats["zoom_window_m"] == 300
    # Without crop_m or zoom_m there is no square to record.
    plain = tool.Entry(
        "98", "Demo_plain", "Demo", [tool.Layer("outputs/demo/fields.gpkg", "Demo")], inset=False
    )
    stats = tool.render(plain, root, tmp_path / "gallery", "composite")
    assert not {"crop_window", "zoom_window", "zoom_window_m", "window"} & set(stats)


def test_local_square_stats_leave_out_the_crop_and_zoom_squares(guarded, tmp_path):
    root = _run_root(tmp_path)
    _copy_area(root, "demo", "demo_local/a")
    entry = guarded.Entry(
        "98",
        "Demo_local_crop_zoom",
        "Demo",
        [guarded.Layer(f"{L_OUT}/a/fields.gpkg", "A")],
        crop_m=600,
        zoom_m=300,
        inset=False,
    )
    stats = guarded.render(entry, root, tmp_path / "gallery", "composite")
    assert not {"crop_window", "zoom_window"} & set(stats)
    assert stats["zoom_window_m"] == 300  # a side, not a place
    text = json.dumps(stats)
    assert "centre_lonlat" not in text and "bounds" not in text


# --------------------------------------------------------------------------- examples 24 to 27

NEW_KEYS = (
    "24",
    "24b",
    "24c",
    "24d",
    "24e",
    "25",
    "25b",
    "25c",
    "26",
    "26b",
    "27",
    "28",
    "28b",
    "28c",
    "sam",
)


def _summary_rows(path: Path) -> list[dict]:
    import csv

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_the_entries_of_examples_24_to_27_are_listed(tool):
    entries = {e.key: e for e in tool.ENTRIES}
    assert set(NEW_KEYS) <= set(entries)
    keys, names = [e.key for e in tool.ENTRIES], [e.name for e in tool.ENTRIES]
    assert len(keys) == len(set(keys)) and len(names) == len(set(names))  # one image each
    roots = {k: tool._entry_root(entries[k]).relative_to(tool.REPO).as_posix() for k in NEW_KEYS}
    assert roots == {
        "24": "outputs/examples_1.1_ex24b",  # the five Nevada figures from one run root
        "24b": "outputs/examples_1.1_ex24b",
        "24c": "outputs/examples_1.1_ex24b",  # with the irrigated-only score
        "24d": "outputs/examples_1.1_ex24b",
        "24e": "outputs/examples_1.1_ex24b",  # the "-opt" runs of --options
        "25": "outputs/examples_1.1",
        "25b": "outputs/examples_1.1_final",  # MULTISTATE_ZONES = True: the zones' run
        "25c": "outputs/examples_1.1",  # its NAIP 2005 panel: outputs/examples_1.1_r3
        "26": "outputs/examples_1.1_sar",
        "26b": "outputs/examples_1.1_sar",
        "27": "outputs/examples_1.1_final",
        "28": "outputs/examples_1.1_r3/ny",
        "28b": "outputs/examples_1.1_r3/ny",
        "28c": "outputs/examples_1.1_r3/ny",
        "sam": "outputs/examples_1.1_r3/sam_eval",  # its comparators: their own run roots
    }
    # Every panel gives the F1 of the metrics file its example wrote beside the output (24c
    # and 24d: the irrigated-only score's file, metrics_<stem>_irrigated.json; 24e's right
    # panels and the SAM entry: their own tests below).
    for key in ("24", "24b", "24c", "24d", "25b", "26", "26b", "27", "28", "28b", "28c"):
        assert entries[key].title_f1
        suffix = "_irrigated" if key in ("24c", "24d") else ""
        for lay in entries[key].layers:
            folder, name = lay.path.rsplit("/", 1)
            assert lay.metrics == f"{folder}/metrics_{name.removesuffix('.gpkg')}{suffix}.json"
            assert lay.metrics_record is None
    # 24, 24b: four panels of the 5 km evaluation square, each on its own input image, against
    # the DRI fields (with their attribution); 24c and 24d the same, scored on the irrigated
    # fields; 24e: Diamond Valley's square again, before and after the options.
    for key, (x, y) in (
        ("24", (584000, 4389000)),
        ("24b", (310000, 4306000)),
        ("24c", (584000, 4389000)),
        ("24d", (310000, 4306000)),
        ("24e", (584000, 4389000)),
    ):
        entry = entries[key]
        assert len(entry.layers) == 4 and entry.ncols == 2
        assert entry.window == (x + 2500.0, y + 2500.0, 5000.0)
        assert entry.per_layer_background and not entry.input_notes
        assert entry.background_from is None and entry.bg_role is None
        credit = "DRI field boundaries (Nevada, 2026-08-20), Desert Research Institute"
        irrigated = key in ("24c", "24d")
        assert entry.reference_label == (
            f"Irrigated in the panel's year: {credit}" if irrigated else credit
        )
        assert (entry.scored_part is not None) == irrigated
    # 28, 28b, 28c: four panels of the region's 5 km evaluation square, each on its own input
    # image, against the region's CLU fields (example 28's REGIONS corners).
    for key, (x, y) in (("28", (545000, 4575000)), ("28b", (365000, 4745000))) + (
        ("28c", (410000, 4845000)),
    ):
        entry = entries[key]
        assert len(entry.layers) == 4 and entry.ncols == 2 and entry.per_layer_background
        assert entry.window == (x + 2500.0, y + 2500.0, 5000.0)
        assert entry.reference.endswith(f"_reference_{'?' * 12}.gpkg")
        assert "Common Land Units" in entry.reference_label
    # 25: the overview map, without layers; its numbered squares are the areas of 25b (two
    # panels each with the zones), in the same order.
    overview = entries["25"]
    assert overview.overview is not None and not overview.layers and not overview.inset
    areas_25b = list(dict.fromkeys(lay.area for lay in entries["25b"].layers))
    assert overview.overview.marks == areas_25b
    assert any("OpenET agricultural field boundaries" in n for n in overview.overview.notes)
    # 26, 26b: per site Sentinel-1, Umbra or Capella, then NAIP read with the radar's
    # settings (Sentinel-2 at Sevilla), in a window inside the site's evaluation area.
    for key, sites in (
        ("26", ["tamu", "intel_oh", "orleans_vt"]),
        ("26b", ["ucd", "sevilla", "centerfield"]),
    ):
        entry = entries[key]
        assert [lay.area for lay in entry.layers] == [s for s in sites for _ in range(3)]
        for k in range(0, len(entry.layers), 3):
            s1, vhr, optical = (lay.path.rsplit("/", 1)[1] for lay in entry.layers[k : k + 3])
            assert s1.startswith("fields_sentinel1_")
            assert vhr.startswith(("fields_umbra_", "fields_capella_"))
            assert optical.startswith(("fields_naip_sar_settings_", "fields_sentinel2_"))
            assert optical.startswith("fields_sentinel2_") == (entry.layers[k].area == "sevilla")
        for lay in entry.layers:
            assert lay.window_within == f"{tool.SAR}/{lay.area}/evaluation_area.geojson"
            assert lay.reference == f"{tool.SAR}/{lay.area}/reference.gpkg"
    # 27: per square the zones, drawn on the Sentinel-2 image, then Delineate-Anything.
    quebec = entries["27"]
    zones = [lay for lay in quebec.layers if lay.noun == "zone"]
    assert len(quebec.layers) == 4 and len(zones) == 2
    for lay in zones:
        assert lay.path.endswith("/zones_aci-sequence_2016-2023.gpkg")
        assert lay.background_from == lay.path.replace(
            "zones_aci-sequence_2016-2023", "fields_sentinel2_da_2023"
        )


def test_the_notes_of_examples_24_to_27_state_their_credits_and_scores(tool):
    entries = {e.key: e for e in tool.ENTRIES}
    # The Copernicus line on the entries that draw Sentinel imagery: 26, 26b and 27 (whose
    # examples' terms give it), 24 to 24d (the Sentinel-1 image at bottom right) and 28 to 28c
    # (the Sentinel-2 image at bottom left); not on 24e (NAIP and SPOT 6/7) or the SAM entry
    # (SPOT 6/7 and NAIP).
    assert {k for k in NEW_KEYS if entries[k].copernicus_credit} == {
        "24",
        "24b",
        "24c",
        "24d",
        "26",
        "26b",
        "27",
        "28",
        "28b",
        "28c",
    }
    # Where the window is smaller than the scored area, a line says what counts and F1 cover.
    for key in ("25b", "26", "26b", "27"):
        scores = [n for n in entries[key].model_notes if n.startswith("Counts: the whole output")]
        assert len(scores) == 1 and "one-to-one at IoU >= 0.5" in scores[0]
    # 25 and 25b number the squares in increasing F1, and 25b says how its windows are chosen.
    order = "the lowest, the two that cut the ranking into thirds, the highest"
    squares = " ".join(entries["25b"].model_notes)
    assert order in squares and order in entries["25"].overview.notes[0]
    assert "among nine candidates half a side apart" in squares
    # California: no vintage (the delivered one cannot be established); Idaho by its name.
    assert "California Department of Water Resources, Statewide Crop Mapping (with Land IQ)" in (
        squares
    )
    assert "provisional" not in squares and "2023" not in squares.split("Reference:")[1]
    assert "Eastern Snake River Plain irrigated lands (IDWRSP" in squares
    # The overview map has a panel title; the entry's title is the caption.
    assert entries["25"].overview.panel_label and entries["25"].title not in (
        tool._overview_title(entries["25"].overview, 47, 0.31)
    )
    # 24 to 24d: the model lines name their panels (two DINOv3 checkpoints in 24 and 24c).
    assert all(entries[k].model_positions for k in ("24", "24b", "24c", "24d"))
    # 27: the ACI credit with the licence statement and version; BDPPAD marked as modified.
    notes_27 = " ".join(entries["27"].model_notes)
    assert (
        "ACI: Agriculture and Agri-Food Canada, Annual Crop Inventory 2016-2023; contains "
        "information licensed under the Open Government Licence – Canada (2.0)"
    ) in notes_27
    assert "CC BY 4.0 (Données Québec), modified (declared parcels only" in notes_27
    assert "SIGPAC recintos of 2025" in " ".join(entries["26b"].model_notes)
    assert "CC BY 4.0, modified" in " ".join(entries["26b"].model_notes)


def test_the_copernicus_line_gives_the_years_of_the_sentinel_composites(tool, tmp_path):
    def s(source, start, end):
        return source, {"date_start": start, "date_end_exclusive": end}

    held_out = [  # entry 26: Sentinel-1 of 2024, 2024 and 2022; Umbra and NAIP do not count
        s("sentinel1", "2024-04-01", "2024-08-16"),
        s("umbra", "2024-04-01", "2024-08-16"),
        s("sentinel1", "2024-06-01", "2024-09-16"),
        s("naip", "2023-01-01", "2024-01-01"),
        s("sentinel1", "2022-06-01", "2022-09-16"),
    ]
    assert tool._copernicus_note(held_out) == (
        "Contains modified Copernicus Sentinel data 2022 and 2024"
    )
    in_sample = [  # entry 26b: Sentinel-1 of 2024, 2023 and 2025, Sentinel-2 of 2023
        s("sentinel1", "2024-04-01", "2024-09-01"),
        s("sentinel2", "2023-01-01", "2024-01-01"),  # the end is exclusive: 2023 only
        s("sentinel1", "2023-01-01", "2023-03-01"),
        s("sentinel1", "2025-04-01", "2025-10-01"),
    ]
    assert (
        tool._copernicus_note(in_sample) == "Contains modified Copernicus Sentinel data 2023-2025"
    )
    assert tool._copernicus_note([s("naip", "2022-01-01", "2023-01-01")]) is None
    assert (
        tool._copernicus_note([("sentinel2", {})]) == "Contains modified Copernicus Sentinel data"
    )
    assert tool._year_list([2022, 2024, 2025]) == "2022 and 2024-2025"
    assert tool._and_join(["top left", "bottom left", "bottom right"]) == (
        "top left, bottom left and bottom right"
    )
    # Drawn by both renderers with the flag (the synthetic composite: Sentinel-2 of 2024).
    root = _run_root(tmp_path)
    single = tool.Entry(
        "86",
        "Demo_copernicus",
        "Demo",
        [tool.Layer("outputs/demo/fields.gpkg", "A")],
        inset=False,
        copernicus_credit=True,
    )
    areas = tool.Entry(
        "85",
        "Demo_copernicus_areas",
        "Demo",
        [tool.Layer("outputs/demo/fields.gpkg", "A")],
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
        inset=False,
        copernicus_credit=True,
    )
    line = "Contains modified Copernicus Sentinel data 2024"
    for entry, renderer in ((single, tool.render), (areas, tool.render_areas)):
        stats = renderer(entry, root, tmp_path / "gallery", "composite")
        assert line in stats["background_note"].split("\n")
        assert stats["copernicus_credit"] == line  # as printed, for the captions
        assert "background_credit" not in stats["layers"][0]  # Sentinel-2 has no tag credit
    import dataclasses

    plain = dataclasses.replace(single, copernicus_credit=False)
    stats = tool.render(plain, root, tmp_path / "gallery", "composite")
    assert "Copernicus" not in stats["background_note"] and "copernicus_credit" not in stats


def test_model_lines_name_their_panels(tool, tmp_path):
    """Entry 24: two fine-tuned DINOv3 checkpoints (NAIP, SPOT) read alike but for the hash."""
    root = _run_root(tmp_path)
    demo = root / "outputs" / "demo"
    base = json.loads((demo / "fields.gpkg.provenance.json").read_text())
    for name, sha in (("naip", "262c121aa"), ("spot", "88006e2bb")):
        gpd.read_file(demo / "fields.gpkg").to_file(demo / f"fields_{name}.gpkg", driver="GPKG")
        sidecar = json.loads(json.dumps(base))
        sidecar["config"].update(engine="dinov3", engine_params={"checkpoint_path": "ft.pth"})
        sidecar["engine_meta"] = {
            "model_name": "dinov3_vitl16",
            "weights": "w.pth",
            "checkpoint_sha256": sha,
        }
        (demo / f"fields_{name}.gpkg.provenance.json").write_text(json.dumps(sidecar))
    layers = [
        tool.Layer("outputs/demo/fields_naip.gpkg", "A"),
        tool.Layer("outputs/demo/fields_spot.gpkg", "B"),
    ]
    entry = tool.Entry("84", "Demo_positions", "Demo", layers, inset=False, model_positions=True)
    stats = tool.render(entry, root, tmp_path / "gallery", "composite")
    assert [m.rsplit(" (", 1)[1] for m in stats["model_notes"]] == ["left)", "right)"]
    assert "sha256 262c121" in stats["model_notes"][0]
    assert "sha256 88006e2" in stats["model_notes"][1]
    # Without the flag, or with one model for every panel, the lines are as before.
    import dataclasses

    stats = tool.render(
        dataclasses.replace(entry, model_positions=False), root, tmp_path / "g2", "composite"
    )
    assert not any(m.endswith(("(left)", "(right)")) for m in stats["model_notes"])
    same = dataclasses.replace(entry, layers=[layers[0], layers[0]])
    stats = tool.render(same, root, tmp_path / "g3", "composite")
    assert len(stats["model_notes"]) == 1 and not stats["model_notes"][0].endswith("right)")


def test_model_lines_name_the_study_areas_of_a_multi_area_entry(tool, tmp_path):
    """Entry 23e: each study area's fine-tuned checkpoints read alike but for the hash."""
    root = _run_root(tmp_path)
    _second_area(root)
    released = {  # the released Delineate-Anything weights, in both areas
        "model_key": "large_v2",
        "model_architecture": "YOLO11x-seg",
        "weights_repo": "MykolaL/DelineateAnything",
        "weights_filename": "DelineateAnythingv2.pt",
        "weights_revision": "369d0b4abc",
        "ultralytics_version": "8.4.163",
    }
    for folder, sha in (("demo", "aaaaaaa11"), ("demo2", "bbbbbbb22")):
        out = root / "outputs" / folder
        side = out / "fields.gpkg.provenance.json"
        sidecar = json.loads(side.read_text())
        sidecar["config"]["engine"] = "delineate-anything"
        sidecar["engine_meta"] = released
        side.write_text(json.dumps(sidecar))
        gpd.read_file(out / "fields.gpkg").to_file(out / "fields_ft.gpkg", driver="GPKG")
        sidecar = json.loads(json.dumps(sidecar))
        sidecar["config"].update(engine="dinov3", engine_params={"checkpoint_path": "ft.pth"})
        sidecar["engine_meta"] = {
            "model_name": "dinov3_vitl16",
            "weights": "w.pth",
            "checkpoint_sha256": sha,
        }
        (out / "fields_ft.gpkg.provenance.json").write_text(json.dumps(sidecar))
    layers = [
        tool.Layer("outputs/demo/fields.gpkg", "A: released", area="a"),
        tool.Layer("outputs/demo/fields_ft.gpkg", "A: fine-tuned", area="a"),
        tool.Layer("outputs/demo2/fields.gpkg", "B: released", area="b"),
        tool.Layer("outputs/demo2/fields_ft.gpkg", "B: fine-tuned", area="b"),
    ]
    entry = tool.Entry(
        "82",
        "Demo_area_positions",
        "Demo",
        layers,
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
        ncols=2,
        inset=False,
        model_positions=True,
    )
    stats = tool.render_areas(entry, root, tmp_path / "gallery", "composite")
    lines = stats["model_notes"]
    assert len(lines) == 3
    assert lines[0].startswith("Delineate-Anything") and lines[0].endswith("(study areas 1-2)")
    assert "sha256 aaaaaaa" in lines[1] and lines[1].endswith("(study area 1)")
    assert "sha256 bbbbbbb" in lines[2] and lines[2].endswith("(study area 2)")
    # Without the flag the lines are as before.
    import dataclasses

    plain = dataclasses.replace(entry, model_positions=False)
    stats = tool.render_areas(plain, root, tmp_path / "g2", "composite")
    assert not any("study area" in m for m in stats["model_notes"])


def test_dinov3_lines_name_the_processing_grid(tool):
    """Entry 14's Landsat run read a 10 m bilinear copy of the 30 m composite (engine_meta
    "processing"); a run on the raster as it is has no such clause."""
    prov = {
        "config": {
            "engine": "dinov3",
            "fine_tune": True,
            "fine_tune_epochs": 30,
            "fine_tune_split": "block",
        },
        "engine_meta": {
            "model_name": "dinov3_vitl16",
            "weights": "giswqs/geoai/dinov3_vitl16_sat493m.pth",
            "weights_revision": "aa2b25d27b2c",
            "geoai-py_version": "0.43.1",
            "use_lora": False,
            "processing": {"gsd_m": 10.0, "scale": 3.0, "resampling": "bilinear"},
        },
    }
    assert tool._model_note(prov) == (
        "DINOv3 dinov3_vitl16 with giswqs/geoai/dinov3_vitl16_sat493m.pth @ aa2b25d "
        "(geoai-py 0.43.1), fine-tuned on the reference polygons (full, up to 30 epochs, "
        "block split), read on a 10 m bilinear copy of the 30 m raster"
    )
    native = json.loads(json.dumps(prov))
    del native["engine_meta"]["processing"]
    assert tool._model_note(native).endswith("(full, up to 30 epochs, block split)")
    # Delineate-Anything's processing grid (the SAR settings) is named by the entries' notes.
    da = {
        "config": {"engine": "delineate-anything"},
        "engine_meta": {
            "model_key": "large_v2",
            "weights": "checkpoint",
            "checkpoint_sha256": "abc1234",
            "processing": {"gsd_m": 3.0, "scale": 3.0, "resampling": "average"},
        },
    }
    assert "copy" not in tool._model_note(da)
    entry = next(e for e in tool.ENTRIES if e.key == "14")
    assert entry.model_positions  # the Landsat line differs from the other three


def test_a_multi_area_figure_sets_its_titles_in_one_size(tool, tmp_path, monkeypatch):
    import matplotlib.axes

    root = _run_root(tmp_path)
    _second_area(root)
    sizes = {}
    real = matplotlib.axes.Axes.set_title

    def spy(self, label, *args, **kwargs):
        sizes[label] = kwargs.get("fontsize")
        return real(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", spy)
    long = "Area B: a much longer panel title that must be set smaller to fit its panel"
    entry = tool.Entry(
        "83",
        "Demo_title_sizes",
        "Demo",
        [
            tool.Layer("outputs/demo/fields.gpkg", "A"),
            tool.Layer("outputs/demo2/fields.gpkg", long),
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
        ncols=2,
        inset=False,
    )
    tool.render_areas(entry, root, tmp_path / "gallery", "composite")
    panel_w = tool.FIG_WIDTH_IN * (tool.MAPS_RIGHT - tool.MAPS_LEFT) / (2 + tool.MAPS_WSPACE)
    titles = {t: fs for t, fs in sizes.items() if t.startswith(("1. ", "2. "))}
    assert len(titles) == 2 and len(set(titles.values())) == 1
    assert next(iter(titles.values())) == min(tool._fit_title(t, panel_w) for t in titles)
    assert next(iter(titles.values())) < tool.TITLE_FS  # the long title sets the size


def test_the_nevada_panels_are_drawn_on_the_images_they_were_delineated_on(tool):
    """24 to 24d (author, 2026-10-11): each panel on its own input composite, NAIP 2022 in the
    top row, SPOT 6/7 2023 at bottom left and Sentinel-1 2024 at bottom right (no shared NAIP
    background), with the Copernicus line of the Sentinel-1 image; all four read the
    re-scoring run of example 24."""
    entries = {e.key: e for e in tool.ENTRIES}
    nevada = [entries[k] for k in ("24", "24b", "24c", "24d")]
    for entry in nevada:
        assert entry.per_layer_background and not entry.input_notes
        assert (entry.background, entry.background_from, entry.bg_role) == (None, None, None)
        assert all(lay.background is None and lay.background_from is None for lay in entry.layers)
        assert entry.copernicus_credit and entry.run_root == tool.EX24_RUN_ROOT
    root = tool.REPO / tool.EX24_RUN_ROOT
    if not (root / tool.NV).is_dir():
        pytest.skip("the re-scoring run of example 24 is not in this checkout")
    for entry in nevada:
        sources, years = [], []
        for lay in entry.layers:
            prov = tool._sidecar(tool._one(lay.path, root))
            tifs, source, _prov, role = tool._background_for(entry, root, prov, "composite", lay)
            assert tifs == [root / prov["facts"]["raster_path"]]  # the panel's own input
            assert role == "the engine's input"
            sources.append(source)
            years.append(int(prov["config"]["year"]))
        assert sources == ["naip", "naip", "spot", "sentinel1"], entry.key
        assert years == [2022, 2022, 2023, 2024], entry.key


def test_the_nevada_panels_follow_their_rule(tool):
    rows = [
        {"basin": "b", "run": "naip da", "source": "naip", "status": "ok", "f1": "0.3"},
        {"basin": "b", "run": "naip da-ft", "source": "naip", "status": "ok", "f1": "0.5"},
        {"basin": "b", "run": "naip dinov3-ft", "source": "naip", "status": "ok", "f1": "0.5"},
        {"basin": "b", "run": "naip da-sam2", "source": "naip", "status": "failed: X", "f1": ""},
        {"basin": "b", "run": "spot da", "source": "spot", "status": "ok", "f1": "0.2"},
        {"basin": "b", "run": "spot dinov3-ft", "source": "spot", "status": "ok", "f1": "0.4"},
        {"basin": "b", "run": "sentinel1 da", "source": "sentinel1", "status": "ok", "f1": "0.1"},
        {"basin": "x", "run": "spot da-ft", "source": "spot", "status": "ok", "f1": "0.9"},
    ]
    # The released NAIP run, then the best other NAIP run (a tie goes to the earlier row),
    # the best SPOT run and the Sentinel-1 run; another basin's rows do not count.
    assert tool._nevada_panels(rows, "b") == [
        "naip da",
        "naip da-ft",
        "spot dinov3-ft",
        "sentinel1 da",
    ]
    no_spot = [r for r in rows if r["source"] != "spot"]
    assert tool._nevada_panels(no_spot, "b") == ["naip da", "naip da-ft", "sentinel1 da"]
    # Ranked by another summary column (the irrigated-only F1 of entries 24c and 24d).
    other = [{**r, "f1_irrigated": str(1 - float(r["f1"] or 0))} for r in rows]
    assert tool._nevada_panels(other, "b", score="f1_irrigated") == [
        "naip da",
        "naip da-ft",  # a tie with naip dinov3-ft: the earlier row
        "spot da",
        "sentinel1 da",
    ]
    entries = {e.key: e for e in tool.ENTRIES}
    for keys, basin in ((("24", "24c"), "diamond_valley"), (("24b", "24d"), "mason_valley")):
        stems = [tool.NEVADA_RUNS[run][0] for run in tool.NEVADA_PANELS[basin]]
        expected = [f"{tool.NV}/{basin}/{stem}.gpkg" for stem in stems]
        for key in keys:
            assert [lay.path for lay in entries[key].layers] == expected
    # The entries' run (example 24 with --options and --sam-auto): the rule picks the runs of
    # NEVADA_PANELS, by the F1 and by the irrigated-only F1 (sam-auto takes three panels).
    assert tool.NEVADA_PANELS == {
        "diamond_valley": ["naip da", "naip sam-auto", "spot sam-auto", "sentinel1 da"],
        "mason_valley": ["naip da", "naip dinov3-ft", "spot sam-auto", "sentinel1 da"],
    }
    summary = tool.REPO / tool.EX24_RUN_ROOT / tool.NV / "summary.csv"
    if summary.exists():
        rows = _summary_rows(summary)
        for basin, panels in tool.NEVADA_PANELS.items():
            assert tool._nevada_panels(rows, basin) == panels
            assert tool._nevada_panels(rows, basin, score="f1_irrigated") == panels
        # Every run the summary holds has a stem and a label, the "-opt" runs aside (24e).
        runs = {r["run"] for r in rows if not r["run"].endswith(tool.NEVADA_OPT_SUFFIX)}
        drawable = {r for r in runs if r.split(" ", 1)[0] in ("naip", "spot", "sentinel1")}
        assert drawable <= set(tool.NEVADA_RUNS)
        assert {r for r in runs if r.endswith(" sam-auto")} <= set(tool.NEVADA_RUNS)
    # The 1.1 run (no sam-auto or "-opt" rows) and its re-scoring: the runs of the 1.1 gallery.
    before = {
        "diamond_valley": ["naip da", "naip dinov3-ft", "spot dinov3-ft", "sentinel1 da"],
        "mason_valley": ["naip da", "naip dinov3-ft", "spot da", "sentinel1 da"],
    }
    for summary in (
        tool.RUN_ROOT / tool.NV / "summary.csv",
        tool.REPO / "outputs/examples_1.1_ex24" / tool.NV / "summary.csv",
    ):
        if summary.exists():
            rows = _summary_rows(summary)
            for basin, panels in before.items():
                assert tool._nevada_panels(rows, basin) == panels


def test_the_nevada_panel_rule_leaves_out_the_opt_runs(tool):
    """Example 24's "-opt" runs (--options) never take a panel of 24-24d, whatever their F1:
    the options get their own before-and-after entry (24e). Its direct SAM runs (--sam-auto)
    compete like the other runs of their source."""
    assert tool.NEVADA_OPT_SUFFIX == "-opt"
    runs = [  # (run, source, F1); every "-opt" run scores higher than the runs it repeats
        ("naip da", "naip", "0.3"),
        ("naip dinov3-ft", "naip", "0.4"),
        ("naip sam-auto", "naip", "0.45"),
        ("spot dinov3-ft", "spot", "0.5"),
        ("sentinel1 da", "sentinel1", "0.2"),
        ("naip da-opt", "naip", "0.9"),
        ("naip dinov3-ft-opt", "naip", "0.8"),
        ("spot dinov3-ft-opt", "spot", "0.9"),
        ("sentinel1 da-opt", "sentinel1", "0.7"),
    ]
    rows = [{"basin": "b", "run": r, "source": s, "status": "ok", "f1": f} for r, s, f in runs]
    picks = ["naip da", "naip sam-auto", "spot dinov3-ft", "sentinel1 da"]
    assert tool._nevada_panels(rows, "b") == picks
    plain = [r for r in rows if not r["run"].endswith("-opt")]
    assert tool._nevada_panels(plain, "b") == picks  # the "-opt" rows change nothing
    irrigated = [{**r, "f1_irrigated": r["f1"]} for r in rows]
    assert tool._nevada_panels(irrigated, "b", score="f1_irrigated") == picks
    # A basin whose only run of a source is an "-opt" run has no panel for that source.
    only_opt = [r for r in rows if r["source"] != "spot" or r["run"].endswith("-opt")]
    assert tool._nevada_panels(only_opt, "b") == ["naip da", "naip sam-auto", "sentinel1 da"]
    # The run of example 24 with --options and --sam-auto (outputs/examples_1.1_ex24b, the
    # entries' run root), when it is present: no "-opt" run is picked, and the rule picks what
    # it picks without them, by either score (`sam-auto` takes three slots: both of Diamond
    # Valley's NAIP and SPOT ones and Mason Valley's SPOT one).
    assert tool.EX24_RUN_ROOT == "outputs/examples_1.1_ex24b"
    summary = tool.REPO / tool.EX24_RUN_ROOT / tool.NV / "summary.csv"
    if summary.exists():
        rows = _summary_rows(summary)
        assert any(r["run"].endswith("-opt") for r in rows)
        expected = {
            "diamond_valley": ["naip da", "naip sam-auto", "spot sam-auto", "sentinel1 da"],
            "mason_valley": ["naip da", "naip dinov3-ft", "spot sam-auto", "sentinel1 da"],
        }
        assert expected == tool.NEVADA_PANELS
        plain = [r for r in rows if not r["run"].endswith("-opt")]
        for basin, panels in expected.items():
            for score in ("f1", "f1_irrigated"):
                assert tool._nevada_panels(rows, basin, score=score) == panels, (basin, score)
                assert tool._nevada_panels(plain, basin, score=score) == panels, (basin, score)


def test_the_irrigated_only_entries_draw_the_runs_of_24_and_24b(tool):
    """24c and 24d: the panels, image, window and notes of 24 and 24b, with example 24's
    irrigated-only score of the same runs, read from the re-scoring run root."""
    import dataclasses

    entries = {e.key: e for e in tool.ENTRIES}
    for key, base in (("24c", "24"), ("24d", "24b")):
        entry, plain = entries[key], entries[base]
        assert entry.run_root == plain.run_root == tool.EX24_RUN_ROOT
        assert tool.EX24_RUN_ROOT == "outputs/examples_1.1_ex24b"
        assert plain.scored_part is None
        assert entry.scored_part == tool.NEVADA_IRRIGATED
        assert [(lay.path, lay.label) for lay in entry.layers] == [
            (lay.path, lay.label) for lay in plain.layers
        ]
        assert [lay.metrics for lay in entry.layers] == [
            lay.metrics.replace(".json", "_irrigated.json") for lay in plain.layers
        ]
        assert entry.reference_label == f"Irrigated in the panel's year: {plain.reference_label}"
        assert entry.pred_label == f"{tool.PRED_LABEL}, scored"
        assert entry.model_notes == [tool.NEVADA_IRRIGATED_PANEL_NOTE, *plain.model_notes[1:]]
        # Everything else is the all-fields entry's (reference, backgrounds, window, layout,
        # credits, run root).
        assert (
            dataclasses.replace(
                entry,
                key=plain.key,
                name=plain.name,
                title=plain.title,
                layers=plain.layers,
                reference_label=plain.reference_label,
                pred_label=None,
                model_notes=plain.model_notes,
                scored_part=None,
            )
            == plain
        )
    part = tool.NEVADA_IRRIGATED
    assert (part.column, part.value, part.title) == (
        "irrigated_{year}",
        "irrigated",
        "irrigated only",
    )
    assert part.left_out_label.format(years=tool._or_join([2022, 2023, 2024])) == (
        "Not irrigated in the panel's year (2022, 2023 or 2024): left out of the score"
    )
    assert "{counts}" in part.note and "count as false positives" in part.note
    # A prediction not scored that is thinner than the outlines drawn over its fill (24c: a
    # 0.28 ha strip between two pivots) does not show; the footer says which outline wins.
    assert part.note.endswith("Red outlines are drawn over grey ones where they coincide")


def _scored_root(tool, tmp_path: Path) -> Path:
    """A run root with an output scored on part of its reference (``Entry.scored_part``):
    one field irrigated in 2024, two not (one without a value), and the metrics file of
    agribound.evaluate with the others as ``ignore``."""
    from agribound.evaluate import evaluate

    root = _run_root(tmp_path)
    demo = root / "outputs" / "demo"
    ref = gpd.GeoDataFrame(
        {"irrigated_2024": ["irrigated", "not irrigated", None]},
        geometry=[
            box(500100, 3999100, 500400, 3999400),
            box(500500, 3999500, 500900, 3999900),
            box(500100, 3999500, 500300, 3999700),
        ],
        crs="EPSG:32613",
    )
    ref.to_file(demo / "reference_part.gpkg", driver="GPKG")
    pred = gpd.GeoDataFrame(
        geometry=[
            box(500110, 3999110, 500390, 3999390),  # on the irrigated field: scored, matched
            box(500520, 3999520, 500880, 3999880),  # point in a field left out: not scored
            box(500850, 3999600, 500950, 3999700),  # point on that field's edge: not scored
            box(500600, 3999000, 500800, 3999200),  # unmapped ground: scored, a false positive
            box(500120, 3999520, 500280, 3999680),  # in the field without a value: not scored
        ],
        crs="EPSG:32613",
    )
    pred.to_file(demo / "fields_part.gpkg", driver="GPKG")
    sidecar = (demo / "fields.gpkg.provenance.json").read_text()
    (demo / "fields_part.gpkg.provenance.json").write_text(sidecar)
    irrigated = (ref["irrigated_2024"] == "irrigated").to_numpy()
    metrics = evaluate(pred, ref[irrigated], ignore=ref[~irrigated])
    record = {"row": {"year": 2024, "f1_irrigated": metrics["f1"]}, "metrics": metrics}
    (demo / "metrics_part.json").write_text(json.dumps(tool.to_jsonable(record)))
    return root


def _scored_entry(tool, **kw):
    part = tool.ScoredPart(
        column="irrigated_{year}",
        value="irrigated",
        left_out_label="Not irrigated in {years}: left out of the score",
        ignored_label="Not scored",
        title="irrigated only",
        note="Irrigated only: {counts}",
    )
    layer = tool.Layer(
        "outputs/demo/fields_part.gpkg", "Demo", metrics="outputs/demo/metrics_part.json"
    )
    return tool.Entry(
        "79",
        "Demo_scored_part",
        "Demo",
        [layer],
        reference="outputs/demo/reference_part.gpkg",
        reference_label="Irrigated in the panel's year",
        title_f1=True,
        inset=False,
        scored_part=part,
        **kw,
    )


def test_a_scored_part_draws_what_its_score_left_out(tool, tmp_path, monkeypatch):
    """Entry.scored_part: the scored field in cyan, the others outlined and hatched; the
    predictions whose representative point lies in a left-out field (inside it or on its
    edge, as agribound.evaluate's ignore) outlined in grey on a lighter grey casing and
    filled under the hatch and every outline, the others in red; a two-line title; the
    legend sample as on the map."""
    import matplotlib.axes
    import matplotlib.figure
    from matplotlib.colors import to_rgba
    from matplotlib.transforms import IdentityTransform

    root = _scored_root(tool, tmp_path)
    # Per collection: edge RGBA (None without an edge), face RGBA (None without a face), number
    # of paths, zorder, line width (points); the panel titles; the handles of each legend.
    drawn, titles, legends = [], [], []
    add_collection, set_title = matplotlib.axes.Axes.add_collection, matplotlib.axes.Axes.set_title
    figure_legend = matplotlib.figure.Figure.legend

    def first(colours):
        return tuple(np.round(colours[0], 4)) if len(colours) else None

    def spy_collection(self, collection, *args, **kwargs):
        widths = np.atleast_1d(collection.get_linewidth())
        drawn.append(
            (
                first(collection.get_edgecolor()),
                first(collection.get_facecolor()),
                len(collection.get_paths()),
                collection.get_zorder(),
                float(widths[0]) if len(widths) else None,
            )
        )
        return add_collection(self, collection, *args, **kwargs)

    def spy_title(self, label, *args, **kwargs):
        titles.append(label)
        return set_title(self, label, *args, **kwargs)

    def spy_legend(self, *args, **kwargs):
        legends.append(kwargs.get("handles"))
        return figure_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "add_collection", spy_collection)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", spy_title)
    monkeypatch.setattr(matplotlib.figure.Figure, "legend", spy_legend)
    stats = tool.render(_scored_entry(tool), root, tmp_path / "gallery", "composite")

    def rgba(colour, alpha=None):  # alpha None keeps the colour's own
        return tuple(np.round(to_rgba(colour, alpha), 4))

    def lines(colour, alpha=1.0):
        return sum(n for edge, _face, n, _z, _lw in drawn if edge == rgba(colour, alpha))

    assert lines(tool.PRED_COLOR) == 2  # the predictions scored
    assert lines(tool.IGNORED_COLOR) == 3  # point in a left-out field, on its edge, no value
    assert lines(tool.REF_COLOR) == 1  # the field scored
    assert lines(tool.LEFT_OUT_EDGE) == 2  # the fields left out, outlined and hatched
    hatch = lines(tool.LEFT_OUT_HATCH, tool.LEFT_OUT_HATCH_ALPHA)
    assert hatch >= 4 and lines(tool.LEFT_OUT_CASING, tool.LEFT_OUT_CASING_ALPHA) == hatch
    # The predictions not scored are also filled, without an edge, above the image and under
    # the hatch and every outline, so that one whose outline lies under a red one still shows.
    fill = rgba(tool.IGNORED_COLOR, tool.IGNORED_FILL_ALPHA)
    assert [(face, edge, n, z) for edge, face, n, z, _lw in drawn if face is not None] == [
        (fill, None, 3, 2.0)
    ]
    assert all(z > 2.0 for edge, _face, _n, z, _lw in drawn if edge is not None)
    # Their grey outline lies on a lighter grey casing, IGNORED_CASING_LW points wider, above
    # the hatch and the left-out outlines (2.5) and under the scored reference (3) and red (4),
    # so that it shows on dark ground (the Sentinel-1 images of 24c and 24d).
    casing = rgba(tool.IGNORED_CASING, tool.IGNORED_CASING_ALPHA)
    cased = [(n, z, lw) for edge, _face, n, z, lw in drawn if edge == casing]
    grey = [(n, z, lw) for edge, _face, n, z, lw in drawn if edge == rgba(tool.IGNORED_COLOR)]
    assert [(n, z) for n, z, _lw in cased] == [(3, 2.6)] and [(n, z) for n, z, _ in grey] == [
        (3, 3.9)
    ]
    assert cased[0][2] == pytest.approx(grey[0][2] + tool.IGNORED_CASING_LW)
    handle = next(h for h in legends[-1] if h.get_label() == "Not scored")
    assert isinstance(handle, tool._NotScoredSwatch)
    assert rgba(handle.get_facecolor()) == fill
    assert rgba(handle.get_edgecolor()) == rgba(tool.IGNORED_COLOR)
    # The legend sample draws the fill, then the casing and the grey edge on it.
    sample = tool.LEGEND_HANDLERS[tool._NotScoredSwatch].create_artists(
        None, handle, 0, 0, 20, 10, 9, IdentityTransform()
    )
    assert [a.get_fill() for a in sample] == [True, False, False]
    assert rgba(sample[0].get_facecolor()) == fill
    assert [rgba(a.get_edgecolor()) for a in sample[1:]] == [casing, rgba(tool.IGNORED_COLOR)]
    assert sample[1].get_linewidth() == pytest.approx(
        handle.get_linewidth() + tool.IGNORED_CASING_LW
    )
    assert titles == ["Demo\n2 of 5 fields scored; F1 0.67, irrigated only"]
    assert stats["layers"][0]["scored_part"] == {
        "year": 2024,
        "column": "irrigated_2024",
        "value": "irrigated",
        "reference_scored": 1,
        "reference_left_out": 2,
        "predicted": 5,
        "predicted_scored": 2,
        "predicted_ignored": 3,
        "f1": pytest.approx(2 / 3),
    }
    assert stats["scored_part"]["left_out_label"] == "Not irrigated in 2024: left out of the score"
    assert stats["scored_part"]["ignored_label"] == "Not scored"
    assert "Irrigated only: 1 of 3 in 2024" in stats["model_notes"]
    assert stats["reference"]["n_polygons"] == 3  # the whole reference, both parts


def test_a_scored_part_must_agree_with_its_metrics_file(tool, tmp_path):
    """The figure is refused when its split differs from the one the metrics file scored."""
    import dataclasses

    root = _scored_root(tool, tmp_path)
    path = root / "outputs" / "demo" / "metrics_part.json"
    data = json.loads(path.read_text())
    for key, value, match in (
        ("count_predicted_ignored", 2, "count_predicted_ignored is 2, but the figure would draw 3"),
        ("count_reference", 3, "count_reference is 3, but the figure would draw 1"),
        ("count_predicted", 3, "3 predictions scored or dropped and 3 not scored, but the output"),
        ("year", None, "the summary row gives no year"),
        ("year", 2023, "the reference has no column 'irrigated_2023'"),
    ):
        bad = json.loads(json.dumps(data))
        bad["row" if key == "year" else "metrics"][key] = value
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=match):
            tool.render(_scored_entry(tool), root, tmp_path / "gallery", "composite")
    path.write_text(json.dumps(data))
    entry = _scored_entry(tool)
    unscored = dataclasses.replace(entry, layers=[tool.Layer("outputs/demo/fields_part.gpkg", "A")])
    for bad_entry in (unscored, dataclasses.replace(entry, zoom_m=200)):
        with pytest.raises(ValueError, match="scored_part needs"):
            tool.render(bad_entry, root, tmp_path / "gallery", "composite")
    areas = dataclasses.replace(
        entry, reference=None, multi_area=True, per_layer_background=True, crop_m=400
    )
    with pytest.raises(ValueError, match="do not support scored_part"):
        tool.render_areas(areas, root, tmp_path / "gallery", "composite")
    overview = dataclasses.replace(
        entry, layers=[], overview=tool.Overview(summary="s.csv", run="naip", metric_label="F1")
    )
    with pytest.raises(ValueError, match="scored_part is for map panels"):
        tool.render_overview(overview, root, tmp_path / "gallery", "composite")
    assert not (tmp_path / "gallery" / "Demo_scored_part.png").exists()


def test_the_predictions_not_scored_are_the_ones_evaluate_ignores(tool, tmp_path):
    """The grey predictions are the ones agribound.evaluate's ignore leaves out: the point of
    the repaired polygon decides. Two invalid predictions (a box and a strip across it, in one
    multipolygon) straddle a scored and a left-out field; the point of each unrepaired
    multipolygon lies in its strip, the point of each repaired one (their union) in the other
    field, so that a rule on the unrepaired polygons would swap the two at equal counts. An
    output in another CRS is taken in the reference's, as example 24 evaluates its outputs."""
    from shapely.geometry import MultiPolygon

    from agribound.evaluate import evaluate

    x0, y0 = 500000, 3999000
    ref = gpd.GeoDataFrame(  # fields 0 and 2 irrigated (scored), field 1 left out
        {"irrigated_2024": ["irrigated", "not irrigated", "irrigated"]},
        geometry=[box(x0 + 200 * k, y0, x0 + 200 * (k + 1), y0 + 200) for k in range(3)],
        crs="EPSG:32613",
    )
    irrigated = (ref["irrigated_2024"] == "irrigated").to_numpy()

    def box_and_strip(x, y, strip_x, strip_y):  # overlapping parts: an invalid multipolygon
        return MultiPolygon(
            [
                box(x0 + x, y0 + y, x0 + x + 150, y0 + y + 90),
                box(x0 + strip_x, y0 + strip_y, x0 + strip_x + 250, y0 + strip_y + 10),
            ]
        )

    invalid = gpd.GeoDataFrame(
        geometry=[
            box_and_strip(0, 0, 100, 40),  # box in field 0, strip into field 1: scored
            box_and_strip(200, 110, 300, 150),  # box in field 1, strip into field 2: not scored
            box(x0 + 220, y0 + 20, x0 + 380, y0 + 100),  # in field 1: not scored
            box(x0 + 420, y0 + 20, x0 + 580, y0 + 100),  # in field 2: scored
        ],
        crs="EPSG:32613",
    )
    # The points of the unrepaired polygons would swap the first two (the same count).
    unrepaired = invalid.representative_point().intersects(ref[~irrigated].union_all())
    assert unrepaired.tolist() == [True, False, True, False]
    part = _scored_entry(tool).scored_part
    for name, pred, expected in (
        ("invalid", invalid, [False, True, True, False]),
        ("geographic", invalid.iloc[2:].to_crs("EPSG:4326"), [True, False]),
    ):
        evaluated = pred.to_crs(ref.crs)  # as example 24 passes its outputs to evaluate
        metrics = evaluate(evaluated, ref[irrigated], ignore=ref[~irrigated])
        path = tmp_path / f"metrics_{name}.json"
        path.write_text(json.dumps(tool.to_jsonable({"row": {"year": 2024}, "metrics": metrics})))
        drawn = tool._scored_panel(part, path, ref, pred)["ignored_mask"].tolist()
        # evaluate's verdict on each prediction alone
        alone = [
            evaluate(evaluated.iloc[[i]], ref[irrigated], ignore=ref[~irrigated])[
                "count_predicted_ignored"
            ]
            == 1
            for i in range(len(pred))
        ]
        assert drawn == alone == expected, name
    assert tool._not_scored(invalid.iloc[:0], ref[~irrigated], None).tolist() == []
    assert tool._not_scored(invalid, ref.iloc[:0], "EPSG:6933").tolist() == [False] * 4


def test_hatch_lines_lie_in_their_fields_on_one_lattice(tool):
    import shapely

    a, b = box(0, 0, 100, 100), box(300, 0, 400, 50)
    alone = tool._hatch_lines(gpd.GeoSeries([a], crs="EPSG:32613"), 10.0).iloc[0]
    both = tool._hatch_lines(gpd.GeoSeries([a, b, None], crs="EPSG:32613"), 10.0).iloc[0]
    assert alone.difference(a.buffer(1e-6)).is_empty
    assert both.difference(shapely.union_all([a, b]).buffer(1e-6)).is_empty
    # One lattice: field a is hatched alike with or without field b.
    assert alone.symmetric_difference(both.intersection(a)).length < 1e-6
    # Lines 10 apart across them: about area / spacing of line in a field.
    assert alone.length == pytest.approx(100 * 100 / 10, rel=0.05)
    assert len(tool._hatch_lines(gpd.GeoSeries([], crs="EPSG:32613"), 10.0)) == 0


def test_scored_part_titles_counts_and_years_read_as_printed(tool):
    lay = tool.Layer("x.gpkg", "NAIP 1 m, 2022: DA v2 as released")
    entry = tool.Entry(
        "78", "Demo", "Demo", [lay], title_f1=True, scored_part=tool.NEVADA_IRRIGATED
    )
    record = {"predicted_scored": 145, "f1": 0.3804347826086957}
    assert tool._panel_title(entry, lay, 216, None, record) == (
        "NAIP 1 m, 2022: DA v2 as released\n145 of 216 fields scored; F1 0.38, irrigated only"
    )
    undefined = {"predicted_scored": 0, "f1": float("nan")}
    assert tool._panel_title(entry, lay, 1, None, undefined).endswith(
        "\n0 of 1 field scored, irrigated only"
    )
    # Without a record the title is the one-line title of the other entries.
    assert tool._panel_title(entry, lay, 216, {"f1": 0.255}) == (
        "NAIP 1 m, 2022: DA v2 as released (216 fields; F1 0.26)"
    )
    # A title of several lines fits when its widest line does.
    wide = "a title line that is much wider than its panel"
    assert tool._fit_title(f"short\n{wide}", 2.0) == tool._fit_title(wide, 2.0) < tool.TITLE_FS
    parts = [
        {"year": 2022, "reference_scored": 39, "reference_left_out": 35},
        {"year": 2022, "reference_scored": 39, "reference_left_out": 35},
        {"year": 2024, "reference_scored": 34, "reference_left_out": 40},
        {"year": 2023, "reference_scored": 40, "reference_left_out": 34},
    ]
    assert tool._scored_counts(parts) == "39 of 74 in 2022, 40 in 2023 and 34 in 2024"
    assert tool._or_join([2022]) == "2022" and tool._or_join(["a", "b"]) == "a or b"


def test_the_irrigated_only_panels_agree_with_the_scores_of_the_run(tool):
    """24c and 24d on the re-scoring run of example 24: each panel's split gives the counts of
    its metrics file (_scored_panel raises otherwise), in the years of NAIP, SPOT and
    Sentinel-1."""
    entries = {e.key: e for e in tool.ENTRIES}
    for key in ("24c", "24d"):
        entry = entries[key]
        root = tool._entry_root(entry)
        try:
            ref = tool._read_reference(entry.reference, root, None)
            files = [
                (tool._one(lay.path, root), tool._one(lay.metrics, root)) for lay in entry.layers
            ]
        except FileNotFoundError:
            pytest.skip("the re-scoring run of example 24 is not in this checkout")
        years = [
            tool._scored_panel(entry.scored_part, metrics, ref, gpd.read_file(path))["year"]
            for path, metrics in files
        ]
        assert years == [2022, 2022, 2023, 2024]


def test_rank_picks_are_evenly_spaced_from_the_lowest_to_the_highest(tool):
    scores = [(f"s{i}", v) for i, v in enumerate([0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8])]
    assert tool._rank_picks(scores, 4) == ["s1", "s3", "s4", "s2"]  # ranks 0, 2, 4, 6 of 7
    assert tool._rank_picks([("a", 0.5), ("b", 0.5), ("c", 0.1)], 2) == ["c", "b"]  # ties: order
    many = [(str(i), float(i)) for i in range(47)]
    assert tool._rank_picks(many, 4) == ["0", "15", "31", "46"]
    with pytest.raises(ValueError):
        tool._rank_picks(many[:3], 4)
    # The notes of entries 25 and 25b number the squares in that order, with these ranks.
    assert tool._rank_positions(47, 4) == [0, 15, 31, 46]
    assert tool._multistate_order("F1") == (
        "1-4 in increasing F1: the lowest, the two that cut the ranking into thirds, the "
        "highest (ranks 1, 16, 32 and 47 of 47)"
    )


def test_the_multistate_panels_follow_their_rule(tool):
    # 25 reads the summary of the 1.1 run and 25b that of its own run root (the final run,
    # with the zones): the rule picks the same squares in both, from the same NAIP rows.
    entries = {e.key: e for e in tool.ENTRIES}
    summaries = {tool._entry_root(entries[k]) / tool.MS / "summary.csv" for k in ("25", "25b")}
    summaries = sorted(s for s in summaries if s.exists())
    if not summaries:
        pytest.skip("the 1.1 run of example 25 is not in this checkout")
    tables = []
    for summary in summaries:
        rows = _summary_rows(summary)
        naip = [
            (r["state"], float(r["f1"]))
            for r in rows
            if r["imagery"] == "naip" and r["status"] == "ok"
        ]
        assert tool._rank_picks(naip, 4) == [state for state, *_ in tool.MULTISTATE_PANELS]
        assert len(naip) == tool.MULTISTATE_N_SQUARES  # the "of 47" of the notes
        tables.append(naip)
    assert all(t == tables[0] for t in tables)


def test_the_tree_crop_fine_tuning_notes_give_the_runs_chip_counts(tool):
    """23b and 23c print the training chips of the Delineate-Anything fine-tuning they draw (and
    the reference polygons behind it), checked against the chips the run recorded."""
    import glob

    entries = {e.key: e for e in tool.ENTRIES}
    for key, area in (("23b", "norpalm"), ("23c", "higaturu")):
        root = tool._entry_root(entries[key]) / tool.TC
        chips = glob.glob(
            str(
                root
                / area
                / ".agribound_cache"
                / "finetune_delineate-anything_*"
                / "finetune_data_delineate-anything_*"
                / "chips.gpkg"
            )
        )
        refs = glob.glob(str(root / "references" / f"{area}_rspo_*.gpkg"))
        if not chips or not refs:
            pytest.skip("the run of example 23 is not in this checkout")
        assert len(chips) == 1 and len(refs) == 1
        ref = gpd.read_file(refs[0])
        train = gpd.read_file(chips[0]).query("split == 'train'").to_crs(ref.crs)
        note = " ".join(entries[key].model_notes)
        assert f"{len(train)} training chips" in note
        if key == "23b":
            assert f"fine-tuned on {len(ref)} blocks of the NORPALM estate" in note
        else:  # the parcels with a part (a positive area) in a training chip
            held = int((ref.geometry.intersection(train.geometry.union_all()).area > 0).sum())
            assert f"holding parts of {held} of the square's {len(ref)} parcels" in note


def test_the_zones_flag_adds_the_cdl_zones_on_the_naip_image(tool):
    assert tool.MULTISTATE_ZONES is True  # the final run of example 25 was checked
    entry = next(e for e in tool.ENTRIES if e.key == "25b")
    assert entry == tool._multistate_squares(True)  # the gallery draws the zones
    plain, zones = tool._multistate_squares(False), tool._multistate_squares(True)
    assert len(plain.layers) == 4 and plain.run_root is None and plain.pred_label is None
    assert len(zones.layers) == 8 and zones.run_root == tool.FINAL_RUN_ROOT
    for naip, zone in zip(zones.layers[::2], zones.layers[1::2], strict=True):
        assert (zone.area, zone.reference) == (naip.area, naip.reference)
        assert zone.path.endswith("/zones_cdl-sequence_2016-2023.gpkg") and zone.noun == "zone"
        assert zone.background_from == naip.path  # the NAIP image, not the CDL stack
        assert zone.metrics.endswith("/metrics_zones_cdl-sequence_2016-2023.json")
    assert "zones" in zones.pred_label and "zones" in zones.bg_role and plain.bg_role is None


def test_panels_take_their_scores_from_the_metrics_file(tool, tmp_path):
    root = _run_root(tmp_path)
    demo = root / "outputs" / "demo"
    metrics = {
        "row": {"run": "naip da", "f1": 0.4567, "n": 3, "collects": "a b"},
        "metrics": {
            "f1": 0.1,
            "precision": 0.5,
            "per_stratum": {"pivot": {"f1": 0.2}},
            "bootstrap": {"n": 200, "f1": {"lo": 0.1}},
        },
    }
    (demo / "metrics_fields.json").write_text(json.dumps(metrics))
    record = tool._metrics_record(demo / "metrics_fields.json", root)
    assert record["file"] == "outputs/demo/metrics_fields.json"
    assert record["f1"] == 0.4567  # the summary row's F1
    assert record["row"]["collects"] == "a b" and record["scores"]["precision"] == 0.5
    assert record["scores"] == {"f1": 0.1, "precision": 0.5}  # METRICS_SCORES only
    # Example 27 writes one set of scores per boundary tolerance.
    per_tolerance = {"row": {}, "metrics": {"30": {"f1": 0.2, "boundary_f1": 0.7}, "60": {}}}
    (demo / "metrics_27.json").write_text(json.dumps(per_tolerance))
    record_27 = tool._metrics_record(demo / "metrics_27.json", root)
    assert record_27["f1"] is None and record_27["scores"] == {
        "30": {"f1": 0.2, "boundary_f1": 0.7}
    }

    lay = tool.Layer("outputs/demo/fields.gpkg", "Demo", metrics="outputs/demo/metrics_fields.json")
    entry = tool.Entry("99", "Demo_scores", "Demo", [lay], inset=False, title_f1=True)
    assert tool._panel_title(entry, lay, 2, record) == "Demo (2 fields; F1 0.46)"
    assert tool._panel_title(entry, lay, 2, {"f1": float("nan")}) == "Demo (2 fields)"
    zones = tool.Layer("outputs/demo/zones.gpkg", "Zones", noun="zone")
    assert tool._panel_title(entry, zones, 1, None) == "Zones (1 zone)"
    plain = tool.Entry("98", "Demo_plain", "Demo", [lay], inset=False)
    assert tool._panel_title(plain, lay, 2, record) == "Demo (2 fields)"
    stats = tool.render(entry, root, tmp_path / "gallery", "composite")
    assert stats["layers"][0]["metrics"]["f1"] == 0.4567
    assert tool._metrics_of("outputs/x/fields_naip_da_20??.gpkg") == (
        "outputs/x/metrics_fields_naip_da_20??.json"
    )
    assert tool._metrics_of("fields.gpkg") == "metrics_fields.json"  # not at the root "/"


def test_panels_on_a_common_background_name_their_own_input(tool, tmp_path):
    import shutil

    root = _run_root(tmp_path)
    demo = root / "outputs" / "demo"
    shutil.copy(demo / "composite.tif", demo / "radar.tif")
    shutil.copy(demo / "fields.gpkg", demo / "fields_radar.gpkg")
    sidecar = json.loads((demo / "fields.gpkg.provenance.json").read_text())
    sidecar["config"]["source"] = "sentinel1"
    sidecar["facts"] = {
        "raster_path": "outputs/demo/radar.tif",
        "composite": {
            "AGRIBOUND_RESOLUTION_M": "10",
            "AGRIBOUND_COMPOSITE_METHOD": "median",
            "AGRIBOUND_DATE_START": "2024-04-01",
            "AGRIBOUND_DATE_END_EXCLUSIVE": "2024-11-01",
            "AGRIBOUND_N_IMAGES": "36",
            "AGRIBOUND_ORBIT_PASS": "DESCENDING",
        },
    }
    (demo / "fields_radar.gpkg.provenance.json").write_text(json.dumps(sidecar))
    entry = tool.Entry(
        "99",
        "Demo_inputs",
        "Demo",
        [
            tool.Layer("outputs/demo/fields.gpkg", "A"),
            tool.Layer("outputs/demo/fields_radar.gpkg", "B"),
        ],
        background_from="outputs/demo/fields.gpkg",
        input_notes=True,
        inset=False,
    )
    stats = tool.render(entry, root, tmp_path / "gallery", "composite")
    note = stats["background_note"].replace("\n", " ")
    assert (
        "Delineated on: right, Sentinel-1 GRD descending (VV, VH, VV-VH) 10 m median composite, "
        "2024-04-01 to 2024-11-01, 36 acquisitions (end dates exclusive)"
    ) in note
    assert "left," not in note  # the left panel is drawn on its own input


def test_an_area_window_lies_in_the_imagery_of_every_panel_of_the_area(tool, tmp_path):
    """A 10 m grid can reach past a 1 m grid of the same area: the window is placed inside
    the imagery that all the area's panels cover, not only the first panel's."""
    root = _run_root(tmp_path)
    demo = root / "outputs" / "demo"
    # Panel 1: fields in the east only. Panel 2: imagery of the western 600 m only.
    gpd.GeoDataFrame(geometry=[box(500700, 3999500, 500950, 3999900)], crs="EPSG:32613").to_file(
        demo / "fields_east.gpkg", driver="GPKG"
    )
    (demo / "fields_east.gpkg.provenance.json").write_text(
        (demo / "fields.gpkg.provenance.json").read_text()
    )
    with rasterio.open(demo / "composite.tif") as src:
        profile = src.profile.copy()
        data = src.read(window=rasterio.windows.Window(0, 0, 60, 120))
        descriptions = src.descriptions
    profile.update(width=60)
    with rasterio.open(demo / "west.tif", "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = descriptions
    sidecar = json.loads((demo / "fields.gpkg.provenance.json").read_text())
    sidecar["facts"]["raster_path"] = "outputs/demo/west.tif"
    (demo / "fields_west.gpkg.provenance.json").write_text(json.dumps(sidecar))
    gpd.read_file(demo / "fields.gpkg").to_file(demo / "fields_west.gpkg", driver="GPKG")
    entry = tool.Entry(
        "88",
        "Demo_area_imagery",
        "Demo",
        [
            tool.Layer("outputs/demo/fields_east.gpkg", "A, 10 m", area="A"),
            tool.Layer("outputs/demo/fields_west.gpkg", "A, 1 m", area="A"),
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
        ncols=2,
        inset=False,
    )
    stats = tool.render_areas(entry, root, tmp_path / "gallery", "composite")
    for lay in stats["layers"]:
        x0, _y0, x1, _y1 = lay["window"]["bounds"]
        assert x0 >= 500000 and x1 <= 500600


def test_window_within_keeps_the_square_inside_the_polygon(tool):
    pts = gpd.GeoDataFrame(
        geometry=[box(x, 0, x + 1, 1) for x in (0, 1, 2)] + [box(60, 60, 61, 61)],
        crs="EPSG:32613",
    )
    record = {}
    free = tool._densest_window(pts, 10, (0, 100, 0, 100), record=record)
    assert free[0] <= 0.5 <= free[1]  # the three clustered fields
    assert record == {"candidates": 19 * 19}  # centres 5, 10, ..., 95: half a side apart
    within = box(40, 40, 100, 100)
    w = tool._densest_window(pts, 10, (0, 100, 0, 100), within, record)
    assert within.contains(box(w[0], w[2], w[1], w[3]))
    assert w[0] <= 60.5 <= w[1] and w[2] <= 60.5 <= w[3]  # the lone field inside it
    assert record == {"candidates": 11 * 11}  # the candidates inside it: centres 45 ... 95
    # No candidate square inside the polygon: the free choice, with a warning.
    with pytest.warns(UserWarning, match="lies inside the given polygon"):
        fallback = tool._densest_window(pts, 10, (0, 100, 0, 100), box(0, 0, 5, 5), record)
    assert fallback == free and record == {"candidates": 19 * 19}


def test_markers_are_moved_apart_only_where_they_would_overlap(tool):
    xy = np.array([[0.0, 0.0], [3.0, 1.0], [100.0, 100.0]])
    out = tool._spread_markers(xy, 10.0)
    assert np.abs(out[1] - out[0]).max() >= 10.0  # apart on one axis (square markers)
    assert (out[2] == xy[2]).all()  # a marker far from the others is not moved
    assert np.allclose(out[:2].mean(axis=0), xy[:2].mean(axis=0))  # pushed apart evenly
    assert (tool._spread_markers(xy, 10.0) == out).all()  # deterministic


@pytest.mark.parametrize(
    ("sources", "kind"),
    [
        ("CLU 100 %", "CLU"),
        ("CLU 80 %, NEDNR 20 %", "CLU"),
        ("IDWRSP 92 %, CLU 8 %", "agency"),
        ("CADWR_2023prov 100 %", "agency"),
        ("V18 100 %", "DRI"),
        ("DRI 100 %", "DRI"),
    ],
)
def test_overview_markers_are_shaped_by_the_main_reference_source(tool, sources, kind):
    assert tool._reference_kind(sources) == kind


def _fake_conus(tool, monkeypatch):
    """Stand-ins for Natural Earth around the conterminous US (no network)."""
    countries = gpd.GeoDataFrame(
        {"ADM0_A3": ["USA", "MEX", "CAN"]},
        geometry=[box(-125, 25, -66, 49), box(-117, 15, -87, 31.3), box(-130, 49, -60, 60)],
        crs=4326,
    )
    states = gpd.GeoDataFrame(
        {
            "name": ["New Mexico", "Texas", "Alabama", "Alaska"],
            "postal": ["NM", "TX", "AL", "AK"],
            "adm0_a3": ["USA", "USA", "USA", "USA"],
        },
        geometry=[
            box(-109.05, 31.33, -103.0, 37.0),
            box(-106.6, 25.8, -93.5, 36.5),
            box(-88.5, 30.2, -84.9, 35.0),
            box(-170, 52, -130, 71),
        ],
        crs=4326,
    )
    lakes = gpd.GeoDataFrame(geometry=[box(-100, 32, -99, 33)], crs=4326)
    layers = {"countries": countries, "states": states, "lakes": lakes}
    monkeypatch.setattr(tool, "_natural_earth", lambda layer: layers[layer])


def _overview_root(tmp_path: Path) -> Path:
    import pandas as pd

    root = _run_root(tmp_path)
    common = {"status": "ok", "year": 2022, "output": "outputs/demo/fields.gpkg"}
    pd.DataFrame(
        [
            {"state": "NM", "imagery": "naip", "f1": 0.2, "sources": "V18 100 %", **common},
            {"state": "TX", "imagery": "naip", "f1": 0.6, "sources": "CLU 100 %", **common},
            {"state": "TX", "imagery": "sentinel2", "f1": 0.9, "sources": "CLU 100 %", **common},
        ]
    ).assign(
        reference="Demo reference",
        centre_lon=[-106.72, -106.70, -106.70],  # two squares close together
        centre_lat=[34.76, 34.70, 34.70],
    ).to_csv(root / "outputs" / "demo" / "summary.csv", index=False)
    return root


def test_the_overview_maps_one_marker_per_square(tool, tmp_path, monkeypatch):
    import dataclasses

    _fake_conus(tool, monkeypatch)
    root = _overview_root(tmp_path)
    entry = tool.Entry(
        "97",
        "Demo_overview_map",
        "Demo overview",
        [],
        overview=tool.Overview(
            summary="outputs/demo/summary.csv",
            run="naip",
            metric_label="Object F1",
            marks=["TX"],
            notes=["Demo note"],
        ),
        inset=False,
    )
    stats = tool.render_overview(entry, root, tmp_path / "gallery", "composite")
    assert (tmp_path / "gallery" / "Demo_overview_map.png").exists()
    assert (tmp_path / "gallery" / "preview" / "Demo_overview_map.webp").exists()
    assert stats["n_squares"] == 2 and stats["no_square"] == ["Alabama"]  # Alaska: not CONUS
    assert [s["state"] for s in stats["squares"]] == ["NM", "TX"]  # the naip rows only
    assert [s["kind"] for s in stats["squares"]] == ["DRI", "CLU"]
    assert stats["marks"] == [{"n": 1, "state": "TX", "f1": 0.6}]
    assert stats["lowest"] == {"state": "NM", "f1": 0.2}
    assert stats["highest"] == {"state": "TX", "f1": 0.6}
    assert all(s["moved_on_map_km"] > 0 for s in stats["squares"])  # 7 km apart: both moved
    note = stats["background_note"].replace("\n", " ")
    assert note.startswith("2 squares, one per state (no square: Alabama); F1 median 0.40")
    assert "2 markers moved apart" in note and "Demo note" in note
    assert stats["map_title"] == "Demo overview"  # no panel label: the entry's title
    # With a panel label the map has a panel title, and the entry's title is the caption.
    labelled = dataclasses.replace(
        entry, overview=dataclasses.replace(entry.overview, panel_label="NAIP 1 m: Demo")
    )
    stats = tool.render_overview(labelled, root, tmp_path / "gallery", "composite")
    assert stats["map_title"] == "NAIP 1 m: Demo (2 squares; median F1 0.40)"
    assert stats["title"] == "Demo overview"
    # The numbered markers' legend label does not depend on the companion's panel count.
    assert tool._marks_label(4) == "Squares 1-4, drawn in the companion image"
    assert tool._marks_label(1) == "Square 1, drawn in the companion image"
    # An overview has no layers (and no locator); map panels need layers.
    with_layer = dataclasses.replace(entry, layers=[tool.Layer("outputs/demo/fields.gpkg", "A")])
    with pytest.raises(ValueError, match="no layers"):
        tool.render_overview(with_layer, root, tmp_path / "other", "composite")
    with pytest.raises(ValueError, match="map panels need layers"):
        tool.render(entry, root, tmp_path / "other", "composite")
    with pytest.raises(ValueError, match="its own locator"):
        tool.render_overview(dataclasses.replace(entry, inset=True), root, tmp_path / "x", "x")


def test_overviews_never_draw_a_restricted_reference_or_place_a_local_square(guarded, tmp_path):
    root = _run_root(tmp_path)
    for summary, match in (
        (f"{R_OUT}/references/summary.csv", "restricted reference"),
        (f"{L_OUT}/summary.csv", "stays local"),
    ):
        overview = guarded.Overview(summary=summary, run="naip", metric_label="F1")
        entry = guarded.Entry("97", "Demo_overview", "Demo", [], overview=overview, inset=False)
        with pytest.raises(ValueError, match=match):
            guarded.render_overview(entry, root, tmp_path / "gallery", "composite")
    assert not (tmp_path / "gallery").exists()


def test_main_reads_each_entry_from_its_own_run_root(tool, tmp_path, monkeypatch):
    root = _run_root(tmp_path)
    entry = tool.Entry(
        "99",
        "Demo_own_root",
        "Demo",
        [tool.Layer("outputs/demo/fields.gpkg", "Demo")],
        inset=False,
        run_root=str(root),  # absolute here; relative run roots are under the repository
    )
    assert tool._entry_root(entry) == root
    relative = tool.Entry("98", "Demo_sar", "Demo", [], run_root="outputs/examples_1.1_sar")
    assert tool._entry_root(relative) == tool.REPO / "outputs" / "examples_1.1_sar"
    monkeypatch.setattr(tool, "ENTRIES", [entry])
    out = tmp_path / "gallery"
    assert tool.main(["--out-dir", str(out), "--no-proj-network"]) == 0
    assert (out / "Demo_own_root.png").exists()
    # An explicit --run-root replaces it (and the outputs are not there).
    elsewhere = ["--run-root", str(tmp_path / "elsewhere"), "--out-dir", str(out)]
    assert tool.main([*elsewhere, "--no-proj-network"]) == 1


def test_notes_keep_numbers_with_the_next_word(tool):
    text = (
        "a 5 km square, F1 from 0.01 (MT) to 0.70 (CA); 3: 2023-01-01 to 2024-01-01, "
        "(© Capella Space Corp, CC BY 4.0, modified) 2,500 m²"
    )
    for width in (0.6, 1.0, 1.5, 2.5, 4.0):
        lines = tool._wrap(text, width)
        assert " ".join(lines) == text
        assert not any(tool._NO_BREAK_AFTER.match(line.split()[-1]) for line in lines[:-1])
    assert tool._wrap("one word two", 0.3) == ["one word", "two"]  # a no-break space


def test_crop_sequence_zones_are_named_in_the_model_note(tool):
    prov = {
        "config": {"engine": "crop-sequence", "source": "aci-sequence"},
        "engine_meta": {
            "implementation": "agribound-crop-sequence-1",
            "dataset": "aci",
            "years": list(range(2016, 2024)),
            "min_crop_years": 5,
            "smooth_radius_px": 3,
            "pixel_size_m": 30.0,
            "params": {"min_region_m2": 10000.0},
        },
    }
    assert tool._model_note(prov) == (
        "Crop-history zones (agribound-crop-sequence-1): AAFC Annual Crop Inventory 2016-2023 "
        "at 30 m, cultivated in at least 5 of 8 years, focal mode of radius 3 px, regions "
        "under 1 ha merged; zones of equal crop sequence"
    )


def test_zone_panels_are_drawn_on_the_image_of_their_square(tool, tmp_path):
    """Entry 27 and entry 25b with zones: the zones' own raster is a stack of crop classes,
    so they are drawn on the imagery of the Delineate-Anything output of the same square."""
    root = _run_root(tmp_path)
    demo = root / "outputs" / "demo"
    gpd.read_file(demo / "fields.gpkg").to_file(demo / "zones.gpkg", driver="GPKG")
    zones_sidecar = {
        "status": "success",
        "config": {"source": "aci-sequence", "engine": "crop-sequence", "year": 2023},
        "facts": {"raster_path": "outputs/demo/no_such_stack.tif"},  # never read
        "engine_meta": {
            "implementation": "agribound-crop-sequence-1",
            "dataset": "aci",
            "years": [2022, 2023],
            "min_crop_years": 2,
            "smooth_radius_px": 3,
            "pixel_size_m": 30.0,
            "params": {"min_region_m2": 10000.0},
        },
    }
    (demo / "zones.gpkg.provenance.json").write_text(json.dumps(zones_sidecar))
    common = {"area": "A", "reference": "outputs/demo/reference.gpkg"}
    entry = tool.Entry(
        "87",
        "Demo_zones",
        "Demo",
        [
            tool.Layer(
                "outputs/demo/zones.gpkg",
                "Zones",
                noun="zone",
                background_from="outputs/demo/fields.gpkg",
                **common,
            ),
            tool.Layer("outputs/demo/fields.gpkg", "Fields", **common),
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
        ncols=2,
        inset=False,
        bg_role="the input of the fields panel, also under the zones",
        pred_label="fields or zones",
    )
    stats = tool.render_areas(entry, root, tmp_path / "gallery", "composite")
    zones, fields = stats["layers"]
    assert zones["engine"] == "crop-sequence" and zones["window"] == fields["window"]
    note = stats["background_note"].replace("\n", " ")
    # One role for both panels: the square's image is listed once.
    assert note.startswith(
        "Background: Sentinel-2 L2A median composites (the input of the fields panel, also "
        "under the zones); end dates exclusive: 1: 2024-01-01 to 2025-01-01, 7 images"
    )
    assert (
        "Crop-history zones (agribound-crop-sequence-1): AAFC Annual Crop Inventory 2022-2023"
        in note
    )


# --------------------------------------------------------------------------- imagery entries


def _imagery_roots(tmp_path: Path) -> tuple[Path, Path]:
    """The synthetic run root, and a second root holding an older RGB image of the same area
    with the record of tools/fetch_naip_history.py (as for entry 25c's NAIP 2005 image)."""
    import hashlib

    root = _run_root(tmp_path)
    other = tmp_path / "r3"
    folder = other / "history"
    folder.mkdir(parents=True)
    tif = folder / "old.tif"
    rng = np.random.default_rng(1)
    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        width=100,
        height=120,
        count=3,
        dtype="uint8",
        crs="EPSG:32613",
        transform=rasterio.transform.from_origin(500000, 4000000, 10, 10),
        nodata=0,
    ) as dst:
        dst.write(rng.integers(1, 255, size=(3, 120, 100)).astype("uint8"))
        dst.descriptions = ("R", "G", "B")
        dst.update_tags(
            AGRIBOUND_COLLECTIONS="USDA/NAIP/DOQQ",
            AGRIBOUND_COMPOSITE_METHOD="mosaic",
            AGRIBOUND_DATE_START="2005-01-01",
            AGRIBOUND_DATE_END_EXCLUSIVE="2006-01-01",
            AGRIBOUND_N_IMAGES="2",
            AGRIBOUND_RESOLUTION_M="10.0",
        )
    record = {
        "collection": "USDA/NAIP/DOQQ",
        "year": 2005,
        "images": [
            {"asset_id": "USDA/NAIP/DOQQ/a", "date": "2005-07-28", "native_resolution_m": 1.0},
            {"asset_id": "USDA/NAIP/DOQQ/b", "date": "2005-07-28", "native_resolution_m": 1.0},
        ],
        "bands": ["R", "G", "B"],
        "export": {"aggregation": "mean of the native pixels"},
        "output": {"file": tif.name, "sha256": hashlib.sha256(tif.read_bytes()).hexdigest()},
        "credit": "NAIP: USDA Farm Service Agency",
    }
    (folder / "old.json").write_text(json.dumps(record))
    return root, other


def _imagery_entry(tool, root: Path, other: Path, **kw):
    return tool.Entry(
        "86",
        "Demo_imagery",
        "Demo imagery",
        [],
        reference="outputs/demo/reference.gpkg",
        reference_label="Demo reference",
        window=(500500.0, 3999400.0, 600.0),
        ncols=2,
        inset=False,
        run_root=str(root),  # absolute here; relative run roots are under the repository
        imagery=tool.Imagery(
            panels=[
                tool.ImagePanel(
                    "Old image",
                    background="history/old.tif",
                    background_source="naip",
                    bg_role="exported for this test",
                    record="history/old.json",
                    run_root=str(other),
                ),
                tool.ImagePanel(
                    "The engine's image",
                    background_from="outputs/demo/fields.gpkg",
                    bg_role="the engine's input",
                ),
            ],
            notes=["No model output", "Credits: demo"],
        ),
        **kw,
    )


def test_imagery_panels_draw_the_reference_on_each_image_in_one_window(tool, tmp_path, monkeypatch):
    root, other = _imagery_roots(tmp_path)
    entry = _imagery_entry(tool, root, other)
    reads = []
    real = tool._read_background

    def spy(tifs, source, bounds, crs, out_px, stretch=None):
        reads.append((tifs, source, bounds))
        return real(tifs, source, bounds, crs, out_px, stretch)

    monkeypatch.setattr(tool, "_read_background", spy)
    plotted = []
    real_plot = gpd.GeoSeries.plot

    def plot_spy(self, *args, **kwargs):
        plotted.append(kwargs.get("color"))
        return real_plot(self, *args, **kwargs)

    monkeypatch.setattr(gpd.GeoSeries, "plot", plot_spy)
    stats = tool.render_imagery(entry, root, tmp_path / "gallery", "composite")
    assert (tmp_path / "gallery" / "Demo_imagery.png").exists()
    assert (tmp_path / "gallery" / "preview" / "Demo_imagery.webp").exists()
    # Each panel on its own image (the second from the output's provenance), in the window.
    window = (500200.0, 3999100.0, 500800.0, 3999700.0)
    assert [(t, s, b) for t, s, b in reads] == [
        ([other / "history" / "old.tif"], "naip", window),
        ([root / "outputs" / "demo" / "composite.tif"], "sentinel2", window),
    ]
    # The reference only: cyan outlines, no model output in red.
    assert plotted == [tool.REF_COLOR, tool.REF_COLOR]
    assert stats["reference"]["n_polygons"] == 1
    assert stats["reference"]["legend"] == "Demo reference (1 field)"
    assert stats["window"] == {
        "crs": "EPSG:32613",
        "centre": [500500.0, 3999400.0],
        "side_m": 600.0,
    }
    old, new = stats["panels"]
    assert (old["panel"], old["label"], old["file"]) == ("Left", "Old image", "history/old.tif")
    assert old["run_root"] == other.as_posix() and new["run_root"] == root.as_posix()
    assert old["record"]["images"] == [
        ["USDA/NAIP/DOQQ/a", "2005-07-28"],
        ["USDA/NAIP/DOQQ/b", "2005-07-28"],
    ]
    assert old["record"]["native_resolution_m"] == [1.0]
    assert old["record"]["bands"] == ["R", "G", "B"] and old["composite"]["n_images"] == "2"
    assert new["from"] == "outputs/demo/fields.gpkg" and "record" not in new
    assert new["composite"]["date_start"] == "2024-01-01"
    note = stats["background_note"].replace("\n", " ")
    assert note.startswith(
        "Background: Left: NAIP 10 m mosaic composite, 2005-01-01 to 2006-01-01 (end "
        "exclusive), 2 images of 2005-07-28; exported for this test Right: Sentinel-2 L2A "
        "median composite, 2024-01-01 to 2025-01-01 (end exclusive), 7 images; the engine's "
        "input No model output Credits: demo"
    )
    assert stats["notes"] == ["No model output", "Credits: demo"]
    assert "layers" not in stats  # no model output


def test_imagery_entries_refuse_what_they_do_not_draw(tool, tmp_path):
    import dataclasses
    import shutil

    root, other = _imagery_roots(tmp_path)
    entry = _imagery_entry(tool, root, other)
    out = tmp_path / "gallery"
    panels = entry.imagery.panels
    cases = [
        (
            dataclasses.replace(entry, layers=[tool.Layer("outputs/demo/fields.gpkg", "A")]),
            "no layers",
        ),
        (dataclasses.replace(entry, window=None), "Entry.window"),
        (dataclasses.replace(entry, reference=None), "reference"),
        (dataclasses.replace(entry, zoom_m=100.0), "do not support zoom_m"),
        (dataclasses.replace(entry, model_notes=["x"]), "do not support model_notes"),
        (dataclasses.replace(entry, bg_role="x"), "do not support bg_role"),
        (dataclasses.replace(entry, copernicus_credit=True), "copernicus_credit"),
        (dataclasses.replace(entry, window=(500950.0, 3999900.0, 400.0)), "not inside the image"),
        (
            dataclasses.replace(
                entry,
                imagery=tool.Imagery(
                    [dataclasses.replace(panels[0], background_from="outputs/demo/fields.gpkg")]
                ),
            ),
            "one of background and background_from",
        ),
    ]
    for bad, match in cases:
        with pytest.raises(ValueError, match=match):
            tool.render_imagery(bad, root, out, "composite")
    with pytest.raises(ValueError, match="composite"):
        tool.render_imagery(entry, root, out, "esri")
    # The other renderers refuse an imagery entry.
    with pytest.raises(ValueError, match="map panels need layers"):
        tool.render(entry, root, out, "composite")
    with pytest.raises(ValueError, match="imagery"):
        tool.render_areas(dataclasses.replace(entry, crop_m=100.0), root, out, "composite")
    with pytest.raises(ValueError, match="an overview entry"):
        tool.render_overview(entry, root, out, "composite")
    # The record must describe the file drawn.
    record = json.loads((other / "history" / "old.json").read_text())
    record["output"]["sha256"] = "0" * 64
    (other / "history" / "old.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="but the panel draws 'old.tif'"):
        tool.render_imagery(entry, root, out, "composite")
    # Images in two CRSs are refused (no image is reprojected).
    shutil.copy(other / "history" / "old.tif", other / "history" / "old_32612.tif")
    with rasterio.open(other / "history" / "old_32612.tif", "r+") as dst:
        dst.crs = "EPSG:32612"
    moved = dataclasses.replace(panels[0], background="history/old_32612.tif", record=None)
    two_crs = dataclasses.replace(entry, imagery=tool.Imagery([moved, panels[1]]))
    with pytest.raises(ValueError, match="one CRS"):
        tool.render_imagery(two_crs, root, out, "composite")
    assert not out.exists()


def test_an_imagery_panel_reads_its_own_run_root_unless_one_is_given(tool):
    entry = tool.Entry("85", "Demo", "Demo", [], run_root="outputs/examples_1.1")
    own = tool.ImagePanel("A", background="a.tif", run_root="outputs/examples_1.1_r3")
    plain = tool.ImagePanel("B", background="b.tif")
    root = tool._entry_root(entry)
    assert tool._panel_root(entry, own, root) == tool.REPO / "outputs" / "examples_1.1_r3"
    assert tool._panel_root(entry, plain, root) == root
    elsewhere = Path("outputs/examples_1.0")  # an explicit --run-root replaces both
    assert tool._panel_root(entry, own, elsewhere) == elsewhere
    assert tool._panel_root(entry, plain, elsewhere) == elsewhere


@pytest.mark.parametrize(
    ("panel_kw", "match"),
    [
        ({"background": "examples/Demo_Restricted_Fields/old.tif"}, "restricted reference"),
        ({"record": f"{R_OUT}/references/old.json"}, "restricted reference"),
        ({"background": f"{L_OUT}/old.tif"}, "stays local"),
    ],
)
def test_imagery_entries_never_draw_a_restricted_reference_or_place_a_local_square(
    guarded, tmp_path, panel_kw, match
):
    import dataclasses

    root, other = _imagery_roots(tmp_path)
    entry = _imagery_entry(guarded, root, other)
    panel = dataclasses.replace(entry.imagery.panels[0], **panel_kw)
    bad = dataclasses.replace(entry, imagery=guarded.Imagery([panel, entry.imagery.panels[1]]))
    with pytest.raises(ValueError, match=match):
        guarded.render_imagery(bad, root, tmp_path / "gallery", "composite")
    assert not (tmp_path / "gallery").exists()


def test_main_renders_and_lists_imagery_entries(tool, tmp_path, monkeypatch, capsys):
    root, other = _imagery_roots(tmp_path)
    monkeypatch.setattr(tool, "ENTRIES", [_imagery_entry(tool, root, other)])
    out = tmp_path / "gallery"
    assert tool.main(["--out-dir", str(out), "--no-proj-network"]) == 0
    assert (out / "Demo_imagery.png").exists()
    stats = json.loads((out / "gallery_stats.json").read_text())
    assert [p["label"] for p in stats["86"]["panels"]] == ["Old image", "The engine's image"]
    capsys.readouterr()
    assert tool.main(["--list"]) == 0
    listing = capsys.readouterr().out
    assert f"history/old.tif (run root: {other})" in listing
    assert "outputs/demo/fields.gpkg" in listing


def test_entry_25c_draws_the_montana_square_on_naip_2005_and_2023(tool):
    """Entry 25c: example 25's Montana square with its reference, no model output; the NAIP
    2005 image (tools/fetch_naip_history.py) from its own run root, the rest from the 1.1 run."""
    import csv

    entry = next(e for e in tool.ENTRIES if e.key == "25c")
    assert entry.title == "Montana: the CLU on NAIP 2005 and 2023" and not entry.layers
    assert tool._entry_root(entry) == tool.RUN_ROOT  # the reference and the 2023 composite
    x, y = tool.MONTANA_SQUARE
    assert entry.window == (x + 2500.0, y + 2500.0, 5000.0)  # the 5 km square
    assert entry.reference == f"{tool.MS}/mt/reference.gpkg"
    assert "Common Land Units" in entry.reference_label and not entry.copernicus_credit
    old, new = entry.imagery.panels
    assert (old.run_root, tool.R3_RUN_ROOT) == ("outputs/examples_1.1_r3",) * 2
    assert old.background == "montana/naip_2005_rgb_2m.tif"
    assert old.record == "montana/naip_2005_rgb_2m.json" and old.background_source == "naip"
    assert new.background_from == f"{tool.MS}/mt/fields_naip_da_2023.gpkg"
    assert new.run_root is None and new.record is None
    notes = " ".join(entry.imagery.notes)
    assert notes.startswith("No model output.")
    assert "NAIP: USDA Farm Service Agency" in notes and tool.OPENET_CREDIT in notes
    assert "Common Land Units (CLU) of the USDA Farm Service Agency" in notes
    # The square and the 2023 flight date are example 25's (its docstring's table).
    text = (tool.REPO / "examples" / "25_multistate_openet_field_boundaries.py").read_text()
    row = next(line for line in text.splitlines() if line.strip().startswith("MT  12N"))
    assert f"{x}, {y}" in row and "2023-07-22" in row and "2023-07-22" in new.bg_role
    summary = tool._entry_root(entry) / tool.MS / "summary.csv"
    if summary.exists():
        with open(summary, newline="") as f:
            mt = [r for r in csv.DictReader(f) if r["state"] == "MT" and r["imagery"] == "naip"]
        assert [(int(r["square_x"]), int(r["square_y"]), r["year"]) for r in mt] == [(x, y, "2023")]
    image = tool.REPO / tool.R3_RUN_ROOT / old.background
    if not image.exists():
        pytest.skip("the NAIP 2005 export of entry 25c is not in this checkout")
    record = tool._image_record(old, tool.REPO / tool.R3_RUN_ROOT, image)  # it matches the file
    assert {date for _id, date in record["images"]} == {"2005-07-28"}
    assert record["native_resolution_m"] == [1.0] and record["bands"] == ["R", "G", "B"]
    assert "native 1 m pixels" in old.bg_role and "R, G and B" in old.bg_role
    prov = json.loads(
        (tool._entry_root(entry) / f"{new.background_from}.provenance.json").read_text()
    )
    assert (tool._entry_root(entry) / prov["facts"]["raster_path"]).exists()


# --------------------------------------------------------------------------- 1.1.0 entries
# 24e (the Diamond Valley options), the SAM entry and 28-28c (New York), and the renderer's
# options they use: Layer.window, Layer.run_root, Layer.metrics_record, an area's reference
# for its panels that name none, titles of two lines in multi-area figures, the sam-auto model
# and credit lines, and NAIP mosaics with images of other years underneath.


def _signed(value: float) -> str:
    """A change as the footers print it: "+0.167", "−0.012" (U+2212), three decimals."""
    return f"{float(value):+.3f}".replace("-", "−")


def test_the_nevada_notes_name_the_fine_tuned_and_the_sam_auto_panels(tool):
    """The fine-tuning and sam-auto lines of 24-24e follow the panels drawn."""
    dv = tool._nevada_notes("diamond_valley", tool.NEVADA_PANELS["diamond_valley"])
    assert len(dv) == 1 and dv[0].startswith("sam-auto (top right and bottom left): ")
    assert "5.0 km south of this square" in dv[0] and "72 settings per source" in dv[0]
    assert "held out from that choice, not from the landscape" in dv[0]
    # The selection rule, not "the best mean F1": for SPOT 6/7 ten settings tied within 0.01
    # and boundary F1 chose one of mean F1 0.578, below the best of the 72, 0.587 (round-2
    # sam-direct DESIGN 5.1 and 6.1, runs/selection.json).
    assert "by mean F1 among 72 settings per source; settings within 0.01 of the highest " in dv[0]
    printed = tool._plain(dv[0])  # "boundary F1" kept on one footer line (a no-break space)
    assert "then the fewest SAM passes per km², then the higher boundary F1" in printed
    assert "best" not in dv[0]
    mv = tool._nevada_notes("mason_valley", tool.NEVADA_PANELS["mason_valley"])
    assert mv[0] == tool.NEVADA_FINE_TUNE_NOTE.format(
        sources="NAIP (top right)", where=tool.NEVADA_FINE_TUNE_WHERE["mason_valley"]
    )
    assert mv[1].startswith("sam-auto (bottom left): ") and "16.0 km north" in mv[1]
    # The panels of the 1.1 gallery give the line it printed.
    old = ["naip da", "naip dinov3-ft", "spot dinov3-ft", "sentinel1 da"]
    assert tool._nevada_notes("diamond_valley", old) == [
        "Fine-tuned: DINOv3 ViT-L/16 (SAT-493M weights), full fine-tuning for up to 20 epochs "
        "on the NAIP (top right) or SPOT (bottom left) composite of a 5 km training square of "
        "the basin 5.0 km south of the evaluated one (54 DRI fields), never on the evaluated "
        "fields"
    ]
    plain = ["naip da", "naip da-sam2", "spot da", "sentinel1 da"]
    assert tool._nevada_notes("diamond_valley", plain) == []
    with pytest.raises(ValueError, match="no fine-tuning line"):  # the line is DINOv3's only
        tool._nevada_notes("diamond_valley", ["naip da", "naip da-ft", "spot da", "sentinel1 da"])


def test_the_nevada_titles_name_the_kinds_of_runs_drawn(tool):
    """The titles of 24-24d (gallery_stats.json, --list) name the kinds of runs their panels
    draw, in the style of 28-28c: Diamond Valley draws no fine-tuned model."""
    kinds_of = {"da": "released", "sam-auto": "sam-auto"}
    entries = {e.key: e for e in tool.ENTRIES}
    for key, basin in (
        ("24", "diamond_valley"),
        ("24b", "mason_valley"),
        ("24c", "diamond_valley"),
        ("24d", "mason_valley"),
    ):
        drawn = set()
        for run in tool.NEVADA_PANELS[basin]:
            label = run.split(" ", 1)[1]
            drawn.add("fine-tuned" if label.endswith("-ft") else kinds_of[label])
        kinds = [k for k in ("released", "fine-tuned", "sam-auto") if k in drawn]
        title = entries[key].title
        assert f": {tool._and_join(kinds)} runs on NAIP, SPOT and Sentinel-1" in title, key
        assert ("fine-tuned" in title) == ("fine-tuned" in drawn), key
        assert "models" not in title, key


def test_entry_24e_draws_the_panels_fixed_before_the_test(tool):
    """24e (PREREG-DV3 section 7): Diamond Valley's NAIP Delineate-Anything v2 as released
    and SPOT DINOv3 fine-tuned as in 1.1, beside the same runs with the options that carried
    (the crop-filter layers of example 24's "-opt" runs, which its metrics files score); the
    footer's numbers are those of the run's options.csv."""
    entries = {e.key: e for e in tool.ENTRIES}
    entry, nevada = entries["24e"], entries["24"]
    folder = f"{tool.NV}/diamond_valley"
    assert [lay.path for lay in entry.layers] == [
        f"{folder}/fields_naip_da_2022.gpkg",
        f"{folder}/fields_naip_da-opt_2022_crop_t0.3.gpkg",
        f"{folder}/fields_spot_dinov3-ft_2023.gpkg",
        f"{folder}/fields_spot_dinov3-ft-opt_2023_crop_t0.3.gpkg",
    ]
    for lay in entry.layers[1::2]:  # the "-opt" results: no record of their own
        output = lay.path.replace("_crop_t0.3", "")
        assert lay.model_from == lay.background_from == output
        assert lay.metrics == tool._metrics_of(output)
        assert lay.label.count("\n") == 1  # the run, then its options
    for lay in entry.layers[::2]:
        assert lay.metrics == tool._metrics_of(lay.path) and lay.model_from is None
    assert entry.layers[0] == nevada.layers[0]  # entry 24's top left
    assert (entry.window, entry.reference, entry.reference_label, entry.run_root) == (
        nevada.window,
        nevada.reference,
        nevada.reference_label,
        tool.EX24_RUN_ROOT,
    )
    assert not entry.copernicus_credit and entry.title_f1 and entry.model_positions
    assert entry.model_notes[:4] == [
        tool.NEVADA_OPTIONS_NOTE,
        tool.NEVADA_OPTIONS_CHANGES,
        tool.NEVADA_OPTIONS_TESTED,
        tool.NEVADA_OPTIONS_MASON,
    ]
    assert entry.model_notes[4] == tool.NEVADA_FINE_TUNE_NOTE.format(
        sources="SPOT (bottom left and bottom right)",
        where=tool.NEVADA_FINE_TUNE_WHERE["diamond_valley"],
    )
    tested = tool.NEVADA_OPTIONS_TESTED
    assert tested.startswith("Reported results, not tests") and "amended 2026-10-13" in tested
    # PREREG-DV3 section 7's label; for the 3 m grid only "partly" holds: example 26's run at
    # 3 m differed (merge_tile_pieces=False; +0.046 against the option's +0.056).
    assert tested.endswith("were partly known before the test")
    assert "example 26's run at 3 m, made with merge_tile_pieces=False" in tested
    root = tool.REPO / tool.EX24_RUN_ROOT / tool.NV
    if not (root / "options.csv").exists():
        pytest.skip("example 24's run with --options is not in this checkout")
    rows = {(r["basin"], r["run"]): r for r in _summary_rows(root / "options.csv")}
    naip, spot = rows["diamond_valley", "naip da-opt"], rows["diamond_valley", "spot dinov3-ft-opt"]
    assert (naip["base"], spot["base"]) == ("naip da", "spot dinov3-ft")  # the left panels
    for name, row in (("NAIP", naip), ("SPOT", spot)):
        for what, key in (
            ("F1", "d_f1"),
            ("irrigated-only F1", "d_f1_irrigated"),
            ("boundary F1", "d_boundary_f1"),
        ):
            change = (
                f"{_signed(row[key])} [{_signed(row[key + '_ci_low'])}, "
                f"{_signed(row[key + '_ci_high'])}]"
            )
            assert change in tool.NEVADA_OPTIONS_CHANGES, (name, what)
    assert f'half_band": {float(spot["growth_px"]):.0f} px here' in tool.NEVADA_OPTIONS_NOTE
    # The pivot counts and the trained width, from the runs' provenance records.
    facts = {}
    for lay in entry.layers[1::2]:
        prov = json.loads((root.parent.parent / f"{lay.model_from}.provenance.json").read_text())
        facts[lay.model_from] = (prov["facts"]["pivots"], prov["engine_meta"])
    naip_pivots, _ = facts[entry.layers[1].model_from]
    spot_pivots, spot_meta = facts[entry.layers[3].model_from]
    assert (
        f"here {naip_pivots['n_groups']} pivots, {naip_pivots['n_groups_multi']} of them "
        f"rejoined from {naip_pivots['n_pieces_merged']} pieces"
    ) in tool.NEVADA_OPTIONS_NOTE
    assert spot_pivots["n_groups_multi"] == 0
    assert f"pivots snapped ({spot_pivots['n_groups']}, none rejoined)" in tool.NEVADA_OPTIONS_NOTE
    assert spot_meta["dilate_interior_px"] == float(spot["growth_px"])
    assert f"against the {spot_meta['trained_dilate_px']} px it was trained with" in (
        tool.NEVADA_OPTIONS_NOTE
    )
    mason = tool.NEVADA_OPTIONS_MASON
    for run in ("naip da-opt", "naip dinov3-ft-opt", "spot da-opt"):
        row = rows["mason_valley", run]
        assert f"{float(row['f1_base']):.3f} to {float(row['f1']):.3f}" in mason, run
        assert (
            f"{_signed(row['d_f1'])} [{_signed(row['d_f1_ci_low'])}, "
            f"{_signed(row['d_f1_ci_high'])}]"
        ) in mason, run
    s1 = rows["mason_valley", "sentinel1 da-opt"]
    assert s1["f1"] == s1["f1_base"] and f"{float(s1['f1']):.3f} unchanged" in mason
    growth = float(rows["mason_valley", "naip dinov3-ft-opt"]["growth_px"])
    assert f"the growth measured {growth:.0f} px" in mason
    assert rows["mason_valley", "naip dinov3-ft-opt"]["pivot_groups_multi"] == "0"
    # Mason Valley's "-opt" runs (PREREG-DV3 section 7), not 24e's panel runs; where the pivot
    # option found no pivot the footer says so, where it snapped some it says "pivots snapped".
    assert mason.startswith('Mason Valley (no figure; example 24\'s "-opt" runs on its other ')
    mv_runs = sorted(run for basin, run in rows if basin == "mason_valley")
    assert mv_runs == ["naip da-opt", "naip dinov3-ft-opt", "sentinel1 da-opt", "spot da-opt"]
    for run, words in (
        ("naip dinov3-ft-opt", "the pivot step found no pivot, so the change is the crop"),
        ("sentinel1 da-opt", "Sentinel-1 (pivot option; no pivot found)"),
    ):
        assert rows["mason_valley", run]["pivot_groups"] == "0" and words in mason, run
    assert int(rows["mason_valley", "spot da-opt"]["pivot_groups"]) > 0
    assert "SPOT 6/7 (pivots snapped, crop filter;" in mason
    # Each right panel's metrics file scores the crop-filter layer drawn.
    import geopandas as gpd

    for lay in entry.layers[1::2]:
        data = json.loads((root.parent.parent / lay.metrics).read_text())
        drawn = len(gpd.read_file(root.parent.parent / lay.path))
        assert data["metrics"]["count_predicted"] == data["row"]["fields"] == drawn
        assert str(data["scored_layer"]).endswith(lay.path.rsplit("/", 1)[1])


def test_the_new_york_panels_follow_their_rule(tool):
    """28-28c (PREREG-NY section 5): ties go to the higher boundary F1, then to the order the
    rule lists the runs in; without sam-auto the best SPOT run, else the other Sentinel-2 run."""

    def row(run, f1, bf1, status="ok"):
        return {"region": "r", "run": run, "status": status, "f1": f1, "boundary_f1": bf1}

    rows = [
        row("naip da", "0.30", "0.5"),
        row("naip da-sam2", "0.40", "0.5"),
        row("naip da-ft", "0.40", "0.6"),  # ties da-sam2's F1: the higher boundary F1
        row("naip dinov3-ft", "0.40", "0.6"),  # ties both: the earlier in the rule's list
        row("sentinel2 da", "0.35", "0.4"),
        row("sentinel2 ftw", "0.35", "0.4"),  # a full tie: sentinel2 da
        row("naip sam-auto", "0.10", "0.2"),  # drawn whatever its F1
        row("naip da-ft-a1", "0.90", "0.9"),  # never a panel (near-sample)
        row("sentinel2 sam-auto", "0.90", "0.9"),  # never a panel
        {**row("naip da", "0.99", "0.9"), "region": "other"},
    ]
    assert tool._new_york_panels(rows, "r") == [
        "naip da",
        "naip da-ft",
        "sentinel2 da",
        "naip sam-auto",
    ]
    no_sam = [r for r in rows if r["run"] != "naip sam-auto"]
    spot = [row("spot da", "0.2", "0.3"), row("spot dinov3-ft", "0.3", "0.3")]
    assert tool._new_york_panels(no_sam + spot, "r")[3] == "spot dinov3-ft"
    assert tool._new_york_panels(no_sam, "r")[3] == "sentinel2 ftw"
    failed = [row("naip sam-auto", "", "", status="failed: X")]
    assert tool._new_york_panels(no_sam + failed, "r")[3] == "sentinel2 ftw"
    assert tool._new_york_panels([], "r") == [None, None, None, None]
    # The entries draw the rule's runs in the New York run (and its panels.json).
    entries = {e.key: e for e in tool.ENTRIES}
    for region, (key, *_rest) in tool.NY_REGIONS.items():
        stems = [tool.NY_RUNS[run][0] for run in tool.NY_PANELS[region]]
        paths = [f"{tool.NY}/{region}/{stem}.gpkg" for stem in stems]
        assert [lay.path for lay in entries[key].layers] == paths
    summary = tool.REPO / tool.NY_RUN_ROOT / tool.NY / "summary.csv"
    if not summary.exists():
        pytest.skip("example 28's run is not in this checkout")
    rows = _summary_rows(summary)
    chosen = json.loads((summary.parent / "panels.json").read_text())
    order = ("top_left", "top_right", "bottom_left", "bottom_right")
    for region, panels in tool.NY_PANELS.items():
        assert tool._new_york_panels(rows, region) == panels
        assert [chosen[region][p] for p in order] == panels


def test_the_new_york_entries_state_the_rule_scores_and_credits(tool):
    """28-28c: the OpenET credit with its source agency, the two sentences of rule C4, the
    Copernicus line of the Sentinel-2 panel, the training squares and the chance baselines of
    the run; SAM 2 refinement marked where it is drawn."""
    entries = {e.key: e for e in tool.ENTRIES}
    for region, (key, *_rest) in tool.NY_REGIONS.items():
        entry = entries[key]
        notes = " ".join(entry.model_notes)
        assert f"Reference: {tool.NY_CREDIT}" in entry.model_notes
        assert "source agency: Common Land Unit (CLU), USDA Farm Service Agency" in tool.NY_CREDIT
        assert "agreement with CLUs (farm-programme units), not with visible fields" in notes
        assert "one square per region is not an accuracy estimate for the county" in notes
        assert entry.copernicus_credit and entry.model_positions and entry.title_f1
        assert entry.run_root == tool.NY_RUN_ROOT
        runs = tool.NY_PANELS[region]
        assert not any(r.endswith("-a1") or r == "sentinel2 sam-auto" for r in runs)
        refined = [lay.highlight for lay in entry.layers]
        assert refined == ["agribound:sam_refined" if r.endswith("-sam2") else None for r in runs]
        assert entry.sam2_note == any(r.endswith("-sam2") for r in runs)
        tuned = [r for r in runs if r.endswith("-ft")]
        assert (tool.NY_TRAINING[region] in notes) == bool(tuned)
        assert "sam-auto (bottom right)" in notes and tool.NY_CHANCE_NOTE[region] in notes
    root = tool.REPO / tool.NY_RUN_ROOT
    summary = root / tool.NY / "summary.csv"
    if not summary.exists():
        pytest.skip("example 28's run is not in this checkout")
    import geopandas as gpd

    for region, (key, *_rest) in tool.NY_REGIONS.items():
        entry = entries[key]
        ref = gpd.read_file(tool._one(entry.reference, root))
        assert len(ref) == tool.NY_N_REFERENCE[region]
        assert f"square's {len(ref):,} CLU fields" in " ".join(entry.model_notes)
        for lay in entry.layers:  # the chance baselines of every panel drawn
            data = json.loads((root / lay.metrics).read_text())
            shifted = data["metrics"]["baselines"]["shifted_reference"]
            line = tool.NY_CHANCE_NOTE[region]
            assert f"moved {shifted['shift_m']:.1f} m east and north" in line
            assert f"matches {shifted['count_tp']} of its {len(ref)} fields" in line
            digits = 2 if region == "black_dirt" else 3
            assert f"F1 {shifted['f1']:.{digits}f}" in line
            assert data["row"]["above_chance"] == (region != "black_dirt")


def test_the_sam_entry_follows_its_rule(tool):
    """The SAM entry (round-2 PLAN 1.5): Diamond Valley on SPOT 6/7 and the E4 square at the
    median of the paired NAIP F1 differences, each with Delineate-Anything v2 as released and
    sam-auto in the whole 5 km square; the footer's numbers are the evaluation's."""
    median = tool._sam_median_square
    assert median([("A", 0.1, 0.5), ("B", 0.3, 0.2), ("C", 0.2, 0.4)]) == "C"
    # An even number: the two middle squares tie; the lower DA F1, then the first by name.
    assert median([("A", 0.1, 0.5), ("B", 0.2, 0.4), ("C", 0.3, 0.3), ("D", 0.4, 0.1)]) == "C"
    assert median([("A", 0.1, 0.5), ("B", 0.2, 0.3), ("C", 0.3, 0.3), ("D", 0.4, 0.1)]) == "B"
    with pytest.raises(ValueError):
        median([])
    entry = next(e for e in tool.ENTRIES if e.key == "sam")
    assert entry.multi_area and entry.run_root == tool.SAM_EVAL_RUN_ROOT
    assert not entry.copernicus_credit  # SPOT 6/7 and NAIP only
    da_dv, sam_dv, da_tn, sam_tn = entry.layers
    assert (da_dv.run_root, da_tn.run_root) == ("outputs/examples_1.1_ex24", tool.FINAL_RUN_ROOT)
    assert sam_dv.run_root is sam_tn.run_root is None  # the evaluation's own outputs
    assert (da_dv.area, sam_dv.area, da_tn.area, sam_tn.area) == ("dv-e", "dv-e", "tn", "tn")
    assert da_dv.window == (586500.0, 4391500.0, 5000.0)  # example 24's DV-E square
    assert da_tn.window == (302500.0, 3962500.0, 5000.0)  # example 25's Tennessee square
    assert sam_dv.window is None and sam_tn.window is None  # the area's window
    assert sam_dv.reference is None and sam_tn.reference is None  # the area's reference
    assert da_dv.path == f"{tool.NV}/diamond_valley/fields_spot_da_2023.gpkg"
    assert da_tn.path == f"{tool.MS}/tn/fields_naip_da_2023.gpkg"
    assert sam_dv.metrics_record == ("E1/DV-E/spot", "sam-auto", "own")
    assert sam_tn.metrics_record == ("E4/TN/naip", "sam-auto", "own")
    notes = tool._plain(" ".join(entry.model_notes))  # no-break spaces as spaces
    credit = entry.model_notes[-1]  # the references drawn, and V18 behind the endpoints
    assert credit.startswith(f"Reference: 1 {tool.NEVADA_CREDIT}; 2 {tool.NY_CREDIT}. ")
    assert "DRI/NMOSE field boundaries (V18), Desert Research Institute and New Mexico" in credit
    # The NAIP and Sentinel-2 endpoints rest on references of several OpenET source agencies,
    # not only the CLU of the drawn Tennessee square.
    assert (
        "The endpoints' other areas: OpenET v2.1.1, with the field boundaries of their source "
        "agencies; in Nevada, the DRI field boundaries (1); in New Mexico, the DRI/NMOSE"
    ) in credit
    # Examples 24 and 25 are new in 1.1.0: their scores were known, not published.
    assert "known before the evaluation (examples 24 and 25's own runs)" in notes
    assert "published" not in notes and "not from the landscape" in notes
    amendment = next(m for m in entry.model_notes if m.startswith("Amendment of 2026-10-13"))
    assert "after their first scores and the endpoints had been read" in amendment
    assert "a recorded deviation from PREREG-SAM section 6" in amendment
    assert amendment.endswith("Mason Valley's scores changed, square 1's did not")
    root = tool.REPO / tool.SAM_EVAL_RUN_ROOT
    if not (root / "E4" / "scores.json").exists():
        pytest.skip("the SAM evaluation's run root is not in this checkout")
    e4 = json.loads((root / "E4" / "scores.json").read_text())
    squares = [
        (key.split("/")[1], v["difference_f1"], v["da"]["f1"])
        for key, v in e4.items()
        if key.endswith("/naip")
    ]
    assert len(squares) == 47
    assert median(squares) == "TN"  # the 47 squares
    assert median([s for s in squares if s[0] != "CA"]) == "TN"  # the 46 of the endpoint
    e1 = json.loads((root / "E1" / "scores.json").read_text())
    for n, pair in (("1", e1["E1/DV-E/spot"]), ("2", e4["E4/TN/naip"])):
        blocks = pair["secondary"]["paired_blocks"]
        low, high = blocks["ci"]["f1"]
        quoted = f"{n}, {_signed(blocks['difference']['f1'])} [{_signed(low)}, {_signed(high)}]"
        assert quoted in notes
        assert blocks["n_blocks"] == 25 and blocks["n_resamples"] == 2000
        # The titles' F1 (each example's own score) equals the paired record's arms.
        assert pair["sam-auto"]["own"]["f1"] == pytest.approx(blocks["a"]["f1"])
        assert pair["da"]["own"]["f1"] == pytest.approx(blocks["b"]["f1"])
    assert "(Tennessee, " + _signed(e4["E4/TN/naip"]["difference_f1"]) + ")" in notes
    endpoints = json.loads((root / "endpoints.json").read_text())
    for key in ("primary_naip", "primary_sentinel2"):
        point = endpoints[key]
        low, high = point["ci"]
        assert f"{_signed(point['mean'])} [{_signed(low)}, {_signed(high)}]" in notes
    spot, landsat = endpoints["primary_spot_E1_E3"], endpoints["primary_landsat_E1_E3"]
    assert f"SPOT 6/7 {_signed(spot['mean'])}" in notes
    assert f"Landsat {_signed(landsat['mean'])} (Landsat 8/9; 7/8/9 on the San Juan box)" in notes
    # The four areas' Landsat composites: 8/9, but 7/8/9 on the San Juan box (2022).
    import rasterio

    missions = {}
    for area in ("E1/DV-E", "E1/MV-E", "E2/lea", "E3/san_juan"):
        (tif,) = sorted((root / area / "cache").glob("landsat_composite_*.tif"))
        with rasterio.open(tif) as src:
            missions[area] = tool._landsat_label(src.tags().get("AGRIBOUND_COLLECTIONS"))
    assert missions == {
        "E1/DV-E": "Landsat 8/9 C2 L2",
        "E1/MV-E": "Landsat 8/9 C2 L2",
        "E2/lea": "Landsat 8/9 C2 L2",
        "E3/san_juan": "Landsat 7/8/9 C2 L2",
    }
    # The pooled pairs mix filter states (PREREG-SAM 3.2: "The report says so"); the mean of
    # the four unfiltered pairs is a secondary result.
    for point in (spot, landsat):
        assert point["note"] == (
            "pairs matched within each area: unfiltered in E1, crop-filtered in E2 and E3"
        )
    assert "of pairs scored without the crop filter in Nevada and with it in New Mexico" in notes
    assert (
        f"with all four pairs unfiltered: {_signed(spot['mean_unfiltered_pairs'])} and "
        f"{_signed(landsat['mean_unfiltered_pairs'])}, a secondary result"
    ) in notes
    assert endpoints["primary_sentinel2"]["verdict"] == "did not transfer"
    # The amendment's counts: Mason Valley's reference and the crop-filter copy E1 first read.
    import geopandas as gpd

    refs = tool.REPO / "outputs/examples_1.1_ex24" / tool.NV / "references"
    (full,) = [
        p
        for p in refs.glob("mason_valley_reference_*.gpkg")
        if re.fullmatch(r"mason_valley_reference_[0-9a-f]{12}\.gpkg", p.name)
    ]
    (copy,) = refs.glob("mason_valley_reference_*_2024_crop_t0.3.gpkg")
    n_copy = len(gpd.read_file(copy, layer="fields"))
    n_full = len(gpd.read_file(full, layer="reference"))
    assert f"({n_copy} of its {n_full} fields)" in amendment
    # The comparators' metrics files (each example's own score) give the DA arm's F1.
    for lay, pair in ((da_dv, e1["E1/DV-E/spot"]), (da_tn, e4["E4/TN/naip"])):
        data = json.loads((tool.REPO / lay.run_root / lay.metrics).read_text())
        assert data["row"]["f1"] == pytest.approx(pair["da"]["own"]["f1"])


def test_multi_area_layers_take_a_given_window_their_own_run_root_and_a_scores_record(
    tool, tmp_path, monkeypatch
):
    """Layer.window (the area's window as given), Layer.run_root (a layer read under its own
    run root, unless --run-root replaces it), Layer.metrics_record (one record of a scores
    file), and the area's reference drawn on its panels that name none."""
    import shutil

    repo = tmp_path / "repo"
    own = _run_root(repo / "own")  # the comparator's run root: repo/own/run
    entry_root = repo / "eval"
    (entry_root / "out").mkdir(parents=True)
    # The evaluation's output: the comparator's polygons, its composite named absolutely.
    shutil.copy(own / "outputs" / "demo" / "fields.gpkg", entry_root / "out" / "sam.gpkg")
    sidecar = {
        "status": "success",
        "config": {"source": "sentinel2", "engine": "sam-auto", "year": 2024},
        "facts": {"raster_path": str(own / "outputs" / "demo" / "composite.tif")},
        "engine_meta": {
            "model": "facebook/sam2.1-hiera-large",
            "checkpoint": {"revision": "665f8e2ad61c"},
            "sam2": {"version": "1.1.0", "n_files": 62, "meta_commit": "2b90b9f5ceec"},
            "settings": {
                "sam_tile_m": 5120.0,
                "points_per_side": 32,
                "pred_iou_thresh": 0.8,
                "stability_score_thresh": 0.9,
            },
            "settings_basis": "default",
            "device": "mps",
        },
    }
    (entry_root / "out" / "sam.gpkg.provenance.json").write_text(json.dumps(sidecar))
    record = {"A/x": {"sam-auto": {"own": {"f1": 0.625, "precision": 0.5, "recall": 0.8}}}}
    (entry_root / "scores.json").write_text(json.dumps(record))
    monkeypatch.setattr(tool, "REPO", repo)
    entry = tool.Entry(
        "93",
        "Demo_given_windows",
        "Demo",
        [
            tool.Layer(
                "outputs/demo/fields.gpkg",
                "Area A: comparator",
                reference="outputs/demo/reference.gpkg",
                area="a",
                window=(500250.0, 3999250.0, 500.0),
                run_root="own/run",
            ),
            tool.Layer(
                "out/sam.gpkg",
                "Area A\nsam-auto",  # a title of two lines
                area="a",
                metrics="scores.json",
                metrics_record=("A/x", "sam-auto", "own"),
            ),
        ],
        multi_area=True,
        per_layer_background=True,
        crop_m=400,
        ncols=2,
        inset=False,
        title_f1=True,
        model_positions=True,
        run_root="eval",
    )
    assert tool._layer_root(entry, entry.layers[0], entry_root) == own
    assert tool._layer_root(entry, entry.layers[1], entry_root) == entry_root
    elsewhere = tmp_path / "elsewhere"  # an explicit --run-root replaces both
    assert tool._layer_root(entry, entry.layers[0], elsewhere) == elsewhere
    stats = tool.render_areas(entry, entry_root, tmp_path / "gallery", "composite")
    a, b = stats["layers"]
    assert a["run_root"] == "own/run" and "run_root" not in b
    assert (a["file"], b["file"]) == ("outputs/demo/fields.gpkg", "out/sam.gpkg")
    for lay in (a, b):
        assert lay["window"]["bounds"] == [500000.0, 3999000.0, 500500.0, 3999500.0]
        assert lay["window"]["given"] is True and lay["window"]["side_m"] == 500.0
        assert lay["reference"]["n_polygons"] == 1  # the area's reference on both panels
    assert b["metrics"]["record"] == ["A/x", "sam-auto", "own"] and b["metrics"]["f1"] == 0.625
    assert b["metrics"]["file"] == "scores.json"
    lines = stats["model_notes"]
    model = (
        "SAM 2.1 automatic masks (sam-auto): facebook/sam2.1-hiera-large @ 665f8e2, the "
        "engine's Sentinel-2 L2A default: 5,120 m tiles"
    )
    assert any(m.startswith(model) for m in lines)
    assert (
        "SAM 2.1: Meta, Apache-2.0; sam2 1.1.0, a third-party PyPI build whose 62 code, "
        "configuration and licence files equal Meta's commit 2b90b9f, run on device mps"
    ) in lines
    # A window outside the imagery, another window on an area's later panel, a missing
    # record and the single-area renderer are refused.
    import dataclasses

    far = dataclasses.replace(entry.layers[0], window=(600000.0, 3999250.0, 500.0))
    with pytest.raises(ValueError, match="not inside the imagery"):
        tool.render_areas(
            dataclasses.replace(entry, layers=[far, entry.layers[1]]),
            entry_root,
            tmp_path / "g2",
            "composite",
        )
    other = dataclasses.replace(entry.layers[1], window=(500300.0, 3999300.0, 400.0))
    with pytest.raises(ValueError, match="another window"):
        tool.render_areas(
            dataclasses.replace(entry, layers=[entry.layers[0], other]),
            entry_root,
            tmp_path / "g3",
            "composite",
        )
    missing = dataclasses.replace(entry.layers[1], metrics_record=("A/x", "da", "own"))
    with pytest.raises(KeyError, match="no record"):
        tool.render_areas(
            dataclasses.replace(entry, layers=[entry.layers[0], missing]),
            entry_root,
            tmp_path / "g4",
            "composite",
        )
    for name, kw in (
        ("Layer.window", {"window": (1.0, 1.0, 1.0)}),
        ("Layer.run_root", {"run_root": "own"}),
    ):
        single = tool.Entry("92", "Demo_single", "Demo", [tool.Layer("x.gpkg", "A", **kw)])
        with pytest.raises(ValueError, match=name):
            tool.render(single, entry_root, tmp_path / "g5", "composite")


def test_a_multi_area_title_of_two_lines_makes_room_above_its_row(tool, tmp_path):
    """Titles of two lines (the SAM entry) add a title line above the maps and between their
    rows, as in render; titles of one line keep the figure's height."""
    from PIL import Image

    root = _run_root(tmp_path)
    _second_area(root)
    heights = {}
    for name, labels in (("one", ("A", "B")), ("two", ("A\nfirst", "B\nsecond"))):
        entry = tool.Entry(
            "91",
            f"Demo_titles_{name}",
            "Demo",
            [
                tool.Layer("outputs/demo/fields.gpkg", labels[0]),
                tool.Layer("outputs/demo2/fields.gpkg", labels[1]),
            ],
            multi_area=True,
            per_layer_background=True,
            crop_m=400,
            ncols=1,
            inset=False,
        )
        tool.render_areas(entry, root, tmp_path / "gallery", "composite")
        with Image.open(tmp_path / "gallery" / f"Demo_titles_{name}.png") as im:
            heights[name] = im.height
    # Two rows: one extra title line above the first row and one between the rows.
    extra = round(2 * tool.TITLE_LINE_IN * tool.DPI)
    assert abs(heights["two"] - heights["one"] - extra) <= 1


def test_naip_mosaics_name_the_images_of_other_years_underneath(tool):
    """A NAIP mosaic of agribound 1.1 that fills gaps from images of other years names them
    by date; other composites keep their line."""
    tags = {
        "AGRIBOUND_COMPOSITE_METHOD": "mosaic",
        "AGRIBOUND_DATE_START": "2022-01-01",
        "AGRIBOUND_DATE_END_EXCLUSIVE": "2023-01-01",
        "AGRIBOUND_N_IMAGES": "5",
        "AGRIBOUND_RESOLUTION_M": "1.0",
        "AGRIBOUND_IMAGE_YEARS": "2022",
        "AGRIBOUND_IMAGE_YEARS_UNDERNEATH": "2021",
        "AGRIBOUND_IMAGE_LIST": json.dumps(
            [["a", "2021-09-20", None]] * 3 + [["b", "2022-10-14", None]] * 2
        ),
    }
    prov = {"facts": {"composite": tags}}
    note = tool._bg_note(([], "naip", prov, "the engine's input"))
    assert note == (
        "NAIP 1 m mosaic composite, 2022-01-01 to 2023-01-01 (end exclusive), 5 images: 2 of "
        "2022-10-14 on top, gaps filled from 3 of 2021-09-20; the engine's input"
    )
    facts = tool._composite_facts(prov)
    assert facts["image_years_underneath"] == "2021" and "image_list" in facts
    plain = {k: v for k, v in tags.items() if "UNDERNEATH" not in k}
    facts = tool._composite_facts({"facts": {"composite": plain}})
    assert "image_list" not in facts and "image_years_underneath" not in facts
    assert tool._bg_note(([], "naip", {"facts": {"composite": plain}}, "x")).endswith("5 images; x")
    broken = {**tags, "AGRIBOUND_IMAGE_LIST": "not json"}
    assert tool._bg_note(([], "naip", {"facts": {"composite": broken}}, "x")).endswith(
        "5 images; x"
    )


def test_sam_auto_lines_name_the_setting_the_build_and_the_licence(tool):
    """sam-auto panels: a model line per setting (it names the source's default) and one
    credit line: SAM 2.1 (Meta, Apache-2.0) and the sam2 build that ran it, from engine_meta."""
    em = {
        "model": "facebook/sam2.1-hiera-large",
        "checkpoint": {"revision": "665f8e2ad61cf5f53d65644ff27c8ee525124610"},
        "sam2": {"version": "1.1.0", "n_files": 62, "meta_commit": "2b90b9f5ceec907a"},
        "settings": {
            "sam_tile_m": 6144.0,
            "points_per_side": 32,
            "pred_iou_thresh": 0.6,
            "stability_score_thresh": 0.95,
        },
        "settings_basis": "default",
        "device": "mps",
    }
    prov = {"config": {"engine": "sam-auto", "source": "spot"}, "engine_meta": em}
    line = tool._model_note(prov)
    assert "SPOT\u00a06/7" in line  # the source's name on one line of the footer
    assert tool._plain(line) == (
        "SAM 2.1 automatic masks (sam-auto): facebook/sam2.1-hiera-large @ 665f8e2, the "
        "engine's SPOT 6/7 default: 6,144 m tiles, 32 points per side, predicted IoU > 0.6, "
        "stability >= 0.95"
    )
    credit = tool._sam_auto_credit([prov, {"config": {"engine": "delineate-anything"}}, prov])
    assert credit == (
        "SAM 2.1: Meta, Apache-2.0; sam2 1.1.0, a third-party PyPI build whose 62 code, "
        "configuration and licence files equal Meta's commit 2b90b9f, run on device mps"
    )
    assert tool._sam_auto_credit([{"config": {"engine": "delineate-anything"}}]) is None
    given = {**prov, "engine_meta": {**em, "settings_basis": "engine_params"}}
    assert "settings (engine_params): 6,144 m tiles" in tool._model_note(given)


def test_a_sam_auto_panel_prints_its_model_and_credit_lines(tool, tmp_path):
    """render: a sam-auto panel adds its model line (the source's name kept on one footer
    line) and the SAM 2.1 credit line; gallery_stats.json holds them with plain spaces."""
    root = _run_root(tmp_path)
    side = root / "outputs" / "demo" / "fields.gpkg.provenance.json"
    sidecar = json.loads(side.read_text())
    sidecar["config"]["engine"] = "sam-auto"
    sidecar["engine_meta"] = {
        "model": "facebook/sam2.1-hiera-large",
        "checkpoint": {"revision": "665f8e2ad61c"},
        "sam2": {"version": "1.1.0", "n_files": 62, "meta_commit": "2b90b9f5ceec"},
        "settings": {"sam_tile_m": 5120.0, "points_per_side": 32},
        "settings_basis": "default",
        "device": "mps",
    }
    side.write_text(json.dumps(sidecar))
    entry = tool.Entry(
        "90", "Demo_sam_auto", "Demo", [tool.Layer("outputs/demo/fields.gpkg", "A")], inset=False
    )
    stats = tool.render(entry, root, tmp_path / "gallery", "composite")
    lines = stats["model_notes"]
    assert lines[0].startswith("SAM 2.1 automatic masks (sam-auto): facebook/sam2.1-hiera-large")
    assert "the engine's Sentinel-2 L2A default: 5,120 m tiles, 32 points per side" in lines[0]
    assert lines[1].startswith("SAM 2.1: Meta, Apache-2.0; sam2 1.1.0, a third-party PyPI build")
    assert tool.NBSP not in json.dumps(stats, ensure_ascii=False)
