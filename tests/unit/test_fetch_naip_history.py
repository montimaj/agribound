"""tools/fetch_naip_history.py exports a square of a year's NAIP with a record of the export.

No test reaches Earth Engine: the grid, band and naming rules are pure functions, the Earth
Engine image is checked against a stand-in ``ee`` module that records the calls, and the
command is run with the image list and the download replaced by stand-ins that write a small
GeoTIFF. The tool is not shipped in the wheel, so the tests are skipped when the source
checkout is not available.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "fetch_naip_history.py"

if not TOOL.exists():  # e.g. running from an installed wheel
    pytest.skip("tools/fetch_naip_history.py is not available", allow_module_level=True)

rasterio = pytest.importorskip("rasterio")

MODULE_NAME = "fetch_naip_history"
CORNER = (375000.0, 5385000.0)  # example 25's Montana square (EPSG:32612)


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(MODULE_NAME, None)


def _images(n: int = 2, bands=("R", "G", "B")) -> list[dict]:
    return [
        {
            "asset_id": f"USDA/NAIP/DOQQ/n_481121{k}_se_12_1_20050728",
            "date": "2005-07-28",
            "time_start_ms": 1122508800000 + k,
            "bands": list(bands),
            "crs": "EPSG:26912",
            "native_resolution_m": 1.0,
        }
        for k in range(n)
    ]


def test_the_grid_is_the_square_from_its_upper_left_corner(tool):
    grid = tool.square_grid("EPSG:32612", CORNER, 5000, 2)
    assert (grid.crs, grid.width, grid.height) == ("EPSG:32612", 2500, 2500)
    assert grid.crs_transform == [2.0, 0.0, 375000.0, 0.0, -2.0, 5390000.0]
    for side, res in ((5001, 2), (5000, 0), (-5000, 2), (5000, 3)):
        with pytest.raises(ValueError):
            tool.square_grid("EPSG:32612", CORNER, side, res)


def test_bands_are_rgb_unless_every_image_has_nir(tool):
    rgb, rgbn = _images(1)[0], _images(1, ("R", "G", "B", "N"))[0]
    assert tool.shared_bands([rgbn, rgbn]) == ["R", "G", "B", "N"]
    assert tool.shared_bands([rgb, rgbn]) == ["R", "G", "B"]
    with pytest.raises(ValueError, match="no image"):
        tool.shared_bands([])
    with pytest.raises(ValueError, match="no band B"):
        tool.shared_bands([{**rgb, "bands": ["R", "G"]}])
    assert tool.output_stem(2005, ["R", "G", "B"], 2.0) == "naip_2005_rgb_2m"
    assert tool.output_stem(2023, ["R", "G", "B", "N"], 0.6) == "naip_2023_rgbn_0.6m"


class _Chain:
    """A stand-in Earth Engine object: every method call is logged and returns a new one."""

    def __init__(self, log: list, info=None):
        self._log, self._info = log, info

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self._log.append((name, args, kwargs))
            if name == "getInfo":
                return self._info
            return _Chain(self._log, self._info)

        return call


def _fake_ee(info=None) -> tuple[types.ModuleType, list]:
    log: list = []
    ee = types.ModuleType("ee")
    ee.ImageCollection = lambda *a: log.append(("ImageCollection", a, {})) or _Chain(log, info)
    ee.Image = lambda obj: obj
    ee.Geometry = lambda *a: "square"
    ee.Feature = lambda *a: a
    ee.Algorithms = types.SimpleNamespace(If=lambda *a: a)
    ee.Reducer = types.SimpleNamespace(mean=lambda: "mean")
    return ee, log


def test_the_image_is_the_documented_mosaic_averaged_to_the_grid(tool, monkeypatch):
    ee, log = _fake_ee()
    monkeypatch.setitem(sys.modules, "ee", ee)
    grid = tool.square_grid("EPSG:32612", CORNER, 5000, 2)
    tool.naip_image(grid, 2005, ["R", "G", "B"])
    calls = {name: (args, kwargs) for name, args, kwargs in log}
    assert calls["ImageCollection"][0] == ("USDA/NAIP/DOQQ",)
    assert calls["filterBounds"][0] == ("square",)
    assert calls["filterDate"][0] == ("2005-01-01", "2006-01-01")  # the calendar year
    assert calls["sort"][0] == ("system:time_start",)  # the latest image on top
    assert calls["reduceResolution"][1] == {"reducer": "mean", "maxPixels": 1024}
    assert calls["reproject"][1] == {"crs": "EPSG:32612", "crsTransform": grid.crs_transform}
    names = [name for name, *_ in log]
    order = ["mosaic", "setDefaultProjection", "reduceResolution", "reproject", "round", "uint8"]
    assert [n for n in names if n in order] == order
    assert (["R", "G", "B"],) in [args for name, args, _ in log if name == "select"]


def test_the_image_list_gives_asset_ids_dates_and_native_pixels(tool, monkeypatch):
    rows = [  # as Earth Engine returns them, out of order, one without a full asset ID
        {"asset_id": "n_4811228_nw_12_1_20050728", "time_start_ms": 1122508800001},
        {"asset_id": "USDA/NAIP/DOQQ/n_4811219_se_12_1_20050728", "time_start_ms": 1122508800000},
    ]
    info = {
        "features": [
            {
                "properties": {
                    **r,
                    "bands": ["R", "G", "B"],
                    "crs": "EPSG:26912",
                    "native_resolution_m": 1,
                }
            }
            for r in rows
        ]
    }
    ee, _log = _fake_ee(info)
    monkeypatch.setitem(sys.modules, "ee", ee)
    images = tool.list_images(tool.square_grid("EPSG:32612", CORNER, 5000, 2), 2005)
    assert [im["asset_id"] for im in images] == [
        "USDA/NAIP/DOQQ/n_4811219_se_12_1_20050728",
        "USDA/NAIP/DOQQ/n_4811228_nw_12_1_20050728",
    ]
    assert {im["date"] for im in images} == {"2005-07-28"}
    assert images[0]["native_resolution_m"] == 1.0 and images[0]["crs"] == "EPSG:26912"


def _stand_ins(tool, monkeypatch, images):
    """Earth Engine replaced: no login, the given image list, a download that writes a small
    GeoTIFF on the grid with the tags it is given (as export_ee_image does)."""
    import agribound.auth

    calls = {"download": 0}
    monkeypatch.setattr(agribound.auth, "setup_gee", lambda project=None: None)
    monkeypatch.setattr(tool, "list_images", lambda grid, year: images)

    def download(grid, year, bands, path, tags):
        calls["download"] += 1
        data = np.full((len(bands), grid.height, grid.width), 100, dtype="uint8")
        data[:, :2, :] = 0  # two rows without data
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=grid.width,
            height=grid.height,
            count=len(bands),
            dtype="uint8",
            crs=grid.crs,
            transform=grid.transform,
            nodata=0,
        ) as dst:
            dst.write(data)
            dst.descriptions = tuple(bands)
            dst.update_tags(**{k: str(v) for k, v in tags.items()})

    monkeypatch.setattr(tool, "download", download)
    return calls


def test_the_command_writes_the_image_and_its_record(tool, monkeypatch, tmp_path):
    calls = _stand_ins(tool, monkeypatch, _images(2))
    out = tmp_path / "montana"
    argv = ["--year", "2005", "--crs", "EPSG:32612", "--corner", "375000", "5385000"]
    argv += ["--side", "200", "--resolution", "2", "--out-dir", str(out)]
    assert tool.main([*argv, "--dry-run"]) == 0
    assert calls["download"] == 0 and not out.exists()  # a dry run writes nothing
    assert tool.main(argv) == 0
    tif, record_path = out / "naip_2005_rgb_2m.tif", out / "naip_2005_rgb_2m.json"
    record = json.loads(record_path.read_text())
    assert record["collection"] == "USDA/NAIP/DOQQ" and record["year"] == 2005
    assert record["date_window"] == ["2005-01-01", "2006-01-01"]
    assert record["square"] == {"crs": "EPSG:32612", "corner": list(CORNER), "side_m": 200.0}
    assert [im["asset_id"] for im in record["images"]] == [im["asset_id"] for im in _images(2)]
    assert record["images"][0] == {
        "asset_id": "USDA/NAIP/DOQQ/n_4811210_se_12_1_20050728",
        "date": "2005-07-28",
        "bands": ["R", "G", "B"],
        "crs": "EPSG:26912",
        "native_resolution_m": 1.0,
    }
    assert record["bands"] == ["R", "G", "B"]
    export = record["export"]
    assert export["crs_transform"] == [2.0, 0.0, 375000.0, 0.0, -2.0, 5385200.0]
    assert (export["width"], export["height"], export["resolution_m"]) == (100, 100, 2.0)
    assert (export["dtype"], export["nodata"]) == ("uint8", 0)
    assert "mean of the native pixels" in export["aggregation"]
    assert "the latest on top" in export["mosaic"]
    assert record["output"]["file"] == tif.name
    assert record["output"]["sha256"] == hashlib.sha256(tif.read_bytes()).hexdigest()
    assert record["output"]["valid_fraction"] == pytest.approx(0.98)
    assert record["credit"].startswith("NAIP: USDA Farm Service Agency")
    assert record["versions"]["agribound"]
    with rasterio.open(tif) as src:
        tags = src.tags()
        assert src.descriptions == ("R", "G", "B")
    # The tags agribound's composites use to describe their contents, and this tool's.
    assert tags["AGRIBOUND_COLLECTIONS"] == "USDA/NAIP/DOQQ"
    assert tags["AGRIBOUND_COMPOSITE_METHOD"] == "mosaic"
    assert (tags["AGRIBOUND_DATE_START"], tags["AGRIBOUND_DATE_END_EXCLUSIVE"]) == (
        "2005-01-01",
        "2006-01-01",
    )
    assert tags["AGRIBOUND_N_IMAGES"] == "2" and tags["AGRIBOUND_RESOLUTION_M"] == "2.0"
    assert tags["AGRIBOUND_IMAGE_DATE_FIRST"] == tags["AGRIBOUND_IMAGE_DATE_LAST"] == "2005-07-28"
    listed = json.loads(tags["AGRIBOUND_IMAGE_LIST"])
    assert [row[0] for row in listed] == [im["asset_id"] for im in _images(2)]
    assert tags["AGRIBOUND_VALID_FRACTION"] == "0.98"
    assert tags["NAIP_HISTORY_RECORD"] == record_path.name
    assert "AGRIBOUND_SOURCE" not in tags and "AGRIBOUND_VERSION" not in tags
    # An existing image is replaced only with --overwrite.
    with pytest.raises(SystemExit, match="exists"):
        tool.main(argv)
    assert tool.main([*argv, "--overwrite"]) == 0
    assert calls["download"] == 2


def test_the_command_refuses_a_year_without_images(tool, monkeypatch, tmp_path):
    _stand_ins(tool, monkeypatch, [])
    argv = ["--year", "1999", "--crs", "EPSG:32612", "--corner", "375000", "5385000"]
    with pytest.raises(SystemExit, match="No USDA/NAIP/DOQQ image of 1999"):
        tool.main([*argv, "--out-dir", str(tmp_path)])
    assert not list(tmp_path.iterdir())
