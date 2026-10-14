"""tools/make_gallery_pampas_0.1x.py: the fitted frame of the 0.1.x Pampas screenshot.

The tool is not shipped in the wheel, so the tests are skipped when the source
checkout is not available.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "make_gallery_pampas_0.1x.py"

if not TOOL.exists():  # e.g. running from an installed wheel
    pytest.skip("tools/make_gallery_pampas_0.1x.py is not available", allow_module_level=True)

pytest.importorskip("matplotlib")
pytest.importorskip("rasterio")
pytest.importorskip("geopandas")


@pytest.fixture(scope="module")
def tool():
    sys.path.insert(0, str(TOOL.parent))
    spec = importlib.util.spec_from_file_location("make_gallery_pampas_0_1x", TOOL)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(TOOL.parent))


def test_the_frame_is_the_documented_scale_and_rotation(tool):
    a, b, _tx, _ty = tool.FRAME
    assert 1 / math.hypot(a, b) == pytest.approx(12.55, abs=0.01)  # metres per screenshot pixel
    assert abs(math.degrees(math.atan2(b, a))) == pytest.approx(32.8, abs=0.1)


def test_the_defaults_are_the_1_1_run_and_gallery_and_the_old_layer_can_be_named(tool, tmp_path):
    """As tools/make_gallery.py: outputs/examples_1.1 into assets/gallery_1.1; the 0.1.x layer
    (not in the repository) is given with --old-layer, and a missing one is an error."""
    assert tool.mg.RUN_ROOT == ROOT / "outputs" / "examples_1.1"
    assert tool.mg.OUT_DIR == ROOT / "assets" / "gallery_1.1"
    assert tool.OLD_LAYER.name == "fields_sam2_tessera_improved_2024.gpkg"
    with pytest.raises(SystemExit):
        tool.main(["--old-layer", str(tmp_path / "missing.gpkg"), "--out-dir", str(tmp_path)])
    assert not list(tmp_path.iterdir())  # nothing written


def test_the_notes_name_the_composite_from_its_tags(tool, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    import numpy as np

    tif = tmp_path / "sentinel2_composite_demo.tif"
    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:32720",
        transform=rasterio.transform.from_origin(0, 20, 10, 10),
    ) as dst:
        dst.write(np.zeros((1, 2, 2), dtype="float32"))
        dst.update_tags(
            AGRIBOUND_DATE_START="2024-10-01",
            AGRIBOUND_DATE_END_EXCLUSIVE="2024-11-01",
            AGRIBOUND_COMPOSITE_METHOD="median",
            AGRIBOUND_N_IMAGES="8",
            AGRIBOUND_RESOLUTION_M="10.0",
        )
    assert tool.composite_note(tif) == (
        "Sentinel-2 L2A 10 m median composite of 2024-10-01 to 2024-11-01 (end exclusive), 8 images"
    )


def test_the_screenshot_footprint_is_about_23_by_18_km(tool):
    poly = tool.frame_polygon(1864, 1458)
    xs, ys = poly.exterior.coords.xy
    top = math.dist((xs[0], ys[0]), (xs[1], ys[1]))
    side = math.dist((xs[1], ys[1]), (xs[2], ys[2]))
    assert top == pytest.approx(23_400, rel=0.01)
    assert side == pytest.approx(18_300, rel=0.01)
    # The frame maps back onto the screenshot corners.
    m = tool.frame_affine()
    col, row = m * (xs[2], ys[2])
    assert (col, row) == pytest.approx((1864, 1458), abs=1e-6)
