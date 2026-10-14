"""Live checks of the scene survey, the image list and the date layers (Earth Engine).

At example 23's Twifo Praso square (Ghana) for 2020, with example 23's Sentinel-2
settings (Cloud Score+ at 0.60, scene cloud cover <= 80 %). The expected values are those
of the Earth Engine catalogue on 2026-10-10, which a pilot of 2026-10-09 also found; a
reprocessing of Sentinel-2 or Cloud Score+ can change them.

They need Earth Engine credentials (``gee`` marker) and are excluded from the offline
suite. Run them with::

    GEE_PROJECT=my-project pytest -m gee tests/unit/test_scenes_live.py -s

``AGRIBOUND_TEST_WORKLOAD_TAG`` sets the Earth Engine workload tag and
``AGRIBOUND_SMOKE_DIR`` keeps the downloaded date layers (default: pytest's ``tmp_path``).
A restricted-mode report stops the session. Requests: one survey, one image-count request
with the image list, and one image count and one download (501 x 501 px x 3 uint16 bands)
for the date layers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio

from agribound.config import AgriboundConfig

pytestmark = pytest.mark.gee

#: Example 23's Twifo Praso square (Ghana), MGRS tile T30NXM.
TWIFO = (
    "POLYGON ((-1.4955032599320377 5.515674790288927, -1.495388865913433 5.560891430400164, "
    "-1.5405184541386092 5.56100479685173, -1.540629420016046 5.515787229421631, "
    "-1.4955032599320377 5.515674790288927))"
)
#: The four acquisitions of 2020 fully clear over the square under Cloud Score+, by
#: mean cs_cdf, with their mean aerosol optical thickness.
FULLY_CLEAR = {"2020-08-04": 0.056, "2020-01-12": 0.59, "2020-01-27": 1.18, "2020-01-07": 1.08}


def _config(out_dir, **kwargs):
    params = {
        "source": "sentinel2",
        "year": 2020,
        "study_area": TWIFO,
        "cloud_cover_max": 80,
        "s2_cloud_mask": "cloud_score_plus",
        "gee_project": os.environ.get("GEE_PROJECT") or None,
        "gee_workload_tag": os.environ.get("AGRIBOUND_TEST_WORKLOAD_TAG") or None,
        "output_path": str(Path(out_dir) / "twifo.gpkg"),
        "lulc_filter": False,
    }
    params.update(kwargs)
    return AgriboundConfig(**params)


def _stop_if_restricted(text: str) -> None:
    if "restricted mode" in text.lower():
        pytest.exit(f"Earth Engine reported restricted mode; stopping: {text}", returncode=1)


@pytest.fixture
def live(caplog):
    """``ee`` and a check of the logs (agribound logs restricted-mode warnings)."""
    ee = pytest.importorskip("ee")
    caplog.set_level("WARNING")
    state = SimpleNamespace(ee=ee, check=lambda: _stop_if_restricted(caplog.text))
    yield state
    state.check()


@pytest.fixture
def out_dir(tmp_path):
    base = os.environ.get("AGRIBOUND_SMOKE_DIR")
    path = Path(base) if base else tmp_path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run(fn, *args, **kwargs):
    """Call *fn*; a restricted-mode error stops the session."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        _stop_if_restricted(str(exc))
        raise


def test_survey_twifo_2020(live, tmp_path):
    import agribound

    table = _run(agribound.survey_scenes, _config(tmp_path))
    live.check()
    assert len(table) == 52 and table.attrs["n_acquisitions"] == 52
    assert table["rank"].tolist() == list(range(1, 53))
    assert table.attrs["scale_m"] == 60.0 and table.attrs["crs"] == "EPSG:32630"
    assert table.attrs["reduced_over"] == "the study area"
    assert (table["n_images"] == 1).all()
    assert table["clear_share"].is_monotonic_decreasing  # ties allowed
    top = table.head(4)
    assert top["date"].tolist() == list(FULLY_CLEAR)
    assert (top["clear_share"] > 0.999).all() and (top["cs_clear_share"] > 0.999).all()
    for date, aot in zip(top["date"], top["aot_mean"], strict=True):
        assert aot == pytest.approx(FULLY_CLEAR[date], abs=0.01), date
    first = table.iloc[0]
    assert first["date_range"] == ("2020-08-04", "2020-08-04")
    assert first["acquisition"] == "Sentinel-2B_2020-08-04_R065"
    assert first["image_ids"] == [
        "COPERNICUS/S2_SR_HARMONIZED/20200804T101559_20200804T103905_T30NXM"
    ]
    assert first["cs_mean"] == pytest.approx(0.903, abs=0.002)


def test_image_list_twifo_2020(live, tmp_path):
    """The rows of the image list, as the builder requests them (no download)."""
    from agribound.auth import ensure_gee
    from agribound.composites import gee

    config = _config(tmp_path)
    ensure_gee(config)
    geom = gee.study_area_geometry_4326(config)
    crs = gee.resolve_export_crs(config.export_crs, geom)
    grid = gee.compute_export_grid(geom, crs, gee.export_resolution_m(config))
    spec = gee._build_sentinel2(config, gee.ee_geometry(gee.grid_footprint_4326(grid)))
    summary = _run(
        gee.collection_summary,
        spec.raw,
        "sentinel2 2020 image count (live test)",
        image_list=True,
        cloud_property=spec.scene_cloud_property,
    )
    live.check()
    tags = gee._image_list_tags(summary["images"], summary["n_images"], summary["dates"])
    rows = json.loads(tags["AGRIBOUND_IMAGE_LIST"])
    assert summary["n_images"] == 52 and len(rows) == 52
    assert tags["AGRIBOUND_IMAGE_DATES"] == 52
    assert tags["AGRIBOUND_IMAGE_DATE_FIRST"] == "2020-01-02"
    assert tags["AGRIBOUND_IMAGE_DATE_LAST"] == "2020-12-27"
    assert "AGRIBOUND_IMAGE_LIST_TRUNCATED" not in tags
    assert all(r[0].startswith("COPERNICUS/S2_SR_HARMONIZED/2020") for r in rows)
    assert [r[1] for r in rows] == sorted(r[1] for r in rows)
    assert all(r[2] is not None and 0 <= r[2] <= 80 for r in rows)
    assert ["COPERNICUS/S2_SR_HARMONIZED/20200804T101559_20200804T103905_T30NXM",
            "2020-08-04", 10.63] in rows  # fmt: skip


@pytest.mark.slow
def test_date_layers_twifo_2020(live, out_dir):
    import agribound
    from agribound.composites.scenes import DATE_LAYER_BANDS, day_to_date

    config = _config(out_dir, cache_dir=str(out_dir / "cache"), gee_max_requests=4)
    path = _run(agribound.build_date_layers, config)
    live.check()
    with rasterio.open(path) as src:
        assert src.count == 3 and src.dtypes[0] == "uint16" and src.nodata == 0
        assert src.descriptions == DATE_LAYER_BANDS
        assert src.crs.to_epsg() == 32630 and src.res == (10.0, 10.0)
        assert (src.height, src.width) == (501, 501)
        tags = src.tags()
        n_valid, first, last = (band.astype(np.int64) for band in src.read())
    # The composite of example 23's 2020 run (agribound 1.0.1 recorded this cache key).
    assert tags["AGRIBOUND_COMPOSITE_CACHE_KEY"] == "e2c329c1855c"
    assert tags["AGRIBOUND_N_IMAGES"] == "52"
    assert int(n_valid.min()) == 10 and int(n_valid.max()) == 30
    assert float(np.median(n_valid)) == 20.0
    assert day_to_date(int(first.min())) == "2020-01-02"
    assert day_to_date(int(last.max())) == "2020-12-27"
    assert (first <= last).all() and float(np.median(last - first)) == 355.0
