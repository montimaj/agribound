"""Pipeline regressions: final minimum area, imagery end date, warnings and composite facts.

Stubbed engines on the local test raster (no GPU, GEE or ML dependencies).
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from agribound.pipeline import _engine_imagery_ends, _imagery_window_end, delineate
from agribound.provenance import read_provenance

X0, Y0 = 500000.0, 4000000.0  # lower-left corner of the 640 m test raster (EPSG:32611)


class Stub:
    """Returns the given boxes (metres from the raster corner) and engine_meta."""

    def __init__(self, sides=((20, 20, 220, 220),), meta=None, warn=()):
        self.sides = sides
        self.meta = meta or {"backend": "stub"}
        self.warn = warn

    def delineate(self, raster_path, config):
        for message in self.warn:
            logging.getLogger("agribound.engines.stub").warning(message)
        geoms = [box(X0 + a, Y0 + b, X0 + c, Y0 + d) for a, b, c, d in self.sides]
        gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:32611")
        gdf.attrs["engine_meta"] = self.meta
        return gdf


@pytest.fixture
def run_kwargs(sample_rgb_tif, tmp_path):
    return dict(
        source="local",
        local_tif_path=sample_rgb_tif,
        engine="delineate-anything",
        output_path=str(tmp_path / "out" / "fields.gpkg"),
        device="cpu",
        lulc_filter=False,
        year=2024,
    )


def _use(monkeypatch, stub):
    monkeypatch.setattr("agribound.engines.get_engine", lambda name: stub)


def test_min_field_area_holds_after_smoothing(monkeypatch, run_kwargs):
    """Regression: polygons shrunk below min_field_area_m2 by the smoothing were kept."""
    # 52 m square: 2704 m² before smoothing, about 2260 m² after the default 3 Chaikin
    # iterations of a 4-corner outline (-16.4 %); the 70 m square stays above 2500 m².
    _use(monkeypatch, Stub(sides=((20, 20, 72, 72), (200, 200, 270, 270))))
    gdf = delineate(**run_kwargs)
    assert len(gdf) == 1
    assert gdf["metrics:area"].min() >= 2500
    record = read_provenance(run_kwargs["output_path"])
    assert record["facts"]["n_postprocessed"] == 1
    assert "again after" in record["facts"]["postprocess"]["min_field_area_applied"]


def test_without_outline_edits_the_area_filter_runs_once(monkeypatch, run_kwargs):
    _use(monkeypatch, Stub(sides=((20, 20, 72, 72),)))
    gdf = delineate(**run_kwargs, simplify_tolerance=0, engine_params={"smooth_iterations": 0})
    assert len(gdf) == 1 and gdf["metrics:area"].iloc[0] == pytest.approx(2704, rel=2e-3)


def test_engine_warnings_reach_the_provenance_record(monkeypatch, run_kwargs, caplog):
    """Regression: engine WARNINGs were logged but warnings was [] in every record."""
    note = "this raster's pixel size (30.00 m) is outside that range"
    _use(monkeypatch, Stub(warn=(note, note)))
    logging.getLogger("other.library").warning("not an agribound warning")
    with caplog.at_level("WARNING"):
        delineate(**run_kwargs)
    record = read_provenance(run_kwargs["output_path"])
    assert record["warnings"].count(note) == 1  # identical messages are kept once
    assert "not an agribound warning" not in record["warnings"]
    assert record["warnings_not_recorded"] == 0
    # The handler is removed when the run ends.
    assert not [
        h
        for h in logging.getLogger("agribound").handlers
        if type(h).__name__ == "_WarningCollector"
    ]


def test_composite_tags_are_copied_into_the_record(monkeypatch, run_kwargs):
    """Regression: the composite's metadata lived only in the cached GeoTIFF's tags."""
    import rasterio

    # A local raster in the expected layout is used as it is, so tag it like a composite.
    with rasterio.open(run_kwargs["local_tif_path"], "r+") as dst:
        dst.update_tags(
            AGRIBOUND_N_IMAGES="69", AGRIBOUND_DATE_START="2024-01-01", TESSERA_YEAR="2024"
        )
        dst.update_tags(UNRELATED="x")
    _use(monkeypatch, Stub())
    delineate(**run_kwargs)
    record = read_provenance(run_kwargs["output_path"])
    composite = record["facts"]["composite"]
    assert composite["AGRIBOUND_N_IMAGES"] == "69"
    assert composite["AGRIBOUND_DATE_START"] == "2024-01-01" and composite["TESSERA_YEAR"] == "2024"
    assert "UNRELATED" not in composite
    assert record["facts"]["raster_path"] == run_kwargs["local_tif_path"]


FTW_WINDOWS = {
    "a": {"start": "2024-08-30", "end": "2024-10-29", "status": "composite"},
    "b": {"start": "2025-05-07", "end": "2025-07-06", "status": "composite"},
}


def test_determination_datetime_uses_the_engines_own_windows(monkeypatch, run_kwargs):
    """Regression: FTW's window B (next year in the south) was ignored."""
    _use(monkeypatch, Stub(meta={"backend": "ftw-tools", "windows": FTW_WINDOWS}))
    gdf = delineate(**run_kwargs)
    assert gdf["determination:datetime"].iloc[0] == pd.Timestamp("2025-07-06T23:59:59Z")


def test_imagery_end_rules():
    from agribound.config import AgriboundConfig

    cfg = AgriboundConfig(
        source="local", local_tif_path="x.tif", engine="ftw", year=2024, output_path="o.gpkg"
    )
    end = pd.Timestamp("2024-12-31T23:59:59Z")
    assert _imagery_window_end(cfg) == end
    assert _imagery_window_end(cfg, {"windows": FTW_WINDOWS}) == pd.Timestamp(
        "2025-07-06T23:59:59Z"
    )
    fallback = {"a": FTW_WINDOWS["a"], "b": {**FTW_WINDOWS["b"], "status": "annual_fallback"}}
    # Window B fell back to the annual composite: the configured year end is the latest.
    assert _imagery_window_end(cfg, {"windows": fallback}) == end
    assert _imagery_window_end(cfg, {"windows": {"single": {"status": "input raster"}}}) == end
    ensemble = {
        "members": [
            {"label": "da", "engine_meta": {"backend": "native"}},
            {"label": "ftw", "engine_meta": {"windows": FTW_WINDOWS}},
        ]
    }
    assert _imagery_window_end(cfg, ensemble) == pd.Timestamp("2025-07-06T23:59:59Z")
    assert _engine_imagery_ends(
        {"windows": {"a": {"status": "composite", "end": "bad"}}}, end.date()
    ) == [
        end.date(),
        end.date(),
    ]
    ranged = cfg.merged(date_range=("2024-03-01", "2024-09-30"))
    assert _imagery_window_end(ranged) == pd.Timestamp("2024-09-30T23:59:59Z")
