"""Helpers of example 26 (examples/26_sar_field_boundaries.py); no network, no model."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pandas as pd
import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "26_sar_field_boundaries.py"


@pytest.fixture(scope="module")
def ex26():
    spec = importlib.util.spec_from_file_location("example_26_sar_field_boundaries", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_keeps_the_rows_of_a_site_whose_reference_failed(ex26, tmp_path):
    # An earlier run of all sites wrote summary.csv ...
    path = tmp_path / "summary.csv"
    earlier = [
        ("ucd", "umbra", 0.485, "00000000000000e1"),  # a hash that reads as a number
        ("ucd", "naip", 0.426, "00000000000000e1"),
        ("sevilla", "capella", 0.513, "fedcba9876543210"),
        ("centerfield", "umbra", 0.163, "0123456789abcdef"),
    ]
    columns = ["site", "run", "f1", "reference_sha256"]
    pd.DataFrame(earlier, columns=columns).to_csv(path, index=False)
    # ... then "--only ucd sevilla" while DWR's service is down: only Sevilla gives rows.
    rows = [dict(zip(columns, ("sevilla", "capella", 0.5, "fedcba9876543210"), strict=True))]
    summary = ex26.merged_summary(path, rows)
    assert list(summary.itertuples(index=False, name=None)) == [
        ("ucd", "umbra", 0.485, "00000000000000e1"),
        ("ucd", "naip", 0.426, "00000000000000e1"),
        ("sevilla", "capella", 0.5, "fedcba9876543210"),
        ("centerfield", "umbra", 0.163, "0123456789abcdef"),
    ]
    # Without an earlier file, only this run's rows.
    assert ex26.merged_summary(tmp_path / "none.csv", rows)["site"].tolist() == ["sevilla"]


def test_reference_digest_ignores_the_order_of_the_outlines(ex26):
    import geopandas as gpd
    from shapely.geometry import box

    ref = gpd.GeoDataFrame(geometry=[box(0, 0, 50, 50), box(60, 0, 120, 50)], crs=32610)
    digest = ex26.reference_digest(ref)
    assert digest == ex26.reference_digest(ref.iloc[::-1])
    assert digest != ex26.reference_digest(ref.iloc[:1])


def test_sigpac_campaigns_from_the_tiles_dates(ex26):
    dates = ["Mon, 10 Feb 2025 10:08:05 GMT", None, "Mon, 10 Feb 2025 10:22:26 GMT"]
    assert ex26.sigpac_campaigns(dates) == [ex26.SIGPAC_CAMPAIGN] == [2025]
    assert ex26.sigpac_campaigns(dates + ["Mon, 19 Jan 2026 13:34:36 GMT"]) == [2025, 2026]
    assert ex26.sigpac_campaigns([None]) == []


def test_dry_run_selection_never_rebuilds_the_index(ex26, tmp_path, monkeypatch, capsys):
    from agribound.composites import vhr_sar

    def no_refresh(*args, **kwargs):
        raise AssertionError("the dry run listed the archive")

    monkeypatch.setattr(vhr_sar, "refresh_index", no_refresh)
    args = types.SimpleNamespace(vhr_sar_cache_dir=str(tmp_path))
    config = types.SimpleNamespace(source="umbra")
    ex26.show_selection("ucd", config, args)
    assert "index not cached" in capsys.readouterr().out
    assert not any(tmp_path.iterdir())
    path = vhr_sar.index_path("umbra", tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"an index of an older agribound")
    ex26.show_selection("ucd", config, args)
    assert "unreadable or of an older format" in capsys.readouterr().out
    assert path.read_bytes() == b"an index of an older agribound"
