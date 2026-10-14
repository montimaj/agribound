"""Example 23's ``--s2-single-date`` option (examples/23_tree_crops.py); no network, no model."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "23_tree_crops.py"
SLUGS = ["twifo", "oro", "madera", "jaen"]
MANY = {"engine_params": {"max_detections": 1000}}


@pytest.fixture(scope="module")
def ex23():
    spec = importlib.util.spec_from_file_location("example_23_tree_crops", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parsed(ex23, *argv):
    args = ex23.parse_args(["--gee-project", "test-project", *argv])
    args.spot = not args.no_spot  # as in a dry run: SPOT access is not checked
    return args


@pytest.mark.parametrize("slug", SLUGS)
def test_off_by_default(ex23, slug):
    args = parsed(ex23)
    assert args.s2_single_date is False
    runs = ex23.site_runs(slug, ex23.SITES[slug], args, {})
    assert [r for r in runs if r[1] == "sentinel2" and "date_range" in r[3]] == []


@pytest.mark.parametrize("slug", SLUGS)
def test_single_date_run_follows_the_sentinel2_run(ex23, slug):
    site = ex23.SITES[slug]
    args = parsed(ex23, "--s2-single-date")
    date = f"{site['year']}-06-15"
    default = ex23.site_runs(slug, site, args, {})
    runs = ex23.site_runs(slug, site, args, {}, s2_date=date)
    at = runs.index(("da", "sentinel2", "delineate-anything", MANY))
    assert runs[: at + 1] + runs[at + 2 :] == default  # the only change
    label, source, engine, overrides = runs[at + 1]
    assert (label, source, engine) == (f"da-{date}", "sentinel2", "delineate-anything")
    assert overrides == {**MANY, "date_range": (date, date)}
    config = ex23.base_config(site, slug, source, engine, label, args, **overrides)
    median = ex23.base_config(site, slug, "sentinel2", engine, "da", args, **MANY)
    assert tuple(config.date_range) == (date, date)
    assert config.composite_method == "median"
    assert Path(config.output_path).name == f"fields_sentinel2_da-{date}_{site['year']}.gpkg"
    # The scene filter and mask of the area's median run.
    for field in ("cloud_cover_max", "s2_cloud_mask", "cloud_score_threshold", "lulc_filter"):
        assert getattr(config, field) == getattr(median, field), field


def survey_table():
    table = pd.DataFrame(
        {
            "rank": [1, 2],
            "date": ["2020-08-04", "2020-01-12"],
            "acquisition": ["S2B 2020-08-04", "S2A 2020-01-12"],
            "n_images": [1, 2],
            "image_ids": [["A"], ["B", "C"]],
            "scene_cloud_pct": [10.6, 3.0],
            "extent_share": [1.0, 1.0],
            "clear_share": [1.0, 0.98],
            "date_range": [("2020-08-04", "2020-08-04"), ("2020-01-12", "2020-01-12")],
        }
    )
    table.attrs["cloud_mask"] = "cloud_score_plus"
    return table


def test_clearest_date_is_the_survey_rank_1(ex23, monkeypatch, tmp_path):
    seen = []

    def fake_survey(config):
        seen.append(config)
        return survey_table()

    monkeypatch.setattr(ex23.agribound, "survey_scenes", fake_survey, raising=False)
    monkeypatch.setattr(ex23, "OUTPUT_DIR", tmp_path)
    (tmp_path / "twifo").mkdir()
    args = parsed(ex23, "--s2-single-date")
    assert ex23.clearest_s2_date(ex23.SITES["twifo"], "twifo", args) == "2020-08-04"
    (config,) = seen
    assert (config.source, config.year, config.date_range) == ("sentinel2", 2020, None)
    assert (config.cloud_cover_max, config.s2_cloud_mask) == (80, "cloud_score_plus")
    saved = pd.read_csv(tmp_path / "twifo" / "scenes_sentinel2_2020.csv")
    assert saved["date"].tolist() == ["2020-08-04", "2020-01-12"]
    assert saved["image_ids"].tolist() == ["A", "B;C"]
    assert "date_range" not in saved.columns


def test_no_acquisition_gives_no_run(ex23, monkeypatch, tmp_path):
    empty = survey_table().iloc[0:0]
    monkeypatch.setattr(ex23.agribound, "survey_scenes", lambda config: empty, raising=False)
    monkeypatch.setattr(ex23, "OUTPUT_DIR", tmp_path)
    (tmp_path / "oro").mkdir()
    args = parsed(ex23, "--s2-single-date")
    assert ex23.clearest_s2_date(ex23.SITES["oro"], "oro", args) is None
    assert (tmp_path / "oro" / "scenes_sentinel2_2021.csv").exists()


def test_dry_run_prints_a_placeholder_and_nothing_else_changes(ex23, capsys):
    ex23.dry_run(["madera"], parsed(ex23, "--dry-run"))
    without = capsys.readouterr().out.splitlines()
    ex23.dry_run(["madera"], parsed(ex23, "--dry-run", "--s2-single-date"))
    with_option = capsys.readouterr().out.splitlines()
    added = [line for line in with_option if line not in without]
    assert len(with_option) == len(without) + 2 and len(added) == 2
    assert "sentinel2 da-<date>" in added[0] and "clearest date of 2022" in added[0]
    assert added[0].rstrip().endswith("fields_sentinel2_da-<date>_2022.gpkg")
    assert "s2_cloud_mask=scl" in added[1] and "cloud_cover_max=20" in added[1]
    # Right after the area's Sentinel-2 median run (and its notes line).
    at = next(i for i, line in enumerate(with_option) if "sentinel2 da " in line)
    assert with_option[at + 2] == added[0]
