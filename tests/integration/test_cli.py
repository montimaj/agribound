"""Integration tests for the agribound CLI."""

from __future__ import annotations

from click.testing import CliRunner

from agribound.cli import main


class TestCliVersion:
    """Test --version flag."""

    def test_version_output(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "agribound" in result.output.lower()


class TestCliListEngines:
    """Test the list-engines command."""

    def test_list_engines_exits_ok(self):
        runner = CliRunner()
        result = runner.invoke(main, ["list-engines"])
        assert result.exit_code == 0
        assert "delineate-anything" in result.output
        assert "ftw" in result.output
        assert "geoai" in result.output
        assert "prithvi" in result.output
        assert "embedding" in result.output
        assert "ensemble" in result.output

    def test_list_engines_shows_approach(self):
        runner = CliRunner()
        result = runner.invoke(main, ["list-engines"])
        # At least one engine approach substring should appear
        assert "segmentation" in result.output.lower() or "clustering" in result.output.lower()


class TestCliListSources:
    """Test the list-sources command."""

    def test_list_sources_exits_ok(self):
        runner = CliRunner()
        result = runner.invoke(main, ["list-sources"])
        assert result.exit_code == 0
        assert "sentinel2" in result.output or "Sentinel" in result.output
        assert "landsat" in result.output or "Landsat" in result.output
        assert "naip" in result.output or "NAIP" in result.output

    def test_list_sources_shows_resolution(self):
        runner = CliRunner()
        result = runner.invoke(main, ["list-sources"])
        # Sentinel-2 is 10m
        assert "10m" in result.output
