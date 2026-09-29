"""Earth Engine project checks: ``agribound tiles gee-project`` and the example scripts.

The region files name no Earth Engine project, so the scripts must find the
user's own (``--gee-project``, ``GEE_PROJECT``, gcloud, the key's
``project_id``) and stop with a clear error before running or submitting
anything when a run needs Earth Engine and none is found.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from agribound.cli import main

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
DRIVER = str(EXAMPLES / "run_region_delineation.sh")
SUBMIT = str(EXAMPLES / "hpc" / "submit_region.sh")
TEST_BBOX = "bbox:1.40,48.10,1.45,48.12"

#: Variables that could supply a project or key from the developer's environment.
_PROJECT_ENV = (
    "GEE_PROJECT",
    "AGRIBOUND_GEE_PROJECT",
    "AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY",
    "AGB_GEE_SERVICE_ACCOUNT_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
)

needs_posix_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="needs a POSIX bash (HPC scripts are not supported on Windows)",
)


@pytest.fixture
def no_project(monkeypatch):
    """No project from the environment or gcloud (for in-process CLI calls)."""
    for name in _PROJECT_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("agribound.auth._get_gcloud_project", lambda: None)


def _key(tmp_path: Path, project: str | None, name: str = "key.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"project_id": project} if project else {}))
    return path


def _base_yaml(tmp_path: Path, **fields) -> Path:
    data = {
        "source": "sentinel2",
        "engine": "delineate-anything",
        "year": 2024,
        "study_area": TEST_BBOX,
        "output_path": str(tmp_path / "base" / "fields.gpkg"),
        "lulc_filter": False,
        "gee_project": None,
    }
    data.update(fields)
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _gee_project(*args: str):
    return CliRunner().invoke(main, ["tiles", "gee-project", *args])


# ---------------------------------------------------------------------------
# agribound tiles gee-project
# ---------------------------------------------------------------------------


class TestGeeProjectCommand:
    def test_missing_project_is_an_actionable_error(self, no_project):
        res = _gee_project("--sources", "sentinel2,tessera-embedding")
        assert res.exit_code == 1
        assert "No Google Earth Engine project found" in res.output
        assert "(source sentinel2, the LULC filter)" in res.output
        for hint in ("--gee-project ID", "GEE_PROJECT", "gcloud config set project", "project_id"):
            assert hint in res.output

    def test_without_sources_a_project_is_always_needed(self, no_project):
        res = _gee_project()
        assert res.exit_code == 1 and "No Google Earth Engine project found" in res.output

    def test_lulc_filter_needs_earth_engine_for_every_source(self, no_project):
        res = _gee_project("--sources", "tessera-embedding")
        assert res.exit_code == 1 and "(the LULC filter)" in res.output

    def test_no_earth_engine_needed_prints_nothing(self, no_project):
        res = _gee_project("--sources", "tessera-embedding usgs-naip-plus", "--no-lulc-filter")
        assert res.exit_code == 0 and res.output == ""

    def test_google_embedding_uses_earth_engine(self, no_project):
        res = _gee_project("--sources", "google-embedding", "--no-lulc-filter")
        assert res.exit_code == 1 and "source google-embedding" in res.output

    def test_unknown_source(self, no_project):
        res = _gee_project("--sources", "sentinel9")
        assert res.exit_code == 2 and "Unknown sources ['sentinel9']" in res.output

    def test_resolution_order(self, no_project, monkeypatch, tmp_path):
        key = _key(tmp_path, "key-proj")
        assert _gee_project("--service-account-key", str(key)).output == "key-proj\n"
        monkeypatch.setenv(
            "AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY", str(_key(tmp_path, "env-key", "k2"))
        )
        assert _gee_project().output == "env-key\n"
        monkeypatch.setattr("agribound.auth._get_gcloud_project", lambda: "gcloud-proj")
        assert _gee_project("--service-account-key", str(key)).output == "gcloud-proj\n"
        monkeypatch.setenv("GEE_PROJECT", "env-proj")
        assert _gee_project().output == "env-proj\n"
        assert _gee_project("--project", "explicit").output == "explicit\n"

    def test_google_application_credentials_project_id(self, no_project, monkeypatch, tmp_path):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(_key(tmp_path, "adc-proj")))
        assert _gee_project("--sources", "landsat").output == "adc-proj\n"

    def test_key_without_project_id_is_not_enough(self, no_project, tmp_path):
        res = _gee_project("--service-account-key", str(_key(tmp_path, None)))
        assert res.exit_code == 1

    def test_config_gee_source_without_project(self, no_project, tmp_path):
        res = _gee_project("--config", str(_base_yaml(tmp_path)))
        assert res.exit_code == 1
        assert "(source sentinel2)" in res.output
        assert "gee_project in the base configuration" in res.output

    def test_config_project_and_key(self, no_project, tmp_path):
        base = _base_yaml(tmp_path, gee_project="yaml-proj")
        assert _gee_project("--config", str(base)).output == "yaml-proj\n"
        key = _key(tmp_path, "key-proj")
        base = _base_yaml(tmp_path, gee_service_account_key=str(key))
        assert _gee_project("--config", str(base)).output == "key-proj\n"

    def test_config_that_needs_no_earth_engine(self, no_project, tmp_path):
        base = _base_yaml(tmp_path, source="tessera-embedding", engine="embedding")
        res = _gee_project("--config", str(base))
        assert res.exit_code == 0 and res.output == ""
        base = _base_yaml(
            tmp_path,
            source="google-embedding",
            engine="embedding",
            google_embedding_backend="source_coop",
        )
        assert _gee_project("--config", str(base)).output == ""

    def test_config_lulc_filter_and_google_embedding(self, no_project, tmp_path):
        base = _base_yaml(
            tmp_path, source="tessera-embedding", engine="embedding", lulc_filter=True
        )
        res = _gee_project("--config", str(base))
        assert res.exit_code == 1 and "(the LULC filter)" in res.output
        base = _base_yaml(tmp_path, source="google-embedding", engine="embedding")
        res = _gee_project("--config", str(base))
        assert res.exit_code == 1 and "google-embedding with the gee backend" in res.output

    def test_config_is_validated(self, no_project, tmp_path):
        base = _base_yaml(tmp_path, no_such_field=1)
        res = _gee_project("--config", str(base))
        assert res.exit_code == 2 and "Invalid base configuration" in res.output
        assert "no_such_field" in res.output

    def test_config_excludes_the_other_options(self, no_project, tmp_path):
        res = _gee_project("--config", str(_base_yaml(tmp_path)), "--sources", "landsat")
        assert res.exit_code == 2 and "cannot be combined" in res.output

    def test_no_gcloud_call_when_not_needed(self, monkeypatch):
        for name in _PROJECT_ENV:
            monkeypatch.delenv(name, raising=False)

        def fail():
            raise AssertionError("gcloud must not be queried")

        monkeypatch.setattr("agribound.auth._get_gcloud_project", fail)
        res = _gee_project("--sources", "tessera-embedding", "--no-lulc-filter")
        assert res.exit_code == 0 and res.output == ""


# ---------------------------------------------------------------------------
# Region files
# ---------------------------------------------------------------------------


def _keys(value, prefix=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield f"{prefix}{k}"
            yield from _keys(v, f"{prefix}{k}.")
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item, prefix)


@pytest.mark.parametrize(
    "path", sorted((EXAMPLES / "regions").glob("*.yaml")), ids=lambda p: p.stem
)
def test_region_files_name_no_earth_engine_project(path):
    data = yaml.safe_load(path.read_text())
    assert not [k for k in _keys(data) if k.split(".")[-1] == "gee_project"]


# ---------------------------------------------------------------------------
# Scripts (subprocess, no gcloud project, no project variables)
# ---------------------------------------------------------------------------


def _agribound_bin() -> str:
    candidate = Path(sys.executable).with_name("agribound")
    if candidate.exists():
        return str(candidate.parent)
    found = shutil.which("agribound")
    if not found:
        pytest.skip("agribound console script not found")
    return str(Path(found).parent)


@pytest.fixture
def bare_env(tmp_path):
    """Environment without a project: a fake gcloud prints '(unset)'."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    (bindir / "gcloud").write_text('#!/usr/bin/env bash\necho "(unset)"\n')
    (bindir / "gcloud").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in _PROJECT_ENV}
    env["PATH"] = os.pathsep.join([str(bindir), _agribound_bin(), env.get("PATH", "")])
    env.update(AGB_GPU_ACCOUNT="g", AGB_CPU_ACCOUNT="c")
    return env


def _driver(tmp_path, env, *args):
    cmd = ["bash", DRIVER, "--region", "beauce_fr", "--test", "--years", "2024"]
    cmd += [*args, "--out-root", str(tmp_path / "out"), "--dry-run"]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


@needs_posix_bash
def test_driver_stops_without_a_project(tmp_path, bare_env):
    proc = _driver(tmp_path, bare_env, "--sources", "sentinel2", "--engines", "delineate-anything")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "No Google Earth Engine project found" in proc.stderr
    assert "[region] ERROR: no Earth Engine project for these runs" in proc.stderr
    assert "nothing was run" in proc.stderr and "--gee-project ID" in proc.stderr
    assert "===" not in proc.stdout  # no run was validated or started
    assert not (tmp_path / "out").exists()


@needs_posix_bash
def test_driver_uses_the_keys_project_for_every_run(tmp_path, bare_env):
    key = _key(tmp_path, "key-proj")
    proc = _driver(
        tmp_path,
        bare_env,
        "--sources",
        "sentinel2",
        "--engines",
        "delineate-anything",
        "--mode",
        "slurm",
        "--profile",
        "delta",
        "--gee-service-account-key",
        str(key),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[region] Earth Engine project: key-proj" in proc.stderr
    # submit_region.sh reads it from the run's config.yaml (written with --gee-project).
    assert "[agribound-hpc] Earth Engine project: key-proj" in proc.stderr
    sbatch = _sbatch_lines(proc.stdout)
    assert len(sbatch) == 3 and not [line for line in sbatch if "GEE_PROJECT=" in line]


@needs_posix_bash
def test_driver_needs_no_project_without_earth_engine(tmp_path, bare_env):
    proc = _driver(
        tmp_path,
        bare_env,
        "--sources",
        "tessera-embedding",
        "--engines",
        "embedding",
        "--no-lulc-filter",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Earth Engine project" not in proc.stderr
    assert "DRY   2024/tessera-embedding__embedding" in proc.stdout


@needs_posix_bash
def test_driver_help_documents_the_project(tmp_path, bare_env):
    proc = subprocess.run(["bash", DRIVER, "--help"], capture_output=True, text=True, env=bare_env)
    assert proc.returncode == 0
    assert "Earth Engine project: the region files name none" in proc.stdout
    assert "--gee-project ID         your Earth Engine project" in proc.stdout


def _submit(tmp_path, env, base, *args):
    cmd = ["bash", SUBMIT, "--profile", "delta", "--config", str(base)]
    cmd += ["--out-dir", str(tmp_path / "out"), "--tile-size-km", "10", *args, "--dry-run"]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _sbatch_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if " sbatch --parsable " in f" {line}"]


@needs_posix_bash
def test_submit_stops_without_a_project(tmp_path, bare_env):
    proc = _submit(tmp_path, bare_env, _base_yaml(tmp_path))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "No Google Earth Engine project found" in proc.stderr
    assert "nothing was submitted" in proc.stderr
    assert _sbatch_lines(proc.stdout) == []
    assert "tiles make" not in proc.stdout


@needs_posix_bash
def test_submit_exports_the_project_found_when_the_config_has_none(tmp_path, bare_env):
    base = _base_yaml(tmp_path, source="google-embedding", engine="embedding")
    proc = _submit(tmp_path, {**bare_env, "GEE_PROJECT": "env-proj"}, base, "--compute", "cpu")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Earth Engine project: env-proj" in proc.stderr
    stage, compute, merge = _sbatch_lines(proc.stdout)
    assert "GEE_PROJECT=env-proj" in stage and "GEE_PROJECT=env-proj" in compute
    assert "GEE_PROJECT=" not in merge  # the merge does not use Earth Engine


@needs_posix_bash
def test_submit_keeps_the_configs_project(tmp_path, bare_env):
    base = _base_yaml(tmp_path, gee_project="yaml-proj")
    proc = _submit(tmp_path, {**bare_env, "GEE_PROJECT": "env-proj"}, base)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Earth Engine project: yaml-proj" in proc.stderr
    assert not [line for line in _sbatch_lines(proc.stdout) if "GEE_PROJECT=" in line]


@needs_posix_bash
def test_submit_sees_the_agb_service_account_key(tmp_path, bare_env):
    key = _key(tmp_path, "key-proj")
    base = _base_yaml(tmp_path, source="google-embedding", engine="embedding")
    proc = _submit(
        tmp_path, {**bare_env, "AGB_GEE_SERVICE_ACCOUNT_KEY": str(key)}, base, "--compute", "cpu"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Earth Engine project: key-proj" in proc.stderr
    stage, compute, _ = _sbatch_lines(proc.stdout)
    assert "GEE_PROJECT=key-proj" in stage and "GEE_PROJECT=key-proj" in compute


@needs_posix_bash
def test_submit_needs_no_project_without_earth_engine(tmp_path, bare_env):
    base = _base_yaml(tmp_path, source="tessera-embedding", engine="embedding")
    proc = _submit(tmp_path, bare_env, base, "--compute", "cpu")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Earth Engine project" not in proc.stderr
    assert not [line for line in _sbatch_lines(proc.stdout) if "GEE_PROJECT=" in line]
