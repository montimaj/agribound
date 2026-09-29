"""Tests for agribound.auth decision logic (Earth Engine fully mocked, no network)."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from agribound import auth

HV_URL = "https://earthengine-highvolume.googleapis.com"


@pytest.fixture
def fake_ee(monkeypatch):
    ee = types.ModuleType("ee")
    ee.Initialize = MagicMock(name="Initialize")
    ee.Authenticate = MagicMock(name="Authenticate")
    ee.ServiceAccountCredentials = MagicMock(name="ServiceAccountCredentials", return_value="SA")
    ee.Number = MagicMock()
    ee.data = types.SimpleNamespace(
        HIGH_VOLUME_API_BASE_URL=HV_URL,
        is_initialized=MagicMock(return_value=False),
        setDefaultWorkloadTag=MagicMock(name="setDefaultWorkloadTag"),
    )
    monkeypatch.setitem(sys.modules, "ee", ee)
    monkeypatch.delenv("AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GEE_PROJECT", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    auth._reset_state()
    yield ee
    auth._reset_state()


@pytest.fixture
def key_file(tmp_path):
    path = tmp_path / "sa.json"
    path.write_text(json.dumps({"type": "service_account", "project_id": "key-project"}))
    return str(path)


def _adc(ok=True):
    # google-auth comes with earthengine-api (the ``gee`` extra); the CI core job
    # does not install it, so the tests that patch ADC are skipped there.
    pytest.importorskip("google.auth")
    if ok:
        return patch("google.auth.default", return_value=("ADC", "adc-project"))
    return patch("google.auth.default", side_effect=Exception("no ADC"))


class TestCredentialOrder:
    def test_explicit_service_account_key(self, fake_ee, key_file):
        auth.setup_gee(project="p", service_account_key=key_file)
        fake_ee.ServiceAccountCredentials.assert_called_once_with(None, key_file=key_file)
        fake_ee.Initialize.assert_called_once_with("SA", project="p", opt_url=None)

    def test_env_service_account_key(self, fake_ee, key_file, monkeypatch):
        monkeypatch.setenv("AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY", key_file)
        auth.setup_gee(project="p")
        fake_ee.ServiceAccountCredentials.assert_called_once()
        fake_ee.Initialize.assert_called_once_with("SA", project="p", opt_url=None)

    def test_explicit_key_wins_over_env(self, fake_ee, key_file, tmp_path, monkeypatch):
        other = tmp_path / "other.json"
        other.write_text("{}")
        monkeypatch.setenv("AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY", str(other))
        auth.setup_gee(project="p", service_account_key=key_file)
        fake_ee.ServiceAccountCredentials.assert_called_once_with(None, key_file=key_file)

    def test_project_from_key(self, fake_ee, key_file):
        with patch.object(auth, "_get_gcloud_project", return_value=None):
            auth.setup_gee(service_account_key=key_file)
        fake_ee.Initialize.assert_called_once_with("SA", project="key-project", opt_url=None)

    def test_missing_key_file(self, fake_ee, tmp_path):
        with pytest.raises(FileNotFoundError):
            auth.setup_gee(project="p", service_account_key=str(tmp_path / "missing.json"))

    def test_persistent_credentials(self, fake_ee):
        with _adc() as adc:
            auth.setup_gee(project="p")
        fake_ee.Initialize.assert_called_once_with(project="p", opt_url=None)
        adc.assert_not_called()
        fake_ee.Authenticate.assert_not_called()

    def test_adc_after_persistent_fails(self, fake_ee):
        fake_ee.Initialize.side_effect = [Exception("no stored creds"), None]
        with _adc() as adc:
            auth.setup_gee(project="p")
        adc.assert_called_once_with(scopes=list(auth.EE_SCOPES))
        assert fake_ee.Initialize.call_args_list[-1].args == ("ADC",)
        assert fake_ee.Initialize.call_args_list[-1].kwargs == {"project": "p", "opt_url": None}
        fake_ee.Authenticate.assert_not_called()

    def test_non_interactive_never_authenticates(self, fake_ee, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        fake_ee.Initialize.side_effect = Exception("no stored creds")
        with _adc(ok=False), pytest.raises(RuntimeError, match="non-interactive") as err:
            auth.setup_gee(project="p")
        fake_ee.Authenticate.assert_not_called()
        assert "service-account key" in str(err.value)
        assert "no stored creds" in str(err.value) and "no ADC" in str(err.value)

    def test_interactive_false_never_authenticates(self, fake_ee):
        fake_ee.Initialize.side_effect = Exception("no stored creds")
        with _adc(ok=False), pytest.raises(RuntimeError):
            auth.setup_gee(project="p", interactive=False)
        fake_ee.Authenticate.assert_not_called()

    def test_interactive_authenticates(self, fake_ee):
        fake_ee.Initialize.side_effect = [Exception("no stored creds"), None]
        with _adc(ok=False):
            auth.setup_gee(project="p", interactive=True)
        fake_ee.Authenticate.assert_called_once()
        assert fake_ee.Initialize.call_count == 2

    def test_interactive_failure_raises(self, fake_ee):
        fake_ee.Initialize.side_effect = Exception("nope")
        with _adc(ok=False), pytest.raises(RuntimeError, match="GEE authentication failed"):
            auth.setup_gee(project="p", interactive=True)


class TestOptions:
    def test_high_volume_endpoint(self, fake_ee):
        auth.setup_gee(project="p", high_volume=True)
        fake_ee.Initialize.assert_called_once_with(project="p", opt_url=HV_URL)

    def test_workload_tag(self, fake_ee):
        auth.setup_gee(project="p", workload_tag="agribound-run1")
        fake_ee.data.setDefaultWorkloadTag.assert_called_once_with("agribound-run1")

    def test_no_workload_tag_by_default(self, fake_ee):
        auth.setup_gee(project="p")
        fake_ee.data.setDefaultWorkloadTag.assert_not_called()

    def test_project_required(self, fake_ee):
        with (
            patch.object(auth, "_get_gcloud_project", return_value=None),
            pytest.raises(ValueError, match="GEE project ID is required"),
        ):
            auth.setup_gee()

    def test_project_from_env(self, fake_ee, monkeypatch):
        monkeypatch.setenv("GEE_PROJECT", "env-p")
        auth.setup_gee()
        fake_ee.Initialize.assert_called_once_with(project="env-p", opt_url=None)

    def test_project_from_google_application_credentials(self, fake_ee, key_file, monkeypatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", key_file)
        with patch.object(auth, "_get_gcloud_project", return_value=None), _adc():
            auth.setup_gee()
        # No service-account key: ADC is used, with the project_id of that file.
        fake_ee.ServiceAccountCredentials.assert_not_called()
        fake_ee.Initialize.assert_called_once_with(project="key-project", opt_url=None)

    def test_project_from_credentials_order(self, fake_ee, tmp_path, monkeypatch):
        def key(name, **content):
            path = tmp_path / name
            path.write_text(json.dumps(content))
            return str(path)

        explicit = key("a.json", project_id="explicit")
        monkeypatch.setenv("AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY", key("b.json", project_id="env"))
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", key("c.json", project_id="adc"))
        assert auth.project_from_credentials(explicit) == "explicit"
        assert auth.project_from_credentials() == "env"
        monkeypatch.delenv("AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY")
        assert auth.project_from_credentials() == "adc"
        # Only the first file that is set is read, even without a project_id.
        user = key("d.json", type="authorized_user")
        assert auth.project_from_credentials(user) is None
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "missing.json"))
        assert auth.project_from_credentials() is None
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS")
        assert auth.project_from_credentials() is None

    def test_gcloud_project_memoised(self, fake_ee):
        with patch.object(auth, "_get_gcloud_project", return_value="gc") as gc:
            assert auth.resolve_project() == "gc"
            assert auth.resolve_project() == "gc"
        gc.assert_called_once()


class TestIdempotency:
    def test_same_signature_not_reinitialised(self, fake_ee):
        auth.setup_gee(project="p")
        fake_ee.data.is_initialized.return_value = True
        auth.setup_gee(project="p", workload_tag="t1")
        assert fake_ee.Initialize.call_count == 1
        fake_ee.data.setDefaultWorkloadTag.assert_called_once_with("t1")

    def test_changed_signature_reinitialises(self, fake_ee):
        auth.setup_gee(project="p")
        fake_ee.data.is_initialized.return_value = True
        auth.setup_gee(project="p", high_volume=True)
        assert fake_ee.Initialize.call_count == 2

    def test_ensure_gee_reads_config(self, fake_ee, key_file):
        cfg = types.SimpleNamespace(
            gee_project="cfg-p",
            gee_service_account_key=key_file,
            gee_high_volume=True,
            gee_workload_tag="tag-1",
        )
        auth.ensure_gee(cfg)
        auth.ensure_gee(cfg)
        fake_ee.data.is_initialized.return_value = True
        auth.ensure_gee(cfg)
        fake_ee.Initialize.assert_any_call("SA", project="cfg-p", opt_url=HV_URL)
        assert fake_ee.data.setDefaultWorkloadTag.call_count == 3


class TestInteractiveDetection:
    def test_override(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "1")
        assert auth.is_interactive(True) is True
        assert auth.is_interactive(False) is False

    def test_slurm(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "1")
        assert auth.is_interactive() is False

    def test_no_tty(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = False
        monkeypatch.setattr(sys, "stdin", fake_stdin)
        assert auth.is_interactive() is False
        fake_stdin.isatty.return_value = True
        assert auth.is_interactive() is True


class TestCheckInitialized:
    def test_not_initialized(self, fake_ee):
        assert auth.check_gee_initialized() is False
        fake_ee.Number.assert_not_called()

    def test_initialized(self, fake_ee):
        fake_ee.data.is_initialized.return_value = True
        assert auth.check_gee_initialized() is True
