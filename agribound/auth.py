"""
Google Earth Engine authentication helpers.

:func:`setup_gee` initialises the Earth Engine client with, in order:

1. an explicit service-account key file,
2. the key named by ``AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY``,
3. persistent credentials from ``earthengine authenticate`` (read by
   ``ee.Initialize``),
4. Application Default Credentials (``google.auth.default`` with the Earth
   Engine scopes; honours ``GOOGLE_APPLICATION_CREDENTIALS``),
5. interactive ``ee.Authenticate()`` -- only in interactive sessions. Under
   SLURM, without a TTY, or with ``interactive=False`` a :class:`RuntimeError`
   with instructions is raised instead, so batch jobs never block on a browser
   prompt.

:func:`ensure_gee` is the idempotent entry point used by every Earth Engine
call site; it reads the relevant fields of an ``AgriboundConfig``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: OAuth scopes requested for Application Default Credentials.
EE_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/cloud-platform",
)

#: Environment variable naming a service-account JSON key.
SERVICE_ACCOUNT_ENV = "AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY"

# Signature (project, key, high_volume) of the last successful initialisation.
_INIT_STATE: dict[str, Any] = {"signature": None, "method": None, "gcloud_project": None}

_NON_INTERACTIVE_HELP = (
    "Earth Engine credentials were not found and interactive authentication is "
    "disabled in this session (non-interactive: SLURM job, no TTY, or "
    "interactive=False). Use one of:\n"
    "  1. A service-account key: gee_service_account_key='/path/key.json' "
    "(or --gee-service-account-key, or env AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY). "
    "The service account needs roles/serviceusage.serviceUsageConsumer and an "
    "Earth Engine role (roles/earthengine.writer for export tasks).\n"
    "  2. Application Default Credentials: 'gcloud auth application-default login "
    "--scopes=https://www.googleapis.com/auth/earthengine,"
    "https://www.googleapis.com/auth/cloud-platform', or set "
    "GOOGLE_APPLICATION_CREDENTIALS to a key file.\n"
    "  3. Run 'earthengine authenticate' once on a login node; the stored "
    "credentials in ~/.config/earthengine are then reused."
)


def _get_gcloud_project() -> str | None:
    """Read the active project from gcloud config, if available."""
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        project = result.stdout.strip()
        if project and project != "(unset)":
            logger.info("Using GEE project from gcloud config: %s", project)
            return project
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def is_interactive(interactive: bool | None = None) -> bool:
    """Return whether interactive browser authentication may be attempted.

    Parameters
    ----------
    interactive : bool or None
        Explicit override. *None* auto-detects: non-interactive when
        ``SLURM_JOB_ID`` is set or standard input is not a TTY.
    """
    if interactive is not None:
        return bool(interactive)
    if os.environ.get("SLURM_JOB_ID"):
        return False
    try:
        return bool(sys.stdin is not None and sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _import_ee():
    try:
        import ee
    except ImportError:
        raise ImportError(
            "earthengine-api is required for GEE operations. "
            'Install with: pip install "agribound[gee]"'
        ) from None
    return ee


def _project_from_key(key_path: Path) -> str | None:
    try:
        with open(key_path) as f:
            return json.load(f).get("project_id")
    except (OSError, ValueError, AttributeError):
        return None


def project_from_credentials(service_account_key: str | None = None) -> str | None:
    """Return the ``project_id`` of the credentials file that will be used.

    The file is *service_account_key*, else ``AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY``
    (the key :func:`setup_gee` initialises with), else
    ``GOOGLE_APPLICATION_CREDENTIALS`` (read by Application Default
    Credentials). Only the first of these that is set is read. Returns *None*
    when that file is missing, unreadable or has no ``project_id`` (user and
    external-account credential files have none).

    Parameters
    ----------
    service_account_key : str or None
        Explicit service-account key path (``gee_service_account_key``).
    """
    key = (
        service_account_key
        or os.environ.get(SERVICE_ACCOUNT_ENV)
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or None
    )
    if not key:
        return None
    project = _project_from_key(Path(key).expanduser())
    if project:
        logger.info("Using GEE project %s from the project_id of %s", project, key)
    return project or None


def resolve_project(project: str | None = None) -> str | None:
    """Resolve a GEE project: argument, then ``GEE_PROJECT``, then gcloud config.

    A project found through ``gcloud`` is remembered for the rest of the
    process, so repeated calls do not spawn ``gcloud`` again.
    """
    if project:
        return project
    env_project = os.environ.get("GEE_PROJECT")
    if env_project:
        return env_project
    if _INIT_STATE.get("gcloud_project"):
        return _INIT_STATE["gcloud_project"]
    found = _get_gcloud_project()
    if found:
        _INIT_STATE["gcloud_project"] = found
    return found


def setup_gee(
    project: str | None = None,
    service_account_key: str | None = None,
    high_volume: bool = False,
    workload_tag: str | None = None,
    interactive: bool | None = None,
) -> None:
    """Authenticate and initialise Google Earth Engine.

    Credentials are tried in the order described in the module docstring.
    Calling again with the same project, key and endpoint does not
    re-initialise the client (only the workload tag is updated).

    Parameters
    ----------
    project : str or None
        GEE Cloud project ID. If *None*: ``GEE_PROJECT``, then ``gcloud config``,
        then the ``project_id`` of the service-account key, or of the
        ``GOOGLE_APPLICATION_CREDENTIALS`` file when no key is given (see
        :func:`project_from_credentials`).
    service_account_key : str or None
        Path to a service-account JSON key. If *None*, the
        ``AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY`` environment variable is used.
    high_volume : bool
        Connect to the high-volume endpoint
        (``ee.data.HIGH_VOLUME_API_BASE_URL``), intended for many parallel
        small requests.
    workload_tag : str or None
        Default workload tag for all subsequent Earth Engine requests
        (``ee.data.setDefaultWorkloadTag``), for EECU accounting.
    interactive : bool or None
        Allow ``ee.Authenticate()`` when no credentials are found. *None*
        auto-detects (see :func:`is_interactive`).

    Raises
    ------
    ImportError
        If ``earthengine-api`` is not installed.
    FileNotFoundError
        If the service-account key file does not exist.
    ValueError
        If no project ID can be determined.
    RuntimeError
        If authentication fails, or credentials are missing in a
        non-interactive session.

    Examples
    --------
    >>> from agribound.auth import setup_gee
    >>> setup_gee(project="my-gee-project")
    """
    ee = _import_ee()

    key = service_account_key or os.environ.get(SERVICE_ACCOUNT_ENV) or None
    key_path = Path(key).expanduser() if key else None
    if key_path is not None and not key_path.exists():
        raise FileNotFoundError(f"Service account key file not found: {key_path}")

    project = resolve_project(project)
    if project is None:
        project = project_from_credentials(str(key_path) if key_path is not None else None)
    if project is None:
        raise ValueError(
            "A GEE project ID is required. Provide it via one of:\n"
            "  1. The 'project' argument / gee_project / --gee-project CLI flag\n"
            "  2. The GEE_PROJECT environment variable\n"
            "  3. gcloud config: gcloud config set project YOUR_PROJECT\n"
            "  4. The project_id of the service-account key (gee_service_account_key, "
            "AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or GOOGLE_APPLICATION_CREDENTIALS)\n"
            "You can find your project ID at https://console.cloud.google.com/"
        )

    url = ee.data.HIGH_VOLUME_API_BASE_URL if high_volume else None
    signature = (project, str(key_path) if key_path else None, bool(high_volume))

    if _INIT_STATE["signature"] == signature and _ee_is_initialized(ee):
        logger.debug("Earth Engine already initialised (%s)", _INIT_STATE["method"])
    else:
        method = _initialize(ee, project, key_path, url, interactive)
        _INIT_STATE["signature"] = signature
        _INIT_STATE["method"] = method
        logger.info(
            "GEE initialised with %s (project=%s%s)",
            method,
            project,
            ", high-volume endpoint" if high_volume else "",
        )

    if workload_tag:
        ee.data.setDefaultWorkloadTag(workload_tag)
        logger.info("GEE default workload tag: %s", workload_tag)


def _ee_is_initialized(ee: Any) -> bool:
    try:
        return bool(ee.data.is_initialized())
    except Exception:
        return False


def _initialize(
    ee: Any,
    project: str,
    key_path: Path | None,
    url: str | None,
    interactive: bool | None,
) -> str:
    """Initialise Earth Engine and return the name of the credential method used."""
    if key_path is not None:
        try:
            credentials = ee.ServiceAccountCredentials(None, key_file=str(key_path))
            ee.Initialize(credentials, project=project, opt_url=url)
        except Exception as exc:
            raise RuntimeError(
                f"GEE initialisation with service-account key {key_path} failed: {exc}"
            ) from exc
        return "service-account key"

    errors: list[str] = []

    # Persistent credentials (earthengine authenticate)
    try:
        ee.Initialize(project=project, opt_url=url)
        return "persistent credentials"
    except Exception as exc:
        errors.append(f"persistent credentials: {exc}")
        logger.debug("GEE persistent credentials unavailable: %s", exc)

    # Application Default Credentials
    try:
        import google.auth

        credentials, _adc_project = google.auth.default(scopes=list(EE_SCOPES))
        ee.Initialize(credentials, project=project, opt_url=url)
        return "application default credentials"
    except Exception as exc:
        errors.append(f"application default credentials: {exc}")
        logger.debug("GEE application default credentials unavailable: %s", exc)

    details = "\n".join(f"  - {e}" for e in errors)
    if not is_interactive(interactive):
        raise RuntimeError(f"{_NON_INTERACTIVE_HELP}\nAttempts:\n{details}")

    # Interactive browser authentication
    try:
        ee.Authenticate()
        ee.Initialize(project=project, opt_url=url)
    except Exception as exc:
        raise RuntimeError(
            f"GEE authentication failed: {exc}\n\n"
            "Troubleshooting steps:\n"
            "1. Ensure you have a GEE-enabled Google Cloud project\n"
            "2. Run 'earthengine authenticate' in your terminal\n"
            "3. Check that your project ID is correct\n"
            "4. For CI/server/HPC environments, use a service account key or ADC\n"
            f"Earlier attempts:\n{details}"
        ) from exc
    return "interactive authentication"


def ensure_gee(config: Any) -> None:
    """Initialise Earth Engine for *config* if needed (idempotent).

    Reads ``gee_project``, ``gee_service_account_key``, ``gee_high_volume``
    and ``gee_workload_tag`` from the configuration and calls
    :func:`setup_gee`.

    Parameters
    ----------
    config : AgriboundConfig
        Pipeline configuration.
    """
    setup_gee(
        project=getattr(config, "gee_project", None),
        service_account_key=getattr(config, "gee_service_account_key", None),
        high_volume=bool(getattr(config, "gee_high_volume", False)),
        workload_tag=getattr(config, "gee_workload_tag", None),
        interactive=None,
    )


def check_gee_initialized() -> bool:
    """Check if GEE is initialised and reachable.

    Returns
    -------
    bool
        *True* if the client is initialised and a trivial request succeeds.
    """
    try:
        import ee

        if not ee.data.is_initialized():
            return False
        # A lightweight call to test the connection
        ee.Number(1).getInfo()
        return True
    except Exception:
        return False


def _reset_state() -> None:
    """Forget the cached initialisation state (for tests)."""
    _INIT_STATE["signature"] = None
    _INIT_STATE["method"] = None
    _INIT_STATE["gcloud_project"] = None


__all__ = [
    "EE_SCOPES",
    "SERVICE_ACCOUNT_ENV",
    "check_gee_initialized",
    "ensure_gee",
    "is_interactive",
    "resolve_project",
    "setup_gee",
]
