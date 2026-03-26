"""
Google Earth Engine authentication helper.

Provides a unified function to authenticate and initialize GEE, handling
the common failure modes with clear error messages.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_gee(
    project: str | None = None,
    service_account_key: str | None = None,
) -> None:
    """Authenticate and initialize Google Earth Engine.

    Attempts authentication in the following order:

    1. Service account key file (if *service_account_key* is provided) — for
       CI/server environments.
    2. Existing credentials from a prior ``ee.Authenticate()`` call.
    3. Interactive browser-based authentication.

    Parameters
    ----------
    project : str or None
        GEE Cloud project ID (e.g. ``"my-gee-project"``). If *None*, reads
        from the ``GEE_PROJECT`` environment variable.
    service_account_key : str or None
        Path to a GEE service account JSON key file. Useful for headless or
        CI environments.

    Raises
    ------
    ImportError
        If ``earthengine-api`` is not installed.
    ValueError
        If no project ID is provided and ``GEE_PROJECT`` is unset.
    RuntimeError
        If authentication fails.

    Examples
    --------
    >>> from agribound.auth import setup_gee
    >>> setup_gee(project="my-gee-project")
    """
    try:
        import ee
    except ImportError:
        raise ImportError(
            "earthengine-api is required for GEE operations. "
            "Install with: pip install agribound[gee]"
        )

    # Resolve project ID
    if project is None:
        project = os.environ.get("GEE_PROJECT")
    if project is None:
        raise ValueError(
            "A GEE project ID is required. Pass it via the 'project' argument "
            "or set the GEE_PROJECT environment variable. "
            "You can find your project ID at https://console.cloud.google.com/"
        )

    # Strategy 1: Service account key
    if service_account_key is not None:
        key_path = Path(service_account_key)
        if not key_path.exists():
            raise FileNotFoundError(
                f"Service account key file not found: {key_path}"
            )
        logger.info("Authenticating GEE with service account key: %s", key_path)
        credentials = ee.ServiceAccountCredentials(None, str(key_path))
        ee.Initialize(credentials=credentials, project=project)
        logger.info("GEE initialized with service account (project=%s)", project)
        return

    # Strategy 2: Try existing credentials
    try:
        ee.Initialize(project=project)
        logger.info("GEE initialized with existing credentials (project=%s)", project)
        return
    except Exception:
        logger.debug("No existing GEE credentials found, attempting authentication")

    # Strategy 3: Interactive browser auth
    try:
        ee.Authenticate()
        ee.Initialize(project=project)
        logger.info("GEE initialized after browser authentication (project=%s)", project)
    except Exception as exc:
        raise RuntimeError(
            f"GEE authentication failed: {exc}\n\n"
            "Troubleshooting steps:\n"
            "1. Ensure you have a GEE-enabled Google Cloud project\n"
            "2. Run 'earthengine authenticate' in your terminal\n"
            "3. Check that your project ID is correct\n"
            "4. For CI/server environments, use a service account key"
        ) from exc


def check_gee_initialized() -> bool:
    """Check if GEE is already initialized.

    Returns
    -------
    bool
        *True* if ``ee.Initialize()`` has been called successfully.
    """
    try:
        import ee

        # A lightweight call to test initialization
        ee.Number(1).getInfo()
        return True
    except Exception:
        return False
