"""
Results versions: releases that change a component's output for the same configuration.

:func:`agribound.provenance.config_hash` covers the configuration, not the
code. When a release changes what a component produces for an unchanged
configuration, its number in :data:`RESULTS_VERSIONS` is raised. The pipeline
records the numbers that apply to a run (:func:`results_versions`) in the
provenance fact ``results_versions``, and an existing output is reused only
when its recorded numbers equal the current ones
(:func:`agribound.provenance.reuse_mismatch`). A record without the fact, or
without a component, counts as version 1 for it.
"""

from __future__ import annotations

from typing import Any

RESULTS_VERSIONS: dict[str, int] = {
    "embedding": 2,
    "sam_refine": 2,
}
"""Version of each component's results; 1 = agribound <= 1.0.0.

Bump a component's number when its output changes for the same
configuration, so that outputs of earlier releases are no longer reused.

- ``"embedding"``: the embedding engine's clustering (2: agribound 1.0.1).
- ``"sam_refine"``: SAM refinement (2: agribound 1.0.1).

A record without an entry for a component counts as version 1, so a
component added later starts at 1 when its output is unchanged, or at 2
when the release that adds it also changes its output; :func:`results_versions`
must say when it applies.
"""


def _ensemble_member_engines(config: Any) -> list[str]:
    """Engine names of the configured ensemble members (malformed entries ignored)."""
    members = (getattr(config, "engine_params", None) or {}).get("engines")
    if not isinstance(members, list | tuple):
        from agribound.registry import ENSEMBLE_DEFAULT_MEMBERS

        return list(ENSEMBLE_DEFAULT_MEMBERS)
    names = []
    for entry in members:
        name = entry.get("engine") if isinstance(entry, dict) else entry
        if isinstance(name, str):
            names.append(name.lower().strip())
    return names


def results_versions(config: Any) -> dict[str, int]:
    """Return the :data:`RESULTS_VERSIONS` entries of the components *config* runs.

    - ``"embedding"``: ``config.engine == "embedding"``, or an ensemble
      member is the embedding engine (no source supports both engines
      today; checked so that a registry change cannot make a stale output
      reusable).
    - ``"sam_refine"``: ``config.sam_refine`` (which also absorbs the legacy
      ``engine_params["sam_refine"]``). The pipeline refines the output of
      every engine, including an ensemble's; the embedding engine refines
      its own polygons.

    Parameters
    ----------
    config : AgriboundConfig
        Configuration of the run.

    Returns
    -------
    dict
        Component name to version number, in :data:`RESULTS_VERSIONS` order;
        empty when no versioned component runs.
    """
    engine = str(getattr(config, "engine", "") or "").lower().strip()
    engine_params = getattr(config, "engine_params", None) or {}
    applies = {
        "embedding": engine == "embedding"
        or (engine == "ensemble" and "embedding" in _ensemble_member_engines(config)),
        "sam_refine": bool(getattr(config, "sam_refine", False))
        or bool(engine_params.get("sam_refine", False)),
    }
    return {name: version for name, version in RESULTS_VERSIONS.items() if applies[name]}


__all__ = ["RESULTS_VERSIONS", "results_versions"]
