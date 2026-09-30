"""
Run plans: frozen, hash-identified Agribound configurations proposed by the agent.

A :class:`Plan` is created by the ``propose_run`` tool. It stores the fully
validated :class:`~agribound.config.AgriboundConfig` as canonical JSON, a
fingerprint of the run's inputs, and a SHA-256 **plan hash** over both. The
confirmation gate (:mod:`agribound.agent.gate`) binds every approval to that
hash, and recomputes it immediately before execution, so an approval can
never authorise a different configuration, a different study-area geometry,
or a modified reference/raster file.

Input fingerprints
------------------
- ``study_area``: :func:`agribound._cache.aoi_fingerprint` (geometry-based for
  files, ``bbox:`` strings and WKT; the asset ID string for GEE assets).
- ``reference_boundaries`` and ``local_tif_path``: absolute path, size and
  modification time (the file contents are not hashed).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "agribound-agent-plan/1"
"""Identifier mixed into every plan hash; bump if the hashed content changes."""

THRESHOLD_FIELDS: tuple[str, ...] = (
    "lulc_filter",
    "lulc_crop_threshold",
    "lulc_nodata_policy",
    "lulc_dataset",
    "lulc_on_error",
    "aoi_selection",
    "min_field_area_m2",
    "simplify_tolerance",
    "sam_refine",
    "sam_min_crop_px",
    "sam_crop_padding",
    "cloud_cover_max",
    "cloud_score_threshold",
)
"""Configuration fields that decide which polygons are kept or how they are shaped.

Includes ``lulc_on_error`` (``"warn"`` keeps the unfiltered polygons when the
LULC filter fails), ``aoi_selection`` (which predictions are kept at, or cut
to, the study-area outline) and ``sam_refine`` (replaces engine geometries
with SAM masks that cover enough of them). A plan that sets any of them to a
value other than the package default carries an explicit warning, so the human
reviewer sees the change before approving.
"""

METHOD_FIELDS: tuple[str, ...] = (
    "usgs_allow_year_fallback",
    "composite_method",
    "date_range",
    "s2_cloud_mask",
    "naip_resolution_m",
    "bands",
    "tessera_version",
    "tessera_variant",
    "google_embedding_backend",
    "lulc_mode",
    "sam_backend",
    "sam_model",
)
"""Configuration fields that change the input data or the method.

For example ``usgs_allow_year_fallback`` lets the USGS NAIP Plus composite use
another year than the requested one, and ``lulc_mode`` switches the LULC
filter between Earth Engine zonal statistics and a downloaded raster. A plan
that sets any of them to a non-default value carries an explicit warning.
"""

DESTINATION_FIELDS: tuple[str, ...] = (
    "usgs_service_url",
    "export_method",
    "gcs_bucket",
)
"""Configuration fields that change which remote service is contacted or where data go.

``usgs_service_url`` names the ImageServer the USGS NAIP Plus composite is
downloaded from; ``export_method="gcs"`` and ``gcs_bucket`` send Earth Engine
exports to a Cloud Storage bucket. A plan that sets any of them to a
non-default value carries an explicit warning.
"""


# ---------------------------------------------------------------------------
# Canonical JSON and hashing
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Serialise *value* deterministically (sorted keys, no whitespace, ASCII)."""
    from agribound.provenance import to_jsonable

    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_plan_hash(config_json: str, inputs_json: str) -> str:
    """Return the SHA-256 hex digest that identifies a plan.

    Parameters
    ----------
    config_json : str
        Canonical JSON of the configuration (:func:`canonical_json`).
    inputs_json : str
        Canonical JSON of the input fingerprints (:func:`input_fingerprints`).

    Returns
    -------
    str
        64 hexadecimal characters.
    """
    payload = f"{PLAN_SCHEMA}\n{config_json}\n{inputs_json}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_signature(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        st = os.stat(resolved)
    except OSError:
        return {"path": str(resolved), "exists": False}
    return {
        "path": str(resolved),
        "exists": True,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def input_fingerprints(config: Any) -> dict[str, Any]:
    """Fingerprint the inputs a configuration reads (see the module docstring).

    Parameters
    ----------
    config : AgriboundConfig
        Validated configuration.

    Returns
    -------
    dict
        JSON-serialisable fingerprints of the study area, reference boundaries
        and local raster.
    """
    from agribound._cache import aoi_fingerprint

    return {
        "study_area": {
            "value": str(config.study_area or ""),
            "aoi_fingerprint": aoi_fingerprint(config),
        },
        "reference_boundaries": _file_signature(config.reference_boundaries),
        "local_tif_path": _file_signature(config.local_tif_path),
    }


# ---------------------------------------------------------------------------
# Defaults comparison
# ---------------------------------------------------------------------------


def config_defaults() -> dict[str, Any]:
    """Return the :class:`~agribound.config.AgriboundConfig` field defaults."""
    from agribound.config import AgriboundConfig

    defaults: dict[str, Any] = {}
    for f in dataclasses.fields(AgriboundConfig):
        if f.default is not dataclasses.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            defaults[f.name] = f.default_factory()
    return defaults


def non_default_fields(config_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return ``{field: {"default": ..., "value": ...}}`` for every field that differs.

    ``date_range`` tuples and lists compare equal.
    """
    from agribound.provenance import to_jsonable

    out: dict[str, dict[str, Any]] = {}
    for name, default in config_defaults().items():
        value = config_dict.get(name)
        if to_jsonable(value) != to_jsonable(default):
            out[name] = {"default": to_jsonable(default), "value": to_jsonable(value)}
    return out


def threshold_warnings(changes: dict[str, dict[str, Any]]) -> list[str]:
    """Warnings for changed :data:`THRESHOLD_FIELDS` and for any ``engine_params``."""
    warnings: list[str] = []
    for name in THRESHOLD_FIELDS:
        if name in changes:
            warnings.append(
                f"{name} is {changes[name]['value']!r} (package default "
                f"{changes[name]['default']!r}); this changes which polygons are kept or how "
                "they are shaped."
            )
    if "engine_params" in changes:
        warnings.append(
            f"engine_params are set to {changes['engine_params']['value']!r}; review them "
            "before approving (they are passed to the engine unchanged)."
        )
    return warnings


def method_warnings(changes: dict[str, dict[str, Any]]) -> list[str]:
    """Warnings for changed :data:`METHOD_FIELDS`."""
    return [
        f"{name} is {changes[name]['value']!r} (package default {changes[name]['default']!r}); "
        "this changes the input data or the method."
        for name in METHOD_FIELDS
        if name in changes
    ]


def destination_warnings(changes: dict[str, dict[str, Any]]) -> list[str]:
    """Warnings for changed :data:`DESTINATION_FIELDS`."""
    return [
        f"{name} is {changes[name]['value']!r} (package default {changes[name]['default']!r}); "
        "this changes which remote service is contacted or where data are written."
        for name in DESTINATION_FIELDS
        if name in changes
    ]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


_AGENT_MARK = "  | "

#: Unicode categories escaped by :func:`display_safe`: control characters (C0,
#: DEL, C1, including ESC, which starts terminal control sequences), format
#: characters (bidirectional overrides, zero-width characters) and the line
#: and paragraph separators.
_UNSAFE_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
_NAMED_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def display_safe(text: Any, *, keep_newlines: bool = False) -> str:
    """Return *text* with every character that could steer a terminal escaped.

    Characters in :data:`_UNSAFE_CATEGORIES` are replaced by a visible
    ``repr``-style escape (``"\\x1b"``, ``"\\u202e"``), so text shown to the
    reviewer cannot move the cursor, erase lines, reorder text or start a new
    line that looks like one of Agribound's own sections.

    Parameters
    ----------
    text : object
        Text to show (converted with :func:`str`).
    keep_newlines : bool
        Keep ``"\\n"`` line breaks (for multi-line reports); every other
        control character is still escaped.

    Returns
    -------
    str
    """
    out = []
    for ch in str(text):
        if keep_newlines and ch == "\n":
            out.append(ch)
        elif unicodedata.category(ch) in _UNSAFE_CATEGORIES:
            code = ord(ch)
            if ch in _NAMED_ESCAPES:
                out.append(_NAMED_ESCAPES[ch])
            elif code <= 0xFF:
                out.append(f"\\x{code:02x}")
            elif code <= 0xFFFF:
                out.append(f"\\u{code:04x}")
            else:
                out.append(f"\\U{code:08x}")
        else:
            out.append(ch)
    return "".join(out)


def _marked(text: str, *, first: str = "", cont: str = "") -> list[str]:
    """Split *text* into lines (every Unicode line break) and prefix each with ``"  | "``.

    Each line is passed through :func:`display_safe`, so escape sequences and
    other control characters are shown, not interpreted.
    """
    parts = str(text).splitlines() or [""]
    return [
        f"{_AGENT_MARK}{first if i == 0 else cont}{display_safe(line)}"
        for i, line in enumerate(parts)
    ]


@dataclass(frozen=True)
class Plan:
    """A proposed Agribound run, identified by the hash of its content.

    Instances are immutable; a changed configuration is a new plan with a new
    ``plan_id`` and ``plan_hash`` and needs its own approval.

    Attributes
    ----------
    plan_id : str
        ``"plan-"`` + the first 12 hex characters of *plan_hash*.
    config_json : str
        Canonical JSON of ``AgriboundConfig.to_dict()``.
    inputs_json : str
        Canonical JSON of :func:`input_fingerprints` at proposal time.
    plan_hash : str
        :func:`compute_plan_hash` of the two fields above.
    created_utc : str
        ISO-8601 creation time (UTC).
    rationale, limitations, alternatives
        The agent's explanation, shown to the reviewer.
    warnings : tuple of str
        Warnings generated by Agribound (not by the model).
    non_default_fields_json : str
        Canonical JSON of :func:`non_default_fields`.
    estimated_cost_json : str
        Canonical JSON of the size/resource estimate.
    yaml_path : str or None
        Where the plan's configuration YAML was written.
    """

    plan_id: str
    config_json: str
    inputs_json: str
    plan_hash: str
    created_utc: str
    rationale: str = ""
    limitations: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    non_default_fields_json: str = "{}"
    estimated_cost_json: str = "{}"
    yaml_path: str | None = None

    # -- accessors -------------------------------------------------------------

    @property
    def config(self) -> dict[str, Any]:
        """A fresh copy of the frozen configuration dictionary."""
        return json.loads(self.config_json)

    @property
    def inputs(self) -> dict[str, Any]:
        """A fresh copy of the input fingerprints recorded at proposal time."""
        return json.loads(self.inputs_json)

    @property
    def non_default_fields(self) -> dict[str, Any]:
        """Fields that differ from the package defaults."""
        return json.loads(self.non_default_fields_json)

    @property
    def estimated_cost(self) -> dict[str, Any]:
        """Size and resource estimate."""
        return json.loads(self.estimated_cost_json)

    def to_config(self) -> Any:
        """Rebuild the validated :class:`~agribound.config.AgriboundConfig`."""
        from agribound.config import AgriboundConfig

        return AgriboundConfig.from_dict(self.config)

    def current_hash(self) -> str:
        """Recompute the plan hash from the stored configuration and *current* inputs.

        Differs from :attr:`plan_hash` when the stored configuration was
        altered or when an input file or the study-area geometry changed.
        """
        current_inputs = canonical_json(input_fingerprints(self.to_config()))
        return compute_plan_hash(self.config_json, current_inputs)

    def stored_hash_is_consistent(self) -> bool:
        """True if :attr:`plan_hash` matches the stored JSON (no input re-check)."""
        return compute_plan_hash(self.config_json, self.inputs_json) == self.plan_hash

    def config_yaml(self) -> str:
        """The configuration as YAML, in :class:`AgriboundConfig` field order."""
        import yaml

        from agribound.config import AgriboundConfig

        data = self.config
        order = AgriboundConfig.field_names()
        ordered = {k: data[k] for k in order if k in data}
        return yaml.safe_dump(ordered, sort_keys=False, default_flow_style=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (used in tool results and the session transcript)."""
        return {
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "created_utc": self.created_utc,
            "config": self.config,
            "inputs": self.inputs,
            "rationale": self.rationale,
            "limitations": list(self.limitations),
            "alternatives": list(self.alternatives),
            "warnings": list(self.warnings),
            "non_default_fields": self.non_default_fields,
            "estimated_cost": self.estimated_cost,
            "yaml_path": self.yaml_path,
        }

    def render(self) -> str:
        """Human-readable description shown by the confirmation gate.

        The sections generated by Agribound (warnings, fields that differ from
        the defaults, estimate) come first. The text written by the agent
        (rationale, limitations, alternatives) follows, with **every** line
        prefixed by ``"  | "``, so agent text cannot pass itself off as an
        Agribound section. The full configuration YAML comes last. Control
        characters in any value (terminal escape sequences, line breaks in
        a single-line field, bidirectional overrides) are shown escaped
        (:func:`display_safe`), so no value can move the cursor, erase or
        reorder the screen, or add lines of its own.
        """
        cfg = self.config
        safe = display_safe
        lines = [
            f"Plan {self.plan_id} (sha256 {self.plan_hash})",
            f"  source={safe(cfg.get('source'))}  engine={safe(cfg.get('engine'))}  "
            f"year={safe(cfg.get('year'))}",
            f"  study_area={safe(cfg.get('study_area'))}",
            f"  output_path={safe(cfg.get('output_path'))}",
            "",
            "Warnings (from Agribound):",
        ]
        lines += [f"  - {safe(item)}" for item in self.warnings] or ["  (none)"]
        changes = self.non_default_fields
        lines += ["", "Fields that differ from the AgriboundConfig defaults:"]
        if changes:
            for name, change in changes.items():
                lines.append(
                    f"  {safe(name)}: {safe(repr(change['default']))} -> "
                    f"{safe(repr(change['value']))}"
                )
        else:
            lines.append("  (none)")
        cost = self.estimated_cost
        if cost:
            lines += ["", "Estimate:"]
            lines += [f"  {safe(key)}: {safe(value)}" for key, value in cost.items()]
        if self.rationale or self.limitations or self.alternatives:
            lines += [
                "",
                f"Text written by the agent (not checked by Agribound; lines marked "
                f"{_AGENT_MARK.strip()!r}):",
            ]
            if self.rationale:
                lines.append("  Rationale:")
                lines += _marked(self.rationale)
            if self.limitations:
                lines.append("  Limitations:")
                for item in self.limitations:
                    lines += _marked(item, first="- ", cont="  ")
            if self.alternatives:
                lines.append("  Alternatives:")
                for item in self.alternatives:
                    lines += _marked(item, first="- ", cont="  ")
        lines += ["", "Full configuration (YAML):"]
        lines += [safe(line) for line in self.config_yaml().rstrip().split("\n")]
        return "\n".join(lines)


def make_plan(
    config: Any,
    *,
    rationale: str = "",
    limitations: tuple[str, ...] | list[str] = (),
    alternatives: tuple[str, ...] | list[str] = (),
    warnings: tuple[str, ...] | list[str] = (),
    estimated_cost: dict[str, Any] | None = None,
    yaml_path: str | None = None,
) -> Plan:
    """Freeze a validated configuration into a :class:`Plan`.

    Parameters
    ----------
    config : AgriboundConfig
        Validated configuration.
    rationale, limitations, alternatives : str / sequence of str
        The agent's explanation (shown to the reviewer; not hashed).
    warnings : sequence of str
        Agribound-generated warnings (not hashed).
    estimated_cost : dict or None
        Size/resource estimate (not hashed).
    yaml_path : str or None
        Path the configuration YAML is (or will be) written to.

    Returns
    -------
    Plan
    """
    config_dict = config.to_dict()
    config_json = canonical_json(config_dict)
    inputs_json = canonical_json(input_fingerprints(config))
    plan_hash = compute_plan_hash(config_json, inputs_json)
    return Plan(
        plan_id=f"plan-{plan_hash[:12]}",
        config_json=config_json,
        inputs_json=inputs_json,
        plan_hash=plan_hash,
        created_utc=_utc_now(),
        rationale=str(rationale or ""),
        limitations=tuple(str(x) for x in limitations),
        alternatives=tuple(str(x) for x in alternatives),
        warnings=tuple(str(x) for x in warnings),
        non_default_fields_json=canonical_json(non_default_fields(config_dict)),
        estimated_cost_json=canonical_json(estimated_cost or {}),
        yaml_path=yaml_path,
    )


def write_plan_yaml(plan: Plan, path: str | Path) -> Path:
    """Write the plan's configuration as a YAML file for ``agribound delineate --config``.

    The file starts with comment lines naming the plan ID and hash; YAML
    loaders ignore them, so :meth:`AgriboundConfig.from_yaml` reads the file
    unchanged.

    Parameters
    ----------
    plan : Plan
        The plan.
    path : str or Path
        Destination file.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Agribound agent plan {plan.plan_id}\n"
        f"# plan sha256: {plan.plan_hash}\n"
        "# Run with: agribound delineate --config <this file>\n"
    )
    path.write_text(header + plan.config_yaml())
    return path


__all__ = [
    "DESTINATION_FIELDS",
    "METHOD_FIELDS",
    "PLAN_SCHEMA",
    "THRESHOLD_FIELDS",
    "Plan",
    "canonical_json",
    "compute_plan_hash",
    "config_defaults",
    "destination_warnings",
    "display_safe",
    "input_fingerprints",
    "make_plan",
    "method_warnings",
    "non_default_fields",
    "threshold_warnings",
    "write_plan_yaml",
]
