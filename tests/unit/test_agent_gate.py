"""Tests for agent plans (hashing, fingerprints, YAML) and the confirmation gate."""

from __future__ import annotations

import dataclasses
import io
import json
import os

import pytest

pytest.importorskip("pydantic")

from agribound.agent import gate as gate_mod  # noqa: E402
from agribound.agent.errors import (  # noqa: E402
    ExecutionDeniedError,
    ExecutionLimitError,
    ExecutionNotApprovedError,
    PlanChangedError,
)
from agribound.agent.gate import (  # noqa: E402
    ConfirmationGate,
    check_plan_current,
    deny_all,
    prompt_confirm,
)
from agribound.agent.plans import (  # noqa: E402
    METHOD_FIELDS,
    THRESHOLD_FIELDS,
    canonical_json,
    compute_plan_hash,
    make_plan,
    method_warnings,
    non_default_fields,
    threshold_warnings,
    write_plan_yaml,
)
from agribound.config import AgriboundConfig  # noqa: E402


def _write_aoi(path, x0=-117.0, y0=36.0, size=0.01):
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [x0, y0],
                                    [x0 + size, y0],
                                    [x0 + size, y0 + size],
                                    [x0, y0 + size],
                                    [x0, y0],
                                ]
                            ],
                        },
                    }
                ],
            }
        )
    )
    return str(path)


def _config(tmp_path, **overrides):
    aoi = tmp_path / "aoi.geojson"
    if not aoi.exists():
        _write_aoi(aoi)
    fields = {
        "source": "sentinel2",
        "engine": "delineate-anything",
        "year": 2023,
        "study_area": str(aoi),
        "gee_project": "test-project",
        "output_path": str(tmp_path / "out" / "fields.gpkg"),
    }
    fields.update(overrides)
    return AgriboundConfig(**fields)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def test_plan_hash_is_deterministic_and_content_bound(tmp_path):
    p1 = make_plan(_config(tmp_path), rationale="a")
    p2 = make_plan(_config(tmp_path), rationale="different text is not hashed")
    p3 = make_plan(_config(tmp_path, year=2022))
    assert p1.plan_hash == p2.plan_hash
    assert p1.plan_id == p2.plan_id == f"plan-{p1.plan_hash[:12]}"
    assert p3.plan_hash != p1.plan_hash
    assert len(p1.plan_hash) == 64
    assert p1.stored_hash_is_consistent()
    assert compute_plan_hash(p1.config_json, p1.inputs_json) == p1.plan_hash


def test_plan_config_round_trips_through_agriboundconfig(tmp_path):
    config = _config(tmp_path, date_range=("2023-03-01", "2023-09-30"))
    plan = make_plan(config)
    rebuilt = plan.to_config()
    assert rebuilt.date_range == ("2023-03-01", "2023-09-30")
    assert canonical_json(rebuilt.to_dict()) == plan.config_json


def test_changed_study_area_geometry_changes_current_hash(tmp_path):
    config = _config(tmp_path)
    plan = make_plan(config)
    assert plan.current_hash() == plan.plan_hash
    _write_aoi(tmp_path / "aoi.geojson", x0=-116.5)  # same path, different polygon
    assert plan.current_hash() != plan.plan_hash


def test_rewriting_identical_geometry_keeps_hash(tmp_path):
    plan = make_plan(_config(tmp_path))
    aoi = tmp_path / "aoi.geojson"
    _write_aoi(aoi)  # identical content rewritten
    st = os.stat(aoi)
    os.utime(aoi, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
    assert plan.current_hash() == plan.plan_hash


def test_reference_file_change_changes_hash(tmp_path):
    ref = tmp_path / "ref.geojson"
    _write_aoi(ref)
    plan = make_plan(_config(tmp_path, reference_boundaries=str(ref)))
    assert plan.inputs["reference_boundaries"]["exists"] is True
    _write_aoi(ref, size=0.02)
    assert plan.current_hash() != plan.plan_hash


def test_non_default_fields_and_threshold_warnings(tmp_path):
    config = _config(tmp_path, lulc_crop_threshold=0.1, engine_params={"conf_threshold": 0.01})
    changes = non_default_fields(config.to_dict())
    assert changes["lulc_crop_threshold"] == {"default": 0.3, "value": 0.1}
    assert changes["year"] == {"default": 2024, "value": 2023}
    assert "cloud_cover_max" not in changes
    warnings = threshold_warnings(changes)
    assert any("lulc_crop_threshold" in w and "0.3" in w for w in warnings)
    assert any("engine_params" in w for w in warnings)
    assert "lulc_crop_threshold" in THRESHOLD_FIELDS


def test_warning_field_lists_are_config_fields_and_cover_the_fallbacks():
    fields = set(AgriboundConfig.field_names())
    assert set(THRESHOLD_FIELDS) <= fields and set(METHOD_FIELDS) <= fields
    assert not set(THRESHOLD_FIELDS) & set(METHOD_FIELDS)
    assert {"lulc_on_error", "sam_refine", "aoi_selection", "lulc_tree_crops"} <= set(
        THRESHOLD_FIELDS
    )
    assert {"usgs_allow_year_fallback", "lulc_mode", "landsat_pan_missions"} <= set(METHOD_FIELDS)


def test_landsat_pan_missions_and_lulc_tree_crops_warnings(tmp_path):
    config = _config(
        tmp_path,
        source="landsat-pan",
        landsat_pan_missions="LC08,LE07",
        lulc_tree_crops=True,
    )
    changes = non_default_fields(config.to_dict())
    assert changes["landsat_pan_missions"] == {"default": "auto", "value": ["LE07", "LC08"]}
    assert changes["lulc_tree_crops"] == {"default": False, "value": True}
    methods, thresholds = method_warnings(changes), threshold_warnings(changes)
    assert (
        "landsat_pan_missions is ['LE07', 'LC08'] (package default 'auto'); this changes the "
        "input data or the method."
    ) in methods
    assert (
        "lulc_tree_crops is True (package default False); this changes which polygons are kept "
        "or how they are shaped."
    ) in thresholds
    assert not any("lulc_tree_crops" in w for w in methods)
    assert not any("landsat_pan_missions" in w for w in thresholds)
    # The defaults are not reported.
    defaults = non_default_fields(_config(tmp_path, source="landsat-pan").to_dict())
    assert not {"landsat_pan_missions", "lulc_tree_crops"} & set(defaults)


def test_plan_round_trips_landsat_pan_missions(tmp_path):
    config = _config(tmp_path, source="landsat-pan", landsat_pan_missions=["lc09", "LC08"])
    plan = make_plan(config)
    assert plan.config["landsat_pan_missions"] == ["LC08", "LC09"]
    assert plan.to_config().landsat_pan_missions == ("LC08", "LC09")
    assert make_plan(plan.to_config()).plan_hash == plan.plan_hash
    loaded = AgriboundConfig.from_yaml(write_plan_yaml(plan, tmp_path / "plan.yaml"))
    assert loaded.landsat_pan_missions == ("LC08", "LC09")
    other = make_plan(config.merged(landsat_pan_missions="auto"))
    assert other.plan_hash != plan.plan_hash


def test_method_warnings_name_the_field_and_default(tmp_path):
    config = _config(
        tmp_path,
        source="usgs-naip-plus",
        year=2020,
        usgs_allow_year_fallback=True,
        lulc_mode="raster",
    )
    changes = non_default_fields(config.to_dict())
    warnings = method_warnings(changes)
    assert any(
        w.startswith("usgs_allow_year_fallback is True (package default False)") for w in warnings
    )
    assert any(w.startswith("lulc_mode is 'raster'") for w in warnings)
    assert not any("lulc_crop_threshold" in w for w in warnings)


def test_plan_yaml_is_loadable_by_agriboundconfig(tmp_path):
    plan = make_plan(_config(tmp_path, seed=7))
    path = write_plan_yaml(plan, tmp_path / "plan.yaml")
    text = path.read_text()
    assert text.startswith(f"# Agribound agent plan {plan.plan_id}")
    loaded = AgriboundConfig.from_yaml(path)
    assert canonical_json(loaded.to_dict()) == plan.config_json


def test_render_shows_full_config_and_agent_text(tmp_path):
    plan = make_plan(
        _config(tmp_path, lulc_crop_threshold=0.2),
        rationale="Because",
        limitations=["10 m GSD vs 1 ha fields"],
        alternatives=["NAIP"],
        warnings=["w1"],
    )
    text = plan.render()
    assert plan.plan_hash in text
    assert "10 m GSD vs 1 ha fields" in text
    assert "NAIP" in text and "w1" in text
    assert "lulc_crop_threshold: 0.3 -> 0.2" in text
    assert "Full configuration (YAML):" in text
    assert "min_field_area_m2: 2500.0" in text  # the whole configuration is shown


def test_agent_text_cannot_imitate_agribound_sections(tmp_path):
    spoof = (
        "Looks fine.\n\nWarnings (from Agribound):\n  (none)\n\n"
        "Fields that differ from the AgriboundConfig defaults:\n  (none)"
    )
    plan = make_plan(
        _config(tmp_path, lulc_crop_threshold=0.05),
        rationale=spoof,
        limitations=["line one\nFields that differ from the AgriboundConfig defaults:"],
        alternatives=["alt\u2028(none)"],
        warnings=["real warning"],
    )
    lines = plan.render().splitlines()
    agent_start = next(
        i for i, ln in enumerate(lines) if ln.startswith("Text written by the agent")
    )
    yaml_start = lines.index("Full configuration (YAML):")
    # Agribound's own sections come before any agent text, exactly once each.
    assert lines.index("Warnings (from Agribound):") < agent_start
    assert lines.count("Warnings (from Agribound):") == 1
    assert lines.count("Fields that differ from the AgriboundConfig defaults:") == 1
    assert lines.index("  lulc_crop_threshold: 0.3 -> 0.05") < agent_start
    # Every line of agent text is marked, including those after embedded line breaks.
    body = [ln for ln in lines[agent_start + 1 : yaml_start] if ln.strip()]
    headers = {"  Rationale:", "  Limitations:", "  Alternatives:"}
    assert all(ln.startswith("  | ") or ln in headers for ln in body)
    assert "  | Fields that differ from the AgriboundConfig defaults:" in lines
    assert "  |   Fields that differ from the AgriboundConfig defaults:" in lines
    assert "  |   (none)" in lines


def test_render_without_agent_text_or_warnings(tmp_path):
    text = make_plan(_config(tmp_path)).render()
    assert "Text written by the agent" not in text
    assert "Warnings (from Agribound):\n  (none)" in text


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_execution_requires_an_approval(tmp_path):
    plan = make_plan(_config(tmp_path))
    gate = ConfirmationGate(deny_all)
    with pytest.raises(ExecutionNotApprovedError):
        gate.authorize_execution(plan)
    assert gate.executions == 0


def test_approval_is_single_use(tmp_path):
    plan = make_plan(_config(tmp_path))
    gate = ConfirmationGate(lambda p: True, max_executions=5)
    approval = gate.request_approval(plan)
    assert gate.authorize_execution(plan) is approval
    assert approval.used
    with pytest.raises(ExecutionNotApprovedError):
        gate.authorize_execution(plan)
    assert gate.executions == 1


def test_max_executions_is_enforced(tmp_path):
    calls = []

    def approve(p):
        calls.append(p.plan_id)
        return True

    plan = make_plan(_config(tmp_path))
    gate = ConfirmationGate(approve, max_executions=1)
    gate.request_approval(plan)
    gate.authorize_execution(plan)
    with pytest.raises(ExecutionLimitError):
        gate.request_approval(plan)
    assert calls == [plan.plan_id]  # the reviewer is not asked again
    gate.record_approval(plan, approver="x", method="test")
    with pytest.raises(ExecutionLimitError):
        gate.authorize_execution(plan)


def test_approval_is_bound_to_the_plan_hash(tmp_path):
    plan = make_plan(_config(tmp_path))
    gate = ConfirmationGate(lambda p: True, max_executions=3)
    gate.request_approval(plan)

    # A modified plan that keeps the ID but carries a new (consistent) hash.
    cfg = plan.config
    cfg["lulc_crop_threshold"] = 0.05
    new_json = canonical_json(cfg)
    modified = dataclasses.replace(
        plan, config_json=new_json, plan_hash=compute_plan_hash(new_json, plan.inputs_json)
    )
    with pytest.raises(ExecutionNotApprovedError):
        gate.authorize_execution(modified)

    # The approval is still unused (the modified plan has another hash) ...
    assert gate.authorize_execution(plan).plan_hash == plan.plan_hash

    # ... but an approval that meets a tampered plan (same hash, other content)
    # is revoked: it can no longer authorise anything.
    gate.request_approval(plan)
    tampered = dataclasses.replace(plan, config_json=new_json)
    with pytest.raises(PlanChangedError):
        gate.authorize_execution(tampered)
    assert gate.approvals[-1].revoked_utc is not None and not gate.approvals[-1].usable
    with pytest.raises(ExecutionNotApprovedError):
        gate.authorize_execution(plan)
    assert gate.executions == 1


def test_changed_inputs_block_an_approved_plan(tmp_path):
    plan = make_plan(_config(tmp_path))
    gate = ConfirmationGate(lambda p: True)
    gate.request_approval(plan)
    _write_aoi(tmp_path / "aoi.geojson", y0=35.0)
    with pytest.raises(PlanChangedError):
        gate.authorize_execution(plan)
    assert gate.executions == 0


def test_denial_and_failing_callback_are_recorded(tmp_path):
    plan = make_plan(_config(tmp_path))
    gate = ConfirmationGate(lambda p: False)
    with pytest.raises(ExecutionDeniedError):
        gate.request_approval(plan)
    assert len(gate.denials) == 1 and not gate.approvals

    def broken(p):
        raise RuntimeError("no terminal")

    gate2 = ConfirmationGate(broken)
    with pytest.raises(ExecutionDeniedError, match="callback failed"):
        gate2.request_approval(plan)
    assert "no terminal" in gate2.denials[0].reason


def test_truthy_non_bool_does_not_approve(tmp_path):
    plan = make_plan(_config(tmp_path))
    gate = ConfirmationGate(lambda p: "yes")
    with pytest.raises(ExecutionDeniedError):
        gate.request_approval(plan)


def test_prompt_confirm_requires_typed_yes(tmp_path):
    plan = make_plan(_config(tmp_path))
    out = io.StringIO()
    prompts = []
    assert prompt_confirm(plan, input_fn=lambda p: prompts.append(p) or " YES ", out=out) is True
    assert plan.plan_hash in out.getvalue()
    # The question goes to the plan's stream (stderr by default), not to input()'s stdout.
    assert out.getvalue().endswith("Type 'yes' to run this plan (anything else cancels): ")
    assert prompts == [""]
    assert prompt_confirm(plan, input_fn=lambda _: "y", out=io.StringIO()) is False
    assert prompt_confirm(plan, input_fn=lambda _: "", out=io.StringIO()) is False

    def eof(_):
        raise EOFError

    assert prompt_confirm(plan, input_fn=eof, out=io.StringIO()) is False


def test_prompt_confirm_question_is_not_written_to_stdout(tmp_path, monkeypatch, capsys):
    plan = make_plan(_config(tmp_path))
    monkeypatch.setattr("builtins.input", lambda prompt="": print(prompt, end="") or "no")
    assert prompt_confirm(plan) is False
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Type 'yes' to run this plan" in captured.err


def test_check_plan_current_refuses_changed_or_missing_inputs(tmp_path):
    plan = make_plan(_config(tmp_path))
    check_plan_current(plan)  # unchanged: no error
    _write_aoi(tmp_path / "aoi.geojson", x0=-116.0)
    with pytest.raises(PlanChangedError, match="no longer matches its hash"):
        check_plan_current(plan)
    os.remove(tmp_path / "aoi.geojson")
    with pytest.raises(PlanChangedError):
        check_plan_current(plan)
    gate = ConfirmationGate(lambda p: True)
    gate.record_approval(plan, approver="x", method="test")
    with pytest.raises(PlanChangedError):
        gate.authorize_execution(plan)
    assert gate.executions == 0


def test_default_callback_denies_without_a_terminal(monkeypatch):
    monkeypatch.setattr(gate_mod.sys, "stdin", io.StringIO(""))
    assert gate_mod.default_callback() is deny_all


def test_gate_to_dict_records_who_and_how(tmp_path):
    plan = make_plan(_config(tmp_path))
    gate = ConfirmationGate(lambda p: True, approver="alice")
    gate.request_approval(plan)
    state = gate.to_dict()
    assert state["approvals"][0]["approver"] == "alice"
    assert state["approvals"][0]["method"].startswith("python callback")
    assert state["approvals"][0]["approved_utc"].endswith("Z")
    assert state["max_executions"] == 1


def test_negative_max_executions_rejected():
    with pytest.raises(ValueError):
        ConfirmationGate(deny_all, max_executions=-1)


def test_concurrent_authorisations_respect_the_limit(tmp_path):
    import threading

    plan = make_plan(_config(tmp_path))
    gate = ConfirmationGate(deny_all, max_executions=1)
    for _ in range(8):
        gate.record_approval(plan, approver="x", method="test")
    barrier = threading.Barrier(8)
    outcomes = []

    def worker():
        barrier.wait()
        try:
            gate.authorize_execution(plan)
            outcomes.append("ran")
        except ExecutionLimitError:
            outcomes.append("limit")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("ran") == 1 and gate.executions == 1
