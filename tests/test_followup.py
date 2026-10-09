import json
from dataclasses import replace

import pytest

from algorithms.nuisance import NuisanceRule
from experiments.followup_runner import FollowupExperimentRunner
from run_scientist import build_demo, main
from scientist.agents.experiment import followup_plan, toy_plan
from scientist.followup import analyze_followup, run_followup
from scientist.orchestrator import CycleStopped
from scientist.provenance import capture_source, make_provenance
from scientist.schemas import Experiment
from scientist.tools.experiment_tracker import json_bytes


@pytest.fixture
def parent(tmp_path):
    orchestrator, initial = build_demo(output_dir=tmp_path / "artifacts", database=tmp_path / "research.sqlite3", emit=lambda _: None)
    final = orchestrator.run_cycle(initial)
    return orchestrator, final


def execute(parent, **kwargs):
    orchestrator, state = parent
    return run_followup(parent_run=state.results[0].provenance.run_id, store=orchestrator.store,
                        source_root=orchestrator.source_root, artifact_root=orchestrator.artifact_root,
                        emit=lambda _: None, **kwargs)


def test_followup_preserves_parent_and_replays(parent):
    orchestrator, original = parent
    parent_id = original.results[0].provenance.run_id
    final = execute(parent)
    assert final.current_round == 2
    assert final.budget.spent_compute_units == 420
    assert final.budget.max_rounds == 2
    assert [r.experiment_id for r in final.results] == ["E001", "E002"]
    assert orchestrator.store.load(parent_id) == original
    assert original.results == final.results[:1]
    assert original.analyses == final.analyses[:1]
    assert original.theory_updates == final.theory_updates[:1]
    assert [h.status for h in final.hypotheses] == ["weakened", "supported", "supported"]
    assert [h.confidence for h in final.hypotheses] == [h.confidence for h in original.hypotheses]
    result = final.results[-1]
    run_id = result.provenance.run_id
    assert orchestrator.store.load(run_id) == final
    orchestrator.store.verify_artifacts(run_id)
    plan = next(e for e in final.experiments if e.id == "E002")
    assert FollowupExperimentRunner().execute(plan, result.provenance) == result
    directory = orchestrator.artifact_root / run_id
    lineage = json.loads((directory / "lineage.json").read_text())
    assert lineage["parent_run_id"] == parent_id
    assert lineage["parent_proposal"] == original.experiments[-1].model_dump(mode="json")
    assert json.loads((directory / "parent_state.json").read_text()) == original.model_dump(mode="json")
    assert "report.md" in json.loads((directory / "manifest.json").read_text())["sha256"]
    (directory / "report.md").write_text("corrupted")
    with pytest.raises(ValueError, match="checksum"):
        orchestrator.store.verify_artifacts(run_id)


def test_paired_observations_matched_counts_and_direction_effect(parent, monkeypatch):
    observations = []
    original = NuisanceRule.observe

    def spy(self, observation):
        assert isinstance(observation, list) and len(observation) == 2
        assert set(vars(self)) == {"threshold", "learning_rate", "observation", "offset", "predictor", "parameter_updates", "state_updates"}
        observations.append(observation)
        return original(self, observation)

    monkeypatch.setattr(NuisanceRule, "observe", spy)
    final = execute(parent)
    result = final.results[-1]
    assert len(observations) == 720  # Execution plus exact replay verification.
    metrics = result.metrics
    assert metrics["toy_residual_gate_loss"] == pytest.approx(0.77)
    assert metrics["reverse_direction_control_loss"] == pytest.approx(0.77)
    assert metrics["matched_update_control_loss"] == pytest.approx(0.01)
    assert metrics["toy_frozen_loss"] == metrics["relevance_filter_loss"] == metrics["predictor_state_loss"] == 0
    assert metrics["predictor_state_final_prediction_mse"] == pytest.approx((2 * 0.9 ** 10) ** 2)
    assert result.compute_cost["parameter_updates"] == 90
    assert result.compute_cost["state_updates"] == 30
    for seed in result.provenance.seeds:
        arms = {}
        for row in result.trajectory:
            if row["seed"] == seed:
                arms.setdefault(row["algorithm"], []).append(row)
        baseline = [(r["step"], r["observation"]) for r in arms["toy_frozen"]]
        for rows in arms.values():
            assert [(r["step"], r["observation"]) for r in rows] == baseline
        times = {name: [r["step"] for r in rows if r["parameter_updates"] or r["state_updates"]]
                 for name, rows in arms.items()}
        assert times["toy_residual_gate"] == times["predictor_state"] == times["matched_update_control"] == times["reverse_direction_control"]
        assert times["toy_frozen"] == times["relevance_filter"] == []


@pytest.mark.parametrize("amplitude,threshold", [(0.0, 0.5), (0.2, 0.5), (2.0, 3.0)])
def test_inactive_trigger_does_not_generate_support(parent, amplitude, threshold):
    orchestrator, state = parent
    plan = followup_plan()
    plan = Experiment.model_validate({**plan.model_dump(), "environment": {**plan.environment, "amplitude": amplitude, "threshold": threshold}})
    runner = FollowupExperimentRunner()
    sources = capture_source(orchestrator.source_root)
    provenance = make_provenance(orchestrator.source_root, "test", plan, sources,
                                 runner_version=runner.runner_version, algorithm_version=runner.algorithm_version)
    result = runner.execute(plan, provenance)
    analysis = analyze_followup(plan, result)
    assert analysis.statistical_comparisons["H002_prediction_met"] == 0
    assert analysis.statistical_comparisons["H003_prediction_met"] == 0
    assert result.adaptation_events == ()


def test_legacy_windows_parent_and_cli(parent):
    orchestrator, state = parent
    parent_id = state.results[0].provenance.run_id
    with orchestrator.store.connect() as db:
        db.execute("UPDATE runs SET artifact_dir = ? WHERE id = ?", ("D:\\old\\artifacts\\" + parent_id, parent_id))
    assert main(["--followup", "E002", "--database", str(orchestrator.store.path),
                 "--output-dir", str(orchestrator.artifact_root), "--parent-run", parent_id]) == 0
    with orchestrator.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2


def test_invalid_parent_never_executes(parent, monkeypatch):
    orchestrator, state = parent
    parent_id = state.results[0].provenance.run_id
    (orchestrator.artifact_root / parent_id / "raw_results.json").write_text("{}")

    def forbidden(*args):
        pytest.fail("Corrupted parent must fail before execution")

    monkeypatch.setattr(FollowupExperimentRunner, "execute", forbidden)
    with pytest.raises(ValueError, match="checksum"):
        execute(parent)
    with orchestrator.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_analysis_failure_keeps_raw_and_no_new_commit(parent, monkeypatch):
    orchestrator, state = parent

    def failing(*args):
        pending = list(orchestrator.artifact_root.glob("run_*/pending.json"))
        assert len(pending) == 1
        assert (pending[0].parent / "raw_results.json").is_file()
        raise RuntimeError("Injected analysis failure")

    monkeypatch.setattr("scientist.followup.analyze_followup", failing)
    with pytest.raises(RuntimeError, match="Injected analysis failure"):
        execute(parent)
    with orchestrator.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert orchestrator.store.load(state.results[0].provenance.run_id) == state


def test_budget_and_stop_are_checked_before_execution(parent, monkeypatch):
    orchestrator, state = parent
    limited = state.revised(budget={**state.budget.model_dump(), "max_compute_units": 100})
    parent_id = state.results[0].provenance.run_id
    directory = orchestrator.artifact_root / parent_id
    (directory / "research_state.json").write_bytes(json_bytes(limited.model_dump(mode="json")))
    import hashlib
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["sha256"]["research_state.json"] = hashlib.sha256((directory / "research_state.json").read_bytes()).hexdigest()
    (directory / "manifest.json").write_bytes(json_bytes(manifest))
    with orchestrator.store.connect() as db:
        db.execute("UPDATE runs SET state_json = ? WHERE id = ?", (limited.model_dump_json(), parent_id))
    with pytest.raises(CycleStopped, match="budget"):
        execute((orchestrator, limited))
    with pytest.raises(CycleStopped, match="Human stop"):
        execute((orchestrator, limited), should_stop=lambda: True)


def test_no_shift_followup_preserves_beliefs(parent):
    orchestrator, _ = parent
    other, initial = build_demo(output_dir=orchestrator.artifact_root, database=orchestrator.store.path, emit=lambda _: None)

    class NoShiftPlanner:
        def propose(self, snapshot):
            from scientist.schemas import ExperimentOutput
            plan = toy_plan()
            plan = Experiment.model_validate({**plan.model_dump(), "environment": {**plan.environment, "amplitude": 0.0}})
            return ExperimentOutput(candidates=(plan,))

    real_theory = other.agents.theory

    class NoShiftTheory:
        def propose(self, snapshot):
            update = real_theory.propose(snapshot)
            proposals = tuple(Experiment.model_validate({**plan.model_dump(),
                              "environment": {**plan.environment, "amplitude": 0.0}})
                              for plan in update.next_experiments)
            return type(update).model_validate({**update.model_dump(), "next_experiments": proposals})

    other.agents = replace(other.agents, experiment=NoShiftPlanner(), theory=NoShiftTheory())
    zero_state = other.run_cycle(initial)
    final = execute((other, zero_state))
    assert final.hypotheses == zero_state.hypotheses
    assert final.theory_updates[-1].changes == ()
    assert final.theory_updates[-1].evidence == ()
