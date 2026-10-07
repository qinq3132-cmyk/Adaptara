import hashlib
import json
import sqlite3
from dataclasses import replace

import pytest
from pydantic import ValidationError

from experiments.runner import ResidualGate, ToyExperimentRunner
from run_scientist import build_demo, main
from scientist.agents.experiment import toy_plan
from scientist.orchestrator import CycleStopped, select_best_experiment
from scientist.schemas import Evidence, ExperimentOutput, HypothesisOutput, Stage
from scientist.state import ScientificState
from scientist.tools.experiment_tracker import json_bytes


@pytest.fixture
def demo(tmp_path):
    return build_demo(output_dir=tmp_path / "artifacts", database=tmp_path / "research.sqlite3", emit=lambda _: None)


def test_complete_cycle_and_persistence(demo):
    orchestrator, initial = demo
    final = orchestrator.run_cycle(initial)
    assert initial.current_round == 0 and initial.hypotheses == ()
    assert final.current_round == 1 and final.stage == Stage.THEORY_UPDATE
    assert [d.stage for d in final.decisions] == list(Stage)[1:]
    h1, h2, h3 = final.hypotheses
    assert h1.status == "weakened" and h1.confidence == 0.25
    assert h1.contradicting_evidence == ("EV001",)
    assert h2.status == h3.status == "proposed"
    assert [e.id for e in final.experiments] == ["E001", "E002"]
    assert [r.experiment_id for r in final.results] == ["E001"]
    assert final.critiques[0].confounds
    result = final.results[0]
    assert result.metrics["excess_loss"] == pytest.approx(0.77)
    assert result.metrics["false_adaptation_events"] == 30
    assert final.budget.spent_compute_units == 60
    run_id = result.provenance.run_id
    directory = orchestrator.artifact_root / run_id
    required = {"question.json", "hypotheses.json", "experiment_plan.json", "raw_results.json",
                "analysis.json", "critique.json", "theory_update.json", "research_state.json"}
    assert required <= {p.name for p in directory.iterdir()}
    assert orchestrator.store.load(run_id) == final
    orchestrator.store.verify_artifacts(run_id)
    with sqlite3.connect(orchestrator.store.path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"questions", "papers", "claims", "hypotheses", "experiments", "runs", "evidence", "decisions"} <= tables
        assert db.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 3
    snapshot = json.loads((directory / "source_snapshot.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(json_bytes(snapshot)).hexdigest() == result.provenance.source_sha256
    assert hashlib.sha256(json_bytes(result.provenance.config)).hexdigest() == result.provenance.config_sha256
    assert ScientificState.model_validate_json(final.model_dump_json()) == final


def test_reproducible_runs_and_runner_replay(demo):
    orchestrator, initial = demo
    a, b = orchestrator.run_cycle(initial), orchestrator.run_cycle(initial)
    ra, rb = a.results[0], b.results[0]
    assert ra.provenance.run_id != rb.provenance.run_id
    assert ra.metrics == rb.metrics
    assert ra.trajectory == rb.trajectory
    assert a.theory_updates == b.theory_updates
    assert ToyExperimentRunner().execute(a.experiments[0], ra.provenance) == ra


@pytest.mark.parametrize("field,value", [("confidence", 1.1), ("confidence", float("nan")),
                                        ("predictions", []), ("falsification_conditions", [])])
def test_invalid_hypotheses_rejected(demo, field, value):
    orchestrator, state = demo
    proposal = orchestrator.agents.hypothesis.propose(state).model_dump()
    proposal["hypotheses"][0][field] = value
    with pytest.raises(ValidationError):
        HypothesisOutput.model_validate(proposal)


def test_unknown_evidence_and_duplicate_ids_rejected(demo):
    orchestrator, initial = demo
    hypotheses = orchestrator.agents.hypothesis.propose(initial).hypotheses
    with pytest.raises(ValidationError, match="globally unique"):
        initial.revised(hypotheses=(hypotheses[0], hypotheses[0]))
    with pytest.raises(ValidationError, match="Unknown evidence source"):
        initial.revised(evidence=(Evidence(id="EV", source_type="experiment", source_id="missing",
                                          claim="x", strength="weak", limitations=("toy",)),))


def test_agent_cannot_mutate_authoritative_state(demo):
    orchestrator, state = demo
    real_critic = orchestrator.agents.critic

    class MutatingCritic:
        def propose(self, snapshot):
            snapshot.results[0].metrics["excess_loss"] = -999
            snapshot.experiments[0].environment["steps"] = 999
            return real_critic.propose(snapshot)

    orchestrator.agents = replace(orchestrator.agents, critic=MutatingCritic())
    final = orchestrator.run_cycle(state)
    assert final.results[0].metrics["excess_loss"] > 0
    assert final.experiments[0].environment["steps"] == 20
    assert state.hypotheses == ()


def test_invalid_output_rolls_back_entire_cycle(demo):
    orchestrator, state = demo

    class BadTheory:
        def propose(self, snapshot):
            return {"made_up": True}

    orchestrator.agents = replace(orchestrator.agents, theory=BadTheory())
    with pytest.raises(ValidationError):
        orchestrator.run_cycle(state)
    with sqlite3.connect(orchestrator.store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    pending = list(orchestrator.artifact_root.glob("run_*/pending.json"))
    assert len(pending) == 1
    assert (pending[0].parent / "raw_results.json").is_file()
    assert not (pending[0].parent / "research_state.json").exists()
    assert state.current_round == 0


def test_unknown_hypothesis_in_plan_never_executes(demo):
    orchestrator, state = demo

    class BadPlanner:
        def propose(self, snapshot):
            candidate = toy_plan().model_copy(update={"hypotheses_tested": ("missing",)})
            return ExperimentOutput(candidates=(candidate,))

    class NeverRun:
        def execute(self, experiment, provenance):
            pytest.fail("Invalid candidate must not execute")

    orchestrator.agents = replace(orchestrator.agents, experiment=BadPlanner())
    orchestrator.runner = NeverRun()
    with pytest.raises(ValidationError, match="tested hypotheses"):
        orchestrator.run_cycle(state)


def test_budget_and_human_stop(demo):
    orchestrator, initial = demo
    low_budget = initial.budget.model_copy(update={"max_compute_units": 1})
    with pytest.raises(CycleStopped, match="budget"):
        orchestrator.run_cycle(initial.revised(budget=low_budget))
    orchestrator.should_stop = lambda: True
    with pytest.raises(CycleStopped, match="Human stop"):
        orchestrator.run_cycle(initial)
    orchestrator.should_stop = lambda: False
    final = orchestrator.run_cycle(initial)
    with pytest.raises(CycleStopped, match="Maximum rounds"):
        orchestrator.run_cycle(final)


def test_score_and_deterministic_tie_break():
    a = toy_plan("A")
    b = toy_plan("B")
    assert a.scores.value == pytest.approx(0.785)
    assert select_best_experiment((b, a), 100).id == "A"
    with pytest.raises(CycleStopped):
        select_best_experiment((a,), 0)


def test_no_hidden_information_leak(demo, monkeypatch):
    orchestrator, state = demo
    observed = []
    original = ResidualGate.observe_and_act

    def spy(self, observation):
        assert isinstance(observation, tuple) and len(observation) == 2
        assert all(isinstance(value, float) for value in observation)
        assert set(vars(self)) == {"threshold", "learning_rate", "offset"}
        observed.append(observation)
        return original(self, observation)

    monkeypatch.setattr(ResidualGate, "observe_and_act", spy)
    orchestrator.run_cycle(state)
    assert len(observed) == 60


def test_theory_does_not_weaken_without_counterevidence(demo):
    orchestrator, state = demo

    class NoShiftPlanner:
        def propose(self, snapshot):
            plan = toy_plan()
            plan = plan.model_copy(update={"environment": {**plan.environment, "amplitude": 0.0}})
            return ExperimentOutput(candidates=(plan,))

    orchestrator.agents = replace(orchestrator.agents, experiment=NoShiftPlanner())
    final = orchestrator.run_cycle(state)
    assert final.results[0].metrics["excess_loss"] == 0
    assert final.hypotheses[0].status == "proposed"
    assert final.evidence == ()


def test_artifact_tampering_is_detected(demo):
    orchestrator, state = demo
    final = orchestrator.run_cycle(state)
    run_id = final.results[0].provenance.run_id
    (orchestrator.artifact_root / run_id / "raw_results.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        orchestrator.store.verify_artifacts(run_id)


def test_cli(tmp_path, capsys):
    assert main(["--demo", "--output-dir", str(tmp_path / "out"), "--database", str(tmp_path / "db.sqlite3")]) == 0
    output = capsys.readouterr().out
    for expected in ("Scientific Cycle 1", "Question loaded", "Evidence contradicts H001",
                     "H001: proposed -> weakened", "Next experiment generated: E002", "Cycle complete"):
        assert expected in output


def test_raw_results_saved_before_analysis(demo):
    orchestrator, state = demo
    original = orchestrator.agents.analysis

    class InspectingAnalysis:
        def analyze(self, experiment, result):
            directory = orchestrator.artifact_root / result.provenance.run_id
            assert json.loads((directory / "raw_results.json").read_text(encoding="utf-8")) == result.model_dump(mode="json")
            assert (directory / "pending.json").exists()
            return original.analyze(experiment, result)

    orchestrator.agents = replace(orchestrator.agents, analysis=InspectingAnalysis())
    final = orchestrator.run_cycle(state)
    assert not (orchestrator.artifact_root / final.results[0].provenance.run_id / "pending.json").exists()


def test_runner_adapter_and_actual_budget_validation(demo):
    orchestrator, state = demo

    class ExternalRunner:
        runner_version = "independent-adapter-v1"
        algorithm_version = "independent-rule-v1"

        def execute(self, experiment, provenance):
            return ToyExperimentRunner().execute(experiment, provenance)

    orchestrator.runner = ExternalRunner()
    final = orchestrator.run_cycle(state)
    assert final.results[0].provenance.runner_version == "independent-adapter-v1"

    class OverBudgetRunner(ExternalRunner):
        def execute(self, experiment, provenance):
            result = super().execute(experiment, provenance)
            return result.model_copy(update={"compute_cost": {"compute_units": 1001.0}})

    orchestrator.runner = OverBudgetRunner()
    with pytest.raises(ValidationError, match="Compute budget exceeded"):
        orchestrator.run_cycle(state)
    with sqlite3.connect(orchestrator.store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_runner_failure_is_not_retried(demo):
    orchestrator, state = demo

    class FailingRunner:
        runner_version = "failing-v1"
        algorithm_version = "toy-v1"
        calls = 0

        def execute(self, experiment, provenance):
            self.calls += 1
            raise RuntimeError("Execution failed")

    runner = FailingRunner()
    orchestrator.runner = runner
    state = state.revised(budget=state.budget.model_copy(update={"max_retries": 2}))
    with pytest.raises(RuntimeError, match="Execution failed"):
        orchestrator.run_cycle(state)
    assert runner.calls == 1
    with sqlite3.connect(orchestrator.store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_invalid_proposal_retry_is_bounded(demo):
    orchestrator, state = demo
    original = orchestrator.agents.hypothesis

    class RetryAgent:
        calls = 0

        def propose(self, snapshot):
            self.calls += 1
            return {} if self.calls == 1 else original.propose(snapshot)

    adapter = RetryAgent()
    orchestrator.agents = replace(orchestrator.agents, hypothesis=adapter)
    state = state.revised(budget=state.budget.model_copy(update={"max_retries": 1}))
    assert orchestrator.run_cycle(state).current_round == 1
    assert adapter.calls == 2


def test_sqlite_transaction_rolls_back_on_record_failure(demo):
    orchestrator, state = demo
    with sqlite3.connect(orchestrator.store.path) as db:
        db.execute("CREATE TRIGGER fail_hypothesis BEFORE INSERT ON hypotheses BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        orchestrator.run_cycle(state)
    with sqlite3.connect(orchestrator.store.path) as db:
        for table in ("runs", "questions", "hypotheses", "experiments"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert len(list(orchestrator.artifact_root.glob("run_*/pending.json"))) == 1
