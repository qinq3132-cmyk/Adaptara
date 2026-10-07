"""A bounded, transactional scientific cycle. Agents propose; code commits."""
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .interfaces import AnalysisAgent, ExperimentRunner, ScientificAgent
from .provenance import capture_source, make_provenance, runtime_versions
from .schemas import (
    Analysis, Critique, Decision, Experiment, ExperimentOutput, ExperimentResult,
    Hypothesis, HypothesisOutput, ResearchBudget, Stage, TheoryUpdate,
)
from .state import ScientificState
from .tools.experiment_tracker import ResearchStore


class CycleStopped(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentBundle:
    hypothesis: ScientificAgent[HypothesisOutput]
    experiment: ScientificAgent[ExperimentOutput]
    analysis: AnalysisAgent
    critic: ScientificAgent[Critique]
    theory: ScientificAgent[TheoryUpdate]


def select_best_experiment(candidates: tuple[Experiment, ...], available: float) -> Experiment:
    feasible = [candidate for candidate in candidates if candidate.estimated_cost <= available]
    if not feasible:
        raise CycleStopped("No experiment fits the remaining compute budget")
    # Stable tie-break, independent of provider output ordering.
    return min(feasible, key=lambda candidate: (-candidate.scores.value, candidate.id))


class Orchestrator:
    def __init__(self, *, agents: AgentBundle, runner: ExperimentRunner, store: ResearchStore,
                 source_root: Path, artifact_root: Path, emit: Callable[[str], None] = print,
                 should_stop: Callable[[], bool] = lambda: False):
        self.agents, self.runner, self.store = agents, runner, store
        self.source_root, self.artifact_root = Path(source_root), Path(artifact_root)
        self.emit, self.should_stop = emit, should_stop

    def _check_stop(self):
        if self.should_stop():
            raise CycleStopped("Human stop requested")

    def _proposal(self, agent, output_type, state):
        # Retry only a stateless proposal, never an experiment with possible side effects.
        for attempt in range(state.budget.max_retries + 1):
            self._check_stop()
            try:
                proposal = agent.propose(state.model_copy(deep=True))
                value = proposal.model_dump() if hasattr(proposal, "model_dump") else proposal
                return output_type.model_validate(value)
            except (ValueError, TypeError):
                if attempt == state.budget.max_retries:
                    raise

    def run_cycle(self, state: ScientificState) -> ScientificState:
        """Commit one full cycle or leave the caller's state and SQLite unchanged.

        Milestone 1 accepts a fresh QUESTION state only. Future scheduling can call
        this interface without giving agents write access to scientific memory.
        """
        self._check_stop()
        state = ScientificState.model_validate_json(state.model_dump_json())
        if state.current_round >= state.budget.max_rounds:
            raise CycleStopped("Maximum rounds reached")
        if state.stage != Stage.QUESTION:
            raise CycleStopped("Milestone 1 runs one fresh cycle; resuming later phases is not implemented")
        if state.hypotheses or state.results or state.experiments:
            raise CycleStopped("Milestone 1 requires an empty initial research state")
        if state.budget.spent_compute_units >= state.budget.max_compute_units:
            raise CycleStopped("Compute budget exhausted")
        run_id = "run_" + uuid4().hex
        sources = capture_source(self.source_root)
        self.emit(f"Scientific Cycle {state.current_round + 1}")
        self.emit("Question loaded")

        def transition(current, stage, summary, **changes):
            self._check_stop()
            order = list(Stage)
            if order.index(stage) != order.index(current.stage) + 1:
                raise ValueError("Illegal scientific state transition")
            decision = Decision(id=f"D{len(current.decisions) + 1:03d}", stage=stage, summary=summary)
            return current.revised(stage=stage, decisions=current.decisions + (decision,), **changes)

        proposal = self._proposal(self.agents.hypothesis, HypothesisOutput, state)
        if any(h.status != "proposed" for h in proposal.hypotheses):
            raise ValueError("New hypotheses must have proposed status")
        state = transition(state, Stage.HYPOTHESIS, "Validated proposed hypotheses", hypotheses=proposal.hypotheses)
        for hypothesis in proposal.hypotheses:
            self.emit(f"Hypothesis generated: {hypothesis.id}")
        plan = self._proposal(self.agents.experiment, ExperimentOutput, state)
        # Validate every candidate before selecting or executing any candidate.
        state.revised(experiments=plan.candidates)
        selected = select_best_experiment(plan.candidates,
                                         state.budget.max_compute_units - state.budget.spent_compute_units)
        state = transition(state, Stage.EXPERIMENT, f"Selected {selected.id} by heuristic score {selected.scores.value:.4f}",
                           experiments=plan.candidates)
        self.emit(f"Experiment selected: {selected.id}")
        provenance = make_provenance(self.source_root, run_id, selected, sources,
                                     runner_version=self.runner.runner_version,
                                     algorithm_version=self.runner.algorithm_version)
        self._check_stop()
        raw = self.runner.execute(selected.model_copy(deep=True), provenance.model_copy(deep=True))
        result = ExperimentResult.model_validate(raw.model_dump())
        if result.experiment_id != selected.id or result.provenance != provenance:
            raise ValueError("Runner returned mismatched experiment or provenance")
        if "compute_units" not in result.compute_cost:
            raise ValueError("Runner must report compute_units")
        spent = state.budget.spent_compute_units + result.compute_cost["compute_units"]
        budget = ResearchBudget.model_validate({**state.budget.model_dump(), "spent_compute_units": spent})
        state = state.revised(results=(result,), budget=budget)
        directory = self.artifact_root / run_id
        self.store.stage_result(run_id, directory, result)
        self.emit("Experiment executed")
        self._check_stop()
        raw_analysis = self.agents.analysis.analyze(selected.model_copy(deep=True), result.model_copy(deep=True))
        analysis = Analysis.model_validate(raw_analysis.model_dump())
        state = transition(state, Stage.ANALYSIS, "Validated descriptive analysis", analyses=(analysis,))
        critique = self._proposal(self.agents.critic, Critique, state)
        state = transition(state, Stage.CRITIQUE, "Recorded confounds and required controls", critiques=(critique,))
        update = self._proposal(self.agents.theory, TheoryUpdate, state)
        if update.analysis_id != analysis.id or update.critique_id != critique.id:
            raise ValueError("Theory update must refer to the current analysis and critique")
        changes = {change.hypothesis_id: change for change in update.changes}
        if len(changes) != len(update.changes):
            raise ValueError("Duplicate hypothesis changes")
        if set(changes) - {h.id for h in state.hypotheses}:
            raise ValueError("Theory update refers to unknown hypotheses")
        all_evidence = state.evidence + update.evidence
        hypotheses = []
        for hypothesis in state.hypotheses + update.new_hypotheses:
            fields = hypothesis.model_dump()
            if hypothesis.id in changes:
                change = changes[hypothesis.id]
                linked = [e for e in all_evidence if e.id in change.evidence_ids]
                direction = "supports" if change.status == "supported" else "contradicts"
                if not linked or any(hypothesis.id not in getattr(e, direction) for e in linked):
                    raise ValueError("Belief change must be backed by evidence in the stated direction")
                fields.update(status=change.status, confidence=change.confidence)
            fields.update(supporting_evidence=tuple(e.id for e in all_evidence if hypothesis.id in e.supports),
                          contradicting_evidence=tuple(e.id for e in all_evidence if hypothesis.id in e.contradicts))
            hypotheses.append(Hypothesis.model_validate(fields))
        state = transition(state, Stage.THEORY_UPDATE, update.interpretation,
                           hypotheses=tuple(hypotheses), evidence=all_evidence,
                           experiments=state.experiments + update.next_experiments,
                           theory_updates=(update,), open_questions=state.open_questions + update.open_questions,
                           current_round=state.current_round + 1)
        exports = {
            "question.json": state.question.model_dump(mode="json"),
            "hypotheses.json": proposal.model_dump(mode="json"),
            "experiment_plan.json": {"selected_id": selected.id, **plan.model_dump(mode="json")},
            "experiment.json": selected.model_dump(mode="json"),
            "raw_results.json": result.model_dump(mode="json"),
            "analysis.json": analysis.model_dump(mode="json"),
            "critique.json": critique.model_dump(mode="json"),
            "theory_update.json": update.model_dump(mode="json"),
            "research_state.json": state.model_dump(mode="json"),
            "source_snapshot.json": sources,
            "runtime.json": runtime_versions(),
        }
        self._check_stop()
        self.store.save(run_id, state, directory, exports)
        for evidence in update.evidence:
            for identifier in evidence.contradicts:
                self.emit(f"Evidence contradicts {identifier}")
        for change in update.changes:
            old = next(h.status for h in proposal.hypotheses if h.id == change.hypothesis_id)
            self.emit(f"{change.hypothesis_id}: {old} -> {change.status}")
        for experiment in update.next_experiments:
            self.emit(f"Next experiment generated: {experiment.id}")
        self.emit("Cycle complete")
        self.emit(f"Artifacts: {directory.resolve()}")
        return state
