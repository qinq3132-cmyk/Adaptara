from scientist.schemas import Evidence, HypothesisChange, ResearchQuestion, TheoryUpdate
from scientist.state import ScientificState
from .experiment import toy_plan


class MockTheoryAgent:
    def propose(self, state: ScientificState) -> TheoryUpdate:
        analysis = state.analyses[-1]
        contradicted = (analysis.statistical_comparisons["excess_loss"] > 0
                        and analysis.statistical_comparisons["false_adaptation_events"] > 0)
        evidence = ()
        changes = ()
        if contradicted:
            evidence = (Evidence(
                id="EV001", source_type="experiment", source_id=state.results[-1].experiment_id,
                claim="The specified gate made unnecessary updates and increased loss on the nuisance-only shift.",
                contradicts=("H001",), strength="weak",
                limitations=("Synthetic counterexample; only one deliberately simple update rule.",
                             "Update counts are unmatched; H002 and H003 remain untested."),
            ),)
            changes = (HypothesisChange(
                hypothesis_id="H001", status="weakened", confidence=0.25, evidence_ids=("EV001",),
                rationale="The observed false updates contradict H001's prediction in this toy setting.",
            ),)
        return TheoryUpdate(
            id="T001", analysis_id=analysis.id, critique_id=state.critiques[-1].id,
            interpretation=("H001 is weakened for this toy gate; no claim of general mechanism necessity follows."
                            if contradicted else "The falsification criterion was not met; beliefs remain unchanged."),
            changes=changes, evidence=evidence,
            next_experiments=(toy_plan("E002", followup=True, seeds=state.results[-1].provenance.seeds),),
            open_questions=(ResearchQuestion(id="Q002", text="Does the effect persist with matched update counts?",
                                            source_ids=("C001",)),),
        )
