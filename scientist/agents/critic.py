from scientist.schemas import Critique
from scientist.state import ScientificState


class MockCriticAgent:
    def propose(self, state: ScientificState) -> Critique:
        return Critique(
            id="C001", analysis_id=state.analyses[-1].id,
            validity="A reproducible counterexample for the specified toy gate, not a general theory of adaptation.",
            confounds=("The gated rule performs more updates than the frozen control.",
                       "The update directly couples a nuisance residual to the action offset by construction."),
            alternative_explanations=("The update direction may be unsuitable even if prediction-error gating is useful.",),
            required_controls=("Match update counts and vary update direction.",
                               "Compare relevance filtering and predictor-state adaptation."),
            recommended_next_experiment="E002: matched-update relevance and predictor-state controls.",
            conclusion_too_strong=True,
        )
