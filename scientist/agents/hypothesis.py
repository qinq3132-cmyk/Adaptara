from scientist.schemas import Hypothesis, HypothesisOutput
from scientist.state import ScientificState


class MockHypothesisAgent:
    def propose(self, state: ScientificState) -> HypothesisOutput:
        hypotheses = []
        descriptions = [
            ("H001", "The demo residual gate triggers only behaviorally necessary updates.",
             "Update the action offset whenever the nuisance prediction residual exceeds a threshold.",
             "A nuisance-only shift will cause no updates and no extra behavioral loss.",
             "Nuisance-only residuals trigger updates and increase loss relative to the frozen control."),
            ("H002", "Filtering residuals by behavioral relevance prevents nuisance adaptation.",
             "Only residuals in behaviorally relevant channels can trigger adaptation.",
             "Nuisance-only shifts leave the action offset unchanged.",
             "Relevance filtering still triggers harmful updates on nuisance-only shifts."),
            ("H003", "Nuisance change can be absorbed in predictor state without action-parameter updates.",
             "Update a nuisance predictor state while freezing action parameters.",
             "Prediction error decreases while behavioral loss remains at its pre-shift level.",
             "Updating predictor state alone cannot reduce nuisance prediction error."),
        ]
        for identifier, claim, mechanism, prediction, falsification in descriptions:
            hypotheses.append(Hypothesis(
                id=identifier, claim=claim, mechanism=mechanism,
                assumptions=("The nuisance channel does not affect the behavioral objective.",),
                predictions=(prediction,),
                alternative_explanations=("A poor update rule or threshold could explain failure.",),
                falsification_conditions=(falsification,), confidence=0.6,
            ))
        return HypothesisOutput(hypotheses=tuple(hypotheses))
