from scientist.schemas import Experiment, ExperimentOutput, ExperimentScore
from scientist.state import ScientificState


def toy_plan(identifier="E001", *, followup=False, seeds=(7, 11, 19)) -> Experiment:
    return Experiment(
        id=identifier, hypotheses_tested=("H001", "H002", "H003"),
        scientific_question="Does a nuisance prediction residual reliably indicate a need to adapt behavior?",
        intervention=("Compare behavior-relevance filtering and predictor-state adaptation at matched update counts."
                      if followup else "Introduce an irrelevant sensor offset while the behavioral target stays fixed."),
        control="Frozen action offset exposed to the identical observations and seeds.",
        environment={"kind": "nuisance_counterexample", "version": "1", "steps": 20,
                     "shift_step": 10, "amplitude": 2.0, "threshold": 0.5, "learning_rate": 0.1},
        algorithms=(("relevance_filter", "predictor_state", "matched_update_control") if followup
                    else ("toy_residual_gate", "toy_frozen")),
        metrics=("gated_loss", "frozen_loss", "excess_loss", "false_adaptation_events"),
        expected_outcomes={"H001": "No false updates or excess loss", "H002": "Filtering should avoid false updates",
                           "H003": "Predictor state should absorb nuisance change"},
        falsification_logic={"H001": "excess_loss > 0 and false_adaptation_events > 0"},
        seeds=seeds, estimated_cost=float(20 * len(seeds)),
        scores=ExperimentScore(discriminability=0.8, falsifiability=1, feasibility=1,
                               scientific_importance=0.7, cost=0.1),
    )


class MockExperimentPlanner:
    def __init__(self, seeds=(7, 11, 19)):
        self.seeds = seeds

    def propose(self, state: ScientificState) -> ExperimentOutput:
        return ExperimentOutput(candidates=(toy_plan(seeds=self.seeds),))
