from scientist.schemas import Experiment, ExperimentOutput, ExperimentScore
from scientist.state import ScientificState


def toy_plan(identifier="E001", *, followup=False, seeds=(7, 11, 19)) -> Experiment:
    if followup:
        return followup_plan(identifier, seeds=seeds)
    return Experiment(
        id=identifier, hypotheses_tested=("H001", "H002", "H003"),
        scientific_question="Does a nuisance prediction residual reliably indicate a need to adapt behavior?",
        intervention="Introduce an irrelevant sensor offset while the behavioral target stays fixed.",
        control="Frozen action offset exposed to the identical observations and seeds.",
        environment={"kind": "nuisance_counterexample", "version": "1", "steps": 20,
                     "shift_step": 10, "amplitude": 2.0, "threshold": 0.5, "learning_rate": 0.1},
        algorithms=("toy_residual_gate", "toy_frozen"),
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


def followup_plan(identifier="E002", *, seeds=(7, 11, 19)) -> Experiment:
    """Refine the saved E002 proposal into explicit executable controls."""
    from algorithms.nuisance import FOLLOWUP_ALGORITHMS

    return Experiment(
        id=identifier, hypotheses_tested=("H001", "H002", "H003"),
        scientific_question="Can relevance filtering or predictor-state adaptation avoid nuisance-induced behavioral loss?",
        intervention="Compare positive, reversed and alternating action updates on a shared observation-only gate, plus relevance filtering and predictor-state adaptation.",
        control="Frozen action offset; positive, reversed, alternating and predictor-state arms have matched update events. Relevance-filter and frozen arms make no updates.",
        environment={"kind": "nuisance_counterexample", "version": "1", "steps": 20,
                     "shift_step": 10, "amplitude": 2.0, "threshold": 0.5, "learning_rate": 0.1},
        algorithms=FOLLOWUP_ALGORITHMS,
        metrics=("loss", "parameter_updates", "state_updates", "post_shift_prediction_mse", "final_prediction_mse"),
        expected_outcomes={"H001": "Positive nuisance action updates incur excess loss",
                           "H002": "Declared relevance filtering makes no action updates and preserves frozen loss",
                           "H003": "Predictor-state adaptation reduces nuisance prediction error without action updates or excess loss"},
        falsification_logic={"H001": "Positive-gate loss exceeds frozen loss with nonzero action updates",
                             "H002": "Relevance filter makes action updates or exceeds frozen loss during an active nuisance shift",
                             "H003": "Predictor error fails to decrease, action parameters change, or loss exceeds frozen loss during an active nuisance shift"},
        seeds=seeds, estimated_cost=float(20 * len(seeds) * len(FOLLOWUP_ALGORITHMS)),
        scores=ExperimentScore(discriminability=0.8, falsifiability=1, feasibility=1,
                               scientific_importance=0.7, cost=0.1),
    )
