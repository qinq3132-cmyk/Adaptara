"""Small arithmetic counterexample; this is not the Phase 2 environment gym."""
import random
from statistics import fmean

from pydantic import Field, model_validator

from scientist.schemas import Experiment, ExperimentResult, Provenance, Record


class ToyConfig(Record):
    kind: str = "nuisance_counterexample"
    version: str = "1"
    steps: int = Field(gt=1)
    shift_step: int = Field(ge=1)
    amplitude: float = Field(ge=0)
    threshold: float = Field(ge=0)
    learning_rate: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def within_run(self):
        if self.shift_step >= self.steps:
            raise ValueError("shift_step must be inside the trajectory")
        return self


class ResidualGate:
    """The candidate sees sensor observations only; no evaluator metadata."""
    def __init__(self, threshold: float, learning_rate: float):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.offset = 0.0

    def observe_and_act(self, observation: tuple[float, float]) -> tuple[float, bool]:
        signal, nuisance = observation
        update = abs(nuisance) > self.threshold  # Fixed predictor expects zero.
        if update:
            self.offset += self.learning_rate * nuisance
        return signal + self.offset, update


class ToyExperimentRunner:
    runner_version = "toy-runner-v1"
    algorithm_version = "toy-rules-v1"

    def execute(self, experiment: Experiment, provenance: Provenance) -> ExperimentResult:
        config = ToyConfig.model_validate(experiment.environment)
        if (config.kind != "nuisance_counterexample" or config.version != "1"
                or experiment.algorithms != ("toy_residual_gate", "toy_frozen")):
            raise ValueError("The toy runner only supports the E001 counterexample contract")
        trajectory, events = [], []
        for seed in experiment.seeds:
            rng = random.Random(seed)
            candidate = ResidualGate(config.threshold, config.learning_rate)
            for step in range(config.steps):
                target = rng.uniform(-1, 1)
                nuisance = config.amplitude if step >= config.shift_step else 0.0
                observation = (target, nuisance)
                action, updated = candidate.observe_and_act(observation)
                row = {"seed": seed, "step": step, "observation": list(observation),
                       "gated_action": action, "frozen_action": observation[0],
                       "gated_loss": (action - target) ** 2,
                       "frozen_loss": (observation[0] - target) ** 2}
                trajectory.append(row)
                if updated:
                    events.append({"seed": seed, "step": step, "offset": candidate.offset})
        gated = fmean(row["gated_loss"] for row in trajectory)
        frozen = fmean(row["frozen_loss"] for row in trajectory)
        return ExperimentResult(
            id="R001", experiment_id=experiment.id, kind="toy",
            metrics={"gated_loss": gated, "frozen_loss": frozen, "excess_loss": gated - frozen,
                     "false_adaptation_events": float(len(events))},
            trajectory=tuple(trajectory), adaptation_events=tuple(events),
            compute_cost={"compute_units": float(len(trajectory)), "samples": float(len(trajectory)),
                          "parameter_updates": float(len(events))},
            artifacts=("raw_results.json",), provenance=provenance,
        )
