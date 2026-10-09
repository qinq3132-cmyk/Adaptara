"""Executable E002 fixture with separately accounted action/state updates."""
from statistics import fmean

from algorithms.nuisance import RULES
from environments.nuisance import NuisanceEnvironment
from experiments.runner import ToyConfig
from scientist.schemas import Experiment, ExperimentResult, Provenance


class FollowupExperimentRunner:
    runner_version = "nuisance-followup-runner-v1"
    algorithm_version = "nuisance-controls-v1"

    def execute(self, experiment: Experiment, provenance: Provenance) -> ExperimentResult:
        config = ToyConfig.model_validate(experiment.environment)
        if config.kind != "nuisance_counterexample" or config.version != "1":
            raise ValueError("Unsupported follow-up environment")
        if len(set(experiment.algorithms)) != len(experiment.algorithms):
            raise ValueError("Duplicate algorithm arms")
        if len(set(experiment.seeds)) != len(experiment.seeds):
            raise ValueError("Seeds must identify distinct replicates")
        if set(experiment.algorithms) - RULES.keys():
            raise ValueError("Unsupported follow-up algorithm")
        if provenance.config != experiment.model_dump(mode="json") or provenance.seeds != experiment.seeds:
            raise ValueError("Provenance does not match execution plan")
        trajectory, events = [], []
        for seed in experiment.seeds:
            for name in experiment.algorithms:
                environment = NuisanceEnvironment(config)
                environment.reset(seed=seed)
                agent = RULES[name](config.threshold, config.learning_rate)
                for step in range(config.steps):
                    observation = environment.observe()
                    agent.observe(list(observation))
                    before = agent.get_internal_state()
                    agent.adapt()  # Equal opportunity every step; rule owns trigger.
                    action = agent.act()
                    after = agent.get_internal_state()
                    environment.step(action)
                    parameter_updates = after["parameter_updates"] - before["parameter_updates"]
                    state_updates = after["state_updates"] - before["state_updates"]
                    row = {"seed": seed, "step": step, "algorithm": name,
                           "observation": observation, "action": action,
                           "loss": environment.hidden_evaluate()["loss"],
                           "prediction_error_before": (observation[1] - before["predictor"]) ** 2,
                           "prediction_error_after": (observation[1] - after["predictor"]) ** 2,
                           "parameter_updates": parameter_updates, "state_updates": state_updates,
                           "internal_state": after}
                    trajectory.append(row)
                    if parameter_updates or state_updates:
                        events.append({"seed": seed, "step": step, "algorithm": name,
                                       "parameter_updates": parameter_updates,
                                       "state_updates": state_updates, "internal_state": after})
        metrics = {}
        for name in experiment.algorithms:
            rows = [r for r in trajectory if r["algorithm"] == name]
            shifted = [r for r in rows if r["step"] >= config.shift_step]
            final = [r for r in rows if r["step"] == config.steps - 1]
            metrics.update({f"{name}_loss": fmean(r["loss"] for r in rows),
                            f"{name}_parameter_updates": float(sum(r["parameter_updates"] for r in rows)),
                            f"{name}_state_updates": float(sum(r["state_updates"] for r in rows)),
                            f"{name}_post_shift_prediction_mse": fmean(r["prediction_error_after"] for r in shifted),
                            f"{name}_final_prediction_mse": fmean(r["prediction_error_after"] for r in final)})
        return ExperimentResult(
            id="R002", experiment_id=experiment.id, kind="toy", metrics=metrics,
            trajectory=tuple(trajectory), adaptation_events=tuple(events),
            compute_cost={"compute_units": float(len(trajectory)), "samples": float(len(trajectory)),
                          "parameter_updates": float(sum(r["parameter_updates"] for r in trajectory)),
                          "state_updates": float(sum(r["state_updates"] for r in trajectory))},
            artifacts=("raw_results.json",), provenance=provenance,
        )
