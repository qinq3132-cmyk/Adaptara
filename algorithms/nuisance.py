"""Observation-only rules for the E002 nuisance-channel control experiment.

Channel relevance is declared by the public tracking objective, not learned here.
The shared trigger uses a fixed zero predictor. Predictor adaptation changes a
separate state, so its update count remains comparable to action-update controls.
"""
from pydantic import JsonValue


class NuisanceRule:
    mode = "frozen"

    def __init__(self, threshold: float, learning_rate: float):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.reset()

    def reset(self) -> None:
        self.observation = (0.0, 0.0)
        self.offset = 0.0
        self.predictor = 0.0
        self.parameter_updates = 0
        self.state_updates = 0

    def observe(self, observation: JsonValue) -> None:
        if not isinstance(observation, list) or len(observation) != 2:
            raise ValueError("Expected signal and nuisance observations only")
        self.observation = tuple(float(value) for value in observation)

    def adapt(self) -> None:
        nuisance = self.observation[1]
        if abs(nuisance) <= self.threshold:
            return
        if self.mode == "predictor":
            self.predictor += self.learning_rate * (nuisance - self.predictor)
            self.state_updates += 1
        elif self.mode in {"positive", "negative", "alternating"}:
            direction = -1 if self.mode == "negative" else 1
            if self.mode == "alternating" and self.parameter_updates % 2:
                direction = -1
            self.offset += direction * self.learning_rate * nuisance
            self.parameter_updates += 1
        # Frozen and relevance-filtered rules do not couple nuisance to action.

    def act(self) -> JsonValue:
        return self.observation[0] + self.offset

    def get_internal_state(self) -> dict[str, JsonValue]:
        return {"offset": self.offset, "predictor": self.predictor,
                "parameter_updates": self.parameter_updates,
                "state_updates": self.state_updates}


class FrozenRule(NuisanceRule):
    mode = "frozen"


class ResidualActionRule(NuisanceRule):
    mode = "positive"


class RelevanceFilterRule(NuisanceRule):
    mode = "relevance"


class PredictorStateRule(NuisanceRule):
    mode = "predictor"


class MatchedUpdateRule(NuisanceRule):
    mode = "alternating"


class ReverseDirectionRule(NuisanceRule):
    mode = "negative"


RULES = {"toy_frozen": FrozenRule, "toy_residual_gate": ResidualActionRule,
         "relevance_filter": RelevanceFilterRule, "predictor_state": PredictorStateRule,
         "matched_update_control": MatchedUpdateRule, "reverse_direction_control": ReverseDirectionRule}
FOLLOWUP_ALGORITHMS = tuple(RULES)
