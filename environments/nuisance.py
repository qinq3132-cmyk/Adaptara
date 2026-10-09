"""Runner-owned tracking fixture; hidden schedule never reaches an agent."""
import random

from pydantic import JsonValue

from experiments.runner import ToyConfig


class NuisanceEnvironment:
    def __init__(self, config: ToyConfig):
        self.config = config

    def reset(self, *, seed: int) -> None:
        self.rng = random.Random(seed)
        self.index = 0
        self.target = self.rng.uniform(-1, 1)
        self.nuisance = 0.0
        self.loss = 0.0

    def apply_shift(self) -> None:
        self.nuisance = self.config.amplitude

    def observe(self) -> JsonValue:
        if self.index >= self.config.shift_step:
            self.apply_shift()
        return [self.target, self.nuisance]

    def step(self, action: JsonValue) -> None:
        self.loss = (float(action) - self.target) ** 2
        self.index += 1
        if self.index < self.config.steps:
            self.target = self.rng.uniform(-1, 1)

    def hidden_evaluate(self) -> dict[str, float]:
        return {"loss": self.loss}
