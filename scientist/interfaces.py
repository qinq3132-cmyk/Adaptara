"""Adapters implement these protocols; the orchestrator is provider agnostic."""
from typing import Protocol, TypeVar

from .schemas import Analysis, Experiment, ExperimentResult, Provenance, Record
from .state import ScientificState

Output = TypeVar("Output", bound=Record, covariant=True)


class ScientificAgent(Protocol[Output]):
    def propose(self, state: ScientificState) -> Output: ...


class AnalysisAgent(Protocol):
    # Analysis receives only configuration and raw results, not scientific beliefs.
    def analyze(self, experiment: Experiment, result: ExperimentResult) -> Analysis: ...


class ExperimentRunner(Protocol):
    runner_version: str
    algorithm_version: str

    def execute(self, experiment: Experiment, provenance: Provenance) -> ExperimentResult: ...
