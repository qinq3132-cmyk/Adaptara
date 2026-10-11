"""Version-one contracts shared by mock and future real implementations."""
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_serializer, model_validator

Text = Annotated[str, Field(min_length=1)]
Score = Annotated[float, Field(ge=0, le=1)]
NonNegative = Annotated[float, Field(ge=0)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Stage(StrEnum):
    QUESTION = "QUESTION"
    HYPOTHESIS = "HYPOTHESIS"
    EXPERIMENT = "EXPERIMENT"
    ANALYSIS = "ANALYSIS"
    CRITIQUE = "CRITIQUE"
    THEORY_UPDATE = "THEORY_UPDATE"


class ScientificQuestion(Record):
    id: Text
    text: Text
    constraints: tuple[Text, ...]


class ResearchQuestion(Record):
    id: Text
    text: Text
    source_ids: tuple[Text, ...] = ()


class Paper(Record):
    id: Text
    title: Text
    source_uri: Text
    doi: str | None = None
    arxiv_id: str | None = None
    version: str | None = None
    content_sha256: str | None = None
    read_level: Literal["title", "abstract", "excerpt", "full_text"] | None = None

    @model_serializer(mode="wrap")
    def serialize_optional(self, handler):
        return {k: v for k, v in handler(self).items() if v is not None}


class Claim(Record):
    id: Text
    statement: Text
    source_ids: tuple[Text, ...] = Field(min_length=1)
    mechanism: Text
    assumptions: tuple[Text, ...] = ()
    limitations: tuple[Text, ...] = ()


class Hypothesis(Record):
    id: Text
    claim: Text
    mechanism: Text
    assumptions: tuple[Text, ...] = Field(min_length=1)
    predictions: tuple[Text, ...] = Field(min_length=1)
    alternative_explanations: tuple[Text, ...] = Field(min_length=1)
    falsification_conditions: tuple[Text, ...] = Field(min_length=1)
    supporting_evidence: tuple[Text, ...] = ()
    contradicting_evidence: tuple[Text, ...] = ()
    parent_hypotheses: tuple[Text, ...] = ()
    source_claims: tuple[Text, ...] = ()
    status: Literal["proposed", "testing", "supported", "weakened", "rejected"] = "proposed"
    confidence: Score = 0.5  # Internal belief score, not a Bayesian probability.


class ExperimentScore(Record):
    discriminability: Score
    falsifiability: Score
    feasibility: Score
    scientific_importance: Score
    cost: Score

    @property
    def value(self) -> float:
        return (0.30 * self.discriminability + 0.25 * self.falsifiability
                + 0.20 * self.feasibility + 0.15 * self.scientific_importance
                - 0.10 * self.cost)


class Experiment(Record):
    id: Text
    hypotheses_tested: tuple[Text, ...] = Field(min_length=1)
    scientific_question: Text
    intervention: Text
    control: Text
    environment: dict[str, JsonValue]
    algorithms: tuple[Text, ...] = Field(min_length=1)
    metrics: tuple[Text, ...] = Field(min_length=1)
    expected_outcomes: dict[str, JsonValue]
    falsification_logic: dict[str, JsonValue]
    seeds: tuple[int, ...] = Field(min_length=1)
    estimated_cost: NonNegative
    scores: ExperimentScore


class Provenance(Record):
    run_id: Text
    code_commit: str | None
    source_sha256: Text
    config_sha256: Text
    environment_version: Text
    algorithm_version: Text
    runner_version: Text
    seeds: tuple[int, ...]
    config: dict[str, JsonValue]


class ExperimentResult(Record):
    id: Text
    experiment_id: Text
    kind: Literal["toy", "simulation", "measurement"]
    metrics: dict[str, float]
    trajectory: tuple[dict[str, JsonValue], ...]
    adaptation_events: tuple[dict[str, JsonValue], ...]
    compute_cost: dict[str, NonNegative]
    artifacts: tuple[Text, ...]
    provenance: Provenance


class Evidence(Record):
    id: Text
    source_type: Literal["paper", "experiment", "analysis"]
    source_id: Text
    claim: Text
    supports: tuple[Text, ...] = ()
    contradicts: tuple[Text, ...] = ()
    strength: Literal["weak", "moderate", "strong"]
    limitations: tuple[Text, ...] = Field(min_length=1)
    quote: Text | None = None
    locator: Text | None = None
    content_sha256: str | None = None
    conditions: tuple[Text, ...] = ()
    verification: Literal["accepted"] | None = None

    @model_serializer(mode="wrap")
    def serialize_optional(self, handler):
        data = {k: v for k, v in handler(self).items() if v is not None}
        if not self.conditions:
            data.pop("conditions", None)
        return data

    @model_validator(mode="after")
    def consistent_direction(self):
        if set(self.supports) & set(self.contradicts):
            raise ValueError("Evidence cannot both support and contradict the same hypothesis")
        return self


class Analysis(Record):
    id: Text
    result_ids: tuple[Text, ...] = Field(min_length=1)
    observations: tuple[Text, ...] = Field(min_length=1)
    statistical_comparisons: dict[str, float]
    unexpected_results: tuple[Text, ...] = ()
    possible_interpretations: tuple[Text, ...] = ()


class Critique(Record):
    id: Text
    analysis_id: Text
    validity: Text
    confounds: tuple[Text, ...]
    alternative_explanations: tuple[Text, ...]
    required_controls: tuple[Text, ...]
    recommended_next_experiment: Text
    conclusion_too_strong: bool


class HypothesisChange(Record):
    hypothesis_id: Text
    status: Literal["supported", "weakened", "rejected"]
    confidence: Score
    evidence_ids: tuple[Text, ...] = Field(min_length=1)
    rationale: Text


class HypothesisOutput(Record):
    hypotheses: tuple[Hypothesis, ...] = Field(min_length=1)


class ExperimentOutput(Record):
    candidates: tuple[Experiment, ...] = Field(min_length=1)


class TheoryUpdate(Record):
    id: Text
    analysis_id: Text
    critique_id: Text
    interpretation: Text
    changes: tuple[HypothesisChange, ...]
    evidence: tuple[Evidence, ...]
    new_hypotheses: tuple[Hypothesis, ...] = ()
    next_experiments: tuple[Experiment, ...] = ()
    open_questions: tuple[ResearchQuestion, ...] = ()


class ResearchBudget(Record):
    max_rounds: Annotated[int, Field(ge=1)] = 1
    max_compute_units: NonNegative = 1000
    spent_compute_units: NonNegative = 0
    max_retries: Annotated[int, Field(ge=0, le=5)] = 0

    @model_validator(mode="after")
    def bounded(self):
        if self.spent_compute_units > self.max_compute_units:
            raise ValueError("Compute budget exceeded")
        return self


class Decision(Record):
    id: Text
    stage: Stage
    summary: Text
    related_ids: tuple[Text, ...] = ()
