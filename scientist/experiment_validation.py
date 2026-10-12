"""GLM implementation review and result interpretation with explicit scope."""
from typing import Literal

from pydantic import Field, model_validator

from .agents.structured import StructuredAgent
from .schemas import Record, Text


class ImplementationCheck(Record):
    hypothesis_id: Text
    verdict: Literal['ready_for_exploratory_run', 'revise']
    implemented_mechanism: Text
    mandatory_repairs: tuple[Text,...]
    scientific_limits: tuple[Text,...] = Field(min_length=1)
    human_summary: Text = Field(max_length=120)

    @model_validator(mode='after')
    def close_repairs(self):
        if (self.verdict == 'ready_for_exploratory_run') == bool(self.mandatory_repairs):
            raise ValueError('Ready requires no mandatory repairs; revise requires specific repairs')
        return self


class ImplementationReview(Record):
    checks: tuple[ImplementationCheck,...] = Field(min_length=1)
    overall_summary: Text


class ImplementationReviewAgent(StructuredAgent):
    role, output_type = 'experiment_implementation_review', ImplementationReview


class ResultInterpretation(Record):
    hypothesis_id: Text
    verdict: Literal['supported_in_scope','opposite_in_scope','inconclusive','invalid_experiment']
    numerical_comparison_ids: tuple[Text,...]
    interpretation: Text
    assumptions_checked: tuple[Text,...]
    unmet_assumptions: tuple[Text,...]
    confounds: tuple[Text,...]
    followup: tuple[Text,...]
    human_summary: Text = Field(max_length=120)


class ResultAnalysis(Record):
    interpretations: tuple[ResultInterpretation,...] = Field(min_length=1)
    overall_summary: Text


class ResultAnalysisAgent(StructuredAgent):
    role, output_type = 'experiment_result_analysis', ResultAnalysis


def validate_targets(output, targets, *, comparisons=None):
    rows = output.checks if hasattr(output,'checks') else output.interpretations
    ids = [r.hypothesis_id for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(targets):
        raise ValueError('Exactly one assessment of every actual target is required')
    if comparisons is not None:
        for row in rows:
            if set(row.numerical_comparison_ids)-set(comparisons):
                raise ValueError('Analysis may reference only computed comparison IDs')
            if row.verdict == 'supported_in_scope' and row.unmet_assumptions:
                raise ValueError('Unmet required assumptions prohibit a support verdict')
