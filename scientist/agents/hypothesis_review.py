"""Pre-experiment hypothesis review, distinct from experimental critique."""
from pydantic import Field
from scientist.literature_schemas import ReviewBatch
from scientist.schemas import Record
from .llm_hypothesis import DraftHypothesis
from .structured import StructuredAgent


class HypothesisRevisionBatch(Record):
    hypotheses: tuple[DraftHypothesis, ...] = Field(min_length=1, max_length=6)


class HypothesisReviewAgent(StructuredAgent):
    role, output_type = "hypothesis_review", ReviewBatch

    def output_schema(self, inputs):
        schema = super().output_schema(inputs)
        targets = inputs["target_hypothesis_ids"]
        # Official structured outputs require explicitly named object keys.
        # Compile the comparison map for this exact reviewed cohort.
        schema["$defs"]["HypothesisComparison"]["properties"]["predictions"] = {
            "type": "object", "properties": {hid: {"type": "string", "minLength": 1} for hid in targets},
            "required": list(targets), "additionalProperties": False}
        return schema


class HypothesisRevisionAgent(StructuredAgent):
    role, output_type = "hypothesis_revision", HypothesisRevisionBatch
