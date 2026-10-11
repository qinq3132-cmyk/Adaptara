"""Contracts for problem clarification, separate from scientific evidence."""
import hashlib
import json
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StringConstraints, model_validator

from .schemas import Record, ScientificQuestion

ProblemText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def problem_digest(value: Record) -> str:
    payload = json.dumps(value.model_dump(mode="json"), ensure_ascii=False,
                         sort_keys=True, indent=2) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ProblemSource(Record):
    id: ProblemText
    title: ProblemText
    text: ProblemText


class ConceptInput(Record):
    text: ProblemText
    background: tuple[ProblemText, ...] = ()
    explicit_constraints: tuple[ProblemText, ...] = ()
    sources: tuple[ProblemSource, ...] = ()

    @model_validator(mode="after")
    def unique_sources(self):
        ids = [s.id for s in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("Source IDs must be unique")
        return self


class ProblemStatement(Record):
    text: ProblemText
    origin: Literal["user", "model_inference", "model_suggestion", "source_supported"]
    source_ids: tuple[ProblemText, ...]

    @model_validator(mode="after")
    def source_required(self):
        if (self.origin == "source_supported") != bool(self.source_ids):
            raise ValueError("Only source-supported statements must have source IDs")
        return self


class TermDefinition(Record):
    term: ProblemText
    interpretation: ProblemStatement
    alternatives: tuple[ProblemText, ...]


class DisciplinaryPerspective(Record):
    discipline: ProblemText
    contribution: ProblemStatement
    definition_implications: tuple[ProblemText, ...]
    conflicts: tuple[ProblemText, ...]
    limitations: tuple[ProblemText, ...]


class SubQuestion(Record):
    id: ProblemText
    text: ProblemText
    root_relationship: ProblemText
    evaluation_criteria: tuple[ProblemText, ...] = Field(min_length=1)


class ClarificationIssue(Record):
    id: ProblemText
    question: ProblemText
    reason: ProblemText
    options: tuple[ProblemText, ...]
    blocking: StrictBool


class ProblemDraft(Record):
    root_question: ProblemStatement
    goal: ProblemStatement
    terms: tuple[TermDefinition, ...]
    scope: tuple[ProblemStatement, ...]
    hard_constraints: tuple[ProblemStatement, ...]
    preferences: tuple[ProblemStatement, ...]
    assumptions: tuple[ProblemStatement, ...]
    perspectives: tuple[DisciplinaryPerspective, ...]
    subquestions: tuple[SubQuestion, ...]
    evaluation_criteria: tuple[ProblemStatement, ...]
    candidate_mechanisms: tuple[ProblemStatement, ...]
    pending_issues: tuple[ClarificationIssue, ...]
    changes: tuple[ProblemText, ...]
    limitations: tuple[ProblemText, ...]

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [q.id for q in self.subquestions] + [q.id for q in self.pending_issues]
        if len(ids) != len(set(ids)):
            raise ValueError("Subquestion and issue IDs must be unique")
        return self

    def validate_context(self, concept: ConceptInput) -> None:
        known = {s.id for s in concept.sources}
        # All source references, including nested term/discipline statements.
        def visit(value):
            if isinstance(value, dict):
                if set(value.get("source_ids", ())) - known:
                    raise ValueError("Unknown problem source reference")
                for child in value.values():
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)
        visit(self.model_dump())
        retained = {s.text for s in self.hard_constraints if s.origin == "user"}
        if set(concept.explicit_constraints) - retained:
            raise ValueError("Explicit user constraints must be retained verbatim")


class HumanFeedback(Record):
    text: ProblemText
    answers: dict[ProblemText, ProblemText] = Field(default_factory=dict)
    # Only an explicit human update may replace the declared hard constraints.
    revised_constraints: tuple[ProblemText, ...] | None = None


class ProblemRevision(Record):
    session_id: ProblemText
    revision: int = Field(ge=1)
    concept: ConceptInput
    draft: ProblemDraft
    feedback: HumanFeedback | None
    previous_draft_sha256: str | None
    created_at: ProblemText

    @model_validator(mode="after")
    def valid_context(self):
        self.draft.validate_context(self.concept)
        if (self.revision == 1) != (self.previous_draft_sha256 is None):
            raise ValueError("Revision lineage is inconsistent")
        if (self.revision == 1) != (self.feedback is None):
            raise ValueError("Revisions after the first require human feedback")
        return self


class ConfirmedProblem(Record):
    schema_version: Literal[1] = 1
    session_id: ProblemText
    revision: int = Field(ge=1)
    original_concept: ConceptInput
    concept: ConceptInput
    draft: ProblemDraft
    draft_sha256: ProblemText
    confirmed_at: ProblemText
    confirmed_by: ProblemText
    question: ScientificQuestion

    @model_validator(mode="after")
    def confirmed_content_matches(self):
        self.draft.validate_context(self.concept)
        if any(issue.blocking for issue in self.draft.pending_issues):
            raise ValueError("Blocking clarification issues must be resolved before confirmation")
        if problem_digest(self.draft) != self.draft_sha256:
            raise ValueError("Confirmed draft checksum mismatch")
        if (self.question.text != self.draft.root_question.text
                or self.question.constraints != tuple(s.text for s in self.draft.hard_constraints)):
            raise ValueError("Scientific question differs from the confirmed definition")
        return self
