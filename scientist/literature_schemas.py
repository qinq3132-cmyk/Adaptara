"""Contracts for literature acquisition, grounded extraction and hypothesis review."""
from typing import Literal

from pydantic import Field, StrictBool, model_validator

from .problem_schemas import ProblemText
from .schemas import Record, Text


class SearchQuery(Record):
    text: ProblemText = Field(max_length=200)
    purpose: Literal["overview", "support", "challenge", "cross_discipline"]
    discipline: ProblemText
    rationale: ProblemText


class SearchPlan(Record):
    queries: tuple[SearchQuery, ...] = Field(min_length=1, max_length=6)
    coverage_plan: tuple[ProblemText, ...] = Field(min_length=1)


class PaperCandidate(Record):
    id: ProblemText
    title: ProblemText
    abstract: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source_uri: ProblemText
    fulltext_uri: str | None = None
    providers: tuple[str, ...] = ()
    query_texts: tuple[str, ...] = ()


class SelectionItem(Record):
    paper_id: ProblemText
    selected: StrictBool
    reason: ProblemText


class PaperSelection(Record):
    decisions: tuple[SelectionItem, ...] = Field(min_length=1)
    coverage_limitations: tuple[ProblemText, ...]


class DocumentSection(Record):
    id: ProblemText
    heading: ProblemText
    text: ProblemText


class PaperDocument(Record):
    paper_id: ProblemText
    source_uri: ProblemText
    read_level: Literal["abstract", "excerpt", "full_text"]
    sections: tuple[DocumentSection, ...] = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fetched_at: ProblemText
    limitations: tuple[ProblemText, ...]


class EvidenceDraft(Record):
    section_id: ProblemText
    quote: ProblemText = Field(min_length=15, max_length=600)
    claim: ProblemText
    conditions: tuple[ProblemText, ...]
    limitations: tuple[ProblemText, ...] = Field(min_length=1)
    strength: Literal["weak", "moderate", "strong"]


class PaperReading(Record):
    findings: tuple[EvidenceDraft, ...] = Field(max_length=5)
    limitations: tuple[ProblemText, ...]


class FindingCheck(Record):
    finding_id: ProblemText
    verdict: Literal["accepted", "rejected", "uncertain"]
    reason: ProblemText


class EvidenceChecks(Record):
    checks: tuple[FindingCheck, ...] = Field(min_length=1)


class ClaimDraft(Record):
    statement: ProblemText
    source_ids: tuple[ProblemText, ...] = Field(min_length=1)
    mechanism: ProblemText
    assumptions: tuple[ProblemText, ...]
    limitations: tuple[ProblemText, ...]


class LiteratureObservation(Record):
    description: ProblemText
    evidence_ids: tuple[ProblemText, ...] = Field(min_length=1)


class LiteratureSynthesis(Record):
    summary: ProblemText
    claims: tuple[ClaimDraft, ...] = Field(min_length=1, max_length=12)
    gaps: tuple[LiteratureObservation, ...]
    conflicts: tuple[LiteratureObservation, ...]
    coverage_limitations: tuple[ProblemText, ...] = Field(min_length=1)


class LiteratureOverview(Record):
    summary: ProblemText
    gaps: tuple[LiteratureObservation, ...]
    conflicts: tuple[LiteratureObservation, ...]
    coverage_limitations: tuple[ProblemText, ...] = Field(min_length=1)


class HypothesisReviewItem(Record):
    hypothesis_id: ProblemText
    verdict: Literal["ready", "revise", "defer"]
    evidence_ids: tuple[ProblemText, ...]
    closest_paper_ids: tuple[ProblemText, ...]
    novelty: Literal["known", "incremental", "not_found_in_search", "uncertain"]
    rationale: ProblemText
    issues: tuple[ProblemText, ...]
    revision_requests: tuple[ProblemText, ...]
    human_summary: ProblemText = Field(max_length=100)
    recommended_test: ProblemText = Field(max_length=160)

    @model_validator(mode="after")
    def actionable_revision(self):
        if self.verdict == "revise" and not self.revision_requests:
            raise ValueError("Revision verdict requires actionable requests")
        if any("\n" in s or "\r" in s for s in (self.human_summary, self.recommended_test)):
            raise ValueError("Human summaries must be single-line")
        return self


class HypothesisComparison(Record):
    condition: ProblemText
    predictions: dict[ProblemText, ProblemText] = Field(min_length=2)
    measurement: ProblemText


class ReviewBatch(Record):
    reviews: tuple[HypothesisReviewItem, ...] = Field(min_length=1)
    comparisons: tuple[HypothesisComparison, ...]
    ranking: tuple[ProblemText, ...] = Field(min_length=1)
    search_requests: tuple[SearchQuery, ...] = Field(max_length=3)
    limitations: tuple[ProblemText, ...] = Field(min_length=1)


class HypothesisReviewRecord(Record):
    id: Text
    round: int = Field(ge=0)
    batch: ReviewBatch
