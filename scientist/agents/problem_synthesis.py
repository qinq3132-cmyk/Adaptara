"""Synthesize independently generated drafts without human confirmation authority."""
from pathlib import Path
from typing import Literal

from pydantic import Field

from scientist.agents.problem_clarification import ProblemClarificationAgent, clarification_output_schema
from scientist.backends.common import unwrap_json_fence
from scientist.problem_schemas import ProblemDraft, ProblemText
from scientist.schemas import Record

SYNTHESIS_PROMPT_FILE = Path(__file__).resolve().parents[1] / "prompts/problem_synthesis.txt"


class CandidateDraft(Record):
    candidate_id: ProblemText
    model: ProblemText
    draft: ProblemDraft


class Contribution(Record):
    candidate_id: ProblemText
    retained: tuple[ProblemText, ...]
    corrected_or_omitted: tuple[ProblemText, ...]


class Disagreement(Record):
    topic: ProblemText
    candidate_ids: tuple[ProblemText, ...] = Field(min_length=2)
    interpretations: tuple[ProblemText, ...] = Field(min_length=2)
    handling: ProblemText
    status: Literal["kept_open", "needs_human_choice", "resolved_from_input"]
    final_issue_ids: tuple[ProblemText, ...]


class IssueMapping(Record):
    candidate_id: ProblemText
    issue_id: ProblemText
    integrated_issue_ids: tuple[ProblemText, ...]
    status: Literal["merged", "kept_open", "deferred", "already_explicit", "unsupported"]
    reason: ProblemText


class SynthesisAnalysis(Record):
    contributions: tuple[Contribution, ...] = Field(min_length=3, max_length=3)
    disagreements: tuple[Disagreement, ...]
    issue_mapping: tuple[IssueMapping, ...]
    limitations: tuple[ProblemText, ...]


class SynthesisOutput(Record):
    draft: ProblemDraft
    analysis: SynthesisAnalysis


class ProblemSynthesisAgent(ProblemClarificationAgent):
    def __init__(self, backend, candidates: tuple[CandidateDraft, ...], *, max_retries: int = 1):
        super().__init__(backend, max_retries=max_retries)
        if len(candidates) != 3 or len({c.candidate_id for c in candidates}) != 3:
            raise ValueError("Synthesis requires three uniquely identified candidates")
        self.candidates = tuple(CandidateDraft.model_validate_json(c.model_dump_json()) for c in candidates)
        self.analysis: SynthesisAnalysis | None = None

    def prepare_request(self, concept, previous_draft=None, feedback=None):
        for candidate in self.candidates:
            candidate.draft.validate_context(concept)
        instructions, request = super().prepare_request(concept, previous_draft, feedback)
        instructions += "\n\n" + SYNTHESIS_PROMPT_FILE.read_text(encoding="utf-8")
        request.update(task="整合三个独立问题草案，返回完整统一草案及可追溯的整合分析，不自行确认。",
                       candidates=[c.model_dump(mode="json") for c in self.candidates],
                       output_schema=clarification_output_schema(SynthesisOutput.model_json_schema()))
        return instructions, request

    def parse_response(self, text):
        output = SynthesisOutput.model_validate_json(unwrap_json_fence(text))
        known = {c.candidate_id for c in self.candidates}
        contributions = [c.candidate_id for c in output.analysis.contributions]
        if len(set(contributions)) != len(contributions) or set(contributions) != known:
            raise ValueError("Synthesis must account for every candidate")
        expected = {(c.candidate_id, i.id) for c in self.candidates for i in c.draft.pending_issues}
        mapped = [(m.candidate_id, m.issue_id) for m in output.analysis.issue_mapping]
        if len(set(mapped)) != len(mapped) or set(mapped) != expected:
            raise ValueError("Synthesis must map every candidate issue exactly once")
        final = {i.id for i in output.draft.pending_issues}
        for mapping in output.analysis.issue_mapping:
            if set(mapping.integrated_issue_ids) - final:
                raise ValueError("Synthesis refers to unknown integrated issues")
            if mapping.status == "merged" and not mapping.integrated_issue_ids:
                raise ValueError("Merged issues need an integrated issue ID")
        for disagreement in output.analysis.disagreements:
            if set(disagreement.candidate_ids) - known or len(set(disagreement.candidate_ids)) < 2:
                raise ValueError("Disagreement refers to invalid candidates")
            if set(disagreement.final_issue_ids) - final:
                raise ValueError("Disagreement refers to unknown integrated issues")
            if disagreement.status == "needs_human_choice" and not disagreement.final_issue_ids:
                raise ValueError("Human choices must appear in the confirmation sheet")
        self.analysis = output.analysis
        return output.draft


def synthesis_report(analysis: SynthesisAnalysis) -> str:
    lines = ["", "## 整合依据（供审阅与后续 Agent 使用）", ""]
    for contribution in analysis.contributions:
        lines.extend([f"### {contribution.candidate_id}", "",
                      *[f"- 采纳：{text}" for text in contribution.retained],
                      *[f"- 修正或舍弃：{text}" for text in contribution.corrected_or_omitted], ""])
    lines.extend(["### 分歧处理", ""])
    for item in analysis.disagreements:
        lines.extend([f"- {item.topic}（{', '.join(item.candidate_ids)}）：{item.handling}",
                      f"  解释：{'；'.join(item.interpretations)}；处理状态：{item.status}"])
    lines.extend(["", "逐项映射与保留限制见 synthesis.json。", ""])
    return "\n".join(lines)
