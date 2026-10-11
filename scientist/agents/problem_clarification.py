"""LLM proposes problem drafts; it has no confirmation or persistence authority."""
import json
from pathlib import Path

from scientist.agents.llm_hypothesis import ReasoningBackend
from scientist.backends.common import unwrap_json_fence
from scientist.problem_schemas import ConceptInput, HumanFeedback, ProblemDraft

PROMPT_FILE = Path(__file__).resolve().parents[1] / "prompts/problem_clarification.txt"
HUMAN_QUESTION_MAX = 40
HUMAN_REASON_MAX = 50
HUMAN_OPTION_MAX = 24
HUMAN_OPTIONS_MAX = 4


def clarification_output_schema(schema: dict | None = None) -> dict:
    # These bounds apply to new model responses, not historical persisted
    # records whose contents and checksums must remain readable unchanged.
    schema = schema if schema is not None else ProblemDraft.model_json_schema()
    fields = schema["$defs"]["ClarificationIssue"]["properties"]
    fields["question"]["maxLength"] = HUMAN_QUESTION_MAX
    fields["reason"]["maxLength"] = HUMAN_REASON_MAX
    fields["options"]["maxItems"] = HUMAN_OPTIONS_MAX
    fields["options"]["items"]["maxLength"] = HUMAN_OPTION_MAX
    return schema


def validate_human_review(draft: ProblemDraft) -> None:
    for issue in draft.pending_issues:
        fields = [(issue.question, HUMAN_QUESTION_MAX), (issue.reason, HUMAN_REASON_MAX)]
        fields.extend((option, HUMAN_OPTION_MAX) for option in issue.options)
        if any(len(text) > limit or "\n" in text or "\r" in text for text, limit in fields):
            raise ValueError("Human review text must be short and single-line")
        if len(issue.options) == 1 or len(issue.options) > HUMAN_OPTIONS_MAX:
            raise ValueError("Human review requires zero or 2-4 options")
        if len({option.casefold() for option in issue.options}) != len(issue.options):
            raise ValueError("Human review options must be distinct")


class ProblemClarificationAgent:
    def __init__(self, backend: ReasoningBackend, *, max_retries: int = 1):
        if not 0 <= max_retries <= 2:
            raise ValueError("At most two validation retries are supported")
        self.backend, self.max_retries = backend, max_retries
        self.attempts = []

    def prepare_request(self, concept: ConceptInput, previous_draft: ProblemDraft | None = None,
                        feedback: HumanFeedback | None = None) -> tuple[str, dict]:
        instructions = PROMPT_FILE.read_text(encoding="utf-8")
        request = {"task": "解析科研概念，返回中文问题草案，不自行确认。",
                   "concept": concept.model_dump(mode="json"),
                   "previous_draft": previous_draft.model_dump(mode="json") if previous_draft else None,
                   "feedback": feedback.model_dump(mode="json") if feedback else None,
                   "output_schema": clarification_output_schema()}
        return instructions, request

    def parse_response(self, text: str) -> ProblemDraft:
        return ProblemDraft.model_validate_json(unwrap_json_fence(text))

    def analyze(self, concept: ConceptInput, previous_draft: ProblemDraft | None = None,
                feedback: HumanFeedback | None = None) -> ProblemDraft:
        concept = ConceptInput.model_validate_json(concept.model_dump_json())
        if (previous_draft is None) != (feedback is None):
            raise ValueError("A revision requires both the previous draft and human feedback")
        instructions, request = self.prepare_request(concept, previous_draft, feedback)
        self.attempts = []
        for attempt in range(self.max_retries + 1):
            prompt = json.dumps(request, ensure_ascii=False, sort_keys=True)
            response = self.backend.complete(instructions=instructions, prompt=prompt)
            entry = {"attempt": attempt + 1, "instructions": instructions, "prompt": prompt,
                     "response_text": response.text, "runtime": response.metadata, "accepted": False}
            self.attempts.append(entry)
            try:
                draft = self.parse_response(response.text)
                validate_human_review(draft)
                draft.validate_context(concept)
                validate_revision(previous_draft, feedback, draft)
            except ValueError:
                entry["validation_error"] = "Invalid structure, source, constraint, unresolved issue or concise human review"
                if attempt == self.max_retries:
                    raise ValueError("Problem draft failed validation; no revision was committed") from None
                request["validation_requirements"] = (
                    "Match the complete schema; preserve explicit constraints verbatim with user origin; "
                    "use only provided sources; retain unanswered blocking issue IDs. Human review: question <=40 "
                    "characters, reason <=50, each option <=24, zero or 2-4 distinct options, no newlines. "
                    "Move detailed explanations to analytical fields; do not truncate or remove decisions. "
                    "When synthesizing, include all candidate contributions and map every candidate issue to valid final IDs "
                    "or explain its omission. Return valid JSON.")
                continue
            entry["accepted"] = True
            return draft
        raise RuntimeError("Unreachable validation state")


def validate_revision(previous: ProblemDraft | None, feedback: HumanFeedback | None,
                      draft: ProblemDraft) -> None:
    if previous is None:
        return
    if feedback is None:
        raise ValueError("Human feedback is required")
    known = {issue.id for issue in previous.pending_issues}
    if set(feedback.answers) - known:
        raise ValueError("Feedback refers to unknown issue IDs")
    pending = {issue.id: issue for issue in draft.pending_issues}
    for issue in previous.pending_issues:
        if issue.blocking and issue.id not in feedback.answers:
            if issue.id not in pending or not pending[issue.id].blocking:
                raise ValueError("Unanswered blocking issues cannot be removed or downgraded")
