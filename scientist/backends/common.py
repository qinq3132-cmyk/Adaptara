"""Runtime-neutral responses and failures."""
from dataclasses import dataclass

TOOL_FREE_ROLES = {"hypothesis", "problem_clarification", "problem_synthesis", "literature_search",
                   "literature_screening", "literature_reading", "evidence_verification",
                   "literature_synthesis", "hypothesis_review", "hypothesis_revision"}


@dataclass(frozen=True)
class ReasoningResponse:
    text: str
    metadata: dict


class BackendError(RuntimeError):
    """A failed runtime invocation; never substitute a mock response."""


def unwrap_json_fence(text: str) -> str:
    """Allow exactly one complete JSON Markdown fence, never extract fragments."""
    raw = text.strip()
    if raw.startswith("```json\n") and raw.endswith("```"):
        return raw[len("```json\n"):-3].strip()
    return raw
