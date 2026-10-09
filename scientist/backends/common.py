"""Runtime-neutral responses and failures."""
from dataclasses import dataclass


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
