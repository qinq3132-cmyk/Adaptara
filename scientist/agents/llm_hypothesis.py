"""Model-authored hypotheses; Python owns IDs, status and scientific memory."""
import json
import re
from pathlib import Path
from typing import Protocol

from pydantic import Field

from scientist.backends.common import ReasoningResponse, unwrap_json_fence
from scientist.schemas import Hypothesis, HypothesisOutput, Record, Text
from scientist.state import ScientificState

PROMPT_FILE = Path(__file__).resolve().parents[1] / "prompts/hypothesis.txt"


class DraftHypothesis(Record):
    claim: Text
    mechanism: Text
    assumptions: tuple[Text, ...] = Field(min_length=1)
    predictions: tuple[Text, ...] = Field(min_length=1)
    alternative_explanations: tuple[Text, ...] = Field(min_length=1)
    falsification_conditions: tuple[Text, ...] = Field(min_length=1)
    parent_hypotheses: tuple[Text, ...]
    source_claims: tuple[Text, ...]


class DraftBatch(Record):
    hypotheses: tuple[DraftHypothesis, ...] = Field(min_length=3, max_length=6)


class ReasoningBackend(Protocol):
    def complete(self, *, instructions: str, prompt: str) -> ReasoningResponse: ...


def hypothesis_context(state: ScientificState) -> dict:
    """Use recorded evidence/metrics, not entire raw trajectories or credentials."""
    context = {"question": state.question.model_dump(mode="json"),
            "knowledge": [c.model_dump(mode="json") for c in state.knowledge],
            "existing_hypotheses": [h.model_dump(mode="json") for h in state.hypotheses],
            "evidence": [e.model_dump(mode="json") for e in state.evidence],
            "results": [{"id": r.id, "experiment_id": r.experiment_id, "kind": r.kind, "metrics": r.metrics}
                        for r in state.results],
            "experiments": [e.model_dump(mode="json") for e in state.experiments],
            "analyses": [a.model_dump(mode="json") for a in state.analyses],
            "critiques": [c.model_dump(mode="json") for c in state.critiques],
            "open_questions": [q.model_dump(mode="json") for q in state.open_questions]}
    if state.problem_definition is not None:
        context["problem_definition"] = state.problem_definition.model_dump(mode="json")
    if state.schema_version >= 3:
        context["papers"] = [p.model_dump(mode="json") for p in state.papers]
        context["literature"] = state.literature.model_dump(mode="json") if state.literature else None
        context["hypothesis_reviews"] = [r.model_dump(mode="json") for r in state.hypothesis_reviews]
    return context


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def allocate_ids(state: ScientificState, count: int) -> tuple[str, ...]:
    data = state.model_dump()
    existing = {state.question.id}
    for value in data.values():
        if isinstance(value, (list, tuple)):
            existing.update(item["id"] for item in value if isinstance(item, dict) and "id" in item)
    maximum = max((int(h.id[1:]) for h in state.hypotheses if re.fullmatch(r"H\d+", h.id)), default=0)
    ids = []
    candidate = maximum + 1
    while len(ids) < count:
        identifier = f"H{candidate:03d}"
        candidate += 1
        if identifier not in existing:
            ids.append(identifier)
    return tuple(ids)


class LLMHypothesisAgent:
    def __init__(self, backend: ReasoningBackend, *, count: int = 3, max_retries: int = 1):
        if not 3 <= count <= 6 or not 0 <= max_retries <= 2:
            raise ValueError("Request 3-6 hypotheses and at most two validation retries")
        self.backend, self.count, self.max_retries = backend, count, max_retries
        self.attempts = []

    def propose(self, state: ScientificState) -> HypothesisOutput:
        state = ScientificState.model_validate_json(state.model_dump_json())
        instructions = PROMPT_FILE.read_text(encoding="utf-8")
        context = hypothesis_context(state)
        schema = DraftBatch.model_json_schema()
        schema["properties"]["hypotheses"].update(minItems=self.count, maxItems=self.count)
        prompt = json.dumps({"task": f"必须恰好生成 {self.count} 条相互竞争的可证伪假设，多一条或少一条都会被拒绝。只返回一个合法 JSON 对象。每个列表保留 1–2 条具体内容，保证简洁。",
                             "required_count": self.count, "scientific_context": context, "output_schema": schema},
                            ensure_ascii=False, sort_keys=True)
        self.attempts = []
        for attempt in range(self.max_retries + 1):
            # Transport failures are not retried or replaced by mock hypotheses.
            response = self.backend.complete(instructions=instructions, prompt=prompt)
            entry = {"attempt": attempt + 1, "instructions": instructions, "prompt": prompt,
                     "response_text": response.text, "runtime": response.metadata, "accepted": False}
            self.attempts.append(entry)
            try:
                batch = self._validate(response.text, state)
            except ValueError:
                entry["validation_error"] = "Invalid structure, count, duplicate content or scientific references"
                if attempt == self.max_retries:
                    raise ValueError("Model hypothesis proposals failed validation; no scientific state was committed") from None
                prompt = json.dumps({"task": f"上一次输出未通过校验。重新生成恰好 {self.count} 条假设（不要额外的第四条等记录），返回一个合法 JSON 对象，禁止尾逗号和多余的大括号。每个列表保留 1–2 条具体内容。",
                                     "required_count": self.count, "scientific_context": context, "output_schema": schema,
                                     "validation_requirements": "Exactly the requested count; all fields present; unique claims/mechanisms; references only to provided IDs; no extra fields."},
                                    ensure_ascii=False, sort_keys=True)
                continue
            ids = allocate_ids(state, len(batch.hypotheses))
            output = HypothesisOutput(hypotheses=tuple(
                Hypothesis(id=identifier, **draft.model_dump(), status="proposed", confidence=0.5)
                for identifier, draft in zip(ids, batch.hypotheses)))
            state.revised(hypotheses=state.hypotheses + output.hypotheses)
            entry["accepted"] = True
            return output
        raise RuntimeError("Unreachable hypothesis validation state")

    def _validate(self, response: str, state: ScientificState) -> DraftBatch:
        raw = unwrap_json_fence(response)
        batch = DraftBatch.model_validate_json(raw)
        if len(batch.hypotheses) != self.count:
            raise ValueError("Wrong hypothesis count")
        prior_claims = {normalized(h.claim) for h in state.hypotheses}
        claims, mechanisms = set(), set()
        existing_hypotheses = {h.id for h in state.hypotheses}
        source_claims = {c.id for c in state.knowledge}
        for draft in batch.hypotheses:
            claim, mechanism = normalized(draft.claim), normalized(draft.mechanism)
            if claim in claims or claim in prior_claims or mechanism in mechanisms:
                raise ValueError("Duplicate hypothesis claim or mechanism")
            if set(draft.parent_hypotheses) - existing_hypotheses:
                raise ValueError(f"Unknown parent_hypotheses; allowed hypothesis IDs: {sorted(existing_hypotheses)}")
            if set(draft.source_claims) - source_claims:
                raise ValueError(f"Unknown source_claims; use knowledge[].id only, not evidence or paper IDs. Allowed claim IDs: {sorted(source_claims)}")
            if state.literature is not None and source_claims and not draft.source_claims:
                raise ValueError("Literature-grounded proposals must cite supplied claims")
            for text in (draft.claim, draft.mechanism, *draft.assumptions, *draft.predictions,
                         *draft.alternative_explanations, *draft.falsification_conditions):
                if not text.strip():
                    raise ValueError("Blank scientific text")
            claims.add(claim)
            mechanisms.add(mechanism)
        return batch
