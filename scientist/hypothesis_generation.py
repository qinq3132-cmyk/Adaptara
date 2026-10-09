"""Generate and persist hypotheses without invoking the fixed toy planner."""
import hashlib
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field

from scientist.agents.llm_hypothesis import LLMHypothesisAgent
from scientist.backends.opencode import OpenCodeBackend
from scientist.backends.codex import CodexBackend
from scientist.provenance import capture_source, runtime_versions
from scientist.schemas import Decision, HypothesisOutput, Record, Stage, Text
from scientist.state import ScientificState
from scientist.tools.experiment_tracker import ResearchStore, json_bytes


class ReasoningConfig(Record):
    backend: Literal["opencode", "codex"] = "opencode"
    model: Text
    executable: str | None = None
    config: str | None = None
    proxy_env_precedence: Literal["inherit", "uppercase"] = "inherit"
    timeout_seconds: float = Field(default=180, gt=0, le=600)
    hypothesis_count: int = Field(default=3, ge=3, le=6)
    validation_retries: int = Field(default=1, ge=0, le=2)
    codex_executable: str | None = None
    codex_auth_mode: Literal["opencode", "codex"] = "opencode"
    codex_credential_provider: str | None = None
    codex_base_url: str | None = None
    codex_provider_id: str = "OpenAI"
    codex_sandbox_backend: Literal["default", "landlock"] = "default"
    codex_model_catalog: str | None = None
    codex_reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    codex_reasoning_summary: Literal["auto", "concise", "detailed", "none"] | None = None


def make_reasoning_backend(config: ReasoningConfig):
    common = dict(model=config.model, config=Path(config.config) if config.config else None,
                  timeout_seconds=config.timeout_seconds, proxy_env_precedence=config.proxy_env_precedence)
    if config.backend == "codex":
        return CodexBackend(**common, executable=config.codex_executable, auth_mode=config.codex_auth_mode,
                            base_url=config.codex_base_url, provider_id=config.codex_provider_id,
                            credential_provider=config.codex_credential_provider,
                            sandbox_backend=config.codex_sandbox_backend,
                            model_catalog=Path(config.codex_model_catalog) if config.codex_model_catalog else None,
                            reasoning_effort=config.codex_reasoning_effort, reasoning_summary=config.codex_reasoning_summary)
    return OpenCodeBackend(**common, executable=config.executable)


def make_hypothesis_agent(config: ReasoningConfig) -> LLMHypothesisAgent:
    backend = make_reasoning_backend(config)
    return LLMHypothesisAgent(backend, count=config.hypothesis_count, max_retries=config.validation_retries)


def hypothesis_report(output: HypothesisOutput, model: str, backend: str = "opencode") -> str:
    lines = ["# 模型生成的候选假设", "", f"执行后端：{backend}；请求模型：`{model}`。", "",
             "所有假设均为 proposed；0.5 是统一初始化的内部信念分数，不是模型给出的概率。", "",
             "本次仅生成和校验候选，尚未执行实验或科学批判，也未验证假设正确性。", ""]
    for h in output.hypotheses:
        lines += [f"## {h.id}", "", h.claim, "", f"机制：{h.mechanism}", ""]
        for label, values in (("假设条件", h.assumptions), ("预测", h.predictions),
                              ("替代解释", h.alternative_explanations), ("可证伪条件", h.falsification_conditions)):
            lines += [f"**{label}**", ""] + [f"- {value}" for value in values] + [""]
        if h.parent_hypotheses or h.source_claims:
            lines += [f"来源：父假设 {list(h.parent_hypotheses)}；已有 claim {list(h.source_claims)}。", ""]
    return "\n".join(lines)


def generate_hypotheses(*, state: ScientificState, agent: LLMHypothesisAgent,
                        store: ResearchStore, source_root: Path, artifact_root: Path, emit=print):
    initial = ScientificState.model_validate_json(state.model_dump_json())
    run_id = "run_" + uuid4().hex
    sources = capture_source(source_root)
    emit(f"Hypothesis generation started ({getattr(agent.backend, 'name', 'test')}); no experiment will execute")
    raw = agent.propose(initial.model_copy(deep=True))
    output = HypothesisOutput.model_validate(raw.model_dump())
    if any(h.status != "proposed" or h.supporting_evidence or h.contradicting_evidence for h in output.hypotheses):
        raise ValueError("New hypotheses must be untested proposals")
    data = initial.model_dump()
    occupied = {initial.question.id}
    for value in data.values():
        if isinstance(value, (list, tuple)):
            occupied.update(item["id"] for item in value if isinstance(item, dict) and "id" in item)
    occupied.update(h.id for h in output.hypotheses)
    number = len(initial.decisions) + 1
    while f"D{number:03d}" in occupied:
        number += 1
    decision = Decision(id=f"D{number:03d}", stage=Stage.HYPOTHESIS,
                        summary="Validated model-authored proposals; experiments and critique are pending",
                        related_ids=tuple(h.id for h in output.hypotheses))
    final = initial.revised(hypotheses=initial.hypotheses + output.hypotheses,
                            decisions=initial.decisions + (decision,), stage=Stage.HYPOTHESIS)
    model = agent.attempts[-1]["runtime"].get("model", "injected test backend")
    backend = agent.attempts[-1]["runtime"].get("backend", "test")
    exports = {"question.json": initial.question.model_dump(mode="json"),
               "hypotheses.json": output.model_dump(mode="json"),
               "input_state.json": initial.model_dump(mode="json"),
               "research_state.json": final.model_dump(mode="json"),
               "source_snapshot.json": sources, "runtime.json": runtime_versions(),
               "reasoning.json": {"purpose": "hypothesis_generation_only", "model": model, "backend": backend,
                                  "attempts": agent.attempts,
                                  "input_state_sha256": hashlib.sha256(json_bytes(initial.model_dump(mode="json"))).hexdigest(),
                                  "source_sha256": hashlib.sha256(json_bytes(sources)).hexdigest()},
               "report.md": hypothesis_report(output, model, backend).encode("utf-8")}
    directory = artifact_root / run_id
    store.save(run_id, final, directory, exports)
    store.verify_artifacts(run_id)
    for h in output.hypotheses:
        emit(f"{h.id} [proposed]: {h.claim}")
    emit(f"Hypotheses saved and verified: {directory.resolve()}")
    return final
