"""Three independent clarification drafts followed by a traceable synthesis."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from .agents.problem_clarification import ProblemClarificationAgent, validate_human_review, validate_revision
from .agents.problem_synthesis import CandidateDraft, ProblemSynthesisAgent, synthesis_report
from .hypothesis_generation import ReasoningConfig, make_reasoning_backend
from .problem_clarification import human_review_report, now, problem_report
from .problem_schemas import ProblemDraft, ProblemRevision, ProblemText
from .schemas import Record
from .tools.experiment_tracker import json_bytes


class ClarificationModel(Record):
    model: ProblemText
    backend: Literal["codex", "opencode"] = "codex"
    auth_mode: Literal["opencode", "codex", "chatgpt"] = "opencode"
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "high"
    timeout_seconds: float = Field(default=300, gt=0, le=600)

    def runtime_config(self, common: ReasoningConfig) -> ReasoningConfig:
        values = {**common.model_dump(), "model": self.model, "backend": self.backend,
                  "codex_auth_mode": self.auth_mode, "codex_reasoning_effort": self.reasoning_effort,
                  "timeout_seconds": self.timeout_seconds}
        if self.auth_mode == "chatgpt":
            if self.backend != "codex":
                raise ValueError("Official ChatGPT authentication requires Codex")
            values.update(codex_base_url=None, codex_provider_id="openai", codex_credential_provider=None)
        return ReasoningConfig.model_validate(values)


class ClarificationConfig(Record):
    mode: Literal["single", "ensemble"] = "ensemble"
    candidates: tuple[ClarificationModel, ...] = Field(min_length=3, max_length=3, default=(
        ClarificationModel(model="glm-5.3"), ClarificationModel(model="deepseek-v4.1-flash"),
        ClarificationModel(model="gpt-6.1-sol", auth_mode="chatgpt", timeout_seconds=600)))
    synthesizer: ClarificationModel = ClarificationModel(
        model="gpt-6.1-sol", auth_mode="chatgpt", timeout_seconds=600)

    @model_validator(mode="after")
    def distinct_candidates(self):
        if len({c.model for c in self.candidates}) != 3:
            raise ValueError("Clarification requires three distinct candidate models")
        return self


def runtime_identity(config: ReasoningConfig) -> dict:
    chatgpt = config.codex_auth_mode == "chatgpt"
    executable = (config.codex_chatgpt_executable or config.codex_executable) if chatgpt else config.codex_executable
    sandbox = (config.codex_chatgpt_sandbox_backend or config.codex_sandbox_backend) if chatgpt else config.codex_sandbox_backend
    # Timeouts and validation retry counts do not change an accepted draft's input.
    return {"backend": config.backend, "model": config.model, "auth_mode": config.codex_auth_mode,
            "reasoning_effort": config.codex_reasoning_effort, "reasoning_summary": config.codex_reasoning_summary,
            "executable": executable if config.backend == "codex" else config.executable,
            "sandbox_backend": sandbox, "base_url": config.codex_base_url,
            "provider_id": config.codex_provider_id, "credential_provider": config.codex_credential_provider,
            "config": config.config, "proxy_env_precedence": config.proxy_env_precedence,
            "model_catalog": config.codex_model_catalog}


def verify_manifest(folder: Path, expected_digest: str | None = None) -> None:
    manifest_path = folder / "manifest.json"
    if expected_digest and hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError("Clarification cache manifest checksum mismatch")
    hashes = json.loads(manifest_path.read_bytes())["sha256"]
    for name, digest in hashes.items():
        if Path(name).name != name:
            raise ValueError("Invalid clarification cache artifact path")
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Clarification cache artifact checksum mismatch")


@dataclass
class StageResult:
    draft: ProblemDraft
    attempts: list
    cache_dir: Path
    reused: bool


class DeferredBackend:
    """Cached results can be read without initializing a credentialed runtime."""
    def __init__(self, factory, config, role):
        self.factory, self.config, self.role = factory, config, role
        self.backend = None

    def complete(self, *, instructions, prompt):
        if self.backend is None:
            self.backend = self.factory(self.config, role=self.role)
        return self.backend.complete(instructions=instructions, prompt=prompt)


class ClarificationCache:
    def __init__(self, artifact_root: Path):
        self.artifact_root = Path(artifact_root)
        self.root = self.artifact_root / "clarification_cache"

    def run(self, *, config, role, agent, concept, previous=None, feedback=None) -> StageResult:
        instructions, request = agent.prepare_request(concept, previous, feedback)
        identity = {"role": role, "runtime": runtime_identity(config),
                    "instructions_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
                    "request_sha256": hashlib.sha256(json_bytes(request)).hexdigest()}
        key = hashlib.sha256(json_bytes(identity)).hexdigest()
        slot = self.root / key
        pointer = slot / "accepted.json"
        if pointer.exists():
            saved = json.loads(pointer.read_bytes())
            name = saved["attempt"]
            if Path(name).name != name:
                raise ValueError("Invalid clarification cache pointer")
            folder = slot / name
            verify_manifest(folder, saved["manifest_sha256"])
            if json.loads((folder / "identity.json").read_bytes()) != identity:
                raise ValueError("Clarification cache identity mismatch")
            attempts = json.loads((folder / "reasoning.json").read_bytes())["attempts"]
            draft = self._parse(agent, attempts, config, instructions, request, concept, previous, feedback)
            return StageResult(draft, attempts, folder, True)

        imported = self._find_existing(config, instructions, request) if role == "problem_clarification" else None
        try:
            if imported:
                source, attempts = imported
                draft = self._parse(agent, attempts, config, instructions, request, concept, previous, feedback)
            else:
                source = None
                draft = agent.analyze(concept, previous, feedback)
                attempts = agent.attempts
                # Verify that the live call used the exact request used to name this cache slot.
                draft = self._parse(agent, attempts, config, instructions, request, concept, previous, feedback)
        except (ValueError, RuntimeError, OSError) as error:
            failed = slot / ("failed_" + uuid4().hex)
            failed.mkdir(parents=True)
            (failed / "reasoning.json").write_bytes(json_bytes({"attempts": agent.attempts, "accepted": False,
                                                               "error_type": type(error).__name__}))
            raise

        record = ProblemRevision(session_id="problem_" + uuid4().hex, revision=1,
                                 concept=concept, draft=draft, feedback=None, previous_draft_sha256=None,
                                 created_at=now())
        report = problem_report(record)
        analysis = getattr(agent, "analysis", None)
        if analysis is not None:
            report += synthesis_report(analysis)
        exports = {"identity.json": identity, "settings.json": config.model_dump(mode="json"),
                   "request.json": request, "instructions.txt": instructions.encode(),
                   "reasoning.json": {"purpose": role, "attempts": attempts},
                   "draft.json": draft.model_dump(mode="json"),
                   "report.md": report.encode(), "review.md": human_review_report(record).encode(),
                   "provenance.json": {"imported_from": str(source) if source else None}}
        if analysis is not None:
            exports["analysis.json"] = analysis.model_dump(mode="json")
        folder = slot / ("attempt_" + uuid4().hex)
        folder.mkdir(parents=True)
        hashes = {}
        for name, value in exports.items():
            data = value if isinstance(value, bytes) else json_bytes(value)
            (folder / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        (folder / "manifest.json").write_bytes(json_bytes({"sha256": hashes}))
        temporary = slot / ("accepted_" + uuid4().hex + ".tmp")
        temporary.write_bytes(json_bytes({"attempt": folder.name,
                                         "manifest_sha256": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()}))
        temporary.replace(pointer)
        return StageResult(draft, attempts, folder, bool(imported))

    @staticmethod
    def _parse(agent, attempts, config, instructions, request, concept, previous, feedback):
        if not attempts or not attempts[-1].get("accepted"):
            raise ValueError("Cache does not contain an accepted model response")
        for attempt in attempts:
            actual = json.loads(attempt["prompt"])
            actual.pop("validation_requirements", None)
            if attempt["instructions"] != instructions or actual != request:
                raise ValueError("Clarification response input mismatch")
            runtime = attempt["runtime"]
            if runtime.get("model") != config.model:
                raise ValueError("Clarification response model mismatch")
        draft = agent.parse_response(attempts[-1]["response_text"])
        validate_human_review(draft)
        draft.validate_context(concept)
        validate_revision(previous, feedback, draft)
        return draft

    def _find_existing(self, config, instructions, request):
        if not self.artifact_root.exists():
            return None
        for path in sorted(self.artifact_root.rglob("reasoning.json")):
            if self.root in path.parents:
                continue
            settings = path.parent / "settings.json"
            if not settings.exists():
                settings = path.parent.parent / "settings.json"
            if not settings.exists() or not (path.parent / "manifest.json").exists():
                continue
            try:
                saved_config = ReasoningConfig.model_validate_json(settings.read_bytes())
                if runtime_identity(saved_config) != runtime_identity(config):
                    continue
                attempts = json.loads(path.read_bytes()).get("attempts", [])
                if not attempts or not attempts[-1].get("accepted"):
                    continue
                last = attempts[-1]
                actual = json.loads(last["prompt"])
                actual.pop("validation_requirements", None)
                if last["instructions"] != instructions or actual != request:
                    continue
                runtime = last["runtime"]
                if runtime.get("model") != config.model or runtime.get("role") != "problem_clarification":
                    continue
                if runtime.get("reasoning_effort_requested") != config.codex_reasoning_effort:
                    continue
                verify_manifest(path.parent)
            except (ValueError, KeyError, OSError):
                continue
            return path, attempts
        return None


class MultiModelProblemClarificationAgent:
    def __init__(self, common: ReasoningConfig, config: ClarificationConfig, artifact_root: Path,
                 *, backend_factory=make_reasoning_backend, emit=print):
        self.common, self.config = common, config
        self.cache = ClarificationCache(artifact_root)
        self.backend_factory, self.emit = backend_factory, emit
        self.attempts = []
        self.exports = {}
        self.report_appendix = ""

    def analyze(self, concept, previous_draft=None, feedback=None):
        self.attempts, self.exports, self.report_appendix = [], {}, ""
        candidates, stages, errors = {}, {}, []

        def candidate_stage(index, member):
            config = member.runtime_config(self.common)
            agent = ProblemClarificationAgent(DeferredBackend(self.backend_factory, config, "problem_clarification"),
                                              max_retries=config.validation_retries)
            stage = self.cache.run(config=config, role="problem_clarification", agent=agent,
                                   concept=concept.model_copy(deep=True),
                                   previous=previous_draft.model_copy(deep=True) if previous_draft else None,
                                   feedback=feedback.model_copy(deep=True) if feedback else None)
            self.emit(f"Clarification {member.model}: {'reused' if stage.reused else 'generated'}")
            return CandidateDraft(candidate_id=f"C{index + 1:03d}", model=member.model, draft=stage.draft), stage

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(candidate_stage, index, member): index
                       for index, member in enumerate(self.config.candidates)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    candidates[index], stages[index] = future.result()
                except (ValueError, RuntimeError, OSError) as error:
                    errors.append((self.config.candidates[index].model, error))
        if errors:
            model, error = errors[0]
            raise RuntimeError(f"Clarification candidate {model} failed: {error}; "
                               "successful drafts were cached; synthesis was not run") from error
        ordered = tuple(candidates[index] for index in range(3))
        config = self.config.synthesizer.runtime_config(self.common)
        agent = ProblemSynthesisAgent(DeferredBackend(self.backend_factory, config, "problem_synthesis"), ordered,
                                      max_retries=config.validation_retries)
        self.emit(f"Synthesis started: {config.model} ({config.codex_reasoning_effort})")
        synthesis = self.cache.run(config=config, role="problem_synthesis", agent=agent,
                                   concept=concept.model_copy(deep=True),
                                   previous=previous_draft.model_copy(deep=True) if previous_draft else None,
                                   feedback=feedback.model_copy(deep=True) if feedback else None)
        self.emit(f"Synthesis {config.model}: {'reused' if synthesis.reused else 'generated'}")
        for index, candidate in enumerate(ordered):
            stage = stages[index]
            self.attempts.extend({**attempt, "phase": "candidate", "candidate_id": candidate.candidate_id,
                                  "reused": stage.reused} for attempt in stage.attempts)
        self.attempts.extend({**attempt, "phase": "synthesis", "reused": synthesis.reused}
                             for attempt in synthesis.attempts)
        self.exports = {"candidates.json": [
            {**candidate.model_dump(mode="json"), "cache_dir": str(stages[index].cache_dir.resolve()),
             "reused": stages[index].reused}
            for index, candidate in enumerate(ordered)],
            "synthesis.json": agent.analysis.model_dump(mode="json")}
        self.report_appendix = synthesis_report(agent.analysis)
        return synthesis.draft
