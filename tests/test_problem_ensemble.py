import hashlib
import json
from pathlib import Path

import pytest
import yaml

from run_scientist import ROOT, main
from scientist.agents.problem_clarification import ProblemClarificationAgent
from scientist.agents.problem_synthesis import CandidateDraft, ProblemSynthesisAgent
from scientist.backends.common import BackendError, ReasoningResponse
from scientist.hypothesis_generation import ReasoningConfig
from scientist.problem_clarification import clarify_problem, confirm_problem
from scientist.problem_ensemble import ClarificationConfig, MultiModelProblemClarificationAgent
from scientist.problem_schemas import ConceptInput, HumanFeedback, ProblemDraft
from scientist.tools.problem_store import ProblemStore
from test_hypothesis_generation import ScriptedBackend
from test_problem_clarification import draft


def synthesis_payload(request, integrated=None):
    candidates = request["candidates"]
    integrated = integrated or candidates[0]["draft"]
    final = [issue["id"] for issue in integrated["pending_issues"]]
    return {"draft": integrated, "analysis": {
        "contributions": [{"candidate_id": c["candidate_id"], "retained": ["A relevant distinction"],
                           "corrected_or_omitted": []} for c in candidates],
        "disagreements": [],
        "issue_mapping": [{"candidate_id": c["candidate_id"], "issue_id": issue["id"],
                           "integrated_issue_ids": final, "status": "merged" if final else "kept_open",
                           "reason": "Equivalent decision retained, or scope explicitly left open"}
                          for c in candidates for issue in c["draft"]["pending_issues"]],
        "limitations": ["Synthesis does not establish scientific truth"]}}


class ModelBackend:
    def __init__(self, config, role, response):
        self.config, self.role, self.response = config, role, response
        self.calls = []

    def complete(self, *, instructions, prompt):
        self.calls.append({"instructions": instructions, "prompt": prompt})
        response = self.response
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(json.loads(prompt))
        return ReasoningResponse(json.dumps(response), {"backend": "test", "model": self.config.model,
                                 "role": self.role, "auth_mode": self.config.codex_auth_mode,
                                 "reasoning_effort_requested": self.config.codex_reasoning_effort})


@pytest.fixture
def setup(tmp_path):
    common = ReasoningConfig(model="glm-5.3", backend="codex", validation_retries=0)
    config = ClarificationConfig()
    concept = ConceptInput(text="Study autonomous adaptation", explicit_constraints=("No task ID",))
    backends = {}

    def candidate_response(request):
        answered = request.get("feedback") and "I001" in request["feedback"]["answers"]
        payload = draft(blocking=not answered)
        if answered:
            payload["changes"] = ["Human answered I001"]
        return payload

    def factory(runtime, *, role):
        key = (role, runtime.model)
        if key not in backends:
            backends[key] = ModelBackend(runtime, role, synthesis_payload if role == "problem_synthesis" else candidate_response)
        backends[key].config = runtime
        return backends[key]

    artifacts = tmp_path / "artifacts"
    store = ProblemStore(tmp_path / "research.sqlite3")

    def agent():
        return MultiModelProblemClarificationAgent(common, config, artifacts, backend_factory=factory, emit=lambda _: None)

    def run(**kwargs):
        return clarify_problem(concept=concept, agent=agent(), store=store, source_root=ROOT,
                               artifact_root=artifacts, emit=lambda _: None, **kwargs)

    return common, config, concept, backends, factory, artifacts, store, agent, run


def call_count(backends):
    return sum(len(backend.calls) for backend in backends.values())


def test_pipeline_persists_three_drafts_and_audited_synthesis_without_confirming(setup):
    _, _, _, backends, _, artifacts, store, _, run = setup
    record = run()
    assert call_count(backends) == 4
    assert store.load_revision(record.session_id, 1) == record
    folder = artifacts / record.session_id / "revision_0001"
    candidates = json.loads((folder / "candidates.json").read_bytes())
    assert [c["model"] for c in candidates] == ["glm-5.3", "deepseek-v4.1-flash", "gpt-6.1-sol"]
    analysis = json.loads((folder / "synthesis.json").read_bytes())
    assert len(analysis["contributions"]) == 3 and len(analysis["issue_mapping"]) == 3
    assert "整合依据" in (folder / "report.md").read_text()
    assert "整合依据" not in (folder / "review.md").read_text()
    synth = backends[("problem_synthesis", "gpt-6.1-sol")]
    request = json.loads(synth.calls[0]["prompt"])
    assert request["concept"] == record.concept.model_dump(mode="json")
    assert [c["candidate_id"] for c in request["candidates"]] == ["C001", "C002", "C003"]
    assert synth.config.codex_auth_mode == "chatgpt" and synth.config.codex_reasoning_effort == "high"
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM problem_confirmations").fetchone()[0] == 0
    with pytest.raises(ValueError, match="Blocking"):
        confirm_problem(session_id=record.session_id, revision=1, store=store, artifact_root=artifacts,
                        confirmed_by="test", emit=lambda _: None)


def test_same_input_reuses_all_four_results(setup):
    _, _, concept, backends, _, _, _, agent, _ = setup
    first = agent().analyze(concept)
    repeated = agent()
    def forbidden_runtime(*args, **kwargs):
        raise AssertionError("Cache reuse must not initialize credentialed runtimes")
    repeated.backend_factory = forbidden_runtime
    assert repeated.analyze(concept) == first
    assert call_count(backends) == 4
    assert all(entry["reused"] for entry in repeated.attempts)
    assert all(c["reused"] for c in repeated.exports["candidates.json"])


def test_feedback_reaches_every_model_and_keeps_history(setup):
    _, _, _, backends, _, _, store, _, run = setup
    first = run()
    feedback = HumanFeedback(text="Allow rewards", answers={"I001": "Allowed"})
    second = run(session_id=first.session_id, revision=1, feedback=feedback)
    assert second.revision == 2 and not second.draft.pending_issues
    assert call_count(backends) == 8
    for backend in backends.values():
        request = json.loads(backend.calls[-1]["prompt"])
        assert request["feedback"] == feedback.model_dump(mode="json")
        assert request["previous_draft"] == first.draft.model_dump(mode="json")
    assert store.load_revision(first.session_id, 1) == first


def test_changed_prompt_invalidates_candidate_and_synthesis_caches(setup, tmp_path, monkeypatch):
    import scientist.agents.problem_clarification as clarification
    _, _, concept, backends, _, _, _, agent, _ = setup
    agent().analyze(concept)
    changed = tmp_path / "prompt.txt"
    changed.write_text(clarification.PROMPT_FILE.read_text() + "\n补充通用说明。\n")
    monkeypatch.setattr(clarification, "PROMPT_FILE", changed)
    agent().analyze(concept)
    assert call_count(backends) == 8


def test_candidate_failure_preserves_successes_and_prevents_partial_synthesis(setup):
    _, _, _, backends, factory, _, store, _, run = setup
    runtime = setup[1].candidates[1].runtime_config(setup[0])
    failed = factory(runtime, role="problem_clarification")
    normal = failed.response
    failed.response = BackendError("provider unavailable")
    with pytest.raises(RuntimeError, match="candidate.*failed"):
        run()
    assert ("problem_synthesis", "gpt-6.1-sol") not in backends
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM problem_revisions").fetchone()[0] == 0
    failed.response = normal
    record = run()
    assert record.revision == 1 and call_count(backends) == 5


def test_corrupt_cached_report_is_rejected_without_new_model_calls(setup):
    _, _, concept, backends, _, _, _, agent, _ = setup
    first = agent()
    first.analyze(concept)
    folder = Path(first.exports["candidates.json"][0]["cache_dir"])
    (folder / "draft.json").write_text("{}")
    with pytest.raises(RuntimeError, match="candidate.*failed"):
        agent().analyze(concept)
    assert call_count(backends) == 4


@pytest.mark.parametrize("invalid", ["missing_mapping", "unknown_issue", "missing_contribution", "human_choice_without_issue", "constraint_removed"])
def test_synthesis_rejects_incomplete_audit_or_invalid_final_definition(invalid):
    candidates = tuple(CandidateDraft(candidate_id=f"C{i:03d}", model=f"model-{i}",
                                      draft=ProblemDraft.model_validate(draft())) for i in range(1, 4))
    payload = synthesis_payload({"candidates": [c.model_dump(mode="json") for c in candidates]})
    if invalid == "missing_mapping":
        payload["analysis"]["issue_mapping"].pop()
    elif invalid == "unknown_issue":
        payload["analysis"]["issue_mapping"][0]["integrated_issue_ids"] = ["unknown"]
    elif invalid == "missing_contribution":
        payload["analysis"]["contributions"].pop()
    elif invalid == "human_choice_without_issue":
        payload["analysis"]["disagreements"] = [{"topic": "scope", "candidate_ids": ["C001", "C002"],
            "interpretations": ["one", "two"], "handling": "Ask human", "status": "needs_human_choice", "final_issue_ids": []}]
    else:
        payload["draft"]["hard_constraints"] = []
    agent = ProblemSynthesisAgent(ScriptedBackend(json.dumps(payload)), candidates, max_retries=0)
    with pytest.raises(ValueError, match="failed validation"):
        agent.analyze(ConceptInput(text="Study adaptation", explicit_constraints=("No task ID",)))


def test_synthesis_cannot_remove_prior_unanswered_blocking_issue(setup):
    _, config, _, backends, factory, _, store, _, run = setup
    first = run()
    synth = factory(config.synthesizer.runtime_config(setup[0]), role="problem_synthesis")
    synth.response = lambda request: synthesis_payload(request, draft(blocking=False))
    with pytest.raises(ValueError, match="failed validation"):
        run(session_id=first.session_id, revision=1, feedback=HumanFeedback(text="Improve wording"))
    assert store.latest_revision(first.session_id) == 1
    assert first.draft.pending_issues[0].blocking


def test_existing_comparison_reports_are_imported_without_repeating_candidates(setup):
    common, config, concept, backends, factory, artifacts, _, agent, _ = setup
    for member in config.candidates:
        runtime = member.runtime_config(common)
        original = ProblemClarificationAgent(factory(runtime, role="problem_clarification"), max_retries=0)
        original.analyze(concept)
        folder = artifacts / "existing_comparison" / member.model
        folder.mkdir(parents=True)
        exports = {"settings.json": runtime.model_dump(mode="json"), "reasoning.json": {"attempts": original.attempts}}
        hashes = {}
        for name, value in exports.items():
            (folder / name).write_text(json.dumps(value))
            hashes[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        (folder / "manifest.json").write_text(json.dumps({"sha256": hashes}))
    pipeline = agent()
    pipeline.analyze(concept)
    assert call_count(backends) == 4
    assert all(c["reused"] for c in pipeline.exports["candidates.json"])


def test_cli_uses_ensemble_configuration_by_default(setup, tmp_path, monkeypatch):
    import scientist.hypothesis_generation as generation
    common, _, concept, backends, factory, _, _, _, _ = setup
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text(yaml.safe_dump(common.model_dump(mode="json")))
    (config_dir / "clarification.yaml").write_text("mode: ensemble\n")
    concept_file = tmp_path / "concept.json"
    concept_file.write_text(concept.model_dump_json())
    monkeypatch.setattr(generation, "make_reasoning_backend", factory)
    args = ["--clarify-question", str(concept_file), "--config-dir", str(config_dir),
            "--database", str(tmp_path / "cli.sqlite3"), "--output-dir", str(tmp_path / "cli_artifacts")]
    assert main(args) == 0 and call_count(backends) == 4
    with pytest.raises(SystemExit) as error:
        main([*args, "--model", "gpt-6.1-sol"])
    assert error.value.code == 2 and call_count(backends) == 4
