import json
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from run_scientist import ROOT, main
from scientist.agents.llm_hypothesis import LLMHypothesisAgent, hypothesis_context
from scientist.agents.problem_clarification import ProblemClarificationAgent
from scientist.backends.common import BackendError
from scientist.backends.opencode import OpenCodeBackend
from scientist.hypothesis_generation import generate_hypotheses
from scientist.problem_clarification import clarify_problem, confirm_problem, human_review_report, read_concept, state_from_confirmed_problem
from scientist.problem_schemas import ConceptInput, ConfirmedProblem, HumanFeedback, ProblemDraft, ProblemRevision, ProblemSource
from scientist.schemas import ResearchBudget, ScientificQuestion
from scientist.state import ScientificState
from scientist.tools.experiment_tracker import ResearchStore
from scientist.tools.problem_store import ProblemStore
from test_hypothesis_generation import ScriptedBackend, batch, events


def statement(text, origin="model_inference", source_ids=()):
    return {"text": text, "origin": origin, "source_ids": list(source_ids)}


def draft(*, blocking=True, constraint="No task ID"):
    return {"root_question": statement("What permits autonomous adaptation?", "user"),
            "goal": statement("Understand adaptation without prescribing a mechanism", "user"),
            "terms": [{"term": "adaptation", "interpretation": statement("Recovery of behavior"),
                       "alternatives": ["Change of internal state"]}],
            "scope": [statement("Changing environments")],
            "hard_constraints": [statement(constraint, "user")], "preferences": [], "assumptions": [],
            "perspectives": [{"discipline": "Control theory",
                              "contribution": statement("Consider stability during recovery"),
                              "definition_implications": ["Assess both final and transient behavior"],
                              "conflicts": ["State change is not necessarily learning"],
                              "limitations": ["No literature has been supplied"]}],
            "subquestions": [{"id": "SQ001", "text": "How can recovery be measured?",
                              "root_relationship": "Operationalizes adaptation",
                              "evaluation_criteria": ["Recovery after a controlled change"]}],
            "evaluation_criteria": [statement("Measure recovery time", "model_suggestion")],
            "candidate_mechanisms": [statement("Internal state adaptation", "model_suggestion")],
            "pending_issues": ([{"id": "I001", "question": "Are rewards allowed?",
                                 "reason": "This changes available information",
                                 "options": ["Allowed", "Excluded"], "blocking": True}] if blocking else []),
            "changes": [], "limitations": ["No scientific claim has been validated"]}


@pytest.fixture
def workspace(tmp_path):
    return ProblemStore(tmp_path / "research.sqlite3"), tmp_path / "artifacts"


def start(workspace, payload=None, concept=None):
    store, artifacts = workspace
    return clarify_problem(concept=concept or ConceptInput(text="Study autonomous adaptation", explicit_constraints=("No task ID",)),
                           agent=ProblemClarificationAgent(ScriptedBackend(json.dumps(payload or draft())), max_retries=0),
                           store=store, source_root=ROOT, artifact_root=artifacts, emit=lambda _: None)


def confirm(workspace, record):
    store, artifacts = workspace
    return confirm_problem(session_id=record.session_id, revision=record.revision, store=store,
                           artifact_root=artifacts, confirmed_by="test-researcher", emit=lambda _: None)


def test_full_lifecycle_preserves_history_and_passes_definition_to_hypotheses(workspace):
    store, artifacts = workspace
    first = start(workspace)
    assert store.load_revision(first.session_id, 1) == first
    with pytest.raises(ValueError, match="Blocking"):
        confirm(workspace, first)
    revised = draft(blocking=False)
    revised["changes"] = ["Human permits rewards; the no-task-ID constraint remains"]
    second = clarify_problem(concept=first.concept,
                             agent=ProblemClarificationAgent(ScriptedBackend(json.dumps(revised))),
                             store=store, source_root=ROOT, artifact_root=artifacts,
                             session_id=first.session_id, revision=1,
                             feedback=HumanFeedback(text="Rewards are allowed", answers={"I001": "Allowed"}),
                             emit=lambda _: None)
    assert store.load_revision(first.session_id, 1) == first
    assert second.revision == 2 and second.previous_draft_sha256
    approved = confirm(workspace, second)
    assert store.load_confirmation(first.session_id, 2) == approved
    state = state_from_confirmed_problem(approved, ResearchBudget())
    assert state.schema_version == 2 and state.current_round == 0
    context = hypothesis_context(state)
    assert context["problem_definition"]["draft"]["perspectives"][0]["discipline"] == "Control theory"
    assert context["problem_definition"]["draft"]["terms"][0]["term"] == "adaptation"
    backend = ScriptedBackend(json.dumps(batch()))
    research_store = ResearchStore(store.path)
    final = generate_hypotheses(state=state, agent=LLMHypothesisAgent(backend), store=research_store,
                                source_root=ROOT, artifact_root=artifacts, emit=lambda _: None)
    request = json.loads(backend.calls[0]["prompt"])
    assert request["scientific_context"]["problem_definition"]["revision"] == 2
    assert final.problem_definition == approved and len(final.hypotheses) == 3
    with research_store.connect() as db:
        run_id, directory = db.execute("SELECT id, artifact_dir FROM runs").fetchone()
    research_store.verify_artifacts(run_id)
    assert research_store.load(run_id) == final
    assert ConfirmedProblem.model_validate_json((Path(directory) / "problem_definition.json").read_bytes()) == approved


@pytest.mark.parametrize("invalid", ["constraint_removed", "constraint_origin", "unknown_source", "fake_confirmation", "duplicate_issue", "blocking_type", "blank"])
def test_invalid_model_drafts_never_commit(workspace, invalid):
    payload = draft()
    if invalid == "constraint_removed":
        payload["hard_constraints"] = []
    elif invalid == "constraint_origin":
        payload["hard_constraints"][0]["origin"] = "model_suggestion"
    elif invalid == "unknown_source":
        payload["perspectives"][0]["contribution"] = statement("A scientific claim", "source_supported", ["invented"])
    elif invalid == "fake_confirmation":
        payload["confirmed"] = True
    elif invalid == "duplicate_issue":
        payload["pending_issues"][0]["id"] = "SQ001"
    elif invalid == "blocking_type":
        payload["pending_issues"][0]["blocking"] = "false"
    else:
        payload["root_question"]["text"] = "   "
    with pytest.raises(ValueError, match="failed validation"):
        start(workspace, payload)
    with workspace[0].connect() as db:
        assert db.execute("SELECT COUNT(*) FROM problem_revisions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM problem_sessions").fetchone()[0] == 0
    assert not workspace[1].exists()


def test_source_support_is_traceable_but_not_assumed_truth(workspace):
    payload = draft(blocking=False)
    payload["perspectives"][0]["contribution"] = statement("A source describes stability", "source_supported", ["S1"])
    concept = ConceptInput(text="Study adaptation", explicit_constraints=("No task ID",),
                           sources=(ProblemSource(id="S1", title="User-supplied note", text="Consider stability"),))
    record = start(workspace, payload, concept)
    assert record.draft.perspectives[0].contribution.source_ids == ("S1",)


@pytest.mark.parametrize("change", ["remove", "downgrade", "rename"])
def test_model_cannot_erase_unanswered_blocking_issues(workspace, change):
    first = start(workspace)
    revised = draft()
    if change == "remove":
        revised["pending_issues"] = []
    elif change == "downgrade":
        revised["pending_issues"][0]["blocking"] = False
    else:
        revised["pending_issues"][0]["id"] = "I002"
    with pytest.raises(ValueError, match="failed validation"):
        clarify_problem(concept=first.concept,
                        agent=ProblemClarificationAgent(ScriptedBackend(json.dumps(revised)), max_retries=0),
                        store=workspace[0], source_root=ROOT, artifact_root=workspace[1],
                        session_id=first.session_id, revision=1, feedback=HumanFeedback(text="Please improve clarity"),
                        emit=lambda _: None)
    assert workspace[0].latest_revision(first.session_id) == 1


def test_constraints_can_change_only_via_explicit_feedback(workspace):
    first = start(workspace, draft(blocking=False))
    second = clarify_problem(concept=first.concept,
                             agent=ProblemClarificationAgent(ScriptedBackend(json.dumps(draft(blocking=False, constraint="No labels")))),
                             store=workspace[0], source_root=ROOT, artifact_root=workspace[1],
                             session_id=first.session_id, revision=1,
                             feedback=HumanFeedback(text="Replace the constraint", revised_constraints=("No labels",)),
                             emit=lambda _: None)
    approved = confirm(workspace, second)
    assert approved.original_concept.explicit_constraints == ("No task ID",)
    assert approved.question.constraints == ("No labels",)
    with pytest.raises(ValueError, match="Stale"):
        clarify_problem(concept=first.concept, agent=ProblemClarificationAgent(ScriptedBackend()),
                        store=workspace[0], source_root=ROOT, artifact_root=workspace[1],
                        session_id=first.session_id, revision=1, feedback=HumanFeedback(text="Late feedback"))
    with pytest.raises(ValueError, match="latest"):
        confirm(workspace, first)


def test_unknown_answer_rejected_before_a_model_call(workspace):
    first = start(workspace)
    backend = ScriptedBackend()
    with pytest.raises(ValueError, match="unknown issue"):
        clarify_problem(concept=first.concept, agent=ProblemClarificationAgent(backend),
                        store=workspace[0], source_root=ROOT, artifact_root=workspace[1],
                        session_id=first.session_id, revision=1,
                        feedback=HumanFeedback(text="Answer", answers={"invented": "Allowed"}))
    assert backend.calls == []


def test_open_research_unknowns_do_not_block_confirmation(workspace):
    payload = draft()
    payload["pending_issues"][0]["blocking"] = False
    approved = confirm(workspace, start(workspace, payload))
    assert not approved.draft.pending_issues[0].blocking
    with pytest.raises(ValueError, match="already been confirmed"):
        confirm_problem(session_id=approved.session_id, revision=1, store=workspace[0],
                        artifact_root=workspace[1], confirmed_by="test")


def test_tampered_artifacts_and_changed_confirmed_questions_are_rejected(workspace):
    first = start(workspace, draft(blocking=False))
    approved = confirm(workspace, first)
    changed = approved.model_dump()
    changed["question"]["text"] = "Another research objective"
    with pytest.raises(ValueError, match="differs"):
        ConfirmedProblem.model_validate(changed)
    path = workspace[1] / first.session_id / "revision_0001" / "draft.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        workspace[0].load_confirmation(first.session_id, 1)


def test_legacy_states_load_and_new_states_bind_to_confirmation(workspace):
    old = ScientificState.model_validate({"schema_version": 1,
                                         "question": {"id": "Q001", "text": "Legacy question", "constraints": []}})
    assert old.problem_definition is None and old.schema_version == 1
    assert "problem_definition" not in old.model_dump()
    with pytest.raises(ValueError, match="matching confirmed"):
        ScientificState(schema_version=2, question=old.question)
    approved = confirm(workspace, start(workspace, draft(blocking=False)))
    with pytest.raises(ValueError, match="schema version 2"):
        ScientificState(question=approved.question, problem_definition=approved)
    state = state_from_confirmed_problem(approved, ResearchBudget())
    with pytest.raises(ValueError, match="matching confirmed"):
        state.revised(question=ScientificQuestion(id="Q001", text="Different question", constraints=()))


def test_retry_bound_and_transport_errors(workspace):
    concept = ConceptInput(text="Study adaptation", explicit_constraints=("No task ID",))
    backend = ScriptedBackend("not json", json.dumps(draft()))
    agent = ProblemClarificationAgent(backend, max_retries=1)
    agent.analyze(concept)
    assert len(backend.calls) == 2 and [a["accepted"] for a in agent.attempts] == [False, True]
    failing = ScriptedBackend(BackendError("provider unavailable"))
    with pytest.raises(BackendError):
        ProblemClarificationAgent(failing, max_retries=2).analyze(concept)
    assert len(failing.calls) == 1


@pytest.mark.parametrize("field,value", [
    ("question", "问" * 41),
    ("reason", "因" * 51),
    ("options", ["选" * 25, "其他"]),
    ("options", ["仅一个"]),
    ("options", ["一", "二", "三", "四", "五"]),
    ("options", ["Allowed", "allowed"]),
    ("question", "第一行\n第二行"),
])
def test_unreadable_human_review_never_commits(workspace, field, value):
    payload = draft()
    payload["pending_issues"][0][field] = value
    with pytest.raises(ValueError, match="failed validation"):
        start(workspace, payload)
    with workspace[0].connect() as db:
        assert db.execute("SELECT COUNT(*) FROM problem_revisions").fetchone()[0] == 0
    assert not workspace[1].exists()


def test_long_review_is_regenerated_without_shortening_agent_analysis(workspace):
    long = draft()
    long["pending_issues"][0]["question"] = "问" * 41
    corrected = draft()
    analysis = "Detailed conditional reasoning for later agents. " * 20
    corrected["scope"] = [statement(analysis)]
    corrected["root_question"] = statement(analysis, "model_inference")
    backend = ScriptedBackend(json.dumps(long), json.dumps(corrected))
    record = clarify_problem(concept=ConceptInput(text="Study adaptation", explicit_constraints=("No task ID",)),
                             agent=ProblemClarificationAgent(backend, max_retries=1),
                             store=workspace[0], source_root=ROOT, artifact_root=workspace[1], emit=lambda _: None)
    assert len(backend.calls) == 2
    assert record.draft.scope[0].text == analysis.strip()
    directory = workspace[1] / record.session_id / "revision_0001"
    assert analysis.strip() in (directory / "report.md").read_text()
    assert analysis.strip() not in (directory / "review.md").read_text()
    assert record.draft.pending_issues[0].question == corrected["pending_issues"][0]["question"]
    workspace[0].load_revision(record.session_id, 1)


def test_human_review_prioritizes_required_choices_and_keeps_all_questions(workspace):
    payload = draft()
    payload["pending_issues"] = [
        {"id": f"I{number:03d}", "question": f"确认条件{number}？", "reason": "决定研究范围。",
         "options": ["允许", "不允许"] if number != 4 else [], "blocking": number != 1}
        for number in range(1, 6)
    ]
    record = start(workspace, payload)
    review = human_review_report(record)
    assert review.index("### 1. 确认条件2？") < review.index("### 2. 确认条件3？")
    assert review.index("### 3. 确认条件4？") < review.index("<details>")
    assert "### 4. 确认条件5？" in review and "### 5. 确认条件1？" in review
    assert "- A. 允许" in review and "请直接说明你的选择。" in review
    assert "I002" not in review
    assert len(record.draft.pending_issues) == 5
    with pytest.raises(ValueError, match="Blocking"):
        confirm(workspace, record)


def test_clear_review_without_pending_questions_does_not_confirm_automatically(workspace):
    record = start(workspace, draft(blocking=False))
    assert "再决定是否确认" in human_review_report(record)
    with workspace[0].connect() as db:
        assert db.execute("SELECT COUNT(*) FROM problem_confirmations").fetchone()[0] == 0


def test_historical_long_review_remains_readable_and_can_be_rephrased(workspace):
    first = start(workspace)
    payload = first.model_dump(mode="json")
    payload["session_id"] = "problem_" + uuid4().hex
    payload["draft"]["pending_issues"][0]["question"] = "Long historic question " * 10
    historical = ProblemRevision.model_validate(payload)
    directory = workspace[1] / historical.session_id / "revision_0001"
    workspace[0].save_revision(historical, directory, {})
    assert workspace[0].load_revision(historical.session_id, 1) == historical
    revised = clarify_problem(concept=historical.concept,
                              agent=ProblemClarificationAgent(ScriptedBackend(json.dumps(draft()))),
                              store=workspace[0], source_root=ROOT, artifact_root=workspace[1],
                              session_id=historical.session_id, revision=1,
                              feedback=HumanFeedback(text="Shorten the wording"), emit=lambda _: None)
    assert revised.draft.pending_issues[0].id == historical.draft.pending_issues[0].id
    assert revised.draft.pending_issues[0].blocking
    assert workspace[0].load_revision(historical.session_id, 1) == historical


def test_injected_agent_cannot_bypass_concise_review_validation(workspace):
    payload = draft()
    payload["pending_issues"][0]["question"] = "问" * 41
    class VerboseAgent:
        attempts = []
        def analyze(self, concept, previous, feedback):
            return ProblemDraft.model_validate(payload)
    with pytest.raises(ValueError, match="short and single-line"):
        clarify_problem(concept=ConceptInput(text="Study adaptation", explicit_constraints=("No task ID",)),
                        agent=VerboseAgent(), store=workspace[0], source_root=ROOT,
                        artifact_root=workspace[1], emit=lambda _: None)
    assert not workspace[1].exists()


def test_injected_agent_cannot_mutate_feedback_to_bypass_confirmation(workspace):
    first = start(workspace)
    feedback = HumanFeedback(text="Improve the wording")
    class MutatingAgent:
        attempts = []
        def analyze(self, concept, previous, supplied_feedback):
            supplied_feedback.answers["I001"] = "Invented approval"
            return ProblemDraft.model_validate(draft(blocking=False))
    with pytest.raises(ValueError, match="Unanswered blocking"):
        clarify_problem(concept=first.concept, agent=MutatingAgent(), store=workspace[0],
                        source_root=ROOT, artifact_root=workspace[1], session_id=first.session_id,
                        revision=1, feedback=feedback, emit=lambda _: None)
    assert feedback.answers == {} and workspace[0].latest_revision(first.session_id) == 1


def test_failed_artifact_write_does_not_commit_a_session(workspace, monkeypatch):
    original = Path.write_bytes
    def fail_report(path, content):
        if path.name == "report.md":
            raise OSError("Simulated disk failure")
        return original(path, content)
    monkeypatch.setattr(Path, "write_bytes", fail_report)
    with pytest.raises(OSError, match="disk failure"):
        start(workspace)
    with workspace[0].connect() as db:
        assert db.execute("SELECT COUNT(*) FROM problem_sessions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM problem_revisions").fetchone()[0] == 0
    assert list(workspace[1].glob("*/revision_0001/pending.json"))


def test_cli_clarify_review_confirm_generate_without_real_provider(tmp_path, monkeypatch):
    import scientist.hypothesis_generation as generation
    backend = ScriptedBackend(json.dumps(draft()), json.dumps(draft(blocking=False)), json.dumps(batch()))
    roles = []
    def make_backend(config, *, role="hypothesis"):
        roles.append(role)
        return backend
    monkeypatch.setattr(generation, "make_reasoning_backend", make_backend)
    concept_path = tmp_path / "concept.json"
    concept_path.write_text(json.dumps({"text": "Study adaptation", "explicit_constraints": ["No task ID"]}))
    database, artifacts = tmp_path / "research.sqlite3", tmp_path / "artifacts"
    common = ["--database", str(database), "--output-dir", str(artifacts)]
    assert main(["--clarify-question", str(concept_path), "--clarification-mode", "single", *common]) == 0
    store = ProblemStore(database)
    with store.connect() as db:
        session = db.execute("SELECT id FROM problem_sessions").fetchone()[0]
    feedback = tmp_path / "feedback.json"
    feedback.write_text(json.dumps({"text": "Rewards are allowed", "answers": {"I001": "Allowed"}}))
    assert main(["--review-question", session, "--revision", "1", "--feedback-file", str(feedback),
                 "--clarification-mode", "single", *common]) == 0
    assert main(["--confirm-question", session, "--revision", "2", "--confirmed-by", "Researcher", *common]) == 0
    assert roles == ["problem_clarification", "problem_clarification"]
    approved_path = artifacts / session / "confirmation_0002" / "confirmed_problem.json"
    assert main(["--generate-hypotheses", "--confirmed-question", str(approved_path), *common]) == 0
    assert roles[-1] == "hypothesis"
    # Passing a draft as a confirmation cannot reach a model.
    with pytest.raises(SystemExit):
        main(["--generate-hypotheses", "--confirmed-question",
              str(artifacts / session / "revision_0002" / "draft.json"), *common])
    assert len(roles) == 3


@pytest.mark.parametrize("args", [
    ["--demo", "--clarification-mode", "ensemble"],
    ["--review-question", "session"], ["--confirm-question", "session"],
    ["--review-question", "session", "--revision", "1"],
    ["--demo", "--feedback-file", "feedback.json"], ["--demo", "--revision", "1"],
    ["--demo", "--confirmed-question", "approved.json"],
    ["--generate-hypotheses", "--confirmed-question", "approved.json", "--state-file", "state.json"],
    ["--confirm-question", "session", "--revision", "1", "--model", "test/model"],
])
def test_invalid_cli_combinations_fail(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("role", ["problem_clarification", "problem_synthesis", "literature_search", "literature_screening",
                                  "literature_reading", "evidence_verification", "literature_synthesis",
                                  "hypothesis_review", "hypothesis_revision"])
def test_plain_concept_input_and_opencode_role(tmp_path, monkeypatch, role):
    path = tmp_path / "concept.md"
    path.write_text("研究自主适应", encoding="utf-8")
    assert read_concept(path).text == "研究自主适应"
    executable = tmp_path / "opencode"
    executable.write_text("fixture")
    executable.chmod(0o755)
    observed = {}
    def run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, events(json.dumps(draft())), "")
    monkeypatch.setattr(subprocess, "run", run)
    backend = OpenCodeBackend(model="test/model", executable=str(executable), role=role)
    response = backend.complete(instructions="Clarify a concept", prompt="concept")
    command = observed["command"]
    assert command[command.index("--agent") + 1] == role
    runtime = json.loads(observed["env"]["OPENCODE_CONFIG_CONTENT"])
    assert runtime["agent"][role]["permission"] == {"*": "deny"}
    assert response.metadata["role"] == role


def test_codex_role_uses_draft_schema_and_disables_tools(tmp_path, monkeypatch):
    from scientist.backends.codex import CodexBackend
    from test_codex_backend import fake_popen
    executable = tmp_path / "codex"
    executable.write_text("fixture")
    executable.chmod(0o755)
    config = tmp_path / "provider.json"
    config.write_text(json.dumps({"provider": {"test": {"models": {"model": {}}, "options": {
        "baseURL": "https://provider.example.test/v1", "apiKey": "dummy-secret-key"}}}}))
    observed = fake_popen(monkeypatch, json.dumps(draft()))
    backend = CodexBackend(model="test/model", executable=str(executable), config=config, role="problem_clarification")
    result = ProblemClarificationAgent(backend).analyze(ConceptInput(text="Study adaptation", explicit_constraints=("No task ID",)))
    assert isinstance(result, ProblemDraft)
    assert "root_question" in observed["schema"]["properties"]
    assert "features.shell_tool=false" in observed["command"]
