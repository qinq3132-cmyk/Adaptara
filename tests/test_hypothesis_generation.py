import json
import subprocess
from pathlib import Path

import pytest

from run_scientist import build_demo, main
from scientist.agents.llm_hypothesis import LLMHypothesisAgent, allocate_ids
from scientist.backends.opencode import BackendError, OpenCodeBackend, ReasoningResponse, parse_events
from scientist.hypothesis_generation import generate_hypotheses
from scientist.schemas import Claim, Paper, Stage


def batch():
    return {"hypotheses": [{"claim": f"Distinct claim {i}", "mechanism": f"Mechanism {i}",
                            "assumptions": ["A declared condition"], "predictions": [f"Intervention {i} changes a measured outcome"],
                            "alternative_explanations": ["A confounding variable could explain the difference"],
                            "falsification_conditions": [f"Outcome {i} stays unchanged under the intervention"],
                            "parent_hypotheses": [], "source_claims": []} for i in range(3)]}


class ScriptedBackend:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, *, instructions, prompt):
        self.calls.append({"instructions": instructions, "prompt": prompt})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ReasoningResponse(response, {"backend": "test", "model": "fixture/test-model"})


@pytest.fixture
def setup(tmp_path):
    return build_demo(output_dir=tmp_path / "artifacts", database=tmp_path / "research.sqlite3", emit=lambda _: None)


def test_model_contents_are_validated_and_persisted_without_experiments(setup):
    orchestrator, state = setup
    backend = ScriptedBackend(json.dumps(batch()))
    agent = LLMHypothesisAgent(backend)
    final = generate_hypotheses(state=state, agent=agent, store=orchestrator.store,
                                source_root=orchestrator.source_root, artifact_root=orchestrator.artifact_root, emit=lambda _: None)
    assert final.stage == Stage.HYPOTHESIS and final.current_round == 0
    assert final.experiments == final.results == final.evidence == ()
    assert state.hypotheses == ()
    request = json.loads(backend.calls[0]["prompt"])
    assert request["output_schema"]["properties"]["hypotheses"]["minItems"] == 3
    assert request["output_schema"]["properties"]["hypotheses"]["maxItems"] == 3
    assert [h.claim for h in final.hypotheses] == [h["claim"] for h in batch()["hypotheses"]]
    assert [h.id for h in final.hypotheses] == ["H001", "H002", "H003"]
    assert all(h.status == "proposed" and h.confidence == 0.5 for h in final.hypotheses)
    with orchestrator.store.connect() as db:
        run_id = db.execute("SELECT id FROM runs").fetchone()[0]
        assert db.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0
    orchestrator.store.verify_artifacts(run_id)
    assert orchestrator.store.load(run_id) == final
    path = orchestrator.artifact_root / run_id
    trace = json.loads((path / "reasoning.json").read_text())
    assert trace["attempts"][0]["response_text"] == json.dumps(batch())
    assert trace["attempts"][0]["accepted"] is True
    snapshot = json.loads((path / "source_snapshot.json").read_text())
    assert "scientist/prompts/hypothesis.txt" in snapshot


def test_previous_results_inform_new_ids_without_replacing_history(setup):
    orchestrator, state = setup
    previous = orchestrator.run_cycle(state)
    drafts = batch()
    drafts["hypotheses"][0]["parent_hypotheses"] = ["H001"]
    backend = ScriptedBackend(json.dumps(drafts))
    proposals = LLMHypothesisAgent(backend).propose(previous)
    assert [h.id for h in proposals.hypotheses] == ["H004", "H005", "H006"]
    assert proposals.hypotheses[0].parent_hypotheses == ("H001",)
    context = json.loads(backend.calls[0]["prompt"])["scientific_context"]
    assert context["results"][0]["metrics"]["excess_loss"] > 0
    assert "trajectory" not in context["results"][0]
    revised = previous.revised(hypotheses=previous.hypotheses + proposals.hypotheses)
    assert revised.results == previous.results
    assert previous.hypotheses[0].status == "weakened"


@pytest.mark.parametrize("invalid", ["extra_status", "unknown_source", "duplicate_claim", "duplicate_mechanism", "wrong_count", "blank", "empty_prediction", "unknown_parent"])
def test_invalid_proposals_never_commit(setup, invalid):
    orchestrator, state = setup
    drafts = batch()
    first = drafts["hypotheses"][0]
    if invalid == "extra_status":
        first["status"] = "supported"
    elif invalid == "unknown_source":
        first["source_claims"] = ["invented"]
    elif invalid == "duplicate_claim":
        drafts["hypotheses"][1]["claim"] = first["claim"]
    elif invalid == "duplicate_mechanism":
        drafts["hypotheses"][1]["mechanism"] = first["mechanism"]
    elif invalid == "wrong_count":
        drafts["hypotheses"].pop()
    elif invalid == "blank":
        first["claim"] = "  "
    elif invalid == "empty_prediction":
        first["predictions"] = []
    elif invalid == "unknown_parent":
        first["parent_hypotheses"] = ["H999"]
    agent = LLMHypothesisAgent(ScriptedBackend(json.dumps(drafts)), max_retries=0)
    with pytest.raises(ValueError, match="failed validation"):
        generate_hypotheses(state=state, agent=agent, store=orchestrator.store,
                            source_root=orchestrator.source_root, artifact_root=orchestrator.artifact_root, emit=lambda _: None)
    with orchestrator.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert state.hypotheses == ()


def test_retry_is_bounded_and_transport_errors_never_fall_back(setup):
    _, state = setup
    backend = ScriptedBackend("not JSON", json.dumps(batch()))
    agent = LLMHypothesisAgent(backend, max_retries=1)
    assert len(agent.propose(state).hypotheses) == 3
    assert len(backend.calls) == 2
    assert [a["accepted"] for a in agent.attempts] == [False, True]
    failing = ScriptedBackend(BackendError("provider unavailable"))
    with pytest.raises(BackendError):
        LLMHypothesisAgent(failing, max_retries=2).propose(state)
    assert len(failing.calls) == 1


def test_references_and_global_id_collisions(setup):
    _, state = setup
    state = state.revised(papers=(Paper(id="H001", title="A source", source_uri="https://example.test/paper"),),
                          knowledge=(Claim(id="CL001", statement="A sourced claim", source_ids=("H001",), mechanism="A prior mechanism"),))
    assert allocate_ids(state, 3) == ("H002", "H003", "H004")
    drafts = batch()
    drafts["hypotheses"][0]["source_claims"] = ["CL001"]
    proposals = LLMHypothesisAgent(ScriptedBackend(json.dumps(drafts))).propose(state)
    assert proposals.hypotheses[0].source_claims == ("CL001",)


def events(text="{}", finish="stop"):
    return "\n".join(json.dumps(e) for e in [
        {"type": "step_start", "sessionID": "ses_test", "part": {}},
        {"type": "text", "sessionID": "ses_test", "part": {"text": text}},
        {"type": "step_finish", "sessionID": "ses_test", "part": {"reason": finish, "tokens": {"input": 12, "output": 8}, "cost": 0}}])


def test_opencode_event_handling():
    response = parse_events(events())
    assert response.text == "{}" and response.metadata["session_id"] == "ses_test"
    for output in (events(finish="length"), 'not json', json.dumps({"type": "tool_use"}),
                   json.dumps({"type": "error", "error": {"message": "fake-secret-key"}})):
        with pytest.raises(BackendError) as error:
            parse_events(output)
        assert "fake-secret-key" not in str(error.value)


def test_opencode_command_uses_existing_config_and_role_permissions(tmp_path, monkeypatch):
    executable = tmp_path / "opencode"
    executable.write_text("fixture")
    executable.chmod(0o755)
    config = tmp_path / "opencode.json"
    config.write_text(json.dumps({"provider": {"test": {"options": {"apiKey": "fake-secret-key"}}}}))
    observed = {}

    def run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, events(json.dumps(batch())), "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("HTTP_PROXY", "http://working.proxy.test:18080")
    monkeypatch.setenv("http_proxy", "http://stale.proxy.test:7899")
    backend = OpenCodeBackend(model="test/model", executable=str(executable), config=config,
                              proxy_env_precedence="uppercase")
    response = backend.complete(instructions="role instructions", prompt="scientific input")
    assert response.metadata["model"] == "test/model"
    assert observed["command"][1:6] == ["run", "--pure", "--format", "json", "--agent"]
    assert observed["env"]["OPENCODE_CONFIG"] == str(config)
    assert observed["env"]["http_proxy"] == "http://working.proxy.test:18080"
    assert "--title" in observed["command"]
    inline = observed["env"]["OPENCODE_CONFIG_CONTENT"]
    assert "fake-secret-key" not in inline
    assert json.loads(inline)["agent"]["hypothesis"]["permission"] == {"*": "deny"}
    assert "--auto" not in observed["command"] and "--continue" not in observed["command"]
    assert not Path(observed["cwd"]).exists()


def test_timeout_and_cli_option_validation(tmp_path, monkeypatch):
    executable = tmp_path / "opencode"
    executable.write_text("fixture")
    executable.chmod(0o755)
    backend = OpenCodeBackend(model="test/model", executable=str(executable), timeout_seconds=1)

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 1, stderr="fake-secret-key")

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(BackendError, match="timed out") as error:
        backend.complete(instructions="role", prompt="question")
    assert "fake-secret-key" not in str(error.value)
    with pytest.raises(SystemExit):
        main(["--demo", "--model", "test/model"])
