import json
import subprocess
from pathlib import Path

import pytest
import scientist.backends.codex as codex_module

from run_scientist import main
from scientist.backends.codex import CodexBackend, CodexError, KEY_ENV, failure_code, load_provider, native_provider, parse_events
from scientist.hypothesis_generation import ReasoningConfig, generate_hypotheses, make_hypothesis_agent
from test_hypothesis_generation import batch


def stream(text, *, tool=False, failed=False, complete=True):
    events = [{"type": "thread.started", "thread_id": "test-thread"}, {"type": "turn.started"}]
    if tool:
        events.append({"type": "item.completed", "item": {"type": "command_execution", "status": "failed" if failed else "completed",
                       "exit_code": 1 if failed else 0, "aggregated_output": "bwrap: No permissions to create a new namespace" if failed else "document text"}})
    events.append({"type": "item.completed", "item": {"type": "agent_message", "text": text}})
    if complete:
        events.append({"type": "turn.completed", "usage": {"input_tokens": 24, "output_tokens": 12}})
    return "\n".join(json.dumps(event) for event in events)


@pytest.fixture
def configured(tmp_path):
    executable = tmp_path / "codex"
    executable.write_text("fixture")
    executable.chmod(0o755)
    config = tmp_path / "opencode.json"
    config.write_text(json.dumps({"provider": {"test": {"models": {"model": {}}, "options": {
        "baseURL": "https://provider.example.test/v1", "apiKey": "dummy-secret-key"}}}}))
    return executable, config


def fake_popen(monkeypatch, answer, *, tool=False, failed=False, mismatch=False, readback=True):
    observed = {}

    class Process:
        returncode = 0
        pid = 12345

        def __init__(self, command, **kwargs):
            observed.update(command=command, **kwargs)
            self.command = command

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def communicate(self, prompt=None, timeout=None):
            observed["input"] = prompt
            observed["schema"] = json.loads(Path(self.command[self.command.index("--output-schema") + 1]).read_text())
            output = Path(self.command[self.command.index("-o") + 1])
            output.write_text("mismatch" if mismatch else answer)
            result = stream(answer, tool=tool, failed=failed)
            source = Path(observed["cwd"]) / "document.txt"
            if tool and source.exists() and not failed and readback:
                result = result.replace("document text", source.read_text().replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n"))
            return result, "provider stderr contains dummy-secret-key"

    monkeypatch.setattr(codex_module, "Popen", Process)
    return observed


def test_codex_command_has_no_secret_or_global_config_writes(configured, monkeypatch):
    executable, config = configured
    observed = fake_popen(monkeypatch, json.dumps(batch()))
    backend = CodexBackend(model="test/model", executable=str(executable), config=config, sandbox_backend="landlock")
    reply = backend.complete(instructions="scientific role", prompt=json.dumps({"output_schema": {"type": "object"}}))
    command = observed["command"]
    assert "dummy-secret-key" not in " ".join(command)
    assert observed["env"][KEY_ENV] == "dummy-secret-key"
    assert "--ignore-user-config" in command and "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "features.shell_tool=false" in command and "features.unified_exec=false" in command
    assert "shell_environment_policy.ignore_default_excludes=false" in command
    assert "features.multi_agent=false" in command
    assert command[-1] == "-" and observed["start_new_session"] is True
    assert reply.metadata["backend"] == "codex" and reply.metadata["tool_permissions"] == "none"
    assert "dummy-secret-key" not in json.dumps(reply.metadata)
    assert not Path(observed["cwd"]).exists()


def test_native_auth_is_left_to_codex(configured, monkeypatch):
    executable, _ = configured
    monkeypatch.setenv("CODEX_API_KEY", "native-dummy-secret")
    monkeypatch.delenv(KEY_ENV, raising=False)
    observed = fake_popen(monkeypatch, '{"ok":true}')
    backend = CodexBackend(model="glm-5.3-flash", executable=str(executable), auth_mode="codex",
                           base_url="https://provider.example.test/v1", reasoning_effort="high")
    backend.complete(instructions="role", prompt=json.dumps({"output_schema": {"type": "object"}}))
    assert KEY_ENV not in observed["env"]
    command = observed["command"]
    assert 'model_providers.OpenAI.requires_openai_auth=true' in command
    assert 'model_reasoning_effort="high"' in command
    assert not any("env_key=" in arg for arg in command)
    assert "native-dummy-secret" not in " ".join(command)


def test_native_auth_json_api_key_and_chatgpt_distinction(monkeypatch):
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: json.dumps({"OPENAI_API_KEY": "native-json-key"}))
    provider = native_provider("model", "https://provider.example.test/v1", "OpenAI")
    assert provider.key == "native-json-key" and "native-json-key" not in repr(provider)
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: json.dumps({"tokens": {"access_token": "private-login-token"}}))
    with pytest.raises(CodexError, match="credential_missing") as error:
        native_provider("model", "https://provider.example.test/v1", "OpenAI")
    assert "private-login-token" not in str(error.value)


def test_provider_references_and_missing_models(configured, monkeypatch):
    _, config = configured
    document = json.loads(config.read_text())
    document["provider"]["test"]["options"]["apiKey"] = "{env:TEST_GATEWAY_API_KEY}"
    config.write_text(json.dumps(document))
    monkeypatch.setenv("TEST_GATEWAY_API_KEY", "referenced-secret")
    assert load_provider("test/model", config).key == "referenced-secret"
    with pytest.raises(CodexError, match="provider_config_invalid"):
        load_provider("test/missing", config)
    document["provider"]["test"]["options"]["baseURL"] = "https://secret:token@provider.example.test"
    config.write_text(json.dumps(document))
    with pytest.raises(CodexError, match="endpoint_invalid") as error:
        load_provider("test/model", config)
    assert "token" not in str(error.value)


def test_ai_scientist_uses_opencode_key_for_an_explicit_gateway_model(configured, monkeypatch):
    executable, config = configured
    observed = fake_popen(monkeypatch, '{"ok":true}')
    # This model need not be listed in OpenCode: the project explicitly selects
    # it on the user's Codex gateway, reusing only the configured credential.
    backend = CodexBackend(model="glm-5.3-flash", credential_provider="test", executable=str(executable),
                           config=config, base_url="https://codex-gateway.example.test/v1", auth_mode="opencode")
    backend.complete(instructions="role", prompt=json.dumps({"output_schema": {"type": "object"}}))
    command = observed["command"]
    assert command[command.index("-m") + 1] == "glm-5.3-flash"
    assert 'model_providers.scientist_opencode.base_url="https://codex-gateway.example.test/v1"' in command
    assert 'model_providers.scientist_opencode.requires_openai_auth=false' in command
    assert observed["env"][KEY_ENV] == "dummy-secret-key"
    assert "login" not in command and "--ignore-user-config" in command


@pytest.mark.parametrize("output,code", [
    ("not JSON", "invalid_events"),
    (stream("{}", complete=False), "incomplete_response"),
    (stream("{}", tool=True), "unexpected_tool"),
    (json.dumps({"type": "turn.failed", "error": {"message": "401 private-login-token"}}), "provider_auth_failed")])
def test_failed_or_unauthorized_events_never_look_successful(output, code):
    with pytest.raises(CodexError) as error:
        parse_events(output)
    assert error.value.code == code
    assert "private-login-token" not in str(error.value)


def test_reading_requires_a_successful_tool_and_consistent_final_output(configured, monkeypatch):
    executable, config = configured
    backend = CodexBackend(model="test/model", executable=str(executable), config=config)
    with pytest.raises(CodexError, match="tool_read_not_verified"):
        parse_events(stream("{}"), allow_read_tools=True)
    with pytest.raises(CodexError, match="sandbox_namespace_denied"):
        parse_events(stream("{}", tool=True, failed=True), allow_read_tools=True)
    fake_popen(monkeypatch, '{"summary":"summary","key_points":["point"],"limitations":[]}', tool=True)
    reply = backend.read_document("A document")
    assert reply.metadata["tool_calls"][0]["exit_code"] == 0
    assert reply.metadata["document_read_verified"] is True
    fake_popen(monkeypatch, "{}", mismatch=True)
    with pytest.raises(CodexError, match="final_output_mismatch"):
        backend.complete(instructions="role", prompt=json.dumps({"output_schema": {}}))


def test_secret_output_is_discarded(configured, monkeypatch):
    executable, config = configured
    backend = CodexBackend(model="test/model", executable=str(executable), config=config)
    fake_popen(monkeypatch, '{"secret":"dummy-secret-key"}')
    with pytest.raises(CodexError, match="credential_in_output") as error:
        backend.complete(instructions="role", prompt=json.dumps({"output_schema": {}}))
    assert "dummy-secret-key" not in str(error.value)


def test_timeout_kills_tool_process_group_without_echoing_credentials(configured, monkeypatch):
    executable, config = configured
    observed = {}

    class Process:
        pid = 45678

        def __init__(self, command, **kwargs):
            observed["cwd"] = kwargs["cwd"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def communicate(self, prompt=None, timeout=None):
            if timeout:
                raise subprocess.TimeoutExpired("codex", timeout, stderr="dummy-secret-key")
            return "", ""

    monkeypatch.setattr(codex_module, "Popen", Process)
    monkeypatch.setattr(codex_module.os, "killpg", lambda pid, sig: observed.update(killed_pid=pid))
    backend = CodexBackend(model="test/model", executable=str(executable), config=config, timeout_seconds=1)
    with pytest.raises(CodexError, match="timeout") as error:
        backend.complete(instructions="role", prompt=json.dumps({"output_schema": {}}))
    assert observed["killed_pid"] == 45678 and not Path(observed["cwd"]).exists()
    assert "dummy-secret-key" not in str(error.value)


def test_document_reader_accepts_only_a_complete_json_wrapper_and_saves_source(configured, tmp_path, monkeypatch):
    from scientist.backend_checks import read_document
    from scientist.backends.common import unwrap_json_fence
    executable, config = configured
    result = {"summary": "An arithmetic example", "key_points": ["The mean is two"], "limitations": ["Synthetic data only"]}
    fake_popen(monkeypatch, "```json\n" + json.dumps(result) + "\n```", tool=True)
    backend = CodexBackend(model="test/model", executable=str(executable), config=config)
    source = tmp_path / "document.md"
    source.write_text("Synthetic data: 1, 2, 3. The mean is 2.\n")
    directory = read_document(source, backend, tmp_path / "outputs", emit=lambda _: None)
    report = json.loads((directory / "report.json").read_text())
    assert report["source_text"] == source.read_text()
    assert report["summary"]["summary"] == result["summary"]
    assert "dummy-secret-key" not in (directory / "report.json").read_text()
    assert "manifest.json" in {p.name for p in directory.iterdir()}
    # Mixed prose around JSON remains invalid, rather than extracting a fragment.
    assert unwrap_json_fence('Explanation\n{"ok":true}') == 'Explanation\n{"ok":true}'


def test_unrelated_successful_command_is_not_a_document_read(configured, monkeypatch):
    executable, config = configured
    fake_popen(monkeypatch, "{}", tool=True, readback=False)
    backend = CodexBackend(model="test/model", executable=str(executable), config=config)
    with pytest.raises(CodexError, match="document_read_incomplete"):
        backend.read_document("This document was never emitted by the tool")


def test_document_regeneration_is_bounded_and_transport_errors_are_not_retried(tmp_path):
    from scientist.backend_checks import read_document
    from scientist.backends.common import ReasoningResponse
    source = tmp_path / "example.md"
    source.write_text("Synthetic example only")
    outputs = ["{}", '{"summary":"example","key_points":["point"],"limitations":[]}']

    class Backend:
        calls = 0

        def read_document(self, text):
            self.calls += 1
            output = outputs.pop(0)
            if isinstance(output, Exception):
                raise output
            return ReasoningResponse(output, {"backend": "fixture"})

    backend = Backend()
    directory = read_document(source, backend, tmp_path / "outputs", emit=lambda _: None)
    report = json.loads((directory / "report.json").read_text())
    assert [a["accepted"] for a in report["attempts"]] == [False, True]
    outputs[:] = [CodexError("timeout", "fixture")]
    backend.calls = 0
    with pytest.raises(CodexError):
        read_document(source, backend, tmp_path / "failed", emit=lambda _: None)
    assert backend.calls == 1 and not (tmp_path / "failed").exists()


def test_codex_hypotheses_persist_the_actual_backend(configured, tmp_path, monkeypatch):
    from run_scientist import build_demo
    executable, config = configured
    fake_popen(monkeypatch, json.dumps(batch()))
    agent = make_hypothesis_agent(ReasoningConfig(backend="codex", model="test/model", config=str(config), codex_executable=str(executable)))
    orchestrator, state = build_demo(output_dir=tmp_path / "artifacts", database=tmp_path / "research.sqlite3", emit=lambda _: None)
    final = generate_hypotheses(state=state, agent=agent, store=orchestrator.store, source_root=orchestrator.source_root,
                                artifact_root=orchestrator.artifact_root, emit=lambda _: None)
    assert len(final.hypotheses) == 3 and not final.experiments
    with orchestrator.store.connect() as db:
        run_id = db.execute("SELECT id FROM runs").fetchone()[0]
    orchestrator.store.verify_artifacts(run_id)
    report = (orchestrator.artifact_root / run_id / "report.md").read_text()
    assert "codex" in report and "OpenCode" not in report
    trace = (orchestrator.artifact_root / run_id / "reasoning.json").read_text()
    assert json.loads(trace)["backend"] == "codex" and "dummy-secret-key" not in trace


def test_failure_diagnoses_are_distinct_and_cli_does_not_use_backend_flags_in_demo():
    assert failure_code("filesystem-restricted execution requires bubblewrap to isolate app-server sockets") == "landlock_incompatible"
    assert failure_code("bwrap: No permissions to create a new namespace") == "sandbox_namespace_denied"
    assert failure_code("HTTP 404 /v1/responses") == "responses_endpoint_missing"
    with pytest.raises(SystemExit):
        main(["--demo", "--backend", "codex"])
