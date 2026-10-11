"""Run a scientific role through OpenCode, reusing its provider credentials."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .common import BackendError, ReasoningResponse, TOOL_FREE_ROLES


def locate_opencode(executable: str | None = None) -> str:
    requested = executable or os.environ.get("SCIENTIST_OPENCODE_BIN")
    if requested:
        path = shutil.which(requested)
        if path:
            return path
        raise ValueError("Configured OpenCode executable was not found")
    path = shutil.which("opencode")
    if path:
        return path
    candidates = [Path.home() / ".opencode/bin/opencode",
                  Path(sys.prefix) / "lib/node_modules/opencode-ai/node_modules/opencode-linux-x64/bin/opencode"]
    # Conda installations in this workspace may not be on the shell PATH.
    candidates.extend(sorted(Path("/opt/conda/envs").glob("*/lib/node_modules/opencode-ai/node_modules/opencode-linux-x64/bin/opencode")))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise ValueError("OpenCode was not found; set SCIENTIST_OPENCODE_BIN")


def parse_events(output: str) -> ReasoningResponse:
    parts, steps, sessions = [], [], set()
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            raise BackendError("OpenCode returned a non-JSON event") from None
        if not isinstance(event, dict):
            raise BackendError("OpenCode returned an invalid event")
        kind = event.get("type")
        if kind == "error":
            # Providers may include keys/headers in errors; never echo them.
            raise BackendError("OpenCode reported a runtime or provider error")
        if kind == "tool_use":
            raise BackendError("Scientific reasoning unexpectedly attempted a tool call")
        session = event.get("sessionID")
        if session:
            sessions.add(session)
        part = event.get("part", {})
        if not isinstance(part, dict):
            raise BackendError("OpenCode returned an invalid event part")
        if kind == "text":
            value = part.get("text")
            if not isinstance(value, str):
                raise BackendError("OpenCode text event has no text")
            parts.append(value)
        elif kind == "step_finish":
            steps.append({key: part[key] for key in ("reason", "cost", "tokens") if key in part})
    if not parts or not steps or steps[-1].get("reason") not in {"stop", "end_turn"}:
        raise BackendError("OpenCode did not finish with a complete text response")
    if len(sessions) != 1:
        raise BackendError("OpenCode events did not identify one session")
    return ReasoningResponse(text="\n".join(parts), metadata={"session_id": next(iter(sessions)), "steps": steps})


class OpenCodeBackend:
    """Each call creates a fresh role session, with no shell/edit/delegation tools.

    This adapter covers tool-free scientific reasoning. Literature/code roles need
    separate, explicit tool configurations. Provider auth stays in OpenCode.
    """
    name = "opencode"
    def __init__(self, *, model: str, executable: str | None = None,
                 config: Path | None = None, timeout_seconds: float = 180, proxy_env_precedence: str = "inherit",
                 role: str = "hypothesis"):
        if "/" not in model or not all(model.split("/", 1)):
            raise ValueError("Model must use the OpenCode provider/model format")
        if not 0 < timeout_seconds <= 600:
            raise ValueError("OpenCode timeout must be in (0, 600] seconds")
        self.model = model
        self.executable = locate_opencode(executable)
        self.config = Path(config).expanduser().resolve() if config else None
        if self.config and not self.config.is_file():
            raise ValueError("OpenCode config file does not exist")
        if proxy_env_precedence not in {"inherit", "uppercase"}:
            raise ValueError("Proxy environment precedence must be inherit or uppercase")
        self.timeout_seconds = timeout_seconds
        self.proxy_env_precedence = proxy_env_precedence
        if role not in TOOL_FREE_ROLES:
            raise ValueError("Unsupported tool-free scientific role")
        self.role = role

    def complete(self, *, instructions: str, prompt: str) -> ReasoningResponse:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"scientist-{self.role}-") as folder:
            workdir = Path(folder)
            role = {"description": ("Generate falsifiable competing scientific hypotheses" if self.role == "hypothesis"
                                    else "Perform a structured, tool-free scientific reasoning role"),
                    "mode": "primary", "prompt": instructions,
                    "permission": {"*": "deny"}, "steps": 2}
            # A one-step limit makes OpenCode append an assistant summary hint.
            # Some compatible models reject that as unsupported prefill.
            runtime_config = {"agent": {self.role: role}, "share": "disabled", "autoupdate": False}
            env = dict(os.environ)
            if self.proxy_env_precedence == "uppercase":
                for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
                    if name in env:
                        env[name.lower()] = env[name]
            env.update({"OPENCODE_CONFIG_CONTENT": json.dumps(runtime_config),
                        "OPENCODE_PERMISSION": json.dumps({"*": "deny"}),
                        "OPENCODE_DISABLE_AUTOUPDATE": "true", "OPENCODE_AUTO_SHARE": "false",
                        "OPENCODE_DISABLE_CLAUDE_CODE": "true",
                        "OPENCODE_DISABLE_MODELS_FETCH": "true"})
            if self.config:
                env["OPENCODE_CONFIG"] = str(self.config)
            title = "Autonomous Scientist hypotheses" if self.role == "hypothesis" else "Autonomous Scientist problem clarification"
            command = [self.executable, "run", "--pure", "--format", "json", "--agent", self.role,
                       "--model", self.model, "--title", title, "--dir", str(workdir), prompt]
            try:
                process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                         env=env, cwd=workdir, timeout=self.timeout_seconds, check=False)
            except subprocess.TimeoutExpired:
                raise BackendError("OpenCode scientific reasoning timed out") from None
            except OSError:
                raise BackendError("OpenCode could not be started") from None
            if process.returncode:
                raise BackendError(f"OpenCode exited with status {process.returncode}; provider errors are not echoed")
            response = parse_events(process.stdout)
        return ReasoningResponse(response.text, {**response.metadata, "backend": "opencode",
                                                 "model": self.model, "elapsed_seconds": time.monotonic() - started,
                                                 "role": self.role, "executable": self.executable,
                                                 "proxy_env_precedence": self.proxy_env_precedence,
                                                 "tool_permissions": {"*": "deny"}})
