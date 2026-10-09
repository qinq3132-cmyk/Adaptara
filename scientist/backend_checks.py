"""Reproducible Codex compatibility checks and a bounded local text reader."""
import hashlib
import json
import subprocess
from pathlib import Path
from uuid import uuid4

from pydantic import Field

from scientist.backends.codex import CodexBackend, CodexError, failure_code
from scientist.backends.common import unwrap_json_fence
from scientist.hypothesis_generation import ReasoningConfig, make_reasoning_backend
from scientist.schemas import Record, Text


class DocumentSummary(Record):
    summary: Text
    key_points: tuple[Text, ...] = Field(min_length=1)
    limitations: tuple[Text, ...]


def save_report(root: Path, purpose: str, payload: dict, markdown: str) -> Path:
    directory = root / (purpose + "_" + uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "report.md").write_text(markdown, encoding="utf-8")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def check_backend(config: ReasoningConfig, output_dir: Path, emit=print) -> bool:
    if config.backend != "codex":
        raise ValueError("--check-backend currently checks Codex; select --backend codex")
    from scientist.backends.codex import locate_codex
    binary = locate_codex(config.codex_executable)
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
    command = [binary, "-c", 'sandbox_mode="read-only"']
    if config.codex_sandbox_backend == "landlock":
        command += ["--enable", "use_legacy_landlock"]
    # These installed versions accept `sandbox -- command`, not `sandbox linux`.
    command += ["sandbox", "--", "/bin/true"]
    sandbox = subprocess.run(command, capture_output=True, text=True, timeout=15)
    cases = [{"name": "sandbox_startup", "passed": sandbox.returncode == 0,
              "failure_code": None if sandbox.returncode == 0 else failure_code(sandbox.stderr)}]
    try:
        backend = make_reasoning_backend(config)
    except CodexError as error:
        cases.append({"name": "credentials", "passed": False, "failure_code": error.code, "detail": str(error)})
        backend = None
    if backend:
        emit("Checking Codex structured response and file-tool continuation...")
        try:
            reply = backend.complete(instructions='Return exactly {"ok":true}. Do not use tools.', prompt=json.dumps({
                "task": 'Return exactly {"ok":true}', "output_schema": {"type": "object", "properties": {
                    "ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}}))
            cases.append({"name": "structured_response", "passed": json.loads(unwrap_json_fence(reply.text)) == {"ok": True}, "runtime": reply.metadata})
        except (CodexError, ValueError) as error:
            cases.append({"name": "structured_response", "passed": False,
                          "failure_code": getattr(error, "code", "invalid_json")})
        # A failed local sandbox check already identifies the tool blocker;
        # avoid an unnecessary provider call when a tool cannot run.
        if sandbox.returncode == 0:
            reply = None
            try:
                reply = backend.read_document("研究结论：冻结动作规则在干扰变化后保持零损失。实验仅包含十步，不能推断真实环境中的泛化能力。\n")
                summary = DocumentSummary.model_validate_json(unwrap_json_fence(reply.text))
                cases.append({"name": "file_tool_continuation", "passed": True,
                              "summary": summary.model_dump(mode="json"), "runtime": reply.metadata})
            except (CodexError, ValueError) as error:
                cases.append({"name": "file_tool_continuation", "passed": False,
                              "failure_code": getattr(error, "code", "invalid_document_structure"),
                              **({"runtime": reply.metadata, "response_text": reply.text} if reply else {})})
        else:
            cases.append({"name": "file_tool_continuation", "passed": False, "skipped": "sandbox startup failed"})
    passed = backend is not None and all(case["passed"] for case in cases)
    payload = {"purpose": "backend_compatibility", "backend": "codex", "model": config.model,
               "auth_mode": config.codex_auth_mode, "executable": binary, "codex_version": version,
               "sandbox_backend": config.codex_sandbox_backend, "passed": passed, "checks": cases}
    markdown = "# Codex 后端兼容性检查\n\n" + f"模型：`{config.model}`；版本：`{version}`；总体通过：{passed}。\n\n"
    markdown += "\n".join(f"- {case['name']}: {'通过' if case['passed'] else '未通过'}" +
                           (f" ({case['failure_code']})" if case.get("failure_code") else "") for case in cases)
    directory = save_report(output_dir, "backend_check", payload, markdown)
    emit(f"Backend check {'passed' if passed else 'failed'}: {directory.resolve()}")
    return passed


def read_document(path: Path, backend: CodexBackend, output_dir: Path, emit=print, *, max_retries: int = 1) -> Path:
    """Initial reader: UTF-8 .txt/.md only, with no automatic truncation."""
    if path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Document reader currently supports UTF-8 .txt/.md. Extract PDF text first; OCR/retrieval is not implemented")
    if not 0 <= max_retries <= 2:
        raise ValueError("Document validation permits at most two retries")
    raw = path.read_bytes()
    if not raw or len(raw) > 32000:
        raise ValueError("Document must contain 1-32000 bytes; split larger documents explicitly")
    text = raw.decode("utf-8")
    emit("Reading the staged document through Codex's read-only shell tool...")
    attempts = []
    for attempt in range(max_retries + 1):
        reply = backend.read_document(text)
        entry = {"attempt": attempt + 1, "response_text": reply.text, "runtime": reply.metadata, "accepted": False}
        attempts.append(entry)
        try:
            summary = DocumentSummary.model_validate_json(unwrap_json_fence(reply.text))
        except ValueError:
            entry["validation_error"] = "Invalid summary, key_points or limitations structure"
            if attempt == max_retries:
                raise ValueError("Document summary failed structural validation; no report was saved") from None
            continue
        entry["accepted"] = True
        break
    payload = {"purpose": "local_document_reading", "source_name": path.name,
               "source_sha256": hashlib.sha256(raw).hexdigest(), "source_text": text,
               "summary": summary.model_dump(mode="json"), "runtime": reply.metadata, "response_text": reply.text, "attempts": attempts,
               "scientific_status": "unreviewed summary; no claims/evidence committed to scientific state"}
    markdown = f"# 文档阅读结果\n\n来源：`{path.name}`。此摘要尚未通过科学批判。\n\n{summary.summary}\n\n"
    markdown += "## 要点\n\n" + "\n".join(f"- {point}" for point in summary.key_points)
    markdown += "\n\n## 局限\n\n" + "\n".join(f"- {point}" for point in summary.limitations)
    directory = save_report(output_dir, "document_reading", payload, markdown)
    emit(f"Document summary saved: {directory.resolve()}")
    return directory
