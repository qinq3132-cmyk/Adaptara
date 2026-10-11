"""Validated role calls with persistent successful responses and bounded retries."""
import json
from pathlib import Path
from uuid import uuid4

from scientist.backends.common import unwrap_json_fence
from scientist.tools.experiment_tracker import json_bytes
from scientist.tools.literature import digest


class StructuredAgent:
    role = ""
    output_type = None

    def __init__(self, backend_factory, settings, cache_root: Path, *, emit=print):
        self.backend_factory, self.settings = backend_factory, settings
        self.cache_root, self.emit = Path(cache_root), emit
        self.attempts = []

    def output_schema(self, inputs):
        schema = self.output_type.model_json_schema()
        query = schema.get("$defs", {}).get("SearchQuery")
        if query:
            # Keep historical query records readable while requiring source choices in new LLM outputs.
            query["required"] = list(query["properties"])
            query["properties"]["providers"].pop("default", None)
        return schema

    def run(self, inputs: dict, validate=lambda value: None):
        instructions = (Path(__file__).resolve().parents[1] / "prompts" / (self.role + ".txt")).read_text(encoding="utf-8")
        request = {"task": self.role, "input": inputs, "output_schema": self.output_schema(inputs)}
        identity = {"role": self.role, "instructions": instructions, "request": request,
                    "settings": self.settings.model_dump(mode="json")}
        slot = self.cache_root / digest(identity)
        pointer = slot / "accepted.json"
        if pointer.exists():
            saved = json.loads(pointer.read_bytes())
            stored_identity = json.loads(json.dumps(saved["payload"]["identity"]))
            # Earlier retries mutated the request referenced by the saved identity.
            # Verify the original payload checksum, then compare its base request.
            stored_identity["request"].pop("validation_feedback", None)
            if digest(saved["payload"]) != saved["sha256"] or stored_identity != identity:
                raise ValueError("Agent cache checksum or identity mismatch")
            trace = saved["payload"]["attempts"]
            value = self.output_type.model_validate_json(unwrap_json_fence(trace[-1]["response_text"]))
            validate(value)
            self.attempts.extend({**a, "reused": True} for a in trace)
            self.emit(f"{self.role}: reused ({self.settings.model})")
            return value
        slot.mkdir(parents=True, exist_ok=True)
        backend = self.backend_factory(self.settings, role=self.role)
        trace = []
        for index in range(self.settings.validation_retries + 1):
            prompt = json.dumps(request, ensure_ascii=False, sort_keys=True)
            try:
                response = backend.complete(instructions=instructions, prompt=prompt)
            except (RuntimeError, OSError) as error:
                (slot / ("transport_failed_" + uuid4().hex + ".json")).write_bytes(
                    json_bytes({"role": self.role, "error_type": type(error).__name__, "identity": identity}))
                raise
            entry = {"role": self.role, "attempt": index + 1, "instructions": instructions, "prompt": prompt,
                     "response_text": response.text, "runtime": response.metadata, "accepted": False}
            trace.append(entry)
            try:
                value = self.output_type.model_validate_json(unwrap_json_fence(response.text))
                validate(value)
            except ValueError as error:
                entry["validation_error"] = str(error)[:1500]
                (slot / ("failed_" + uuid4().hex + ".json")).write_bytes(json_bytes(entry))
                if index == self.settings.validation_retries:
                    self.attempts.extend(trace)
                    raise ValueError(f"{self.role} failed validation; no stage was committed") from None
                request = {**request, "validation_feedback": str(error)[:1500]}
                continue
            entry["accepted"] = True
            payload = {"identity": identity, "attempts": trace}
            temporary = slot / ("accepted_" + uuid4().hex + ".tmp")
            temporary.write_bytes(json_bytes({"payload": payload, "sha256": digest(payload)}))
            temporary.replace(pointer)
            self.attempts.extend({**a, "reused": False} for a in trace)
            self.emit(f"{self.role}: generated ({self.settings.model})")
            return value
        raise RuntimeError("Unreachable structured agent state")
