"""SQLite is the authoritative commit; JSON files are verified exports."""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from scientist.state import ScientificState


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class ResearchStore:
    TABLES = {"questions": None, "papers": "papers", "claims": "knowledge",
              "hypotheses": "hypotheses", "experiments": "experiments", "results": "results",
              "evidence": "evidence", "decisions": "decisions", "analyses": "analyses",
              "critiques": "critiques", "theory_updates": "theory_updates"}

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, state_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
            for table in self.TABLES:
                db.execute(f"CREATE TABLE IF NOT EXISTS {table} (run_id TEXT NOT NULL REFERENCES runs(id), id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(run_id, id))")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def stage_result(self, run_id: str, directory: Path, result) -> None:
        """Save raw data before analysis. A marker distinguishes uncommitted runs."""
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "pending.json").write_bytes(json_bytes({"run_id": run_id}))
        (directory / "raw_results.json").write_bytes(json_bytes(result.model_dump(mode="json")))

    def save(self, run_id: str, state: ScientificState, directory: Path, exports: dict) -> None:
        state = ScientificState.model_validate_json(state.model_dump_json())
        if directory.exists():
            marker = json.loads((directory / "pending.json").read_text(encoding="utf-8"))
            if marker != {"run_id": run_id} or {p.name for p in directory.iterdir()} != {"pending.json", "raw_results.json"}:
                raise ValueError("Refusing to overwrite an existing run")
            if (directory / "raw_results.json").read_bytes() != json_bytes(exports["raw_results.json"]):
                raise ValueError("Staged raw results changed before commit")
        else:
            directory.mkdir(parents=True, exist_ok=False)
        checksums = {}
        for name, value in exports.items():
            if Path(name).name != name:
                raise ValueError("Artifact names must be plain filenames")
            payload = json_bytes(value)
            (directory / name).write_bytes(payload)
            checksums[name] = hashlib.sha256(payload).hexdigest()
        (directory / "manifest.json").write_bytes(json_bytes({"schema_version": 1, "run_id": run_id, "sha256": checksums}))
        # A failed file write or DB transaction leaves no authoritative run.
        # An uncommitted artifact directory is retained for diagnosis, never loaded as state.
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES (?, ?, ?)",
                       (run_id, state.model_dump_json(), str(directory.resolve())))
            for table, field in self.TABLES.items():
                records = (state.question,) + state.open_questions if field is None else getattr(state, field)
                db.executemany(f"INSERT INTO {table} VALUES (?, ?, ?)",
                               [(run_id, record.id, record.model_dump_json()) for record in records])
        # SQLite remains authoritative even if cleanup is interrupted after commit.
        (directory / "pending.json").unlink(missing_ok=True)

    def load(self, run_id: str) -> ScientificState:
        with self.connect() as db:
            row = db.execute("SELECT state_json FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return ScientificState.model_validate_json(row[0])

    def verify_artifacts(self, run_id: str) -> None:
        with self.connect() as db:
            row = db.execute("SELECT artifact_dir FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        directory = Path(row[0])
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["sha256"].items():
            if Path(name).name != name:
                raise ValueError("Invalid artifact path in manifest")
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        if ScientificState.model_validate_json((directory / "research_state.json").read_bytes()) != self.load(run_id):
            raise ValueError("JSON state does not match SQLite state")
