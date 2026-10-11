"""SQLite-authoritative, append-only clarification revisions and confirmations."""
import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from scientist.problem_schemas import ConceptInput, ConfirmedProblem, ProblemRevision, problem_digest
from scientist.tools.experiment_tracker import json_bytes


class ProblemStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS problem_sessions "
                       "(id TEXT PRIMARY KEY, concept_json TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS problem_revisions "
                       "(session_id TEXT NOT NULL REFERENCES problem_sessions(id), revision INTEGER NOT NULL, "
                       "payload_json TEXT NOT NULL, draft_sha256 TEXT NOT NULL, artifact_dir TEXT NOT NULL, "
                       "PRIMARY KEY(session_id, revision))")
            db.execute("CREATE TABLE IF NOT EXISTS problem_confirmations "
                       "(session_id TEXT NOT NULL, revision INTEGER NOT NULL, payload_json TEXT NOT NULL, "
                       "artifact_dir TEXT NOT NULL, PRIMARY KEY(session_id, revision), "
                       "FOREIGN KEY(session_id, revision) REFERENCES problem_revisions(session_id, revision))")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def check_session_id(session_id: str):
        if not re.fullmatch(r"problem_[0-9a-f]{32}", session_id):
            raise ValueError("Invalid problem session ID")

    def original_concept(self, session_id: str) -> ConceptInput:
        self.check_session_id(session_id)
        with self.connect() as db:
            row = db.execute("SELECT concept_json FROM problem_sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown problem session")
        return ConceptInput.model_validate_json(row[0])

    def latest_revision(self, session_id: str) -> int:
        self.check_session_id(session_id)
        with self.connect() as db:
            number = db.execute("SELECT MAX(revision) FROM problem_revisions WHERE session_id=?",
                                (session_id,)).fetchone()[0]
        if number is None:
            raise ValueError("Unknown problem session")
        return number

    @staticmethod
    def _write_artifacts(directory: Path, exports: dict) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "pending.json").write_bytes(json_bytes({"status": "uncommitted"}))
        checksums = {}
        for name, value in exports.items():
            if Path(name).name != name:
                raise ValueError("Artifact names must be plain filenames")
            data = value if isinstance(value, bytes) else json_bytes(value)
            (directory / name).write_bytes(data)
            checksums[name] = hashlib.sha256(data).hexdigest()
        (directory / "manifest.json").write_bytes(json_bytes({"schema_version": 1, "sha256": checksums}))

    @staticmethod
    def _verify_artifacts(directory: Path, record_name: str, expected) -> None:
        manifest = json.loads((directory / "manifest.json").read_bytes())
        checksums = manifest["sha256"]
        if record_name not in checksums:
            raise ValueError("Required problem record is missing from manifest")
        for name, digest in checksums.items():
            if Path(name).name != name:
                raise ValueError("Invalid problem artifact path")
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Problem artifact checksum mismatch")
        saved = type(expected).model_validate_json((directory / record_name).read_bytes())
        if saved != expected:
            raise ValueError("Problem artifact differs from SQLite record")

    def save_revision(self, record: ProblemRevision, directory: Path, exports: dict) -> None:
        self.check_session_id(record.session_id)
        record = ProblemRevision.model_validate_json(record.model_dump_json())
        # Holding the write transaction prevents two revisions being committed
        # against the same parent. Files remain pending if any operation fails.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            latest = db.execute("SELECT MAX(revision) FROM problem_revisions WHERE session_id=?",
                                (record.session_id,)).fetchone()[0]
            if record.revision != (latest or 0) + 1:
                raise ValueError("Stale problem revision; reload the latest version")
            if latest is None:
                db.execute("INSERT INTO problem_sessions VALUES (?, ?)",
                           (record.session_id, record.concept.model_dump_json()))
            else:
                parent_json, parent_digest = db.execute(
                    "SELECT payload_json, draft_sha256 FROM problem_revisions WHERE session_id=? AND revision=?",
                    (record.session_id, latest)).fetchone()
                if record.previous_draft_sha256 != parent_digest:
                    raise ValueError("Problem revision lineage mismatch")
                from scientist.agents.problem_clarification import validate_revision
                parent = ProblemRevision.model_validate_json(parent_json)
                validate_revision(parent.draft, record.feedback, record.draft)
                constraints = (record.feedback.revised_constraints if record.feedback.revised_constraints is not None
                               else parent.concept.explicit_constraints)
                effective = ConceptInput.model_validate({**parent.concept.model_dump(),
                                                         "explicit_constraints": constraints})
                if record.concept != effective:
                    raise ValueError("Revision changed original input without explicit human feedback")
            exports = {**exports, "revision.json": record.model_dump(mode="json")}
            self._write_artifacts(directory, exports)
            db.execute("INSERT INTO problem_revisions VALUES (?, ?, ?, ?, ?)",
                       (record.session_id, record.revision, record.model_dump_json(),
                        problem_digest(record.draft), str(directory.resolve())))
        (directory / "pending.json").unlink(missing_ok=True)

    def load_revision(self, session_id: str, revision: int) -> ProblemRevision:
        self.check_session_id(session_id)
        with self.connect() as db:
            row = db.execute("SELECT payload_json, draft_sha256, artifact_dir FROM problem_revisions "
                             "WHERE session_id=? AND revision=?", (session_id, revision)).fetchone()
        if row is None:
            raise ValueError("Unknown problem revision")
        record = ProblemRevision.model_validate_json(row[0])
        if problem_digest(record.draft) != row[1]:
            raise ValueError("Stored problem draft checksum mismatch")
        self._verify_artifacts(Path(row[2]), "revision.json", record)
        return record

    def save_confirmation(self, problem: ConfirmedProblem, directory: Path) -> None:
        problem = ConfirmedProblem.model_validate_json(problem.model_dump_json())
        record = self.load_revision(problem.session_id, problem.revision)
        if (problem.draft != record.draft or problem.concept != record.concept
                or problem.original_concept != self.original_concept(problem.session_id)):
            raise ValueError("Confirmation does not match the recorded problem revision")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            latest = db.execute("SELECT MAX(revision) FROM problem_revisions WHERE session_id=?",
                                (problem.session_id,)).fetchone()[0]
            if problem.revision != latest:
                raise ValueError("Only the latest problem revision can be confirmed")
            if db.execute("SELECT 1 FROM problem_confirmations WHERE session_id=? AND revision=?",
                          (problem.session_id, problem.revision)).fetchone():
                raise ValueError("This problem revision has already been confirmed")
            self._write_artifacts(directory, {"confirmed_problem.json": problem.model_dump(mode="json")})
            db.execute("INSERT INTO problem_confirmations VALUES (?, ?, ?, ?)",
                       (problem.session_id, problem.revision, problem.model_dump_json(), str(directory.resolve())))
        (directory / "pending.json").unlink(missing_ok=True)

    def load_confirmation(self, session_id: str, revision: int) -> ConfirmedProblem:
        self.check_session_id(session_id)
        with self.connect() as db:
            row = db.execute("SELECT payload_json, artifact_dir FROM problem_confirmations "
                             "WHERE session_id=? AND revision=?", (session_id, revision)).fetchone()
        if row is None:
            raise ValueError("Problem revision has not been confirmed")
        problem = ConfirmedProblem.model_validate_json(row[0])
        self._verify_artifacts(Path(row[1]), "confirmed_problem.json", problem)
        record = self.load_revision(session_id, revision)
        if problem.draft != record.draft or problem.concept != record.concept:
            raise ValueError("Confirmed problem differs from recorded revision")
        return problem
