import hashlib
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

from .schemas import Experiment, Provenance
from .tools.experiment_tracker import json_bytes


def capture_source(root: Path) -> dict[str, str]:
    """Store exact executable/configuration sources, including uncommitted edits."""
    paths = [root / "run_scientist.py", root / "pyproject.toml"]
    for folder in ("scientist", "experiments", "algorithms", "environments"):
        paths.extend((root / folder).rglob("*.py"))
    paths.extend((root / "config").glob("*.yaml"))
    return {path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted(set(paths)) if path.is_file()}


def make_provenance(root: Path, run_id: str, experiment: Experiment, sources: dict[str, str],
                    *, runner_version: str, algorithm_version: str) -> Provenance:
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    config = experiment.model_dump(mode="json")
    return Provenance(
        run_id=run_id, code_commit=commit,
        source_sha256=hashlib.sha256(json_bytes(sources)).hexdigest(),
        config_sha256=hashlib.sha256(json_bytes(config)).hexdigest(),
        environment_version=str(experiment.environment.get("version", "unspecified")),
        algorithm_version=algorithm_version, runner_version=runner_version,
        seeds=experiment.seeds, config=config,
    )


def runtime_versions() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "pydantic": version("pydantic"), "PyYAML": version("PyYAML")}
