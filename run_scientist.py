"""Run exactly one offline Milestone 1 demonstration."""
import argparse
from pathlib import Path

import yaml
from pydantic import Field

from experiments.runner import ToyExperimentRunner
from scientist.agents.analysis import MockAnalysisAgent
from scientist.agents.critic import MockCriticAgent
from scientist.agents.experiment import MockExperimentPlanner
from scientist.agents.hypothesis import MockHypothesisAgent
from scientist.agents.theory import MockTheoryAgent
from scientist.orchestrator import AgentBundle, Orchestrator
from scientist.schemas import Record, ResearchBudget, ScientificQuestion
from scientist.state import ScientificState
from scientist.tools.experiment_tracker import ResearchStore

ROOT = Path(__file__).resolve().parent


class ProjectConfig(Record):
    question: ScientificQuestion
    budget: ResearchBudget


class ExperimentConfig(Record):
    seeds: tuple[int, ...] = Field(min_length=1)


def build_demo(*, output_dir: Path = ROOT / "artifacts", database: Path = ROOT / "database/research.sqlite3",
               config_dir: Path = ROOT / "config", emit=print, should_stop=lambda: False):
    project = ProjectConfig.model_validate(yaml.safe_load((config_dir / "project.yaml").read_text(encoding="utf-8")))
    experiments = ExperimentConfig.model_validate(yaml.safe_load((config_dir / "experiments.yaml").read_text(encoding="utf-8")))
    state = ScientificState(question=project.question, budget=project.budget)
    agents = AgentBundle(MockHypothesisAgent(), MockExperimentPlanner(experiments.seeds),
                         MockAnalysisAgent(), MockCriticAgent(), MockTheoryAgent())
    orchestrator = Orchestrator(agents=agents, runner=ToyExperimentRunner(),
                                store=ResearchStore(database), source_root=ROOT,
                                artifact_root=output_dir, emit=emit, should_stop=should_stop)
    return orchestrator, state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run the offline demo (also the default)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--database", type=Path, default=ROOT / "database/research.sqlite3")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    args = parser.parse_args(argv)
    try:
        orchestrator, state = build_demo(output_dir=args.output_dir, database=args.database, config_dir=args.config_dir)
        orchestrator.run_cycle(state)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Scientific cycle failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
