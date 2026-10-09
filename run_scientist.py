"""Run toy experiments or scientific roles through OpenCode/Codex."""
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Run the offline E001 demo (also the default)")
    mode.add_argument("--generate-hypotheses", action="store_true", help="Generate model-authored hypotheses without running an experiment")
    mode.add_argument("--check-backend", action="store_true", help="Check Codex structured output, sandbox and file tools")
    mode.add_argument("--read-document", type=Path, help="Read a local UTF-8 .txt/.md through Codex")
    mode.add_argument("--followup", choices=("E002",), help="Execute E002 from a saved E001 run")
    parser.add_argument("--backend", choices=("opencode", "codex"), help="Reasoning runtime override")
    parser.add_argument("--model", help="Provider/model for OpenCode bridging; native model ID for Codex auth")
    parser.add_argument("--codex-bin", help="Codex executable override")
    parser.add_argument("--codex-auth", choices=("opencode", "codex"), help="Bridge OpenCode API credentials or use native Codex auth.json")
    parser.add_argument("--state-file", type=Path, help="Existing scientific state to inform hypothesis generation")
    parser.add_argument("--opencode-bin", help="OpenCode executable override")
    parser.add_argument("--opencode-config", type=Path, help="OpenCode config override; credentials are not copied")
    parser.add_argument("--parent-run", help="Parent run ID; defaults to the latest completed E001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--database", type=Path, default=ROOT / "database/research.sqlite3")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    args = parser.parse_args(argv)
    if args.parent_run and not args.followup:
        parser.error("--parent-run requires --followup E002")
    reasoning_mode = args.generate_hypotheses or args.check_backend or args.read_document
    if any((args.backend, args.model, args.codex_bin, args.codex_auth, args.opencode_bin, args.opencode_config)) and not reasoning_mode:
        parser.error("Backend options require --generate-hypotheses, --check-backend or --read-document")
    if args.state_file and not args.generate_hypotheses:
        parser.error("--state-file requires --generate-hypotheses")
    try:
        if reasoning_mode:
            from scientist.hypothesis_generation import ReasoningConfig, make_hypothesis_agent, make_reasoning_backend, generate_hypotheses
            config = yaml.safe_load((args.config_dir / "models.yaml").read_text(encoding="utf-8"))
            if args.backend:
                config["backend"] = args.backend
            if args.model:
                config["model"] = args.model
            if args.codex_bin:
                config["codex_executable"] = args.codex_bin
            if args.codex_auth:
                config["codex_auth_mode"] = args.codex_auth
            if args.opencode_bin:
                config["executable"] = args.opencode_bin
            if args.opencode_config:
                config["config"] = str(args.opencode_config)
            reasoning = ReasoningConfig.model_validate(config)
            if args.opencode_bin and reasoning.backend != "opencode":
                parser.error("--opencode-bin requires the opencode backend; use --codex-bin for Codex")
            if (args.codex_bin or args.codex_auth) and reasoning.backend != "codex":
                parser.error("Codex options require the codex backend")
            if args.check_backend:
                from scientist.backend_checks import check_backend
                return 0 if check_backend(reasoning, args.output_dir) else 1
            if args.read_document:
                from scientist.backend_checks import read_document
                if reasoning.backend != "codex":
                    raise ValueError("--read-document currently requires the Codex backend")
                read_document(args.read_document, make_reasoning_backend(reasoning), args.output_dir,
                              max_retries=reasoning.validation_retries)
                return 0
            if args.state_file:
                state = ScientificState.model_validate_json(args.state_file.read_bytes())
            else:
                project = ProjectConfig.model_validate(yaml.safe_load((args.config_dir / "project.yaml").read_text(encoding="utf-8")))
                state = ScientificState(question=project.question, budget=project.budget)
            agent = make_hypothesis_agent(reasoning)
            generate_hypotheses(state=state, agent=agent, store=ResearchStore(args.database),
                                source_root=ROOT, artifact_root=args.output_dir)
            return 0
        if args.followup:
            from scientist.followup import run_followup
            run_followup(parent_run=args.parent_run, store=ResearchStore(args.database),
                         source_root=ROOT, artifact_root=args.output_dir)
            return 0
        orchestrator, state = build_demo(output_dir=args.output_dir, database=args.database, config_dir=args.config_dir)
        orchestrator.run_cycle(state)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.exit(1, f"Scientific cycle failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
