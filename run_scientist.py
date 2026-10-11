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
    mode.add_argument("--research-literature", action="store_true", help="Research literature for a confirmed problem and save a grounded state")
    mode.add_argument("--review-hypotheses", action="store_true", help="Review and revise existing literature-grounded hypotheses")
    mode.add_argument("--research", action="store_true", help="Run literature research, hypothesis generation, review and bounded revision")
    mode.add_argument("--check-backend", action="store_true", help="Check Codex structured output, sandbox and file tools")
    mode.add_argument("--read-document", type=Path, help="Read a local UTF-8 .txt/.md through Codex")
    mode.add_argument("--followup", choices=("E002",), help="Execute E002 from a saved E001 run")
    mode.add_argument("--clarify-question", type=Path, help="Analyze a concept .txt/.md or ConceptInput .json for human review")
    mode.add_argument("--review-question", help="Revise a problem session using explicit human feedback")
    mode.add_argument("--confirm-question", help="Confirm a reviewed problem session revision without calling a model")
    parser.add_argument("--revision", type=int, help="Exact problem revision to review or confirm")
    parser.add_argument("--clarification-mode", choices=("single", "ensemble"),
                        help="Override clarification.yaml: one model or three drafts plus synthesis")
    parser.add_argument("--feedback-file", type=Path, help="HumanFeedback JSON for --review-question")
    parser.add_argument("--confirmed-by", help="Confirmation author; defaults to local-user")
    parser.add_argument("--confirmed-question", type=Path, help="Recorded confirmed_problem.json for literature or hypothesis research")
    parser.add_argument("--backend", choices=("opencode", "codex"), help="Reasoning runtime override")
    parser.add_argument("--model", help="Provider/model for OpenCode bridging; native model ID for Codex auth")
    parser.add_argument("--codex-bin", help="Codex executable override")
    parser.add_argument("--codex-auth", choices=("opencode", "codex", "chatgpt"),
                        help="Bridge OpenCode API credentials, use native API-key auth, or reuse official ChatGPT login")
    parser.add_argument("--state-file", type=Path, help="Existing scientific state to inform hypothesis generation")
    parser.add_argument("--opencode-bin", help="OpenCode executable override")
    parser.add_argument("--opencode-config", type=Path, help="OpenCode config override; credentials are not copied")
    parser.add_argument("--parent-run", help="Parent run ID; defaults to the latest completed E001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--database", type=Path, default=ROOT / "database/research.sqlite3")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--literature-config", type=Path, help="Literature agent/model and acquisition budget configuration")
    args = parser.parse_args(argv)
    if args.parent_run and not args.followup:
        parser.error("--parent-run requires --followup E002")
    problem_edit = args.review_question or args.confirm_question
    if problem_edit and (args.revision is None or args.revision < 1):
        parser.error("--review-question and --confirm-question require --revision >= 1")
    if args.revision is not None and not problem_edit:
        parser.error("--revision requires --review-question or --confirm-question")
    if args.review_question and not args.feedback_file:
        parser.error("--review-question requires --feedback-file")
    if args.feedback_file and not args.review_question:
        parser.error("--feedback-file requires --review-question")
    if args.confirmed_by and not args.confirm_question:
        parser.error("--confirmed-by requires --confirm-question")
    literature_mode = args.research_literature or args.research or args.review_hypotheses
    state_mode = args.generate_hypotheses or literature_mode
    if args.confirmed_question and (not (args.generate_hypotheses or args.research_literature or args.research) or args.state_file):
        parser.error("--confirmed-question requires literature/hypothesis generation and cannot be combined with --state-file")
    if literature_mode and not (args.confirmed_question or args.state_file):
        parser.error("Literature research requires --confirmed-question or --state-file")
    if args.review_hypotheses and not args.state_file:
        parser.error("--review-hypotheses requires --state-file")
    if args.literature_config and not literature_mode:
        parser.error("--literature-config requires a literature/review research mode")
    reasoning_mode = (args.generate_hypotheses or args.check_backend or args.read_document
                      or args.clarify_question or args.review_question or literature_mode)
    if any((args.backend, args.model, args.codex_bin, args.codex_auth, args.opencode_bin, args.opencode_config)) and not reasoning_mode:
        parser.error("Backend options require a model reasoning mode")
    if args.state_file and not state_mode:
        parser.error("--state-file requires a scientific state reasoning mode")
    if args.clarification_mode and not (args.clarify_question or args.review_question):
        parser.error("--clarification-mode requires --clarify-question or --review-question")
    try:
        if args.confirm_question:
            from scientist.problem_clarification import confirm_problem
            from scientist.tools.problem_store import ProblemStore
            confirm_problem(session_id=args.confirm_question, revision=args.revision,
                            store=ProblemStore(args.database), artifact_root=args.output_dir,
                            confirmed_by=args.confirmed_by or "local-user")
            return 0
        if reasoning_mode:
            from scientist.hypothesis_generation import ReasoningConfig, make_hypothesis_agent, make_reasoning_backend, generate_hypotheses
            config = yaml.safe_load((args.config_dir / "models.yaml").read_text(encoding="utf-8"))
            if args.backend:
                config["backend"] = args.backend
            if args.model:
                config["model"] = args.model
            if args.codex_bin:
                config["codex_executable"] = args.codex_bin
                config["codex_chatgpt_executable"] = args.codex_bin
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
            if args.clarify_question or args.review_question:
                from scientist.agents.problem_clarification import ProblemClarificationAgent
                from scientist.problem_clarification import clarify_problem, read_concept
                from scientist.problem_schemas import HumanFeedback
                from scientist.tools.problem_store import ProblemStore
                from scientist.problem_ensemble import ClarificationConfig, MultiModelProblemClarificationAgent
                ensemble_path = args.config_dir / "clarification.yaml"
                ensemble = ClarificationConfig.model_validate(
                    yaml.safe_load(ensemble_path.read_text(encoding="utf-8")) if ensemble_path.exists()
                    else {"mode": "single"})
                if args.clarification_mode:
                    ensemble = ensemble.model_copy(update={"mode": args.clarification_mode})
                if ensemble.mode == "ensemble" and any((args.model, args.backend, args.codex_auth)):
                    parser.error("Per-model overrides require --clarification-mode single; configure ensemble members in clarification.yaml")
                store = ProblemStore(args.database)
                feedback = HumanFeedback.model_validate_json(args.feedback_file.read_bytes()) if args.review_question else None
                concept = store.original_concept(args.review_question) if args.review_question else read_concept(args.clarify_question)
                if args.review_question:
                    if store.latest_revision(args.review_question) != args.revision:
                        raise ValueError("Stale feedback; review the latest problem revision")
                    store.load_revision(args.review_question, args.revision)
                if ensemble.mode == "ensemble":
                    agent = MultiModelProblemClarificationAgent(reasoning, ensemble, args.output_dir,
                                                               backend_factory=make_reasoning_backend)
                else:
                    agent = ProblemClarificationAgent(make_reasoning_backend(reasoning, role="problem_clarification"),
                                                      max_retries=reasoning.validation_retries)
                clarify_problem(concept=concept, agent=agent, store=store, source_root=ROOT,
                                artifact_root=args.output_dir, session_id=args.review_question,
                                revision=args.revision, feedback=feedback)
                return 0
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
            if args.confirmed_question:
                from scientist.problem_clarification import state_from_confirmed_problem
                from scientist.problem_schemas import ConfirmedProblem
                from scientist.tools.problem_store import ProblemStore
                problem = ConfirmedProblem.model_validate_json(args.confirmed_question.read_bytes())
                recorded = ProblemStore(args.database).load_confirmation(problem.session_id, problem.revision)
                if problem != recorded:
                    raise ValueError("Confirmed problem file differs from the recorded human confirmation")
                project = ProjectConfig.model_validate(yaml.safe_load((args.config_dir / "project.yaml").read_text(encoding="utf-8")))
                state = state_from_confirmed_problem(recorded, project.budget)
            elif args.state_file:
                state = ScientificState.model_validate_json(args.state_file.read_bytes())
            else:
                project = ProjectConfig.model_validate(yaml.safe_load((args.config_dir / "project.yaml").read_text(encoding="utf-8")))
                state = ScientificState(question=project.question, budget=project.budget)
            if literature_mode:
                from scientist.literature_pipeline import LiteratureConfig, LiteraturePipeline, research_literature
                from scientist.hypothesis_pipeline import HypothesisPipeline, reviewed_hypotheses
                from scientist.tools.problem_store import ProblemStore
                if state.problem_definition is None:
                    raise ValueError("Literature research requires a confirmed problem definition")
                recorded = ProblemStore(args.database).load_confirmation(
                    state.problem_definition.session_id, state.problem_definition.revision)
                if recorded != state.problem_definition:
                    raise ValueError("Research state differs from recorded human confirmation")
                literature_path = args.literature_config or args.config_dir / "literature.yaml"
                settings = LiteratureConfig.model_validate(yaml.safe_load(literature_path.read_text(encoding="utf-8")))
                pipeline = LiteraturePipeline(reasoning, settings, args.output_dir, backend_factory=make_reasoning_backend)
                store = ResearchStore(args.database)
                if args.research_literature or (args.research and state.literature is None):
                    state = research_literature(state=state, pipeline=pipeline, store=store,
                                               source_root=ROOT, artifact_root=args.output_dir)
                if args.research or args.review_hypotheses:
                    reviewed_hypotheses(state=state, pipeline=HypothesisPipeline(pipeline), store=store,
                                       source_root=ROOT, artifact_root=args.output_dir, generate=not args.review_hypotheses)
                return 0
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
