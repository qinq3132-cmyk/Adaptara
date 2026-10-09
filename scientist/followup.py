"""Execute the specific E001 -> E002 follow-up, not a general scheduler."""
import hashlib
import json
import re
from pathlib import Path
from statistics import fmean
from uuid import uuid4

from experiments.followup_runner import FollowupExperimentRunner
from experiments.runner import ToyConfig
from scientist.agents.experiment import followup_plan
from scientist.orchestrator import CycleStopped
from scientist.provenance import capture_source, make_provenance, runtime_versions
from scientist.schemas import Analysis, Critique, Decision, Evidence, ExperimentResult, Hypothesis, HypothesisChange, ResearchQuestion, Stage, TheoryUpdate
from scientist.state import ScientificState
from scientist.tools.experiment_tracker import ResearchStore, json_bytes


def latest_parent(store: ResearchStore) -> str:
    with store.connect() as db:
        rows = db.execute("SELECT id, state_json FROM runs ORDER BY rowid DESC").fetchall()
    for run_id, payload in rows:
        state = ScientificState.model_validate_json(payload)
        if (state.current_round == 1 and state.stage == Stage.THEORY_UPDATE
                and any(e.id == "E002" for e in state.experiments)
                and not any(r.experiment_id == "E002" for r in state.results)):
            return run_id
    raise ValueError("No completed E001 run with a pending E002 was found")


def verified_parent(store: ResearchStore, run_id: str, artifact_root: Path):
    if not re.fullmatch(r"run_[0-9a-f]{32}", run_id):
        raise ValueError("Invalid parent run ID")
    state = store.load(run_id)
    with store.connect() as db:
        saved = Path(db.execute("SELECT artifact_dir FROM runs WHERE id = ?", (run_id,)).fetchone()[0])
    # Support a copied project whose existing DB still records Windows paths.
    directory = saved if saved.is_dir() else artifact_root / run_id
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["run_id"] != run_id:
        raise ValueError("Parent manifest run ID mismatch")
    if not {"research_state.json", "raw_results.json", "source_snapshot.json", "experiment.json"} <= manifest["sha256"].keys():
        raise ValueError("Parent manifest is missing required artifacts")
    for name, expected in manifest["sha256"].items():
        if Path(name).name != name or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Parent artifact checksum mismatch: {name}")
    if ScientificState.model_validate_json((directory / "research_state.json").read_bytes()) != state:
        raise ValueError("Parent artifacts do not match authoritative SQLite state")
    result = ExperimentResult.model_validate_json((directory / "raw_results.json").read_bytes())
    if result not in state.results or result.experiment_id != "E001":
        raise ValueError("Parent raw result is not the recorded E001")
    sources = json.loads((directory / "source_snapshot.json").read_text(encoding="utf-8"))
    if hashlib.sha256(json_bytes(sources)).hexdigest() != result.provenance.source_sha256:
        raise ValueError("Parent source snapshot does not match provenance")
    return state, directory, sources


def analyze_followup(experiment, result):
    metrics = result.metrics
    frozen = metrics["toy_frozen_loss"]
    active = metrics["toy_residual_gate_parameter_updates"] > 0
    flags = {
        "H002_prediction_met": float(active and metrics["relevance_filter_parameter_updates"] == 0
                                     and metrics["relevance_filter_loss"] <= frozen + 1e-12),
        "H003_prediction_met": float(active and metrics["predictor_state_state_updates"] > 0
                                     and metrics["predictor_state_parameter_updates"] == 0
                                     and metrics["predictor_state_loss"] <= frozen + 1e-12
                                     and metrics["predictor_state_final_prediction_mse"]
                                     < metrics["toy_frozen_final_prediction_mse"]),
    }
    return Analysis(
        id="A002", result_ids=(result.id,),
        observations=(f"Executed {len(experiment.algorithms)} arms on identical per-seed sensor observations.",
                      "Positive, negative, alternating action updates and predictor-state updates share the fixed-zero residual trigger.",
                      f"Positive/negative/alternating loss: {metrics['toy_residual_gate_loss']:.6g} / {metrics['reverse_direction_control_loss']:.6g} / {metrics['matched_update_control_loss']:.6g}.",
                      f"Predictor final squared residual: {metrics['predictor_state_final_prediction_mse']:.6g}; frozen: {metrics['toy_frozen_final_prediction_mse']:.6g}."),
        statistical_comparisons={**metrics, **flags},
        possible_interpretations=("Descriptive paired comparisons only; seeds do not provide independent replication of the constructed loss effect.",),
    )


def per_seed_results(experiment, result):
    summaries = []
    for seed in experiment.seeds:
        for algorithm in experiment.algorithms:
            rows = [r for r in result.trajectory if r["seed"] == seed and r["algorithm"] == algorithm]
            summaries.append({"seed": seed, "algorithm": algorithm,
                              "mean_loss": fmean(r["loss"] for r in rows),
                              "parameter_updates": sum(r["parameter_updates"] for r in rows),
                              "state_updates": sum(r["state_updates"] for r in rows),
                              "final_prediction_mse": rows[-1]["prediction_error_after"]})
    return summaries


def report_text(parent_run, experiment, result, state):
    metrics = result.metrics
    lines = ["# E002 实验报告", "", f"父运行：`{parent_run}`", "",
             f"种子：{list(experiment.seeds)}；每个算法每个种子运行 {experiment.environment['steps']} 步。", "",
             "| 算法 | 平均平方跟踪损失 | 行动参数更新 | 预测器状态更新 | 最终预测残差平方 |",
             "|---|---:|---:|---:|---:|"]
    for name in experiment.algorithms:
        lines.append(f"| {name} | {metrics[name + '_loss']:.6f} | {metrics[name + '_parameter_updates']:g} | {metrics[name + '_state_updates']:g} | {metrics[name + '_final_prediction_mse']:.6f} |")
    lines += ["", "更新次数在全部种子上累计；最终预测误差是每个种子末步误差的均值。", "",
              "## 解释与范围", "",
              state.theory_updates[-1].interpretation, "",
              "相关性过滤预先知道第二通道与公开的跟踪目标无关；本实验没有学习通道相关性。",
              "正向、反向、交替方向和预测器状态规则都由观测到的固定零预测器残差触发；算法不读取变化时刻、隐藏目标或 evaluator 的损失。",
              "这四组匹配的是更新事件次数，行动参数更新与预测器状态更新不是等价计算操作。过滤组与冻结组的更新次数为零。",
              "预测器状态适应使用独立状态吸收偏移，触发器本身保持不变。交替方向控制的方向由自身更新计数决定。",
              "种子改变信号轨迹，损失差在这个构造中基本不随信号变化；不做显著性、独立复现或一般适应能力声明。",
              "假设的 confidence 保留父状态分数，不将本次结果转成校准概率。", "",
              "## 复现", "", "用 research_state.json 中 E002 的计划与 R002.provenance 调用 FollowupExperimentRunner.execute，结果应完全一致。", ""]
    return "\n".join(lines)


def run_followup(*, parent_run: str | None, store: ResearchStore, source_root: Path,
                 artifact_root: Path, emit=print, should_stop=lambda: False):
    def check_stop():
        if should_stop():
            raise CycleStopped("Human stop requested")

    check_stop()
    parent_run = parent_run or latest_parent(store)
    parent, parent_directory, parent_sources = verified_parent(store, parent_run, artifact_root)
    if parent.current_round != 1 or parent.stage != Stage.THEORY_UPDATE:
        raise ValueError("E002 follow-up requires a completed first cycle")
    if any(r.experiment_id == "E002" for r in parent.results):
        raise ValueError("Parent already executed E002")
    proposal = next((e for e in parent.experiments if e.id == "E002"), None)
    if proposal is None:
        raise ValueError("Parent has no E002 proposal")
    config = ToyConfig.model_validate(proposal.environment)
    plan = followup_plan(seeds=proposal.seeds)
    plan = type(plan).model_validate({**plan.model_dump(), "environment": proposal.environment,
                                     "estimated_cost": float(config.steps * len(plan.seeds) * len(plan.algorithms))})
    available = parent.budget.max_compute_units - parent.budget.spent_compute_units
    if plan.estimated_cost > available:
        raise CycleStopped("E002 exceeds remaining compute budget")
    budget = type(parent.budget).model_validate({**parent.budget.model_dump(),
                                               "max_rounds": max(2, parent.budget.max_rounds)})
    working = parent.revised(experiments=tuple(plan if e.id == "E002" else e for e in parent.experiments), budget=budget)
    run_id = "run_" + uuid4().hex
    directory = artifact_root / run_id
    sources = capture_source(source_root)
    runner = FollowupExperimentRunner()
    provenance = make_provenance(source_root, run_id, plan, sources,
                                 runner_version=runner.runner_version, algorithm_version=runner.algorithm_version)
    emit(f"E002 follow-up; parent: {parent_run}")
    check_stop()
    result = ExperimentResult.model_validate(runner.execute(plan, provenance).model_dump())
    if result.provenance != provenance or result.experiment_id != plan.id:
        raise ValueError("Mismatched follow-up result")
    budget = type(budget).model_validate({**budget.model_dump(),
                                        "spent_compute_units": budget.spent_compute_units + result.compute_cost["compute_units"]})
    store.stage_result(run_id, directory, result)
    emit("E002 executed; raw results saved before analysis")
    check_stop()
    analysis = analyze_followup(plan, result)
    critique = Critique(
        id="C002", analysis_id=analysis.id,
        validity="A deterministic nuisance-only fixture, with event-matched direction and state controls.",
        confounds=("Relevance is declared in advance rather than discovered.",
                   "Action parameters and predictor state have different effects and update costs.",
                   "Relevance-filter and frozen arms have zero updates; only the other four arms are event-matched.",
                   "Signal seeds do not independently vary the constructed nuisance effect."),
        alternative_explanations=("Alternating directions reduce cumulative offset without learning behavioral relevance.",),
        required_controls=("Introduce a behaviorally relevant shift with online relevance inference.",
                           "Vary shift amplitude, duration, trigger threshold and update direction."),
        recommended_next_experiment="Test unknown relevance on both nuisance-only and behaviorally relevant shifts.",
        conclusion_too_strong=False,
    )
    evidence, changes = [], []
    definitions = (("H002", "EV002", "Declared relevance filtering preserved tracking loss without action updates."),
                   ("H003", "EV003", "Predictor-state adaptation reduced nuisance prediction error without action-parameter changes or excess tracking loss."))
    for hypothesis_id, evidence_id, claim in definitions:
        if analysis.statistical_comparisons[f"{hypothesis_id}_prediction_met"]:
            evidence.append(Evidence(id=evidence_id, source_type="experiment", source_id="E002", claim=claim,
                                     supports=(hypothesis_id,), strength="weak",
                                     limitations=critique.confounds))
            previous = next(h for h in working.hypotheses if h.id == hypothesis_id)
            changes.append(HypothesisChange(hypothesis_id=hypothesis_id, status="supported",
                                            confidence=previous.confidence, evidence_ids=(evidence_id,),
                                            rationale="The measured prediction holds in this fixture; retain the uncalibrated internal belief score."))
    interpretation = ("E002 provides weak, fixture-specific support for " + ", ".join(c.hypothesis_id for c in changes)
                      if changes else "E002 did not meet the predeclared support conditions; beliefs remain unchanged.")
    interpretation += ". H001 retains its previous status. Equal event counts produce different losses under different update directions; no general mechanism necessity follows."
    update = TheoryUpdate(id="T002", analysis_id=analysis.id, critique_id=critique.id,
                          interpretation=interpretation, changes=tuple(changes), evidence=tuple(evidence),
                          open_questions=(ResearchQuestion(id="Q003", text="Can relevance be inferred online when the changing channel is not known in advance?", source_ids=("C002",)),))
    all_evidence = working.evidence + update.evidence
    change_map = {c.hypothesis_id: c for c in changes}
    hypotheses = []
    for h in working.hypotheses:
        fields = h.model_dump()
        if h.id in change_map:
            change = change_map[h.id]
            fields.update(status=change.status, confidence=change.confidence)
        fields.update(supporting_evidence=tuple(e.id for e in all_evidence if h.id in e.supports),
                      contradicting_evidence=tuple(e.id for e in all_evidence if h.id in e.contradicts))
        hypotheses.append(Hypothesis.model_validate(fields))
    decisions = tuple(Decision(id=f"D{len(working.decisions) + i:03d}", stage=stage, summary=summary,
                               related_ids=(identifier,))
                      for i, (stage, summary, identifier) in enumerate((
                          (Stage.EXPERIMENT, "Explicitly authorize one E002 follow-up; preserve the parent run and refine its proposal", "E002"),
                          (Stage.ANALYSIS, "Record paired descriptive comparisons", "A002"),
                          (Stage.CRITIQUE, "Record event-matching limits and declared relevance", "C002"),
                          (Stage.THEORY_UPDATE, interpretation, "T002")), 1))
    final = working.revised(hypotheses=tuple(hypotheses), results=working.results + (result,),
                            analyses=working.analyses + (analysis,), critiques=working.critiques + (critique,),
                            theory_updates=working.theory_updates + (update,), evidence=all_evidence,
                            open_questions=working.open_questions + update.open_questions,
                            decisions=working.decisions + decisions, current_round=2, budget=budget)
    lineage = {"parent_run_id": parent_run, "parent_artifact_dir": str(parent_directory.resolve()),
               "parent_state_sha256": hashlib.sha256(json_bytes(parent.model_dump(mode="json"))).hexdigest(),
               "parent_proposal": proposal.model_dump(mode="json"),
               "refinement": "Add explicit frozen, original and direction controls; report predictor errors and separate state/action update costs; preserve environment and seeds.",
               "round_limit_extension": {"before": parent.budget.max_rounds, "after": budget.max_rounds}}
    report = report_text(parent_run, plan, result, final)
    exports = {"question.json": final.question.model_dump(mode="json"),
               "hypotheses.json": {"hypotheses": [h.model_dump(mode="json") for h in parent.hypotheses]},
               "experiment_plan.json": {"selected_id": "E002", "candidates": [plan.model_dump(mode="json")]},
               "experiment.json": plan.model_dump(mode="json"), "raw_results.json": result.model_dump(mode="json"),
               "analysis.json": analysis.model_dump(mode="json"), "critique.json": critique.model_dump(mode="json"),
               "theory_update.json": update.model_dump(mode="json"), "research_state.json": final.model_dump(mode="json"),
               "source_snapshot.json": sources, "runtime.json": runtime_versions(), "lineage.json": lineage,
               "parent_state.json": parent.model_dump(mode="json"), "parent_source_snapshot.json": parent_sources,
               "results_by_seed.json": per_seed_results(plan, result), "report.md": report.encode("utf-8")}
    check_stop()
    store.save(run_id, final, directory, exports)
    store.verify_artifacts(run_id)
    replayed = runner.execute(plan, result.provenance)
    if replayed != result:
        raise RuntimeError("E002 replay differs from the saved result")
    emit("E002 committed; checksums and exact runner replay verified")
    for name in plan.algorithms:
        emit(f"{name}: loss={result.metrics[name + '_loss']:.6f}, action_updates={result.metrics[name + '_parameter_updates']:g}, state_updates={result.metrics[name + '_state_updates']:g}")
    emit(f"Artifacts: {directory.resolve()}")
    return final
