"""Review-gated scientific experiment proposals; no experiment execution."""
import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, StrictBool, model_validator

from .agents.structured import StructuredAgent
from .hypothesis_pipeline import load_source_documents
from .provenance import capture_source, runtime_versions
from .schemas import Experiment, ExperimentScore, Record, Stage, Text
from .tools.experiment_tracker import json_bytes
from .tools.literature import normalize


class ExperimentDesignConfig(Record):
    max_candidates: int = Field(default=2, ge=1, le=3)
    max_target_hypotheses: int = Field(default=3, ge=1, le=6)
    seeds: tuple[int, ...] = (7, 11, 19)
    target_hypothesis_ids: tuple[Text, ...] = ()

    @model_validator(mode='after')
    def distinct_seeds(self):
        if len(self.seeds) < 3 or len(self.seeds) != len(set(self.seeds)):
            raise ValueError('Provide at least three distinct pilot seeds')
        return self


class TaskDesign(Record):
    name: Text
    data_source: Text
    construction: Text
    permitted_agent_inputs: tuple[Text, ...] = Field(min_length=1)
    evaluator_only_inputs: tuple[Text, ...]
    training_protocol: Text
    evaluation_protocol: Text
    information_boundary_check: Text


class ExperimentArm(Record):
    name: Text
    role: Literal['intervention', 'control']
    intervention: Text
    parameter_learning: StrictBool
    resource_matching: Text


class ExperimentMetric(Record):
    name: Text
    definition: Text
    unit: Text
    aggregation: Text
    evaluation_window: Text
    primary: StrictBool


class ExperimentComparison(Record):
    hypothesis_id: Text
    arm_names: tuple[Text, ...] = Field(min_length=2)
    metric_names: tuple[Text, ...] = Field(min_length=1)
    predicted_difference: Text
    falsification_condition: Text


class ExperimentDesignDraft(Record):
    title: Text
    hypotheses_tested: tuple[Text, ...] = Field(min_length=1, max_length=6)
    scientific_question: Text
    task: TaskDesign
    arms: tuple[ExperimentArm, ...] = Field(min_length=2, max_length=8)
    metrics: tuple[ExperimentMetric, ...] = Field(min_length=1, max_length=10)
    comparisons: tuple[ExperimentComparison, ...] = Field(min_length=1)
    sample_unit: Text
    samples_per_seed: int = Field(ge=1, le=100000)
    seeds: tuple[int, ...] = Field(min_length=3)
    sample_rationale: Text
    statistical_analysis: Text
    confound_controls: tuple[Text, ...] = Field(min_length=1)
    implementation_steps: tuple[Text, ...] = Field(min_length=1)
    limitations: tuple[Text, ...] = Field(min_length=1)
    estimated_cost: float = Field(ge=0)
    cost_basis: Text
    scores: ExperimentScore
    human_summary: Text = Field(max_length=120)

    @model_validator(mode='after')
    def coherent_design(self):
        if len(self.hypotheses_tested) != len(set(self.hypotheses_tested)):
            raise ValueError('Repeated tested hypothesis')
        names = [a.name for a in self.arms]
        metrics = [m.name for m in self.metrics]
        if len(set(names)) != len(names) or len(set(metrics)) != len(metrics):
            raise ValueError('Arm and metric names must each be unique')
        if not {'intervention', 'control'} <= {a.role for a in self.arms}:
            raise ValueError('Include both intervention and control arms')
        if not any(m.primary for m in self.metrics):
            raise ValueError('Specify a primary measurement')
        if {c.hypothesis_id for c in self.comparisons} != set(self.hypotheses_tested):
            raise ValueError('Provide a distinguishing comparison for every tested hypothesis')
        for comparison in self.comparisons:
            if (len(set(comparison.arm_names)) < 2 or set(comparison.arm_names) - set(names)
                    or set(comparison.metric_names) - set(metrics)):
                raise ValueError('Comparisons must cite distinct known arms and known measurements')
        allowed = {normalize(s).casefold() for s in self.task.permitted_agent_inputs}
        oracle = {normalize(s).casefold() for s in self.task.evaluator_only_inputs}
        if allowed & oracle:
            raise ValueError('Evaluator-only signals cannot be permitted agent inputs')
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError('Repeated random seed')
        if '\n' in self.human_summary or '\r' in self.human_summary:
            raise ValueError('Keep the human summary on one line')
        return self


class ExperimentDesignBatch(Record):
    candidates: tuple[ExperimentDesignDraft, ...] = Field(min_length=1, max_length=3)
    confirmation_items: tuple[Text, ...] = Field(max_length=3)

    @model_validator(mode='after')
    def concise_confirmation(self):
        if any(len(s) > 100 or '\n' in s or '\r' in s for s in self.confirmation_items):
            raise ValueError('At most three single-line confirmation items, each at most 100 characters')
        return self


class ExperimentDesignAgent(StructuredAgent):
    role, output_type = 'experiment_design', ExperimentDesignBatch


def ready_hypotheses(state):
    ancestors = {p for h in state.hypotheses for p in h.parent_hypotheses}
    latest = {r.hypothesis_id: r for record in state.hypothesis_reviews for r in record.batch.reviews}
    return tuple(h for h in state.hypotheses if h.id not in ancestors and h.status == 'proposed'
                 and h.id in latest and latest[h.id].verdict == 'ready')


def _plan_report(plans, batch):
    lines = ['# 实验设计报告', '', '这些是待实现、未执行的方案，预期结果和设计评分均不是实验结果。', '']
    for experiment, draft in zip(plans, batch.candidates):
        lines += [f'## {experiment.id}：{draft.title}', '', draft.scientific_question, '',
                  f'候选：{", ".join(draft.hypotheses_tested)}', '', '### 任务与信息边界', '',
                  draft.task.construction, '', '数据来源：' + draft.task.data_source,
                  '智能体可用：' + '；'.join(draft.task.permitted_agent_inputs),
                  '仅评估者可用：' + '；'.join(draft.task.evaluator_only_inputs),
                  draft.task.information_boundary_check, '', '训练：' + draft.task.training_protocol,
                  '评价：' + draft.task.evaluation_protocol, '', '### 干预与对照', '']
        lines += [f'- {a.name}（{a.role}，参数学习={a.parameter_learning}）：{a.intervention}；资源控制：{a.resource_matching}' for a in draft.arms]
        lines += ['', '### 测量与判据', '']
        lines += [f'- {m.name}（{"主要" if m.primary else "次要"}）：{m.definition}；单位：{m.unit}；窗口：{m.evaluation_window}；汇总：{m.aggregation}' for m in draft.metrics]
        for c in draft.comparisons:
            lines += ['', f'{c.hypothesis_id}：{", ".join(c.arm_names)}；指标：{", ".join(c.metric_names)}',
                      '预测：' + c.predicted_difference, '反证条件：' + c.falsification_condition]
        lines += ['', '### 抽样、统计与实现', '',
                  f'随机种子：{list(draft.seeds)}；每种子样本数：{draft.samples_per_seed}；单位：{draft.sample_unit}。',
                  draft.sample_rationale, '', draft.statistical_analysis, '',
                  f'估计成本：{draft.estimated_cost} 抽象预算单位；{draft.cost_basis}', '',
                  *['- 混杂控制：' + s for s in draft.confound_controls], '',
                  *[f'{i}. {s}' for i, s in enumerate(draft.implementation_steps, 1)], '',
                  *['- 限制：' + s for s in draft.limitations], '']
    return '\n'.join(lines)


def design_experiments(*, state, pipeline, config, store, source_root, artifact_root, emit=print):
    if state.problem_definition is None or state.literature is None:
        raise ValueError('Experiment design requires a confirmed, literature-grounded state')
    targets = ready_hypotheses(state)
    if config.target_hypothesis_ids:
        requested = set(config.target_hypothesis_ids)
        if len(requested) != len(config.target_hypothesis_ids) or requested - {h.id for h in targets}:
            raise ValueError('Explicit design targets must be distinct current ready hypotheses')
        targets = tuple(h for h in targets if h.id in requested)
    if not targets or len(targets) > config.max_target_hypotheses:
        raise ValueError('Design requires current proposed hypotheses with a latest ready review, within the target budget')
    if state.budget.max_compute_units <= state.budget.spent_compute_units:
        raise ValueError('No remaining experiment budget')
    # A file edit must not silently attach a saved ready verdict to an altered hypothesis.
    expected = state.model_dump(mode='json')
    with store.connect() as db:
        records = db.execute('SELECT id, state_json FROM runs ORDER BY rowid DESC').fetchall()
    matching = next((identifier for identifier, raw in records if json.loads(raw) == expected), None)
    if matching is None:
        raise ValueError('Experiment design requires an unchanged committed research state and its original database')
    store.verify_artifacts(matching)
    # Validate the original evidence snapshots even though the planner receives the grounded context.
    load_source_documents(state, store)
    target_ids = {h.id for h in targets}
    latest = {r.hypothesis_id: r for record in state.hypothesis_reviews for r in record.batch.reviews}
    sources = capture_source(source_root)
    agent = pipeline.agent(ExperimentDesignAgent)
    remaining = state.budget.max_compute_units - state.budget.spent_compute_units
    def validate(batch):
        if len(batch.candidates) > config.max_candidates:
            raise ValueError('Experiment candidate budget exceeded')
        for candidate in batch.candidates:
            if set(candidate.hypotheses_tested) - target_ids:
                raise ValueError(f'Design may only test latest ready hypotheses: {sorted(target_ids)}')
            if candidate.seeds != config.seeds:
                raise ValueError('Use the configured reproducible pilot seeds in their original order')
            if candidate.estimated_cost > remaining:
                raise ValueError('Each candidate must fit the remaining budget; candidates are alternatives')
    claim_ids = {identifier for h in targets for identifier in h.source_claims}
    claims = tuple(c for c in state.knowledge if c.id in claim_ids)
    evidence_ids = {identifier for c in claims for identifier in c.source_ids}
    evidence_ids.update(identifier for h in targets for identifier in latest[h.id].evidence_ids)
    inputs = {'research_definition': state.problem_definition.model_dump(mode='json'),
              'hypotheses': [h.model_dump(mode='json') for h in targets],
              'latest_reviews': [latest[h.id].model_dump(mode='json') for h in targets],
              'knowledge': [c.model_dump(mode='json') for c in claims],
              'accepted_evidence': [e.model_dump(mode='json') for e in state.evidence if e.verification == 'accepted'
                                    and e.id in evidence_ids],
              'allowed_hypothesis_ids': sorted(target_ids), 'maximum_candidates': config.max_candidates,
              'seeds': list(config.seeds), 'remaining_abstract_compute_units': remaining,
              'execution_status': 'proposal_only'}
    try:
        batch = agent.run(inputs, validate)
    except (ValueError, RuntimeError, OSError) as error:
        folder = Path(artifact_root) / ('failed_design_' + uuid4().hex)
        folder.mkdir(parents=True)
        (folder/'failure.json').write_bytes(json_bytes({'error_type': type(error).__name__,
            'input_state': state.model_dump(mode='json'), 'attempts': agent.attempts}))
        raise
    occupied = {state.question.id}
    for value in state.model_dump(mode='json').values():
        if isinstance(value, (list, tuple)):
            occupied.update(r['id'] for r in value if isinstance(r, dict) and 'id' in r)
    plans = []
    number = 1
    for candidate in batch.candidates:
        while f'E{number:03d}' in occupied:
            number += 1
        identifier = f'E{number:03d}'; occupied.add(identifier)
        plans.append(Experiment(id=identifier, hypotheses_tested=candidate.hypotheses_tested,
            scientific_question=candidate.scientific_question,
            intervention='；'.join(a.intervention for a in candidate.arms if a.role == 'intervention'),
            control='；'.join(a.intervention for a in candidate.arms if a.role == 'control'),
            environment={'kind': 'research_proposal', 'version': '1', 'execution_status': 'not_implemented',
                         'design': candidate.model_dump(mode='json')},
            algorithms=tuple(a.name for a in candidate.arms), metrics=tuple(m.name for m in candidate.metrics),
            expected_outcomes={hid: [c.predicted_difference for c in candidate.comparisons if c.hypothesis_id == hid]
                               for hid in candidate.hypotheses_tested},
            falsification_logic={hid: [c.falsification_condition for c in candidate.comparisons if c.hypothesis_id == hid]
                                 for hid in candidate.hypotheses_tested},
            seeds=candidate.seeds, estimated_cost=candidate.estimated_cost, scores=candidate.scores))
    final = state.revised(experiments=state.experiments + tuple(plans), stage=Stage.EXPERIMENT)
    summary = ['# 实验设计简约单', '', '方案尚未实现或执行。', '']
    summary += [f'- {p.id}（{", ".join(p.hypotheses_tested)}）：{d.human_summary}' for p, d in zip(plans, batch.candidates)]
    if batch.confirmation_items:
        summary += ['', '需确认：', '', *['- ' + s for s in batch.confirmation_items]]
    summary += ['', '[完整实验设计](report.md)', '']
    run_id = 'run_' + uuid4().hex
    directory = Path(artifact_root)/run_id
    store.save(run_id, final, directory, {'input_state.json': state.model_dump(mode='json'),
        'research_state.json': final.model_dump(mode='json'), 'experiment_designs.json': batch.model_dump(mode='json'),
        'experiments.json': [p.model_dump(mode='json') for p in plans],
        'configuration.json': {'design': config.model_dump(mode='json'), 'model': agent.settings.model_dump(mode='json')},
        'reasoning.json': {'purpose': 'experiment_design_only', 'attempts': agent.attempts},
        'source_snapshot.json': sources, 'runtime.json': runtime_versions(),
        'report.md': _plan_report(plans, batch).encode('utf-8'), 'confirmation.md': '\n'.join(summary).encode('utf-8')})
    store.verify_artifacts(run_id)
    emit(f'Experiment proposals saved and verified: {directory.resolve()}; no experiment executed')
    return final
