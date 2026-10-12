"""Proposal-only design gating, provenance, information boundaries and budgets."""
import json

import pytest

from run_scientist import ROOT, main
from scientist.backends.common import ReasoningResponse
from scientist.experiment_design import ExperimentDesignConfig, design_experiments, ready_hypotheses
from scientist.hypothesis_generation import ReasoningConfig
from scientist.hypothesis_pipeline import HypothesisPipeline, reviewed_hypotheses
from scientist.literature_pipeline import LiteratureConfig, LiteraturePipeline, research_literature
from scientist.schemas import Stage
from scientist.tools.experiment_tracker import ResearchStore
from test_literature_agents import FakeTools, FixtureFactory, confirmed_state


def draft(inputs):
    hid = inputs['allowed_hypothesis_ids'][0]
    return {'title': 'Feedback-conditioned response pilot', 'hypotheses_tested': [hid],
        'scientific_question': 'Does available history cause the response difference?',
        'task': {'name': 'Hidden response context', 'data_source': 'Locally generated episodes',
            'construction': 'Generate two hidden reward mappings with identical observations.',
            'permitted_agent_inputs': ['observation', 'reward history'], 'evaluator_only_inputs': ['context truth'],
            'training_protocol': 'Warm up online using permitted feedback only.',
            'evaluation_protocol': 'Pause parameter learning and use matched episode streams.',
            'information_boundary_check': 'Assert that context truth never enters agent inputs.'},
        'arms': [{'name': 'normal', 'role': 'intervention', 'intervention': 'Use normal history',
                  'parameter_learning': False, 'resource_matching': 'Same fixed capacity and history length'},
                 {'name': 'shuffle', 'role': 'control', 'intervention': 'Shuffle history while preserving marginals',
                  'parameter_learning': False, 'resource_matching': 'Same capacity, samples and evaluation stream'}],
        'metrics': [{'name': 'error', 'definition': 'Fraction of incorrect responses', 'unit': 'proportion',
                     'aggregation': 'Per-seed paired mean', 'evaluation_window': 'After five permitted feedback observations',
                     'primary': True}],
        'comparisons': [{'hypothesis_id': hid, 'arm_names': ['normal', 'shuffle'], 'metric_names': ['error'],
                         'predicted_difference': 'Normal history reduces error by at least 0.05',
                         'falsification_condition': 'Matched errors fall inside the predeclared +/-0.05 equivalence band'}],
        'sample_unit': 'independent episode', 'samples_per_seed': 10, 'seeds': inputs['seeds'],
        'sample_rationale': 'Exploratory pilot; estimate variance before a powered confirmation study.',
        'statistical_analysis': 'Use paired per-seed differences and report uncertainty without claiming power.',
        'confound_controls': ['Freeze all parameter updates during evaluation'],
        'implementation_steps': ['Implement task generator and permitted-input interface', 'Implement controls and metrics'],
        'limitations': ['Synthetic local test does not establish general adaptive success'],
        'estimated_cost': 60, 'cost_basis': 'Three seeds x ten episodes x two arms; abstract pilot estimate, not measured time.',
        'scores': {'discriminability': .8, 'falsifiability': .8, 'feasibility': .8, 'scientific_importance': .6, 'cost': .1},
        'human_summary': '暂停学习，比较正常与打乱历史，判断反馈历史是否影响行为。'}


class DesignFactory(FixtureFactory):
    def __init__(self):
        super().__init__()
        self.design_mutate = None

    def __call__(self, settings, *, role):
        if role != 'experiment_design':
            return super().__call__(settings, role=role)
        owner = self
        class Backend:
            def complete(self, *, instructions, prompt):
                inputs = json.loads(prompt)['input']
                owner.calls.append((role, inputs))
                data = {'candidates': [draft(inputs)], 'confirmation_items': []}
                if owner.design_mutate:
                    owner.design_mutate(data)
                return ReasoningResponse(json.dumps(data), {'model': settings.model, 'role': role})
        return Backend()


@pytest.fixture
def prepared(tmp_path):
    factory = DesignFactory()
    pipeline = LiteraturePipeline(ReasoningConfig(model='fixture', validation_retries=0),
        LiteratureConfig(max_queries=1, max_candidates=2, max_selected_papers=1,
                         max_revision_rounds=0, max_supplement_rounds=0),
        tmp_path/'artifacts', backend_factory=factory, tools=FakeTools(), emit=lambda _: None)
    store = ResearchStore(tmp_path/'research.sqlite3')
    state = research_literature(state=confirmed_state(), pipeline=pipeline, store=store,
                               source_root=ROOT, artifact_root=pipeline.artifact_root, emit=lambda _: None)
    state = reviewed_hypotheses(state=state, pipeline=HypothesisPipeline(pipeline), store=store,
                               source_root=ROOT, artifact_root=pipeline.artifact_root, emit=lambda _: None)
    return state, pipeline, store, factory


def design(prepared):
    state, pipeline, store, _ = prepared
    return design_experiments(state=state, pipeline=pipeline, config=ExperimentDesignConfig(),
        store=store, source_root=ROOT, artifact_root=pipeline.artifact_root, emit=lambda _: None)


def test_plan_commits_without_executing_or_creating_scientific_evidence(prepared):
    initial, pipeline, store, factory = prepared
    final = design(prepared)
    assert final.stage == Stage.EXPERIMENT and len(final.experiments) == 1
    assert final.results == initial.results == ()
    assert final.evidence == initial.evidence and final.hypotheses == initial.hypotheses
    assert final.budget == initial.budget
    assert final.experiments[0].environment['kind'] == 'research_proposal'
    assert final.experiments[0].environment['execution_status'] == 'not_implemented'
    assert factory.calls[-1][0] == 'experiment_design'
    with store.connect() as db:
        run_id = db.execute('SELECT id FROM runs ORDER BY rowid DESC').fetchone()[0]
    store.verify_artifacts(run_id)
    assert store.load(run_id) == final
    folder = pipeline.artifact_root/run_id
    assert (folder/'confirmation.md').exists() and (folder/'report.md').exists()


def test_explicit_target_selects_only_current_ready_hypothesis(prepared):
    state,pipeline,store,factory = prepared
    hid = ready_hypotheses(state)[0].id
    final = design_experiments(state=state,pipeline=pipeline,
        config=ExperimentDesignConfig(target_hypothesis_ids=(hid,)),store=store,
        source_root=ROOT,artifact_root=pipeline.artifact_root,emit=lambda _:None)
    assert factory.calls[-1][1]['allowed_hypothesis_ids'] == [hid]
    assert final.experiments[-1].hypotheses_tested == (hid,)


@pytest.mark.parametrize('targets',[('H_missing',),('H001','H001')])
def test_unknown_or_repeated_explicit_design_targets_never_call_model(prepared,targets):
    state,pipeline,store,factory = prepared
    calls = len(factory.calls)
    with pytest.raises(ValueError,match='distinct current ready'):
        design_experiments(state=state,pipeline=pipeline,
            config=ExperimentDesignConfig(target_hypothesis_ids=targets),store=store,
            source_root=ROOT,artifact_root=pipeline.artifact_root,emit=lambda _:None)
    assert len(factory.calls)==calls


def test_latest_review_controls_eligibility_instead_of_older_ready_reviews(prepared):
    state, pipeline, store, factory = prepared
    record = state.hypothesis_reviews[-1].model_dump(mode='json')
    record['id'] = 'HR_new'; record['round'] += 1
    for item in record['batch']['reviews']:
        item['verdict'] = 'defer'
    from scientist.literature_schemas import HypothesisReviewRecord
    blocked = state.revised(hypothesis_reviews=state.hypothesis_reviews + (HypothesisReviewRecord.model_validate(record),))
    assert not ready_hypotheses(blocked)
    calls = len(factory.calls)
    with pytest.raises(ValueError, match='latest ready'):
        design((blocked, pipeline, store, factory))
    assert len(factory.calls) == calls


def test_superseded_or_already_testing_versions_are_not_design_targets(prepared):
    state = prepared[0]
    first = state.hypotheses[0]
    child = first.model_copy(update={'id': 'H_child', 'parent_hypotheses': (first.id,)})
    revised = state.revised(hypotheses=state.hypotheses + (child,))
    assert first.id not in {h.id for h in ready_hypotheses(revised)}
    assert child.id not in {h.id for h in ready_hypotheses(revised)}
    testing = state.revised(hypotheses=tuple(h.model_copy(update={'status': 'testing'}) for h in state.hypotheses))
    assert not ready_hypotheses(testing)


@pytest.mark.parametrize('damage', ['unready', 'over_budget', 'wrong_seeds', 'duplicate_arms',
                                  'unknown_measurement', 'oracle_input', 'no_primary', 'long_confirmation'])
def test_invalid_proposals_never_commit(prepared, damage):
    state, pipeline, store, factory = prepared
    def corrupt(batch):
        p = batch['candidates'][0]
        if damage == 'unready':
            p['hypotheses_tested'] = ['H_unknown']; p['comparisons'][0]['hypothesis_id'] = 'H_unknown'
        elif damage == 'over_budget': p['estimated_cost'] = 1001
        elif damage == 'wrong_seeds': p['seeds'] = [1, 2, 3]
        elif damage == 'duplicate_arms': p['arms'][1]['name'] = p['arms'][0]['name']
        elif damage == 'unknown_measurement': p['comparisons'][0]['metric_names'] = ['invented']
        elif damage == 'oracle_input': p['task']['permitted_agent_inputs'].append('context truth')
        elif damage == 'no_primary': p['metrics'][0]['primary'] = False
        else: batch['confirmation_items'] = ['x'*101]
    factory.design_mutate = corrupt
    with store.connect() as db:
        before = db.execute('SELECT count(*) FROM runs').fetchone()[0]
    with pytest.raises(ValueError, match='failed validation'):
        design(prepared)
    with store.connect() as db:
        assert db.execute('SELECT count(*) FROM runs').fetchone()[0] == before
    assert state.experiments == state.results == ()
    failure = next(pipeline.artifact_root.glob('failed_design_*/failure.json'))
    assert json.loads(failure.read_text())['attempts'][-1]['accepted'] is False


def test_changed_original_document_is_rejected_before_design_model_call(prepared):
    state, pipeline, store, factory = prepared
    with store.connect() as db:
        directory = db.execute('SELECT artifact_dir FROM runs ORDER BY rowid DESC').fetchone()[0]
    from pathlib import Path
    path = Path(directory)/'document_P_test.json'
    path.write_text('{}')
    calls = len(factory.calls)
    with pytest.raises(ValueError, match='checksum'):
        design(prepared)
    assert len(factory.calls) == calls


def test_file_edit_cannot_reuse_a_ready_verdict_for_an_altered_claim(prepared):
    state, pipeline, store, factory = prepared
    altered = state.hypotheses[0].model_copy(update={'claim': 'An altered claim which has never been reviewed'})
    forged = state.revised(hypotheses=(altered,) + state.hypotheses[1:])
    calls = len(factory.calls)
    with pytest.raises(ValueError, match='unchanged committed research state'):
        design((forged, pipeline, store, factory))
    assert len(factory.calls) == calls


def test_successful_design_cache_reuses_model_response_and_revalidates(prepared):
    first = design(prepared)
    factory = prepared[-1]
    count = sum(role == 'experiment_design' for role, _ in factory.calls)
    assert design(prepared) == first
    assert sum(role == 'experiment_design' for role, _ in factory.calls) == count


def test_implemented_design_cannot_be_sent_to_toy_runner(prepared):
    from experiments.runner import ToyExperimentRunner
    plan = design(prepared).experiments[0]
    with pytest.raises(ValueError):
        ToyExperimentRunner().execute(plan, None)


def test_design_cli_requires_existing_state():
    with pytest.raises(SystemExit):
        main(['--design-experiments'])


@pytest.mark.parametrize('seeds', [(), (1, 2), (1, 1, 3)])
def test_pilot_seeds_must_be_reproducible_and_distinct(seeds):
    with pytest.raises(ValueError):
        ExperimentDesignConfig(seeds=seeds)
