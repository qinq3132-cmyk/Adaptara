"""Resume the exact screened problem without another search or screening call."""
import json

import pytest

from run_scientist import ROOT, main
from scientist.hypothesis_generation import ReasoningConfig
from scientist.literature_pipeline import LiteratureConfig, LiteraturePipeline, research_literature
from scientist.tools.experiment_tracker import ResearchStore
from scientist.tools.literature import digest
from test_literature_agents import FakeTools, FixtureFactory, confirmed_state


def prepared(tmp_path):
    factory = FixtureFactory()
    pipeline = LiteraturePipeline(ReasoningConfig(model='fixture', validation_retries=0),
                                  LiteratureConfig(max_queries=1, max_candidates=2, max_selected_papers=1),
                                  tmp_path/'artifacts', backend_factory=factory, tools=FakeTools(), emit=lambda _: None)
    state = confirmed_state()
    pipeline.discover(state)
    path = pipeline.save_discovery_checkpoint(state, tmp_path/'discovery.json')
    return pipeline, factory, state, path


def test_resume_reads_and_verifies_without_new_search_and_can_load_original_snapshots(tmp_path):
    from scientist.hypothesis_pipeline import load_source_documents
    pipeline, factory, state, path = prepared(tmp_path)
    calls = len(factory.calls)
    pipeline.tools.search = lambda *a, **k: pytest.fail('Resume must not search again')
    store = ResearchStore(tmp_path/'research.sqlite3')
    final = research_literature(state=state, pipeline=pipeline, store=store, source_root=ROOT,
                               artifact_root=pipeline.artifact_root, discovery_checkpoint=path, emit=lambda _: None)
    assert [r for r, _ in factory.calls[calls:]] == ['literature_reading', 'evidence_verification', 'literature_synthesis']
    assert len(final.evidence) == 1 and final.evidence[0].verification == 'accepted'
    assert pipeline.exports['discovery_resume.json']['reused_screening']
    assert pipeline.attempts[0]['reused']
    assert load_source_documents(final, store)['P_test'].read_level == 'full_text'
    assert pipeline.exports['reading_P_test.json']['findings']
    assert pipeline.exports['verification_P_test.json']['checks'][0]['verdict'] == 'accepted'


def test_changed_checkpoint_is_rejected_before_model_invocation(tmp_path):
    pipeline, factory, state, path = prepared(tmp_path)
    saved=json.loads(path.read_text())
    saved['payload']['exports']['search_results.json'][0]['abstract'] = 'Changed scientific content'
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='checksum'):
        pipeline.run(state, discovery_checkpoint=path)
    assert len(factory.calls) == 2


def test_checkpoint_from_other_state_or_budget_is_rejected(tmp_path):
    pipeline, factory, state, path = prepared(tmp_path)
    changed = state.revised(budget=state.budget.model_copy(update={'max_compute_units': 99}))
    with pytest.raises(ValueError, match='different research definition or input state'):
        pipeline.run(changed, discovery_checkpoint=path)
    assert len(factory.calls) == 2


@pytest.mark.parametrize('damage', ['duplicate', 'unknown', 'summary', 'unrelated'])
def test_even_checksummed_inconsistent_checkpoints_are_rejected(tmp_path, damage):
    pipeline, factory, state, path = prepared(tmp_path)
    saved=json.loads(path.read_text())
    exports=saved['payload']['exports']
    if damage == 'duplicate':
        exports['search_results.json'].append(exports['search_results.json'][0])
    elif damage == 'unknown':
        exports['selection.json']['decisions'][0]['paper_id'] = 'P_unknown'
    elif damage == 'summary':
        exports['retrieval_summary.json']['selected_papers'] = 0
    else:
        exports['selection.json']['decisions'][0]['category'] = 'unrelated'
    saved['sha256']=digest(saved['payload'])
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        pipeline.run(state, discovery_checkpoint=path)
    assert len(factory.calls) == 2


def test_checkpoint_configuration_and_supplement_cannot_be_silently_replaced(tmp_path):
    pipeline, factory, state, path = prepared(tmp_path)
    pipeline.config = pipeline.config.model_copy(update={'max_selected_papers': 2})
    with pytest.raises(ValueError, match='recorded literature configuration'):
        pipeline.run(state, discovery_checkpoint=path)
    with pytest.raises(ValueError, match='not both'):
        pipeline.run(state, discovery_checkpoint=path, queries=())
    assert len(factory.calls) == 2


def test_resume_allows_configuring_later_roles_without_changing_screening(tmp_path):
    from scientist.literature_pipeline import AgentSettings
    pipeline, factory, state, path = prepared(tmp_path)
    pipeline.config = pipeline.config.model_copy(update={'roles': {'evidence_verification': AgentSettings(model='other-fixture')},
                                                        'http_timeout_seconds': 30})
    pipeline.run(state, discovery_checkpoint=path)
    verification = next(a for a in pipeline.attempts if a['role'] == 'evidence_verification')
    assert verification['runtime']['model'] == 'other-fixture'
    assert pipeline.exports['discovery_resume.json']['literature']['roles'] == {}
    assert [r for r, _ in factory.calls].count('literature_search') == 1


def test_checkpoint_is_immutable(tmp_path):
    pipeline, _, state, path = prepared(tmp_path)
    pipeline.save_discovery_checkpoint(state, path)
    pipeline.exports['selection_budget.json']['note'] = 'Different checkpoint'
    with pytest.raises(ValueError, match='overwrite'):
        pipeline.save_discovery_checkpoint(state, path)


def test_normal_run_saves_a_portable_screening_checkpoint_before_reading(tmp_path):
    pipeline, factory, state, _ = prepared(tmp_path)
    def fail_reading(role, data, inputs):
        if role == 'literature_reading':
            data['findings'][0]['quote'] = 'Invented quote absent from this paper'
    factory.mutate = fail_reading
    with pytest.raises(ValueError, match='failed validation'):
        pipeline.run(state)
    paths=list((pipeline.artifact_root/'literature_discoveries').glob('*.json'))
    assert len(paths) == 1
    saved=json.loads(paths[0].read_text())
    assert saved['payload']['stage'] == 'literature_screening'
    assert saved['sha256'] == digest(saved['payload'])
    assert pipeline.attempts[-1]['role'] == 'literature_reading' and not pipeline.attempts[-1]['accepted']


def test_validation_retry_receives_previous_output_and_specific_quote_location(tmp_path):
    pipeline, factory, state, path = prepared(tmp_path)
    pipeline.common = pipeline.common.model_copy(update={'validation_retries': 1})
    def damage_first(role, data, inputs):
        if role == 'literature_reading' and sum(r == role for r, _ in factory.calls) == 1:
            data['findings'][0]['quote'] = 'A changed quotation absent from the original section'
    factory.mutate = damage_first
    pipeline.run(state, discovery_checkpoint=path)
    attempts = [a for a in pipeline.attempts if a['role'] == 'literature_reading']
    retry = json.loads(attempts[1]['prompt'])
    assert retry['previous_response'] == attempts[0]['response_text']
    assert 'Finding 1, S001' in retry['validation_feedback']
    assert 'repeated numbers' in retry['validation_feedback']
    assert retry['validation_task']
    assert attempts[1]['accepted']


def test_retry_reports_all_quote_locations_and_strength_errors_together(tmp_path):
    pipeline, factory, state, path = prepared(tmp_path)
    pipeline.common = pipeline.common.model_copy(update={'validation_retries': 1})
    pipeline.tools.level = 'excerpt'
    def damage_first(role, data, inputs):
        if role == 'literature_reading' and sum(r == role for r, _ in factory.calls) == 1:
            data['findings'][0].update(section_id='S_wrong', strength='strong')
            second=dict(data['findings'][0])
            second.update(section_id='S001', quote='This quote does not appear anywhere in the source')
            data['findings'].append(second)
    factory.mutate = damage_first
    pipeline.run(state, discovery_checkpoint=path)
    attempts=[a for a in pipeline.attempts if a['role'] == 'literature_reading']
    feedback=json.loads(attempts[1]['prompt'])['validation_feedback']
    assert 'Finding 1 references unknown section_id S_wrong' in feedback
    assert 'Finding 1: excerpt evidence must be weak' in feedback
    assert 'Finding 2, S001: quoted evidence' in feedback
    assert 'Finding 2: excerpt evidence must be weak' in feedback


def test_retry_supplies_original_source_context_without_accepting_edited_quotes(tmp_path):
    pipeline, factory, state, path = prepared(tmp_path)
    pipeline.common = pipeline.common.model_copy(update={'validation_retries': 1})
    def damage_first(role, data, inputs):
        if role == 'literature_reading' and sum(r == role for r, _ in factory.calls) == 1:
            data['findings'][0]['quote'] = 'Altered prefix ' + data['findings'][0]['quote']
    factory.mutate = damage_first
    pipeline.run(state, discovery_checkpoint=path)
    attempts = [a for a in pipeline.attempts if a['role'] == 'literature_reading']
    feedback = json.loads(attempts[1]['prompt'])['validation_feedback']
    original = json.loads(attempts[0]['prompt'])['input']['document']['sections'][0]['text']
    assert 'Nearby ORIGINAL source text' in feedback
    assert original in feedback
    assert not attempts[0]['accepted'] and attempts[1]['accepted']


def test_quote_feedback_keeps_the_ligature_at_the_end_of_a_long_span(tmp_path):
    from scientist.literature_schemas import DocumentSection
    from test_literature_agents import QUOTE
    pipeline, factory, state, path = prepared(tmp_path)
    pipeline.common = pipeline.common.model_copy(update={'validation_retries': 1})
    original = QUOTE * 5 + ' The signiﬁcance of the adaptation mechanism.'
    fetch = pipeline.tools.fetch_document
    def long_document(candidate, **kwargs):
        sections = (DocumentSection(id='S001', heading='Results', text=original),)
        return fetch(candidate, **kwargs).model_copy(update={'sections': sections,
            'content_sha256': digest([s.model_dump(mode='json') for s in sections])})
    pipeline.tools.fetch_document = long_document
    def change_ligature(role, data, inputs):
        if role == 'literature_reading':
            first = sum(r == role for r, _ in factory.calls) == 1
            data['findings'][0]['quote'] = original.replace('signiﬁcance', 'significance') if first else original
    factory.mutate = change_ligature
    pipeline.run(state, discovery_checkpoint=path)
    attempts = [a for a in pipeline.attempts if a['role'] == 'literature_reading']
    feedback = json.loads(attempts[1]['prompt'])['validation_feedback']
    assert original in feedback and 'signiﬁcance' in feedback
    assert not attempts[0]['accepted'] and attempts[1]['accepted']


@pytest.mark.parametrize('mode', ['--demo', '--generate-hypotheses', '--review-hypotheses'])
def test_cli_rejects_resume_in_unrelated_modes(mode):
    with pytest.raises(SystemExit):
        main([mode, '--literature-discovery', 'discovery.json'])


@pytest.mark.parametrize('code,failures,expected_calls,success,transport_retries', [
    ('timeout', 1, 2, True, '1'), ('timeout', 5, 6, True, '5'),
    ('timeout', 6, 6, False, '5'), ('provider_auth_failed', 1, 1, False, '5'),
])
def test_transient_invocation_retry_is_bounded_and_preserves_screening_cache(tmp_path, code, failures, expected_calls, success, transport_retries, monkeypatch):
    from scientist.backends.codex import CodexError
    monkeypatch.setenv('SCIENTIST_TRANSPORT_BACKOFF_SECONDS', '0')  # no real sleeping
    pipeline, factory, state, path = prepared(tmp_path)
    calls = []
    def flaky(settings, *, role):
        backend = factory(settings, role=role)
        class Backend:
            def complete(self, **kwargs):
                if role == 'literature_reading':
                    calls.append(kwargs['prompt'])
                    if len(calls) <= failures:
                        raise CodexError(code, 'Synthetic invocation failure')
                return backend.complete(**kwargs)
        return Backend()
    pipeline.backend_factory = flaky
    if success:
        final = pipeline.run(state, discovery_checkpoint=path)
        assert len(final.evidence) == 1
    else:
        with pytest.raises(CodexError):
            pipeline.run(state, discovery_checkpoint=path)
        assert not state.evidence
    assert len(calls) == expected_calls
    assert all(prompt == calls[0] for prompt in calls)
    assert [r for r, _ in factory.calls].count('literature_search') == 1
    records = [json.loads(p.read_text()) for p in (pipeline.artifact_root / 'research_agent_cache').glob('*/transport_failed_*.json')]
    assert len(records) == min(failures, expected_calls)
    assert all(r['error_code'] == code for r in records)
