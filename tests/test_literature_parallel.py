"""Concurrent paper sessions must preserve acquisition pacing and scientific provenance."""
import json
from collections import Counter
from threading import Barrier, Event, Lock, get_ident

import pytest

from scientist.backends.codex import CodexError
from scientist.hypothesis_generation import ReasoningConfig
from scientist.literature_pipeline import AgentSettings, LiteratureConfig, LiteraturePipeline
from test_literature_agents import FakeTools, FixtureFactory, confirmed_state


class ParallelFactory(FixtureFactory):
    def __init__(self, overlap=True, fail=False):
        super().__init__()
        self.overlap, self.fail = overlap, fail
        self.barrier, self.second_done, self.lock = Barrier(2), Event(), Lock()
        self.active, self.maximum, self.reads = 0, 0, Counter()

    def response(self, role, inputs):
        data = super().response(role, inputs)
        if role == 'literature_screening':
            for decision in data['decisions']:
                decision.update(selected=True, category='direct')
        return data

    def __call__(self, settings, *, role):
        backend = super().__call__(settings, role=role)
        owner = self
        class Backend:
            def complete(self, **kwargs):
                if role != 'literature_reading':
                    return backend.complete(**kwargs)
                pid = json.loads(kwargs['prompt'])['input']['paper']['id']
                with owner.lock:
                    owner.active += 1
                    owner.maximum = max(owner.maximum, owner.active)
                    owner.reads[pid] += 1
                try:
                    if owner.overlap:
                        owner.barrier.wait(timeout=3)
                        if pid == 'P_test':
                            assert owner.second_done.wait(timeout=3)
                    if owner.fail and pid == 'P_test':
                        raise CodexError('provider_auth_failed', 'Synthetic failure')
                    return backend.complete(**kwargs)
                finally:
                    if pid == 'P_other':
                        owner.second_done.set()
                    with owner.lock:
                        owner.active -= 1
        return Backend()


def prepared(tmp_path, concurrency=2, *, fail=False):
    factory = ParallelFactory(overlap=concurrency > 1, fail=fail)
    tools = FakeTools()
    tools.fetch_threads = []
    original_fetch = tools.fetch_document
    def fetch(candidate, **kwargs):
        tools.fetch_threads.append(get_ident())
        return original_fetch(candidate, **kwargs)
    tools.fetch_document = fetch
    config = LiteratureConfig(max_queries=1, max_candidates=2, max_selected_papers=2,
                              min_selected_papers=2, reading_concurrency=concurrency,
                              roles={'literature_reading': AgentSettings(model='glm-5.3-flash'),
                                     'evidence_verification': AgentSettings(model='glm-5.3')})
    pipeline = LiteraturePipeline(ReasoningConfig(model='fixture', validation_retries=0), config,
                                  tmp_path, backend_factory=factory, tools=tools, emit=lambda _: None)
    state = confirmed_state()
    pipeline.discover(state)
    path = pipeline.save_discovery_checkpoint(state)
    return pipeline, factory, state, path


def test_parallel_reading_overlaps_but_merges_in_input_order_and_keeps_acquisition_serial(tmp_path):
    main_thread = get_ident()
    pipeline, factory, state, path = prepared(tmp_path)
    final = pipeline.run(state, discovery_checkpoint=path)
    assert factory.maximum == 2 and factory.active == 0
    assert pipeline.tools.fetch_threads == [main_thread, main_thread]
    assert [p.id for p in final.papers] == ['P_test', 'P_other']
    assert [e.source_id for e in final.evidence] == ['P_test', 'P_other']
    assert final.evidence[0].id != final.evidence[1].id
    assert all(e.verification == 'accepted' for e in final.evidence)
    assert [a['paper_id'] for a in pipeline.exports['extraction_audit.json']] == ['P_test', 'P_other']
    assert {a['runtime']['model'] for a in pipeline.attempts if a['role'] == 'literature_reading'} == {'glm-5.3-flash'}
    assert {a['runtime']['model'] for a in pipeline.attempts if a['role'] == 'evidence_verification'} == {'glm-5.3'}
    assert pipeline.exports['paper_execution.json']['reading_concurrency'] == 2


def test_serial_and_parallel_runs_produce_the_same_scientific_state(tmp_path):
    parallel, _, state, path = prepared(tmp_path/'parallel')
    serial, serial_factory, _, serial_path = prepared(tmp_path/'serial', concurrency=1)
    assert parallel.run(state, discovery_checkpoint=path) == serial.run(state, discovery_checkpoint=serial_path)
    assert serial_factory.maximum == 1


def test_one_failed_job_does_not_corrupt_other_caches_or_commit_partial_science(tmp_path):
    pipeline, factory, state, path = prepared(tmp_path, fail=True)
    with pytest.raises(CodexError):
        pipeline.run(state, discovery_checkpoint=path)
    assert not state.papers and not state.evidence
    audits = pipeline.exports['extraction_audit.json']
    assert audits[0]['status'] == 'failed'
    assert audits[1]['checks']['checks'][0]['verdict'] == 'accepted'
    assert 'reading_P_other.json' in pipeline.exports
    # Recover with serial execution; the completed paper must not call the model again.
    factory.fail, factory.overlap = False, False
    pipeline.config = pipeline.config.model_copy(update={'reading_concurrency': 1})
    final = pipeline.run(state, discovery_checkpoint=path)
    assert len(final.papers) == len(final.evidence) == 2
    assert factory.reads == {'P_test': 2, 'P_other': 1}


def test_rejected_first_finding_does_not_discard_later_accepted_findings(tmp_path):
    factory = FixtureFactory()
    def mixed(role, data, inputs):
        if role == 'literature_reading':
            data['findings'].append({**data['findings'][0], 'claim': 'A later supported finding'})
        if role == 'evidence_verification':
            data['checks'][0]['verdict'] = 'rejected'
    factory.mutate = mixed
    tools = FakeTools()
    pipeline = LiteraturePipeline(ReasoningConfig(model='fixture', validation_retries=0), LiteratureConfig(),
                                  tmp_path, backend_factory=factory, tools=tools, emit=lambda _: None)
    result = pipeline.read_paper(tools.paper, tools.fetch_document(tools.paper),
                                 confirmed_state().problem_definition.model_dump(mode='json'))
    assert result.error is None
    assert [e.claim for e in result.evidence] == ['A later supported finding']


@pytest.mark.parametrize('concurrency', [0, 9])
def test_parallelism_has_a_bounded_configurable_limit(concurrency):
    with pytest.raises(ValueError):
        LiteratureConfig(reading_concurrency=concurrency)
