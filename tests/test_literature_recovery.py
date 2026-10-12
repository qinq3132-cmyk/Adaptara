"""Paging, independent screening, evidence coverage, and exact metadata repair."""
import json
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from scientist.hypothesis_generation import ReasoningConfig
from scientist.literature_pipeline import LiteratureConfig, LiteraturePipeline
from scientist.literature_schemas import PaperCandidate, SearchQuery
from scientist.tools.literature import LiteratureTools
from test_literature_agents import FakeTools, FixtureFactory, QUOTE, confirmed_state


QUERY = SearchQuery(text="adaptation mechanisms", purpose="overview", discipline="Learning", rationale="Compare mechanisms")


def page_body(provider, numbers, more=False):
    if provider == 'arxiv':
        entries = ''.join(f'<entry><id>https://arxiv.org/abs/2401.0000{n}</id><title>Paper {n}</title>'
                          f'<summary>Abstract {n}</summary><published>2024-01-01</published></entry>' for n in numbers)
        return (f'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:o="http://a9.com/-/spec/opensearch/1.1/">'
                f'<o:totalResults>3</o:totalResults>{entries}</feed>').encode(), 'application/xml'
    records = [{'title': f'Paper {n}', 'id': str(n), 'DOI': f'10.1234/{n}'} for n in numbers]
    if provider == 'europe_pmc':
        payload = {'resultList': {'result': [{'id': p['id'], 'title': p['title'], 'doi': p['DOI']} for p in records]}}
        if more:
            payload['nextCursorMark'] = 'NEXT2'
    elif provider == 'openalex':
        payload = {'results': [{'id': 'https://openalex.org/W' + p['id'], 'title': p['title'], 'doi': p['DOI']} for p in records],
                   'meta': {'count': 3}}
    elif provider == 'crossref':
        payload = {'message': {'items': [{'title': [p['title']], 'DOI': p['DOI']} for p in records], 'total-results': 3}}
    else:
        payload = {'data': [{'paperId': p['id'], 'title': p['title'], 'abstract': 'Reported adaptation',
                            'externalIds': {'DOI': p['DOI']}} for p in records]}
        if more:
            payload['next'] = 2
    return json.dumps(payload).encode(), 'application/json'


@pytest.mark.parametrize('provider,parameter,value', [
    ('semantic_scholar', 'offset', '2'), ('arxiv', 'start', '2'), ('crossref', 'offset', '2'),
    ('europe_pmc', 'cursorMark', 'NEXT2'), ('openalex', 'page', '2')])
def test_each_provider_advances_pages_and_stops_when_exhausted(provider, parameter, value, monkeypatch):
    monkeypatch.setenv('SEMANTIC_SCHOLAR_API_KEY', 'fixture-secret')
    class HTTP:
        calls = []
        def get(self, url, headers=None):
            self.calls.append((url, headers))
            page = len(self.calls)
            assert page <= 2
            if page == 2:
                assert parse_qs(urlsplit(url).query)[parameter] == [value]
            return page_body(provider, [0, 1] if page == 1 else [2], more=page == 1)
    http = HTTP()
    tools = LiteratureTools(http)
    papers = tools.search((QUERY,), (provider,), per_query=2, max_pages=5)
    assert len(papers) == 3 and len(http.calls) == 2
    assert [e['page'] for e in tools.events] == [1, 2]
    if provider == 'semantic_scholar':
        assert all(headers == {'x-api-key': 'fixture-secret'} for _, headers in http.calls)
        assert 'fixture-secret' not in json.dumps(tools.events)


def test_later_page_rate_limit_preserves_first_page_and_uses_other_sources():
    class HTTP:
        calls = []
        def get(self, url, headers=None):
            self.calls.append(url)
            if 'semanticscholar.org' in url:
                if 'offset=2' in url:
                    raise RuntimeError('Literature HTTP request failed (status 429)')
                return page_body('semantic_scholar', [0, 1], more=True)
            return page_body('crossref', [2])
    tools = LiteratureTools(HTTP())
    other = QUERY.model_copy(update={'text': 'different mechanism'})
    papers = tools.search((QUERY, other), ('semantic_scholar', 'crossref'), per_query=2, max_pages=3)
    assert len(papers) == 3
    assert any(e['status'] == 'failed' and e['page'] == 2 for e in tools.events)
    assert any(e['status'] == 'skipped' and e['provider'] == 'semantic_scholar' for e in tools.events)
    assert any(e['provider'] == 'crossref' and e['status'] == 'ok' for e in tools.events)


def test_repeated_page_stops_and_old_candidates_do_not_consume_new_budget():
    class HTTP:
        calls = 0
        def get(self, url, headers=None):
            self.calls += 1
            return page_body('crossref', [0, 1], more=True)
    http = HTTP()
    tools = LiteratureTools(http)
    old = tools.search((QUERY,), ('crossref',), per_query=2, max_pages=1)[0]
    new = tools.search((QUERY,), ('crossref',), per_query=2, max_candidates=1, max_pages=5, exclude_ids={old.id})
    assert len(new) == 1 and new[0].id != old.id
    assert http.calls == 3  # Initial call plus two identical pages, then stop.
    assert tools.events[-1]['new_records'] == 0


def candidate(n, abstract=QUOTE):
    return PaperCandidate(id=f'P_{n}', title=f'Mechanism {n}', abstract=abstract,
                          doi=f'10.1234/{n}', source_uri=f'https://example.org/{n}')


class RoundTools(FakeTools):
    def __init__(self, rounds):
        super().__init__()
        self.rounds = rounds

    def search(self, queries, providers, **kwargs):
        index = len(self.search_calls)
        self.search_calls.append(queries)
        self.events.append({'operation': 'search', 'provider': 'crossref', 'status': 'ok', 'query': queries[0].text})
        return self.rounds[min(index, len(self.rounds) - 1)]


class CoverageFactory(FixtureFactory):
    def __init__(self, categories):
        super().__init__()
        self.categories = categories

    def response(self, role, inputs):
        data = super().response(role, inputs)
        if role == 'literature_search':
            data['queries'][0]['text'] += ' ' + str(len(inputs['previous_queries']))
        if role == 'literature_screening':
            for decision, paper in zip(data['decisions'], inputs['candidates']):
                decision.update(selected=bool(paper['abstract']), category=self.categories.get(paper['id'], 'direct'),
                                priority=3, needs_metadata=not bool(paper['abstract']))
                if not paper['abstract']:
                    decision['category'] = 'uncertain'
        return data


def pipeline_for(tmp_path, rounds, categories=None, **options):
    base = dict(max_queries=1, max_candidates=20, max_selected_papers=4, max_search_rounds=4)
    base.update(options)
    factory = CoverageFactory(categories or {})
    pipeline = LiteraturePipeline(ReasoningConfig(model='fixture', validation_retries=0), LiteratureConfig(**base),
                                  tmp_path, backend_factory=factory, tools=RoundTools(rounds), emit=lambda _: None)
    return pipeline, factory


def test_numeric_target_does_not_stop_before_missing_direction_is_found(tmp_path):
    pipeline, factory = pipeline_for(tmp_path, [[candidate(0), candidate(1)], [candidate(2)]],
                                    {'P_2': 'boundary'}, required_categories=('direct', 'boundary'), max_selected_papers=2)
    _, selection = pipeline.discover(confirmed_state())
    assert len(pipeline.tools.search_calls) == 2
    assert pipeline.exports['search_history.json'][0]['missing_categories'] == ['boundary']
    assert {d.category for d in selection.decisions if d.selected} == {'direct', 'boundary'}
    recovery = next(i for role, i in factory.calls if role == 'literature_search' and i['phase'] == 'recovery')
    assert recovery['feedback'][0]['missing_categories'] == ['boundary']
    assert pipeline.exports['selection_budget.json']['deferred_paper_ids']
    assert all(d['selected'] for d in pipeline.exports['screening_decisions.json'])


def test_large_new_relevant_yield_continues_even_when_reading_budget_is_full(tmp_path):
    pipeline, _ = pipeline_for(tmp_path, [[candidate(0), candidate(1)], [candidate(2), candidate(3)], []],
                              max_selected_papers=2, min_search_rounds=2, saturation_rounds=1,
                              max_new_relevant_per_round=0, required_categories=('direct',))
    pipeline.discover(confirmed_state())
    history = pipeline.exports['search_history.json']
    assert [r['selected_count'] for r in history] == [2, 2, 2]
    assert [r['new_relevant_candidates'] for r in history] == [2, 2, 0]
    assert len(history) == 3 and pipeline.exports['retrieval_summary.json']['status'] == 'target_met'


def test_batch_screening_only_calls_for_new_candidates(tmp_path):
    first = [candidate(n) for n in range(3)]
    pipeline, factory = pipeline_for(tmp_path, [first, first + [candidate(3), candidate(4)]],
                                    min_search_rounds=2, screening_batch_size=2)
    pipeline.discover(confirmed_state())
    batches = [i['candidates'] for role, i in factory.calls if role == 'literature_screening']
    assert [len(b) for b in batches] == [2, 1, 2]
    assert sorted(p['id'] for b in batches for p in b) == [f'P_{n}' for n in range(5)]


def test_failed_sources_cannot_be_mistaken_for_search_saturation(tmp_path):
    pipeline, _ = pipeline_for(tmp_path, [[candidate(0)], []], min_search_rounds=2, max_search_rounds=2,
                              saturation_rounds=1, required_categories=('direct',))
    search = pipeline.tools.search
    def failing_search(*args, **kwargs):
        records = search(*args, **kwargs)
        if len(pipeline.tools.search_calls) == 2:
            pipeline.tools.events[-1]['status'] = 'failed'
        return records
    pipeline.tools.search = failing_search
    pipeline.discover(confirmed_state())
    summary = pipeline.exports['retrieval_summary.json']
    assert summary['status'] == 'partial' and summary['stop_reason'] == 'search_round_budget_exhausted'
    assert pipeline.exports['search_history.json'][-1]['low_yield_rounds'] == 0


def test_pending_metadata_is_repaired_and_rescreened_before_reading(tmp_path):
    pipeline, factory = pipeline_for(tmp_path, [[candidate(0, '')]], max_metadata_requests=1,
                                    required_categories=('direct',))
    def repair(papers, providers, max_requests):
        assert max_requests == 1
        return [p.model_copy(update={'abstract': QUOTE}) for p in papers], 1
    pipeline.tools.enrich_candidates = repair
    pool, selection = pipeline.discover(confirmed_state())
    assert pool[0].abstract == QUOTE and selection.decisions[0].selected
    assert [role for role, _ in factory.calls].count('literature_screening') == 2
    assert pipeline.exports['retrieval_summary.json']['pending_metadata_ids'] == []
    assert pipeline.exports['retrieval_summary.json']['metadata_requests'] == 1


def test_title_only_candidate_remains_pending_and_does_not_satisfy_coverage(tmp_path):
    pipeline, _ = pipeline_for(tmp_path, [[candidate(0, '')]], max_search_rounds=1, required_categories=('direct',))
    pipeline.discover(confirmed_state())
    summary = pipeline.exports['retrieval_summary.json']
    assert summary['status'] == 'partial' and summary['pending_metadata_ids'] == ['P_0']
    assert summary['missing_categories'] == ['direct']
    assert not pipeline.exports['screening_decisions.json'][0]['selected']


def test_screened_direction_without_verified_evidence_is_partial(tmp_path):
    pipeline, factory = pipeline_for(tmp_path, [[candidate(0), candidate(1)]], {'P_1': 'boundary'},
                                    required_categories=('direct', 'boundary'))
    def mutate(role, data, inputs):
        if role == 'literature_reading' and inputs['paper']['id'] == 'P_1':
            data['findings'] = []
    factory.mutate = mutate
    pipeline.run(confirmed_state())
    summary = pipeline.exports['retrieval_summary.json']
    assert summary['category_coverage']['boundary'] == 1
    assert summary['verified_category_coverage']['boundary'] == 0
    assert summary['status'] == 'partial' and summary['stop_reason'] == 'verified_coverage_incomplete'


@pytest.mark.parametrize('provider', ['europe_pmc', 'openalex', 'crossref'])
@pytest.mark.parametrize('matching', [False, True])
def test_exact_metadata_lookup_accepts_only_matching_doi_and_preserves_identity(provider, matching):
    class HTTP:
        calls = []
        def get(self, url, headers=None):
            self.calls.append(url)
            doi = 'https://doi.org/10.1234/0' if matching else '10.1234/wrong'
            if provider == 'europe_pmc':
                payload = {'resultList': {'result': [{'id': '123', 'title': 'Alternate index title',
                            'doi': doi, 'abstractText': QUOTE, 'pmcid': 'PMC123', 'isOpenAccess': 'Y'}]}}
            elif provider == 'openalex':
                payload = {'doi': doi, 'abstract_inverted_index': {'Recovered': [0], 'abstract': [1]},
                           'ids': {'pmid': 'https://pubmed.ncbi.nlm.nih.gov/123/', 'pmcid': 'https://ncbi.nlm.nih.gov/pmc/PMC123/'},
                           'best_oa_location': {'is_oa': True, 'pdf_url': 'https://repo.org/paper.pdf'}}
            else:
                payload = {'message': {'DOI': doi, 'abstract': '<jats:p>' + QUOTE + '</jats:p>',
                           'link': [{'content-type': 'application/pdf', 'URL': 'https://repo.org/paper.pdf'}]}}
            return json.dumps(payload).encode(), 'application/json'
    tools = LiteratureTools(HTTP())
    original = candidate(0, '')
    repaired, used = tools.enrich_candidates([original], (provider,), max_requests=1)
    assert used == 1 and repaired[0].id == original.id and repaired[0].title == original.title
    assert bool(repaired[0].abstract) == matching
    assert tools.events[0]['status'] == ('ok' if matching else 'failed')
    if matching:
        assert repaired[0].fulltext_uris
        if provider == 'openalex':
            assert repaired[0].pmid == '123' and repaired[0].pmcid == 'PMC123'


def test_metadata_request_budget_is_shared_across_papers_and_providers():
    class HTTP:
        calls = []
        def get(self, url, headers=None):
            self.calls.append(url)
            raise RuntimeError('No record')
    http = HTTP()
    tools = LiteratureTools(http)
    originals = [candidate(0, ''), candidate(1, '')]
    unchanged, used = tools.enrich_candidates(originals, ('europe_pmc', 'openalex', 'crossref'), max_requests=2)
    assert unchanged == originals and used == len(http.calls) == 2
    unchanged, used = tools.enrich_candidates(originals, ('crossref',), max_requests=0)
    assert used == 0 and len(http.calls) == 2


def test_normal_configuration_exposes_real_research_budgets():
    from run_scientist import ROOT
    config = LiteratureConfig.model_validate(yaml.safe_load((ROOT / 'config/literature.yaml').read_text()))
    assert config.max_candidates == 120 and config.max_selected_papers == 20
    assert config.max_pages_per_query == 3 and config.min_search_rounds == 2
    assert set(config.required_categories) == {'direct', 'theory', 'boundary', 'cross_discipline'}
    assert config.max_metadata_requests == 20 and config.saturation_rounds == 1


@pytest.mark.parametrize('options', [
    {'min_search_rounds': 2}, {'saturation_rounds': 2}, {'saturation_rounds': 1},
    {'required_categories': ('direct', 'direct')},
    {'max_selected_papers': 1, 'required_categories': ('direct', 'boundary')},
])
def test_impossible_stopping_requirements_are_rejected(options):
    with pytest.raises(ValueError):
        LiteratureConfig(**options)
