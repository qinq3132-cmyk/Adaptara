import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest

from run_scientist import ROOT, main
from scientist.agents.structured import StructuredAgent
from scientist.agents.hypothesis_review import HypothesisReviewAgent
from scientist.backends.common import ReasoningResponse, TOOL_FREE_ROLES
from scientist.hypothesis_generation import ReasoningConfig
from scientist.hypothesis_pipeline import HypothesisPipeline, hypothesis_suggestion_sheet, load_source_documents, reviewed_hypotheses
from scientist.literature_pipeline import LiteratureConfig, LiteraturePipeline, research_literature
from scientist.literature_schemas import DocumentSection, PaperCandidate, PaperDocument, SearchQuery
from scientist.problem_clarification import state_from_confirmed_problem
from scientist.problem_schemas import ConceptInput, ConfirmedProblem, ProblemDraft, problem_digest
from scientist.schemas import Paper, ResearchBudget
from scientist.state import ScientificState
from scientist.tools.experiment_tracker import ResearchStore
from scientist.tools.literature import ArticleHTML, LiteratureHTTP, LiteratureTools, deduplicate, digest, public_url
from test_hypothesis_generation import batch
from test_problem_clarification import draft

QUOTE = "The adaptation response recovered after a change, but only under bounded changes."


def confirmed_state():
    payload = draft(blocking=False)
    parsed = ProblemDraft.model_validate(payload)
    concept = ConceptInput(text="Study autonomous adaptation", explicit_constraints=("No task ID",))
    from scientist.schemas import ScientificQuestion
    problem = ConfirmedProblem(session_id="problem_" + "a" * 32, revision=1, original_concept=concept,
        concept=concept, draft=parsed, draft_sha256=problem_digest(parsed), confirmed_at="2026-10-11T00:00:00Z",
        confirmed_by="test", question=ScientificQuestion(id="Q001", text=parsed.root_question.text, constraints=("No task ID",)))
    return state_from_confirmed_problem(problem, ResearchBudget())


class FakeTools:
    def __init__(self, level="full_text"):
        self.events, self.level = [], level
        self.paper = PaperCandidate(id="P_test", title="Adaptation under bounded change",
             abstract=QUOTE, source_uri="https://example.org/paper", doi="10.1234/test", providers=("arxiv",))
        self.other = PaperCandidate(id="P_other", title="An irrelevant question", abstract="Unrelated",
                                   source_uri="https://example.org/other")
        self.search_calls = []

    def search(self, queries, providers, **kwargs):
        self.search_calls.append(queries)
        return [self.paper, self.other]

    def fetch_document(self, candidate, **kwargs):
        sections = (DocumentSection(id="S001", heading="Results", text=QUOTE + " An independent replication is absent."),)
        return PaperDocument(paper_id=candidate.id, source_uri=candidate.source_uri, read_level=self.level,
            sections=sections, content_sha256=digest([s.model_dump(mode="json") for s in sections]),
            fetched_at="2026-10-11T00:00:00Z", limitations=("No independent replication",))


class FixtureFactory:
    def __init__(self, mutate=None, revise=False, supplement=False):
        self.calls, self.mutate = [], mutate
        self.revise, self.supplement = revise, supplement

    def __call__(self, settings, *, role):
        owner = self
        class Backend:
            def complete(self, *, instructions, prompt):
                inputs = json.loads(prompt)["input"]
                owner.calls.append((role, inputs))
                response = owner.response(role, inputs)
                if owner.mutate:
                    owner.mutate(role, response, inputs)
                return ReasoningResponse(json.dumps(response), {"model": settings.model, "role": role})
        return Backend()

    def response(self, role, inputs):
        if role == "literature_search":
            return {"queries": [{"text": "autonomous adaptation", "purpose": "overview", "discipline": "Learning",
                                  "rationale": "Identify bounded adaptation mechanisms"}], "coverage_plan": ["Compare competing mechanisms"]}
        if role == "literature_screening":
            return {"decisions": [{"paper_id": p["id"], "selected": p["id"] == "P_test", "reason": "Assess actual relevance"}
                                  for p in inputs["candidates"]], "coverage_limitations": ["Small retrieval budget"]}
        if role == "literature_reading":
            return {"findings": [{"section_id": "S001", "quote": QUOTE, "claim": "Recovery was reported for bounded changes",
                      "conditions": ["Bounded changes"], "limitations": ["Not arbitrary change"], "strength": "weak"}],
                    "limitations": ["Limited experimental detail"]}
        if role == "evidence_verification":
            return {"checks": [{"finding_id": f["finding_id"], "verdict": "accepted", "reason": "Faithful conditional extraction"}
                               for f in inputs["findings"]]}
        if role == "literature_synthesis":
            eid = inputs["accepted_evidence"][0]["id"]
            return {"summary": "Conditional adaptation evidence; the mechanism needs further investigation",
               "claims": [{"statement": "Some bounded changes permit recovery", "source_ids": [eid], "mechanism": "Mechanism not established",
                           "assumptions": ["Bounded change"], "limitations": ["Local report only"]}],
               "gaps": [{"description": "This reviewed material does not explain autonomous regulation", "evidence_ids": [eid]}],
               "conflicts": [], "coverage_limitations": ["One relevant paper does not cover the field"]}
        if role == "hypothesis":
            data = batch()
            for h in data["hypotheses"]:
                h["source_claims"] = [inputs["scientific_context"]["knowledge"][0]["id"]]
            return data
        if role == "hypothesis_review":
            ids = inputs["target_hypothesis_ids"]
            context = inputs["scientific_context"]
            first = not context["hypothesis_reviews"]
            items = [{"hypothesis_id": hid, "verdict": "revise" if self.revise and first and i == 1 else "ready",
                      "evidence_ids": [context["evidence"][0]["id"]], "closest_paper_ids": ["P_test"],
                      "novelty": "uncertain", "rationale": "Evidence grounds premises, not the new hypothesis",
                      "issues": ["Define the intervention"], "revision_requests": ["Clarify an intervention"] if i == 1 else [],
                      "human_summary": "需要实验区分该机制。", "recommended_test": "控制相同变化，比较干预后的恢复。"}
                     for i, hid in enumerate(ids)]
            return {"reviews": items, "comparisons": [{"condition": "Same change and controlled intervention",
                     "predictions": {hid: f"Distinct measurable response {i}" for i, hid in enumerate(ids)}, "measurement": "Recovery latency"}],
                    "ranking": ids,
                    "search_requests": ([{"text": "adaptation counterexample", "purpose": "challenge", "discipline": "Learning",
                                         "rationale": "Check contrary evidence"}] if self.supplement and first else []),
                    "limitations": ["No experiment executed"]}
        if role == "hypothesis_revision":
            h = batch()["hypotheses"][1]
            h.update(mechanism="Revised intervention-specific mechanism", parent_hypotheses=[inputs["targets"][0]["id"]],
                     source_claims=[inputs["scientific_context"]["knowledge"][0]["id"]])
            return {"hypotheses": [h]}
        raise AssertionError(role)


@pytest.fixture
def setup(tmp_path):
    settings = ReasoningConfig(model="fixture", validation_retries=0)
    config = LiteratureConfig(max_queries=1, max_candidates=2, max_selected_papers=1, max_supplement_rounds=0)
    store = ResearchStore(tmp_path / "research.sqlite3")
    factory = FixtureFactory()
    pipeline = LiteraturePipeline(settings, config, tmp_path / "artifacts", backend_factory=factory,
                                  tools=FakeTools(), emit=lambda _: None)
    return store, pipeline, factory


def run_literature(setup):
    store, pipeline, _ = setup
    return research_literature(state=confirmed_state(), pipeline=pipeline, store=store, source_root=ROOT,
                               artifact_root=pipeline.artifact_root, emit=lambda _: None)


def test_grounded_literature_chain_and_hypothesis_review_persist(setup):
    store, pipeline, factory = setup
    state = run_literature(setup)
    assert state.schema_version == 3 and len(state.papers) == len(state.evidence) == len(state.knowledge) == 1
    assert state.evidence[0].quote == QUOTE and state.evidence[0].verification == "accepted"
    assert state.knowledge[0].source_ids == (state.evidence[0].id,)
    assert state.hypotheses == state.experiments == ()
    final = reviewed_hypotheses(state=state, pipeline=HypothesisPipeline(pipeline), store=store,
                              source_root=ROOT, artifact_root=pipeline.artifact_root, emit=lambda _: None)
    assert len(final.hypotheses) == 3 and len(final.hypothesis_reviews) == 1
    assert final.experiments == final.results == ()
    assert all(h.status == "proposed" and h.confidence == 0.5 for h in final.hypotheses)
    assert [r for r, _ in factory.calls] == ["literature_search", "literature_screening", "literature_reading",
                                            "evidence_verification", "literature_synthesis", "hypothesis", "hypothesis_review"]
    request = next(i for r, i in factory.calls if r == "hypothesis")
    assert request["scientific_context"]["papers"][0]["content_sha256"]
    assert request["scientific_context"]["literature"]["gaps"]
    review = next(i for r, i in factory.calls if r == "hypothesis_review")
    assert review["original_documents"][0]["sections"][0]["text"].startswith(QUOTE)
    with store.connect() as db:
        runs = db.execute("select id from runs order by rowid").fetchall()
        assert len(runs) == 2
        assert db.execute("select count(*) from hypothesis_reviews").fetchone()[0] == 1
    for run_id, in runs:
        store.verify_artifacts(run_id)
    assert store.load(runs[-1][0]) == final
    assert (pipeline.artifact_root / runs[-1][0] / "suggestions.md").exists()
    config = json.loads((pipeline.artifact_root / runs[-1][0] / 'pipeline_configuration.json').read_text())
    assert config['literature']['max_queries'] == 1 and config['reasoning']['model'] == 'fixture'


def test_identical_completed_calls_reuse_cache_without_backend(setup):
    store, pipeline, factory = setup
    state = run_literature(setup)
    first = HypothesisPipeline(pipeline).run(state, store=store, source_root=ROOT, artifact_root=pipeline.artifact_root)
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed agent should not initialize a model backend")
    pipeline.backend_factory = forbidden
    repeated_state = run_literature(setup)
    assert repeated_state == state
    second = HypothesisPipeline(pipeline).run(repeated_state, store=store, source_root=ROOT, artifact_root=pipeline.artifact_root)
    assert second == first
    assert all(a["reused"] for a in pipeline.attempts)


@pytest.mark.parametrize('legacy_identity', [False, True])
def test_successful_validation_retry_can_be_reused_without_new_model_call(setup, legacy_identity):
    _, pipeline, factory = setup
    pipeline.common = pipeline.common.model_copy(update={'validation_retries': 1})
    def mutate(role, data, inputs):
        if role == 'literature_reading' and sum(r == role for r, _ in factory.calls) == 1:
            data['findings'][0]['quote'] = 'Invented quotation absent from source material.'
    factory.mutate = mutate
    first = run_literature(setup)
    assert [role for role, _ in factory.calls].count('literature_reading') == 2
    if legacy_identity:
        for pointer in (pipeline.artifact_root / 'research_agent_cache').glob('*/accepted.json'):
            saved = json.loads(pointer.read_text())
            if saved['payload']['identity']['role'] == 'literature_reading':
                saved['payload']['identity']['request']['validation_feedback'] = 'Legacy retry feedback'
                saved['sha256'] = digest(saved['payload'])
                pointer.write_text(json.dumps(saved))
    def forbidden(*args, **kwargs):
        raise AssertionError('Accepted retry must be reusable without model initialization')
    pipeline.backend_factory = forbidden
    assert run_literature(setup) == first


def test_empty_search_recovers_using_agent_feedback(setup):
    store, pipeline, factory = setup
    pipeline.config = pipeline.config.model_copy(update={'max_search_rounds': 2, 'min_candidates': 2})
    original_search = pipeline.tools.search
    def search(queries, providers, **kwargs):
        if not pipeline.tools.search_calls:
            pipeline.tools.search_calls.append(queries)
            pipeline.tools.events.append({'provider': 'arxiv', 'query': queries[0].text, 'returned': 0, 'status': 'ok'})
            return []
        return original_search(queries, providers, **kwargs)
    pipeline.tools.search = search
    def mutate(role, data, inputs):
        if role == 'literature_search' and inputs['phase'] == 'recovery':
            assert inputs['feedback'][0]['candidate_count'] == 0
            assert inputs['previous_queries'][0]['text'] == 'autonomous adaptation'
            data['queries'][0]['text'] = 'continual learning'
    factory.mutate = mutate
    final = run_literature(setup)
    assert len(final.papers) == 1
    summary = pipeline.exports['retrieval_summary.json']
    assert summary['search_rounds'] == 2 and summary['status'] == 'target_met'
    assert summary['unique_candidates'] == 2 and summary['documents_read'] == 1
    assert [role for role, _ in factory.calls][:3] == ['literature_search', 'literature_search', 'literature_screening']


def test_recovery_does_not_repeat_failed_queries(setup):
    store, pipeline, factory = setup
    pipeline.config = pipeline.config.model_copy(update={'max_search_rounds': 2})
    pipeline.tools.search = lambda *args, **kwargs: []
    with pytest.raises(ValueError, match='failed validation'):
        run_literature(setup)
    with store.connect() as db:
        assert db.execute('select count(*) from runs').fetchone()[0] == 0


def test_all_empty_searches_stop_at_round_budget_without_committing(setup):
    store, pipeline, factory = setup
    pipeline.config = pipeline.config.model_copy(update={'max_search_rounds': 3})
    pipeline.tools.search = lambda *args, **kwargs: []
    def mutate(role, data, inputs):
        if role == 'literature_search' and inputs['phase'] == 'recovery':
            data['queries'][0]['text'] = f"recovery {len(inputs['previous_queries'])}"
    factory.mutate = mutate
    with pytest.raises(RuntimeError, match='after recovery'):
        run_literature(setup)
    assert len(factory.calls) == 3 and pipeline.exports['retrieval_summary.json']['search_rounds'] == 3
    with store.connect() as db:
        assert db.execute('select count(*) from runs').fetchone()[0] == 0


def test_selected_count_does_not_hide_shortfall_in_verified_papers(setup):
    _, pipeline, factory = setup
    pipeline.config = pipeline.config.model_copy(update={'max_selected_papers': 2, 'min_selected_papers': 2})
    def mutate(role, data, inputs):
        if role == 'literature_screening':
            for item in data['decisions']: item['selected'] = True
        if role == 'literature_reading' and inputs['paper']['id'] == 'P_other':
            data['findings'] = []
    factory.mutate = mutate
    final = run_literature(setup)
    summary = pipeline.exports['retrieval_summary.json']
    assert summary['selected_papers'] == summary['documents_read'] == 2
    assert summary['papers_with_accepted_evidence'] == 1 and summary['status'] == 'partial'
    assert summary['stop_reason'] == 'readable_evidence_target_not_met'
    assert any('可核对证据覆盖不足' in text for text in final.literature.coverage_limitations)


def test_partial_coverage_is_explicit_after_search_budget(setup):
    store, pipeline, factory = setup
    pipeline.config = pipeline.config.model_copy(update={'max_candidates': 4, 'min_candidates': 3, 'max_search_rounds': 2})
    factory.mutate = lambda role, data, inputs: data['queries'][0].update(text='continual learning') if role == 'literature_search' and inputs['phase'] == 'recovery' else None
    final = run_literature(setup)
    summary = pipeline.exports['retrieval_summary.json']
    assert summary['status'] == 'partial' and summary['stop_reason'] == 'search_round_budget_exhausted'
    assert summary['search_rounds'] == 2 and summary['unique_candidates'] == 2
    assert any('检索覆盖目标未达成' in text for text in final.literature.coverage_limitations)
    assert [role for role, _ in factory.calls].count('literature_screening') == 1


def test_candidate_budget_stops_recovery_without_filling_irrelevant_papers(setup):
    _, pipeline, factory = setup
    pipeline.config = pipeline.config.model_copy(update={'max_selected_papers': 2, 'min_selected_papers': 2, 'max_search_rounds': 3})
    _, selection = pipeline.discover(confirmed_state())
    assert sum(item.selected for item in selection.decisions) == 1
    assert pipeline.exports['retrieval_summary.json']['stop_reason'] == 'candidate_budget_exhausted'
    assert [role for role, _ in factory.calls].count('literature_search') == 1


@pytest.mark.parametrize("kind", ["invented_quote", "unknown_section", "missing_selection", "unknown_evidence", "missing_finding_check", "all_rejected"])
def test_invalid_or_unverified_literature_cannot_commit(setup, kind):
    store, pipeline, factory = setup
    def mutate(role, data, inputs):
        if kind == "invented_quote" and role == "literature_reading":
            data["findings"][0]["quote"] = "This sentence does not appear in the cited original source."
        if kind == "unknown_section" and role == "literature_reading":
            data["findings"][0]["section_id"] = "S999"
        if kind == "missing_selection" and role == "literature_screening":
            data["decisions"].pop()
        if kind == "unknown_evidence" and role == "literature_synthesis":
            data["claims"][0]["source_ids"] = ["EV_invented"]
        if kind == "missing_finding_check" and role == "evidence_verification":
            data["checks"][0]["finding_id"] = "F999"
        if kind == "all_rejected" and role == "evidence_verification":
            data["checks"][0]["verdict"] = "rejected"
    factory.mutate = mutate
    with pytest.raises((ValueError, RuntimeError)):
        run_literature(setup)
    with store.connect() as db:
        assert db.execute("select count(*) from runs").fetchone()[0] == 0


def test_failed_stage_preserves_successful_calls_for_retry(setup):
    store, pipeline, factory = setup
    factory.mutate = lambda role, data, inputs: data["findings"][0].update(quote="Invented observation not present in original material.") if role == "literature_reading" else None
    with pytest.raises(ValueError):
        run_literature(setup)
    assert len(factory.calls) == 3
    factory.mutate = None
    final = run_literature(setup)
    assert len(final.evidence) == 1
    assert [r for r, _ in factory.calls][3:] == ["literature_reading", "evidence_verification", "literature_synthesis"]


def test_cache_corruption_is_detected_without_new_call(setup):
    store, pipeline, factory = setup
    run_literature(setup)
    pointer = next((pipeline.artifact_root / "research_agent_cache").glob("*/accepted.json"))
    saved = json.loads(pointer.read_bytes())
    saved["payload"]["identity"]["role"] = "tampered"
    pointer.write_text(json.dumps(saved))
    calls = len(factory.calls)
    with pytest.raises(ValueError, match="cache"):
        run_literature(setup)
    assert len(factory.calls) == calls


@pytest.mark.parametrize("level", ["abstract", "excerpt"])
def test_partial_evidence_is_limited_and_cannot_be_strong(setup, level):
    store, pipeline, factory = setup
    pipeline.tools.level = level
    state = run_literature(setup)
    assert state.papers[0].read_level == level and state.evidence[0].strength == "weak"
    factory.mutate = lambda role, data, inputs: data["findings"][0].update(strength="strong") if role == "literature_reading" else None
    pipeline.common = pipeline.common.model_copy(update={"model": "changed-model"})
    with pytest.raises(ValueError):
        run_literature(setup)


def test_revision_preserves_parent_and_prior_review_history(setup):
    store, pipeline, factory = setup
    factory.revise = True
    state = run_literature(setup)
    final = HypothesisPipeline(pipeline).run(state, store=store, source_root=ROOT, artifact_root=pipeline.artifact_root)
    assert [h.id for h in final.hypotheses] == ["H001", "H002", "H003", "H004"]
    assert final.hypotheses[-1].parent_hypotheses == ("H002",)
    assert final.hypotheses[-1].claim == final.hypotheses[1].claim
    assert len(final.hypothesis_reviews) == 2
    assert final.hypothesis_reviews[0].batch.reviews[1].verdict == "revise"
    assert set(final.hypothesis_reviews[-1].batch.ranking) == {"H001", "H004", "H003"}
    assert all(h.status == "proposed" for h in final.hypotheses)


def test_revision_budget_does_not_turn_unresolved_candidates_ready(setup):
    store, pipeline, factory = setup
    factory.revise = True
    pipeline.config = pipeline.config.model_copy(update={"max_revision_rounds": 0})
    state = run_literature(setup)
    hp = HypothesisPipeline(pipeline)
    final = hp.run(state, store=store, source_root=ROOT, artifact_root=pipeline.artifact_root)
    assert len(final.hypotheses) == 3
    assert final.hypothesis_reviews[-1].batch.reviews[1].verdict == "revise"
    assert hp.exports["review_budget.json"]["remaining_revision_requests"] == ["H002"]


def test_review_existing_candidates_preserves_history_and_advances_round(setup):
    store, pipeline, factory = setup
    state = run_literature(setup)
    hp = HypothesisPipeline(pipeline)
    first = hp.run(state, store=store, source_root=ROOT, artifact_root=pipeline.artifact_root)
    calls = len(factory.calls)
    second = hp.run(first, store=store, source_root=ROOT, artifact_root=pipeline.artifact_root, generate=False)
    assert second.hypotheses == first.hypotheses
    assert second.hypothesis_reviews[:-1] == first.hypothesis_reviews
    assert tuple(record.round for record in second.hypothesis_reviews) == (0, 1)
    assert second.hypothesis_reviews[0].id != second.hypothesis_reviews[1].id
    assert [role for role, _ in factory.calls[calls:]] == ['hypothesis_review']


@pytest.mark.parametrize("kind", ["unknown_reference", "wrong_targets", "wrong_ranking", "unavailable_search", "no_comparison", "missing_parent"])
def test_invalid_reviews_and_revisions_rejected(setup, kind):
    store, pipeline, factory = setup
    state = run_literature(setup)
    factory.revise = kind == "missing_parent"
    def mutate(role, data, inputs):
        if role == "hypothesis_review":
            if kind == "unknown_reference": data["reviews"][0]["evidence_ids"] = ["EV_unknown"]
            if kind == "wrong_targets": data["reviews"][0]["hypothesis_id"] = "H999"
            if kind == "wrong_ranking": data["ranking"] = ["H001"] * 3
            if kind == "unavailable_search": data["search_requests"] = [{"text": "adaptation", "purpose": "challenge", "discipline": "Learning", "rationale": "Check"}]
            if kind == "no_comparison": data["comparisons"] = []
        if role == "hypothesis_revision" and kind == "missing_parent": data["hypotheses"][0]["parent_hypotheses"] = []
    factory.mutate = mutate
    with pytest.raises(ValueError):
        HypothesisPipeline(pipeline).run(state, store=store, source_root=ROOT, artifact_root=pipeline.artifact_root)
    assert state.hypotheses == ()


def test_failed_supplement_search_records_failure_and_keeps_evidence(setup):
    store, pipeline, factory = setup
    factory.supplement = True
    pipeline.config = pipeline.config.model_copy(update={"max_supplement_rounds": 1})
    state = run_literature(setup)
    # Existing papers are removed, so all available new candidates are rejected.
    hp = HypothesisPipeline(pipeline)
    final = hp.run(state, store=store, source_root=ROOT, artifact_root=pipeline.artifact_root)
    assert final.evidence == state.evidence and len(final.hypothesis_reviews) == 2
    assert hp.exports["review_budget.json"]["supplement_rounds"] == 1
    assert hp.exports["review_budget.json"]["supplement_failures"]
    assert final.hypothesis_reviews[-1].batch.search_requests == ()


def test_successful_supplement_adds_sources_and_rereviews_once(setup):
    store, pipeline, factory = setup
    factory.supplement = True
    pipeline.config = pipeline.config.model_copy(update={"max_supplement_rounds": 1})
    state = run_literature(setup)
    pipeline.tools.paper = pipeline.tools.paper.model_copy(update={
        'id': 'P_supp', 'title': 'Independent adaptation study', 'doi': '10.1234/supp'})
    def mutate(role, data, inputs):
        if role == 'literature_screening':
            for item in data['decisions']: item['selected'] = item['paper_id'] == 'P_supp'
        if role == 'literature_synthesis':
            data['claims'][0].update(statement='A second conditional finding',
                                     source_ids=[inputs['accepted_evidence'][-1]['id']])
    factory.mutate = mutate
    hp = HypothesisPipeline(pipeline)
    final = reviewed_hypotheses(state=state, pipeline=hp, store=store,
                                source_root=ROOT, artifact_root=pipeline.artifact_root, emit=lambda _: None)
    assert len(final.papers) == len(final.evidence) == len(final.knowledge) == 2
    assert len(final.hypothesis_reviews) == 2
    assert len(load_source_documents(final, store)) == 2
    assert hp.exports['review_budget.json']['supplement_rounds'] == 1
    assert hp.exports['review_budget.json']['supplement_failures'] == []
    assert [role for role, _ in factory.calls].count('literature_search') == 1
    last = [inputs for role, inputs in factory.calls if role == 'hypothesis_review'][-1]
    assert last['supplementary_search_available'] is False and len(last['original_documents']) == 2
    assert final.hypothesis_reviews[-1].batch.search_requests == ()


def test_document_tampering_prevents_hypothesis_calls(setup):
    store, pipeline, factory = setup
    state = run_literature(setup)
    path = next(pipeline.artifact_root.glob("run_*/document_P_test.json"))
    path.write_text("{}")
    count = len(factory.calls)
    with pytest.raises(ValueError, match="checksum"):
        load_source_documents(state, store)
    assert len(factory.calls) == count


def test_historical_wire_contract_and_version3_roundtrip():
    legacy = ScientificState(question=confirmed_state().question)
    assert "literature" not in legacy.model_dump() and "hypothesis_reviews" not in legacy.model_dump()
    version2 = confirmed_state()
    assert "literature" not in version2.model_dump()
    paper = Paper(id="P", title="Old source", source_uri="https://example.org")
    assert paper.model_dump() == {"id": "P", "title": "Old source", "source_uri": "https://example.org"}
    assert ScientificState.model_validate_json(version2.model_dump_json()) == version2


class FixtureHTTP:
    def __init__(self, body, kind="text/html"):
        self.body, self.kind, self.calls = body, kind, []
    def get(self, url, headers=None):
        self.calls.append(url)
        if isinstance(self.body, Exception): raise self.body
        return self.body, self.kind


def test_html_locations_and_excerpt_budget():
    html = '<html><nav>Ignore navigation</nav><article><h2>Results</h2>' + ''.join('<p>'+QUOTE*3+'</p>' for _ in range(10)) + '</article><footer>Ignore footer</footer></html>'
    tools = LiteratureTools(FixtureHTTP(html.encode()))
    paper = FakeTools().paper.model_copy(update={"fulltext_uri": "https://example.org/article"})
    document = tools.fetch_document(paper, max_characters=1000)
    assert document.read_level == "excerpt"
    assert sum(len(s.text) for s in document.sections) <= 1000
    assert any(s.heading == "Results" for s in document.sections)
    assert not any("navigation" in s.text or "footer" in s.text for s in document.sections)


def test_http_cache_preserves_timestamp_and_document_identity(tmp_path):
    url = "https://example.org/article"
    body = ('<article><h2>Results</h2><p>' + QUOTE * 12 + '</p></article>').encode()
    cache = tmp_path / hashlib.sha256(url.encode()).hexdigest()
    cache.mkdir()
    (cache / 'body').write_bytes(body)
    (cache / 'receipt.json').write_text(json.dumps({'url': url, 'sha256': hashlib.sha256(body).hexdigest(),
        'content_type': 'text/html', 'fetched_at': '2026-10-11T00:00:00Z'}))
    http = LiteratureHTTP(tmp_path)
    class NoNetwork:
        def open(self, *args, **kwargs):
            raise AssertionError('Cached document must not use the network')
    http.opener = NoNetwork()
    tools = LiteratureTools(http)
    paper = FakeTools().paper.model_copy(update={'fulltext_uri': url})
    first = tools.fetch_document(paper)
    http.fetched_at = 'changed'
    second = tools.fetch_document(paper)
    assert first == second and second.fetched_at == '2026-10-11T00:00:00Z'
    (cache / 'body').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        http.get(url)


def test_review_schema_has_closed_cohort_prediction_keys(tmp_path):
    agent = HypothesisReviewAgent(None, ReasoningConfig(model='fixture'), tmp_path)
    schema = agent.output_schema({'target_hypothesis_ids': ['H001', 'H004']})
    predictions = schema['$defs']['HypothesisComparison']['properties']['predictions']
    assert predictions['required'] == ['H001', 'H004']
    assert set(predictions['properties']) == {'H001', 'H004'}
    def check(node):
        if isinstance(node, dict):
            if node.get('type') == 'object':
                assert node.get('additionalProperties') is False
                assert set(node.get('required', [])) == set(node.get('properties', {}))
                assert 'propertyNames' not in node
            for value in node.values(): check(value)
        elif isinstance(node, list):
            for value in node: check(value)
    check(schema)


def test_human_suggestion_sheet_excludes_detailed_review_limitations(setup):
    store, pipeline, factory = setup
    detailed = 'Detailed technical limitations. ' * 300
    factory.mutate = lambda role, data, inputs: data.update(limitations=[detailed]) if role == 'hypothesis_review' else None
    state = run_literature(setup)
    final = reviewed_hypotheses(state=state, pipeline=HypothesisPipeline(pipeline), store=store,
                               source_root=ROOT, artifact_root=pipeline.artifact_root, emit=lambda _: None)
    sheet = hypothesis_suggestion_sheet(final)
    assert detailed not in sheet and len(sheet) < 1000 and 'report.md' in sheet
    assert any(detailed.rstrip() in p.read_text() for p in pipeline.artifact_root.glob('run_*/report.md'))


def test_fulltext_failure_abstract_fallback_is_explicit():
    tools = LiteratureTools(FixtureHTTP(RuntimeError("Unavailable")))
    paper = FakeTools().paper
    document = tools.fetch_document(paper)
    assert document.read_level == "abstract" and document.sections[0].text == QUOTE
    assert any("仅取得摘要" in s for s in document.limitations)
    with pytest.raises(RuntimeError): tools.fetch_document(paper, allow_abstract=False)


def test_title_only_paper_is_retained_without_reading_or_evidence(setup):
    _, pipeline, factory = setup
    pipeline.config = pipeline.config.model_copy(update={'max_selected_papers': 2})
    original = pipeline.tools.fetch_document
    def fetch(candidate, **kwargs):
        if candidate.id != 'P_other':
            return original(candidate, **kwargs)
        sections = (DocumentSection(id='S001', heading='Title', text=candidate.title),)
        return PaperDocument(paper_id=candidate.id, source_uri=candidate.source_uri, read_level='title',
            sections=sections, content_sha256=digest([s.model_dump(mode='json') for s in sections]),
            fetched_at='not_recorded', limitations=('Only title available',))
    pipeline.tools.fetch_document = fetch
    def mutate(role, data, inputs):
        if role == 'literature_screening':
            for decision in data['decisions']:
                decision['selected'] = True
    factory.mutate = mutate
    state = run_literature(setup)
    assert [p.read_level for p in state.papers] == ['full_text', 'title']
    assert len(state.evidence) == 1 and state.evidence[0].source_id == 'P_test'
    assert [role for role, _ in factory.calls].count('literature_reading') == 1
    assert pipeline.exports['retrieval_summary.json']['read_levels']['title'] == 1
    title_evidence = state.evidence[0].model_copy(update={'source_id': 'P_other', 'content_sha256': state.papers[1].content_sha256})
    with pytest.raises(ValueError, match='Title-only'):
        state.revised(evidence=(title_evidence,))


def test_pdf_page_location():
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=800, height=800)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(('BT /F1 10 Tf 10 700 Td (' + (QUOTE + ' ') * 8 + ') Tj ET').encode())
    page[NameObject('/Contents')] = writer._add_object(stream)
    buf = BytesIO()
    writer.write(buf)
    tools = LiteratureTools(FixtureHTTP(buf.getvalue(), "application/pdf"))
    document = tools.fetch_document(FakeTools().paper)
    assert document.read_level == "full_text"
    assert document.sections[0].heading == "Page 1" and QUOTE in document.sections[0].text


def test_arxiv_and_doi_version_dedup():
    p = FakeTools().paper
    p1 = p.model_copy(update={"arxiv_id": "2401.00001v1", "providers": ("semantic_scholar",)})
    p2 = p.model_copy(update={"id": "different", "title": "Changed title", "arxiv_id": "2401.00001v2",
                             "providers": ("arxiv",), "fulltext_uri": "https://arxiv.org/pdf/2401.00001v2"})
    merged = deduplicate([p1, p2])
    assert len(merged) == 1 and merged[0].arxiv_id.endswith("v2")
    assert set(merged[0].providers) == {"semantic_scholar", "arxiv"}


def test_dedup_merges_previously_separate_doi_and_arxiv_records():
    base = FakeTools().paper
    doi = base.model_copy(update={'providers': ('crossref',)})
    arxiv = base.model_copy(update={'id': 'P_arxiv', 'title': 'Preprint title', 'doi': None,
                                  'arxiv_id': '2401.00001v1', 'providers': ('arxiv',)})
    bridge = base.model_copy(update={'id': 'P_bridge', 'title': 'Indexed title',
                                   'arxiv_id': '2401.00001v2', 'providers': ('semantic_scholar',)})
    papers = deduplicate([doi, arxiv, bridge])
    assert len(papers) == 1 and papers[0].arxiv_id == '2401.00001v2'
    assert set(papers[0].providers) == {'crossref', 'arxiv', 'semantic_scholar'}


def test_arxiv_atom_search_adapter():
    xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2401.00001v2</id><title>Adaptation study</title><summary>Conditional finding</summary><published>2024-01-01</published><author><name>A Researcher</name></author></entry></feed>'
    tools = LiteratureTools(FixtureHTTP(xml.encode(), "application/atom+xml"))
    query = SearchQuery(text="continual adaptation", purpose="overview", discipline="Learning", rationale="Explore")
    results = tools.search((query,), ("arxiv",), per_query=2)
    assert results[0].arxiv_id == "2401.00001v2" and results[0].year == 2024
    assert "all%3Acontinual+AND+all%3Aadaptation" in tools.http.calls[0]


def test_crossref_search_reads_deposited_metadata_and_abstract():
    record = {'title': ['Online adaptation'], 'DOI': '10.1234/online', 'abstract': '<jats:p>Reported conditional recovery.</jats:p>',
              'author': [{'given': 'A', 'family': 'Researcher'}], 'issued': {'date-parts': [[2024, 2]]},
              'link': [{'content-type': 'application/pdf', 'URL': 'https://example.org/paper.pdf'}]}
    tools = LiteratureTools(FixtureHTTP(json.dumps({'message': {'items': [record]}}).encode(), 'application/json'))
    query = SearchQuery(text='online adaptation', purpose='overview', discipline='Learning', rationale='Explore')
    results = tools.search((query,), ('crossref',), per_query=2)
    assert len(results) == 1 and results[0].doi == '10.1234/online'
    assert results[0].abstract == 'Reported conditional recovery.' and results[0].year == 2024
    assert results[0].fulltext_uri.endswith('.pdf') and 'query.bibliographic=' in tools.http.calls[0]


def test_rate_limited_source_is_skipped_while_other_source_continues():
    class Tools(LiteratureTools):
        def _search(self, query, provider, count):
            if provider == 'semantic_scholar':
                raise RuntimeError('Literature HTTP request failed (status 429)')
            return [FakeTools().paper.model_copy(update={'providers': (provider,), 'query_texts': (query.text,)})]
    tools = Tools(None)
    queries = tuple(SearchQuery(text=text, purpose='overview', discipline='Learning', rationale='Explore') for text in ['adaptation', 'learning'])
    results = tools.search(queries, ('semantic_scholar', 'arxiv'))
    assert len(results) == 1
    assert [e['status'] for e in tools.events] == ['failed', 'ok', 'skipped', 'ok']


def test_candidate_cap_balances_sources_instead_of_filling_with_first_source():
    class Tools(LiteratureTools):
        def _search(self, query, provider, count):
            return [FakeTools().paper.model_copy(update={'id': f'{provider}_{n}', 'title': f'{provider} {n}',
                'doi': None, 'providers': (provider,), 'query_texts': (query.text,)}) for n in range(3)]
    tools = Tools(None)
    query = SearchQuery(text='adaptation', purpose='overview', discipline='Learning', rationale='Explore')
    results = tools.search((query,), ('semantic_scholar', 'arxiv'), max_candidates=2)
    assert {p.providers[0] for p in results} == {'semantic_scholar', 'arxiv'}


@pytest.mark.parametrize('payload', [{'data': None}, {'data': [{}]}, {'data': [{'title': 'Study', 'authors': None}]}, {'data': [{'title': 'Study'}]}])
def test_invalid_metadata_provider_response_is_logged_without_inventing_papers(payload):
    tools = LiteratureTools(FixtureHTTP(json.dumps(payload).encode(), 'application/json'))
    query = SearchQuery(text='adaptation', purpose='overview', discipline='Learning', rationale='Explore')
    assert tools.search((query,), ('semantic_scholar',)) == []
    assert tools.events[0]['status'] == 'failed'


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1", "http://169.254.169.254", "https://user:pass@example.org", "http://localhost"])
def test_document_acquisition_does_not_accept_local_or_credential_urls(url):
    with pytest.raises(ValueError): public_url(url)


@pytest.mark.parametrize("args", [["--research"], ["--research-literature"], ["--review-hypotheses"], ["--demo", "--literature-config", "config/literature.yaml"]])
def test_new_cli_requires_explicit_confirmed_state(args):
    with pytest.raises(SystemExit): main(args)
