"""Cross-disciplinary search, access fallbacks, and bounded throttling."""
import json
from datetime import datetime, timezone
from email.message import Message
from email.utils import format_datetime
from io import BytesIO
from urllib.error import HTTPError

import pytest

from scientist.literature_pipeline import LiteratureConfig
from scientist.literature_schemas import PaperCandidate, SearchQuery
from scientist.tools.literature import LiteratureHTTP, LiteratureTools, deduplicate


QUERY = SearchQuery(text="synaptic plasticity", purpose="cross_discipline", discipline="Neuroscience", rationale="Explore mechanisms")
PAPER = PaperCandidate(id="P_neuro", title="Synaptic adaptation", doi="10.1234/neuro",
                       abstract="The authors report adaptation under specific experimental conditions.",
                       source_uri="https://example.org/article")
BODY = "The authors measured adaptation under controlled experimental conditions. " * 15


class RoutingHTTP:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, headers=None):
        self.calls.append((url, headers or {}))
        for fragment, response in self.routes.items():
            if fragment in url:
                if isinstance(response, Exception):
                    raise response
                return response
        raise RuntimeError("Unavailable fixture")


def europe_record(**changes):
    return {"id": "123456", "source": "MED", "title": PAPER.title, "doi": PAPER.doi,
            "pmcid": "PMC123456", "pubYear": "2024", "isOpenAccess": "Y",
            "abstractText": "<p>Adaptation was reported.</p>",
            "authorList": {"author": [{"fullName": "A Researcher"}]}, **changes}


def europe_payload(*records):
    return json.dumps({"resultList": {"result": list(records)}}).encode(), "application/json"


def test_europe_pmc_search_preserves_abstract_and_ids_without_oa_filter():
    http = RoutingHTTP({"/search?": europe_payload(europe_record(), europe_record(id="123457", title="Closed paper", doi="10.1234/closed", pmcid=None, isOpenAccess="N"))})
    papers = LiteratureTools(http).search((QUERY,), ("europe_pmc",))
    assert len(papers) == 2
    assert papers[0].pmid == "123456" and papers[0].pmcid == "PMC123456"
    assert papers[0].abstract == "Adaptation was reported." and papers[0].authors == ("A Researcher",)
    assert papers[0].fulltext_uri.endswith("/fullTextXML")
    assert papers[1].fulltext_uri is None  # Closed articles remain in the relevance pool.
    assert "resultType=core" in http.calls[0][0] and "OPEN_ACCESS" not in http.calls[0][0]


def test_openalex_reconstructs_abstract_and_retains_multiple_oa_locations(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "fixture-secret")
    record = {"id": "https://openalex.org/W1", "title": PAPER.title, "doi": "https://doi.org/" + PAPER.doi,
              "abstract_inverted_index": {"adaptation": [2], "The": [0], "authors": [1]},
              "publication_year": 2024, "authorships": [{"author": {"display_name": "A Researcher"}}],
              "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/123456/"},
              "locations": [{"is_oa": True, "pdf_url": "https://repo.org/accepted.pdf"},
                            {"is_oa": True, "pdf_url": "https://publisher.org/paper.pdf"}]}
    http = RoutingHTTP({"openalex.org/works?": (json.dumps({"results": [record]}).encode(), "application/json")})
    paper = LiteratureTools(http).search((QUERY,), ("openalex",))[0]
    assert paper.abstract == "The authors adaptation" and paper.pmid == "123456"
    assert len(paper.fulltext_uris) == 2
    assert http.calls[0][1] == {"Authorization": "Bearer fixture-secret"}
    assert "fixture-secret" not in http.calls[0][0] and "fixture-secret" not in paper.model_dump_json()


def test_agent_source_choice_controls_executed_sources():
    http = RoutingHTTP({"/search?": europe_payload(europe_record())})
    query = QUERY.model_copy(update={"providers": ("europe_pmc",)})
    papers = LiteratureTools(http).search((query,), ("arxiv", "europe_pmc", "openalex"))
    assert len(papers) == len(http.calls) == 1
    with pytest.raises(ValueError, match="configured"):
        LiteratureTools(http).search((query,), ("arxiv",))


def test_pmid_merge_retains_all_download_locations():
    first = PAPER.model_copy(update={"pmid": "123456", "fulltext_uri": "https://repo.org/one.pdf"})
    second = PAPER.model_copy(update={"id": "other", "title": "Different index title", "doi": None,
                                     "pmid": "123456", "pmcid": "PMC123456",
                                     "fulltext_uris": ("https://repo.org/two.pdf",)})
    merged = deduplicate([first, second])
    assert len(merged) == 1 and merged[0].pmcid == "PMC123456"
    assert set(merged[0].fulltext_uris) == {"https://repo.org/one.pdf", "https://repo.org/two.pdf"}


def test_pmc_xml_uses_body_sections_and_locations():
    xml = ("<article><front><abstract><p>Short abstract</p></abstract></front>"
           "<body><sec><title>Results</title><p>" + BODY + "</p></sec></body></article>").encode()
    http = RoutingHTTP({"/fullTextXML": (xml, "application/xml")})
    document = LiteratureTools(http).fetch_document(PAPER.model_copy(update={"pmcid": "PMC123456"}))
    assert document.read_level == "full_text"
    assert any(s.heading == "Results" and s.text == BODY.strip() for s in document.sections)
    assert len(http.calls) == 1


def test_publisher_abstract_page_is_not_mislabeled_full_text():
    html = ("<article><h2>Abstract</h2><p>" + BODY + "</p><h2>References</h2><p>Some citations</p></article>").encode()
    http = RoutingHTTP({"example.org/article": (html, "text/html")})
    tools = LiteratureTools(http, resolve_open_access=False)
    document = tools.fetch_document(PAPER)
    assert document.read_level == "abstract" and document.sections[0].text == PAPER.abstract
    assert any(e["operation"] == "download" and e["status"] == "failed" for e in tools.events)


@pytest.mark.parametrize('fragment', [
    '<section id="abstract"><h2>Results</h2><p>' + BODY + '</p></section>',
    '<h2>Abstract</h2><h3>Methods</h3><p>' + BODY + '</p><h3>Results</h3><p>' + BODY + '</p>',
])
def test_structured_abstract_sections_do_not_count_as_full_text(fragment):
    http = RoutingHTTP({'example.org/article': (('<article>' + fragment + '</article>').encode(), 'text/html')})
    document = LiteratureTools(http, resolve_open_access=False).fetch_document(PAPER)
    assert document.read_level == 'abstract'


def test_same_location_uses_html_when_pdf_is_denied(monkeypatch):
    monkeypatch.setenv('UNPAYWALL_EMAIL', 'researcher@example.org')
    location = {'url_for_pdf': 'https://repo.org/blocked.pdf', 'url_for_landing_page': 'https://repo.org/full',
                'version': 'acceptedVersion'}
    http = RoutingHTTP({'unpaywall.org/v2/': (json.dumps({'best_oa_location': location, 'oa_locations': [location]}).encode(), 'application/json'),
                        'blocked.pdf': RuntimeError('Denied'),
                        'repo.org/full': (('<article><h2>Results</h2><p>' + BODY + '</p></article>').encode(), 'text/html'),
                        '/search?': europe_payload()})
    document = LiteratureTools(http).fetch_document(PAPER)
    assert document.read_level == 'full_text' and document.source_uri == 'https://repo.org/full'


def test_unpaywall_tries_another_location_after_download_failure(monkeypatch):
    monkeypatch.setenv("UNPAYWALL_EMAIL", "researcher@example.org")
    first = {"url_for_pdf": "https://closed.org/first.pdf", "version": "publishedVersion"}
    second = {"url_for_landing_page": "https://repo.org/accepted", "version": "acceptedVersion"}
    http = RoutingHTTP({"unpaywall.org/v2/": (json.dumps({"best_oa_location": first, "oa_locations": [first, second]}).encode(), "application/json"),
                        "closed.org": RuntimeError("Literature HTTP request failed (status 403)"),
                        "repo.org/accepted": (("<article><h2>Methods</h2><p>" + BODY + "</p></article>").encode(), "text/html"),
                        "/search?": europe_payload()})
    tools = LiteratureTools(http)
    document = tools.fetch_document(PAPER)
    assert document.read_level == "full_text" and document.source_uri == "https://repo.org/accepted"
    assert any("acceptedVersion" in s for s in document.limitations)
    assert any(e["operation"] == "resolve" and e["status"] == "ok" for e in tools.events)
    assert "researcher@example.org" not in json.dumps(tools.events)


def test_doi_lookup_finds_pmc_when_publisher_is_unavailable(monkeypatch):
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    xml = ("<article><body><sec><title>Results</title><p>" + BODY + "</p></sec></body></article>").encode()
    http = RoutingHTTP({"/search?": europe_payload(europe_record()), "/fullTextXML": (xml, "application/xml")})
    document = LiteratureTools(http).fetch_document(PAPER)
    assert document.read_level == "full_text" and document.source_uri.endswith("/fullTextXML")


def test_only_title_survives_unavailable_access_without_inventing_abstract():
    tools = LiteratureTools(RoutingHTTP({}), resolve_open_access=False)
    document = tools.fetch_document(PAPER.model_copy(update={"abstract": ""}))
    assert document.read_level == "title" and document.sections[0].text == PAPER.title
    with pytest.raises(RuntimeError):
        tools.fetch_document(PAPER, allow_abstract=False)


@pytest.mark.parametrize("provider", ["europe_pmc", "openalex"])
def test_invalid_new_provider_response_is_reported(provider):
    http = RoutingHTTP({"search?": (b'{"resultList":{"result":null}}', "application/json"),
                        "works?": (b'{"results":null}', "application/json")})
    tools = LiteratureTools(http)
    assert tools.search((QUERY,), (provider,)) == []
    assert tools.events[0]["status"] == "failed"


@pytest.mark.parametrize("http_date", [False, True])
def test_long_retry_after_stops_retries_and_cools_down_host(tmp_path, monkeypatch, http_date):
    now = datetime.now(timezone.utc)
    header = format_datetime(datetime.fromtimestamp(now.timestamp() + 120, timezone.utc)) if http_date else "120"
    headers = Message()
    headers["Retry-After"] = header
    class Denied:
        calls = 0
        def open(self, *args, **kwargs):
            self.calls += 1
            raise HTTPError("https://example.org/paper", 429, "Rate limited", headers, BytesIO())
    monkeypatch.setattr("scientist.tools.literature.time.sleep", lambda delay: None)
    http = LiteratureHTTP(tmp_path)
    http.opener = Denied()
    with pytest.raises(RuntimeError, match="429"):
        http.get("https://example.org/first")
    with pytest.raises(RuntimeError, match="cooling down"):
        http.get("https://example.org/second")
    assert http.opener.calls == 1
    assert not list(tmp_path.glob("*/receipt.json"))


def test_new_sources_are_configurable():
    assert {"europe_pmc", "openalex"} <= set(LiteratureConfig().providers)
    assert LiteratureConfig(providers=("europe_pmc",), resolve_open_access=False).providers == ("europe_pmc",)
