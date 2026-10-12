"""Bounded public literature search and immutable, located document snapshots."""
import hashlib
import ipaddress
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from xml.etree import ElementTree

from scientist.literature_schemas import DocumentSection, PaperCandidate, PaperDocument, SearchQuery


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def digest(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def public_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Literature URLs must be public HTTP(S) URLs without credentials")
    if parsed.hostname.lower() in {"localhost", "metadata.google.internal"} or parsed.hostname.endswith(".localhost"):
        raise ValueError("Local literature URLs are not supported")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return url
    if not address.is_global:
        raise ValueError("Private network literature URLs are not supported")
    return url


class LiteratureHTTP:
    """Keys are read only into request headers; cached responses contain no headers with credentials."""
    def __init__(self, cache_root: Path, *, timeout=25, max_bytes=20_000_000, retries=1):
        self.cache_root = Path(cache_root)
        self.timeout, self.max_bytes, self.retries = timeout, max_bytes, retries
        proxies = {p: os.environ.get(p.upper() + "_PROXY", os.environ.get(p + "_proxy", ""))
                   for p in ("http", "https")}
        self.opener = build_opener(ProxyHandler({k: v for k, v in proxies.items() if v}))
        self.last_requests, self.cooldowns = {}, {}
        self.fetched_at = "not_recorded"

    def get(self, url: str, headers: dict | None = None) -> tuple[bytes, str]:
        public_url(url)
        key = hashlib.sha256(url.encode()).hexdigest()
        directory = self.cache_root / key
        if (directory / "receipt.json").exists():
            receipt = json.loads((directory / "receipt.json").read_bytes())
            data = (directory / "body").read_bytes()
            if receipt["url"] != url or hashlib.sha256(data).hexdigest() != receipt["sha256"]:
                raise ValueError("Literature HTTP cache checksum mismatch")
            self.fetched_at = receipt.get("fetched_at", "not_recorded")
            return data, receipt["content_type"]
        request = Request(url, headers={"User-Agent": "AutonomousScientist/0.1 (literature research)", **(headers or {})})
        host = urlsplit(url).hostname
        for attempt in range(self.retries + 1):
            if self.cooldowns.get(host, 0) > time.monotonic():
                raise RuntimeError("Literature HTTP request failed (status 429); source cooling down")
            interval = 3 if host == "export.arxiv.org" else 1
            time.sleep(max(0, interval - (time.monotonic() - self.last_requests.get(host, 0))))
            self.last_requests[host] = time.monotonic()
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    public_url(response.geturl())
                    data = response.read(self.max_bytes + 1)
                    kind = response.headers.get("Content-Type", "")
                if len(data) > self.max_bytes:
                    raise ValueError("Literature document exceeds download budget")
                break
            except HTTPError as error:
                delay = min(30, 2 ** attempt + random.uniform(0, 0.5))
                retry_after = error.headers.get("Retry-After") if error.headers else None
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        try:
                            delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - time.time())
                        except (ValueError, TypeError, OverflowError):
                            pass
                if error.code == 429:
                    self.cooldowns[host] = time.monotonic() + delay
                if error.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise RuntimeError(f"Literature HTTP request failed (status {error.code})") from None
                if delay > 30:
                    raise RuntimeError(f"Literature HTTP request failed (status {error.code}); Retry-After exceeds wait budget") from None
                time.sleep(delay)
            except (URLError, TimeoutError, OSError):
                if attempt == self.retries:
                    raise RuntimeError("Literature HTTP request failed or timed out") from None
                time.sleep(1)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "body").write_bytes(data)
        self.fetched_at = datetime.now(timezone.utc).isoformat()
        receipt = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "content_type": kind,
                   "fetched_at": self.fetched_at}
        temporary = directory / "receipt.tmp"
        temporary.write_text(json.dumps(receipt, ensure_ascii=False), encoding="utf-8")
        temporary.replace(directory / "receipt.json")
        return data, kind


def candidate_id(title: str) -> str:
    return "P_" + hashlib.sha256(normalize(title).casefold().encode()).hexdigest()[:16]


def normalized_doi(value):
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value or "", flags=re.IGNORECASE).strip().casefold()


def arxiv_parts(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    value = re.sub(r"^https?://[^/]+/(?:abs|pdf|html)/", "", value).removesuffix(".pdf")
    match = re.fullmatch(r"(.+?)(v\d+)?", value)
    return (match.group(1), match.group(2)) if match else (None, None)


def deduplicate(candidates: list[PaperCandidate]) -> list[PaperCandidate]:
    merged, aliases = [], {}
    def combine(previous, candidate):
        _, ver_old = arxiv_parts(previous.arxiv_id)
        _, ver_new = arxiv_parts(candidate.arxiv_id)
        newer = bool(ver_new and (not ver_old or int(ver_new[1:]) > int(ver_old[1:])))
        return previous.model_copy(update={
            "abstract": max((previous.abstract, candidate.abstract), key=len),
            "doi": previous.doi or candidate.doi,
            "arxiv_id": candidate.arxiv_id if newer else previous.arxiv_id or candidate.arxiv_id,
            "pmid": previous.pmid or candidate.pmid,
            "pmcid": previous.pmcid or candidate.pmcid,
            "fulltext_uri": candidate.fulltext_uri if newer else previous.fulltext_uri or candidate.fulltext_uri,
            "fulltext_uris": tuple(dict.fromkeys(previous.fulltext_uris + candidate.fulltext_uris +
                                                tuple(u for u in (previous.fulltext_uri, candidate.fulltext_uri) if u))),
            "providers": tuple(dict.fromkeys(previous.providers + candidate.providers)),
            "query_texts": tuple(dict.fromkeys(previous.query_texts + candidate.query_texts))})
    for candidate in candidates:
        aid, _ = arxiv_parts(candidate.arxiv_id)
        keys = ["title:" + normalize(candidate.title).casefold()]
        if candidate.doi:
            keys.append("doi:" + candidate.doi.lower().removeprefix("https://doi.org/"))
        if aid:
            keys.append("arxiv:" + aid)
        if candidate.pmid:
            keys.append("pmid:" + candidate.pmid)
        if candidate.pmcid:
            keys.append("pmcid:" + candidate.pmcid.upper())
        matching = sorted({aliases[k] for k in keys if k in aliases})
        if not matching:
            index = len(merged)
            merged.append(candidate)
        else:
            index = matching[0]
            for other in matching[1:]:
                merged[index] = combine(merged[index], merged[other])
                merged[other] = None
            if len(matching) > 1:
                aliases = {key: index if value in matching else value for key, value in aliases.items()}
            merged[index] = combine(merged[index], candidate)
        for key in keys:
            aliases[key] = index
    return [paper for paper in merged if paper is not None]


class ArticleHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.parts, self.blocks = [], [], []
        self.body_blocks = []
        self.heading = "Document"
        self.article_seen = False
        self.abstract_section_level = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        in_article = any(a for _, a, _, _ in self.stack)
        article = tag in {"article", "main"} or "ltx_document" in attrs.get("class", "").split()
        excluded = tag in {"script", "style", "nav", "footer", "button"} or any(e for _, _, e, _ in self.stack)
        abstract = "abstract" in (attrs.get("id", "") + " " + attrs.get("class", "")).lower() or any(a for _, _, _, a in self.stack)
        if article:
            if not self.article_seen:
                self.blocks, self.parts, self.body_blocks = [], [], []
            self.article_seen = True
        if tag in {"p", "h1", "h2", "h3", "h4", "li", "figcaption"}:
            self.flush()
        if tag not in {"meta", "link", "br", "img", "hr", "input", "source", "wbr"}:
            self.stack.append((tag, article or in_article, excluded, abstract))

    def handle_endtag(self, tag):
        if tag in {"p", "h1", "h2", "h3", "h4", "li", "figcaption", "div"}:
            text = self.flush()
            if tag in {"h1", "h2", "h3", "h4"} and text:
                self.heading = text
                level = int(tag[1])
                if re.match(r"^(abstract|摘要)\b", text, re.IGNORECASE):
                    self.abstract_section_level = level
                elif self.abstract_section_level is not None and level <= self.abstract_section_level:
                    self.abstract_section_level = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if any(e for _, _, e, _ in self.stack):
            return
        if self.article_seen and not any(a for _, a, _, _ in self.stack):
            return
        self.parts.append(data)

    def flush(self):
        text = normalize(" ".join(self.parts))
        self.parts = []
        if text:
            self.blocks.append((self.heading, text))
            if self.abstract_section_level is None and not any(a for _, _, _, a in self.stack):
                self.body_blocks.append((self.heading, text))
        return text


class LiteratureTools:
    def __init__(self, http: LiteratureHTTP, *, resolve_open_access=True):
        self.http = http
        self.resolve_open_access = resolve_open_access
        self.events = []
        self.unavailable_providers = {}
        self.next_page = None

    def search(self, queries: tuple[SearchQuery, ...], providers: tuple[str, ...], *, per_query=5, max_candidates=15,
               max_pages=1, exclude_ids=()):
        found = []
        for query in queries:
            chosen_providers = query.providers or providers
            if set(chosen_providers) - set(providers) or len(chosen_providers) != len(set(chosen_providers)):
                raise ValueError("Query providers must be distinct configured sources")
            for provider in chosen_providers:
                if provider in self.unavailable_providers:
                    self.events.append({"operation": "search", "provider": provider, "query": query.text,
                                        "status": "skipped", "error": self.unavailable_providers[provider]})
                    continue
                seen, offset, cursor = set(), 0, None
                for page in range(max_pages):
                    try:
                        self.next_page = None
                        candidates = (self._search(query, provider, per_query) if page == 0 else
                                      self._search(query, provider, per_query, offset=offset, cursor=cursor))
                        found.extend(candidates)
                        ids = {c.id for c in candidates}
                        new_ids = ids - seen
                        seen.update(ids)
                        self.events.append({"operation": "search", "provider": provider, "query": query.text,
                                            "returned": len(candidates), "new_records": len(new_ids), "page": page + 1, "status": "ok"})
                        if not candidates or not new_ids:
                            break
                        if self.next_page is None:
                            if len(candidates) < per_query:
                                break
                            offset += per_query
                        else:
                            if self.next_page.get("exhausted"):
                                break
                            offset = self.next_page.get("offset", offset + per_query)
                            cursor = self.next_page.get("cursor")
                    except RuntimeError as error:
                        if "status 429" in str(error) or "status 403" in str(error):
                            self.unavailable_providers[provider] = "Provider rate-limited or denied access for this search run"
                        self.events.append({"operation": "search", "provider": provider, "query": query.text,
                                            "page": page + 1, "status": "failed", "error": str(error)})
                        break
        # Round-robin query coverage avoids letting the first query fill the budget.
        unique = [p for p in deduplicate(found) if p.id not in exclude_ids]
        ordered = []
        pools = [[c for c in unique if q.text in c.query_texts and p in c.providers] for q in queries for p in providers]
        for n in range(max((len(p) for p in pools), default=0)):
            for pool in pools:
                if n < len(pool) and pool[n].id not in {c.id for c in ordered}:
                    ordered.append(pool[n])
        return ordered[:max_candidates]

    def _search(self, query, provider, count, *, offset=0, cursor=None):
        if provider == "europe_pmc":
            params = {"query": query.text, "format": "json", "resultType": "core", "pageSize": count,
                      "cursorMark": cursor or "*"}
            data, _ = self.http.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urlencode(params))
            records = self._europe_pmc_records(data, query.text)
            next_cursor = json.loads(data).get("nextCursorMark")
            if next_cursor is not None and not isinstance(next_cursor, str):
                raise RuntimeError("Europe PMC returned an invalid pagination cursor")
            self.next_page = {"cursor": next_cursor, "exhausted": not next_cursor or next_cursor == (cursor or "*")}
            return records
        if provider == "openalex":
            params = {"search": query.text, "per_page": count}
            if offset:
                params["page"] = offset // count + 1
            headers = {"Authorization": "Bearer " + os.environ["OPENALEX_API_KEY"]} if os.environ.get("OPENALEX_API_KEY") else {}
            data, _ = self.http.get("https://api.openalex.org/works?" + urlencode(params), headers)
            try:
                payload = json.loads(data)
                records = payload["results"]
                if not isinstance(records, list):
                    raise ValueError("Invalid results")
                result = []
                for p in records:
                    title = normalize(p.get("title") or p.get("display_name") or "")
                    if not title:
                        continue
                    positions = {}
                    for word, indices in (p.get("abstract_inverted_index") or {}).items():
                        for index in indices:
                            positions[int(index)] = word
                    locations = [p.get("best_oa_location")] + (p.get("locations") or [])
                    urls = tuple(dict.fromkeys(url for location in locations if location and location.get("is_oa")
                                               for url in (location.get("pdf_url"), location.get("landing_page_url")) if url))
                    ids = p.get("ids") or {}
                    doi = (p.get("doi") or "").removeprefix("https://doi.org/") or None
                    pmid = (ids.get("pmid") or "").rstrip("/").rsplit("/", 1)[-1] or None
                    pmcid = (ids.get("pmcid") or "").rstrip("/").rsplit("/", 1)[-1] or None
                    result.append(PaperCandidate(id=candidate_id(title), title=title,
                        abstract=" ".join(positions[n] for n in sorted(positions)),
                        authors=tuple(a["author"]["display_name"] for a in p.get("authorships", [])),
                        year=p.get("publication_year"), doi=doi, pmid=pmid, pmcid=pmcid,
                        source_uri="https://doi.org/" + doi if doi else p["id"],
                        fulltext_uri=urls[0] if urls else None, fulltext_uris=urls,
                        providers=(provider,), query_texts=(query.text,)))
                total = (payload.get("meta") or {}).get("count")
                self.next_page = {"offset": offset + count, "exhausted": len(records) < count or
                                  (isinstance(total, int) and offset + len(records) >= total)}
                return result
            except (ValueError, KeyError, TypeError, AttributeError):
                raise RuntimeError("OpenAlex returned invalid metadata records") from None
        if provider == "crossref":
            params = {"query.bibliographic": query.text, "rows": count}
            if offset:
                params["offset"] = offset
            data, _ = self.http.get("https://api.crossref.org/works?" + urlencode(params))
            try:
                payload = json.loads(data)
                items = payload["message"]["items"]
                if not isinstance(items, list):
                    raise ValueError("Invalid result list")
                result = []
                for p in items:
                    titles = p.get("title") or []
                    if not titles or not p.get("DOI"):
                        continue
                    title = normalize(titles[0])
                    # Crossref abstracts are often JATS/XML; parse deposited text only.
                    abstract = ArticleHTML()
                    abstract.feed(p.get("abstract") or "")
                    abstract.flush()
                    dates = (p.get("issued") or {}).get("date-parts") or []
                    links = [v.get("URL") for v in p.get("link", [])
                             if v.get("content-type") == "application/pdf" and v.get("URL")]
                    doi = p["DOI"]
                    result.append(PaperCandidate(id=candidate_id(title), title=title,
                        abstract=" ".join(t for _, t in abstract.blocks),
                        authors=tuple(normalize(" ".join([a.get("given", ""), a.get("family", "")]))
                                      for a in p.get("author", [])),
                        year=dates[0][0] if dates and dates[0] else None, doi=doi,
                        source_uri="https://doi.org/" + doi, fulltext_uri=links[0] if links else None,
                        fulltext_uris=tuple(dict.fromkeys(links)),
                        providers=(provider,), query_texts=(query.text,)))
                total = payload["message"].get("total-results")
                self.next_page = {"offset": offset + count, "exhausted": len(items) < count or
                                  (isinstance(total, int) and offset + len(items) >= total)}
                return result
            except (ValueError, KeyError, TypeError, AttributeError):
                raise RuntimeError("Crossref returned invalid metadata records") from None
        if provider == "semantic_scholar":
            params = {"query": query.text, "limit": count,
                      "fields": "title,abstract,authors,year,url,externalIds,openAccessPdf"}
            if offset:
                params["offset"] = offset
            headers = {"x-api-key": os.environ["SEMANTIC_SCHOLAR_API_KEY"]} if os.environ.get("SEMANTIC_SCHOLAR_API_KEY") else {}
            data, _ = self.http.get("https://api.semanticscholar.org/graph/v1/paper/search?" + urlencode(params), headers)
            try:
                payload = json.loads(data)
                if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                    raise ValueError("Missing results")
                for paper in payload["data"]:
                    if not isinstance(paper, dict) or not isinstance(paper.get("title"), str) or not paper["title"].strip():
                        raise ValueError("Invalid paper metadata")
                    if not isinstance(paper.get("authors", []), list):
                        raise ValueError("Invalid author metadata")
                    if any(not isinstance(a, dict) or not isinstance(a.get("name"), str) for a in paper.get("authors", [])):
                        raise ValueError("Invalid author metadata")
                    for field in ("externalIds", "openAccessPdf"):
                        if paper.get(field) is not None and not isinstance(paper[field], dict):
                            raise ValueError("Invalid source metadata")
            except ValueError:
                raise RuntimeError("Semantic Scholar returned an invalid metadata response") from None
            try:
                next_offset = payload.get("next")
                self.next_page = {"offset": next_offset, "exhausted": type(next_offset) is not int or next_offset <= offset}
                return [PaperCandidate(id=candidate_id(p["title"]), title=p["title"], abstract=p.get("abstract") or "",
                            authors=tuple(a["name"] for a in p.get("authors", [])), year=p.get("year"),
                            doi=(p.get("externalIds") or {}).get("DOI"), arxiv_id=(p.get("externalIds") or {}).get("ArXiv"),
                            pmid=(p.get("externalIds") or {}).get("PubMed"), pmcid=(p.get("externalIds") or {}).get("PubMedCentral"),
                            source_uri=p.get("url") or "https://www.semanticscholar.org/paper/" + p["paperId"],
                            fulltext_uri=(p.get("openAccessPdf") or {}).get("url"),
                            providers=(provider,), query_texts=(query.text,)) for p in payload["data"]]
            except (ValueError, KeyError, TypeError):
                raise RuntimeError("Semantic Scholar returned invalid paper records") from None
        if provider != "arxiv":
            raise ValueError("Unknown literature provider")
        tokens = re.findall(r'[\w-]+', query.text)
        expression = " AND ".join("all:" + t for t in tokens)
        params = {"search_query": expression, "max_results": count, "sortBy": "relevance"}
        if offset:
            params["start"] = offset
        data, _ = self.http.get("https://export.arxiv.org/api/query?" + urlencode(params))
        ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
        try:
            root = ElementTree.fromstring(data)
        except ElementTree.ParseError:
            raise RuntimeError("arXiv returned an invalid metadata response") from None
        if root.tag != "{http://www.w3.org/2005/Atom}feed":
            raise RuntimeError("arXiv response is not an Atom feed")
        result = []
        entries = root.findall("a:entry", ns)
        total_text = root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
        total = int(total_text) if total_text and total_text.isdigit() else None
        self.next_page = {"offset": offset + count, "exhausted": len(entries) < count or
                          (total is not None and offset + len(entries) >= total)}
        for entry in root.findall("a:entry", ns):
            uri = entry.findtext("a:id", namespaces=ns)
            if not uri or "/abs/" not in uri:
                continue
            title = normalize(entry.findtext("a:title", namespaces=ns) or "")
            aid = uri.split("/abs/", 1)[1]
            result.append(PaperCandidate(id=candidate_id(title), title=title,
                abstract=normalize(entry.findtext("a:summary", namespaces=ns) or ""),
                authors=tuple(a.findtext("a:name", namespaces=ns) or "" for a in entry.findall("a:author", ns)),
                year=int((entry.findtext("a:published", namespaces=ns) or "0000")[:4]),
                doi=entry.findtext("x:doi", namespaces=ns), arxiv_id=aid,
                source_uri="https://arxiv.org/abs/" + aid, fulltext_uri="https://arxiv.org/pdf/" + aid,
                providers=(provider,), query_texts=(query.text,)))
        return result

    @staticmethod
    def _europe_pmc_records(data, query_text):
        try:
            records = json.loads(data)["resultList"]["result"]
            if not isinstance(records, list):
                raise ValueError("Invalid results")
            result = []
            for p in records:
                title = normalize(p.get("title") or "")
                if not title:
                    continue
                parser = ArticleHTML()
                parser.feed(p.get("abstractText") or "")
                parser.flush()
                pmcid = p.get("pmcid")
                urls = []
                if pmcid and p.get("isOpenAccess") == "Y":
                    urls.append("https://www.ebi.ac.uk/europepmc/webservices/rest/" + quote(pmcid, safe="") + "/fullTextXML")
                for item in (p.get("fullTextUrlList") or {}).get("fullTextUrl", []):
                    if item.get("availabilityCode") == "OA" and item.get("url"):
                        urls.append(item["url"])
                doi = p.get("doi")
                source = p.get("source") or "MED"
                identifier = p["id"]
                names = (p.get("authorList") or {}).get("author", [])
                result.append(PaperCandidate(id=candidate_id(title), title=title,
                    abstract=" ".join(t for _, t in parser.blocks),
                    authors=tuple(a.get("fullName") or "" for a in names),
                    year=int(p["pubYear"]) if p.get("pubYear") else None, doi=doi,
                    pmid=identifier if source == "MED" else None, pmcid=pmcid,
                    source_uri="https://doi.org/" + doi if doi else "https://europepmc.org/article/" + source + "/" + identifier,
                    fulltext_uri=urls[0] if urls else None, fulltext_uris=tuple(dict.fromkeys(urls)),
                    providers=("europe_pmc",), query_texts=(query_text,)))
            return result
        except (ValueError, KeyError, TypeError, AttributeError):
            raise RuntimeError("Europe PMC returned invalid metadata records") from None

    def enrich_candidates(self, papers, providers, *, max_requests):
        """Bounded exact-identifier lookups; never copy an abstract from a similar title."""
        updated, used = [], 0
        for paper in papers:
            current = paper
            for provider in ("europe_pmc", "openalex", "crossref"):
                if current.abstract.strip() or used >= max_requests:
                    break
                if provider not in providers or provider in self.unavailable_providers:
                    continue
                if not current.doi and not (provider == "europe_pmc" and current.pmid):
                    continue
                used += 1
                try:
                    donor = self._metadata_lookup(current, provider)
                    if donor is not None:
                        # Match exact IDs, rejecting conflicting DOI records even if another ID agrees.
                        if current.doi and donor.doi and normalized_doi(current.doi) != normalized_doi(donor.doi):
                            raise ValueError("Metadata lookup returned a conflicting DOI")
                        match = ((current.doi and donor.doi and normalized_doi(current.doi) == normalized_doi(donor.doi))
                                 or (current.pmid and donor.pmid and current.pmid == donor.pmid))
                        if not match:
                            raise ValueError("Metadata lookup did not match the requested identifier")
                        current = deduplicate([current, donor])[0]
                    self.events.append({"operation": "metadata", "provider": provider, "paper_id": paper.id,
                                        "status": "ok", "abstract_available": bool(current.abstract.strip())})
                except (RuntimeError, ValueError, KeyError, TypeError, AttributeError) as error:
                    if "status 429" in str(error) or "status 403" in str(error):
                        self.unavailable_providers[provider] = "Provider rate-limited or denied access for this search run"
                    self.events.append({"operation": "metadata", "provider": provider, "paper_id": paper.id,
                                        "status": "failed", "error": str(error)})
            updated.append(current)
        return updated, used

    def _metadata_lookup(self, paper, provider):
        if provider == "europe_pmc":
            expression = 'DOI:"' + normalized_doi(paper.doi) + '"' if paper.doi else 'EXT_ID:' + paper.pmid + ' AND SRC:MED'
            params = {"query": expression, "format": "json", "resultType": "core", "pageSize": 1}
            data, _ = self.http.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urlencode(params))
            records = self._europe_pmc_records(data, "Metadata lookup")
            return records[0] if records else None
        if provider == "openalex":
            headers = {"Authorization": "Bearer " + os.environ["OPENALEX_API_KEY"]} if os.environ.get("OPENALEX_API_KEY") else {}
            data, _ = self.http.get("https://api.openalex.org/works/https://doi.org/" + quote(normalized_doi(paper.doi), safe="/"), headers)
            record = json.loads(data)
            positions = {int(index): word for word, indices in (record.get("abstract_inverted_index") or {}).items() for index in indices}
            locations = [record.get("best_oa_location")] + (record.get("locations") or [])
            urls = tuple(dict.fromkeys(url for location in locations if location and location.get("is_oa")
                                      for url in (location.get("pdf_url"), location.get("landing_page_url")) if url))
            identifiers = record.get("ids") or {}
            return PaperCandidate(id=paper.id, title=paper.title, doi=record.get("doi"),
                pmid=(identifiers.get("pmid") or "").rstrip("/").rsplit("/", 1)[-1] or None,
                pmcid=(identifiers.get("pmcid") or "").rstrip("/").rsplit("/", 1)[-1] or None,
                abstract=" ".join(positions[i] for i in sorted(positions)), source_uri=paper.source_uri,
                fulltext_uri=urls[0] if urls else None, fulltext_uris=urls,
                providers=(provider,), query_texts=paper.query_texts)
        data, _ = self.http.get("https://api.crossref.org/works/" + quote(normalized_doi(paper.doi), safe=""))
        record = json.loads(data)["message"]
        parser = ArticleHTML()
        parser.feed(record.get("abstract") or "")
        parser.flush()
        links = tuple(dict.fromkeys(v['URL'] for v in record.get('link', [])
                                    if v.get('content-type') == 'application/pdf' and v.get('URL')))
        return PaperCandidate(id=paper.id, title=paper.title, doi=record["DOI"],
            abstract=" ".join(t for _, t in parser.blocks), source_uri=paper.source_uri,
            fulltext_uri=links[0] if links else None, fulltext_uris=links,
            providers=(provider,), query_texts=paper.query_texts)

    def _resolve_fulltext(self, paper):
        """Resolve only selected papers, keeping metadata and access failures separate."""
        urls, versions = [], {}
        if not self.resolve_open_access or not paper.doi:
            return urls, versions
        email = os.environ.get("UNPAYWALL_EMAIL")
        if email:
            try:
                data, _ = self.http.get("https://api.unpaywall.org/v2/" + quote(paper.doi.removeprefix("https://doi.org/"), safe="") + "?" + urlencode({"email": email}))
                payload = json.loads(data)
                if not isinstance(payload, dict) or "oa_locations" not in payload:
                    raise ValueError("Invalid locations")
                locations = [payload.get("best_oa_location")] + payload["oa_locations"]
                for location in locations:
                    if not location:
                        continue
                    for url in (location.get("url_for_pdf"), location.get("url_for_landing_page")):
                        if url:
                            urls.append(url)
                            versions[url] = location.get("version") or "unknown"
                self.events.append({"operation": "resolve", "provider": "unpaywall", "paper_id": paper.id,
                                    "status": "ok", "locations": [{"url": u, "version": versions[u]} for u in dict.fromkeys(urls)]})
            except (RuntimeError, ValueError, KeyError, TypeError, AttributeError) as error:
                self.events.append({"operation": "resolve", "provider": "unpaywall", "paper_id": paper.id,
                                    "status": "failed", "error": str(error)})
        else:
            self.events.append({"operation": "resolve", "provider": "unpaywall", "paper_id": paper.id,
                                "status": "skipped", "error": "UNPAYWALL_EMAIL is not configured"})
        if not paper.pmcid:
            try:
                params = {"query": 'DOI:"' + paper.doi.replace('"', '') + '"', "format": "json", "resultType": "core", "pageSize": 1}
                data, _ = self.http.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urlencode(params))
                matches = self._europe_pmc_records(data, "DOI lookup")
                for match in matches:
                    if match.doi and match.doi.casefold() == paper.doi.removeprefix("https://doi.org/").casefold():
                        urls.extend(match.fulltext_uris)
                self.events.append({"operation": "resolve", "provider": "europe_pmc", "paper_id": paper.id,
                                    "status": "ok", "returned": len(matches)})
            except (RuntimeError, ValueError) as error:
                self.events.append({"operation": "resolve", "provider": "europe_pmc", "paper_id": paper.id,
                                    "status": "failed", "error": str(error)})
        return list(dict.fromkeys(urls)), versions

    @staticmethod
    def _xml_blocks(data):
        root = ElementTree.fromstring(data)
        article = root if root.tag == "article" else root.find(".//article")
        if article is None or article.find("body") is None:
            raise RuntimeError("XML has no article body")
        if len(normalize(" ".join(article.find("body").itertext()))) < 500:
            raise RuntimeError("XML has no usable full-text body")
        blocks = []
        def visit(node, heading):
            if node.tag == "sec":
                title = node.find("title")
                if title is not None:
                    heading = normalize(" ".join(title.itertext())) or heading
            if node.tag in {"p", "caption", "table-wrap"}:
                text = normalize(" ".join(node.itertext()))
                if text:
                    blocks.append((heading, text))
                return
            for child in node:
                visit(child, heading)
        for name in ("front", "body", "back"):
            node = article.find(name)
            if node is not None:
                visit(node, {"front": "Abstract", "body": "Body", "back": "References"}[name])
        return blocks

    def fetch_document(self, paper: PaperCandidate, *, max_characters=36000, allow_abstract=True) -> PaperDocument:
        urls = []
        if paper.pmcid:
            urls.append("https://www.ebi.ac.uk/europepmc/webservices/rest/" + quote(paper.pmcid, safe="") + "/fullTextXML")
        if paper.arxiv_id:
            urls.append("https://arxiv.org/html/" + paper.arxiv_id)
        if paper.fulltext_uri:
            urls.append(paper.fulltext_uri)
        urls.extend(paper.fulltext_uris)
        urls.append(paper.source_uri)
        urls = list(dict.fromkeys(urls))[:8]
        limitations, blocks, chosen = [], [], paper.source_uri
        versions, resolved, index = {}, False, 0
        while True:
            if index >= len(urls):
                if resolved:
                    break
                additional, versions = self._resolve_fulltext(paper)
                urls.extend(u for u in additional if u not in urls)
                urls = urls[:12]
                resolved = True
                if index >= len(urls):
                    break
            url = urls[index]
            index += 1
            try:
                data, kind = self.http.get(url)
                if data.startswith(b"%PDF") or "application/pdf" in kind:
                    try:
                        from pypdf import PdfReader
                    except ImportError:
                        raise RuntimeError("PDF reader unavailable; install the literature optional dependency") from None
                    try:
                        blocks = [(f"Page {n}", normalize(p.extract_text() or ""))
                                  for n, p in enumerate(PdfReader(BytesIO(data)).pages, 1)]
                    except Exception:
                        raise RuntimeError("PDF could not be parsed as readable text") from None
                elif "xml" in kind or url.endswith("/fullTextXML"):
                    blocks = self._xml_blocks(data)
                else:
                    parser = ArticleHTML()
                    parser.feed(data.decode("utf-8", errors="replace"))
                    parser.flush()
                    if not parser.article_seen:
                        raise RuntimeError("Page does not identify an article body")
                    blocks = parser.blocks
                    # Publisher abstract/landing pages also use <article>. Require a body section.
                    body_heading = r"\b(introduction|background|methods|materials|results|discussion|conclusion|main|正文|结果|方法)\b"
                    if sum(len(t) for h, t in parser.body_blocks if re.search(body_heading, h, re.IGNORECASE)) < 500:
                        raise RuntimeError("HTML has no identifiable full-text scientific body")
                blocks = [(h, t) for h, t in blocks if len(t) > 20]
                if sum(len(t) for _, t in blocks) < 500:
                    raise RuntimeError("Document has no usable article body")
                chosen = url
                if versions.get(url):
                    limitations.append("开放全文版本：" + versions[url] + "；来源：" + url)
                self.events.append({"operation": "download", "paper_id": paper.id, "source_uri": url,
                                    "status": "ok", "content_version": versions.get(url)})
                break
            except (RuntimeError, ValueError, OSError, ElementTree.ParseError) as error:
                blocks = []
                reason = str(error)
                limitations.append(f"正文获取失败：{reason}；{url}")
                self.events.append({"operation": "download", "paper_id": paper.id, "source_uri": url,
                                    "status": "failed", "error": reason})
        level = "full_text"
        if not blocks:
            if not allow_abstract:
                raise RuntimeError("Selected paper has no readable text within acquisition policy")
            if paper.abstract.strip():
                blocks = [("Abstract", paper.abstract)]
                level = "abstract"
                limitations.append("仅取得摘要；标题用于相关性判断，不能据此推断正文实验细节或独立验证作者结论。")
            else:
                blocks = [("Title", paper.title)]
                level = "title"
                limitations.append("仅取得标题，只能初步判断相关性，不能作为科学结论的证据。")
        if sum(len(t) for _, t in blocks) > max_characters:
            # Keep the start and end; explicitly declare that middle sections were not read.
            start_budget, end_budget = max_characters * 2 // 3, max_characters // 3
            def take(values, budget):
                selected = []
                for heading, text in values:
                    if budget <= 20:
                        break
                    excerpt = text[:budget]
                    selected.append((heading, excerpt))
                    budget -= len(excerpt)
                return selected
            blocks = take(blocks, start_budget) + list(reversed(take(list(reversed(blocks)), end_budget)))
            level = "excerpt" if level == "full_text" else level
            limitations.append("受阅读预算限制，仅提供正文开头与末尾摘录；中间部分未完整审阅。")
        sections = tuple(DocumentSection(id=f"S{n:03d}", heading=h, text=t) for n, (h, t) in enumerate(blocks, 1))
        content_hash = digest([s.model_dump(mode="json") for s in sections])
        document = PaperDocument(paper_id=paper.id, source_uri=chosen, read_level=level,
                                 sections=sections, content_sha256=content_hash,
                                 fetched_at=getattr(self.http, "fetched_at", "not_recorded"), limitations=tuple(limitations))
        self.events.append({"operation": "document", "paper_id": paper.id, "read_level": level,
                            "content_sha256": content_hash, "source_uri": chosen})
        return document
