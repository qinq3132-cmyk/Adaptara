"""Bounded public literature search and immutable, located document snapshots."""
import hashlib
import ipaddress
import json
import os
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
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
        self.last_arxiv_request = 0.0
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
        for attempt in range(self.retries + 1):
            if urlsplit(url).hostname == "export.arxiv.org":
                time.sleep(max(0, 3 - (time.monotonic() - self.last_arxiv_request)))
                self.last_arxiv_request = time.monotonic()
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    public_url(response.geturl())
                    data = response.read(self.max_bytes + 1)
                    kind = response.headers.get("Content-Type", "")
                if len(data) > self.max_bytes:
                    raise ValueError("Literature document exceeds download budget")
                break
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise RuntimeError(f"Literature HTTP request failed (status {error.code})") from None
                time.sleep(min(3, 1 + attempt))
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
            "fulltext_uri": candidate.fulltext_uri if newer else previous.fulltext_uri or candidate.fulltext_uri,
            "providers": tuple(dict.fromkeys(previous.providers + candidate.providers)),
            "query_texts": tuple(dict.fromkeys(previous.query_texts + candidate.query_texts))})
    for candidate in candidates:
        aid, _ = arxiv_parts(candidate.arxiv_id)
        keys = ["title:" + normalize(candidate.title).casefold()]
        if candidate.doi:
            keys.append("doi:" + candidate.doi.lower().removeprefix("https://doi.org/"))
        if aid:
            keys.append("arxiv:" + aid)
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
        self.heading = "Document"
        self.article_seen = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        in_article = any(a for _, a, _ in self.stack)
        article = tag in {"article", "main"} or "ltx_document" in attrs.get("class", "").split()
        excluded = tag in {"script", "style", "nav", "footer", "button"} or any(e for _, _, e in self.stack)
        if article:
            if not self.article_seen:
                self.blocks, self.parts = [], []
            self.article_seen = True
        if tag in {"p", "h1", "h2", "h3", "h4", "li", "figcaption"}:
            self.flush()
        if tag not in {"meta", "link", "br", "img", "hr", "input", "source", "wbr"}:
            self.stack.append((tag, article or in_article, excluded))

    def handle_endtag(self, tag):
        if tag in {"p", "h1", "h2", "h3", "h4", "li", "figcaption", "div"}:
            text = self.flush()
            if tag in {"h1", "h2", "h3", "h4"} and text:
                self.heading = text
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if any(e for _, _, e in self.stack):
            return
        if self.article_seen and not any(a for _, a, _ in self.stack):
            return
        self.parts.append(data)

    def flush(self):
        text = normalize(" ".join(self.parts))
        self.parts = []
        if text:
            self.blocks.append((self.heading, text))
        return text


class LiteratureTools:
    def __init__(self, http: LiteratureHTTP):
        self.http = http
        self.events = []
        self.unavailable_providers = {}

    def search(self, queries: tuple[SearchQuery, ...], providers: tuple[str, ...], *, per_query=5, max_candidates=15):
        found = []
        for query in queries:
            for provider in providers:
                if provider in self.unavailable_providers:
                    self.events.append({"operation": "search", "provider": provider, "query": query.text,
                                        "status": "skipped", "error": self.unavailable_providers[provider]})
                    continue
                try:
                    candidates = self._search(query, provider, per_query)
                    found.extend(candidates)
                    self.events.append({"operation": "search", "provider": provider, "query": query.text,
                                        "returned": len(candidates), "status": "ok"})
                except RuntimeError as error:
                    if "status 429" in str(error) or "status 403" in str(error):
                        self.unavailable_providers[provider] = "Provider rate-limited or denied access for this search run"
                    self.events.append({"operation": "search", "provider": provider, "query": query.text,
                                        "status": "failed", "error": str(error)})
        # Round-robin query coverage avoids letting the first query fill the budget.
        unique = deduplicate(found)
        ordered = []
        pools = [[c for c in unique if q.text in c.query_texts and p in c.providers] for q in queries for p in providers]
        for n in range(max((len(p) for p in pools), default=0)):
            for pool in pools:
                if n < len(pool) and pool[n].id not in {c.id for c in ordered}:
                    ordered.append(pool[n])
        return ordered[:max_candidates]

    def _search(self, query, provider, count):
        if provider == "crossref":
            params = {"query.bibliographic": query.text, "rows": count}
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
                        providers=(provider,), query_texts=(query.text,)))
                return result
            except (ValueError, KeyError, TypeError, AttributeError):
                raise RuntimeError("Crossref returned invalid metadata records") from None
        if provider == "semantic_scholar":
            params = {"query": query.text, "limit": count,
                      "fields": "title,abstract,authors,year,url,externalIds,openAccessPdf"}
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
                return [PaperCandidate(id=candidate_id(p["title"]), title=p["title"], abstract=p.get("abstract") or "",
                            authors=tuple(a["name"] for a in p.get("authors", [])), year=p.get("year"),
                            doi=(p.get("externalIds") or {}).get("DOI"), arxiv_id=(p.get("externalIds") or {}).get("ArXiv"),
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
        data, _ = self.http.get("https://export.arxiv.org/api/query?" + urlencode(params))
        ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
        try:
            root = ElementTree.fromstring(data)
        except ElementTree.ParseError:
            raise RuntimeError("arXiv returned an invalid metadata response") from None
        if root.tag != "{http://www.w3.org/2005/Atom}feed":
            raise RuntimeError("arXiv response is not an Atom feed")
        result = []
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

    def fetch_document(self, paper: PaperCandidate, *, max_characters=36000, allow_abstract=True) -> PaperDocument:
        urls = []
        if paper.arxiv_id:
            urls.append("https://arxiv.org/html/" + paper.arxiv_id)
        if paper.fulltext_uri:
            urls.append(paper.fulltext_uri)
        if not urls:
            urls.append(paper.source_uri)
        limitations, blocks, chosen = [], [], paper.source_uri
        for url in dict.fromkeys(urls):
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
                else:
                    parser = ArticleHTML()
                    parser.feed(data.decode("utf-8", errors="replace"))
                    parser.flush()
                    if not parser.article_seen:
                        raise RuntimeError("Page does not identify an article body")
                    blocks = parser.blocks
                blocks = [(h, t) for h, t in blocks if len(t) > 20]
                if sum(len(t) for _, t in blocks) < 500:
                    raise RuntimeError("Document has no usable article body")
                chosen = url
                break
            except (RuntimeError, ValueError, OSError) as error:
                blocks = []
                limitations.append(f"正文获取失败：{type(error).__name__}；{url}")
        level = "full_text"
        if not blocks:
            if not allow_abstract or not paper.abstract.strip():
                raise RuntimeError("Selected paper has no readable text within acquisition policy")
            blocks = [("Abstract", paper.abstract)]
            level = "abstract"
            limitations.append("仅取得摘要；不能据此推断正文实验细节或独立验证作者结论。")
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
