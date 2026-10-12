"""Literature agents produce a single grounded ScientificState, with auditable checkpoints."""
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from .agents.literature import (LiteratureSearchAgent, LiteratureScreeningAgent, LiteratureReadingAgent,
                               EvidenceVerificationAgent, LiteratureSynthesisAgent)
from .hypothesis_generation import ReasoningConfig, make_reasoning_backend
from .literature_schemas import CoverageCategory, LiteratureOverview, LiteratureProvider, PaperCandidate, PaperSelection, SearchQuery
from .provenance import capture_source, runtime_versions
from .schemas import Claim, Evidence, Paper, Record
from .state import ScientificState
from .tools.experiment_tracker import ResearchStore
from .tools.literature import LiteratureHTTP, LiteratureTools, arxiv_parts, deduplicate, digest, normalize

ROLES = {"literature_search", "literature_screening", "literature_reading", "evidence_verification",
         "literature_synthesis", "hypothesis_review", "hypothesis_revision", "experiment_design"}


class AgentSettings(Record):
    model: str | None = None
    auth_mode: Literal["opencode", "codex", "chatgpt"] | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    timeout_seconds: float | None = Field(default=None, gt=0, le=600)

    def runtime(self, common: ReasoningConfig):
        data = common.model_dump()
        for name, target in (("model", "model"), ("auth_mode", "codex_auth_mode"),
                             ("reasoning_effort", "codex_reasoning_effort"), ("timeout_seconds", "timeout_seconds")):
            if getattr(self, name) is not None:
                data[target] = getattr(self, name)
        if self.auth_mode == "chatgpt":
            data.update(backend="codex", codex_base_url=None, codex_provider_id="openai", codex_credential_provider=None)
        return ReasoningConfig.model_validate(data)


class LiteratureConfig(Record):
    providers: tuple[LiteratureProvider, ...] = ("semantic_scholar", "arxiv", "crossref", "europe_pmc", "openalex")
    max_queries: int = Field(default=3, ge=1, le=6)
    results_per_query: int = Field(default=5, ge=1, le=50)
    max_pages_per_query: int = Field(default=1, ge=1, le=10)
    max_candidates: int = Field(default=15, ge=1, le=500)
    max_candidates_per_round: int = Field(default=40, ge=1, le=200)
    max_selected_papers: int = Field(default=3, ge=1, le=50)
    reading_concurrency: int = Field(default=1, ge=1, le=8)
    min_candidates: int = Field(default=1, ge=1, le=500)
    min_selected_papers: int = Field(default=1, ge=1, le=50)
    max_search_rounds: int = Field(default=1, ge=1, le=8)
    min_search_rounds: int = Field(default=1, ge=1, le=8)
    required_categories: tuple[CoverageCategory, ...] = ()
    saturation_rounds: int = Field(default=0, ge=0, le=4)
    max_new_relevant_per_round: int = Field(default=2, ge=0, le=20)
    screening_batch_size: int = Field(default=30, ge=1, le=50)
    max_metadata_requests: int = Field(default=0, ge=0, le=100)
    http_retries: int = Field(default=2, ge=0, le=5)
    max_characters_per_paper: int = Field(default=36000, ge=1000, le=100000)
    allow_abstract: bool = True
    resolve_open_access: bool = True
    http_timeout_seconds: float = Field(default=25, gt=0, le=60)
    max_revision_rounds: int = Field(default=1, ge=0, le=3)
    max_supplement_rounds: int = Field(default=1, ge=0, le=2)
    roles: dict[str, AgentSettings] = Field(default_factory=dict)

    @model_validator(mode="after")
    def bounded(self):
        if not self.providers or len(self.providers) != len(set(self.providers)):
            raise ValueError("Choose distinct literature providers")
        if set(self.roles) - ROLES:
            raise ValueError("Unknown literature agent role")
        if self.max_selected_papers > self.max_candidates:
            raise ValueError("Paper selection budget exceeds candidate budget")
        if self.min_candidates > self.max_candidates or self.min_selected_papers > self.max_selected_papers:
            raise ValueError("Literature coverage target exceeds the configured budget")
        if self.min_search_rounds > self.max_search_rounds or (self.saturation_rounds and
                                                             self.saturation_rounds >= self.max_search_rounds):
            raise ValueError("Search stopping requirement exceeds round budget")
        if len(self.required_categories) != len(set(self.required_categories)) or len(self.required_categories) > self.max_selected_papers:
            raise ValueError("Choose distinct coverage categories within the reading budget")
        return self


@dataclass
class PaperReadingResult:
    paper: Paper | None = None
    evidence: list[Evidence] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    exports: dict = field(default_factory=dict)
    attempts: list = field(default_factory=list)
    audit: dict | None = None
    error: Exception | None = None


class LiteraturePipeline:
    def __init__(self, common: ReasoningConfig, config: LiteratureConfig, artifact_root: Path,
                 *, backend_factory=make_reasoning_backend, tools=None, emit=print):
        self.common, self.config, self.artifact_root = common, config, Path(artifact_root)
        self.backend_factory, self.emit = backend_factory, emit
        self.tools = tools or LiteratureTools(LiteratureHTTP(self.artifact_root / "literature_http_cache",
                                                            timeout=config.http_timeout_seconds, retries=config.http_retries),
                                             resolve_open_access=config.resolve_open_access)
        self.attempts, self.exports = [], {}

    def agent(self, cls):
        settings = self.config.roles.get(cls.role, AgentSettings()).runtime(self.common)
        return cls(self.backend_factory, settings, self.artifact_root / "research_agent_cache", emit=self.emit)

    def call(self, cls, inputs, validate=lambda value: None):
        agent = self.agent(cls)
        try:
            return agent.run(inputs, validate)
        finally:
            self.attempts.extend(agent.attempts)

    def save_discovery_checkpoint(self, state: ScientificState, path: Path | None = None):
        """Save screened inputs without claiming that literature evidence has been read."""
        if state.problem_definition is None:
            raise ValueError("A discovery checkpoint requires a confirmed problem definition")
        required = {"search_results.json", "selection.json", "retrieval_summary.json", "selection_budget.json"}
        if not required <= self.exports.keys():
            raise ValueError("Discovery must finish before saving its checkpoint")
        payload = {"schema_version": 1, "stage": "literature_screening",
                   "input_state_sha256": digest(state.model_dump(mode="json")),
                   "research_definition_sha256": digest(state.problem_definition.model_dump(mode="json")),
                   "reasoning": self.common.model_dump(mode="json"), "literature": self.config.model_dump(mode="json"),
                   "exports": self.exports, "attempts": [{**a, 'reused': False} for a in self.attempts]}
        from .tools.experiment_tracker import json_bytes
        checksum = digest(payload)
        data = json_bytes({"payload": payload, "sha256": checksum})
        path = Path(path) if path is not None else self.artifact_root / 'literature_discoveries' / f'discovery_{checksum}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("Refusing to overwrite a different discovery checkpoint")
            return path
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        return path

    def resume_discovery(self, state: ScientificState, path: Path):
        saved = json.loads(Path(path).read_bytes())
        payload = saved["payload"]
        if digest(payload) != saved["sha256"]:
            raise ValueError("Discovery checkpoint checksum mismatch")
        if payload.get("schema_version") != 1 or payload.get("stage") != "literature_screening":
            raise ValueError("Unsupported discovery checkpoint")
        if (state.problem_definition is None or
                payload['research_definition_sha256'] != digest(state.problem_definition.model_dump(mode="json")) or
                payload['input_state_sha256'] != digest(state.model_dump(mode="json"))):
            raise ValueError("Discovery checkpoint belongs to a different research definition or input state")
        recorded_config = LiteratureConfig.model_validate(payload['literature'])
        reading_settings = {'roles', 'max_characters_per_paper', 'allow_abstract', 'resolve_open_access',
                            'http_timeout_seconds', 'http_retries', 'max_revision_rounds', 'max_supplement_rounds',
                            'reading_concurrency'}
        if recorded_config.model_dump(exclude=reading_settings) != self.config.model_dump(exclude=reading_settings):
            raise ValueError("Discovery checkpoint requires its recorded literature configuration")
        exports = payload['exports']
        pool = [PaperCandidate.model_validate(p) for p in exports['search_results.json']]
        selection = PaperSelection.model_validate(exports['selection.json'])
        ids, decided = [p.id for p in pool], [d.paper_id for d in selection.decisions]
        if (not pool or len(ids) != len(set(ids)) or len(decided) != len(set(decided)) or set(ids) != set(decided)
                or len(deduplicate(pool)) != len(pool) or set(ids) & {p.id for p in state.papers}):
            raise ValueError("Discovery checkpoint has inconsistent or duplicate candidates")
        if len(pool) > self.config.max_candidates or sum(d.selected for d in selection.decisions) > self.config.max_selected_papers:
            raise ValueError("Discovery checkpoint exceeds the recorded candidate or reading budget")
        summary = exports['retrieval_summary.json']
        if summary['unique_candidates'] != len(pool) or summary['selected_papers'] != sum(d.selected for d in selection.decisions):
            raise ValueError("Discovery checkpoint summary disagrees with its screening decisions")
        if any(d.selected and d.category == 'unrelated' for d in selection.decisions):
            raise ValueError("Discovery checkpoint selected an unrelated candidate")
        self.exports = exports
        self.attempts = [{**a, "reused": True} for a in payload['attempts']]
        self.tools.events = list(exports.get('retrieval.json', []))
        self.tools.unavailable_providers = {
            e['provider']: 'Provider rate-limited or denied access in the saved discovery'
            for e in self.tools.events if e.get('status') == 'failed'
            and any(f'status {code}' in e.get('error', '') for code in (429, 403))}
        self.exports['discovery_resume.json'] = {"checkpoint_sha256": saved['sha256'],
                                               "reasoning": payload['reasoning'],
                                               "literature": recorded_config.model_dump(mode='json'), "reused_screening": True}
        self.emit(f"Literature screening resumed: {len(pool)} candidates; {summary['selected_papers']} selected")
        return pool, selection

    def discover(self, state: ScientificState, *, queries: tuple[SearchQuery, ...] | None = None):
        state = ScientificState.model_validate_json(state.model_dump_json())
        if state.problem_definition is None:
            raise ValueError("Literature research requires a confirmed problem definition")
        self.attempts, self.exports = [], {}
        self.tools.events = []
        self.tools.unavailable_providers = {}
        definition = state.problem_definition.model_dump(mode="json")
        pool, history, used_queries = [], [], []
        decisions, screened, screening_limits = {}, {}, []
        metadata_used, low_yield_rounds = 0, 0
        previous_relevant = set()
        selection = None
        stop_reason = "search_round_budget_exhausted"
        existing_titles = {normalize(p.title).casefold() for p in state.papers}
        existing_dois = {p.doi.casefold() for p in state.papers if p.doi}
        existing_arxiv = {arxiv_parts(p.arxiv_id)[0] for p in state.papers if p.arxiv_id}
        def fresh_only(candidates):
            return [p for p in candidates if normalize(p.title).casefold() not in existing_titles
                    and (not p.doi or p.doi.casefold() not in existing_dois)
                    and (not p.arxiv_id or arxiv_parts(p.arxiv_id)[0] not in existing_arxiv)]

        def screen_changed():
            changed = [p for p in pool if screened.get(p.id) != digest(p.model_dump(mode="json"))]
            for start in range(0, len(changed), self.config.screening_batch_size):
                batch = changed[start:start + self.config.screening_batch_size]
                candidate_ids = {p.id for p in batch}
                def check_selection(value):
                    ids = [d.paper_id for d in value.decisions]
                    if set(ids) != candidate_ids or len(ids) != len(candidate_ids):
                        raise ValueError("Screening must decide each candidate exactly once")
                    if any(d.selected and d.category == "unrelated" for d in value.decisions):
                        raise ValueError("An unrelated candidate cannot be eligible for reading")
                    if any(d.needs_metadata and next(p for p in batch if p.id == d.paper_id).abstract.strip()
                           for d in value.decisions):
                        raise ValueError("Metadata repair is only for candidates without an abstract")
                result = self.call(LiteratureScreeningAgent, {
                    "research_definition": definition, "candidates": [p.model_dump(mode="json") for p in batch],
                    "max_selected": len(batch), "target_selected": self.config.min_selected_papers,
                    "global_reading_budget": self.config.max_selected_papers,
                    "required_categories": list(self.config.required_categories),
                    "selection_policy": "逐篇判断是否值得阅读，不按数量目标截断；程序另行分配阅读预算"}, check_selection)
                decisions.update({d.paper_id: d for d in result.decisions})
                screened.update({p.id: digest(p.model_dump(mode="json")) for p in batch})
                screening_limits.extend(result.coverage_limitations)

        def allocate():
            papers = {p.id: p for p in pool}
            def ready(d):
                return bool(papers[d.paper_id].abstract.strip()) and not d.needs_metadata
            eligible = sorted((d for d in decisions.values() if d.selected),
                              key=lambda d: (not ready(d), -d.priority, d.paper_id))
            chosen = []
            # Reserve a readable paper for each requested direction before filling by priority.
            for category in self.config.required_categories:
                match = next((d for d in eligible if ready(d) and d.category == category), None)
                if match:
                    chosen.append(match.paper_id)
            for decision in eligible:
                if len(chosen) >= self.config.max_selected_papers:
                    break
                if decision.paper_id not in chosen:
                    chosen.append(decision.paper_id)
            coverage = {c: sum(d.paper_id in chosen and ready(d) and d.category == c for d in eligible)
                        for c in self.config.required_categories}
            pending = [d.paper_id for d in decisions.values()
                       if (d.selected or d.needs_metadata) and not ready(d)]
            missing = [c for c, count in coverage.items() if not count]
            limits = list(dict.fromkeys(screening_limits))
            if missing:
                limits.append("缺少可依据摘要判断的方向：" + ", ".join(missing))
            if pending:
                limits.append(f"{len(pending)} 篇相关性待定候选缺少摘要；标题不构成科研证据。")
            selected = PaperSelection(decisions=tuple(decisions[p.id].model_copy(update={"selected": p.id in chosen})
                                                      for p in pool), coverage_limitations=tuple(limits))
            relevant = {d.paper_id for d in eligible if ready(d)}
            return selected, coverage, missing, pending, relevant

        for round_number in range(self.config.max_search_rounds):
            forbidden_queries = {normalize(q.text).casefold() for q in used_queries}
            def check_plan(plan):
                if len(plan.queries) > self.config.max_queries:
                    raise ValueError("Search plan exceeds max_queries")
                if len({normalize(q.text).casefold() for q in plan.queries}) != len(plan.queries):
                    raise ValueError("Search queries must be distinct")
                if any(normalize(q.text).casefold() in forbidden_queries for q in plan.queries):
                    raise ValueError("Recovery queries must differ from already executed queries")
                for query in plan.queries:
                    if set(query.providers) - set(self.config.providers) or len(query.providers) != len(set(query.providers)):
                        raise ValueError("Query must choose distinct configured providers")
            if round_number == 0 and queries is not None:
                if not queries or len(queries) > self.config.max_queries:
                    raise ValueError("Supplementary queries exceed search budget")
                for query in queries:
                    if set(query.providers) - set(self.config.providers) or len(query.providers) != len(set(query.providers)):
                        raise ValueError("Supplementary query must choose distinct configured providers")
                plan_data = {"queries": [q.model_dump(mode="json") for q in queries], "origin": "hypothesis_review"}
                round_queries = queries
            else:
                plan = self.call(LiteratureSearchAgent, {"research_definition": definition,
                         "max_queries": self.config.max_queries, "phase": "initial" if round_number == 0 else "recovery",
                         "available_providers": list(self.config.providers),
                         "unavailable_providers": dict(self.tools.unavailable_providers),
                         "provider_guidance": {"europe_pmc": "生命科学、神经科学、生物医学；包含 PubMed 和 PMC",
                                               "openalex": "跨学科论文、开放版本链接",
                                               "semantic_scholar": "跨学科学术索引", "crossref": "跨学科 DOI 元数据",
                                               "arxiv": "预印本，含计算机科学、数学、物理等"},
                         "acquisition_policy": "优先获取全文；不可用时依据标题和摘要判断相关性，不因全文不可用排除论文",
                         "coverage_targets": {"min_candidates": self.config.min_candidates,
                                              "min_selected_papers": self.config.min_selected_papers,
                                              "required_categories": list(self.config.required_categories)},
                         "remaining_candidate_budget": self.config.max_candidates - len(pool),
                         "previous_queries": [q.model_dump(mode="json") for q in used_queries],
                         "feedback": history}, check_plan)
                round_queries, plan_data = plan.queries, plan.model_dump(mode="json")
            if round_number == 0:
                self.exports["search_plan.json"] = plan_data
            used_queries.extend(round_queries)
            self.emit(f"Literature tools: searching academic sources (round {round_number + 1})")
            before = len(self.tools.events)
            found = self.tools.search(round_queries, self.config.providers, per_query=self.config.results_per_query,
                                      max_candidates=min(self.config.max_candidates_per_round, self.config.max_candidates - len(pool)),
                                      max_pages=self.config.max_pages_per_query, exclude_ids={p.id for p in pool})
            previous_ids = {paper.id for paper in pool}
            pool = deduplicate(pool + fresh_only(found))[:self.config.max_candidates]
            if pool:
                screen_changed()
                repair = sorted((p for p in pool if not p.abstract.strip()
                                 and (decisions[p.id].selected or decisions[p.id].needs_metadata)),
                                key=lambda p: -decisions[p.id].priority)
                if repair and metadata_used < self.config.max_metadata_requests:
                    updated, used = self.tools.enrich_candidates(repair, self.config.providers,
                                         max_requests=self.config.max_metadata_requests - metadata_used)
                    metadata_used += used
                    replacements = {p.id: p for p in updated}
                    pool = [replacements.get(p.id, p) for p in pool]
                    screen_changed()
                selection, coverage, missing, pending, relevant = allocate()
            else:
                coverage = {c: 0 for c in self.config.required_categories}
                missing, pending, relevant = list(self.config.required_categories), [], set()
            selected_count = sum(d.selected for d in selection.decisions) if selection else 0
            new_relevant = len(relevant - previous_relevant)
            retrieval_ok = any(e.get('operation') == 'search' and e.get('status') == 'ok'
                               for e in self.tools.events[before:])
            low_yield_rounds = (low_yield_rounds + 1 if round_number > 0 and retrieval_ok and
                               new_relevant <= self.config.max_new_relevant_per_round else 0)
            previous_relevant = relevant
            history.append({"round": round_number + 1, "plan": plan_data,
                            "retrieval_events": list(self.tools.events[before:]), "new_candidates": sum(p.id not in previous_ids for p in pool),
                            "candidate_count": len(pool), "selected_count": selected_count,
                            "eligible_count": sum(d.selected for d in decisions.values()),
                            "new_relevant_candidates": new_relevant, "low_yield_rounds": low_yield_rounds,
                            "successful_search_this_round": retrieval_ok,
                            "category_coverage": coverage, "missing_categories": missing,
                            "pending_metadata_ids": pending,
                            "screening": selection.model_dump(mode="json") if selection else None})
            ready_selected = sum(d.selected and d.paper_id in relevant for d in selection.decisions) if selection else 0
            if (len(pool) >= self.config.min_candidates and ready_selected >= self.config.min_selected_papers
                    and not missing and round_number + 1 >= self.config.min_search_rounds
                    and low_yield_rounds >= self.config.saturation_rounds):
                stop_reason = "coverage_targets_met"
                break
            if len(pool) >= self.config.max_candidates:
                stop_reason = "candidate_budget_exhausted"
                break
            self.emit("Literature search continues: check direction coverage and newly relevant papers")
        selected_count = sum(d.selected for d in selection.decisions) if selection else 0
        self.exports.update({"search_history.json": history,
             "search_results.json": [p.model_dump(mode="json") for p in pool], "retrieval.json": self.tools.events,
             "retrieval_summary.json": {"status": "target_met" if stop_reason == "coverage_targets_met" else "partial",
                 "stop_reason": stop_reason, "search_rounds": len(history), "queries_executed": len(used_queries),
                 "unique_candidates": len(pool), "selected_papers": selected_count,
                 "raw_search_records": sum(e.get('returned', 0) for e in self.tools.events
                                           if e.get('operation') == 'search' and e.get('status') == 'ok'),
                 "budgets": {"max_candidates": self.config.max_candidates,
                             "max_selected_papers": self.config.max_selected_papers,
                             "max_pages_per_query": self.config.max_pages_per_query,
                             "max_search_rounds": self.config.max_search_rounds},
                 "eligible_papers": sum(d.selected for d in decisions.values()),
                 "category_coverage": coverage, "missing_categories": missing,
                 "pending_metadata_ids": pending, "metadata_requests": metadata_used,
                 "targets": {"min_candidates": self.config.min_candidates, "min_selected_papers": self.config.min_selected_papers,
                             "required_categories": list(self.config.required_categories)},
                 "source_failures": [e for e in self.tools.events if e.get("status") == "failed"]}})
        if not pool:
            raise RuntimeError("Literature search returned no new readable candidate records after recovery; no evidence was invented")
        if selection is None:
            raise RuntimeError("No literature screening decision is available")
        self.exports["selection.json"] = selection.model_dump(mode="json")
        self.exports["screening_decisions.json"] = [decisions[p.id].model_dump(mode="json") for p in pool]
        self.exports["selection_budget.json"] = {
            "reading_budget": self.config.max_selected_papers,
            "deferred_paper_ids": [d.paper_id for d in decisions.values() if d.selected
                                   and not next(s for s in selection.decisions if s.paper_id == d.paper_id).selected],
            "note": "预算暂缓不等于不相关；原始筛选判断见 screening_decisions.json"}
        return pool, selection

    def read_paper(self, candidate, document, definition):
        """Read and verify one paper using isolated sessions and local result buffers."""
        result = PaperReadingResult()
        def call(cls, inputs, validate=lambda value: None):
            agent = self.agent(cls)
            agent.emit = lambda message: self.emit(f"{candidate.id}: {message}")
            try:
                return agent.run(inputs, validate)
            finally:
                result.attempts.extend(agent.attempts)
        try:
            if document.read_level == "title":
                result.paper = Paper(id=candidate.id, title=candidate.title, source_uri=document.source_uri,
                                        doi=candidate.doi, arxiv_id=candidate.arxiv_id,
                                        content_sha256=document.content_sha256, read_level="title")
                result.limitations.extend(document.limitations)
                result.audit = {"paper_id": candidate.id, "reading": None, "checks": [],
                                         "reason": "仅标题用于相关性判断，不提取科研证据"}
                return result
            sections = {s.id: s for s in document.sections}
            def check_reading(reading):
                errors = []
                for finding_index, finding in enumerate(reading.findings, 1):
                    if finding.section_id not in sections:
                        errors.append(f"Finding {finding_index} references unknown section_id {finding.section_id}")
                    elif normalize(finding.quote) not in normalize(sections[finding.section_id].text):
                        matching = [sid for sid, section in sections.items()
                                    if normalize(finding.quote) in normalize(section.text)]
                        location = ' Exact quotation found in: ' + ', '.join(matching) if matching else ''
                        if not matching:
                            quote = normalize(finding.quote)
                            source = normalize(sections[finding.section_id].text)
                            anchor = SequenceMatcher(None, quote, source, autojunk=False).find_longest_match()
                            if anchor.size >= 15:
                                start = max(0, anchor.b - anchor.a - 20)
                                context = source[start:start + min(len(quote) + 80, 700)]
                                location = ' Nearby ORIGINAL source text (verify support before quoting): ' + json.dumps(context, ensure_ascii=False)
                        errors.append(f"Finding {finding_index}, {finding.section_id}: quoted evidence is not a verbatim contiguous span.{location}")
                    if document.read_level in {"abstract", "excerpt"} and finding.strength != "weak":
                        errors.append(f"Finding {finding_index}: {document.read_level} evidence must be weak; received {finding.strength}")
                if errors:
                    raise ValueError("Fix ALL listed errors. Copy exact quotations, preserving repeated numbers, words and extraction artifacts; "
                                     "do not repair or paraphrase quotes. For abstract/excerpt material EVERY finding must have strength=weak. "
                                     + '; '.join(errors))
            reading = call(LiteratureReadingAgent, {"research_definition": definition,
                                "paper": candidate.model_dump(mode="json"), "document": document.model_dump(mode="json")},
                                check_reading)
            result.exports[f"reading_{candidate.id}.json"] = reading.model_dump(mode="json")
            findings = [{"finding_id": f"F{n:03d}", **f.model_dump(mode="json")}
                        for n, f in enumerate(reading.findings, 1)]
            _, version = arxiv_parts(candidate.arxiv_id)
            result.paper = Paper(id=candidate.id, title=candidate.title, source_uri=document.source_uri,
                                    doi=candidate.doi, arxiv_id=candidate.arxiv_id, version=version,
                                    content_sha256=document.content_sha256, read_level=document.read_level)
            result.limitations.extend(document.limitations + reading.limitations)
            if not findings:
                result.audit = {"paper_id": candidate.id, "reading": reading.model_dump(mode="json"), "checks": []}
                return result
            def check_verification(checks):
                ids = [c.finding_id for c in checks.checks]
                if set(ids) != {f["finding_id"] for f in findings} or len(ids) != len(findings):
                    raise ValueError("Verification must check every finding exactly once")
            checks = call(EvidenceVerificationAgent, {"research_definition": definition,
                               "paper": candidate.model_dump(mode="json"), "document": document.model_dump(mode="json"),
                               "findings": findings}, check_verification)
            result.exports[f"verification_{candidate.id}.json"] = checks.model_dump(mode="json")
            result.audit = {"paper_id": candidate.id, "reading": reading.model_dump(mode="json"),
                                     "checks": checks.model_dump(mode="json")}
            accepted = {c.finding_id for c in checks.checks if c.verdict == "accepted"}
            for n, finding in enumerate(reading.findings, 1):
                if f"F{n:03d}" not in accepted:
                    continue
                identifier = "EVL_" + digest({"paper": candidate.id, "finding": finding.model_dump(mode="json")})[:16]
                result.evidence.append(Evidence(id=identifier, source_type="paper", source_id=candidate.id,
                    claim=finding.claim, strength="weak" if document.read_level == "abstract" else finding.strength,
                    limitations=tuple(dict.fromkeys(finding.limitations + document.limitations)),
                    quote=finding.quote, locator=f"{finding.section_id} | {sections[finding.section_id].heading}",
                    content_sha256=document.content_sha256, conditions=finding.conditions, verification="accepted"))
            return result
        except (ValueError, RuntimeError, OSError) as error:
            result.error = error
            result.audit = {"paper_id": candidate.id, "reading": result.exports.get(f"reading_{candidate.id}.json"),
                            "checks": [], "status": "failed", "error_type": type(error).__name__,
                            "error_code": getattr(error, "code", None)}
            return result

    def run(self, state: ScientificState, *, queries: tuple[SearchQuery, ...] | None = None,
            discovery_checkpoint: Path | None = None):
        if discovery_checkpoint is not None and queries is not None:
            raise ValueError("Choose a discovery checkpoint or supplementary queries, not both")
        fresh, selection = (self.resume_discovery(state, discovery_checkpoint) if discovery_checkpoint is not None
                            else self.discover(state, queries=queries))
        if discovery_checkpoint is None:
            checkpoint_path = self.save_discovery_checkpoint(state)
            self.exports['discovery_checkpoint.json'] = json.loads(checkpoint_path.read_bytes())
            self.emit(f"Literature screening checkpoint saved: {checkpoint_path}")
        definition = state.problem_definition.model_dump(mode="json")
        chosen_ids = {d.paper_id for d in selection.decisions if d.selected}
        chosen = [p for p in fresh if p.id in chosen_ids]
        if not chosen:
            raise RuntimeError("Screening found no relevant papers; no literature state was committed")
        new_papers, new_evidence, limitations, extraction_audit = [], [], [], []
        self.exports['extraction_audit.json'] = extraction_audit
        pending = []
        with ThreadPoolExecutor(max_workers=self.config.reading_concurrency) as executor:
            for index, candidate in enumerate(chosen, 1):
                self.emit(f"Literature acquisition: {index}/{len(chosen)} {candidate.id} {candidate.title}")
                try:
                    # Acquisition stays on the main thread: provider pacing/cache state are not shared with workers.
                    document = self.tools.fetch_document(candidate, max_characters=self.config.max_characters_per_paper,
                                                         allow_abstract=self.config.allow_abstract)
                except RuntimeError as error:
                    limitations.append(f"{candidate.id} 未能取得可读材料：{error}")
                    continue
                self.exports[f"document_{candidate.id}.json"] = document.model_dump(mode="json")
                self.emit(f"Literature material: {candidate.id} {document.read_level}")
                if self.config.reading_concurrency == 1:
                    result = self.read_paper(candidate, document, definition)
                    pending.append(result)
                    if result.error is not None:
                        break
                else:
                    pending.append(executor.submit(self.read_paper, candidate, document, definition))
            # Wait for all submitted jobs, then merge in input order regardless of completion order.
            results = [job if isinstance(job, PaperReadingResult) else job.result() for job in pending]
        self.exports['paper_execution.json'] = {
            'reading_concurrency': self.config.reading_concurrency,
            'reading_model': self.config.roles.get('literature_reading', AgentSettings()).runtime(self.common).model,
            'verification_model': self.config.roles.get('evidence_verification', AgentSettings()).runtime(self.common).model,
            'tasks': [r.audit for r in results]}
        for result in results:
            self.attempts.extend(result.attempts)
            self.exports.update(result.exports)
            limitations.extend(result.limitations)
            if result.paper is not None:
                new_papers.append(result.paper)
            new_evidence.extend(result.evidence)
            if result.audit is not None:
                extraction_audit.append(result.audit)
        errors = [r.error for r in results if r.error is not None]
        if errors:
            # Successful jobs remain cached; no partially verified scientific state is committed.
            raise errors[0]
        self.exports["extraction_audit.json"] = extraction_audit
        self.exports["retrieval.json"] = self.tools.events
        summary = self.exports["retrieval_summary.json"]
        summary.update(documents_read=len(new_papers), papers_with_accepted_evidence=len({e.source_id for e in new_evidence}),
                       accepted_findings=len(new_evidence),
                       read_levels={level: sum(p.read_level == level for p in new_papers)
                                    for level in ("full_text", "excerpt", "abstract", "title")})
        accepted_papers = {e.source_id for e in new_evidence}
        summary['verified_category_coverage'] = {
            category: sum(d.paper_id in accepted_papers and d.category == category for d in selection.decisions)
            for category in self.config.required_categories}
        missing_verified = [c for c, count in summary['verified_category_coverage'].items() if not count]
        if summary['papers_with_accepted_evidence'] < self.config.min_selected_papers:
            summary['discovery_stop_reason'] = summary['stop_reason']
            summary.update(status='partial', stop_reason='readable_evidence_target_not_met')
            limitations.append(f"可核对证据覆盖不足：有 accepted 证据的论文 {summary['papers_with_accepted_evidence']}/{self.config.min_selected_papers}；不能把检索或筛选数量当作实际证据覆盖。")
        elif missing_verified:
            summary['discovery_stop_reason'] = summary['stop_reason']
            summary.update(status='partial', stop_reason='verified_coverage_incomplete')
        if missing_verified:
            limitations.append("缺少已核对证据的方向：" + ", ".join(missing_verified))
        if summary["status"] != "target_met":
            limitations.append(f"检索覆盖目标未达成：候选 {len(fresh)}/{self.config.min_candidates}；入选 {len(chosen)}/{self.config.min_selected_papers}；{summary['stop_reason']}。")
        deferred = self.exports['selection_budget.json']['deferred_paper_ids']
        if deferred:
            limitations.append(f"另有 {len(deferred)} 篇值得阅读的候选因本轮阅读预算暂缓，尚未核查其证据。")
        if not new_evidence:
            raise RuntimeError("No findings passed evidence verification; no knowledge was invented")
        accepted_evidence = [e for e in state.evidence if e.verification == "accepted"] + new_evidence
        evidence_ids = {e.id for e in accepted_evidence}
        def check_synthesis(synthesis):
            for claim in synthesis.claims:
                if set(claim.source_ids) - evidence_ids:
                    raise ValueError("Synthesis claim cites unknown or unaccepted evidence")
            for item in synthesis.gaps + synthesis.conflicts:
                if set(item.evidence_ids) - evidence_ids:
                    raise ValueError("Literature observations cite unknown evidence")
            statements = [normalize(c.statement).casefold() for c in synthesis.claims]
            if len(statements) != len(set(statements)):
                raise ValueError("Synthesis claims must be distinct")
        synthesis = self.call(LiteratureSynthesisAgent, {"research_definition": definition,
                              "papers": [p.model_dump(mode="json") for p in state.papers + tuple(new_papers)],
                              "accepted_evidence": [e.model_dump(mode="json") for e in accepted_evidence],
                              "previous_knowledge": [c.model_dump(mode="json") for c in state.knowledge],
                              "coverage": list(selection.coverage_limitations) + limitations,
                              "retrieval_events": self.tools.events}, check_synthesis)
        prior = {normalize(c.statement).casefold() for c in state.knowledge}
        claims = tuple(Claim(id="CLL_" + digest(c.model_dump(mode="json"))[:16], **c.model_dump())
                       for c in synthesis.claims if normalize(c.statement).casefold() not in prior)
        retrieval_limits = tuple("检索失败：" + e["provider"] + "；" + e["query"]
                                 for e in self.tools.events if e.get("status") == "failed" and e.get("operation") == "search")
        overview = LiteratureOverview(summary=synthesis.summary, gaps=synthesis.gaps, conflicts=synthesis.conflicts,
                    coverage_limitations=tuple(dict.fromkeys(synthesis.coverage_limitations + selection.coverage_limitations
                                                             + tuple(limitations) + retrieval_limits)))
        final = state.revised(schema_version=3, papers=state.papers + tuple(new_papers),
                              evidence=state.evidence + tuple(new_evidence), knowledge=state.knowledge + claims,
                              literature=overview)
        self.exports.update({"extraction_audit.json": extraction_audit,
                             "retrieval.json": self.tools.events, "synthesis.json": synthesis.model_dump(mode="json")})
        return final


def literature_report(state, retrieval_summary=None):
    lines = ["# 文献调研报告", ""]
    if retrieval_summary:
        s = retrieval_summary
        status = '达到本轮配置的覆盖目标' if s['status'] == 'target_met' else '覆盖不足'
        lines += [f"检索候选 {s['unique_candidates']} 篇 → 入选 {s['selected_papers']} 篇 → 阅读 {s.get('documents_read', 0)} 篇 → 有核查证据 {s.get('papers_with_accepted_evidence', 0)} 篇。{status}；不代表穷尽领域文献。", ""]
        if s.get('eligible_papers') is not None:
            lines += [f"值得阅读 {s['eligible_papers']} 篇；其中 {s['selected_papers']} 篇排入本轮预算。缺摘要待定 {len(s.get('pending_metadata_ids', []))} 篇。", ""]
        if s.get("read_levels"):
            counts = s["read_levels"]
            lines += [f"材料范围：全文 {counts['full_text']}；正文摘录 {counts['excerpt']}；摘要 {counts['abstract']}；仅标题 {counts['title']}。", ""]
    lines += [state.literature.summary, "", "## 原文证据", ""]
    papers = {p.id: p for p in state.papers}
    for e in state.evidence:
        if e.verification != "accepted":
            continue
        p = papers[e.source_id]
        lines += [f"### {e.id}", "", f"[{p.title}]({p.source_uri})；阅读范围：{p.read_level}；位置：{e.locator}", "",
                  e.claim, "", f"原文：{e.quote}", "", "条件：" + "；".join(e.conditions),
                  "限制：" + "；".join(e.limitations), ""]
    lines += ["## 带来源的科研主张", ""]
    for c in state.knowledge:
        lines += [f"- {c.id}：{c.statement}", f"  证据：{', '.join(c.source_ids)}；限制：{'；'.join(c.limitations)}"]
    for title, items in (("研究缺口（限本次材料范围）", state.literature.gaps), ("争议", state.literature.conflicts)):
        lines += ["", "## " + title, ""]
        lines += [f"- {i.description}（{', '.join(i.evidence_ids)}）" for i in items]
    lines += ["", "## 覆盖限制", "", *["- " + s for s in state.literature.coverage_limitations], "",
              "证据核查通过表示提取忠实于材料，不代表论文结论已被独立证实。", ""]
    return "\n".join(lines)


def research_literature(*, state, pipeline: LiteraturePipeline, store: ResearchStore, source_root: Path,
                        artifact_root: Path, queries=None, discovery_checkpoint=None, emit=print):
    sources = capture_source(source_root)
    try:
        final = pipeline.run(state, queries=queries, discovery_checkpoint=discovery_checkpoint)
    except (ValueError, RuntimeError, OSError) as error:
        from .tools.experiment_tracker import json_bytes
        folder = Path(artifact_root) / ("failed_literature_" + uuid4().hex)
        folder.mkdir(parents=True)
        (folder / "failure.json").write_bytes(json_bytes({"error_type": type(error).__name__,
                          "input_state": state.model_dump(mode="json"), "attempts": pipeline.attempts,
                          "exports": pipeline.exports, "retrieval": pipeline.tools.events}))
        raise
    run_id = "run_" + uuid4().hex
    exports = {**pipeline.exports, "input_state.json": state.model_dump(mode="json"),
               "research_state.json": final.model_dump(mode="json"),
               "papers.json": [p.model_dump(mode="json") for p in final.papers],
               "evidence.json": [e.model_dump(mode="json") for e in final.evidence],
               "knowledge.json": [c.model_dump(mode="json") for c in final.knowledge],
               "literature.json": final.literature.model_dump(mode="json"),
               "pipeline_configuration.json": {"reasoning": pipeline.common.model_dump(mode="json"),
                                                "literature": pipeline.config.model_dump(mode="json")},
               "reasoning.json": {"purpose": "literature_research", "attempts": pipeline.attempts},
               "source_snapshot.json": sources, "runtime.json": runtime_versions(),
               "report.md": literature_report(final, pipeline.exports.get('retrieval_summary.json')).encode("utf-8")}
    directory = Path(artifact_root) / run_id
    store.save(run_id, final, directory, exports)
    store.verify_artifacts(run_id)
    emit(f"Literature state saved and verified: {directory.resolve()}")
    return final
