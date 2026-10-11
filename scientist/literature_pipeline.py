"""Literature agents produce a single grounded ScientificState, with auditable checkpoints."""
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from .agents.literature import (LiteratureSearchAgent, LiteratureScreeningAgent, LiteratureReadingAgent,
                               EvidenceVerificationAgent, LiteratureSynthesisAgent)
from .hypothesis_generation import ReasoningConfig, make_reasoning_backend
from .literature_schemas import LiteratureOverview, SearchQuery
from .provenance import capture_source, runtime_versions
from .schemas import Claim, Evidence, Paper, Record
from .state import ScientificState
from .tools.experiment_tracker import ResearchStore
from .tools.literature import LiteratureHTTP, LiteratureTools, arxiv_parts, deduplicate, digest, normalize

ROLES = {"literature_search", "literature_screening", "literature_reading", "evidence_verification",
         "literature_synthesis", "hypothesis_review", "hypothesis_revision"}


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
    providers: tuple[Literal["semantic_scholar", "arxiv", "crossref"], ...] = ("semantic_scholar", "arxiv")
    max_queries: int = Field(default=3, ge=1, le=6)
    results_per_query: int = Field(default=5, ge=1, le=10)
    max_candidates: int = Field(default=15, ge=1, le=50)
    max_selected_papers: int = Field(default=3, ge=1, le=8)
    min_candidates: int = Field(default=1, ge=1, le=50)
    min_selected_papers: int = Field(default=1, ge=1, le=8)
    max_search_rounds: int = Field(default=1, ge=1, le=4)
    max_characters_per_paper: int = Field(default=36000, ge=1000, le=100000)
    allow_abstract: bool = True
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
        return self


class LiteraturePipeline:
    def __init__(self, common: ReasoningConfig, config: LiteratureConfig, artifact_root: Path,
                 *, backend_factory=make_reasoning_backend, tools=None, emit=print):
        self.common, self.config, self.artifact_root = common, config, Path(artifact_root)
        self.backend_factory, self.emit = backend_factory, emit
        self.tools = tools or LiteratureTools(LiteratureHTTP(self.artifact_root / "literature_http_cache",
                                                            timeout=config.http_timeout_seconds))
        self.attempts, self.exports = [], {}

    def agent(self, cls):
        settings = self.config.roles.get(cls.role, AgentSettings()).runtime(self.common)
        return cls(self.backend_factory, settings, self.artifact_root / "research_agent_cache", emit=self.emit)

    def call(self, cls, inputs, validate=lambda value: None):
        agent = self.agent(cls)
        result = agent.run(inputs, validate)
        self.attempts.extend(agent.attempts)
        return result

    def discover(self, state: ScientificState, *, queries: tuple[SearchQuery, ...] | None = None):
        state = ScientificState.model_validate_json(state.model_dump_json())
        if state.problem_definition is None:
            raise ValueError("Literature research requires a confirmed problem definition")
        self.attempts, self.exports = [], {}
        self.tools.events = []
        self.tools.unavailable_providers = {}
        definition = state.problem_definition.model_dump(mode="json")
        pool, history, used_queries = [], [], []
        selection = None
        stop_reason = "search_round_budget_exhausted"
        existing_titles = {normalize(p.title).casefold() for p in state.papers}
        existing_dois = {p.doi.casefold() for p in state.papers if p.doi}
        existing_arxiv = {arxiv_parts(p.arxiv_id)[0] for p in state.papers if p.arxiv_id}
        def fresh_only(candidates):
            return [p for p in candidates if normalize(p.title).casefold() not in existing_titles
                    and (not p.doi or p.doi.casefold() not in existing_dois)
                    and (not p.arxiv_id or arxiv_parts(p.arxiv_id)[0] not in existing_arxiv)]
        for round_number in range(self.config.max_search_rounds):
            forbidden_queries = {normalize(q.text).casefold() for q in used_queries}
            def check_plan(plan):
                if len(plan.queries) > self.config.max_queries:
                    raise ValueError("Search plan exceeds max_queries")
                if len({normalize(q.text).casefold() for q in plan.queries}) != len(plan.queries):
                    raise ValueError("Search queries must be distinct")
                if any(normalize(q.text).casefold() in forbidden_queries for q in plan.queries):
                    raise ValueError("Recovery queries must differ from already executed queries")
            if round_number == 0 and queries is not None:
                if not queries or len(queries) > self.config.max_queries:
                    raise ValueError("Supplementary queries exceed search budget")
                plan_data = {"queries": [q.model_dump(mode="json") for q in queries], "origin": "hypothesis_review"}
                round_queries = queries
            else:
                plan = self.call(LiteratureSearchAgent, {"research_definition": definition,
                         "max_queries": self.config.max_queries, "phase": "initial" if round_number == 0 else "recovery",
                         "coverage_targets": {"min_candidates": self.config.min_candidates,
                                              "min_selected_papers": self.config.min_selected_papers},
                         "previous_queries": [q.model_dump(mode="json") for q in used_queries],
                         "feedback": history}, check_plan)
                round_queries, plan_data = plan.queries, plan.model_dump(mode="json")
            if round_number == 0:
                self.exports["search_plan.json"] = plan_data
            used_queries.extend(round_queries)
            self.emit(f"Literature tools: searching academic sources (round {round_number + 1})")
            before = len(self.tools.events)
            found = self.tools.search(round_queries, self.config.providers, per_query=self.config.results_per_query,
                                      max_candidates=self.config.max_candidates)
            previous_pool = pool
            previous_ids = {paper.id for paper in pool}
            pool = deduplicate(pool + fresh_only(found))[:self.config.max_candidates]
            if pool:
                candidate_ids = {p.id for p in pool}
                def check_selection(value):
                    ids = [d.paper_id for d in value.decisions]
                    if set(ids) != candidate_ids or len(ids) != len(candidate_ids):
                        raise ValueError("Screening must decide each candidate exactly once")
                    if sum(d.selected for d in value.decisions) > self.config.max_selected_papers:
                        raise ValueError("Screening exceeds max_selected")
                if selection is None or pool != previous_pool:
                    selection = self.call(LiteratureScreeningAgent, {"research_definition": definition,
                             "candidates": [p.model_dump(mode="json") for p in pool],
                             "max_selected": self.config.max_selected_papers,
                             "target_selected": self.config.min_selected_papers}, check_selection)
            selected_count = sum(d.selected for d in selection.decisions) if selection else 0
            history.append({"round": round_number + 1, "plan": plan_data,
                            "retrieval_events": list(self.tools.events[before:]), "new_candidates": sum(p.id not in previous_ids for p in pool),
                            "candidate_count": len(pool), "selected_count": selected_count,
                            "screening": selection.model_dump(mode="json") if selection else None})
            if len(pool) >= self.config.min_candidates and selected_count >= self.config.min_selected_papers:
                stop_reason = "coverage_targets_met"
                break
            if len(pool) >= self.config.max_candidates:
                stop_reason = "candidate_budget_exhausted"
                break
            self.emit("Literature coverage below target; search agent will receive retrieval and screening feedback")
        selected_count = sum(d.selected for d in selection.decisions) if selection else 0
        self.exports.update({"search_history.json": history,
             "search_results.json": [p.model_dump(mode="json") for p in pool], "retrieval.json": self.tools.events,
             "retrieval_summary.json": {"status": "target_met" if stop_reason == "coverage_targets_met" else "partial",
                 "stop_reason": stop_reason, "search_rounds": len(history), "queries_executed": len(used_queries),
                 "unique_candidates": len(pool), "selected_papers": selected_count,
                 "targets": {"min_candidates": self.config.min_candidates, "min_selected_papers": self.config.min_selected_papers},
                 "source_failures": [e for e in self.tools.events if e.get("status") == "failed"]}})
        if not pool:
            raise RuntimeError("Literature search returned no new readable candidate records after recovery; no evidence was invented")
        if selection is None:
            raise RuntimeError("No literature screening decision is available")
        self.exports["selection.json"] = selection.model_dump(mode="json")
        return pool, selection

    def run(self, state: ScientificState, *, queries: tuple[SearchQuery, ...] | None = None):
        fresh, selection = self.discover(state, queries=queries)
        definition = state.problem_definition.model_dump(mode="json")
        chosen_ids = {d.paper_id for d in selection.decisions if d.selected}
        chosen = [p for p in fresh if p.id in chosen_ids]
        if not chosen:
            raise RuntimeError("Screening found no relevant papers; no literature state was committed")
        new_papers, new_evidence, limitations, extraction_audit = [], [], [], []
        for candidate in chosen:
            try:
                document = self.tools.fetch_document(candidate, max_characters=self.config.max_characters_per_paper,
                                                     allow_abstract=self.config.allow_abstract)
            except RuntimeError as error:
                limitations.append(f"{candidate.id} 未能取得可读材料：{error}")
                continue
            self.exports[f"document_{candidate.id}.json"] = document.model_dump(mode="json")
            sections = {s.id: s for s in document.sections}
            def check_reading(reading):
                for finding in reading.findings:
                    if finding.section_id not in sections:
                        raise ValueError("Reading references an unknown section_id")
                    if normalize(finding.quote) not in normalize(sections[finding.section_id].text):
                        raise ValueError("Quoted evidence is not a verbatim contiguous span of the cited section")
                    if document.read_level in {"abstract", "excerpt"} and finding.strength != "weak":
                        raise ValueError("Abstract-only or excerpt evidence must be weak")
            reading = self.call(LiteratureReadingAgent, {"research_definition": definition,
                                "paper": candidate.model_dump(mode="json"), "document": document.model_dump(mode="json")},
                                check_reading)
            findings = [{"finding_id": f"F{n:03d}", **f.model_dump(mode="json")}
                        for n, f in enumerate(reading.findings, 1)]
            _, version = arxiv_parts(candidate.arxiv_id)
            new_papers.append(Paper(id=candidate.id, title=candidate.title, source_uri=document.source_uri,
                                    doi=candidate.doi, arxiv_id=candidate.arxiv_id, version=version,
                                    content_sha256=document.content_sha256, read_level=document.read_level))
            limitations.extend(document.limitations + reading.limitations)
            if not findings:
                extraction_audit.append({"paper_id": candidate.id, "reading": reading.model_dump(mode="json"), "checks": []})
                continue
            def check_verification(checks):
                ids = [c.finding_id for c in checks.checks]
                if set(ids) != {f["finding_id"] for f in findings} or len(ids) != len(findings):
                    raise ValueError("Verification must check every finding exactly once")
            checks = self.call(EvidenceVerificationAgent, {"research_definition": definition,
                               "paper": candidate.model_dump(mode="json"), "document": document.model_dump(mode="json"),
                               "findings": findings}, check_verification)
            extraction_audit.append({"paper_id": candidate.id, "reading": reading.model_dump(mode="json"),
                                     "checks": checks.model_dump(mode="json")})
            accepted = {c.finding_id for c in checks.checks if c.verdict == "accepted"}
            for n, finding in enumerate(reading.findings, 1):
                if f"F{n:03d}" not in accepted:
                    continue
                identifier = "EVL_" + digest({"paper": candidate.id, "finding": finding.model_dump(mode="json")})[:16]
                new_evidence.append(Evidence(id=identifier, source_type="paper", source_id=candidate.id,
                    claim=finding.claim, strength="weak" if document.read_level == "abstract" else finding.strength,
                    limitations=tuple(dict.fromkeys(finding.limitations + document.limitations)),
                    quote=finding.quote, locator=f"{finding.section_id} | {sections[finding.section_id].heading}",
                    content_sha256=document.content_sha256, conditions=finding.conditions, verification="accepted"))
        self.exports["extraction_audit.json"] = extraction_audit
        self.exports["retrieval.json"] = self.tools.events
        summary = self.exports["retrieval_summary.json"]
        summary.update(documents_read=len(new_papers), papers_with_accepted_evidence=len({e.source_id for e in new_evidence}),
                       accepted_findings=len(new_evidence))
        if summary['papers_with_accepted_evidence'] < self.config.min_selected_papers:
            summary['discovery_stop_reason'] = summary['stop_reason']
            summary.update(status='partial', stop_reason='readable_evidence_target_not_met')
            limitations.append(f"可核对证据覆盖不足：有 accepted 证据的论文 {summary['papers_with_accepted_evidence']}/{self.config.min_selected_papers}；不能把检索或筛选数量当作实际证据覆盖。")
        if summary["status"] != "target_met":
            limitations.append(f"检索覆盖目标未达成：候选 {len(fresh)}/{self.config.min_candidates}；入选 {len(chosen)}/{self.config.min_selected_papers}；{summary['stop_reason']}。")
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
                                 for e in self.tools.events if e.get("status") == "failed")
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
        status = '达到本轮数量目标' if s['status'] == 'target_met' else '覆盖不足'
        lines += [f"检索候选 {s['unique_candidates']} 篇 → 入选 {s['selected_papers']} 篇 → 阅读 {s.get('documents_read', 0)} 篇 → 有核查证据 {s.get('papers_with_accepted_evidence', 0)} 篇。{status}；不代表穷尽领域文献。", ""]
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
                        artifact_root: Path, queries=None, emit=print):
    sources = capture_source(source_root)
    try:
        final = pipeline.run(state, queries=queries)
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
