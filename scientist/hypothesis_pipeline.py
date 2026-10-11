"""Literature-grounded proposals, bounded external review, and immutable revisions."""
import json
from pathlib import Path
from uuid import uuid4

from .agents.hypothesis_review import HypothesisReviewAgent, HypothesisRevisionAgent
from .agents.llm_hypothesis import DraftBatch, LLMHypothesisAgent, allocate_ids, hypothesis_context, normalized
from .agents.structured import StructuredAgent
from .hypothesis_generation import hypothesis_report
from .literature_pipeline import LiteraturePipeline, research_literature
from .literature_schemas import HypothesisReviewRecord, PaperDocument
from .provenance import capture_source, runtime_versions
from .schemas import Hypothesis, HypothesisOutput, Stage
from .tools.literature import digest, normalize


class GroundedHypothesisAgent(StructuredAgent):
    role, output_type = "hypothesis", DraftBatch


def load_source_documents(state, store):
    """Only load checksum-verified document exports matching the state's exact paper snapshots."""
    wanted = {p.id: p.content_sha256 for p in state.papers if p.content_sha256}
    documents = {}
    with store.connect() as db:
        runs = db.execute("SELECT id, artifact_dir FROM runs ORDER BY rowid DESC").fetchall()
    for run_id, directory in runs:
        paths = {pid: Path(directory) / ("document_" + pid + ".json") for pid in wanted if pid not in documents}
        available = {pid: path for pid, path in paths.items() if path.exists()}
        if not available:
            continue
        store.verify_artifacts(run_id)
        for pid, path in available.items():
            document = PaperDocument.model_validate_json(path.read_bytes())
            if document.content_sha256 != digest([s.model_dump(mode="json") for s in document.sections]):
                raise ValueError("Original document content digest mismatch")
            if document.paper_id == pid and document.content_sha256 == wanted[pid]:
                documents[pid] = document
    if set(wanted) - set(documents):
        raise ValueError("Verified paper document snapshots are missing; use the literature state and its original database")
    for evidence in state.evidence:
        if evidence.verification == "accepted":
            document = documents[evidence.source_id]
            section_id = evidence.locator.split(" | ", 1)[0]
            section = next((s for s in document.sections if s.id == section_id), None)
            if section is None or normalize(evidence.quote) not in normalize(section.text):
                raise ValueError("Stored evidence quote differs from the verified original document")
    return documents


class HypothesisPipeline:
    def __init__(self, literature: LiteraturePipeline):
        self.literature = literature
        self.attempts, self.exports = [], {}

    def call(self, cls, inputs, validate=lambda value: None):
        if cls.role == "hypothesis":
            agent = cls(self.literature.backend_factory, self.literature.common,
                        self.literature.artifact_root / "research_agent_cache", emit=self.literature.emit)
        else:
            agent = self.literature.agent(cls)
        result = agent.run(inputs, validate)
        self.attempts.extend(agent.attempts)
        return result

    def generate(self, state):
        if state.literature is None or not state.knowledge or not state.evidence:
            raise ValueError("Reviewed hypothesis generation requires completed literature research")
        count = self.literature.common.hypothesis_count
        validator = LLMHypothesisAgent(None, count=count, max_retries=0)
        batch = self.call(GroundedHypothesisAgent, {"scientific_context": hypothesis_context(state),
                         "allowed_source_claim_ids": [c.id for c in state.knowledge],
                         "allowed_parent_hypothesis_ids": [h.id for h in state.hypotheses],
                         "required_count": count}, lambda b: validator._validate(b.model_dump_json(), state))
        ids = allocate_ids(state, count)
        new = tuple(Hypothesis(id=hid, **d.model_dump(), status="proposed", confidence=0.5)
                    for hid, d in zip(ids, batch.hypotheses))
        return state.revised(hypotheses=state.hypotheses + new, stage=Stage.HYPOTHESIS), ids

    def review(self, state, targets, documents, *, search_available, round_number):
        targets = tuple(targets)
        known_evidence = {e.id for e in state.evidence}
        known_papers = {p.id for p in state.papers}
        def validate(batch):
            ids = [r.hypothesis_id for r in batch.reviews]
            if set(ids) != set(targets) or len(ids) != len(targets):
                raise ValueError("Review must address each target once")
            if set(batch.ranking) != set(targets) or len(batch.ranking) != len(targets):
                raise ValueError("Review ranking must contain each target once")
            for item in batch.reviews:
                if set(item.evidence_ids) - known_evidence or set(item.closest_paper_ids) - known_papers:
                    raise ValueError("Review cites unknown evidence or paper")
            if batch.search_requests and not search_available:
                raise ValueError("Supplementary literature search budget exhausted")
            if len(batch.search_requests) > min(3, self.literature.config.max_queries):
                raise ValueError("Review search requests exceed query budget")
            for comparison in batch.comparisons:
                if set(comparison.predictions) - set(targets):
                    raise ValueError("Comparison references an unreviewed hypothesis")
            if len(targets) > 1 and not batch.comparisons and any(r.verdict == "ready" for r in batch.reviews):
                raise ValueError("Ready candidates require at least one distinguishing comparison")
        inputs = {"scientific_context": hypothesis_context(state), "target_hypothesis_ids": list(targets),
                  "original_documents": [d.model_dump(mode="json") for d in documents.values()],
                  "supplementary_search_available": search_available,
                  "max_search_queries": min(3, self.literature.config.max_queries)}
        batch = self.call(HypothesisReviewAgent, inputs, validate)
        record = HypothesisReviewRecord(id="HR_" + digest({"round": round_number, "batch": batch.model_dump(mode="json")})[:16],
                                        round=round_number, batch=batch)
        return state.revised(hypothesis_reviews=state.hypothesis_reviews + (record,)), batch

    def revise(self, state, targets, review):
        existing = {h.id: h for h in state.hypotheses}
        claims = {c.id for c in state.knowledge}
        def validate(batch):
            if len(batch.hypotheses) != len(targets):
                raise ValueError("Revise every requested target once")
            parents, texts, mechanisms = [], set(), set()
            for draft in batch.hypotheses:
                if len(draft.parent_hypotheses) != 1 or draft.parent_hypotheses[0] not in targets:
                    raise ValueError("Revision requires exactly its target parent ID")
                if not draft.source_claims or set(draft.source_claims) - claims:
                    raise ValueError("Revision must cite known literature claims")
                text, mechanism = normalized(draft.claim), normalized(draft.mechanism)
                if text in texts or mechanism in mechanisms:
                    raise ValueError("Revised candidates must have distinct claims and mechanisms")
                if any(not normalize(t) for t in (draft.claim, draft.mechanism, *draft.assumptions, *draft.predictions,
                       *draft.alternative_explanations, *draft.falsification_conditions)):
                    raise ValueError("Revision scientific fields cannot be blank")
                parents.extend(draft.parent_hypotheses)
                texts.add(text)
                mechanisms.add(mechanism)
            if set(parents) != set(targets) or len(set(parents)) != len(parents):
                raise ValueError("Revision must preserve a one-to-one parent mapping")
        batch = self.call(HypothesisRevisionAgent, {"scientific_context": hypothesis_context(state),
                         "targets": [existing[t].model_dump(mode="json") for t in targets],
                         "review": review.model_dump(mode="json")}, validate)
        ids = allocate_ids(state, len(targets))
        new = tuple(Hypothesis(id=hid, **d.model_dump(), status="proposed", confidence=0.5)
                    for hid, d in zip(ids, batch.hypotheses))
        replacements = {h.parent_hypotheses[0]: h.id for h in new}
        return state.revised(hypotheses=state.hypotheses + new), replacements

    def run(self, state, *, store, source_root, artifact_root, generate=True):
        self.attempts, self.exports = [], {}
        documents = load_source_documents(state, store)
        if generate:
            state, targets = self.generate(state)
        else:
            ancestors = {p for h in state.hypotheses for p in h.parent_hypotheses}
            targets = tuple(h.id for h in state.hypotheses if h.id not in ancestors and h.status == "proposed")
            if not targets or len(targets) > 6 or state.literature is None:
                raise ValueError("Review requires 1-6 current untested hypotheses and completed literature")
        supplements, retrieval_failures = 0, []
        round_number = max((record.round for record in state.hypothesis_reviews), default=-1) + 1
        for revision_round in range(self.literature.config.max_revision_rounds + 1):
            state, batch = self.review(state, targets, documents,
                          search_available=supplements < self.literature.config.max_supplement_rounds,
                          round_number=round_number)
            round_number += 1
            while batch.search_requests and supplements < self.literature.config.max_supplement_rounds:
                supplements += 1
                try:
                    state = research_literature(state=state, pipeline=self.literature, store=store,
                                source_root=source_root, artifact_root=artifact_root, queries=batch.search_requests,
                                emit=self.literature.emit)
                    documents = load_source_documents(state, store)
                except RuntimeError as error:
                    retrieval_failures.append(str(error))
                    self.literature.emit("Supplementary literature acquisition failed; existing evidence retained")
                self.attempts.extend(self.literature.attempts)
                state, batch = self.review(state, targets, documents,
                              search_available=supplements < self.literature.config.max_supplement_rounds,
                              round_number=round_number)
                round_number += 1
            revise_targets = tuple(r.hypothesis_id for r in batch.reviews if r.verdict == "revise")
            if not revise_targets or revision_round == self.literature.config.max_revision_rounds:
                break
            state, replacements = self.revise(state, revise_targets, batch)
            targets = tuple(replacements.get(t, t) for t in targets)
        self.exports = {"reviewed_candidate_ids.json": list(targets),
                        "hypothesis_reviews.json": [r.model_dump(mode="json") for r in state.hypothesis_reviews],
                        "review_budget.json": {"revision_rounds": revision_round, "supplement_rounds": supplements,
                                               "supplement_failures": retrieval_failures,
                                               "remaining_revision_requests": [r.hypothesis_id for r in batch.reviews if r.verdict == "revise"]}}
        self.exports.update({f"document_{pid}.json": d.model_dump(mode="json") for pid, d in documents.items()})
        return state


def hypothesis_suggestion_sheet(state):
    batch = state.hypothesis_reviews[-1].batch
    items = {r.hypothesis_id: r for r in batch.reviews}
    lines = ["# 候选假设建议单", "", "评审决定检验优先级，不代表假设已被证实。", ""]
    labels = {"ready": "可进入实验设计", "revise": "仍需修订", "defer": "暂缓"}
    for n, hid in enumerate(batch.ranking, 1):
        item = items[hid]
        lines += [f"## {n}. {hid}：{labels[item.verdict]}", "", item.human_summary, "",
                  "建议检验：" + item.recommended_test, ""]
    lines += ["完整依据、限制与修订要求见 [完整报告](report.md)。", ""]
    return "\n".join(lines)


def reviewed_hypotheses(*, state, pipeline: HypothesisPipeline, store, source_root, artifact_root,
                        generate=True, emit=print):
    sources = capture_source(source_root)
    final = pipeline.run(state, store=store, source_root=source_root, artifact_root=artifact_root, generate=generate)
    run_id = "run_" + uuid4().hex
    report = hypothesis_report(HypothesisOutput(hypotheses=final.hypotheses), pipeline.literature.common.model,
                               pipeline.literature.common.backend, reviewed=True)
    report += "\n## 假设审查与修订记录\n\n"
    for record in final.hypothesis_reviews:
        report += f"### {record.id}（第 {record.round} 轮）\n\n"
        for item in record.batch.reviews:
            report += f"- {item.hypothesis_id}：{item.verdict}；新颖性：{item.novelty}\n  {item.rationale}\n"
            report += f"  证据：{', '.join(item.evidence_ids)}；近似文献：{', '.join(item.closest_paper_ids)}\n"
            report += "\n".join("  问题：" + s for s in item.issues) + "\n"
            report += "\n".join("  修订要求：" + s for s in item.revision_requests) + "\n"
        report += "\n评审限制：\n" + "\n".join("- " + s for s in record.batch.limitations) + "\n"
        if record.batch.search_requests:
            report += "\n补充检索请求：\n" + "\n".join(
                f"- {q.text}：{q.rationale}" for q in record.batch.search_requests) + "\n"
        for comparison in record.batch.comparisons:
            report += "\n共同条件：" + comparison.condition + "\n测量：" + comparison.measurement + "\n"
            report += "\n".join(f"- {hid}：{p}" for hid, p in comparison.predictions.items()) + "\n"
    exports = {**pipeline.exports, "input_state.json": state.model_dump(mode="json"),
               "research_state.json": final.model_dump(mode="json"),
               "hypotheses.json": [h.model_dump(mode="json") for h in final.hypotheses],
               "pipeline_configuration.json": {"reasoning": pipeline.literature.common.model_dump(mode="json"),
                                                "literature": pipeline.literature.config.model_dump(mode="json")},
               "reasoning.json": {"purpose": "reviewed_hypotheses", "attempts": pipeline.attempts},
               "source_snapshot.json": sources, "runtime.json": runtime_versions(),
               "report.md": report.encode("utf-8"), "suggestions.md": hypothesis_suggestion_sheet(final).encode("utf-8")}
    directory = Path(artifact_root) / run_id
    store.save(run_id, final, directory, exports)
    store.verify_artifacts(run_id)
    emit(f"Reviewed hypotheses saved and verified: {directory.resolve()}")
    return final
