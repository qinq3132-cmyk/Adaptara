"""Problem clarification lifecycle: propose, review, explicitly confirm."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .agents.problem_clarification import ProblemClarificationAgent, validate_human_review, validate_revision
from .problem_schemas import ConceptInput, ConfirmedProblem, HumanFeedback, ProblemDraft, ProblemRevision, problem_digest
from .provenance import capture_source, runtime_versions
from .schemas import ResearchBudget, ScientificQuestion
from .state import ScientificState
from .tools.problem_store import ProblemStore


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_concept(path: Path) -> ConceptInput:
    if path.suffix.lower() == ".json":
        return ConceptInput.model_validate_json(path.read_bytes())
    if path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Concept input must be UTF-8 .txt, .md or ConceptInput .json")
    return ConceptInput(text=path.read_text(encoding="utf-8"))


def human_review_report(record: ProblemRevision) -> str:
    issues = sorted(record.draft.pending_issues, key=lambda issue: not issue.blocking)
    lines = ["# 请确认研究边界", "", "研究问题及详细分析：[查看完整草案](report.md)。", "",
             "每题可选一个字母，也可以用自己的话回答。未回答的事项会保留。", ""]
    if not issues:
        lines.extend(["目前没有待回答的问题。请核对完整草案后，再决定是否确认。", ""])
        return "\n".join(lines)

    def issue_block(number, issue):
        lines.extend([f"### {number}. {issue.question}", "",
                      "需确认" if issue.blocking else "可后续再定", "",
                      *[f"- {chr(65 + index)}. {option}" for index, option in enumerate(issue.options)],
                      "" if issue.options else "请直接说明你的选择。",
                      f"影响：{issue.reason}", ""])

    lines.extend(["## 先回答这几项", ""])
    for number, issue in enumerate(issues[:3], 1):
        issue_block(number, issue)
    if len(issues) > 3:
        remaining = issues[3:]
        blocking = sum(issue.blocking for issue in remaining)
        lines.extend(["<details>", f"<summary>其余 {len(remaining)} 项，其中 {blocking} 项需确认</summary>", ""])
        for number, issue in enumerate(remaining, 4):
            issue_block(number, issue)
        lines.extend(["</details>", ""])
    lines.extend(["可以按“1选A，2选B”回复，也可以只回答其中几题。", "",
                  "确认整份草案前，需要回答所有标为“需确认”的事项。", ""])
    return "\n".join(lines)


def problem_report(record: ProblemRevision) -> str:
    draft = record.draft
    lines = ["# 科研问题解析草案（详细版）", "", f"会话：{record.session_id}；版本：{record.revision}；尚未确认。", "",
             "## 人类原始概念", "", record.concept.text, "", "## 根问题", "", draft.root_question.text,
             "", "## 研究目标", "", draft.goal.text, ""]
    def statements(title, items):
        lines.extend([f"## {title}", ""])
        lines.extend(f"- {s.text}（{s.origin}；来源：{', '.join(s.source_ids) or '无'}）" for s in items)
        lines.append("")
    statements("研究边界", draft.scope)
    statements("硬约束", draft.hard_constraints)
    statements("偏好", draft.preferences)
    statements("隐含前提与待验证判断", draft.assumptions)
    lines.extend(["## 核心术语", ""])
    for term in draft.terms:
        lines.extend([f"- {term.term}：{term.interpretation.text}（{term.interpretation.origin}）",
                      f"  其他解释：{'；'.join(term.alternatives) or '无'}"])
    lines.extend(["", "## 跨学科审视", ""])
    for view in draft.perspectives:
        lines.extend([f"### {view.discipline}", "", f"贡献：{view.contribution.text}（{view.contribution.origin}）",
                      f"定义影响：{'；'.join(view.definition_implications) or '无'}",
                      f"冲突：{'；'.join(view.conflicts) or '无'}",
                      f"限制：{'；'.join(view.limitations) or '无'}", ""])
    lines.extend(["## 子问题", ""])
    for sub in draft.subquestions:
        lines.extend([f"- {sub.id}：{sub.text}", f"  与根问题的关系：{sub.root_relationship}",
                      f"  评价条件：{'；'.join(sub.evaluation_criteria)}"])
    lines.append("")
    statements("评价条件", draft.evaluation_criteria)
    statements("候选机制（未验证，不是问题前提）", draft.candidate_mechanisms)
    lines.extend(["## 待确认事项", ""])
    for issue in draft.pending_issues:
        lines.extend([f"- {issue.id} [{'必须处理' if issue.blocking else '可保留的未知'}]：{issue.question}",
                      f"  原因：{issue.reason}", f"  选项：{'；'.join(issue.options) or '自由回答'}"])
    lines.extend(["", "## 本轮修改", "", *[f"- {s}" for s in draft.changes], "",
                  "## 限制", "", *[f"- {s}" for s in draft.limitations], "",
                  "修改请提交反馈；确认前必须处理 blocking=true 的事项。人类确认不代表科学判断成立。", ""])
    return "\n".join(lines)


def clarify_problem(*, concept: ConceptInput, agent: ProblemClarificationAgent, store: ProblemStore,
                    source_root: Path, artifact_root: Path, session_id: str | None = None,
                    revision: int | None = None, feedback: HumanFeedback | None = None, emit=print) -> ProblemRevision:
    previous = None
    if session_id is not None:
        if revision is None or feedback is None:
            raise ValueError("Review requires an explicit revision and human feedback")
        if store.latest_revision(session_id) != revision:
            raise ValueError("Stale feedback; review the latest problem revision")
        previous = store.load_revision(session_id, revision)
        concept = previous.concept
        if feedback.revised_constraints is not None:
            concept = ConceptInput.model_validate({**concept.model_dump(),
                                                   "explicit_constraints": feedback.revised_constraints})
        unknown = set(feedback.answers) - {i.id for i in previous.draft.pending_issues}
        if unknown:
            raise ValueError("Feedback refers to unknown issue IDs")
    elif feedback is not None or revision is not None:
        raise ValueError("Initial clarification cannot include revision feedback")
    session_id = session_id or "problem_" + uuid4().hex
    sources = capture_source(source_root)
    raw = agent.analyze(concept.model_copy(deep=True),
                        previous.draft.model_copy(deep=True) if previous else None,
                        feedback.model_copy(deep=True) if feedback else None)
    draft = ProblemDraft.model_validate(raw.model_dump() if hasattr(raw, "model_dump") else raw)
    validate_human_review(draft)
    draft.validate_context(concept)
    validate_revision(previous.draft if previous else None, feedback, draft)
    record = ProblemRevision(session_id=session_id, revision=(revision or 0) + 1, concept=concept,
                             draft=draft, feedback=feedback,
                             previous_draft_sha256=problem_digest(previous.draft) if previous else None,
                             created_at=now())
    directory = Path(artifact_root) / session_id / f"revision_{record.revision:04d}"
    store.save_revision(record, directory, {
        **getattr(agent, "exports", {}),
        "draft.json": draft.model_dump(mode="json"), "concept.json": concept.model_dump(mode="json"),
        "reasoning.json": {"purpose": "problem_clarification", "attempts": agent.attempts},
        "source_snapshot.json": sources, "runtime.json": runtime_versions(),
        "review.md": human_review_report(record).encode("utf-8"),
        "report.md": (problem_report(record) + getattr(agent, "report_appendix", "")).encode("utf-8")})
    store.load_revision(session_id, record.revision)
    emit(f"Problem draft saved: {session_id}, revision {record.revision}")
    emit(f"Review report: {(directory / 'review.md').resolve()}")
    emit(f"Detailed report: {(directory / 'report.md').resolve()}")
    for number, issue in enumerate(sorted(draft.pending_issues, key=lambda i: not i.blocking)[:3], 1):
        emit(f"{number}. {issue.question}")
        if issue.options:
            emit("  " + " / ".join(f"{chr(65 + index)}. {option}" for index, option in enumerate(issue.options)))
        emit(f"  影响：{issue.reason}")
    return record


def confirm_problem(*, session_id: str, revision: int, store: ProblemStore,
                    artifact_root: Path, confirmed_by: str, emit=print) -> ConfirmedProblem:
    record = store.load_revision(session_id, revision)
    problem = ConfirmedProblem(
        session_id=session_id, revision=revision, original_concept=store.original_concept(session_id),
        concept=record.concept, draft=record.draft, draft_sha256=problem_digest(record.draft),
        confirmed_at=now(), confirmed_by=confirmed_by,
        question=ScientificQuestion(id="Q001", text=record.draft.root_question.text,
                                    constraints=tuple(s.text for s in record.draft.hard_constraints)))
    directory = Path(artifact_root) / session_id / f"confirmation_{revision:04d}"
    store.save_confirmation(problem, directory)
    store.load_confirmation(session_id, revision)
    emit(f"Problem confirmed: {(directory / 'confirmed_problem.json').resolve()}")
    return problem


def state_from_confirmed_problem(problem: ConfirmedProblem, budget: ResearchBudget) -> ScientificState:
    problem = ConfirmedProblem.model_validate_json(problem.model_dump_json())
    return ScientificState(schema_version=2, question=problem.question, problem_definition=problem, budget=budget)
