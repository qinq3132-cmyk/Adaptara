"""The sole scientific state, with referential integrity at every commit."""
from typing import Literal

from pydantic import Field, model_validator

from .schemas import (
    Analysis, Claim, Critique, Decision, Evidence, Experiment, ExperimentResult,
    Hypothesis, Paper, Record, ResearchBudget, ResearchQuestion, ScientificQuestion,
    Stage, TheoryUpdate,
)


class ScientificState(Record):
    schema_version: Literal[1] = 1
    question: ScientificQuestion
    knowledge: tuple[Claim, ...] = ()
    papers: tuple[Paper, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    experiments: tuple[Experiment, ...] = ()
    results: tuple[ExperimentResult, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    open_questions: tuple[ResearchQuestion, ...] = ()
    analyses: tuple[Analysis, ...] = ()
    critiques: tuple[Critique, ...] = ()
    theory_updates: tuple[TheoryUpdate, ...] = ()
    decisions: tuple[Decision, ...] = ()
    current_round: int = Field(default=0, ge=0)
    stage: Stage = Stage.QUESTION
    budget: ResearchBudget = Field(default_factory=ResearchBudget)

    @model_validator(mode="after")
    def references_exist(self):
        collections = (self.papers, self.knowledge, self.hypotheses, self.experiments,
                       self.results, self.evidence, self.open_questions, self.analyses,
                       self.critiques, self.theory_updates, self.decisions)
        all_ids = [self.question.id] + [item.id for group in collections for item in group]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Record IDs must be globally unique")

        def require(refs, records, label):
            missing = set(refs) - {item.id for item in records}
            if missing:
                raise ValueError(f"Unknown {label}: {sorted(missing)}")

        for claim in self.knowledge:
            require(claim.source_ids, self.papers + self.evidence, "claim sources")
        for hypothesis in self.hypotheses:
            require(hypothesis.parent_hypotheses, self.hypotheses, "parent hypotheses")
            require(hypothesis.source_claims, self.knowledge, "source claims")
            require(hypothesis.supporting_evidence + hypothesis.contradicting_evidence,
                    self.evidence, "hypothesis evidence")
            for evidence in self.evidence:
                if (evidence.id in hypothesis.supporting_evidence) != (hypothesis.id in evidence.supports):
                    raise ValueError("Inconsistent supporting evidence links")
                if (evidence.id in hypothesis.contradicting_evidence) != (hypothesis.id in evidence.contradicts):
                    raise ValueError("Inconsistent contradicting evidence links")
        for experiment in self.experiments:
            require(experiment.hypotheses_tested, self.hypotheses, "tested hypotheses")
        for result in self.results:
            require((result.experiment_id,), self.experiments, "result experiment")
        for evidence in self.evidence:
            sources = {"paper": self.papers, "experiment": self.experiments, "analysis": self.analyses}
            require((evidence.source_id,), sources[evidence.source_type], "evidence source")
            require(evidence.supports + evidence.contradicts, self.hypotheses, "evidence hypotheses")
            if evidence.source_type == "experiment" and not any(
                r.experiment_id == evidence.source_id for r in self.results
            ):
                raise ValueError("Experimental evidence requires a recorded result")
        for analysis in self.analyses:
            require(analysis.result_ids, self.results, "analysis results")
        for critique in self.critiques:
            require((critique.analysis_id,), self.analyses, "critique analysis")
        for update in self.theory_updates:
            require((update.analysis_id,), self.analyses, "theory analysis")
            require((update.critique_id,), self.critiques, "theory critique")
            for change in update.changes:
                require((change.hypothesis_id,), self.hypotheses, "changed hypothesis")
                require(change.evidence_ids, self.evidence, "change evidence")
        for question in self.open_questions:
            if set(question.source_ids) - set(all_ids):
                raise ValueError("Unknown open-question source")
        if self.current_round > self.budget.max_rounds:
            raise ValueError("Round budget exceeded")
        return self

    def revised(self, **changes) -> "ScientificState":
        """Unlike model_copy(update=...), always validate replacements."""
        return type(self).model_validate({**self.model_dump(), **changes})
