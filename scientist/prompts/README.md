Scientific role prompts are loaded by their agents and recorded in each model invocation.

- `problem_clarification.txt`, `problem_synthesis.txt`: independent clarification and cross-model synthesis.
- `literature_search.txt`, `literature_screening.txt`: search planning and explicit paper selection.
- `literature_reading.txt`, `evidence_verification.txt`: located extraction and independent source checks.
- `literature_synthesis.txt`: evidence-backed claims, conflicts and scoped research gaps.
- `hypothesis.txt`, `hypothesis_review.txt`, `hypothesis_revision.txt`: proposals, external review and immutable revisions.

Python owns IDs, retrieval tools, source checks, budgets and commits. These model roles have no direct tool permissions.
