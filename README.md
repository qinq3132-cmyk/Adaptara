# Autonomous Scientist — Milestone 1

One offline scientific cycle, with deterministic mock reasoning and an executable
arithmetic counterexample. No API key, model download, literature search, real
benchmark, autonomous multi-round scheduler, or hardware integration is required.

## Run

Python 3.12+:

```powershell
cd D:\AI4Science\autonomous_scientist
python -m pip install -e ".[test]"
python run_scientist.py --demo
python -m pytest -q
```

`python run_scientist.py` also runs the demo. Paths default to the project directory,
independent of your working directory. Override with `--output-dir`, `--database`,
and `--config-dir`. The latter must contain `project.yaml` and `experiments.yaml`.
`models.yaml` is reserved documentation; no provider is loaded.

Each invocation creates a distinct `artifacts/run_<uuid>/` and adds one SQLite run
to `database/research.sqlite3`. Repeated IDs such as H001 are scoped to their run
in the database. An existing run is never overwritten.

Expected terminal milestones:

```text
Scientific Cycle 1
Question loaded
Hypothesis generated: H001
Hypothesis generated: H002
Hypothesis generated: H003
Experiment selected: E001
Experiment executed
Evidence contradicts H001
H001: proposed -> weakened
Next experiment generated: E002
Cycle complete
```

Required exports: `question.json`, `hypotheses.json`, `experiment_plan.json`,
`raw_results.json`, `analysis.json`, `critique.json`, `theory_update.json`, and
`research_state.json`. Additional exports include `experiment.json`,
`source_snapshot.json`, `runtime.json`, and a checksum `manifest.json`.
`hypotheses.json` contains initial proposals; final beliefs are in `research_state.json`.
E002 is proposed and saved but **never executed** in this milestone.

## What the toy demonstrates

The public behavioral objective is to track an observed signal. The observation
also contains an irrelevant sensor channel. Halfway through the run, the nuisance
channel acquires an offset of 2, while the behavioral target remains unchanged.
A deliberately simple rule changes its action offset by 0.1 times the nuisance
residual whenever its magnitude exceeds 0.5. The frozen control sees the same
observations but leaves its action offset at zero. Evaluation computes squared
tracking loss; neither evaluator losses nor shift metadata are supplied to the rule.

With seeds 7, 11, and 19 and 20 steps per seed, the executed arithmetic produces
mean excess loss 0.77 and 30 unnecessary updates. The fixed seeds control signal
trajectories; this construction makes the loss difference invariant to that signal.
No statistical significance or independent replication claim is made.

H001 concerns this specific residual gate. H002 proposes relevance filtering;
H003 proposes predictor-state adaptation. They are explicitly candidate mechanisms,
not restrictions on the root scientific question. H002 and H003 remain untested.
The critic flags unequal update counts and the deliberately chosen update direction.
The mock theory adapter weakens H001 only when measured toy outputs meet the
falsification condition. Its confidence 0.25 is a fixture's **internal belief score**,
not a calibrated probability. E002 proposes matched-update controls. These results
do not establish a general scientific principle or real-world adaptation capability.

## Architecture and extension contracts

`scientist/state.py` defines versioned `ScientificState`; `scientist/schemas.py`
defines validated, closed schemas. The orchestrator controls this fixed sequence:

```text
QUESTION -> HYPOTHESIS -> EXPERIMENT -> ANALYSIS -> CRITIQUE -> THEORY_UPDATE
```

Agent outputs are proposals. The orchestrator revalidates them, checks ID links,
selects an affordable experiment by the specified heuristic score, and applies
evidence-backed belief changes. Score ties resolve lexicographically by ID.
Schema objects are frozen; nested mutable values are isolated through deep copies.
This is an API integrity boundary, not a sandbox for untrusted executable plugins.

To replace mock reasoning, inject an `AgentBundle` with adapters implementing
`ScientificAgent.propose(state) -> structured output`. Analysis has a deliberately
narrower `analyze(experiment, result) -> Analysis` interface and cannot inspect
hypotheses or belief scores. It receives the relevant projection of scientific state.
Adapters may return dictionaries for proposals; the orchestrator validates them.

To replace execution, inject an `ExperimentRunner` implementing
`execute(experiment, provenance) -> ExperimentResult` and declaring `runner_version`
and `algorithm_version`. Record seeds, config, trajectories, costs, and artifacts
in the same output contract. The orchestrator does not inspect algorithm type or
calculate conclusions from terminal output. The demo runner alone validates its
supported toy configuration. Future environment/algorithm adapters have protocols
in `environments/base.py` and `algorithms/base.py`; they are not implemented here.

Use `build_demo()` as the composition-root example. Adding new adapters changes
composition, not `ScientificState` or `Orchestrator.run_cycle(state) -> ScientificState`.
Milestone 1 only accepts a fresh cycle. Multi-round scheduling and resume semantics
remain deferred; loading a completed state is supported for inspection and replay.
Any future incompatible persisted schema requires an explicit version migration.

## Persistence and provenance

The result is saved as `raw_results.json` **before analysis executes**. A
`pending.json` marker identifies a staged directory. The complete validated state
and individual records are then committed in one SQLite transaction after all
stages succeed. On failure, the original state and SQLite remain unchanged; raw
data may remain in a pending directory for diagnosis. SQLite is authoritative:
filesystem writes and SQLite are not a distributed transaction. No automatic
resume from incomplete directories is attempted.

SQLite includes questions, papers, claims, hypotheses, experiments, runs, results,
evidence, decisions, analyses, critiques, and theory updates. Paper/claim tables
are empty in the demo. JSON payloads preserve all schema fields; run-scoped primary
keys prevent collisions across demonstrations. Scientific references are validated
before the transaction, and row-to-run foreign keys are enforced by SQLite.

Trace a belief change through `theory_update.json` -> evidence ID -> experiment
ID -> result -> its provenance/configuration -> `raw_results.json` and the exact
`source_snapshot.json`. A Git commit is recorded when available; otherwise it is
`null`. The source snapshot and SHA-256 remain available even without Git or with
uncommitted edits. Python and package versions are recorded in `runtime.json`.
Checksums detect accidental corruption; they are not cryptographic signatures.

```python
from pathlib import Path
from scientist.tools.experiment_tracker import ResearchStore
from experiments.runner import ToyExperimentRunner

store = ResearchStore(Path("database/research.sqlite3"))
state = store.load("run_<uuid>")
store.verify_artifacts("run_<uuid>")
saved = state.results[0]
plan = next(e for e in state.experiments if e.id == saved.experiment_id)
replayed = ToyExperimentRunner().execute(plan, saved.provenance)
assert replayed == saved
```

Run this with the saved source and recorded runtime for exact replay. Run IDs and
runtime metadata differ between invocations; scientific outputs and raw trajectories
are deterministic for a given configuration and implementation.

## Bounds and deferred work

The orchestrator checks round limits, estimated affordability, reported compute
cost, and an injectable human-stop callback. Toy compute units count observation
steps, not FLOPs, time, money, or energy. Agents may retry invalid proposals within
the configured bound; execution is never automatically retried. Real runners will
need process isolation, wall-time limits, and enforcement of actual resource budgets.
These are not claimed by the current accounting checks.

The full requested specification is preserved in `SPEC.md`. Only its Milestone 1
is implemented. Future literature, prompts, tools, real environments, baselines,
generated code execution, adaptive experiment selection across rounds, and hardware
work remain deferred. Reserved directories contain notes rather than executable
stubs that could be mistaken for working later-phase functionality.
