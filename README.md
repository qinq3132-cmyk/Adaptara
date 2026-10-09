# Autonomous Scientist

The default demo runs one offline scientific cycle, with deterministic mock
reasoning and an executable arithmetic counterexample. An explicit hypothesis
generation mode calls Codex (or OpenCode) with a configured model provider, and a
bounded local document reader uses Codex file tools. The offline demo needs no
API key or model download; reasoning modes reuse existing OpenCode credentials.
Literature retrieval, real benchmarks, an autonomous multi-round scheduler and
hardware integration remain future work.

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
`models.yaml` configures the explicit `--generate-hypotheses` mode;
the default demo does not load a provider.

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
The default E001 demo proposes and saves E002 without executing it. The explicit
`--followup E002` command below executes the saved follow-up in a new run.

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
The default Milestone 1 orchestrator only accepts a fresh cycle. Multi-round scheduling and resume semantics
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

The full requested specification is preserved in `SPEC.md`. Its Milestone 1 and the explicit E002 follow-up described below
are implemented, along with model-authored hypothesis generation through Codex/OpenCode
and local text/Markdown document reading. Future literature retrieval, real environments, baselines,
generated code execution, adaptive experiment selection across rounds, and hardware
work remain deferred. Reserved directories contain notes rather than executable
stubs that could be mistaken for working later-phase functionality.

## Execute the E002 follow-up

After a saved E001 demo, run:

```bash
python run_scientist.py --followup E002
# Or select a particular parent run:
python run_scientist.py --followup E002 --parent-run run_<uuid>
```

The default parent is the latest completed E001 with a proposed E002. The command
verifies the parent's checksums and SQLite state before execution. Copied projects
with stale Windows artifact paths are supported through the matching local
`artifacts/run_<uuid>` directory, without modifying the parent DB row.

E002 refines the parent proposal into six executable arms: frozen, positive
residual action updates, declared relevance filtering, predictor-state adaptation,
alternating-direction matched-update control, and reversed-direction control.
The parent's environment and seeds are preserved. The refined plan, original
proposal, parent state, parent source snapshot and reason for refinement are saved.
The new state retains E001 history, extends the round limit to two for this explicit
follow-up, and accumulates E002 costs. This is not a general multi-round scheduler.

All arms see identical sensor trajectories. Positive, negative, alternating and
predictor-state arms use the same observation-only fixed-zero residual gate;
the predictor state is distinct from that gate. Those four arms match update event
counts, not FLOPs or operation cost. Relevance-filter and frozen arms make zero
updates. Relevance is declared by the tracking objective, not discovered online.
No hidden shift metadata, evaluator target or evaluator loss reaches the algorithms.

The result records per-arm action loss, action-parameter updates, predictor-state
updates and nuisance prediction errors, with separate per-seed exports. Raw data
is saved before analysis. H002/H003 receive weak, fixture-specific supporting
evidence only if their measured predictions hold during an active trigger; their
existing internal belief scores are retained. H001 retains its earlier status.
There is no statistical significance or general adaptation claim.

Each follow-up saves the standard artifacts plus `lineage.json`,
`parent_state.json`, `parent_source_snapshot.json`, `results_by_seed.json`, and
`report.md`; the report is covered by the checksum manifest. Checksums and exact
runner replay are verified. Reported compute units count experimental arm-steps;
verification replay performs the same work again and is outside the experiment's
reported budget. No failed experiment is automatically retried.

```python
from experiments.followup_runner import FollowupExperimentRunner

state = store.load("run_<followup_uuid>")
result = next(r for r in state.results if r.experiment_id == "E002")
plan = next(e for e in state.experiments if e.id == "E002")
assert FollowupExperimentRunner().execute(plan, result.provenance) == result
```

## Generate hypotheses through Codex or OpenCode

Scientific roles and execution runtimes are separate: one Hypothesis Agent
proposes 3-6 competing hypotheses; Codex or OpenCode runs that role using the selected
provider/model configuration. Each hypothesis is a scientific record, not a
separate coding-agent process. Literature, planning and critique roles remain
future work.

```bash
python run_scientist.py --generate-hypotheses
python run_scientist.py --check-backend
python run_scientist.py --read-document examples/document_reading_smoke.md
# Use the original OpenCode runtime:
python run_scientist.py --generate-hypotheses --backend opencode --model Lemon-Domestic/glm-5.3
# Select another model already configured in OpenCode:
python run_scientist.py --generate-hypotheses --backend opencode --model Paratera/Qwen3.8-Max
# Use previous recorded research as context:
python run_scientist.py --generate-hypotheses --state-file artifacts/run_<uuid>/research_state.json
```

`config/models.yaml` contains only nonsecret execution settings. This project
now uses Codex with `glm-5.3-flash` at the supplied gateway, reusing the same
API key from OpenCode's `Lemon-Domestic` provider in its child process only.
It does not change global Codex config/auth or the existing ChatGPT login.
The verified standalone Codex 0.150.1 and its read-only Landlock sandbox are
selected in `config/models.yaml`; the IDE Codex has a namespace restriction here.
See [Codex setup, isolation and diagnostics](docs/codex_backend.md).

For the OpenCode backend, override the executable with `--opencode-bin` or
`SCIENTIST_OPENCODE_BIN`, and the configuration path with `--opencode-config`.
No OpenCode config or credential values are copied into project artifacts.
The adapter calls the installed `opencode run --pure --format json --agent
hypothesis --model <provider/model>` in a fresh temporary working directory.
The role uses a two-step ceiling to avoid the one-step assistant-summary prefill
that this compatible model rejects. The supplied role has no tool permissions; hypothesis generation needs only the
provided question, claims and recorded research. Literature roles can later use
a separate configuration for reading and retrieval tools.

The model supplies claims, mechanisms, assumptions, predictions, alternatives
and falsification conditions. Python assigns globally noncolliding H IDs,
initializes `proposed` status and internal belief score 0.5, and validates fields,
count, exact duplicate claims/mechanisms and references to existing claims or
parent hypotheses. Semantic independence and scientific usefulness still require
critique and evaluation. Hypotheses may propose mechanisms not yet executable.

This command saves only hypotheses and scientific context; it never invokes the
fixed E001 planner or executes an experiment. New proposals are appended to an
input state rather than replacing its history. `current_round` and experimental
compute spend do not advance; stage becomes `HYPOTHESIS`. A new SQLite run and
verified artifacts include `input_state.json`, `hypotheses.json`,
`research_state.json`, `reasoning.json`, `source_snapshot.json`, `runtime.json`,
and `report.md`. The reasoning trace includes prompts, output text, requested
model, session ID, reported tokens/cost and validation attempts. Model inference
is not claimed to be exactly reproducible; the saved output can be revalidated.

Malformed proposals may be regenerated within the configured validation bound.
Runtime/provider failures stop without committing research or falling back to
mock hypotheses. The subprocess timeout bounds each invocation; reported model
usage is recorded separately from toy experiment compute units. OpenCode may
perform its own provider retries within that timeout. A fixed session title
avoids an extra model call for automatic title generation.

The current environment has a working uppercase proxy configuration and stale
lowercase proxy variables. `proxy_env_precedence: uppercase` normalizes these
variables only for the runtime child process. Set it to `inherit` to preserve
another environment unchanged; global proxy settings are never modified.

OpenCode's role/model/JSON-event interface is documented in its official
[CLI documentation](https://opencode.ai/docs/cli/) and
[agent configuration](https://opencode.ai/docs/agents/).
The Codex adapter maps OpenCode credentials into an invocation-scoped custom
provider, with Responses streaming, structured output and file-tool checks.
See the official [OpenAI Docs on custom providers](https://learn.chatgpt.com/docs/config-file/config-advanced#custom-model-providers).
