# Autonomous AI Scientist for Discovering Principles of Autonomous Adaptation

## 0. Project Goal

构建一个可执行的多 Agent AI-for-Science 系统，使 AI 能围绕一个开放科学问题自动完成：

**Scientific Question → Literature → Hypotheses → Experiments → Code → Results → Critique → Theory Update → Next Experiment**

第一阶段的目标不是发表论文，也不是寻找某一种预设的“类脑算法”，而是验证：

> 一个 AI Scientist 是否能够围绕明确的科学问题，持续提出可证伪假设、设计实验、执行实验，并根据实验结果修改自己的理论。

---

# 1. Root Scientific Question

系统的初始科学问题固定为：

> **What computational principles enable an artificial agent to autonomously and continually adapt to unknown environmental changes without externally supplied sample-level teaching signals, explicit task identities, or manually triggered learning?**

中文：

> **什么样的计算原理能够使人工智能体在没有外部提供的样本级教学信号、显式任务标识和人为触发学习的条件下，自主、持续地适应未知环境变化？**

注意：

这个问题只规定：

- dynamic environment
- autonomous adaptation
- continual adaptation
- no sample-level labels
- no explicit task ID
- no manual adaptation trigger

不得预设：

- neural network
- BP
- SNN
- STDP
- predictive coding
- reinforcement learning
- memory
- local plasticity
- meta-learning
- world model
- RRAM

这些都属于待探索的 hypothesis space。

---

# 2. Core Design Principle

整个系统必须严格区分：

## Scientific objective

我们想知道：

> What mechanisms are actually necessary for autonomous adaptation?

和：

## Candidate hypothesis

例如：

- prediction error
- novelty
- intrinsic motivation
- active exploration
- memory
- meta-learning
- parameter adaptation
- state adaptation
- architecture routing

不能把 hypothesis 写进 scientific question。

---

# 3. Overall Architecture

系统结构：

```text
Scientific Question
        │
        ▼
┌─────────────────────┐
│ Problem Formalizer  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Literature Agent    │
│ + Evidence Graph    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Hypothesis Agent    │
│ H1 H2 H3 ... Hn     │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Experiment Planner  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Code / Experiment   │
│ Runner              │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Analysis Agent      │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Critic / Falsifier  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Theory Update Agent │
└──────────┬──────────┘
           │
           ├── hypothesis supported
           ├── hypothesis weakened
           ├── hypothesis rejected
           └── new hypothesis generated
           │
           └──────────────► NEXT LOOP
```

所有模块共享：

```text
Scientific State / Research Memory
```

---

# 4. Critical Architectural Rule

不要让多个 Agent 通过自由对话控制系统。

采用：

```text
Python Orchestrator
        │
        ├── calls Literature Agent
        ├── calls Hypothesis Agent
        ├── calls Experiment Planner
        ├── calls Code Agent
        ├── calls Analysis Agent
        ├── calls Critic Agent
        └── calls Theory Agent
```

即：

> **Agents reason. Code controls.**

Python 决定：

- 当前处于哪个阶段
- 输入输出 schema
- 哪个 Agent 被调用
- 是否进入下一轮
- 实验是否执行
- 数据存储
- budget
- retry
- provenance
- reproducibility

LLM 不允许直接改变全局 scientific state。

所有修改必须通过结构化 output。

---

# 5. Repository Structure

创建：

```text
autonomous_scientist/
│
├── README.md
├── SPEC.md
├── pyproject.toml
├── .env.example
│
├── config/
│   ├── project.yaml
│   ├── models.yaml
│   └── experiments.yaml
│
├── scientist/
│   │
│   ├── orchestrator.py
│   ├── state.py
│   ├── schemas.py
│   │
│   ├── agents/
│   │   ├── literature.py
│   │   ├── hypothesis.py
│   │   ├── experiment.py
│   │   ├── analysis.py
│   │   ├── critic.py
│   │   └── theory.py
│   │
│   ├── tools/
│   │   ├── literature_search.py
│   │   ├── code_execution.py
│   │   ├── statistics.py
│   │   └── experiment_tracker.py
│   │
│   └── prompts/
│       ├── literature.md
│       ├── hypothesis.md
│       ├── experiment.md
│       ├── analysis.md
│       ├── critic.md
│       └── theory.md
│
├── environments/
│   ├── base.py
│   ├── stationary.py
│   ├── sensor_shift.py
│   ├── dynamics_shift.py
│   ├── gradual_drift.py
│   └── recurring_env.py
│
├── algorithms/
│   ├── base.py
│   └── generated/
│
├── experiments/
│   ├── runner.py
│   ├── evaluator.py
│   └── metrics.py
│
├── database/
│   ├── papers/
│   ├── hypotheses/
│   ├── experiments/
│   └── results/
│
├── artifacts/
│
├── tests/
│
└── run_scientist.py
```

---

# 6. Scientific State

整个项目必须维护一个唯一的：

```python
ScientificState
```

建议 schema：

```python
class ScientificState:
    question: ScientificQuestion

    knowledge:
        list[Claim]

    hypotheses:
        list[Hypothesis]

    experiments:
        list[Experiment]

    results:
        list[ExperimentResult]

    evidence:
        list[Evidence]

    open_questions:
        list[ResearchQuestion]

    current_round: int

    budget:
        ResearchBudget
```

任何 Agent 必须读取 ScientificState。

任何 Agent 不可以自由修改 ScieH1ntificState。

必须返回 structured output，由 orchestrator 验证之后写入。

---

# 7. Core Data Structures

## 7.1 Hypothesis

```python
class Hypothesis:
    id: str

    claim: str

    mechanism: str

    assumptions: list[str]

    predictions: list[str]

    alternative_explanations: list[str]

    falsification_conditions: list[str]

    supporting_evidence: list[str]

    contradicting_evidence: list[str]

    parent_hypotheses: list[str]

    status: Literal[
        "proposed",
        "testing",
        "supported",
        "weakened",
        "rejected"
    ]

    confidence: float
```

例如：

```yaml
id: H003

claim:
  Prediction-error-triggered adaptation is sufficient
  for autonomous adaptation after environmental shift.

prediction:
  Adaptation intensity should correlate with useful
  prediction errors.

falsification:
  Large prediction errors that do not impair behavior
  should not trigger useful adaptation.

status:
  proposed
```

---

# 8. Evidence Structure

不能简单存：

> Paper A says X.

使用：

```python
class Evidence:
    id: str

    source_type:
        paper | experiment | analysis

    source_id: str

    claim: str

    supports:
        list[hypothesis_id]

    contradicts:
        list[hypothesis_id]

    strength:
        weak | moderate | strong

    limitations:
        list[str]
```

目标最终形成：

```text
Evidence Graph

Paper
  ↓
Claim
  ↓
Hypothesis
  ↓
Experiment
  ↓
Evidence
  ↓
Theory
```

---

# 9. Agent 1 — Literature Agent

输入：

```text
Scientific Question
Current Hypotheses
Current Open Questions
Existing Knowledge
```

任务：

1. 搜索已有相关理论和工作；
2. 不仅总结论文；
3. 提取：
   - mechanism
   - assumptions
   - evidence
   - limitations
   - contradictions
4. 输出结构化 Claims；
5. 发现 unresolved questions。

输出：

```json
{
  "claims": [],
  "contradictions": [],
  "knowledge_gaps": [],
  "candidate_mechanisms": []
}
```

不要生成长篇 review。

目标是建立：

```text
Knowledge Map
```

而不是文章摘要数据库。

---

# 10. Agent 2 — Hypothesis Agent

输入：

```text
Scientific Question
Knowledge Map
Existing Hypotheses
Previous Experimental Results
```

任务：

生成 3–6 个：

```text
competing hypotheses
```

重要：

假设应该相互具有区别。

例如不能生成：

```text
H1: prediction error helps adaptation
H2: prediction error may improve adaptation
```

而应该：

```text
H1:
prediction error is sufficient

H2:
novelty is sufficient

H3:
behavioral controllability loss is required

H4:
adaptation requires active exploration

H5:
parameter update is unnecessary; internal state is sufficient
```

每个 hypothesis 必须：

```text
Claim
Mechanism
Prediction
Alternative explanation
Falsification condition
```

缺一不可。

---

# 11. Agent 3 — Experiment Planner

这里是整个系统最重要的 Agent。

它的目标不是：

> 找最高 accuracy。

而是：

> 设计最能区分 hypotheses 的实验。

Experiment schema：

```python
class Experiment:
    id: str

    hypotheses_tested:
        list[str]

    scientific_question:
        str

    intervention:
        str

    control:
        str

    environment:
        dict

    algorithms:
        list[str]

    metrics:
        list[str]

    expected_outcomes:
        dict

    falsification_logic:
        dict

    seeds:
        list[int]

    estimated_cost:
        float
```

---

# 12. Experiment Selection

MVP 不需要一开始实现复杂 Bayesian Information Gain。

定义 heuristic score：

```text
experiment_score =

0.30 × discriminability
+
0.25 × falsifiability
+
0.20 × feasibility
+
0.15 × scientific_importance
-
0.10 × cost
```

每个维度：

```text
0–1
```

Experiment Planner 输出候选实验。

Orchestrator 选择最高 score。

以后再升级成：

```text
Expected Information Gain
```

---

# 13. Experimental World

不要一开始进入真实机器人或者 RRAM。

首先构造：

# Adaptive Intelligence Gym

使用统一环境 interface：

```python
class AdaptiveEnvironment:

    def reset():
        ...

    def observe():
        ...

    def step(action):
        ...

    def apply_shift():
        ...

    def hidden_evaluate():
        ...
```

最重要：

```text
environment change information
```

不得暴露给 agent。

agent 不知道：

```text
task_id
shift_time
environment_id
```

---

# 14. 第一代环境

至少实现五种：

## E0 — Stationary

环境不变化。

测试算法是否：

> 没有变化却过度学习。

---

## E1 — Sudden Sensor Shift

例如：

```text
x → Ax + noise
```

或者：

```text
normal image → noisy image
```

测试：

```text
perceptual adaptation
```

---

## E2 — Dynamics Shift

例如：

```text
action → motion
```

映射发生变化。

测试：

```text
control adaptation
```

---

## E3 — Gradual Drift

环境参数缓慢变化：

```text
θ_t → θ_t + Δθ
```

测试：

```text
continuous adaptation
```

---

## E4 — Recurring Environment

```text
A → B → C → A
```

测试：

```text
adaptation
+
retention
+
reuse
```

---

# 15. Important Experimental Rule

Agent during adaptation:

不得获得：

```text
ground-truth label
task ID
environment ID
change point
```

但是 Evaluator 可以访问 hidden ground truth。

因此：

```text
Agent information
        !=
Evaluation information
```

需要在代码上严格隔离。

增加 automated test：

```python
test_no_hidden_information_leak()
```

---

# 16. What Does “No Teaching Signal” Mean?

必须明确：

项目禁止的是：

```text
sample-level label
explicit desired output
task ID
manual adaptation trigger
```

但不禁止预先定义：

```text
a global behavioral objective
```

例如：

```text
maintain stability
navigate successfully
avoid failure
```

第二阶段进一步建立更严格 benchmark：

```text
Mode A:
No labels, but global task utility exists.

Mode B:
No online external reward;
only observation-action consequences are available.

Mode C:
Agent additionally chooses exploratory actions.
```

不要在第一版混合这些概念。

---

# 17. Algorithm Interface

无论 AI 自动生成什么算法，都必须遵循：

```python
class AdaptiveAgent:

    def reset(self):
        ...

    def observe(self, observation):
        ...

    def act(self):
        ...

    def adapt(self):
        ...

    def get_internal_state(self):
        ...
```

禁止 experiment runner 根据算法类型写特殊逻辑。

---

# 18. Seed Baselines

这些不是系统最终答案。

只是基础 comparison library：

```text
B0 — Frozen model
B1 — Continuous self-supervised update
B2 — Prediction-error-gated update
B3 — Novelty-gated update
B4 — Memory/context adaptation
B5 — Meta-adaptation
B6 — Random update control
```

Hypothesis Agent 可以：

```text
reuse baseline
modify baseline
combine mechanisms
create new algorithm
```

但不能被限定在这些算法里。

---

# 19. Core Metrics

不能只测 accuracy。

至少记录：

## Performance

[
P(t)
]

---

## Adaptation Recovery

[
R =
\frac{P\_{\text{adapted}}-P\_{\text{after shift}}}
{P\_{\text{before shift}}-P\_{\text{after shift}}}
]

---

## Adaptation Latency

[
T\_{adapt}
]

---

## Forgetting

P\_{\text{old,after}}
]

---

## Update Cost

```text
number of parameter updates
number of samples
FLOPs
memory
```

以后加入：

```text
energy
hardware writes
```

---

## False Adaptation

环境没有发生变化的时候：

```text
unnecessary adaptation events
```

非常重要。

因为：

> 一个一直更新的模型不等于 autonomous adaptation。

---

# 20. Code Agent

Code Agent 不参与科学结论。

它只负责：

```text
implement experiment
implement candidate algorithm
write tests
execute code
debug
save artifacts
```

每个实验创建：

```text
experiment_id
git commit
config
seed
environment version
algorithm version
```

保证完全 reproducible。

---

# 21. Experiment Runner

实验 runner 必须：

```python
result = run_experiment(
    agent=agent,
    environment=environment,
    seed=seed,
    config=config
)
```

输出：

```json
{
  "metrics": {},
  "trajectory": [],
  "adaptation_events": [],
  "compute_cost": {},
  "artifacts": []
}
```

不要让 LLM 自己“观察 terminal 然后总结”。

Raw data 必须先结构化保存。

---

# 22. Analysis Agent

Analysis Agent 只能读取：

```text
experiment configuration
raw results
statistics
plots
```

任务：

```text
What happened?
```

不能回答：

```text
Why must this be true?
```

输出：

```text
observations
statistical comparisons
unexpected results
possible interpretations
```

---

# 23. Critic / Falsifier Agent

这是系统核心组件。

它必须专门尝试：

> 推翻当前解释。

输入：

```text
Hypothesis
Experiment
Result
Analysis
```

强制回答：

```text
1. Is there an alternative explanation?

2. Is there a confound?

3. Does the experiment actually discriminate
   between competing hypotheses?

4. Is improvement simply caused by:
   - more updates?
   - more parameters?
   - more data?
   - larger compute?

5. What experiment could falsify
   the current conclusion?

6. Is the conclusion stronger than the evidence?
```

输出：

```json
{
  "validity": "...",
  "confounds": [],
  "alternative_explanations": [],
  "required_controls": [],
  "recommended_next_experiment": ""
}
```

---

# 24. Theory Update Agent

只有 Theory Agent 可以提出：

```text
scientific interpretation
```

但最终修改仍由 orchestrator 执行。

例如：

```text
H1: 0.60 → 0.25
H2: 0.30 → 0.65
H3: unchanged
```

不要把 confidence 当成真实 Bayesian probability。

第一版仅表示：

```text
internal belief score
```

同时更新：

```text
supported
weakened
rejected
```

并允许：

```text
H2 + experimental anomaly
        ↓
generate H7
```

---

# 25. Full Scientific Loop

实现：

```python
while not stop_condition:

    state = load_state()

    gaps = literature_agent(
        question=state.question,
        state=state
    )

    hypotheses = hypothesis_agent(
        knowledge=state.knowledge,
        previous_results=state.results
    )

    candidate_experiments = experiment_agent(
        hypotheses=hypotheses,
        state=state
    )

    experiment = select_best_experiment(
        candidate_experiments
    )

    implementation = code_agent(
        experiment
    )

    result = execute(
        implementation
    )

    analysis = analysis_agent(
        result
    )

    critique = critic_agent(
        experiment,
        result,
        analysis
    )

    state = theory_agent(
        state,
        result,
        analysis,
        critique
    )

    save_state(state)
```

---

# 26. Important Improvement

不要每轮重新做全部 literature review。

Literature Agent 应该根据：

```text
new hypothesis
new contradiction
new experimental anomaly
```

执行 targeted search。

即：

```text
Experiment anomaly
        ↓
new scientific question
        ↓
targeted literature search
```

形成真正的：

```text
Research Loop
```

---

# 27. Stopping Conditions

允许以下停止条件：

```text
max_rounds reached
compute budget exceeded
no testable hypothesis generated
all major hypotheses rejected
results repeatedly replicated
human stop
```

不能让 LLM 自己说：

> “The problem is solved.”

---

# 28. Scientific Provenance

每个结论必须能够反向追踪：

```text
Conclusion
     ↓
Evidence
     ↓
Experiment
     ↓
Raw Data
     ↓
Code Commit
     ↓
Configuration
```

同样：

```text
Hypothesis
     ↓
Claim
     ↓
Literature Source
```

这是必须实现的 feature，不是后期优化。

---

# 29. Storage

MVP：

```text
SQLite
+
JSON/YAML
+
filesystem artifacts
```

不要第一版就使用复杂 vector database。

数据库至少建立：

```text
questions
papers
claims
hypotheses
experiments
runs
evidence
decisions
```

实验大文件：

```text
artifacts/
```

单独保存。

---

# 30. Recommended Technology Stack

MVP：

```text
Python 3.12+
Pydantic
OpenAI Agents SDK
SQLite
SQLAlchemy
MLflow
Gymnasium
PyTorch
NumPy
Pandas
SciPy
Matplotlib
pytest
Git
```

可选：

```text
Docker
PostgreSQL
vector database
distributed workers
```

第一版不要上 Kubernetes。

---

# 31. Development Phases

## Phase 1 — Skeleton

目标：

跑通：

```text
Question
→ Hypothesis
→ Fake Experiment
→ Analysis
→ Theory Update
```

实验可以先是 mock。

验收：

```bash
python run_scientist.py
```

运行后生成：

```text
research_state.json
hypotheses.json
experiment.json
analysis.json
theory_update.json
```

---

# Phase 2 — Real Experimental Environment

实现：

```text
Stationary
Sensor Shift
Dynamics Shift
Gradual Drift
Recurring Environment
```

加入：

```text
Frozen baseline
Random adaptation baseline
```

验收：

系统能够自动运行不同 shift。

---

# Phase 3 — Candidate Learning Mechanisms

加入最基础的：

```text
prediction-based adaptation
novelty-based adaptation
continuous adaptation
memory/state adaptation
```

不要追求 SOTA。

目的是验证：

```text
scientific loop works
```

---

# Phase 4 — Falsification Loop

必须实现：

```text
Hypothesis
→ Experiment
→ Unexpected result
→ Critic
→ Control experiment
→ Updated hypothesis
```

这是整个 MVP 最关键的 milestone。

成功标准不是：

```text
accuracy ↑
```

而是系统真实完成一次：

```text
H1 proposed

↓ experiment

H1 contradicted

↓ critique

H1 weakened

↓ new hypothesis

H2 generated

↓ discriminative experiment
```

---

# Phase 5 — Literature Integration

加入真实论文检索。

实现：

```text
search
→ paper
→ claim extraction
→ evidence graph
```

Literature Agent 必须提供 source provenance。

---

# Phase 6 — Autonomous Multi-Round Discovery

允许系统：

```text
run 5–20 scientific cycles
```

自动决定：

```text
next hypothesis
next experiment
next search
```

最后自动生成：

```text
Scientific Discovery Report
```

包括：

```text
1. Question

2. Existing knowledge

3. Hypotheses considered

4. Experiments performed

5. Rejected hypotheses

6. Supported mechanisms

7. Remaining uncertainty

8. New scientific questions
```

---

# 32. MVP Success Criterion

第一版不要要求：

> AI 发现全新的类脑学习理论。

真正验收标准：

### Criterion 1

AI 能生成至少三个 genuinely competing hypotheses。

### Criterion 2

AI 能设计可以区分两个 hypothesis 的实验。

### Criterion 3

AI 可以自动实现并执行实验。

### Criterion 4

实验数据可以自动进入 ScientificState。

### Criterion 5

Critic 可以发现至少一种 confound。

### Criterion 6

系统能够因为实验结果修改 hypothesis。

### Criterion 7

系统自动选择下一轮实验。

### Criterion 8

完整结果可复现。

只要完成这八点：

> AI Scientist MVP 成立。

---

# 33. First Concrete Scientific Run

第一轮不要从复杂视觉网络开始。

构造一个简单动态 control environment。

流程：

```text
Environment A
     ↓
Agent operates normally
     ↓
Unknown dynamics shift
     ↓
Agent must detect/adapt
     ↓
Environment B
```

AI Scientist 首先回答：

> What information is actually necessary to detect and adapt to this change?

让 Hypothesis Agent 自动生成候选机制。

第一轮可以重点比较：

```text
H1:
continuous adaptation is sufficient

H2:
prediction-error-triggered adaptation is sufficient

H3:
novelty/change detection is required

H4:
behavioral failure signal is required

H5:
parameter adaptation is unnecessary;
latent-state adaptation is sufficient
```

然后让系统自己设计 discriminative experiments。

---

# 34. Second Scientific Run

第一轮稳定后，再加入：

```text
A → B → A
```

科学问题升级：

> Adaptation 与 continual adaptation 是否需要不同的机制？

观察：

```text
Does adaptation destroy A?

Does returning to A require relearning?

Can previous internal state be reused?
```

系统可能自然产生：

```text
memory-related hypotheses
```

而不是我们提前规定必须有 memory。

---

# 35. Third Scientific Run

加入：

```text
active exploration
```

允许 agent 主动选择：

```text
what action to perform
```

研究：

> Passive observation 是否足以适应，还是 intelligent exploration 是必要机制？

---

# 36. Future Extension — RRAM / Neuromorphic Hardware

只有 computational principles 得到一定结论以后，才进入：

```text
Device-aware Scientist
```

新的问题变成：

> 哪些物理器件性质适合实现已经发现的 adaptive mechanisms？

再引入：

```text
RRAM dynamics
conductance relaxation
stochasticity
asymmetric update
endurance
retention
threshold switching
```

研究：

```text
nonideality
```

究竟是：

```text
problem
```

还是：

```text
useful computational primitive
```

不要提前假定答案一定是 RRAM。

---

# 37. Development Priority

严格按照以下顺序开发：

```text
P0
ScientificState + schemas

P1
Orchestrator

P2
Toy dynamic environment

P3
Experiment runner + tracking

P4
Hypothesis Agent

P5
Experiment Planner

P6
Analysis Agent

P7
Critic/Falsifier

P8
Theory Update

P9
Literature Agent

P10
Multi-round autonomous loop

P11
UI / visualization

P12
Hardware extension
```

不要先开发漂亮 UI。

不要先开发十几个 Agent。

不要先开发 RRAM。

---

# 38. First Coding Task

现在首先完成：

```text
Milestone 1:
Minimal Scientific Loop
```

要求：

### 1. Create repository structure.

### 2. Implement Pydantic schemas:

```text
ScientificQuestion
Claim
Hypothesis
Experiment
ExperimentResult
Evidence
ScientificState
```

### 3. Implement deterministic orchestrator state machine:

```text
QUESTION
→ HYPOTHESIS
→ EXPERIMENT
→ ANALYSIS
→ CRITIQUE
→ THEORY_UPDATE
```

### 4. Agents initially can be mocked.

### 5. Implement persistent state using SQLite + JSON artifacts.

### 6. Add unit tests.

### 7. Implement one toy experiment demonstrating:

```text
Hypothesis H1
↓
experiment
↓
contradictory evidence
↓
H1 becomes weakened
↓
new experiment proposed
```

Do not implement literature search yet.

Do not implement RRAM yet.

Do not optimize model performance yet.

---

# 39. Definition of Done for Milestone 1

Executing:

```bash
python run_scientist.py --demo
```

must create:

```text
artifacts/run_xxx/

question.json

hypotheses.json

experiment_plan.json

raw_results.json

analysis.json

critique.json

theory_update.json

research_state.json
```

Terminal should show:

```text
Scientific Cycle 1

Question loaded
Hypothesis generated: H001
Experiment selected: E001
Experiment executed
Evidence contradicts H001
H001: proposed → weakened
Next experiment generated: E002

Cycle complete
```

If this works reliably and reproducibly, proceed to Milestone 2.

---

# 40. Guiding Principle

Whenever there is a design choice, prioritize:

```text
scientific traceability
>
agent autonomy

reproducibility
>
complexity

falsification
>
idea generation

structured state
>
conversation history

simple controlled experiments
>
large benchmark performance
```

The purpose of this project is not to build another chatbot swarm.

The purpose is to build:

> **a machine that maintains an explicit scientific state and iteratively changes its beliefs through executable evidence.**
