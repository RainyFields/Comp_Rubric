<title>RubricComp: Segment-Level Rubric-Based Process Rewards for Long-Horizon Agentic Reinforcement Learning</title>

# Abstract

Long-horizon agentic reinforcement learning faces a fundamental credit-assignment problem: terminal rewards do not reveal which intermediate decisions caused success or failure. RubricComp asks whether context-management operations—invoked as tool calls for branching, summarization or compaction—can provide a useful granularity for decomposing trajectories into segments that support process supervision. Our core hypothesis is that these learned boundaries can expose locally coherent, independently evaluable units of progress; the first experiments explicitly test this assumption rather than taking it for granted.

We formulate three linked hypotheses. **H1:** segments produced by FoldGRPO, SUPO and CompactionRL are independently evaluable, with identifiable goals, evidence and completion status. **H2:** dynamically generated, task-conditioned rubrics provide more informative feedback about segment quality and action utility than hand-designed context-management penalties. **H3:** segment-level rubric feedback can improve long-horizon RL beyond terminal rewards and manually specified process rewards, including when the rubric PRM and policy co-evolve.

The first experimental round uses Qwen3.5-9B and Qwen3.5-27B to compare GRPO/PPO baselines with FoldGRPO, SUPO and CompactionRL, asking both whether the agents learn to compact in the intended way and whether their boundaries support reliable process feedback. Additional context-management techniques, such as ReSum, may be added in later rounds after the evaluation framework is validated.

**Keywords:** Long-horizon agents; Agentic RL; Context compaction; Process reward models; Rubric generation; Credit assignment; Test-time scaling.

---

# 1. Background and Motivation

Existing methods address limited working context through different learned operations: FoldGRPO uses branch tool calls to execute bounded subtasks and return compact results; SUPO trains an agent to summarize and resume when its working context reaches a threshold; and CompactionRL jointly optimizes compaction and continuation with PPO. These mechanisms are the first-round candidates for testing whether policies can learn useful compaction behavior and whether their induced boundaries provide an effective granularity for credit assignment. Additional mechanisms may be evaluated after this framework is validated.

These methods introduce two related research opportunities.

First, their context-management operations produce execution boundaries that may be suitable for process-level evaluation. However, a compaction boundary is not necessarily a meaningful subtask boundary. A branch may be poorly defined, unnecessary or impossible to evaluate independently. A summary may preserve the preceding trajectory accurately while contributing little to subsequent task completion.

Second, reducing active context does not necessarily reduce end-to-end computational cost. An agent may compensate for missing information by repeating searches, generating additional reasoning or delaying termination.

RubricComp's primary deliverable is a segment-level rubric PRM that uses compaction as a tool-call boundary and is co-trained with the policy to improve long-horizon task solving. The parallel token-efficiency analysis is supporting work: it measures information retention, repeated search, stopping and total cost, but it does not displace the PRM objective. Both efforts share rollout infrastructure and token accounting.

## 1.1 Related work and research positioning

Prior rubric-RL methods differ mainly in evaluation granularity and rubric construction. At the trajectory level, DR Tulu constructs instance-specific rubrics from retrieved evidence and maintains an evolving rubric buffer using on-policy responses, then scores complete long-form research answers during RL [7]. EvoLM alternates training a response policy with an instance-specific rubric generator whose criteria are optimized for discriminative utility on temporal comparisons between checkpoints, likewise yielding response-level rewards [8]. At finer granularity, ARCO generates per-step rubrics and co-evolves its rubric model with the policy [4], whereas ARBOR reuses criteria through a rubric buffer for online process rewards [5].

RubricComp differs in both boundary choice and training target. It treats context-management actions themselves—branch, summary and compaction tool calls—as candidate decomposition points, validates whether the resulting segments are coherent and independently evaluable, and then co-trains a segment-level rubric PRM with the policy. The goal is not rubric generation alone; it is to use learned compaction decisions to assign credit at a granularity that improves end-to-end long-horizon task solving.

# 2. Research Questions and Hypotheses

| Hypothesis | Research question | Primary experiment |
|-|-|-|
| H1 — Segment validity | Are segments produced by existing context-management methods independently evaluable, with identifiable goals, evidence and completion status? | Side-by-side method evaluation and post-training segment analysis |
| H2 — Rubric utility | Can dynamically generated, task-conditioned rubrics provide more informative process feedback than manually designed rewards and improve agent decisions? | Offline rubric validation; selective retry; best-of-K |
| H3 — RL effectiveness | Can segment-level rubric feedback improve long-horizon RL beyond terminal rewards and hand-crafted process rewards? | PRM-guided RL |

The hypotheses form explicit decision gates. H1 first tests whether context-management boundaries provide a defensible unit of decomposition. H2 then tests whether generated rubrics evaluate those units reliably and guide real decisions. PRM-guided RL under H3 begins only if both conditions are supported on held-out data.

# 3. Experimental Setup

## 3.1 Models

The primary models are Qwen3.5-9B and Qwen3.5-27B. Both will participate in the initial evaluation, segmentation study and subsequent RL experiments, so that the effects of context management are not inferred exclusively from the smaller model.

Both models natively support 262,144 tokens. The primary 64k setting is deliberately a constrained working-context experiment, rather than the default configuration recommended for unrestricted Qwen3.5 deployment. We will include 128k no-compaction references to test whether observed gains depend on the imposed context bottleneck.

An additional experiment will use Qwen3-8B to compare preserving and discarding historical thinking traces. As Qwen3-8B has a native context length of 32,768 tokens, this comparison will initially use 32k rather than introducing context extension as a confounding factor.

## 3.2 Tasks and datasets

BrowseComp-Plus will serve as the primary environment. It supports long-horizon search, evidence acquisition, dynamic task decomposition and verifiable final answers. It also provides a common setting for comparing the context-management methods.

All methods will use an identical training, validation and held-out test split on BrowseComp-Plus. Dataset versions, retrieval corpus, tool schemas, result limits and final-answer verification must be fixed before collecting the main experimental data. We will not directly compare our scores with the original papers when their data splits, model backbones or tool environments differ.

# 4. Context Budgets and Experimental Controls

## 4.1 Original paper configurations

The following settings have been verified against the original papers. They define the native-reference experiments rather than the unified cross-method comparison.

| Method | Original working context | Theoretical effective budget | Context-management limit |
|-|-|-|-|
| FoldGRPO | 32,768 | 327,680 | Up to 10 branches |
| SUPO, BrowseComp-Plus | 64k | 192k | Up to 2 summaries; trigger at 95% of working context |
| CompactionRL, GLM-4.7-Flash | 64k | 256k | Up to 3 compactions; trigger when remaining budget is below 10,240 tokens |

FoldGRPO's effective budget describes the theoretical allowance across branches. It is not equivalent to SUPO's sequence of summarized windows or CompactionRL's sequence of compacted execution segments.

## 4.2 Proposed budget-matched setting

## Unified primary comparison

Peak working-context limit

**64k**

Global unfolded allowance

**256k**

---

The 256k limit applies to newly accumulated logical trajectory content, not repeated branch prefixes or cumulative prefill. Each model call must also satisfy the 64k working-context constraint.

| Method | Unified setting |
|-|-|
| PPO / GRPO | Single 64k context; no compaction during training |
| SUPO | 64k; at most 3 summaries |
| CompactionRL | 64k; at most 3 compactions |
| FoldGRPO | 64k per active thread; up to 10 branches, subject to the global allowance |
| RubricComp | The same FoldGRPO constraints, with only the evaluator or reward mechanism changed |

For FoldGRPO, the branch limit will remain distinct from the summarization limits of the other methods. All methods will additionally share independently enforced limits for newly generated assistant tokens, tool invocations and wall-clock time. The exact generation and interaction caps will be calibrated from pilot trajectories before the main comparison, rather than chosen solely to make the theoretical context budgets appear equal.

A 128k no-compaction reference will be run for both model sizes. Where hardware permits, an additional near-native long-context reference will be evaluated with sufficient generation headroom.

## 4.3 Context and compute measurements

All metrics will be recorded per complete rollout, including unsuccessful, truncated and timed-out runs.

| Metric | Definition |
|-|-|
| Active context | Tokens visible to the model at an individual generation call |
| Peak prompt length | Maximum observed input length across the rollout |
| Peak occupied context | Maximum prompt plus generated tokens in a single model call |
| Logical trajectory length | Newly accumulated transcript content, counting each new message once |
| Generated tokens | All assistant-generated tokens, including reasoning, actions, summaries and returns |
| Prefill tokens | Total input tokens processed across model calls |
| Uncached prefill | Tokens actually recomputed after accounting for prefix-cache reuse |
| Tool-observation tokens | Environment content delivered to the agent |
| Context-management overhead | Additional generation and calls used for compaction or folding |
| Evaluator overhead | Rubric generation, verification tokens, calls and latency |

These measurements will be used to produce accuracy–compute trade-off curves. Reporting only peak context or effective context length would obscure repeated prefix processing and additional end-to-end generation.

# 5. Mandatory Preflight and Trajectory Integrity Validation

Before training any method, the finalized rollout harness must successfully generate and archive raw full trajectories using the exact configuration intended for training.

For Qwen3.5, we will continue using standard tool responses for environment observations. The official template serializes tool responses within dedicated tool-response wrappers. Historical thinking retention is handled separately from current-turn thinking generation.

VERL's `use_inference_chat_template` setting will be tested explicitly. Its documented default preserves the recorded full model output for subsequent rollout and training, whereas enabling it uses the inference template and may remove historical reasoning.

The preflight suite must cover:

| Area | Verification requirement |
|-|-|
| Chat-template fidelity | Compare actual rollout token prefixes against reconstructed `apply_chat_template` outputs and identify the first divergence |
| Thinking blocks | Verify reasoning delimiters, empty-block handling and historical thinking retention |
| Tool parser | Verify valid, malformed and interrupted tool calls |
| Multi-tool turns | Verify multiple calls in one assistant turn, execution order and response pairing |
| Turn accounting | Enforce assistant-turn and tool-invocation limits separately |
| Context reconstruction | Verify exact pre-compaction, post-compaction, branch and resumed-main-thread states |
| Training consistency | Verify loss masks, token IDs and log-probability calculations against the actual generation context |
| Termination | Verify finish, context exhaustion, tool failure, turn cap and timeout outcomes |

Initial manual inspection will cover at least 20 targeted Qwen3.5-9B trajectories, supplemented by automated tests on both 9B and 27B. Parallel tool execution will initially be disabled and enabled only after the multi-tool parser tests pass.

The initial primary configuration will preserve historical thinking. The Qwen3-8B extension will separately compare full-history preservation against discarding historical reasoning while keeping current-turn thinking enabled.

## 5.1 Mandatory raw trajectory archive

At least 20 complete, inspectable pilot rollouts per model and method must be saved before formal RL training. Samples should include successful execution, long trajectories, context-management events and abnormal termination.

The archive must contain the original prompts and tool schemas; raw model messages; prompt and generated token IDs; thinking and action spans; complete tool calls and observations; parent–child branch relationships; summary inputs and outputs; loss masks; rewards; token accounting; and termination metadata.

Model, tokenizer, training code, inference backend, chat template and environment revisions must be pinned. After RL begins, further trajectories will be collected from early, intermediate and final checkpoints. This archive is a mandatory experimental deliverable, not an optional debugging artifact.

# 6. Experiment I — Side-by-Side Context-Management Evaluation and Segment Analysis

**Scope:** H1; establishes the empirical base required for H2 and H3.

## Objective

Experiment I will first produce a side-by-side account of how the candidate context-management methods behave end to end. The immediate goal is to determine whether each method learns to branch, summarize or compact in the intended way; how that behavior changes task success, completion and failure modes; and what token and compute costs it creates. This comparison will be completed for both Qwen3.5-9B and Qwen3.5-27B before formal segment-validity analysis begins.

The segment study is therefore a downstream analysis of trained, validated policies rather than the starting point. It asks whether the boundaries observed in those completed runs provide a useful granularity for credit assignment and rubric-based process supervision.

## Design overview

| Stage | Primary question | Required output |
|-|-|-|
| I-A — System verification | Does each implementation preserve the intended prompts, reasoning history, tools, boundaries and token accounting? | A validated rollout harness and archived raw traces for every method. |
| I-B — Matched training | Can each 9B and 27B policy learn its intended context-management behavior? | Completed checkpoints, training curves and behavior diagnostics under native-reference and unified settings. |
| I-C — End-to-end evaluation | How do the methods compare in task performance, token use, compute and failure patterns? | Paired method comparisons and accuracy–cost curves on the held-out set. |
| I-D — Post-training segment analysis | Do the learned boundaries form coherent, independently evaluable units of progress? | A stratified segment dataset, human annotations and a go/no-go decision for rubric-PRM development. |

## Experimental matrix and controls

| Dimension | Levels and controls |
|-|-|
| Model scale | Qwen3.5-9B and Qwen3.5-27B. |
| Methods | No-context-management GRPO/PPO baseline; FoldGRPO; SUPO; CompactionRL. |
| Evaluation environment | BrowseComp-Plus only, using the fixed training, validation and held-out test split defined in Section 3. |
| Protocols | Native-reference settings for implementation validation; unified 64k active-context and 256k logical-trajectory allowance for the primary cross-method comparison. |
| Long-context reference | 128k no-compaction evaluation for both model scales, with sufficient reserved generation headroom. |
| Matched factors | Tokenizer and chat template, tool schemas, retrieval corpus, result limits, task IDs, rollout count, stopping rules, final-answer verifier and seed schedule. |
| Training accounting | Report prompts, rollouts, optimizer updates, generated tokens, prefill, wall time and GPU time. Native optimizers may differ, so native reproductions and unified mechanism comparisons will be reported separately. |

Experiment I will not use the proposed rubric PRM as a training reward. It is a characterization and model-production stage: terminal and original method-specific rewards are retained so that the learned context-management behavior can be measured before RubricComp changes the objective.

## Phase I-A — Preflight and implementation verification

Each method–model cell must pass the same preflight before formal training. The preflight will verify token-prefix identity between training and inference, correct chat-template rendering, preservation of historical reasoning and tool observations, stable parsing of tool calls and returns, deterministic turn accounting, and correct enforcement of context, generation, tool-call and wall-clock limits.

- Archive complete raw trajectories rather than only training tensors or scalar rewards.
- Confirm that FoldGRPO branches return to the correct main-thread state, SUPO summaries replace the intended history, and CompactionRL compactions occur at the configured trigger.
- Recompute token metrics from the archived transcript and reconcile them with server-side counters on a sampled subset.
- Run short over-limit and parser-failure tests to confirm truncation, retry and terminal-state behavior.

A method cannot enter the formal comparison until its trajectory reconstruction, parser behavior and token ledger pass these checks.

## Phase I-B — Training at 9B and 27B

All four conditions will be trained or reproduced at both model scales. Within a scale, runs will share the same task split, rollout infrastructure and evaluation harness. Native-reference runs will first confirm that each implementation learns its advertised context-management behavior. The unified runs will then impose the common resource constraints needed for cross-method comparison.

At minimum, base, early and final checkpoints will be retained. For every checkpoint, the archive will include the exact configuration, model identifier, optimizer state, reward components, training-step counters and a sample of raw rollouts. Learning curves will separately show terminal success, completion, context-management event frequency, event timing, summary or return length, tool failures and reward components. A run will be considered behaviorally valid only if the learned policy actually invokes its context-management mechanism at non-trivial frequency without collapsing into immediate compaction, uncontrolled branching or systematic avoidance.

## Phase I-C — Paired end-to-end evaluation

After training, every final checkpoint will be evaluated on the same held-out task IDs. Where sampling is stochastic, each method will receive the same number of rollouts per task and a predeclared seed schedule. The unit of comparison is the attempted task, and unsuccessful, truncated and timed-out rollouts remain in all primary analyses.

### Outcome metrics

| Metric | Interpretation |
|-|-|
| Task success | Verified correctness of the final answer; the primary quality outcome. |
| Completion rate | Fraction of attempts that produce a valid terminal answer before any limit. |
| Accuracy given completion | Separates answer quality from the ability to stop and submit. |
| Budget exhaustion and timeout | Identifies failures caused by generation, tool, context or wall-clock limits. |

### Token and compute ledger

The definitions in Section 4.3 will be applied to every rollout, not only successful ones.

| Metric family | Recorded quantities |
|-|-|
| Working-context occupancy | Active context at each call, peak prompt length and peak prompt-plus-generation occupancy. |
| Trajectory volume | Logical trajectory length, total assistant-generated tokens and tool-observation tokens. |
| Reprocessing cost | Total prefill tokens and uncached prefill after prefix-cache reuse. |
| Context-management overhead | Number and timing of branches, summaries or compactions; tokens and calls used to produce them; compression ratio; return or summary length. |
| End-to-end cost | Tool calls, model calls, wall-clock latency, GPU time and peak memory where available. |

### Behavior and failure diagnostics

Method behavior will be analyzed alongside aggregate scores. Diagnostics will include trigger position within the trajectory, events per rollout, information retained or lost across the boundary, repeated searches, repeated tool calls, recovery after compaction, premature stopping, failure to stop, invalid tool syntax and empty or unsupported returns. Error cases will be coded against a shared taxonomy so that a method cannot appear efficient merely because it fails early.

### Primary comparisons

Primary comparisons will be task-paired within each model scale under the unified budget. We will report task-clustered bootstrap confidence intervals for success and completion, together with accuracy versus generated tokens, accuracy versus uncached prefill, and accuracy versus GPU or wall-clock time. Native-reference results, unified-budget results and 128k no-compaction references will appear in separate panels. All-attempt metrics are primary; completed-only accuracy and success-conditioned token use are secondary diagnostics.

## Phase I-D — Post-training segment analysis

Formal segment analysis begins only after final checkpoints for FoldGRPO, SUPO and CompactionRL are available and validated at both 9B and 27B. Natural boundaries will be extracted from branch invocation and return for FoldGRPO, the pre-summary and resumed states for SUPO, and the pre-compaction and post-compaction states for CompactionRL. From the same trajectories, length-matched fixed-token and tool-turn boundaries will be constructed as controls.

The target is approximately 200 candidate boundaries per method–model-scale cell, subject to boundary availability. Sampling will be stratified by task difficulty, final outcome, boundary position, segment length and number of prior context-management events. The exact sample size will be finalized after a pilot estimates boundary prevalence and annotation variance, and the sampling rule will be frozen before the main annotation pass.

Reviewers will not see the method label or terminal answer. They will receive the local state needed to interpret the proposed segment, its tool interactions and its return or summary. At least 20% of segments will be independently annotated by two reviewers; disagreements will be adjudicated after agreement is measured with an ordinal reliability statistic.

### Segment-level annotation rubric

| Dimension | Operational question |
|-|-|
| Goal specificity | Is there a clear, bounded objective that can be stated without hindsight? |
| Independent verifiability | Can success or failure be judged without seeing the terminal answer? |
| Evidence sufficiency | Does the segment contain enough local evidence to support its claims? |
| Boundary coherence | Does the selected boundary contain one meaningful unit of work rather than an arbitrary transcript slice? |
| Return fidelity | Does the return or summary preserve findings, provenance, uncertainty and unresolved questions? |
| Downstream reuse | Is the segment's information subsequently used by the continuing policy? |

The primary segment outcome is the rate of independently verifiable boundaries. Secondary outcomes include the complete annotation profile, unscorable rate, reviewer agreement, method and scale effects, and association with downstream task success, repeated search and token cost. These analyses will distinguish whether a boundary is locally valid from whether it is causally useful.

## Deliverables and decision gate

Experiment I will deliver trained 9B and 27B checkpoints for every method, a reproducible paired comparison, a complete token-and-compute ledger, an error taxonomy, an archived trajectory set and a labeled segment dataset. H2 proceeds only if at least one natural boundary type is sufficiently common and reliably scorable across both scales to support rubric training. If no method produces stable evaluable segments, the next step will be to refine the boundary representation or add an explicit subtask-proposal action before investing in a rubric PRM.

# 7. Experiment II — Rubric Design and Offline Validation

(H2 · Offline)

The proposed rubric will cover the complete branch lifecycle rather than only judging a finished subtask.

## 7.1 Four-stage rubric

| Stage | Evaluation dimensions | Available information |
|-|-|-|
| Before branching | Necessity, expected information gain, cost and opportunity cost | Current main-thread state |
| Subtask proposal | Relevance, specificity, feasibility, scope and independent verifiability | Main-thread state and proposed branch description |
| Branch completion | Goal achievement, evidence-grounded correctness, useful progress and return fidelity | Branch description, execution trace and return |
| Downstream validation | Contribution to final success, later cost and information reuse | Matched continuation outcomes; offline validation only |

The first two stages will be evaluated separately. A well-defined subtask may still be unnecessary, while an otherwise valuable decision to branch may be implemented with a poorly formulated subtask.

## 7.2 Rubric generation and scoring

For each proposed branch, a task-conditioned rubric generator will construct concrete evaluation criteria from the current main-thread state and subtask description. The criteria must be fixed before observing branch completion.

A verifier will subsequently score the completed branch against the previously generated rubric. Initial implementation will keep rubric generation and verification frozen to establish reliability before attempting co-evolution.

The experiment will compare manually designed criteria, static generated rubrics and state-conditioned generated rubrics. These will be evaluated against human annotations, verifiable evidence and held-out error cases. Ratings must retain separate criterion-level scores, uncertainty and reasons rather than being reduced immediately to a single reward.

Downstream utility will initially be an independent validation target. It will not be inferred from a verifier's unsupported judgment that a branch probably helped.

# 8. Experiment III — Inference-Time Rubric Interventions

(H2 · Online)

The two proposed inference-time strategies will use the same validated rubric framework.

## 8.1 Selective retry

After a branch completes, the evaluator will determine whether the return satisfies the subtask rubric. Branches with identifiable, potentially repairable failures may be retried within the remaining budget.

The treatment will be compared with unmodified execution, random retry and self-reflection under equal additional generation allowances. Primary outcomes will include final task success, successful error correction, retry frequency and total additional inference cost.

## 8.2 Best-of-K branch selection

At a fixed branch-creation state, the agent will sample K candidate executions of the same proposed subtask. The verifier will rank candidates according to the frozen rubric, and the selected return will be supplied to an otherwise identical main-thread continuation.

Pilot experiments will use K = 2 and K = 4. Controls will include random selection and selection using the original hand-designed reward criteria. All candidate generation, verification and continuation costs will be counted.

## 8.3 Counterfactual branch utility

A smaller, controlled experiment will evaluate whether the decision to branch is itself useful. From the same main-thread checkpoint, the agent will continue either by creating the proposed branch or by remaining in the main thread. Paired continuations with repeated sampling will estimate the effect on final success, token consumption and task completion.

This comparison also provides evidence for calibrating a prospective branch-utility predictor. Such a predictor will only use information available at the original decision point.

The inference-time results will determine whether the available process signal is sufficiently reliable and actionable to justify integration into RL.

# 9. Experiment IV — Baseline Training and Cross-Condition Evaluation

## 9.1 Training baselines

Both Qwen3.5 model sizes will be evaluated using the base checkpoint, standard GRPO, PPO or Turn-PPO, FoldGRPO, SUPO and CompactionRL.

Original reward assignments, normalization methods and compaction mechanisms will be preserved in native-reference runs. Unified-budget experiments will isolate differences attributable to method rather than context allowance.

In addition, standard PPO and GRPO will be trained without exposure to compaction tools. The resulting checkpoints will be evaluated in three conditions.

| Inference setting | Compaction tool in prompt | Compaction behavior |
|-|-|-|
| No compaction | No | Original execution |
| Tool exposed | Yes | Autonomous, zero-shot use |
| Forced compaction | Not required | Harness-triggered summary and reconstruction |

The tool-exposed prompt must include the tool schema and a concise description of its effects. It will not be included during the original PPO/GRPO training. Forced compaction will be treated as a separate intervention rather than evidence of learned tool-use ability.

Trained context-management checkpoints will also undergo native versus altered-inference comparisons where the tool schema and checkpoint remain compatible.

# 10. Experiment V — PRM-Guided Reinforcement Learning

(H3)

## Objective

Test whether validated, segment-level rubric rewards improve long-horizon credit assignment beyond terminal rewards and FoldGRPO's existing hand-designed process rewards.

The primary training comparison will hold the underlying FoldGRPO context-management architecture, data, rollout budget and optimization configuration constant while modifying only the reward source.

| Condition | Reward |
|-|-|
| Terminal only | Final task outcome |
| Original FoldGRPO | Terminal reward plus original hand-designed process rewards |
| Fixed rubric PRM | Terminal reward plus frozen segment-level rubric feedback |
| Co-evolving rubric PRM | Terminal reward plus periodically updated rubric feedback |
| Co-evolving PRM with safeguards | Updated rubric feedback with independent verification and drift controls |

The initial rubric-conditioned reward will distinguish branch-proposal decisions, branch execution and return-summary quality. Each reward will be assigned at its corresponding decision boundary and propagated through the selected advantage estimator. It will not simply be copied without normalization to every token in a segment.

Training implementation must specifically test whether variable numbers of branches distort loss weighting or reward propagation. PPO-based experiments will maintain correct temporal propagation across compaction boundaries, while FoldGRPO-based experiments must preserve rollout-level group normalization and account for variable-length branches.

## 10.1 Co-evolution

The rubric generator and verifier will initially be pretrained or calibrated using independent annotated trajectories. During RL, candidate rubric updates will be produced from newly collected on-policy behavior.

An update will only be accepted after evaluation against a frozen reference set and independent verification. Historical rubrics, evaluator checkpoints and reward distributions will be archived to identify drift, reward saturation and reward hacking.

## 10.2 Ablations

The main ablations will isolate rubric adaptation, reward granularity, inclusion of proposal-stage reward, inclusion of summary-fidelity reward and independent verifier safeguards.

A separate turn-level PRM condition will be evaluated with an otherwise matched training pipeline to determine whether segment-level credit assignment provides benefits beyond simply adding process feedback.

# 11. Parallel Work — Token-Efficient Compaction

The efficiency study will reuse all baseline trajectories while maintaining a separate hypothesis and optimization objective.

Its central question is whether compaction changes the agent's continuation policy in ways that increase total generated tokens, even when active context is reduced.

The initial analysis will decompose excess generation into repeated evidence acquisition, additional reasoning, redundant tool calls, delayed stopping and summary overhead. It will compare token consumption at equal task success, rather than treating shorter unsuccessful rollouts as more efficient.

Candidate interventions will include progress-aware summaries, explicit stopping signals and budget-aware compaction. These will be evaluated independently of RubricComp rewards. Any successful intervention can subsequently be included as an additional context-management source in RubricComp's generalization experiments.

# 12. Statistical Analysis and Reporting

The primary task-level outcome will be final success on held-out tasks. Results will be reported separately for 9B and 27B, with identical task splits and matched sampling conditions.

The initial evaluation target is five rollouts per task. The number of tasks and training seeds will be finalized following pilot variance estimates and a prospective power analysis. Paired comparisons will use task-clustered bootstrap confidence intervals, ensuring that multiple rollouts of the same question are not treated as independent tasks.

Segment-level analyses will account for segments nested within trajectories and tasks. Human-annotation agreement, rubric calibration and the prevalence of unscorable subtasks will be reported. Training experiments will additionally report variation across independent runs.

Primary comparisons must include all attempted tasks. Completed-only accuracy, success-conditioned token usage and overlong rates will be reported as secondary diagnostics.

## Planned figures and tables

| Output | Main purpose |
|-|-|
| Figure 1 — RubricComp framework | Show branch proposal, execution, rubric evaluation and RL feedback |
| Figure 2 — Budget-aligned baselines | Compare success under native and unified context budgets |
| Figure 3 — Segment validity | Compare natural branches against matched alternative boundaries |
| Figure 4 — Rubric reliability | Show human agreement, calibration and evidence-based error detection |
| Figure 5 — Inference-time interventions | Compare selective retry and best-of-K with equal-budget controls |
| Figure 6 — RL results | Compare terminal, hand-designed, fixed-rubric and co-evolving rewards |
| Figure 7 — Accuracy–compute trade-off | Show success against generated tokens, prefill and GPU time |

All headline tables should include the number of tasks, number of rollouts, budget configuration and uncertainty estimates.

# 13. Execution Plan and Decision Gates

1. **Phase 0 — Infrastructure and preflight** Deploy both primary models, validate chat templates and parsers, implement unified token accounting and archive raw full trajectories. No formal RL run may begin before its own finalized harness passes this gate.
2. **Phase 1 — Baseline collection and FoldGRPO training** Collect paired baseline evaluations for both model sizes. Train and archive the missing FoldGRPO checkpoints in parallel with rubric-schema development.
3. **Phase 2 — Segment validity and rubric validation** Complete human and automated segment evaluations, establish verifiability rates, and test frozen rubric reliability. If natural boundaries prove unreliable, examine subtask-boundary refinement before introducing PRM rewards.
4. **Phase 3 — Inference-time experiments** Run selective retry and best-of-K with matched compute controls. Establish whether rubric feedback can change downstream outcomes, not merely retrospectively classify branches.
5. **Phase 4 — PRM-guided RL** Compare fixed and co-evolving rubrics against terminal-only and hand-designed process rewards in both model sizes. Archive training-stage trajectories and evaluator versions.
6. **Phase 5 — Generalization and efficiency** Evaluate a second task environment, longer effective horizons and the parallel token-efficiency interventions. Produce the final accuracy–cost analysis.

Each phase will produce a concise report containing the hypothesis, completed experiments, principal findings, key figures and numbers, observed failures, active jobs and the next experimental decision. Raw trajectories, configurations and checkpoint identifiers will be maintained separately so that reports remain readable without losing reproducibility.

# 14. Anticipated Limitations

The main methodological risk is that a branch may be easy to evaluate locally but have little causal effect on the final outcome. We will distinguish local task correctness from downstream utility through matched continuations, rather than treating their correlation as proof of causal contribution.

A second risk is evaluator–policy co-adaptation. The evolving policy may learn to satisfy its rubric while failing the original task. Independent evidence checks, frozen evaluation sets and terminal-success measurements are therefore required throughout training.

Finally, differences between the original papers' environments and budget definitions may prevent exact numerical reproduction. Native-reference results and newly controlled comparisons will be clearly separated. Any modifications to loss normalization, branch semantics or compaction triggers will be documented rather than described as faithful reproductions.

# References

[1] [Scaling Long-Horizon LLM Agent via Context-Folding](https://arxiv.org/abs/2510.11967). Context-Folding and FoldGRPO; foundational branch structure and original process rewards.

[2] [Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management](https://miaolu3.github.io/ArXiv_SUPO.pdf). SUPO; summarization-based training and effective context budgeting.

[3] [CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents](https://arxiv.org/abs/2607.05378). PPO-based compaction training, cross-trajectory credit assignment and original budget configurations.

[4] [ARCO: Adaptive Rubrics with Co-Evolution for Multi-Step LLM-Based Agents](https://arxiv.org/abs/2606.21262). Dynamic step-level rubrics and policy–rubric co-evolution.

[5] [ARBOR: Online Process Rewards via a Reusable Rubric Buffer for Search Agents](https://arxiv.org/abs/2606.03239). Reusable process-reward criteria and online rubric updates.

[6] [VERL: Multi-turn Rollout Support](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html). Rollout tokenization, historical reasoning preservation and training–inference template consistency.

[7] [DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research](https://arxiv.org/abs/2511.19399). Retrieved-evidence-conditioned, instance-specific rubrics that evolve with on-policy long-form research responses.

[8] [EvoLM: Self-Evolving Language Models through Co-Evolved Discriminative Rubrics](https://arxiv.org/abs/2605.03871). Alternating co-training of a response policy and an instance-specific rubric generator optimized for discriminative utility.

The central experimental dependency is now explicit: segment validity must be established before trusting segment-level rewards, and inference-time utility must be measured before committing to full co-evolving PRM training. At the same time, baseline training and raw trajectory collection can proceed in parallel for both model sizes.