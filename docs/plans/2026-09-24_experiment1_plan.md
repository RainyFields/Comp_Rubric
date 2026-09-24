# Experiment I — side-by-side context-management evaluation on BrowseComp-Plus (Qwen3.5-9B / 27B): execution plan

Written 2026-09-24 (SF time) from the RubricComp plan (Lark doc `PhVAd96Pno0xw2xM1q5u8qN4sTf`, snapshot:
`docs/plans/2026-09-24_rubriccomp_experiment_plan_lark.md`, §5–§6, §13 Phase 0–1). Scope = H1 only: characterise how
GRPO/PPO, FoldGRPO, SUPO and CompactionRL behave end to end at 9B and 27B, produce validated checkpoints, a paired
evaluation with a full token/compute ledger, an archived trajectory set, and the segment dataset for the H2 gate.
No rubric PRM is used as a reward anywhere in Experiment I.

Status legend: ✅ exists and verified · 🔧 exists, needs adaptation · ❌ missing · ⏳ in progress.

## 1. Inventory: what the repo already provides vs what the plan requires

| Plan requirement | Repo state | Gap / work item |
|---|---|---|
| Qwen3.5 stack (§3.1) | 🔧 venv `fold_train_q35` built (verl 0.9.1, vLLM 0.24, transformers 5.9); 1-GPU smoke in flight | WP3 port of the vendored verl 0.7 code to pip verl 0.9.1 (`docs/handoff/NEXT_STEPS.md` step 4) — prerequisite for every cell |
| Qwen3.5-27B | ❌ never run here; weights on shared HDFS | multi-node (2×8 H100) training: TP=8 rollout, FSDP 27B + critic does not fit one node at 64k; reuse the SUPO multi-node ray recipe (`supo_codegym/scripts/merlin/job_entrypoint.sh`, `NCCL_NET_GDR_LEVEL`, IPv4 rack check) |
| GRPO/PPO baseline, no context management (§6 matrix, §9) | 🔧 `grpo` arm = fold scaffold **with** the branch tool exposed; `norl` = plain ReAct (search/open_page/finish) | add `grpo_nobranch` (= `workflow=search`, single window) as the plan's baseline; keep `grpo` (tool exposed) as the §9 "tool exposed" variant if budget allows |
| FoldGRPO | ✅ `foldgrpo` (paper process rewards; Qwen3-8B campaign done) | 64k window + the **global 256k allowance** for branches (new cap in `agents/fold_agent.py`) |
| SUPO | ❌ not in this repo; `~/xiaoxuan/supo_codegym` has `SupoAgentLoop` (CodeGym, verl HEAD + patch: `supo` estimator, overlong mask) | new arm `supo`: summarise-and-resume with SUPO semantics (trigger at 0.95·W after an (action, observation) pair which is dropped, `v_sum` prompt, continuation = original prompt + summary, no tail steps, overlong rollouts masked, rollout-level GRPO broadcast). Two routes — see Q1 |
| CompactionRL | 🔧 rollout + `compaction_gae` (PPO + critic) implemented and CPU-tested, never run on GPU; `compactiongrpo` (critic-free) ran on Qwen3-8B | first GPU run of the paper-faithful arm is the 9B shakeout; 27B critic memory — see Q3 |
| Unified budget 64k / 256k (§4.2) | 🔧 windows are `prompt_length + response_length` (8k + 32k today); compaction/summary caps exist (`max_compactions`, `val_max_compactions`); FoldGRPO has `max_session=10`, no global token allowance | `PROMPT_LENGTH=8192 RESPONSE_LENGTH=57344` (64k per call); `MAX_COMPACTIONS=3` / `MAX_SUMMARIES=3`; new **global logical-trajectory cap 256k** (counts new content once; forbids new branches/compactions and forces finish when hit); shared caps: `turn_max_new_tokens`, `max_calls_per_turn=8`, `max_consecutive_no_call`, tool calls per rollout, `session_timeout` — calibrated from the pilot captures |
| 128k no-compaction reference (§4.2) | ❌ | eval-only cells: `RESPONSE_LENGTH=122880`, `VAL_MAX_COMPACTIONS=0` / no branch tool, on base + baseline checkpoints (memory: 248k vocab × 128k logits → `use_fused_kernels`, TP=8) |
| Token/compute ledger (§4.3, I-C) | 🔧 `agents/e2e_ledger.py` = exact per-turn ids for main + branches (fold scaffold); `scripts/e2e_metrics.py` **missing from git** | ledger v2 for all scaffolds (compaction/SUPO segments, summary overhead), + prefill / uncached-prefill from the vLLM server metrics (`prefix_cache_hit`), wall-clock and GPU time per rollout; recover `scripts/e2e_metrics.py` from the old box |
| Preflight suite (§5 table) | ✅ `docs/PROTOCOL.md` + `PROTOCOL_VERIFICATION.md` rows E1–E9 (prefix identity, think blocks, grammar incl. multi-call turns, turn caps, branch/compaction reconstruction, masks, termination) on Qwen3-8B; template-level tests on Qwen3.5 | rerun the captured audits on Qwen3.5-9B/27B rollouts (`scripts/shakeout_audit.py`), add a `use_inference_chat_template` note (our loops build ids themselves; verl's own ToolAgentLoop setting is irrelevant but must stay off), 20 manually inspected 9B trajectories |
| Raw trajectory archive (§5.1) | 🔧 `FOLD_PROMPT_CAPTURE=1` (exact prompt/completion ids, masks, fork/tail events) + `FOLD_ROLLOUT_DUMP` (train) + val dumps | one archive schema per rollout (prompts + tool schemas, raw messages, ids, think/action spans, calls/observations, parent–child branches, summary in/out, masks, reward, ledger, termination) written by the capture hook; ≥ 20 pilot rollouts per model × method before RL |
| Data split (§3.2) | 🔧 680 train / 150 test; **in-training validation uses the 150 test tasks** | carve a validation set out of train (Q6) so the held-out 150 stay untouched until I-C |
| Judge / retriever | ✅ gpt-oss-120b judge, Qwen3-Embedding-8B retriever, Tevatron corpus (fixed since Sep) | unchanged |
| Statistics (§12) | 🔧 `scripts/analyze_fix_campaign.py` per-step tables | task-clustered bootstrap CIs, paired comparisons, accuracy–cost curves (`--model-tag` layout) |

## 2. Cells

Models M = {Qwen3.5-9B, Qwen3.5-27B}. Methods A = {`grpo_nobranch` (baseline), `foldgrpo`, `supo`, `compactionrl`}.

| Setting | Baseline | FoldGRPO | SUPO | CompactionRL |
|---|---|---|---|---|
| Native reference | 64k single window (same as unified) | **32k** window, 10 branches, no global cap (= Qwen3-8B campaign config on 9B) | 64k, ≤ 2 summaries, trigger 0.95·W | 64k, ≤ 3 compactions, T_comp 10 240 |
| Unified | 64k single window | 64k per thread, ≤ 10 branches, 256k allowance | 64k, ≤ 3 summaries, 256k | 64k, ≤ 3 compactions, 256k |
| 128k reference (eval only) | base + baseline ckpt at 128k | – | – | – |

Proposal to keep the matrix tractable (Q4): for SUPO and CompactionRL the native and unified settings differ only in the
summary cap / trigger, so **one training run per model serves both** (evaluate the checkpoint under both caps). Only
FoldGRPO needs a separate native (32k) run. Training runs per model = 4 unified + 1 FoldGRPO-native = 5; total 10 runs
(+ evals). Each run: 100 steps, 32 prompts × 8 samples, lr 1e-6 (Q3 for the PPO arm), val every 10 steps on the new
validation set, checkpoints base/every 10/final kept on HDFS.

## 3. Phases, gates, work items (owner: this agent unless noted)

### Phase 0 — infrastructure (plan §13.1; ≈ 2 weeks; no training)
0.1 ⏳ Qwen3.5 venv GPU smoke (`infra/jobs/fold_smoke_q35_venv_1gpu_h100.json`): torch cu130 / flash-attn / fla GDN / HF load / vLLM 0.24 + thinking completion. Gate: RC 0.
0.2 ❌ **WP3**: pip verl 0.9.1 + plugin (`foldgrpo`, `compaction_gae`, `compaction_grpo`, `supo` estimators via `register_adv_est`; `need_critic`; `global_token_mean` if 0.9.1 lacks it), agent loops on the 0.9 `AgentLoopBase` (7-arg ctor, `AgentLoopOutput` fields), `use_remove_padding=True`, `use_fused_kernels=True`, `causal_conv1d_implementation=fla`; vendored `verl/` moved to a branch. Gate: 73/73 CPU tests on both tokenizers with pip verl; 3-step `grpo_nobranch` shakeout on 9B (finite loss, grad_norm > 0, rollout/actor log-prob corr ≥ 0.95, GDN fast path, dumped batch masks) — closes verification row E11.
0.3 ❌ `supo` arm (Q1) + `grpo_nobranch` arm; prompts from `supo_codegym/supo/prompts.py`; CPU tests replaying the SUPO scripted-LLM tests from that repo.
0.4 ❌ unified-budget knobs: 64k windows, `MAX_SUMMARIES`, global 256k allowance, shared turn/call/time caps; 128k eval config; `MODEL_TAG` HDFS layout `fold_replication/<model_tag>/…`.
0.5 ❌ ledger v2 + archive schema + `scripts/e2e_metrics.py` recovery; prefill/uncached-prefill from vLLM metrics.
0.6 ❌ **I-A preflight on 9B** (no-RL captures, 18-task shakeout file, one job per scaffold: nobranch / branch / supo / compaction; `FOLD_PROMPT_CAPTURE=1`): `shakeout_audit.py` all rows PASS on the Qwen3.5 capture; over-limit and parser-failure tests; ledger reconciled with the vLLM counters on a sample; 20 trajectories read by hand (docs/traces). Repeat automated part on 27B (needs 0.7).
0.7 ❌ 27B rollout path: 2-node job spec (infra pod + 16 GPUs), TP=8 vLLM, FSDP + offload; smoke = no-RL eval on 18 tasks.
Gate P0→P1: 0.1–0.6 green on 9B; 0.7 green before any 27B training.

### Phase I-B — training (plan §6 Phase I-B; ≈ 4 days wall per 9B run, ≈ 10–12 per 27B run, plus queue time)
Order (9B first, one 8×H100 pod each + the shared infra pod; concurrency limited by the ark queue, see Q4):
1. `grpo_nobranch` unified → 2. `foldgrpo` unified → 3. `compactionrl` unified (3-step shakeout with `CRITIC_WARMUP=1` first) → 4. `supo` unified → 5. `foldgrpo` native 32k. Then the same on 27B.
Behavioural validity check per run (plan): context-management event frequency and timing, summary/return length, tool failures, reward components from the `[COMPACTION]`/branch log lines — a run that never compacts/branches or collapses into immediate compaction is flagged before I-C.
Every run: W&B `context_folding`, `infra/jobs/JOBS.tsv`, checkpoints to `fold_replication/<model_tag>/ckpt/<arm>/`, rollout dumps at steps 1, 50, 100 (archive samples).

### Phase I-C — paired evaluation (≈ 1 day per model)
Held-out 150 tasks, greedy + T=1 n=5 (plan: five rollouts per task) for every final checkpoint and the base model; compaction/SUPO arms under ×1 and their native/unified caps; FoldGRPO with/without the branch tool; baseline in the §9 three conditions (no tool / tool exposed / forced compaction) when the harness supports forced compaction (`FOLD_AGENT=compaction` on a non-compaction checkpoint already does). 128k references. Outputs: all-attempt success, completion, accuracy | completion, exhaustion/timeout; ledger families; task-clustered bootstrap CIs; accuracy vs generated tokens / uncached prefill / GPU time. Report per house standard (PDF + MD + assets + builder script).

### Phase I-D — segment dataset (after I-C at both scales)
Boundary extraction per method (branch call/return; pre-summary/resume; pre/post compaction) + length-matched fixed-token and tool-turn controls from the same trajectories; ≈ 200 boundaries per method × scale, stratified (difficulty, outcome, position, segment length, prior events); annotation packets with method label and terminal answer hidden; 20 % double-annotated. Decision gate for H2.

## 4. Compute estimate (from the Qwen3-8B 32k runs: 26–36 min/step on 8×H100 incl. val; 4 h reclamation grid)

| run | GPUs | est. wall (100 steps) | notes |
|---|---|---|---|
| 9B, 64k, GRPO-type arm | 8×H100 | ≈ 3–4 days | ≈ 2× the 32k step time; 248k vocab log-probs chunked |
| 9B, CompactionRL (PPO + critic) | 8×H100 | ≈ 5–6 days | critic forward/backward; `CRITIC_WARMUP=50` |
| 27B any arm | 16×H100 (2 nodes) | ≈ 10–12 days | untested here; SUPO repo ran Qwen2.5-32B GRPO multi-node |
| evals (150 tasks × 6 rollouts × settings) | 8×H100 | ≈ 3–6 h per checkpoint-setting | |

Ten training runs sequentially ≈ 2.5 months; with 3 concurrent 9B pods the 9B block fits in ≈ 1.5 weeks. Queue reality:
the ark H100 queue has pended 8-GPU jobs for hours/days; A100 schedules in minutes but is ≈ 2× slower.

## 5. Open questions for the user (answers change the work)

- **Q1 SUPO route.** (A, recommended) implement `supo` inside this repo by reusing the compaction rollout (`agents/compaction_agent.py`) with SUPO semantics (SUPO prompts, 0.95·W trigger dropping the last pair, no tail, overlong mask, rollout-level GRPO); or (B) port `supo_codegym/supo/agent_loop.py` (CodeGym loop) to the search tools. A keeps one protocol/verification path and the token ledger; B is closer to the replicated code.
- **Q2 Baseline.** The plan's baseline is "no context management": train `grpo_nobranch` (search/open_page/finish only, single 64k window). Our existing `grpo` arm exposes the branch tool. Train only `grpo_nobranch` in round 1 and treat `grpo` (tool exposed) as the §9 variant?
- **Q3 CompactionRL optimiser.** Paper = PPO + critic (`compactionrl`, never run on GPU). At 27B the critic doubles memory → likely 3 nodes or heavy offload. Proposal: paper PPO at 9B; decide 27B after the 9B run (fall back to the critic-free `compactiongrpo` only if PPO does not fit, and label it as an ablation).
- **Q4 Matrix and concurrency.** Accept the 10-run reduction in §2 (native = unified for SUPO/CompactionRL)? How many 8×H100 pods may run concurrently, and is the 2-node 27B configuration (16 H100) available under group 765? If not, 27B waits or goes to A100.
- **Q5 64k split and caps.** 8k prompt + 56k response per call; per-turn cap 4 096 new tokens (9B thinking; 2 048 today), `max_calls_per_turn=8`, `max_consecutive_no_call=3`, 100 model calls per rollout, 2 h wall-clock per rollout — to be calibrated from the Phase 0.6 pilots as the plan says. OK as starting values?
- **Q6 Validation split.** Today the 150-task test set is also the in-training validation set. Proposal: hold out 100 of the 680 train tasks as the validation set (seeded, fixed file `data/bc_val.parquet`), train on 580, keep the 150 test untouched until I-C. This changes the data split relative to the Qwen3-8B campaigns (documented, not comparable).
- **Q7 Seeds.** One seed per cell in round 1 (plan §12 asks for run-to-run variation later); a second seed only for the 9B baseline and FoldGRPO if pods are available.
- **Q8 Judge / retriever** stay gpt-oss-120b / Qwen3-Embedding-8B (unchanged from the Qwen3-8B campaign)?

Work that proceeds regardless of the answers: 0.1 (smoke), 0.2 (WP3 port), 0.4 window knobs, 0.5 ledger/archive.

## 6. Decisions (user, 2026-09-24) — supersede §5 where they differ

| # | Decision |
|---|---|
| D1 SUPO | Reuse the search/compaction rollout; reproduce SUPO semantics faithfully on BC-Plus: policy-generated summaries, trigger at 95 % of the working context, native max 2 summaries, SUPO overlong masking and advantage treatment. Port the CodeGym `SupoAgentLoop` only if the current rollout cannot express these exactly. |
| D2 Baseline | New `grpo_no_compaction` arm: no branch/summary/compaction tool exposed in training. Existing branch-tool-exposed `grpo` = additional ablation only. Evaluate the no-compaction checkpoint under (a) no CM tool, (b) CM tool exposed zero-shot, (c) harness-forced compaction. |
| D3 27B | 9B first for smoke/debugging; prepare the 27B path in parallel; 27B is NOT contingent on 9B gains (capacity is a variable of interest). |
| D4 Matrix | §2 reduction accepted for the initial phase. Before launching, deliver a complete table: method × model × native/unified × seeds × nodes/GPUs × estimated GPU-hours. Do not launch the full matrix yet. |
| D5 Caps | No fixed 8k/56k split. **64k per-call occupied-context ceiling** (prompt + generation of one call), initial **8k assistant-generation cap per call**, 100 assistant turns, 100 tool invocations, one parallel tool invocation. Calibrate the total generated-token cap from pilot trajectories. |
| D6 Validation | Hold out 100 of the 680 training tasks as fixed `data/bc_val.parquet` (`scripts/make_val_split.py`, seed 20260924, record `docs/splits/bc_val_split.json`; training file `data/bc_train_580.parquet`). The 150 test tasks stay untouched until the protocol and checkpoint-selection rules are frozen. |
| D7 Seeds | 1 seed for smoke/pilot runs only; ≥ 3 independent training seeds for the final key comparisons, staged. |
| D8 Judge/retriever | gpt-oss-120b + Qwen3-Embedding-8B fixed; also freeze corpus/index revision, retriever top-k, snippet/page truncation, judge prompt/version, answer normalisation (to be written into `docs/PROTOCOL.md` as the frozen eval contract). |
| D9 Gate | Before every training method: generate and save raw full trajectories with the exact finalised harness; 9B: ≥ 20 manually inspected. Archive = raw messages, prompt/generated token ids, thinking/action spans, loss masks, tool calls/responses, CM events, per-call active/occupied context lengths, token ledger, terminal reward, termination reason. |
| D10 verl 0.9.1 preflight | Before formal RL: compare the actual rollout token stream against the `apply_chat_template` reconstruction; test `use_inference_chat_template`; never suppress tokenisation sanity-check failures; confirm tool parser, turn accounting, branch/summary reconstruction, loss masks, multi-tool-call handling. |
| D11 Budget tracks | (A) native-reference per paper; (B) unified controlled budget. Do not hard-code 256k yet: collect native pilot distributions first, then pick the primary unified cap (192k or 256k), keep the other as sensitivity analysis. |
| D12 Order | Start with the verl 0.9.1 port, Phase-0 preflight and raw-trajectory collection; no training matrix until those pass. |

Implementation notes for D5 in this harness: the per-call ceiling maps to `max_model_len` = 65 536 with `prompt_length` +
`response_length` no longer a fixed split — the agent must cap each call at `min(8 192, 65 536 − occupied)` new tokens and
treat "occupied ≥ ceiling − margin" as window exhaustion (CM trigger for compaction/SUPO; forced finish for the baseline).
Turn accounting: `max_session`/turn counters become "assistant turns ≤ 100" and "tool invocations ≤ 100", `max_calls_per_turn=1`.
