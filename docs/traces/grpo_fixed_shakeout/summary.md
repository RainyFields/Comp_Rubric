# GRPO rollout shakeout + trajectory audit — summary (2026-09-23)

Checkpoint: GRPO step 50 (Qwen3-8B, fix campaign). Code under test: **7c04e37** (= 3f697bf four audit fixes + 83a3070 shared grammar/capture +
shakeout-#1 fixes). Shakeout set: 18 hard/problematic BC-Plus test tasks (`data/bc_test_shakeout.parquet`), greedy, one 8×H100 pod per arm.
Arms: **A** compaction OFF with the branch tool (the training scaffold), **A'** compaction OFF without branch, **B** compaction ON
(q_sum/summary/resume, raw-id tail). Full-set (150 tasks) runs are in §7. All numbers below come from the captured token ids
(`trace_audit.jsonl`), not from reconstructed text.

## 1. Completion criteria (captured end-to-end, shakeout #2)

| criterion | evidence (A / A' / B) | status |
|---|---|---|
| no execution of think-block calls | calls inside `<think>` seen 4 / 9 / 13; executed-vs-executable sequence identical in 214/214, 122/122, 248/248 turns; think-internal calls never appear in `results` | **pass** |
| multiple calls: execution + observation mapping | multi-call turns 0 / 6 / 5 (up to 6 `open_page` in one turn, B rollout 1095); per-call results map 1:1 in order and the observation the policy saw is the concatenation of the per-call observations in 198/198, 113/113, 241/241 turns | **pass** |
| 2 048-token turn cap reaches the engine | max completion 2 048 in every arm (`cap_over` 0/0/0); turns at the cap 4 / 8 / 4 — all but one are thinks cut mid-way (no call emitted) | **pass** (behavioural cost in §4) |
| exact history preservation | every earlier completion's ids are a contiguous subsequence of every later prompt of the same context, and prompt_t = prompt_{t−1} − gen-prompt + stored turn + observation, in all turns except the *designed* rollbacks (branch forced-return 12, compaction q_sum 7) | **pass** |
| branch inheritance / compaction tail | fork sha1 exact 25/25 (A); tail sha1 exact 190/190 (B); `<|im_end|><|im_start|>` never glued (0 prompts, vs 603/694 in the old-code run) | **pass** |
| masks | stored-turn masks valid 239/239, 122/122, 297/297 (trainable = sampled ids incl. eos; generation prompt and terminator newline untrained); inherited/tail turns are appended with all-zero masks (unit-tested) | **pass** |
| termination + accounting | `rollout_end` event in 18/18 per arm with exact stop reasons; prompt sha1 replays 246/246, 138/138, 297/297; per-category token totals in §5 | **pass** |
| no loops | longest branch 4 turns → longest branch 13 steps; no rollout hit `max_turn`; loop protection did not need to fire in #2 | **pass** |

Both tokenizers: 58/58 unit tests (`tests/test_action_grammar.py`, `test_loop_protection.py`, `test_trace_audit_fixes.py`, …) on Qwen3-8B and Qwen3.5-9B.

## 2. Confirmed bugs (found by this shakeout) and what was done

| # | finding | evidence | fix (commit) |
|---|---|---|---|
| S1 | **Strict grammar + this checkpoint = branch loops.** The checkpoint closes `return` with `<function>` instead of `</function>` (241 of its `return` calls in the old-code 150-task run; 277/869 turns in shakeout #1). The old lenient harness ended the branch on the first `<function=return>` *substring*, so training never penalised it. Under the strict grammar the env replied "No function call was detected" and the branch re-emitted the same call: 5 branches ≥ 50 steps (max 101), 7 branches with ≥ 5 consecutive no-call turns, 246 degenerate `<think>\n`+EOS completions | `work/shakeout1/`, `readable_traces.md` last section | 7c04e37: an unclosed *terminal* opener ends the branch as a flagged **malformed return** (message extracted; `branch_end=malformed_return`, 14/25 branches in #2); `plugin.max_consecutive_no_call=3` forces a return in branches and stops main/compaction rollouts (`stop_reason=no_call_loop`) |
| S2 | env marked `is_finish=True` before validating the finish (empty answer / `must_search` / double-check), so a refused finish counted as finished in stats | unit test | 83a3070 |
| S3 | env `[Error] … not supported` used `=` and clobbered earlier calls' observations in a multi-call turn | unit test | 83a3070 |
| S4 | old env grammar executed a line-anchored call inside `<think>` when it sat within 4 newlines of the real call (the "adjacent-group" edge case) | unit test | 83a3070: think blocks are stripped before parsing |
| S5 | an accepted `finish` was not recorded in per-call results | audit | 7c04e37 |

Earlier fixes re-confirmed end-to-end by this capture: dead turn cap (F4), branch re-tokenisation (F1), two parsers (F3), missing `<|im_end|>\n` (F5).

## 3. Compaction OFF vs ON (same checkpoint; 18 tasks; NOT a training effect)

| | A branch (OFF) | A' no-branch (OFF) | B compaction (ON) | old code, A branch (18) |
|---|---|---|---|---|
| correct / finished / unfinished | **5** / 11 / 7 | 2 / 2 / 16 | 1 / 4 / 14 | 5 / 14 / 4 |
| stop reasons | finish 11, window exhausted 7 | window exhausted 16, finish 2 | budget exhausted (3 compactions) 14, finish 4 | finish 14, window 4 |
| main turns (mean) / branches / compactions | 5.1 / 1.4 / – | 7.7 / – / – | 5.8 / – / 2.7 | 6.9 / 4.1 / – |
| generated tokens per task: main / branch / summary / **total** | 3 245 / 825 / – / **4 069** | 3 121 / – / – / **3 121** | 3 054 / – / 2 227 / **5 281** | 2 832 / 2 908 / – / 5 740 |
| prefill tokens per task (sum of prompts) | 267 k | 174 k | 355 k | 822 k |
| peak active context (mean / max) | 34.9 k / 44.5 k | 39.6 k / 47.7 k | 34.0 k / 37.3 k | 36.6 k / 43.4 k |
| tool calls: search / open_page / branch / finish | 106 / 67 / 25 / 11 | 78 / 45 / – / 2 | 158 / 105 / – / 4 | 268 / 274 / 74 / 14 |
| duplicate search queries | 32 (30 %) | 20 (26 %) | 68 (43 %) | 113 (42 %) |
| turns at the cap (think cut) | 4 (3) | 8 (7) | 4 (3) | 4 over the cap |
| think: present / empty / none | 43 / 186 / 17 | 40 / 74 / 24 | 47 / 242 / 8 | 50 / 636 / 8 |
| citations grounded (cited docids ⊆ observed) | 5/5 rollouts | 2/2 | 4/4 | 3/3 |

Paired outcomes (A vs B): both correct 1, only A 4, only B 0, both wrong 13. A vs A': only A 5, only A' 2. So on this hard subset the
branch scaffold the checkpoint was trained with is the only configuration that finishes reliably; **compaction ON does not rescue the
no-branch policy**: 14/18 exhaust all three compactions.

**Why B fails (trajectory reading, not score reading).** The summaries themselves are well-formed (49/49 carry `<summary>` tags, median
777 tokens, always list the searches/pages already done) and the resume context is exact (tail sha1 190/190). The policy nevertheless
repeats: **59 % of the searches issued after a compaction repeat a query it had already run** (55/94), against 26 % repeats in the
no-branch arm without compaction. In most exhausted rollouts the loop already exists *before* the first compaction (e.g. rollout 178:
`search "VETLAM I … completion date"` ↔ `open_page 84381` five times in segment 0, then the same pair with docid 45588 in segment 1).
Compaction does not corrupt history (checked on ids) and does not lose the evidence (summaries name the candidate answers); it simply
gives a policy that was never trained to read a summary three more windows in which to repeat itself. The one compaction success (1095,
1 compaction, then `finish` in segment 1 with 3 grounded citations) shows the mechanism works when the policy has the answer.

**A' (no branch, no compaction)** exhausts the 40k window in 16/18: observations are large (10–20k tokens for one multi-result search; the
policy opens 5–6 pages in a turn) and nothing bounds the main context — the last observation can push the prompt past the window
(max 47.7k > 40 960), after which the next step returns `None` (`llm_none`). That is a harness limitation, not a policy decision.

## 4. Trajectory quality (what the rollouts actually do)

- **Sensible behaviour exists**: A/1095 (search → verification branch → grounded finish, correct), B/1095 (1 compaction, correct with citations),
  A'/581 (11 turns, correct). Correct answers cite docids that were actually observed (no ungrounded citation in any arm).
- **Dominant failure mode = repetition, not reasoning**: identical search re-issued 3× in a row (A/254 until the window dies), identical
  search/open pairs 5× (B/178), branches that re-run the same two actions (A/581 branch). 30–43 % of searches are duplicates. These are
  **policy failures** (the empty-think RL policy re-emits the last action pattern), amplified by the harness only in that observations
  are never de-duplicated.
- **Cap cost**: with the 2 048 cap now real, first-turn thinks that ran to 3–5k tokens under the old code are cut (A'/1095 loses its
  first two turns, 4 096 tokens, to "No function call was detected" before it starts). 3–7 turns per arm. Loop protection (3 consecutive)
  did not have to fire in #2, but a cap-cut think followed by two more would end a rollout early — a `max_consecutive_no_call` of 3 is
  tight for a thinking policy; consider 5 (see §6).
- **Malformed returns are the norm for branches** (14/25): the checkpoint learned `</parameter>\n<function>` as a return closer. Treating it
  as a return is the faithful reading of what it was trained on; the flag is in every capture/audit record.
- **Empty thinks**: 186/246 main+branch turns in A are `<think>\n</think>` — the trained policy's habit (known from the fix campaign), preserved
  byte-exactly in branches now.
- **Wrong-but-finished answers** (A/581 "1/2 cup" vs gold "90g"): the evidence for the specific fact was never found; the policy answered
  from a plausible default. Judge verdicts are clear-cut in these cases (no ambiguous-evaluation case found in the 18).
- **Window exhaustion** dominates the OFF arms' unfinished rollouts (A 7, A' 16): a **budget/harness failure** (unbounded observations),
  not a reasoning stop.

## 5. Cost accounting (shakeout #2, totals over 18 tasks)

| tokens | A | A' | B |
|---|---|---|---|
| generated main | 58 401 | 56 173 | 54 972 |
| generated branch | 14 841 | – | – |
| generated summary | – | – | 40 086 |
| prefill (sum of prompt ids over all steps) | 4.80 M | 3.14 M | 6.39 M |
| observations > 8k tokens | 14 | 17 | 35 |
| wall-clock per task (mean; 18-way concurrent, throughput-bound) | 43 s | 24 s | 71 s |

Compaction ON costs 1.3× the generated tokens and 1.3× the prefill of the branch arm on this subset (summaries are 43 % of B's generated
tokens) while finishing fewer tasks.

## 6. Uncertainties and prioritized recommendations

1. **Greedy decoding is not reproducible** on this stack: with byte-identical first prompts (sha1 18/18), shakeouts #1 and #2 produced the
   same first completion in only 1/18, 2/18, 3/18 tasks (vLLM TP-4 batching nondeterminism). Paired OFF/ON differences on 18 tasks are
   therefore noisy; the 150-task runs (§7) are needed for any accuracy claim, and even they compare *distributions*, not paired rollouts.
2. **Old-vs-new differences are not attributable**: the old run (glued boundaries, no cap, lenient returns) had 4.1 branches/task vs 1.4 now;
   part of that is the changed harness (F5 newline is out-of-distribution for a checkpoint trained without it; the cap; malformed-return
   handling), part is nondeterminism. A clean ablation would toggle F5 only. Recommendation: make the terminator newline a
   `plugin.turn_end_newline` flag (default on for new training, **off when evaluating checkpoints trained before 3f697bf**).
3. **Bound observations**: cap a single observation (e.g. 12k tokens, `keep_first_n_words` per page and per-search) so the main context
   cannot overshoot the window; today 16/18 no-branch rollouts die that way.
4. **Loop protection threshold**: raise `max_consecutive_no_call` to 5 for thinking policies, and add an explicit *duplicate-action* guard
   (identical call twice in a row → observation "you already ran this call; results unchanged") — the cheapest fix for the dominant failure.
5. **Compaction for this checkpoint** is not a fair test of CompactionRL (the policy was never trained with q_sum); the ON arm should be
   read as "inference-time compaction on a branch-trained policy". The mechanism itself (trigger, summary, exact tail, resume) is verified.
6. Judge/ground truth: one ambiguous case type to watch in the full set — answers correct in substance but in a different unit/format.

## 7. Full 150-task runs (compaction OFF vs ON, same GRPO step-50 checkpoint, code 7c04e37)

Jobs `abec53a537251edc` (A), `a2339ece197b992f` (A'), `7f8f41600dda264f` (B); greedy; 150-way concurrency per pod; results under
`results/valonly_grpo_50_greedy_fx_full_{A_branch,A_nobranch,B_compact}_sc/`; per-arm audits `work/full_*`, table `work/comparison_full.md`.
Integrity on captured ids (A / A' / B): prompt sha1 replay 3 186/3 186, 1 061/1 061, 2 061/2 061; masks 3 168/3 168, 936/936, 2 061/2 061;
fork exact 353/353; compaction tails exact 1 190/1 190; executed-vs-parsed identical 2 813/2 816 (the 3 are branch-guard / branch-limit
turns, i.e. tool artefacts), 936/936, 1 748/1 748; observation = concatenation of per-call observations 2 575/2 575, 865/866, 1 631/1 631;
`cap_over` 0 everywhere; no unexpected history loss or prefix break; `rollout_end` 150/150 per arm. **All completion criteria hold on 450 rollouts.**

### 7.1 Outcomes and stopping

| | A branch (OFF) | A' no-branch (OFF) | B compaction (ON) | old code, A branch (reference) |
|---|---|---|---|---|
| **correct** (of 150) | **55 (0.367)** | 13 (0.087) | 36 (0.240) | 51 (0.340) |
| finished / unfinished | 132 / 18 | 24 / 126 | 65 / 85 | — |
| stop reasons | finish 132, window exhausted 18 | window exhausted 125, finish 24, no-call loop 1 | budget exhausted (3 compactions) 83, finish 65, no-call loop 2 | — |
| wrong but finished | 77 | 11 | 29 | — |
| main turns (mean) / branches / compactions | 5.3 / 2.4 / – | 7.1 / – / – | 5.7 / – / 2.1 | – / ~2.8 / – |
| tasks that used 0 / 1 / 2 / 3 compactions | – | – | 28 / 22 / 9 / 91 | – |

Paired outcomes (same task, same checkpoint):
- **A vs B**: both correct 29, only A 26, only B 7, both wrong 88 → compaction ON is 19 tasks worse than the branch scaffold.
- **A' vs B**: both 8, only B 28, only A' 5 → compaction ON is 23 tasks better than the same single-thread agent without compaction.
- **A vs A'**: only A 43, only A' 1, both 12.

Reading: for this checkpoint the ordering is *branch scaffold (trained) > compaction > single window*. Compaction rescues the
single-thread agent from window exhaustion (A' dies at the window in 125/150; B never exceeds the window, `prompts_over_window` 0) and
finishes 65 tasks, but it does not reach the branch scaffold, because the policy repeats itself after a reset: **60 % of the searches issued
after a compaction repeat a query already run in that rollout (335/554), versus 12 % inside the first segment (56/480)** and 22–31 % in
the OFF arms. Accuracy falls with the number of compactions used: 17/28 with none, 10/22 with one, 4/9 with two, **5/91 with three** —
the 91 tasks that used all three compactions are essentially the tasks the policy could not solve, and 83 of them end by budget exhaustion.
Summaries are well-formed (313/313 tagged, median 763 tokens, 1 hit the cap) and the resume context is exact; the loss is behavioural
(a policy never trained to act on a summary), not mechanical.

### 7.2 Cost and behaviour

| per task (mean) | A | A' | B |
|---|---|---|---|
| generated: main / branch / summary / **total** | 2 757 / 1 537 / – / **4 293** | 2 848 / – / – / **2 848** | 3 420 / – / 1 690 / **5 110** |
| prefill (sum of prompt ids over steps) | 424 k | 164 k | 285 k |
| peak active context (mean / max) | 31.5 k / 45.8 k | 42.0 k / **234.3 k** | 31.2 k / 36.9 k |
| wall-clock per task (throughput-bound, 150-way) | 396 s | 112 s | 476 s |
| tool calls: search / open_page / branch / finish | 1 165 / 1 105 / 355 / 133 | 597 / 570 / – / 25 | 1 034 / 744 / – / 66 |
| duplicate searches | 364 (31 %) | 131 (22 %) | 391 (38 %) |
| multi-call turns / max calls in one turn | 35 / 5 | 45 / **92** | 51 / 24 |
| observations > 30 k chars | 290 | 234 | 304 |
| prompts beyond the 40 960 window | 11 | 70 | 0 |
| turns at the 2 048 cap (all think cuts) | 43 | 51 (38) | 50 (45) |
| "No function call" observations | 108 | 46 | 52 |
| branch endings: proper return / malformed return / forced | 161 / 190 / 1 | – | – |
| think: present / empty / none | 351 / 2 759 / 76 | 333 / 547 / 181 | 485 / 1 519 / 57 |
| correct answers with citations / fabricated-docid rollouts | 31 of 55 / 2 | 13 of 13 / 1 | 29 of 36 / 0 |

Per correct answer, B costs 21 k generated tokens (A: 11.7 k, A': 33 k) and 1.19 M prefill tokens (A: 1.16 M, A': 1.9 M) — compaction is
not cheaper than branching for this policy; its summaries alone are 33 % of B's generated tokens.

### 7.3 New findings from the full set

- **F-full-1 — unbounded multi-call turns blow the window.** With every parsed call now executed, the no-branch policy emitted turns with
  up to **92 `open_page` calls** (A' rollouts 1195, 885: 930 k / 780 k-char observations, prompt 234 k tokens → `llm_none`). 70 A' prompts and
  11 A prompts exceed the 40 960 window; every such rollout dies at the next step. Old code executed at most the last adjacent group and so
  hid this. Fix applied after these runs (commit below): `plugin.max_calls_per_turn` (default 8; the rest are rejected with an explicit
  error text); an observation token budget per turn remains recommended.
- **F-full-2 — malformed `return` is the majority ending of branches** (190/352): the flagged malformed-return path is load-bearing for this
  checkpoint.
- **F-full-3 — fabricated citations**: 3 wrong rollouts cite placeholder docids (`[12345]`, `[1234]`, `[123]`) that never appeared in any
  observation; all other cited docids were observed. Grounding is otherwise honest.
- **F-full-4 — cap cuts are think cuts**: 43–51 turns per arm are thinks cut at 2 048 tokens with no action (≈1.5–5 % of steps); loop
  protection then fired only 3 times in 450 rollouts.
- **F-full-5 — repetition dominates failures** in every arm: 22–38 % of searches are exact repeats; `no_call_loop` and window exhaustion,
  not reasoning stops, account for 145 of the 229 unfinished rollouts.
- Old-code reference (same checkpoint, run 222c7dd69460d364): 51/150 with 30 turns over the (dead) cap and every turn boundary glued;
  the new harness reaches 55/150 with the cap enforced. Given the greedy nondeterminism measured in §6.1, the two are indistinguishable.

### 7.4 Answer to the audit question

The GRPO trajectories are *behaviourally* sensible when they succeed (search → verify → grounded finish) and *mechanically* clean
throughout (history, masks, boundaries, compaction reconstruction all exact). Their failures are repetition and budget exhaustion, not
corrupted context. Inference-time compaction on this branch-trained checkpoint changes the outcome distribution (−19 tasks vs the branch
scaffold, +23 vs single window), the stopping profile (budget exhaustion replaces window exhaustion), search behaviour (repeat rate 12 % →
60 % after a reset) and cost (+19 % generated tokens, +33 % summaries); it does not change accuracy through history corruption.
