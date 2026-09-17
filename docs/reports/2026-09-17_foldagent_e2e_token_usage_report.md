# FoldAgent on BrowseComp-Plus: context efficiency vs end-to-end token usage

*Build: `python3 docs/reports/build_e2e_report.py` (this file, assets and figures are regenerated from the HDFS ledgers). Repo commit at build time: `50e4e8b`.*

## 1. Question and design

Does FoldAgent's folding reduce **active context** while *increasing* the total interaction needed to solve the same task?
Three efficiencies are separated: **context efficiency** (peak / mean active context per forward pass), **trajectory efficiency**
(turns, tool calls, tokens generated), and **end-to-end inference efficiency** (every token generated or returned before anything was
folded away, and the cumulative model input over all forward passes).

**Cells.** Training ∈ {no-RL Qwen3-8B base, GRPO checkpoint 60 (59 real steps), FoldGRPO checkpoint 50 (46 real steps)} ×
inference ∈ {**no fold**: `plugin.workflow=search` (the branch tool is removed from the prompt; single-thread ReAct with search/open_page/finish),
**fold**: `plugin.workflow=search_branch` (the training-time prompt with the branch/return tools; the policy decides when to branch, a branch
inherits the main context, runs its own sub-trajectory and is folded to its return message)}. Both RL checkpoints were trained with the
branch tool available: "vanilla GRPO" is the authors' baseline (same agent, plain GRPO, no fold-specific process reward), not a branch-free
ReAct-GRPO. The no-fold cells are therefore an inference-time ablation for every model.

**Evaluation.** Identical to the fix-campaign final evals: 150-item BC-Plus test split, two decodings — greedy (n=1) and T=1.0 top-p=1.0 n=4 —
prompt 8k + response window 32k per thread, `max_turn` 100 across threads, up to 10 branches, gpt-oss-120b judge, self-contained 8×H100 val-only
batch jobs (`infra/worker_val_selfcontained.sh`, `FOLD_WORKFLOW`, `FOLD_E2E_LEDGER=1`). Same tasks and same sampling settings across cells; task-level
pairing by `instance_id` (t1n4 pairs the per-task mean of 4 samples).

**Ledger.** `agents/e2e_ledger.py` records, for every thread of every rollout, each turn's exact token ids the policy generated or was fed
(`scripts/e2e_metrics.py` defines every metric). Folded-away branch tokens are counted in full; a branch's inherited context is counted as
model input of its forward passes, not as new tokens.

| cell | mode | results dir tag | rollouts | status | trainer val score |
|---|---|---|---|---|---|
| base_nofold | greedy | norl_base_greedy_e2e_nobranch_sc | 150 | DONE | 0.087 |
| base_nofold | t1n4 | norl_base_t1n4_e2e_nobranch_sc | 600 | DONE | 0.073 |
| base_fold | greedy | norl_base_greedy_e2e_branch_sc | 150 | DONE | 0.147 |
| base_fold | t1n4 | norl_base_t1n4_e2e_branch_sc | 600 | DONE | 0.180 |
| grpo_nofold | greedy | grpo_60_greedy_e2e_nobranch_sc | 150 | DONE | 0.153 |
| grpo_nofold | t1n4 | grpo_60_t1n4_e2e_nobranch_sc | 600 | DONE | 0.182 |
| grpo_fold | greedy | grpo_60_greedy_e2e_branch_sc | 150 | DONE | 0.327 |
| grpo_fold | t1n4 | grpo_60_t1n4_e2e_branch_sc | 600 | DONE | 0.317 |
| fold_nofold | greedy | foldgrpo_50_greedy_e2e_nobranch_sc | 150 | DONE | 0.067 |
| fold_nofold | t1n4 | foldgrpo_50_t1n4_e2e_nobranch_sc | 600 | DONE | 0.053 |
| fold_fold | greedy | foldgrpo_50_greedy_e2e_branch_sc | 150 | DONE | 0.373 |
| fold_fold | t1n4 | foldgrpo_50_t1n4_e2e_branch_sc | 600 | DONE | 0.317 |

## 2. Main tables

### 2.1 All rollouts — greedy
*tokens:*

| cell (train / inference) | n | success | e2e tokens | e2e median | generated | observations | fold in | fold gen | cum. input |
|---|---|---|---|---|---|---|---|---|---|
| base / no fold | 150 | 0.087 | 36,478 | 39,906 | 4,272 | 27,040 | 0 | 0 | 117,601 |
| base / fold | 150 | 0.147 | 25,545 | 20,538 | 4,581 | 14,594 | 1,374 | 3,199 | 103,720 |
| GRPO / no fold | 150 | 0.153 | 36,754 | 39,044 | 3,159 | 28,430 | 0 | 0 | 110,043 |
| GRPO / fold | 150 | 0.327 | 62,627 | 45,094 | 4,301 | 51,531 | 1,755 | 2,872 | 379,794 |
| FoldGRPO / no fold | 150 | 0.067 | 39,315 | 39,814 | 2,064 | 32,082 | 0 | 0 | 150,772 |
| FoldGRPO / fold | 150 | 0.373 | 101,087 | 80,578 | 4,311 | 88,538 | 3,154 | 3,384 | 595,451 |

*interaction and context:*

| cell (train / inference) | n | turns | tool calls | branches | fwd passes | peak ctx | mean ctx | main growth | overlong | post-fold tokens | post-fold turns | post-fold tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base / no fold | 150 | 5.6 | 5.3 | 0.00 | 5.6 | 31,778 | 19,033 | 31,329 | 0.76 | – | – | – |
| base / fold | 150 | 7.3 | 2.6 | 1.72 | 7.3 | 20,213 | 12,276 | 9,394 | 0.11 | 8,678 | 4.4 | 1.1 |
| GRPO / no fold | 150 | 5.4 | 5.0 | 0.00 | 5.4 | 30,734 | 19,335 | 31,605 | 0.71 | – | – | – |
| GRPO / fold | 150 | 19.5 | 12.0 | 2.43 | 19.5 | 31,135 | 18,293 | 10,832 | 0.11 | 38,554 | 14.4 | 8.3 |
| FoldGRPO / no fold | 150 | 6.8 | 6.7 | 0.00 | 6.8 | 33,871 | 21,637 | 34,166 | 0.91 | – | – | – |
| FoldGRPO / fold | 150 | 31.4 | 18.6 | 4.43 | 31.4 | 32,426 | 17,343 | 4,894 | 0.03 | 73,970 | 25.0 | 14.6 |


### 2.2 All rollouts — T=1.0, n=4
*tokens:*

| cell (train / inference) | n | success | e2e tokens | e2e median | generated | observations | fold in | fold gen | cum. input |
|---|---|---|---|---|---|---|---|---|---|
| base / no fold | 600 | 0.073 | 37,334 | 39,758 | 3,526 | 28,641 | 0 | 0 | 133,625 |
| base / fold | 600 | 0.180 | 25,703 | 21,678 | 4,685 | 14,647 | 1,374 | 3,272 | 109,332 |
| GRPO / no fold | 600 | 0.182 | 43,786 | 39,118 | 3,312 | 35,309 | 0 | 0 | 112,726 |
| GRPO / fold | 600 | 0.317 | 63,203 | 48,729 | 4,511 | 51,908 | 1,743 | 2,970 | 391,104 |
| FoldGRPO / no fold | 600 | 0.053 | 39,117 | 40,035 | 1,972 | 31,975 | 0 | 0 | 152,424 |
| FoldGRPO / fold | 600 | 0.317 | 82,285 | 84,103 | 3,784 | 70,840 | 2,597 | 3,024 | 477,142 |

*interaction and context:*

| cell (train / inference) | n | turns | tool calls | branches | fwd passes | peak ctx | mean ctx | main growth | overlong | post-fold tokens | post-fold turns | post-fold tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base / no fold | 600 | 6.2 | 6.0 | 0.00 | 6.2 | 32,517 | 20,020 | 32,185 | 0.82 | – | – | – |
| base / fold | 600 | 7.7 | 2.5 | 1.75 | 7.7 | 20,170 | 11,977 | 8,692 | 0.08 | 8,631 | 4.6 | 1.1 |
| GRPO / no fold | 600 | 5.5 | 5.1 | 0.00 | 5.5 | 31,155 | 19,510 | 38,638 | 0.67 | – | – | – |
| GRPO / fold | 600 | 19.8 | 12.3 | 2.38 | 19.8 | 31,924 | 18,698 | 11,261 | 0.12 | 38,808 | 14.6 | 8.6 |
| FoldGRPO / no fold | 600 | 7.0 | 6.9 | 0.00 | 7.0 | 33,864 | 21,238 | 33,968 | 0.91 | – | – | – |
| FoldGRPO / fold | 600 | 25.8 | 15.4 | 3.61 | 25.8 | 32,447 | 17,491 | 4,860 | 0.04 | 53,846 | 19.0 | 11.0 |


### 2.3 Successful rollouts only — greedy
*tokens:*

| cell (train / inference) | n | success | e2e tokens | e2e median | generated | observations | fold in | fold gen | cum. input |
|---|---|---|---|---|---|---|---|---|---|
| base / no fold | 13 | 1.000 | 22,031 | 20,648 | 2,619 | 14,249 | 0 | 0 | 60,636 |
| base / fold | 22 | 1.000 | 20,096 | 18,512 | 3,412 | 10,830 | 864 | 1,862 | 78,467 |
| GRPO / no fold | 23 | 1.000 | 23,118 | 22,492 | 2,336 | 15,617 | 0 | 0 | 79,277 |
| GRPO / fold | 49 | 1.000 | 38,621 | 29,233 | 2,921 | 29,515 | 1,171 | 1,917 | 237,970 |
| FoldGRPO / no fold | 10 | 1.000 | 23,426 | 24,462 | 1,587 | 16,669 | 0 | 0 | 92,984 |
| FoldGRPO / fold | 56 | 1.000 | 58,653 | 48,424 | 3,086 | 48,343 | 2,184 | 2,430 | 330,826 |

*interaction and context:*

| cell (train / inference) | n | turns | tool calls | branches | fwd passes | peak ctx | mean ctx | main growth | overlong | post-fold tokens | post-fold turns | post-fold tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base / no fold | 13 | 4.1 | 2.9 | 0.00 | 4.1 | 22,031 | 14,410 | 16,880 | 0.00 | – | – | – |
| base / fold | 22 | 5.9 | 2.3 | 1.09 | 5.9 | 18,626 | 11,818 | 7,807 | 0.00 | 2,466 | 1.9 | 0.2 |
| GRPO / no fold | 23 | 4.7 | 3.6 | 0.00 | 4.7 | 23,118 | 16,208 | 17,966 | 0.00 | – | – | – |
| GRPO / fold | 49 | 13.8 | 8.6 | 1.59 | 13.8 | 26,864 | 15,600 | 6,593 | 0.00 | 15,682 | 7.7 | 4.3 |
| FoldGRPO / no fold | 10 | 5.7 | 4.6 | 0.00 | 5.7 | 23,426 | 15,839 | 18,274 | 0.00 | – | – | – |
| FoldGRPO / fold | 56 | 20.6 | 12.0 | 3.07 | 20.6 | 27,852 | 14,494 | 2,754 | 0.00 | 33,427 | 14.1 | 7.8 |


### 2.4 Successful rollouts only — T=1.0, n=4
*tokens:*

| cell (train / inference) | n | success | e2e tokens | e2e median | generated | observations | fold in | fold gen | cum. input |
|---|---|---|---|---|---|---|---|---|---|
| base / no fold | 44 | 1.000 | 22,177 | 22,110 | 2,141 | 14,871 | 0 | 0 | 86,203 |
| base / fold | 108 | 1.000 | 23,787 | 21,483 | 3,380 | 14,350 | 1,066 | 1,923 | 86,989 |
| GRPO / no fold | 109 | 1.000 | 23,650 | 23,772 | 2,259 | 16,232 | 0 | 0 | 81,853 |
| GRPO / fold | 190 | 1.000 | 38,794 | 32,678 | 3,093 | 29,444 | 1,241 | 1,990 | 240,487 |
| FoldGRPO / no fold | 32 | 1.000 | 23,408 | 24,499 | 1,498 | 16,753 | 0 | 0 | 94,055 |
| FoldGRPO / fold | 190 | 1.000 | 48,835 | 44,372 | 2,925 | 39,079 | 1,802 | 2,351 | 265,507 |

*interaction and context:*

| cell (train / inference) | n | turns | tool calls | branches | fwd passes | peak ctx | mean ctx | main growth | overlong | post-fold tokens | post-fold turns | post-fold tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base / no fold | 44 | 5.3 | 4.2 | 0.00 | 5.3 | 22,177 | 15,321 | 17,029 | 0.00 | – | – | – |
| base / fold | 108 | 6.7 | 2.8 | 1.30 | 6.7 | 20,383 | 12,018 | 7,255 | 0.00 | 5,926 | 3.2 | 0.9 |
| GRPO / no fold | 109 | 4.9 | 3.7 | 0.00 | 4.9 | 23,650 | 16,178 | 18,505 | 0.00 | – | – | – |
| GRPO / fold | 190 | 14.1 | 8.5 | 1.68 | 14.1 | 27,556 | 15,966 | 7,001 | 0.00 | 14,635 | 7.3 | 3.7 |
| FoldGRPO / no fold | 32 | 5.9 | 4.9 | 0.00 | 5.9 | 23,408 | 15,678 | 18,268 | 0.00 | – | – | – |
| FoldGRPO / fold | 190 | 17.1 | 10.0 | 2.53 | 17.1 | 27,759 | 14,414 | 2,637 | 0.00 | 23,159 | 10.3 | 5.6 |


Columns: *e2e tokens* = prompt + generated + framing + observations + fold prompts/returns (everything before discard); *fold in* = branch prompts,
overflow-summary prompts and folded return messages injected into contexts; *fold gen* = policy tokens spent on branch-call / return / summary turns;
*cum. input* = Σ input length over all forward passes; *peak ctx* = max input+output length of any forward pass; *post-fold* = after the first fold
(rollouts that folded; NaN otherwise). Means with bootstrap CIs are in `assets/cell_summary.csv`.

### 2.5 Termination
Greedy:

| cell | n | finish | max_turn | session timeout | window exhausted | overlong (main ≥ 32k) | folded ≥1 | branch limit |
|---|---|---|---|---|---|---|---|---|
| base / no fold | 150 | 24% | 0% | 0% | 76% | 76% | 0% | 0% |
| base / fold | 150 | 89% | 0% | 0% | 11% | 11% | 87% | 0% |
| GRPO / no fold | 150 | 29% | 0% | 0% | 71% | 71% | 0% | 0% |
| GRPO / fold | 150 | 89% | 0% | 0% | 11% | 11% | 85% | 1% |
| FoldGRPO / no fold | 150 | 8% | 0% | 0% | 92% | 91% | 0% | 0% |
| FoldGRPO / fold | 150 | 97% | 0% | 0% | 3% | 3% | 98% | 13% |

T=1.0 n=4:

| cell | n | finish | max_turn | session timeout | window exhausted | overlong (main ≥ 32k) | folded ≥1 | branch limit |
|---|---|---|---|---|---|---|---|---|
| base / no fold | 600 | 18% | 0% | 0% | 82% | 82% | 0% | 0% |
| base / fold | 600 | 92% | 0% | 0% | 8% | 8% | 87% | 0% |
| GRPO / no fold | 600 | 33% | 0% | 0% | 67% | 67% | 0% | 0% |
| GRPO / fold | 600 | 78% | 0% | 10% | 12% | 12% | 85% | 0% |
| FoldGRPO / no fold | 600 | 9% | 0% | 0% | 91% | 91% | 0% | 0% |
| FoldGRPO / fold | 600 | 62% | 0% | 34% | 4% | 4% | 98% | 0% |

## 3. Paired task-level comparisons

Greedy (one rollout per task):

| comparison | metric | n tasks | mean Δ [95% CI] | median Δ | % tasks Δ>0 | sign-test p |
|---|---|---|---|---|---|---|
| FoldGRPO/fold − GRPO/fold (same tasks) | success | 150 | 0.047 [-0.027, 0.120] | 0.000 | 13% | 0.281 |
| FoldGRPO/fold − GRPO/fold (same tasks) | total_e2e_tokens | 150 | 38,459 [26,422, 50,763] | 20,903 | 69% | 0.000 |
| FoldGRPO/fold − GRPO/fold (same tasks) | turns | 150 | 11.9 [8.0, 16.1] | 9.0 | 69% | 0.000 |
| FoldGRPO/fold − GRPO/fold (same tasks) | tool_calls | 150 | 6.6 [4.0, 9.3] | 5.0 | 64% | 0.000 |
| FoldGRPO/fold − GRPO/fold (same tasks) | peak_ctx | 150 | 1,291 [-138, 2,639] | 322 | 54% | 0.369 |
| FoldGRPO/fold − GRPO/fold (same tasks) | cum_prompt_tokens | 150 | 215,657 [132,849, 303,859] | 138,373 | 63% | 0.001 |
| FoldGRPO: fold − no fold | success | 150 | 0.307 [0.227, 0.387] | 0.000 | 33% | 0.000 |
| FoldGRPO: fold − no fold | total_e2e_tokens | 150 | 61,772 [50,780, 72,857] | 38,906 | 83% | 0.000 |
| FoldGRPO: fold − no fold | turns | 150 | 24.6 [21.3, 28.0] | 19.0 | 90% | 0.000 |
| FoldGRPO: fold − no fold | tool_calls | 150 | 11.9 [9.7, 14.0] | 8.5 | 81% | 0.000 |
| FoldGRPO: fold − no fold | peak_ctx | 150 | -1,444 [-2,766, -231] | 234 | 54% | 0.326 |
| FoldGRPO: fold − no fold | cum_prompt_tokens | 150 | 444,679 [368,854, 520,038] | 295,167 | 85% | 0.000 |
| GRPO: fold − no fold | success | 150 | 0.173 [0.100, 0.247] | 0.000 | 21% | 0.000 |
| GRPO: fold − no fold | total_e2e_tokens | 150 | 25,873 [18,899, 33,639] | 8,762 | 65% | 0.000 |
| GRPO: fold − no fold | turns | 150 | 14.1 [11.5, 16.7] | 9.5 | 81% | 0.000 |
| GRPO: fold − no fold | tool_calls | 150 | 6.9 [5.2, 8.8] | 4.0 | 66% | 0.000 |
| GRPO: fold − no fold | peak_ctx | 150 | 401 [-1,282, 1,904] | 766 | 52% | 0.683 |
| GRPO: fold − no fold | cum_prompt_tokens | 150 | 269,751 [212,670, 328,635] | 168,095 | 76% | 0.000 |
| FoldGRPO/no fold − GRPO/no fold | success | 150 | -0.087 [-0.147, -0.027] | 0.000 | 3% | 0.011 |
| FoldGRPO/no fold − GRPO/no fold | total_e2e_tokens | 150 | 2,561 [710, 4,314] | 630 | 57% | 0.086 |
| FoldGRPO/no fold − GRPO/no fold | turns | 150 | 1.4 [0.9, 1.9] | 2.0 | 68% | 0.000 |
| FoldGRPO/no fold − GRPO/no fold | tool_calls | 150 | 1.7 [1.2, 2.3] | 2.0 | 69% | 0.000 |
| FoldGRPO/no fold − GRPO/no fold | peak_ctx | 150 | 3,136 [1,878, 4,434] | 1,714 | 61% | 0.009 |
| FoldGRPO/no fold − GRPO/no fold | cum_prompt_tokens | 150 | 40,728 [27,185, 54,122] | 46,297 | 73% | 0.000 |
| base: fold − no fold | success | 150 | 0.060 [-0.013, 0.127] | 0.000 | 13% | 0.136 |
| base: fold − no fold | total_e2e_tokens | 150 | -10,933 [-13,864, -7,929] | -13,123 | 27% | 0.000 |
| base: fold − no fold | turns | 150 | 1.7 [0.8, 2.7] | 1.0 | 55% | 0.012 |
| base: fold − no fold | tool_calls | 150 | -2.7 [-3.4, -2.0] | -3.0 | 21% | 0.000 |
| base: fold − no fold | peak_ctx | 150 | -11,566 [-13,656, -9,514] | -14,278 | 23% | 0.000 |
| base: fold − no fold | cum_prompt_tokens | 150 | -13,880 [-35,702, 7,751] | -24,714 | 37% | 0.001 |
| FoldGRPO/fold − GRPO/no fold | success | 150 | 0.220 [0.140, 0.300] | 0.000 | 26% | 0.000 |
| FoldGRPO/fold − GRPO/no fold | total_e2e_tokens | 150 | 64,332 [53,559, 75,039] | 43,157 | 88% | 0.000 |
| FoldGRPO/fold − GRPO/no fold | turns | 150 | 26.0 [22.7, 29.4] | 21.0 | 97% | 0.000 |
| FoldGRPO/fold − GRPO/no fold | tool_calls | 150 | 13.6 [11.4, 15.7] | 10.0 | 89% | 0.000 |
| FoldGRPO/fold − GRPO/no fold | peak_ctx | 150 | 1,692 [311, 3,142] | 1,466 | 63% | 0.001 |
| FoldGRPO/fold − GRPO/no fold | cum_prompt_tokens | 150 | 485,407 [412,540, 561,705] | 334,877 | 94% | 0.000 |

T=1.0 n=4 (per-task mean of 4 samples):

| comparison | metric | n tasks | mean Δ [95% CI] | median Δ | % tasks Δ>0 | sign-test p |
|---|---|---|---|---|---|---|
| FoldGRPO/fold − GRPO/fold (same tasks) | success | 150 | 0.000 [-0.035, 0.035] | 0.000 | 19% | 0.392 |
| FoldGRPO/fold − GRPO/fold (same tasks) | total_e2e_tokens | 150 | 19,082 [15,006, 23,338] | 17,582 | 73% | 0.000 |
| FoldGRPO/fold − GRPO/fold (same tasks) | turns | 150 | 6.0 [4.6, 7.5] | 6.4 | 76% | 0.000 |
| FoldGRPO/fold − GRPO/fold (same tasks) | tool_calls | 150 | 3.1 [2.2, 4.1] | 3.1 | 71% | 0.000 |
| FoldGRPO/fold − GRPO/fold (same tasks) | peak_ctx | 150 | 523 [-92, 1,142] | 526 | 57% | 0.086 |
| FoldGRPO/fold − GRPO/fold (same tasks) | cum_prompt_tokens | 150 | 86,039 [56,464, 115,781] | 76,748 | 68% | 0.000 |
| FoldGRPO: fold − no fold | success | 150 | 0.263 [0.210, 0.315] | 0.000 | 49% | 0.000 |
| FoldGRPO: fold − no fold | total_e2e_tokens | 150 | 43,168 [37,974, 48,048] | 45,832 | 89% | 0.000 |
| FoldGRPO: fold − no fold | turns | 150 | 18.8 [17.3, 20.3] | 19.9 | 99% | 0.000 |
| FoldGRPO: fold − no fold | tool_calls | 150 | 8.5 [7.4, 9.5] | 9.6 | 87% | 0.000 |
| FoldGRPO: fold − no fold | peak_ctx | 150 | -1,417 [-2,288, -615] | 290 | 57% | 0.121 |
| FoldGRPO: fold − no fold | cum_prompt_tokens | 150 | 324,718 [289,135, 360,140] | 360,851 | 89% | 0.000 |
| GRPO: fold − no fold | success | 150 | 0.135 [0.093, 0.178] | 0.000 | 35% | 0.000 |
| GRPO: fold − no fold | total_e2e_tokens | 150 | 19,416 [3,118, 29,492] | 22,652 | 85% | 0.000 |
| GRPO: fold − no fold | turns | 150 | 14.3 [12.9, 15.7] | 12.4 | 97% | 0.000 |
| GRPO: fold − no fold | tool_calls | 150 | 7.2 [6.2, 8.3] | 6.2 | 89% | 0.000 |
| GRPO: fold − no fold | peak_ctx | 150 | 768 [-44, 1,579] | 1,196 | 62% | 0.004 |
| GRPO: fold − no fold | cum_prompt_tokens | 150 | 278,378 [244,449, 312,000] | 250,535 | 92% | 0.000 |
| FoldGRPO/no fold − GRPO/no fold | success | 150 | -0.128 [-0.168, -0.090] | 0.000 | 2% | 0.000 |
| FoldGRPO/no fold − GRPO/no fold | total_e2e_tokens | 150 | -4,669 [-19,438, 3,268] | 1,479 | 66% | 0.000 |
| FoldGRPO/no fold − GRPO/no fold | turns | 150 | 1.5 [1.2, 1.9] | 1.5 | 76% | 0.000 |
| FoldGRPO/no fold − GRPO/no fold | tool_calls | 150 | 1.9 [1.5, 2.2] | 1.8 | 81% | 0.000 |
| FoldGRPO/no fold − GRPO/no fold | peak_ctx | 150 | 2,709 [2,055, 3,397] | 2,070 | 72% | 0.000 |
| FoldGRPO/no fold − GRPO/no fold | cum_prompt_tokens | 150 | 39,698 [30,495, 48,890] | 35,758 | 79% | 0.000 |
| base: fold − no fold | success | 150 | 0.107 [0.067, 0.145] | 0.000 | 26% | 0.000 |
| base: fold − no fold | total_e2e_tokens | 150 | -11,631 [-13,012, -10,206] | -12,720 | 10% | 0.000 |
| base: fold − no fold | turns | 150 | 1.5 [0.8, 2.3] | 0.5 | 59% | 0.021 |
| base: fold − no fold | tool_calls | 150 | -3.4 [-3.8, -3.1] | -3.5 | 7% | 0.000 |
| base: fold − no fold | peak_ctx | 150 | -12,347 [-13,247, -11,449] | -13,014 | 1% | 0.000 |
| base: fold − no fold | cum_prompt_tokens | 150 | -24,293 [-39,598, -7,234] | -44,094 | 28% | 0.000 |
| FoldGRPO/fold − GRPO/no fold | success | 150 | 0.135 [0.087, 0.185] | 0.000 | 35% | 0.000 |
| FoldGRPO/fold − GRPO/no fold | total_e2e_tokens | 150 | 38,499 [21,564, 49,430] | 47,647 | 93% | 0.000 |
| FoldGRPO/fold − GRPO/no fold | turns | 150 | 20.3 [18.9, 21.8] | 21.4 | 100% | 0.000 |
| FoldGRPO/fold − GRPO/no fold | tool_calls | 150 | 10.4 [9.4, 11.3] | 11.2 | 97% | 0.000 |
| FoldGRPO/fold − GRPO/no fold | peak_ctx | 150 | 1,292 [447, 2,074] | 1,474 | 73% | 0.000 |
| FoldGRPO/fold − GRPO/no fold | cum_prompt_tokens | 150 | 364,417 [329,779, 399,260] | 391,489 | 96% | 0.000 |

## 4. Figures

![](./2026-09-17_foldagent_e2e_token_usage_report_assets/fig1_success_vs_e2e_tokens.png)
*Fig. 1 — Success vs total end-to-end tokens (top: cell means ± 95% CI; bottom: fraction of rollouts solved within an end-to-end token budget).*

![](./2026-09-17_foldagent_e2e_token_usage_report_assets/fig2_peak_ctx_vs_e2e_tokens.png)
*Fig. 2 — Peak active context vs total end-to-end tokens per rollout (large markers = cell means; axes clipped at the 99.5th percentile — one GRPO/no-fold T=1 rollout received a single 4.2M-token observation). Points below the dotted diagonal discarded tokens.*

![](./2026-09-17_foldagent_e2e_token_usage_report_assets/fig3_post_first_fold_usage.png)
*Fig. 3 — Interaction after the first fold, for rollouts that folded.*

![](./2026-09-17_foldagent_e2e_token_usage_report_assets/fig4_paired_task_diffs.png)
*Fig. 4 — Paired per-task differences (y-axes clipped to the 1st–99th percentile of task differences).*

![](./2026-09-17_foldagent_e2e_token_usage_report_assets/fig5_token_composition.png)
*Fig. 5 — Where the end-to-end tokens go.*

## 5. Findings

**Headline.** On the same 150 tasks, FoldAgent's folding does **not** buy shorter active context for free. For both RL checkpoints the peak
active context is the same with or without folding (every policy saturates the 32k window somewhere in the trajectory; paired Δpeak ≈ 0),
the mean context per forward pass drops only 10–20 %, and in exchange the trajectory becomes 3–5× longer in turns and 1.7–2.6× longer in
end-to-end tokens, with 3.5–5.5× the cumulative model input. Most of that extra interaction happens **after the first fold**. FoldGRPO
training does not learn to compensate for this: it learns to fold more and to spend ~60 % more end-to-end tokens than the plain-GRPO policy
on the same tasks, for a success gain that is not significant (greedy +0.047 [−0.03, 0.12]; T=1 n=4: 0.00).

**1. Context efficiency (what folding actually changes).** Greedy, all rollouts: peak active context 31.1k (GRPO/fold) vs 30.7k (GRPO/no fold)
and 32.4k (FoldGRPO/fold) vs 33.9k (FoldGRPO/no fold) — paired Δpeak +0.4k [−1.3k, 1.9k] and −1.4k [−2.8k, −0.2k]. The window is hit either
in the main thread (no fold) or inside a branch (fold). Mean input length per forward pass: 18.3k / 17.3k with folding vs 19.3k / 21.6k without.
Only the untrained base model gets a real context reduction from folding (peak 20.2k vs 31.8k, −11.6k paired), because it returns from
branches after ~2 tool calls.

**2. Trajectory efficiency (interaction).** Folding multiplies interaction: turns 19.5 vs 5.4 (GRPO) and 31.4 vs 6.8 (FoldGRPO); tool calls
12.0 vs 5.0 and 18.6 vs 6.7; branches per rollout 2.4 (GRPO) and 4.4 (FoldGRPO). Of the fold cells' end-to-end tokens, 62 % (GRPO) and 73 %
(FoldGRPO) are generated or returned after the first fold (38.6k / 74.0k tokens, 14.4 / 25.0 turns; Fig. 3). Restricting to successful
rollouts the pattern is the same: a solved task costs 23.1k e2e tokens for GRPO/no fold, 38.6k for GRPO/fold and 58.7k for FoldGRPO/fold
(20.1k for base/fold).

**3. End-to-end inference efficiency.** Greedy means (medians): e2e tokens 36.8k (39.0k) GRPO/no fold → 62.6k (45.1k) GRPO/fold → 101.1k (80.6k)
FoldGRPO/fold; cumulative model input 110k → 380k → 595k tokens per rollout. Paired on the same tasks, FoldGRPO/fold − GRPO/fold = +38.5k
e2e tokens [26k, 51k], +11.9 turns, +6.6 tool calls, +216k cumulative input (all p < 0.001), for +0.047 success (p = 0.28). The
solved-within-budget curves (Fig. 1, bottom) show GRPO/fold dominating FoldGRPO/fold up to an end-to-end budget of ~100k tokens
(B = 50k: 0.253 vs 0.193; 75k: 0.307 vs 0.287); FoldGRPO/fold only pulls ahead when > 100k tokens per task are allowed (0.373 vs 0.327 at 400k).

**4. Why the no-fold cells are not simply "cheaper".** Without the branch tool both RL policies die at the window: 71 % (GRPO) and 91 %
(FoldGRPO) of rollouts end with the 32k window exhausted before an answer (success 0.153 and 0.067; FoldGRPO/no fold is *below the base model*,
0.087). Their low e2e token counts are the cost of a killed trajectory, not of an efficient one. Folding converts the hard window limit into a
soft budget on interaction: finish rate 89 % / 97 %, window exhaustion 11 % / 3 %. So the honest reading is: folding buys *survival past the
window*, and the trained policies pay for it with interaction, not with a smaller active context.

**5. What FoldGRPO training learned.** Relative to GRPO with the same tool: +2 branches per rollout, +60 % e2e tokens, 13 % of greedy rollouts
hit the 10-branch limit (GRPO 1 %), and complete dependence on the tool (no-fold success 0.067 vs GRPO's 0.153). It did learn to *stop*:
97 % finish and 3 % window exhaustion (GRPO 89 % / 11 %), and its main thread is shorter — but the stopping is achieved by delegating more
work to branches, i.e. by spending more end-to-end tokens, not by needing less post-fold interaction. Neither "reduces context occupancy
without increasing future interaction" nor "learns to compensate for post-compaction interaction" holds; "trades shorter (mean) context for
more search/reasoning" is the description the data supports.

**6. Sampling (T=1.0, n=4) agrees on tokens and disagrees only where an artefact intervenes.** The paired FoldGRPO/fold − GRPO/fold
differences are +19.1k e2e tokens [15k, 23k], +6.0 turns, +86k cumulative input at Δsuccess 0.000 [−0.035, 0.035]. However the 1 h
session timeout of the val worker killed 10 % (GRPO/fold) and 34 % (FoldGRPO/fold) of the 600 concurrent sampled rollouts (0 % in greedy and
in all no-fold cells: the shared search + judge pod is throughput-bound at 600-way concurrency, median session 51 min for FoldGRPO/fold vs
12 min in greedy). Timed-out rollouts score 0 and have truncated token counts, so the T=1 fold-cell numbers are lower bounds on both success
and cost; among rollouts that did not time out FoldGRPO/fold scores 0.479 (biased upward: the long ones were dropped). Both fold cells are
being re-run at T=1 with a 4 h session timeout (jobs c5220b294491facc, 7ea9d06bbfe4d595; tag `_e2e_branch_t4h`); this report is rebuilt
automatically from them when they finish. **Greedy is the clean comparison in this version.**


## 6. Caveats

- Both RL checkpoints are truncated runs (46 / 59 real steps) from the fix campaign and were trained *with* the branch tool; the no-fold cells
  are an inference-time ablation with a different system prompt (`search` vs `search_branch`), so they measure "the same policy without the
  ability to fold", not a branch-free-trained baseline.
- "Compaction" here is FoldAgent's branch/fold; the CompactionRL-style summary compaction (`compactiongrpo` arm) was still training and is not included.
- Judge = gpt-oss-120b (not the paper's GPT models); success = judge-graded correctness of the final answer (0/1).
- The bottom of the 32k window: a thread that fills its window keeps its partial context; `overlong` marks main threads that reached the window.
- Token counts are exact ids from the ledger; the branch's inherited context is re-rendered by the chat template (±1 token per inherited turn).
- Greedy decoding on vLLM is not bit-reproducible across nodes; t1n4 CIs reflect sampling noise over 4 samples × 150 tasks.

## 7. Reproduce

```
# 1. jobs (already submitted; specs in infra/jobs/fold_e2e_*_h100.json, ledger in infra/jobs/JOBS.tsv)
cd ~/xiaoxuan/FoldAgent/infra/jobs && merlin-cli --control-plane i18n-tt job-v2 runs create --from-file fold_e2e_grpo60_branch_h100.json   # ... x6
# 2. report (reads /mnt/hdfs/mlsys/xiaoxuan/fold_replication/results/valonly_*_e2e_*_sc/ledger)
python3 docs/reports/build_e2e_report.py
# tests: ~/xiaoxuan/envs/fold_train/bin/python -m unittest tests.test_e2e_ledger tests.test_e2e_metrics
```
Artifacts: `docs/reports/2026-09-17_foldagent_e2e_token_usage_report_assets/` (rollouts.csv = one row per rollout with every metric; cell_summary.csv; paired_diffs.csv; figures as PNG+PDF with JSON data).
