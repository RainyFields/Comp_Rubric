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
