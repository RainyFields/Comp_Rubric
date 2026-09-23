# Rollout traces (sanity checks of the experiment setup)

Rendered by `scripts/render_rollout_traces.py` from the trainer's JSONL dumps on HDFS
(`/mnt/hdfs/mlsys/xiaoxuan/fold_replication/{val_dump_fix,rollout_dump_fix}/…`). Every file starts with a
table of rollout kinds in the dump, a table of the rendered rollouts, and then one section per rollout with a
**sanity checklist** followed by the full trace, turn by turn (think block, action, observation, compaction
prompt, policy summary, resume context, tail steps).

| file | source | what to look at |
|---|---|---|
| `2026-09-23_compactiongrpo_val_step100.md` | greedy validation, step 100 of the full `compactiongrpo` run (val = 0.353) | last segment of 6 rollouts: 1 single-window correct, 1 single-window wrong, 2 compacted correct, 1 compacted wrong, 1 budget-exhausted. Validation dumps hold the **last** context segment only, so compacted rollouts open with the resume template and the policy's own summary. |
| `2026-09-23_compactiongrpo_train_shakeout2_step2.md` | training rollouts, shakeout2 step 2 (`FOLD_ROLLOUT_DUMP=1`, 256 rollouts → 777 segments) | **full multi-segment chains** (segments matched by summary text): 4-segment budget-exhausted rollout, 2–4-segment finished rollouts, one single-window rollout, one broken chain. Shows q_sum → `<summary>` → resume → 2 verbatim tail steps, rollback of a step before the summary, and the observation clipped at the segment end. |
| `2026-09-23_foldgrpo_val_step40.md` | greedy validation, FoldGRPO step 40 (val = 0.353, clean) | main context of the fold agent for comparison (branch sub-contexts are folded away in the dump). |
| `2026-09-23_grpo_val_step50.md` | greedy validation, GRPO step 50 (val = 0.300, clean) | same agent scaffold trained with plain GRPO. |

Note: for the fold arms, step 50 (foldgrpo) and step 60 (grpo) in-training dumps are the infra-gap ones
(val read 0.0); the clean numbers come from val-only re-evals whose dumps live under
`results/valsc_<arm>_<step>_greedy_sc/dump/`. The steps rendered above are clean in-training vals.

## What the checklist verifies

| check | why it matters |
|---|---|
| every tool call is followed by an intact `<tool_response>…</tool_response>` | the Qwen3 template bug (Sep 8) dropped observations; the `wrap_tool_response` + `render_single_turn` fix must hold in every trace. An observation cut at the *end* of a segment is the dumped sample clipped to `response_length` (mask 0); the agent rolls that step back before summarising. |
| every assistant turn has a `<think>` block; count of empty thinks | thinking mode is on; the RL'd Fold/GRPO policies collapsed to ~80 % empty thinks, the compaction policy did not (0–1 %). |
| no leaked `<|im_start|>` / `<|im_end|>` | decoded dumps must not contain special tokens (would mean template double-rendering). |
| only the arm's tools are called; no text after a call | prompt format compliance. |
| rollout ends with `finish` or with the compaction budget exhausted | anything else = max_turn / timeout / cut. |
| written summary == summary resumed by the next segment | the resume context carries the policy's summary verbatim. |
| tail steps re-rendered verbatim from the previous segment | k = 2 (assistant, observation) pairs; a tail step that is *not* in the previous segment's trained text was rolled back before the summary (expected when `q_sum` + `summary_max_tokens` did not fit). |
| per-segment token budget | generated side ≤ `response_length` (32 768); compaction triggers when the remaining budget < `compaction_threshold` (8 192). Counts are re-tokenised approximations. |

## Regenerate / render other dumps

```bash
P=~/xiaoxuan/envs/fold_train/bin/python; D=/mnt/hdfs/mlsys/xiaoxuan/fold_replication
$P scripts/render_rollout_traces.py --dump $D/val_dump_fix/compactiongrpo/100.jsonl --out docs/traces/<name>.md \
    --select single_correct:1,compacted_correct:2,exhausted:1 --obs-chars 0      # full observations
$P scripts/render_rollout_traces.py --dump $D/rollout_dump_fix/compactiongrpo_shakeout2/2.jsonl --out <md> --select all   # every chain (~50 MB)
# fold arms: --tools search,open_page,finish,branch,return
```

Kinds: `single_*` (no compaction), `compacted_*` (≥1 compaction, finished), `exhausted` (budget gone after
`max_compactions`), `other` (ended without finish: max_turn / timeout / broken chain in a training dump).
