# docs/ index

| where | what |
|---|---|
| `PROTOCOL.md` | **training–inference interaction protocol**: token stream, action grammar, `plugin.protocol` legacy/v2 compatibility setting, training masks, limitations |
| `PROTOCOL_VERIFICATION.md` | PASS/FAIL/NOT-YET-VERIFIED table for every protocol check, with reproduction commands and trace references |
| `handoff/QWEN35_HANDOFF.md`, `handoff/NEXT_STEPS.md` | handoff for the Qwen3.5 devbox: objective, repo/commit, environment, commands, gates |
| `TODO.md` | **the to-do list** — A: replicate CompactionRL (PPO arm), B: compactiongrpo wrap-up, C: Qwen3.5 migration, D: moving to a new dev box, E: housekeeping |
| `STATUS.md` | **project status** — methods implemented, benchmarks, results per arm, live state, open items (updated 2026-09-23) |
| `baselines/README.md` | arm table, CompactionRL paper→code mapping, knobs, shakeout results, how to add an arm; `baselines/refs/` = paper PDF + Markdown |
| `traces/` | rendered full rollout traces with per-rollout sanity checklists (`scripts/render_rollout_traces.py`); see `traces/README.md` |
| `plans/2026-09-23_qwen3.5_migration_plan.md` | Qwen3.5 migration: verified facts, decisions D1–D5, work packages WP1–WP5 |
| `reports/` | house-standard reports (PDF + md + assets + rerunnable builders): `2026-08-12_*` replication, `2026-09-14_*` fix campaign, `2026-09-17_*` E2E token study; `paper/`, `paper_fix/` = LaTeX research-paper versions (`paper/` carries the user's uncommitted review edits — never touch) |

The August draft `report/` was removed on 2026-09-23 (canonical copy: `reports/2026-08-12_*`; still in git history).

Hand-offs live outside the repo in `~/xiaoxuan/handoffs/` (newest first: `ls -t`).
