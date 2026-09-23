# docs/ index

| where | what |
|---|---|
| `STATUS.md` | **project status** — methods implemented, benchmarks, results per arm, live state, open items (updated 2026-09-23) |
| `baselines/README.md` | arm table, CompactionRL paper→code mapping, knobs, shakeout results, how to add an arm; `baselines/refs/` = paper PDF + Markdown |
| `traces/` | rendered full rollout traces with per-rollout sanity checklists (`scripts/render_rollout_traces.py`); see `traces/README.md` |
| `plans/2026-09-23_qwen3.5_migration_plan.md` | Qwen3.5 migration: verified facts, decisions D1–D5, work packages WP1–WP5 |
| `reports/` | house-standard reports (PDF + md + assets + rerunnable builders): `2026-08-12_*` replication, `2026-09-14_*` fix campaign, `2026-09-17_*` E2E token study; `paper/`, `paper_fix/` = LaTeX research-paper versions (`paper/` carries the user's uncommitted review edits — never touch) |

Superseded: `../report/` (August draft of the replication report; canonical copy is `reports/2026-08-12_*`).

Hand-offs live outside the repo in `~/xiaoxuan/handoffs/` (newest first: `ls -t`).
