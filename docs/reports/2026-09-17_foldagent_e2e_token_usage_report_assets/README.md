# 2026-09-17_foldagent_e2e_token_usage_report — assets
Regenerate with `python3 docs/reports/build_e2e_report.py`. Inputs: HDFS ledgers listed in the report §1.
- rollouts.csv — one row per rollout (cell, mode, instance_id, every metric of scripts/e2e_metrics.py)
- cell_summary.csv — per cell × mode × (all|success): mean, 95% bootstrap CI, median of each metric; stop-reason counts
- paired_diffs.csv — task-level paired differences with bootstrap CI and sign-test p
- fig1..fig5 .png/.pdf with the plotted numbers in the matching .json
