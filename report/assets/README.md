# FoldAgent replication — report assets
- fig1_scores.{pdf,png}: pass@1 by arm at both operating points; data = ../results.json (greedy N=150, T1.0 n=4 = 600 samples; 95% binomial CIs).
- fig2_behavior.{pdf,png}: behavioral signatures (overlong rate, branches/traj, main-thread chars) from greedy trajectory dumps; data = ../results.json.
- Rebuild: python3 ../collect_results.py && python3 ../build_figures.py && python ../build_pdf.py
- Raw per-run data: ../../results/valonly_<tag>/ (valonly.log, val_metrics.txt, dump/*.jsonl, judge_calls.jsonl).
