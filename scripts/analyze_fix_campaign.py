#!/usr/bin/env python3
"""Analyze the post-fix RL retrain campaign (FoldGRPO vs GRPO, Qwen3-8B, BrowseComp-Plus).

Reads the HDFS mirrors written by the batch jobs (see infra/jobs/JOBS.tsv):
  <root>/val_dump_fix/<arm>/<step>.jsonl   greedy validation dumps (150 items; main-context text)
  <root>/train_logs_fix/<arm>/train_<arm>.log   trainer log (5-min mirror), one metrics line per step

Per validation step it reports accuracy, empty-<think> rate, observation integrity
(every non-finish tool call followed by a <tool_response> block), branch/search calls
per trajectory, main-context turns, and finish rate.  Per training step it reports the
reward/turn/length curve and flags "gap" steps (infra reclaimed -> zero reward).

Usage:
  python scripts/analyze_fix_campaign.py [--root /mnt/hdfs/mlsys/xiaoxuan/fold_replication]
                                         [--out results/fix_campaign] [--arms foldgrpo grpo]
Writes <out>/val_<arm>.csv, <out>/train_<arm>.csv, <out>/summary.md and prints the tables.
"""
import argparse, csv, glob, json, math, os, re, sys
from collections import Counter

TURN_SPLIT = re.compile(r"\nassistant\n")
THINK = re.compile(r"<think>(.*?)</think>", re.S)
CALL = re.compile(r"<function=(\w+)>")
METRIC = re.compile(r"([\w/\-@.]+):(-?(?:\d+\.?\d*(?:e[+-]?\d+)?|nan|inf))(?= - |$)")


def parse_traj(text):
    """Return per-trajectory stats from the detokenized main-context text."""
    turns = TURN_SPLIT.split(text)
    if turns and turns[0].startswith("assistant\n"):
        turns[0] = turns[0][len("assistant\n"):]
    st = Counter(turns=len(turns))
    for t in turns:
        th = THINK.search(t)
        if th is None:
            st["no_think"] += 1
        else:
            st["think"] += 1
            if not th.group(1).strip():
                st["empty_think"] += 1
        calls = CALL.findall(t)
        for c in calls:
            st["call_" + c] += 1
        non_finish = [c for c in calls if c != "finish"]
        if non_finish:
            st["obs_expected"] += 1
            if "<tool_response>" in t and "</tool_response>" in t:
                st["obs_intact"] += 1
            elif "<tool_response>" in t:
                st["obs_partial"] += 1
        if "finish" in calls:
            st["finished"] += 1
    st["chars"] = len(text)
    return st


def analyze_dump(path):
    rows = [json.loads(l) for l in open(path)]
    agg = Counter()
    n = len(rows)
    traj_empty_any = traj_all_empty = 0
    for r in rows:
        s = parse_traj(r["output"])
        agg.update(s)
        agg["score"] += float(r["score"])
        if s["empty_think"] > 0:
            traj_empty_any += 1
        if s["think"] and s["empty_think"] == s["think"]:
            traj_all_empty += 1
    r0 = rows[0]
    acc = agg["score"] / n
    return {
        "n": n,
        "acc": round(acc, 4),
        "ci95": round(1.96 * math.sqrt(acc * (1 - acc) / n), 3),
        "empty_think_turn_pct": round(100 * agg["empty_think"] / max(1, agg["think"]), 1),
        "traj_any_empty_pct": round(100 * traj_empty_any / n, 1),
        "traj_all_empty_pct": round(100 * traj_all_empty / n, 1),
        "no_think_turns": agg["no_think"],
        "obs_intact_pct": round(100 * agg["obs_intact"] / max(1, agg["obs_expected"]), 1),
        "obs_partial": agg["obs_partial"],
        "obs_missing": agg["obs_expected"] - agg["obs_intact"] - agg["obs_partial"],
        "main_turns_per_traj": round(agg["turns"] / n, 2),
        "branch_per_traj": round(agg["call_branch"] / n, 2),
        "search_per_traj": round(agg["call_search"] / n, 2),
        "open_per_traj": round(agg["call_open_page"] / n, 2),
        "finish_pct": round(100 * agg["finished"] / n, 1),
        "chars_per_traj": int(agg["chars"] / n),
        "all_turns_per_traj(log)": round(float(r0.get("avg_num_turns", float("nan"))), 2),
        "overlong_pct(log)": round(100 * float(r0.get("overlong_rate", float("nan"))), 1),
    }


def parse_train_log(path):
    steps = {}
    for line in open(path, errors="replace"):
        if "training/global_step:" not in line:
            continue
        d = {k: float(v) for k, v in METRIC.findall(line)}
        if "training/global_step" not in d:
            continue
        s = int(d["training/global_step"])
        steps[s] = d  # keep last occurrence (resume re-logs)
    out = []
    for s in sorted(steps):
        d = steps[s]
        val = d.get("val-core/unknown/reward/mean@1")
        out.append({
            "step": s,
            "reward_mean": round(d.get("critic/score/mean", float("nan")), 4),
            "reward_avg_score": round(d.get("reward/avg_score", float("nan")), 4),
            "overlong_rate": round(d.get("reward/overlong_rate", float("nan")), 3),
            "turns_mean": round(d.get("num_turns/mean", float("nan")), 2),
            "resp_len_mean": int(d.get("response_length/mean", 0)),
            "entropy": round(d.get("actor/entropy", float("nan")), 4),
            "grad_norm": round(d.get("actor/grad_norm", float("nan")), 4),
            "step_s": int(d.get("timing_s/step", 0)),
            # gap = infra reclaimed for the whole rollout (zero reward everywhere);
            # partial = short rollouts + near-zero reward (infra died mid-rollout)
            "gap": int(d.get("critic/score/max", 1) == 0.0),
            "partial": int(d.get("critic/score/max", 1) != 0.0 and s > 3
                           and d.get("response_length/mean", 1e9) < 8000),
            "val_acc": "" if val is None else round(val, 4),
            "val_overlong": "" if val is None else round(d.get("val-aux/unknown/overlong_rate/mean@1", float("nan")), 3),
            "val_turns": "" if val is None else round(d.get("val-aux/unknown/avg_num_turns/mean@1", float("nan")), 2),
        })
    return out


def read_val_metrics(path):
    d = {}
    for line in open(path):
        m = re.match(r"'([^']+)':\s*([-\d.e]+)", line.strip())
        if m:
            d[m.group(1)] = float(m.group(2))
    return d


def t1n4_rows(root, arm):
    """T=1.0, n=4 val-only results (600 trajectories): mean@4, best@4, worst@4 + dump behaviour stats."""
    out = {}
    cands = [(0, f"{root}/results/valonly_norl_base_t1n4_fix_sc")]
    cands += [(int(p.split(f"valonly_{arm}_")[1].split("_")[0]), p)
              for p in glob.glob(f"{root}/results/valonly_{arm}_*_t1n4_fix_sc")]
    for step, d in cands:
        mp, dp = f"{d}/val_metrics.txt", f"{d}/dump/0.jsonl"
        if not os.path.exists(mp):
            continue
        m = read_val_metrics(mp)
        n = m.get("val/num_unique_gen_uids", 600)
        acc = m.get("val/avg_score", float("nan"))
        row = {"step": step, "mean@4": round(acc, 4), "ci95": round(1.96 * math.sqrt(acc * (1 - acc) / n), 3),
               "best@4": round(m.get("val-aux/unknown/reward_score/best@4/mean", float("nan")), 4),
               "worst@4": round(m.get("val-aux/unknown/reward_score/worst@4/mean", float("nan")), 4),
               "overlong_pct": round(100 * m.get("val/overlong_rate", float("nan")), 1),
               "turns(log)": round(m.get("val/avg_num_turns", float("nan")), 2)}
        if os.path.exists(dp):
            a = analyze_dump(dp)
            row.update({k: a[k] for k in ("empty_think_turn_pct", "obs_intact_pct", "branch_per_traj", "search_per_traj", "finish_pct")})
        out[step] = row
    return [out[k] for k in sorted(out)]


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/hdfs/mlsys/xiaoxuan/fold_replication")
    ap.add_argument("--out", default="results/fix_campaign")
    ap.add_argument("--arms", nargs="+", default=["foldgrpo", "grpo"])
    ap.add_argument("--dump-dir", default="val_dump_fix")
    ap.add_argument("--log-dir", default="train_logs_fix")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    md = []
    val_cols = ["step", "src", "acc", "ci95", "empty_think_turn_pct", "traj_any_empty_pct", "traj_all_empty_pct",
                "obs_intact_pct", "obs_missing", "main_turns_per_traj", "branch_per_traj", "search_per_traj",
                "open_per_traj", "finish_pct", "chars_per_traj", "all_turns_per_traj(log)", "overlong_pct(log)"]
    train_cols = ["step", "reward_mean", "overlong_rate", "turns_mean", "resp_len_mean", "entropy", "step_s", "gap", "partial", "val_acc"]
    for arm in a.arms:
        vrows = {}
        base = f"{a.root}/results/valonly_norl_base_greedy_fix_sc/dump/0.jsonl"
        if os.path.exists(base):
            vrows[0] = {"step": 0, "src": "no-RL base (val-only job)", **analyze_dump(base)}
        for p in glob.glob(f"{a.root}/{a.dump_dir}/{arm}/*.jsonl"):
            step = int(os.path.basename(p).split(".")[0])
            vrows[step] = {"step": step, "src": "in-training", **analyze_dump(p)}
        # val-only re-evals from the uploaded checkpoint override validations that ran inside an infra gap
        for p in glob.glob(f"{a.root}/results/valonly_{arm}_*_greedy_fix_sc/dump/0.jsonl"):
            step = int(p.split(f"valonly_{arm}_")[1].split("_")[0])
            old = vrows.get(step, {}).get("acc")
            vrows[step] = {"step": step, "src": f"re-eval (in-gap val read {old})", **analyze_dump(p)}
        vrows = [vrows[k] for k in sorted(vrows)]
        write_csv(f"{a.out}/val_{arm}.csv", vrows)
        md.append(f"## {arm} — greedy validation dumps (150 items)\n\n" + md_table(vrows, val_cols))
        trows4 = t1n4_rows(a.root, arm)
        if trows4:
            write_csv(f"{a.out}/t1n4_{arm}.csv", trows4)
            md.append(f"## {arm} — T=1.0 n=4 val-only evals (600 trajectories)\n\n" + md_table(trows4, list(trows4[0].keys())))
        logp = f"{a.root}/{a.log_dir}/{arm}/train_{arm}.log"
        if os.path.exists(logp):
            trows = parse_train_log(logp)
            write_csv(f"{a.out}/train_{arm}.csv", trows)
            gaps = [r["step"] for r in trows if r["gap"]]
            partial = [r["step"] for r in trows if r["partial"]]
            md.append(f"## {arm} — training log ({len(trows)} steps; gap steps = {gaps or 'none'}; partial-gap steps = {partial or 'none'})\n\n" + md_table(trows, train_cols))
    text = "\n\n".join(md)
    open(f"{a.out}/summary.md", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
