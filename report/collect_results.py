#!/usr/bin/env python3
"""Aggregate FoldAgent replication results into report/results.json (rerunnable).
Reads results/valonly_*/ dirs: scores from valonly.log, behavior stats from logs+dumps.
Contamination guard: runs with >50 judge/tool errors are marked invalid."""
import json, glob, os, re, statistics

R = '/home/tiger/xiaoxuan/FoldAgent/results'
out = {}
for d in sorted(glob.glob(f'{R}/valonly_*/')):
    tag = os.path.basename(d.rstrip('/')).replace('valonly_', '')
    log = os.path.join(d, 'valonly.log')
    if not os.path.exists(log):
        continue
    txt = open(log, errors='ignore').read()
    def last(pat):
        m = re.findall(pat, txt)
        return float(m[-1]) if m else None
    rec = {
        'score': last(r"'val/avg_score': ([0-9.]+)"),
        'overlong_rate': last(r"'val/overlong_rate': ([0-9.]+)"),
        'avg_num_turns': last(r"'val/avg_num_turns': ([0-9.]+)"),
        'n_errors': txt.count('Error after 3 attempts') + txt.count('CALL OPENAI] Error'),
    }
    rec['valid'] = rec['score'] is not None and rec['n_errors'] <= 50
    dumps = glob.glob(os.path.join(d, 'dump', '*.jsonl'))
    if dumps:
        br, ml, sc = [], [], []
        for line in open(dumps[0], errors='ignore'):
            try:
                j = json.loads(line)
            except Exception:
                continue
            o = j.get('output') or ''
            br.append(o.count('<function=branch>')); ml.append(len(o))
            if j.get('score') is not None: sc.append(j['score'])
        if br:
            rec['avg_branches'] = round(statistics.mean(br), 2)
            rec['avg_main_chars'] = round(statistics.mean(ml))
            rec['n_traj'] = len(br)
    out[tag] = rec

os.makedirs('/home/tiger/xiaoxuan/FoldAgent/report', exist_ok=True)
with open('/home/tiger/xiaoxuan/FoldAgent/report/results.json', 'w') as f:
    json.dump(out, f, indent=1, sort_keys=True)
for k, v in sorted(out.items()):
    print(f"{k:30s} score={v['score']} valid={v['valid']} overlong={v.get('overlong_rate')} branches={v.get('avg_branches')}")
