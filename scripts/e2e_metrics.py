"""Per-rollout end-to-end token metrics from the fold-agent ledger (agents/e2e_ledger.py).

Definitions (all token counts are exact ids the policy generated or was fed; nothing folded
away is dropped):
  gen_tokens        tokens sampled by the policy over all threads (main + branches)
  obs_tokens        tool observations returned to the policy (search / open_page results,
                    error strings, branch-limit notices) over all threads
  comp_in_tokens    tokens injected by the folding mechanism itself: branch task prompts,
                    branch-overflow summary prompts, and the folded return messages the main
                    thread receives (branch_return_obs)
  comp_gen_tokens   policy tokens spent on the mechanism: the branch-call turns and the
                    return turns (whole turn incl. its reasoning), summary responses
  frame_tokens      chat-template framing around generated turns (generation prompt + eos)
  prompt_tokens     the task prompt (system + user), counted once
  total_e2e_tokens  prompt + gen + frame + obs + comp_in  (everything before any discard)
  forward_passes    number of generation calls; ctx_before of each = model input length
  cum_prompt_tokens sum of ctx_before over all forward passes (cumulative model input)
  peak_ctx          max over passes of (ctx_before + tokens generated in that pass)
  mean_ctx          mean ctx_before over passes
  main_ctx_final    final main-thread context; overlong = main growth >= response_length
  post_*            same quantities restricted to events after the first fold (first
                    branch return); NaN when the rollout never folded
"""
import glob, json, math, os

TOOL_KINDS = {'search', 'open_page'}
OBS_KINDS = {'obs', 'branch_limit_obs'}
COMP_IN_KINDS = {'branch_prompt', 'summary_prompt', 'branch_return_obs'}
COMP_GEN_KINDS = {'branch', 'return'}


def load_ledgers(d):
    recs = []
    for f in sorted(glob.glob(os.path.join(d, 'ledger_*.jsonl'))):
        for line in open(f):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _events(rec):
    """Global event timeline: main turns in order, each branch's own turns inserted where it ran
    (after main turn inherited_turns-1, before the return observation at main turn inherited_turns)."""
    agents = rec['agents']
    main = agents[0]
    branches = sorted([a for a in agents[1:]], key=lambda a: (a['inherited_turns'], a['name']))
    ev, bi = [], 0
    for t in main['turns']:
        while bi < len(branches) and branches[bi]['inherited_turns'] <= t['i']:
            b = branches[bi]
            for bt in b['turns']:
                if not bt['inherited']:
                    ev.append(dict(bt, agent=b['name'], branch_idx=bi))
            ev.append({'role': 'event', 'kind': 'fold', 'n': 0, 'gen': 0, 'agent': b['name'], 'branch_idx': bi})
            bi += 1
        ev.append(dict(t, agent='main', branch_idx=None))
    # branches that never got a return observation (rollout ended inside a branch)
    while bi < len(branches):
        b = branches[bi]
        for bt in b['turns']:
            if not bt['inherited']:
                ev.append(dict(bt, agent=b['name'], branch_idx=bi))
        bi += 1
    return ev


def _agg(events):
    m = dict(gen=0, frame=0, obs=0, comp_in=0, comp_gen=0, turns=0, tool_calls=0, passes=0,
             cum_prompt=0, peak_ctx=0, ctx_sum=0, branch_calls=0, summary_gen=0, empty_think=0)
    prev_summary = False
    for e in events:
        if e['role'] == 'assistant':
            m['gen'] += e['gen']; m['frame'] += e['n'] - e['gen']; m['turns'] += 1; m['passes'] += 1
            if e.get('ctx_before') is not None:
                m['cum_prompt'] += e['ctx_before']; m['ctx_sum'] += e['ctx_before']
                m['peak_ctx'] = max(m['peak_ctx'], e['ctx_before'] + e['gen'])
            k = e.get('kind')
            if k in TOOL_KINDS: m['tool_calls'] += 1
            if k in COMP_GEN_KINDS or prev_summary: m['comp_gen'] += e['gen']
            if prev_summary: m['summary_gen'] += e['gen']
            if k == 'branch': m['branch_calls'] += 1
            m['empty_think'] += e.get('think_empty', 0)
            prev_summary = False
        elif e['role'] == 'user':
            k = e.get('kind')
            if k in OBS_KINDS: m['obs'] += e['n']
            elif k in COMP_IN_KINDS: m['comp_in'] += e['n']
            prev_summary = (k == 'summary_prompt')
    return m


def rollout_metrics(rec):
    main = rec['agents'][0]
    prompt_tokens = sum(t['n'] for t in main['turns'] if t['inherited'])
    ev = _events(rec)
    all_m = _agg(ev)
    fold_idx = next((i for i, e in enumerate(ev) if e.get('kind') == 'fold'), None)
    post = _agg(ev[fold_idx + 1:]) if fold_idx is not None else None
    pre = _agg(ev[:fold_idx + 1]) if fold_idx is not None else None
    main_growth = main['final_context'] - prompt_tokens - (main['final_context'] - sum(t['n'] for t in main['turns']))
    out = {
        'instance_id': rec['instance_id'], 'uid': rec['uid'], 'gen_uid': rec['gen_uid'], 'workflow': rec['workflow'],
        'score': rec['score'], 'success': int(rec['score'] > 0), 'is_finish': int(rec['is_finish']),
        'stop_reason': rec['stop_reason'], 'n_branches': rec['n_branches'], 'branch_limit_hit': rec.get('branch_limit_hit', 0),
        'prompt_tokens': prompt_tokens,
        'gen_tokens': all_m['gen'], 'frame_tokens': all_m['frame'], 'obs_tokens': all_m['obs'],
        'comp_in_tokens': all_m['comp_in'], 'comp_gen_tokens': all_m['comp_gen'], 'summary_gen_tokens': all_m['summary_gen'],
        'total_e2e_tokens': prompt_tokens + all_m['gen'] + all_m['frame'] + all_m['obs'] + all_m['comp_in'],
        'turns': all_m['turns'], 'tool_calls': all_m['tool_calls'], 'branch_calls': all_m['branch_calls'],
        'forward_passes': all_m['passes'], 'cum_prompt_tokens': all_m['cum_prompt'],
        'peak_ctx': all_m['peak_ctx'], 'mean_ctx': all_m['ctx_sum'] / all_m['passes'] if all_m['passes'] else float('nan'),
        'main_ctx_final': main['final_context'], 'main_growth': main_growth,
        'overlong': int(main_growth >= rec['response_length']),
        'empty_think_turns': all_m['empty_think'],
        'has_fold': int(fold_idx is not None),
        'search_calls': rec.get('env_stats', {}).get('search', None), 'open_page_calls': rec.get('env_stats', {}).get('open_page', None),
        'session_time': rec.get('env_stats', {}).get('session_time', None),
    }
    for tag, m in (('post', post), ('pre', pre)):
        for k in ('gen', 'obs', 'comp_in', 'comp_gen', 'turns', 'tool_calls', 'passes', 'cum_prompt', 'branch_calls'):
            out[f'{tag}_{k}'] = m[k] if m else float('nan')
        out[f'{tag}_total'] = (m['gen'] + m['frame'] + m['obs'] + m['comp_in']) if m else float('nan')
    return out


def metrics_table(records):
    return [rollout_metrics(r) for r in records]
