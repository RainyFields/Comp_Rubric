"""Per-rollout end-to-end token ledger for the fold agent (eval instrumentation).

Enabled only when ``FOLD_E2E_LEDGER_DIR`` is set: after a rollout, every agent thread
(main + each branch) is walked turn by turn and the *exact* token ids the policy saw are
counted — nothing that was folded away is dropped. One JSON line per rollout is appended
to ``<dir>/ledger_<pid>.jsonl`` (per-process files: no cross-process write interleaving).

Per turn: role, token count, generated-token count (assistant turns; includes the
generation prompt/eos framing as non-generated), the context length the model was fed
for that generation (``ctx_before``), and a coarse ``kind`` (search / open_page / branch
call / return / finish / observation / branch prompt / summary prompt / branch return
observation). Branch threads inherit the main context up to the branch call: those
``inherited`` turns are counted as prompt tokens of the branch's forward passes, not as
newly generated or newly returned tokens.
"""
import json
import os
import re
import time

_FN = re.compile(r'<function=([^>]+)>')


def _kind_assistant(text: str) -> str:
    fns = _FN.findall(text or '')
    if not fns:
        return 'no_call'
    return fns[-1]  # search / open_page / branch / return / finish / ...


def _kind_user(text: str, i: int, inherited: int, is_branch: bool) -> str:
    t = text or ''
    if is_branch and i == inherited:
        return 'branch_prompt'
    if t.startswith('The context limit has been exceeded'):
        return 'summary_prompt'
    body = t
    if body.startswith('<tool_response>'):
        body = body[len('<tool_response>'):].lstrip()
    if body.startswith('Branch has finished its task'):
        return 'branch_return_obs'
    if body.startswith("You've already reached the limit of"):
        return 'branch_limit_obs'
    return 'obs'


def build_ledger(agents: dict, prompt_turn: int) -> list:
    """agents: {'main': Agent, '#0-desc': Agent, ...} in creation order."""
    out = []
    for name, ag in agents.items():
        is_branch = name != 'main'
        inherited = ag.init_len if is_branch else prompt_turn
        gp = len(ag.get_generation_prompt())
        turns, ctx = [], 0
        for i, (ids, mask, turn) in enumerate(zip(ag.chat_ids, ag.token_mask, ag.chat)):
            n, g = len(ids), int(sum(mask))
            role = turn.get('role')
            rec = {'i': i, 'role': role, 'n': n, 'gen': g, 'inherited': i < inherited}
            if role == 'assistant':
                rec['ctx_before'] = ctx + gp if i >= inherited else None
                rec['kind'] = _kind_assistant(turn.get('content'))
                rec['think_empty'] = int(bool(re.search(r'<think>\s*</think>', turn.get('content') or '')))
            else:
                rec['kind'] = _kind_user(turn.get('content'), i, inherited, is_branch) if i >= inherited else 'prompt'
            turns.append(rec)
            ctx += n
        out.append({
            'name': name,
            'is_branch': is_branch,
            'inherited_turns': inherited,
            'inherited_tokens': int(sum(len(x) for x in ag.chat_ids[:inherited])),
            'final_context': int(ctx + gp),
            'turns': turns,
        })
    return out


def write_ledger(record: dict) -> None:
    d = os.environ.get('FOLD_E2E_LEDGER_DIR')
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        record = dict(record, wall_time=time.time())
        with open(os.path.join(d, f'ledger_{os.getpid()}.jsonl'), 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:  # never let instrumentation break a rollout
        print(f'[E2E-LEDGER] write failed: {e}')
