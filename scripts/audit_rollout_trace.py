#!/usr/bin/env python3
"""Auditable rollout trace: does the policy receive its previous <think> blocks, and do tool calls / boundaries alter that history?

Inputs
------
--dump   trainer validation dump (one record per rollout; ``input`` = decoded prompt ids, ``output`` = decoded response ids,
         special tokens stripped). For the fold arms this is the MAIN context only (branch sub-contexts are not dumped).
--capture (optional) directory of ``capture_*.jsonl`` written by ``agents.utils._capture_step`` (env FOLD_PROMPT_CAPTURE_DIR):
         the EXACT token ids sent to the policy at every step (decoded WITH special tokens), main + branches.
--tokenizer  the checkpoint's tokenizer dir (defaults to the step-50 GRPO checkpoint) — used to re-render turns with the real
         chat template, count tokens, and reconstruct the serialised prompt.

Evidence chain used for the dump (see agents/utils.py):
  Agent.context() = sum(chat_ids) + generation_prompt;  Agent.step() sends exactly context() to the policy (CallLLM);
  assistant turns are stored as the RAW sampled ids (+eos), user turns as template-rendered ids (render_single_turn);
  AgentContext.get_data(): prompt_ids = sum(chat_ids[:prompt_turn]), response_ids = sum(chat_ids[prompt_turn:]) → the dump.
So ``input + output`` is the decoded (special-token-stripped) concatenation of every prompt the policy saw in the main
context; the prompt at turn t is its prefix up to turn t plus the generation prompt. The audit verifies the user turns by
re-rendering them with the real template and marks reconstructions explicitly; captured prompts (when given) are the
ground truth and are compared against the reconstruction.

Outputs: <out>.md (aggregate diagnostics + detailed traces) and <out>.jsonl (one record per turn; observations intact).
"""
import argparse
import glob
import json
import os
import re
import sys
import types
from collections import Counter, OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agents.parsing import FN_BLOCK, PARAM, extract_fn_call, find_think  # noqa: E402
from agents.prompts import COMPACTION_RESUME_TEMPLATE, COMPACTION_SUMMARY_PROMPT  # noqa: E402
from envs.local_search import extract_fn_call as env_extract_fn_call  # noqa: E402  (the parser the environment really runs)

TURN_MAX_NEW_TOKENS = 2048   # plugin.turn_max_new_tokens in every script — see finding: not applied by CallLLM

DEFAULT_TOK = "/mnt/hdfs/mlsys/xiaoxuan/fold_replication/ckpt_fix/grpo/global_step_50/actor/huggingface"
QSUM_HEAD = COMPACTION_SUMMARY_PROMPT.split("\n")[0]
RESUME_HEAD = COMPACTION_RESUME_TEMPLATE.split("<summary>")[0].strip()[:40]
BRANCH_SUMMARY_HEAD = "The context limit has been exceeded"
TURN_START = re.compile(
    r"(?:^|(?<=\n))(assistant|user)\n"
    r"|(user)\n(?=<tool_response>|" + re.escape(QSUM_HEAD[:30]) + r"|" + re.escape(RESUME_HEAD[:30]) + r")"
    r"|(assistant)\n(?=<think>|<function=)", re.M)
FN_NAME = re.compile(r"<function=([^>]+)>")
QUESTION = re.compile(r"Question: (.*?)\n\nYour response should contain", re.S)
OBS_KIND = OrderedDict([
    ("branch_return", "Branch has finished its task"),
    ("branch_limit", "You've already reached the limit"),
    ("no_call", "No function call was detected"),
    ("search", "[Search Results for"),
    ("open_page", "[Opened Page Content]"),
])
EXPECTED_OBS = {"search": ("search",), "open_page": ("open_page",), "branch": ("branch_return", "branch_limit"),
                "none": ("no_call",), "finish": (), "unsupported": ("other",)}


# ---------------------------------------------------------------------------------------------- tokenizer helpers
class Renderer:
    """Re-renders turns with the real tokenizer + chat template through the repo's own Agent machinery."""

    def __init__(self, path):
        from transformers import AutoTokenizer
        from agents.utils import Agent
        from agents.model_profile import detect_profile
        self.tok = AutoTokenizer.from_pretrained(path)
        self.path = path
        self.profile = detect_profile(self.tok)
        cfg = types.SimpleNamespace(prompt_length=8192, response_length=32768, plugin=types.SimpleNamespace(retry_cjk=0))
        self._agent = Agent(None, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], self.tok, cfg, prompt_turn=2)
        self.gen_prompt_ids = self._agent.get_generation_prompt()
        self.gen_prompt_text = self.tok.decode(self.gen_prompt_ids, skip_special_tokens=False)
        self.eos_text = self.tok.decode([self.tok.eos_token_id], skip_special_tokens=False)

    def n(self, text):
        return len(self.tok.encode(text, add_special_tokens=False))

    def render_turn(self, role, content):
        ids = self._agent.render_single_turn({"role": role, "content": content})
        return ids, self.tok.decode(ids, skip_special_tokens=False), self.tok.decode(ids, skip_special_tokens=True)

    def render_prompt(self, system, user):
        ids = self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                           add_generation_prompt=False, tokenize=True)
        return ids, self.tok.decode(ids, skip_special_tokens=False), self.tok.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------------------------- parsing
def split_turns(text):
    bounds = [(m.start(), m.end(), next(g for g in m.groups() if g)) for m in TURN_START.finditer(text)]
    turns, notes = [], []
    lead = text[: bounds[0][0]] if bounds else text
    if lead.strip():
        turns.append(["assistant", lead, "lead"])
        notes.append("text before first role marker")
    for k, (b, e, role) in enumerate(bounds):
        content = text[e: bounds[k + 1][0] if k + 1 < len(bounds) else len(text)]
        glued = not (b == 0 or text[b - 1] == "\n")      # marker glued to the previous turn: previous turn ended with <|im_end|>, no newline
        if role == "user":
            c = content.lstrip()
            if not (c.startswith("<tool_response>") or c.startswith(QSUM_HEAD) or c.startswith(RESUME_HEAD)):
                if turns:
                    turns[-1][1] += text[b:e] + content
                    notes.append("suspicious 'user' line merged into previous turn")
                    continue
        turns.append([role, content, "glued" if glued else "newline"])
    return turns, notes


def obs_kind(body):
    for k, head in OBS_KIND.items():
        if body.startswith(head):
            return k
    return "other"


def parse_assistant(text):
    reasoning, span = find_think(text)
    calls = []
    for m in FN_BLOCK.finditer(text):
        inside = span is not None and span[0] <= m.start() < span[1]
        calls.append({"name": m.group(1), "arguments": dict(PARAM.findall(m.group(2))), "inside_think": inside,
                      "pos": m.start()})
    names_all = FN_NAME.findall(text)                 # includes unclosed ones
    agent_parse = extract_fn_call(text)               # agents.parsing (fold_agent): lenient, LAST <function=...>...</function> anywhere
    env_calls = env_extract_fn_call(text) or []       # envs.local_search: line-anchored, LAST adjacent group, ALL calls of it execute
    if isinstance(env_calls, dict):
        env_calls = [env_calls]
    # what actually ran in the rollout (fold_agent.process_item): 'branch' is intercepted by the agent BEFORE the env sees the turn
    if agent_parse and agent_parse["function"] == "branch":
        executed_names, executed_by = ["branch"], "agent(branch)"
    elif env_calls:
        executed_names, executed_by = [c["function"] for c in env_calls], "env"
    else:
        executed_names, executed_by = [], "env(no call detected)"
    executed = {"function": executed_names[0], "arguments": env_calls[0]["arguments"] if env_calls else agent_parse["arguments"]} if executed_names else None
    exec_inside_think = False
    if executed and span is not None:
        pos = text.rfind(f"<function={executed['function']}>")
        exec_inside_think = span[0] <= pos < span[1]
    discrepancies = []
    if agent_parse and executed_names and agent_parse["function"] != executed_names[-1] and executed_by == "env":
        discrepancies.append(f"lenient_parse={agent_parse['function']} vs env_executed={executed_names}")
    if agent_parse and not executed_names:
        discrepancies.append(f"lenient_parse={agent_parse['function']} but env detected NO call (unclosed / not line-anchored)")
    if len(executed_names) > 1:
        discrepancies.append(f"env_executed_{len(executed_names)}_calls_in_one_turn={executed_names}")
    if len(calls) > len(executed_names) and executed_by == "env" and executed_names:
        discrepancies.append(f"{len(calls)}_closed_calls_generated_but_{len(executed_names)}_executed")
    d = {
        "agent_parse": agent_parse, "env_calls": env_calls, "executed_names": executed_names, "executed_by": executed_by,
        "discrepancies": discrepancies,
        "has_think": reasoning is not None,
        "think_empty": reasoning is not None and reasoning == "",
        "think_text": reasoning,
        "think_span": span,
        "unclosed_think": ("<think>" in text and "</think>" not in text),
        "stray_close_think": ("</think>" in text and "<think>" not in text),
        "calls": calls,
        "n_function_tags": len(names_all),
        "unclosed_call": len(names_all) > len(calls),
        "executed_call": executed,
        "executed_inside_think": exec_inside_think,
        "ends_with_newline": text.endswith("\n"),
        "suffix_after_call": (text.rsplit("</function>", 1)[-1].strip() if calls else ""),
        "chars": len(text),
    }
    return d


def question_of(prompt_text):
    m = QUESTION.search(prompt_text)
    return m.group(1).strip() if m else prompt_text[-300:]


def split_prompt(input_text):
    """dump input = 'system\\n<sys>\\nuser\\n<user>\\nassistant\\n' (special tokens stripped) -> (system, user)."""
    assert input_text.startswith("system\n"), "unexpected prompt layout"
    body = input_text[len("system\n"):]
    i = body.find("\nuser\n")
    system = body[:i]
    rest = body[i + len("\nuser\n"):]
    if rest.endswith("\nassistant\n"):
        rest = rest[: -len("\nassistant\n")]
    return system, rest


# ---------------------------------------------------------------------------------------------- per-rollout audit
def audit_rollout(rec, ridx, R):
    input_text, output = rec["input"], rec["output"]
    system, user = split_prompt(input_text)
    p_ids, p_text_sp, p_text = R.render_prompt(system, user)
    prompt_verified = p_text.rstrip("\n") == input_text.rsplit("\nassistant\n", 1)[0].rstrip("\n")
    turns, notes = split_turns(output)
    records = []
    serial = p_text_sp                         # reconstructed serialised context (WITH special tokens)
    main_tokens = len(p_ids)
    history_thinks = []                        # (turn_idx, think_text) of previous assistant turns
    pending_call = None                        # executed call awaiting its observation
    branch_calls = 0
    for t, (role, content, joint) in enumerate(turns):
        rec_t = {"rollout": ridx, "turn": t, "role": role, "text": content, "joint_with_previous": joint}
        if role == "assistant":
            a = parse_assistant(content)
            rec_t.update(a)
            rec_t["tokens_est"] = R.n(content) + 1           # + eos, as stored by Agent.append
            rec_t["exceeds_turn_max_new_tokens"] = rec_t["tokens_est"] > TURN_MAX_NEW_TOKENS
            # What the policy saw before generating this turn = serialised prefix + generation prompt
            model_input = serial + R.gen_prompt_text
            rec_t["model_input_tokens_est"] = main_tokens + len(R.gen_prompt_ids)
            rec_t["model_input_reconstructed_tail"] = model_input[-600:]
            # history think status: every earlier think block must be in the model input verbatim
            st = []
            for (ti, th) in history_thinks:
                if th is None:
                    st.append({"turn": ti, "status": "no_think_in_original"})
                elif th == "":
                    st.append({"turn": ti, "status": "preserved_empty_model_generated"})
                else:
                    st.append({"turn": ti, "status": "preserved" if th in model_input else "missing_or_modified"})
            rec_t["history_think_status"] = st
            # branch view: how this turn would be re-rendered from text when a branch inherits the history
            _, sp, plain = R.render_turn("assistant", content)
            same = plain[len("assistant\n"):].rstrip("\n") == content.rstrip("\n") and plain.startswith("assistant\n")
            if a["has_think"]:
                rec_t["branch_view"] = "identical" if same else "modified_whitespace_or_layout"
            else:
                rec_t["branch_view"] = "empty_think_inserted_by_template" if "<think>\n\n</think>" in sp else ("identical" if same else "modified")
            rec_t["branch_view_text_with_special"] = sp
            anomalies = []
            if a["unclosed_think"]:
                anomalies.append("unclosed_think")
            if a["stray_close_think"]:
                anomalies.append("stray_close_think")
            if a["unclosed_call"]:
                anomalies.append("unclosed_function_tag")
            if len(a["calls"]) > 1:
                anomalies.append(f"multiple_calls({len(a['calls'])}): executed last = {a['executed_call']['function'] if a['executed_call'] else None}")
            if any(c["inside_think"] for c in a["calls"]):
                anomalies.append("tool_call_xml_inside_think")
            if a["executed_inside_think"]:
                anomalies.append("EXECUTED_call_taken_from_inside_think")
            if a["suffix_after_call"]:
                anomalies.append("text_after_function_call")
            if rec_t["exceeds_turn_max_new_tokens"]:
                anomalies.append(f"turn_longer_than_turn_max_new_tokens({TURN_MAX_NEW_TOKENS})")
            if a["unclosed_call"] and rec_t["tokens_est"] < TURN_MAX_NEW_TOKENS:
                anomalies.append("unclosed_call_emitted_by_model(short_turn,not_a_cap_cut)")
            anomalies += a["discrepancies"]
            if not a["has_think"]:
                anomalies.append("no_think_block")
            if pending_call is not None:
                anomalies.append("assistant_turn_without_observation_for_previous_call")
            rec_t["anomalies"] = anomalies
            history_thinks.append((t, a["think_text"] if a["has_think"] else None))
            if a["executed_names"] and a["executed_names"][0] == "finish":
                pending_call = None
            elif not a["executed_names"]:
                pending_call = "none"
            elif a["executed_names"][0] in EXPECTED_OBS:
                pending_call = a["executed_names"][0]
            else:
                pending_call = "unsupported"
            if a["executed_names"] == ["branch"]:
                branch_calls += 1
            # serialised form of a sampled assistant turn: generation prompt + raw text + eos (no trailing newline: Agent.append)
            serial += R.gen_prompt_text + content + R.eos_text
            main_tokens += rec_t["tokens_est"] + len(R.gen_prompt_ids)
        else:
            body = content.strip()
            is_obs = body.startswith("<tool_response>")
            inner = body[len("<tool_response>"):].strip() if is_obs else body
            intact = is_obs and body.endswith("</tool_response>")
            if intact:
                inner = inner[: -len("</tool_response>")].strip()
            kind = obs_kind(inner) if is_obs else ("summary_prompt" if body.startswith(QSUM_HEAD) else "resume" if body.startswith(RESUME_HEAD) else "other_user")
            rec_t.update({"kind": kind, "wrapped": is_obs, "intact": intact, "observation": inner, "chars": len(content)})
            ids, sp, plain = R.render_turn("user", content.strip())
            rec_t["tokens_est"] = len(ids)
            rec_t["rerender_matches_dump"] = plain.startswith("user\n") and plain[len("user\n"):].strip() == content.strip()
            anomalies = []
            if not is_obs:
                anomalies.append(f"user_turn_not_wrapped({kind})")
            if is_obs and not intact:
                anomalies.append("observation_not_closed" + ("_at_end_of_dump(clipped_to_response_length)" if t == len(turns) - 1 else ""))
            if pending_call is None:
                anomalies.append("observation_after_finish_or_at_rollout_start")
            else:
                exp = EXPECTED_OBS.get(pending_call, ())
                rec_t["executed_call"] = pending_call
                if kind not in exp:
                    anomalies.append(f"observation_kind_mismatch(executed={pending_call},obs={kind})")
            if not rec_t["rerender_matches_dump"]:
                anomalies.append("rerender_differs_from_dump")
            rec_t["anomalies"] = anomalies
            pending_call = None
            serial += sp
            main_tokens += len(ids)
        records.append(rec_t)
    finished = any(r["role"] == "assistant" and r["executed_names"][:1] == ["finish"] for r in records)
    summary = {
        "rollout": ridx, "question": question_of(input_text), "score": float(rec["score"]), "finished": finished,
        "n_turns": len(turns), "n_assistant": sum(r["role"] == "assistant" for r in records),
        "prompt_tokens": len(p_ids), "prompt_rerender_matches_dump": prompt_verified,
        "main_tokens_est_total": main_tokens, "branch_calls": branch_calls,
        "branch_tokens": None, "combined_tokens": None,     # filled from capture / ledger when available
        "notes": notes,
    }
    return summary, records


# ---------------------------------------------------------------------------------------------- capture (ground truth)
def load_capture(capture_dir):
    recs = []
    for f in sorted(glob.glob(os.path.join(capture_dir, "capture_*.jsonl"))):
        for l in open(f):
            recs.append(json.loads(l))
    by = defaultdict(list)
    for r in recs:
        tag = r.get("tag") or {}
        by[tag.get("instance_id") or tag.get("uid")].append(r)
    for k in by:
        by[k].sort(key=lambda r: r["ts"])
    return by


def attach_capture(summary, records, cap_rows, R):
    """Compare captured main-context prompts with the reconstruction, and add branch usage."""
    mains = [r for r in cap_rows if r.get("agent") == "main"]
    branches = [r for r in cap_rows if r.get("agent") not in (None, "main")]
    a_turns = [r for r in records if r["role"] == "assistant"]
    matched = 0
    for r_t, c in zip(a_turns, mains):
        r_t["captured_prompt_text"] = c["prompt_text"]
        r_t["captured_prompt_tokens"] = c["prompt_tokens"]
        r_t["captured_completion_text"] = c["completion_text"]
        r_t["captured_completion_tokens"] = c["completion_tokens"]
        recon_ok = c["prompt_text"].endswith(r_t["model_input_reconstructed_tail"][-200:])
        r_t["capture_matches_reconstruction_tail"] = recon_ok
        for st in r_t["history_think_status"]:
            th = next((x["think_text"] for x in a_turns if x["turn"] == st["turn"]), None)
            if th:
                st["status_captured"] = "preserved" if th in c["prompt_text"] else "missing_or_modified"
        matched += 1
    summary["captured_main_steps"] = len(mains)
    summary["captured_steps_matched"] = matched
    summary["captured_prompt_tokens_last"] = mains[-1]["prompt_tokens"] if mains else None
    bt = defaultdict(lambda: {"steps": 0, "prompt_tokens_max": 0, "completion_tokens": 0, "calls": Counter(), "template_empty_think_in_inherited": None})
    for c in branches:
        b = bt[c["agent"]]
        b["steps"] += 1
        b["prompt_tokens_max"] = max(b["prompt_tokens_max"], c["prompt_tokens"])
        b["completion_tokens"] += c["completion_tokens"] or 0
        fc = extract_fn_call(c.get("completion_text"))
        if fc:
            b["calls"][fc["function"]] += 1
        if b["steps"] == 1:
            # inherited history as the branch actually saw it: count template-inserted empty thinks in it
            head = c["prompt_text"]
            b["inherited_prompt_tokens"] = c["prompt_tokens"]
            b["template_empty_think_in_inherited"] = head.count("<think>\n\n</think>")
            b["inherited_prompt_text"] = head
    summary["branches_captured"] = {k: {**v, "calls": dict(v["calls"])} for k, v in bt.items()}
    summary["branch_tokens"] = {"generated": sum(v["completion_tokens"] for v in bt.values()),
                                "peak_prompt": max([v["prompt_tokens_max"] for v in bt.values()] or [0])}
    summary["combined_tokens"] = {"main_generated": sum(c["completion_tokens"] or 0 for c in mains),
                                  "branch_generated": summary["branch_tokens"]["generated"],
                                  "total_generated": sum(c["completion_tokens"] or 0 for c in cap_rows),
                                  "cumulative_prompt_tokens_all_steps": sum(c["prompt_tokens"] for c in cap_rows)}


# ---------------------------------------------------------------------------------------------- rendering
def _fence(text, lang=""):
    fence = "````" if "```" in text else "```"
    return f"{fence}{lang}\n{text.rstrip()}\n{fence}\n"


def _trunc(text, n, label="observation"):
    if n and len(text) > n:
        return text[:n] + f"\n… [{label} truncated for readability: {len(text) - n:,} more chars; full text in the .jsonl]"
    return text


def render_detail(summary, records, args, R, cap=None):
    out = [f"\n## Rollout {summary['rollout']} — score {summary['score']:.0f} — {summary['n_assistant']} assistant turns, "
           f"{summary['branch_calls']} branch call(s) — {'finished' if summary['finished'] else 'NOT finished'}\n\n",
           f"**Question:** {summary['question']}\n\n",
           f"Prompt: {summary['prompt_tokens']:,} tokens (system + task, re-rendered with the checkpoint template; "
           f"{'matches' if summary['prompt_rerender_matches_dump'] else '⚠ DIFFERS from'} the dump text). "
           f"Main context at the end ≈ {summary['main_tokens_est_total']:,} tokens (re-tokenised estimate).\n\n"]
    if cap is not None:
        out.append(f"Captured steps: {summary.get('captured_main_steps')} main, branches: "
                   f"{json.dumps({k: {'steps': v['steps'], 'peak_prompt': v['prompt_tokens_max'], 'generated': v['completion_tokens']} for k, v in summary['branches_captured'].items()})}; "
                   f"combined generated tokens {summary['combined_tokens']}\n\n")
    for r in records:
        t = r["turn"]
        if r["role"] == "assistant":
            out.append(f"### [{t}] assistant — generated output ({r['tokens_est']:,} tokens est.)\n\n")
            flags = ", ".join(r["anomalies"]) or "none"
            think = "none" if not r["has_think"] else ("EMPTY (model-generated: raw sampled ids)" if r["think_empty"] else f"{len(r['think_text']):,} chars")
            out.append(f"think: **{think}** · generated closed calls: {[c['name'] + (' (inside think!)' if c['inside_think'] else '') for c in r['calls']]} · "
                       f"lenient parse (agent): `{r['agent_parse']['function'] if r['agent_parse'] else None}` · **executed: {r['executed_names'] or 'none'}** by {r['executed_by']} · anomalies: {flags}\n\n")
            out.append(_fence(r["text"]))
            hs = r["history_think_status"]
            if hs:
                c = Counter(x["status"] for x in hs)
                cc = Counter(x.get("status_captured", "n/a") for x in hs) if cap is not None else None
                out.append(f"History think blocks in this turn's model input (reconstructed): {dict(c)}"
                           + (f"; in the CAPTURED prompt: {dict(cc)}" if cc else "") + "\n\n")
            if r["branch_view"] != "identical":
                out.append(f"Branch view of this turn (template re-render when a branch inherits the history): **{r['branch_view']}**\n\n"
                           + _fence(r["branch_view_text_with_special"]))
            if args.checkpoints and (t == 0 or (records[t - 1]["role"] == "user")):
                label = "captured" if r.get("captured_prompt_text") else "reconstructed from the dump under the documented tokenisation path (NOT a capture)"
                text = r.get("captured_prompt_text") or r["model_input_reconstructed_tail"]
                out.append(f"<details><summary>model input before this turn ({label}; tail)</summary>\n\n" + _fence(_trunc(text[-args.ckpt_chars:], 0)) + "</details>\n\n")
        else:
            head = f"### [{t}] user · {r['kind']} ({r['chars']:,} chars, {r['tokens_est']:,} tokens" + (", NOT closed" if r["wrapped"] and not r["intact"] else "") + \
                   (", re-render == dump ✓" if r["rerender_matches_dump"] else ", ⚠ re-render ≠ dump") + \
                   (f", executed call: `{r.get('executed_call')}`" if r.get("executed_call") else "") + ")\n\n"
            out.append(head)
            if r["anomalies"]:
                out.append("anomalies: " + ", ".join(r["anomalies"]) + "\n\n")
            out.append(_fence(_trunc(r["observation"], args.obs_chars), "text"))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True, help="output stem: <out>.md and <out>.jsonl")
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--capture", default=None, help="capture dir from a FOLD_PROMPT_CAPTURE=1 eval (exact prompts)")
    ap.add_argument("--detail", default="", help="comma list of rollout indices to render in full; default = the original trace's picks")
    ap.add_argument("--select", default="single_correct:2,single_wrong:1,other:1")
    ap.add_argument("--extra-branch-examples", type=int, default=2, help="add rollouts with most branch calls")
    ap.add_argument("--obs-chars", type=int, default=2500)
    ap.add_argument("--ckpt-chars", type=int, default=3000)
    ap.add_argument("--checkpoints", action="store_true", default=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--preamble", default=None, help="markdown file inserted after the header (verified config, findings)")
    args = ap.parse_args()

    R = Renderer(args.tokenizer)
    dump = [json.loads(l) for l in open(args.dump)]
    summaries, all_records = [], []
    for i, rec in enumerate(dump):
        s, rs = audit_rollout(rec, i, R)
        summaries.append(s)
        all_records.append(rs)
    cap = load_capture(args.capture) if args.capture else None
    if cap is not None:
        for s, rs, rec in zip(summaries, all_records, dump):
            key = None
            for k, rows in cap.items():
                # match by question text inside the first captured prompt
                if rows and s["question"][:120] in rows[0]["prompt_text"]:
                    key = k
                    break
            if key:
                attach_capture(s, rs, cap[key], R)

    # ---------------- aggregate
    A = [r for rs in all_records for r in rs if r["role"] == "assistant"]
    U = [r for rs in all_records for r in rs if r["role"] == "user"]
    agg = OrderedDict()
    agg["rollouts"] = len(dump)
    agg["mean_score"] = round(sum(s["score"] for s in summaries) / len(summaries), 4)
    agg["finished"] = sum(s["finished"] for s in summaries)
    agg["assistant_turns"] = len(A)
    agg["think: present / empty(model-generated) / absent"] = (sum(r["has_think"] for r in A), sum(r["think_empty"] for r in A), sum(not r["has_think"] for r in A))
    agg["unclosed_think"] = sum(r["unclosed_think"] for r in A)
    agg["turns_with_multiple_calls"] = sum(len(r["calls"]) > 1 for r in A)
    agg["turns_with_call_xml_inside_think"] = sum(any(c["inside_think"] for c in r["calls"]) for r in A)
    agg["turns_whose_EXECUTED_call_came_from_inside_think"] = sum(r["executed_inside_think"] for r in A)
    agg["unclosed_function_tag"] = sum(r["unclosed_call"] for r in A)
    agg["text_after_call"] = sum(bool(r["suffix_after_call"]) for r in A)
    agg["executed_calls (first of turn)"] = dict(Counter(r["executed_names"][0] if r["executed_names"] else "none" for r in A))
    agg["executed_by"] = dict(Counter(r["executed_by"] for r in A))
    agg["turns where env executed >1 call"] = sum(len(r["executed_names"]) > 1 for r in A)
    agg["turns where the env executed a call sitting inside <think>"] = sum(r["executed_inside_think"] for r in A)
    agg["generated-vs-executed discrepancies"] = dict(Counter(d.split("=")[0] if "vs" not in d else "lenient_parse_vs_env" for r in A for d in r["discrepancies"]))
    agg["unclosed calls emitted by the model in short turns"] = sum("unclosed_call_emitted_by_model(short_turn,not_a_cap_cut)" in r["anomalies"] for r in A)
    agg[f"assistant turns longer than turn_max_new_tokens={TURN_MAX_NEW_TOKENS} (cap not applied)"] = (sum(r["exceeds_turn_max_new_tokens"] for r in A), max(r["tokens_est"] for r in A))
    agg["assistant_turn_ends_with_newline_before_eos"] = sum(r["ends_with_newline"] for r in A)
    agg["user_turns"] = len(U)
    agg["observation kinds"] = dict(Counter(r["kind"] for r in U))
    agg["observations_not_wrapped"] = sum(not r["wrapped"] for r in U)
    agg["observations_not_closed (at end of dump)"] = (sum(r["wrapped"] and not r["intact"] for r in U),
                                                       sum(r["wrapped"] and not r["intact"] and "observation_not_closed_at_end_of_dump(clipped_to_response_length)" in r["anomalies"] for r in U))
    agg["observation_after_finish_or_at_start"] = sum("observation_after_finish_or_at_rollout_start" in r["anomalies"] for r in U)
    agg["observation_kind_mismatch"] = sum(any(a.startswith("observation_kind_mismatch") for a in r["anomalies"]) for r in U)
    agg["user_turn_rerender_matches_dump"] = (sum(r["rerender_matches_dump"] for r in U), len(U))
    agg["prompt_rerender_matches_dump"] = (sum(s["prompt_rerender_matches_dump"] for s in summaries), len(summaries))
    hist = Counter(x["status"] for r in A for x in r["history_think_status"])
    agg["history think blocks in later model inputs (reconstructed)"] = dict(hist)
    agg["branch_view of assistant turns (template re-render in a branch)"] = dict(Counter(r["branch_view"] for r in A))
    agg["rollouts_with_branch_calls"] = sum(s["branch_calls"] > 0 for s in summaries)
    agg["branch_calls_total"] = sum(s["branch_calls"] for s in summaries)
    if cap is not None:
        capd = [s for s in summaries if s.get("captured_main_steps")]
        agg["captured rollouts"] = len(capd)
        agg["captured steps matched to dump turns"] = sum(s["captured_steps_matched"] for s in capd)
        agg["captured prompt tail == reconstruction"] = sum(r.get("capture_matches_reconstruction_tail", False) for r in A)
        agg["history think blocks in CAPTURED prompts"] = dict(Counter(x.get("status_captured") for r in A for x in r["history_think_status"] if "status_captured" in x))
        agg["branch tokens (generated) total"] = sum(s["branch_tokens"]["generated"] for s in capd if s.get("branch_tokens"))
        agg["template-inserted empty thinks in inherited branch histories"] = sum(v.get("template_empty_think_in_inherited") or 0 for s in capd for v in s["branches_captured"].values())

    # ---------------- selection for detail
    def kind_of(s):
        return ("single_correct" if s["score"] > 0 else "single_wrong") if s["finished"] else "other"
    if args.detail:
        picks = [int(x) for x in args.detail.split(",")]
        extra = sorted(summaries, key=lambda s: -s["branch_calls"])[: args.extra_branch_examples]
        picks += [s["rollout"] for s in extra if s["rollout"] not in picks]
    else:
        picks = []
        want = OrderedDict((k, int(v)) for k, v in (x.split(":") for x in args.select.split(",")))
        for k, v in want.items():
            pool = [s for s in summaries if kind_of(s) == k]
            pool.sort(key=lambda s: -s["n_assistant"])
            picks += [s["rollout"] for s in pool[:v]]
        extra = sorted(summaries, key=lambda s: -s["branch_calls"])[: args.extra_branch_examples]
        picks += [s["rollout"] for s in extra if s["rollout"] not in picks]

    # ---------------- write jsonl
    with open(args.out + ".jsonl", "w") as f:
        for s, rs in zip(summaries, all_records):
            f.write(json.dumps({"type": "rollout", **s}, ensure_ascii=False) + "\n")
            for r in rs:
                f.write(json.dumps({"type": "turn", **r}, ensure_ascii=False) + "\n")

    # ---------------- write md
    title = args.title or f"Trace audit — {os.path.basename(os.path.dirname(args.dump))} step {dump[0].get('step')} ({os.path.basename(args.dump)})"
    md = [f"# {title}\n\n",
          f"Dump: `{args.dump}` ({len(dump)} rollouts). Tokenizer/template: `{R.path}` — profile "
          f"strips_history_think={R.profile.strips_history_think}, think_prefilled={R.profile.think_prefilled}, "
          f"generation prompt = `{R.gen_prompt_text!r}`.\n\n",
          "Evidence chain: the dump's `input`+`output` is the decoded concatenation of the token ids the policy received in the main context "
          "(`Agent.context()` → `CallLLM`; `AgentContext.get_data()`), with special tokens stripped. Serialised prompts shown below are "
          "**reconstructed** from that text (assistant turns = generation prompt + raw text + eos, user turns re-rendered with the real template) "
          "unless labelled *captured*. Observations in the .jsonl are intact; here they are truncated with a label.\n\n",
          "## Aggregate diagnostics (all rollouts)\n\n| metric | value |\n|---|---|\n"]
    for k, v in agg.items():
        md.append(f"| {k} | {v} |\n")
    if args.preamble and os.path.exists(args.preamble):
        md.append("\n" + open(args.preamble).read() + "\n")
    md.append("\n## Detailed traces\n")
    for p in picks:
        md.append(render_detail(summaries[p], all_records[p], args, R, cap))
    with open(args.out + ".md", "w") as f:
        f.write("".join(md))
    print(f"wrote {args.out}.md ({sum(len(x) for x in md):,} chars) and {args.out}.jsonl; detail rollouts {picks}")
    print(json.dumps(agg, indent=1, default=str))


if __name__ == "__main__":
    main()
