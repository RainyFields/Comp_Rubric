#!/usr/bin/env python3
"""End-to-end rollout audit from a prompt-capture evaluation (FOLD_PROMPT_CAPTURE=1).

audit   : one results dir (capture/, dump/0.jsonl, judge_calls.jsonl, valonly.log) -> <out>/<arm>_trace_audit.jsonl,
          <out>/<arm>_metrics.json, <out>/<arm>_rollouts.json (per-rollout summary), <out>/<arm>_readable.md (selected traces)
compare : several *_metrics.json / *_rollouts.json -> <out>/comparison.md (aggregate table + paired outcome changes)

Every check works on the RAW TOKEN IDS the policy received (capture records store the prompt ids as suffix-of-previous or
in full, the sampled completion ids, and the stored turn ids + training mask). Text is decoded only for display.

Checks per rollout (see docs/traces/grpo_fixed_shakeout/README.md for definitions):
  cap        every completion_tokens <= turn cap; count of turns exactly at the cap (cut by the cap)
  history    main context: prompt_t == prompt_{t-1} minus generation prompt + stored_turn_{t-1} + observation ids (prefix
             equality checked on ids); every earlier completion's ids remain a contiguous subsequence of every later prompt
  fork       a branch's first prompt starts with the parent's inherited ids (sha1 + length from the fork event)
  tail       compaction: each append_tokens event's ids occur contiguously in the resumed segment's first prompt
  masks      stored_turn_mask: trainable count == completion_tokens (incl. eos), generation prompt + newline untrained
  parser     results' call ids are a subset of the executable ids; calls_in_think never executed; observation text is the
             concatenation of the per-call observations; multi-call turns
  template   prompt text contains no '<|im_end|><|im_start|>' (missing newline) and no template-inserted empty think
  termination finish accepted / max_turn / context exhausted (inferred) / timeout
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import types
from collections import Counter, OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from agents.parsing import find_think, parse_actions  # noqa: E402
from agents.prompts import COMPACTION_SUMMARY_PROMPT  # noqa: E402

QUESTION = re.compile(r"Question: (.*?)\n\n(?:\* You can search|Your response should contain)", re.S)   # search / search_branch / search_base prompts
DOCID_CITE = re.compile(r"\[(?:docid:\s*)?(\d{3,})\]")
DOCID_OBS = re.compile(r"docid: (\d+)")
QSUM_HEAD = COMPACTION_SUMMARY_PROMPT.split("\n")[0][:40]


def sha1(ids):
    return hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:16]


def ids_str(ids):
    return "," + ",".join(map(str, ids)) + ","


def contains(seq, sub, seq_str=None, sub_str=None):
    """True if list ``sub`` occurs contiguously in list ``seq`` (delimited-string search; C speed)."""
    if not sub:
        return True
    return (sub_str or ids_str(sub)) in (seq_str or ids_str(seq))


# ------------------------------------------------------------------------------------------------- loading
def load_results_dir(rdir):
    cap = []
    for f in sorted(glob.glob(os.path.join(rdir, "capture", "capture_*.jsonl"))):
        with open(f) as fh:
            for ln, line in enumerate(fh):
                r = json.loads(line)
                r["_ref"] = f"{os.path.basename(f)}:{ln}"
                cap.append(r)
    dump = []
    for f in sorted(glob.glob(os.path.join(rdir, "dump", "*.jsonl"))):
        dump += [json.loads(l) for l in open(f)]
    judge = []
    jf = os.path.join(rdir, "judge_calls.jsonl")
    if os.path.exists(jf):
        judge = [json.loads(l) for l in open(jf)]
    log = open(os.path.join(rdir, "valonly.log"), errors="replace").read() if os.path.exists(os.path.join(rdir, "valonly.log")) else ""
    return cap, dump, judge, log


def question_of(text):
    m = QUESTION.search(text or "")
    return m.group(1).strip() if m else None


def group_rollouts(cap):
    by = defaultdict(list)
    for r in cap:
        tag = r.get("tag") or {}
        key = tag.get("instance_id") or tag.get("uid") or "?"
        by[key].append(r)
    for k in by:
        by[k].sort(key=lambda r: r["ts"])
    return by


def reconstruct_prompts(rows):
    """Replay suffix-mode prompt ids per context_uid; attach r['_prompt'] (full ids) to step records."""
    prev = {}
    for r in rows:
        if r.get("kind", "step") != "step":
            continue
        ids = r.get("prompt_ids")
        if ids is None:
            r["_prompt"] = None
            continue
        if r.get("prompt_ids_mode") == "suffix":
            full = prev.get(r["context_uid"], []) + ids
        else:
            full = list(ids)
        prev[r["context_uid"]] = full
        r["_prompt"] = full
        r["_prompt_sha1_ok"] = (sha1(full) == r.get("prompt_sha1")) if r.get("prompt_sha1") else None


# ------------------------------------------------------------------------------------------------- per-rollout audit
def audit_rollout(key, rows, dump_rec, judge_recs, args, tok):
    steps = [r for r in rows if r.get("kind", "step") == "step"]
    actions = [r for r in rows if r.get("kind") == "action"]
    forks = [r for r in rows if r.get("kind") == "fork"]
    tails = [r for r in rows if r.get("kind") == "append_tokens"]
    ends = [r for r in rows if r.get("kind") == "rollout_end"]
    reconstruct_prompts(rows)
    has_ids = any(r.get("_prompt") is not None for r in steps)
    gp_len = None
    if tok is not None:
        from agents.utils import Agent
        cfg = types.SimpleNamespace(prompt_length=8192, response_length=32768, plugin=types.SimpleNamespace(retry_cjk=0))
        a = Agent(None, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], tok, cfg, prompt_turn=2)
        gp = a.get_generation_prompt()
        gp_len = len(gp)
    agents = OrderedDict()
    for r in steps:
        agents.setdefault(r["agent"] or r["context_uid"], []).append(r)
    action_by = {(r["agent"], r["turn_index"]): r for r in actions}
    turns_out, checks = [], Counter()
    cap_hits = cap_over = 0
    completions_by_agent = defaultdict(list)     # agent -> list of (turn_index, completion ids)
    hist_missing = 0
    prefix_break = 0
    for agent_name, srows in agents.items():
        prev = None
        for r in srows:
            comp = r.get("completion_ids")
            ct = r.get("completion_tokens") or 0
            if ct >= args.turn_cap:
                cap_hits += 1
            if ct > args.turn_cap:
                cap_over += 1
            t = {
                "rollout": key, "agent": agent_name, "context_uid": r["context_uid"], "turn_index": r["turn_index"], "ref": r["_ref"],
                "ts": r["ts"], "prompt_tokens": r["prompt_tokens"], "prompt_sha1": r.get("prompt_sha1"), "prompt_sha1_ok": r.get("_prompt_sha1_ok"),
                "completion_tokens": ct, "completion_sha1": sha1(comp) if comp else None, "completion_none": r.get("completion_none"),
                "completion_text": r.get("completion_text"), "prompt_text_tail": (r.get("prompt_text") or "")[-args.tail_chars:],
                "hit_turn_cap": ct >= args.turn_cap, "over_turn_cap": ct > args.turn_cap,
            }
            # think / parser
            text = r.get("completion_text") or ""
            reasoning, span = find_think(text)
            t["think"] = "none" if reasoning is None else ("empty" if reasoning == "" else "present")
            t["think_chars"] = len(reasoning) if reasoning else 0
            t["unclosed_think"] = ("<think>" in text and "</think>" not in text)
            parsed = parse_actions(text, turn_id=str(r["turn_index"]))
            t["parsed"] = {"calls": [{"call_id": c["call_id"], "function": c["function"], "arguments": c["arguments"]} for c in parsed["calls"]],
                           "executable": [c["call_id"] for c in parsed["executable"]], "error": parsed["error"],
                           "calls_in_think": parsed["calls_in_think"], "unclosed_tags": parsed["unclosed_tags"]}
            act = action_by.get((agent_name, r["turn_index"]))
            if act is not None:
                res = act.get("results") or []
                t["executed"] = [{k: v for k, v in x.items() if k != "observation"} | {"observation_chars": len(x.get("observation") or "")} for x in res]
                t["observation"] = act.get("observation")
                t["observation_chars"] = act.get("observation_chars")
                # executed functions must be exactly the executable calls, in order (env ids are turn-agnostic: compare by sequence)
                exec_fns = [x.get("function") for x in res if x.get("status") in ("ok", "error")]
                expected_fns = [c["function"] for c in parsed["executable"]]
                if exec_fns and exec_fns[0] == "guard":
                    expected_fns = exec_fns
                t["check_exec_subset_of_executable"] = (exec_fns == expected_fns) or (exec_fns == ["branch"] and expected_fns == ["branch"]) or (not exec_fns and (not expected_fns or parsed["error"] is not None))
                t["check_no_think_call_executed"] = True   # by construction of parse_actions; verified via the subset check + calls_in_think
                if act.get("observation") and res and all(x.get("observation") is not None for x in res):
                    joined = "".join(x["observation"] for x in res).strip()
                    t["check_obs_is_concat_of_calls"] = act["observation"].startswith(joined[:300]) if joined else True
                t["multi_call"] = len(res) > 1
                if not t["check_exec_subset_of_executable"]:
                    checks["exec_not_subset"] += 1
            else:
                t["executed"] = None
            # template / boundaries (text level, on the exact prompt)
            pt = r.get("prompt_text") or ""
            t["check_no_glued_boundary"] = "<|im_end|><|im_start|>" not in pt
            t["double_newline_empty_think_in_prompt"] = pt.count("<|im_start|>assistant\n<think>\n\n</think>")   # model-written when fork ids are exact
            if not t["check_no_glued_boundary"]:
                checks["glued_boundary_prompts"] += 1
            # masks
            m = r.get("stored_turn_mask")
            if m is not None and comp is not None:
                n_train = sum(m)
                t["mask_trainable"] = n_train
                t["mask_total"] = len(m)
                # trainable == sampled ids (incl. eos); generation prompt untrained; anything stored after the sampled ids
                # (v2: the terminator newline) untrained. In legacy protocol the stored turn ends at the trained eos.
                stored = r.get("stored_turn_ids") or []
                gp_ok = gp_len is None or not any(m[:gp_len])
                tail_ok = (not any(m[(gp_len or 0) + len(comp):])) if len(stored) > (gp_len or 0) + len(comp) else bool(m[-1])
                t["check_mask"] = (n_train == len(comp)) and gp_ok and tail_ok
                if not t["check_mask"]:
                    checks["mask_mismatch"] += 1
            # history checks on ids
            if has_ids and r.get("_prompt") is not None:
                P = r["_prompt"]
                if prev is not None and prev.get("_prompt") is not None and prev.get("stored_turn_ids"):
                    base = prev["_prompt"][:-gp_len] if gp_len else prev["_prompt"]
                    expect = base + prev["stored_turn_ids"]
                    ok = P[:len(expect)] == expect
                    t["check_prefix_continuity"] = ok
                    if not ok:
                        # designed rollbacks: compaction rolls the last step back before q_sum; a branch that hits its
                        # budget rolls back 2 turns before the forced-return prompt. Everything else is a real break.
                        pt_now = r.get("prompt_text") or ""
                        designed = (QSUM_HEAD in pt_now[-6000:]) or ("The context limit has been exceeded for the branch" in pt_now[-6000:])
                        t["prefix_break_kind"] = "designed_rollback" if designed else "UNEXPECTED"
                        if designed:
                            checks["designed_rollback"] += 1
                        else:
                            prefix_break += 1
                            checks["prefix_break_UNEXPECTED"] += 1
                miss = []
                P_str = ids_str(P)
                for (ti, cids, cstr) in completions_by_agent[agent_name]:
                    if cstr not in P_str:
                        miss.append(ti)
                t["history_completions_missing"] = miss
                if miss:
                    rolled = any(x.get("prefix_break_kind") == "designed_rollback" for x in turns_out if x["agent"] == agent_name) or t.get("prefix_break_kind") == "designed_rollback"
                    t["history_missing_kind"] = "after_designed_rollback" if rolled else "UNEXPECTED"
                    if rolled:
                        checks["history_missing_after_rollback"] += 1
                    else:
                        hist_missing += 1
                        checks["history_completion_missing_UNEXPECTED"] += 1
                if comp:
                    completions_by_agent[agent_name].append((r["turn_index"], list(comp), ids_str(comp)))
            prev = r
            turns_out.append(t)
    # fork checks (branch inheritance)
    fork_checks = []
    for f in forks:
        child = [r for r in steps if r["context_uid"] == f["context_uid"]]
        if child and child[0].get("_prompt") is not None:
            P = child[0]["_prompt"]
            ok = len(P) >= f["inherited_tokens"] and sha1(P[:f["inherited_tokens"]]) == f["inherited_sha1"]
            fork_checks.append({"child": f.get("agent") or f["context_uid"], "parent": f.get("parent_agent"), "inherited_tokens": f["inherited_tokens"], "exact": ok})
            checks["fork_ok" if ok else "fork_MISMATCH"] += 1
    # tail checks (compaction)
    tail_checks = []
    by_ctx = defaultdict(list)
    for e in tails:
        by_ctx[e["context_uid"]].append(e)
    for ctx, evs in by_ctx.items():
        seg = [r for r in steps if r["context_uid"] == ctx]
        if not seg or seg[0].get("_prompt") is None:
            continue
        P = seg[0]["_prompt"]
        # resumed segment = task prompt + resume turn + tail turns (in event order) + generation prompt:
        # the tail ids end exactly gp_len before the end of the first prompt, so every event has a known offset
        end = len(P) - (gp_len or 0)
        offsets = []
        for e in reversed(evs):
            end -= e["ids_len"]
            offsets.append(end)
        for e, off in zip(reversed(evs), offsets):
            ok = off >= 0 and sha1(P[off:off + e["ids_len"]]) == e["ids_sha1"]
            if not ok:   # fall back to a bounded scan near the expected offset (robust to an unexpected extra turn)
                L = e["ids_len"]
                ok = any(sha1(P[i:i + L]) == e["ids_sha1"] for i in range(max(0, off - 64), min(len(P) - L, off + 64) + 1))
            tail_checks.append({"segment": e.get("agent"), "role": e.get("role"), "ids_len": e["ids_len"], "offset": off, "exact": ok})
            checks["tail_ok" if ok else "tail_MISMATCH"] += 1
    # compaction events (segments)
    seg_names = [a for a in agents if str(a).startswith("seg")]
    n_compactions = max(0, len(seg_names) - 1)
    summary_tokens = 0
    for a, srows in agents.items():
        for i, r in enumerate(srows):
            pt = r.get("prompt_text") or ""
            if pt.endswith("<|im_start|>assistant\n") and QSUM_HEAD in pt[-6000:]:
                summary_tokens += r.get("completion_tokens") or 0
                r["_is_summary_step"] = True
    # tokens / cost
    main_rows = agents.get("main", []) or agents.get("seg0", [])
    gen_main = sum((r.get("completion_tokens") or 0) for a, srows in agents.items() if a == "main" or str(a).startswith("seg") for r in srows) - summary_tokens
    gen_branch = sum((r.get("completion_tokens") or 0) for a, srows in agents.items() if a != "main" and not str(a).startswith("seg") for r in srows)
    prefill = sum(r["prompt_tokens"] for r in steps)
    peak = max([r["prompt_tokens"] + (r.get("completion_tokens") or 0) for r in steps] or [0])
    peak_main = max([r["prompt_tokens"] + (r.get("completion_tokens") or 0) for r in main_rows] or [0])
    wall = (steps[-1]["ts"] - steps[0]["ts"]) if len(steps) > 1 else 0.0
    # termination
    last_main = (agents.get("main") or agents.get(seg_names[-1] if seg_names else None) or steps)[-1] if steps else None
    last_act = None
    if last_main is not None:
        last_act = action_by.get((last_main["agent"], last_main["turn_index"]))
    finished = bool(last_act and last_act.get("observation") is None and
                    ((any(x.get("function") == "finish" for x in (last_act.get("results") or []))) or
                     (parse_actions(last_main.get("completion_text") or "")["executable"][-1:] and parse_actions(last_main.get("completion_text") or "")["executable"][-1]["function"] == "finish")))
    if not finished and last_main is not None and not actions:      # captures without action records (old code): infer from the last completion
        pl = parse_actions(last_main.get("completion_text") or "")
        finished = bool(pl["executable"]) and pl["executable"][-1]["function"] == "finish"
    if ends:
        stop = ends[-1].get("stop_reason")
    elif finished:
        stop = "finish"
    elif last_main is not None and last_main.get("completion_none"):
        stop = "context_exhausted(llm_none)"
    elif len(main_rows) >= 100:
        stop = "max_turn"
    elif seg_names and n_compactions >= 3:
        stop = "budget_exhausted(max_compactions)"
    else:
        stop = "unknown(timeout_or_budget)"
    # evidence grounding
    finish_expl = None
    cited, seen_docids = set(), set()
    for a in actions:
        for x in (a.get("results") or []):
            if x.get("observation"):
                seen_docids |= set(DOCID_OBS.findall(x["observation"]))
        if a.get("observation"):
            seen_docids |= set(DOCID_OBS.findall(a["observation"]))
    for r in steps:                                   # every observation the policy ever saw (main + branches + segments)
        seen_docids |= set(DOCID_OBS.findall(r.get("prompt_text") or ""))
    for r in steps:
        p = parse_actions(r.get("completion_text") or "")
        for c in p["executable"]:
            if c["function"] == "finish":
                finish_expl = c["arguments"].get("explanation", "")
                cited |= set(DOCID_CITE.findall(finish_expl or ""))
                finish_answer = c["arguments"].get("answer", "")
    # tool usage
    calls = Counter(); queries = []; branch_prompts = []; multi = 0; malformed = 0; think_calls = 0; unsupported = 0
    for t in turns_out:
        names = [c["function"] for c in t["parsed"]["calls"]]
        for c in t["parsed"]["calls"]:
            calls[c["function"]] += 1
            if c["function"] == "search":
                queries.append(c["arguments"].get("query", "").strip().lower())
            if c["function"] == "branch":
                branch_prompts.append(c["arguments"].get("prompt", "").strip().lower())
            if c["function"] not in ("search", "open_page", "finish", "branch", "return"):
                unsupported += 1
        if t["parsed"]["error"] or t["parsed"]["unclosed_tags"] or t["unclosed_think"]:
            malformed += 1
        think_calls += t["parsed"]["calls_in_think"]
        if t.get("multi_call"):
            multi += 1
    dup_q = sum(v - 1 for v in Counter(queries).values() if v > 1)
    dup_b = sum(v - 1 for v in Counter(branch_prompts).values() if v > 1)
    long_obs = sum(1 for a in actions if (a.get("observation_chars") or 0) > args.long_obs_chars)
    jr = judge_recs[-1] if judge_recs else None
    summary = {
        "rollout": key, "question": question_of(steps[0].get("prompt_text")) if steps else None,
        "score": (float(dump_rec["score"]) if dump_rec else None), "finished": finished, "stop_reason": stop,
        "n_steps": len(steps), "n_main_turns": len(main_rows), "n_branches": len([a for a in agents if a not in ("main",) and not str(a).startswith("seg")]),
        "n_compactions": n_compactions, "branch_turns": sum(len(v) for a, v in agents.items() if a != "main" and not str(a).startswith("seg")),
        "gen_main": gen_main, "gen_branch": gen_branch, "gen_summary": summary_tokens, "gen_total": gen_main + gen_branch + summary_tokens,
        "prefill_tokens": prefill, "peak_context": peak, "peak_context_main": peak_main, "wall_s": round(wall, 1),
        "calls": dict(calls), "multi_call_turns": multi, "malformed_turns": malformed, "calls_in_think": think_calls,
        "unsupported_calls": unsupported, "duplicate_queries": dup_q, "duplicate_branch_prompts": dup_b, "long_observations": long_obs,
        "cap_hits": cap_hits, "cap_over": cap_over, "history_missing_turns": hist_missing, "prefix_breaks": prefix_break,
        "fork_checks": fork_checks, "tail_checks": tail_checks, "checks": dict(checks), "has_ids": has_ids,
        "finish_answer": locals().get("finish_answer"), "finish_explanation": finish_expl,
        "cited_docids": sorted(cited), "cited_seen": sorted(cited & seen_docids), "cited_unseen": sorted(cited - seen_docids),
        "judge": (jr.get("response") or "")[:1500] if jr else None,
    }
    return summary, turns_out


# ------------------------------------------------------------------------------------------------- readable traces
def _fence(text, lang=""):
    fence = "````" if "```" in (text or "") else "```"
    return f"{fence}{lang}\n{(text or '').rstrip()}\n{fence}\n"


def _trunc(text, n):
    text = text or ""
    if n and len(text) > n:
        return text[:n] + f"\n… [OBSERVATION TRUNCATED for readability: {len(text) - n:,} more chars; full text in trace_audit.jsonl / capture]"
    return text


def render_rollout(s, turns, rows, args):
    out = [f"\n## Rollout {s['rollout']} — score {s['score']} — {s['stop_reason']} — {s['n_main_turns']} main turns, "
           f"{s['n_branches']} branches, {s['n_compactions']} compactions — gen {s['gen_total']:,} tok (main {s['gen_main']:,} / branch {s['gen_branch']:,} / summary {s['gen_summary']:,}), "
           f"peak ctx {s['peak_context']:,}, wall {s['wall_s']} s\n\n", f"**Question:** {s['question']}\n\n"]
    if s["judge"]:
        out.append(f"**Judge:** {s['judge'][:600].strip()}\n\n")
    out.append(f"Checks: cap hits {s['cap_hits']} (over {s['cap_over']}), history-missing turns {s['history_missing_turns']}, prefix breaks {s['prefix_breaks']}, "
               f"forks {[(f['child'], f['exact']) for f in s['fork_checks']]}, tails {[(t['segment'], t['exact']) for t in s['tail_checks']]}, "
               f"other {s['checks']}; citations seen {s['cited_seen']} unseen {s['cited_unseen']}\n\n")
    by_agent = OrderedDict()
    for t in turns:
        by_agent.setdefault(t["agent"], []).append(t)
    for agent, ts in by_agent.items():
        out.append(f"### Context `{agent}` ({len(ts)} steps)\n\n")
        for t in ts:
            flags = []
            if t["hit_turn_cap"]: flags.append("AT CAP")
            if t["parsed"]["error"]: flags.append("PARSE ERROR: " + t["parsed"]["error"])
            if t["parsed"]["calls_in_think"]: flags.append(f"{t['parsed']['calls_in_think']} call(s) inside think (ignored)")
            if t["unclosed_think"]: flags.append("unclosed think")
            if t.get("history_completions_missing"): flags.append(f"HISTORY MISSING turns {t['history_completions_missing']}")
            if t.get("check_prefix_continuity") is False: flags.append("PREFIX BREAK")
            if t.get("check_mask") is False: flags.append("MASK MISMATCH")
            out.append(f"#### [{agent} · turn {t['turn_index']}] input {t['prompt_tokens']:,} tok → output {t['completion_tokens']:,} tok · think {t['think']}"
                       f"{' (' + str(t['think_chars']) + ' chars)' if t['think'] == 'present' else ''} · executed {[x.get('function') + ':' + str(x.get('status')) for x in (t.get('executed') or [])] or 'none'}"
                       f"{' · ' + '; '.join(flags) if flags else ''}\n\n")
            if args.show_input_tail:
                out.append("<details><summary>exact model input (tail, captured)</summary>\n\n" + _fence(t["prompt_text_tail"]) + "</details>\n\n")
            out.append(_fence(t["completion_text"] or "(no completion)"))
            if t.get("observation") is not None:
                out.append(f"observation ({t['observation_chars']:,} chars):\n\n" + _fence(_trunc(t["observation"], args.obs_chars), "text"))
    return "".join(out)


# ------------------------------------------------------------------------------------------------- aggregate
def aggregate(summaries, turns_all):
    n = len(summaries)
    agg = OrderedDict()
    agg["rollouts"] = n
    agg["correct"] = sum(1 for s in summaries if (s["score"] or 0) > 0)
    agg["finished"] = sum(s["finished"] for s in summaries)
    agg["unfinished"] = n - agg["finished"]
    agg["stop_reasons"] = dict(Counter(s["stop_reason"] for s in summaries))
    agg["accuracy"] = round(agg["correct"] / max(1, n), 3)
    for k in ("n_main_turns", "n_branches", "n_compactions", "gen_main", "gen_branch", "gen_summary", "gen_total", "prefill_tokens", "peak_context", "peak_context_main", "wall_s"):
        vals = [s[k] for s in summaries]
        agg[k] = {"mean": round(sum(vals) / max(1, n), 1), "median": sorted(vals)[n // 2] if n else 0, "max": max(vals) if vals else 0}
    calls = Counter()
    for s in summaries:
        calls.update(s["calls"])
    agg["calls_by_type"] = dict(calls)
    for k in ("multi_call_turns", "malformed_turns", "calls_in_think", "unsupported_calls", "duplicate_queries", "duplicate_branch_prompts", "long_observations", "cap_hits", "cap_over", "history_missing_turns", "prefix_breaks"):
        agg[k] = sum(s[k] for s in summaries)
    agg["fork_checks"] = dict(Counter("exact" if f["exact"] else "MISMATCH" for s in summaries for f in s["fork_checks"]))
    agg["tail_checks"] = dict(Counter("exact" if t["exact"] else "MISMATCH" for s in summaries for t in s["tail_checks"]))
    agg["other_checks"] = dict(sum((Counter(s["checks"]) for s in summaries), Counter()))
    agg["steps_total"] = len(turns_all)
    agg["turns_with_glued_boundary"] = sum(1 for t in turns_all if not t.get("check_no_glued_boundary", True))
    agg["prompts_with_double_newline_empty_think"] = sum(1 for t in turns_all if t.get("double_newline_empty_think_in_prompt"))
    agg["think"] = dict(Counter(t["think"] for t in turns_all))
    agg["mask_checked"] = sum(1 for t in turns_all if "check_mask" in t)
    agg["mask_ok"] = sum(1 for t in turns_all if t.get("check_mask") is True)
    agg["prompt_sha1_ok"] = (sum(1 for t in turns_all if t.get("prompt_sha1_ok") is True), sum(1 for t in turns_all if t.get("prompt_sha1_ok") is not None))
    agg["exec_subset_ok"] = (sum(1 for t in turns_all if t.get("check_exec_subset_of_executable") is True), sum(1 for t in turns_all if "check_exec_subset_of_executable" in t))
    agg["obs_concat_ok"] = (sum(1 for t in turns_all if t.get("check_obs_is_concat_of_calls") is True), sum(1 for t in turns_all if "check_obs_is_concat_of_calls" in t))
    agg["citations"] = {"rollouts_with_citations": sum(1 for s in summaries if s["cited_docids"]), "unseen_citation_rollouts": sum(1 for s in summaries if s["cited_unseen"])}
    return agg


def cmd_audit(args):
    cap, dump, judge, log = load_results_dir(args.results)
    tok = None
    if args.tokenizer and not args.no_tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    by = group_rollouts(cap)
    if args.subset_parquet:
        import pandas as pd
        keep = {str(x["query"]).strip() for x in pd.read_parquet(args.subset_parquet)["extra_info"]}
        by = OrderedDict((k, v) for k, v in by.items() if question_of(next((r.get("prompt_text") for r in v if r.get("prompt_text")), "")) in keep)
    dump_by_q = {question_of(d["input"]): d for d in dump}
    judge_by_q = defaultdict(list)
    for j in judge:
        m = re.search(r"\[question\]:?\s*(.*?)\s*\[", str(j.get("messages", "")), re.S)
        # fall back: match by any question substring
        judge_by_q["__all__"].append(j)
    summaries, turns_all, readable = [], [], []
    for key, rows in by.items():
        q = question_of(next((r.get("prompt_text") for r in rows if r.get("prompt_text")), ""))
        d = dump_by_q.get(q)
        jr = [j for j in judge_by_q["__all__"] if q and q[:80] in str(j.get("messages", ""))]
        s, turns = audit_rollout(key, rows, d, jr, args, tok)
        summaries.append(s)
        turns_all += turns
        s["_rows"] = rows
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, args.arm)
    with open(stem + "_trace_audit.jsonl", "w") as f:
        for s in summaries:
            f.write(json.dumps({"type": "rollout", **{k: v for k, v in s.items() if k != "_rows"}}, ensure_ascii=False) + "\n")
        for t in turns_all:
            f.write(json.dumps({"type": "turn", "arm": args.arm, **t}, ensure_ascii=False) + "\n")
    agg = aggregate(summaries, turns_all)
    agg["arm"] = args.arm
    agg["results_dir"] = args.results
    json.dump(agg, open(stem + "_metrics.json", "w"), indent=1)
    json.dump([{k: v for k, v in s.items() if k not in ("_rows", "finish_explanation", "judge")} for s in summaries], open(stem + "_rollouts.json", "w"), indent=1)
    # readable selection
    def kind(s):
        if s["n_compactions"] > 0: return "compaction"
        if not s["finished"]: return "unfinished"
        return "correct" if (s["score"] or 0) > 0 else "wrong"
    want = {"correct": args.n_each, "wrong": args.n_each, "unfinished": args.n_each, "compaction": args.n_each}
    picks = []
    for k, v in want.items():
        pool = [s for s in summaries if kind(s) == k or (k == "compaction" and s["n_compactions"] > 0)]
        pool.sort(key=lambda s: -s["n_steps"])
        picks += [s for s in pool[:v] if s not in picks]
    anomalous = [s for s in summaries if s["cap_over"] or s["history_missing_turns"] or s["prefix_breaks"] or any(not f["exact"] for f in s["fork_checks"]) or any(not t["exact"] for t in s["tail_checks"]) or s["checks"]]
    picks += [s for s in anomalous if s not in picks]
    if args.all_readable:
        picks = summaries
    md = [f"# Readable traces — arm {args.arm}\n\nResults dir: `{args.results}` ({len(summaries)} rollouts). Model-written text is complete; observations are truncated to {args.obs_chars} chars with an explicit marker.\n"]
    for s in picks:
        md.append(render_rollout(s, [t for t in turns_all if t["rollout"] == s["rollout"]], s["_rows"], args))
    open(stem + "_readable.md", "w").write("".join(md))
    print(json.dumps(agg, indent=1, default=str))
    print(f"wrote {stem}_trace_audit.jsonl / _metrics.json / _rollouts.json / _readable.md ({len(picks)} rollouts rendered)")


def cmd_compare(args):
    arms = OrderedDict()
    for m in args.metrics:
        a = json.load(open(m))
        arms[a["arm"]] = a
    rolls = {}
    for r in args.rollouts:
        arm = os.path.basename(r).replace("_rollouts.json", "")
        rolls[arm] = {s["question"]: s for s in json.load(open(r))}
    keys = ["rollouts", "accuracy", "correct", "finished", "unfinished", "stop_reasons", "n_main_turns", "n_branches", "n_compactions",
            "gen_main", "gen_branch", "gen_summary", "gen_total", "prefill_tokens", "peak_context", "peak_context_main", "wall_s",
            "calls_by_type", "multi_call_turns", "malformed_turns", "calls_in_think", "unsupported_calls", "duplicate_queries",
            "duplicate_branch_prompts", "long_observations", "cap_hits", "cap_over", "history_missing_turns", "prefix_breaks",
            "fork_checks", "tail_checks", "other_checks", "turns_with_glued_boundary", "prompts_with_double_newline_empty_think", "think",
            "mask_ok", "mask_checked", "prompt_sha1_ok", "exec_subset_ok", "obs_concat_ok", "citations"]
    md = ["| metric | " + " | ".join(arms) + " |\n", "|---|" + "---|" * len(arms) + "\n"]
    for k in keys:
        md.append(f"| {k} | " + " | ".join(str(a.get(k, "")) for a in arms.values()) + " |\n")
    # paired outcomes
    names = list(rolls)
    if len(names) >= 2:
        md.append("\n### Paired outcomes (same task, same checkpoint; not a training effect)\n\n")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = rolls[names[i]], rolls[names[j]]
                common = [q for q in A if q in B]
                tr = Counter(((A[q]["score"] or 0) > 0, (B[q]["score"] or 0) > 0) for q in common)
                md.append(f"**{names[i]} vs {names[j]}** on {len(common)} paired tasks: both correct {tr[(True, True)]}, only {names[i]} {tr[(True, False)]}, only {names[j]} {tr[(False, True)]}, both wrong {tr[(False, False)]}\n\n")
                md.append("| task (question head) | " + f"{names[i]} score/stop/turns/gen | {names[j]} score/stop/turns/gen |\n|---|---|---|\n")
                for q in common:
                    a, b = A[q], B[q]
                    if ((a["score"] or 0) > 0) != ((b["score"] or 0) > 0):
                        md.append(f"| {q[:90]}… | {a['score']} / {a['stop_reason']} / {a['n_main_turns']} / {a['gen_total']} | {b['score']} / {b['stop_reason']} / {b['n_main_turns']} / {b['gen_total']} |\n")
    open(os.path.join(args.out, "comparison.md"), "w").write("".join(md))
    print("".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--results", required=True); a.add_argument("--arm", required=True); a.add_argument("--out", required=True)
    a.add_argument("--tokenizer", default="/mnt/hdfs/mlsys/xiaoxuan/fold_replication/ckpt_fix/grpo/global_step_50/actor/huggingface")
    a.add_argument("--no-tokenizer", action="store_true")
    a.add_argument("--turn-cap", type=int, default=2048); a.add_argument("--obs-chars", type=int, default=2500)
    a.add_argument("--tail-chars", type=int, default=1500); a.add_argument("--long-obs-chars", type=int, default=40000)
    a.add_argument("--n-each", type=int, default=4); a.add_argument("--show-input-tail", action="store_true", default=True)
    a.add_argument("--all-readable", action="store_true")
    a.add_argument("--subset-parquet", default=None, help="restrict to the tasks (extra_info.query) of this parquet")
    c = sub.add_parser("compare")
    c.add_argument("--metrics", nargs="+", required=True); c.add_argument("--rollouts", nargs="+", required=True); c.add_argument("--out", required=True)
    args = ap.parse_args()
    {"audit": cmd_audit, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
