#!/usr/bin/env python3
"""Render full rollout traces from the trainer's JSONL dumps as Markdown, with a per-rollout sanity checklist.

Reads either kind of dump written by the training stack (see infra/jobs/fold_train_entrypoint.sh):

* validation dumps  ``val_dump_fix/<arm>/<step>.jsonl``  (``trainer.validation_data_dir``): one record per task.
  For the compaction arms the record holds the *last* context segment only (eval returns the last segment),
  so a compacted rollout starts with the resume template + the policy-written summary.
* training-rollout dumps ``rollout_dump_fix/<tag>/<step>.jsonl`` (``trainer.rollout_data_dir``, ``FOLD_ROLLOUT_DUMP=1``):
  one record per *training sample* = one context segment. Segments of one rollout are chained here by matching
  the summary a segment wrote against the summary the next segment resumes from (same task prompt), so the
  rendered trace is the complete multi-segment rollout.

Each record has ``input`` (decoded prompt ids: system + task prompt + generation prompt) and ``output`` (decoded
response ids: everything the policy generated or was re-rendered after the prompt), special tokens stripped, so
turn boundaries appear as lines ``assistant`` / ``user``.

Usage::

  python scripts/render_rollout_traces.py --dump <jsonl> --out <md> [--select kind:n,...] [--obs-chars 3000]
                                          [--tokenizer Qwen/Qwen3-8B] [--response-length 32768] [--threshold 8192]

``--select`` picks rollouts by kind (``single_correct``, ``single_wrong``, ``compacted_correct``, ``compacted_wrong``,
``exhausted``, ``other``). ``--obs-chars 0`` keeps tool observations in full. Model-generated text is never truncated.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.parsing import find_think, strip_think  # noqa: E402
from agents.prompts import COMPACTION_RESUME_TEMPLATE, COMPACTION_SUMMARY_PROMPT  # noqa: E402

TURN_SPLIT = re.compile(r"(?m)^(assistant|user)\n")
FN = re.compile(r"<function=([^>]+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.S)
SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.S)
RESUME_HEAD = COMPACTION_RESUME_TEMPLATE.split("<summary>")[0].strip()
QSUM_HEAD = COMPACTION_SUMMARY_PROMPT.split("\n")[0]
QUESTION = re.compile(r"Question: (.*?)\n\nYour response should contain", re.S)
SPECIAL = re.compile(r"<\|im_(start|end)\|>|<\|endoftext\|>")


# ----------------------------------------------------------------------------------------------- parsing
TURN_START = re.compile(
    r"(?:^|(?<=\n))(assistant|user)\n"                      # role marker on its own line
    r"|(user)\n(?=<tool_response>|" + re.escape(QSUM_HEAD[:30]) + r"|" + re.escape(RESUME_HEAD[:30]) + r")"
    r"|(assistant)\n(?=<think>)",                            # marker glued to the previous turn's last token
    re.M)


def split_turns(text):
    """Split decoded segment text into [(role, content)].

    Special tokens are stripped in the dumps, so a turn boundary is ``<|im_start|>ROLE\n`` rendered as ``ROLE\n``.
    When the previous assistant turn ended with its sampled ``<|im_end|>`` (no trailing newline) the marker is glued
    to the previous text (``</function>user\n<tool_response>``), hence the look-ahead alternatives.
    """
    bounds = [(m.start(), m.end(), next(g for g in m.groups() if g)) for m in TURN_START.finditer(text)]
    turns, notes = [], []
    lead = text[: bounds[0][0]] if bounds else text
    if lead.strip():
        turns.append(["assistant", lead])   # response text before any role marker (should not happen)
        notes.append("text before first role marker")
    for k, (b, e, role) in enumerate(bounds):
        content = text[e: bounds[k + 1][0] if k + 1 < len(bounds) else len(text)]
        real = True
        if role == "user":
            c = content.lstrip()
            real = c.startswith("<tool_response>") or c.startswith(QSUM_HEAD) or c.startswith(RESUME_HEAD[:40])
        if not real and turns:
            turns[-1][1] += text[b:e] + content   # a literal 'user'/'assistant' line inside content
            notes.append("suspicious role line merged")
            continue
        turns.append([role, content])
    return [(r, c) for r, c in turns], notes


def classify_user(content):
    c = content.strip()
    if c.startswith("<tool_response>"):
        return "observation"
    if c.startswith(QSUM_HEAD):
        return "summary_prompt"
    if c.startswith(RESUME_HEAD[:40]):
        return "resume"
    return "user_other"


def parse_assistant(content):
    reasoning, span = find_think(content)
    calls = FN.findall(content)
    d = {
        "has_think": reasoning is not None,
        "think": reasoning or "",
        "think_span": span,
        "think_empty": reasoning is not None and not reasoning,
        "calls": [(name, dict(PARAM.findall(body))) for name, body in calls],
        "summary": None,
    }
    sm = SUMMARY.findall(content)
    if sm:
        d["summary"] = sm[-1].strip()
    after = content.rsplit("</function>", 1)[-1] if calls else ""
    d["suffix_after_call"] = after.strip() if calls else ""
    d["unclosed_call"] = ("<function=" in content) and not calls
    return d


def parse_segment(text):
    """Structured view of one segment's decoded response text."""
    turns, notes = split_turns(text)
    seg = {"turns": [], "notes": notes, "resume_summary": None, "written_summary": None,
           "tail": [], "finished": False, "n_calls": Counter(), "obs_expected": 0, "obs_intact": 0,
           "think_total": 0, "think_empty": 0, "no_think": 0, "special_tokens": len(SPECIAL.findall(text))}
    pending_obs = False
    after_qsum = False
    for role, content in turns:
        if role == "user":
            kind = classify_user(content)
            t = {"role": "user", "kind": kind, "content": content}
            if kind == "observation":
                t["intact"] = content.strip().endswith("</tool_response>")
                if pending_obs:
                    seg["obs_expected"] += 1
                    seg["obs_intact"] += int(t["intact"])
                    pending_obs = False
                else:
                    seg["notes"].append("observation without preceding tool call")
            elif kind == "resume":
                sm = SUMMARY.findall(content)
                seg["resume_summary"] = sm[-1].strip() if sm else None
                if not sm:
                    seg["notes"].append("resume turn without <summary> block")
            elif kind == "summary_prompt":
                if pending_obs:
                    seg["notes"].append("summary prompt issued while an observation was pending")
                    pending_obs = False
                after_qsum = True
            seg["turns"].append(t)
            continue
        a = parse_assistant(content)
        a.update({"role": "assistant", "content": content, "kind": "step"})
        if after_qsum:
            a["kind"] = "summary"
            if a["summary"] is None:      # agent fallback: the think-stripped response becomes the summary
                seg["notes"].append("summary turn without <summary> block (agent used the think-stripped response)")
                a["summary"] = strip_think(content)
            seg["written_summary"] = a["summary"]
            after_qsum = False
        seg["think_total"] += int(a["has_think"])
        seg["think_empty"] += int(a["think_empty"])
        seg["no_think"] += int(not a["has_think"])
        names = [n for n, _ in a["calls"]]
        for n in names:
            seg["n_calls"][n] += 1
        if "finish" in names:
            seg["finished"] = True
        elif names:
            pending_obs = True
        seg["turns"].append(a)
    _fix_tail_kinds(seg)   # a resumed segment starts with k re-rendered (assistant, observation) pairs
    if pending_obs:
        seg["notes"].append("last tool call has no observation (segment ended at the budget or a cut)")
    return seg


def _fix_tail_kinds(seg):
    """A resumed segment starts: resume, then k x (assistant, observation). Everything after is new steps."""
    turns = seg["turns"]
    if not turns or turns[0].get("kind") != "resume":
        for t in turns:
            if t["role"] == "assistant" and t["kind"] == "tail":
                t["kind"] = "step"
        seg["tail"] = []
        return
    i, tail = 1, []
    while i + 1 < len(turns) and turns[i]["role"] == "assistant" and turns[i + 1]["role"] == "user" \
            and turns[i + 1]["kind"] == "observation":
        # heuristic: a tail assistant turn must call a tool and is immediately followed by its observation.
        # We cannot distinguish the last tail pair from the first new step by text alone; the training dump
        # chain check (`tail_verbatim`) resolves it against the previous segment. Bound by 2 (config default).
        if len(tail) >= 2:
            break
        turns[i]["kind"] = "tail"
        tail.append(turns[i]["content"])
        i += 2
    for j in range(i, len(turns)):
        if turns[j]["role"] == "assistant" and turns[j]["kind"] == "tail":
            turns[j]["kind"] = "step"
    seg["tail"] = tail


def question_of(prompt_text):
    m = QUESTION.search(prompt_text)
    return m.group(1).strip() if m else prompt_text[-400:].strip()


# ----------------------------------------------------------------------------------------------- chaining
def load(path):
    """Load a dump; drop exact duplicate records (verl pads the training batch to the world size by repetition)."""
    seen, out = set(), []
    for l in open(path):
        r = json.loads(l)
        key = (r["input"], r["output"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def build_rollouts(records, tail_steps):
    """Group records into rollouts. Val dump: one record = one rollout (last segment). Train dump: chain segments."""
    parsed = []
    for k, r in enumerate(records):
        seg = parse_segment(r["output"])
        seg["idx"], seg["score"], seg["question"], seg["prompt"] = k, float(r["score"]), question_of(r["input"]), r["input"]
        parsed.append(seg)
    firsts = [s for s in parsed if s["resume_summary"] is None]
    by_written = {}
    for s in parsed:
        if s["written_summary"]:
            by_written.setdefault((s["question"], s["written_summary"]), []).append(s)
    used = set()
    rollouts = []
    for s in firsts:
        chain = [s]
        used.add(s["idx"])
        cur = s
        while cur["written_summary"]:
            cands = [c for c in parsed if c["idx"] not in used and c["question"] == cur["question"]
                     and c["resume_summary"] == cur["written_summary"]]
            if not cands:
                chain[-1]["notes"].append("wrote a summary but no resumed segment found (chain broken)")
                break
            nxt = cands[0]
            used.add(nxt["idx"])
            # tail verbatim check against the previous segment's assistant turns
            prev_assist = [t["content"] for t in cur["turns"] if t["role"] == "assistant" and t["kind"] != "summary"]
            nxt["tail_verbatim"] = [tc in prev_assist for tc in nxt["tail"]]
            chain.append(nxt)
            cur = nxt
        rollouts.append(chain)
    orphans = [s for s in parsed if s["idx"] not in used]
    for s in orphans:                                  # val dumps: every compacted record is an orphan by design
        rollouts.append([s])
    return rollouts, len(orphans)


def rollout_kind(chain):
    last = chain[-1]
    compacted = len(chain) > 1 or last["resume_summary"] is not None
    if last["finished"]:
        return ("compacted_" if compacted else "single_") + ("correct" if last["score"] > 0 else "wrong")
    if compacted and not last["finished"] and last["written_summary"] is None:
        return "exhausted"
    return "other"


def rollout_stats(chain):
    calls = Counter()
    turns = think = empty = obs_e = obs_i = 0
    for s in chain:
        calls.update(s["n_calls"])
        turns += sum(1 for t in s["turns"] if t["role"] == "assistant" and t["kind"] == "step")
        think += s["think_total"]
        empty += s["think_empty"]
        obs_e += s["obs_expected"]
        obs_i += s["obs_intact"]
    undumped = chain[0]["resume_summary"] is not None      # val dump: earlier segments of this rollout were not dumped
    n_sum = sum(1 for s in chain if s["written_summary"])
    return {"segments": len(chain), "steps": turns, "search": calls["search"], "open_page": calls["open_page"],
            "finish": calls["finish"], "summaries": f"≥{n_sum + 1} (earlier segments not dumped)" if undumped else n_sum,
            "think": think, "empty_think": empty, "obs_expected": obs_e, "obs_intact": obs_i,
            "chars": sum(len(t["content"]) for s in chain for t in s["turns"])}


# ----------------------------------------------------------------------------------------------- rendering
def _trunc(text, n):
    if n and len(text) > n:
        return text[:n] + f"\n… [truncated: {len(text) - n:,} more chars; rerun with --obs-chars 0 for the full observation]"
    return text


def _fence(text, lang=""):
    fence = "````" if "```" in text else "```"
    return f"{fence}{lang}\n{text.rstrip()}\n{fence}\n"


def _quote(text):
    return "\n".join("> " + l for l in text.rstrip().split("\n")) + "\n"


def render_assistant(t, i, tok):
    kind = {"step": "assistant", "tail": "assistant · TAIL (re-rendered from the previous segment, loss mask 0)",
            "summary": "assistant · SUMMARY (policy-written, trainable)"}[t["kind"]]
    out = [f"#### [{i}] {kind}\n"]
    if tok is not None:
        out.append(f"*{len(tok.encode(t['content'], add_special_tokens=False)):,} tokens*\n")
    body = t["content"]
    if t["has_think"]:
        a, b = t["think_span"]
        pre, post = body[:a], body[b:]
        think = t["think"]
        out.append("**think**" + (" *(empty)*" if t["think_empty"] else "") + "\n\n")
        if think.strip():
            out.append(_quote(think) + "\n")
        rest = (pre + post).strip("\n")
    else:
        out.append("**think**: *none — no `<think>` block in this turn*\n\n")
        rest = body.strip("\n")
    if t["kind"] == "summary" and t["summary"] is not None:
        out.append("**summary** (`<summary>` block, becomes the resume context of the next segment)\n\n")
        out.append(_fence(t["summary"], "text"))
        rest = SUMMARY.sub("", rest).strip()
        if rest:
            out.append("text outside the summary block:\n\n" + _fence(rest))
        return "".join(out)
    if t["calls"]:
        for name, params in t["calls"]:
            arg = "\n".join(f"<parameter={k}>{v.strip()}</parameter>" for k, v in params.items())
            out.append(f"**action** `{name}`\n\n" + _fence(f"<function={name}>\n{arg}\n</function>", "xml"))
        pre_call = rest.split("<function=")[0].strip()
        if pre_call:
            out.append("text before the call:\n\n" + _fence(pre_call))
        if t["suffix_after_call"]:
            out.append("⚠ text after the call (format says NO suffix):\n\n" + _fence(t["suffix_after_call"]))
    elif rest:
        out.append(("⚠ unclosed `<function=` call\n\n" if t["unclosed_call"] else "**no tool call**\n\n") + _fence(rest))
    return "".join(out)


def render_user(t, i, obs_chars, tok):
    kind = t["kind"]
    if kind == "observation":
        body = t["content"].strip()
        inner = body[len("<tool_response>"):].strip()
        if inner.endswith("</tool_response>"):
            inner = inner[: -len("</tool_response>")].strip()
        head = f"#### [{i}] tool_response ({len(body):,} chars"
        if tok is not None:
            head += f", {len(tok.encode(t['content'], add_special_tokens=False)):,} tokens"
        head += ")" + ("" if t["intact"] else " ⚠ NOT closed with `</tool_response>`") + "\n\n"
        return head + _fence(_trunc(inner, obs_chars), "text")
    if kind == "summary_prompt":
        exact = t["content"].strip() == COMPACTION_SUMMARY_PROMPT.strip()
        return (f"#### [{i}] user · COMPACTION PROMPT (q_sum, fixed text{'' if exact else ' ⚠ differs from agents/prompts.py'})\n\n"
                "<details><summary>show prompt</summary>\n\n" + _fence(t["content"].strip(), "text") + "</details>\n")
    if kind == "resume":
        body = t["content"].strip()
        sm = SUMMARY.search(body)
        pre = body[: sm.start()].strip() if sm else body
        out = f"#### [{i}] user · RESUME CONTEXT (u_resume, fixed template + the summary below; non-trainable)\n\n" + _quote(pre) + "\n"
        if sm:
            out += "**resumed summary**\n\n" + _fence(sm.group(1).strip(), "text")
        return out
    return f"#### [{i}] user · ⚠ unexpected user turn\n\n" + _fence(_trunc(t["content"], obs_chars))


def checklist(chain, args, tok):
    st = rollout_stats(chain)
    last = chain[-1]
    rows = []

    def row(ok, what, detail=""):
        rows.append(f"| {'✅' if ok else '⚠️'} | {what} | {detail} |")

    cut_last = sum(1 for s in chain for i, t in enumerate(s["turns"])
                   if t["role"] == "user" and t["kind"] == "observation" and not t["intact"] and i == len(s["turns"]) - 1)
    row(st["obs_intact"] == st["obs_expected"], "every tool call got an intact `<tool_response>…</tool_response>` observation",
        f"{st['obs_intact']}/{st['obs_expected']}" + (f"; {cut_last} observation(s) cut at the END of a segment = the dumped sample's response was "
        "clipped to response_length (the observation overflowed the window; mask 0, no loss; the agent rolled that step back before summarising)"
        if cut_last else ""))
    no_think = sum(s["no_think"] for s in chain)
    row(no_think == 0, "every assistant turn carries a think block (thinking mode on; Qwen3.5 pre-fills the opener)", f"missing in {no_think} turn(s)")
    row(True, "empty `<think></think>` turns", f"{st['empty_think']}/{st['think']}")
    row(all(s["special_tokens"] == 0 for s in chain), "no leaked special tokens (`<|im_start|>`, `<|im_end|>`) in decoded text", "")
    allowed = tuple(x.strip() for x in args.tools.split(","))
    bad_calls = [n for s in chain for n in s["n_calls"] if n not in allowed]
    row(not bad_calls, f"only {' / '.join(allowed)} are called",
        f"unexpected: {bad_calls}" if bad_calls else f"search {st['search']}, open_page {st['open_page']}, finish {st['finish']}")
    suffix = sum(1 for s in chain for t in s["turns"] if t["role"] == "assistant" and t.get("suffix_after_call"))
    row(suffix == 0, "no text after a function call (prompt: 'NO suffix')", f"{suffix} turn(s) with a suffix")
    row(last["finished"] or rollout_kind(chain) == "exhausted", "rollout ends with `finish` or with the compaction budget exhausted",
        "finish" if last["finished"] else ("budget exhausted after max compactions" if rollout_kind(chain) == "exhausted" else "⚠ ended without finish (max_turn / timeout / cut)"))
    for i, s in enumerate(chain):
        if s["written_summary"] is not None:
            nxt = chain[i + 1] if i + 1 < len(chain) else None
            if nxt is not None:
                row(nxt["resume_summary"] == s["written_summary"], f"segment {i + 1}: written summary == summary resumed by segment {i + 2}", "")
                tv = nxt.get("tail_verbatim", [])
                row(all(tv) if tv else True, f"segment {i + 2}: {len(nxt['tail'])} tail step(s) re-rendered verbatim from segment {i + 1}",
                    "" if all(tv) else "some tail step is not in the previous segment's trained text → it was rolled back before the summary (expected when q_sum + summary_max_tokens did not fit)")
            else:
                row(False, f"segment {i + 1} wrote a summary but the resumed segment is not in this dump",
                    "expected for validation dumps (only the last segment is dumped)" if args.kind == "val" else "chain broken")
        if s["resume_summary"] is not None and i == 0:
            row(True, "segment 1 of this record is a RESUMED segment (earlier segments were not dumped: validation returns the last segment only)", "")
    if tok is not None:
        for i, s in enumerate(chain):
            n = len(tok.encode("".join(t["content"] for t in s["turns"]), add_special_tokens=False))
            ok = n <= args.response_length
            sum_pos = None
            for j, t in enumerate(s["turns"]):
                if t["role"] == "user" and t["kind"] == "summary_prompt":
                    sum_pos = len(tok.encode("".join(x["content"] for x in s["turns"][:j]), add_special_tokens=False))
            detail = f"{n:,} generated-side tokens (approx., re-tokenised) ≤ response_length {args.response_length:,}"
            if sum_pos is not None:
                detail += f"; compaction triggered at ≈{sum_pos:,} tokens (threshold: remaining < {args.threshold:,} ⇒ ≥ {args.response_length - args.threshold:,} minus rollback)"
            row(ok, f"segment {i + 1} fits the per-segment budget", detail)
    notes = [n for s in chain for n in s["notes"]]
    if notes:
        row(False, "parser notes", "; ".join(sorted(set(notes))))
    return "| | check | detail |\n|---|---|---|\n" + "\n".join(rows) + "\n"


def render_rollout(chain, n, args, tok):
    st = rollout_stats(chain)
    last = chain[-1]
    kind = rollout_kind(chain)
    out = [f"\n## Rollout {n} — {kind.replace('_', ' ')} — score {last['score']:.0f} — {st['segments']} segment(s), "
           f"{st['steps']} step(s), {st['summaries']} compaction(s)\n\n",
           f"**Question:** {chain[0]['question']}\n\n"]
    if args.kind == "val" and last["resume_summary"] is not None:
        out.append("*Validation dump: only the last context segment of this rollout was dumped; it opens with the resume context "
                   "(the policy's own summary of the earlier, undumped segments).*\n\n")
    out.append("### Sanity checklist\n\n" + checklist(chain, args, tok) + "\n")
    for si, s in enumerate(chain):
        label = "first window" if s["resume_summary"] is None else f"resumed (summary + {len(s['tail'])} tail step(s))"
        out.append(f"### Segment {si + 1}/{len(chain)} — {label}\n\n")
        for i, t in enumerate(s["turns"], 1):
            out.append(render_assistant(t, i, tok) if t["role"] == "assistant" else render_user(t, i, args.obs_chars, tok))
            out.append("\n")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=["auto", "val", "train"], default="auto")
    ap.add_argument("--select", default="single_correct:1,single_wrong:1,compacted_correct:2,compacted_wrong:1,exhausted:1,other:1",
                    help="kind:count list; 'all' renders every rollout")
    ap.add_argument("--obs-chars", type=int, default=3000, help="truncate tool observations to this many chars (0 = full)")
    ap.add_argument("--tokenizer", default=os.environ.get("QWEN3_TOKENIZER_PATH", "Qwen/Qwen3-8B"))
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--response-length", type=int, default=32768)
    ap.add_argument("--threshold", type=int, default=8192)
    ap.add_argument("--tail-steps", type=int, default=2)
    ap.add_argument("--title", default=None)
    ap.add_argument("--tools", default="search,open_page,finish", help="tool names the arm may call (checklist)")
    args = ap.parse_args()

    records = load(args.dump)
    if args.kind == "auto":
        # training dumps carry several records per prompt (n=8 samples x segments); validation dumps one per task
        qs = Counter(question_of(r["input"]) for r in records)
        args.kind = "train" if max(qs.values()) > 1 else "val"
    tok = None
    if not args.no_tokenizer:
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:  # pragma: no cover
            print(f"[warn] tokenizer unavailable ({e}); token counts omitted", file=sys.stderr)

    rollouts, n_orphans = build_rollouts(records, args.tail_steps)
    kinds = Counter(rollout_kind(c) for c in rollouts)
    if args.select == "all":
        chosen = rollouts
    else:
        want = OrderedDict((k, int(v)) for k, v in (x.split(":") for x in args.select.split(",")))
        chosen = []
        for k, v in want.items():
            pool = [c for c in rollouts if rollout_kind(c) == k]
            # prefer the most interesting: most segments, then most steps
            pool.sort(key=lambda c: (-len(c), -rollout_stats(c)["steps"]))
            chosen += pool[:v]

    step = records[0].get("step", "?")
    title = args.title or f"Rollout traces — {os.path.basename(os.path.dirname(args.dump))} / step {step} ({args.kind} dump)"
    md = [f"# {title}\n\n",
          f"Source: `{args.dump}` — {len(records)} records, {len(rollouts)} rollouts"
          + (f" ({n_orphans} records are resumed segments whose earlier segments are not in the dump)" if args.kind == "val" and n_orphans else
             f" ({n_orphans} unchained resumed segments)" if n_orphans else "") + ".\n\n",
          "How to read: `input` = system + task prompt (identical for every segment of a task); each **segment** is one training sample "
          "(prompt = task prompt, response = everything after it). A compaction = the fixed `q_sum` prompt → the policy's `<summary>` (trainable) → "
          "a new segment that starts with the fixed resume template carrying that summary plus the last k steps re-rendered (mask 0). "
          "Tool observations are wrapped in `<tool_response>` so Qwen3's chat template keeps earlier `<think>` blocks. "
          f"Observations are truncated to {args.obs_chars or 'no limit'} chars here; model-written text is shown in full.\n\n",
          "## Rollout kinds in this dump\n\n| kind | rollouts | mean score |\n|---|---|---|\n"]
    for k in ["single_correct", "single_wrong", "compacted_correct", "compacted_wrong", "exhausted", "other"]:
        if kinds[k]:
            sc = [c[-1]["score"] for c in rollouts if rollout_kind(c) == k]
            md.append(f"| {k} | {kinds[k]} | {sum(sc) / len(sc):.3f} |\n")
    md.append("\n## Rendered rollouts\n\n| # | kind | score | segments | steps | search | open_page | compactions | obs intact | empty think | chars |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    for n, c in enumerate(chosen, 1):
        st = rollout_stats(c)
        md.append(f"| {n} | {rollout_kind(c)} | {c[-1]['score']:.0f} | {st['segments']} | {st['steps']} | {st['search']} | {st['open_page']} | "
                  f"{st['summaries']} | {st['obs_intact']}/{st['obs_expected']} | {st['empty_think']}/{st['think']} | {st['chars']:,} |\n")
    for n, c in enumerate(chosen, 1):
        md.append(render_rollout(c, n, args, tok))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("".join(md))
    print(f"wrote {args.out}: {len(chosen)} rollouts, {sum(len(x) for x in md):,} chars; kinds in dump: {dict(kinds)}")


if __name__ == "__main__":
    main()
