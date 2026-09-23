"""Text-side parsing of policy responses, shared by the agents, the ledger and the analysis scripts.

One home for every regex that looks at generated text, so a model-family change (Qwen3 -> Qwen3.5) is made here once:

* ``find_think`` / ``strip_think`` accept a **missing opening ``<think>`` tag**. Qwen3.5's chat template pre-fills
  ``<think>\\n`` in the generation prompt, so the sampled text starts mid-thought and only contains ``</think>``;
  Qwen3 samples contain both tags. Both forms are handled identically.
* ``extract_fn_call`` tolerates the ``<tool_call>…</tool_call>`` wrapper of Qwen3.5's native tool format (the prompt
  asks for bare ``<function=…>`` blocks, but the model may emit the wrapper spontaneously).
* ``extract_summary`` returns the last ``<summary>…</summary>`` block (compaction / fold summaries).
"""
import re
from typing import Optional

THINK_TAGGED = re.compile(r"<think>(.*?)</think>", re.S)
THINK_PREFILLED = re.compile(r"\A\s*(.*?)</think>", re.S)      # opener lives in the generation prompt (Qwen3.5)
FN_NAME = re.compile(r"<function=([^>]+)>")
FN_BLOCK = re.compile(r"<function=([^>]+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.S)
SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.S)
TOOL_CALL_WRAP = re.compile(r"</?tool_call>")


def find_think(text: Optional[str]):
    """Return ``(reasoning, span)`` for the first think block, or ``(None, None)`` when the text has no ``</think>``.

    ``<think>…</think>`` is preferred; a bare leading ``…</think>`` (pre-filled opener) is accepted too.
    ``reasoning`` is stripped; an empty string means an empty think block.
    """
    if not text:
        return None, None
    m = THINK_TAGGED.search(text)
    if m is None:
        m = THINK_PREFILLED.search(text)
    if m is None:
        return None, None
    return m.group(1).strip(), m.span()


def has_think(text: Optional[str]) -> bool:
    return find_think(text)[0] is not None


def think_is_empty(text: Optional[str]) -> bool:
    reasoning, _ = find_think(text)
    return reasoning is not None and reasoning == ""


def strip_think(text: Optional[str]) -> str:
    """Remove the (first) think block, tagged or pre-filled, and strip whitespace."""
    _, span = find_think(text)
    if span is None:
        return (text or "").strip()
    return (text[: span[0]] + text[span[1]:]).strip()


def extract_fn_call(text: Optional[str]):
    """Last ``<function=NAME>…</function>`` call as ``{'function': NAME, 'arguments': {param: value}}`` (or None)."""
    if text is None:
        return None
    text = TOOL_CALL_WRAP.sub("", text)
    func_matches = FN_NAME.findall(text)
    if not func_matches:
        return None
    last_function = func_matches[-1]
    last_func_pos = text.rfind(f"<function={last_function}>")
    text_after_last_func = text[last_func_pos:]
    params = dict(PARAM.findall(text_after_last_func))
    return {"function": last_function, "arguments": params}


def fn_call_names(text: Optional[str]) -> list:
    return FN_NAME.findall(text or "")


STRICT_FN = re.compile(r"(?m)^[ \t]*<function=([^>]+)>\s*(.*?)\s*</function>", re.S)
_ROLE_MARK = re.compile(r"<\[[^\]]+\]>")
TERMINAL_CALLS = ("finish", "return")     # end the (main | branch) episode; must be the only call of the turn
SINGLETON_CALLS = ("branch",)             # the prompt says "Branch one task at a time": one branch, alone, per turn


def strip_all_think(text: str) -> str:
    """Remove every think block (tagged, or with the opener pre-filled by the template) from ``text``."""
    out, guard = text, 0
    while guard < 32:
        reasoning, span = find_think(out)
        if span is None:
            break
        out = out[: span[0]] + out[span[1]:]
        guard += 1
    return out


MAX_CALLS_PER_TURN_DEFAULT = 8


def parse_actions(text: Optional[str], turn_id: Optional[str] = None, max_calls: Optional[int] = None) -> dict:
    """The ONE action grammar shared by the agents and the environment (2026-09-23).

    * think blocks are removed first — nothing inside ``<think>…</think>`` is ever an action (no adjacency edge case);
    * a call must start a line and be closed (``<function=NAME> … </function>``); every such call in the remaining text is
      parsed **in order** and gets a ``call_id`` (``<turn_id>.<k>``, 1-based);
    * ``search`` / ``open_page`` may be repeated in one turn (the prompt invites "multiple <function=search> actions");
    * ``finish`` / ``return`` are terminal and must be the only call of the turn; ``branch`` must be the only call of the
      turn (the prompt says "Branch one task at a time"). Violations make the whole turn non-executable with ``error`` set
      (the environment returns ``[Error] …`` and executes nothing) — the calls are still listed for diagnostics.

    Returns ``{'calls': [...], 'executable': [...], 'error': str|None, 'calls_in_think': int, 'unclosed_tags': int}``.
    """
    body = text or ""
    n_think_calls = 0
    if body:
        stripped = strip_all_think(body)
        n_think_calls = len(FN_NAME.findall(body)) - len(FN_NAME.findall(stripped))
        body = _ROLE_MARK.split(stripped)[-1].strip()
    calls = []
    for k, m in enumerate(STRICT_FN.finditer(body), 1):
        calls.append({"call_id": f"{turn_id}.{k}" if turn_id else f"c{k}", "function": m.group(1),
                      "arguments": dict(PARAM.findall(m.group(2))), "span": m.span()})
    names = [c["function"] for c in calls]
    error = None
    if any(n in TERMINAL_CALLS for n in names) and len(names) > 1:
        error = f"a terminal call ({'/'.join(n for n in names if n in TERMINAL_CALLS)}) must be the only function call in the turn; nothing was executed"
    elif names.count("branch") > 1:
        error = "only one branch call is allowed per turn; nothing was executed"
    elif "branch" in names and len(names) > 1:
        error = "a branch call must be the only function call in the turn; nothing was executed"
    truncated_calls = 0
    limit = MAX_CALLS_PER_TURN_DEFAULT if max_calls is None else max_calls
    if not error and limit and len(calls) > limit:
        # full 150-task run: a single-window policy emitted up to 92 open_page calls in one turn (930k-char observation,
        # 234k-token prompt). Execute the first `limit` calls; the rest are reported, not executed.
        truncated_calls = len(calls) - limit
        error = None
    unclosed = max(0, len(FN_NAME.findall(body)) - len(calls))
    # A line-anchored terminal opener (finish/return) with no closing tag after the last closed call: not executable, but
    # the harness may treat it as a malformed terminal attempt (the checkpoint trained under the old lenient harness
    # closes `return` with `<function>` in ~30% of branch turns) instead of looping on "No function call was detected".
    unclosed_terminal, unclosed_terminal_args = None, {}
    openers = list(re.finditer(r"(?m)^[ \t]*<function=([^>\s]+)>", body))
    last_closed_end = calls[-1]["span"][1] if calls else -1
    if openers and openers[-1].start() >= last_closed_end and openers[-1].group(1) in TERMINAL_CALLS:
        unclosed_terminal = openers[-1].group(1)
        unclosed_terminal_args = dict(PARAM.findall(body[openers[-1].end():]))
    executable = [] if error else (calls[:limit] if (limit and len(calls) > limit) else calls)
    return {"calls": calls, "executable": executable, "error": error, "truncated_calls": truncated_calls,
            "calls_in_think": n_think_calls, "unclosed_tags": unclosed,
            "unclosed_terminal": unclosed_terminal, "unclosed_terminal_args": unclosed_terminal_args}


def extract_fn_calls_strict(text: Optional[str]) -> list:
    """Executable calls of a turn under the shared grammar (empty when the turn is malformed). Kept for callers/tests."""
    return parse_actions(text)["executable"]


def last_strict_call(text: Optional[str]) -> Optional[dict]:
    """Last executable call of the turn, or None."""
    calls = extract_fn_calls_strict(text)
    return calls[-1] if calls else None


def extract_summary(text: Optional[str]) -> Optional[str]:
    matches = SUMMARY.findall(text or "")
    return matches[-1].strip() if matches else None
