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


def extract_fn_calls_strict(text: Optional[str], max_line_gap: int = 4) -> list:
    """The parser the environment executes (moved here from envs/local_search.py so the agents use the SAME one).

    Rules: a call must start at the beginning of a line and be closed (``</function>``); calls separated by fewer than
    ``max_line_gap`` newlines form a group; only the LAST group is returned and every call in it is executed, in order.
    Calls inside ``<think>`` blocks that happen to be line-anchored are accepted here exactly as the environment accepts
    them. Returns ``[{'function', 'arguments'}, ...]`` (possibly empty).
    """
    if not text:
        return []
    text = _ROLE_MARK.split(text)[-1].strip()
    matches = list(STRICT_FN.finditer(text))
    if not matches:
        return []
    groups = [[matches[0]]]
    for m in matches[1:]:
        prev = groups[-1][-1]
        line_gap = text.count("\n", prev.end(), m.start())
        groups[-1].append(m) if line_gap < max_line_gap else groups.append([m])
    return [{"function": m.group(1), "arguments": dict(PARAM.findall(m.group(2)))} for m in groups[-1]]


def last_strict_call(text: Optional[str]) -> Optional[dict]:
    """Last call of the executed group (what the environment acts on last), or None."""
    calls = extract_fn_calls_strict(text)
    return calls[-1] if calls else None


def extract_summary(text: Optional[str]) -> Optional[str]:
    matches = SUMMARY.findall(text or "")
    return matches[-1].strip() if matches else None
