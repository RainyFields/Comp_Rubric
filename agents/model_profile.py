"""Model-family profile detected from a tokenizer's chat template.

The agents never hard-code a model name; the few behaviours that differ between families are read off the template:

* ``strips_history_think`` — the template drops earlier assistant ``<think>`` blocks before the latest plain user query
  (Qwen3, Qwen3.5). Observations must then be wrapped in ``<tool_response>`` (``agents.utils.wrap_tool_response``) and
  post-prompt turns rendered standalone (``Agent.render_single_turn``).
* ``think_prefilled`` — the generation prompt already ends with ``<think>\\n`` (Qwen3.5), so sampled text has no opening
  tag (see ``agents.parsing``); ``Agent.get_generation_prompt`` picks the prefill up from the template automatically.
* ``requires_user_query`` — the template raises when no plain user message exists (Qwen3.5: ``No user query found``),
  so a system-only prefix cannot be rendered on its own (``Agent._render_prefix`` uses an anchor query).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    strips_history_think: bool
    think_prefilled: bool
    requires_user_query: bool
    non_thinking_prefill: str = ""      # what enable_thinking=False pre-fills, if the template supports it

    @property
    def wrap_tool_response(self) -> bool:
        return self.strips_history_think


_ANCHOR = [{"role": "user", "content": "anchor"}]


def detect_profile(tokenizer) -> ModelProfile:
    tpl = tokenizer.chat_template or ""
    strips = "last_query_index" in tpl or "tool_response" in tpl
    base = tokenizer.apply_chat_template(_ANCHOR, add_generation_prompt=False, tokenize=False)
    gen = tokenizer.apply_chat_template(_ANCHOR, add_generation_prompt=True, tokenize=False)
    gen_prompt = gen[len(base):]
    prefilled = "<think>" in gen_prompt
    requires_query = True
    try:
        tokenizer.apply_chat_template([{"role": "system", "content": "s"}], add_generation_prompt=False, tokenize=False)
        requires_query = False
    except Exception:
        pass
    non_thinking = ""
    try:
        nt = tokenizer.apply_chat_template(_ANCHOR, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        non_thinking = nt[len(base):].replace(gen_prompt.split("<think>")[0], "", 1) if prefilled else nt[len(base):]
    except Exception:
        pass
    name = getattr(tokenizer, "name_or_path", "") or "unknown"
    return ModelProfile(name=name, strips_history_think=strips, think_prefilled=prefilled,
                        requires_user_query=requires_query, non_thinking_prefill=non_thinking)
