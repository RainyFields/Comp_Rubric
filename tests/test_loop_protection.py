"""Shakeout #1 regression: a `return` closed with `<function>` instead of `</function>` (what this checkpoint emits) must end
the branch as a malformed return, and repeated turns without a valid call must not loop to max_turn."""
import asyncio
import os
import sys
import types
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tests.tokenizers import load_tokenizer  # noqa: E402
from agents.parsing import parse_actions  # noqa: E402

TOK = load_tokenizer()
CFG = types.SimpleNamespace(prompt_length=8192, response_length=32768, plugin=types.SimpleNamespace(retry_cjk=0))
CHAT = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
MALFORMED_RETURN = "<think>\n</think>\n\n<function=return>\n<parameter=message>\n**RESULT**: found it\n</parameter>\n<function>"
GOOD_SEARCH = "<think>\n</think>\n\n<function=search>\n<parameter=query>q</parameter>\n</function>"


class TestUnclosedTerminal(unittest.TestCase):
    def test_malformed_return_is_reported_not_executed(self):
        p = parse_actions(MALFORMED_RETURN)
        self.assertEqual(p["executable"], [])
        self.assertEqual(p["unclosed_terminal"], "return")
        self.assertEqual(p["unclosed_terminal_args"]["message"].strip(), "**RESULT**: found it")
        self.assertIsNone(parse_actions(GOOD_SEARCH)["unclosed_terminal"])
        self.assertIsNone(parse_actions("<function=search>\n<parameter=query>q</parameter>\n")["unclosed_terminal"])   # non-terminal
        # an unclosed terminal that precedes a closed call is not "the last opener"
        self.assertIsNone(parse_actions("<function=finish>\n<parameter=answer>a</parameter>\n" + GOOD_SEARCH)["unclosed_terminal"])


class _LLM:
    def __init__(self, tok, texts):
        self.tok, self.texts, self.i = tok, texts, 0

    async def create_completion(self, prompt, uid=None, max_len=None, messages=None):
        text = self.texts[min(self.i, len(self.texts) - 1)]
        self.i += 1
        ids = self.tok.encode(text, add_special_tokens=False) + [self.tok.eos_token_id]
        return {"choices": [{"message": {"content": self.tok.decode(ids, skip_special_tokens=True), "raw_output_ids": ids,
                                         "response_log_probs": [0.0] * len(ids)}}]}


async def _no_call_action(response):
    return "No function call was detected in the model response."


@unittest.skipIf(TOK is None, "tokenizer not available offline")
class TestBranchLoopProtection(unittest.TestCase):
    def _branch(self, texts, **kw):
        from agents.utils import Agent
        ag = Agent(_LLM(TOK, texts), CHAT, TOK, CFG, prompt_turn=2)
        ag.append({"role": "user", "content": "branch prompt"})
        return ag, asyncio.run(ag.react(_no_call_action, max_turn=50,
                                         should_continue=lambda r: not (any(c["function"] == "return" for c in parse_actions(r)["executable"])
                                                                        or parse_actions(r)["unclosed_terminal"] == "return"), **kw))

    def test_malformed_return_ends_the_branch_on_the_first_turn(self):
        ag, out = self._branch([MALFORMED_RETURN])
        self.assertEqual(out["iteration"], 1)
        self.assertEqual(out["last_response"], MALFORMED_RETURN)
        self.assertIsNone(out["forced_reason"])

    def test_consecutive_no_call_turns_force_a_return(self):
        ag, out = self._branch(["just prose, no call"] * 10, max_consecutive_no_call=3)
        self.assertEqual(out["iteration"], 3)
        self.assertEqual(out["forced_reason"], "no_call_loop")
        self.assertEqual(out["last_response"], "just prose, no call")

    def test_protection_can_be_disabled(self):
        ag, out = self._branch(["just prose, no call"] * 6, max_consecutive_no_call=0, max_tokens=None)
        self.assertEqual(out["iteration"], 50)    # without protection the branch loops to max_turn (the shakeout #1 behaviour)
        self.assertIsNone(out["forced_reason"])


if __name__ == "__main__":
    unittest.main()
