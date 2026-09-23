"""plugin.protocol = "legacy" (pre-3f697bf training-time rollout format) vs "v2" (corrected). Both must be reproducible
and explicitly selectable; nothing may switch silently. See agents/protocol.py for the table."""
import asyncio
import os
import sys
import types
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tests.tokenizers import load_tokenizer  # noqa: E402
from agents.protocol import resolve, DEFAULTS  # noqa: E402
from agents.parsing import extract_fn_calls_legacy, legacy_branch_call, legacy_return_requested, parse_actions  # noqa: E402

TOK = load_tokenizer()
CHAT = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
S1 = "<function=search>\n<parameter=query>q1</parameter>\n</function>"
OP = "<function=open_page>\n<parameter=docid>42</parameter>\n</function>"
BR = "<function=branch>\n<parameter=description>d</parameter>\n<parameter=prompt>p</parameter>\n</function>"
MALFORMED_RETURN = "<think>\n</think>\n\n<function=return>\n<parameter=message>\nresult\n</parameter>\n<function>"


def cfg(protocol=None, **over):
    plugin = types.SimpleNamespace(retry_cjk=0, turn_max_new_tokens=100, **over)
    if protocol:
        plugin.protocol = protocol
    return types.SimpleNamespace(prompt_length=8192, response_length=32768, plugin=plugin)


def completion(tok, text):
    ids = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
    return {"choices": [{"message": {"content": tok.decode(ids, skip_special_tokens=True), "raw_output_ids": ids,
                                     "response_log_probs": [0.0] * len(ids)}}]}


class TestResolve(unittest.TestCase):
    def test_defaults_and_overrides(self):
        self.assertEqual(resolve(None).name, "v2")
        p = resolve(types.SimpleNamespace(protocol="legacy"))
        self.assertEqual((p.turn_end_newline, p.enforce_turn_cap, p.branch_inherit, p.tail_inherit, p.action_grammar),
                         (False, False, "render", "render", "legacy"))
        v = resolve(types.SimpleNamespace())
        self.assertEqual((v.turn_end_newline, v.enforce_turn_cap, v.branch_inherit, v.tail_inherit, v.action_grammar),
                         (True, True, "ids", "ids", "strict"))
        mixed = resolve(types.SimpleNamespace(protocol="legacy", turn_end_newline=True, action_grammar="strict"))
        self.assertTrue(mixed.turn_end_newline); self.assertEqual(mixed.action_grammar, "strict"); self.assertFalse(mixed.enforce_turn_cap)
        with self.assertRaises(ValueError):
            resolve(types.SimpleNamespace(protocol="v3"))
        self.assertEqual(set(DEFAULTS), {"legacy", "v2"})


class TestLegacyGrammar(unittest.TestCase):
    def test_legacy_env_parser_last_group_and_think_calls(self):
        text = S1 + "\n\n\n\n\n\n" + OP + "\n" + S1                      # >= 4 newlines: only the last group executes
        self.assertEqual([c["function"] for c in extract_fn_calls_legacy(text)], ["open_page", "search"])
        anchored = "<think>\n" + S1 + "\n</think>\n\n" + OP                # the pre-fix edge case: think call joins the group
        self.assertEqual([c["function"] for c in extract_fn_calls_legacy(anchored)], ["search", "open_page"])
        self.assertEqual([c["function"] for c in parse_actions(anchored)["executable"]], ["open_page"])   # strict
        self.assertEqual(extract_fn_calls_legacy("<function=return>\nno close"), [])

    def test_legacy_branch_and_return_rules(self):
        self.assertIsNotNone(legacy_branch_call("<think>\n" + BR + "\n</think>\n\nno action"))    # anywhere, think included
        self.assertIsNone(parse_actions("<think>\n" + BR + "\n</think>\n\nno action")["executable"] or None)
        self.assertTrue(legacy_return_requested(MALFORMED_RETURN))
        self.assertTrue(legacy_return_requested("<think>I might <function=return> later</think>\n" + S1))   # substring rule
        self.assertFalse(legacy_return_requested(S1))

    def test_env_uses_the_selected_grammar(self):
        from tests.test_action_grammar import _env
        anchored = "<think>\n" + S1 + "\n</think>\n\n" + OP
        env = _env(); env.config.plugin.protocol = "legacy"
        asyncio.run(env.run_action(anchored))
        self.assertEqual([r["function"] for r in env.last_action_results], ["search", "open_page"])
        env2 = _env(); env2.config.plugin.protocol = "v2"
        asyncio.run(env2.run_action(anchored))
        self.assertEqual([r["function"] for r in env2.last_action_results], ["open_page"])


@unittest.skipIf(TOK is None, "tokenizer not available offline")
class TestLegacyVsV2Tokens(unittest.TestCase):
    def test_turn_terminator(self):
        from agents.utils import Agent
        eos = TOK.decode([TOK.eos_token_id], skip_special_tokens=False)
        for proto, expect_nl in (("legacy", False), ("v2", True)):
            ag = Agent(None, CHAT, TOK, cfg(proto), prompt_turn=2)
            ag.append({"role": "assistant", "content": S1}, completion(TOK, S1))
            text = TOK.decode(ag.chat_ids[-1], skip_special_tokens=False)
            self.assertEqual(text.endswith(eos + "\n"), expect_nl, proto)
            ag.append({"role": "user", "content": "<tool_response>\nOBS\n</tool_response>"})
            ctx = TOK.decode(ag.context(), skip_special_tokens=False)
            self.assertEqual(("<|im_end|><|im_start|>user" in ctx), not expect_nl, proto)     # glued boundary only in legacy

    def test_turn_cap_enforcement(self):
        from agents.utils import CallLLM
        from tests.test_trace_audit_fixes import _FakeServer
        for proto, expect in (("legacy", 4950), ("v2", 100)):
            server = _FakeServer([1, 2, 3]); loop = asyncio.new_event_loop()
            try:
                llm = CallLLM(server, TOK, cfg(proto), loop)
                loop.run_until_complete(llm._create_completion([5] * 50, max_len=5000))
            finally:
                loop.close()
            self.assertEqual(server.calls[-1]["max_tokens"], expect, proto)

    def test_branch_inheritance_modes(self):
        from agents.utils import Agent
        empty = "<think>\n</think>\n\n" + BR
        for proto in ("legacy", "v2"):
            ag = Agent(None, CHAT, TOK, cfg(proto), prompt_turn=2)
            ag.append({"role": "assistant", "content": empty}, completion(TOK, empty))
            ag.append({"role": "user", "content": "<tool_response>\nOBS\n</tool_response>"})
            if proto == "v2":
                child = ag.fork(None)
                self.assertEqual(child.chat_ids, ag.chat_ids)
            else:
                child = Agent(None, ag.messages(), TOK, cfg(proto), prompt_turn=2)     # what fold_agent does in legacy mode
                self.assertNotEqual(child.chat_ids[2], ag.chat_ids[2])
                self.assertIn("<think>\n\n</think>", TOK.decode(child.chat_ids[2]))      # template-normalised empty think

    def test_compaction_tail_modes(self):
        from tests.test_compaction_agent import _run
        for proto in ("legacy", "v2"):
            out, _, _ = _run({}, protocol=proto)
            seg0, seg1 = out["segments"][0], out["segments"][1]
            self.assertEqual(seg1.protocol.name, proto)
            a = seg1.chat[3]; self.assertEqual(a["role"], "assistant")
            stored_in_seg0 = next(ids for t, ids in zip(seg0.chat, seg0.chat_ids) if t is not a and t.get("content") == a["content"] and t["role"] == "assistant")
            if proto == "v2":
                self.assertEqual(seg1.chat_ids[3], stored_in_seg0, "v2 carries the previous segment's exact ids")
            else:
                self.assertEqual(seg1.chat_ids[3], seg1.render_single_turn(a), "legacy re-renders the tail through the template")
                self.assertNotEqual(seg1.chat_ids[3], stored_in_seg0, "legacy stored ids end at eos (no newline) so they differ from the render")
            self.assertFalse(any(any(m) for m in seg1.token_mask[:5]), "resume + tail never trainable in either mode")


if __name__ == "__main__":
    unittest.main()
