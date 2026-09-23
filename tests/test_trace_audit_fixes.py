"""Regression tests for the 2026-09-23 trace-audit fixes (docs/traces/grpo_step50_trace_audit_preamble.md, F1/F3/F4/F5).

F4  CallLLM applies plugin.turn_max_new_tokens.
F5  a sampled assistant turn is stored as generation prompt + raw ids + eos + the template's turn-end newline.
F1  Agent.fork() inherits exact token ids (branches); compaction tails carry raw ids (no template re-render).
F3  one strict parser (environment grammar) drives branch/return/finish detection.
"""
import asyncio
import os
import sys
import types
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tests.tokenizers import load_tokenizer  # noqa: E402
from agents.parsing import extract_fn_calls_strict, last_strict_call  # noqa: E402

TOK = load_tokenizer()
CFG = types.SimpleNamespace(prompt_length=8192, response_length=32768, plugin=types.SimpleNamespace(retry_cjk=0, turn_max_new_tokens=100))
CHAT = [{"role": "system", "content": "You are a search agent."}, {"role": "user", "content": "Q: which theater?"}]


class _Out:
    def __init__(self, ids):
        self.token_ids, self.log_probs = ids, [0.0] * len(ids)


class _FakeServer:
    """Records the sampling params CallLLM sends and returns a fixed completion."""

    def __init__(self, ids):
        self.ids, self.calls = ids, []

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None):
        self.calls.append(sampling_params)
        return _Out(self.ids)


def _completion(tok, text):
    ids = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
    return {"choices": [{"message": {"content": tok.decode(ids, skip_special_tokens=True), "raw_output_ids": ids,
                                     "response_log_probs": [0.0] * len(ids)}}]}


class TestStrictParser(unittest.TestCase):
    A = "<function=search>\n<parameter=query>a</parameter>\n</function>"
    B = "<function=open_page>\n<parameter=docid>7</parameter>\n</function>"

    def test_unclosed_and_indented_calls_are_not_executed(self):
        self.assertEqual(extract_fn_calls_strict("<function=return>\n<parameter=message>x</parameter>\n"), [])
        self.assertEqual(extract_fn_calls_strict("text " + self.A), [])          # not at line start
        self.assertEqual(last_strict_call(None), None)

    def test_all_calls_outside_think_execute_in_order(self):
        text = self.A + "\n\n\n\n\n\n" + self.B + "\n" + self.A     # gaps no longer split the turn into groups
        calls = extract_fn_calls_strict(text)
        self.assertEqual([c["function"] for c in calls], ["search", "open_page", "search"])
        self.assertEqual(last_strict_call(text)["function"], "search")
        self.assertEqual(calls[1]["arguments"], {"docid": "7"})

    def test_calls_inside_think_are_excluded_even_when_line_anchored(self):
        """The old environment grammar accepted a line-anchored call inside <think> that sat within 4 newlines of the
        real call; the shared grammar strips think blocks first, so this can no longer happen."""
        inside = "<think>\nmaybe " + self.A + "\n</think>\n\n" + self.B
        self.assertEqual([c["function"] for c in extract_fn_calls_strict(inside)], ["open_page"])
        anchored_inside = "<think>\n" + self.A + "\n</think>\n\n" + self.B
        self.assertEqual([c["function"] for c in extract_fn_calls_strict(anchored_inside)], ["open_page"])

    def test_env_uses_the_same_parser(self):
        from envs.local_search import extract_fn_call as env_parse
        text = self.A + "\n" + self.B
        self.assertEqual(env_parse(text), extract_fn_calls_strict(text))
        self.assertIsNone(env_parse("<function=return>\nno close"))


@unittest.skipIf(TOK is None, "tokenizer not available offline")
class TestTurnCap(unittest.TestCase):
    def test_turn_max_new_tokens_is_sent(self):
        from agents.utils import CallLLM
        server = _FakeServer([1, 2, 3])
        loop = asyncio.new_event_loop()
        try:
            llm = CallLLM(server, TOK, CFG, loop)
            loop.run_until_complete(llm._create_completion([5] * 50, max_len=5000))
            self.assertEqual(server.calls[-1]["max_tokens"], 100)
            # explicit smaller max_new_tokens still wins
            loop.run_until_complete(llm._create_completion([5] * 50, max_len=5000, max_new_tokens=20))
            self.assertEqual(server.calls[-1]["max_tokens"], 20)
            # without the plugin cap the remaining budget is used
            cfg2 = types.SimpleNamespace(prompt_length=8192, response_length=32768, plugin=types.SimpleNamespace(retry_cjk=0))
            llm2 = CallLLM(server, TOK, cfg2, loop)
            loop.run_until_complete(llm2._create_completion([5] * 50, max_len=5000))
            self.assertEqual(server.calls[-1]["max_tokens"], 4950)
        finally:
            loop.close()


@unittest.skipIf(TOK is None, "tokenizer not available offline")
class TestTurnTerminatorAndFork(unittest.TestCase):
    def _agent(self):
        from agents.utils import Agent
        return Agent(None, CHAT, TOK, CFG, prompt_turn=2)

    def test_sampled_turn_ends_with_eos_and_newline(self):
        ag = self._agent()
        from agents.model_profile import detect_profile
        opener = "" if detect_profile(TOK).think_prefilled else "<think>\n"     # Qwen3.5 pre-fills the opener
        text = opener + "r\n</think>\n\n<function=search>\n<parameter=query>q</parameter>\n</function>"
        ag.append({"role": "assistant", "content": text}, _completion(TOK, text))
        ids = ag.chat_ids[-1]
        eos_text = TOK.decode([TOK.eos_token_id], skip_special_tokens=False)
        self.assertTrue(TOK.decode(ids, skip_special_tokens=False).endswith(eos_text + "\n"))
        # trained tokens = exactly the sampled ids (incl. eos); generation prompt and newline are not trained
        n_sampled = len(_completion(TOK, text)["choices"][0]["message"]["raw_output_ids"])
        self.assertEqual(sum(ag.token_mask[-1]), n_sampled)
        self.assertFalse(ag.token_mask[-1][-1])
        # the concatenated context now equals the template's canonical rendering of the same chat
        ag.append({"role": "user", "content": "<tool_response>\nOBS\n</tool_response>"})
        direct = TOK.apply_chat_template(ag.chat, add_generation_prompt=True, tokenize=True)
        self.assertEqual(ag.context(), direct)

    def test_fork_inherits_exact_ids_and_is_non_trainable(self):
        ag = self._agent()
        empty = "<think>\n</think>\n\n<function=branch>\n<parameter=description>d</parameter>\n<parameter=prompt>p</parameter>\n</function>"
        ag.append({"role": "assistant", "content": empty}, _completion(TOK, empty))
        ag.append({"role": "user", "content": "<tool_response>\nOBS\n</tool_response>"})
        child = ag.fork(llm_client="client")
        self.assertEqual(child.chat_ids, ag.chat_ids)
        self.assertEqual(child.context(), ag.context())
        self.assertFalse(any(any(m) for m in child.token_mask))
        self.assertEqual(child.init_len, len(ag.chat))
        self.assertNotEqual(child.context_uid, ag.context_uid)
        self.assertEqual(child.llm_client, "client")
        # the old path (re-tokenising from text) would have rewritten '<think>\n</think>' -> '<think>\n\n</think>'
        from agents.utils import Agent
        retok = Agent(None, ag.messages(), TOK, CFG, prompt_turn=2)
        self.assertNotEqual(retok.chat_ids[2], ag.chat_ids[2])
        self.assertIn("<think>\n</think>", TOK.decode(child.chat_ids[2]))
        # appending to the child does not touch the parent
        child.append({"role": "user", "content": "branch prompt"})
        self.assertEqual(len(ag.chat), 4)

    def test_compaction_tail_carries_raw_ids(self):
        from tests.test_compaction_agent import _run
        out, _, _ = _run({})
        seg0, seg1 = out["segments"][0], out["segments"][1]
        k = out["segment_info"][1]["tail_steps"]
        self.assertGreaterEqual(k, 1)
        eos, nl = TOK.eos_token_id, seg1._turn_end_newline_ids()
        # seg1.chat = [system, user, resume, (assistant, observation) x k, ...]; each tail assistant turn must be the RAW
        # sampled ids (generation prompt + encode(text) + eos + newline), i.e. exactly what Agent.append stored in seg0 —
        # not a template re-render (which would rewrite the think block layout)
        for j in range(k):
            a, o = seg1.chat[3 + 2 * j], seg1.chat[4 + 2 * j]
            self.assertEqual(a["role"], "assistant"); self.assertEqual(o["role"], "user")
            expected = seg1.get_generation_prompt() + TOK.encode(a["content"], add_special_tokens=False) + [eos] + nl
            self.assertEqual(seg1.chat_ids[3 + 2 * j], expected)
            self.assertEqual(seg1.chat_ids[4 + 2 * j], seg1.render_single_turn(o))
        self.assertFalse(any(any(m) for m in seg1.token_mask[:3 + 2 * k]), "resume + tail are non-trainable")
        # a rolled-back step (removed from seg0 before the summary) is still carried verbatim in the tail
        n_rolled = out["stats"]["rollback_before_summary"]
        if n_rolled:
            last_tail_assistant = seg1.chat[3 + 2 * (k - 1)]["content"]
            self.assertNotIn(last_tail_assistant, [t["content"] for t in seg0.chat], "rolled-back step lives only in the tail")


if __name__ == "__main__":
    unittest.main()
