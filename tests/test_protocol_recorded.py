"""Deterministic protocol tests on RECORDED model outputs (tests/fixtures/recorded_turns.json: real completions + their
sampled token ids from the Qwen3-8B GRPO step-50 capture runs of 2026-09-23). No vLLM sampling is involved: the recorded
ids are replayed through Agent/parser/environment exactly as a rollout would, under both protocols.

Cases: multiple search calls, multiple open_page calls, malformed and valid branch returns, empty and non-empty thinking,
a think cut by the 2048 cap, finish, a call inside <think>, a compaction summary, a giant multi-call turn (per-turn cap)."""
import asyncio
import json
import os
import sys
import types
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tests.tokenizers import load_tokenizer  # noqa: E402
from agents.parsing import extract_fn_calls_legacy, extract_summary, find_think, parse_actions  # noqa: E402

FX = json.load(open(os.path.join(REPO, "tests", "fixtures", "recorded_turns.json")))
TOK = load_tokenizer()
# the recorded ids are Qwen3-8B ids (vocab 151 936); id-level replay only makes sense with that tokenizer family
IDS_MATCH_TOKENIZER = TOK is not None and TOK.decode(FX["finish"]["completion_ids"], skip_special_tokens=True) == FX["finish"]["completion_text"]
CHAT = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]


def names(text, grammar="strict"):
    if grammar == "legacy":
        return [c["function"] for c in extract_fn_calls_legacy(text)]
    return [c["function"] for c in parse_actions(text)["executable"]]


class TestRecordedParsing(unittest.TestCase):
    def test_multi_call_turns_execute_all_in_order(self):
        for key, fn in (("multi_call_search", "search"), ("multi_call_open_page", "open_page")):
            p = parse_actions(FX[key]["completion_text"])
            self.assertIsNone(p["error"]); self.assertGreaterEqual(len(p["executable"]), 3)
            self.assertTrue(all(c["function"] == fn for c in p["executable"]))
            self.assertEqual([c["call_id"] for c in p["executable"]], [f"c{i}" for i in range(1, len(p["executable"]) + 1)])
            self.assertEqual(names(FX[key]["completion_text"], "legacy"), names(FX[key]["completion_text"]))   # both grammars agree

    def test_malformed_and_valid_return(self):
        m = parse_actions(FX["malformed_return"]["completion_text"])
        self.assertEqual(m["executable"], []); self.assertEqual(m["unclosed_terminal"], "return")
        self.assertTrue(m["unclosed_terminal_args"].get("message", "").strip())
        self.assertTrue(FX["malformed_return"]["completion_text"].rstrip().endswith("<function>"))
        self.assertEqual(names(FX["malformed_return"]["completion_text"], "legacy"), [])          # old env also rejected it …
        self.assertIn("<function=return>", FX["malformed_return"]["completion_text"])             # … but the old substring rule ended the branch
        v = parse_actions(FX["valid_return"]["completion_text"])
        self.assertEqual([c["function"] for c in v["executable"]], ["return"]); self.assertIsNone(v["unclosed_terminal"])

    def test_thinking_variants(self):
        e = FX["empty_think_search"]["completion_text"]
        self.assertTrue(e.startswith("<think>\n</think>")); self.assertEqual(find_think(e)[0], ""); self.assertEqual(names(e), ["search"])
        n = FX["nonempty_think_branch"]["completion_text"]
        self.assertTrue(len(find_think(n)[0]) > 50); self.assertEqual(names(n), ["branch"])
        c = FX["cap_cut_think"]["completion_text"]
        self.assertEqual(len(FX["cap_cut_think"]["completion_ids"]), 2048)
        self.assertIsNone(find_think(c)[0]); self.assertEqual(names(c), []); self.assertEqual(names(c, "legacy"), [])

    def test_finish_call_inside_think_and_summary(self):
        self.assertEqual(names(FX["finish"]["completion_text"]), ["finish"])
        t = FX["call_inside_think"]["completion_text"]
        p = parse_actions(t)
        self.assertGreater(p["calls_in_think"], 0)
        self.assertGreaterEqual(len(extract_fn_calls_legacy(t)), len(p["executable"]))     # legacy could execute a think call
        self.assertTrue(extract_summary(FX["compaction_summary"]["completion_text"]))

    def test_giant_multi_call_turn_is_capped(self):
        t = FX["giant_multi_call"]["completion_text"]
        all_calls = parse_actions(t, max_calls=0)["calls"]
        self.assertGreaterEqual(len(all_calls), 20)
        self.assertEqual(len(parse_actions(t)["executable"]), 8)
        self.assertEqual(parse_actions(t)["truncated_calls"], len(all_calls) - 8)


@unittest.skipIf(not IDS_MATCH_TOKENIZER, "recorded ids are Qwen3-8B ids; replay skipped for other tokenizer families (parse tests above still run)")
class TestRecordedReplay(unittest.TestCase):
    """Replay recorded ids through Agent under both protocols and check the serialised prompt and masks."""

    def _cfg(self, protocol):
        return types.SimpleNamespace(prompt_length=8192, response_length=32768,
                                     plugin=types.SimpleNamespace(retry_cjk=0, turn_max_new_tokens=2048, protocol=protocol))

    def _comp(self, key):
        ids = FX[key]["completion_ids"]
        return {"choices": [{"message": {"content": FX[key]["completion_text"], "raw_output_ids": ids, "response_log_probs": [0.0] * len(ids)}}]}

    def test_recorded_ids_are_the_tokenization_of_the_text(self):
        for key, fx in FX.items():
            ids = fx["completion_ids"]
            self.assertEqual(TOK.decode(ids, skip_special_tokens=True), fx["completion_text"], key)
            self.assertIn(ids[-1], (TOK.eos_token_id, TOK.convert_tokens_to_ids("<|endoftext|>")) if len(ids) < 2048 else (ids[-1],), key)

    def test_replay_serialisation_both_protocols(self):
        from agents.utils import Agent
        eos = TOK.decode([TOK.eos_token_id], skip_special_tokens=False)
        for protocol in ("v2", "legacy"):
            ag = Agent(None, CHAT, TOK, self._cfg(protocol), prompt_turn=2)
            ag.append({"role": "assistant", "content": FX["empty_think_search"]["completion_text"]}, self._comp("empty_think_search"))
            ag.append({"role": "user", "content": "<tool_response>\nOBS-1\n</tool_response>"})
            ag.append({"role": "assistant", "content": FX["nonempty_think_branch"]["completion_text"]}, self._comp("nonempty_think_branch"))
            ag.append({"role": "user", "content": "<tool_response>\nBranch has finished its task, the returned message is:\n\nR\n</tool_response>"})
            text = TOK.decode(ag.context(), skip_special_tokens=False)
            # v2 invariant: prompt + [gen prompt + RAW sampled ids + eos + "\n" | rendered user turn]* + gen prompt.
            # (Not "== canonical template render": the model wrote '<think>\n</think>' and the template would normalise it to
            # '<think>\n\n</think>\n\n' — raw sampled whitespace is preserved on purpose, see docs/PROTOCOL.md §2.)
            gp = ag.get_generation_prompt(); nl = ag._turn_end_newline_ids()
            expected = list(TOK.apply_chat_template(CHAT, tokenize=True, return_dict=False))
            for turn, comp in zip(ag.chat[2:], ag.chat_completions[2:]):
                if comp is not None:
                    ids = comp["choices"][0]["message"]["raw_output_ids"]
                    expected += gp + list(ids) + ([] if ids[-1] == TOK.eos_token_id else [TOK.eos_token_id]) + (nl if protocol == "v2" else [])
                else:
                    expected += ag.render_single_turn(turn)
            expected += gp
            self.assertEqual(ag.context(), expected, protocol)
            if protocol == "v2":
                self.assertNotIn(eos + "<|im_start|>", text)
            else:
                self.assertIn(eos + "<|im_start|>user", text, "legacy keeps the glued boundary the checkpoint was trained on")
            # think blocks of both recorded turns are in the model input verbatim, ids exact
            for key in ("empty_think_search", "nonempty_think_branch"):
                self.assertIn(FX[key]["completion_text"], text)
                ids = FX[key]["completion_ids"]
                ctx = ag.context()
                self.assertTrue(any(ctx[i:i + len(ids)] == ids for i in range(len(ctx) - len(ids) + 1)), f"{protocol}: recorded ids of {key} are a contiguous subsequence of the context")
            d = asyncio.run(ag.get_data())
            self.assertEqual(sum(d["response_mask"]), len(FX["empty_think_search"]["completion_ids"]) + len(FX["nonempty_think_branch"]["completion_ids"]))

    def test_environment_execution_of_recorded_multi_call(self):
        from tests.test_action_grammar import _env
        env = _env()
        ret = asyncio.run(env.run_action(FX["multi_call_search"]["completion_text"]))
        res = env.last_action_results
        self.assertEqual(len(res), len(parse_actions(FX["multi_call_search"]["completion_text"])["executable"]))
        self.assertTrue(all(r["status"] == "ok" for r in res))
        self.assertTrue(ret["observation"].startswith('[Search Results for'))
        env2 = _env()
        ret2 = asyncio.run(env2.run_action(FX["cap_cut_think"]["completion_text"]))
        self.assertIn("No function call was detected", ret2["observation"])


if __name__ == "__main__":
    unittest.main()
