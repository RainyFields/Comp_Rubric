"""The shared action grammar (agents.parsing.parse_actions) and its use by the environment and the fold agent.

Legal tools (from agents/tool_spec.py + agents/prompts.py, workflow ``search_branch``):
  MAIN   : search, open_page (repeatable in one turn: "you can search multiple queries in one turn"), branch ("one task
           at a time" -> exactly one branch, alone), finish (terminal, alone)
  BRANCH : search, open_page (repeatable), return (terminal, alone); finish/branch are refused by the branch guard
Nothing inside <think> is an action. Calls must start a line and be closed.
"""
import asyncio
import os
import sys
import types
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from agents.parsing import extract_fn_calls_strict, last_strict_call, parse_actions  # noqa: E402

S1 = "<function=search>\n<parameter=query>q1</parameter>\n</function>"
S2 = "<function=search>\n<parameter=query>q2</parameter>\n<parameter=topk>5</parameter>\n</function>"
OP = "<function=open_page>\n<parameter=docid>42</parameter>\n</function>"
BR = "<function=branch>\n<parameter=description>d</parameter>\n<parameter=prompt>p</parameter>\n</function>"
FIN = "<function=finish>\n<parameter=answer>A</parameter>\n<parameter=explanation>E [42]</parameter>\n</function>"
RET = "<function=return>\n<parameter=message>m</parameter>\n</function>"
THINK = "<think>\nlet me search\n</think>\n\n"


def names(text):
    return [c["function"] for c in parse_actions(text)["executable"]]


class TestGrammar(unittest.TestCase):
    def test_multiple_search_open_calls_execute_in_order_with_ids(self):
        p = parse_actions(THINK + S1 + "\n" + OP + "\n\n\n\n\n\n" + S2, turn_id="7")     # a big gap no longer splits groups
        self.assertIsNone(p["error"])
        self.assertEqual([c["function"] for c in p["executable"]], ["search", "open_page", "search"])
        self.assertEqual([c["call_id"] for c in p["executable"]], ["7.1", "7.2", "7.3"])
        self.assertEqual(p["executable"][2]["arguments"], {"query": "q2", "topk": "5"})

    def test_calls_inside_think_are_never_actions(self):
        p = parse_actions("<think>\n" + S1 + "\n</think>\n\n" + OP)          # line-anchored inside think: the old edge case
        self.assertEqual([c["function"] for c in p["executable"]], ["open_page"])
        self.assertEqual(p["calls_in_think"], 1)
        p = parse_actions("<think>\n" + FIN + "\n</think>\n\n" + S1)          # terminal call inside think does not poison the turn
        self.assertIsNone(p["error"]); self.assertEqual(names("<think>\n" + FIN + "\n</think>\n\n" + S1), ["search"])
        p = parse_actions("only thinking " + S1 + "\n</think>\n\n" + OP)      # pre-filled opener (Qwen3.5 style)
        self.assertEqual([c["function"] for c in p["executable"]], ["open_page"])
        self.assertEqual(names("<think>\n" + S1 + "\n</think>"), [])          # nothing outside think

    def test_malformed_calls(self):
        self.assertEqual(names("<function=return>\n<parameter=message>x</parameter>\n"), [])       # unclosed
        self.assertEqual(parse_actions("<function=return>\n<parameter=message>x</parameter>\n")["unclosed_tags"], 1)
        self.assertEqual(names("text " + S1), [])                                                 # not at line start
        self.assertEqual(names("  " + S1), ["search"])                                             # indentation is fine
        self.assertEqual(names(None), []); self.assertEqual(names(""), [])
        self.assertEqual(parse_actions("<function=search>\n<parameter=query>q</parameter>\n</function>")["executable"][0]["arguments"], {"query": "q"})

    def test_terminal_calls_must_be_standalone(self):
        for text in (S1 + "\n" + FIN, FIN + "\n" + S1, RET + "\n" + OP, FIN + "\n" + RET):
            p = parse_actions(text)
            self.assertIsNotNone(p["error"], text)
            self.assertEqual(p["executable"], [])
            self.assertGreaterEqual(len(p["calls"]), 2)         # still listed for diagnostics
        self.assertEqual(names(THINK + FIN), ["finish"])
        self.assertEqual(names(RET), ["return"])
        self.assertEqual(last_strict_call(THINK + FIN)["function"], "finish")

    def test_branch_must_be_alone_and_single(self):
        self.assertEqual(names(THINK + BR), ["branch"])
        for text in (BR + "\n" + BR, BR + "\n" + S1, S1 + "\n" + BR):
            p = parse_actions(text)
            self.assertIsNotNone(p["error"], text); self.assertEqual(p["executable"], [])
        self.assertIn("one branch", parse_actions(BR + "\n" + BR)["error"])

    def test_legacy_helpers_follow_the_same_grammar(self):
        self.assertEqual([c["function"] for c in extract_fn_calls_strict(S1 + "\n" + OP)], ["search", "open_page"])
        self.assertEqual(extract_fn_calls_strict(S1 + "\n" + FIN), [])


class _FakeClient:
    async def search(self, query, k):
        return [{"docid": f"d-{query}", "url": "u", "text": f"snippet for {query}"}]

    async def open(self, url, docid):
        return [{"docid": docid or url, "url": "u", "text": "page body"}]


def _env(must_search=False):
    from envs.local_search import LocalSearch
    cfg = types.SimpleNamespace(plugin=types.SimpleNamespace(double_check=False, must_search=must_search))
    os.environ.setdefault("LOCAL_SEARCH_URL", "http://127.0.0.1:1")
    env = LocalSearch(cfg, None, "search")
    env.client = _FakeClient()
    env.question, env.label_answer = "Q", "A"
    return env


class TestEnvironmentExecution(unittest.TestCase):
    def test_multi_call_results_map_one_to_one(self):
        env = _env()
        ret = asyncio.run(env.run_action(THINK + S1 + "\n" + OP + "\n" + S2))
        res = env.last_action_results
        self.assertEqual([r["function"] for r in res], ["search", "open_page", "search"])
        self.assertEqual([r["call_id"] for r in res], ["c1", "c2", "c3"])
        self.assertTrue(all(r["status"] == "ok" for r in res))
        self.assertTrue(res[0]["observation"].startswith('[Search Results for "q1"]'))
        self.assertTrue(res[1]["observation"].startswith("[Opened Page Content]"))
        self.assertTrue(res[2]["observation"].startswith('[Search Results for "q2"]'))
        # the observation the policy sees is exactly the concatenation of the per-call observations (+ the fixed reminder)
        joined = "".join(r["observation"] for r in res)
        self.assertTrue(ret["observation"].startswith(joined.strip()[:200]))
        self.assertIn("* Please reflect", ret["observation"])
        self.assertEqual(ret["results"], res)

    def test_per_call_errors_do_not_clobber_earlier_calls(self):
        env = _env()
        ret = asyncio.run(env.run_action(S1 + "\n<function=verify>\n<parameter=x>1</parameter>\n</function>\n<function=open_page>\n</function>"))
        res = env.last_action_results
        self.assertEqual([r["status"] for r in res], ["ok", "error", "error"])
        self.assertIn('[Search Results for "q1"]', ret["observation"])
        self.assertIn('The function "verify" is not supported', ret["observation"])
        self.assertIn('requires either a "docid" or a "url"', ret["observation"])

    def test_terminal_with_other_calls_is_rejected_without_execution(self):
        env = _env()
        ret = asyncio.run(env.run_action(S1 + "\n" + FIN))
        self.assertTrue(ret["observation"].startswith("[Error] a terminal call"))
        self.assertFalse(env.is_finish)
        self.assertEqual([r["status"] for r in env.last_action_results], ["rejected", "rejected"])
        self.assertEqual(env.stats["search"], 0)

    def test_finish_alone_finishes_and_calls_in_think_are_ignored(self):
        env = _env()
        ret = asyncio.run(env.run_action("<think>\n" + S1 + "\n</think>\n\n" + FIN))
        self.assertEqual(ret.get("action"), "finish")
        self.assertTrue(env.is_finish)
        self.assertEqual(env.stats["search"], 0, "the search inside <think> must not run")
        # env rule plugin.must_search=True (the training config): finish before any search is refused with an observation
        env3 = _env(must_search=True)
        ret = asyncio.run(env3.run_action(FIN))
        self.assertIsNone(ret.get("action")); self.assertFalse(env3.is_finish); self.assertIn("observation", ret)
        env2 = _env()
        ret = asyncio.run(env2.run_action("<think>\n" + S1 + "\n</think>\n\nno call here"))
        self.assertIn("No function call was detected", ret["observation"])
        self.assertEqual(env2.last_action_results, [])


if __name__ == "__main__":
    unittest.main()


class TestCallCap(unittest.TestCase):
    def test_calls_beyond_the_per_turn_cap_are_not_executed(self):
        many = "\n".join(OP for _ in range(12))
        p = parse_actions(many)
        self.assertEqual(len(p["calls"]), 12)
        self.assertEqual(len(p["executable"]), 8)
        self.assertEqual(p["truncated_calls"], 4)
        self.assertEqual(len(parse_actions(many, max_calls=3)["executable"]), 3)
        self.assertEqual(len(parse_actions(many, max_calls=0)["executable"]), 12)      # 0 = no cap
        env = _env()
        env.config.plugin.max_calls_per_turn = 2
        ret = asyncio.run(env.run_action(many))
        self.assertEqual(len(env.last_action_results), 2)
        self.assertIn("Only the first 2 function calls", ret["observation"])
