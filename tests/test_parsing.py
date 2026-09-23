"""agents.parsing: think blocks with or without the opening tag (Qwen3 vs Qwen3.5), tool-call wrapper tolerance."""
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from agents.parsing import extract_fn_call, extract_summary, find_think, has_think, strip_think, think_is_empty  # noqa: E402

ACTION = "<function=search>\n<parameter=query>q</parameter>\n</function>"
TAGGED = "<think>\nreasoning here\n</think>\n\n" + ACTION          # Qwen3 sample
PREFILLED = "reasoning here\n</think>\n\n" + ACTION                # Qwen3.5 sample (opener in the generation prompt)
EMPTY_TAGGED = "<think>\n\n</think>\n\n" + ACTION
EMPTY_PREFILLED = "\n</think>\n\n" + ACTION


class TestThink(unittest.TestCase):
    def test_tagged_and_prefilled_are_equivalent(self):
        for text in (TAGGED, PREFILLED):
            self.assertTrue(has_think(text))
            self.assertEqual(find_think(text)[0], "reasoning here")
            self.assertFalse(think_is_empty(text))
            self.assertEqual(strip_think(text), ACTION)

    def test_empty_think(self):
        for text in (EMPTY_TAGGED, EMPTY_PREFILLED):
            self.assertTrue(think_is_empty(text))
            self.assertEqual(strip_think(text), ACTION)

    def test_no_think(self):
        self.assertFalse(has_think(ACTION))
        self.assertFalse(think_is_empty(ACTION))
        self.assertEqual(strip_think(ACTION), ACTION)
        self.assertEqual(find_think(None), (None, None))

    def test_prefilled_opener_does_not_swallow_a_later_tagged_block(self):
        text = "<think>\nx\n</think>\n\nanswer"
        self.assertEqual(find_think(text)[0], "x")


class TestCalls(unittest.TestCase):
    def test_bare_function_block(self):
        self.assertEqual(extract_fn_call(TAGGED), {"function": "search", "arguments": {"query": "q"}})

    def test_tool_call_wrapper_is_tolerated(self):
        wrapped = "<tool_call>\n" + ACTION + "\n</tool_call>"
        self.assertEqual(extract_fn_call(wrapped), {"function": "search", "arguments": {"query": "q"}})

    def test_last_call_wins_and_multiline_params(self):
        text = ACTION + "\n<function=finish>\n<parameter=answer>A\nB</parameter>\n<parameter=explanation>e</parameter>\n</function>"
        self.assertEqual(extract_fn_call(text), {"function": "finish", "arguments": {"answer": "A\nB", "explanation": "e"}})

    def test_no_call(self):
        self.assertIsNone(extract_fn_call("just text"))
        self.assertIsNone(extract_fn_call(None))

    def test_summary(self):
        self.assertEqual(extract_summary("<summary>\n s1 \n</summary> x <summary>s2</summary>"), "s2")
        self.assertIsNone(extract_summary("none"))


if __name__ == "__main__":
    unittest.main()
