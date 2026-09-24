"""SUPO semantics on the compaction rollout (decision D1, 2026-09-24; arXiv:2510.06727 Alg. 2 / Sec. 4.2), CPU:
trigger when the occupied context reaches summary_ratio x (prompt_length + response_length) after an (action, observation)
pair; that pair is dropped; the summary is written with the SUPO v_sum prompt; the next trajectory = original prompt +
continuation template (no verbatim tail); summaries count towards H; unfinished rollouts (max summaries) are overlong ->
masked; plus the D5 tool-invocation cap."""
import asyncio
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tests.test_compaction_agent import TOK, OPENER, PROMPT, SEARCH, FINISH, _cfg, _act  # noqa: E402


class SupoScriptedLLM:
    def __init__(self, tok, finish_at, think_words=40, summary_words=30):
        self.tok, self.finish_at, self.think_words, self.summary_words = tok, finish_at, think_words, summary_words
        self.calls, self.step_calls, self.summary_calls, self.summary_prompts = 0, 0, 0, []

    async def create_completion(self, prompt, uid=None, max_len=None, messages=None):
        from agents.prompts import COMPACTION_SUMMARY_PROMPT, SUPO_SUMMARY_PROMPT
        self.calls += 1
        last = messages[-1]["content"] if messages else ""
        if messages and messages[-1]["role"] == "user" and last in (COMPACTION_SUMMARY_PROMPT, SUPO_SUMMARY_PROMPT):
            self.summary_calls += 1; self.summary_prompts.append(last)
            text = OPENER + "summarising\n</think>\n\n<summary>\nSUMMARY#%d %s\n</summary>" % (self.summary_calls, "finding " * self.summary_words)
        else:
            self.step_calls += 1
            i = self.step_calls
            action = FINISH if i >= self.finish_at else SEARCH.format(i=i)
            text = OPENER + "THINK#%d %s\n</think>\n\n%s" % (i, ("reason " * self.think_words), action)
        ids = self.tok.encode(text, add_special_tokens=False) + [self.tok.eos_token_id]
        return {"choices": [{"message": {"content": self.tok.decode(ids, skip_special_tokens=True), "raw_output_ids": ids,
                                         "response_log_probs": [0.0] * len(ids)}}]}


def _run(cc_kwargs, response_length=1600, prompt_length=200, finish_at=12, env=None):
    from agents.compaction_agent import CompactionConfig, run_compaction_rollout
    llm = SupoScriptedLLM(TOK, finish_at=finish_at)
    cc = CompactionConfig(**{"summary_protocol": "supo", "summary_ratio": 0.95, "max_compactions": 2, "tail_steps": 0,
                             "summary_max_tokens": 120, "max_turn": 50, "mask_unfinished": True, **cc_kwargs})
    cfg = _cfg(response_length); cfg.prompt_length = prompt_length
    out = asyncio.run(run_compaction_rollout(PROMPT, llm, TOK, cfg, _act, cc, env=env))
    return out, llm, cfg, cc


@unittest.skipIf(TOK is None, "tokenizer not available offline")
class TestSupoSemantics(unittest.TestCase):
    def test_trigger_drop_pair_prompts_and_continuation(self):
        from agents.prompts import SUPO_SUMMARY_PROMPT, SUPO_CONTINUATION_TEMPLATE, COMPACTION_RESUME_TEMPLATE
        out, llm, cfg, cc = _run({})
        st = out["stats"]
        self.assertTrue(out["finished"]); self.assertGreaterEqual(st["compactions"], 1)
        self.assertEqual(st["dropped_pairs"], st["compactions"], "SUPO discards the pair that crossed L before every summary")
        self.assertEqual(st["tail_steps"], 0)
        self.assertTrue(all(p == SUPO_SUMMARY_PROMPT for p in llm.summary_prompts))
        ceiling = cfg.prompt_length + cfg.response_length
        # every non-final segment was summarised at/after the 95 % line, and only after a pair was dropped it sits below it
        for seg in out["segments"][:-1]:
            self.assertLess(len(seg.context()), ceiling)
        seg1 = out["segments"][1]
        text = TOK.decode(seg1.context())
        self.assertIn(SUPO_CONTINUATION_TEMPLATE.format(summary="")[:30], text)
        self.assertNotIn(COMPACTION_RESUME_TEMPLATE.format(summary="")[:30], text)
        # next trajectory = original prompt turns + one continuation turn, then generated turns only (no verbatim tail)
        self.assertEqual(seg1.chat[len(PROMPT)]["role"], "user")
        self.assertTrue(seg1.chat[len(PROMPT)]["content"].startswith("We are in the following stage"))
        self.assertEqual(seg1.chat[len(PROMPT) + 1]["role"], "assistant")
        # summaries count towards H
        self.assertEqual(st["turns"], llm.step_calls + llm.summary_calls)

    def test_overlong_when_summaries_exhausted_is_masked(self):
        from agents.compaction_agent import _segment_outputs
        out, llm, cfg, cc = _run({"max_compactions": 1}, finish_at=10_000)
        self.assertFalse(out["finished"]); self.assertEqual(out["stop_reason"], "budget_exhausted")
        self.assertEqual(out["stats"]["compactions"], 1)
        outs = asyncio.run(_segment_outputs(out, 0.0, True, {}, True))
        self.assertTrue(all(o.extra_fields["mask_rollout"] for o in outs))
        from agents.verl_plugin.agent_loop import apply_rollout_masks
        apply_rollout_masks(outs, "u_0")
        self.assertTrue(all(sum(o.response_mask) == 0 for o in outs), "overlong rows carry no loss")
        self.assertTrue(all(o.reward_score == 0.0 for o in outs), "but keep their reward for the group statistics")

    def test_compactionrl_protocol_unchanged(self):
        from agents.prompts import COMPACTION_SUMMARY_PROMPT
        out, llm, cfg, cc = _run({"summary_protocol": "compactionrl", "threshold": 350, "tail_steps": 2, "max_compactions": 3})
        self.assertTrue(out["finished"]); self.assertEqual(out["stats"]["dropped_pairs"], 0)
        self.assertTrue(all(p == COMPACTION_SUMMARY_PROMPT for p in llm.summary_prompts))

    def test_tool_call_cap(self):
        class Env:
            stats = {"tool_calls": 0}
        env = Env()
        from agents.compaction_agent import CompactionConfig, run_compaction_rollout
        llm = SupoScriptedLLM(TOK, finish_at=10_000)
        cc = CompactionConfig(summary_protocol="compactionrl", threshold=100, max_compactions=3, tail_steps=0, max_turn=50, max_tool_calls=4)
        cfg = _cfg(4000)

        async def act(response):
            env.stats["tool_calls"] += 1          # what agents.utils.run_action does for executed calls
            return await _act(response)
        out = asyncio.run(run_compaction_rollout(PROMPT, llm, TOK, cfg, act, cc, env=env))
        self.assertEqual(out["stop_reason"], "max_tool_calls"); self.assertEqual(out["stats"]["turns"], 4)

    def test_from_plugin_supo_knobs(self):
        import types
        from agents.compaction_agent import CompactionConfig
        pl = types.SimpleNamespace(summary_protocol="supo", max_summaries=2, val_max_summaries=3, summary_ratio=0.9, max_tool_calls=100, max_compactions=7)
        self.assertEqual((CompactionConfig.from_plugin(pl, True).max_compactions, CompactionConfig.from_plugin(pl, False).max_compactions), (2, 3))
        c = CompactionConfig.from_plugin(pl, True); self.assertTrue(c.supo); self.assertEqual(c.summary_ratio, 0.9); self.assertEqual(c.max_tool_calls, 100)


if __name__ == "__main__":
    unittest.main()
