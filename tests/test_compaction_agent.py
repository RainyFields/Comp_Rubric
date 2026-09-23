"""CompactionRL rollout tests (agents/compaction_agent.py) with a scripted LLM and the real Qwen3 tokenizer.

Offline: ``~/xiaoxuan/envs/fold_train/bin/python -m unittest tests.test_compaction_agent -v``
Set ``QWEN3_TOKENIZER_PATH`` to any Qwen3 tokenizer directory or hub id if the default candidates are absent.
"""
import asyncio
import os
import sys
import types
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tests.tokenizers import load_tokenizer, profile_of  # noqa: E402

TOK = load_tokenizer()
PROFILE = profile_of(TOK) if TOK is not None else None
OPENER = "" if (PROFILE is not None and PROFILE.think_prefilled) else "<think>\n"   # Qwen3.5 pre-fills the opener
PROMPT = [{"role": "system", "content": "You are a search agent. Tools: search, open_page, finish."},
          {"role": "user", "content": "Q: which theater was built by a local during the Depression?"}]
SEARCH = "<function=search>\n<parameter=query>theater depression {i}</parameter>\n</function>"
FINISH = "<function=finish>\n<parameter=answer>The Rialto</parameter>\n</function>"


def _cfg(response_length):
    return types.SimpleNamespace(prompt_length=8192, response_length=response_length,
                                 plugin=types.SimpleNamespace(retry_cjk=0, turn_max_new_tokens=-1))


class ScriptedLLM:
    """Returns think + search for the first ``finish_at`` steps, then finish; a <summary> for q_sum."""

    def __init__(self, tok, finish_at, think_words=40, summary_words=30):
        self.tok, self.finish_at, self.think_words, self.summary_words = tok, finish_at, think_words, summary_words
        self.calls, self.step_calls, self.summary_calls = 0, 0, 0

    async def create_completion(self, prompt, uid=None, max_len=None, messages=None):
        from agents.prompts import COMPACTION_SUMMARY_PROMPT
        self.calls += 1
        if messages and messages[-1]["role"] == "user" and messages[-1]["content"] == COMPACTION_SUMMARY_PROMPT:
            self.summary_calls += 1
            text = OPENER + "summarising\n</think>\n\n<summary>\nSUMMARY#%d %s\n</summary>" % (
                self.summary_calls, "finding " * self.summary_words)
        else:
            self.step_calls += 1
            i = self.step_calls
            action = FINISH if i >= self.finish_at else SEARCH.format(i=i)
            text = OPENER + "THINK#%d %s\n</think>\n\n%s" % (i, ("reason " * self.think_words), action)
        ids = self.tok.encode(text, add_special_tokens=False) + [self.tok.eos_token_id]
        decoded = self.tok.decode(ids, skip_special_tokens=True)
        return {"choices": [{"message": {"content": decoded, "raw_output_ids": ids,
                                         "response_log_probs": [0.0] * len(ids)}}]}


async def _act(response):
    if "<function=finish>" in response:
        return None
    n = response.count("#")
    return "[Search Results] " + " ".join(f"doc{n}_{j} snippet about theaters" for j in range(12))


def _run(cc_kwargs, response_length=1600, finish_at=12, **llm_kwargs):
    from agents.compaction_agent import CompactionConfig, run_compaction_rollout
    llm = ScriptedLLM(TOK, finish_at=finish_at, **llm_kwargs)
    cc = CompactionConfig(**{"max_compactions": 3, "threshold": 350, "tail_steps": 2, "summary_max_tokens": 120,
                             "max_turn": 50, **cc_kwargs})
    cfg = _cfg(response_length)
    out = asyncio.run(run_compaction_rollout(PROMPT, llm, TOK, cfg, _act, cc))
    return out, llm, cfg


@unittest.skipIf(TOK is None, "Qwen3 tokenizer not available offline; set QWEN3_TOKENIZER_PATH")
class TestCompactionRollout(unittest.TestCase):
    def test_compaction_triggers_and_segments_are_well_formed(self):
        from agents.compaction_agent import _segment_outputs
        from agents.prompts import COMPACTION_RESUME_TEMPLATE
        out, llm, cfg = _run({})
        self.assertTrue(out["finished"]); self.assertEqual(out["stop_reason"], "finish")
        n_seg = len(out["segments"])
        self.assertGreaterEqual(n_seg, 2, "budget must force at least one compaction")
        self.assertEqual(out["stats"]["compactions"], n_seg - 1)
        self.assertEqual(llm.summary_calls, n_seg - 1)
        # every segment's generated tokens stay within the budget
        base = len(out["segments"][0].context()) - sum(len(t) for t in out["segments"][0].chat_ids[2:])
        for seg in out["segments"]:
            self.assertLessEqual(len(seg.context()) - base, cfg.response_length, "segment overran its budget")
        outs = asyncio.run(_segment_outputs(out, 1.0, False, {}, True))
        self.assertEqual(len(outs), n_seg)
        n_opt = [o.extra_fields["optimized_tokens"] for o in outs]
        for i, o in enumerate(outs):
            self.assertEqual(o.extra_fields["tokens_after"], sum(n_opt[i + 1:]))
            self.assertEqual(o.extra_fields["segment_index"], i)
            self.assertEqual(o.reward_score, 1.0)
            self.assertEqual(len(o.response_ids), len(o.response_mask))
        self.assertEqual(outs[-1].extra_fields["tokens_after"], 0)
        # summary tokens are optimised (mask 1) in every non-final segment
        for o in outs[:-1]:
            self.assertGreater(o.extra_fields["summary_tokens"], 0)
        # the second segment starts with prompt + resume(summary) + verbatim tail with think preserved
        seg1 = out["segments"][1]
        text = TOK.decode(seg1.context())
        self.assertIn(COMPACTION_RESUME_TEMPLATE.format(summary="")[:40], text)
        self.assertIn("SUMMARY#1", text)
        self.assertGreaterEqual(outs[1].extra_fields["tail_steps"], 1)
        self.assertIn("<tool_response>", text)
        self.assertIn("THINK#", text, "tail assistant turns keep their think block")
        # tail + resume turns are not trainable; only newly generated turns are
        n_prompt_turns = 2
        for t, (turn_ids, mask) in enumerate(zip(seg1.chat_ids, seg1.token_mask)):
            if t < n_prompt_turns + 1 + 2 * outs[1].extra_fields["tail_steps"]:
                self.assertFalse(any(mask), f"turn {t} of segment 1 must be non-trainable")
        self.assertTrue(any(any(m) for m in seg1.token_mask), "segment 1 has trainable tokens")

    def test_no_summary_training_masks_summary_tokens(self):
        from agents.compaction_agent import _segment_outputs
        out, _, _ = _run({"train_summary": False})
        outs = asyncio.run(_segment_outputs(out, 0.0, True, {}, True))
        seg0 = out["segments"][0]
        self.assertFalse(any(seg0.token_mask[-1]), "summary response must carry no loss")
        self.assertEqual(outs[0].extra_fields["summary_tokens"], 0)

    def test_max_compactions_bounds_the_rollout(self):
        out, llm, _ = _run({"max_compactions": 0}, finish_at=100)
        self.assertEqual(len(out["segments"]), 1)
        self.assertEqual(out["stop_reason"], "budget_exhausted")
        self.assertEqual(llm.summary_calls, 0)
        out, _, _ = _run({"max_compactions": 1}, finish_at=100)
        self.assertEqual(len(out["segments"]), 2)
        self.assertEqual(out["stop_reason"], "budget_exhausted")

    def test_paper_resume_context_drops_the_task_prompt(self):
        """resume_keep_task_prompt=False = paper Eq. 9: (system) + u_resume + tail; the question lives in the summary."""
        from agents.compaction_agent import _segment_outputs
        out, _, cfg = _run({"resume_keep_task_prompt": False})
        self.assertGreaterEqual(len(out["segments"]), 2)
        seg0, seg1 = out["segments"][0], out["segments"][1]
        self.assertEqual(seg1.prompt_turn, 1, "only the system turn is a prompt turn in a resumed segment")
        self.assertEqual(seg1.chat[0]["role"], "system")
        self.assertEqual(seg1.chat[1]["role"], "user")
        self.assertTrue(seg1.chat[1]["content"].startswith("Your context window was compacted"))
        self.assertNotIn(PROMPT[1]["content"], TOK.decode(seg1.context()), "user instruction is not re-inserted")
        self.assertIn(PROMPT[0]["content"], TOK.decode(seg1.context()), "system prompt is kept")
        outs = asyncio.run(_segment_outputs(out, 1.0, False, {}, True))
        self.assertEqual(outs[1].prompt_ids, seg1.chat_ids[0])
        # budget accounting uses each segment's own prompt length
        from agents.compaction_agent import _generated_tokens
        for seg in out["segments"]:
            self.assertLessEqual(_generated_tokens(seg), cfg.response_length)
        # default keeps the task prompt
        out2, _, _ = _run({})
        self.assertEqual(out2["segments"][1].prompt_turn, 2)
        self.assertIn(PROMPT[1]["content"], TOK.decode(out2["segments"][1].context()))

    def test_eval_returns_last_segment_only(self):
        from agents.compaction_agent import _segment_outputs
        out, _, _ = _run({})
        outs = asyncio.run(_segment_outputs(out, 1.0, False, {}, False))
        self.assertEqual(len(outs), 1)
        self.assertEqual(outs[0].extra_fields["is_last_segment"], 1)

    def test_tail_shrinks_when_it_does_not_fit(self):
        # observations are large relative to the budget: k must drop below 2 for the tail to fit
        out, _, _ = _run({"threshold": 700}, response_length=1100, think_words=10)
        self.assertGreaterEqual(len(out["segments"]), 2)
        self.assertLessEqual(out["segment_info"][1]["tail_steps"], 2)


if __name__ == "__main__":
    unittest.main()
