"""Qwen3.5 tokenizer/template behaviour, checked against the ACTUAL Qwen3.5-9B chat template (offline copy at
~/xiaoxuan/tokenizers/Qwen3.5-9B or fold-job-assets/tokenizers/Qwen3.5-9B). These are TOKENIZER/TEMPLATE tests only:
no Qwen3.5 checkpoint runs on this devbox; the inference-side checks are done by the capture audit on a real run.
Every assertion is also executed for the Qwen3-8B template so the two families' differences are explicit."""
import os
import sys
import types
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agents.model_profile import detect_profile  # noqa: E402
from agents.utils import Agent, wrap_tool_response  # noqa: E402

CANDS35 = [os.environ.get("QWEN35_TOKENIZER_PATH"), os.path.expanduser("~/xiaoxuan/tokenizers/Qwen3.5-9B"),
           "/mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets/tokenizers/Qwen3.5-9B"]
CANDS3 = [os.environ.get("QWEN3_TOKENIZER_PATH"), "Qwen/Qwen3-8B",
          "/mnt/hdfs/mlsys/xiaoxuan/fold_replication/ckpt_fix/grpo/global_step_50/actor/huggingface"]


def _load(cands):
    from transformers import AutoTokenizer
    for c in cands:
        if not c:
            continue
        try:
            return AutoTokenizer.from_pretrained(c)
        except Exception:
            continue
    return None


T35, T3 = _load(CANDS35), _load(CANDS3)
CFG = types.SimpleNamespace(prompt_length=8192, response_length=32768, plugin=types.SimpleNamespace(retry_cjk=0))
SYS = {"role": "system", "content": "SYS"}
USR = {"role": "user", "content": "QUESTION"}
A_THINK = {"role": "assistant", "content": "<think>\nreason-1\n</think>\n\n<function=search>\n<parameter=query>q</parameter>\n</function>"}
OBS = {"role": "user", "content": wrap_tool_response("OBS")}
PLAIN = {"role": "user", "content": "PLAIN INSTRUCTION"}


def _render(tok, msgs, gen=False, **kw):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, **kw)


class _Common:
    tok = None
    prefilled = None

    def test_profile(self):
        p = detect_profile(self.tok)
        self.assertTrue(p.strips_history_think)
        self.assertEqual(p.think_prefilled, self.prefilled)

    def test_plain_user_turn_strips_history_think_but_tool_response_keeps_it(self):
        self.assertNotIn("reason-1", _render(self.tok, [SYS, USR, A_THINK, PLAIN], gen=True))
        self.assertIn("<think>\nreason-1\n</think>", _render(self.tok, [SYS, USR, A_THINK, OBS], gen=True))

    def test_generation_prompt(self):
        base = _render(self.tok, [SYS, USR])
        gp = _render(self.tok, [SYS, USR], gen=True)[len(base):]
        self.assertEqual(gp, "<|im_start|>assistant\n<think>\n" if self.prefilled else "<|im_start|>assistant\n")
        nt = _render(self.tok, [SYS, USR], gen=True, enable_thinking=False)[len(base):]
        self.assertEqual(nt, "<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def test_agent_context_equals_canonical_render_for_a_sampled_turn(self):
        ag = Agent(None, [SYS, USR], self.tok, CFG, prompt_turn=2)
        opener = "" if self.prefilled else "<think>\n"
        text = opener + "reason-1\n</think>\n\n<function=search>\n<parameter=query>q</parameter>\n</function>"
        ids = self.tok.encode(text, add_special_tokens=False) + [self.tok.eos_token_id]
        comp = {"choices": [{"message": {"content": self.tok.decode(ids, skip_special_tokens=True), "raw_output_ids": ids,
                                         "response_log_probs": [0.0] * len(ids)}}]}
        ag.append({"role": "assistant", "content": text}, comp)
        ag.append(OBS)
        self.assertEqual(ag.context(), self.tok.apply_chat_template(ag.chat, add_generation_prompt=True, tokenize=True, return_dict=False))
        self.assertEqual(ag._turn_end_newline_ids(), self.tok.encode("\n", add_special_tokens=False))

    def test_system_only_prefix_and_generation_prompt_for_a_resumed_segment(self):
        # Qwen3.5 raises 'No user query found' for [system] alone; Agent must still tokenize prompt turns and the gen prompt
        ag = Agent(None, [SYS], self.tok, CFG, prompt_turn=1)
        self.assertEqual(sum(ag.chat_ids, []), self.tok.apply_chat_template([SYS, {"role": "user", "content": "anchor"}], tokenize=True, return_dict=False)[: len(sum(ag.chat_ids, []))])
        self.assertTrue(len(ag.get_generation_prompt()) > 0)

    def test_special_tokens_are_single_tokens(self):
        for t in ("<think>", "</think>", "<tool_response>", "</tool_response>", "<|im_end|>"):
            self.assertEqual(len(self.tok.encode(t, add_special_tokens=False)), 1, t)


@unittest.skipIf(T35 is None, "Qwen3.5-9B tokenizer not available offline (QWEN35_TOKENIZER_PATH)")
class TestQwen35Template(_Common, unittest.TestCase):
    tok, prefilled = T35, True

    def test_qwen35_specifics(self):
        self.assertIn("No user query found", self.tok.chat_template)      # the prefix-render guard is needed
        self.assertIn("<|im_start|>assistant\n<think>\n", _render(self.tok, [SYS, USR], gen=True))
        with self.assertRaises(Exception):
            self.tok.apply_chat_template([SYS], tokenize=False)


@unittest.skipIf(T3 is None, "Qwen3-8B tokenizer not available offline")
class TestQwen3Template(_Common, unittest.TestCase):
    tok, prefilled = T3, False


if __name__ == "__main__":
    unittest.main()
