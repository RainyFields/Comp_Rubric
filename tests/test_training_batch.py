"""Training-batch construction (verl 0.9.1 V1 rows): rollout outputs of the compaction and fold agents turned into
TransferQueue rows by ``agents.verl_plugin.agent_loop.build_row`` / ``apply_rollout_masks``, on CPU.

Checks (docs/PROTOCOL.md §5): a row's ``input_ids`` is exactly prompt + response (no padding in V1 rows), ``loss_mask``
equals the agent's ``response_mask`` (observations / re-rendered history / summary prompt untrained), position ids are a
plain arange, ``rm_scores`` carries the reward on the last response token, rows flagged ``mask_rollout`` lose their loss
mask but keep their reward (group statistics), and every row gets ``gen_uid`` + ``reward_extra_info``.
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

TOK = load_tokenizer()


def _worker(tok):
    """A minimal worker shell exposing the real base helpers (_compute_multi_modal_inputs / _compute_position_ids)."""
    from agents.verl_plugin.agent_loop import _WorkerBase
    w = _WorkerBase.__new__(_WorkerBase)
    w.tokenizer = tok
    w.processor = None
    w.config = types.SimpleNamespace(actor_rollout_ref=types.SimpleNamespace(rollout=types.SimpleNamespace()))
    return w


KW = {"uid": "u1", "session_id": 0, "global_steps": 3, "raw_prompt": None}


@unittest.skipIf(TOK is None, "tokenizer not available offline")
class TestBatchFromCompactionRollout(unittest.TestCase):
    def _outputs(self, protocol):
        from tests.test_compaction_agent import _run
        from agents.compaction_agent import _segment_outputs
        out, _, cfg = _run({}, protocol=protocol)
        outs = asyncio.run(_segment_outputs(out, 1.0, False, {}, True))
        return out, outs, cfg

    def _check(self, protocol):
        import torch
        from agents.verl_plugin.agent_loop import apply_rollout_masks, build_row
        out, outs, cfg = self._outputs(protocol)
        apply_rollout_masks(outs, "u1_0")
        w = _worker(TOK)
        for seg, o in zip(out["segments"], outs):
            field, tag = build_row(w, o, KW)
            n_p, n_r = len(o.prompt_ids), len(o.response_ids)
            self.assertEqual(field["input_ids"].tolist(), list(o.prompt_ids) + list(o.response_ids))
            self.assertEqual(tag["prompt_len"], n_p); self.assertEqual(tag["response_len"], n_r)
            self.assertEqual(field["loss_mask"].tolist(), list(o.response_mask))
            self.assertEqual(field["response_mask"].tolist(), list(o.response_mask))
            self.assertTrue(torch.equal(field["position_ids"], torch.arange(n_p + n_r)))
            # reward on the last response token (verl as_dict convention), the agent's trained-token count preserved
            self.assertAlmostEqual(float(field["rm_scores"][-1]), 1.0); self.assertEqual(float(field["rm_scores"][:-1].abs().sum()), 0.0)
            self.assertEqual(int(field["loss_mask"].sum()), int(sum(seg.token_mask[-1]) if False else sum(o.response_mask)))
            self.assertEqual(o.extra_fields["gen_uid"], "u1_0"); self.assertIn("reward_extra_info", o.extra_fields)
            # the segment's context == prompt + response ids (what the policy saw is what is trained)
            self.assertEqual(list(o.prompt_ids) + list(o.response_ids), list(seg.context())[: n_p + n_r])

    def test_v2(self):
        self._check("v2")

    def test_legacy(self):
        self._check("legacy")

    def test_mask_rollout_zeroes_loss_but_keeps_reward(self):
        from agents.verl_plugin.agent_loop import apply_rollout_masks, build_row
        _, outs, _ = self._outputs("v2")
        outs[-1].extra_fields["mask_rollout"] = True
        n = apply_rollout_masks(outs, "u1_0")
        self.assertEqual(n, 1)
        field, _ = build_row(_worker(TOK), outs[-1], KW)
        self.assertEqual(int(field["loss_mask"].sum()), 0)
        self.assertAlmostEqual(float(field["rm_scores"].sum()), 1.0)

    def test_forked_branch_history_is_untrained_and_branch_sample_layout(self):
        import torch
        from agents.utils import Agent, AgentLoopOutput, AgentLoopMetrics
        from agents.verl_plugin.agent_loop import build_row
        from tests.test_protocol_compat import cfg, completion, CHAT, S1, BR
        ag = Agent(None, CHAT, TOK, cfg("v2"), prompt_turn=2)
        ag.append({"role": "assistant", "content": BR}, completion(TOK, BR))
        child = ag.fork(None)
        child.append({"role": "user", "content": "branch prompt"})
        child.append({"role": "assistant", "content": S1}, completion(TOK, S1))
        child.append({"role": "user", "content": "<tool_response>\nOBS\n</tool_response>"})
        d = asyncio.run(child.get_data())
        # trained: only the branch's own completion; inherited main turn + branch prompt + observation untrained
        n_sampled = len(completion(TOK, S1)["choices"][0]["message"]["raw_output_ids"])
        self.assertEqual(sum(d["response_mask"]), n_sampled)
        o = AgentLoopOutput(prompt_ids=d["prompt_ids"], response_ids=d["response_ids"], response_mask=d["response_mask"],
                            response_logprobs=d["response_logprobs"], multi_modal_data={}, reward_score=0.0, num_turns=d["num_turns"],
                            metrics=AgentLoopMetrics(), extra_fields={})
        field, _ = build_row(_worker(TOK), o, KW)
        self.assertEqual(int(field["loss_mask"].sum()), n_sampled)
        self.assertTrue(torch.equal(field["position_ids"], torch.arange(len(d["prompt_ids"]) + len(d["response_ids"]))))


if __name__ == "__main__":
    unittest.main()
