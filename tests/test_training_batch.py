"""Training batch semantics: what the GRPO actor loss actually sees, built by the SAME code path as the trainer
(``verl.experimental.agent_loop.AgentLoopWorker._agent_loop_postprocess``) from Agent outputs, on CPU.

Checks: input_ids = left-padded prompt + right-padded response; attention_mask 1 on real tokens only; position_ids are
cumsum(attention)-1 (contiguous across the prompt/response boundary); response_mask (the actor's loss mask) is 1 exactly on
the sampled ids (incl. eos) and 0 on the generation prompt, the terminator newline, tool observations, the compaction
resume turn and the copied tail (v2 and legacy), and on padding; rollback boundaries (a rolled-back step never appears in
the sample that rolled it back). No GPU trainer is involved: this is the deterministic tensor construction; the GPU-side
batch of a real run was not dumped (NOT YET VERIFIED on device, see docs/PROTOCOL.md)."""
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


def _worker(tok, prompt_length, response_length):
    """A minimal AgentLoopWorker shell exposing the real _agent_loop_postprocess."""
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
    cls = getattr(AgentLoopWorker, "__ray_actor_class__", None) or getattr(getattr(AgentLoopWorker, "__ray_metadata__", None), "modified_class", None) or AgentLoopWorker
    w = cls.__new__(cls)
    w.tokenizer = tok
    w.processor = None
    w.use_reward_loop = False
    w.reward_router_address = None
    w.reward_loop_worker = None
    w.config = types.SimpleNamespace(actor_rollout_ref=types.SimpleNamespace(rollout=types.SimpleNamespace(
        prompt_length=prompt_length, response_length=response_length, multi_turn=types.SimpleNamespace(enable=False))),
        reward_model=types.SimpleNamespace(enable_resource_pool=False, enable=False))
    return w


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
        from verl.utils.model import compute_position_id_with_mask
        out, outs, cfg = self._outputs(protocol)
        w = _worker(TOK, 8192, cfg.response_length)
        for seg, o in zip(out["segments"], outs):
            b = asyncio.run(w._agent_loop_postprocess(o, raw_prompt=None))
            P, R = 8192, cfg.response_length
            self.assertEqual(b.input_ids.shape, (1, P + R))
            n_p, n_r = len(o.prompt_ids), len(o.response_ids)
            # layout: [pad]*(P-n_p) + prompt | response + [pad]*(R-n_r)
            self.assertEqual(b.input_ids[0, P - n_p:P].tolist(), o.prompt_ids)
            self.assertEqual(b.input_ids[0, P:P + n_r].tolist(), o.response_ids)
            am = b.attention_mask[0]
            self.assertEqual(am[:P - n_p].sum().item(), 0); self.assertEqual(am[P - n_p:P + n_r].sum().item(), n_p + n_r); self.assertEqual(am[P + n_r:].sum().item(), 0)
            self.assertTrue(torch.equal(b.position_ids, compute_position_id_with_mask(b.attention_mask)))
            self.assertEqual(b.position_ids[0, P - n_p:P + n_r].tolist(), list(range(n_p + n_r)), "contiguous positions across the prompt/response boundary")
            # loss mask == the Agent's token_mask over the response, padding 0
            rm = b.response_mask[0]
            expect = [1 if m else 0 for turn in seg.token_mask[seg.prompt_turn:] for m in turn][:R]
            self.assertEqual(rm[:n_r].tolist(), expect); self.assertEqual(rm[n_r:].sum().item(), 0)
            # semantics on the segment: every trainable token is a sampled id of a completion turn (never gen prompt / obs / tail / resume)
            for turn_ids, mask, comp, turn in zip(seg.chat_ids, seg.token_mask, seg.chat_completions, seg.chat):
                if comp is None:
                    self.assertFalse(any(mask), f"{turn['role']} turn without a sampled completion must be untrained")
                else:
                    gp = len(seg.get_generation_prompt())
                    self.assertFalse(any(mask[:gp]), "generation prompt untrained")
                    n_sampled = len(comp["choices"][0]["message"]["raw_output_ids"])
                    self.assertEqual(sum(mask), n_sampled, "trained tokens == sampled ids incl. eos")
                    self.assertTrue(all(mask[gp:gp + n_sampled]))
                    self.assertFalse(any(mask[gp + n_sampled:]), "terminator newline untrained")
            self.assertEqual(int(b.response_mask.sum()), o.extra_fields["optimized_tokens"])
        # rollback boundary: a step rolled back before the summary is not in seg0's ids but is in seg1's tail (v2: exact ids)
        n_rolled = out["stats"]["rollback_before_summary"]
        if n_rolled:
            seg0, seg1 = out["segments"][0], out["segments"][1]
            tail_assistant = seg1.chat[3 + 2 * (out["segment_info"][1]["tail_steps"] - 1)]["content"]
            self.assertNotIn(tail_assistant, [t["content"] for t in seg0.chat])

    def test_v2(self):
        self._check("v2")

    def test_legacy(self):
        self._check("legacy")


@unittest.skipIf(TOK is None, "tokenizer not available offline")
class TestBatchFromBranchAgent(unittest.TestCase):
    def test_forked_branch_history_is_untrained_and_branch_sample_layout(self):
        import torch
        from agents.utils import Agent, AgentLoopOutput, AgentLoopMetrics
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
        w = _worker(TOK, 8192, 32768)
        b = asyncio.run(w._agent_loop_postprocess(o, raw_prompt=None))
        self.assertEqual(int(b.response_mask.sum()), n_sampled)
        self.assertTrue(torch.equal(b.position_ids, __import__("verl.utils.model", fromlist=["x"]).compute_position_id_with_mask(b.attention_mask)))


if __name__ == "__main__":
    unittest.main()
