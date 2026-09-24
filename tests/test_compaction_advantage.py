"""CompactionRL advantage estimators (agents/verl_plugin/estimators.py, verl 0.9.1 plugin): cross-trajectory GAE (Eqs. 13-15 of
arXiv:2607.05378) and the protocol-matched group-relative variant. CPU only:
``~/xiaoxuan/envs/fold_train/bin/python -m unittest tests.test_compaction_advantage -v``
"""
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agents.verl_plugin import estimators as core_algos  # noqa: E402  (verl 0.9.1 plugin; bodies = the former fork code)


def _terminal_reward_batch(lengths, rewards, L=12):
    bs = len(lengths)
    r = torch.zeros(bs, L)
    mask = torch.zeros(bs, L)
    for i, (n, rw) in enumerate(zip(lengths, rewards)):
        mask[i, :n] = 1
        r[i, n - 1] = rw
    return r, mask


class TestCompactionGAE(unittest.TestCase):
    def test_position_correction_matches_eq15(self):
        # rollout A = two segments (4 then 3 optimised tokens), rollout B = one segment (5 tokens)
        lengths, rewards = [4, 3, 5], [1.0, 1.0, 0.0]
        tokens_after = torch.tensor([3, 0, 0])
        r, mask = _terminal_reward_batch(lengths, rewards)
        values = torch.zeros_like(r)
        gamma, lam = 1.0, 0.8
        adv, ret = core_algos.compute_compaction_gae_advantage_return(
            r, values, mask, gamma, lam, tokens_after, lam_alpha=None, whiten=False)
        for i, (n, na) in enumerate(zip(lengths, tokens_after.tolist())):
            for t in range(n):
                expected = rewards[i] * (gamma * lam) ** (na + n - 1 - t)   # Eq. 15 distance to the final outcome
                self.assertAlmostEqual(adv[i, t].item(), expected, places=6)
            self.assertTrue(torch.all(adv[i, n:] == 0))
        # critic targets keep the local (uncorrected) return
        self.assertAlmostEqual(ret[0, 0].item(), (gamma * lam) ** 3, places=6)

    def test_length_adaptive_lambda(self):
        lengths, rewards = [4, 8], [1.0, 1.0]
        r, mask = _terminal_reward_batch(lengths, rewards)
        adv, _ = core_algos.compute_compaction_gae_advantage_return(
            r, torch.zeros_like(r), mask, 1.0, 1.0, torch.tensor([0, 0]), lam_alpha=1.5, whiten=False)
        for i, n in enumerate(lengths):
            lam_i = 1 - 1 / (1.5 * n)
            self.assertAlmostEqual(adv[i, 0].item(), lam_i ** (n - 1), places=6)

    def test_observation_tokens_are_skipped(self):
        r = torch.zeros(1, 6); mask = torch.tensor([[1, 1, 0, 0, 1, 1]], dtype=torch.float)
        r[0, 5] = 1.0
        adv, _ = core_algos.compute_compaction_gae_advantage_return(
            r, torch.zeros_like(r), mask, 1.0, 0.5, torch.tensor([0]), whiten=False)
        self.assertAlmostEqual(adv[0, 1].item(), 0.5 ** 2, places=6)   # two optimised tokens after position 1
        self.assertEqual(adv[0, 2].item(), 0.0)

    def test_whitening_keeps_zero_on_padding(self):
        r, mask = _terminal_reward_batch([4, 6], [1.0, 0.0])
        adv, _ = core_algos.compute_compaction_gae_advantage_return(
            r, torch.zeros_like(r), mask, 1.0, 0.9, torch.tensor([0, 0]), whiten=True)
        self.assertTrue(torch.all(adv[mask == 0] == 0))


class TestGlobalTokenMean(unittest.TestCase):
    def test_verl_token_mean_is_global(self):
        """verl 0.9.1 agg_loss(token-mean) normalises every micro-batch by the global mini-batch token count (the
        paper's token-level loss, formerly our global_token_mean patch): summing the DP-averaged micro-batch losses
        equals the plain mean over all tokens, whatever the segment lengths."""
        from verl.trainer.ppo.core_algos import agg_loss
        torch.manual_seed(0)
        lengths = [5, 50, 500]
        losses = [torch.rand(1, n) for n in lengths]
        global_tokens = sum(lengths)
        expected = torch.cat(losses, dim=1).sum() / global_tokens
        for dp_size in (1, 4):
            total = sum(agg_loss(l, torch.ones_like(l), "token-mean", dp_size=dp_size, batch_num_tokens=global_tokens)
                        for l in losses) / dp_size
            self.assertAlmostEqual(total.item(), expected.item(), places=5)
        old = sum(l.mean() for l in losses) / len(losses)     # per-segment weighting differs whenever lengths differ
        self.assertNotAlmostEqual(old.item(), expected.item(), places=3)


class TestCompactionGRPO(unittest.TestCase):
    def test_rollout_level_groups_and_broadcast(self):
        # prompt p1: rollout a (2 segments, reward 1), rollout b (1 segment, reward 0)
        lengths, rewards = [4, 3, 5], [1.0, 1.0, 0.0]
        r, mask = _terminal_reward_batch(lengths, rewards)
        index = np.array(["p1", "p1", "p1"]); gen_uid = np.array(["a", "a", "b"])
        adv, _ = core_algos.compute_compaction_grpo_advantage(r, mask, index, gen_uid, torch.tensor([3, 0, 0]))
        std = torch.std(torch.tensor([1.0, 0.0]))
        expected_a = (1.0 - 0.5) / (std + 1e-6)
        self.assertAlmostEqual(adv[0, 0].item(), expected_a.item(), places=5)
        self.assertAlmostEqual(adv[1, 0].item(), expected_a.item(), places=5, msg="segments of one rollout share the advantage")
        self.assertAlmostEqual(adv[2, 0].item(), -expected_a.item(), places=5)
        self.assertTrue(torch.all(adv[mask == 0] == 0))

    def test_position_correction_identity_at_gamma_lam_one_and_active_otherwise(self):
        lengths, rewards = [4, 3], [1.0, 1.0]
        r, mask = _terminal_reward_batch(lengths, rewards)
        index = np.array(["p", "p"]); gen_uid = np.array(["a", "a"])
        adv1, _ = core_algos.compute_compaction_grpo_advantage(r, mask, index, gen_uid, torch.tensor([3, 0]), norm_adv_by_std_in_grpo=False)
        self.assertAlmostEqual(adv1[0, 0].item(), adv1[1, 0].item(), places=6)   # single rollout: mean 0 -> both 0
        index = np.array(["p", "p", "p"]); gen_uid = np.array(["a", "a", "b"])
        r, mask = _terminal_reward_batch([4, 3, 5], [1.0, 1.0, 0.0])
        adv, _ = core_algos.compute_compaction_grpo_advantage(r, mask, index, gen_uid, torch.tensor([3, 0, 0]), lam=0.5, norm_adv_by_std_in_grpo=False)
        self.assertAlmostEqual(adv[0, 0].item(), 0.5 * 0.5 ** 3, places=6)
        self.assertAlmostEqual(adv[1, 0].item(), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
