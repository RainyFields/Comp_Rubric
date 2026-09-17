import os, sys, unittest, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_e2e_ledger import TestE2ELedger, TOK


@unittest.skipIf(TOK is None, "Qwen3 tokenizer not available offline")
class TestE2EMetrics(unittest.TestCase):
    def test_rollout_metrics(self):
        from agents.e2e_ledger import build_ledger
        from scripts.e2e_metrics import rollout_metrics, _events
        agents = TestE2ELedger()._agents()
        rec = {'instance_id': '1', 'uid': 'u', 'gen_uid': 'g', 'workflow': 'search_branch', 'score': 1.0, 'is_finish': True,
               'stop_reason': 'finish', 'n_branches': 1, 'response_length': 32768, 'agents': build_ledger(agents, 2)}
        ev = _events(rec)
        self.assertEqual([e.get('kind') for e in ev], ['prompt', 'prompt', 'branch', 'branch_prompt', 'search', 'obs', 'return', 'fold', 'branch_return_obs', 'finish'])
        m = rollout_metrics(rec)
        self.assertEqual(m['turns'], 4); self.assertEqual(m['tool_calls'], 1); self.assertEqual(m['branch_calls'], 1)
        self.assertEqual(m['has_fold'], 1); self.assertEqual(m['post_turns'], 1); self.assertEqual(m['post_tool_calls'], 0)
        self.assertEqual(m['pre_turns'], 3)
        gen = sum(t['gen'] for a in rec['agents'] for t in a['turns'] if t['role'] == 'assistant' and not t['inherited'])
        self.assertEqual(m['gen_tokens'], gen)
        self.assertEqual(m['forward_passes'], 4)
        self.assertGreaterEqual(m['peak_ctx'], m['mean_ctx'])
        self.assertEqual(m['total_e2e_tokens'], m['prompt_tokens'] + m['gen_tokens'] + m['frame_tokens'] + m['obs_tokens'] + m['comp_in_tokens'])
        self.assertGreater(m['comp_in_tokens'], 0); self.assertGreater(m['comp_gen_tokens'], 0)
        # no-fold rollout -> post metrics NaN
        rec2 = dict(rec, agents=rec['agents'][:1]); m2 = rollout_metrics(rec2)
        self.assertEqual(m2['has_fold'], 0); self.assertTrue(math.isnan(m2['post_turns']))

if __name__ == '__main__':
    unittest.main()
