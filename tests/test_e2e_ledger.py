"""CPU test for agents/e2e_ledger.py: main thread + one branch, exact token accounting."""
import asyncio, json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_qwen3_tool_response_wrapping import TOK, CFG, SYSTEM_USER, _completion


@unittest.skipIf(TOK is None, "Qwen3 tokenizer not available offline")
class TestE2ELedger(unittest.TestCase):
    def _agents(self):
        from agents.utils import Agent, wrap_tool_response
        from agents.prompts import BRANCH_MESSAGE_SEARCH

        class LLM:
            def __init__(self, script): self.script = list(script)
            async def create_completion(self, input_ids, **kw):
                return _completion(TOK, "t " * 5, self.script.pop(0))[1]

        main = Agent(LLM(["<function=branch><parameter=prompt>find x</parameter></function>",
                          "<function=finish><parameter=answer>42</parameter></function>"]),
                     list(SYSTEM_USER), TOK, CFG, prompt_turn=2)
        asyncio.run(main.step())                                   # branch call
        history = main.messages()
        br = Agent(LLM(["<function=search><parameter=query>q</parameter></function>",
                        "<function=return><parameter=message>x=42</parameter></function>"]),
                   history, TOK, CFG, prompt_turn=2)
        br.append({'role': 'user', 'content': BRANCH_MESSAGE_SEARCH.format(message='find x')})
        async def act(_): return "[Search Results] doc doc"
        asyncio.run(br.react(act, max_turn=5, should_continue=lambda r: '<function=return>' not in r))
        main.append({'role': 'user', 'content': wrap_tool_response('Branch has finished its task, the returned message is:\n\nx=42')})
        asyncio.run(main.step())                                   # finish
        return {'main': main, '#0-find_x': br}

    def test_ledger_accounting(self):
        from agents.e2e_ledger import build_ledger, write_ledger
        agents = self._agents()
        led = build_ledger(agents, prompt_turn=2)
        m, b = led[0], led[1]
        self.assertEqual([t['kind'] for t in m['turns']], ['prompt', 'prompt', 'branch', 'branch_return_obs', 'finish'])
        self.assertEqual(b['inherited_turns'], 3)                  # system, user, branch call
        self.assertEqual([t['kind'] for t in b['turns'][3:]], ['branch_prompt', 'search', 'obs', 'return'])
        # inherited tokens of the branch ~ main context at the branch call (assistant turn re-rendered by the
        # template instead of raw sampled ids: +-1 token of framing). The ledger reports what the branch saw.
        self.assertAlmostEqual(b['inherited_tokens'], sum(t['n'] for t in m['turns'][:3]), delta=2)
        # every generated assistant turn records the context it was fed; it grows monotonically
        gens = [t for t in b['turns'] if t['role'] == 'assistant' and not t['inherited']]
        self.assertTrue(all(t['gen'] > 0 and t['ctx_before'] > 0 for t in gens))
        self.assertLess(gens[0]['ctx_before'], gens[1]['ctx_before'])
        self.assertEqual(gens[0]['ctx_before'], b['inherited_tokens'] + b['turns'][3]['n'] + len(agents['#0-find_x'].get_generation_prompt()))
        # final_context == what context() returns
        self.assertEqual(m['final_context'], len(agents['main'].context()))
        self.assertEqual(b['final_context'], len(agents['#0-find_x'].context()))
        # writer: gated by env var, one line per rollout
        with tempfile.TemporaryDirectory() as d:
            os.environ['FOLD_E2E_LEDGER_DIR'] = d
            write_ledger({'instance_id': '1', 'agents': led})
            del os.environ['FOLD_E2E_LEDGER_DIR']
            files = os.listdir(d); self.assertEqual(len(files), 1)
            rec = json.loads(open(os.path.join(d, files[0])).readline())
            self.assertEqual(len(rec['agents']), 2); self.assertIn('wall_time', rec)
        write_ledger({'x': 1})   # unset env -> silent no-op

if __name__ == '__main__':
    unittest.main()
