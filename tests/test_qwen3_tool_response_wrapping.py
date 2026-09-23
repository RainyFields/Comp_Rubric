"""Regression tests: observations must survive Qwen3's chat-template think-stripping.

Background
----------
``AgentContext.get_turn_context`` tokenizes a text-only turn (observations) by rendering
``chat[:i]`` and ``chat[:i+1]`` with the chat template and slicing the difference. Qwen3's
template strips every earlier assistant ``<think>...</think>`` block whenever the last message
is a plain user message (a "new query"). The two renders then no longer share a prefix, and the
slice is offset by the length of the previous think block: the observation loses its header and
its first tokens, or disappears entirely when the think is longer than the observation.

The fix wraps environment/tool observations as ``<tool_response>...</tool_response>``, which the
template treats as tool output belonging to the current query, so prior reasoning is preserved
and the diff stays aligned. Genuine instructions (branch task prompt, summary prompt) stay
unwrapped and remain new-query boundaries.

These tests use the real Qwen3 tokenizer + chat template (identical across Qwen3 dense models).
They run offline: ``python -m unittest tests.test_qwen3_tool_response_wrapping -v``
Set ``QWEN3_TOKENIZER_PATH`` to point at any Qwen3 tokenizer directory or hub id.
"""
import asyncio
import os
import re
import sys
import types
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tests.tokenizers import load_tokenizer, profile_of  # noqa: E402

_load_qwen3_tokenizer = load_tokenizer   # legacy name used by the other test modules

TOK = load_tokenizer()
PROFILE = profile_of(TOK) if TOK is not None else None
CFG = types.SimpleNamespace(prompt_length=8192, response_length=32768, plugin=types.SimpleNamespace(retry_cjk=0))
SYSTEM_USER = [{"role": "system", "content": "You are a search agent."},
               {"role": "user", "content": "Q: which theater was built by a local during the Depression?"}]


def _completion(tok, think_body, action):
    """Build a CallLLM-style completion (raw sampled ids + decoded text) with a think block.

    Qwen3 samples ``<think>…</think>``; Qwen3.5 pre-fills ``<think>\n`` in the generation prompt, so its samples
    start mid-thought (no opening tag) — mimic whichever family the tokenizer belongs to.
    """
    opener = "" if PROFILE.think_prefilled else "<think>\n"
    text = f"{opener}{think_body}\n</think>\n\n{action}"
    ids = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
    decoded = tok.decode(ids, skip_special_tokens=True)
    return decoded, {"choices": [{"message": {"content": decoded, "raw_output_ids": ids,
                                              "response_log_probs": [0.0] * len(ids)}}]}


_ANCHOR = [{"role": "user", "content": "anchor"}]


def _render_single_user_turn(tok, content):
    """Reference tokens for one user turn rendered on its own by the template (behind an anchor query, because
    Qwen3.5's template refuses a chat whose only user message is a <tool_response>)."""
    full = tok.apply_chat_template(_ANCHOR + [{"role": "user", "content": content}], add_generation_prompt=False, tokenize=True, return_dict=False)
    prev = tok.apply_chat_template(_ANCHOR, add_generation_prompt=False, tokenize=True, return_dict=False)
    assert full[:len(prev)] == prev
    return full[len(prev):]


@unittest.skipIf(TOK is None, "Qwen3 tokenizer not available offline; set QWEN3_TOKENIZER_PATH")
class TestToolResponseWrapping(unittest.TestCase):
    N_TURNS = 3

    def _run_turns(self, wrap):
        from agents.utils import AgentContext, wrap_tool_response
        ctx = AgentContext(SYSTEM_USER, TOK, CFG, prompt_turn=2)
        thinks, observations, stored_turns = [], [], []
        for t in range(self.N_TURNS):
            think = f"THINK{t} " + ("let me reason carefully about hop %d. " % t) * (60 + 40 * t)
            decoded, comp = _completion(TOK, think, f"<function=search>\n<parameter=query>q{t}</parameter>\n</function>")
            ctx.append({"role": "assistant", "content": decoded}, comp)
            obs = f'[Search Results for "q{t}"]\n' + "\n".join(f"--- #{i}: doc{t}{i} --- content about hop {t}" for i in range(8))
            content = wrap_tool_response(obs) if wrap else obs
            ctx.append({"role": "user", "content": content})
            thinks.append(f"THINK{t}")
            observations.append(obs)
            stored_turns.append((content, list(ctx.chat_ids[-1])))
        return ctx, thinks, observations, stored_turns

    # 1. Observation integrity ---------------------------------------------------------------
    def test_observation_integrity_across_turns(self):
        ctx, _, observations, stored = self._run_turns(wrap=True)
        context_text = TOK.decode(ctx.context())
        for obs in observations:
            self.assertIn(obs, context_text, "complete observation must be present in the model-facing context")
        for content, ids in stored:
            self.assertIn(content, TOK.decode(ids), "stored user-turn tokens must contain the full wrapped observation")

    # 2. Think preservation across tool turns ------------------------------------------------
    def test_think_blocks_preserved_across_tool_turns(self):
        ctx, thinks, _, _ = self._run_turns(wrap=True)
        context_text = TOK.decode(ctx.context())
        # one per assistant turn (sampled by Qwen3 / pre-filled by Qwen3.5) + the pre-filled opener of the pending generation prompt
        self.assertEqual(context_text.count("<think>"), self.N_TURNS + int(PROFILE.think_prefilled))
        for marker in thinks:
            self.assertIn(marker, context_text)
        # Template semantics: with tool responses the query boundary does not move, so even a
        # fresh text render of the same chat keeps every earlier think block.
        rendered = TOK.apply_chat_template(ctx.chat, add_generation_prompt=True, tokenize=False)
        for marker in thinks:
            self.assertIn(marker, rendered)

    # 3. Prefix/diff correctness -------------------------------------------------------------
    def test_incremental_turn_tokens_match_template_render(self):
        _, _, _, stored = self._run_turns(wrap=True)
        for content, ids in stored:
            expected = _render_single_user_turn(TOK, content)
            self.assertEqual(ids, expected, "get_turn_context diff must equal the template's own rendering of the turn")

    def test_full_text_chat_concatenation_matches_direct_render(self):
        """Text-only path: chat_ids built turn-by-turn must concatenate to the direct render."""
        from agents.utils import AgentContext, wrap_tool_response
        chat = list(SYSTEM_USER)
        for t in range(self.N_TURNS):
            chat.append({"role": "assistant", "content": f"<think>\nTHINK{t} " + "reason " * 100 + f"\n</think>\n\n<function=search><parameter=query>q{t}</parameter></function>"})
            chat.append({"role": "user", "content": wrap_tool_response(f"[Search Results for q{t}] " + "doc " * 20)})
        ctx = AgentContext(chat, TOK, CFG, prompt_turn=2)
        built = sum(ctx.chat_ids, [])
        direct = TOK.apply_chat_template(chat, add_generation_prompt=False, tokenize=True, return_dict=False)
        self.assertEqual(built, direct)

    def test_unwrapped_observation_is_also_intact(self):
        """Standalone per-turn rendering makes the slice independent of the query boundary, so
        even a plain (unwrapped) observation keeps its header and all its tokens."""
        _, _, _, stored = self._run_turns(wrap=False)
        for content, ids in stored:
            self.assertEqual(ids, _render_single_user_turn(TOK, content))
            self.assertIn("<|im_start|>user", TOK.decode(ids))

    def test_prefix_diff_would_truncate_plain_observation(self):
        """Guard that the underlying template hazard is real, so the tests above stay meaningful:
        a chat[:i+1] minus chat[:i] diff loses the observation after a longer think block."""
        chat = list(SYSTEM_USER) + [
            {"role": "assistant", "content": "<think>\n" + "reason " * 200 + "\n</think>\n\n<function=search><parameter=query>q</parameter></function>"},
            {"role": "user", "content": "[Search Results for q] doc doc doc"},
        ]
        full = TOK.apply_chat_template(chat, add_generation_prompt=False, tokenize=True, return_dict=False)
        prev = TOK.apply_chat_template(chat[:-1], add_generation_prompt=False, tokenize=True, return_dict=False)
        self.assertLess(len(full[len(prev):]), len(_render_single_user_turn(TOK, chat[-1]["content"])))

    def test_branch_and_summary_prompts_survive_after_think_turns(self):
        """The unwrapped instructions appended after wrapped tool turns move the query boundary;
        the per-turn tokens must still contain the whole instruction (regression: they were
        sliced to zero tokens when earlier think blocks outweighed the prompt)."""
        from agents.prompts import SUMMARY_PROMPT_SEARCH, BRANCH_MESSAGE_SEARCH
        from agents.utils import AgentContext
        ctx, thinks, _, _ = self._run_turns(wrap=True)
        branch_prompt = BRANCH_MESSAGE_SEARCH.format(message="find the traffic-hours ranking")
        branch = AgentContext(ctx.messages(), TOK, CFG, prompt_turn=2)  # branch forks main's history
        branch.append({"role": "user", "content": branch_prompt})
        self.assertEqual(branch.chat_ids[-1], _render_single_user_turn(TOK, branch_prompt))
        branch_text = TOK.decode(branch.context())
        self.assertIn(branch_prompt, branch_text)
        for marker in thinks:  # forked history keeps main's reasoning, as main itself saw it
            self.assertIn(marker, branch_text)
        ctx.append({"role": "assistant", "content": ""})
        ctx.append({"role": "user", "content": SUMMARY_PROMPT_SEARCH})
        self.assertEqual(ctx.chat_ids[-1], _render_single_user_turn(TOK, SUMMARY_PROMPT_SEARCH))
        self.assertIn(SUMMARY_PROMPT_SEARCH, TOK.decode(ctx.context()))

    # 4. New-query boundary remains intact ---------------------------------------------------
    def test_genuine_instruction_stays_new_query_boundary(self):
        from agents.prompts import SUMMARY_PROMPT_SEARCH, BRANCH_MESSAGE_SEARCH
        ctx, thinks, _, _ = self._run_turns(wrap=True)
        summary_prompt = SUMMARY_PROMPT_SEARCH
        ctx.append({"role": "assistant", "content": ""})
        ctx.append({"role": "user", "content": summary_prompt})
        self.assertFalse(ctx.chat[-1]["content"].startswith("<tool_response>"))
        rendered = TOK.apply_chat_template(ctx.chat, add_generation_prompt=True, tokenize=False)
        self.assertIn(summary_prompt, rendered)
        for marker in thinks:  # a genuine instruction moves the query boundary: earlier thinks are stripped
            self.assertNotIn(marker, rendered)
        branch_prompt = BRANCH_MESSAGE_SEARCH.format(message="find the traffic-hours ranking")
        self.assertFalse(branch_prompt.startswith("<tool_response>"))

    # End-to-end through Agent.react(): observations wrapped, summary prompt not ---------------
    def test_react_wraps_observations_but_not_summary_prompt(self):
        from agents.utils import Agent

        class FakeLLM:
            def __init__(self):
                self.n = 0
            async def create_completion(self, input_ids, **kwargs):
                t = self.n
                self.n += 1
                if t < 2:
                    return _completion(TOK, f"THINK{t} " + "reason " * 80, f"<function=search><parameter=query>q{t}</parameter></function>")[1]
                return _completion(TOK, "", "<function=return><parameter=message>done</parameter></function>")[1]

        async def fake_run_action(response):
            return "[Search Results] " + "doc " * 10

        agent = Agent(FakeLLM(), list(SYSTEM_USER), TOK, CFG, prompt_turn=2)
        asyncio.run(agent.react(fake_run_action, max_turn=10,
                                should_continue=lambda r: "<function=return>" not in r,
                                observation_prompt="* You are now in branch mode."))
        user_turns = [m["content"] for m in agent.chat[2:] if m["role"] == "user"]
        self.assertEqual(len(user_turns), 2)
        for c in user_turns:
            self.assertTrue(c.startswith("<tool_response>\n") and c.endswith("\n</tool_response>"))
            self.assertIn("* You are now in branch mode.", c)
        for i, m in enumerate(agent.chat):
            if m["role"] == "user" and i >= 2:
                self.assertEqual(agent.chat_ids[i], _render_single_user_turn(TOK, m["content"]))
        # summary path: a genuine instruction appended by react() must stay unwrapped
        class SummaryLLM(FakeLLM):
            async def create_completion(self, input_ids, **kwargs):
                if self.n == 0:
                    self.n += 1
                    return _completion(TOK, "THINK0 " + "reason " * 80, "<function=search><parameter=query>q</parameter></function>")[1]
                return _completion(TOK, "", "<summary>progress</summary>")[1]
        agent = Agent(SummaryLLM(), list(SYSTEM_USER), TOK, CFG, prompt_turn=2)
        asyncio.run(agent.react(fake_run_action, max_turn=1, summary_prompt="Please summarize the sub task progress."))
        self.assertEqual(agent.chat[-2]["role"], "user")
        self.assertEqual(agent.chat[-2]["content"], "Please summarize the sub task progress.")


class TestFoldAgentSourceGuard(unittest.TestCase):
    """process_item is not unit-testable without the RL stack; guard the call sites by source."""

    def setUp(self):
        self.src = open(os.path.join(REPO, "agents", "fold_agent.py")).read()

    def test_main_agent_observation_is_wrapped(self):
        self.assertIn("agent['main'].append({'role': 'user', 'content': wrap_tool_response(observation)})", self.src)
        self.assertNotIn("agent['main'].append({'role': 'user', 'content': observation})", self.src)

    def test_instruction_prompts_are_not_wrapped(self):
        self.assertIn("agent[current].append({'role': 'user', 'content': summary_prompt})", self.src)
        self.assertIn("agent[current].append({'role': 'user', 'content': next_session_prompt})", self.src)
        self.assertIn("agent[agent_name].append({'role': 'user', 'content': branch_prompt_formatted})", self.src)
        self.assertEqual(len(re.findall(r"wrap_tool_response\(", self.src)), 1, "exactly one call site in fold_agent.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
