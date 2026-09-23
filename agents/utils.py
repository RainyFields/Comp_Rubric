import os
import time
import copy
import uuid
from unittest.mock import patch
from itertools import groupby
import re, unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, AutoTokenizer
import aiohttp
import torch
from pydantic import BaseModel
from typing import Any, Optional
import asyncio, httpx
from envs.local_search import LocalSearch
from agents.parsing import parse_actions
from envs.repo_env import GymEnv


def select_env(ability, config, extra_info=None):
    # Select env
    if 'LocalSearch' in ability:
        EnvClass = LocalSearch
    elif ability.startswith('SWEModalEnv@') or 'SWEModalEnv' in ability:
        from envs.swe_modal_env import SWEModalEnv
        EnvClass = SWEModalEnv
    else:
        EnvClass = GymEnv
    return EnvClass


async def call_openai(messages, model='gpt-5-nano', max_retries=3):
    openai_url = os.getenv("OPENAI_URL")
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=300.0) as c:
                r = await c.post(openai_url, json={
                    "model": model,
                    "messages": messages
                })
                r.raise_for_status()
                return r.json()["content"]
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[CALL OPENAI] Error after {max_retries} attempts: {str(e)}")
                return f"Error after {max_retries} attempts: {str(e)}"
            await asyncio.sleep(1 * (attempt + 1))
    return ""

def decode_conversation(input_ids: list[int], tokenizer) -> tuple[list[dict[str, str]], str]:
    decoded_str = tokenizer.decode(input_ids, skip_special_tokens=False)
    pattern = re.compile(
        re.escape(tokenizer.bos_token)
        + r'(system|user|assistant|tool)\n'
        + r'(.*?)'
        + r'(?=' + re.escape(tokenizer.eos_token) + r')',
        re.DOTALL,
    )
    matches = pattern.findall(decoded_str)
    conversation = [{'role': role, 'content': content} for role, content in matches]
    return conversation, decoded_str

def truncate_text(
        text: str,
        max_lines: int | None = None,
        max_length: int | None = None,
        merge_repeat: bool = False,
        merge_num: int = 128,
        keep_tail_lines: int = 5,
) -> str:
    lines = text.splitlines()

    # 1) Merge repeated lines if requested
    if merge_repeat:
        merged: list[str] = []
        for line, group in groupby(lines):
            grp = list(group)
            cnt = len(grp)
            if cnt > merge_num:
                merged += [line] * 2
                merged.append(f"[This line repeated {cnt - 4} more times]")
                merged += [line] * 2
            else:
                merged += grp
        lines = merged

    # 2) Line-count truncation (keep last keep_tail_lines)
    if max_lines is not None and len(lines) > max_lines:
        total = len(lines)
        if max_lines <= keep_tail_lines + 1:
            lines = lines[:max_lines]
        else:
            head_count = max_lines - keep_tail_lines - 1
            head = lines[:head_count]
            tail = lines[-keep_tail_lines:]
            omitted = total - head_count - keep_tail_lines
            lines = head + [f"… {omitted} lines omitted …"] + tail

    # 3) Per-line character-length truncation
    if max_length is not None:
        truncated_lines: list[str] = []
        for line in lines:
            if len(line) > max_length:
                truncated_lines.append(line[:max_length] + "… (truncated)")
            else:
                truncated_lines.append(line)
        lines = truncated_lines
    return "\n".join(lines)


def is_weird(text, repeat_n=128, cjk_limit=128):
    s = unicodedata.normalize('NFKC', text)
    if re.search(rf'(.)\1{{{repeat_n - 1},}}|(.{{2,12}})\2{{{repeat_n - 1},}}', s):
        return True
    CJK = ((0x4E00, 0x9FFF), (0x3040, 0x309F), (0x30A0, 0x30FF), (0xAC00, 0xD7AF))
    cjk_count = sum(any(a <= ord(c) <= b for a, b in CJK) for c in s)
    return cjk_count >= cjk_limit or (len(s) > 0 and cjk_count / len(s) > 0.8)

class LLMClass:
    async def create_completion(self, input_ids, **kwargs):
        raise NotImplemented

class CallLLM(LLMClass):  # Call LLM in Verl RL env
    def __init__(self, url, tokenizer, config, loop, **kwargs):
        self.server_manager = url
        self.tokenizer = tokenizer
        self.config = config
        self.loop = loop
        self.call_openai = getattr(config.plugin, "call_openai", None)

    async def _create_completion(self, input_ids, **kwargs):
        from uuid import uuid4

        max_len = kwargs.pop('max_len', None) or self.config.prompt_length + self.config.response_length
        max_len = min(max_len, self.config.prompt_length + self.config.response_length)
        max_new_tokens = max_len - len(input_ids)
        # Per-turn cap (plugin.turn_max_new_tokens). Before 2026-09-23 the capped value was computed into an unused
        # variable and never sent (audit finding F4: turns up to 6k tokens with a 2048 cap configured).
        # plugin.protocol=legacy reproduces that (enforce_turn_cap=False).
        from agents.protocol import resolve as _resolve_protocol
        if hasattr(self.config, 'plugin') and getattr(self.config.plugin, 'turn_max_new_tokens', -1) > 0 \
                and _resolve_protocol(getattr(self.config, 'plugin', None)).enforce_turn_cap:
            max_new_tokens = min(max_new_tokens, self.config.plugin.turn_max_new_tokens)
        if 'max_new_tokens' in kwargs:
            max_new_tokens = min(max_new_tokens, kwargs['max_new_tokens'])

        if max_new_tokens < 10:
            print(f"[DEBUG] max_new_tokens {max_new_tokens}, skip rollout")
            return None

        uid = kwargs.pop('uid', None) or uuid4().hex

        sampling_params = kwargs.pop('sampling_params', None) or {}
        sampling_params = {
            'temperature': sampling_params.get('temperature', 1.0),
            'top_p': sampling_params.get('top_p', 1.0),
            'max_tokens': max_new_tokens,
            'logprobs': True,
        }

        output = await self.server_manager.generate(
            request_id=uid,
            prompt_ids=input_ids,
            sampling_params=sampling_params,
            image_data=None,
        )

        if output is None or len(output.token_ids) == 0:
            return None

        response_text = await self.loop.run_in_executor( None, lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True))

        return {
            "choices": [{
                "message": {
                    "content": response_text,
                    "raw_output_ids": output.token_ids,
                    "response_log_probs": output.log_probs if getattr(output, 'log_probs', None) is not None else [0.0] * len(
                        output.token_ids),
                    "extra_data": {"input_ids": input_ids},
                    "metrics": {}
                }
            }]
        }

    async def create_completion(self, input_ids, **kwargs):
        completion = await self._create_completion(input_ids, **kwargs)
        return completion


class CallAPI(LLMClass):  # Call external API (OpenAI)
    def __init__(self, url, tokenizer, config, **kwargs):
        self.tokenizer = tokenizer
        self.config = config
        self.model = url
        from openai import AsyncOpenAI
        import os
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", None)  # Optional custom base URL
        )

    async def create_completion(self, input_ids, **kwargs):
        max_len = kwargs.pop('max_len', None) or self.config.prompt_length + self.config.response_length
        max_tokens = min(max_len, self.config.prompt_length + self.config.response_length) - len(input_ids)

        if getattr(self.config.plugin, 'turn_max_new_tokens', -1) > 0:
            max_tokens = min(max_tokens, self.config.plugin.turn_max_new_tokens)
        if 'max_new_tokens' in kwargs:
            max_tokens = min(max_tokens, kwargs.pop('max_new_tokens'))

        if max_tokens < 10:
            return None
        messages = kwargs.get('messages', None)
        if messages is None:
            messages = decode_conversation(input_ids, self.tokenizer)[0]

        for attempt in range(5):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                )

                text = response.choices[0].message.content or ""
                text_ids = self.tokenizer.encode(text, add_special_tokens=False)

                return {
                    "choices": [{
                        "message": {
                            "content": text,
                            "raw_output_ids": text_ids,
                            "response_log_probs": [0.0] * len(text_ids),
                            "extra_data": {"input_ids": input_ids},
                            "metrics": {"usage": {
                                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                                "completion_tokens": response.usage.completion_tokens if response.usage else len(text_ids),
                                "total_tokens": response.usage.total_tokens if response.usage else 0,
                            }}
                        }
                    }]
                }
            except Exception as e:
                if attempt == 4:
                    print(f"[CallAPI ERROR] Failed after 5 attempts: {e}")
                    return None
                wait_time = 2 ** attempt
                print(f"[CallAPI] Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        return None


_ANCHOR_QUERY = [{'role': 'user', 'content': 'anchor'}]


def render_prefix_len(tokenizer, msgs):
    """Token length of ``msgs`` as the chat template renders them; templates that refuse a chat without a plain user
    query (Qwen3.5: 'No user query found') are rendered with a trailing anchor query whose tokens are subtracted."""
    try:
        return len(tokenizer.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True))
    except Exception as e:
        if 'user query' not in str(e):
            raise
        full = tokenizer.apply_chat_template(list(msgs) + _ANCHOR_QUERY, add_generation_prompt=False, tokenize=True)
        return len(full) - len(tokenizer.apply_chat_template(_ANCHOR_QUERY, add_generation_prompt=False, tokenize=True))


def truncate_prompt(chat, prompt_length, tokenizer, prompt_turn):
    exceed_len = render_prefix_len(tokenizer, chat[:prompt_turn]) + 8 - prompt_length
    _cut_idx = 0
    while exceed_len > 0:  # truncate long user prompt
        print('[PROMPT] now exceed', exceed_len, 'work on cut turn', _cut_idx)
        chat[_cut_idx]['content'] = tokenizer.decode(
            tokenizer.encode(chat[_cut_idx]['content'], add_special_tokens=False)[
                exceed_len + 4:], add_special_tokens=False)
        exceed_len = render_prefix_len(tokenizer, chat[:prompt_turn]) + 8 - prompt_length
        _cut_idx = _cut_idx + 1
        if _cut_idx >= prompt_turn:
            break
    return chat


class AgentContext:
    # Manage context of an agent
    def __init__(self, chat, tokenizer, config, prompt_turn=2):
        self.tokenizer = tokenizer
        self.config = config
        self.init_len = len(chat)
        self.prompt_turn = prompt_turn

        # Support both inference and training config styles
        if hasattr(config, 'actor_rollout_ref'):
            # Training config (VERL)
            self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
            self.response_length = config.actor_rollout_ref.rollout.response_length
        else:
            # Inference config
            self.prompt_length = config.prompt_length
            self.response_length = config.response_length

        self.context_uid = str(uuid.uuid4())
        from agents.protocol import resolve_from_config
        self.protocol = resolve_from_config(config)

        self.chat = copy.deepcopy([turn for turn in chat])
        self.chat = truncate_prompt(self.chat, config.prompt_length, tokenizer, prompt_turn)
        self.chat_completions = [None for _ in range(len(self.chat))]
        self.chat_ids = [self.get_turn_context(i) for i in range(len(self.chat))]
        self.log_probs = [[0.0] * len(turn) for turn in self.chat_ids]
        self.token_mask = [[False] * len(turn) for turn in self.chat_ids]
        self.additional_info = [None for _ in self.chat_ids]
        self.generation_prompt = None
        self.metrics = None
        self.prompt_ids_len = len(sum(self.chat_ids[:prompt_turn], []))

    def get_turn_context(self, i):
        """Token ids for turn ``i`` of ``self.chat``.

        Prompt turns (``i < prompt_turn``: system + task prompt) are sliced from the full
        template render so any template-level preamble (tool docs, default system text) is
        kept exactly once. Every later text-only turn is rendered standalone by
        ``render_single_turn``: a prefix diff of ``chat[:i+1]`` vs ``chat[:i]`` is not safe
        with templates that rewrite earlier turns depending on the last message. Qwen3, for
        example, strips every assistant ``<think>`` block before the latest plain user query,
        so the diff came out shifted by the previous think block and the observation (or a
        branch/summary prompt) lost its header, its first tokens, or all of it.
        """
        if i >= self.prompt_turn:
            return self.render_single_turn(self.chat[i])
        tokens = self._render_prefix(i + 1)
        prev = self._render_prefix(i)
        turn_tokens = tokens[len(prev):]
        return turn_tokens

    _ANCHOR = _ANCHOR_QUERY

    def _render_prefix(self, k):
        """Token ids of ``chat[:k]`` (prompt turns only) as the template renders them.

        Qwen3's template renders any prefix; Qwen3.5's raises ``No user query found`` for a prefix without a
        plain user message (e.g. the system turn alone). In that case the prefix is rendered followed by a plain
        anchor query whose own tokens are sliced off, which yields the same prefix tokens (user/system turns are
        rendered as a pure append).
        """
        if k <= 0:
            return []
        try:
            return self.tokenizer.apply_chat_template(self.chat[:k], add_generation_prompt=False, tokenize=True)
        except Exception as e:  # jinja2 TemplateError from templates that need a user query
            if 'user query' not in str(e):
                raise
            full = self.tokenizer.apply_chat_template(self.chat[:k] + self._ANCHOR, add_generation_prompt=False, tokenize=True)
            tail = self.tokenizer.apply_chat_template(self._ANCHOR, add_generation_prompt=False, tokenize=True)
            assert full[len(full) - len(tail):] == tail, 'chat template does not render the anchor query as a pure append'
            return full[:len(full) - len(tail)]

    def render_single_turn(self, turn):
        """Render one message on its own, independent of the rest of the conversation.

        The turn is rendered behind a fixed anchor (the chat's system message, if any, plus a
        plain user query) whose own tokens are sliced off. The anchor keeps templates from
        injecting a default system prompt, and the plain user query in it makes an assistant
        turn render *after* a query boundary, so its ``<think>`` block is kept (Qwen3 strips
        reasoning only from assistant turns at or before the latest query). A re-tokenized
        assistant turn therefore matches what the policy saw when it generated it.
        """
        anchor = self.chat[:1] if self.chat and self.chat[0].get('role') == 'system' else []
        anchor = anchor + self._ANCHOR
        tokens = self.tokenizer.apply_chat_template(anchor + [turn], add_generation_prompt=False, tokenize=True)
        prev = self.tokenizer.apply_chat_template(anchor, add_generation_prompt=False, tokenize=True)
        assert tokens[:len(prev)] == prev, 'chat template does not render turns as a pure append'
        return tokens[len(prev):]

    def get_generation_prompt(self):
        if self.generation_prompt is None:
            try:
                tokens = self.tokenizer.apply_chat_template(self.chat, add_generation_prompt=False, tokenize=True)
                add_tokens = self.tokenizer.apply_chat_template(self.chat, add_generation_prompt=True,
                                                                tokenize=True)
            except Exception as e:  # template needs a plain user query (Qwen3.5) and the chat has none yet
                if 'user query' not in str(e):
                    raise
                anchor = (self.chat[:1] if self.chat and self.chat[0].get('role') == 'system' else []) + self._ANCHOR
                tokens = self.tokenizer.apply_chat_template(anchor, add_generation_prompt=False, tokenize=True)
                add_tokens = self.tokenizer.apply_chat_template(anchor, add_generation_prompt=True, tokenize=True)
            self.generation_prompt = add_tokens[len(tokens):]
        return self.generation_prompt

    def messages(self):
        return self.chat

    def context_ids(self, messages=None):
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def context(self, turn_cut: int=None):
        if turn_cut is not None:
            return sum(self.chat_ids[:turn_cut], []) + self.get_generation_prompt()
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def append(self, turn, completion=None, additional_info=None):
        self.chat.append(turn)
        self.chat_completions.append(completion)
        self.additional_info.append(additional_info)
        if completion is None:
            self.chat_ids.append(self.get_turn_context(len(self.chat) - 1))
            self.log_probs.append([0.0] * len(self.chat_ids[-1]))
            self.token_mask.append([False] * len(self.chat_ids[-1]))
        else:
            completion_tokens = completion["choices"][0]["message"]["raw_output_ids"]
            completion_log_probs = completion["choices"][0]["message"]["response_log_probs"]
            self.chat_ids.append(self.get_generation_prompt() + completion_tokens)
            self.log_probs.append([0.0] * len(self.get_generation_prompt()) + completion_log_probs)
            self.token_mask.append([False] * len(self.get_generation_prompt()) + [True] * len(completion_tokens))
            if len(completion_tokens) == 0 or completion_tokens[-1] != self.tokenizer.eos_token_id:
                self.chat_ids[-1].append(self.tokenizer.eos_token_id)
                self.log_probs[-1].append(0.0)
                self.token_mask[-1].append(False)
            # canonical turn terminator is '<|im_end|>\n'; sampled turns stopped at <|im_end|>, so the next turn's
            # '<|im_start|>' followed it with no newline (audit finding F5). Append the newline as a non-trained token.
            # plugin.protocol=legacy (pre-3f697bf checkpoints) keeps the glued form the checkpoint was trained on.
            if self.protocol.turn_end_newline:
                for t in self._turn_end_newline_ids():
                    self.chat_ids[-1].append(t)
                    self.log_probs[-1].append(0.0)
                    self.token_mask[-1].append(False)

    def _turn_end_newline_ids(self):
        """Token ids the template puts after the end-of-turn token (a single '\n' for Qwen); [] if the template does not."""
        if getattr(self, '_nl_ids', None) is None:
            anchor = self._ANCHOR
            with_turn = self.tokenizer.apply_chat_template(anchor + [{'role': 'assistant', 'content': 'x'}],
                                                           add_generation_prompt=False, tokenize=False)
            eos = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
            tail = with_turn.rsplit(eos, 1)[-1] if eos in with_turn else ''
            self._nl_ids = self.tokenizer.encode(tail, add_special_tokens=False) if tail.strip() == '' and tail else []
        return self._nl_ids

    def append_tokens(self, turn, ids):
        """Append a turn with PRE-TOKENISED ids (non-trainable): used to carry raw sampled turns into another context
        (compaction tail, branch history) without re-rendering them through the chat template."""
        self.chat.append(turn)
        self.chat_completions.append(None)
        self.additional_info.append(None)
        self.chat_ids.append(list(ids))
        self.log_probs.append([0.0] * len(ids))
        self.token_mask.append([False] * len(ids))
        capture_event(self, "append_tokens", role=turn.get("role"), turn_index=len(self.chat) - 1, ids_len=len(ids),
                      ids_sha1=_ids_sha1(ids))

    def rollback(self, k=1):
        self.chat = self.chat[:-k]
        self.chat_completions = self.chat_completions[:-k]
        self.chat_ids = self.chat_ids[:-k]
        self.log_probs = self.log_probs[:-k]
        self.token_mask = self.token_mask[:-k]
        self.additional_info = self.additional_info[:-k]

    def get_metrics(self):
        if self.metrics is None:
            return {}
        return self.metrics

    async def get_data(self):
        prompt_turn = self.prompt_turn
        prompt_length = self.prompt_length
        response_length = self.response_length

        prompt_ids = sum(self.chat_ids[:prompt_turn], [])
        if len(prompt_ids) > prompt_length:
            print('[PROMPT] prompt truncated')
            prompt_ids = prompt_ids[-prompt_length:]

        response_ids = sum(self.chat_ids[prompt_turn:], [])[:response_length]
        response_logprobs = sum(self.log_probs[prompt_turn:], [])[:response_length]
        response_mask = [1 if m else 0 for turn in self.token_mask[self.prompt_turn:] for m in turn][:response_length]
        process_reward_mask = sum([[info.get('process_reward', 0) if isinstance(info, dict) else 0] * len(turn)
                                   for turn, info in zip(self.chat_ids, self.additional_info)][prompt_turn:], [])
        process_reward_mask = [p * m for p, m in zip(process_reward_mask, response_mask)][:response_length]
        return {
            'prompt_ids': prompt_ids,
            'response_ids': response_ids,
            'response_logprobs': response_logprobs,
            'response_mask': response_mask,
            'process_reward_mask': process_reward_mask,
            'num_turns': len(self.chat_ids),
            'messages': self.chat,
        }


import contextvars
import json as _json

# Prompt capture for audits (env FOLD_PROMPT_CAPTURE_DIR): every Agent.step appends one JSON line with the EXACT token
# ids sent to the policy (decoded with special tokens) and the sampled completion. process_item sets CAPTURE_TAG
# (rollout uid etc.) so records can be grouped; Agent.capture_name tells main / branch / segment apart.
CAPTURE_TAG: contextvars.ContextVar = contextvars.ContextVar("fold_capture_tag", default=None)


def _capture_write(rec):
    d = os.environ.get("FOLD_PROMPT_CAPTURE_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        rec.setdefault("ts", time.time()); rec.setdefault("pid", os.getpid()); rec.setdefault("tag", CAPTURE_TAG.get())
        with open(os.path.join(d, f"capture_{os.getpid()}.jsonl"), "a") as f:
            f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:  # never break a rollout because of the audit hook
        print(f"[CAPTURE] failed: {e}")


def _ids_sha1(ids):
    import hashlib
    return hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:16]


def _capture_step(agent, prompt, completion, max_len):
    """One record per policy call: the EXACT prompt ids (stored as the suffix beyond this agent's previously captured
    prompt when the prefix is unchanged, else in full — reconstructable exactly), the sampled completion ids, and the
    stored turn (ids + training mask) after ``Agent.append``."""
    if not os.environ.get("FOLD_PROMPT_CAPTURE_DIR"):
        return
    try:
        msg = completion["choices"][0]["message"] if completion else {}
        ids = list(msg.get("raw_output_ids") or [])
        prev = getattr(agent, "_cap_prev_prompt", None)
        if prev is not None and len(prompt) >= len(prev) and prompt[: len(prev)] == prev:
            prompt_ids, mode = prompt[len(prev):], "suffix"
        else:
            prompt_ids, mode = list(prompt), "full"
        agent._cap_prev_prompt = list(prompt)
        stored_turn = agent.chat_ids[-1] if completion else None
        rec = {
            "kind": "step", "agent": getattr(agent, "capture_name", None), "context_uid": agent.context_uid,
            "turn_index": len(agent.chat) - (1 if completion else 0),
            "prompt_tokens": len(prompt), "prompt_ids_len": agent.prompt_ids_len, "max_len": max_len,
            "prompt_sha1": _ids_sha1(prompt), "prompt_ids_mode": mode, "prompt_ids": prompt_ids,
            "prompt_text": agent.tokenizer.decode(prompt, skip_special_tokens=False),
            "completion_tokens": len(ids), "completion_ids": ids, "completion_text": msg.get("content"),
            "completion_text_with_special": agent.tokenizer.decode(ids, skip_special_tokens=False) if ids else None,
            "completion_none": completion is None,
            "stored_turn_ids": stored_turn, "stored_turn_mask": [int(m) for m in agent.token_mask[-1]] if completion else None,
        }
        _capture_write(rec)
    except Exception as e:
        print(f"[CAPTURE] failed: {e}")


def capture_action(agent, parsed, results, observation):
    """After the environment acted on a turn: parsed calls, executed calls with per-call status/observation, and the
    observation text that will be appended (before wrapping)."""
    if not os.environ.get("FOLD_PROMPT_CAPTURE_DIR"):
        return
    _capture_write({"kind": "action", "agent": getattr(agent, "capture_name", None), "context_uid": agent.context_uid,
                    "turn_index": len(agent.chat) - 1, "parsed": parsed, "results": results,
                    "observation": observation, "observation_chars": len(observation) if observation else 0})


def capture_event(agent, kind, **extra):
    if not os.environ.get("FOLD_PROMPT_CAPTURE_DIR"):
        return
    _capture_write({"kind": kind, "agent": getattr(agent, "capture_name", None), "context_uid": agent.context_uid, **extra})


class Agent(AgentContext):
    # Agent utils
    def __init__(self, llm_client, conversations, tokenizer, config, prompt_turn=2):
        super().__init__(conversations, tokenizer, config, prompt_turn=prompt_turn)
        self.llm_client = llm_client
        self.retry_cjk = getattr(config.plugin, "retry_cjk", 0)
        self.info_cache = {}
        self.capture_name = None

    def fork(self, llm_client=None):
        """A new Agent that inherits this context's exact token ids (all turns non-trainable), e.g. a branch."""
        new = copy.copy(self)
        new.llm_client = llm_client if llm_client is not None else self.llm_client
        new.context_uid = str(uuid.uuid4())
        new.chat = copy.deepcopy(self.chat)
        new.chat_ids = [list(t) for t in self.chat_ids]
        new.chat_completions = [None] * len(self.chat)
        new.log_probs = [[0.0] * len(t) for t in self.chat_ids]
        new.token_mask = [[False] * len(t) for t in self.chat_ids]
        new.additional_info = [None] * len(self.chat)
        new.init_len = len(self.chat)
        new.info_cache = {}
        new.capture_name = None
        new.metrics = None
        new._cap_prev_prompt = None
        inherited = sum(self.chat_ids, [])
        capture_event(new, "fork", parent_context_uid=self.context_uid, parent_agent=getattr(self, "capture_name", None),
                      inherited_turns=len(self.chat), inherited_tokens=len(inherited), inherited_sha1=_ids_sha1(inherited))
        return new

    async def step(self, max_new_tokens=None, retry_cjk=0):
        prompt = self.context()
        max_len = self.prompt_ids_len + self.config.response_length
        if max_new_tokens is not None:
            max_len = min(len(prompt) + max_new_tokens, 131072)
        completion = await self.llm_client.create_completion(
            prompt, uid=self.context_uid, max_len=max_len, messages=self.chat)
        if completion is None:
            _capture_step(self, prompt, None, max_len)
            return None
        response = completion["choices"][0]["message"]["content"]
        self.append({'role': 'assistant', 'content': response}, completion)
        _capture_step(self, prompt, completion, max_len)
        return response

    async def react(self, run_action, max_turn=64, max_tokens=None, session_timeout=60 * 60,
                    should_continue=None, summary_prompt=None, safe_finish=None, observation_prompt=None, env=None,
                    max_consecutive_no_call=3):
        # Run react for max_turn turn
        if should_continue is None:
            should_continue = lambda st: True
        no_call_run, forced_reason = 0, None
        session_start_time = time.time()
        iteration = 0
        if max_tokens is not None:
            max_tokens = max_tokens - 512
        else:
            max_tokens = self.config.response_length - 512

        last_response = None
        response = None
        init_len = len(self.context(turn_cut=self.prompt_turn))
        while iteration < max_turn:
            if time.time() - session_start_time > session_timeout:  # TODO add session timeout
                print('[SESSION] Session Timeout')
                break
            if len(self.context()) - init_len > max_tokens:  # summary
                break

            iteration += 1
            response = await self.step()
            if response is None:
                break

            if not should_continue(response):
                last_response = response
                break
            if safe_finish is not None and safe_finish(response) is not None:
                observation = safe_finish(response)
                capture_action(self, parse_actions(response), [{"call_id": None, "function": "guard", "status": "rejected",
                                                                 "observation": observation}], observation)
            else:
                observation = await run_action(response)
                capture_action(self, parse_actions(response), getattr(env, "last_action_results", None) if env is not None else None, observation)
            if observation is None:
                break
            # loop protection: N consecutive turns without a valid call end the branch (forced return with the last
            # message) instead of burning the turn budget re-emitting the same malformed call (shakeout #1: 101-step branches)
            no_call_run = no_call_run + 1 if str(observation).startswith('No function call was detected') else 0
            if max_consecutive_no_call and no_call_run >= max_consecutive_no_call:
                last_response, forced_reason = response, 'no_call_loop'
                print(f'[BRANCH] forced return after {no_call_run} consecutive turns without a valid call')
                break
            if observation_prompt:
                observation += '\n' + observation_prompt
            self.append({'role': 'user', 'content': wrap_tool_response(observation), })

        if last_response is None and summary_prompt is not None:
            if len(self.context()) - init_len > self.config.response_length - 1024:  # summary
                self.rollback(k=2)
            if self.chat[-1]['role'] == 'user':
                self.append({'role': 'assistant', 'content': "", })
            self.append({'role': 'user', 'content': summary_prompt, })
            last_response = await self.step(max_new_tokens=4096)
        elif last_response is None:
            last_response = str(response)

        return {'last_response': last_response, 'iteration': iteration, 'forced_reason': forced_reason}

    def set_process_reward(self, turn, reward):
        if isinstance(turn, str) and turn.lower() == 'all':
            turn = [i for i in range(len(self.chat))]
        if not isinstance(turn, list):
            turn = [turn]
        for i in turn:
            if i <= 0:
                continue
            if i > len(self.chat) - 1:
                continue
            if self.chat_completions[i] is None:
                continue
            if self.additional_info[i] is None:
                self.additional_info[i] = {}
            self.additional_info[i]['process_reward'] = reward

    def set_cache(self, key, value):
        self.info_cache[key] = value


@dataclass
class TaskContext:
    config: DictConfig
    global_step: int
    is_train: bool
    tokenizer: PreTrainedTokenizer | AutoTokenizer | None = None
    llm_client: LLMClass = None

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput  # single class identity for pydantic

class _UnusedLocalAgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"


def wrap_tool_response(observation) -> str:
    """Mark an environment/tool observation as a tool response for the chat template.

    Qwen3's chat template treats a user message as a *new query* unless it starts with
    ``<tool_response>`` and ends with ``</tool_response>``. A new query boundary makes the
    template strip every earlier assistant ``<think>...</think>`` block, so rendering
    ``chat[:i]`` and ``chat[:i+1]`` no longer share a prefix and the per-turn token diff in
    ``AgentContext.get_turn_context`` drops the start of the observation (or all of it).
    Wrapping observations keeps the query boundary at the last genuine instruction, so prior
    reasoning is preserved across tool turns and the diff stays aligned.

    Only environment/tool results must be wrapped. Genuine instructions (task prompt, branch
    task prompt, summary prompt) are appended unwrapped so they remain new-query boundaries.
    """
    return f"{TOOL_RESPONSE_OPEN}\n{observation}\n{TOOL_RESPONSE_CLOSE}"


async def run_action(env, response):
    try:
        try:
            act = time.time()
            env_return = await asyncio.wait_for(env.run_action(response), timeout=120.0)
            if time.time() - act > 10:
                print('Action Cost', time.time() - act)
        except asyncio.TimeoutError:
            print('[ACTION] Action timed out after 120 seconds')
            env_return = {'observation': 'Action timed out after 120 seconds'}
        if 'action' in env_return:
            action, arguments = env_return['action'], env_return.get('arguments', {})
            if action == 'finish':
                return None
        elif env_return.get('observation', None) == 'finish':
            return None
        observation = env_return.pop('observation', 'Empty')
    except Exception as e:
        observation = f"Error: {e}"
    return observation
