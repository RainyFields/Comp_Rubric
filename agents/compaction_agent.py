"""CompactionRL rollout for the FoldAgent stack.

Implements the rollout side of *CompactionRL: Reinforcement Learning with Context Compaction for
Long-Horizon Agents* (Li et al., arXiv:2607.05378) as a baseline arm next to GRPO and FoldGRPO:

* A single-thread ReAct agent (search / open_page / finish; no branch tool) runs under a fixed
  generated-token budget ``B = config.response_length`` per context *segment*.
* When the remaining budget falls below ``T_comp`` (``plugin.compaction_threshold``) and fewer
  than ``plugin.max_compactions`` compactions have happened, a fixed summarisation instruction
  ``q_sum`` is appended and the **policy itself** samples the summary ``S_t``. Those summary tokens
  are trainable (``plugin.train_summary=False`` masks them: the paper's "w/o sum." ablation).
* The rollout resumes from a reconstructed context ``(prompt) + u_resume(S_t) + (z_{t-k+1..t})``:
  the original task prompt, a fixed resume template carrying the summary, and the ``k`` most
  recent (assistant, observation) steps verbatim (``plugin.compaction_tail_steps``, default 2,
  reduced when they do not fit). Tail turns are re-rendered text (mask 0): they were already
  optimised in the previous segment.
* The summary always fits: if ``q_sum`` plus ``summary_max_tokens`` would exceed the segment budget,
  the most recent steps are rolled back out of the segment (they remain verbatim in the tail).
* Every segment becomes one training sample (prompt = task prompt; response = everything after
  it). All segments of a rollout share the final task reward ``R``; there is no summary-quality
  reward. Each sample carries ``tokens_after`` = the number of optimised tokens generated after
  it in the same rollout, which the ``compaction_gae`` / ``compaction_grpo`` advantage estimators
  use for the paper's cross-trajectory correction ``(gamma*lambda)^{N_>s}``.

The rollout core (:func:`run_compaction_rollout`) is independent of the environment and the
trainer so it can be unit-tested with a scripted LLM client (see tests/test_compaction_agent.py).
"""
import asyncio
import collections
import copy
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Union
from uuid import uuid4

from verl import DataProto

from .fold_agent import print_chat
from .parsing import extract_summary, strip_think
from .prompts import COMPACTION_RESUME_TEMPLATE, COMPACTION_SUMMARY_PROMPT, create_chat
from .parsing import parse_actions
from .utils import CAPTURE_TAG, Agent, AgentLoopMetrics, AgentLoopOutput, TaskContext, capture_action, run_action, select_env, wrap_tool_response


@dataclass
class CompactionConfig:
    """Rollout hyper-parameters (defaults follow the paper where the FoldAgent stack allows)."""
    max_compactions: int = 3          # paper: at most three compaction operations per rollout (x4 budget)
    threshold: int = 6144             # T_comp: remaining generated-token budget that triggers compaction
    tail_steps: int = 2               # k: most recent (assistant, observation) steps kept verbatim
    summary_max_tokens: int = 2048    # max new tokens for the summary response
    train_summary: bool = True        # False = "w/o summary training" ablation (loss masked on summary tokens)
    max_turn: int = 100               # total assistant turns across all segments
    session_timeout: float = 3600.0
    mask_unfinished: bool = True      # FoldAgent convention: unfinished zero-reward rollouts carry no gradient
    resume_keep_task_prompt: bool = True   # True: resumed segment = (system + task) + u_resume + tail (FoldAgent runs so far);
                                           # False: paper Eq. 9, (system) + u_resume + tail — the task survives only via the summary

    @classmethod
    def from_plugin(cls, plugin: Any, is_train: bool) -> "CompactionConfig":
        g = lambda k, d: getattr(plugin, k, d) if plugin is not None else d  # noqa: E731
        max_comp = g("max_compactions", cls.max_compactions)
        if not is_train:
            max_comp = g("val_max_compactions", max_comp)
        return cls(
            max_compactions=int(max_comp),
            threshold=int(g("compaction_threshold", cls.threshold)),
            tail_steps=int(g("compaction_tail_steps", cls.tail_steps)),
            summary_max_tokens=int(g("summary_max_tokens", cls.summary_max_tokens)),
            train_summary=bool(g("train_summary", cls.train_summary)),
            max_turn=int(g("val_max_turn", g("max_turn", cls.max_turn)) if not is_train else g("max_turn", cls.max_turn)),
            session_timeout=float(g("session_timeout", cls.session_timeout)),
            mask_unfinished=bool(g("mask_unfinished", cls.mask_unfinished)),
            resume_keep_task_prompt=bool(g("resume_keep_task_prompt", cls.resume_keep_task_prompt)),
        )


def _summary_from_response(response: str) -> str:
    """The <summary> block if present, else the response without its think block (tagged or pre-filled opener)."""
    return extract_summary(response) or strip_think(response)


def _generated_tokens(seg: Agent) -> int:
    """Tokens of the segment that count against the generated-side budget: everything after the prompt turns
    (resume turn, tail, generated turns, summary) excluding the pending generation prompt."""
    return len(seg.context()) - seg.prompt_ids_len - len(seg.get_generation_prompt())


async def _build_segment(llm_client, user_prompt, tokenizer, config, prompt_turn, summary, tail, keep_task_prompt=True):
    """Resumed context: (prompt) + u_resume(S) + tail steps, all as non-trainable turns.

    ``keep_task_prompt=False`` follows the paper's Eq. 9 literally — only the system prompt precedes
    ``u_resume``; the user instruction is expected to be carried by the summary (q_sum item 1).
    """
    if keep_task_prompt:
        seg_prompt, seg_prompt_turn = user_prompt, prompt_turn
    else:
        seg_prompt = [t for t in user_prompt[:prompt_turn] if t.get("role") == "system"] + list(user_prompt[prompt_turn:])
        seg_prompt_turn = sum(1 for t in user_prompt[:prompt_turn] if t.get("role") == "system")
    seg = Agent(llm_client, seg_prompt, tokenizer, config, prompt_turn=seg_prompt_turn)
    seg.capture_name = "segment"      # renumbered below by the caller
    seg.append({"role": "user", "content": COMPACTION_RESUME_TEMPLATE.format(summary=summary)})
    for assistant_text, observation_wrapped, assistant_ids, observation_ids in tail:
        # verbatim tail: the previous segment's exact token ids (raw sampled assistant turn incl. its think block, and the
        # rendered observation), non-trainable — not a re-render through the chat template (audit finding F1)
        seg.append_tokens({"role": "assistant", "content": assistant_text}, assistant_ids)
        seg.append_tokens({"role": "user", "content": observation_wrapped}, observation_ids)
    return seg


async def run_compaction_rollout(
    user_prompt: list[dict],
    llm_client,
    tokenizer,
    config,
    act: Callable[[str], Awaitable[Optional[str]]],
    cc: CompactionConfig,
    prompt_turn: Optional[int] = None,
    env=None,
) -> dict:
    """Environment-agnostic CompactionRL rollout.

    ``act(response)`` returns the observation text for an assistant response, or ``None`` when the
    response finished the task. Returns a dict with ``segments`` (list of :class:`Agent`, one per
    context segment, in rollout order), ``segment_info`` (per-segment dicts: ``summary_tokens``,
    ``tail_steps``), ``session_message`` (full transcript across segments), ``finished``,
    ``stop_reason`` and ``stats``.
    """
    prompt_turn = len(user_prompt) if prompt_turn is None else prompt_turn
    budget = config.response_length
    stats: collections.Counter = collections.Counter()
    seg = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
    seg.capture_name = "seg0"
    segments, seg_info = [seg], [{"summary_tokens": 0, "tail_steps": 0}]
    steps: list[tuple] = []                  # (assistant text, wrapped observation, assistant ids, observation ids) of the current segment
    session_message: list[dict] = []
    t0, iteration, finished, stop_reason = time.time(), 0, False, "max_turn"

    used = _generated_tokens                 # per segment: its own prompt length is subtracted (prompts may differ)

    while iteration < cc.max_turn:
        if time.time() - t0 > cc.session_timeout:
            stop_reason = "timeout"
            break
        remaining = budget - used(seg)
        if remaining < cc.threshold:
            if stats["compactions"] >= cc.max_compactions:
                stop_reason = "budget_exhausted"
                break
            tail = list(steps[-cc.tail_steps:]) if cc.tail_steps > 0 else []
            # The summary must fit in this segment's budget: q_sum + up to summary_max_tokens of summary.
            # Roll the most recent steps back out of the segment until it does; they stay in the tail,
            # so the model still sees them verbatim after compaction (only the loss on them is forgone).
            q_len = len(seg.render_single_turn({"role": "user", "content": COMPACTION_SUMMARY_PROMPT}))
            room = q_len + cc.summary_max_tokens
            while budget - used(seg) < room and len(steps) >= 1:
                seg.rollback(k=2)
                steps.pop()
                stats["rollback_before_summary"] += 1
            seg.append({"role": "user", "content": COMPACTION_SUMMARY_PROMPT})   # plain user turn: new query boundary
            summary_resp = await seg.step(max_new_tokens=max(1, min(cc.summary_max_tokens, budget - used(seg))))
            if summary_resp is None:
                stop_reason = "llm_none"
                break
            n_sum = int(sum(seg.token_mask[-1]))
            if not cc.train_summary:                      # "w/o summary training": generated but no loss
                seg.token_mask[-1] = [False] * len(seg.token_mask[-1])
            summary = _summary_from_response(summary_resp)
            session_message.append({"role": "user", "content": COMPACTION_SUMMARY_PROMPT})
            session_message.append({"role": "assistant", "content": summary_resp})
            seg_info[-1]["summary_tokens"] = n_sum if cc.train_summary else 0   # trained summary tokens
            stats["compactions"] += 1
            stats["summary_tokens_generated"] += n_sum
            stats["summary_tokens"] += seg_info[-1]["summary_tokens"]
            stats["summary_chars"] += len(summary)
            # reconstruct: shrink the tail until the new context leaves at least T_comp of budget
            k = len(tail)
            while True:
                new_seg = await _build_segment(llm_client, user_prompt, tokenizer, config, prompt_turn, summary,
                                               tail[len(tail) - k:], keep_task_prompt=cc.resume_keep_task_prompt)
                if budget - used(new_seg) >= cc.threshold or k == 0:
                    break
                k -= 1
            stats["tail_steps"] += k
            seg = new_seg
            seg.capture_name = f"seg{len(segments)}"
            segments.append(seg)
            seg_info.append({"summary_tokens": 0, "tail_steps": k})
            steps = list(tail[len(tail) - k:])
            session_message.append({"role": "user", "content": COMPACTION_RESUME_TEMPLATE.format(summary=summary)})
            continue

        iteration += 1
        response = await seg.step()
        if response is None:
            stop_reason = "llm_none"
            break
        session_message.append({"role": "assistant", "content": response})
        observation = await act(response)
        capture_action(seg, parse_actions(response, turn_id=str(len(seg.chat) - 1)),
                       getattr(env, "last_action_results", None) if env is not None else None, observation)
        if observation is None:                     # finish tool
            finished, stop_reason = True, "finish"
            break
        wrapped = wrap_tool_response(observation)
        assistant_ids = list(seg.chat_ids[-1])
        seg.append({"role": "user", "content": wrapped})
        session_message.append({"role": "user", "content": observation})
        steps.append((response, wrapped, assistant_ids, list(seg.chat_ids[-1])))

    stats["segments"] = len(segments)
    stats["turns"] = iteration
    return {
        "segments": segments,
        "segment_info": seg_info,
        "session_message": session_message,
        "finished": finished,
        "stop_reason": stop_reason,
        "stats": stats,
    }


def segment_outputs(rollout: dict, reward: float, mask_rollout: bool, extra: dict, is_train: bool) -> list:
    """Turn segments into AgentLoopOutputs (train: every segment; eval: last segment only)."""
    return asyncio.run(_segment_outputs(rollout, reward, mask_rollout, extra, is_train))


async def _segment_outputs(rollout: dict, reward: float, mask_rollout: bool, extra: dict, is_train: bool) -> list:
    segments, infos = rollout["segments"], rollout["segment_info"]
    datas = [await s.get_data() for s in segments]
    n_opt = [int(sum(d["response_mask"])) for d in datas]
    n_seg = len(segments)
    outs = []
    indices = range(n_seg) if is_train else [n_seg - 1]
    for i in indices:
        d, info = datas[i], infos[i]
        tokens_after = int(sum(n_opt[i + 1:]))
        fields = dict(extra)
        fields.update({
            "messages": d["messages"],
            "mask_rollout": mask_rollout,
            "process_reward_mask": d["process_reward_mask"],
            "agent_name": f"seg{i}",
            "meta_info": f"N: {n_seg} | seg{i}",
            "segment_index": i,
            "num_segments": n_seg,
            "tokens_after": tokens_after,
            "optimized_tokens": n_opt[i],
            "summary_tokens": info["summary_tokens"],
            "tail_steps": info["tail_steps"],
            "is_last_segment": int(i == n_seg - 1),
        })
        outs.append(AgentLoopOutput(
            prompt_ids=d["prompt_ids"],
            response_ids=d["response_ids"],
            response_mask=d["response_mask"],
            response_logprobs=d["response_logprobs"],
            multi_modal_data={},
            reward_score=reward,
            num_turns=d["num_turns"],
            metrics=AgentLoopMetrics(),
            extra_fields=fields,
        ))
    return outs


async def process_item(item: DataProto, context: TaskContext) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train
    if not is_train and getattr(config.plugin, "val_response_length", None):
        config.response_length = getattr(config.plugin, "val_response_length", None)

    ability = item.non_tensor_batch["ability"][0]
    uid = item.non_tensor_batch.get("uid", uuid4().hex)
    gen_uid = item.non_tensor_batch.get("gen_uid", None)
    if hasattr(uid, "__len__") and not isinstance(uid, str):
        uid = uid[0]
    if gen_uid is not None and hasattr(gen_uid, "__len__") and not isinstance(gen_uid, str):
        gen_uid = gen_uid[0]

    EnvClass = select_env(ability, config)
    env = EnvClass(config, tokenizer, ability)
    try:
        await env.init_env(item)
    except Exception as e:
        print(f"[Error] during environment init: {str(e)}")
        raise

    workflow = item.non_tensor_batch["extra_info"][0].get("workflow", None) or getattr(config.plugin, "workflow", "search")
    user_prompt = create_chat(env.instance_info["problem_statement"], workflow, item)
    cc = CompactionConfig.from_plugin(config.plugin, is_train)
    CAPTURE_TAG.set({"uid": str(uid), "gen_uid": str(gen_uid), "is_train": bool(is_train),
                     "instance_id": str(env.instance_info.get("instance_id", env.instance_info.get("query_id", "")))})

    session_start = time.time()
    rollout = await run_compaction_rollout(
        user_prompt, context.llm_client, tokenizer, config,
        act=lambda response: run_action(env, response), cc=cc, env=env,
    )
    env.stats["session_time"] = time.time() - session_start

    print("[TASK] Task Finish, Start Reward")
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, user_prompt + rollout["session_message"], context), timeout=60 * 10)
        score = (score_msg, reward)
    except Exception as e:
        print(f"[Error] Getting reward: {e}")
        score = ("", 0)

    is_finish = bool(getattr(env, "is_finish", False) or getattr(env, "finish", False) or rollout["finished"])
    if getattr(config.plugin, "must_finish", None) and not is_finish:
        score = ("", 0)
    mask_rollout = bool(cc.mask_unfinished and not is_finish and score[1] <= 0)

    st = rollout["stats"]
    env.stats.update({
        "get_final_score": score[1], "traj_num": st["segments"], "compactions": st["compactions"],
        "summary_tokens": st["summary_tokens"], "summary_tokens_generated": st["summary_tokens_generated"],
        "tail_steps": st["tail_steps"], "turns": st["turns"],
        "budget_exhausted": int(rollout["stop_reason"] == "budget_exhausted"),
        "rollback_before_summary": st["rollback_before_summary"], "is_finish": int(is_finish),
        "main_len": min(len(rollout["segments"][-1].context()), config.response_length),
        "total_token": len(tokenizer.encode(print_chat(user_prompt + rollout["session_message"]))),
    })
    print(f"[COMPACTION] uid={uid} gen_uid={gen_uid} stop={rollout['stop_reason']} finished={int(is_finish)} "
          f"score={score[1]} segments={st['segments']} compactions={st['compactions']} turns={st['turns']} "
          f"summary_tokens={st['summary_tokens_generated']} rollback={st['rollback_before_summary']} mask_rollout={int(mask_rollout)}")
    extra = {
        "env_stats": copy.deepcopy(env.stats) if hasattr(env, "stats") else {},
        "num_branches": 0, "branch_names": [], "is_finish": is_finish,
        "message_str": print_chat(rollout["session_message"]),
        "stop_reason": rollout["stop_reason"], "uid": uid, "gen_uid": gen_uid,
    }
    return await _segment_outputs(rollout, score[1], mask_rollout, extra, is_train)
