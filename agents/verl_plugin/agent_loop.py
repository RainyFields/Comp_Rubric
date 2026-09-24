"""Agent-loop worker/manager for verl 0.9.1's V1 (TransferQueue) rollout path, adapted to multi-output rollouts.

Differences from ``verl.trainer.ppo.v1.agent_loop_tq.AgentLoopWorkerTQ`` (0.9.1, L150-227):
* ``validate`` is forwarded to ``AgentLoopBase.run(**kwargs)`` (the stock worker drops it; our loops need it to pick the
  eval path and the val knobs);
* every output keeps its own ``reward_score`` (FoldGRPO's per-branch flat rewards); the stock worker copies the final
  output's reward onto all rows and requires ``extra_fields["reward_extra_info"]`` on it;
* rows flagged ``extra_fields["mask_rollout"]`` (unfinished zero-reward rollouts under the FoldAgent convention, SUPO
  "overlong" rollouts) get ``response_mask = 0``: no policy loss and no share of the token-mean denominator, while their
  ``rm_scores`` still enter the group statistics (the fork's ``overlong_mask`` and SUPO's treatment);
* ``extra_fields["gen_uid"] = f"{uid}_{session_id}"`` (also the prefix of the TransferQueue row key).
Select with ``+actor_rollout_ref.rollout.agent.agent_loop_manager_class=agents.verl_plugin.agent_loop.FoldAgentLoopManagerTQ``.
"""
import logging

import ray
import torch
import transfer_queue as tq

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__file__)

_WorkerBase = AgentLoopWorkerTQ.__ray_metadata__.modified_class.__ray_actor_class__


def apply_rollout_masks(outputs: list[AgentLoopOutput], gen_uid: str) -> int:
    """In-place: guarantee reward_extra_info, stamp gen_uid, zero response_mask of mask_rollout rows. Returns #masked."""
    masked = 0
    final_reward = outputs[-1].reward_score
    for output in outputs:
        ef = output.extra_fields
        ef.setdefault("reward_extra_info", {})
        ef["gen_uid"] = gen_uid
        if output.reward_score is None:
            output.reward_score = final_reward
        if ef.get("mask_rollout"):
            output.response_mask = [0] * len(output.response_mask)
            masked += 1
    return masked


def build_row(worker, out: AgentLoopOutput, kwargs: dict) -> tuple[dict, dict]:
    """One TransferQueue row (field dict + tag) for an AgentLoopOutput: unpadded input_ids = prompt + response,
    loss_mask = response_mask, position ids from the worker's ``_compute_position_ids`` (verl 0.9.1 layout).
    ``worker`` only needs ``processor`` (None = text) and the two base helpers; kept pure for the CPU tests."""
    prompts = torch.tensor(out.prompt_ids, dtype=torch.int64)
    responses = torch.tensor(out.response_ids, dtype=torch.int64)
    input_ids = torch.cat([prompts, responses], dim=0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
    multi_modal_inputs = worker._compute_multi_modal_inputs(out, input_ids)
    position_ids = worker._compute_position_ids(input_ids.unsqueeze(0), attention_mask.unsqueeze(0),
                                                multi_modal_inputs).squeeze(0)
    field = out.as_dict()
    field.update(kwargs)
    field.pop("multi_modal_data", None)
    field["loss_mask"] = field["response_mask"]
    field["input_ids"] = input_ids
    field["position_ids"] = position_ids
    field["multi_modal_inputs"] = multi_modal_inputs
    prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
    tag = {
        "status": "success", "prompt_len": prompt_len, "response_len": response_len,
        "seq_len": prompt_len + response_len, "global_steps": kwargs.get("global_steps"),
        "min_global_steps": field["extra_fields"].get("min_global_steps"),
        "max_global_steps": field["extra_fields"].get("max_global_steps"),
    }
    return field, tag


@ray.remote
class FoldAgentLoopWorkerTQ(_WorkerBase):
    async def _run_agent_loop(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        kwargs["validate"] = bool(trajectory.get("validate", False))
        return await super()._run_agent_loop(sampling_params, trajectory, agent_name=agent_name, trace=trace, **kwargs)

    async def _agent_loop_postprocess(self, output, validate, **kwargs) -> None:
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)
        final_output = outputs[-1]
        await self._compute_teacher_logprobs(final_output, prompt_ids=final_output.prompt_ids,
                                             response_ids=final_output.response_ids, validate=validate, sample_kwargs=kwargs)
        apply_rollout_masks(outputs, f"{uid}_{session_id}")

        keys, fields, tags = [], [], []
        for i, out in enumerate(outputs):
            field, tag = build_row(self, out, kwargs)
            keys.append(f"{uid}_{session_id}_{i}")
            fields.append(field)
            tags.append(tag)
        await tq.async_kv_batch_put(keys=keys, fields=list_of_dict_to_tensordict(fields), tags=tags,
                                    partition_id="train" if not validate else "val")


class FoldAgentLoopManagerTQ(AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = FoldAgentLoopWorkerTQ
