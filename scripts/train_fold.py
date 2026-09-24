"""Agent loops for verl 0.9.1 (V1 rollout path) + the training entry point.

Registered loops (``infra/agent_loop_config.yaml``): ``fold_agent`` (branch scaffold, arms grpo / foldgrpo /
grpo_no_compaction) and ``compaction_agent`` (summarise-and-resume, arms compactionrl / compactiongrpo / supo).

verl 0.9.1 specifics handled here (docs/plans/2026-09-24_verl091_port_design.md):
* ``AgentLoopBase.__init__`` takes (trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config,
  hf_model_type, **kwargs); ``self.config`` is the full trainer DictConfig;
* ``run(sampling_params, **kwargs)`` receives the dataset row fields + uid / index / global_steps / session_id, and
  ``validate`` only through agents.verl_plugin.agent_loop.FoldAgentLoopWorkerTQ;
* the rollout plugin knobs live under ``actor_rollout_ref.rollout.custom.plugin`` (the RolloutConfig dataclass rejects
  ``rollout.plugin``); they are exposed to agents/* and envs/* as ``rollout.plugin`` on a per-worker copy of the config;
* ``sampling_params`` (temperature / top_p / top_k; greedy for validation when val_kwargs.do_sample=False) are honoured
  by CallLLM — before the port every rollout, validation included, sampled at temperature 1.
"""
import copy
import logging
import os
from typing import Any, Union

import numpy as np
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

from agents.utils import CallLLM, TaskContext

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_PLUGIN_CFG_CACHE: dict[int, Any] = {}


def with_plugin_config(config):
    """Return a config whose ``actor_rollout_ref.rollout.plugin`` exists (copied from ``rollout.custom.plugin``).

    One copy per worker process and source config (cached by id), so per-loop mutations such as
    ``config.response_length = plugin.val_response_length`` behave as they did with the vendored verl.
    """
    rollout = config.actor_rollout_ref.rollout
    if rollout.get("plugin", None) is not None:
        return config
    key = id(config)
    if key in _PLUGIN_CFG_CACHE:
        return _PLUGIN_CFG_CACHE[key]
    custom = rollout.get("custom", None) or {}
    plugin = custom.get("plugin", None) if hasattr(custom, "get") else None
    plugin = OmegaConf.to_container(plugin, resolve=True) if plugin is not None else {}
    cfg = copy.deepcopy(config)
    with open_dict(cfg):
        cfg.actor_rollout_ref.rollout.plugin = OmegaConf.create(plugin)
    _PLUGIN_CFG_CACHE[key] = cfg
    return cfg


@register("fold_agent")
class FoldAgentLoop(AgentLoopBase):
    process_item = None  # set below (import at call time keeps the module import light)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = with_plugin_config(self.config)

    async def _process(self, item, context):
        from agents.fold_agent import process_item
        return await process_item(item, context)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
        validate = bool(kwargs.get("validate", False))
        uid = kwargs.get("uid")
        gen_uid = f"{uid}_{kwargs.get('session_id', 0)}"
        non_tensors = {k: np.array([v], dtype=object) for k, v in kwargs.items()}
        non_tensors["gen_uid"] = np.array([gen_uid], dtype=object)
        item = DataProto.from_dict(non_tensors=non_tensors)
        llm_client = CallLLM(url=self.server_manager, tokenizer=self.tokenizer,
                             config=self.config.actor_rollout_ref.rollout, loop=self.loop,
                             sampling_params=sampling_params)
        context = TaskContext(config=self.config, global_step=int(kwargs.get("global_steps", 0) or 0),
                              llm_client=llm_client, is_train=not validate, tokenizer=self.tokenizer)
        return await self._process(item, context)


@register("compaction_agent")
class CompactionAgentLoop(FoldAgentLoop):
    """CompactionRL / SUPO rollout (agents/compaction_agent.py): same client/context plumbing as FoldAgentLoop."""

    async def _process(self, item, context):
        from agents.compaction_agent import process_item
        return await process_item(item, context)


def main():
    """Entry point for training: verl 0.9.1 main_ppo with the Comp_Rubric plugin (agents/verl_plugin) registered."""
    from agents.verl_plugin.runner import main as plugin_main
    plugin_main()


if __name__ == "__main__":
    main()
