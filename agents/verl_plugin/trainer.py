"""FoldSyncTrainer: verl 0.9.1 V1 synchronous PPO trainer + the plugin's advantage estimators.

Selected with ``trainer.use_v1=True trainer.v1.trainer_mode=fold_sync``. The class registers itself under
``fold_sync`` (verl's ``register_trainer``) and, once instantiated, behaves as the stock ``sync`` trainer (it rewrites
``trainer.v1.trainer_mode`` to ``sync`` before ``PPOTrainer.__init__`` reads it, because that value also selects the
replay buffer and the ``trainer.v1.sync`` sub-config).

Only ``_compute_advantage`` is overridden: for the custom estimators it additionally fetches each row's ``extra_fields``
from the TransferQueue (``tokens_after``, ``process_reward_mask``, ``mask_rollout``) and derives ``gen_uid`` from the
row key ``{uid}_{session_id}_{index}``. Everything else (KL-in-reward, rollout correction, write-back) mirrors
``PPOTrainer._compute_advantage`` of verl 0.9.1 (trainer_base.py L1743-1802).
"""
import numpy as np
import transfer_queue as tq
from omegaconf import open_dict
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import apply_kl_penalty
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.utils.padding import response_to_nested

from . import estimators as E

TRAINER_NAME = "fold_sync"


@register_trainer(TRAINER_NAME)
class FoldSyncTrainer(PPOTrainerSync):
    def __init__(self, config, *args, **kwargs):
        with open_dict(config):
            config.trainer.v1.trainer_mode = "sync"
        super().__init__(config, *args, **kwargs)

    def _compute_advantage(self, batch, metrics: dict):
        est = str(self.config.algorithm.adv_estimator)
        if est not in E.CUSTOM_ESTIMATORS:
            return super()._compute_advantage(batch, metrics)

        fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "values",
                  "extra_fields"]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        extra_col = data.pop("extra_fields")
        extra = [e if isinstance(e, dict) else {} for e in extra_col.tolist()]

        response_mask = data["response_mask"]
        data = DataProto(batch=data.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)

        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(data, kl_ctrl=self.kl_ctrl_in_reward,
                                                kl_penalty=self.config.algorithm.kl_penalty)
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass = bool(rollout_corr_config and rollout_corr_config.get("bypass_mode", False))
        rollout_correction = rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass
        if rollout_correction:
            data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
            metrics.update(is_metrics)

        advantages, returns = E.compute_custom_advantage(est, data, list(batch.keys), extra, self.config.algorithm, metrics)
        data.batch["advantages"], data.batch["returns"] = advantages, returns

        out_fields = ["advantages", "returns"]
        if self.config.algorithm.use_kl_in_reward:
            out_fields.append("token_level_rewards")
        if rollout_correction:
            out_fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                out_fields.append("rollout_is_weights")
        output = TensorDict({f: response_to_nested(data.batch[f], response_mask) for f in out_fields}, batch_size=len(batch))
        return tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)
