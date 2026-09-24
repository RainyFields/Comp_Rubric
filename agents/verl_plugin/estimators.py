"""Advantage estimators of the FoldAgent / CompactionRL / SUPO arms as an out-of-tree plugin for verl 0.9.1.

Bodies are the vendored fork's (verl/trainer/ppo/core_algos.py, commits 85391fa + f22260d) verbatim; only the
registration changed: verl 0.9.1's ``AdvantageEstimator`` enum cannot be extended, so the estimators are registered
under plain string names and selected with ``algorithm.adv_estimator=<name>``. verl's generic dispatch passes only
``token_level_rewards, response_mask, config, index``, therefore :class:`agents.verl_plugin.trainer.FoldSyncTrainer`
calls :func:`compute_custom_advantage`, which reconstructs ``gen_uid`` (= ``{uid}_{session_id}`` of the TransferQueue
row key), ``tokens_after`` and ``process_reward_mask`` from the rows' ``extra_fields``.

Registration is idempotent (the same function object may be registered many times; verl rejects a *different* one).
"""
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY, register_adv_est

FOLDGRPO, COMPACTION_GAE, COMPACTION_GRPO, SUPO = "foldgrpo", "compaction_gae", "compaction_grpo", "supo"
CUSTOM_ESTIMATORS = (FOLDGRPO, COMPACTION_GAE, COMPACTION_GRPO, SUPO)


def _register(name):
    def deco(fn):
        if ADV_ESTIMATOR_REGISTRY.get(name) is fn:
            return fn
        return register_adv_est(name)(fn)
    return deco


@_register(FOLDGRPO)
def compute_foldgrpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    gen_uid: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    fix_bad_positive_adv: bool = False,
    process_reward_mask: Optional[torch.Tensor] = None,
    raw_token_level_scores: Optional[torch.Tensor] = None,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FoldGRPO advantage (arXiv:2510.11967): rollout-level group normalisation (rows of one rollout are de-duplicated
    via ``gen_uid``), broadcast to every token; ``process_reward_mask`` (+1 / -1 / 0 per token) replaces the advantage of
    the marked tokens by the group max / min (the paper's process rewards)."""
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2gen_uid = defaultdict(list)
    id2mean = {}
    id2std = {}
    raw_scores = None if raw_token_level_scores is None else raw_token_level_scores.sum(-1)
    if fix_bad_positive_adv:
        assert raw_scores is not None, "raw_scores should not be None when fix_bad_positive_adv"
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if gen_uid[i] in id2gen_uid[index[i]]:
                continue
            id2gen_uid[index[i]].append(gen_uid[i])
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
            if fix_bad_positive_adv and scores[i] > 0 and raw_scores[i] < 0:
                scores[i] = 0.0 * scores[i]
        if process_reward_mask is not None:
            gmin, gmax = {}, {}
            for k, s in zip(index, scores):
                gmin[k], gmax[k] = torch.minimum(s, gmin.get(k, s)), torch.maximum(s, gmax.get(k, s))
            for k in gmin:  # patch all 0/1
                if gmin[k] >= 0:
                    gmin[k] = gmin[k] * 0 - 1
                if gmax[k] <= 0:
                    gmax[k] = gmax[k] * 0 + 1
            gmax = torch.stack([gmax[k] for k in index]).unsqueeze(dim=1).tile([1, response_length])
            gmin = torch.stack([gmin[k] for k in index]).unsqueeze(dim=1).tile([1, response_length])
            scores = scores.unsqueeze(dim=1).tile([1, response_length])
            scores = (1 - process_reward_mask.abs()) * scores + \
                torch.relu(process_reward_mask) * gmax + torch.relu(-process_reward_mask) * gmin
            scores = scores * response_mask
        else:
            scores = scores.unsqueeze(dim=1).tile([1, response_length]) * response_mask
    return scores, scores


def compaction_position_discount(gamma, lam, tokens_after: torch.Tensor) -> torch.Tensor:
    """CompactionRL trajectory-position correction (arXiv:2607.05378, Eq. 14): (gamma*lam)^{N_>s}."""
    lam_t = lam if torch.is_tensor(lam) else torch.full_like(tokens_after, float(lam), dtype=torch.float32)
    return (float(gamma) * lam_t.float()) ** tokens_after.float()


@_register(COMPACTION_GAE)
def compute_compaction_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    lam: float,
    tokens_after: torch.Tensor,
    lam_alpha: Optional[float] = None,
    whiten: bool = True,
    config=None,
):
    """Cross-trajectory GAE for compacted rollouts (CompactionRL, Eqs. 13-15): local GAE per segment with the
    length-adaptive lambda_i = 1 - 1/(lam_alpha * l_i); policy advantage discounted by (gamma*lambda_i)^{tokens_after};
    the critic target keeps the local GAE (returns = A_loc + V)."""
    with torch.no_grad():
        bs, gen_len = token_level_rewards.shape
        resp_len = response_mask.sum(dim=-1).clamp(min=1).float()
        if lam_alpha is not None and lam_alpha > 0:
            lam_vec = 1.0 - 1.0 / (float(lam_alpha) * resp_len)
        else:
            lam_vec = torch.full((bs,), float(lam), dtype=torch.float32, device=token_level_rewards.device)
        lam_vec = lam_vec.to(token_level_rewards.dtype)
        nextvalues = torch.zeros(bs, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
        lastgaelam = torch.zeros_like(nextvalues)
        advantages_reversed = []
        for t in reversed(range(gen_len)):
            m = response_mask[:, t]
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam_vec * lastgaelam
            nextvalues = values[:, t] * m + (1 - m) * nextvalues          # skip observation tokens
            lastgaelam = lastgaelam_ * m + (1 - m) * lastgaelam
            advantages_reversed.append(lastgaelam)
        adv_local = torch.stack(advantages_reversed[::-1], dim=1)
        returns = adv_local + values
        tokens_after = torch.as_tensor(tokens_after, device=adv_local.device)
        disc = compaction_position_discount(gamma, lam_vec, tokens_after).to(adv_local.dtype)
        advantages = adv_local * disc.unsqueeze(-1)
        if whiten:
            advantages = verl_F.masked_whiten(advantages, response_mask)
        advantages = advantages * response_mask
    return advantages, returns


@_register(COMPACTION_GRPO)
def compute_compaction_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    gen_uid: np.ndarray,
    tokens_after: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 1.0,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Critic-free CompactionRL variant: rollout-level group normalisation (one entry per gen_uid inside each prompt
    group), broadcast to every segment, position correction (gamma*lam)^{N_>s} (identity at gamma = lam = 1)."""
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score, id2gen, id2mean, id2std = defaultdict(list), defaultdict(list), {}, {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if gen_uid[i] in id2gen[index[i]]:
                continue
            id2gen[index[i]].append(gen_uid[i])
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx], id2std[idx] = torch.tensor(0.0), torch.tensor(1.0)
            else:
                st = torch.stack(id2score[idx])
                id2mean[idx], id2std[idx] = torch.mean(st), torch.std(st)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        tokens_after = torch.as_tensor(tokens_after, device=scores.device)
        scores = scores * compaction_position_discount(gamma, lam, tokens_after).to(scores.dtype)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    return scores, scores


@_register(SUPO)
def compute_supo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    gen_uid: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SUPO advantage (arXiv:2510.06727, Eq. 3): group-relative advantage over the G rollouts of a prompt (one reward
    per rollout; trajectories of one rollout de-duplicated via gen_uid), broadcast to every token of every trajectory.
    Overlong rollouts keep their reward in the group statistics but carry response_mask = 0 (no loss), as in the paper;
    the masking happens in the agent-loop worker. Identical to compaction_grpo with gamma = lam = 1."""
    zeros = torch.zeros(token_level_rewards.shape[0], device=token_level_rewards.device)
    return compute_compaction_grpo_advantage(token_level_rewards, response_mask, index, gen_uid, zeros, 1.0, 1.0,
                                             epsilon, norm_adv_by_std_in_grpo, config)


def pad_process_reward_mask(extra: list[dict], bs: int, length: int, like: torch.Tensor) -> Optional[torch.Tensor]:
    """Right-pad the per-row ``process_reward_mask`` lists (values in {-1, 0, 1}) to ``(bs, length)``; None if absent."""
    if not any(e.get("process_reward_mask") is not None for e in extra):
        return None
    out = torch.zeros(bs, length, dtype=like.dtype, device=like.device)
    for i, e in enumerate(extra):
        m = e.get("process_reward_mask")
        if m is None:
            continue
        m = torch.as_tensor(list(m), dtype=like.dtype, device=like.device)[:length]
        out[i, : m.numel()] = m
    return out


def compute_custom_advantage(name: str, data, keys: list[str], extra: list[dict], algo, metrics: Optional[dict] = None):
    """Dispatch used by FoldSyncTrainer._compute_advantage. ``data`` is the padded DataProto (token_level_rewards,
    token_level_scores, response_mask, uid, [values]); ``keys`` the TransferQueue row keys ``{uid}_{session}_{index}``;
    ``extra`` the rows' extra_fields dicts (same order)."""
    tlr = data.batch["token_level_rewards"]
    rm = data.batch["response_mask"]
    bs, length = rm.shape
    index = data.non_tensor_batch["uid"]
    gen_uid = np.array([k.rsplit("_", 1)[0] for k in keys], dtype=object)
    tokens_after = torch.tensor([int(e.get("tokens_after", 0) or 0) for e in extra], device=tlr.device)
    gamma, lam = float(algo.get("gamma", 1.0)), float(algo.get("lam", 1.0))
    norm = bool(algo.get("norm_adv_by_std_in_grpo", True))
    if metrics is not None:
        metrics["adv/rows"] = float(bs)
        metrics["adv/rollouts"] = float(len(set(gen_uid.tolist())))
        metrics["adv/masked_rows"] = float(sum(1 for e in extra if e.get("mask_rollout")))
    if name == FOLDGRPO:
        prm = pad_process_reward_mask(extra, bs, length, rm)
        return compute_foldgrpo_advantage(tlr, rm, index, gen_uid, norm_adv_by_std_in_grpo=norm,
                                          fix_bad_positive_adv=bool(algo.get("fix_bad_positive_adv", False)),
                                          process_reward_mask=prm, raw_token_level_scores=data.batch.get("token_level_scores"),
                                          config=algo)
    if name == COMPACTION_GAE:
        return compute_compaction_gae_advantage_return(tlr, data.batch["values"], rm, gamma, lam, tokens_after,
                                                       lam_alpha=algo.get("compaction_lam_alpha", None),
                                                       whiten=bool(algo.get("compaction_whiten", True)), config=algo)
    if name == COMPACTION_GRPO:
        return compute_compaction_grpo_advantage(tlr, rm, index, gen_uid, tokens_after, gamma, lam,
                                                 norm_adv_by_std_in_grpo=norm, config=algo)
    if name == SUPO:
        return compute_supo_advantage(tlr, rm, index, gen_uid, norm_adv_by_std_in_grpo=norm, config=algo)
    raise ValueError(f"unknown custom estimator {name}")
