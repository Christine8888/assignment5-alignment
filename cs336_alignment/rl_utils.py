from typing import Callable, List, Literal, Dict
import torch
import cs336_alignment.utils as utils
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch

loss_types = Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"]

def compute_group_normalized_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: List[str],
        repeated_ground_truths: List[str],
        group_size: int,
        advantage_eps: float,
        normalize_by_std: bool = True):
    
    assert len(rollout_responses) == len(repeated_ground_truths)
    all_raw_rewards = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        raw_rewards = reward_fn(response, gt)
        all_raw_rewards.append(raw_rewards)
    
    raw_rewards = torch.tensor([r["reward"] for r in all_raw_rewards])

    # copy + reshape raw rewards to be n_prompts x group_size; get n_prompts mean
    reshaped_rewards = raw_rewards.clone().view(len(rollout_responses) // group_size, group_size)
    group_means = reshaped_rewards.mean(keepdim = True, dim = -1)
    advantages = reshaped_rewards - group_means
    group_stds = reshaped_rewards.std(keepdim = True, dim = -1)
    
    if normalize_by_std:
        advantages /= (group_stds + advantage_eps)
    
    advantages = advantages.view(len(rollout_responses))

    metadata = {"mean_advantage": torch.mean(advantages).detach().item(),
                "std_advantage": torch.std(advantages).detach().item(),
                "min_advantage": torch.min(advantages).detach().item(),
                "max_advantage": torch.max(advantages).detach().item(),
                "per_group_mean_reward": torch.mean(group_means).detach().item(), # average over groups
                "per_group_std_reward": torch.mean(group_stds).detach().item(), # average over groups
                "per_group_min_reward": torch.mean(reshaped_rewards.min(dim = -1).values).detach().item(), # average over groups
                "per_group_max_reward": torch.mean(reshaped_rewards.max(dim = -1).values).detach().item()} # average over groups
    
    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor
) -> torch.Tensor:
    # apply same reward to every token, broadcast
    batch_size, seq_len = policy_log_probs.shape
    broadcast_a = raw_rewards_or_advantages.expand(batch_size, seq_len)
    
    return -1 * policy_log_probs * broadcast_a # use negative to do gradient ascent

def compute_grpo_clip_loss(
    advantages: torch.Tensor, # per-example advantages
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    clip: bool = True
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, seq_len = policy_log_probs.shape
    broadcast_a = advantages.expand(batch_size, seq_len)
    
    ratio = torch.exp(policy_log_probs - old_log_probs)
    if clip:
        clip_min = 1 - cliprange
        clip_max = 1 + cliprange
        clip = torch.where((ratio > clip_max) | (ratio < clip_min), 1.0, 0.0)
        clipped_ratio = torch.clamp(ratio, min = clip_min, max = clip_max)
    else:
        clip = torch.zeros_like(ratio) # no clipping
        clipped_ratio = ratio

    # compute clipped and unclipped advantages
    lhs = broadcast_a * ratio
    rhs = broadcast_a * clipped_ratio

    # take min
    grpo_loss = torch.minimum(lhs, rhs)

    metadata = {"clip_fraction": clip.mean().detach().item()}

    return -grpo_loss, metadata # use negative to do gradient ascent


def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: loss_types,
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None, # assume these come from grpo
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metadata = {}
    
    if loss_type == "grpo_clip" or loss_type == "grpo_no_clip":
        if advantages is None:
            raise ValueError("advantages must be provided for grpo_clip loss")
        if old_log_probs is None:
            raise ValueError("old_log_probs must be provided for grpo_clip loss")
        if cliprange is None:
            raise ValueError("cliprange must be provided for grpo_clip loss")
        
        clip = loss_type == "grpo_clip" # only do clipping if loss_type is grpo_clip

        loss, grpo_metadata = compute_grpo_clip_loss(advantages = advantages,
                                                     policy_log_probs = policy_log_probs,
                                                     old_log_probs = old_log_probs,
                                                     cliprange = cliprange,
                                                     clip = clip)
        metadata.update(grpo_metadata)

    elif loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards must be provided for no_baseline loss")
        loss = compute_naive_policy_gradient_loss(raw_rewards_or_advantages = raw_rewards,
                                                  policy_log_probs = policy_log_probs)
    
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages must be provided for reinforce_with_baseline loss")
        loss = compute_naive_policy_gradient_loss(raw_rewards_or_advantages = advantages,
                                                  policy_log_probs = policy_log_probs)
        metadata = {}
    
    return loss, metadata

def masked_mean(tensor: torch.Tensor,
                mask: torch.Tensor,
                dim: int | None = None): # get mean, normalized by (unmasked) length
    
    lengths = mask.sum(dim = dim) 
    masked = tensor * mask
    means = masked.sum(dim = dim) / lengths

    return means

def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: loss_types,
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        length_normalize: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # loss has shape (batch_size, seq_len)
    loss, metadata = compute_policy_gradient_loss(policy_log_probs = policy_log_probs,
                                                  loss_type = loss_type,
                                                  raw_rewards = raw_rewards,
                                                  advantages = advantages,
                                                  old_log_probs = old_log_probs,
                                                  cliprange = cliprange)
    
    # per-token loss -> per-example loss
    if length_normalize:
        loss = masked_mean(loss, mask = response_mask, dim = -1)
    else:
        # normalize by length of longest response
        longest_response_length = response_mask.sum(dim = -1).max()
        loss = utils.masked_normalize(loss, mask = response_mask, dim = -1, normalize_constant = longest_response_length)
    
    scaled_loss = torch.mean(loss) / gradient_accumulation_steps

    # backward pass
    scaled_loss.backward()

    return scaled_loss.detach(), metadata