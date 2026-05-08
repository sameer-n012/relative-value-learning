r"""
src/rollout.py

Rollout buffer for PPO + RV that stores trajectories and samples 
pairwise critic batches. We set p=0.33, T=128 by default.

Paper references:
    - Batch pair sampling distribution \mu: Eq. 28
    - Biased same-episode pairing:          App. E
    - Rollout length:                       App. D
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Iterator


@dataclass
class RolloutBatch:
    r"""
    A single minibatch for PPO + RV training.
    """

    # Policy gradient items
    obs: torch.Tensor           # (B, C, H, W)
    actions: torch.Tensor       # (B,)
    log_probs: torch.Tensor     # (B,)  log \pi_old(a|s)
    advantages: torch.Tensor    # (B,)  normalized R-GAE A_t
    returns: torch.Tensor       # (B,)  discounted returns

    # Critic loss items
    obs_i: torch.Tensor         # (B, C, H, W)  first state of pair
    obs_j: torch.Tensor         # (B, C, H, W)  second state of pair
    r_i: torch.Tensor           # (B,)  reward at i
    r_j: torch.Tensor           # (B,)  reward at j
    d_i: torch.Tensor           # (B,)  done flag for s_{i+1}
    d_j: torch.Tensor           # (B,)  done flag for s_{j+1}
    obs_i_next: torch.Tensor    # (B, C, H, W)
    obs_j_next: torch.Tensor    # (B, C, H, W)


class RolloutBuffer:
    r"""
    Stores a single rollout of T steps * N_envs parallel environments.

    Provides:
        - store()         : add one timestep of data
        - compute_rgae()  : compute R-GAE advantages in-place (called after rollout)
        - get_minibatches(): yield shuffled minibatches for PPO epochs
        - sample_pairs()  : sample critic pairs with the biased same-episode strategy
    """

    def __init__(self, rollout_length: int, n_envs: int, obs_shape: tuple, same_ep_prob: float = 0.33, device: str = "cpu"):
        self.T = rollout_length
        self.N = n_envs
        self.obs_shape = obs_shape
        self.same_ep_prob = same_ep_prob
        self.device = device

        self._ptr = 0
        self._allocate()

    def _allocate(self):
        r"""
        Pre-allocate tensors for the rollout.
        """

        T, N, S = self.T, self.N, self.obs_shape
        self.obs = torch.zeros(T, N, *S)
        self.obs_next = torch.zeros(T, N, *S)
        self.actions = torch.zeros(T, N, dtype=torch.long)
        self.rewards = torch.zeros(T, N)
        self.dones = torch.zeros(T, N)
        self.log_probs = torch.zeros(T, N)

        # create episode IDs
        self.ep_ids = torch.zeros(T, N, dtype=torch.long)

        # empty tensors to pass to R-GAE
        self.advantages = torch.zeros(T, N)
        self.returns = torch.zeros(T, N)
        self.V_rel = torch.zeros(T + 1, N)

    def store(
        self,
        obs: torch.Tensor,
        obs_next: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        log_prob: torch.Tensor,
        ep_id: torch.Tensor,
    ):
        r"""
        Store one timestep of experience.

        Args:
            obs:        N observations of shape (N, *obs_shape).
            obs_next:   next N observations of shape (N, *obs_shape).
            action:     actions as (N,).
            reward:     rewards as (N,).
            done:       done flags as (N,).
            log_prob:   log probs as (N,).
            ep_id:      episode counter per env as (N,)
        """
        t = self._ptr
        self.obs[t] = obs
        self.obs_next[t] = obs_next
        self.actions[t] = action
        self.rewards[t] = reward
        self.dones[t] = done
        self.log_probs[t] = log_prob
        self.ep_ids[t] = ep_id
        self._ptr += 1

    def reset(self):
        self._ptr = 0

    def compute_rgae(
        self,
        delta_fn,
        gamma: float,
        lam:   float,
        use_offset: bool = True,
    ):
        r"""
        Compute R-GAE advantages and returns for the stored rollout. Does this by
        computing relative values V_\theta (with or without offset), computing R-GAE,
        and normalizing advantages.

        Args:
            delta_fn:       \Delta_\theta as Callable: (obs_i, obs_j) -> (B,).
            gamma:          discount factor.
            lam:            GAE lambda.
            use_offset:     whether to apply trajectory ranking offset.
        """

        # TODO: implement per-env relative value computation using
        #   core.rgae.compute_relative_values  and (if use_offset)
        #   core.rgae.compute_trajectory_offsets + apply_offset_to_rollout
        # Then call core.rgae.compute_rgae for each env and store results.
        raise NotImplementedError(
            "Fill in: loop over N envs, compute V_rel (or V_bar with offset), "
            "then call compute_rgae() from core.rgae."
        )

    def sample_pairs(self, n_pairs: int) -> Tuple[torch.Tensor]:
        r"""
        Sample n_pairs of (i, j) transition indices for the critic loss. It uses 
        the following pairing strategy:
          - with probability same_ep_prob: j is drawn from the same episode as i.
          - with probability 1-same_ep_prob: j is drawn uniformly random from the batch.

        Args:
            n_pairs:    the number of pairs

        Returns:
            sample:     tuple of (obs_i, r_i, d_i, obs_i_next, obs_j, r_j, d_j, obs_j_next,
                        where each is of shape (n_pairs, ...).
        """

        T, N = self.T, self.N
        flat_size = T * N

        # flatten (T, N) -> (T*N,)
        obs_flat = self.obs.view(flat_size, *self.obs_shape)
        obs_next_flat = self.obs_next.view(flat_size, *self.obs_shape)
        rewards_flat = self.rewards.view(flat_size)
        dones_flat = self.dones.view(flat_size)
        ep_ids_flat = self.ep_ids.view(flat_size)

        # sample reference indices i
        idx_i = torch.randint(0, flat_size, (n_pairs,))

        # sample j with cross-episode strategy
        use_same = torch.rand(n_pairs) < self.same_ep_prob
        idx_j = torch.randint(0, flat_size, (n_pairs,))

        for k in range(n_pairs):

            if use_same[k]:

                # get all transitions from the same episode as i
                ep_id = ep_ids_flat[idx_i[k]].item()
                same_ep_mask = (ep_ids_flat == ep_id).nonzero(as_tuple=True)[0]

                if len(same_ep_mask) > 1:
                    idx_j[k] = same_ep_mask[torch.randint(len(same_ep_mask), (1,))]
                else: 
                    # fall through to random j (already set)
                    pass

        return (
            obs_flat[idx_i],
            rewards_flat[idx_i],
            dones_flat[idx_i],
            obs_next_flat[idx_i],
            obs_flat[idx_j],
            rewards_flat[idx_j],
            dones_flat[idx_j],
            obs_next_flat[idx_j],
        )

    def get_minibatches(self, minibatch_size: int, n_pairs_per_batch: Optional[int] = None) -> Iterator[RolloutBatch]:
        r"""
        Iterator to yield shuffled minibatches for PPO epoch updates. Each minibatch 
        contains policy gradient data and critic pair data.

        Args:
            minibatch_size:     number of policy gradient samples per batch, B.
            n_pairs_per_batch:  number of (i,j) pairs for critic loss (default=B).

        Returns:
            it:                 the rollout batch iterator
        """

        if n_pairs_per_batch is None:
            n_pairs_per_batch = minibatch_size

        T, N = self.T, self.N
        flat_size = T * N

        # flatten
        obs_flat      = self.obs.view(flat_size, *self.obs_shape)
        actions_flat  = self.actions.view(flat_size)
        logp_flat     = self.log_probs.view(flat_size)
        adv_flat      = self.advantages.view(flat_size)
        ret_flat      = self.returns.view(flat_size)

        # shuffle
        perm = torch.randperm(flat_size)
        for start in range(0, flat_size, minibatch_size):
            idx = perm[start : start + minibatch_size]
            if len(idx) < minibatch_size:
                 # drop last incomplete batch
                continue 

            # sample critic pairs for each minibatch
            pairs = self.sample_pairs(n_pairs_per_batch)

            yield RolloutBatch(
                obs = obs_flat[idx].to(self.device),
                actions = actions_flat[idx].to(self.device),
                log_probs = logp_flat[idx].to(self.device),
                advantages = adv_flat[idx].to(self.device),
                returns = ret_flat[idx].to(self.device),
                obs_i = pairs[0].to(self.device),
                r_i = pairs[1].to(self.device),
                d_i = pairs[2].to(self.device),
                obs_i_next = pairs[3].to(self.device),
                obs_j = pairs[4].to(self.device),
                r_j = pairs[5].to(self.device),
                d_j = pairs[6].to(self.device),
                obs_j_next = pairs[7].to(self.device),
            )