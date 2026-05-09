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

from src.rgae import compute_relative_values, extract_start_states, compute_trajectory_offsets, apply_offset_to_rollout, compute_rgae as _compute_rgae


@dataclass
class RolloutBatch:
    r"""
    A single minibatch for PPO + RV training.
    """

    # policy gradient stuff
    obs: torch.Tensor           # (B, C, H, W)
    actions: torch.Tensor       # (B,)
    log_probs: torch.Tensor     # (B,)  log \pi_old(a|s)
    advantages: torch.Tensor    # (B,)  normalized R-GAE A_t
    returns: torch.Tensor       # (B,)  discounted returns

    # critic loss stuff
    obs_i: torch.Tensor         # (B, C, H, W)  first state of pair
    obs_j: torch.Tensor         # (B, C, H, W)  second state of pair
    r_i: torch.Tensor           # (B,)  reward at i
    r_j: torch.Tensor           # (B,)  reward at j
    d_i: torch.Tensor           # (B,)  done flag for s_{i+1}
    d_j: torch.Tensor           # (B,)  done flag for s_{j+1}
    obs_i_next: torch.Tensor    # (B, C, H, W)
    obs_j_next: torch.Tensor    # (B, C, H, W)

    # for n step loss
    traj_i = None
    traj_j = None


class RolloutBuffer:
    r"""
    Stores a single rollout of T steps * N_envs parallel environments.
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
        self.obs = torch.zeros(T, N, *S, device=self.device)
        self.obs_next = torch.zeros(T, N, *S, device=self.device)
        self.actions = torch.zeros(T, N, dtype=torch.long, device=self.device)
        self.rewards = torch.zeros(T, N, device=self.device)
        self.dones = torch.zeros(T, N, device=self.device)
        self.log_probs = torch.zeros(T, N, device=self.device)

        # create episode IDs
        self.ep_ids = torch.zeros(T, N, dtype=torch.long, device=self.device)

        # empty tensors to pass to R-GAE
        self.advantages = torch.zeros(T, N, device=self.device)
        self.returns = torch.zeros(T, N, device=self.device)
        self.V_rel = torch.zeros(T + 1, N, device=self.device)

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
        self.obs[t] = obs.to(self.device)
        self.obs_next[t] = obs_next.to(self.device)
        self.actions[t] = action.to(self.device)
        self.rewards[t] = reward.to(self.device)
        self.dones[t] = done.to(self.device)
        self.log_probs[t] = log_prob.to(self.device)
        self.ep_ids[t] = ep_id.to(self.device)
        self._ptr += 1

    def reset(self):
        self._ptr = 0

    def compute_rgae(
        self,
        delta_fn,
        model,
        gamma: float,
        lam: float,
        use_offset: bool = True,
    ):
        r"""
        Compute R-GAE advantages and returns for the stored rollout. Does this by
        computing relative values V_\theta (with or without offset), computing R-GAE,
        and normalizing advantages.

        Args:
            delta_fn:       \Delta_\theta as Callable: (obs_i, obs_j) -> (B,).
            model:          PPORV actor critic model
            gamma:          discount factor.
            lam:            GAE lambda.
            use_offset:     whether to apply trajectory ranking offset.
        """

        # collect all start states globally for cross-env ranking.
        all_start_states = []
        all_start_idx = []

        for env_idx in range(self.N):
            states_env = torch.cat([ # shape (T+1, *obs_shape)
                self.obs[:, env_idx],
                self.obs_next[-1, env_idx].unsqueeze(0),
            ], dim=0)
            s_starts, s_idx = extract_start_states(states_env, self.dones[:, env_idx])
            all_start_states.append(s_starts)
            all_start_idx.append(s_idx)

        if use_offset:
            global_starts = torch.cat(all_start_states, dim=0) # (N, *obs_shape)
            V_hat_global  = compute_trajectory_offsets(model, global_starts)

        # per-env R-GAE with offset or zero-anchor.
        ptr = 0
        for env_idx in range(self.N):
            states_env = torch.cat([
                self.obs[:, env_idx],
                self.obs_next[-1, env_idx].unsqueeze(0),
            ], dim=0)
            dones_env = self.dones[:, env_idx]
            start_idx = all_start_idx[env_idx]
            n_starts = len(start_idx)

            if use_offset:
                V_hat_env = V_hat_global[ptr : ptr + n_starts]
                ptr += n_starts
                V_rel = apply_offset_to_rollout(
                    delta_fn, states_env, dones_env, V_hat_env, start_idx
                )
            else:
                V_rel = compute_relative_values(delta_fn, states_env)

            adv_env, _ = _compute_rgae(
                rewards = self.rewards[:, env_idx],
                V_rel = V_rel,
                done = dones_env,
                gamma = gamma,
                lam = lam,
            )
            self.advantages[:, env_idx] = adv_env
            self.returns[:, env_idx] = adv_env + V_rel[:-1]

        # normalize
        adv = self.advantages.view(-1)
        mean = adv.mean()
        std  = adv.std()
        self.advantages.copy_(((adv - mean) / (std + 1e-8)).view(self.T, self.N))
        # self.advantages = (
        #     (adv - adv.mean()) / (adv.std() + 1e-8)
        # ).view(self.T, self.N)

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

        # T, N = self.T, self.N
        # flat_size = T * N

        # # flatten (T, N) -> (T*N,)
        # obs_flat = self.obs.view(flat_size, *self.obs_shape)
        # obs_next_flat = self.obs_next.view(flat_size, *self.obs_shape)
        # rewards_flat = self.rewards.view(flat_size)
        # dones_flat = self.dones.view(flat_size)
        # ep_ids_flat = self.ep_ids.view(flat_size)

        # # sample reference indices i
        # idx_i = torch.randint(0, flat_size, (n_pairs,))

        # # sample j with cross-episode strategy
        # use_same = torch.rand(n_pairs) < self.same_ep_prob
        # idx_j = torch.randint(0, flat_size, (n_pairs,))

        # for k in range(n_pairs):

        #     if use_same[k]:

        #         # get all transitions from the same episode as i
        #         ep_id = ep_ids_flat[idx_i[k]].item()
        #         same_ep_mask = (ep_ids_flat == ep_id).nonzero(as_tuple=True)[0]

        #         if len(same_ep_mask) > 1:
        #             idx_j[k] = same_ep_mask[torch.randint(len(same_ep_mask), (1,))]
        #         else: 
        #             # fall through to random j (already set)
        #             pass

        # return (
        #     obs_flat[idx_i],
        #     rewards_flat[idx_i],
        #     dones_flat[idx_i],
        #     obs_next_flat[idx_i],
        #     obs_flat[idx_j],
        #     rewards_flat[idx_j],
        #     dones_flat[idx_j],
        #     obs_next_flat[idx_j],
        # )

        # vectorized
        T, N = self.T, self.N
        flat_size = T * N

        obs_flat = self.obs.view(flat_size, *self.obs_shape)
        obs_next_flat = self.obs_next.view(flat_size, *self.obs_shape)
        rewards_flat = self.rewards.view(flat_size)
        dones_flat = self.dones.view(flat_size)
        ep_ids_flat = self.ep_ids.view(flat_size)  # (T*N,)

        idx_i = torch.randint(0, flat_size, (n_pairs,), device=self.device)

        # Vectorized same-episode pairing — no Python loop, no .item() calls.
        # For each i, pick a random j from the same episode by:
        #   1. computing a random offset within [0, flat_size)
        #   2. looking up ep_ids_flat at idx_i to get the episode ID
        #   3. building a candidate idx_j as a random same-id index via modular search
        #
        # Simpler equivalent: sample a random offset k in [0, flat_size), then
        # find the nearest index with the same ep_id using a prebuilt lookup.
        # But the simplest correct vectorization: for same-ep pairs, randomly shift
        # within the episode by sampling a second index and accepting if same ep_id,
        # falling back to random otherwise — all in tensor ops.

        # Build same-episode j: sample random indices, replace with i's ep_id match
        idx_j_random = torch.randint(0, flat_size, (n_pairs,), device=self.device)
        
        # For same-episode candidates: randomly pick among all indices, then
        # check if they share ep_id with i. Where they don't, use random j.
        # Do this with a small number of candidate draws (3 tries is enough for
        # short episodes; at least one will match with high probability).
        use_same = torch.rand(n_pairs, device=self.device) < self.same_ep_prob
        ep_ids_i = ep_ids_flat[idx_i]  # (n_pairs,) — no .item(), stays on CPU

        idx_j = idx_j_random.clone()
        for _ in range(4):  # 4 vectorized attempts, no Python loop over pairs
            candidates = torch.randint(0, flat_size, (n_pairs,), device=self.device)
            same_ep    = (ep_ids_flat[candidates] == ep_ids_i)  # (n_pairs,) bool
            # Accept this candidate for pairs that (a) want same-ep and (b) got one
            accept = use_same & same_ep
            idx_j[accept] = candidates[accept]

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

    def sample_trajectory_pairs(self, n_pairs: int, n_steps: int) -> Tuple:
        r"""
        Sample n_pairs of trajectory windows of length n_steps for the n-step critic loss.
        For each sampled start index i, collect the next n_steps transitions from the same 
        env.

        Returns:
            traj_i: list of n_steps tuples (obs, r, done, obs_next), each (n_pairs, *obs).
            traj_j: list of n_steps tuples, same structure.
        """

        # T, N = self.T, self.N

        # # sample start positions within [0, T - 1] for each env
        # env_idx_i = torch.randint(0, N, (n_pairs,))
        # t_idx_i = torch.randint(0, T, (n_pairs,))
        # env_idx_j = torch.randint(0, N, (n_pairs,))
        # t_idx_j = torch.randint(0, T, (n_pairs,))

        # # apply same episode bias for j
        # use_same = torch.rand(n_pairs) < self.same_ep_prob
        # for k in range(n_pairs):
        #     if use_same[k]:
        #         env_idx_j[k] = env_idx_i[k]
        #         ep_id = self.ep_ids[t_idx_i[k], env_idx_i[k]].item()

        #         # find timesteps in the same env with the same episode id
        #         same_mask = (self.ep_ids[:, env_idx_i[k]] == ep_id).nonzero(as_tuple=True)[0]
        #         if len(same_mask) > 0:
        #             t_idx_j[k] = same_mask[torch.randint(len(same_mask), (1,))]

        # # create trajectory lists
        # # for step k in [0, n_steps), get t+k (clipped later)
        # traj_i, traj_j = [], []
        # for k in range(n_steps):
        #     t_i = torch.clamp(t_idx_i + k, 0, T - 1)
        #     t_j = torch.clamp(t_idx_j + k, 0, T - 1)

        #     traj_i.append((
        #         self.obs[t_i, env_idx_i],
        #         self.rewards[t_i, env_idx_i],
        #         self.dones[t_i, env_idx_i],
        #         self.obs_next[t_i, env_idx_i],
        #     ))
        #     traj_j.append((
        #         self.obs[t_j, env_idx_j],
        #         self.rewards[t_j, env_idx_j],
        #         self.dones[t_j, env_idx_j],
        #         self.obs_next[t_j, env_idx_j],
        #     ))

        # return traj_i, traj_j

        # vectorized?
        T, N = self.T, self.N

        # Sample start positions
        env_idx_i = torch.randint(0, N, (n_pairs,), device=self.device)
        t_idx_i = torch.randint(0, T, (n_pairs,), device=self.device)
        env_idx_j = torch.randint(0, N, (n_pairs,), device=self.device)
        t_idx_j = torch.randint(0, T, (n_pairs,), device=self.device)

        # Vectorized same-episode bias — no Python loop, no .item()
        use_same  = torch.rand(n_pairs, device=self.device) < self.same_ep_prob
        ep_ids_i  = self.ep_ids[t_idx_i, env_idx_i]   # (n_pairs,)

        for _ in range(4):
            cand_t   = torch.randint(0, T, (n_pairs,), device=self.device)
            # For same-ep we want j in the same env as i
            cand_env = env_idx_i
            same_ep  = (self.ep_ids[cand_t, cand_env] == ep_ids_i)
            accept   = use_same & same_ep
            t_idx_j[accept]   = cand_t[accept]
            env_idx_j[accept] = cand_env[accept]

        # Build trajectory lists with pure tensor indexing — no Python loop over pairs
        # t_idx_i/j: (n_pairs,); adding k and clamping gives the window for step k
        traj_i, traj_j = [], []
        for k in range(n_steps):
            t_i = torch.clamp(t_idx_i + k, 0, T - 1)  # (n_pairs,)
            t_j = torch.clamp(t_idx_j + k, 0, T - 1)

            traj_i.append((
                self.obs[t_i, env_idx_i],
                self.rewards[t_i, env_idx_i],
                self.dones[t_i, env_idx_i],
                self.obs_next[t_i, env_idx_i],
            ))
            traj_j.append((
                self.obs[t_j, env_idx_j],
                self.rewards[t_j, env_idx_j],
                self.dones[t_j, env_idx_j],
                self.obs_next[t_j, env_idx_j],
            ))

        return traj_i, traj_j

    def get_minibatches(self, minibatch_size: int, n_pairs_per_batch: Optional[int] = None, n_step: int = 1) -> Iterator[RolloutBatch]:
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
        obs_flat = self.obs.view(flat_size, *self.obs_shape)
        actions_flat = self.actions.view(flat_size)
        logp_flat = self.log_probs.view(flat_size)
        adv_flat = self.advantages.view(flat_size)
        ret_flat = self.returns.view(flat_size)

        # shuffle
        perm = torch.randperm(flat_size)
        for start in range(0, flat_size, minibatch_size):
            idx = perm[start : start + minibatch_size]

            # drop last incomplete batch
            if len(idx) < minibatch_size:
                continue 

            # sample critic pairs for each minibatch
            if n_step > 1:
                traj_i, traj_j = self.sample_trajectory_pairs(n_pairs_per_batch, n_step)
                pairs = self.sample_pairs(n_pairs_per_batch) # needed for policy loss i think?
            else:
                pairs = self.sample_pairs(n_pairs_per_batch)
                traj_i, traj_j = None, None

            yield RolloutBatch(
                obs = obs_flat[idx],
                actions = actions_flat[idx],
                log_probs = logp_flat[idx],
                advantages = adv_flat[idx],
                returns = ret_flat[idx],
                obs_i = pairs[0],
                r_i = pairs[1],
                d_i = pairs[2],
                obs_i_next = pairs[3],
                obs_j = pairs[4],
                r_j = pairs[5],
                d_j = pairs[6],
                obs_j_next = pairs[7],
            )