r"""
src/rgae.py

Relative Generalized Advantage Estimation (R-GAE) and trajectory ranking.

Paper references:
    - Relative value sequence:      Eq. 9
    - Relative TD residuals:        Eq. 10
    - R-GAE:                        Eq. 11
    - GAE vs R-GAE relationship:    Lem. 3.2
    - Unbiasedness:                 Cor. 3.3
    - Trajectory ranking:           Eq. 23, Eq. 24, Eq. 25, Eq. 26, Sec. 4
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def compute_relative_values(
    delta_fn,
    states: torch.Tensor,   # (T+1, *obs_shape) — rollout states including final
) -> torch.Tensor:
    r"""
    Build the relative value sequence V_\theta(s_t) by telescoping pairwise 
    differences. This is done as follows:
    V_\theta(s_0) = 0
    V_\theta(s_t) = \sum_{k=0}^{t-1} \Delta_\theta(s_{k+1}, s_k)

    Note V_\theta(s_t) = \Delta_\theta(s_t, s_0) is the same as above if using 
    s_0 as the anchor (\Delta_\theta = \Delta^\pi) but is more efficient.

    Args:
        delta_fn:   relative value difference, a Callable (s_i, s_j) -> (B,).
        states:     states of shape (T+1, *obs_shape).

    Returns:
        V_rel:      relative values anchored at 0 of shape (T+1,).
    """

    T_plus1 = states.shape[0]
    V_rel = torch.zeros(T_plus1)

    # telescoping (slow)
    for t in range(1, T_plus1):
        V_rel[t] = V_rel[t - 1] + delta_fn(states[t], states[t - 1])

    # TODO
    # direct anchor (fast)
    # s0 = states[0].unsqueeze(0).expand_as(states[1:])
    # V_rel[1:] = delta_fn(states[1:], s0)

    return V_rel


def compute_rgae(
    rewards: torch.Tensor,
    V_rel:   torch.Tensor,
    done:   torch.Tensor,
    gamma:   float,
    lam:     float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute Relative GAE advantages A_t and relative TD residuals \delta_t.

    \delta_t = r_t + \gamma * V_\theta(s_{t+1}) - V_\theta(s_t)
    A_t = \sum_{l=0}^{T-t} (\gamma * \lambda)^l \delta_{t+l}

    Note that A_t = A'_t + B_t where B_t = (1-\gamma) * C * \sum(\gamma * \lambda)^l
    and A'_t is the absolute advantage is a trajectory-constant offset. It does not bias
    the policy gradient because B_t is independent of a_t.

    Args:
        rewards:        immediate rewards r_t for t=0,...,T-1 of shape (T,).
        V_rel:          relative state values V_\theta(s_t) for t=0,...,T of shape (T+1,).
        done:           episode termination flags of shape (T,).
        gamma:          discount factor
        lam:            GAE lambda parameter.

    Returns:
        advantages:     A_t of shape (T,).
        td_residuals:   \delta_t, shape (T,).
    """

    T = rewards.shape[0]
    advantages   = torch.zeros(T)
    td_residuals = torch.zeros(T)

    # relative TD residuals
    for t in range(T):

        # mask next-state value if episode ended
        next_val = V_rel[t + 1] * (1.0 - done[t])
        td_residuals[t] = rewards[t] + gamma * next_val - V_rel[t]

    # RGAE with backward recursion
    gae = 0.0
    for t in reversed(range(T)):
        gae = td_residuals[t] + gamma * lam * (1.0 - done[t]) * gae
        advantages[t] = gae

    return advantages, td_residuals


def extract_start_states(states: torch.Tensor, done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Extract start-state indices K^m from a rollout.

    K^m = {0} \union { t | d_{t-1} = 1 }

    Args:
        states:         states of shape (T, *obs_shape).
        done:          done flags of shape (T,).

    Returns:
        start_states:   start states from the rollout of shape (T, *obs_shape)
        start_indices:  rollout integer indices of shape (T,).
    """

    indices = [0]
    for t in range(1, states.shape[0]):
        if done[t - 1].item() == 1:
            indices.append(t)
    
    start_indices  = torch.tensor(indices, dtype=torch.long)
    start_states   = states[start_indices]

    return start_states, start_indices


def compute_trajectory_offsets(delta_fn, all_start_states: torch.Tensor) -> torch.Tensor:
    r"""
    Estimate value offsets via row-wise averaging of \Delta_\theta for each start
    state. Estimates are non-negative.
    \Delta_ij = \Delta_\theta(s^(i)_start, s^(j)_start)
    O(s^n_start) = (1/N) \sum_j \Delta_nj
    V_\theta(s^n_start) = O(n) - min_\gamma O(\gamma)

    TODO
    Note that the O(N^2) pairwise computation is feasible for small N. For large N (Atari), 
    subsample pairs.

    Args:
        delta_fn:           delta function as Callable (s_i_batch, s_j_batch) -> (N, N) or looped scalar.
        all_start_states:   N start states in shape (N, *obs_shape).

    Returns:
        V_hat:              non-negative offset estimates (N,)
    """

    N = all_start_states.shape[0]

    # construct N*N pairwise matrix
    # TODO: vectorize with broadcasting for efficiency
    delta_matrix = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            delta_matrix[i, j] = delta_fn(
                all_start_states[i].unsqueeze(0),
                all_start_states[j].unsqueeze(0),
            ).squeeze()

    # row-wise mean
    O = delta_matrix.mean(dim=1)

    # non-negation
    V_hat = O - O.min()

    return V_hat


def apply_offset_to_rollout(states: torch.Tensor, done: torch.Tensor, V_hat_starts: torch.Tensor, start_indices: torch.Tensor) -> torch.Tensor:
    r"""
    Compute offset-corrected relative values V_\theta(s). For each 
    state, we use the most recent start state in the same episode.

    V_\theta(s) = V_\theta(s_start) + \Delta_\theta(s, s_start)

    Args:
        delta_fn:       delta function as Callable (s_i_batch, s_j_batch) -> (N, N) or looped scalar.
        states:         states as (T+1, *obs_shape).
        done:           done vector as (T,).
        V_hat_starts:   trajectory offsets as (N_starts,).
        start_indices:  start indices as (N_starts,).

    Returns:
        V_bar:          offset-corrected relative values of shape (T+1,).
    """

    T_plus1 = states.shape[0]
    V_bar   = torch.zeros(T_plus1)

    # timestep -> most recent state mapping
    start_ptr = 0
    for t in range(T_plus1):
        if start_ptr + 1 < len(start_indices) and t >= start_indices[start_ptr + 1]:
            start_ptr += 1

        s_start = states[start_indices[start_ptr]].unsqueeze(0)
        offset = V_hat_starts[start_ptr]
        delta_t = delta_fn(states[t].unsqueeze(0), s_start).squeeze()
        V_bar[t] = offset + delta_t

    return V_bar