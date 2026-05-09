r"""
src/bellman.py

Pairwise Bellman operator and value targets for Relative Value Learning.

References:
    - Pairwise Bellman operator:        Eq. 8 , Thm. 3.1
    - 1-step target:                    Eq. 19, Eq. 20, App. A
    - n-step target:                    Eq. 21
    - lambda-return target:             Eq. 22
""" 

import torch
import torch.nn.functional as F
from typing import Tuple

def _bootstrap_delta(
    delta_fn,
    s_i: torch.Tensor,
    d_i: torch.Tensor,
    s_i_next: torch.Tensor,
    r_i: torch.Tensor,
    s_j: torch.Tensor,
    d_j: torch.Tensor,
    s_j_next: torch.Tensor,
    r_j: torch.Tensor,
) -> torch.Tensor:
    r"""
    Compute the bootstrap term delta_ij.
 
    Rewrites \Delta(s_{i,next}, s_{j,next}) in terms of non-terminal
    pairwise differences and observable rewards, for all four terminal
    combinations:
        case 1 (d_i=0, d_j=0): delta = \Delta(s_{i+1}, s_{j+1})
        case 2 (d_i=1, d_j=0): delta = \Delta(s_i, s_{j+1}) - r_i
        case 3 (d_i=0, d_j=1): delta = \Delta(s_{i+1}, s_j) + r_j
        case 4 (d_i=1, d_j=1): delta = 0
 
    Returns:
        delta_ij: unscaled bootstrap term of shape (B,).
    """

    c1 = (1.0 - d_i) * (1.0 - d_j)
    c2 = d_i * (1.0 - d_j)
    c3 = (1.0 - d_i) * d_j
 
    delta_ij = (
        c1 * delta_fn(s_i_next, s_j_next)
        + c2 * (delta_fn(s_i, s_j_next) - r_i)
        + c3 * (delta_fn(s_i_next, s_j) + r_j)
    )

    return delta_ij


def compute_step_target(
    delta_fn,                   # callable: (s_i, s_j) -> scalar tensor
    s_i, r_i, d_i, s_i_next,    # transition i
    s_j, r_j, d_j, s_j_next,    # transition j
    gamma: float,
) -> torch.Tensor:
    r"""
    Compute the 1-step pairwise TD target y^(1)_ij.

    Handles all four terminal/non-terminal combinations for successor states
    by rewriting \Delta(s_{i+1}, s_{j+1}) in terms of observable rewards
    and non-terminal pairwise differences.

    Args:
        delta_fn:           The learned ∆_θ(·,·) — called with (s_i, s_j) -> R.
        s_i, s_j:           Current states (batch tensors).
        r_i, r_j:           Immediate rewards (batch scalars).
        d_i, d_j:           Terminal flags (1 = terminal) for successor states.
        s_i_next, s_j_next: Successor states.
        gamma:              Discount factor.

    Returns:
        y_ij:               1-step target, shape (B,).
    """

    # reward_diff = r_i - r_j

    # # compute delta_ij bootstrap term
    # # case 1: both non-terminal: \Delta(s_{i+1}, s_{j+1})
    # # case 2: i terminal, j non-terminal: \Delta(s_i, s_{j+1}) - r_i
    # # case 3: i non-terminal, j terminal: \Delta(s_{i+1}, s_j) + r_j
    # # case 4: both terminal: 0 (implicitly handled by absorbing states)

    # both_live  = (1 - d_i) * (1 - d_j)
    # i_terminal = d_i * (1 - d_j)
    # j_terminal = (1 - d_i) * d_j

    # d1 = both_live  * delta_fn(s_i_next, s_j_next)
    # d2 = i_terminal * (delta_fn(s_i, s_j_next) - r_i)
    # d3 = j_terminal * (delta_fn(s_i_next, s_j) + r_j)

    # delta_ij = d1 + d2 + dc3

    # y_ij = reward_diff + gamma * delta_ij
    # return y_ij

    delta_ij = _bootstrap_delta(
        delta_fn, s_i, d_i, s_i_next, r_i, s_j, d_j, s_j_next, r_j
    )
    return (r_i - r_j) + gamma * delta_ij


def compute_nstep_target(delta_fn, traj_i, traj_j, gamma: float, n: int) -> torch.Tensor:
    r"""
    Compute the n-step pairwise target y^(n)_ij.

    Assumes neither trajectory terminates within the n-step window. The final 
    bootstrapped difference uses the case split from compute_step_target
    for the successor pair at step n. The window is clipped to the minimum
    trajectory length.

    Args:
        traj_i:     List of (state, reward, done, next_state) tuples, length >= n.
        traj_j:     List of (state, reward, done, next_state) tuples, length >= n.
        gamma:      Discount factor.
        n:          Number of steps.

    Returns:
        y^n_ij:     N-step target tensor.
    """

    # satisfy termination assumption
    window = min(n, len(traj_i), len(traj_j))
    B = traj_i[0][1].shape[0]

    # Accumulate discounted reward differences over n steps
    # y^(n)_ij = \sum_{k=0}^{n-1} gamma^k (r_{i+k} - r_{j+k}) + gamma^n * \Delta(s_{i+n}, s_{j+n})
    # TODO: match batch shape
    reward_sum = torch.zeros(B)
    alive_i = torch.ones(B)
    alive_j = torch.ones(B)

    for k in range(window):
        s_i, r_i, d_i, _ = traj_i[k]
        s_j, r_j, d_j, _ = traj_j[k]
        reward_sum = reward_sum + (gamma ** k) * (
            alive_i * r_i - alive_j * r_j
        )

        # update alive masks
        alive_i = alive_i * (1.0 - d_i)
        alive_j = alive_j * (1.0 - d_j)


    s_i_n, r_i_n, d_i_n, sn_i_n = traj_i[window - 1]
    s_j_n, r_j_n, d_j_n, sn_j_n = traj_j[window - 1]

    # extract s_next and apply case split
    bootstrap = _bootstrap_delta(
        delta_fn,
        s_i_n,
        torch.clamp(d_i_n + (1.0 - alive_i), 0.0, 1.0),
        sn_i_n,
        r_i_n,
        s_j_n,
        torch.clamp(d_j_n + (1.0 - alive_j), 0.0, 1.0),
        sn_j_n,
        r_j_n,
    )

    return reward_sum + (gamma ** window) * bootstrap


def compute_lambda_return(
    delta_fn,
    traj_i,
    traj_j,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    r"""
    Compute the pairwise lambda-return y^(\lambda)_ij. Truncates at first terminal
    state to length L. 

        y^(\lambda)_ij = (1 - \lambda) * \sum_{n=1}^\infty \lambda^{n-1} * y^(n)_ij
  
    Implemented with backward TD(\lambda) recursion:
    Instead of evaluating each y^(n) separately (slow), we use the TD(\lambda) 
    identity. Define the pairwise 1-step TD residual at step k:

        delta^pair_k = (r_{i+k} - r_{j+k})
                        + gamma * \Delta_\theta(s_{i+k+1}, s_{j+k+1})
                        - \Delta_\theta(s_{i+k}, s_{j+k})
 
    where the \Delta_\theta(s_{i+k+1}, s_{j+k+1}) bootstrap uses the
    case split at the final step and a direct call for all other steps.
 
    By telescoping identity:
 
        y^(lambda)_ij = \Delta_\theta(s_i, s_j)
                        + sum_{k=0}^{L-1} (\gamma * \lambda)^k * \delta^pair_k
 
    The sum is accumulated in a single backward pass:
 
        G_k = \delta^pair_k + \gamma * \lambda * mask_k * G_{k+1}
 
    where mask_k = (1 - d_{i+k}) * (1 - d_{j+k}) cuts the carry across episode boundaries.
 
    Then: 
        
        y^(\lambda)_ij = \Delta_\theta(s_i, s_j) + G_0.
 
    For special cases (not implemented):
    lam = 0 -> 1-step TD target (G = delta^pair_0)
    lam = 1 -> full Monte-Carlo return (no bootstrapping cutoff)
 
    Args:
        delta_fn:  \Delta_\theta function as Callable: (B, *obs),(B, *obs) -> (B,).
        traj_i:    full trajectory tuple, where each is (s, r, d, s_next).
        traj_j:    full trajectory tuple, where each is (s, r, d, s_next).
        gamma:     discount factor.
        lam:       lambda parameter.
 
    Returns:
        y_lam: lambda-return pairwise target of shape (B,).
    """

    L = min(len(traj_i), len(traj_j))
    B = traj_i[0][1].shape[0]
 
    # pre-compute \Delta_\theta values needed for TD residuals.
    # delta_current[k] = \Delta_\theta(s_{i+k}, s_{j+k}),  for k = 0 ... L-1
    # delta_next[k] = \Delta_\theta(s_{i+k+1}, s_{j+k+1}), for k = 0 ... L-1
    #   - for k < L-1: direct call (both s_next are available in the trajectory).
    #   - for k = L-1: apply the case split, since s_{i+L} or s_{j+L}
    #                  may be terminal.
    with torch.no_grad():
        delta_current = [
            delta_fn(traj_i[k][0], traj_j[k][0])
            for k in range(L)
        ]
 
        # bootstrap for the final step with case split
        s_i_n, r_i_n, d_i_n, sn_i_n = traj_i[L - 1]
        s_j_n, r_j_n, d_j_n, sn_j_n = traj_j[L - 1]
        delta_next_final = _bootstrap_delta(
            delta_fn,
            s_i_n,
            d_i_n,
            sn_i_n,
            r_i_n,
            s_j_n,
            d_j_n,
            sn_j_n,
            r_j_n,
        )
 
        delta_next = [
            delta_fn(traj_i[k][3], traj_j[k][3])
            if k < L - 1
            else delta_next_final
            for k in range(L)
        ]
 

    G = torch.zeros(B)
 
    for k in reversed(range(L)):
        step_i = traj_i[k]
        step_j = traj_j[k]
 
        # pairwise 1-step td residual at step k
        delta_pair_k = (
            (step_i[1] - step_j[1])
            + gamma * delta_next[k]
            - delta_current[k]
        )
 
        # mask the lambda at episode boundaries
        # if either traj ends, future residuals should not backprop
        mask = (1.0 - step_i[2]) * (1.0 - step_j[2])
 
        G = delta_pair_k + gamma * lam * mask * G
 
    y_lam = delta_current[0] + G
    return y_lam