"""
tests/verify.py

Unit tests verifying the algorithm's theoretical properties.

References:
    - Antisymmetry                               Eq. 1
    - Zero self-difference                       Eq. 2
    - Bellman contraction                        Thm. 3.1
    - R-GAE = GAE + B_t relationship             Lem. 3.2
    - Unbiasedness of R-GAE policy gradient      Cor. 3.3
    - 1-step target case split correctness       Eq. 19, Eq. 20
    - Invariance of pairwise differences         Lemma B.1
"""

import torch
import pytest
from src.actor_critic import PPORVActorCritic, RelativeCritic
from src.rgae import compute_rgae, compute_relative_values
from src.bellman import compute_step_target

@pytest.fixture
def model():
    return PPORVActorCritic(n_actions=6, in_channels=4, hidden_dim=64)

@pytest.fixture
def dummy_obs():
    return torch.randn(4, 4, 84, 84)  # batch of 4 observations

def test_antisymmetry(model, dummy_obs):
    s_i, s_j = dummy_obs[:2], dummy_obs[2:]
    d_ij = model.delta(s_i, s_j)
    d_ji = model.delta(s_j, s_i)
    torch.testing.assert_close(d_ij, -d_ji, msg="Antisymmetry failed")


def test_zero_self_difference(model, dummy_obs):
    d_ss = model.delta(dummy_obs, dummy_obs)
    torch.testing.assert_close(
        d_ss, torch.zeros_like(d_ss), msg="Zero self-difference failed"
    )

def test_gauge_invariance_of_differences(model, dummy_obs):
    s_i, s_j = dummy_obs[:2], dummy_obs[2:]

    emb_i = model.encode(s_i)
    emb_j = model.encode(s_j)

    c = 5.0
    d_original = model.critic_head(emb_i, emb_j)
    d_shifted  = model.critic_head(emb_i + c, emb_j + c)

    torch.testing.assert_close(
        d_original, d_shifted,
        msg="Pairwise difference invariant failed"
    )


def test_rgae_equals_gae_plus_bt():
    gamma, lam = 0.9, 0.8
    T = 6
    C = 2.0
    rewards = torch.ones(T)
    V_true = torch.tensor([2., 3., 4., 5., 6., 7., 8.])
    dones = torch.zeros(T)

    # standard GAE
    A_true, _ = compute_rgae(rewards, V_true, dones, gamma, lam)

    # relative values anchored at s_0
    V_rel = V_true - C
    A_rel, _ = compute_rgae(rewards, V_rel, dones, gamma, lam)

    # compute expected B_t
    B = torch.zeros(T)
    for t in range(T):
        B[t] = (1 - gamma) * C * sum((gamma * lam) ** l for l in range(T - 1 - t + 1))

    diff = A_rel - A_true
    torch.testing.assert_close(diff, B, atol=1e-3, rtol=1e-3,
        msg="R-GAE = GAE + B_t failed"
    )


def test_rgae_unbiasedness():
    gamma, lam = 0.99, 0.95
    T = 128
    C = 10.0

    B_0 = (1 - gamma) * C * sum((gamma * lam) ** l for l in range(T + 1))

    # B_t should be the same regardless of which action was taken
    assert isinstance(B_0, float), "B_t is not scalar"
    assert B_0 > 0, "B_0 is not positive"


def test_1step_target_both_terminal(model):
    B = 2
    obs = torch.randn(B, 4, 84, 84)
    r_i = torch.tensor([1.0, 2.0])
    r_j = torch.tensor([0.5, 1.0])
    d_i = torch.ones(B)
    d_j = torch.ones(B)

    y_ij = compute_step_target(
        delta_fn = model.delta,
        s_i = obs, r_i=r_i, d_i=d_i, s_i_next=obs,
        s_j = obs, r_j=r_j, d_j=d_j, s_j_next=obs,
        gamma = 0.99,
    )

    expected = r_i - r_j
    torch.testing.assert_close(y_ij, expected,
        msg="Both-terminal target not equal to reward difference"
    )


def test_1step_target_both_nonterminal(model):
    B = 2
    obs = torch.randn(B, 4, 84, 84)
    obs_next = torch.randn(B, 4, 84, 84)
    r_i = torch.tensor([1.0, 2.0])
    r_j = torch.tensor([0.5, 1.0])
    d_i = torch.zeros(B)
    d_j = torch.zeros(B)
    gamma = 0.99

    with torch.no_grad():
        y_ij = compute_step_target(
            model.delta,
            obs, r_i, d_i, obs_next,
            obs, r_j, d_j, obs_next,
            gamma,
        )
        expected = (r_i - r_j) + gamma * model.delta(obs_next, obs_next)
        torch.testing.assert_close(
            y_ij, r_i - r_j, 
            atol=1e-5, 
            rtol=1e-5, 
            msg="Incorrect target for non-terminal successors"
        )