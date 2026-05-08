r"""
src/ppo.py

PPO + RV loss functions and update step for training loop.

Paper references:
    - PPO clipped surrogate:    Eq. 27
    - Critic loss:              Eq. 28
    - Entropy bonus:            Eq. 28
    - Combined loss:            Eq. 29
    - Hyperparameters:          App. D, App. E
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dataclasses import dataclass
from typing import Dict

from src.actor_critic import PPORVActorCritic
from src.rollout import RolloutBatch, RolloutBuffer
from src.bellman import compute_step_target


@dataclass
class PPORVConfig:
    r"""
    Set hyperparameters.
    """
    
    gamma:          float = 0.99
    lam:            float = 0.95    # GAE lambda
    n_step:         int   = 5       # n-step return target for critic
    clip_eps:       float = 0.1     # PPO clip epsilon
    n_epochs:       int   = 5       # PPO epochs per rollout
    minibatch_size: int   = 128
    n_envs:         int   = 8
    rollout_length: int   = 128     # T
    lr:             float = 2.5e-4  # learning rate
    adam_eps:       float = 1e-5
    entropy_coef:   float = 0.01    # c_e
    critic_coef:    float = 1.25    # c_v
    rv_clip:        float = 0.15    # RV value clipping
    max_grad_norm:  float = 0.5
    same_ep_prob:   float = 0.33    # Appendix E


class PPORVUpdater:
    r"""
    Computes PPO+RV losses and performs one gradient update step.
    Loss is L = -L_policy(\theta) + c_v * L_critic(\theta) + c_e * L_entropy(\theta)
    """

    def __init__(self, model: PPORVActorCritic, config: PPORVConfig):
        self.model  = model
        self.cfg    = config
        self.optim  = Adam(model.parameters(), lr=config.lr, eps=config.adam_eps)

    def policy_loss(
        self,
        batch: RolloutBatch,
    ) -> torch.Tensor:
        r"""
        PPO clipped surrogate objective.

        L_policy = E_t [ min(
            r_t(\theta) A_t, 
            clip(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
        ) ]

        where r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_old(a_t|s_t)
        and A_t is the relative advantages.
        """
        
        dist = self.model.policy(batch.obs)
        log_prob = dist.log_prob(batch.actions)

        ratio = torch.exp(log_prob - batch.log_probs)
        
        # normalize advantages within the minibatch
        adv = batch.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        clipped_ratio = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
        L_policy = torch.min(ratio * adv, clipped_ratio * adv).mean()

        return L_policy

    def critic_loss(self, batch: RolloutBatch) -> torch.Tensor:
        r"""
        Pairwise critic MSE loss using n-step targets.

        L_critic = 0.5 * E_{(i,j) \sim \mu} [ (\Delta_\theta(s_i, s_j) - y^(n)_ij)^2 ]

        Uses the 1-step target. 
        TODO Extend to n-step by composing compute_step_target.
        """

        # current pred
        delta_pred = self.model.delta(batch.obs_i, batch.obs_j)

        # compute 1-step target
        with torch.no_grad():
            y_ij = compute_step_target(
                delta_fn=self.model.delta,
                s_i=batch.obs_i, r_i=batch.r_i, d_i=batch.d_i, s_i_next=batch.obs_i_next,
                s_j=batch.obs_j, r_j=batch.r_j, d_j=batch.d_j, s_j_next=batch.obs_j_next,
                gamma=self.cfg.gamma,
            )

        # clip the value target (not needed?)
        # delta_pred_clipped = torch.clamp(
        #     delta_pred,
        #     y_ij - self.cfg.rv_clip,
        #     y_ij + self.cfg.rv_clip,
        # )

        L_critic = 0.5 * F.mse_loss(delta_pred, y_ij)
        return L_critic

    def entropy_loss(self, batch: RolloutBatch) -> torch.Tensor:
        r"""
        Policy entropy loss.

        L_ent = -E_t [ H(\pi_\theta(.|s_t)) ]
        """
        dist = self.model.policy(batch.obs)
        return -dist.entropy().mean()

    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        r"""
        Compute combined loss and perform one gradient step. Returns dict 
        of scalar loss values for logging.
        """
        
        L_policy = self.policy_loss(batch)
        L_critic = self.critic_loss(batch)
        L_entropy    = self.entropy_loss(batch)

        loss = (
            -L_policy + 
            self.cfg.critic_coef * L_critic + 
            self.cfg.entropy_coef * L_entropy
        )

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optim.step()

        return {
            "loss_total":   loss.item(),
            "loss_policy":  L_policy.item(),
            "loss_critic":  L_critic.item(),
            # "loss_entropy": -L_ent.item(),   # log as positive entropy
            "loss_entropy": L_entropy.item(),
        }

    def train_epoch(self, buffer: RolloutBuffer) -> Dict[str, float]:
        r"""
        Run one PPO epoch over the rollout buffer. Called n_epochs times per 
        rollout.
        """

        epoch_logs: Dict[str, list] = {
            "loss_total":   [],
            "loss_policy":  [],
            "loss_critic":  [],
            "loss_entropy": [],
        }

        for batch in buffer.get_minibatches(self.cfg.minibatch_size):
            logs = self.update(batch)
            for k, v in logs.items():
                epoch_logs[k].append(v)

        return {k: sum(v) / len(v) for k, v in epoch_logs.items()}