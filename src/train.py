r"""
src/train.py

Main PPO + RV training loop. Handles environment rollout, trajectory
ranking, computation of R-GAE, and gradient updates using PPO + RV.
"""

import torch
import numpy as np
from typing import Optional

from src.actor_critic import PPORVActorCritic
from src.rollout import RolloutBuffer
from src.ppo import PPORVUpdater, PPORVConfig
from src.rgae import (
    extract_start_states,
    compute_trajectory_offsets,
    apply_offset_to_rollout,
    compute_rgae,
    compute_relative_values
)


def collect_rollout(
    envs,
    model: PPORVActorCritic,
    buffer: RolloutBuffer,
    obs: torch.Tensor,
    ep_ids: torch.Tensor,
    device: str,
) -> torch.Tensor:
    r"""
    Collect T steps of experience from N_envs parallel environments. Stores 
    transitions into buffer and returns the next starting observation.

    Args:
        envs:   vectorized gym environments of shape (N_envs,).
        model:  PPORVActorCritic model.
        buffer: rollout buffer.
        obs:    current observations of shpae (N_envs, C, H, W).
        ep_ids: episode ID counter per env.
        device: torch device.

    Returns:
        obs:    the next starting observation.
        ep_ids: the updated episode ID counter per env.
    """

    model.eval()
    with torch.no_grad():
        for i in range(buffer.T):
            dist = model.policy(obs.to(device))
            actions = dist.sample()
            log_prob = dist.log_prob(actions)

            # step envs
            # TODO: adapt to gym vectorized env API
            obs_np = obs.cpu().numpy()
            actions_np = actions.cpu().numpy()
            next_obs_np, rewards_np, dones_np, infos = envs.step(actions_np)

            next_obs = torch.tensor(next_obs_np, dtype=torch.float32)
            rewards = torch.tensor(rewards_np, dtype=torch.float32)
            dones = torch.tensor(dones_np, dtype=torch.float32)

            # update episode IDs when done with episode
            ep_ids = ep_ids + dones.long()

            buffer.store(
                obs      = obs,
                obs_next = next_obs,
                action   = actions.cpu(),
                reward   = rewards,
                done     = dones,
                log_prob = log_prob.cpu(),
                ep_id    = ep_ids.clone(),
            )

            obs = next_obs

    model.train()

    return obs, ep_ids


def compute_advantages(
    buffer: RolloutBuffer,
    model: PPORVActorCritic,
    cfg: PPORVConfig,
    device: str,
    use_offset: bool = True,
):
    r"""
    Compute R-GAE advantages for the current rollout. Does this by extracting
    start states per environment, computing pairwise offsets (trajectory ranking),
    and calculating V_\theta and R-GAE.
    """

    model.eval()
    with torch.no_grad():
        delta_fn = lambda s_i, s_j: model.delta(s_i.to(device), s_j.to(device)).cpu()

        # per-env advantage computation
        for env_idx in range(buffer.N):

            states_env = torch.cat([
                buffer.obs[:, env_idx],
                buffer.obs_next[-1, env_idx].unsqueeze(0),
            ], dim=0)

            dones_env  = buffer.dones[:, env_idx]

            if use_offset:
                start_states, start_idx = extract_start_states(states_env, dones_env)

                # get all start states across envs for global ranking
                # TODO currently rank per-env; paper implementation pools across envs
                V_hat = compute_trajectory_offsets(delta_fn, start_states)
                V_rel = apply_offset_to_rollout(delta_fn, states_env, dones_env, V_hat, start_idx)
            else:
                # zero init
                V_rel = compute_relative_values(delta_fn, states_env)

            advantages_env, _ = compute_rgae(
                rewards = buffer.rewards[:, env_idx],
                V_rel = V_rel,
                dones = dones_env,
                gamma = cfg.gamma,
                lam = cfg.lam,
            )

            buffer.advantages[:, env_idx] = advantages_env
            buffer.returns[:, env_idx] = advantages_env + V_rel[:-1]

    model.train()


def train(
    envs,
    n_actions: int,
    total_frames: int = 4e7,
    cfg: PPORVConfig = None,
    device: str,
    use_offset: bool = True,
    log_interval: int = 10,
):
    """
    Full PPO+RV training loop.

    Args:
        envs:          vectorized gym environment (N_envs parallel).
        n_actions:     size of the discrete action space.
        total_frames:  total environment frames (paper: 40M = 10M steps × 4 frame skip).
        cfg:           PPORVConfig; defaults to paper hyperparameters.
        device:        torch device.
        use_offset:    whether to use trajectory ranking offset (Section 4).
        log_interval:  log every k rollouts.
    """

    if cfg is None:
        cfg = PPORVConfig()

    model = PPORVActorCritic(n_actions=n_actions).to(device)
    updater = PPORVUpdater(model, cfg)
    buffer = RolloutBuffer(
        rollout_length = cfg.rollout_length,
        n_envs  = cfg.n_envs,
        obs_shape = (4, 84, 84), # TODO matches atari
        same_ep_prob = cfg.same_ep_prob,
        device = device,
    )

    obs_np = envs.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32) / 255.0
    ep_ids = torch.zeros(cfg.n_envs, dtype=torch.long)

    frames_per_rollout = cfg.rollout_length * cfg.n_envs
    n_rollouts = int(total_frames / frames_per_rollout)
    frames_collected = 0

    for rollout_idx in range(n_rollouts):
        buffer.reset()

        # collect rollout
        obs, ep_ids = collect_rollout(envs, model, buffer, obs, ep_ids, device)
        frames_collected += frames_per_rollout

        # compute r-gae
        compute_advantages(buffer, model, cfg, device, use_offset=use_offset)

        # gradient update
        epoch_logs = {}
        for _ in range(cfg.n_epochs):
            epoch_logs = updater.train_epoch(buffer)


        if rollout_idx % log_interval == 0:
            print(
                f"[{frames_collected:>10,} frames, rollout {rollout_idx}]"
                f"policy={epoch_logs.get('loss_policy', 0):.4f}"
                f"critic={epoch_logs.get('loss_critic', 0):.4f}"
                f"entropy={epoch_logs.get('loss_entropy', 0):.4f}"
            )

    return model