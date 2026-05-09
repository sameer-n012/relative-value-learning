r"""
src/train.py

Main PPO + RV training loop. Handles environment rollout, trajectory
ranking, computation of R-GAE, and gradient updates using PPO + RV.
"""

import torch
import numpy as np
from typing import Optional
from tqdm import tqdm
from dataclasses import asdict
import json
import time



from src.actor_critic import PPORVActorCritic, PPOAVActorCritic
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

    returns = []
    eps_lengths = []
    curr_returns = torch.zeros(buffer.N)
    curr_lengths = torch.zeros(buffer.N)

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

            curr_returns += rewards
            curr_lengths += 1
            for env_idx, done in enumerate(dones):
                if done:
                    returns.append(curr_returns[env_idx].item())
                    eps_lengths.append(curr_lengths[env_idx].item())
                    curr_returns[env_idx] = 0.0
                    curr_lengths[env_idx] = 0.0

            # update episode IDs when done with episode
            ep_ids = ep_ids + dones.long()

            buffer.store(
                obs = obs,
                obs_next = next_obs,
                action = actions.cpu(),
                reward = rewards,
                done = dones,
                log_prob = log_prob.cpu(),
                ep_id = ep_ids.clone(),
            )

            obs = next_obs

    model.train()

    mean_return = np.mean(returns) if returns else float("nan")
    mean_length = np.mean(eps_lengths) if eps_lengths else float("nan")

    return obs, ep_ids, {"mean_return": mean_return, "mean_length": mean_length}


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

    if not model.is_relative:
        # Standard GAE with absolute values
        model.eval()
        with torch.no_grad():
            all_obs = torch.cat([
                buffer.obs.view(-1, *buffer.obs_shape),
                buffer.obs_next[-1].view(buffer.N, *buffer.obs_shape),
            ], dim=0)
            # values for all T*N obs + N final obs
            obs_flat = buffer.obs.reshape(-1, *buffer.obs_shape)
            V = model.value(obs_flat).view(buffer.T, buffer.N)
            V_next_last = model.value(buffer.obs_next[-1])

        for env_idx in range(buffer.N):
            V_env = torch.cat([V[:, env_idx], V_next_last[env_idx:env_idx+1]])
            adv_env, _ = compute_rgae(
                rewards=buffer.rewards[:, env_idx],
                V_rel=V_env,
                done=buffer.dones[:, env_idx],
                gamma=cfg.gamma, lam=cfg.lam,
            )
            buffer.advantages[:, env_idx] = adv_env
            buffer.returns[:, env_idx]    = adv_env + V[:, env_idx]
        model.train()
        return

    model.eval()
    with torch.no_grad():
        delta_fn = lambda s_i, s_j: model.delta(s_i.to(device), s_j.to(device))

        # pool start states across all envs for global ranking.
        all_start_states, all_start_idx = [], []
        for env_idx in range(buffer.N):
            states_env = torch.cat([
                buffer.obs[:, env_idx],
                buffer.obs_next[-1, env_idx].unsqueeze(0),
            ], dim=0)
            s_starts, s_idx = extract_start_states(states_env, buffer.dones[:, env_idx])
            all_start_states.append(s_starts)
            all_start_idx.append(s_idx)

        if use_offset:
            V_hat_global = compute_trajectory_offsets(
                model, torch.cat(all_start_states, dim=0)
            )

        # per env R-GAE.
        ptr = 0
        for env_idx in range(buffer.N):
            states_env = torch.cat([
                buffer.obs[:, env_idx],
                buffer.obs_next[-1, env_idx].unsqueeze(0),
            ], dim=0)
            dones_env = buffer.dones[:, env_idx]
            start_idx = all_start_idx[env_idx]

            if use_offset:
                n_starts = len(start_idx)
                V_hat_env = V_hat_global[ptr : ptr + n_starts]
                ptr += n_starts
                V_rel = apply_offset_to_rollout(
                    delta_fn, states_env, dones_env, V_hat_env, start_idx
                )
            else:
                V_rel = compute_relative_values(delta_fn, states_env)

            adv_env, _ = compute_rgae(
                rewards=buffer.rewards[:, env_idx],
                V_rel=V_rel, done=dones_env, gamma=cfg.gamma, lam=cfg.lam,
            )
            buffer.advantages[:, env_idx] = adv_env
            buffer.returns[:, env_idx]    = adv_env + V_rel[:-1]

    model.train()


def train(
    envs,
    n_actions: int,
    total_frames: int = 4e7,
    cfg: PPORVConfig = None,
    device: Optional[str] = None,
    log_interval: int = 100,
    args = None # extra argparse args
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

    if device is None:
        device = ("cuda" if torch.cuda.is_available() else "cpu")

    train_results = asdict(cfg)
    train_results.update(vars(args))
    train_results["device"] = device
    train_results["log"] = []

    obs_np = envs.reset()
    obs_shape = obs_np.shape[1:] # (C,H,W)


    if not args.baseline:
        model = PPORVActorCritic(
            n_actions=n_actions,
            in_channels=obs_shape[0],
            input_size=obs_shape[1],
        ).to(device)
    else:
        model = PPOAVActorCritic(
            n_actions=n_actions,
            in_channels=obs_shape[0],
            input_size=obs_shape[1],
        ).to(device)
    updater = PPORVUpdater(model, cfg)
    buffer = RolloutBuffer(
        rollout_length = cfg.rollout_length,
        n_envs  = cfg.n_envs,
        obs_shape = obs_shape,
        same_ep_prob = cfg.same_ep_prob,
        device = device,
    )

    # obs = torch.tensor(obs_np, dtype=torch.float32) / 255.0
    obs = torch.tensor(obs_np, dtype=torch.float32)
    ep_ids = torch.zeros(cfg.n_envs, dtype=torch.long)

    frames_per_rollout = cfg.rollout_length * cfg.n_envs
    n_rollouts = int(total_frames / frames_per_rollout)
    frames_collected = 0

    pbar = tqdm(range(n_rollouts), total=n_rollouts, desc="Training", unit="rollout")
    for rollout_idx in pbar:
        buffer.reset()
        timings = {}

        t0 = time.perf_counter()

        # collect rollout
        obs, ep_ids, rollout_stats = collect_rollout(envs, model, buffer, obs, ep_ids, device)
        frames_collected += frames_per_rollout

        # compute r-gae
        compute_advantages(buffer, model, cfg, device, use_offset=not args.no_offset)

        # gradient update
        epoch_logs = {}
        for _ in range(cfg.n_epochs):
            epoch_logs = updater.train_epoch(buffer)

        t0 = time.perf_counter() - t0

        train_results["log"].append(epoch_logs)
        train_results["log"][-1]["frames_collected"] = frames_collected
        train_results["log"][-1]["rollout_idx"] = rollout_idx
        train_results["log"][-1]["mean_return"] = rollout_stats["mean_return"]
        train_results["log"][-1]["mean_eps_length"] = rollout_stats["mean_length"]
        train_results["log"][-1]["time"] = t0

        if rollout_idx % log_interval == 0:
            tqdm.write(
                f"[{frames_collected} frames, rollout {rollout_idx}]" +
                f" policy={epoch_logs.get('loss_policy', 0):.4f}" +
                f" critic={epoch_logs.get('loss_critic', 0):.4f}" +
                f" entropy={epoch_logs.get('loss_entropy', 0):.4f}" +
                f" total={epoch_logs.get('loss_total', 0):.4f}" +
                f" return={rollout_stats["mean_return"]:.4f}" +
                f" length={rollout_stats["mean_length"]:.4f}"
            )

        pbar.set_postfix({"loss": f"{epoch_logs.get('loss_total', 0):.3f}"})
        
        if args.results_file is not None:
            with open(args.results_file, "w") as f:
                json.dump(train_results, f, indent=2)

    return model