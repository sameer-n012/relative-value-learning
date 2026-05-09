r"""
src/main.py

Main script for running PPO + RV on MinAtar and full Atari benchmarks.

MinAtar is the recommended starting point given GPU constraints (see proposal).
MinAtar games: asterix, breakout, freeway, seaquest, space_invaders
  - 10x10 observation grid vs 84x84 Atari frames
  - Much cheaper: ~1-2 GPU-hours per game vs 65 A100-hours on Atari

Usage:
    - `python experiments/run.py --env minatar:asterix --seed 0`
    - `python experiments/run.py --env ALE/Pong-v5    --seed 0 --atari`

References:
    - EnvPool usage: Sec. 6

"""

import argparse
import torch
import numpy as np
from minatar import Environment
# import envpool
import re

from src.train import train, PPORVConfig


def slugify(s):
    return re.sub(r'[^A-Za-z0-9_]', '_', s)


class MinAtarVecEnv:
    r"""
    Simple sequential vectorized wrapper for MinAtar environments.

    API matches the subset used in train.py:
        - reset() -> np.ndarray (N, C, H, W)
        - step(actions) -> (obs, rewards, dones, infos)

    MinAtar observations are originally (H, W, C); this wrapper
    converts them to channel-first (C, H, W).
    """

    def __init__(self, game: str, n_envs: int):
        self.n_envs = n_envs
        self.envs = [Environment(game) for _ in range(n_envs)]

        # infer observation shape
        sample_obs = self.envs[0].state()  # (H, W, C)
        h, w, c = sample_obs.shape
        self.obs_shape = (c, h, w)

    @staticmethod
    def _transpose(obs: np.ndarray) -> np.ndarray:
        # (H, W, C) -> (C, H, W)
        return np.transpose(obs, (2, 0, 1)).astype(np.float32)

    def reset(self) -> np.ndarray:
        observations = []

        for env in self.envs:
            env.reset()
            observations.append(self._transpose(env.state()))

        return np.stack(observations, axis=0)

    def step(self, actions):
        next_obs = []
        rewards = []
        dones = []
        infos = []

        for env, action in zip(self.envs, actions):
            reward, done = env.act(int(action))

            if done:
                env.reset()

            obs = self._transpose(env.state())

            next_obs.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append({})

        return (
            np.stack(next_obs, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            infos,
        )


def make_minatar_envs(game: str, n_envs: int):
    r"""
    Create vectorized MinAtar environments. The MinAtar observation
    shapes are (H, W, C) — note channel-last, needs to be transposed.
    """

    return MinAtarVecEnv(game, n_envs)


def make_atari_envs(game: str, n_envs: int):
    r"""
    Create vectorized Atari envs. Uses EnvPool for faster simulation.
    """

    envs = envpool.make(
        game,
        env_type = "gymnasium",
        num_envs = n_envs,
        episodic_life = True,
        reward_clip = True,
    )

    return envs


def main():
    parser = argparse.ArgumentParser(description="PPO + RV Training")
    parser.add_argument("--env", type=str, default="minatar:asterix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-frames", type=int, default=1e7)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--no-offset", action="store_true") # disable trajectory ranking (ablation: zero offset)
    parser.add_argument("--results-file", type=str, default=None)
    parser.add_argument("--checkpoint-file", type=str, default=None)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    # set rand seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running PPO + RV (env={args.env}, seed={args.seed}) on {device}...")

    cfg = PPORVConfig(n_envs=args.n_envs, n_step=args.n_step)
    # cfg = PPORVConfig(
    #     rollout_length = 32,
    #     n_envs         = 8,
    #     n_epochs       = 4,
    #     minibatch_size = 64,
    #     lr             = 3e-4,
    #     gamma          = 0.99,
    #     lam            = 0.95,
    #     clip_eps       = 0.2,   # standard PPO uses 0.2, not 0.1
    #     entropy_coef   = 0.01,
    #     critic_coef    = 1.25,
    #     n_step         = 1,
    # )
    # cfg = PPORVConfig(
    #     rollout_length = 128,
    #     n_envs         = 8,
    #     n_epochs       = 5,
    #     minibatch_size = 128,
    #     lr             = 2.5e-4,
    #     clip_eps       = 0.1,
    #     entropy_coef   = 0.01,
    #     critic_coef    = 1.25,
    #     n_step         = 1,
    # )

    # create envs
    if args.env.startswith("minatar:"):
        game = args.env.split(":")[1]
        envs = make_minatar_envs(game, args.n_envs)
        n_actions = envs.envs[0].num_actions()
    elif args.env.startswith("atari:"):
        envs = make_atari_envs(args.env, args.n_envs)
        n_actions = envs.action_space.n
    else:
        print(f"\tUnknown env: {args.env}")
        return

    if args.results_file is None:
        results_path = f"results/results_{slugify(args.env)}"
        if args.baseline:
            results_path += "_baseline"
        results_path += ".json"
        args.results_file = results_path

    print(f"Arguments: {vars(args)}")


    # train
    model = train(
        envs = envs,
        n_actions = n_actions,
        total_frames = args.total_frames,
        cfg = cfg,
        device = device,
        args=args
    )

    # checkpoint
    if args.checkpoint_file is None:
        ckpt_path = f"checkpoints/checkpoint_{slugify(args.env)}"
        if args.baseline:
            ckpt_path += "_baseline"
        ckpt_path += ".pt"
    else:
        ckpt_path = args.checkpoint_file
    torch.save(model.state_dict(), ckpt_path)
    print(f"\tSaved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()