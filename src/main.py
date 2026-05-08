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
import envpool
import re

from src.train import train, PPORVConfig


def make_minatar_envs(game: str, n_envs: int):
    r"""
    Create vectorized MinAtar environments. The MinAtar observation
    shapes are (H, W, C) — note channel-last, needs to be transposed.
    """

    # TODO: wrap in a vectorized env class compatible with train.py's envs API
    # MinAtar is not natively vectorized — simple sequential wrapper needed.
    raise NotImplementedError(
        "Implement a vectorized wrapper for MinAtar. "
        "Each env.act(action) returns (reward, done); obs = env.state()."
    )


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
    args = parser.parse_args()

    # set rand seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running PPO+RV (env={args.env}, seed={args.seed}) on {device}...")

    cfg = PPORVConfig(n_envs=args.n_envs)

    # create envs
    if args.env.startswith("minatar:"):
        game = args.env.split(":")[1]
        envs = make_minatar_envs(game, args.n_envs)
        # MinAtar: 10x10xC obs, small action space (~5-6 actions)
        n_actions = 6 # TODO: query from env
    elif args.env.startswith("atari:"):
        envs = make_atari_envs(args.env, args.n_envs)
        n_actions = 18 # TODO: query from env
    else:
        print(f"\tUnknown env: {args.env}")
        return

    # train
    model = train(
        envs         = envs,
        n_actions    = n_actions,
        total_frames = args.total_frames,
        cfg          = cfg,
        device       = device,
        use_offset   = not args.no_offset,
    )

    # checkpoint

    ckpt_path = f"checkpoints/{re.sub(r'[^A-Za-z0-9_]', '_', args.env)}_seed{args.seed}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\tSaved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()