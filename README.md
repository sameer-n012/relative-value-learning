# Relative Value Learning
Sameer Narendran, CS 443

Implementation of **Höftmann, Robine & Harmeling (ICLR 2026)**.

---

## Project Structure
```
relative-value-learning/
|-- src/
|  |-- bellman.py       # Pairwise Bellman operator + value targets
|  |-- rgae.py          # R-GAE + trajectory ranking
|  |-- actor_critic.py  # CNN encoder + policy head + siamese relative critic
|  |-- rollout.py       # Rollout storage + pair sampling
|  |-- ppo.py           # Loss functions + gradient update
|  |-- train.py         # Main training loop
|  |-- main.py          # Entry point
|-- tests/
|  |-- test_theory.py   # Unit tests
|-- checkpoints/        # Model checkpoints   
|-- results/            # Results + figures
```

---

## Running

```bash
pip install -r requirements.txt

PYTHONPATH=. pytest -v

# train on MinAtar
python src/main.py --env minatar:asterix --seed 0

# ablation: zero-offset vs trajectory ranking
python src/main.py --env minatar:asterix --no-offset --seed 0
```

---

## TODOs

- [x] `src/bellman.py`: implement `compute_lambda_return` and the full `compute_nstep_target` (with terminal flag handling at step n)
- [x] `src/rgae.py`: vectorize `compute_trajectory_offsets` (currently O(N^2) loop)
- [x] `tests/test_theory.py`: run all tests: `PYTHONPATH=. pytest -v`
- [x] `src/rollout_buffer.py`: implement `compute_rgae()` method (calls src.rgae per env)
- [ ] `src/train.py`: hook up a real vectorized gym env; verify observation normalization
- [x] `src/run.py`: implement `make_minatar_envs` vectorized wrapper
- [ ] Run ablation: `--no-offset` vs default (Figure 4 in paper)
- [ ] Compare against PPO baseline (same architecture, absolute critic head)
- [ ] Switch to EnvPool for high-throughput simulation
- [x] n-step target (n=5 per Appendix D) instead of 1-step
- [ ] Evaluate on subset of 49 ALE games; compare Table 1