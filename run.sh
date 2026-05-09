#!/bin/bash

export PYTHONPATH=.

GAMES="seaquest"
# GAMES="asterix breakout freeway seaquest space_invaders"
SEED=123
TOTAL_FRAMES=5000000

game=${GAMES}
python src/main.py \
    --env minatar:${game} \
    --seed ${SEED} \
    --total-frames ${TOTAL_FRAMES} \
    --baseline \
    --n-step 5 \
    --n-envs 8 \
    --results-file results/baseline_${game}.json \
    --checkpoint-file checkpoints/baseline_${game}.pt


python src/main.py \
    --env minatar:${game} \
    --seed ${SEED} \
    --total-frames ${TOTAL_FRAMES} \
    --n-step 5 \
    --n-envs 8 \
    --results-file results/rvo_${game}.json \
    --checkpoint-file checkpoints/rvo_${game}.pt

python src/main.py \
    --env minatar:${game} \
    --seed ${SEED} \
    --total-frames ${TOTAL_FRAMES} \
    --no-offset \
    --n-step 5 \
    --n-envs 8 \
    --results-file results/rv_${game}.json \
    --checkpoint-file checkpoints/rv_${game}.pt