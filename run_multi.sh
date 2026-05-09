#!/bin/bash

export PYTHONPATH=.



GAMES="freeway"
# GAMES="asterix breakout freeway seaquest space_invaders"
SEED=123
TOTAL_FRAMES=10000000


# pids=()

# for game in $GAMES; do
#     # RV run
#     python src/main.py \
#         --env minatar:${game} \
#         --seed ${SEED} \
#         --total-frames ${TOTAL_FRAMES} \
#         --no-offset \
#         --n-step 1 \
#         --n-envs 8 \
#         --results-file results/rv_${game}.json \
#         --checkpoint-file checkpoints/rv_${game}.pt \
#         > logs/rv_${game}.log 2>&1 &
#     pids+=($!)

#     python src/main.py \
#         --env minatar:${game} \
#         --seed ${SEED} \
#         --total-frames ${TOTAL_FRAMES} \
#         --n-step 1 \
#         --n-envs 8 \
#         --results-file results/rvo_${game}.json \
#         --checkpoint-file checkpoints/rvo_${game}.pt \
#         > logs/rvo_${game}.log 2>&1 &
#     pids+=($!)

#     # Baseline run
#     python src/main.py \
#         --env minatar:${game} \
#         --seed ${SEED} \
#         --total-frames ${TOTAL_FRAMES} \
#         --baseline \
#         --n-envs 8 \
#         --results-file results/baseline_${game}.json \
#         --checkpoint-file checkpoints/baseline_${game}.pt \
#         > logs/baseline_${game}_seed${seed}.log 2>&1 &
#     pids+=($!)

# done

# echo "Launched ${#pids[@]} jobs: ${pids[@]}"

# # Wait for all and report which failed
# for pid in "${pids[@]}"; do
#     wait $pid || echo "Job $pid failed"
# done

# echo "All done"