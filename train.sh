#!/usr/bin/env bash

# declare -a dataset=(mvtec)
declare -a dataset=(mvtec visa)
# declare -a dataset=(visa)
source ./config.sh

for i in "${dataset[@]}"; do
    for seed in 1 2 4; do
        echo "dataset=$i, seed=$seed"
        python train.py --dataset "$i" --dino_arch "$DINO_ARCH" --dino_weights "$DINO_WEIGHTS" --clip_name "$CLIP_NAME" --seed "$seed" --branch "$BRANCH"
    done
done
