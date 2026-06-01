# !/usr/bin/env bash
# train_datasets=(mvtec)
train_datasets=(mvtec visa)
# train_datasets=(visa)

source ./config.sh

for train_dataset in "${train_datasets[@]}"; do
    if [[ "$train_dataset" == "mvtec" ]]; then
        test_datasets=(visa)
    else
        test_datasets=(mpdd mvtec btad)
    fi

    for test_dataset in "${test_datasets[@]}"; do
        for seed in 1 2 4; do
            echo "train_dataset=$train_dataset, test_dataset=$test_dataset, seed=$seed"
            python test.py --dataset "$test_dataset" --dino_arch "$DINO_ARCH" --dino_weights "$DINO_WEIGHTS" --clip_name "$CLIP_NAME" --seed "$seed" --branch "$BRANCH" --train_dataset "$train_dataset"
        done
    done
done
