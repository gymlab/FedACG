#!/bin/bash

export WANDB_API_KEY=$(echo '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' | tr -d '\n')
export WANDB_LOGIN_METHOD=apikey
export WANDB_MODE=online

data_sets=(cifar100)
alpha_values=(0.1 0.3 0.6)
DEVICE=0

# Iterate over datasets
for DATASET in "${data_sets[@]}"; do
    # Set default BATCH_SIZE based on dataset
    if [ "$DATASET" = "tinyimagenet" ]; then
        BATCH_SIZE=100
    else
        BATCH_SIZE=50
    fi

    # Iterate over split modes
    for SPLIT_MODE in "dirichlet" "iid"; do

        if [ "$SPLIT_MODE" = "iid" ]; then
            # For iid mode, no need to iterate over alpha
            ALPHA=0.6
            EXP_NAME=FedAvg_iid
            python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                batch_size="$BATCH_SIZE" wandb=True model=resnet18 project="loo"
        else
            # For non-iid mode, iterate over alpha values
            for ALPHA in "${alpha_values[@]}"; do
                EXP_NAME=FedAvg_"$ALPHA"
                python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                    exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                    split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                    batch_size="$BATCH_SIZE" wandb=True model=resnet18 project="loo"
            done
        fi
    done
done
