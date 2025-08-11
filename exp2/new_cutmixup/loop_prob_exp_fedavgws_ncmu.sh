#!/bin/bash

data_sets=(cifar10 cifar100)
prob_values=(0.1 0.15 0.2)
ALPHA=0.3
CM_PROB=0.1
MU_PROB=0.1
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
            EXP_NAME=FedAvgWS_ncmu"$CM_PROB"_iid_num1
            python federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                dataset.new_cutmixup.use=true dataset.new_cutmixup.use_reg=true \
                dataset.new_cutmixup.cutmix_prob=${CM_PROB} dataset.new_cutmixup.mixup_prob=${MU_PROB} \
                batch_size="$BATCH_SIZE" wandb=True model=resnet18_WS project="ICLR"
        else
            # For non-iid mode, iterate over alpha values
            for PROB in "${prob_values[@]}"; do
                EXP_NAME=FedAvgWS_ncmu"$PROB"_"$ALPHA"_num1
                python federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                    exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                    split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                    dataset.new_cutmixup.use=true dataset.new_cutmixup.use_reg=true \
                    dataset.new_cutmixup.cutmix_prob=${PROB} dataset.new_cutmixup.mixup_prob=${PROB} \
                    batch_size="$BATCH_SIZE" wandb=True model=resnet18_WS project="ICLR"
            done
        fi
    done
done
