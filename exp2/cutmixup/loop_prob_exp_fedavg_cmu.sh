#!/bin/bash

data_sets=(cifar10 cifar100)
prob_values=(0.1 0.3 0.4)
ALPHA=0.3
CM_PROB=0.2
MU_PROB=0.2
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
            EXP_NAME=FedAvg_cmu"$CM_PROB"_"$MU_PROB"_iid
            python federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                dataset.cutmixup.use=true dataset.cutmixup.use_reg=true \
                dataset.cutmixup.cutmix_prob=${CM_PROB} dataset.cutmixup.mixup_prob=${MU_PROB} \
                batch_size="$BATCH_SIZE" wandb=True project="ICLR"
        else
            # For non-iid mode, iterate over alpha values
            for PROB in "${prob_values[@]}"; do
                EXP_NAME=FedAvg_cmu"$PROB"_"$PROB"_"$ALPHA"
                python federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                    exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                    split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                    dataset.cutmixup.use=true dataset.cutmixup.use_reg=true \
                    dataset.cutmixup.cutmix_prob=${PROB} dataset.cutmixup.mixup_prob=${PROB} \
                    batch_size="$BATCH_SIZE" wandb=True project="ICLR"
            done
        fi
    done
done
