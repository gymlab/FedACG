#!/bin/bash

data_sets=(cifar10)
alpha_values=(0.1 0.3 0.6)
CM_PROB=0.2
DEVICE=2

# Iterate over datasets
for DATASET in "${data_sets[@]}"; do
    # Set default BATCH_SIZE based on dataset
    if [ "$DATASET" = "tinyimagenet" ]; then
        BATCH_SIZE=100
    else
        BATCH_SIZE=50
    fi

    # Iterate over split modes
    for SPLIT_MODE in "iid" "dirichlet"; do
 
        if [ "$SPLIT_MODE" = "iid" ]; then
            # For iid mode, no need to iterate over alpha
            ALPHA=0.6
            EXP_NAME=FedAvgWS_cm"$CM_PROB"_iid_num1
            python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                dataset.cutmix.use=true dataset.cutmix.cutmix_reg=true dataset.cutmix.prob="$CM_PROB" \
                batch_size="$BATCH_SIZE" wandb=True model=resnet18_WS project="ICLR"
        else
            # For non-iid mode, iterate over alpha values
            for ALPHA in "${alpha_values[@]}"; do
                EXP_NAME=FedAvgWS_cm"$CM_PROB"_"$ALPHA"_num1
                python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                    exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                    split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                    dataset.cutmix.use=true dataset.cutmix.cutmix_reg=true dataset.cutmix.prob="$CM_PROB" \
                    batch_size="$BATCH_SIZE" wandb=True model=resnet18_WS project="ICLR"
            done
        fi
    done
done
