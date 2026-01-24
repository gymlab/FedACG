#!/bin/bash

data_sets=(cifar100)
alpha_values=(0.6)
CM_PROB=0.3
DEVICE=3

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
            EXP_NAME=FedWS_cm"$CM_PROB"_iid_no_reg
            python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                dataset.cutmix.use=true dataset.cutmix.cutmix_reg=false dataset.cutmix.prob="$CM_PROB" \
                batch_size="$BATCH_SIZE" model=resnet18_WS wandb=True project="ICLR"
        else
            # For non-iid mode, iterate over alpha values
            for ALPHA in "${alpha_values[@]}"; do
                EXP_NAME=FedWS_cm"$CM_PROB"_"$ALPHA"_no_reg
                python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                    exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                    split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                    dataset.cutmix.use=true dataset.cutmix.cutmix_reg=false dataset.cutmix.prob="$CM_PROB" \
                    batch_size="$BATCH_SIZE" model=resnet18_WS wandb=True project="ICLR"
            done
        fi
    done
done
