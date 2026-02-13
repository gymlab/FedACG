#!/bin/bash

data_sets=(cifar10)
alpha_values=(0.1 0.3 0.6)
DEVICE=6
MU_PROB=0.3

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
            EXP_NAME=FedRDN_msda"$MU_PROB"_iid__noreg
            python3 federated_train.py client=RDN server=base visible_devices=\'$DEVICE\' \
                exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                batch_size="$BATCH_SIZE" wandb=True model=resnet18 project="CVPR_REBUTTAL" \
                dataset.mixup.use=true dataset.mixup.mixup_reg=false dataset.mixup.prob="$MU_PROB" 
        else
            # For non-iid mode, iterate over alpha values
            for ALPHA in "${alpha_values[@]}"; do
                EXP_NAME=FedRDN_msda"$MU_PROB"_"$ALPHA"_noreg
                python3 federated_train.py client=RDN server=base visible_devices=\'$DEVICE\' \
                    exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                    split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                    batch_size="$BATCH_SIZE" wandb=True model=resnet18 project="CVPR_REBUTTAL" \
                    dataset.mixup.use=true dataset.mixup.mixup_reg=false dataset.mixup.prob="$MU_PROB" 
            done
        fi
    done
done
