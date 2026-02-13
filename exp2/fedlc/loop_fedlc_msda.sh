#!/bin/bash

data_sets=(cifar100)
alpha_values=(0.3)
seeds=(4)
MU_PROB=0.3
DEVICE=3

for seed in "${seeds[@]}"; do

    if ! [[ "$seed" =~ ^-?[0-9]+$ ]]; then
        echo "Invalid seed (not an integer): $seed"
        exit 1
    fi

    # Iterate over datasets
    for DATASET in "${data_sets[@]}"; do
        # Set default BATCH_SIZE based on dataset
        if [ "$DATASET" = "tinyimagenet" ]; then
            BATCH_SIZE=100
        else
            BATCH_SIZE=50
        fi

        # Iterate over split modes
        for SPLIT_MODE in "dirichlet"; do
    
            if [ "$SPLIT_MODE" = "iid" ]; then
                # For iid mode, no need to iterate over alpha
                ALPHA=0.6
                EXP_NAME=FedLC_msda"$MU_PROB"_iid_seed"$seed"_noreg
                python3 federated_train.py client=LC server=base visible_devices=\'$DEVICE\' \
                    exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                    split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                    dataset.mixup.use=true dataset.mixup.mixup_reg=false dataset.mixup.prob="$MU_PROB" \
                    batch_size="$BATCH_SIZE" wandb=True project="CVPR_REBUTTAL" seed=$seed
            else
                # For non-iid mode, iterate over alpha values
                for ALPHA in "${alpha_values[@]}"; do
                    EXP_NAME=FedLC_msda"$MU_PROB"_"$ALPHA"_seed"$seed"_noreg
                    python3 federated_train.py client=LC server=base visible_devices=\'$DEVICE\' \
                        exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                        split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                        dataset.mixup.use=true dataset.mixup.mixup_reg=false dataset.mixup.prob="$MU_PROB" \
                        batch_size="$BATCH_SIZE" wandb=True project="CVPR_REBUTTAL" seed=$seed
                done
            fi
        done
    done
done