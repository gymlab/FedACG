#!/bin/bash

data_sets=(tinyimagenet cifar100 cifar10)
alpha_values=(0.1)
seeds=(1 2 3 4)
MU_PROB=0.3
DEVICE=0

for seed in "${seeds[@]}"; do
    if [ $seed = 1 ]; then
        seed=1
    elif [ $seed = 2 ]; then
        seed=2
    elif [ $seed = 3 ]; then
        seed=3
    elif [ $seed = 4 ]; then
        seed=4
    elif [ $seed = 5 ]; then
        seed=5
    elif [ $seed = 6 ]; then
        seed=6
    else
        echo "Unknown seed: $seed"
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
                EXP_NAME=FedDyn_msda"$MU_PROB"_iid_seed"$seed"_noreg
                python3 federated_train.py client=Dyn server=FedDyn visible_devices=\'$DEVICE\' \
                    exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                    split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                    dataset.mixup.use=true dataset.mixup.mixup_reg=false dataset.mixup.prob="$MU_PROB" \
                    batch_size="$BATCH_SIZE" wandb=True project="CVPR_REBUTTAL" seed=$seed
            else
                # For non-iid mode, iterate over alpha values
                for ALPHA in "${alpha_values[@]}"; do
                    EXP_NAME=FedDyn_msda"$MU_PROB"_"$ALPHA"_seed"$seed"_noreg
                    python3 federated_train.py client=Dyn server=FedDyn visible_devices=\'$DEVICE\' \
                        exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                        split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                        dataset.mixup.use=true dataset.mixup.mixup_reg=false dataset.mixup.prob="$MU_PROB" \
                        batch_size="$BATCH_SIZE" wandb=True project="CVPR_REBUTTAL" seed=$seed
                done
            fi
        done
    done
done