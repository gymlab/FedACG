#!/bin/bash

data_sets=(cifar10)
alpha_values=(0.1)
seeds=(1 2 3 4 5)
CM_PROB=0.3
DEVICE=1

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
                EXP_NAME=FedAvg_cm"$CM_PROB"_iid_num1_"$seed"
                python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                    exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                    split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                    dataset.cutmix.use=true dataset.cutmix.cutmix_reg=true dataset.cutmix.prob="$CM_PROB" \
                    batch_size="$BATCH_SIZE" wandb=True project="ICLR" seed=$seed
            else
                # For non-iid mode, iterate over alpha values
                for ALPHA in "${alpha_values[@]}"; do
                    EXP_NAME=FedAvg_cm"$CM_PROB"_"$ALPHA"_num1_"$seed"
                    python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                        exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                        split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                        dataset.cutmix.use=true dataset.cutmix.cutmix_reg=true dataset.cutmix.prob="$CM_PROB" \
                        batch_size="$BATCH_SIZE" wandb=True project="ICLR" seed=$seed
                done
            fi
        done
    done
done