#!/bin/bash

export WANDB_API_KEY=$(echo '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' | tr -d '\n')
export WANDB_LOGIN_METHOD=apikey
export WANDB_MODE=online

data_sets=(cifar100)
alpha_values=(0.1 0.3 0.6)
loo_values=(0.3)
seeds=(0)
CM_PROB=0.15
MU_PROB=0.15
DEVICE=1

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
        for SPLIT_MODE in "dirichlet" "iid"; do
    
            if [ "$SPLIT_MODE" = "iid" ]; then
                for LOO in "${loo_values[@]}"; do       
                    # For iid mode, no need to iterate over alpha
                    ALPHA=0.6
                    EXP_NAME=FedAvg_rmsda"$CM_PROB"_"$MU_PROB"_iid
                    python federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                        exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                        split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                        dataset.new_cutmixup.use=true dataset.new_cutmixup.use_reg=true \
                        dataset.new_cutmixup.cutmix_prob=$CM_PROB dataset.new_cutmixup.mixup_prob=$MU_PROB \
                        batch_size=$BATCH_SIZE wandb=True project="loo" seed=$seed
                done
            else
                # For non-iid mode, iterate over alpha values
                for ALPHA in "${alpha_values[@]}"; do
                    for LOO in "${loo_values[@]}"; do                        
                        EXP_NAME=FedAvg_rmsda"$CM_PROB"_"$MU_PROB"_"$ALPHA"
                        python federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                            exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                            split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                            dataset.new_cutmixup.use=true dataset.new_cutmixup.use_reg=true \
                            dataset.new_cutmixup.cutmix_prob=$CM_PROB dataset.new_cutmixup.mixup_prob=$MU_PROB \
                            batch_size=$BATCH_SIZE wandb=True project="loo" seed=$seed
                    done
                done
            fi
        done
    done
done