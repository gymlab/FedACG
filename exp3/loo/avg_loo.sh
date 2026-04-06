#!/bin/bash

data_sets=(cifar100)
alpha_values=(0.1 0.3)
loo_values=(0.2)
seeds=(0)
CM_PROB=0.15
MU_PROB=0.15
DEVICE=1
MODE=safe_loo


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
                for LOO in "${loo_values[@]}"; do       
                    # For iid mode, no need to iterate over alpha
                    ALPHA=0.6
                    EXP_NAME=FedAvg_iid_seed"$seed"_loo"$LOO"_"$MODE"
                    python federated_train.py client=base server=FedLOO visible_devices=\'$DEVICE\' \
                        exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                        split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                        dataset.new_cutmixup.use=false dataset.new_cutmixup.use_reg=true \
                        dataset.new_cutmixup.cutmix_prob=$CM_PROB dataset.new_cutmixup.mixup_prob=$MU_PROB \
                        server.update_mode="$MODE" server.loo_temp=1.0 server.loo_lambda=$LOO \
                        batch_size=$BATCH_SIZE wandb=True project="loo" seed=$seed
                done
            else
                # For non-iid mode, iterate over alpha values
                for ALPHA in "${alpha_values[@]}"; do
                    for LOO in "${loo_values[@]}"; do                        
                        EXP_NAME=FedAvg_"$ALPHA"_loo"$LOO"_"$MODE"
                        python federated_train.py client=base server=FedLOO visible_devices=\'$DEVICE\' \
                            exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                            split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                            dataset.new_cutmixup.use=false dataset.new_cutmixup.use_reg=true \
                            dataset.new_cutmixup.cutmix_prob=$CM_PROB dataset.new_cutmixup.mixup_prob=$MU_PROB \
                            server.update_mode="$MODE" server.loo_temp=1.0 server.loo_lambda=$LOO \
                            batch_size=$BATCH_SIZE wandb=True project="loo" seed=$seed
                    done
                done
            fi
        done
    done
done