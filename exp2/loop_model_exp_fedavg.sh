#!/bin/bash

data_sets=(cifar100)
alpha_values=(0.1)
models=(SqueezeNet_base)
DEVICE=6

for MODEL in "${models[@]}"; do
    if [ "$MODEL" = "MobileViT" ]; then
        MODEL_NAME="MobileViT"
    elif [ "$MODEL" = "VGG9_base" ]; then
        MODEL_NAME="VGG9_base"
    elif [ "$MODEL" = "ShuffleNet_base" ]; then
        MODEL_NAME="ShuffleNet_base"
    elif [ "$MODEL" = "SqueezeNet_base" ]; then
        MODEL_NAME="SqueezeNet_base"
    else
        echo "Unknown model: $MODEL"
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
                EXP_NAME=FedAvg_"$MODEL_NAME"_iid_num1
                python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                    exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                    split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                    batch_size="$BATCH_SIZE" wandb=True project="ICLR" model="$MODEL_NAME"
            else
                # For non-iid mode, iterate over alpha values
                for ALPHA in "${alpha_values[@]}"; do
                    EXP_NAME=FedAvg_"$MODEL_NAME"_"$ALPHA"_num1_A6000
                    python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' \
                        exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                        split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                        batch_size="$BATCH_SIZE" wandb=True project="ICLR" model="$MODEL_NAME"
                done
            fi
        done
    done
done