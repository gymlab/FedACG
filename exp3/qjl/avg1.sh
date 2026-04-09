#!/bin/bash

data_sets=(cifar100)
alpha_values=(0.1)
DEVICE=0
qjl_ratios=(0.5)
block_sizes=(2048)
ORTH=true
weight_mode=raw
tau=0.05


if [ "$ORTH" = true ]; then
    ORTH_TAG="_orth"
else
    ORTH_TAG=""
fi


for BLOCK in "${block_sizes[@]}"; do

    for QJL in "${qjl_ratios[@]}"; do

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
                    EXP_NAME=FedAvg_iid_QJL"$QJL"_B"$BLOCK"${ORTH_TAG}_tau"$tau"_"$weight_mode"_way
                    python federated_train.py client=base server=FedQJL visible_devices=\'$DEVICE\' \
                        exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                        split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                        server.tau="$tau" server.weight_mode="$weight_mode" \
                        server.qjl_ratio="$QJL" server.use_orthogonal=$ORTH server.skip_small_tensors=true \
                        server.small_tensor_threshold=256 server.block_size="$BLOCK" server.min_m=32 server.max_m=256 \
                        batch_size="$BATCH_SIZE" wandb=True model=resnet18 project="QJL"
                else
                    # For non-iid mode, iterate over alpha values
                    for ALPHA in "${alpha_values[@]}"; do
                        EXP_NAME=FedAvg_"$ALPHA"_QJL"$QJL"_B"$BLOCK"${ORTH_TAG}_tau"$tau"_"$weight_mode"_way
                        python federated_train.py client=base server=FedQJL visible_devices=\'$DEVICE\' \
                            exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                            split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                            server.tau="$tau" server.weight_mode="$weight_mode" \
                            server.qjl_ratio="$QJL" server.use_orthogonal=$ORTH server.skip_small_tensors=true \
                            server.small_tensor_threshold=256 server.block_size="$BLOCK" server.min_m=32 server.max_m=256 \
                            batch_size="$BATCH_SIZE" wandb=True model=resnet18 project="QJL"
                    done
                fi
            done
        done
    done
done