#!/bin/bash

data_sets=(tinyimagenet)
alpha_values=(0.3)
PROB=0.3
DEVICE=3

for DATASET in "${data_sets[@]}"; do
    if [ "$DATASET" = "tinyimagenet" ]; then
        BATCH_SIZE=100
    else
        BATCH_SIZE=50
    fi

    # 실험 종류별 반복 (cutmix, mixup, cutout)
    for AUG_TYPE in "cm" "mu" "co"; do
        # augmentation 설정
        if [ "$AUG_TYPE" = "cm" ]; then
            AUG_ARGS="dataset.cutmix.use=true dataset.cutmix.cutmix_reg=true dataset.cutmix.prob=$PROB"
        elif [ "$AUG_TYPE" = "mu" ]; then
            AUG_ARGS="dataset.mixup.use=true dataset.mixup.mixup_reg=true dataset.mixup.prob=$PROB"
        elif [ "$AUG_TYPE" = "co" ]; then
            AUG_ARGS="dataset.cutout.use=true dataset.cutout.use_reg=true dataset.cutout.prob=$PROB model=resnet18"
        fi

        # split mode 반복
        for SPLIT_MODE in "dirichlet"; do
            if [ "$SPLIT_MODE" = "iid" ]; then
                ALPHA=0.6
                EXP_NAME="FedDyn_${AUG_TYPE}${PROB}_iid_num1"

                python3 federated_train.py client=Dyn server=FedDyn visible_devices=\'$DEVICE\' \
                    exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                    split.mode="$SPLIT_MODE" trainer.participation_rate=0.05 \
                    batch_size="$BATCH_SIZE" wandb=True project="ICLR" \
                    $AUG_ARGS
            else
                for ALPHA in "${alpha_values[@]}"; do
                    EXP_NAME="FedDyn_${AUG_TYPE}${PROB}_${ALPHA}_num1"

                    python3 federated_train.py client=Dyn server=FedDyn visible_devices=\'$DEVICE\' \
                        exp_name="$EXP_NAME" dataset="$DATASET" trainer.num_clients=100 \
                        split.mode="$SPLIT_MODE" split.alpha="$ALPHA" trainer.participation_rate=0.05 \
                        batch_size="$BATCH_SIZE" wandb=True project="ICLR" \
                        $AUG_ARGS
                done
            fi
        done
    done
done
