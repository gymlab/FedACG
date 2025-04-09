CUDA_VISIBLE_DEVICES=0
DATASET=cifar10
MODEL=resnet18_WS_LPT
ALPHA=0.1

for NBITS in 1 2 4 6 8
do
    if [ "${DATASET}" = "tinyimagenet" ]; then
        BATCH_SIZE=100
    else
        BATCH_SIZE=50
    fi

    python3 federated_train.py client=base server=base exp_name="${MODEL}_${ALPHA}_B${NBITS}" \
    dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 model.moving_average=False \
    batch_size=${BATCH_SIZE} quantizer=LPT quantizer.quantization_bits=${NBITS} \
    wandb=True model=${MODEL} project="LPT_WS_MERGE"
    # split.mode=iid
done
