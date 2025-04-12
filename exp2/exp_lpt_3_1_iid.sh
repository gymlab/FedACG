DATASET=cifar100
MODEL=resnet18_LPT
ALPHA=0.6

for NBITS in 5 6 8 4 2
do
    if [ "${DATASET}" = "tinyimagenet" ]; then
        BATCH_SIZE=100
    else
        BATCH_SIZE=50
    fi

    python3 federated_train.py visible_devices="1" client=base server=base exp_name="${MODEL}_IID_B${NBITS}" \
    dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 model.moving_average=False \
    batch_size=${BATCH_SIZE} quantizer=LPT quantizer.quantization_bits=${NBITS} \
    wandb=True model=${MODEL} project="BMVC_2025_2" \
    split.mode=iid
done
