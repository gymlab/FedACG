DATASET=cifar10
MODEL=resnet18_WS_LPT2
ALPHA=0.1
cd ..

for NBITS in 6
do
    if [ "${DATASET}" = "tinyimagenet" ]; then
        BATCH_SIZE=100
    else
        BATCH_SIZE=50
    fi

    python3 federated_train.py visible_devices="3" client=base server=base exp_name="${MODEL}_iid_B${NBITS}_WGB_zero_mask" \
    dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 model.moving_average=False \
    batch_size=${BATCH_SIZE} quantizer=LPT quantizer.quantization_bits=${NBITS} quantizer.block_dim="B" quantizer.uniform_mode="DANUQ"  \
    wandb=True model=${MODEL} project="BMVC_2025_4" \
    split.mode=iid
done
