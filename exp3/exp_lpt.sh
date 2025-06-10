DATASET=cifar10
MODEL=resnet18_WS_LPT2
ALPHA=0.3
cd ..

for NBITS in 6
do
    if [ "${DATASET}" = "tinyimagenet" ]; then
        BATCH_SIZE=100
    else
        BATCH_SIZE=50
    fi

    python3 federated_train.py visible_devices="3" client=base server=base exp_name="${MODEL}_${ALPHA}_B${NBITS}_23H_t30%" \
    dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 model.moving_average=False \
    batch_size=${BATCH_SIZE} quantizer=LPT quantizer.quantization_bits=${NBITS} quantizer.small_block="None"  \
    wandb=True model=${MODEL} project="BMVC_2025_5" model.init_mode='kaiming_normal' quantizer.uniform_mode='DANUQ' \
    quantizer.weight_quantization=True quantizer.gradient_quantization=True quantizer.activation_quantization=False \
    model.blk1_uni_quant=False model.blk_start_uni_quant=False model.blk_end_uni_quant=False \
    model.conv1_uni_quant=False model.conv2_uni_quant=False model.downsample_uni_quant=False
done

