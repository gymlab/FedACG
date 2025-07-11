DATASET=cifar100
MODEL=resnet18_WS_LPT2
ALPHA=0.3
LUT=E3M1
lr=0.1
cd ..

for NBITS in 5
do
    if [ "${DATASET}" = "tinyimagenet" ]; then
        BATCH_SIZE=100
    else
        BATCH_SIZE=50
    fi

    python3 federated_train.py visible_devices="2" client=base server=base exp_name=WS_"$ALPHA"_B"$NBITS"_1_"$LUT"_23U \
    dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 model.moving_average=False \
    batch_size=${BATCH_SIZE} quantizer=LPT quantizer.quantization_bits=${NBITS} quantizer.small_block="None"  \
    wandb=True model=${MODEL} project="BMVC_2025_5" model.init_mode='kaiming_uniform' \
    quantizer.weight_quantization=True quantizer.gradient_quantization=True quantizer.activation_quantization=True \
    quantizer.weight_uniform_mode="BFP" quantizer.grad_uniform_mode="BFP" quantizer.act_uniform_mode="occ" \
    model.blk1_uni_quant=True model.blk_start_uni_quant=True model.blk_end_uni_quant=True \
    model.conv1_uni_quant=True model.conv2_uni_quant=True model.downsample_uni_quant=True \
    quantizer.lut_mode=${LUT} quantizer.comp=True trainer.local_lr=${lr} trainer.lr_fix=False
done