CUDA_VISIBLE_DEVICES=1
DATASET=cifar10
BATCH_SIZE=50
MODEL=resnet18_LPT
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.1
NBITS=8

python3 federated_train.py client=base server=base exp_name="$MODEL"_"$ALPHA"_"B$NBITS" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
batch_size=${BATCH_SIZE} quantizer=LPT quantizer.quantization_bits=${NBITS} \
wandb=True model=${MODEL} project="LPT_WS_MERGE" \
# split.mode=iid
