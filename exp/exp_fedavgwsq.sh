CUDA_VISIBLE_DEVICES=0
DATASET=tinyimagenet
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.3
NBITS=8

python3 federated_train.py client=base server=base visible_devices=\'1\' exp_name=FedAvgWSQ_"$ALPHA"_"B$NBITS" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.02 \
quantizer=WSQ quantizer.wt_bit=${NBITS} \
batch_size=${BATCH_SIZE} wandb=True model=resnet18_WS project="dev_quant2" \
# split.mode=iid
