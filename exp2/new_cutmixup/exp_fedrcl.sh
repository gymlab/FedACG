DEVICE=0
DATASET=tinyimagenet
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.3

python federated_train.py visible_devices=\'$DEVICE\' client=fedrcl server=base exp_name=FedRCL_"$ALPHA"_s0 \
 dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
 batch_size=${BATCH_SIZE} wandb=True model=resnet18 project="ICLR" \
#  split.mode=iid