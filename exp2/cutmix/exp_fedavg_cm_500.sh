DEVICE=0
DATASET=tinyimagenet
BATCH_SIZE=10
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=20
fi 
ALPHA=0.3
CM_PROB=0.2

python3 federated_train.py visible_devices=\'$DEVICE\' client=base server=base exp_name=FedAvg_cm"$CM_PROB"_"$ALPHA"_num1_500 \
 dataset=${DATASET} trainer.num_clients=500 split.alpha=${ALPHA} trainer.participation_rate=0.02 \
 dataset.cutmix.use=true dataset.cutmix.cutmix_reg=true dataset.cutmix.prob=${CM_PROB} \
 batch_size=${BATCH_SIZE} wandb=True model=resnet18 project="ICLR" \
# split.mode=iid