DEVICE=1
DATASET=cifar10
BATCH_SIZE=10
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=20
fi 
ALPHA=0.3
MU_PROB=0.2

python federated_train.py visible_devices=\'$DEVICE\' client=base server=base exp_name=FedAvg_mu"$MU_PROB"_"$ALPHA"_num1_500 \
 dataset=${DATASET} trainer.num_clients=500 split.alpha=${ALPHA} trainer.participation_rate=0.02 \
 dataset.mixup.use=true dataset.mixup.mixup_reg=true dataset.mixup.prob=${MU_PROB} \
 batch_size=${BATCH_SIZE} wandb=True model=resnet18 project="ICLR" \
# split.mode=iid