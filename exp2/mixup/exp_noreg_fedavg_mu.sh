DEVICE=0
DATASET=cifar10
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.1
MU_PROB=0.3

python3 federated_train.py visible_devices=\'$DEVICE\' client=base server=base exp_name=FedAvg_mu"$MU_PROB"_"$ALPHA"_num1_noreg \
 dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
 dataset.mixup.use=true dataset.mixup.mixup_reg=false dataset.mixup.prob=${MU_PROB} \
 batch_size=${BATCH_SIZE} wandb=True model=resnet18 project="ICLR" \
# split.mode=iid