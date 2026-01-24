DEVICE=1
DATASET=cifar10
BATCH_SIZE=10
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=20
fi 
ALPHA=0.3
CM_PROB=0.1
MU_PROB=0.1

python3 federated_train.py visible_devices=\'$DEVICE\' client=base server=base exp_name=FedAvg_ncmu"$CM_PROB"_"$MU_PROB"_"$ALPHA"_num1_500 \
 dataset=${DATASET} trainer.num_clients=500 split.alpha=${ALPHA} trainer.participation_rate=0.02 \
 dataset.new_cutmixup.use=true dataset.new_cutmixup.use_reg=true \
 dataset.new_cutmixup.cutmix_prob=${CM_PROB} dataset.new_cutmixup.mixup_prob=${MU_PROB} \
 batch_size=${BATCH_SIZE} wandb=True model=resnet18 project="ICLR" \
#  split.mode=iid