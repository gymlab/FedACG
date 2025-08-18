DEVICE=4
DATASET=cifar10
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.6
CM_PROB=0.15
MU_PROB=0.15

python3 federated_train.py visible_devices=\'$DEVICE\' client=Dyn server=FedDyn exp_name=FedDyn_ncmu"$CM_PROB"_"$MU_PROB"_"$ALPHA"_num1 \
 dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
 dataset.new_cutmixup.use=true dataset.new_cutmixup.use_reg=true \
 dataset.new_cutmixup.cutmix_prob=${CM_PROB} dataset.new_cutmixup.mixup_prob=${MU_PROB} \
 batch_size=${BATCH_SIZE} wandb=True model=resnet18 project="ICLR" \
#  split.mode=iid