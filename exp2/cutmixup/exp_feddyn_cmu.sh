DEVICE=3
DATASET=cifar10
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.1
CM_PROB=0.2
MU_PROB=0.2

python3 federated_train.py visible_devices=\'$DEVICE\' client=Dyn server=FedDyn exp_name=FedDyn_cmu"$CM_PROB""$MU_PROB"_"$ALPHA" \
 dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
 dataset.cutmixup.use=true dataset.cutmixup.use_reg=true \
 dataset.cutmixup.cutmix_prob=${CM_PROB} dataset.cutmixup.mixup_prob=${MU_PROB} \
 batch_size=${BATCH_SIZE} wandb=True model=resnet18 project="ICLR" \
#  split.mode=iid