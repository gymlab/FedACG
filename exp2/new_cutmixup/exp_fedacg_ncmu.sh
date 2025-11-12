DEVICE=0
DATASET=tinyimagenet
BATCH_SIZE=50
DECAY=0.995
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
    DECAY=0.998
fi 
ALPHA=0.3
CM_PROB=0.15
MU_PROB=0.15

python3 federated_train.py visible_devices=\'$DEVICE\' client=ACG server=FedACG exp_name=FedACG_ncmu"$CM_PROB"_"$MU_PROB"_"$ALPHA" \
 dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
 dataset.new_cutmixup.use=true dataset.new_cutmixup.use_reg=true trainer.local_lr_decay=${DECAY} \
 dataset.new_cutmixup.cutmix_prob=${CM_PROB} dataset.new_cutmixup.mixup_prob=${MU_PROB} \
 batch_size=${BATCH_SIZE} wandb=True model=resnet18 project="ICLR" \
#  split.mode=iid