DEVICE=3
DATASET=tinyimagenet
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.6
CM_PROB=0.15
MU_PROB=0.15

python3 federated_train.py visible_devices=\'$DEVICE\' client=base server=base exp_name=FedAvg_rmsda"$CM_PROB"_"$MU_PROB"_iid_s1 \
 dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
 dataset.new_cutmixup.use=true dataset.new_cutmixup.use_reg=true \
 dataset.new_cutmixup.cutmix_prob=${CM_PROB} dataset.new_cutmixup.mixup_prob=${MU_PROB} \
 batch_size=${BATCH_SIZE} wandb=True model=resnet18 project="CVPR_REBUTTAL" seed=3 \
 split.mode=iid