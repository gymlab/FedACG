DEVICE=0
DATASET=tinyimagenet
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.1
cm_prob=0.2

python3 federated_train.py visible_devices=\'$DEVICE\' client=base server=base exp_name=FedAvg_cm"$cm_prob"_"$ALPHA" \
 dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
 dataset.cutmix.use=true dataset.cutmix.cutmix_reg=true dataset.cutmix.prob=${cm_prob} \
 batch_size=${BATCH_SIZE} wandb=True model=resnet18 project="ICLR" \
# split.mode=iid