DEVICE=0
DATASET=tinyimagenet
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.6
MU_PROB=0.2

python3 federated_train.py visible_devices=\'$DEVICE\' client=base server=base exp_name=FedAvgWS_mu"$MU_PROB"_iid_num1 \
 dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
 dataset.mixup.use=true dataset.mixup.mixup_reg=true dataset.mixup.prob=${MU_PROB} \
 batch_size=${BATCH_SIZE} wandb=True model=resnet18_WS project="ICLR" \
 split.mode=iid