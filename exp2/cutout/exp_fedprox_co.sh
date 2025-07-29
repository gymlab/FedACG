DEVICE=0
DATASET=tinyimagenet
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.1
CO_PROB=0.2

python3 federated_train.py visible_devices=\'$DEVICE\' client=Prox server=base exp_name=FedProx_co"$CO_PROB"_"$ALPHA" \
 dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
 dataset.cutout.use=true dataset.cutout.use_reg=true dataset.cutout.prob=${CO_PROB} \
 batch_size=${BATCH_SIZE} wandb=True model=resnet18 project="ICLR" \
# split.mode=iid