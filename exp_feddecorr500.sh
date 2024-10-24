CUDA_VISIBLE_DEVICES=0
DATASET=tinyimagenet
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.3

python3 federated_train.py client=Decorr server=base exp_name=FedDecorr_"$ALPHA" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.02 \
batch_size=${BATCH_SIZE} wandb=True project="FedWS_2_100" \
# split.mode=iid