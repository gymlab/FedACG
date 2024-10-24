CUDA_VISIBLE_DEVICES=0
DATASET=cifar10
BATCH_SIZE=10
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=20
fi 
ALPHA=0.3

python3 federated_train.py client=base server=FedAdam exp_name=FedAdam_"$ALPHA" \
dataset=${DATASET} trainer.num_clients=500 split.alpha=${ALPHA} trainer.participation_rate=0.02 trainer.global_lr=0.01 \
batch_size=${BATCH_SIZE} wandb=True project="FedWS_2_500" \
# split.mode=iid