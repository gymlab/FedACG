CUDA_VISIBLE_DEVICES=0
DATASET=cifar100
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.3

python3 federated_train.py client=Won server=base exp_name=FedAvgWon_"$ALPHA" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
batch_size=${BATCH_SIZE} wandb=False model=resnet18_Won project="dev_quant" \
# split.mode=iid
