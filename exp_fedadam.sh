CUDA_VISIBLE_DEVICES=0
DATASET=cifar10
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.05

python3 federated_train.py client=base server=FedAdam exp_name=FedAdam_IID \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 trainer.global_lr=0.01 \
batch_size=${BATCH_SIZE} wandb=True \
split.mode=iid