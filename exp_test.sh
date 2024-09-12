CUDA_VISIBLE_DEVICES=0
DATASET=cifar10
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
LR_DECAY=0.995

python3 federated_train.py server=FedACG client=ACG exp_name=FedACGGN_rho1e-3 dataset=${DATASET} project=test \
trainer.num_clients=100 split.alpha=0.3 trainer.participation_rate=0.05 \
batch_size=${BATCH_SIZE} wandb=True trainer.local_lr_decay=${LR_DECAY} model=resnet18_WS \
trainer.local_epochs=5 trainer.global_rounds=1000