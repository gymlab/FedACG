CUDA_VISIBLE_DEVICES=0,1 \
python federated_train.py multiprocessing=True main_gpu=0 client=base server=base exp_name=FedAvgBN dataset=cifar100 trainer.num_clients=100 split.alpha=0.3 trainer.participation_rate=0.05 wandb=True trainer.local_lr_decay=0.995 batch_size=50 model=resnet18_BNWS