CUDA_VISIBLE_DEVICES=0, \
python federated_train.py server=FedACG client=ACG exp_name=FedACG dataset=cifar100 trainer.num_clients=100 split.alpha=0.3 trainer.participation_rate=0.05 batch_size=50 wandb=True trainer.local_lr_decay=0.995
