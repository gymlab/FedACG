CUDA_VISIBLE_DEVICES=1, \
python federated_train.py client=base server=FedAdam exp_name=FedAdam dataset=cifar100 trainer.num_clients=100 split.alpha=0.3 trainer.participation_rate=0.05 batch_size=50 wandb=True trainer.global_lr=0.01
