CUDA_VISIBLE_DEVICES=0 \
python federated_train.py client=base server=base exp_name=FedAvg dataset=cifar10 trainer.num_clients=100 split.alpha=0.3 trainer.participation_rate=0.05 batch_size=50 wandb=True
