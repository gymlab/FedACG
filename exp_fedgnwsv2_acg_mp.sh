CUDA_VISIBLE_DEVICES=0,1 \
python federated_train.py multiprocessing=True main_gpu=0 client=base server=FedACG exp_name=FedGNWSV2_baseACG dataset=cifar100 trainer.num_clients=100 split.alpha=0.3 trainer.participation_rate=0.05 wandb=True model=resnet18_GNWS trainer.local_lr_decay=0.995
