@echo off
set DATASET=cifar100
cd ..
set BATCH_SIZE=50

set ALPHA=0.1
set NBITS=8

python federated_train.py client=base server=base exp_name=ConvNet_%ALPHA%_B%NBITS% ^
dataset=%DATASET% trainer.num_clients=100 split.alpha=%ALPHA% trainer.participation_rate=0.05 ^
batch_size=%BATCH_SIZE% quantizer=LPT ^
wandb=True model=ConvNet project="LPT_WS_MERGE"