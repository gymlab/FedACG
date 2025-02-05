@echo off

cd ..\

set CUDA_VISIBLE_DEVICES=0

set DATASET=cifar10
set BATCH_SIZE=50
if "%DATASET%" == "tinyimagenet" (
    set BATCH_SIZE=100
)

set PARTICIPATION_RATE=0.05
set QUANTIZERS=PAQ HQ


for %%Q in (PAQ HQ) do(
    for %%W in (1 2 4) do (
        echo Running experiment with ALPHA=%%A and WT_BIT=%%W
        python federated_train.py client=base server=base quantizer=%%Q ^
        exp_name=Fed%%Q_IID_B%%W dataset=%DATASET% trainer.num_clients=100 ^
        split.mode=iid trainer.participation_rate=%%Q ^
        batch_size=%BATCH_SIZE% wandb=True model=resnet18 quantizer.wt_bit=%%W ^
        project="FedAVG_Quant"
    )
)

