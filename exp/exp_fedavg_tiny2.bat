@echo off

cd ..\

set CUDA_VISIBLE_DEVICES=0

set DATASET=tinyimagenet
set BATCH_SIZE=50
if "%DATASET%" == "tinyimagenet" (
    set BATCH_SIZE=100
)

set PARTICIPATION_RATE=0.02
set QUANTIZERS=HQ

for %%A in (0.3) do (
    for %%W in (2) do (
        echo Running experiment with ALPHA=%%A and WT_BIT=%%W
        python federated_train.py client=base server=base quantizer=%QUANTIZERS% ^
        exp_name=Fed%QUANTIZERS%_%%A_B%%W_0.02 dataset=%DATASET% trainer.num_clients=100 ^
        split.alpha=%%A trainer.participation_rate=%PARTICIPATION_RATE% ^
        batch_size=%BATCH_SIZE% wandb=True model=resnet18 quantizer.wt_bit=%%W ^
        project="FedAVG_Quant"
    )
)
