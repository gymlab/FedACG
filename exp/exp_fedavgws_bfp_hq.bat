@echo off

cd ..\

set CUDA_VISIBLE_DEVICES=0

set DATASET=cifar100
set BATCH_SIZE=50
if "%DATASET%" == "tinyimagenet" (
    set BATCH_SIZE=100
)

set ALPHA=0.3
set PARTICIPATION_RATE=0.05
set QUANTIZERS=HQ

for %%B in (1, 2, 3, 4, 8) do (
    echo Running experiment with quantizer: %QUANTIZERS% and WT_BIT : %%B
    python federated_train.py client=base server=base quantizer=%QUANTIZERS% ^
    exp_name=FedAvgWS_%QUANTIZERS%_%ALPHA%_%%B_bit__%PARTICIPATION_RATE%_BFP dataset=%DATASET% trainer.num_clients=100 ^
    split.alpha=%ALPHA% trainer.participation_rate=%PARTICIPATION_RATE% ^
    batch_size=%BATCH_SIZE% wandb=True model=resnet18_WS quantizer.wt_bit=%%B ^
    project="FedAVG_WS_Quant"
)
