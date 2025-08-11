@echo off
setlocal enabledelayedexpansion

set MODEL=resnet18_WS
set CUDA_VISIBLE_DEVICES=0
set DATASET=cifar10
set ALPHA=0.3
cd ../..

for %%P in (0.2 0.3 0.4) do (
    call set CM_PROB=%%P

    if "%DATASET%"=="tinyimagenet" (
    set BATCH_SIZE=100
    ) else (
        set BATCH_SIZE=50
    )

    set EXP_NAME=FedAvgWS_cm%%P_%ALPHA%_num1

    echo Running experiment for %DATASET%

    python federated_train.py client=base server=base ^
        visible_devices='%CUDA_VISIBLE_DEVICES%' ^
        exp_name=!EXP_NAME! ^
        dataset=%DATASET% ^
        trainer.num_clients=100 ^
        split.alpha=%ALPHA% ^
        trainer.participation_rate=0.05 ^
        batch_size=!BATCH_SIZE! ^
        wandb=True ^
        model=%MODEL% ^
        project="ICLR" ^
        dataset.cutmix.use=True ^
        dataset.cutmix.cutmix_reg=True ^
        dataset.cutmix.prob=%%P
)
endlocal
