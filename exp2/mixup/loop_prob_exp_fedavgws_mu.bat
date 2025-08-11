@echo off
setlocal enabledelayedexpansion

set MODEL=resnet18_WS
set BATCH_SIZE=100
set CUDA_VISIBLE_DEVICES=0
set DATASET=cifar10
set ALPHA=0.3
cd ../..

for %%P in (0.1 0.15 0.2) do (
    call set MU_PROB=%%P
    set EXP_NAME=FedAvgWS_mu%%P_%ALPHA%_num1

    if "%DATASET%"=="tinyimagenet" (
        set BATCH_SIZE=100
    ) else (
        set BATCH_SIZE=50
    )
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
        dataset.mixup.use=True ^
        dataset.mixup.mixup_reg=True ^
        dataset.mixup.prob=%%P
)

endlocal
