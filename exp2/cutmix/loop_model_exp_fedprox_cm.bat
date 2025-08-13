@echo off
setlocal enabledelayedexpansion

set BATCH_SIZE=50
set CUDA_VISIBLE_DEVICES=0
set DATASET=cifar10
set ALPHA=0.3
set CM_PROB=0.2
cd ../..

for %%M in (MobileViT VGG9_base ShuffleNet_base SqueezeNet_base) do (
    set EXP_NAME=FedProx_%%M_cm%CM_PROB%_%ALPHA%_num1

    if "%DATASET%"=="tinyimagenet" (
        set BATCH_SIZE=100
    ) else (
        set BATCH_SIZE=50
    )
    echo Running experiment for %DATASET%

    python federated_train.py client=Prox server=base ^
        visible_devices='%CUDA_VISIBLE_DEVICES%' ^
        exp_name=!EXP_NAME! ^
        dataset=%DATASET% ^
        trainer.num_clients=100 ^
        split.alpha=%ALPHA% ^
        trainer.participation_rate=0.05 ^
        batch_size=!BATCH_SIZE! ^
        wandb=True ^
        model=%%M ^
        project="ICLR" ^
        dataset.cutmix.use=True ^
        dataset.cutmix.cutmix_reg=True ^
        dataset.cutmix.prob=%CM_PROB%
)
endlocal
