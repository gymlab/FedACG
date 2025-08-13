@echo off
setlocal enabledelayedexpansion

set CM_PROB=0.1
set MU_PROB=0.1
set BATCH_SIZE=50
set CUDA_VISIBLE_DEVICES=0
set DATASET=cifar10
set ALPHA=0.3
cd ../..

for %%M in (MobileViT VGG9_base ShuffleNet_base SqueezeNet_base) do (
    set EXP_NAME=FedAvg_%%M_ncmu%CM_PROB%_%MU_PROB%_%ALPHA%_num1

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
        model=%%M ^
        project="ICLR" ^
        dataset.new_cutmixup.use=True ^
        dataset.new_cutmixup.use_reg=True ^
        dataset.new_cutmixup.cutmix_prob=%CM_PROB% ^
        dataset.new_cutmixup.mixup_prob=%MU_PROB%
)

endlocal
