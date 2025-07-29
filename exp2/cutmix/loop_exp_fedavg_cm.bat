@echo off
setlocal enabledelayedexpansion

set DATASET=cifar10
set CM_PROB=0.2
set MODEL=resnet18
cd ../..
for %%A in (0.1 0.3 0.6) do (
    set ALPHA=%%A
            
    if "%DATASET%"=="tinyimagenet" (
        set BATCH_SIZE=100
    ) else (
        set BATCH_SIZE=50
    )

    set EXP_NAME=FedAvg_cm%CM_PROB%_%%A
    
    echo Running experiment 
    set CUDA_VISIBLE_DEVICES=0

    python federated_train.py client=base server=base ^
        visible_devices='%CUDA_VISIBLE_DEVICES%' ^
        exp_name=!EXP_NAME! ^
        dataset=%DATASET% ^
        trainer.num_clients=100 ^
        split.alpha=!ALPHA! ^
        trainer.participation_rate=0.05 ^
        batch_size=!BATCH_SIZE! ^
        wandb=True ^
        model=%MODEL% ^
        project="ICLR" ^
        dataset.cutmix.use=True ^
        dataset.cutmix.cutmix_reg=True ^
        dataset.cutmix.prob=%CM_PROB%

)