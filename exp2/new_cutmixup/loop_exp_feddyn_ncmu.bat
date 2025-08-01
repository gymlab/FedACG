@echo off
setlocal enabledelayedexpansion

set DATASET=cifar100
set CM_PROB=0.1
set MU_PROB=0.1
set MODEL=resnet18
cd ../..
for %%A in (0.1 0.3 0.6) do (
    set ALPHA=%%A
            
    if "%DATASET%"=="tinyimagenet" (
        set BATCH_SIZE=100
    ) else (
        set BATCH_SIZE=50
    )

    set EXP_NAME=FedDyn_ncmu%CM_PROB%_%MU_PROB%_%%A
    
    echo Running experiment 
    set CUDA_VISIBLE_DEVICES=0

    python federated_train.py client=Dyn server=FedDyn ^
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
        dataset.new_cutmixup.use=True ^
        dataset.new_cutmixup.use_reg=True ^
        dataset.new_cutmixup.cutmix_prob=%CM_PROB% ^
        dataset.new_cutmixup.mixup_prob=%MU_PROB%
)


set EXP_NAME=FedDyn_ncmu%CM_PROB%_%MU_PROB%_iid

if "%DATASET%"=="tinyimagenet" (
    set BATCH_SIZE=100
) else (
    set BATCH_SIZE=50
)

echo Running experiment 
set CUDA_VISIBLE_DEVICES=0

python federated_train.py client=Dyn server=FedDyn ^
    visible_devices='%CUDA_VISIBLE_DEVICES%' ^
    exp_name=%EXP_NAME% ^
    dataset=%DATASET% ^
    trainer.num_clients=100 ^
    split.mode=iid ^
    trainer.participation_rate=0.05 ^
    batch_size=!BATCH_SIZE! ^
    wandb=True ^
    model=%MODEL% ^
    project="ICLR" ^
    dataset.new_cutmixup.use=True ^
    dataset.new_cutmixup.use_reg=True ^
    dataset.new_cutmixup.cutmix_prob=%CM_PROB% ^
    dataset.new_cutmixup.mixup_prob=%MU_PROB%