@echo off
setlocal enabledelayedexpansion

set MODEL=resnet18
set BATCH_SIZE=50
set CUDA_VISIBLE_DEVICES=0
set DATASET=cifar10
set ALPHA=0.3
cd ../..

for %%P in (0.1 0.3 0.4) do (
    call set CM_PROB=%%P
    call set MU_PROB=%%P
    set EXP_NAME=FedDyn_cmu%%P_%%P_%ALPHA%

    echo Running experiment for %DATASET%

    python federated_train.py client=Dyn server=FedDyn ^
        visible_devices='%CUDA_VISIBLE_DEVICES%' ^
        exp_name=!EXP_NAME! ^
        dataset=%DATASET% ^
        trainer.num_clients=100 ^
        split.alpha=%ALPHA% ^
        trainer.participation_rate=0.05 ^
        batch_size=%BATCH_SIZE% ^
        wandb=True ^
        model=%MODEL% ^
        project="ICLR" ^
        dataset.cutmixup.use=True ^
        dataset.cutmixup.use_reg=True ^
        dataset.cutmixup.cutmix_prob=%%P ^
        dataset.cutmixup.mixup_prob=%%P 
)

endlocal
