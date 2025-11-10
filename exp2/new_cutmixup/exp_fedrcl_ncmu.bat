@echo off
setlocal enabledelayedexpansion

set CUDA_VISIBLE_DEVICES=0
set DATASET=cifar10
set MODEL=resnet18
set ALPHA=0.1
set CM_PROB=0.15
set MU_PROB=0.15
cd ../..


if "%DATASET%"=="tinyimagenet" (
    set BATCH_SIZE=100
) else (
    set BATCH_SIZE=50
)

python federated_train.py client=fedrcl server=base ^
    visible_devices='%CUDA_VISIBLE_DEVICES%' ^
    exp_name="FedRCL_ncmu%CM_PROB%_%MU_PROB%_%ALPHA%_num1" ^
    dataset=%DATASET% ^
    trainer.num_clients=100 ^
    split.alpha=%ALPHA% ^
    split.mode=iid ^
    seed=3 ^
    trainer.participation_rate=0.05 ^
    batch_size=!BATCH_SIZE! ^
    wandb=True ^
    model=%MODEL% ^
    project="ICLR" ^
    dataset.new_cutmixup.use=True ^
    dataset.new_cutmixup.use_reg=True ^
    dataset.new_cutmixup.cutmix_prob=%CM_PROB% ^
    dataset.new_cutmixup.mixup_prob=%MU_PROB% 
pause

