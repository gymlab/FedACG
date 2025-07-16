@echo off
setlocal enabledelayedexpansion

set CUDA_VISIBLE_DEVICES=0
set DATASET=cifar100
set MODEL=resnet18
set ALPHA=0.3
set cm_prob=0.2
cd ..


if "%DATASET%"=="tinyimagenet" (
    set BATCH_SIZE=100
) else (
    set BATCH_SIZE=50
)

python federated_train.py client=Dyn server=FedDyn ^
    visible_devices='%CUDA_VISIBLE_DEVICES%' ^
    exp_name="FedDyn_cm%cm_prob%_%ALPHA%" ^
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
    dataset.cutmix.prob=%cm_prob%

pause

