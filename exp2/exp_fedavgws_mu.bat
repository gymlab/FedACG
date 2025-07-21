@echo off
setlocal enabledelayedexpansion

set CUDA_VISIBLE_DEVICES=0
set DATASET=cifar10
set MODEL=resnet18_WS
set ALPHA=0.6
set cm_prob=0.2
cd ..


if "%DATASET%"=="tinyimagenet" (
    set BATCH_SIZE=100
) else (
    set BATCH_SIZE=50
)

python federated_train.py client=base server=base ^
    visible_devices='%CUDA_VISIBLE_DEVICES%' ^
    exp_name="FedAvgWS_mu%cm_prob%_%ALPHA%" ^
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
    dataset.mixup.prob=%cm_prob%

pause

