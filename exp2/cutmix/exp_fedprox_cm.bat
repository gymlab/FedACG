@echo off
setlocal enabledelayedexpansion

set CUDA_VISIBLE_DEVICES=0
set DATASET=cifar100
set MODEL=resnet18
set ALPHA=0.3
set CM_PROB=0.2
cd ../..


if "%DATASET%"=="tinyimagenet" (
    set BATCH_SIZE=100
) else (
    set BATCH_SIZE=50
)

python federated_train.py client=Prox server=base ^
    visible_devices='%CUDA_VISIBLE_DEVICES%' ^
    exp_name="FedProx_cm%CM_PROB%_iid_num1" ^
    dataset=%DATASET% ^
    trainer.num_clients=100 ^
    split.mode=iid ^
    trainer.participation_rate=0.05 ^
    batch_size=!BATCH_SIZE! ^
    wandb=True ^
    model=%MODEL% ^
    project="ICLR" ^
    dataset.cutmix.use=True ^
    dataset.cutmix.cutmix_reg=True ^
    dataset.cutmix.prob=%CM_PROB%

pause

