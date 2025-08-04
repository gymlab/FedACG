@echo off
setlocal enabledelayedexpansion

set CUDA_VISIBLE_DEVICES=0
set DATASET=cifar10
set MODEL=resnet18
set ALPHA=0.3
set MU_PROB=0.2
cd ../..


if "%DATASET%"=="tinyimagenet" (
    set BATCH_SIZE=100
) else (
    set BATCH_SIZE=50
)

python federated_train.py client=Dyn server=FedDyn ^
    visible_devices='%CUDA_VISIBLE_DEVICES%' ^
    exp_name="FedDyn_mu%MU_PROB%_iid_num1" ^
    dataset=%DATASET% ^
    trainer.num_clients=100 ^
    split.mode=iid ^
    trainer.participation_rate=0.05 ^
    batch_size=!BATCH_SIZE! ^
    wandb=True ^
    model=%MODEL% ^
    project="ICLR" ^
    dataset.mixup.use=True ^
    dataset.mixup.mixup_reg=True ^
    dataset.mixup.prob=%MU_PROB%

pause

