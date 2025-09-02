@echo off
setlocal enabledelayedexpansion

set DATASET=cifar10
set MODEL=resnet18_FA
cd ..

for %%A in (0.3 0.1 0.6) do (
    set "ALPHA=%%A"

    if "%DATASET%"=="tinyimagenet" (
        set "BATCH_SIZE=100"
    ) else (
        set "BATCH_SIZE=50"
    )

    set "EXP_NAME=FedFA_!ALPHA!"

    echo Running experiment !EXP_NAME!
    set CUDA_VISIBLE_DEVICES=0

    python federated_train.py client=base server=FedFA ^
        visible_devices='!CUDA_VISIBLE_DEVICES!' ^
        exp_name=!EXP_NAME! ^
        dataset=!DATASET! ^
        trainer.num_clients=100 ^
        split.alpha=!ALPHA! ^
        trainer.participation_rate=0.05 ^
        batch_size=!BATCH_SIZE! ^
        wandb=True ^
        model=!MODEL! ^
        project="ICLR" 
)

set "EXP_NAME=FedFA_iid"

if "%DATASET%"=="tinyimagenet" (
    set "BATCH_SIZE=100"
) else (
    set "BATCH_SIZE=50"
)

echo Running experiment 
set CUDA_VISIBLE_DEVICES=0

python federated_train.py client=base server=FedFA ^
    visible_devices='!CUDA_VISIBLE_DEVICES!' ^
    exp_name=!EXP_NAME! ^
    dataset=!DATASET! ^
    trainer.num_clients=100 ^
    split.mode=iid ^
    trainer.participation_rate=0.05 ^
    batch_size=!BATCH_SIZE! ^
    wandb=True ^
    model=!MODEL! ^
    project="ICLR" ^