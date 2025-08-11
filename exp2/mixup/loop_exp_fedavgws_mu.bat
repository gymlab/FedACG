@echo off
setlocal enabledelayedexpansion

set DATASET=cifar10
set MU_PROB=0.2
set MODEL=resnet18_WS
cd ../..
for %%A in (0.1 0.3 0.6) do (
    set ALPHA=%%A
            
    if "%DATASET%"=="tinyimagenet" (
        set BATCH_SIZE=100
    ) else (
        set BATCH_SIZE=50
    )

    set EXP_NAME=FedAvgWS_mu%MU_PROB%_%%A_num1
    
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
        dataset.mixup.use=True ^
        dataset.mixup.mixup_reg=True ^
        dataset.mixup.prob=%MU_PROB%

)

            
set EXP_NAME=FedAvgWS_mu%MU_PROB%_iid_num1

if "%DATASET%"=="tinyimagenet" (
    set BATCH_SIZE=100
) else (
    set BATCH_SIZE=50
)

echo Running experiment 
set CUDA_VISIBLE_DEVICES=0

python federated_train.py client=base server=base ^
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
    dataset.mixup.use=True ^
    dataset.mixup.mixup_reg=True ^
    dataset.mixup.prob=%MU_PROB%
