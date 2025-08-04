@echo off
setlocal enabledelayedexpansion

set PROB=0.3
set MODEL=resnet18
set ALPHA=0.3
set CUDA_VISIBLE_DEVICES=0
cd ..

for %%D in (cifar10) do (
    set DATASET=%%D
    
            
    if "!DATASET!"=="tinyimagenet" (
        set BATCH_SIZE=100
    ) else (
        set BATCH_SIZE=50
    )

    python federated_train.py client=Prox server=base ^
        visible_devices='%CUDA_VISIBLE_DEVICES%' ^
        exp_name="FedProx_cm%PROB%_%ALPHA%_num1" ^
        dataset=!DATASET! ^
        trainer.num_clients=100 ^
        split.alpha=%ALPHA% ^
        trainer.participation_rate=0.05 ^
        batch_size=!BATCH_SIZE! ^
        wandb=True ^
        model=%MODEL% ^
        project="ICLR" ^
        dataset.cutmix.use=True ^
        dataset.cutmix.cutmix_reg=True ^
        dataset.cutmix.prob=%PROB%
    
)


for %%D in (cifar10) do (
    set DATASET=%%D
    
            
    if "!DATASET!"=="tinyimagenet" (
        set BATCH_SIZE=100
    ) else (
        set BATCH_SIZE=50
    )

    python federated_train.py client=Prox server=base ^
        visible_devices='%CUDA_VISIBLE_DEVICES%' ^
        exp_name="FedProx_mu%PROB%_%ALPHA%_num1" ^
        dataset=!DATASET! ^
        trainer.num_clients=100 ^
        split.alpha=%ALPHA% ^
        trainer.participation_rate=0.05 ^
        batch_size=!BATCH_SIZE! ^
        wandb=True ^
        model=%MODEL% ^
        project="ICLR" ^
        dataset.mixup.use=True ^
        dataset.mixup.mixup_reg=True ^
        dataset.mixup.prob=%PROB%

)

for %%D in (cifar10) do (
    set DATASET=%%D
    
            
    if "!DATASET!"=="tinyimagenet" (
        set BATCH_SIZE=100
    ) else (
        set BATCH_SIZE=50
    )

    python federated_train.py client=Prox server=base ^
        visible_devices='%CUDA_VISIBLE_DEVICES%' ^
        exp_name="FedProx_co%PROB%_%ALPHA%_num1" ^
        dataset=!DATASET! ^
        trainer.num_clients=100 ^
        split.alpha=%ALPHA% ^
        trainer.participation_rate=0.05 ^
        batch_size=!BATCH_SIZE! ^
        wandb=True ^
        model=%MODEL% ^
        project="ICLR" ^
        dataset.cutout.use=True ^
        dataset.cutout.use_reg=True ^
        dataset.cutout.prob=%PROB%


)