@echo off
setlocal enabledelayedexpansion
set DATASET=cifar10
set MODEL=resnet18_WS_LPT
set ALPHA=0.1
cd ..

for %%B in (6) do (
    if "%DATASET%"=="tinyimagenet" (
        set BATCH_SIZE=100
    ) else (
        set BATCH_SIZE=50
    )

    echo Using BATCH_SIZE=!BATCH_SIZE! with BITS=%%B
    python federated_train.py visible_devices="0" client=base server=base ^
    exp_name="%MODEL%_IID_B%%B_RAU_BC_only_weight" ^
    dataset=%DATASET% ^
    trainer.num_clients=100 ^
    split.alpha=%ALPHA% ^
    trainer.participation_rate=0.05 ^
    model.moving_average=False ^
    batch_size=!BATCH_SIZE! ^
    quantizer=LPT quantizer.quantization_bits=%%B quantizer.block_dim="BC" ^
    wandb=True ^
    model=%MODEL% ^
    project="BMVC_2025_3" ^
    split.mode=iid ^
    model.init_mode="kaiming_normal" quantizer.uniform_mode="DANUQ"
)
pause