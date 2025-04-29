setlocal enabledelayedexpansion

@echo off
set DATASET=cifar10
set MODEL=resnet18_WS_LPT2
set ALPHA=0.6
cd ..

for %%N in (6) do (
    if "%DATASET%"=="tinyimagenet" (
        set BATCH_SIZE=100
    ) else (
        set BATCH_SIZE=50
    )

    python federated_train.py visible_devices="0" client=base server=base ^
        exp_name="%MODEL%_IID_B%%N_DANUQ" ^
        dataset=%DATASET% ^
        trainer.num_clients=100 ^
        split.alpha=%ALPHA% ^
        trainer.participation_rate=0.05 ^
        model.moving_average=False ^
        batch_size=!BATCH_SIZE! ^
        quantizer=LPT ^
        quantizer.quantization_bits=%%N ^
        wandb=True ^
        model=%MODEL% ^
        project="BMVC_2025_3" ^
        split.mode=iid
)

pause
