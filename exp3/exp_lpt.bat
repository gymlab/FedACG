@echo off
setlocal enabledelayedexpansion
set DATASET=cifar10
set MODEL=resnet18_WS_LPT2
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
    exp_name="%MODEL%_%ALPHA%_B%%B_1_E3M2_occBW_23N_BFPBW" ^
    dataset=%DATASET% ^
    trainer.num_clients=100 ^
    split.alpha=%ALPHA% ^
    trainer.participation_rate=0.05 ^
    model.moving_average=False ^
    batch_size=!BATCH_SIZE! ^
    quantizer=LPT quantizer.quantization_bits=%%B ^
    quantizer.small_block="None" ^
    wandb=True ^
    model=%MODEL% ^
    project="BMVC_2025_5" ^
    model.init_mode="kaiming_normal" quantizer.uniform_mode="DANUQ" ^
    quantizer.weight_quantization=True quantizer.gradient_quantization=True quantizer.activation_quantization=True ^
    model.blk1_uni_quant=True model.blk_start_uni_quant=True model.blk_end_uni_quant=True ^
    model.conv1_uni_quant=True model.conv2_uni_quant=True model.downsample_uni_quant=True ^
    quantizer.lut_mode="E3M2"
)
pause