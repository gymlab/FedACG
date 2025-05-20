DATASET=cifar10
ALPHA=0.6

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate fl

cd ~/projects/LPT_FL/FedACG

for MODEL in resnet18_WS_LPT resnet18_LPT
do
    for NBITS in 6
    do
        
        if [ "${DATASET}" = "tinyimagenet" ]; then
            BATCH_SIZE=100

        else
            BATCH_SIZE=50

        fi

        EXP_NAME="${MODEL}_${ALPHA}_B${NBITS}_Avg"

        CUDA_VISIBLE_DEVICES=0 python3 federated_train.py \
        client=base server=base \
        exp_name="${EXP_NAME}" \
        dataset=${DATASET} \
        trainer.num_clients=100 \
        split.alpha=${ALPHA} \
        trainer.participation_rate=0.02 \
        batch_size=${BATCH_SIZE} \
        wandb=True \
        model=${MODEL} \
        project="BMVC_2025_3" \
        quantizer=LPT\
        quantizer.quantization_bits=${NBITS} \
        &

        # split.mode=iid
    done
done
