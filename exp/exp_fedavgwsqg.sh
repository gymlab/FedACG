DATASET=cifar10
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.3
NBITS=4
DEVICE=3

<<<<<<< HEAD
python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' exp_name=FedAvgWSQGV1_0.5_"$ALPHA"_"B$NBITS" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
quantizer=WSQG quantizer.wt_bit=${NBITS} quantizer.momentum=0.5 quantizer.wt_clip_prob=-1 \
=======
python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' exp_name=FedAvgWSQGV3_0.2_"$ALPHA"_"B$NBITS" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
quantizer=WSQG quantizer.wt_bit=${NBITS} quantizer.momentum=0.2 quantizer.wt_clip_prob=-1 \
>>>>>>> 51bea890ae78495d38b0ff2c1ea355e8c9f12572
batch_size=${BATCH_SIZE} wandb=True model=resnet18_WS project="dev_quant3" \
# split.mode=iid
# quantizer=WSQ quantizer.wt_bit=${NBITS} \d
# python3 federated_train.py client=base server=base visible_devices=\'$DEVICE\' exp_name=FedAvgWSQ_"$ALPHA"_"B$NBITS" \
