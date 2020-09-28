#!/bin/bash
# CPC unsupervised extration of language features
source /home/zhangwq01/anaconda3/bin/activate torch_py37_yhb
echo $CONDA_DEFAULT_ENV

CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

PATH_AUDIO_FILES='/home/zhangwq01/yuhaibin/data/olr18_train_cpc/data/cpc_train'
PATH_CHECKPOINT_DIR='/home/zhangwq01/yuhaibin/CPC_audio/LID_checkpoint/trial1'
TRAINING_SET='/home/zhangwq01/yuhaibin/data/olr18_train_cpc/data/train3.txt'
VAL_SET='/home/zhangwq01/yuhaibin/data/olr18_train_cpc/data/val3.txt'
CHECKPOINT_TO_LOAD='/home/zhangwq01/yuhaibin/CPC_audio/LID_checkpoint/trial1/extract/checkpoint_85.pt'
PATH_CHECKPOINT='/home/zhangwq01/yuhaibin/CPC_audio/LID_checkpoint/trial1/classify'
EXTENSION='wav'
N_GPU=1
BATCHSIZE=32
N_EPOCH=50

python cpc/eval/linear_separability.py  $PATH_AUDIO_FILES \
                                        $TRAINING_SET\
                                        $VAL_SET\
                                        $CHECKPOINT_TO_LOAD\
                                        --pathCheckpoint $PATH_CHECKPOINT\
                                        --file_extension $EXTENSION\
                                        --nGPU $N_GPU\
                                        --batchSizeGPU $BATCHSIZE\
                                        --n_epoch $N_EPOCH\
                                        --beta1 0.85\
                                        --beta2 0.99
echo 'done'
