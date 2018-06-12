#!/bin/bash


set -x
set -e

export PYTHONUNBUFFERED="True"

DEV=$1
DEV_ID=$2
NET=$3
DATASET=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}


LOG="./logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_net.py --device CPU \
                                 --device_id 0 \
                                 --weights ./model/pretrain/VGG_imagenet.npy \
                                 --imdb voc_2007_train \
                                 --iters 100 \
                                 --cfg ./cfgs/faster_rcnn_end2end_vgg.yml \
                                 --network VGGnet_train \
                                 ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x
