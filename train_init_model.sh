#!/bin/bash

DATASET_NAME=$1

python beyond-part-models/script/experiment/train_pcb.py -d '(0,1,2)' --only_test false --dataset $DATASET_NAME --trainset_part trainval --local_conv_out_channels 256 --exp_dir './models/'$DATASET_NAME --steps_per_log 20 --epochs_per_val 1
