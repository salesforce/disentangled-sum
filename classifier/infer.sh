#!/usr/bin/env bash
# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

ORG_DATA=$1
EXP=$2
ID=$3

# "startswith". Splits like this to allow paralleled inference on multiple gpus independently.
if [[ $ID == "0" ]]; then
    SPLITS=( train_0 )
elif [[ $ID == "1" ]]; then
    SPLITS=( train_1 )
elif [[ $ID == "2" ]]; then
    SPLITS=( train_2 )
elif [[ $ID == "3" ]]; then
    SPLITS=( train_3 )
elif [[ $ID == "4" ]]; then
    SPLITS=( train_4 )
elif [[ $ID == "5" ]]; then
    SPLITS=( train_5 )
elif [[ $ID == "6" ]]; then
    SPLITS=( train_6 )
elif [[ $ID == "7" ]]; then
    SPLITS=( train_7 )
elif [[ $ID == "8" ]]; then
    SPLITS=( train_8 )
elif [[ $ID == "9" ]]; then
    SPLITS=( valid_0 )
elif [[ $ID == "10" ]]; then
    SPLITS=( test_0 )
else
    SPLITS=( None )
fi

for split in ${SPLITS[@]}; do
    CUDA_VISIBLE_DEVICES=$4 python run_classifier.py \
        --data_dir $ORG_DATA \
        --output_dir $EXP \
        --do_eval \
        --fp16 \
        --overwrite_output_dir \
        --logdir bert_exp \
        --infer-jsonl $split
done

