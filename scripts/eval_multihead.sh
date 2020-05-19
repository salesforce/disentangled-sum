#!/bin/bash
# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

EXP=${1:-"both_multihead_old_release"}
TARGET_TYPE=$2
RESULT_DIR=$3
SPLIT=$4

usage (){
    echo
    echo "Run with: ./this_script EXP TARGET_TYPE RESULT_DIR"
    echo "Args:"
    echo "    EXP -- Experiment directory (checkpoint directory)"
    echo "    TARGET_TYPE -- Either of (contrib, context)."
    echo "    RESULT_DIR  -- Directory to store results."
    echo "    SPLIT       -- val or test."
    echo
    echo "e.g.) ./scripts/eval_both_multihead.sh \ "
    echo "          results/both_multihead/2020-08-04-16:48:09/checkpoint_78000 \ "
    echo "          contrib \ "
    echo "          decode_result_multihead_contrib \ "
    echo "          test"
    echo
    exit 0
}

[[ -z $EXP ]] && echo "Checkpoint path empty" && usage;
[[ -z $TARGET_TYPE ]] && echo "Target_type empty" && usage;
[[ -z $RESULT_DIR ]] && echo "Result directory must be specified" && usage;

shift 4

if [[ $TARGET_TYPE == "contrib" ]]; then
    HEAD=0
    DEV=0,1,2,3
    PORT=1234
else
    HEAD=1
    DEV=4,5,6,7
    PORT=5678
fi

# CUDA_VISIBLE_DEVICES=0 python -m pdb -m src.main \
CUDA_VISIBLE_DEVICES=$DEV python -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT -m src.main \
    --batch_size 5 \
    --gradient_accumulation_steps 16 \
    --max_input_length 1024 \
    --eval_every 6000 \
    --datadir /export/scisumm/filtered_s2orc_20190928/  \
    --eval_batch_size 32  \
    --max_eval_batches 5000 \
    --learning_rate 3e-5 \
    --logdir $RESULT_DIR \
    --tokenizer_name facebook/bart-large \
    --model_name_or_path $EXP \
    --input_type paper \
    --target_type both \
    --use_multi_head \
    --use_apex \
    --no_greedy_decode \
    --no_nucleus_decode \
    --no_modify_prefix \
    --force_head $HEAD \
    --test_only \
    --test_split $SPLIT \
    $@
