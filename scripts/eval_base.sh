#!/bin/bash
# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


EXP=$1
RESULT_DIR=$2
SPLIT=$3

usage (){
    echo
    echo "Run with: ./this_script EXP RESULT_DIR SPLIT"
    echo "Args:"
    echo "    EXP -- Experiment directory (checkpoint directory)"
    echo "    RESULT_DIR  -- Directory to store results"
    echo "    SPLIT       -- val or test"
    echo
    echo "e.g.) ./scripts/eval_base.sh \ "
    echo "          path/to/checkpoint \ "
    echo "          decode_result_contrib \ "
    echo "          test"
    echo
    exit 0
}

[[ -z $EXP ]] && echo "Checkpoint path empty" && usage && return 1;
[[ -z $RESULT_DIR ]] && echo "Result directory must be specified" && usage && return 1;

shift 3

python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 -m src.main \
    --batch_size 10 \
    --gradient_accumulation_steps 8 \
    --max_input_length 1024 \
    --eval_every 6000 \
    --datadir /export/scisumm/filtered_s2orc_20190928 \
    --eval_batch_size 32  \
    --max_eval_batches 5000 \
    --learning_rate 3e-5 \
    --logdir $RESULT_DIR \
    --tokenizer_name facebook/bart-large \
    --model_name_or_path $EXP \
    --input_type paper \
    --target_type full \
    --use_apex \
    --no_greedy_decode \
    --no_nucleus_decode \
    --test_only \
    --test_split $SPLIT \
    $@
