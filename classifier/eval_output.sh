#!/usr/bin/env bash
# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

MODE=$1
EXP=$2
shift 2

CUDA_VISIBLE_DEVICES= python run_classifier.py \
    --decode-results $@ \
    --mode $MODE \
    --data_dir dummy \
    --output_dir $EXP \
    --do_eval \
    --fp16 \
    --overwrite_output_dir \
    --logdir unused

