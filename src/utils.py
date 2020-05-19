# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
import pdb
import random

import numpy as np
import torch
from transformers import AdamW


def check_nan(tensor):
    if len(tensor.size()) == 0:
        if torch.isinf(tensor) or torch.isnan(tensor):
            pdb.set_trace()

    elif any(torch.isinf(tensor)) or any(torch.isnan(tensor)):
        pdb.set_trace()
    return


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)


def get_optimizer(args, model):
    # Taken from huggingface/transformers `run_language_modeling.py`
    NO_DECAY = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in NO_DECAY)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in NO_DECAY)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
