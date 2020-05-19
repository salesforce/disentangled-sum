# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
import logging
import timeit

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init(args):
    args.cuda = args.world_size >= 1
    if args.cuda:
        if args.first_rank:
            logger.info("Initializing CUDA")
        t0 = timeit.default_timer()
        torch.randn(1, device="cuda")
        elapsed = timeit.default_timer() - t0
        if args.first_rank:
            logger.info(f"CUDA initailization takes {elapsed:.2f}. Should be <~5s.")
