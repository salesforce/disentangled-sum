# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
import logging
import os

import torch

try:
    from apex.parallel import DistributedDataParallel as AmpDistributedDataParallel

    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init(args):
    args.batch_size = args.per_unit_batch_size

    if "WORLD_SIZE" in os.environ:  # set by torch.distributed.launch
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)  # set the device local to this node
        args.world_size = torch.distributed.get_world_size()
        args.world_rank = torch.distributed.get_rank()
        args.distributed = True
        args.batch_size *= args.world_size
        logger.info(f"Initialized Distributed Environment")
        logger.info(f"world rank: {args.world_rank}")
        logger.info(f"node: {args.node_rank}")
        logger.info(f"global batch size: {args.batch_size}")
        if hasattr(args, "gradient_accumulation_steps"):
            args.gradient_accumulation_steps //= args.world_size
            logger.info(f"grad acc steps: {args.gradient_accumulation_steps}")
    else:
        args.world_size = 1 if torch.cuda.is_available() and not args.no_cuda else 0
        args.world_rank = 0 if args.world_size else -1
        args.local_rank = 0 if args.world_size else -1
        args.distributed = False
        logger.info(f"Initialized Non-Distributed Environment")
        logger.info(f"world rank: {args.world_rank}")
        logger.info(f"node: {args.node_rank}")
        logger.info(f"global batch size: {args.batch_size}")

    args.first_rank = args.world_rank <= 0


def wrap(model, args):
    if args.use_apex and HAS_APEX:
        return AmpDistributedDataParallel(model)
    return torch.nn.parallel.DistributedDataParallel(
        model, [args.local_rank], args.local_rank, find_unused_parameters=True
    )
