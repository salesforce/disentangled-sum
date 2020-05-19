# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
import glob
import itertools
import json
import logging
import math
import os
import random

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


def shift_left(input_ids, prepend_token_id):
    new = torch.ones_like(input_ids) * prepend_token_id
    new[:, 1:] = input_ids[:, :-1]
    return new


class SequentialJSONIterableDataset(IterableDataset):
    def __init__(
        self,
        data_path,
        args,
        process_lines=True,
        reservoir_shuffle=False,
        reservoir_size=500,
        repeat=False,
    ):
        if not os.path.exists(os.path.dirname(data_path)):
            raise RuntimeError("Data path does not exist:", data_path)

        # store information about environment
        self.batch_size = args.per_unit_batch_size
        self.world_rank = args.world_rank
        self.world_size = args.world_size
        self.repeat = repeat

        # TODO: make this part of args?
        self.process_lines = process_lines

        self.reservoir_shuffle = reservoir_shuffle
        self.reservoir_size = reservoir_size
        # Set a common random number generator that will be consistent
        # across different workers
        if self.reservoir_shuffle:
            self.reservoir = []
            self.rng = random.Random(args.seed)
        else:
            self.reservoir = None
            self.rng = None

        # prepare data and datastreams
        self.assigned_files = self.assign_files(data_path, args)
        self.stream_size = self.get_stream_size(self.assigned_files)
        self.wrapped_size = args.batch_size * math.ceil(
            self.stream_size / args.batch_size
        )
        self.num_batches = self.wrapped_size / args.batch_size
        self.num_wrapped = self.wrapped_size - self.stream_size
        if self.repeat:
            self.data_stream = self.get_data_stream(
                self.assigned_files, max_examples=None
            )
        else:
            self.data_stream = self.get_data_stream(
                self.assigned_files, max_examples=self.wrapped_size
            )
        logger.info(
            "Stream size: %d; Wrapped size: %d; Num wrapped: %d"
            % (self.stream_size, self.wrapped_size, self.num_wrapped)
        )

    def __iter__(self):
        """
        Return data iterator
        """
        return self.data_stream

    def get_data_stream(self, assigned_files, max_examples):
        """
        Create data stream from assigned files.
        Reads each file line by line, processes example and yields it in a lazy way
        :param assigned_files: files assigned to this worker
        """

        def data_generator(self, assigned_files, max_examples):

            examples_generated = 0
            while True:
                for processed_file in assigned_files:
                    with open(processed_file, "r") as input_stream:
                        for idx, line in enumerate(input_stream):
                            if idx == 0 and "size" in json.loads(line):
                                continue  # skip metadata
                            if (not self.repeat) and (
                                examples_generated >= max_examples
                            ):
                                return

                            if self.process_lines:
                                line = self.process_json_line(line)
                            else:
                                line = json.loads(line)

                            if self.reservoir_shuffle:
                                self.reservoir.append(line)
                                if len(self.reservoir) >= self.reservoir_size:
                                    r_idx = self.rng.randrange(0, len(self.reservoir))
                                    yield self.reservoir.pop(r_idx)
                                    examples_generated += 1
                            else:
                                yield line
                                examples_generated += 1
                if not self.repeat:
                    return

        return data_generator(self, assigned_files, max_examples)

    @staticmethod
    def get_stream_size(assigned_files):
        total_size = 0
        for processed_file in assigned_files:
            with open(processed_file, "r") as input_stream:
                metadata = json.loads(next(input_stream))
            if "size" in metadata:
                logger.info("Get line count from metadata: %s" % metadata["size"])
                total_size += metadata["size"]
            else:
                # Read line count one by one
                logger.info("Read full file size %s ..." % processed_file)
                with open(processed_file, "r") as input_stream:
                    for line_count, _ in enumerate(input_stream, start=1):
                        pass
                logger.info("Line count: %s" % (line_count))
                total_size += line_count

        return total_size

    @staticmethod
    def wrap_data_stream(data_stream, max_examples):
        return itertools.islice(data_stream, stop=max_examples)

    @staticmethod
    def process_json_line(line):
        """
        Process a line read from the input stream into a Python dictionary
        :param line: input line read from data file
        """
        example = json.loads(line)
        if example["example_ix"] is not None:
            example["chunk_mask_ix"] = example["example_ix"]
        del example["example_ix"]
        return example

    @staticmethod
    def assign_files(data_path, args):
        if os.path.isfile(data_path):
            return [data_path]
        else:  # has to be pattern
            files = sorted(list(glob.glob(data_path)))
            return files
        # another case will be here if we want to shuffle data and assign to workers at random


def worker_init_fn(worker_id):
    """
    Initialize Dataset workers.
    Filters out the input data stream based on the worker mask using `itertools.compress`.
    :param worker_id: id of the current worker process
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # dataset copy in this worker process
    worker_id = worker_info.id
    num_workers = worker_info.num_workers
    batch_size = dataset.batch_size
    world_rank = dataset.world_rank
    world_size = dataset.world_size

    worker_mask = itertools.cycle(
        compute_worker_mask(worker_id, num_workers, world_rank, world_size, batch_size)
    )
    dataset.data_stream = itertools.compress(dataset.data_stream, worker_mask)


def compute_worker_mask(worker_rank, num_workers, world_rank, world_size, batch_size):
    """
    Computes a worker specific mask that will be used to filter out the examples
    from the input stream that the given worker should handle.
    :param worker_rank: rank of current DataLoader worker
    :param num_workers: total number of DataLoader worker, per distributed proces
    :param world_rank: rank of current process in distributed setting
    :param world_size: total number of distributed processes
    :param batch_size: size of batch handled by each worker
    """
    if num_workers == 0:
        raise RuntimeError(
            "Computing worker mask requires at least `1` DataLoader worker"
        )

    if num_workers >= 1 and world_size <= 1:
        worker_mask = [False] * (num_workers * batch_size)
        worker_start_ix = batch_size * worker_rank
        worker_end_ix = batch_size * (worker_rank + 1)
        worker_mask[worker_start_ix:worker_end_ix] = [True] * batch_size
        logger.debug(f"Worker Mask (1st case): {worker_mask}")
        return worker_mask
    elif num_workers == 1 and world_size > 1:
        world_mask = [False] * (world_size * batch_size)
        world_start_ix = batch_size * world_rank
        world_end_ix = batch_size * (world_rank + 1)
        world_mask[world_start_ix:world_end_ix] = [True] * batch_size
        logger.debug(f"Worker Mask (2nd case): {world_mask}")
        return world_mask
    else:  # num_worker > 1 and world_size > 1
        world_mask = [False] * (world_size * batch_size)
        world_start_ix = batch_size * world_rank
        world_end_ix = batch_size * (world_rank + 1)
        world_mask[world_start_ix:world_end_ix] = [True] * batch_size

        global_mask = [False] * (world_size * num_workers * batch_size)
        global_start_ix = len(world_mask) * worker_rank
        global_end_ix = len(world_mask) * (worker_rank + 1)
        global_mask[global_start_ix:global_end_ix] = world_mask
        logger.debug(f"Worker Mask (3rd case): {world_mask}")
        return global_mask
