# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
import json
import os
import subprocess
from argparse import ArgumentParser, Namespace
from datetime import datetime

import src.utils_cuda as utils_cuda
import src.utils_dist as utils_dist
from tensorboardX import SummaryWriter

IGNORED_ARGS_WHEN_SAVE = {"device"}


def get_parser():
    parser = ArgumentParser()

    parser.add_argument("--seed", default=54135, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=10, type=int)
    parser.add_argument("--max_eval_batches", default=None, type=int)
    parser.add_argument("--max_input_length", default=1024, type=int)
    parser.add_argument("--max_output_length", default=200, type=int)
    parser.add_argument("--max_external_input_length", default=512, type=int)
    parser.add_argument("--generate_max_length", default=200, type=int)
    parser.add_argument("--eval_every", default=25, type=int)
    parser.add_argument("--log_every", default=5, type=int)
    parser.add_argument("--logdir", default=None, type=str)
    parser.add_argument("--datadir", required=True, type=str)
    parser.add_argument("--tokenizer_name", default="facebook/bart-large", type=str)
    parser.add_argument("--model_name_or_path", default="facebook/bart-large", type=str)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--use_apex", action="store_true")
    parser.add_argument(
        "--target_type",
        type=str,
        choices=["full", "contrib", "context", "both"],
        default="full",
    )
    parser.add_argument("--no_modify_prefix", action="store_true")
    parser.add_argument(
        "--input_type", type=str, choices=["all", "paper"], default="paper"
    )
    parser.add_argument(
        "--aux_scale",
        type=float,
        default=1,
        help="Lambda, controlling the strength of auxiliary losses.",
    )
    parser.add_argument(
        "--use_adaptive_scale",
        action="store_true",
        help="Only activate the loss in the *maximize similarity* direction.",
    )

    # For reservoir shuffle
    parser.add_argument("--reservoir_shuffle_size", type=int, default=2000)

    # Evaluation only
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--test_split", default="val", choices=["train", "val", "test"])

    # For nucleus and beam search generation
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)

    parser.add_argument("--no_beam_decode", action="store_true")
    parser.add_argument("--no_greedy_decode", action="store_true")
    parser.add_argument("--no_nucleus_decode", action="store_true")

    parser.add_argument("--use_multi_head", action="store_true")
    parser.add_argument("--use_informativeness", action="store_true")
    parser.add_argument("--force_head", type=int, default=None)

    return parser


def save_args(args, with_tensorboard=True, output_filename="config.json"):
    args_to_save = {
        k: v for k, v in vars(args).items() if k not in IGNORED_ARGS_WHEN_SAVE
    }
    if args.logdir is not None:
        with open(os.path.join(args.logdir, output_filename), "w") as f:
            json.dump(args_to_save, f, indent=2, sort_keys=True)

    hparams = {
        k: v
        for k, v in args_to_save.items()
        if (isinstance(v, int) or isinstance(v, float))
    }

    if with_tensorboard and args.logdir:
        with SummaryWriter(logdir=args.logdir) as tb_writer:
            tb_writer.add_hparams(hparams, metric_dict={})


def load_args(config_filepath):
    with open(config_filepath, "r") as f:
        args = Namespace(**json.load(f))

    current_commit_hash = get_current_commit_hash()
    if args.commit_hash != current_commit_hash:
        # TODO: use proper warnings here
        print(
            "Warning: this config is for an older code version: %s" % args.commit_hash
        )
    return set_gpu_args(args)


def get_current_commit_hash():
    # https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    try:
        return (
            subprocess.check_output(["git", "describe", "--always"])
            .decode("utf-8")
            .strip()
        )
    except:
        return None


def set_gpu_args(args):
    args.per_unit_batch_size = args.batch_size
    utils_dist.init(args)
    utils_cuda.init(args)
    return args


def get_args():
    parser = get_parser()
    utils_dist.add_group(parser)

    args = set_gpu_args(parser.parse_args())

    args.commit_hash = get_current_commit_hash()
    args.timestamp = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
    if args.logdir is not None:
        if args.run_name is not None:
            args.logdir = os.path.join(args.logdir, args.run_name, args.timestamp)
        else:
            args.logdir = os.path.join(args.logdir, args.timestamp)
        os.makedirs(args.logdir, exist_ok=True)
    else:
        print("Warning: no logdir specified. This run will not be saved")

    return args
