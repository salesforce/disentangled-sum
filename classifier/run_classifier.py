# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
""" Finetuning SciBERT on a binary classifier task. """

import argparse
import glob
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import ujson as json
from nltk import sent_tokenize
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    xnli_compute_metrics as compute_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class InputFeature:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(
            logdir=os.path.join(args.logdir, args.output_dir.split("/")[-1])
        )

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (
            len(train_dataloader) // args.gradient_accumulation_steps
        )
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps
        )

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoch",
            steps_trained_in_current_epoch,
        )

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert"] else None
                )  # XLM and DistilBERT don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[
                0
            ]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir
                    )

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def bin_position(max_val):
    """returns position features using some symbols. Concatenate them at the end of
    sentences to represent sentence lengths in terms of one of the three buckets.
    """
    symbol_map = {0: " `", 1: " _", 2: " @"}
    if max_val <= 3:
        return [symbol_map[i] for i in range(max_val)]

    first = max_val // 3
    second = 2 * first
    return [" `"] * first + [" _"] * (second - first) + [" @"] * (max_val - second)


def inference_on_paper_text(args, model, tokenizer, prefix=""):
    """run model inference on the summarization outputs after evaluation."""

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.eval()

    # List of experiments
    if args.infer_paper == "train":
        exps = ["train_{i:02}" for i in range(90)]
    elif args.infer_paper == "valid":
        exps = ["valid_{i:02}" for i in range(4)]
    else:
        exps = ["test_{i:02}" for i in range(6)]

    for exp in tqdm(exps, ncols=80, desc="Files"):
        with open(f"/export/scisumm/filtered_s2orc_20190928/{exp}.clf.jsonl", "r") as f:
            count = 0
            targets = []
            separate_indices = [0]
            for line in f:
                count += 1
                obj = json.loads(line)
                paper_texts = [
                    sent_tokenize(p["text"])
                    for p in obj["paper"]["grobid_parse"]["body_text"]
                ]
                separate_indices.append(separate_indices[-1] + len(paper_texts))
                targets += paper_texts
                if count > 100:
                    break
            with open(f"separate_indices_{exp}.txt", "w") as fsep:
                for i in separate_indices:
                    print(i, file=fsep)

        out_stream = Path(f"classified_indices_{exp}.txt").open("w")

        max_length = (
            tokenizer.max_len if args.max_seq_length is None else args.max_seq_length
        )

        # each "batch" is a summary -- list of sentences
        batch_encodings = [
            tokenizer.batch_encode_plus(
                target, max_length=max_length, pad_to_max_length=True, truncation=True
            )
            for target in targets
        ]

        purity_scores = []

        for t in tqdm(range(len(targets)), ncols=80, desc="Evaluating"):
            features = []
            batch_encoding = batch_encodings[t]
            for i in range(len(targets[t])):
                inputs = {k: batch_encoding[k][i] for k in batch_encoding}

                feature = InputFeature(**inputs, label=0)
                features.append(feature)

            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor(
                [f.input_ids for f in features], dtype=torch.long
            ).to(args.device)
            all_attention_mask = torch.tensor(
                [f.attention_mask for f in features], dtype=torch.long
            ).to(args.device)
            all_token_type_ids = torch.tensor(
                [f.token_type_ids for f in features], dtype=torch.long
            ).to(args.device)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long).to(
                args.device
            )

            with torch.no_grad():
                inputs = {
                    "input_ids": all_input_ids,
                    "attention_mask": all_attention_mask,
                    "labels": all_labels,
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1).tolist()

            selected_indices = [
                str(i)
                for i in range(len(preds))
                if preds[i] == (1 if args.mode == "contrib" else 0)
            ]
            purity_scores.append(len(selected_indices) / len(preds))
            # purity, total, indices
            print(
                f"{len(selected_indices) / len(preds):.3f}",
                len(preds),
                " ".join(selected_indices),
                sep="\t",
                file=out_stream,
            )
        tqdm.write(f"Average purity: {np.mean(purity_scores):.3f}")


def inference_on_summary_outputs(args, model, tokenizer, prefix=""):
    """run model inference on the summarization outputs after evaluation."""

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.eval()

    # List of experiments
    exps = list(args.decode_results)

    for exp in tqdm(exps, ncols=80, desc="Files"):
        result_file = list(exp.glob("*/*_results.json"))[0]
        tqdm.write(f"Evaluting: {result_file}")

        with result_file.open("r") as f:
            obj = json.load(f)
        examples = [sent_tokenize(o[args.decode_type]) for o in obj["outputs"]]
        targets = []
        # approximated -- not exactly correct binning.
        for e in examples:
            suffixes = bin_position(len(e))
            targets.append([s + suffixes[i] for i, s in enumerate(e)])

        out_stream = (result_file.parent / f"classified_indices.txt").open("w")

        max_length = (
            tokenizer.max_len if args.max_seq_length is None else args.max_seq_length
        )

        # each "batch" is a summary -- list of sentences
        batch_encodings = [
            tokenizer.batch_encode_plus(
                target, max_length=max_length, pad_to_max_length=True, truncation=True
            )
            for target in targets
        ]

        purity_scores = []

        for t in tqdm(range(len(targets)), ncols=80, desc="Evaluating"):
            features = []
            batch_encoding = batch_encodings[t]
            for i in range(len(targets[t])):
                inputs = {k: batch_encoding[k][i] for k in batch_encoding}

                feature = InputFeature(**inputs, label=0)
                features.append(feature)

            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor(
                [f.input_ids for f in features], dtype=torch.long
            ).to(args.device)
            all_attention_mask = torch.tensor(
                [f.attention_mask for f in features], dtype=torch.long
            ).to(args.device)
            all_token_type_ids = torch.tensor(
                [f.token_type_ids for f in features], dtype=torch.long
            ).to(args.device)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long).to(
                args.device
            )

            with torch.no_grad():
                inputs = {
                    "input_ids": all_input_ids,
                    "attention_mask": all_attention_mask,
                    "labels": all_labels,
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1).tolist()

            selected_indices = [
                str(i)
                for i in range(len(preds))
                if preds[i] == (1 if args.mode == "contrib" else 0)
            ]
            purity_scores.append(len(selected_indices) / len(preds))
            # purity, total, indices
            print(
                f"{len(selected_indices) / len(preds):.3f}",
                len(preds),
                " ".join(selected_indices),
                sep="\t",
                file=out_stream,
            )
        tqdm.write(f"Average purity: {np.mean(purity_scores):.3f}")


def inference(args, model, tokenizer, prefix=""):
    """run inference on the dataset to obtain contrib_indices field for each example."""

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.eval()

    assert args.infer_jsonl is not None
    files = list(args.data_dir.glob(f"{args.infer_jsonl}*.jsonl"))
    for ff in files:
        logger.info(f"{ff.name}")
    for orgfile in tqdm(files, ncols=80, desc="Files"):
        results = {}
        # make features
        with orgfile.open("r") as f:
            obj = [json.loads(l) for l in f]

        examples = [o["target"] for o in obj]
        targets = []
        for e in examples:
            suffixes = bin_position(len(e))
            targets.append([s + suffixes[i] for i, s in enumerate(e)])

        out_stream = (
            args.data_dir / "clf" / f'{orgfile.with_suffix("").name}.clf.jsonl'
        ).open("w")

        max_length = (
            tokenizer.max_len if args.max_seq_length is None else args.max_seq_length
        )

        batch_encodings = [
            tokenizer.batch_encode_plus(
                target, max_length=max_length, pad_to_max_length=True, truncation=True
            )
            for target in targets
        ]
        for t in tqdm(range(len(targets)), ncols=80, desc="Evaluating"):
            features = []
            batch_encoding = batch_encodings[t]
            for i in range(len(targets[t])):
                inputs = {k: batch_encoding[k][i] for k in batch_encoding}

                feature = InputFeature(**inputs, label=0)
                features.append(feature)

            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor(
                [f.input_ids for f in features], dtype=torch.long
            ).to(args.device)
            all_attention_mask = torch.tensor(
                [f.attention_mask for f in features], dtype=torch.long
            ).to(args.device)
            all_token_type_ids = torch.tensor(
                [f.token_type_ids for f in features], dtype=torch.long
            ).to(args.device)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long).to(
                args.device
            )

            with torch.no_grad():
                inputs = {
                    "input_ids": all_input_ids,
                    "attention_mask": all_attention_mask,
                    "labels": all_labels,
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1).tolist()

            raw_ex = obj[t]
            raw_ex["contrib_indices"] = [i for i in range(len(preds)) if preds[i] == 1]
            print(json.dumps(raw_ex), file=out_stream)


def evaluate(args, model, tokenizer, prefix=""):
    """run evaluation on the contribution classifier data."""

    results = {}
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert"] else None
                )  # XLM and DistilBERT don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = compute_metrics("xnli", preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        str(args.data_dir),
        "cached_{}_{}_{}".format(
            "test" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if evaluate:
            with (args.data_dir / "test.tsv").open("r") as f:
                examples = [l.split("\t") for l in f.read().strip().split("\n")]
        else:
            with (args.data_dir / "train.tsv").open("r") as f:
                examples = [l.split("\t") for l in f.read().strip().split("\n")]

        labels = [int(e[1]) for e in examples]

        max_length = (
            tokenizer.max_len if args.max_seq_length is None else args.max_seq_length
        )

        batch_encoding = tokenizer.batch_encode_plus(
            [example for example, _ in examples],
            max_length=max_length,
            pad_to_max_length=True,
        )
        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            feature = InputFeature(**inputs, label=labels[i])
            features.append(feature)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # for evaluating on paper. Specify the split (e.g. train/valid/test)
    parser.add_argument("--infer-paper", type=str, default=None)
    # for evaluating data (to generate contrib_indices)
    parser.add_argument("--infer-jsonl", type=str, default=None)
    # for evaluating system outputs
    parser.add_argument("--decode-type", type=str, default="beam")
    parser.add_argument(
        "--decode-results",
        type=Path,
        nargs="+",
        help="Paths to evaluation experiment directories.",
        default=None,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["contrib", "other"],
        help="Side to check purity scores.",
    )
    # Required parameters
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument(
        "--data_dir", default=None, type=Path, required=True,
    )
    parser.add_argument(
        "--model_name_or_path", default="allenai/scibert_scivocab_cased", type=str,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--tokenizer_name", default="allenai/scibert_scivocab_cased", type=str,
    )
    parser.add_argument(
        "--max_seq_length", default=128, type=int,
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the test set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Rul evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=5.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=500, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    args.model_type = config.model_type
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:

        if args.decode_results is not None:
            checkpoint = args.output_dir
            prefix = (
                checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            )
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            inference_on_summary_outputs(args, model, tokenizer, prefix=prefix)

        elif args.infer_jsonl is not None:
            inference(args, model, tokenizer, prefix=prefix)

        elif args.infer_paper is not None:
            inference_on_paper_text(args, model, tokenizer, prefix="")

        else:
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(
                        glob.glob(
                            args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True
                        )
                    )
                )
                logging.getLogger("transformers.modeling_utils").setLevel(
                    logging.WARN
                )  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = (
                    checkpoint.split("/")[-1]
                    if checkpoint.find("checkpoint") != -1
                    else ""
                )

                model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
                model.to(args.device)
                result = evaluate(args, model, tokenizer, prefix=prefix)
                result = dict(
                    (k + "_{}".format(global_step), v) for k, v in result.items()
                )
                results.update(result)

    return results


if __name__ == "__main__":
    main()
