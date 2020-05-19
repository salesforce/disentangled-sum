# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
import functools
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer

import src.utils_dist as utils_dist
from src.args import get_args, save_args
from src.modeling_bart import (
    MultiHeadBartForConditionalGeneration,
    MultiInputBartForConditionalGeneration,
    MultiInputMultiHeadBartForConditionalGeneration,
)
from src.utils import check_nan, get_optimizer, set_seed
from src.utils_data import SequentialJSONIterableDataset, shift_left, worker_init_fn
from src.utils_rouge import nltk_download_no_ssl, run_rouge
from tensorboardX import SummaryWriter

try:
    from apex import amp

    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


nltk_download_no_ssl("punkt")


CROSS_ENTROPY_IGNORE_INDEX = -100
BOS_OUTPUT_TOKEN = "<extra_id_0>"
EOS_TOKEN_ID = 5


def collate_fn(
    batch,
    input_type: str,
    target_type: str,
    modify_prefix: bool = False,
    input_prefix: str = "",
    input_suffix: str = " </s>",
    target_prefix: str = "",
    target_suffix: str = " </s>",
    return_pid: bool = True,
):
    """Organize input output pairs by the type of model we use.
    input_type:
        - paper: returns source document as input
        - all: returns a tuple of (source doc, inbound citations, outbound citations)
    target_type:
        - full: returns a full summary.
        - contrib: returns a contribution summary.
        - context: returns a context summary.
        - both: returns a tuple of (contribution, context) summaries.
    """

    # Input preprocessing
    if input_type == "paper":
        ip_preproc = lambda x: " ".join(
            [p["text"] for p in x["paper"]["grobid_parse"]["body_text"]]
        )
    elif input_type == "all":
        ip_preproc = lambda x: (
            " ".join([p["text"] for p in x["paper"]["grobid_parse"]["body_text"]]),
            " ".join([i[1] for i in x["inbound"]]),
            " ".join([i[1] for i in x["outbound"]]),
        )

    # Target postprocessing
    tgt_postproc = lambda x: target_prefix + x + target_suffix

    # For "both" targets, inputs need to be duplicated, with proper prefixes
    if target_type == "both":
        # Input postprocessing
        ip_postproc = lambda x: x[0] + x[1] + input_suffix

        if modify_prefix:
            # NOTE: We actually used "other" as the exact code, but it should not
            # matter as long as the same code is used for train/test.
            c_pref, o_pref = "contribution: ", "other: "
        else:
            c_pref, o_pref = "", ""

        if input_type == "paper":
            inputs = [
                (
                    ip_postproc((c_pref, ip_preproc(doc))),
                    ip_postproc((o_pref, ip_preproc(doc))),
                )
                for doc in batch
            ]
        else:
            inputs = []
            for doc in batch:
                ip_docs = ip_preproc(doc)  # input, incite, outcite
                inputs.append(
                    [
                        list(
                            (ip_postproc((c_pref, ip_docs[0])), ip_docs[1], ip_docs[2])
                        ),
                        list(
                            (ip_postproc((o_pref, ip_docs[0])), ip_docs[1], ip_docs[2])
                        ),
                    ]
                )

        # Target preprocessing
        tgt_preproc = lambda x: (" ".join(x["target"]), " ".join(x["target"]))
        targets = [list(map(tgt_postproc, tgt_preproc(doc))) for doc in batch]
    else:
        # Input postprocessing
        ip_postproc = lambda x: input_prefix + x + input_suffix

        if input_type == "paper":
            inputs = [ip_postproc(ip_preproc(doc)) for doc in batch]
        else:
            # only apply prefixing on the first element, i.e., document to summarize
            inputs = [
                [
                    ip_postproc(d) if idx == 0 else d
                    for idx, d in enumerate(ip_preproc(doc))
                ]
                for doc in batch
            ]

        # Target preprocessing
        if target_type == "full":
            tgt_preproc = lambda x: " ".join(x["target"])

        elif target_type == "contrib":
            if modify_prefix:
                input_prefix = "contribution: "
            tgt_preproc = lambda x: " ".join(x["target"])

        elif target_type == "context":
            if modify_prefix:
                # NOTE: We actually used "other" as the exact code, but it should not
                # matter as long as the same code is used for train/test.
                input_prefix = "other: "
            tgt_preproc = lambda x: " ".join(x["target"])

        targets = [tgt_postproc(tgt_preproc(doc)) for doc in batch]

    if not return_pid:
        return inputs, targets
    return [doc["paper"]["paper_id"] for doc in batch], inputs, targets


def tokenize_batch(input_texts, output_texts, model, tokenizer, args):
    """Tokenize text. If necessary, apply the tokenization on all the elements in the
    list of sources.
    """

    def tokenize_input(ip, max_length, model):
        # Idk if this is the best way to do it...
        input_max_length = tokenizer.batch_encode_plus(
            ip, return_tensors="pt", pad_to_max_length=True, add_special_tokens=False
        )["input_ids"].size(1)

        tok_input = tokenizer.batch_encode_plus(
            ip,
            return_tensors="pt",
            pad_to_max_length=True,
            max_length=min(input_max_length, max_length),
            add_special_tokens=False,
        )

        if args.distributed:
            model = model.module

        return tok_input

    if isinstance(input_texts[0], list):
        tok_input = [
            tokenize_input(i, l, model)
            for i, l in zip(
                zip(*input_texts),
                [
                    args.max_input_length,
                    args.max_external_input_length,
                    args.max_external_input_length,
                ],
            )
        ]
    else:
        tok_input = tokenize_input(input_texts, args.max_input_length, model)

    output_max_length = tokenizer.batch_encode_plus(
        output_texts,
        return_tensors="pt",
        pad_to_max_length=True,
        add_special_tokens=False,
    )["input_ids"].size(1)

    tok_output = tokenizer.batch_encode_plus(
        output_texts,
        return_tensors="pt",
        pad_to_max_length=True,
        max_length=min(output_max_length, args.max_output_length),
        add_special_tokens=False,
    )

    labels = tok_output["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = CROSS_ENTROPY_IGNORE_INDEX

    if args.cuda:
        if isinstance(tok_input, list):
            tok_input = [{k: v.cuda() for k, v in ti.items()} for ti in tok_input]
        else:
            tok_input = {k: v.cuda() for k, v in tok_input.items()}
        tok_output = {k: v.cuda() for k, v in tok_output.items()}
        labels = labels.cuda()

    return tok_input, tok_output, labels


def evaluation(args, model, global_step, tokenizer, eval_iter, write_summary=True):
    """Evaluation in between training steps."""

    generate_max_length = args.generate_max_length

    if args.max_eval_batches == 0:
        print("Skipping evaluation (`args.max_eval_batches` set to 0)")
        return

    print("Run evaluation...")

    eval_losses = []
    all_ground_truth = []
    all_generated = {k: [] for k in ["beam", "greedy", "nucleus"]}
    outstream = open((Path(args.logdir) / f"eval.jsonl.{args.world_rank}"), "w")

    if args.distributed:
        model.require_forward_param_sync = False
        model.require_backward_grad_sync = False

    if args.distributed:
        model_for_eval = model.module
    else:
        model_for_eval = model

    t0 = time.time()
    batch_idx = 0
    with torch.no_grad():
        for (pids, input_texts, output_texts) in eval_iter:

            if len(input_texts) == 0 or sum(len(out) for out in output_texts) == 0:
                continue

            additional_kwargs = {}
            bsz = len(input_texts)

            # single model.forward with multiple outputs
            # MultiHead model
            if args.use_multi_head and args.target_type == "both":
                all_labels = []
                all_dec_inputs = []
                if args.input_type == "paper":
                    ips = [i for ip in input_texts for i in ip]
                elif args.input_type == "all":
                    ips = [list(ip[0]) for ip in input_texts]

                ops = [o for op in output_texts for o in op]
                tok_input, tok_output, labels = tokenize_batch(
                    ips, ops, model, tokenizer, args
                )
                # Input is only a paper, so take the first slice of the tensor
                if args.input_type == "paper":
                    tok_input["input_ids"] = tok_input["input_ids"].view(bsz, 2, -1)[
                        :, 0, :
                    ]
                    tok_input["attention_mask"] = tok_input["attention_mask"].view(
                        bsz, 2, -1
                    )[:, 0, :]
                elif args.input_type == "all":
                    # make it compatible
                    new_tok_input = {}
                    new_tok_input["input_ids"] = [t["input_ids"] for t in tok_input]
                    new_tok_input["attention_mask"] = [
                        t["attention_mask"] for t in tok_input
                    ]
                    tok_input = new_tok_input
                    additional_kwargs = {
                        "final_layer": [None, None, None],
                        "input_modes": ["LogL", "MI_inbound", "MI_outbound"],
                        "informativeness": args.use_informativeness,
                    }

                tok_output["input_ids"] = tok_output["input_ids"].view(bsz, 2, -1)
                tok_output["attention_mask"] = tok_output["attention_mask"].view(
                    bsz, 2, -1
                )
                labels = labels.view(bsz, 2, -1)
                for idx in range(2):
                    all_dec_inputs.append(
                        dict(
                            input_ids=tok_output["input_ids"][:, idx, :],
                            attention_mask=tok_output["attention_mask"][:, idx, :],
                        )
                    )
                    all_labels.append(labels[:, idx, :])

                out_with_loss = model_for_eval(
                    input_ids=tok_input["input_ids"],
                    attention_mask=tok_input["attention_mask"],
                    decoder_input_ids=[
                        shift_left(tok_output["input_ids"], tokenizer.bos_token_id)
                        for tok_output in all_dec_inputs
                    ],
                    decoder_attention_mask=[
                        tok_output["attention_mask"] for tok_output in all_dec_inputs
                    ],
                    lm_labels=all_labels,
                    **additional_kwargs,
                )

            else:
                # only evaluate contribution for simplicity when training with "both"
                if args.target_type == "both":  # not use_multi_head
                    input_texts, output_texts = (
                        [i[0] for i in input_texts],
                        [i[0] for i in output_texts],
                    )

                tok_input, tok_output, labels = tokenize_batch(
                    input_texts, output_texts, model, tokenizer, args
                )

                # Run it in train mode for comparison with training loss
                model_for_eval.train()
                if args.input_type == "all":
                    ip, mask = (
                        [t["input_ids"] for t in tok_input],
                        [t["attention_mask"] for t in tok_input],
                    )
                else:
                    ip, mask = tok_input["input_ids"], tok_input["attention_mask"]

                # In the case of multi_head model, this specify which head to use.
                # 0 for contribution and 1 for context
                if args.force_head is not None:
                    additional_kwargs["final_layer"] = args.force_head

                out_with_loss = model_for_eval(
                    input_ids=ip,
                    attention_mask=mask,
                    decoder_input_ids=shift_left(
                        tok_output["input_ids"], tokenizer.bos_token_id
                    ),
                    decoder_attention_mask=tok_output["attention_mask"],
                    lm_labels=labels,
                    **additional_kwargs,
                )

            # Multiple objectives
            if args.input_type == "all" and args.use_multi_head:
                if args.use_informativeness:
                    contrib_loss = (
                        out_with_loss["LogL"][0][0]
                        + args.aux_scale * out_with_loss["MI_outbound"][0][0]
                    )
                    context_loss = (
                        out_with_loss["LogL"][1][0]
                        + args.aux_scale * out_with_loss["MI_inbound"][1][0]
                    )
                    loss = (contrib_loss + context_loss) / 2
                    losses = [
                        loss,
                        out_with_loss["MI_inbound"][0][0],
                        out_with_loss["MI_outbound"][0][0],
                    ]
                else:  # Mutual Information instead of informativeness
                    contrib_loss = (
                        out_with_loss["LogL"][0][0]
                        - args.aux_scale * out_with_loss["MI_inbound"][0][0]
                        + (
                            args.aux_scale * out_with_loss["MI_outbound"][0][0]
                            if not args.use_adaptive_scale
                            else 0
                        )
                    )
                    context_loss = (
                        out_with_loss["LogL"][1][0]
                        + (
                            args.aux_scale * out_with_loss["MI_inbound"][1][0]
                            if not args.use_adaptive_scale
                            else 0
                        )
                        - args.aux_scale * out_with_loss["MI_outbound"][1][0]
                    )
                    loss = (contrib_loss + context_loss) / 2
                    losses = [
                        loss,
                        out_with_loss["MI_inbound"][0][0],
                        out_with_loss["MI_outbound"][0][0],
                    ]

                eval_losses.append(loss)
                input_ids = tok_input["input_ids"][0]
                attn_mask = tok_input["attention_mask"][0]

            # Multiple Inputs: Informativeness loss
            elif args.input_type == "all":
                losses = out_with_loss[0]

                if args.target_type == "contrib":
                    if args.use_informativeness:
                        coeff = [1, 0, args.aux_scale]
                    else:
                        coeff = [
                            1,
                            -args.aux_scale,
                            args.aux_scale if not args.use_adaptive_scale else 0,
                        ]
                elif args.target_type == "context":
                    if args.use_informativeness:
                        coeff = [1, args.aux_scale, 0]
                    else:
                        coeff = [
                            1,
                            args.aux_scale if not args.use_adaptive_scale else 0,
                            -args.aux_scale,
                        ]
                else:
                    coeff = [
                        1,
                        -args.aux_scale,
                        args.aux_scale,
                    ]  # maximize MI against inbound.

                eval_losses.append(sum(l * c for l, c in zip(losses, coeff)))
                input_ids = tok_input[0]["input_ids"]
                attn_mask = tok_input[0]["attention_mask"]

            # MultiHead but without auxiliary losses
            elif args.use_multi_head:
                # contrib, context
                losses = [o[0] for o in out_with_loss]
                eval_losses.append((sum(losses) / len(losses)).detach())
                input_ids = tok_input["input_ids"]
                attn_mask = tok_input["attention_mask"]

            else:
                eval_losses.append(out_with_loss[0].detach())
                input_ids = tok_input["input_ids"]
                attn_mask = tok_input["attention_mask"]

            generate_args = {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "max_length": generate_max_length,
                "early_stopping": True,
                "repetition_penalty": args.repetition_penalty,
                "bos_token_id": tokenizer.bos_token_id,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            decode_dump = [dict(paper_id=p) for p in pids]

            if args.use_multi_head:
                assert args.force_head < 2
                generate_args["final_layer"] = args.force_head
                output_texts = [o[generate_args["final_layer"]] for o in output_texts]

            model_for_eval.eval()

            # Decode
            if not args.no_beam_decode:
                out_generate_beam = model_for_eval.generate(
                    do_sample=False, num_beams=args.num_beams, **generate_args
                )
                generated = [
                    tokenizer.decode(out, skip_special_tokens=True)
                    for out in out_generate_beam.to("cpu")
                ]
                all_generated["beam"] += generated
                for idx, g in enumerate(generated):
                    decode_dump[idx]["decoded"] = g

            if not args.no_greedy_decode:
                out_generate_greedy = model_for_eval.generate(
                    do_sample=False, **generate_args
                )
                generated = [
                    tokenizer.decode(out, skip_special_tokens=True)
                    for out in out_generate_greedy.to("cpu")
                ]
                all_generated["greedy"] += generated

            if not args.no_nucleus_decode:
                out_generate_nucleus = model_for_eval.generate(
                    do_sample=True, top_k=args.top_k, top_p=args.top_p, **generate_args
                )
                generated = [
                    tokenizer.decode(out, skip_special_tokens=True)
                    for out in out_generate_nucleus.to("cpu")
                ]
                all_generated["nucleus"] += generated

            all_ground_truth += [
                o.replace(tokenizer.bos_token, "").strip() for o in output_texts
            ]
            for idx, g in enumerate(
                [o.replace(tokenizer.bos_token, "").strip() for o in output_texts]
            ):
                decode_dump[idx]["gold"] = g.replace(" </s>", "")

            for rec in decode_dump:
                assert "gold" in rec and "decoded" in rec and "paper_id" in rec
                print(json.dumps(rec, ensure_ascii=False), file=outstream)
            outstream.flush()

            batch_idx += 1
            if (args.max_eval_batches is not None) and (
                batch_idx >= args.max_eval_batches
            ):
                break

            if args.first_rank and batch_idx % 10 == 0:
                diff = (time.time() - t0) / 60
                print(f"Decoding: {batch_idx}. ({diff:.1f} min)")
                t0 = time.time()

    if args.distributed:
        model.require_forward_param_sync = True
        model.require_backward_grad_sync = True

    avg_eval_loss = torch.mean(torch.stack(eval_losses))

    all_rouges = {
        k: run_rouge(all_generated[k], all_ground_truth, generate_max_length)
        for k in ["beam", "greedy", "nucleus"]
        if len(all_generated[k]) > 0
    }

    # Average loss and ROUGE values across GPUs
    if args.distributed:
        dist.all_reduce(avg_eval_loss)
        avg_eval_loss = avg_eval_loss.to("cpu").item() / args.world_size

        avg_rouges = {}

        for name, scores in all_rouges.items():
            rouge_keys = []
            rouge_values = []
            # Sort by key to ensure consistent ordering across GPUs
            for k, vv in sorted(scores.items()):
                rouge_keys.append(k)
                rouge_values.append(vv["f"])

            rouge_values = torch.tensor(rouge_values).cuda()

            dist.all_reduce(rouge_values)
            rouge_values = (rouge_values / args.world_size).to("cpu").tolist()
            avg_rouges[name] = {k: {"f": vv} for k, vv in zip(rouge_keys, rouge_values)}

        all_rouges = avg_rouges
    else:
        avg_eval_loss = avg_eval_loss.to("cpu").item()

    if args.logdir is not None and write_summary and args.first_rank:
        tb_writer.add_scalar("eval_loss", avg_eval_loss, global_step)
        tb_writer.add_text("eval_ground_truth_sample", all_ground_truth[0], global_step)
        for k, v in all_generated.items():
            if len(v) > 0:
                tb_writer.add_text(f"eval_{k}_sample", v[0], global_step)

        eval_dump = open(Path(args.logdir) / "eval_scores.txt", "a")
        print(global_step, f"{avg_eval_loss:2.3f}", end="\t", file=eval_dump, sep="\t")
        for name, scores in all_rouges.items():
            for k, vv in scores.items():
                tb_writer.add_scalar(
                    "eval_rouge_%s_%s-F" % (name, k), vv["f"], global_step
                )
                print(f'{vv["f"]:1.5f}', file=eval_dump, sep="\t", end="\t")
        print(file=eval_dump)
        eval_dump.close()

    gen_results = [
        {k: "" for k in all_generated.keys()} for _ in enumerate(all_ground_truth)
    ]
    for k, v in all_generated.items():
        for i, text in enumerate(v):
            gen_results[i][k] = text

    return {
        "rouge_scores": all_rouges,
        "outputs": [
            {"ground_truth": gt, **gen}
            for (gt, gen) in zip(all_ground_truth, gen_results)
        ],
    }


def update_step(
    args,
    model,
    tokenizer,
    optimizer,
    loss,
    losses,
    eval_iter,
    step,
    accumulation_step,
    loss_log,
):
    loss_bp = loss / args.gradient_accumulation_steps
    if args.use_apex and HAS_APEX:
        with amp.scale_loss(loss_bp, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss_bp.backward()

    if (accumulation_step + 1) % args.gradient_accumulation_steps == 0:
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        model.zero_grad()

        if step % args.log_every == 0:
            # Local mean
            loss_value = torch.mean(torch.stack(loss_log))
            if args.distributed:
                dist.all_reduce(loss_value)
                loss_value = loss_value.to("cpu").item() / args.world_size
            else:
                loss_value = loss_value.to("cpu").item()
            loss_log = []

            if args.first_rank:
                if args.input_type == "all":
                    print(
                        f"Training step: {step}, LOSS: {loss_value:4.2f}"
                        f" C: ({losses[1].item():.4f} / {losses[2].item():.4f})"
                        f" O: ({losses[4].item():.4f} / {losses[5].item():.4f})"
                    )
                else:
                    print(f"Training step: {step}, LOSS: {loss_value:4.2f}")
                if args.logdir is not None:
                    tb_writer.add_scalar("train_loss", loss_value, step)

        if args.eval_every != 0 and ((step + 1) % args.eval_every == 0):
            evaluation(args, model, step + 1, tokenizer, eval_iter, write_summary=True)

            if args.first_rank and args.logdir is not None:
                checkpoint_path = os.path.join(
                    args.logdir, "checkpoint_%s" % (step + 1)
                )
                print("SAVE MODEL: %s" % checkpoint_path)
                os.makedirs(checkpoint_path, exist_ok=True)
                if args.distributed:
                    model.module.save_pretrained(checkpoint_path)
                else:
                    model.save_pretrained(checkpoint_path)

            model.train()
        accumulation_step = 0
        step += 1
    else:
        accumulation_step += 1

    return step, accumulation_step, loss_log


def run(args):
    save_args(args, with_tensorboard=True)

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    tokenizer.bos_token = BOS_OUTPUT_TOKEN  # For decoding specifically

    train_dataset, eval_dataset, test_dataset = [
        SequentialJSONIterableDataset(
            os.path.join(args.datadir, f"cord_*.jsonl",),
            args=args,
            process_lines=False,
            reservoir_shuffle=shuffle,
            repeat=repeat,
            reservoir_size=args.reservoir_shuffle_size,
        )
        for (split, shuffle, repeat) in [
            ("train", True, True),
            ("valid", False, True),
            ("test", False, False),
        ]
    ]

    # Multiple inputs. Use Informativeness
    if args.input_type == "all":
        # ControlCode or generic Bart
        model = MultiInputBartForConditionalGeneration.from_pretrained(
            args.model_name_or_path
        )
        # MultiHead
        if args.use_multi_head:
            model = MultiInputMultiHeadBartForConditionalGeneration.from_pretrained_multi(
                args.model_name_or_path
            )

    elif args.use_multi_head:
        # MultiHead
        model = MultiHeadBartForConditionalGeneration.from_pretrained_multi(
            args.model_name_or_path
        )
    else:
        # ControlCode or generic Bart
        model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # Set special token IDs for eval function
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )  # Might not be necessary, but idk

    if args.cuda:
        model = model.to("cuda")
    if args.distributed:
        model = utils_dist.wrap(model, args)

    optimizer = get_optimizer(args, model)
    if args.use_apex and HAS_APEX:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    collate_fn_filled = functools.partial(
        collate_fn,
        input_type=args.input_type,
        modify_prefix=(not args.no_modify_prefix),
        target_type=args.target_type,
    )

    if args.test_only:  # run on test set
        print("=== TEST/EVAL ONLY, no training")

        named_splits = {
            "train": train_dataset,
            "valid": eval_dataset,
            "test": test_dataset,
        }

        selected_split = named_splits[args.test_split]
        eval_iter = DataLoader(
            selected_split,
            batch_size=args.eval_batch_size,
            collate_fn=collate_fn_filled,
            num_workers=1,
            worker_init_fn=worker_init_fn,
        )
        results = evaluation(args, model, 0, tokenizer, eval_iter, write_summary=False)

        print(results["rouge_scores"])

        # Save results in JSON file
        results_filename = Path(args.logdir) / f"{args.test_split}_results.json"

        with results_filename.open("w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        return

    model.train()

    global_step = 0
    grad_acc_step = 0
    loss_tensor_log = []

    train_iter = DataLoader(
        train_dataset,
        batch_size=args.per_unit_batch_size,
        collate_fn=collate_fn_filled,
        num_workers=args.num_data_workers,
        worker_init_fn=worker_init_fn,
    )

    eval_iter = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn_filled,
        num_workers=args.num_data_workers,
        worker_init_fn=worker_init_fn,
    )

    for _, (_, input_texts, output_texts) in enumerate(train_iter):
        if len(input_texts) == 0:
            continue

        # Prohibit batches with no contribution summaries at all
        if sum(len(out) for out in output_texts) == 0:
            continue

        # MultiHead + Auxiliary loss (Informativeness)
        if args.target_type == "both" and args.use_multi_head:
            if args.input_type == "paper":
                ips = [i for ip in input_texts for i in ip]
            elif args.input_type == "all":
                ips = [list(ip[0]) for ip in input_texts]

            ops = [o for op in output_texts for o in op]
            tok_input, tok_output, labels = tokenize_batch(
                ips, ops, model, tokenizer, args
            )

            # Prepare inputs
            if args.input_type == "paper":
                tok_input["input_ids"] = tok_input["input_ids"].view(
                    args.per_unit_batch_size, 2, -1
                )[:, 0, :]
                tok_input["attention_mask"] = tok_input["attention_mask"].view(
                    args.per_unit_batch_size, 2, -1
                )[:, 0, :]
                additional_kwargs = {}
            elif args.input_type == "all":
                new_tok_input = {}
                new_tok_input["input_ids"] = [t["input_ids"] for t in tok_input]
                new_tok_input["attention_mask"] = [
                    t["attention_mask"] for t in tok_input
                ]
                tok_input = new_tok_input
                additional_kwargs = {
                    "final_layer": [None, None, None],
                    "input_modes": ["LogL", "MI_inbound", "MI_outbound"],
                    "informativeness": args.use_informativeness,
                }

            # b x [cont, ctx] x seq_len
            tok_output["input_ids"] = tok_output["input_ids"].view(
                args.per_unit_batch_size, 2, -1
            )
            tok_output["attention_mask"] = tok_output["attention_mask"].view(
                args.per_unit_batch_size, 2, -1
            )
            labels = labels.view(args.per_unit_batch_size, 2, -1)

            # Fixing the strange behavior of torch.distributed where some values
            # are overwritten when the sequence length is just one.
            for b in range(args.per_unit_batch_size):
                tok_output["input_ids"][b][tok_output["input_ids"][b][:, 0] == 1, 0] = 2
                tok_output["attention_mask"][b][
                    tok_output["attention_mask"][b][:, 0] == 0, 0
                ] = 1
                labels[b][labels[b][:, 0] == -100, 0] = 2

            all_labels = []
            all_dec_inputs = []
            for idx in range(2):
                all_dec_inputs.append(
                    dict(
                        input_ids=tok_output["input_ids"][:, idx, :],
                        attention_mask=tok_output["attention_mask"][:, idx, :],
                    )
                )
                all_labels.append(labels[:, idx, :])

            # Disable sync except at the beginning and the end of gradient accumulation
            if args.distributed:
                if (grad_acc_step == 0) or (
                    (grad_acc_step + 1) % args.gradient_accumulation_steps == 0
                ):
                    model.require_forward_param_sync = True
                    model.require_backward_grad_sync = True
                else:
                    model.require_forward_param_sync = False
                    model.require_backward_grad_sync = False

            outs = model(
                input_ids=tok_input["input_ids"],
                attention_mask=tok_input["attention_mask"],
                decoder_input_ids=[
                    shift_left(tok_output["input_ids"], tokenizer.bos_token_id)
                    for tok_output in all_dec_inputs
                ],
                decoder_attention_mask=[
                    tok_output["attention_mask"] for tok_output in all_dec_inputs
                ],
                lm_labels=all_labels,
                **additional_kwargs,
            )

            # MultiHead + Informativeness
            if args.input_type == "all":
                # losses for generating both contrib & context
                if args.use_informativeness:
                    # MI_outbound: informativeness
                    contrib_loss = (
                        outs["LogL"][0][0] + args.aux_scale * outs["MI_outbound"][0][0]
                    )
                    context_loss = (
                        outs["LogL"][1][0] + args.aux_scale * outs["MI_inbound"][1][0]
                    )
                else:
                    contrib_loss = (
                        outs["LogL"][0][0]
                        - args.aux_scale * outs["MI_inbound"][0][0]
                        + (
                            args.aux_scale * outs["MI_outbound"][0][0]
                            if not args.use_adaptive_scale
                            else 0
                        )
                    )
                    context_loss = (
                        outs["LogL"][1][0]
                        + (
                            args.aux_scale * outs["MI_inbound"][1][0]
                            if not args.use_adaptive_scale
                            else 0
                        )
                        - args.aux_scale * outs["MI_outbound"][1][0]
                    )
                loss = (contrib_loss + context_loss) / 2
                losses = [
                    outs["LogL"][0][0],
                    outs["MI_inbound"][0][0],
                    outs["MI_outbound"][0][0],
                    outs["LogL"][1][0],
                    outs["MI_inbound"][1][0],
                    outs["MI_outbound"][1][0],
                ]

            # multihead
            else:
                # contrib, context
                losses = [o[0] for o in outs]
                loss = sum(losses) / len(losses)

            check_nan(loss)

            # reporting logL only
            loss_tensor_log.append(
                (losses[0] if args.input_type == "all" else loss).detach()
            )

            global_step, grad_acc_step, loss_tensor_log = update_step(
                args,
                model,
                tokenizer,
                optimizer,
                loss,
                losses,
                eval_iter,
                global_step,
                grad_acc_step,
                loss_tensor_log,
            )

        else:
            # For compatibility of training loop
            if args.target_type != "both":
                input_texts, output_texts = ([input_texts], [output_texts])

            input_texts, output_texts = zip(*input_texts), zip(*output_texts)
            heads = ["contrib", "context"]
            losses = []

            # loop over the two targets
            for input_text, output_text, head in zip(input_texts, output_texts, heads):
                tok_input, tok_output, labels = tokenize_batch(
                    input_text, output_text, model, tokenizer, args
                )

                if args.distributed:
                    if (grad_acc_step == 0) or (
                        (grad_acc_step + 1) % args.gradient_accumulation_steps == 0
                    ):
                        model.require_forward_param_sync = True
                        model.require_backward_grad_sync = True
                    else:
                        model.require_forward_param_sync = False
                        model.require_backward_grad_sync = False

                # Auxiliary loss: informativeness
                if args.input_type == "all":
                    outs = model(
                        input_ids=[t["input_ids"] for t in tok_input],
                        attention_mask=[t["attention_mask"] for t in tok_input],
                        decoder_input_ids=shift_left(
                            tok_output["input_ids"], tokenizer.bos_token_id
                        ),
                        decoder_attention_mask=tok_output["attention_mask"],
                        lm_labels=labels,
                    )
                else:
                    outs = model(
                        input_ids=tok_input["input_ids"],
                        attention_mask=tok_input["attention_mask"],
                        decoder_input_ids=shift_left(
                            tok_output["input_ids"], tokenizer.bos_token_id
                        ),
                        decoder_attention_mask=tok_output["attention_mask"],
                        lm_labels=labels,
                    )

                if args.input_type == "all":
                    losses += outs[0]
                    if args.target_type == "contrib":
                        if args.use_informativeness:
                            coeff = [
                                1,
                                0,
                                args.aux_scale,
                            ]
                        else:
                            coeff = [
                                1,
                                -args.aux_scale,
                                args.aux_scale if not args.use_adaptive_scale else 0,
                            ]
                    elif args.target_type == "context":
                        if args.use_informativeness:
                            coeff = [
                                1,
                                args.aux_scale,
                                0,
                            ]
                        else:
                            coeff = [
                                1,
                                args.aux_scale if not args.use_adaptive_scale else 0,
                                -args.aux_scale,
                            ]

                    loss = sum(l * c for l, c in zip(outs[0], coeff))

                elif args.use_multi_head:
                    loss = outs[0 if head == "contrib" else 1][0]

                else:
                    loss = outs[0]

                check_nan(loss)

                loss_tensor_log.append(
                    (losses[0] if args.input_type == "all" else loss).detach()
                )

                global_step, grad_acc_step, loss_tensor_log = update_step(
                    args,
                    model,
                    tokenizer,
                    optimizer,
                    loss,
                    losses,
                    eval_iter,
                    global_step,
                    grad_acc_step,
                    loss_tensor_log,
                )


if __name__ == "__main__":
    args = get_args()

    set_seed(args)
    tb_writer = SummaryWriter(logdir=args.logdir)
    run(args)
