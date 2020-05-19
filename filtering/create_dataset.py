# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""create_dataset.py: generates torch dataset from preprocessed s2orc dumps."""

import argparse
import json
import pickle
import random
import sys
from itertools import product
from pathlib import Path
from typing import Iterable, List, Tuple

import flutes
import numpy as np
from nltk import sent_tokenize  # requires punkt
from rouge_score import rouge_scorer
from tqdm import tqdm


class Processor(flutes.PoolState):
    def __init__(self):
        self.results = {}

    @flutes.exception_wrapper()
    def process_pkl(self, file_: Tuple[Path, Path]):
        file_, out_file = file_

        with file_.open("rb") as f:
            records = pickle.load(f)
            self.results[out_file.with_suffix("").name] = len(records)

        total = []
        for rec in records:
            inbound, outbound, target_paper = rec
            if args.legacy:
                if target_paper["metadata"]["abstract"] is not None:
                    abstract = target_paper["metadata"]["abstract"]
                elif len(target_paper["grobid_parse"]["abstract"]) > 0:
                    abstract = target_paper["grobid_parse"]["abstract"][0]["text"]
                else:
                    continue
                abstract: List[str] = sent_tokenize(abstract)
            else:
                abstract: List[str] = sent_tokenize(target_paper["abstract"][0]["text"])

            # format to correct type
            # citation context: List of (section, sentences)
            inbound_cc: List[Tuple[str, str]] = [
                s for pid, sents in inbound if sents is not None for s in sents
            ]
            outbound_cc: List[Tuple[str, str]] = [
                s for pid, sents, abs_ in outbound if len(sents) > 0 for s in sents
            ]
            outbound_abs: List[str] = [
                s
                for pid, sents, abs_ in outbound
                if abs_ is not None
                for s in sent_tokenize(abs_)
            ]
            if len(outbound) == 0 or len(inbound) == 0:
                continue

            # obtain indices that look like contribution
            # THIS SHOULD NOW BE POPULATED BY THE CLASSIFIER
            # contrib_indices = similarity(abstract, inbound, outbound, metric="rougeL")

            rec = dict(
                inbound=inbound_cc,
                outbound=outbound_cc,
                ouutbound_abs=outbound_abs,
                paper=target_paper,
                target=abstract,
                contrib_indices={},
            )
            total.append(rec)

        with out_file.open("w") as f:
            for r in total:
                print(json.dumps(r, ensure_ascii=False), file=f)


def progress(iterable: Iterable, **kwargs) -> Iterable:
    """Better defaults"""
    return tqdm(iterable, ncols=88, ascii=True, **kwargs)


def make_splits(seq, train_ratio: float):
    """split valid and test set from the remaining data."""
    tr_idx = int(train_ratio * len(seq))
    vl_idx = int((1 - train_ratio) * len(seq) / 2)
    return seq[:tr_idx], seq[tr_idx : tr_idx + vl_idx], seq[tr_idx + vl_idx :]


def make_splits_indices(total: int, train_ratio: float):
    """split valid and test set from the remaining data."""
    tr_idx = int(train_ratio * total)
    vl_idx = int((1 - train_ratio) * total / 2)
    arr = list(range(total))
    random.shuffle(arr)
    return arr[:tr_idx], arr[tr_idx : tr_idx + vl_idx], arr[tr_idx + vl_idx :]


# TODO: merge this function with analysis branch's
def similarity(
    targets: List[str],
    inbound: List[str],
    outbound: List[str],
    metric: str = "rougeL",
    batch_size: int = 6000,
):
    """For each target sentence, calculate similarity scores against
    inbound/outbound texts. Use the scores to determine the membership to the
    contribution summary.

    ROUGE-L, where aggregation done for inbound: max, outbound: mean, achieves the closest
    agreement scores to human annotators.

    :param targets: Target abstract, sentence- and word-tokenized.
    :param inbound: Inbound texts.
    :param outbound: Outbound texts.
    :param metric: Either of "rouge[1,2,L]" or "bertscore". Used to calculate
        similarity score of a pair.
    :param batch_size: How much chunk of sentences to compute scores at once.
    """

    if metric.startswith("rouge"):
        assert metric in ["rouge1", "rouge2", "rougeL"]
        scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)

        def calculate(candidates, references):
            F1 = []
            for c, r in zip(candidates, references):
                F1.append(scorer.score(c, r)[metric].fmeasure)
            return F1

    elif metric == "bertscore":

        def calculate(candidates, references):
            _, _, F1 = bscore(candidates, references, lang="en")
            return F1.cpu().tolist()

    in_n_out = inbound + outbound
    split = len(inbound)

    # pre-calculate everything
    scores = []
    # all combinations
    cands, refs = list(zip(*list(product(targets, in_n_out))))
    n_batches = len(refs) // batch_size
    for b in range(n_batches + (1 if len(refs) % batch_size > 0 else 0)):
        scores += calculate(
            cands[b * batch_size : min((b + 1) * batch_size, len(refs))],
            refs[b * batch_size : min((b + 1) * batch_size, len(refs))],
        )

    selected = []
    for sidx, _ in enumerate(targets):
        score_chunk = scores[sidx * len(in_n_out) : (sidx + 1) * len(in_n_out)]
        in_sim, out_sim = (
            np.max(score_chunk[:split]),
            np.mean(score_chunk[split:]),
        )

        if in_sim > out_sim:
            selected.append(sidx)
        elif any([word in targets[sidx].lower() for word in ["we "]]):
            selected.append(sidx)
        # elif any([word in targets[sidx].lower() for word in ["show", "propose", "present"]]):
        #     selected.append(sidx)
        # elif any([word in targets[sidx].lower() for word in ["developed"]]):
        #     selected.append(sidx)

    return selected


def main(args):

    random.seed(args.seed)

    if (not args.output_dir.exists()) or args.overwrite:
        args.output_dir.mkdir(exist_ok=True, parents=True)
    else:
        print(f"Directory {args.output_dir} already exists.")
        sys.exit(0)

    tr_indices, vl_indices, ts_indices = make_splits_indices(100, args.split_ratio)
    with (args.output_dir / "file_map.txt").open("w") as f:
        print("train", file=f)
        for i in tr_indices:
            print(f"{i} ", file=f, end="")
        print(file=f)
        print("valid", file=f)
        for i in vl_indices:
            print(f"{i} ", file=f, end="")
        print(file=f)
        print("test")
        for i in ts_indices:
            print(f"{i} ", file=f, end="")
        print(file=f)

    # in/out file pairs
    files = (
        [
            (
                (args.dump_dir / f"{i}.pkl"),
                (args.output_dir / f"train_{new_idx:02}.jsonl"),
            )
            for new_idx, i in enumerate(tr_indices)
        ]
        + [
            (
                (args.dump_dir / f"{i}.pkl"),
                (args.output_dir / f"valid_{new_idx:02}.jsonl"),
            )
            for new_idx, i in enumerate(vl_indices)
        ]
        + [
            (
                (args.dump_dir / f"{i}.pkl"),
                (args.output_dir / f"test_{new_idx:02}.jsonl"),
            )
            for new_idx, i in enumerate(ts_indices)
        ]
    )

    # files = sorted(
    #     list((args.dump_dir).glob("*.pkl")), key=lambda x: int(x.with_suffix("").name)
    # )

    total = {}
    with flutes.safe_pool(processes=args.njobs, state_class=Processor) as pool:
        for idx, _ in enumerate(
            pool.imap_unordered(Processor.process_pkl, files, chunksize=1)
        ):
            flutes.log(f"Processed {(idx + 1)} files")

        states = pool.get_states()
        for state in states:
            total.update(state.results)
    print(f"Train: {sum(v for k, v in total.items() if k.startswith('train'))}")
    print(f"Valid: {sum(v for k, v in total.items() if k.startswith('valid'))}")
    print(f"Test:  {sum(v for k, v in total.items() if k.startswith('test'))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1726)
    parser.add_argument(
        "--split-ratio", type=float, default=0.8, help="ratio of training set."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--njobs", type=int, default=6)
    parser.add_argument("--legacy", action="store_true", help="TODO")
    from IPython.core import ultratb
    import sys

    sys.excepthook = ultratb.FormattedTB(
        mode="Context", color_scheme="Linux", call_pdb=1
    )
    args = parser.parse_args()
    main(args)
