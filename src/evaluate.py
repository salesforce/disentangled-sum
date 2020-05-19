# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""evaluate.py: takes arbitrary decoded files and return ROUGE and BERTScore measures
for each examples.

Example usage 1: Comparing references from exp1 and decoded texts from exp2.

    python evaluate.py \
        --exps /path/to/exp1 /path/to/exp2  \
        --summary-type reference decoded \
        --output-file exp1-ref_exp2-dec.jsonl

Example usage 2: Comparing references and decoded texts within exp1.

    python evaluate.py \
        --exps /path/to/exp1 \
        --output-file exp1-ref_exp1-dec.jsonl

"""
import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from summ_eval.bert_score_metric import BertScoreMetric
from summ_eval.rouge_metric import RougeMetric


def wrap_results(args, pids, texts, rouges, bertscores, reverse=False):
    wrapped = []
    for pid, text, rouge, bertscore in zip(pids, texts, rouges, bertscores):
        wrapped.append(
            {
                "paper_id": pid,
                args.summary_type[0]: text[0],
                args.summary_type[1]: text[1],
                "rouge": rouge["rouge"],
                "bert_score": bertscore,
            }
        )
    return wrapped


def replace_html(text):
    return re.sub("</?([a-zA-Z]{,4})>", "[\\1]", text)


def batch_data(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def main(args):
    try:
        assert len(args.exps) == len(args.summary_type) == 2
    except AssertionError:
        # do a single file evaluation.
        assert len(args.exps) > 0, "At least one experiment must be specified."
        print(f"Performing single-file evaluation on {args.exps[0]}")
        args.exps = [args.exps[0], args.exps[0]]
        args.summary_type = ['reference', 'decoded']

    batch_size = 4096

    inputs = []
    for exp, key in zip(args.exps, args.summary_type):
        ipfiles = exp.glob("eval.jsonl*")
        data = {}
        # legacy compatibility to existing code
        if key == "reference":
            key = "gold"
        for ipfile in ipfiles:
            with ipfile.open("r") as fd:
                d = [json.loads(line) for line in fd]
            data.update({ex["paper_id"]: ex[key] for ex in d})
        inputs.append(data)

    # rearrange to pid-to-(system_1, system_2) dict.
    data = list({pid: (inputs[0][pid], inputs[1][pid]) for pid in inputs[0]}.items())

    rouge_metric = RougeMetric()
    bertscore_metric = BertScoreMetric()

    results = []
    for batch in tqdm(
        batch_data(data, batch_size), total=len(data) / batch_size, ncols=80, ascii=True
    ):
        batch_pids, batch_texts = zip(*batch)
        clean_refs = [replace_html(ex[0]) for ex in batch_texts]
        clean_preds = [replace_html(ex[1]) for ex in batch_texts]

        # assign zero-scores for empty reference cases
        empty_ref_indices = [i for i, x in enumerate(clean_refs) if x == ""]
        if len(empty_ref_indices) > 0:
            wrapped = []
            for i in empty_ref_indices:
                wrapped.append(
                    {
                        "paper_id": batch_pids[i],
                        args.summary_type[0]: batch_texts[i][0],
                        args.summary_type[1]: batch_texts[i][1],
                        "rouge": {
                            "rouge_1_f_score": 0,
                            "rouge_2_f_score": 0,
                            "rouge_3_f_score": 0,
                            "rouge_4_f_score": 0,
                            "rouge_l_f_score": 0,
                            "rouge_w_1.2_f_score": 0,
                            "rouge_s*_f_score": 0,
                            "rouge_su*_f_score": 0,
                        },
                        "bert_score": dict(bert_score_f1=0),
                    }
                )
            results.extend(wrapped)
            batch_pids = [
                ex for i, ex in enumerate(batch_pids) if i not in empty_ref_indices
            ]
            clean_refs = [
                ex for i, ex in enumerate(clean_refs) if i not in empty_ref_indices
            ]
            clean_preds = [
                ex for i, ex in enumerate(clean_preds) if i not in empty_ref_indices
            ]

        try:
            rouges = rouge_metric.evaluate_batch(
                clean_preds, clean_refs, aggregate=False
            )
            bertscores = bertscore_metric.evaluate_batch(
                clean_preds, clean_refs, aggregate=False
            )
        except:
            import pdb

            pdb.set_trace()
        results.extend(wrap_results(args, batch_pids, batch_texts, rouges, bertscores))

    with args.output_file.open("w") as fd:
        for example in results:
            print(json.dumps(example), file=fd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Calculate ROUGE and BERTScore between any two outputs."
    )
    parser.add_argument("--exps", nargs="+", type=Path)
    parser.add_argument(
        "--summary-type",
        nargs="+",
        type=str,
        help="Decoded, or Gold, for each input file.",
        choices=["decoded", "reference"],
    )
    parser.add_argument("--output-file", type=Path)

    args = parser.parse_args()
    main(args)
