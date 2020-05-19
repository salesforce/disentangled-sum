# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""calc_metrics.py: script to calculate all the scores given appropriate outputs.
The output file are created via eval scripts in scripts/

For each measure, following files must be specified:
    Relevance:
        --contrib-rel:
            (s_con, y_con) - Eval results between contribution summary and contribution reference.
        --context-rel:
            (s_ctx, y_ctx) - Eval results between context summary and contex reference.

    Purity:
        --contrib-rel:
            (s_con, y_con) - Eval results between contribution summary and contribution reference.
        --context-rel:
            (s_ctx, y_ctx) - Eval results between context summary and contex reference.
        --contrib-cross:
            (s_con, y_ctx) - Eval results between contribution summary and context reference.
        --context-cross:
            (s_ctx, y_con) - Eval results between context summary and contribution reference.

    Disentanglement:
        --contrib-context:
            (s_con, s_ctx) - Eval results between contribution summary and context summary.

Example:
    Relevance:
        python calc_metrics.py \
            --contrib-rel /path/to/s_con_y_con.jsonl \
            --context-rel /path/to/s_ctx_y_ctx.jsonl \
            --measure relevance

    Purity:
        python calc_metrics.py \
            --contrib-rel /path/to/s_con_y_con.jsonl \
            --context-rel /path/to/s_ctx_y_ctx.jsonl \
            --contrib-cross /path/to/s_con_y_ctx.jsonl \
            --context-cross /path/to/s_ctx_y_con.jsonl \
            --measure purity

    Disentanglement:
        python calc_metrics.py \
            --contrib-context /path/to/s_con_s_ctx.jsonl \
            --measure disentanglement

"""


import argparse
import json
from pathlib import Path

import numpy as np
from tabulate import tabulate

ROUGE_KEYS = [
    "rouge_1_f_score",
    "rouge_2_f_score",
    "rouge_3_f_score",
    "rouge_4_f_score",
    "rouge_l_f_score",
    "rouge_w_1.2_f_score",
    "rouge_s*_f_score",
    "rouge_su*_f_score",
]


def nouveau(con, cros, rouge=1):
    # c0 + c1 * R(Contrib) + c2 * R(Other)
    # Cf. Table 2 @ https://www.aclweb.org/anthology/J11-1001.pdf
    NR_COEFF = np.array([[-0.0271, 13.4227, -7.355], [0.9126, 21.1556, -5.4536]])
    contrib = con["rouge"][f"rouge_{rouge}_f_score"]
    context = cros["rouge"][f"rouge_{rouge}_f_score"]
    nr = NR_COEFF[rouge - 1, :] @ np.array([1, contrib, context])
    return nr


def extract_scores(path: Path):

    res = {}
    with path.open("r") as f:
        for line in f:
            obj = json.loads(line)
            pid = obj["paper_id"]
            res[pid] = dict(rouge=obj["rouge"], bert_score=obj["bert_score"])
    return res


def format_2digit(arr):
    arr = arr * 100
    return [f"{v:.2f}" for v in arr]


def main(args):
    if args.measure == "relevance":
        assert args.contrib_rel.exists() or args.context_rel.exists()

        pid2scores = {}
        pid2scores["contrib"] = extract_scores(args.contrib_rel)
        pid2scores["context"] = extract_scores(args.context_rel)

        for k, v in pid2scores.items():
            scores = []
            for pid, score in v.items():
                scores.append(
                    [score["rouge"][k] for k in ROUGE_KEYS]
                    + [score["bert_score"]["bert_score_f1"]]
                )
            mean = format_2digit(np.mean(scores, axis=0))
            print(args.measure, k)
            print(tabulate(mean, headers=ROUGE_KEYS + ["bert_score_f1"]))

    elif args.measure == "purity":
        assert args.contrib_rel.exists() or args.context_rel.exists()
        assert args.contrib_cross.exists() and args.context_cross.exists()

        pid2scores = {}
        pid2scores["contrib"] = extract_scores(args.contrib_rel)
        pid2scores["context"] = extract_scores(args.context_rel)
        pid2scores["cross"] = extract_scores(args.contrib_cross)
        pid2scores["revcross"] = extract_scores(args.context_cross)

        contrib_nr, context_nr = [], []
        for pid in pid2scores["contrib"].keys():
            nr1 = nouveau(pid2scores["contrib"][pid], pid2scores["cross"][pid], rouge=1)
            nr2 = nouveau(pid2scores["contrib"][pid], pid2scores["cross"][pid], rouge=2)
            rev_nr1 = nouveau(
                pid2scores["context"][pid], pid2scores["revcross"][pid], rouge=1
            )
            rev_nr2 = nouveau(
                pid2scores["context"][pid], pid2scores["revcross"][pid], rouge=2
            )
            contrib_nr.append([nr1, nr2])
            context_nr.append([rev_nr1, rev_nr2])

        contrib_nr = np.mean(contrib_nr, axis=0)
        context_nr = np.mean(contrib_nr, axis=0)
        purity_score = format_2digit(contrib_nr + context_nr / 2)
        print(args.measure)
        print(tabulate(purity_score, headers=["P1", "P2"]))

    elif args.measure == "disentanglement":
        assert args.contrib_context.exists()
        pid2scores = extract_scores(args.contrib_context)
        scores = []
        for pid, score in pid2scores.items():
            scores.append(
                [score["rouge"][k] for k in ROUGE_KEYS]
                + [score["bert_score"]["bert_score_f1"]]
            )
        mean = format_2digit(1 - np.mean(scores, axis=0))
        print(args.measure)
        print(
            tabulate(
                mean, headers=["D1", "D2", "D3", "D4", "DL", "DW", "DS*", "DSU", "DBS"]
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Calculate metrics using the individual automatic measurse."
    )
    # For Relevance
    parser.add_argument(
        "--contrib-rel", type=Path, help="Path to the eval file between (s_con, y_con)."
    )
    parser.add_argument(
        "--context-rel", type=Path, help="Path to the eval file between (s_ctx, y_ctx)."
    )
    # For Purity
    parser.add_argument(
        "--contrib-cross",
        type=Path,
        help="Path to the eval file between (s_con, y_ctx).",
    )
    parser.add_argument(
        "--context-cross",
        type=Path,
        help="Path to the eval file between (s_ctx, y_con).",
    )
    # For Disentanglement
    parser.add_argument(
        "--contrib-context",
        type=Path,
        help="Path to the eval file between (s_con, y_ctx).",
    )
    parser.add_argument(
        "--measure", type=str, choices=["relevance", "purity", "disentanglement"]
    )

    args = parser.parse_args()

    main(args)
