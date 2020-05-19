# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List


def dump_tsv(data_dict: Dict[str, List[str]], output_path: Path):

    output_path.mkdir(parents=True, exist_ok=True)

    print("Split/#contrib/#other")
    for k, v in data_dict.items():
        labels = Counter()
        with (output_path / f"{k}.tsv").open("w") as f:
            for contrib, other in v:
                for c in contrib:
                    print(c, 1, file=f, sep="\t")
                for o in other:
                    print(o, 0, file=f, sep="\t")
                labels.update([1 for _ in contrib] + [0 for _ in other])
        print(f"{k}/{labels[1]}/{labels[0]}")


def bin_position(max_val):
    """ ` _ @ """
    symbol_map = {0: " `", 1: " _", 2: " @"}
    if max_val <= 3:
        return [symbol_map[i] for i in range(max_val)]

    first = max_val // 3
    second = 2 * first
    return [" `"] * first + [" _"] * (second - first) + [" @"] * (max_val - second)


def main(args):

    # load annotation -- random seed = 14, max_count = 1_000
    with args.annotation_file.open("r") as f:
        annot = []
        for line in f:
            line = line.strip().split(" ")
            if len(line) == 1:
                continue

            ids = tuple(int(i) for i in line[0].split("-"))
            contrib_indices = [int(i) for i in line[1].split(",")]
            annot.append((ids, contrib_indices))

    # load data file
    with args.data_file.open("r") as f:
        data = list(enumerate([json.loads(l) for l in f]))
        random.shuffle(data)
        data = data[: args.max_count]
        data = {i: d for i, d in data}

    texts = []  # contrib, other
    total_contrib = 0
    for (ids, indices) in annot:
        obj = data[ids[0]]
        if args.add_special_position_token:
            special_tokens = bin_position(len(obj["target"]))
        else:
            special_tokens = [""] * len(obj["target"])
        contrib = [
            s + special_tokens[i] for i, s in enumerate(obj["target"]) if i in indices
        ]
        other = [
            s + special_tokens[i]
            for i, s in enumerate(obj["target"])
            if i not in indices
        ]
        texts.append((contrib, other))
        total_contrib += len(indices)

    # make sentence-level instances, but keep same abstract in same split
    target_contrib_count = int(total_contrib * args.split_ratio)
    train_contrib = 0
    train_contrib_idx = -1
    for idx, (con, oth) in enumerate(texts):
        if train_contrib + len(con) > target_contrib_count:
            train_contrib_idx = idx
            break
        train_contrib += len(con)

    train, test = texts[:train_contrib_idx], texts[train_contrib_idx:]

    dump_tsv(dict(train=train, test=test), args.target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation-file", type=Path, help="Path to the annotation file."
    )
    parser.add_argument("--data-file", type=Path, help="Path to dataset file.")
    parser.add_argument(
        "--seed", type=int, help="Random seed when subsample annotations.", default=14
    )
    parser.add_argument(
        "--max-count", type=int, help="Number of examples to show.", default=1000
    )
    parser.add_argument("--target-dir", type=Path, help="Where to save the files.")
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--add-special-position-token", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    main(args)
