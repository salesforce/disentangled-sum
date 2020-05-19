# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""script to postprocess the contribution density on a single file. Optionally
visualize.
"""
import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

USE_SCIENCE_PLOTS = False
try:
    plt.style.use(["science"])
    USE_SCIENCE_PLOTS = True
except:
    print("Install the matplotlib style with `pip install scienceplots`.")


def transform(s):
    res = []
    total = sum([int(i[1]) for i in s])
    cumm = 0
    for i in s:
        if len(i) > 2:
            res += [j + cumm for j in map(int, i[2].split(" "))]
        cumm += int(i[1])
    return (res, total)


def postprocess(file_):
    para = []
    with open(file_, "r") as f:
        for line in f:
            para.append(line.strip().split("\t"))

    sep = []
    with open(file_.replace("classified", "separate"), "r") as f:
        sep = f.read().strip().split("\n")

    separated = []
    for i in range(len(sep) - 1):
        begin = int(sep[i])
        end = int(sep[i + 1])
        separated.append(para[begin:end])

    trans = [transform(s) for s in separated]
    normalized = [
        [(val / t[1], 1 if val in t[0] else 0) for val in range(t[1])] for t in trans
    ]
    return normalized


def visualize(heatmap):
    total_denom = []
    total_counts = []
    for n in heatmap:
        # total counts of sentences in each bin
        denom = [
            len([i for i, _ in n if i < 0.1 * j and i > 0.1 * (j - 1)])
            for j in range(1, 11)
        ]
        # total counts of CONTRIBUTION sentences in each bin
        counts = [
            len([i for i, con in n if i < 0.1 * j and i > 0.1 * (j - 1) and con == 1])
            for j in range(1, 11)
        ]
        total_counts.append(counts)
        total_denom.append(denom)

    total_denom = np.sum(total_denom, axis=0)
    total_counts = np.sum(total_counts, axis=0)
    ratio = total_counts / total_denom

    plt.bar(np.arange(10) + 0.5, ratio * 100, alpha=1)
    plt.xticks(np.arange(10) + 0.5, [10 * i for i in range(10)])
    percentage = "\%" if USE_SCIENCE_PLOTS else "%"
    plt.xlabel(f"Position in a paper [{percentage}]")
    plt.ylabel(f"Percentage of\n contribution-labeled sentences. [{percentage}]")
    plt.savefig("density.pdf", bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Take soem stats on classified sentences.")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--prefix", type=str, help="Data path prefix.")
    args = parser.parse_args()

    all_files = glob.glob(os.path.join(args.prefix, "classified_indices*.txt"))
    all_normalized = []
    for f in all_files:
        all_normalized += postprocess(f)

    with open("normalized_heatmap.pkl", "wb") as f:
        pickle.dump(all_normalized, f)

    if args.visualize:
        visualize(all_normalized)
