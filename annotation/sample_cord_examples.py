# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

"""get_cord_examples.py: Extract the CORD paper parses from 2020 and save the object
in jsonl format. The script randomly extracts papers.

Usage:
    python get_cord_examples.py --cord-dir <Unzipped directory> --total K --seed N

"""
import argparse
import csv
import json
import random
from pathlib import Path

from nltk import sent_tokenize


def main(args):

    metadata_file = args.cord_dir / "metadata.csv"
    parse_dir = args.cord_dir / "document_parses" / "pdf_json"
    papers_from_2020 = []

    with metadata_file.open("r") as f:
        lines = f.read().strip().split("\n")

    for l in csv.reader(
        lines,
        quotechar='"',
        delimiter=",",
        quoting=csv.QUOTE_ALL,
        skipinitialspace=True,
    ):
        if l[9].startswith("2020"):
            papers_from_2020.append(l)

    random.shuffle(papers_from_2020)

    count = 0
    target_file = open("cord_2020_papers.jsonl", "w")
    for p in papers_from_2020:
        sha = p[1]
        if (parse_dir / f"{sha}.json").exists():
            with (parse_dir / f"{sha}.json").open("r") as f:
                paper = json.load(f)

            # additionally tokenize the target
            try:
                abstract = paper["abstract"][0]["text"]
            except IndexError:
                continue

            sents = sent_tokenize(abstract)
            # necessary fields for compatibility
            paper = dict(
                paper_id=paper["paper_id"],
                grobid_parse=dict(body_text=paper["body_text"]),
            )
            obj = dict(target=sents, paper=paper)
            print(json.dumps(obj), file=target_file)
            count += 1

        if count >= args.total:
            break

    target_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pick CORD examples randomly.")
    parser.add_argument("--cord-dir", type=Path, required=True)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--total", type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    main(args)
