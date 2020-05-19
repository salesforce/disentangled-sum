# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""construct_triples.py: prepares (inbound info, outbound info, target abstract)
triples. This is then used for oracle ROUGE evaluation.

Args:
    paper-id-data: output of filter_instances


"""
import argparse
import functools
import pickle
from collections import defaultdict
from pathlib import Path

import flutes
from processor import Filter, LegacyFilter
from tqdm import tqdm
from utils import progress


def main(args) -> None:

    assert len(args.slice) == 2
    if args.legacy:
        file_format = "{}.jsonl.gz"
        Proc = LegacyFilter
        file_total = 10000
    else:
        file_format = "pdf_parses_{}.jsonl.gz"
        Proc = Filter
        file_total = 100

    args.output_dir.mkdir(exist_ok=True, parents=True)

    # pid-to-citation
    with args.paper_id_data.open("rb") as f:
        # batchnum -> List[Tuple[pid, _, List[in], List[out]]]
        paper_ids = pickle.load(f)

    target_valid_paper_lines = {i: [] for i in range(file_total)}
    #  which line numbers should I care when reading the paper contents?
    with args.target_pid_location_dict.open("r") as f:
        for line in f:
            pid, bnum, lnum = line.strip().split("\t")
            target_valid_paper_lines[int(bnum)].append(int(lnum))
    for k, v in target_valid_paper_lines.items():
        target_valid_paper_lines[k] = sorted(v)

    flutes.log("Done loading target_valid_paper_lines.")

    if Path("per_file_citation_pid2loc_20190928_min5max20.pkl").exists():
        with open("per_file_citation_pid2loc_20190928_min5max20.pkl", "rb") as f:
            per_file_citation_pid2loc = pickle.load(f)
        flutes.log("Found and loaded a saved dump of per_file_citation_pid2loc.")
    else:
        required_paper_ids = {
            int(k): [i + o for _, i, o in v] for k, v in paper_ids.items()
        }
        for k, v in progress(required_paper_ids.items()):
            ids = []
            for nums in v:
                ids += nums
            required_paper_ids[k] = set(ids)

        flutes.log("Done loading required paper ids.")

        # pid-to-location; only load those we need
        # pre-computed list of valid (citation) papers around the target papers

        # 2. where are the citation papers located?
        citation_pid2loc = dict()
        with args.citation_pid_location_dict.open("r") as f:
            for line in f:
                pid, bnum, lnum = line.strip().split("\t")
                citation_pid2loc[pid] = (int(bnum), int(lnum))

        flutes.log("Done loading citation_pid2loc.")

        # further split into per-file, so that parallel jobs won't be looking at the same dict
        per_file_citation_pid2loc = {k: {} for k in range(file_total)}
        for trg_bnum in range(file_total):
            for id_ in required_paper_ids[trg_bnum]:
                per_file_citation_pid2loc[trg_bnum][id_] = citation_pid2loc.get(
                    id_, None
                )

        with open("per_file_citation_pid2loc_20190928_min5max20.pkl", "wb") as f:
            pickle.dump(per_file_citation_pid2loc, f)
            flutes.log("Saved citation_pid2loc.")

        del citation_pid2loc

    files = (
        (
            (args.input_dir / file_format.format(i)),
            {k: (ib, ob) for k, ib, ob in paper_ids[str(i)]},
            args.output_dir,
            target_valid_paper_lines[i],
            per_file_citation_pid2loc[i],
        )
        for i in range(args.slice[0], args.slice[1])
    )

    # non-parallel, smaller subset for debugging purpose.
    if args.debug:
        flutes.log("Debug mode.")
        worker = Proc()
        files = list(files)[:1]
        for f in tqdm(files, ascii=True, ncols=80):
            worker.gather_abstracts(f)

    else:
        with flutes.work_in_progress("Parallel"):
            with flutes.safe_pool(
                processes=args.njobs, state_class=Proc
            ) as pool_stateful:
                for idx, _ in enumerate(
                    pool_stateful.imap_unordered(
                        Proc.gather_abstracts, files, chunksize=10
                    )
                ):
                    flutes.log(f"Processed {(idx + 1)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-id-data", required=True, type=Path)
    parser.add_argument("--target-pid-location-dict", required=True, type=Path)
    parser.add_argument("--citation-pid-location-dict", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--njobs", type=int, default=10)
    parser.add_argument("--legacy", action="store_true", help="TODO")
    parser.add_argument("--slice", nargs="+", type=int, help="TODO")

    args = parser.parse_args()
    main(args)
