# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""Filter examples by certain conditions.

examples:
    python filter_papers.py \
        --parse-type both \
        --input-dir /export/share/hhayashi/data/repos/s2orc/gorc/metadata \
        --output-dir ./stats.pkl \

"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import flutes
from processor import Filter, LegacyFilter


def main(args) -> None:
    Proc = LegacyFilter if args.legacy else Filter

    if args.target == "paper":
        processing = Proc.filter_ids_complete  # gathers pid, in/out cite pids
    elif args.target == "citation":
        processing = Proc.filter_ids_text  # gathers pids

    if args.valid_citations is not None and args.valid_citations.exists():
        with args.valid_citations.open("rb") as f:
            d = pickle.load(f)
        dict_valid_citations = {k: True for _, pids in d.items() for k in pids}
        del d
    else:
        dict_valid_citations = {}

    files = [
        (f, dict_valid_citations, args.min_cite, args.max_cite, args.seed)
        for f in list(args.input_dir.glob("*"))
    ]

    with flutes.work_in_progress("Parallel"):
        total_results = defaultdict(list)
        with flutes.safe_pool(processes=args.njobs, state_class=Proc) as pool_stateful:
            for idx, _ in enumerate(
                pool_stateful.imap_unordered(processing, files, chunksize=10)
            ):
                flutes.log(f"Processed {(idx + 1)} files")
            with flutes.work_in_progress("Get states"):
                states = pool_stateful.get_states()
            for state in states:
                total_results.update(state.results)

    with args.output_file.open("wb") as f:
        # Dict[batchnum, List[obj]]
        pickle.dump(total_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Filter instances according to criteria.")
    parser.add_argument("--legacy", action="store_true", help="20190928")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--njobs", type=int, default=6)
    parser.add_argument("--target", type=str, choices=["paper", "citation"])
    parser.add_argument("--valid-citations", type=Path, default=None)
    parser.add_argument("--min-cite", type=int, default=5)
    parser.add_argument("--max-cite", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2564)

    args = parser.parse_args()
    main(args)
