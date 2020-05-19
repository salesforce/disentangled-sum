# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""create_pid_locator.py: generates the map between paper id and the corresponding
file number and the line number to locate the paper.
"""
import argparse
import pickle
from pathlib import Path

import flutes
from processor import Filter, LegacyFilter
from tqdm import tqdm


def main(args) -> None:

    pids_with_text = set()
    if args.valid_pids and args.valid_pids.exists():
        pids_with_text = set()
        with args.valid_pids.open("rb") as f:
            data = pickle.load(f)
            buf = set()
            if args.mode == "citation":
                for k, d in tqdm(enumerate(data.values()), ncols=88, ascii=True):
                    buf = buf.union(set([pid for i in d for pid in i[1] + i[2]]))
                    if k % 500 == 0:
                        pids_with_text = pids_with_text.union(buf)
                        buf = set()
            elif args.mode == "paper":
                for k, d in tqdm(data.items(), ncols=88, ascii=True):
                    buf = buf.union(set([i[0] for i in d]))
                    if k % 500 == 0:
                        pids_with_text = pids_with_text.union(buf)
                        buf = set()

            # remaining one
            pids_with_text = pids_with_text.union(buf)

    flutes.log(f"# of valid pids to consider: {len(pids_with_text)}")

    if args.legacy:
        # glob takes more time than this?
        files = (
            (args.input_dir / f"{i}.jsonl.gz", pids_with_text) for i in range(10000)
        )
        Proc = LegacyFilter
    else:
        files = (
            (args.input_dir / f"pdf_parses_{i}.jsonl.gz", pids_with_text)
            for i in range(100)
        )
        Proc = Filter

    with flutes.work_in_progress("Parallel"):
        total_map = {}
        with flutes.safe_pool(processes=args.njobs, state_class=Proc) as pool_stateful:
            for idx, _ in enumerate(
                pool_stateful.imap_unordered(Proc.make_map, files, chunksize=10)
            ):
                flutes.log(f"Processed {(idx + 1)} files")

            with flutes.work_in_progress("Get states"):
                states = pool_stateful.get_states()
            for state in states:
                # TODO: Incorporate incite number
                total_map.update(state.results)

        flutes.log(f"Total map size: {len(total_map)}")

    with args.output.open("w") as f:
        for k, v in total_map.items():
            print(k, v[0], v[1], sep="\t", file=f)

        flutes.log(f"Dumped to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument(
        "--valid-pids", type=Path, help="Filtered list of pids, if any."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--njobs", type=int, default=6)
    parser.add_argument(
        "--mode", type=str, required=True, choices=["paper", "citation"]
    )
    parser.add_argument("--legacy", action="store_true", help="20190928")
    args = parser.parse_args()
    main(args)
