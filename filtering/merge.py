# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""merge.py: Merge 100 files into 1 file (useful for the old release of 20190928)."""
import argparse
import json
import pickle
from pathlib import Path

import flutes


class Worker(flutes.PoolState):
    def __init__(self):
        self.results = {}

    @flutes.exception_wrapper()
    def merge(self, file_):
        output_dir, file_ = file_
        """construct a dictionary of pid - (batchnum, line#)"""
        batchnum = int(file_.name.replace(".pkl", ""))
        for i in range(batchnum, batchnum + 100):
            p = file_.parent / file_.name.replace(f"{batchnum}", f"{i}")
            assert p.exists()

        gathered = []
        for i in range(batchnum, batchnum + 100):
            p = file_.parent / file_.name.replace(f"{batchnum}", f"{i}")
            with p.open("rb") as f:
                gathered += pickle.load(f)

        with (output_dir / f"{batchnum}_{batchnum+100}.pkl").open("wb") as f:
            pickle.dump(gathered, f)


def main(args) -> None:

    files = [
        (args.output_dir, args.input_dir / f"{i}.pkl") for i in range(0, 10000, 100)
    ]

    total_map = {}
    with flutes.work_in_progress("Parallel"):
        with flutes.safe_pool(
            processes=args.njobs, state_class=Worker
        ) as pool_stateful:
            for idx, _ in enumerate(
                pool_stateful.imap_unordered(Worker.merge, files, chunksize=1)
            ):
                flutes.log(f"Processed {(idx + 1)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--njobs", type=int, default=6)
    args = parser.parse_args()
    main(args)
