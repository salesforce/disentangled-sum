# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""processor.py: collections of functions to execute for each metadta/paper.
Individual scripts in this directory use methods defined other Filter class.
"""
import bisect
import pickle
import random
import re
from ast import literal_eval as eval
from pathlib import Path

import flutes
import mgzip as gzip
import numpy as np
import ujson as json
from nltk import sent_tokenize

PID_PAT = re.compile(r"\"paper_id\":\s+\"(\d+)\"")


def load_bg_info_legacy(target_pid, obj, content="cite_context"):

    if obj is None:
        flutes.log(f"{target_pid} not found.")
        return None

    if content == "abstract":
        if obj["metadata"]["abstract"] is not None:
            return obj["metadata"]["abstract"]
        elif len(obj["grobid_parse"]["abstract"]) > 0:
            return obj["grobid_parse"]["abstract"][0]["text"]
        return obj["metadata"]["abstract"]

    elif content == "cite_context":
        parse_data = obj["grobid_parse"]
        if parse_data is None:
            flutes.log(f"parse not found.")
            return None

        try:
            # find the right BIBREFX
            bibdict = {
                v["links"]: k
                for k, v in parse_data["bib_entries"].items()
                if v["links"] is not None
            }
            bibref = bibdict[target_pid]

        except KeyError:
            # data bug
            flutes.log(f"links not found.")
            return None

        match = []
        for block in parse_data["body_text"]:
            sec = block["section"]
            spans = [span for span in block["cite_spans"] if span["ref_id"] == bibref]
            if len(spans) == 0:
                continue

            # look for the right sent idx where the span belongs to.
            sents = sent_tokenize(block["text"].replace("al.", "al@"))
            # cumulative sum + stripped (hopefully just one) spaces for each sent.
            sent_start_pos = np.cumsum([len(s) for s in sents]) + np.arange(
                len(sents), dtype=np.int
            )
            for sp in spans:
                # TODO: try to pick up section information here once merged!
                sent_idx = bisect.bisect_left(sent_start_pos, sp["start"])
                if sent_idx == len(sents):  # de-tokenize error
                    sent_idx = len(sents) - 1
                match.append((sec, sents[sent_idx].replace("al@", "al.")))

        return match


def load_bg_info(target_pid, obj, content="cite_context"):

    if obj is None:
        flutes.log(f"{target_pid} not found.")
        return None

    if content == "abstract":
        return obj["abstract"][0]["text"]

    elif content == "cite_context":
        try:
            # find the right BIBREFX
            bibdict = {
                v["link"]: k
                for k, v in obj["bib_entries"].items()
                if v["link"] is not None
            }
            bibref = bibdict[target_pid]

        except KeyError:
            # data bug
            flutes.log(f"links not found.")
            return None

        if obj["body_text"] is None:
            # data bug
            flutes.log(f"{bnum} - {idx} - pid: {obj['paper_id']} body_text not found.")
            return None

        match = []
        for block in obj["body_text"]:
            sec = block["section"]
            spans = [span for span in block["cite_spans"] if span["ref_id"] == bibref]
            if len(spans) == 0:
                continue

            # look for the right sent idx where the span belongs to.
            sents = sent_tokenize(block["text"].replace("al.", "al@"))
            # cumulative sum + stripped (hopefully just one) spaces for each sent.
            sent_start_pos = np.cumsum([len(s) for s in sents]) + np.arange(
                len(sents), dtype=np.int
            )
            for sp in spans:
                # TODO: try to pick up section information here once merged!
                sent_idx = bisect.bisect_left(sent_start_pos, sp["start"])
                if sent_idx == len(sents):  # de-tokenize error
                    sent_idx = len(sents) - 1
                match.append((sec, sents[sent_idx].replace("al@", "al.")))

        return match


class LegacyFilter(flutes.PoolState):
    def __init__(self):
        self.results = {}

    @staticmethod
    def validate_grobid(obj):
        """
        (7, 'has_pdf'),
        (8, 'has_grobid'),
        (9, 'has_grobid_text'),
        (17, 'grobid_num_abs_para'),
        (18, 'grobid_num_body_para'),
        (19, 'grobid_num_sections'),
        (20, 'grobid_abs_cite_spans'),
        (21, 'grobid_body_cite_spans'),
        (22, 'grobid_num_bib_entries'),
        (23, 'grobid_num_linked_bibs'),
        (24, 'grobid_bib_links'),
        """
        # has valid text parse
        if obj[8] == "True" and obj[9] == "True":
            return "0" not in obj[21:24] + [obj[18]]

        return False

    @flutes.exception_wrapper()
    def filter_ids_complete(self, file_: Path):
        """go over the metadata and get the in/out-bound citation data for each VALID
        paper. Validity is determined by whether the paper has a proper grobid parse.
        This method accumulates data into the state, as it won't take so much space.
        Used by filtering/filter_ids.py.
        """
        file_, valid_pids, min_cite, max_cite, seed = file_
        # make sure
        random.seed(seed)

        with file_.open("r") as f:
            # first line in metadata is the header
            fname = file_.with_suffix("").name
            next(f)
            self.results[fname] = []
            for line in f:
                obj = line.strip().split("\t")
                # has_outbound + has_inbound
                if (
                    obj[-2] != "[]"
                    and obj[-1] != "[]"
                    and LegacyFilter.validate_grobid(obj)
                ):
                    ob, ib = eval(obj[-2]), eval(obj[-1])
                    # filter to valid ones
                    vob = [pid for pid in ob if valid_pids.get(pid, False)]
                    vib = [pid for pid in ib if valid_pids.get(pid, False)]

                    if len(vob) >= min_cite and len(vib) >= min_cite:
                        random.shuffle(vob)
                        random.shuffle(vib)
                        self.results[fname].append(
                            (obj[1], vob[:max_cite], vib[:max_cite])
                        )

    @flutes.exception_wrapper()
    def filter_ids_text(self, file_: Path):
        """go over the metadata and get the list of paper_ids for each paper with
        pdf_parse text. Validity is determined by whether the paper has a proper grobid parse.
        Used to create a list of parseable papers.

        This method accumulates data into the state, as it won't take so much space.

        Used by filtering/filter_ids.py.
        """
        # ignore irrelevant entry (cf. filter_ids.py)
        file_, _, _, _, _ = file_

        with open(file_, "r") as f:
            # ID.tsv
            fname = int(file_.name.replace(".tsv", ""))
            self.results[fname] = []
            for line in f:
                obj = line.strip().split("\t")
                if LegacyFilter.validate_grobid(obj):
                    self.results[fname].append(obj[1])

    @flutes.exception_wrapper()
    def make_map(self, file_):
        """construct a dictionary of pid - (batchnum, line#)"""
        file_, allowed_pids = file_
        use_allowed_pids = len(allowed_pids) > 0
        batchnum = int(file_.name.replace(".jsonl.gz", ""))

        pid2idx = {}
        result = set()
        with gzip.open(str(file_), "r") as f:
            for idx, line in enumerate(f):
                pid = re.search(PID_PAT, line.decode("utf8")).group(1)
                result.add(pid)
                pid2idx[pid] = idx
            if use_allowed_pids:
                result = list(result.intersection(allowed_pids))
        for pid in result:
            self.results[pid] = (batchnum, pid2idx[pid])

    @flutes.exception_wrapper()
    def gather_abstracts(self, file_):
        file_, paper_ids, output_dir, valid_lines, pid2loc = file_
        target_batchnum = int(file_.name.replace(".jsonl.gz", ""))

        # load citations
        bnum_bucketed_citations = {i: [] for i in range(10000)}
        for pid, (bnum, lnum) in pid2loc.items():
            bnum_bucketed_citations[bnum].append((lnum, pid))

        # batch load citations instead of reading files on the fly for each citations
        pid2citation = {}
        for bnum in range(10000):
            if len(bnum_bucketed_citations[bnum]) == 0:
                continue
            cite_valid_lines = sorted(bnum_bucketed_citations[bnum], key=lambda x: x[0])
            pdf_parse = file_.parent / file_.name.replace(
                f"{target_batchnum}", f"{bnum}"
            )
            f_idx, l_idx = 0, 0
            with gzip.open(str(pdf_parse), "rb", thread=8) as f:
                for f_idx, line in enumerate(f):
                    if f_idx > cite_valid_lines[-1][0]:
                        break

                    if f_idx != cite_valid_lines[l_idx][0]:
                        continue
                    else:
                        l_idx += 1

                    pid = re.search(PID_PAT, line.decode("utf8")).group(1)
                    pid2citation[pid] = line

        result = []
        f_idx, l_idx = 0, 0
        with gzip.open(str(file_), "r", thread=8) as f:
            for f_idx, line in enumerate(f):
                if f_idx > valid_lines[-1]:
                    break

                if f_idx != valid_lines[l_idx]:
                    continue
                else:
                    l_idx += 1

                # json parse takes a lot of time
                obj = json.loads(line)
                pid = obj["paper_id"]

                outb_pids, inb_pids = paper_ids[pid]

                # random pick strategy
                out_data = []
                for out_pid in outb_pids:
                    out_obj = pid2citation.get(out_pid, None)
                    if out_obj is not None:
                        out_obj = json.loads(out_obj)
                    out_abs = load_bg_info_legacy(out_pid, out_obj, content="abstract")
                    # target is the paper itself, not the citation
                    bginfo = load_bg_info_legacy(out_pid, obj, content="cite_context")
                    if bginfo is None:
                        bginfo = []
                    out_data.append((out_pid, bginfo, out_abs))

                # sort by number of times a paper is cited (i.e., number of matches.)
                out_data = sorted(out_data, key=lambda x: len(x[1]), reverse=True)

                # max 10, look up 20 in case there are bunch of None's
                in_data = []
                # TODO: better inbound citation selection strategy
                for in_pid in inb_pids:
                    in_obj = pid2citation.get(in_pid, None)
                    if in_obj is not None:
                        in_obj = json.loads(in_obj)
                    cite_context = load_bg_info_legacy(
                        pid, in_obj, content="cite_context"
                    )
                    if cite_context != []:
                        in_data.append((in_pid, cite_context))

                result.append((in_data, out_data, obj))

        with (output_dir / f"{target_batchnum}.pkl").open("wb") as f:
            pickle.dump(result, f)


class Filter(flutes.PoolState):
    def __init__(self):
        self.results = {}

    @flutes.exception_wrapper()
    def filter_ids_complete(self, file_: Path):
        """go over the metadata and get the in/out-bound citation data for each VALID
        paper. Validity is determined by whether the paper has a proper grobid parse,
        plus the existence of at least one of in/out citations.

        This method accumulates data into the state, as it won't take so much space.

        Used by filtering/filter_ids.py.
        """
        file_, valid_pids, min_cite, max_cite, seed = file_
        # make sure
        random.seed(seed)

        with gzip.open(str(file_), "r") as f:
            # metadata_ID.jsonl.gz
            fname = int(file_.name.replace(".jsonl.gz", "").replace("metadata_", ""))
            self.results[fname] = []
            for line in f:
                obj = json.loads(line)
                # has_pdf_parse, has_pdf_parsed_(bib_entries, abstract, body_text)
                if (
                    valid_pids.get(obj["paper_id"], False)
                    and obj["has_inbound_citations"]
                    and obj["has_outbound_citations"]
                    and len(obj["inbound_citations"]) > min_cite
                    and len(obj["outbound_citations"]) > min_cite
                ):
                    # filter valid citations
                    ibc = [
                        pid
                        for pid in obj["inbound_citations"]
                        if valid_pids.get(pid, False)
                    ]
                    obc = [
                        pid
                        for pid in obj["outbound_citations"]
                        if valid_pids.get(pid, False)
                    ]
                    random.shuffle(ibc)
                    random.shuffle(obc)
                    if len(ibc) > 0 and len(obc) > 0:
                        self.results[fname].append(
                            (obj["paper_id"], ibc[:max_cite], obc[:max_cite])
                        )

    def filter_ids_text(self, file_: Path):
        """go over the metadata and get the list of paper_ids for each paper with
        pdf_parse text. Validity is determined by whether the paper has a proper grobid parse.
        Used to create a list of parseable papers.

        This method accumulates data into the state, as it won't take so much space.

        Used by filtering/filter_ids.py.
        """
        # ignore irrelevant entry (cf. filter_ids.py)
        file_, *_ = file_

        with gzip.open(str(file_), "r") as f:
            # metadata_ID.jsonl.gz
            fname = int(file_.name.replace(".jsonl.gz", "").replace("metadata_", ""))
            self.results[fname] = []
            for line in f:
                obj = json.loads(line)
                if (
                    obj["has_pdf_parse"]
                    and obj["has_pdf_parsed_abstract"]
                    and obj["has_pdf_parsed_bib_entries"]
                    and obj["has_pdf_parsed_body_text"]
                ):
                    self.results[fname].append(obj["paper_id"])

    @flutes.exception_wrapper()
    def filter_examples(self, args):
        """Extract examples specified by ids_file. This script dumps without storing
        filtered data into a state because it will blow up the ram.

        Used by filter_instances.py.
        """
        file_, required_ids, out_dir = args
        batchnum = int(file_.name.replace(".jsonl.gz", ""))
        # inbound_map = {r[0]: len(r[2]) for r in required_ids}
        # to_take = list(inbound_map.keys())
        to_take = required_ids

        result = []
        with open(file_, "r") as f:
            for line in f:
                # json parse takes a lot of time
                obj = json.loads(line.strip())
                if obj["paper_id"] in to_take:
                    # add the paper info as well as the number of inbound cites
                    # result.append((obj, inbound_map[obj['paper_id']]))
                    result.append(obj)

        with open(out_dir / file_.name, "w") as f:
            for r in result:
                print(json.dumps(r), file=f)
        # self.results.append((batchnum, result))

    @flutes.exception_wrapper()
    def make_map(self, file_):
        """construct a dictionary of pid - (batchnum, line#)"""
        file_, allowed_pids = file_
        use_allowed_pids = len(allowed_pids) > 0
        batchnum = int(file_.name.replace(".jsonl.gz", "").replace("pdf_parses_", ""))

        pid2idx = {}
        result = set()
        with gzip.open(str(file_), "r") as f:
            for idx, line in enumerate(f):
                obj = json.loads(line)
                result.add(obj["paper_id"])
                pid2idx[obj["paper_id"]] = idx
            if use_allowed_pids:
                result = list(result.intersection(allowed_pids))
        for pid in result:
            self.results[pid] = (batchnum, pid2idx[pid])

    # @flutes.exception_wrapper()
    def gather_abstracts(self, file_):
        file_, paper_ids, output_dir, valid_lines, pid2loc = file_
        target_batchnum = int(
            file_.name.replace(".jsonl.gz", "").replace("pdf_parses_", "")
        )

        # load citations
        bnum_bucketed_citations = {i: [] for i in range(100)}
        for pid, (bnum, lnum) in pid2loc.items():
            bnum_bucketed_citations[bnum].append((lnum, pid))

        # batch load citations instead of reading files on the fly for each citations
        pat = re.compile(r"paper_id\":\s+\"(\d+)\"")
        pid2citation = {}
        for bnum in range(100):
            cite_valid_lines = sorted(bnum_bucketed_citations[bnum], key=lambda x: x[0])
            pdf_parse = file_.parent / file_.name.replace(
                f"{target_batchnum}", f"{bnum}"
            )
            f_idx, l_idx = 0, 0
            with gzip.open(str(pdf_parse), "rb", thread=8) as f:
                for f_idx, line in enumerate(f):
                    if f_idx > cite_valid_lines[-1][0]:
                        break

                    if f_idx != cite_valid_lines[l_idx][0]:
                        continue
                    else:
                        l_idx += 1

                    pid = re.search(pat, line.decode("utf8")).group(1)
                    pid2citation[pid] = line

        result = []
        f_idx, l_idx = 0, 0
        with gzip.open(str(file_), "r", thread=8) as f:
            for f_idx, line in enumerate(f):
                if f_idx > valid_lines[-1]:
                    break

                if f_idx != valid_lines[l_idx]:
                    continue
                else:
                    l_idx += 1

                # json parse takes a lot of time
                obj = json.loads(line)
                pid = obj["paper_id"]

                inb_pids, outb_pids = paper_ids[pid]

                # random pick strategy
                out_data = []
                for out_pid in outb_pids:
                    out_obj = pid2citation.get(out_pid, None)
                    if out_obj is not None:
                        out_obj = json.loads(out_obj)
                    out_abs = load_bg_info(out_pid, out_obj, content="abstract")
                    # target is the paper itself, not the citation
                    bginfo = load_bg_info(out_pid, obj, content="cite_context")
                    if bginfo is None:
                        bginfo = []
                    out_data.append((out_pid, bginfo, out_abs))

                # sort by number of times a paper is cited (i.e., number of matches.)
                out_data = sorted(out_data, key=lambda x: len(x[1]), reverse=True)

                # max 10, look up 20 in case there are bunch of None's
                in_data = []
                # TODO: better inbound citation selection strategy
                for in_pid in inb_pids:
                    in_obj = pid2citation.get(in_pid, None)
                    if in_obj is not None:
                        in_obj = json.loads(in_obj)
                    cite_context = load_bg_info(pid, in_obj, content="cite_context")
                    if cite_context != []:
                        in_data.append((in_pid, cite_context))

                result.append((in_data, out_data, obj))

        with (output_dir / f"{target_batchnum}.pkl").open("wb") as f:
            pickle.dump(result, f)
