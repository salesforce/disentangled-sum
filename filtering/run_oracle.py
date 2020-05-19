# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""run_oracle.py: calculates oracle ROUGE score by selecting oracle sentence set
greedily. The algorithm is borrowed from Yang Liu's projects.
"""
import argparse
import pickle
import re
import sys
from itertools import product
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
from bert_score import score as bscore
from nltk import sent_tokenize, word_tokenize
from rouge_score import rouge_scorer
from rouge_score.rouge import scoring
from tqdm import tqdm
from utils import progress


class Example(NamedTuple):
    cite_src: List[List[str]]
    paper_src: List[List[str]]
    tgt: List[List[str]]
    tgt_n_sents: int


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i : i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_bert_selection(
    doc_sent_list: List[str], abstract_sent_list: List[str], summary_size: int
):
    max_score = 0.0
    batch_size = 6000

    # pre-calculate everything
    scores = []
    # all combinations
    cands, refs = list(zip(*list(product(doc_sent_list, abstract_sent_list))))
    n_batches = len(refs) // batch_size
    for b in progress(
        range(n_batches + (1 if len(refs) % batch_size > 0 else 0)),
        desc="B-score for selection",
    ):
        ref = [
            " ".join(r)
            for r in refs[b * batch_size : min((b + 1) * batch_size, len(refs))]
        ]
        cand = [
            " ".join(c)
            for c in cands[b * batch_size : min((b + 1) * batch_size, len(refs))]
        ]
        _, _, F1 = bscore(cand, ref, lang="en")
        scores += F1.cpu().tolist()

    selected = []
    for s in range(summary_size):
        cur_max_score = 0.0
        cur_id = -1
        for i, _ in enumerate(doc_sent_list):
            if i in selected:
                continue

            # the cand vs every ref sentence
            score = scores[
                i * len(abstract_sent_list) : (i + 1) * len(abstract_sent_list)
            ]
            # TODO: weight the vector to bias towards positions
            score = np.mean(np.asarray(score))

            if score > cur_max_score:
                cur_max_score = score
                cur_id = i

        if cur_id == -1:
            return selected
        selected.append(cur_id)
        max_score = cur_max_score

    return sorted(selected)


def greedy_selection(
    doc_sent_list: List[str], abstract_sent_list: List[str], summary_size: int
):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents = [_rouge_clean(" ".join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def preprocess(triple):
    in_papers, out_papers, obj = triple
    in_tok_abs, out_tok_abs = [], []

    # process target abstract
    if obj["metadata"]["abstract"] is not None:
        abs_string = obj["metadata"]["abstract"]
    elif len(obj["grobid_parse"]["abstract"]) > 0:
        abs_string = obj["grobid_parse"]["abstract"][0]["text"]
    else:
        print("-")
        # no point doin the rest
        return None

    target_abs = [word_tokenize(s) for s in sent_tokenize(abs_string)]

    for _, sents in in_papers:
        if sents is not None:
            in_tok_abs.append([word_tokenize(s) for s in sents])
    for _, sents in out_papers:
        if sents is not None:
            out_tok_abs.append([word_tokenize(s) for s in sents])

    target_source = [
        word_tokenize(s.replace("al@", "al."))
        for para in obj["grobid_parse"]["body_text"]
        for s in sent_tokenize(para["text"].replace("al.", "al@"))
    ]

    return in_tok_abs, out_tok_abs, target_source, target_abs


def citation_selection(examples, strategy="concat_in_out"):
    # the most naive source construction
    # for idx, e in examples:
    for idx, e in examples:
        in_joined = sum(e[0], [])
        out_joined = sum(e[1], [])

        if strategy == "concat_in_out":
            yield idx, Example(in_joined + out_joined, e[2], e[3], len(e[3]))
        elif strategy == "concat_out_in":
            yield idx, Example(out_joined + in_joined, e[2], e[3], len(e[3]))
        elif strategy == "inbound_only":
            yield idx, Example(in_joined, e[2], e[3], len(e[3]))
        elif strategy == "outbound_only":
            yield idx, Example(out_joined, e[2], e[3], len(e[3]))
        elif strategy == "in_main":
            yield idx, Example(e[2] + in_joined, e[2], e[3], len(e[3]))
        elif strategy == "out_main":
            yield idx, Example(e[2] + out_joined, e[2], e[3], len(e[3]))
        elif strategy == "all":
            yield idx, Example(e[2] + in_joined + out_joined, e[2], e[3], len(e[3]))
        elif strategy == "source_matched_with_inbound":
            # For each inbound cite context, look for K similar sentences from paper text.
            paper_sent_ids = greedy_selection(e[2], in_joined, 30)
            total = [e[2][p] for p in paper_sent_ids]
            yield idx, Example(total, e[2], e[3], len(e[3]))
        elif strategy == "keep_both":
            # select inbound, and return both inbound and outbound separately
            # this is for evaluateing auto-annotation
            paper_sent_ids = greedy_selection(e[2], in_joined, 30)
            total = [e[2][p] for p in paper_sent_ids]
            yield idx, Example(total, out_joined, e[3], len(e[3]))
        else:
            raise NotImplementedError


def nouveau_rouge(contrib_rouge, other_rouge, N=1):
    # Value taken from Table 2 @
    # https://www.mitpressjournals.org/doi/pdfplus/10.1162/coli_a_00033
    a0, a1, a2 = 0, 1, 1
    if N == 1:
        a0, a1, a2 = -0.0271, -7.355, 13.4227
    elif N == 2:
        a0, a1, a2 = 0.9126, -5.4536, 21.1556
    else:
        raise NotImplementedError
    return a0 + a1 * other_rouge + a2 * contrib_rouge


def similarity(
    targets: List[List[str]],
    inbound: List[List[str]],
    outbound: List[List[str]],
    metric: str = "rougeL",
    batch_size: int = 6000,
):
    """For each target sentence, calculate similarity scores against
    inbound/outbound texts. Use the scores to determine the membership to the
    contribution summary.

    :param targets: Target abstract, sentence- and word-tokenized.
    :param inbound: Inbound texts.
    :param outbound: Outbound texts.
    :param metric: Either of "rouge[1,2,L]" or "bertscore". Used to calculate
        similarity score of a pair.
    :param batch_size: How much chunk of sentences to compute scores at once.
    """
    if metric.startswith("rouge"):
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        def calculate(candidates, references):
            F1 = []
            for c, r in zip(candidates, references):
                F1.append(scorer.score(c, r)[metric].fmeasure)
            return F1

    else:

        def calculate(candidates, references):
            _, _, F1 = bscore(candidates, references, lang="en")
            return F1.cpu().tolist()

    targets = [" ".join(s) for s in targets]
    in_n_out = [" ".join(s) for s in inbound + outbound]
    split = len(inbound)

    # pre-calculate everything
    scores = []
    # all combinations
    cands, refs = list(zip(*list(product(targets, in_n_out))))
    n_batches = len(refs) // batch_size
    for b in range(n_batches + (1 if len(refs) % batch_size > 0 else 0)):
        ref = refs[b * batch_size : min((b + 1) * batch_size, len(refs))]
        cand = cands[b * batch_size : min((b + 1) * batch_size, len(refs))]

        scores += calculate(cand, ref)

    selected = []
    for sidx, _ in enumerate(targets):
        score_chunk = scores[sidx * len(in_n_out) : (sidx + 1) * len(in_n_out)]
        in_sim, out_sim = (
            np.max(score_chunk[:split]),
            np.mean(score_chunk[split:]),
        )

        if in_sim > out_sim:
            selected.append(sidx)
        elif any([word in targets[sidx].lower() for word in ["we "]]):
            selected.append(sidx)
        # elif any([word in targets[sidx].lower() for word in ["show", "propose", "present"]]):
        #     selected.append(sidx)
        # elif any([word in targets[sidx].lower() for word in ["developed"]]):
        #     selected.append(sidx)

    return selected


def agreement_score(s1: set, s2: set, method="jaccard") -> float:
    intersection = s1.intersection(s2)
    union = s1.union(s2)
    if method == "jaccard":
        # no contribution, both agreed
        if len(s1) == 0 and len(s2) == 0:
            return 1.0
        elif len(union) == 0:
            return 0.0
        return len(intersection) / len(union)
    else:
        raise NotImplementedError


def main(args):

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # prepare rouge scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    p_aggregator = scoring.BootstrapAggregator()
    c_aggregator = scoring.BootstrapAggregator()

    # load annotation
    if args.contrib_annot is not None and args.contrib_annot.exists():
        with args.contrib_annot.open("r") as f:
            indices = {}
            for line in f:
                try:
                    idx, annot = line.strip().split(" ")
                    contrib_sent_idx = [int(i) for i in annot.split(",")]
                except ValueError:
                    # no annotation for that index
                    idx = line.strip()
                    contrib_sent_idx = []
                indices[int(idx)] = contrib_sent_idx

    # load data & dump the preprocessed texts
    if (args.data_dir / "preprocessed_instances.pkl").exists():
        with (args.data_dir / "preprocessed_instances.pkl").open("rb") as f:
            instances = pickle.load(f)
    else:
        instances = []
        for file_idx in range(100):
            if (args.data_dir / f"{file_idx}.pkl").exists():
                with (args.data_dir / f"{file_idx}.pkl").open("rb") as f:
                    instances += [preprocess(t) for t in pickle.load(f) if t[0] != []]

        # remove non-target instances (cf. preprocess)
        instances = [i for i in instances if i is not None]

        with (args.output_dir / "preprocessed_instances.pkl").open("wb") as f:
            pickle.dump(instances, f)

    # resolve unfortuante mismatch due to working on GROBID parse only
    with open("/export/home/grobid_outcite_matching_examples.pkl", "rb") as f:
        grobid_instances = [(i, preprocess(j)) for i, j in pickle.load(f) if i < 101]

    filtered_instances = [
        (idx, i)
        for idx, i in enumerate(instances)
        if idx in [j[0] for j in grobid_instances]
    ]

    instances = []
    for f, g in zip(filtered_instances, grobid_instances):
        instances.append((g[0], (g[1][0], f[1][1], f[1][2], g[1][-1])))

    print(f"Loaded {len(instances)} instances.")

    example_generator = citation_selection(instances, args.citation_selection)

    if args.eval_auto_annotate:
        auto_annotated = {}
        jaccard = []
        for ex_id, ex in progress(
            example_generator, total=len(instances), desc="Auto-annotating"
        ):
            if ex_id not in indices:
                print(f"{ex_id} not found.")

            if len(ex.cite_src) == 0 or len(ex.cite_src) == 0:
                continue

            selected = similarity(ex.tgt, ex.cite_src, ex.paper_src)
            # selected = list(range(len(ex.tgt)))
            agreement = agreement_score(set(indices[ex_id]), set(selected))
            tqdm.write(
                f"{ex_id} \t {indices[ex_id]} -\t {selected} -\t {agreement:.3f}"
            )

            jaccard.append(agreement)
            auto_annotated[ex_id] = selected

        # compare_agreement
        print(np.mean(jaccard))

        sys.exit(0)

    if args.dump_output:
        cite_handler = (args.output_dir / f"{args.citation_selection}_output.txt").open(
            "w"
        )
        paper_handler = (args.output_dir / "selected_papertext_output.txt").open("w")
        target_handler = (args.output_dir / "target_contrib_summary.txt").open("w")

    # TODO: large refactoring needed.
    if args.use_nouveau_rouge:
        assert len(indices) > 0

        p_nouvs, c_nouvs = [], []
        p_nouvs2, c_nouvs2 = [], []
        p_rouges, c_rouges = [], []
        p_bs, c_bs = [], []

        stats = []
        # for each line, calculate the sentence indices that maximize avg rouge 1&2
        for ex_id, ex in tqdm(
            example_generator,
            ascii=True,
            ncols=88,
            total=len(instances),
            desc="Greedy selection",
        ):
            if ex_id not in indices:
                print(f"{ex_id} not found.")

            tgt_contrib = []
            tgt_else = []
            for i, s in enumerate(ex.tgt):
                if i in indices[ex_id]:
                    tgt_contrib.append(s)
                else:
                    tgt_else.append(s)

            tgt_n_sents = len(tgt_contrib)
            if tgt_n_sents == 0:
                continue

            tgt_contrib_str = " ".join([" ".join(s) for s in tgt_contrib])
            tgt_else_str = " ".join([" ".join(s) for s in tgt_else])

            stats.append(len(ex.cite_src))
            # we want to select sentences that maximizes rouge against contrib. summaries
            cite_sent_ids = greedy_selection(ex.cite_src, tgt_contrib, tgt_n_sents)
            cite_sent_str = " ".join([" ".join(ex.cite_src[i]) for i in cite_sent_ids])

            cite_c_rouge = scorer.score(tgt_contrib_str, cite_sent_str)
            cite_e_rouge = scorer.score(tgt_else_str, cite_sent_str)

            cite_nouv1 = nouveau_rouge(
                cite_c_rouge["rouge1"].fmeasure, cite_e_rouge["rouge1"].fmeasure, N=1
            )
            cite_nouv2 = nouveau_rouge(
                cite_c_rouge["rouge2"].fmeasure, cite_e_rouge["rouge2"].fmeasure, N=2
            )
            c_nouvs.append(cite_nouv1)
            c_nouvs2.append(cite_nouv2)
            c_rouges.append(
                (
                    cite_c_rouge["rouge1"].fmeasure,
                    cite_e_rouge["rouge1"].fmeasure,
                    cite_c_rouge["rouge2"].fmeasure,
                    cite_e_rouge["rouge2"].fmeasure,
                )
            )

            if args.use_bert_score:
                # bert_score
                _, _, F1c = bscore([cite_sent_str], [tgt_contrib_str], lang="en")
                _, _, F1e = bscore([cite_sent_str], [tgt_else_str], lang="en")
                c_bs.append((F1c.cpu().numpy(), F1e.cpu().numpy()))

            paper_sent_ids = greedy_selection(ex.paper_src, tgt_contrib, tgt_n_sents)
            paper_sent_str = " ".join(
                [" ".join(ex.paper_src[i]) for i in paper_sent_ids]
            )
            paper_c_rouge = scorer.score(tgt_contrib_str, paper_sent_str)
            paper_e_rouge = scorer.score(tgt_else_str, paper_sent_str)
            paper_nouv1 = nouveau_rouge(
                paper_c_rouge["rouge1"].fmeasure, paper_e_rouge["rouge1"].fmeasure, N=1
            )
            paper_nouv2 = nouveau_rouge(
                paper_c_rouge["rouge2"].fmeasure, paper_e_rouge["rouge2"].fmeasure, N=2
            )
            p_nouvs.append(paper_nouv1)
            p_nouvs2.append(paper_nouv2)
            p_rouges.append(
                (
                    paper_c_rouge["rouge1"].fmeasure,
                    paper_e_rouge["rouge1"].fmeasure,
                    paper_c_rouge["rouge2"].fmeasure,
                    paper_e_rouge["rouge2"].fmeasure,
                )
            )

            if args.dump_output:
                print(
                    " || ".join([" ".join(ex.cite_src[i]) for i in cite_sent_ids]),
                    cite_nouv1,
                    cite_nouv2,
                    file=cite_handler,
                    sep="\t",
                )
                print(
                    " || ".join([" ".join(ex.paper_src[i]) for i in paper_sent_ids]),
                    paper_nouv1,
                    paper_nouv2,
                    file=paper_handler,
                    sep="\t",
                )
                print(
                    " || ".join([" ".join(s) for s in tgt_contrib]),
                    file=target_handler,
                )

        print(f"Cite  (Nouveau-ROUGE): {np.mean(c_nouvs):.3f} {np.mean(c_nouvs2):.3f}")
        print(f"Paper (Nouveau-ROUGE): {np.mean(p_nouvs):.3f} {np.mean(p_nouvs2):.3f}")

        for kind, rouge_scores in zip(["Cite", "Paper"], [c_rouges, p_rouges]):
            print(f"{kind}")
            cr1, er1, cr2, er2 = np.mean(np.asarray(rouge_scores), axis=0)
            print(f" - Contrib.    R-1 & R-2: {cr1:.4f} {cr2:.4f}")
            print(f" - Other       R-1 & R-2: {er1:.3f} {er2:.4f}")

        if len(c_bs) > 0:
            print(f"BERTScore: {np.mean(np.asarray(c_bs), axis=0):1.3f}")

        print(f"Avg source # sents: {np.mean(stats):3.2f}")

        if args.dump_output:
            cite_handler.close()
            paper_handler.close()
            target_handler.close()

    else:
        gold_outstream = (args.output_dir / "gold_tgt.txt").open("w")
        for ex in tqdm(
            example_generator,
            ascii=True,
            ncols=88,
            total=len(instances),
            desc="Greedy selection",
        ):
            tgt_str = " ".join([" ".join(s) for s in ex.tgt])
            print(" || ".join([" ".join(s) for s in ex.tgt]), file=gold_outstream)

            cite_sent_ids = greedy_selection(ex.cite_src, ex.tgt, ex.tgt_n_sents)
            cite_sent_str = " ".join([" ".join(ex.cite_src[i]) for i in cite_sent_ids])
            cite_rouge = scorer.score(tgt_str, cite_sent_str)
            c_aggregator.add_scores(cite_rouge)

            paper_sent_ids = greedy_selection(ex.paper_src, ex.tgt, ex.tgt_n_sents)
            paper_sent_str = " ".join(
                [" ".join(ex.paper_src[i]) for i in paper_sent_ids]
            )
            paper_rouge = scorer.score(tgt_str, paper_sent_str)
            p_aggregator.add_scores(paper_rouge)

        gold_outstream.close()

        c_agg = c_aggregator.aggregate()
        p_agg = p_aggregator.aggregate()
        with (args.output_dir / f"{args.citation_selection}_scores.txt").open("w") as f:

            cf_score, pf_score = [], []
            for metric in ["rouge1", "rouge2", "rougeL"]:
                cf_score.append(getattr(getattr(c_agg[metric], "mid"), "fmeasure"))
                pf_score.append(getattr(getattr(p_agg[metric], "mid"), "fmeasure"))

            print("Cite:", "\t".join(map(str, cf_score)), sep="\t", file=f)
            print("Papr:", "\t".join(map(str, pf_score)), sep="\t", file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--citation-selection",
        type=str,
        choices=[
            "inbound_only",
            "outbound_only",
            "concat_in_out",
            "concat_out_in",
            "in_main",
            "out_main",
            "all",
            "source_matched_with_inbound",
            "keep_both",
        ],
        default="inbound_only",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contrib-annot", type=Path)
    parser.add_argument("--use-nouveau-rouge", action="store_true")
    parser.add_argument("--use-bert-score", action="store_true")
    parser.add_argument("--dump-output", action="store_true")
    parser.add_argument(
        "--eval-auto-annotate",
        action="store_true",
        help="Evaluate the accuracy of automatic annotation. Use this with `keep_both` option for citation selection.",
    )
    args = parser.parse_args()

    main(args)
