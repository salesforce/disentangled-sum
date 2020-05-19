# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
import ssl

import nltk
import rouge


def run_rouge(generated, references, max_length):

    if isinstance(generated, list):
        assert isinstance(references, list)
        assert len(generated) == len(references)
    else:
        assert isinstance(generated, str)
        assert isinstance(references, str)

        generated = [generated]
        references = [references]

    # most args are from `run_summarization.py`
    rouge_evaluator = rouge.Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=max_length,
        length_limit_type="words",
        apply_avg=True,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.2,
        stemming=True,
    )

    scores = rouge_evaluator.get_scores(generated, references)
    return scores


def nltk_download_no_ssl(package):
    """NLTK default download script is broken. This fixes it.
    More: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
    """

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download(package)
