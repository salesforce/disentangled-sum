# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

"""serve.py: runs a simple web UI for annotating sentences and recording the annotation
onto a file. The annotation starts after launching. With /view/N URI, one can also check
N-th example.
"""
import argparse
import json
import random
import uuid
from pathlib import Path
from typing import List

from flask import Flask, render_template, request
from flask_htpasswd import HtPasswdAuth

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, help="Path to dataset file.")
parser.add_argument("--user", type=str, help="Who's annotating.", required=True)
parser.add_argument("--host", type=str, help="Hostname.", default="localhost")
parser.add_argument("--port", type=int, help="Port.", default=5294)
parser.add_argument(
    "--seed", type=int, help="Random seed when subsample annotations.", default=14
)
parser.add_argument(
    "--max-count", type=int, help="Number of examples to show.", default=1000
)
args = parser.parse_args()

app = Flask(__name__)
app.debug = True
app.config["FLASK_HTPASSWD_PATH"] = ""
app.config["FLASK_SECRET"] = ""
app.config["FLASK_AUTH_ALL"] = True

htpasswd = HtPasswdAuth(app)

outstream = None

SUMMARIES = []
INDEX = 0
marked = []

random.seed(args.seed)


def load_summaries(path: Path, max_count: int) -> None:
    global SUMMARIES, outstream
    id_ = uuid.uuid4().hex[:6]
    outstream = open(f"results/{args.user}-{id_}.txt", "w")
    with path.open("r") as f:
        obj = f.read().strip().split("\n")
        total = len(obj)
        indices = list(range(total))
        SUMMARIES = [(i, obj[i]) for i in indices[:max_count]]


def tag_sentence(summary: List[str], highlighted_sent_ids: List[int]) -> str:
    """assumes the summaries are already sentence-split."""

    def maybe_mark(s: str, cond: bool) -> str:
        return "<mark>" + s + "</mark>" if cond else s

    tok = [
        f'<span id="{idx}">' + maybe_mark(s, idx in highlighted_sent_ids) + "</span>"
        for idx, s in enumerate(summary)
    ]
    return " ".join(tok)


@app.route("/", methods=["GET"])
@htpasswd.required
def index(user):
    global SUMMARIES, INDEX, args, marked
    if len(SUMMARIES) == 0:
        load_summaries(args.data, args.max_count)

    if len(marked) == 0:
        INDEX = 0
    elif len(marked) == len(SUMMARIES):
        return render_template(
            "summary_view.html", summary="", exid="... you are done.", hide_button=True
        )

    # first example
    line_num, summ = SUMMARIES[INDEX]
    summ = json.loads(summ)
    pid, summ_text = summ["paper"]["paper_id"], summ["target"]
    exid = f"{line_num}-{pid}"

    if INDEX in marked:
        INDEX += 1
    return render_template(
        "summary_view.html",
        summary=tag_sentence(summ_text, []),
        exid=exid,
        hide_button=False,
    )


@app.route("/", methods=["POST"])
def render_example():
    global SUMMARIES, INDEX, marked, outstream
    print(INDEX)
    req_data = request.get_json()
    # dump selections
    marked.append(req_data["exid"])
    print(req_data["exid"], req_data["value"], file=outstream)
    # save progress at every example
    outstream.flush()

    # done
    if INDEX == len(SUMMARIES):
        outstream.close()
        return (
            json.dumps({"success": True, "is_done": True}),
            200,
            {"ContentType": "application/json"},
        )

    # not done yet
    INDEX += 1
    line_num, next_summ = SUMMARIES[INDEX]
    next_summ = json.loads(next_summ)
    pid, summ_text = next_summ["paper"]["paper_id"], next_summ["target"]
    exid = f"{line_num}-{pid}"
    return (
        json.dumps(
            {
                "success": True,
                "is_done": False,
                "new_exid": exid,
                "new_text": tag_sentence(summ_text, []),
            }
        ),
        200,
        {"ContentType": "application/json"},
    )


@app.route("/view/<path:text>", methods=["GET", "POST"])
@htpasswd.required
def view(text):
    global SUMMARIES, args, marked
    marked_sent_ids = []
    if "mark" in request.args:
        marked_sent_ids = list(map(int, request.args["mark"].split(",")))

    if len(SUMMARIES) == 0:
        load_summaries(args.summaries, -1)

    exid = int(text)
    # first example
    if exid > len(SUMMARIES):
        exid = 0
    summ = SUMMARIES[exid]
    return render_template(
        "summary_view.html",
        summary=tag_sentence(summ, marked_sent_ids),
        exid=exid,
        hide_button=True,
    )


if __name__ == "__main__":
    app.run(host=args.host, port=args.port)
