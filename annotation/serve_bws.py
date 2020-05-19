# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

"""serve_bws.py: runs a simple web UI for annotating sentences and recording the annotation
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
# app.config["FLASK_HTPASSWD_PATH"] = "/export/home/.htpasswd"
# app.config["FLASK_SECRET"] = "6@#6S8adMI"
# app.config["FLASK_AUTH_ALL"] = True

# htpasswd = HtPasswdAuth(app)

outstream = None

SUMMARIES = []
MODEL_ORDER = []
INDEX = 0
marked = []

random.seed(args.seed)


def load_summaries(path: Path, max_count: int) -> None:
    global SUMMARIES, outstream, MODEL_ORDER
    id_ = uuid.uuid4().hex[:6]
    outstream = open(f"results/{args.user}-{id_}.txt", "w")
    with path.open("r") as f:
        obj = [json.loads(s) for s in f.read().strip().split("\n")]
        total = len(obj)
        recs = [
            [
                ("cc", (obj[i]["ccco"], obj[i]["ccba"])),
                ("mh", (obj[i]["mhco"], obj[i]["mhba"])),
                ("ccmi", (obj[i]["cccomi"], obj[i]["ccbami"])),
                ("mhmi", (obj[i]["mhcomi"], obj[i]["mhbami"])),
            ]
            for i in range(total)
        ]

        for r in recs:
            random.shuffle(r)
            MODEL_ORDER.append([m for m, _ in r])

        SUMMARIES = [(i, [t for _, t in r]) for i, r in enumerate(recs[:max_count])]


@app.route("/", methods=["GET"])
def index():
    user = "a"
    global SUMMARIES, INDEX, args, marked
    if len(SUMMARIES) == 0:
        load_summaries(args.data, args.max_count)

    if len(marked) == 0:
        INDEX = 0
    elif len(marked) == len(SUMMARIES):
        return render_template(
            "summary_compare.html",
            summary="",
            exid="... you are done.",
            hide_button=True,
        )

    # first example
    pid, summ_text = SUMMARIES[INDEX]
    exid = f"{pid+1}"

    if INDEX in marked:
        INDEX += 1
    return render_template(
        "summary_compare.html",
        summary=make_table(summ_text),
        exid=exid,
        hide_button=False,
        max_count=len(SUMMARIES),
    )


@app.route("/", methods=["POST"])
def render_example():
    global SUMMARIES, INDEX, marked, outstream, MODEL_ORDER
    print(INDEX)
    req_data = request.get_json()
    # dump selections
    marked.append(req_data)
    print(
        ord(req_data["most_dis"]) - 65,
        ord(req_data["least_dis"]) - 65,
        " ".join(MODEL_ORDER[INDEX]),
        sep="\t",
        file=outstream,
    )
    # save progress at every example
    outstream.flush()

    # done
    if INDEX == len(SUMMARIES) - 1:
        outstream.close()
        return (
            json.dumps({"success": True, "is_done": True}),
            200,
            {"ContentType": "application/json"},
        )

    # not done yet
    INDEX += 1
    pid, summ_text = SUMMARIES[INDEX]
    exid = f"{pid+1}"
    return (
        json.dumps(
            {
                "success": True,
                "is_done": False,
                "new_exid": exid,
                "new_text": make_table(summ_text),
                "max_count": len(SUMMARIES),
            }
        ),
        200,
        {"ContentType": "application/json"},
    )


def make_table(summ):
    string = ""
    headers = "A B C D".split()
    for h, s in zip(headers, summ):
        string += f"<td style='width: 1%'><b>{h}</b></td>"
        string += f"<td style='width: 48%; vertical-align: top'>{s[0]}</td>"
        string += f"<td style='width: 48%; vertical-align: top'>{s[1]}</td>"
        string = "<tr>" + string + "</tr>"
    return string


# @htpasswd.required
@app.route("/view/<path:text>", methods=["GET", "POST"])
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
        "summary_compare.html",
        summary=summ,
        exid=exid,
        hide_button=True,
        max_count=len(SUMMARIES),
    )


if __name__ == "__main__":
    app.run(host=args.host, port=args.port)
