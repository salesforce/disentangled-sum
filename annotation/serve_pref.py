# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

"""serve_pref.py: runs a simple web UI for annotating sentences and recording the annotation
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
parser.add_argument("--data", type=Path, help="Path to cord eval data directory.")
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
INDEX = 0
marked = []

random.seed(args.seed)


def load_summaries(path: Path, max_count: int) -> None:
    global SUMMARIES, outstream

    def load_jsonl(path: Path):
        objs = {}
        with path.open("r") as f:
            for l in f:
                obj = json.loads(l)
                objs[obj["paper_id"]] = obj
        return objs

    id_ = uuid.uuid4().hex[:6]
    outstream = open(f"results/{args.user}-{id_}.txt", "w")
    examples = {}
    base = load_jsonl(path / "base" / "eval.jsonl.-1")
    con = load_jsonl(path / "cc_inf_contrib_cord" / "eval.jsonl.-1")
    ctx = load_jsonl(path / "cc_inf_other_cord" / "eval.jsonl.-1")
    SUMMARIES = [(idx, con[i], ctx[i], base[i]) for idx, i in enumerate(base)][:args.max_count]


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
            "summary_pref.html",
            summary="",
            exid="... you are done.",
            hide_button=True,
        )

    # first example
    pid, *summ_text = SUMMARIES[INDEX]
    exid = f"{pid+1}"

    if INDEX in marked:
        INDEX += 1
    return render_template(
        "summary_pref.html",
        summary=make_table(summ_text),
        exid=exid,
        hide_button=False,
        max_count=len(SUMMARIES),
    )


@app.route("/", methods=["POST"])
def render_example():
    global SUMMARIES, INDEX, marked, outstream
    print(INDEX)
    req_data = request.get_json()
    # dump selections
    marked.append(req_data)
    print(
        ord(req_data["useful"]) - 65,
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
    pid, *summ_text = SUMMARIES[INDEX]
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
    strings = [f"<td style='width: 33%; vertical-align: top'>{s['decoded']}</td>" for s in summ[:-1]]
    strings += [f"<td style='width: 33%; vertical-align: top' class='abstract'>{summ[-1]['decoded']}</td>"]
    string = "<tr>" + " ".join(strings) + "</tr>"
    return string


if __name__ == "__main__":
    app.run(host=args.host, port=args.port)
