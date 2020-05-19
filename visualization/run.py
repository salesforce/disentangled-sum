# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""run.py: runs a streamlit demo for visualizing the generated summaries from multiple
domains. `cd` to this directory and run:

$ python run.py

to start the service.
"""
import streamlit as st
import numpy as np
import json


def load_jsonl(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def load_data():
    cord = [json.loads(line) for line in open("./cord_demo.jsonl")]
    cs = [json.loads(line) for line in open("./cs_demo.jsonl")]
    med = [json.loads(line) for line in open("./med_demo.jsonl")]
    soc = [json.loads(line) for line in open("./soc_demo.jsonl")]
    db_data = [
        [dom[i]["distilbart"] for dom in [cord, cs, med, soc]]
        for i, _ in enumerate(cord)
    ]
    gd_data = [
        [dom[i]["gold"] for dom in [cord, cs, med, soc]] for i, _ in enumerate(cord)
    ]
    co_data = [
        [dom[i]["contribution"] for dom in [cord, cs, med, soc]]
        for i, _ in enumerate(cord)
    ]
    ct_data = [
        [dom[i]["context"] for dom in [cord, cs, med, soc]] for i, _ in enumerate(cord)
    ]
    meta_data = [
        [
            (dom[i]["title"].replace('"', ""), dom[i]["url"])
            for dom in [cord, cs, med, soc]
        ]
        for i, _ in enumerate(cord)
    ]
    return (db_data, gd_data, co_data, ct_data, meta_data)


def main():

    st.beta_set_page_config(
        page_title="Contribusum Demo", page_icon="📝", initial_sidebar_state="expanded"
    )
    header = ["CORD (COVID-19)", "Computer Science", "Medicine", "Sociology"]

    st.title("What's new? Summarizing Contributions in Scientific Literature")
    st.markdown(
        "This is a summarization demo for the paper [*What's new? Summarizing Contributions in Scientific Literature*](#). "
        "The proposed model takes as input a paper and summarize into disentangled summaries, specifically contribution and context information. "
        "In this demo, we provide 50 summaries for 4 different domains below including our disentangled summaries, the original paper abstact, and a system-generated abstract. "
        "Code for the paper is available [here](https://github.com/salesforce/contribusum)."
    )

    # distilbart, gold, contrib, context, meta
    db, gd, co, ct, meta = load_data()

    # a new random index
    index = np.random.randint(1, len(db) + 1)

    # UIs
    # sidebar: pulldown
    option = st.selectbox("Please select a domain.", header)
    domain = header.index(option)
    # generate_random buttom
    do_random = st.button("Generate another random summary!")

    # overwrite if specific ID is chosen.
    target_index = st.selectbox(
        "or, jump to a specific instance. [1-50]",
        [f"{i}: {m[domain][0]}..." for i, m in zip(range(1, len(db) + 1), meta)],
    )
    if do_random:
        index = np.random.randint(1, len(db) + 1)
        display_text = [dat[index - 1][domain] for dat in [gd, db, co, ct]]

    else:
        index = int(target_index.split(":")[0])
    display_text = [dat[index - 1][domain] for dat in [gd, db, co, ct]]

    # Show the original paper information
    with st.beta_expander("Paper Information"):
        title, url = meta[index - 1][domain]
        url = url.split(";")[0]
        st.markdown(f"Demo ID: {index}")
        st.markdown(f"Title: [{title}]({url})")

    st.header("Summaries")

    # Gold
    with st.beta_expander("Gold", expanded=True):
        st.markdown(display_text[0])

    # DistilBART
    with st.beta_expander("DistilBART", expanded=True):
        st.markdown(display_text[1])

    # our model
    with st.beta_expander("Our model (ControlCode + Informativeness)", expanded=True):
        con, ctx = st.beta_columns(2)
        with con:
            st.subheader("Contribution")
            st.markdown(display_text[2])
        with ctx:
            st.subheader("Context")
            st.markdown(display_text[3])


if __name__ == "__main__":
    main()
