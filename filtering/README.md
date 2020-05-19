# 4 steps to make the dataset

## Download S2ORC.

Please contact the S2ORC authors [here](https://github.com/allenai/s2orc/#download-instructions).
It is **not** necessary to unzip the `*.jsonl.gz` format files after downloading.

## Filter IDs for citations, followed by papers
Not all the papers in S2ORC are of our interest. First we filter papers of interest both in terms of target papers and citations.

* `citation`: Filter all the papers that can be potential citations to targets. i.e.,
    * Relevant metadata fields: `has_pdf_parse, has_pdf_parsed_abstract, has_pdf_parsed_bib_entries, has_pdf_parsed_body_text`
* `paper`: Filter all the papers that can be the target papers. i.e.,
    * Minimum of K valid citations (+ maximum of K citations), where `valid` is the same condition as above.

*NOTE: Put `--legacy` for each script if running on the S2ORC dump of `20190928`.*

```sh

DATA_DIR=/export/scisumm/s2orc/s2orc/full
METADATA=$DATA_DIR/metadata

# citation
python filter_ids.py \
    --input-dir $METADATA \
    --output-file citation_candidates_20200705.pkl \
    --njobs 20 \
    --target citation

# paper
python filter_ids.py \
    --input-dir $METADATA \
    --output-file papers_20200705_min10max50.pkl \
    --njobs 20 \
    --target paper \
    --valid-citations citation_candidates_20200705.pkl \
    --min-cite 10 \
    --max-cite 50 \
```

## Create paper ID index for valid citation candidates
Using those two filtered paper information, we create paper ID-to-paper location (in the files) index.
Do this for both `paper` and `citation` modes.

```sh
MODE=paper
for MODE in paper citation
do
    python create_pid_locator.py \
        --input-dir $METADATA \
        --output ${MODE}_pid_location_20200705.tsv \
        --njobs 20 \
        --valid-pids papers_20200705_min10max50.pkl \
        --mode $MODE
done
```

## Extract relevant information
This is the primary part where data instances are formed.
This script launches a data reader that harvests citation contexts by identifying sentences 
that includes citation span locations.

```sh
python construct_triples.py \
    --paper-id-data paper_20200705.pkl \
    --target-pid-location-dict paper_pid_location_20200705.tsv \
    --citation-pid-location-dict citation_pid_location_20200705.tsv \
    --input-dir /export/scisumm/s2orc/s2orc/full/pdf_parses \
    --output-dir extracted_20200705 \
    --njobs 20
```

## Create dataset
Preprocessing is done. Run the following script to combine them all and make splits.

```sh
python create_dataset.py \
    --dump-dir extracted_20200705 \
    --output-dir old_release_full_records \
    --split-ratio 0.9 \
    --overwrite \
    --njobs 60 \
```

The output does NOT contain contribution sentence indices yet. Follow [here](classifier/README.md)
for the details on how to populate the "gold" labels.


