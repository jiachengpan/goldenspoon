#! /bin/bash

set -e
set -x

result_path="result-top2-sample1000"
dates="2021-06-30 2021-09-30 2021-12-31 2022-03-31"
dates="2023-03-31"

for date in $dates; do
  ./summarize.py $(ls -d ${result_path}-${date}) \
    --models output/voting_models \
    --output summary.voting.${date}.xlsx \
    --include-stocks
done
