#! /bin/bash

set -e
set -x

dates="
       2020-03-31
       2020-06-30
       2020-09-30
       2020-12-31
       2021-03-31
       2021-06-30
       2021-09-30
       2021-12-31
       "

[[ -z $past_quarters ]] && past_quarters=4
[[ -z $output_dir ]]    && output_dir="data/past_quarters_${past_quarters}/"

mkdir -p $output_dir

for end_date in $dates; do
  python -u generate_data.py \
    -ed $end_date \
    --output ${output_dir} \
    --past-quarters $past_quarters &

  if [[ -z $first ]]; then
    wait
    first=1
  fi
done

wait
