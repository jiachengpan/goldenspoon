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

[[ -z $past_quarters ]]   && past_quarters=4
[[ -z $output_dir ]]      && output_dir="data/past_quarters_${past_quarters}/"
[[ -z $value_threshold ]] && value_threshold=1e9
[[ -z $data_path ]]       && data_path=raw_data

mkdir -p $output_dir
mkdir -p logs

for end_date in $dates; do
  python -u generate_data.py \
    --path $data_path \
    -ed $end_date \
    --output ${output_dir} \
    --value-threshold ${value_threshold} \
    --past-quarters ${past_quarters} \
    | tee logs/gen.end_date.${end_date}.run.$(date -Idate).log &
done

wait
