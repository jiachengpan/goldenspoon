#! /bin/bash

set -e
set -x

dates="
       2020-09-30
       2020-12-31
       2021-03-31
       2021-06-30
       2021-09-30
       2021-12-31
       "

[[ -z $past_quarters ]] && past_quarters=4
[[ -z $output_dir ]]    && output_dir="../linear_regression/regress_data/past_quater_${past_quarters}/"

mkdir -p $output_dir

for end_date in $dates; do
  python main.py ${end_date} ${output_dir} ${past_quarters} &
done

wait
