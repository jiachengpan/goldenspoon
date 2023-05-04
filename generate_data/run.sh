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
       2022-03-31
       2022-06-30
       2022-09-30
       2022-12-31
       2023-03-31
       "

# dates="
#        2021-12-31
#        " 
[[ -z $past_quarters ]]   && past_quarters=4
[[ -z $output_dir ]]      && output_dir="/home/devdata/xiaoying.zhang/goldenspoon/v0408/goldenspoon/linear_regression/"
[[ -z $value_threshold ]] && value_threshold=1e9
[[ -z $data_path ]]       && data_path=raw_data

mkdir -p $output_dir
mkdir -p logs

dateflag=230504
run_label_for_priormonth=T
if [ "${run_label_for_priormonth}" = "T" ]; then
  output_dir=${output_dir}/regress_data_${dateflag}-label_one_month_prior/past_quarters_${past_quarters}/
  label_for_priormonth='--label_for_priormonth'
else
  output_dir=${output_dir}/regress_data_${dateflag}-label_base0_month/past_quarters_${past_quarters}/
  label_for_priormonth=''
fi


for end_date in $dates; do
  python -u generate_data.py \
    --path $data_path \
    -ed $end_date \
    --output ${output_dir} \
    --value-threshold ${value_threshold} \
    --past-quarters ${past_quarters} \
    ${label_for_priormonth} \
    | tee logs/gen.end_date.${end_date}.run.$(date -Idate).log &
done
echo "output_dir:"
echo ${output_dir}

wait
