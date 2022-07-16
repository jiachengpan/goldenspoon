#! /bin/bash


################################################################################
## debug
# train_date_len=1
# train_date_list='2021-06-30'

# train_date_len=2
# train_date_list="2020-09-30 2020-12-31"
# test_date_list="2021-03-31"

# train_date_len=4
# train_date_list="2020-06-30 2020-09-30 2020-12-31 2021-03-31"
# test_date_list="2021-06-30"

# train_date_len=4
# train_date_list="2020-09-30 2020-12-31 2021-03-31 2021-06-30"
# test_date_list="2021-09-30"

train_date_len=6
train_date_list="2020-06-30 2020-09-30 2020-12-31 2021-03-31 2021-06-30 2021-09-30"
test_date_list="2021-12-31"

if [[ $test_suite = "2021-09-30" ]]; then
  train_date_len=4
  train_date_list="2020-09-30 2020-12-31 2021-03-31 2021-06-30"
  test_date_list="2021-09-30"
fi

if [[ $test_suite = "2021-06-30" ]]; then
  train_date_len=4
  train_date_list="2020-06-30 2020-09-30 2020-12-31 2021-03-31"
  test_date_list="2021-06-30"
fi

if [[ $test_suite = "2021-12-31" ]]; then
  train_date_len=6
  train_date_list="2020-06-30 2020-09-30 2020-12-31 2021-03-31 2021-06-30 2021-09-30"
  test_date_list="2021-12-31"
fi

if [[ $test_suite = "2022-03-31" ]]; then
  train_date_len=7
  train_date_list="2020-06-30 2020-09-30 2020-12-31 2021-03-31 2021-06-30 2021-09-30 2021-12-31"
  test_date_list="2022-03-31"
fi

if [[ $test_suite = "2022-06-30" ]]; then
  train_date_len=8
  train_date_list="2020-06-30 2020-09-30 2020-12-31 2021-03-31 2021-06-30 2021-09-30 2021-12-31 2022-03-31"
  test_date_list="2022-06-30"
fi

################################################################################
debug_mode=T
run_with_standardscaler=F
################################################################################
#p0

training_model=$1

modelflag=voting_ada_gdbt_200 # change
[[ -z $repeat_run ]] && repeat_run=1
[[ -z $training_model ]] && training_model=votingclassifier # votingclassifier # gbdtclassifier #adaboostclassifier # votingclassifier     # ['linear', 'ridge', 'lasso', 'randomforest', 'randomforestclssifier', 'adaboostregressor', 'adaboostclassifier']

run_with_label_norm=T
sample_number=1000
data_flag=label_one_month_prior #[label_zero-new,label_zero-pre,label_zero-cur,label_one_month_prior]
indicator_use_type=all_with_premonth_label #['all','all_with_premonth_label', 'industrial_static', 'industrial_dynamic', 'stock_dynamic', 'static_stock_dynamic','all_dynamic']

label_type=class # ['regress','class']

#p1
#data_path=regress_data_220408-${data_flag}/past_quarters_4/
data_path=regress_data_220605-${data_flag}/past_quarters_4/

date=`date +%Y-%m-%d`
#result_flag=regress_result_220408-${modelflag}-sample${sample_number}-${test_date_list}
#result_flag="result-${date}-sample${sample_number}-${test_date_list}"
result_flag="result-top2-sample${sample_number}-${test_date_list}"

train_drop_ponit=0.0
test_drop_ponit=0.0
################################################################################
if [ "${indicator_use_type}" = "None" ]; then
  flag=XY_${label_type}_${training_model}_${train_date_len}_Sample_Drop_${train_drop_ponit}
else
  flag=XY_${label_type}_${training_model}_${indicator_use_type}_${train_date_len}_Sample_Drop_${train_drop_ponit}
fi


if [ "${debug_mode}" = "T" ]; then
    #savepath=${result_flag}/test/${data_flag}/test_drop_ponit_${test_drop_ponit}/${training_model}/${flag}
    savepath=${result_flag}/test/${data_flag}/test_drop_ponit_${test_drop_ponit}/${flag}
else
    savepath=${result_flag}/xy/${data_flag}/test_drop_ponit_${test_drop_ponit}/${flag}
fi

echo "--start"
# echo "flag:"
# echo ${flag}
echo "savepath: $savepath"

if [ "${run_with_standardscaler}" = "T" ]; then
  savepath=${savepath}/data_standardscaler/
  data_standardscaler='--data_standardscaler'
else
  savepath=${savepath}/no_standardscaler/
  data_standardscaler=''
fi

if [ "${run_with_label_norm}" = "T" ]; then
  savepath=${savepath}/label_norm/
  label_norm='--label_norm'
else
  savepath=${savepath}/label_original/
  label_norm=''
fi

if [ ! -d "${savepath}" ]; then
  mkdir -p ${savepath}
fi

  python regress.py \
  --data_path=${data_path} \
  --save_path=${savepath}/ \
  --drop_small_change_stock_fortrain=True --train_drop_ponit=${train_drop_ponit} \
  --drop_small_change_stock_fortest=True --test_drop_ponit=${test_drop_ponit} \
  --training_model=${training_model} \
  --indicator_use_type=${indicator_use_type} \
  --train_date_list ${train_date_list} \
  --test_date_list ${test_date_list} \
  --sample_number ${sample_number} \
  --repeat_run ${repeat_run} \
  ${label_norm} \
  ${data_standardscaler} \
  >& ${savepath}/result.log

echo "--end"
echo "flag:"
echo ${flag}
echo "savepath:"
echo ${savepath}
