#! /bin/bash

debug_mode=N

training_model=randomforest     # ['linear', 'ridge', 'lasso', 'randomforest']
indicator_use_type='stock_dynamic' # ['None', 'industrial_static', 'industrial_dynamic', 'stock_dynamic', 'static_stock_dynamic','all_dynamic']
train_drop_ponit=0.0
test_drop_ponit=0.15

# train_date_len=1
# train_date_list='2021-06-30'

train_date_len=4
train_date_list="2020-09-30 2020-12-31 2021-03-31 2021-06-30"


if [ "${indicator_use_type}" = "None" ]; then
  flag=XY_${training_model}_${train_date_len}_Sample_Drop_${train_drop_ponit}
else
  flag=XY_${training_model}_${indicator_use_type}_${train_date_len}_Sample_Drop_${train_drop_ponit}
fi

if [ "${debug_mode}" = "T" ]; then
  savepath=regress_result/test/${flag}
else
  savepath=regress_result/xy/test_drop_ponit_${test_drop_ponit}/${training_model}/${flag}
fi

echo "flag:"
echo ${flag}
echo "savepath:"
echo ${savepath}

if [ ! -d "${savepath}" ]; then
  mkdir -p ${savepath}
fi

python regress.py \
--data_path=regress_data/past_quater_4/ \
--save_path=${savepath}/ \
--drop_small_change_stock_fortrain=True --train_drop_ponit=${train_drop_ponit} \
--drop_small_change_stock_fortest=True --test_drop_ponit=${test_drop_ponit} \
--training_model=${training_model} \
--indicator_use_type=${indicator_use_type} \
--train_date_list ${train_date_list} \
>& ${savepath}/result.log
