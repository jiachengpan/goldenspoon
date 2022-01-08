#! /bin/bash

training_model=ridge
train_drop_ponit=0.1

# train_date_len=1
# train_date_list='2021-06-30'

train_date_len=4
train_date_list="2020-09-30 2020-12-31 2021-03-31 2021-06-30"

flag=XY_${training_model}_${train_date_len}_Sample_Drop_${train_drop_ponit}
echo ${flag}

savepath=regress_result/${flag}
if [ ! -d "${savepath}" ]; then
  mkdir ${savepath}
fi

python regress.py \
--data_path=regress_data/past_quater_4/ \
--save_path=${savepath}/ \
--drop_small_change_stock_fortrain=True --train_drop_ponit=${train_drop_ponit} \
--drop_small_change_stock_fortest=True --test_drop_ponit=0.15 \
--training_model=${training_model} \
--train_date_list ${train_date_list} \
>& ${savepath}/result.log



