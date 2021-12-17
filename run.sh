#! /bin/bash

set -e
set -x


# dates="
#        2021-03-31
#        "

dates="
       2020-09-30
       2020-12-31
       2021-03-31
       2021-06-30
       2021-09-30
       "


past_quater_number='2'
regress_dir='./regress_data_new_'${past_quater_number}'/'
for end_date in $dates; do
  python main.py $end_date ${regress_dir} ${past_quater_number}&
done

wait
