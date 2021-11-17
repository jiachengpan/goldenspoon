#! /bin/bash

set -e
set -x

dates="2019-03-31
       2019-06-30
       2019-09-30
       2019-12-31
       2020-03-31
       2020-06-30
       2020-09-30
       2020-12-31
       2021-03-31
       2021-06-30
       2021-09-30
       "
for end_date in $dates; do
  python main.py $end_date &
done

wait


