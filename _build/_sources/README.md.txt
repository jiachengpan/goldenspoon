# GoldenSpoon
## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
    - [Generate Data](#generate-data)
    - [Linear Regression](#linear-regression)

## Introduction
[web introduction](http://172.16.15.11:8193)
## Usage
### Generate Data  
1. cd ./generate_data folder.  
2. modify the **run.sh** to set parameters: `dates`, `past_quater_number`, `regress_dir`.  
3. bash run.sh  

### Linear Regression  
1. cd ./linear_regression folder.  
2. modify the **run.sh** and **regress.py** to set parameters. if no, use the default value.  
3. bash run.sh  
4. for details about parameter settings, see [**regress.py**](./linear_regression/regress.py)
```python
    parser.add_argument(
        "--data_path",
        default='regress_data/past_quater_4/',
        help="Provide regression data path.")
    parser.add_argument(
        "--save_path",
        default='regress_result/past_quater_4/',
        help="Provide regression result path.")

    parser.add_argument(
        "--indicator_use_list",
        nargs="+",
        help="Set the indicators you want to train.",
        default=[])

    parser.add_argument(
        "--train_date_list",
        help="Set the enddate data for the training set. e.g. ['2021-03-31','2021-06-30'].",
        nargs="+",
        default=['2021-06-30'])
    parser.add_argument(
        "--test_date_list",
        nargs="+",
        help="Set the enddate data for the test set.",
        default=['2021-09-30'])

    parser.add_argument(
        "--n_month_predict",
        type=int,
        default=3,
        help="N month to predict.")
    parser.add_argument(
        "--predict_mode", 
        default="predict_changerate_price",
        help="Using 'predict_changerate_price' is currently the best approach.",
        choices=['predict_changerate_price', 'predict_absvalue_price'])

    parser.add_argument(
        "--drop_small_change_stock_fortrain",
        type=bool,
        default=True,
        help="Drop small change stock for train data.")
    parser.add_argument(
        "--train_drop_ponit",
        type=float,
        default=0.15,
        help="drop ponit for task 'drop_small_change_stock_fortrain'.")
    parser.add_argument(
        "--drop_small_change_stock_fortest",
        type=bool,
        default=True,
        help="Drop small change stock for test data.")
    parser.add_argument(
        "--test_drop_ponit",
        type=float,
        default=0.15,
        help="drop ponit for task 'drop_small_change_stock_fortest'.")
```