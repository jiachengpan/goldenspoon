import goldenspoon
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Run!')
parser.add_argument('end_date', type=str, help='')

args = parser.parse_args()

db  = goldenspoon.Database('data')
ind = goldenspoon.Indicator(db, end_date=args.end_date)

database = goldenspoon.compute_indicators(ind)
print(database.describe())
database.to_pickle(f'indicators.{args.end_date}.pickle')
