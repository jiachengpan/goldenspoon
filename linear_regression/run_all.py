import os
import glob
import random
import multiprocessing

random.seed(0)

def run_one(filename):
    os.system(f'timeout 600s ./run.sh {filename}')

models = list(glob.glob('output/models/*.pkl', recursive=True))

pool = multiprocessing.Pool(processes=8)
pool.map(run_one, random.sample(models, 500))
