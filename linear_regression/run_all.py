import os
import re
import glob
import random
import subprocess
import multiprocessing

random.seed(0)

def run_one(filename):
    for test_suite in (
            '2023-03-31',
            #'2021-06-30',
            #'2021-09-30',
            #'2021-12-31',
            #'2022-03-31',
            ):
        repeat  = 30
        timeout = 300 * repeat

        os.system(f'test_suite={test_suite} repeat_run={repeat} timeout {timeout}s ./run.sh {filename}')

models = list(glob.glob('output/voting_models/*.pkl', recursive=True))


only = [
    'top3_select5.1217',
    #'top3_select5.1398',
    #'top3_select4.440',
    #'top3_select5.1361',
    #'top3_select4.697',
    #'top3_select5.1228',
    'top3_select4.796',
    #'top3_select4.811',
    #'top3_select3.293',
    #'top3_select2.82',
    ]

pattern = re.compile(r'(%s)' % ('|'.join(['%s.pkl' % s for s in only])))
models = [m for m in models if pattern.search(m)]

pool = multiprocessing.Pool(processes=4)
pool.map(run_one, models)

