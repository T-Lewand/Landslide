import numpy as np
import pandas as pd
import gc
import warnings
from functions import *
from model_func import *
from bayes_opt import BayesianOptimization
from datetime import datetime

import xgboost as xgb
import contextlib

AOI = 'BDunajec'
data_dir, label_dir, lsm_dir, target_shape, LCF_files, LCF_names = header(AOI)

start_time = datetime.now().time()
log_text = '{}\nBayesian Optimization - START {}\n'.format('*'*130, start_time)
logger('Bayesian_opt', log_text)



optimizer = BayesianOptimization(xgb_func, {
                                     'step': (200, 500),
                                     'max_depth': (2, 9),
                                     'eta': (0.005, 0.1),
                                     'gamma': (0.001, 10.0),
                                     'min_child_weight': (0, 20),
                                     'max_delta_step': (0, 10),
                                     'subsample': (0.4, 1.0),
                                     'colsample_bytree': (0.2, 1.0),
                                     'scale_pos_weight': (1, 7)
                                    })

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    optimizer.maximize(init_points=2, n_iter=10, acq='ei', xi=0.0)

print(optimizer.max)
stop_time = datetime.now().time()
log_text = '{}\nBayesian Optimization - STOP {}\n{}\n'.format('-'*130, stop_time, optimizer.max)
logger('Bayesian_opt', log_text)
