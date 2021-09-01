from functions import *
from model_func import *
from sklearn.metrics import accuracy_score, precision_score, recall_score
import gc
from bayes_opt import BayesianOptimization
from datetime import datetime


#______HEADER______
AOI = 'Roznow'
data_dir, label_dir, lsm_dir, target_shape, LCF_files, LCF_names = header(AOI)

rows = target_shape[0]
#zmienne wejściowe:
nowy_model = True
parameters = {'booster': 'gbtree', 'verbosity': 2, 'eval_metric': 'auc', 'seed': 3,
              'eta': 0.043685, 'max_depth': 2, 'gamma': 8.361176,
              'gpu_id': 0, 'tree_method': 'gpu_hist', 'objective': 'binary:logistic', 'max_delta_step': 0.88685,
              'subsample': 0.81576, 'colsample_bytree': 0.75138, 'scale_pos_weight': 6.23676, 'min_child_weight': 8.07124}
step = 423
drop_list2 = ['Aspect', 'CTI', 'Curvature plan', 'Curvature profile', 'Curvature', 'TPI', 'River proximity']
drop_list_pearson = ['River proximity', 'Curvature plan', 'Curvature', 'Curvature profile', 'TPI', 'Soil texture']
drop_list_pearson2 = ['River proximity', 'Curvature plan', 'Curvature', 'Curvature profile', 'TPI', 'Soil texture',
                      'IMI', 'Aspect', 'Soil type']
drop_list_Anova = ['Aspect', 'Curvature plan', 'Curvature profile', 'Curvature', 'Flow direction', 'LC1', 'LC2', 'LC3',
                   'LC4', 'LC5', 'Tectonics', 'River proximity', 'TPI']
drop_list_SU = ['CTI', 'Curvature plan', 'Curvature', 'DEM', 'Fault proximity', 'IMI', 'River proximity',
                'Thrust proximity', 'TPI']


drop_list = drop_list_pearson
comment = 'Two AOI, Pearson2, Bayesian opt for Pearson'

model_dev(AOI, step, parameters, drop_list, target_shape, single_AOI=False, comment=comment, crop=0)
#create_model_data(AOI, target_shape, drop_list, single_AOI=False, crop=0.2)
#create_Dmatrix(train=True)
#create_Dmatrix(train=False)

