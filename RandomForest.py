import Dataset
from Utilities import *
from functions import *
from sklearn.ensemble import RandomForestClassifier
import Dataset

AOI = 'BDunajec'
data_dir, label_dir, lsm_dir, target_shape, LCF_files, LCF_names = header(AOI)

drop_list2 = ['Aspect', 'CTI', 'Curvature plan', 'Curvature profile', 'Curvature', 'TPI', 'River proximity']
drop_list_pearson = ['River proximity', 'Curvature plan', 'Curvature', 'Curvature profile', 'TPI', 'Soil texture']
drop_list_pearson2 = ['River proximity', 'Curvature plan', 'Curvature', 'Curvature profile', 'TPI', 'Soil texture',
                      'IMI', 'Aspect', 'Soil type']
drop_list_Anova = ['Aspect', 'Curvature plan', 'Curvature profile', 'Curvature', 'Flow direction', 'LC1', 'LC2', 'LC3',
                   'LC4', 'LC5', 'Tectonics', 'River proximity', 'TPI']
drop_list_SU = ['CTI', 'Curvature plan', 'Curvature', 'DEM', 'Fault proximity', 'IMI', 'River proximity',
                'Thrust proximity', 'TPI']


params = {'criterion':'gini', 'max_depth': None, 'min_samples_split':2, 'min_samples_leaf':1,
          'min_weight_fraction_leaf':0.0, 'max_features':'auto', 'max_leaf_nodes':None, 'min_impurity_decrease':0.0,
          'min_impurity_split':None, 'bootstrap':True, 'oob_score':False, 'n_jobs':None, 'random_state':None, 'verbose':0,
          'warm_start':False, 'class_weight':None, 'ccp_alpha':0.0, 'max_samples':None}

calsiffier = RandomForestClassifier(n_estimators=100, **params)

data = Dataset.Dataset(data_dir, target_shape, 'BDunajec_dataset.fth', label_dir)
data_set = data.read_feather()
data.train_test_set(drop_list=drop_list_pearson)