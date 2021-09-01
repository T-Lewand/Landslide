from functions import *
from sklearn.metrics import accuracy_score, precision_score, recall_score

#______HEADER______
AOI = 'Roznow'
BDunajec_shape = (7588, 5222)
Roznow_shape = (6817, 5451)

directory = os.getcwd()
os.chdir(directory[0:-7])
lsm_dir = '{}\\LSM\\'.format(os.getcwd())
label_dir = '{}\\Data\\{}_label\\'.format(os.getcwd(), AOI)
if AOI == 'Roznow':
    target_shape = Roznow_shape
else:
    target_shape = BDunajec_shape


fs_style = ["Pearson", "Anova", 'SU']

bounds = [0.207703916, 0.335228327, 0.352940051, 0.530057288,
          0.207579361, 0.345725527, 0.342183318, 0.498040532,
          0.158193273, 0.248042688, 0.268814092, 0.396719414]
bounds = np.array(bounds).reshape(len(fs_style), 4)

multi_lsm_stats(AOI, fs_style, bounds, lsm_dir, label_dir, target_shape)

