from functions import *
from interface_func import *

pd.options.display.float_format = '{:.4f}'.format
#______HEADER______
AOI = 'BDunajec'
data_dir, label_dir, lsm_dir, target_shape, LCF_files, LCF_names = header(AOI)

#print(LCF_names)

rows = target_shape[0]
drop_list = ['River proximity', 'Curvature plan', 'Curvature',
                     'Curvature profile', 'TPI', 'Soil texture']

categorical = ['LC', 'Soil suitability', 'Soil type', 'Soil texture', 'Tectonics']

numerical = LCF_names
for i in categorical:
    numerical.remove(i)

dataset_file = 'Datasets\\{}_dataset_tect_norm_LC.fth'.format(AOI)
DMatrix_file = '{}_DMatrix'.format(AOI)
raster_out_file = 'LSM\\{}_Pearson_9704.tif'.format(AOI)
#!!!!CODE HERE!!!!

#dmatrixer(dataset_file, target_shape, drop_list, output=DMatrix_file, rows=rows)
#data_to_feather(data_dir, target_shape, output='BDunajec_dataset.fth')
#preds = predictor(target_shape, DMatrix_file, 'Model1', rows=rows, plot=True)
#auc = test_model(rows, 'Model\\Model1', 'Model\\{}_DMatrix'.format(AOI), label_dir, AOI, target_shape, plot=True)
#print(auc)
#raster_save(raster_out_file, preds, target_shape, label_dir, AOI, rows=rows)

string = "BDunajec_dataset.fth"
print(string.split('_')[0])