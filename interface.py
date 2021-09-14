import matplotlib.pyplot as plt

import Dataset
import LSM
from functions import *
from interface_func import *

pd.options.display.float_format = '{:.4f}'.format
#______HEADER______
AOI = 'BDunajec'

data_dir, label_dir, lsm_dir, target_shape, LCF_files, LCF_names = header(AOI)

#print(LCF_names)

#rows = target_shape[0]
#drop_list = ['River proximity', 'Curvature plan', 'Curvature',
                #     'Curvature profile', 'TPI', 'Soil texture']

#categorical = ['LC', 'Soil suitability', 'Soil type', 'Soil texture', 'Tectonics']

#numerical = LCF_names
#for i in categorical:
 #   numerical.remove(i)

#dataset_file = 'Datasets\\{}_dataset_tect_norm_LC.fth'.format(AOI)
#DMatrix_file = '{}_DMatrix'.format(AOI)
#raster_out_file = 'LSM\\{}_Pearson_9704.tif'.format(AOI)
#!!!!CODE HERE!!!!

#dmatrixer(dataset_file, target_shape, drop_list, output=DMatrix_file, rows=rows)
#data_to_feather(data_dir, target_shape, output='BDunajec_dataset.fth')
#preds = predictor(target_shape, DMatrix_file, 'Model1', rows=rows, plot=True)
#auc = test_model(rows, 'Model\\Model1', 'Model\\{}_DMatrix'.format(AOI), label_dir, AOI, target_shape, plot=True)
#print(auc)
#raster_save(raster_out_file, preds, target_shape, label_dir, AOI, rows=rows)

#dataset = Dataset.Dataset(data_dir, target_shape, 'BDunajec_dataset.fth', label_dir)
#dataset.create_feather()
#dataset.pearson_matrix(show=True)
#dataset.anova_score(show=True)
#dataset.su_score(show=True)

fs_methods = ['Pearson', 'Anova', 'SU']
boundaries_BD_BD = [[0.511729343, 0.753884009], [0.507618078, 0.749672891], [0.503931830, 0.730482308],
                    [0.507478719, 0.745566130]]
boundaries_BD_R = [[0.449215417, 0.644901493], [0.374304437, 0.558332374], [0.412933700, 0.567603048],
                   [0.376585295, 0.551247105]]
boundaries_R_BD = [[0.462852978, 0.495515977], [0.461333327, 0.491138283], [0.469922922, 0.503463395],
                   [0.462607948, 0.493166988]]
boundaries_R_R = [[0.532897665, 0.767999574], [0.532845502, 0.767924395], [0.517310417, 0.740694454],
                  [0.524947894, 0.759999189]]

cat_boundaries_BD_BD = [[0.5215369189, 0.772500954], [0.52938136, 0.780347323], [0.423304339, .0662392871]]
cat_boundaries_BD_R = [[0.377925925, 0.58631005], [0.386928739, 0.585373415], [0.393967587, 0.560590737]]
cat_boundaries_R_BD = [[0.552893594, 0.788167456], [0.552864139, 0.79204648], [0.529241064, 0.740937478]]
cat_boundaries_R_R = [[0.433344022, 0.575645858], [0.457745976, 0.582995496], [0.438484624, 0.561824099]]

#boundaries = [boundaries_BD_BD, boundaries_BD_R, boundaries_R_BD, boundaries_R_R]
#cat_boundaries = [cat_boundaries_BD_BD, cat_boundaries_BD_R, cat_boundaries_R_BD, cat_boundaries_R_R]

#lsm_instance = LSM.LSM(AOI, 'Roznow', lsm_dir, target_shape)
#lsm_instance.evaluate(fs_methods, cat_boundaries[3], label_dir)
AOI_train = ['BDunajec', 'Roznow']
AOI_test = ['BDunajec', 'Roznow']
titles = [['Biały Dunajec - Biały Dunajec', 'Biały Dunajec - Rożnów'],
          ['Rożnów - Rożnów', 'Rożnów - Biały Dunajec']]

auc_plot(AOI_train, AOI_test, titles, create_target_sets=False)
#data = rio_reader('F:\\Projekt_Staz\\LSM-non_cat\\BDunajec-BDunajec_Pearson_9808.tif', target_shape)
#print(data)