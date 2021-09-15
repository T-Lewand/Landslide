import Dataset
import LSM
from functions import *
from interface_func import *

pd.options.display.float_format = '{:.4f}'.format
#______HEADER______
AOI = 'BDunajec'

data_dir, label_dir, lsm_dir, target_shape, LCF_files, LCF_names = header(AOI)

fs_methods = ['Pearson', 'Anova', 'SU']

#Zakresy do klasyfikacji LSM
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

auc_plot([AOI_train[0]], AOI_test, [titles[0]], 'test', create_target_sets=False)
