import matplotlib.pyplot as plt

from Utilities import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split


def rio_reader(file_directory: str, data_shape: tuple):
    """Funkcja zwraca wskazany w file_directory plik .tif jako macierz numpy w określonym w data_shape rozmiarze"""
    raster_file = rio.open(file_directory)
    raster_data = raster_file.read(
        out_shape=(raster_file.count, data_shape[0], data_shape[1]),
        resampling=Resampling.bilinear)[0]  # tu w przyszłości zmienić ten shape żeby wymagał mniejszej ingerencji
    raster_file.close()
    return raster_data


def read_data(data_directory: str, data_shape: tuple, file_name: list = None):
    """Funkcja zwraca cały zestaw danych w folderze data_directory jako 3 wymiarowy numpy array, gdzie pierwszy wymiar
    określa numer zmiennej. Funkcja zwraca tylko podane zmienne w file_name jeśli określone. file_name musi być listą"""
    dataset = []

    if file_name is None:
        LCF_files = list_files(data_directory)
        for i in LCF_files:
            raster_data = rio_reader(data_directory+i, data_shape)
            if str(raster_data.dtype.name) == 'float32':
                raster_data[raster_data < -3.4e+38] = np.nan
                if 'DEM' in i:
                    raster_data = raster_data.round(decimals=2)
                if 'proximity' in i:
                    raster_data = raster_data.round(decimals=2)

                raster_data = raster_data.round(decimals=6)

            dataset.append(raster_data)
    else:
        for i in file_name:
            raster_data = rio_reader(data_directory + i, data_shape)

            if str(raster_data.dtype.name) == 'float32':
                raster_data[raster_data < -3.4e+38] = np.nan
                if 'DEM' in i:
                    raster_data = np.around(raster_data, decimals=2)
                    pass
                elif 'proximity' in i:
                    raster_data = raster_data.round(decimals=2)
                    pass
                else:
                    pass
                    raster_data = raster_data.round(decimals=6)

            dataset.append(raster_data)

    dataset = np.stack(dataset, axis=0)

    return dataset


def flatten(dataset):
    """Przetwarza zestaw danych dwówymiarowych na jednowymiarowe"""
    variables = []
    variables_count = dataset.shape[0]
    variable_len = dataset.shape[1]*dataset.shape[2]
    for i in dataset:
        variable = i.flatten()
        variables.append(variable)

    variables = np.array(variables).reshape(variables_count, variable_len)
    return variables


def PCC_matrix(file_name: str, dataset): #dump func
    """Dataset musi być flat. Jeden wiersz jedna zmienna."""
    PCC = np.corrcoef(dataset)
    np.savetxt('{}.txt'.format(file_name), PCC, delimiter=';')
    print("Done")


def train_test_set(AOI: str, target_shape: tuple, drop_list: list = None):
    """Dzieli zestaw danych na treningowe i testowe. Zapisuje na dysku jako DataFrame w formacie feather"""

    from sklearn.model_selection import train_test_split

    if drop_list is None:
        dataset_flat = pd.read_feather('Datasets\\{}_dataset_LC.fth'.format(AOI))
    else:
        dataset_flat = pd.read_feather('Datasets\\{}_dataset_LC.fth'.format(AOI)).drop(drop_list, axis=1)
    print(dataset_flat.columns.tolist())

    label_dir = '{}\\Data\\{}_label\\'.format(os.getcwd(), AOI)
    label = rio_reader(label_dir + '{}_Label.tif'.format(AOI), target_shape).flatten()
    label = pd.DataFrame(label, columns=['Label'])

    X = dataset_flat
    Y = label
    del dataset_flat
    gc.collect()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=12)

    X_train.reset_index().to_feather('Model\\X_train')
    X_test.reset_index().to_feather('Model\\X_test')
    Y_train.reset_index().to_feather('Model\\Y_train')
    Y_test.reset_index().to_feather('Model\\Y_test')
    del X_train
    del Y_train
    del X_test
    del Y_test
    gc.collect()


def train_test_set2(AOI: str, drop_list: list = None, test=True, crop: float = 0):
    """Funkcja do tworzenie zestawów test/train z dwóch obszarów.
    Jeśli test True to tworzy zestaw testowy jeśli False - treningowy.
    Zapisuje na dysku jako DataFrame w formacie feather.
    """

    from sklearn.model_selection import train_test_split

    if AOI == 'Roznow':
        target_shape = (6817, 5451)
    else:
        target_shape = (7588, 5222)

    if drop_list is None:
        dataset_flat = pd.read_feather('Datasets\\{}_dataset_LC.fth'.format(AOI))
    else:
        dataset_flat = pd.read_feather('Datasets\\{}_dataset_LC.fth'.format(AOI)).drop(drop_list, axis=1)
    print(dataset_flat.columns.tolist())

    label_dir = '{}\\Data\\{}_label\\'.format(os.getcwd(), AOI)
    label = rio_reader(label_dir + '{}_Label.tif'.format(AOI), target_shape).flatten()
    label = pd.DataFrame(label, columns=['Label'])

    X = dataset_flat
    Y = label
    if crop > 0:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=crop, random_state=12)
        X = X_train
        Y = Y_train

    del dataset_flat
    del label
    gc.collect()

    if test is True:
        X.reset_index().to_feather('Model\\X_test')
        Y.reset_index().to_feather('Model\\Y_test')
    else:
        X.reset_index().to_feather('Model\\X_train')
        Y.reset_index().to_feather('Model\\Y_train')

    del X
    del Y
    gc.collect()


def predictor(target_shape: tuple, D_matrix_file: str, Model_file: str, rows: int = 0, plot=False):
    """Zwraca prawdobodobieństwa wystąpienia zjawiska jako macierz numpy."""

    import xgboost as xgb

    D_matrix = xgb.DMatrix('Model\\{}'.format(D_matrix_file))
    model = xgb.Booster()
    model.load_model('Model\\{}'.format(Model_file))
    if rows == 0:
        preds = model.predict(D_matrix).reshape(target_shape)
    else:
        preds = model.predict(D_matrix).reshape(rows, target_shape[1])
    if plot is True:
        fig, ax = plt.subplots()
        heatmap = sns.heatmap(preds)

        plt.show()

    return preds


def test_model(rows, model_file: str, D_test_file: str, label_dir, AOI, target_shape, plot=False):
    """Funkcja testuje model"""

    import xgboost as xgb

    model = xgb.Booster()
    model.load_model(model_file)
    D_test = xgb.DMatrix(D_test_file)
    Y_test = rio_reader(label_dir + '{}_Label.tif'.format(AOI), target_shape)[0:rows, :].flatten()
    Y_test = pd.DataFrame(Y_test, columns=['Label'])

    preds = model.predict(D_test)
    fpr, tpr, treshold = roc_curve(Y_test, preds)
    fpr = np.array(fpr).reshape(-1,1)[::2, :][::2, :][::2, :][::2, :]
    tpr = np.array(tpr).reshape(-1,1)[::2, :][::2, :][::2, :][::2, :]

    fpr_tpr = np.append(fpr, tpr, axis=1)
    pixel_ratios = pd.DataFrame(fpr_tpr, columns=['fpr', 'tpr'])
    auc_score = roc_auc_score(Y_test, preds)

    if plot is True:
        print(treshold)
        fig, ax = plt.subplots()
        auc = sns.lineplot(data=pixel_ratios, x='fpr', y='tpr')
        ref_data = pd.DataFrame(np.array([0, 0, 1, 1]).reshape((2, 2)), columns=['fpr', 'tpr'])
        ref_line = sns.lineplot(data=ref_data, x='fpr', y='tpr')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        ax.axis('equal')
        plt.show()

    return auc_score


def lsm_read(AOI: str, AOI_test, lsm_directory: str, fs_style: str, raster_shape: tuple=None):
    """Funkcja czyta mapy podatnośći osuwiskowej jako macierz numpy"""
    lsm_files = list_files(lsm_directory)
    for i in lsm_files:
        if '{}-{}_{}'.format(AOI, AOI_test, fs_style) in i:
            lsm = i

    if raster_shape is not None:
        lsm_data = rio_reader('{}{}'.format(lsm_directory, lsm), raster_shape)
        return lsm_data
    else:
        return lsm


def lsm_stats(AOI: str, AOI_test: str,  fs_style: str, boundaries: list, lsm_directory: str, label_directory: str,
              raster_shape: tuple, df_return=False):
    """Funkcja analizuje dokładność rastra podatności osuwiskowej.

        AOI - obszar zainteresowania,
        AOI_test - obszar, na którym testowano model
        fs_style - nazwa metody użytej do feature selection
        boundaries - granica poniżej której wartości badanego rastra przyjmują 0, a powyżej 1
        lsm_directory - ścieżka do folderu z rastrami LSM
        label_directory - scieżka do folderu z etykietami
        raster_shape - wymiary rastra w pikselach

    Zapisuje wyniki w pliku csv, kolumny to granice, a wiersze wartości poszczególnego parametru (accuracy, precision,
    recall i F1"""

    label_data = read_data(label_directory, raster_shape)[0]
    label_data = label_data.flatten()
    label_data[label_data == 255] = 0  # UWAGA z tym

    lsm = lsm_read(AOI, AOI_test,  lsm_directory, fs_style)

    indexes = ['ACCURACY', 'PRECISION', 'RECALL', 'F1']
    stats_df = pd.DataFrame(np.zeros([len(indexes), len(boundaries)]), columns=boundaries, index=indexes)

    for i in boundaries:
        lsm_data = rio_reader('{}{}'.format(lsm_directory, lsm), raster_shape)
        lsm_data = lsm_data.flatten()
        lsm_data[lsm_data >= i] = 1
        lsm_data[lsm_data < i] = 0

        accuracy = accuracy_score(label_data, lsm_data)
        precision = precision_score(label_data, lsm_data)
        recall = recall_score(label_data, lsm_data)
        F1_score = 2 * (precision * recall) / (precision + recall)

        stats_df[i][indexes[0]] = accuracy
        stats_df[i][indexes[1]] = precision
        stats_df[i][indexes[2]] = recall
        stats_df[i][indexes[3]] = F1_score

    stats_df.to_csv('{}{}'.format(lsm_directory, '{}-{}_{}_Stats.csv'.format(AOI, AOI_test, fs_style)))
    if df_return is True:
        return stats_df


def multi_lsm_stats(AOI: str, fs_style: list, boundaries, lsm_directory: str, label_directory: str,
                    raster_shape: tuple):
    """Funkcja robi to samo co lsm_stats tylko, że zwraca DataFrame dla wszystkich fs_style"""
    indexes = ['ACCURACY', 'PRECISION', 'RECALL', 'F1']
    merged_df = pd.DataFrame(np.zeros([len(indexes), 1]), columns=[0], index=indexes)

    for k in range(len(fs_style)):
        df = lsm_stats(AOI, fs_style[k], boundaries[k], lsm_directory, label_directory, raster_shape, df_return=True)
        merged_df = pd.concat([merged_df, df], axis=1)
    merged_df.drop(columns=[merged_df.columns.tolist()[0]])
    print(merged_df)
    merged_df.to_csv('{}{}_Stats.csv'.format(lsm_directory, AOI))


def get_ratios(label, predicted):
    """Funkcja zwraca wartość False Positive Rate i True Positive Rate jako DataFrame z kolumnami ['fpr','tpr']"""
    fpr, tpr, treshold = roc_curve(label, predicted)
    fpr_last, tpr_last = np.array(fpr[-1]).reshape(-1, 1), np.array(tpr[-1]).reshape(-1, 1)
    fpr = np.array(fpr).reshape(-1, 1)[::100000, :]
    fpr = np.append(fpr, fpr_last, axis=0)
    tpr = np.array(tpr).reshape(-1, 1)[::100000, :]
    tpr = np.append(tpr, tpr_last, axis=0)
    fpr_tpr = np.append(fpr, tpr, axis=1)
    pixel_ratios = pd.DataFrame(fpr_tpr, columns=['fpr', 'tpr'])
    return pixel_ratios


def auc_scoring(label, predicted, return_ratios=False):
    """Funkcja zwraca wartość AUC."""
    pixel_ratios = get_ratios(label, predicted)
    auc_score = roc_auc_score(label, predicted)

    if return_ratios is True:
        return auc_score, pixel_ratios
    else:
        return auc_score


def auc_plot_old(AOI: str, label_directory: str, lsm_directory: str, raster_shape: tuple, fs_styles: list = ['Pearson'],
             reference_line=True):
    """Funkcja robi wykres ROC w liczbie takiej ile elementów w fs_style. Funkcja podczytuje dane sama, wystarczy dać
    jej scieżki do folderów z danymi."""
    fig, ax = plt.subplots()
    plt.title("Receiver Operating Characteristic - {}".format(AOI))
    label = read_data(label_directory, raster_shape)[0]
    label[label == 255] = 0
    label = label.flatten()
    for i in range(len(fs_styles)):
        predicted = lsm_read(AOI, lsm_directory, fs_styles[i], raster_shape).flatten()
        auc = auc_scoring(label, predicted)
        ratios = get_ratios(label, predicted)
        roc = plt.plot(ratios['fpr'], ratios['tpr'], label='{}, AUC = {:.4f}'.format(fs_styles[i], auc))

    if reference_line is True:
        ref_data = pd.DataFrame(np.array([0, 0, 1, 1]).reshape((2, 2)), columns=['fpr', 'tpr'])
        ref_line = plt.plot(ref_data['fpr'], ref_data['tpr'], linestyle='--', label='Reference line')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='lower right')
    plt.show()


def lcf_stats(AOI1, AOI2):
    data_directory1, label_dir, lsm_dir, target_shape1, LCF_files1, LCF_names = header(AOI1, reset_origin=False)
    data_directory2, label_dir, lsm_dir, target_shape2, LCF_files2, LCF_names = header(AOI2, reset_origin=False)
    indexes = ['Mean', 'Max', 'Min', 'St.dev']
    stats_merged = pd.DataFrame([0, 0, 0, 0], index=indexes)

    for i in range(len(LCF_files1)):
        print(LCF_names[i])
        data1 = read_data(data_directory1, target_shape1, file_name=[LCF_files1[i]])
        mean1 = data1.mean()
        maxim1 = data1.max()
        minim1 = data1.min()
        st_dev1 = data1.std()
        stats1 = pd.Series([mean1, maxim1, minim1, st_dev1], index=indexes, name='{}_{}'.format(AOI1, LCF_names[i]))
        data2 = read_data(data_directory2, target_shape2, file_name=[LCF_files2[i]])
        mean2 = data2.mean()
        maxim2 = data2.max()
        minim2 = data2.min()
        st_dev2 = data2.std()
        stats2 = pd.Series([mean2, maxim2, minim2, st_dev2], index=indexes, name='{}_{}'.format(AOI2, LCF_names[i]))
        stats_df = pd.concat([stats1, stats2], axis=1)
        stats_merged = pd.concat([stats_merged, stats_df], axis=1)

    stats_merged.to_csv('{}LCF_Stats.csv'.format('Data\\'))

def target_set():
    """Tworzy Target datasets dla obu obszarów w folderze Datasets"""
    import Dataset

    data_dir, label_dir, lsm_dir, target_shape, LCF_files, LCF_names = header('BDunajec', reset_origin=False)
    set_BD = Dataset.Dataset(data_dir, target_shape, 'BDunajec_dataset.fth', label_dir)
    set_BD.label_feather()
    data_dir, label_dir, lsm_dir, target_shape, LCF_files, LCF_names = header('Roznow', reset_origin=False)
    set_R = Dataset.Dataset(data_dir, target_shape, 'Roznow_dataset.fth', label_dir)
    set_R.label_feather()

def auc_plot(AOI_train: list, AOI_test: list, titles: list, image_name: str, create_target_sets=False, show=True):
    """Tworzy wykresy krzywej ROC dla obszarów podanych w AOI_train i AOI_test.
    AOI_train: lista obszarów treningowych
    AOI_test: lista obszarów testowych
    titles: tytuły poszczególnych wykresów, i liczba musi być równa sumie elementów list AOI_train i AOI_test
    """

    import Dataset
    import LSM

    if create_target_sets:
        target_set()

    fig, ax = plt.subplots(len(AOI_train), len(AOI_test))

    fig.suptitle("Receiver Operating Characteristic")
    iterator_k = 0

    for k in AOI_train:
        iterator_j = 0
        data_dir, label_dir, lsm_dir, target_shape, LCF_files, LCF_names = header(k, reset_origin=False)
        try:
            label_set = Dataset.Dataset(label_dir, target_shape, '{}_Target.fth'.format(k), label_dir)
            label_train = label_set.read_feather()
        except:
            print("No Target datasets, create first")
            exit()

        if len(AOI_train) > 1 and k == AOI_train[-1]:
            AOI_test.reverse()
        for j in AOI_test:
            data_dir, label_dir, lsm_dir, target_shape_test, LCF_files, LCF_names = header(j, reset_origin=False)
            label_set = Dataset.Dataset(label_dir, target_shape, '{}_Target.fth'.format(j), label_dir)
            label_test = label_set.read_feather()

            lsm = LSM.LSM(k, j, lsm_dir, target_shape_test)
            fs_methods = ['Pearson', 'Anova', 'SU']
            ref_data = pd.DataFrame(np.array([0, 0, 1, 1]).reshape((2, 2)), columns=['fpr', 'tpr'])
            if len(AOI_test) == 1 and len(AOI_train) == 1:
                axes = ax
            else:
                if len(ax.shape) == 1 and len(AOI_train) == 1:
                    axes = ax[iterator_j]
                elif len(ax.shape) == 1 and len(AOI_test) == 1:
                    axes = ax[iterator_k]
                else:
                    axes = ax[iterator_k, iterator_j]

            ref_line = axes.plot(ref_data['fpr'], ref_data['tpr'], linestyle='--', label='Reference line')

            for i in fs_methods:
                lsm_set = lsm.read(i).flatten()
                auc, ratios = auc_scoring(label_test, lsm_set, return_ratios=True)
                roc = axes.plot(ratios['fpr'], ratios['tpr'], label='{}, AUC = {:.4f}'.format(i, auc))

                axes.set_xlabel('False Positive Rate')
                axes.set_ylabel('True Positive Rate')
                axes.set_title(titles[iterator_k][iterator_j])
                axes.legend(loc='lower right')

            iterator_j += 1

        iterator_k += 1

    fig.tight_layout()
    fig.set_size_inches((12.0, 9.0))
    plt.subplots_adjust(wspace=0.2, hspace=0.3, top=0.9)
    plt.savefig("{}.jpg".format(image_name), dpi=2000)
    if show:
        plt.show()