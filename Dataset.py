from functions import *
import ITMO_FS


class Dataset:
    def __init__(self, directory, shape, name, label_directory):
        """

        :param directory: ścieżka do folderu z LCF
        :param shape: rozmiar rastra
        :param name: Pełna nazwa zestawu danych (np. BDunajec_dataset.fth
        :param label_directory: scieżka do folderu label
        """
        self.directory = directory
        self.label_dir = label_directory
        self.shape = shape
        self.name = name
        self.aoi = name.split('_')[0]
        self.LCF_files = list_files(self.directory)
        self.columns = []
        for i in self.LCF_files:
            self.columns.append(i.split('_')[1].split('.')[0])

        self.categorical = ['LC', 'Soil suitability', 'Soil type', 'Soil texture', 'Tectonics']
        self.numerical = self.columns
        for i in self.categorical:
            self.numerical.remove(i)

    def create_feather(self):
        """Podczytuje obrazy w folderze data_dir  i gromadzi je w jedną macierz, gdzie
        każda kolumna to kolejna zmienna."""

        dataset = read_data(self.directory, self.shape)
        dataset_flat = flatten(dataset).T

        dataset_df = pd.DataFrame(dataset_flat, columns=self.columns)

        dataset_df.to_feather('Datasets\\{}'.format(self.name))

    def read_feather(self):
        """Czyta i zwraca zestaw danych zapisanych w formacie feather"""
        dataset = pd.read_feather(self.name)
        return dataset

    def label_read(self):
        """Czyta i zwraca raster z etykietami jako numpy array wymiary (n_samples, 1)"""
        label_data = read_data(self.label_dir, self.shape)[0]
        label_data = label_data.flatten().reshape(-1, 1)
        return label_data

    def anova_score(self, show=False):
        """Zapisuje wyniki ANOVA dla każdej zmiennej"""

        dataset = self.read_feather()
        label_data = self.label_read()
        pd.options.display.float_format = '{:.3f}'.format
        selection = SelectKBest(score_func=f_classif)
        fit = selection.fit(dataset, label_data)
        scores = pd.DataFrame(fit.scores_.T, index=[self.columns], columns=['score'])
        scores.sort_values(by=['score'], inplace=True, ascending=False)
        scores.to_csv('Feature_selection\\{}_Anova_score.csv'.format(self.name))
        if show is True:
            print(scores)

    def su_score(self, show=False):
        """Zapisuje wartości Symmetrical Uncertainty dla każdej zmiennej"""

        label_data = self.label_read()
        dataset = self.read_feather().to_numpy().T
        pd.options.display.float_format = '{:.3f}'.format
        scores = ITMO_FS.su_measure(dataset, label_data.flatten())
        scores = pd.DataFrame([scores], columns=self.columns)
        scores.to_csv('Feature_selection\\{}_SU_score.csv'.format(self.name))
        if show is True:
            print(scores)

    def pearson_matrix(self, show=False):
        """Zapisuje macierz korelacji Pearsona"""

        dataset = self.read_feather()
        label = self.label_read()
        dataset_labeled = np.append(dataset, label, axis=1)
        pearson_matrix = np.corrcoef(dataset_labeled)
        columns = self.columns.append('Target')
        pearson_matrix = pd.DataFrame([pearson_matrix], columns=columns, index=columns)
        pearson_matrix.to_csv('Feature_selection\\{}_Pearson.txt'.format(self.name))

        target_pearson = pearson_matrix['Target'].drop(index=['Target'])
        target_pearson.to_csv('Feature_selection\\{}_Target_Pearson.txt'.format(self.name))

        if show is True:
            fig, ax = plt.subplots(tight_layout=False)
            cbar_kws = dict(shrink=1)
            heatmap = sns.heatmap(pearson_matrix, annot=True, fmt='.2f', xticklabels=True, yticklabels=True,
                                  cmap='rocket', cbar_kws=cbar_kws)
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=50, ha='left')
            heatmap.xaxis.tick_top()
            heatmap.xaxis.set_label_position('bottom')

            ax.set_position([0.15, 0.14, 0.62, 0.65])
            plt.subplots_adjust(top=0.8, left=0.15)
            plt.show()

    def train_test_set(self, drop_list: list = None, single=True, test=True, crop: float = 0, test_size=0.3):
        """Dzieli zestaw danych na treningowe i testowe. Zapisuje na dysku jako DataFrame w formacie feather"""

        if drop_list is None:
            dataset_flat = pd.read_feather('Datasets\\{}'.format(self.name))
        else:
            dataset_flat = pd.read_feather('Datasets\\{}'.format(self.name)).drop(drop_list, axis=1)
        print(dataset_flat.columns.tolist())

        label = self.label_read()
        label = pd.DataFrame(label, columns=['Target'])

        X = dataset_flat
        Y = label
        del dataset_flat
        gc.collect()
        if single is False and crop > 0.0:
            test_size = crop
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=12)

        if single is True:
            X_train.reset_index().to_feather('Model\\X_train')
            X_test.reset_index().to_feather('Model\\X_test')
            Y_train.reset_index().to_feather('Model\\Y_train')
            Y_test.reset_index().to_feather('Model\\Y_test')
        else:
            X = X_train
            Y = Y_train
            if test is True:
                set_name = 'test'
            else:
                set_name = 'train'

            X.reset_index().to_feather('Model\\X_{}'.format(set_name))
            Y.reset_index().to_feather('Model\\Y_{}'.format(set_name))

        del X_train
        del Y_train
        del X_test
        del Y_test
        gc.collect()


    def train_test_set2(self, AOI: str, drop_list: list = None, test=True, crop: float = 0):
        """Funkcja do tworzenie zestawów test/train z dwóch obszarów.
        Jeśli test True to tworzy zestaw testowy jeśli False - treningowy.
        Zapisuje na dysku jako DataFrame w formacie feather.
        """


        if drop_list is None:
            dataset_flat = pd.read_feather('Datasets\\{}_dataset_LC.fth'.format(AOI))
        else:
            dataset_flat = pd.read_feather('Datasets\\{}_dataset_LC.fth'.format(AOI)).drop(drop_list, axis=1)
        print(dataset_flat.columns.tolist())

        label_dir = '{}\\Data\\{}_label\\'.format(os.getcwd(), AOI)
        label = rio_reader(label_dir + '{}_Label.tif'.format(AOI), self.shape).flatten()
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