from functions import *

class LSM:
    def __init__(self, AOI_train, AOI_test, directory, shape):
        """

        :param name: Nazwa pliku tif
        :param directory: folder z plikami tif
        :param shape: wymiary rastra (takie jak wymiary DEM)
        """
        self.AOI_train = AOI_train
        self.AOI_test = AOI_test
        self.name = '{}-{}'.format(AOI_train, AOI_test)
        self.directory = directory
        self.shape = shape

    def read(self, fs_method):
        """Funkcja czyta mapy podatnośći osuwiskowej jako macierz numpy
        fs_method: nazwa metody feature selection
        """

        lsm_files = list_files(self.directory)
        for i in lsm_files:
            if '{}_{}'.format(self.name, fs_method) in i:
                lsm = i
        try:
            lsm_data = rio_reader('{}{}'.format(self.directory, lsm), self.shape)
            return lsm_data
        except:
            print("Nie istniejący plik: {}_{}".format(self.name, fs_method))

    def evaluate_one(self, fs_method: str, boundaries: list, label_directory: str, df_return=False):
        """Funkcja analizuje dokładność pojedyńczego rastra podatności osuwiskowej.

            fs_style - nazwa metody użytej do feature selection
            boundaries - lista z wartościami granic poniżej której wartości badanego rastra przyjmują 0, a powyżej 1
            label_directory - scieżka do folderu z etykietami

        Zapisuje wyniki w pliku csv, kolumny to granice, a wiersze - wartości poszczególnego parametru (accuracy,
        precision, recall i F1)
        """

        label_data = read_data(label_directory, self.shape)[0]
        label_data = label_data.flatten()
        label_data[label_data == 255] = 0  # UWAGA z tym

        lsm_data = self.read(fs_method)

        indexes = ['ACCURACY', 'PRECISION', 'RECALL', 'F1']
        stats_df = pd.DataFrame(np.zeros([len(indexes), len(boundaries)]), columns=boundaries, index=indexes)

        for i in boundaries:
            lsm_clasiff = lsm_data.flatten().copy()
            lsm_clasiff[lsm_clasiff >= i] = 1
            lsm_clasiff[lsm_clasiff < i] = 0

            accuracy = accuracy_score(label_data, lsm_clasiff)
            precision = precision_score(label_data, lsm_clasiff)
            recall = recall_score(label_data, lsm_clasiff)
            F1_score = 2 * (precision * recall) / (precision + recall)

            stats_df[i][indexes[0]] = accuracy
            stats_df[i][indexes[1]] = precision
            stats_df[i][indexes[2]] = recall
            stats_df[i][indexes[3]] = F1_score

        stats_df.to_csv('{}{}'.format(self.directory,
                                      'Stats\\{}_{}.csv'.format(self.name, fs_method)))
        if df_return is True:
            return stats_df

    def evaluate(self, fs_methods: list, boundaries, label_directory: str):
        """Funkcja robi to samo co evaluate_one tylko, że zwraca DataFrame dla wszystkich fs_style
        boundaries: lista list granic"""

        indexes = ['ACCURACY', 'PRECISION', 'RECALL', 'F1']
        merged_df = pd.DataFrame(np.zeros([len(indexes), 1]), columns=[0], index=indexes)

        for k in range(len(fs_methods)):
            df = self.evaluate_one(fs_methods[k], boundaries[k], label_directory, df_return=True)
            merged_df = pd.concat([merged_df, df], axis=1)

        merged_df.drop(columns=[merged_df.columns.tolist()[0]])
        print(merged_df)
        merged_df.to_csv('{}Stats\\{}_Stats.csv'.format(self.directory, self.name))