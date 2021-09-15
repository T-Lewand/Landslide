from Utilities import *
from functions import *
import Dataset
import xgboost as xgb


class Model:
    def __init__(self, name, parameters, AOI_train, AOI_test=None, test_size=0.3, silent=False):
        """

        :param name: Nazwa modelu (np. Model.model)
        :param parameters: Hiperparametry modelu
        :param AOI_train: Obszar treningowy
        :param AOI_test: Obszar testowy, nie podawać jeśli ten sam co treningowy
        :param test_size: Część danych przeznaczona jako dane testowe
        :param silent: Komunikaty, jeśli True nie wyświetli komunikatów
        """
        self.name = name
        self.parameters = parameters
        self.AOI_train = AOI_train
        self.AOI_test = AOI_test
        data_dir_train, label_dir_train, lsm_dir, target_shape_train, LCF_files, LCF_names = header(AOI_train,
                                                                                                    reset_origin=False)
        self.dataset_train = Dataset.Dataset(data_dir_train, target_shape_train, '{}_dataset.fth'.format(AOI_train),
                                             label_dir_train)
        if AOI_test is not None:
            self.single = False
            data_dir_test, label_dir_test, lsm_dir, target_shape_test, LCF_files, LCF_names = header(AOI_test,
                                                                                                     reset_origin=False)
            self.dataset_test = Dataset.Dataset(data_dir_test, target_shape_test, '{}_dataset.fth'.format(AOI_test),
                                                label_dir_test)
        else:
            self.single = True
        self.test_size = test_size
        self.silent = silent

    def create_data(self, drop_list=[], crop=0):
        """
        Tworzy zestaw danych X_train, Y_train, X_test i Y_test
        """
        if self.single:
            self.dataset_train.train_test_set(drop_list, crop, self.test_size)
        else:
            self.dataset_train.train_test_set(drop_list, crop, self.test_size, single=self.single, test=False)
            self.dataset_test.train_test_set(drop_list, crop, self.test_size, single=self.single, test=True)

    def create_Dmatrix(self, train):
        """
        Przetwarza dane X, Y na dane w formacie DMatrix do XGBoosta
        train: jeśli True tworzy DMatrix z danymi treningowymi jeśli False z testowymi
        """
        if train is True:
            set_type = 'train'
        else:
            set_type = 'test'

        X_set = pd.read_feather('Model\\X_{}'.format(set_type)).set_index('index').to_numpy()
        Y_set = pd.read_feather('Model\\Y_{}'.format(set_type)).set_index('index').to_numpy()
        D_matrix = xgb.DMatrix(X_set, label=Y_set)
        D_matrix.save_binary('Model\\D_{}'.format(set_type))
        del X_set
        del Y_set
        gc.collect()
        return D_matrix

    def initiate(self, drop_list=[], crop=0):
        """
        Inicjuje dane do trenowania
        """
        model_files = os.listdir('{}\\Model'.format(os.getcwd()))
        if 'D_train' in model_files:
            self.D_train = xgb.DMatrix('Model\\D_train')
        else:
            if 'X_train' in model_files and 'Y_train'in model_files:
                self.D_train = self.create_Dmatrix(train=True)
            else:
                self.create_data(drop_list, crop)
                self.D_train = self.create_Dmatrix(train=True)

        if 'D_test' not in model_files:
            if 'X_test' in model_files and 'Y_test' in model_files:
                self.create_Dmatrix(train=False)
            else:
                self.create_data(drop_list, crop)
                self.create_Dmatrix(train=False)

    def train(self, step, drop_list=[], crop=0, early_stop=False):
        """
        Trenuje model i zapisuje go
        :param step: ilość drzew
        :param drop_list: lista LCF do odrzucenia
        :param crop: stopień przycięcia zestawu danych
        :param early_stop: Jeśli True model jest trenowany z funkcją early stop
        """
        self.initiate(drop_list, crop)
        self.step_ = step
        start_time = datetime.now().time()
        if self.silent is False:
            print("!!!Tworze model...")
            print(start_time)

        if early_stop is False:
            self.model = xgb.train(self.parameters, self.D_train, self.step_)
        else:
            D_test = xgb.DMatrix('Model\\D_test')
            self.model = xgb.train(
                self.parameters, self.D_train, self.step_, evals=[(D_test, 'auc')], early_stopping_rounds=10
            )

        end_time = datetime.now().time()
        if self.silent is False:
            print(end_time)

        self.train_time_ = (end_time.hour*60+end_time.minute+end_time.second/60) -\
                     (start_time.hour*60+start_time.minute+start_time.second/60)
        self.model.save_model('Model\\{}'.format(self.name))
        self.model.dump_model('Model\\{}.dump'.format(self.name))

    def test(self):
        """
        Testuje istniejący model
        """
        Y_test = pd.read_feather('Model\\Y_test').set_index('index').to_numpy()
        D_test = xgb.DMatrix('Model\\D_test')
        preds_test = self.model.predict(D_test)
        self.auc_score_test_ = roc_auc_score(Y_test, preds_test)
        preds_test = preds_test.round()
        self.accuracy_ = accuracy_score(Y_test, preds_test)
        self.precision_ = precision_score(Y_test, preds_test)
        self.recall_ = recall_score(Y_test, preds_test)
        self.F1_score_ = 2 * (self.precision_ * self.recall_) / (self.precision_ + self.recall_)

        Y_train = pd.read_feather('Model\\Y_train').set_index('index').to_numpy()

        preds_train = self.model.predict((self.D_train))
        self.auc_score_train_ = roc_auc_score(Y_train, preds_train)
        if self.silent is False:
            print('Accuracy = {:.4f}'.format(self.accuracy_))
            print('Precision = {:.4f}'.format(self.precision_))
            print('Recall = {:.4f}'.format(self.recall_))
            print('F1_score = {:.4f}'.format(self.F1_score_))
            print('AUC score for test set = {:.4f}'.format(self.auc_score_test_))
            print('AUC score for train set = {:.4f}'.format(self.auc_score_train_))

    def log(self, comment, drop_list):
        """
        Tworzy log o parametrach modelu, czasu trenowania i odrzuconych LCF i wyniki testu
        :param comment: Komentarz dotyczący modelu
        :param drop_list: Lista LCF do odrzucenia
        """
        log = "Model {}\n{}\nTrain set {}\n---\nTrain time = {} min\nStep = {}\n---\nDrop List\n{}\n---\nParameters" \
              "\n{}\n---\nAccuracy = {:.4f}\nPrecision = {:.4f}\nRecall = {:.4f}\nF1_score = {:.4f}" \
              "\nAUC (test set) = {:.4f}\nAUC (train set) = {:.4f}\n/-/-/-/-/\n\n"\
            .format(datetime.now(), comment, self.AOI_train, self.train_time_, self.step_, str(drop_list),
                    str(self.parameters), self.accuracy_, self.precision_, self.recall_, self.F1_score_,
                    self.auc_score_test_, self.auc_score_train_)
        logger('Model_log', log)
        logger('Model_setting', log, 'w+', 'Model')

    def develop(self, step, drop_list=[], crop=0, early_stop=False, comment='No comment'):
        """
        Full model development
        :param step: ilość drzew
        :param drop_list: lista LCF do odrzucenia
        :param crop: stopień przycięcia zestawu danych
        :param early_stop: Jeśli True model jest trenowany z funkcją early stop
        :param comment: Komentarz dotyczący modelu
        """
        self.train(step, drop_list, crop, early_stop)
        self.test()
        self.log(comment, drop_list)

    def optimize_func(self,
                      step,
                      max_depth,
                      eta,
                      gamma,
                      min_child_weight,
                      max_delta_step,
                      subsample,
                      colsample_bytree,
                      scale_pos_weight
                      ):
        """Funkcja do optymalizacji, zwraca wynik F1, modelu wytrenowanego na zadanych parametrach"""

        paramt = {
                  'max_depth': round(max_depth),
                  'gamma': gamma,
                  'eta': eta,
                  'subsample': max(min(subsample, 1), 0),
                  'colsample_bytree': max(min(colsample_bytree, 1), 0),
                  'min_child_weight': min_child_weight,
                  'max_delta_step': max_delta_step,
                  'scale_pos_weight': scale_pos_weight,
                  'seed': 3,
                  'tree_method': 'gpu_hist',
                  'eval_metric': 'auc',
                  'booster': 'gbtree',
                  'objective': 'binary:logistic'
                  }
        try:
            dtrain = xgb.DMatrix('Model\\D_train')
            dtest = xgb.DMatrix('Model\\D_test')
            Y_test = pd.read_feather('Model\\Y_test').set_index('index').to_numpy()
        except:
            print('Brak zestawów danych w folderze Model. Uruchom najpierw metodę initiate()')
            exit()

        model = xgb.train(paramt, dtrain, round(step))
        preds_test = model.predict(dtest)
        val_score = roc_auc_score(Y_test, preds_test)

        preds_test = preds_test.round()
        accuracy = accuracy_score(Y_test, preds_test)
        precision = precision_score(Y_test, preds_test)
        recall = recall_score(Y_test, preds_test)
        F1_score = 2 * (precision * recall) / (precision + recall)
        mean_score = (accuracy + precision + recall + 2*val_score)/5
        log = '{}\n   Step = {}  AUC = {:.4f}  ACC = {:.4f}  PREC = {:.4f}  REC = {:.4f}\n   Parameters:\n   {}\n'\
            .format('-'*130, round(step), val_score, accuracy, precision, recall, paramt)
        logger('Bayesian_opt', log)

        return F1_score

    def bayesian_opt(self, step: tuple, max_depth: tuple, eta: tuple, gamma: tuple, min_child_weight: tuple,
                     max_delta_step: tuple, subsample: tuple, colsample_bytree: tuple, scale_pos_weight: tuple):
        """
        Szuka optymalnych parametrów modelu wykorzystując twierdzenie Bayesa. Wyniki zapisywane są w logach w folderze
        Logs
        :param step: zakres wartości step
        :param max_depth: zakres wartości max_depth
        :param eta: zakres wartości eta
        :param gamma: zakres wartości gamma
        :param min_child_weight: zakres wartości min_child_weight
        :param max_delta_step: zakres wartości max_delta_step
        :param subsample: zakres wartości subsample
        :param colsample_bytree: zakres wartości colsample_bytree
        :param scale_pos_weight: zakres wartości scale_pos_weight
        :return: Aktualizuje atrybut parameters o nowe zoptymalizowane wartość
        """
        from bayes_opt import BayesianOptimization
        import warnings

        start_time = datetime.now().time()
        log_text = '{}\nBayesian Optimization - START {}\n'.format('*' * 130, start_time)
        logger('Bayesian_opt', log_text)

        optimizer = BayesianOptimization(self.optimize_func, {
            'step': step,
            'max_depth': max_depth,
            'eta': eta,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'max_delta_step': max_delta_step,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'scale_pos_weight': scale_pos_weight
        })

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            optimizer.maximize(init_points=2, n_iter=10, acq='ei', xi=0.0)

        self.parameters.update(optimizer.max['params'])
        if self.silent is False:
            print(optimizer.max)

        stop_time = datetime.now().time()
        log_text = '{}\nBayesian Optimization - STOP {}\n{}\n'.format('-' * 130, stop_time, optimizer.max)
        logger('Bayesian_opt', log_text)