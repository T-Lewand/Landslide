from Utilities import *
from functions import *

from bayes_opt import BayesianOptimization
import xgboost as xgb


def dmatrixer(dataset_file: str, target_shape: tuple, drop_list: list = None, output: str = None, rows: int = 0):
    """Tworzy zestaw danych do xgboost w formacie DMatrix i zapisuje na dysku."""

    if drop_list is None:
        dataset_flat = pd.read_feather(dataset_file)
    else:
        dataset_flat = pd.read_feather(dataset_file).drop(drop_list, axis=1)
    if rows == 0:
        X = dataset_flat.to_numpy()
    else:
        X = dataset_flat.to_numpy()[0:(rows * target_shape[1]), :]

    del dataset_flat
    gc.collect()

    D_matrix = xgb.DMatrix(X)
    if output is None:
        D_matrix.save_binary('Model\\D_matrix')
    else:
        D_matrix.save_binary(('Model\\{}'.format(output)))


def model_dev(AOI: str, step: int, parameters: dict, drop_list: list, target_shape, nowy_model: bool=True,
              single_AOI=True, crop: float = 0, silent=False, plot=False, memory_friendly=True, early_stop=False,
              comment: str=''):

    """Funkcja uczy model XGBoost na podstawie danych w folderze Model. Jeśli folder Model jest pusty, tworzy zestaw
    danych na podstawie datasetów w folderze Datasets.

        AOI - obszar na, których danych model ma się uczyć
        step - ilość drzew
        parameters - parametry modelu
        drop_list - lista warstw, które nie wezmą udziału w trenowaniu modelu
        target_shape - wymiary obrazów na danym AOI
        nowy_model - jeśli True funkcja uczy nowy model. W innych wypadku jedynie ocenia istniejący już model
        single_AOI - jeśli True funkcja uczy model i testuje na jednym AOI. W innym wypadku, uczy na BDunajec, a testuje
            na Roznow
        silent - jeśli True funkcja nie daje komunikatów
        plot - jeśli True funkcja tworzy wykres istotności cech
        memory_friendly - Jeśli True, zrzuca zmienne, które nie są już potrzebne do dalszych obliczeń. Lepiej zostawić
            True chyba, że dostępne więcej niż 16 Gb RAMu
        early_stop - Jeśli True model trenuje się z wlączoną funkcją early_stop
        comment - dowolny komentarz do logów modeli

    Funkcja dodatkowo na koniec zapisuje bierzące ustawienia modelu w folderze Model oraz w ogólnym logu w folerze
    Logs"""

    model_files = os.listdir('{}\\Model'.format(os.getcwd()))
    if silent is False:
        print(model_files)

    #Data load
    if nowy_model is True:
        if silent is False:
            print("!!!Czytam dane...")

        if 'D_train' in model_files:
            D_train = xgb.DMatrix('Model\\D_train')
        else:
            if silent is False:
                print("!!!Brak danych. Robię...")

            if 'X_train' not in model_files:
                if silent is False:
                    print('   Brak train test set. Robię...')
                # Tworzy zestaw treningowy i testowy
                create_model_data(AOI, target_shape, drop_list, single_AOI, crop=crop)

            #Robi Dmacierz dla danych treningowych
            D_train = create_Dmatrix(train=True)
            if silent is False:
                print('   Train DMatrix gotowe')


            gc.collect()
        # Uczy model
        start_time = datetime.now().time()
        if silent is False:
            print("!!!Tworze model...")
            print(start_time)

        if early_stop is False:
            model = xgb.train(parameters, D_train, step)
        else:
            D_test = xgb.DMatrix('Model\\D_test')
            model = xgb.train(parameters, D_train, step, evals=[(D_test, 'auc')], early_stopping_rounds=10)

        end_time = datetime.now().time()
        if silent is False:
            print(end_time)

        train_time = (end_time.hour*60+end_time.minute+end_time.second/60) -\
                     (start_time.hour*60+start_time.minute+start_time.second/60)
        model.save_model('Model\\Model1')
        model.dump_model('Model\\Model1.dump')
        # Koniec uczenia
        if memory_friendly is True:
            if silent is False:
                print('   Model gotowy, zrzucam D_train...')

            del D_train
            gc.collect()
    else:
        if silent is False:
            print("Wczytuje model")

        model = xgb.Booster()
        model.load_model('Model\\Model1')

    if silent is False:
        print('!!!Ocena modelu')

    if 'D_test' in model_files:
        D_test = xgb.DMatrix('Model\\D_test')
        Y_test = pd.read_feather('Model\\Y_test').set_index('index').to_numpy()

    else:
        if silent is False:
            print('   Brak D_test. Robię...')
        # Robi D macierz danych testowych
        D_test = create_Dmatrix(train=False)
        Y_test = pd.read_feather('Model\\Y_test').set_index('index').to_numpy()
        if silent is False:
            print('   Zrzucam X_test set')


    if plot is True:
        xgb.plot_importance(model, importance_type='gain')
        plt.show()
    # Testowanie i ocena modelu
    preds_test = model.predict(D_test)
    auc_score_test = roc_auc_score(Y_test, preds_test)
    preds_test = preds_test.round()
    accuracy = accuracy_score(Y_test, preds_test)
    precision = precision_score(Y_test, preds_test)
    recall = recall_score(Y_test, preds_test)
    F1_score = 2 * (precision * recall)/(precision + recall)
    if memory_friendly is True:
        del D_test
        del Y_test
        gc.collect()

    D_train = xgb.DMatrix('Model\\D_train')
    Y_train = pd.read_feather('Model\\Y_train').set_index('index').to_numpy()

    preds_train = model.predict((D_train))
    auc_score_train = roc_auc_score(Y_train, preds_train)
    if silent is False:
        print('Accuracy = {:.4f}'.format(accuracy))
        print('Precision = {:.4f}'.format(precision))
        print('Recall = {:.4f}'.format(recall))
        print('F1_score = {:.4f}'.format(F1_score))
        print('AUC score for test set = {:.4f}'.format(auc_score_test))
        print('AUC score for train set = {:.4f}'.format(auc_score_train))
        print('!!!Koniec!!!')

    log = "Model {}\n{}\nTrain set {}\n---\nTrain time = {} min\nStep = {}\n---\nDrop List\n{}\n---\nParameters\n{}\n" \
          "---\nAccuracy = {:.4f}\nPrecision = {:.4f}\nRecall = {:.4f}\nF1_score = {:.4f}\nAUC (test set) = {:.4f}\n" \
          "AUC (train set) = {:.4f}\n/-/-/-/-/\n\n".format(datetime.now(), comment, AOI, train_time,  step,
                                                           str(drop_list), str(parameters), accuracy, precision, recall,
                                                           F1_score, auc_score_test, auc_score_train)
    logger('Model_log', log)
    logger('Model_setting', log, 'w+', 'Model')

    return auc_score_test, auc_score_train


def tuner(AOI: str, step: int, parameters, drop_list, parameter_tuning: str, parameter_tune_val: list, rows,
          double_test=False, AOI2: str=None, comment: str=''): #dump func
    print('{}'.format(parameter_tuning))
    for i in parameter_tune_val:
        target_shape = (7588, 5222)
        parameters[parameter_tuning] = i
        auc_test, auc_train = model_dev(AOI, step, parameters, drop_list, target_shape, single_AOI=False, silent=True,
                                        comment=comment)
        text = 'For {}---AUC TEST = {:.4f}---AUC TRAIN = {:.4f}'.format(i, auc_test, auc_train)
        print(text)
        if double_test is True:
            label_dir2 = '{}\\Data\\{}_label\\'.format(os.getcwd(), AOI2)
            target_shape2 = (6817, 5451)
            auc2 = test_model(rows, 'Model\\Model1', 'Model\\{}_DMatrix'.format(AOI2), label_dir2, AOI2, target_shape2)
            print('{:.4f}'.format(auc2))

        log = 'For {}---AUC TEST = {:.4f}---AUC TRAIN = {:.4f}\n'.format(i, auc_test, auc_train)
        with open('Logs\\{}.txt'.format(parameter_tuning), 'a+') as log_file:
            log_file.write(log)
            log_file.close()


def xgb_func(step,
          max_depth,
          eta,
          gamma,
          min_child_weight,
          max_delta_step,
          subsample,
          colsample_bytree,
          scale_pos_weight
         ):
    """Funkcja do optymalizacji, zwraca wynik AUC, modelu wytrenowanego na zadanych parametrach"""

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

    dtrain = xgb.DMatrix('Model\\D_train')
    dtest = xgb.DMatrix('Model\\D_test')
    Y_test = pd.read_feather('Model\\Y_test').set_index('index').to_numpy()

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

def create_model_data(AOI, target_shape, drop_list, single_AOI, crop: float=0, crop_BD: float=0.2):
    if single_AOI is True:
        train_test_set(AOI, target_shape, drop_list)
    else:
        train_test_set2(AOI, drop_list, test=False, crop=crop)
        if AOI == 'Roznow':
            train_test_set2('Bdunajec', drop_list, test=True, crop=crop_BD)
        else:
            train_test_set2('Roznow', drop_list, test=True)

def create_Dmatrix(train=True):

    if train is True:
        type = 'train'
    else:
        type = 'test'

    X_set = pd.read_feather('Model\\X_{}'.format(type)).set_index('index').to_numpy()
    Y_set = pd.read_feather('Model\\Y_{}'.format(type)).set_index('index').to_numpy()
    D_matrix = xgb.DMatrix(X_set, label=Y_set)
    D_matrix.save_binary('Model\\D_{}'.format(type))
    del X_set
    del Y_set
    gc.collect()
    return D_matrix
