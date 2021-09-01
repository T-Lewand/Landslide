from Utilities import *
from functions import *

from sklearn.preprocessing import MinMaxScaler

def nanseeker(data, columns):
    for i in columns:
        print('{}---MIN = {:.4f}, MAX = {:.4f}'.format(i, data[i].min(), data[i].max()))

def nanner(data, column, nan_value):
    data[column].replace(nan_value, np.nan)

def normalize(data, columns):
    """Przerabia zmienne liczbowe do zakresu wartości od 0 do 1, gdzie 1 to maksymalna wartość cechy, a 0 minimalna"""


    scaler = MinMaxScaler()
    for i in columns:
        data[i] = scaler.fit_transform(data[i].to_numpy().reshape(-1, 1))
    return data


def one_hot_encoder(AOI, columns):
    """Funkcja in progress. Na razie działa tylko na LC"""
    data = pd.read_feather('Datasets\\{}_dataset.fth'.format(AOI))
    LC = data[columns[0]]
    LC_dum = pd.get_dummies(LC).rename(columns={1: 'LC1', 2: 'LC2', 3: 'LC3', 4: 'LC4', 5: 'LC5'})
    data = data.drop(columns=['LC'])
    data = pd.concat([data, LC_dum], axis=1)
    data.to_feather('Datasets\\{}_dataset_LC.fth'.format(AOI))

def raster_save(output_raster: str, data, target_shape: tuple, label_dir: str, AOI: str, rows: int=0):
    """Funkcja zapisuje podaną macierz data jako obraz tiff z georeferencją taką jak obraz Label w label_dir"""
    label_file = rio.open(label_dir+'{}_Label.tif'.format(AOI))
    label_crs = label_file.crs
    label_trans = label_file.transform
    if rows == 0:
        rows = target_shape[0]

    kwargs = {'driver': 'GTiff', 'height': rows, 'width': target_shape[1], 'count': 1, 'dtype': rio.float32,
              'transform': label_trans}
    output_file = rio.open(output_raster, 'w', crs=label_crs, **kwargs)
    output_file.write(data, 1)