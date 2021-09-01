from datetime import datetime
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.enums import Resampling


from matplotlib import pyplot as plt
import seaborn as sns

import os
import gc


def header(AOI, reset_origin=True):
    """Funkcja nagłówek. Napisana do uproszecznia kodu. Nie trzeba teraz robić nagłówka dużego w każdym pliku .py
    Wystarczy wywołać funkcje"""

    BDunajec_shape = (7588, 5222)
    Roznow_shape = (6817, 5451)
    if reset_origin is True:
        directory = os.getcwd()
        os.chdir(directory[0:-7])

    data_dir = '{}\\Data\\{}_LCFs\\'.format(os.getcwd(), AOI)
    label_dir = '{}\\Data\\{}_label\\'.format(os.getcwd(), AOI)
    lsm_dir = '{}\\LSM\\'.format(os.getcwd())
    if AOI == 'Roznow':
        target_shape = Roznow_shape
    else:
        target_shape = BDunajec_shape

    LCF_files = list_files(data_dir)
    LCF_names = []
    for i in LCF_files:
        LCF_names.append(i.split('_')[1].split('.')[0])

    return data_dir, label_dir, lsm_dir, target_shape, LCF_files, LCF_names


def list_files(data_directory: str):
    """Funkcja zwraca nazwy obrazów .tif znajdujących się w danym folderze data_directory jako listę stringów"""

    LCF_files = []
    all_files = os.listdir(data_directory)
    for i in all_files:
        if '.tif.' in i:
            pass
        else:
            if '.tif' in i:
                LCF_files.append(i)

    return LCF_files


def logger(log_file: str, log_text: str, type: str = 'a+', folder: str = 'Logs'):
    with open('{}\\{}.txt'.format(folder, log_file), type) as log_file:
        log_file.write(log_text)
        log_file.close()
