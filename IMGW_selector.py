import pandas as pd
import numpy as np

file_dir = "F:\\Projekt_Staz\\Data\\IMGW\\"
file_name = "s_m_d_2020.csv"
file_name2 = "o_m_2020.csv"
file = open(file_dir + file_name, 'r')
file2 = open(file_dir + file_name2, 'r')
data = file.readlines()
data2 = file2.readlines()
stations = ['BIELSKO-BIAŁA', 'ZAKOPANE', 'KASPROWY WIERCH', 'NOWY SĄCZ', 'KRAKÓW-BALICE']
stations2 = ['RATUŁÓW', 'GUBAŁÓWKA', 'BIAŁKA TATRZAŃSKA', 'SZAFLARY', 'KOŚCIELISKO-KIRY', 'ŁOPUSZNA']
#WYBIERANIE LOGÓW DLA STACJI
selected_stations = []
selected_stations2 = []
selected_file = open(file_dir + "opady_wybrane.txt", 'w+')
selected_file2 = open(file_dir + "opady_wybrane2.txt", 'w+')

for i in stations:
    index = 0
    for j in data:
        if i in j:
            selected_stations.append(j)
            selected_file.write(j)
        index += 1

for i in stations2:
    index = 0
    for j in data2:
        if i in j:
            selected_stations2.append(j)
            selected_file2.write(j)
        index += 1

selected_file.close()
selected_file2.close()

#SUMOWANIE OPADÓW
selected_file = open(file_dir + "opady_wybrane2.txt", 'r')
selected_data = np.genfromtxt(selected_file, delimiter=',')

suma_opadow_month = selected_data[:, 4].reshape(6, 12)
suma_opadow_year = np.zeros((6, 1))
for i in range(suma_opadow_month.shape[0]):
    suma_opadow_year[i] = np.sum(suma_opadow_month[i])
selected_file.close()

suma_opadow_file = open(file_dir + 'suma_opadow2.txt', 'a+')
data_string = []
for i in range(len(suma_opadow_year)):
    data_string.append("{},{:.2f}".format(stations2[i], suma_opadow_year[i][0]))

print(data_string)
for i in data_string:
    suma_opadow_file.write(i + '\n')

suma_opadow_file.close()
