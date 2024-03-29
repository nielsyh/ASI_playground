import datetime
import ftplib
import matplotlib

import matplotlib.pyplot as plt
from builtins import enumerate
import os

import pandas as pd
import numpy as np
from datetime import time
from os import listdir, path
import cv2
import data.data_visuals
import metrics
import pvlib_playground
import features
from tqdm import tqdm
import calendar
import pickle

from os import listdir
from os.path import isfile, join

def filter_0_list_LS(actual, pred):
    res_act, res_pred = [], []

    for i in range(0, len(actual)):
        new_actual, new_pred = filter_0_list(actual[i], pred[i])
        res_act.append(new_actual)
        res_pred.append(new_pred)

    return res_act, res_pred


def filter_0_list(actual, pred):    # remove 0's
    new_actual = []
    new_pred = []

    for idx, val in enumerate(actual):
        if val != 0:
            new_actual.append(val)
            new_pred.append(pred[idx])

    return new_actual, new_pred


def getColor_binairy(N, idx):
    import matplotlib as mpl
    c = 'binary'
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))

def getColor_racket(N, idx):
    import matplotlib as mpl
    c = 'jet'
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))


def get_folders_ann():
    folders = ls_dir_folder('results 26 march/ann single/')
    names = build_name(folders)
    return folders, names


def get_folders_rf():
    folders = ls_dir_folder('results_18march/single/rf/')
    names = build_name(folders)
    return folders, names

def get_folders_lstm():
    folders = ls_dir_folder('results 26 march/lstm single/')
    names = build_name(folders)
    return folders, names


def get_folders_best():
    folders = [
        # 'persistence',
               'prem results/LSTM 5 IMG/LSTM_SEQUENCE_epochs_40_sequence_5CAM_1_img_Truepredhor_',
               'prem results/LSTM 10 IMG/LSTM_SEQUENCE_epochs_40_sequence_10CAM_1_img_Truepredhor_',
               'prem results/LSTM 10 NOIMG/LSTM_BETA_SEQUENCE_epochs_40CAM_1_sequence_10predhor_',
               'prem results/RF 60 IMG/RF SEQUENCE PREM__sequence_60CAM_1_img_Truepredhor_',
               'prem results/RF 60 NOIMG/RF SEQUENCE PREM__sequence_60CAM_1_img_Falsepredhor_',
               'prem results/RF 120 IMG/RF SEQUENCE PREM__sequence_120CAM_1_img_Truepredhor_',
               'prem results/RF 120 NOIMG/RF SEQUENCE PREM__sequence_120CAM_1_img_Falsepredhor_'
               ]

    return folders

def get_files_best_multi():

    files = ['persistence',
             'ann/ANN_SEQUENCE_MULTIepochs_40_sqnc_20data_all data.txt',
             'lstm/LSTM_SEQUENCE_MULTIepochs_50_sqnc_5data_all data.txt',
             'lstm/LSTM_SEQUENCE_MULTIepochs_50_sqnc_5data_onsite_only.txt',
             'rf/RF SEQUENCE multi_sqnc_30data_all data.txt'
             ]

    names = ['Persistence',
             'ANN 20 alldata',
             'LSTM 5 all data',
             'LSTM 5 onsite only',
             'RF 30 all data']

    return files, names

def get_files_ann_multi():
    folders = ls_dir('results_18march/multi/ann/')
    names = build_name(folders)
    return folders, names



def get_files_lstm_multi():
    folders = ls_dir('results_18march/multi/lstm/')
    names = build_name(folders)
    return folders, names


def get_files_cnn():
    fix_directory()
    folders = ['persistence', 'results 26 march/cnn/fb/CNN_predhor_LK_2', 'results_18march/cnn/lukaskanade/CNN_predhor_LK']
    names = ['Persistence', 'CNN-Farneback', 'CNN-LK']
    return folders, names

def get_files_rf_multi():
    folders = ls_dir('results_18march/multi/rf/')
    names = build_name(folders)
    return folders, names


def build_name(arr):
    names = []

    for str in arr:
        tmp_name = ''
        if 'RF' in str:
            tmp_name = 'RF'
        elif 'ANN' in str:
            tmp_name = 'ANN'
        elif 'LSTM' in str:
            tmp_name = 'LSTM'

        if 'single' in str:
            tmp_name = tmp_name + ' S'
        else:
            tmp_name = tmp_name + ' M'


        if 'sqnc_5' in str:
            tmp_name = tmp_name + ' 5'
        elif 'sqnc_10' in str:
            tmp_name = tmp_name + ' 10'
        elif 'sqnc_20' in str:
            tmp_name = tmp_name + ' 20'
        elif 'sqnc_30' in str:
            tmp_name = tmp_name + ' 30'
        elif 'sqnc_40' in str:
            tmp_name = tmp_name + ' 40'
        elif 'sqnc_60' in str:
            tmp_name = tmp_name + ' 60'
        elif 'sqnc_120' in str:
            tmp_name = tmp_name + ' 120'
        elif 'sqnc_3' in str:  # seq length
            tmp_name = tmp_name + ' 3'

        if 'all data' in str:  # data type
            tmp_name = tmp_name + ' all data'
        elif 'PXL' in str:
            tmp_name = tmp_name + ' PXL'
        elif 'onsite' in str:
            tmp_name = tmp_name + ' onsite'
        elif 'img only' in str:
            tmp_name = tmp_name + ' img'
        elif 'meteor' in str:
            tmp_name = tmp_name + ' meteor'

        if 'cams_2' in str:  # 2 cam?
            tmp_name = tmp_name + ' 2CAM'
        if 'clrsky' in str:
            tmp_name = tmp_name + ' Prz'

        names.append(tmp_name)

    return names

def get_files_ann_test():

    files = ls_dir('Results test set/ANN')
    names = build_name(files)

    return files,names

def get_files_rf_test():
    files = ls_dir('Results test set/RF/grad')
    names = build_name(files)
    return files, names

def ls_dir_folder(dir):
    fix_directory()
    ls =  [dir + '/' + f[0:f.find('ph_')+3] for f in listdir(dir) if isfile(join(dir, f))]
    return [i for n, i in enumerate(ls) if i not in ls[:n]]


def ls_dir(dir):
    fix_directory()
    return [dir + '/' + f for f in listdir(dir) if isfile(join(dir, f))]

def get_files_best_test():
    f, n = get_files_lstm_test()
    f = [f[1], f[3]]
    n = [n[1], n[3]]

    fann, nann = get_files_ann_test()
    fann, nann = [fann[4]], [nann[4]]

    frf, nrf = get_files_rf_test()
    frf, nrf = [frf[5]], [nrf[5]]

    f.extend(fann)
    f.extend(frf)

    n.extend(nann)
    n.extend(nrf)

    return f, n

def get_files_lstm_test():
    files = ls_dir('Results test set/LSTM')
    names = build_name(files)
    return files, names


def dump_list(nm, ls):
    with open('persistence/' + nm, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(ls, filehandle)

def load_list(nm):
    with open('persistence/' + nm, 'rb') as filehandle:
        # read the data as binary data stream
        ls = pickle.load(filehandle)
        return ls

def load_features():
    fix_directory()
    return np.load('x_22_d6to19_m7to12.npy')

def get_feature_data(features, month, day, start, end):
    start_time_idx = (start - 6) * 60
    end_time_idx = (end - 6) * 60
    previous_days = 0

    month_list = list(range(7, month + 1))
    for i in month_list:
        if i == 7:
            continue
        if i == 8:
            previous_days += 5
        else:
            previous_days += calendar.monthrange(2019, i - 1)[1]
    if month == 7:
        day_idx = day - 26
    else:
        day_idx = previous_days + day
    return features[day_idx, start_time_idx:end_time_idx]

def get_features_month(start,end, month):
    f = load_features()
    f = f.reshape(f.shape[0]* f.shape[1], f.shape[2])
    f  = f.astype(int)[f.astype(int)[:, 1] == month]  ##filter month
    f = f.astype(int)[f.astype(int)[:, 3] > (start-1)] ##fiter start
    return f.astype(int)[f.astype(int)[:, 3] < end] ##filter end


def month_to_year(month):
    if month < 4:
        return '2020'
    else:
        return '2019'


def process_csv(csv_name):
    tmp = pd.read_csv(csv_name, sep=';', header=None)
    if (len(tmp.columns) > 1):
        arr = pd.read_csv(csv_name, sep=';', header=None,
                          usecols=[0, 1, 2, 4, 7, 15])  # colums = ["DATE", "TIME", "RHUA", "TMPA", "PIRA"]

        # remove rows containing c,v,r not sure what it means..
        arr = arr[arr[0] != 'V-----']
        arr = arr[arr[0] != 'C-----']
        arr = arr[arr[0] != 'R-----']

        # date, time, humidity, tempertature, ghi
        c = [1, 2, 4, 7, 15]
        for index, row in arr.iterrows():
            for i in c:
                vals = row[i].split('=')
                row[i] = vals[1]

        arr[c].to_csv(csv_name, index=False)
    del tmp

def get_credentials():
    f = open('cred.txt', 'r')
    lines = f.read().split(',')
    f.close()
    return lines[0], lines[1], lines[2]

def download_data(cam, overwrite=False, process=True):  # download data
    cam_url = 0
    file_url = 0

    if (cam == 1):
        cam_url = "../asi16_data/asi_16124/"
        file_url = "asi_16124/"
        # todo add second

    server, username, passwd = get_credentials()
    ftp = ftplib.FTP(server)
    ftp.login(user=username.strip(), passwd=passwd.strip())

    ftp.cwd(cam_url)
    files = ftp.nlst()

    for f in tqdm(files, total=len(files)):
        ftp.cwd((cam_url + str(f)))
        tmp_path = file_url + f + "/"

        f_ = ftp.nlst()
        for i in f_:
            file_name = (str(i))
            tmp_name = (tmp_path + str(i))
            if not path.isfile(tmp_name) or overwrite:  # check if file exists
                if not os.path.exists(tmp_path):
                    os.mkdir(tmp_path)
                if '.jpg' in i:  # if image
                    image = open(tmp_name, 'wb')
                    ftp.retrbinary('RETR ' + file_name, image.write, 1024)
                    image.close()
                    # #TODO now you can do pre_processing
                elif '.csv' in i:
                    csv = open(tmp_name, 'wb')
                    ftp.retrbinary('RETR ' + file_name, csv.write, 1024)
                    csv.close()
                    try:  # some have wrong encoding..
                        process_csv(tmp_name)
                    except:
                        print('Error processing: ' + file_name)

def process_all_csv(cam):
    print("downloading all csv files")
    cam_url = 0
    file_url = 0

    if cam == 1:
        cam_url = "../asi16_data/asi_16124/"
        file_url = "asi_16124/"
        # todo add second
    if cam == 2:
        cam_url = "/asi16_data/asi_16133/"
        file_url = "asi_16133/"

    if not os.path.exists(file_url):
        os.mkdir(file_url)

    server, username, passwd = get_credentials()
    ftp = ftplib.FTP(server)
    ftp.login(user=username.strip(), passwd=passwd.strip())

    ftp.cwd(cam_url)
    files = ftp.nlst()

    for f in tqdm(files, total=len(files)):
        ftp.cwd((cam_url + str(f)))
        tmp_path = file_url + f + "/"

        f_ = ftp.nlst()
        for i in f_:
            file_name = (str(i))
            tmp_name = (tmp_path + str(i))
            if '2020' in file_name:  # skip 2020
                continue
            if not path.isfile(tmp_name):
                if not os.path.exists(tmp_path):
                    os.mkdir(tmp_path)
                if '.jpg' in i:  # if image
                    continue
                elif '.csv' in i:
                    csv = open(tmp_name, 'wb')
                    ftp.retrbinary('RETR ' + file_name, csv.write, 1024)
                    csv.close()
                    try:  # some have wrong encoding..
                        process_csv(tmp_name)
                    except:
                        print('Error processing: ' + file_name)

def extract_time(time_str):
    return (time_str[8:14])

def exract_formatted_time(time_str):
    s = time_str[8:14]
    return s[0:2] + ':' + s[2:4] + ':' + s[4:6]


def extract_time_less_accurate(time_str):
    return (time_str[8:12])

def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(zip(wordlist, wordfreq))

def resize_image(img, height, width):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

def get_avg_var_by_minute(df, hour, minute):
    rows = df[df[:, 3] == hour]  # filter on hours
    rows = rows[rows[:, 4] == minute]  # filter on minutes
    return np.mean(rows[:, [6,7,8,9,18,19,20,21]], axis=0), np.var(rows[:, [6,7,8,9,18,19,20,21]], axis=0)


def get_ghi_temp_by_minute(df, hour, minute):

    rows = df[np.where(df[:, 3] == hour)]
    rows = rows[rows[:, 4] == minute]  # filter on minutes
    return rows

def get_prev_day(day):
    previous_day = str(int(day) - 1)  # cao ni mam fix this
    if len(previous_day) == 1:
        previous_day = '0' + previous_day
    return previous_day

def flatten(l):
    f = [item for sublist in l for item in sublist]
    return f

def images_information():
    server, username, passwd = get_credentials()
    ftp = ftplib.FTP(server)
    ftp.login(user=username, passwd=passwd)

    ftp.cwd("/asi16_data/asi_16124")  # cam 1
    files = ftp.nlst()
    del files[0]  # data not valid

    start_times, stop_times, times = ([] for i in range(3))  # times not used. too much data. unable to plot..
    todo = len(files)
    done = 0

    for f in files:
        ftp.cwd(("/asi16_data/asi_16124/" + str(f)))
        f_ = ftp.nlst()
        start_times.append(extract_time_less_accurate(f_[0]))
        stop_times.append(extract_time_less_accurate(f_[-2]))

        done += 1
        print(str(done) + '/' + str(todo))

    start_dict = wordListToFreqDict(sorted(start_times))
    stop_dict = wordListToFreqDict(sorted(stop_times))

    print(start_dict)
    print(stop_dict)


    data.data_visuals.plot_freq(start_dict, 'Frequency start times')
    data.data_visuals.plot_freq(stop_dict, 'Frequency stop times')


def get_df_csv_month(month, start, end,
                     step):  # get data frame for a month with start and end time not inc. image
    fix_directory()
    folders = listdir('asi_16124')  # select cam
    del folders[0:3]  # first 3 are bad data
    index = 0
    queries = int(31 * ((end - start) * 60 / step))
    df = np.empty([queries, 9])  # create df

    for folder in folders:  # fill df
        if (int(folder[4:6]) == month):  # only check for month


            path = 'asi_16124/' + str(folder) + '/'
            files = listdir(path)

            process_csv(path + files[-1])  # process csv
            tmp_df = pd.read_csv(path + files[-1], sep=',', header=0, usecols=[0, 1, 2, 3, 4])  # load csv
            tmp_temp = None

            for row in tmp_df.iterrows():
                # check seconds 0, check step

                if (int(row[1].values[1][6:8]) == 0 and int(row[1].values[1][3:5]) % step == 0 and int(
                        row[1].values[1][0:2]) >= start and int(row[1].values[1][0:2]) < end):
                    df[index][0:9] = np.array(
                        [row[1].values[0][0:2], row[1].values[0][3:5], row[1].values[0][6:8],  # date
                         row[1].values[1][0:2], row[1].values[1][3:5], row[1].values[1][6:8],  # time
                         row[1].values[2],  # temp
                         row[1].values[3],  # humidity
                         row[1].values[4]])  # ghi  # set csv data to df

                    if (df[index][6] == 0):
                        df[index][6] = tmp_temp
                    else:
                        tmp_temp = df[index][6]

                    index += 1
                    # print(index)
    # # YEAR, MONTH, DAY, HOURS, MINUTES, SECONDS, TEMP, IRRADIANCE, IMAGE
    # print('filled queries: ' + str(index) + 'out of: ' + str(queries))
    return df.astype(int)

def fix_directory():
    dir = os.getcwd()
    if dir[-4:] == 'data':
        os.chdir("..")
        print(os.getcwd())


def get_df_csv_day_RP(month, day, start, end,
                      step, cam=1):  # replaces missing values with value of 15 seconds later.

    if cam == 1:
        path = 'asi_16124/2019' + features.int_to_str(month) + features.int_to_str(day) + '/'
        file_name = 'peridata_16124_' + month_to_year(month) + features.int_to_str(month) + features.int_to_str(day) + '.csv'
    elif cam == 2:
        # print('CAM 2 DATA')
        path = 'asi_16133/2019' + features.int_to_str(month) + features.int_to_str(day) + '/'
        file_name = 'peridata_16133_' + month_to_year(month) + features.int_to_str(month) + features.int_to_str(day) + '.csv'

    index = 0

    # data frame
    queries = int(((end - start) * 60 / step))
    df = np.empty([queries, 9])  # create df

    dir = os.getcwd()
    if dir[-4:] == 'data':
        os.chdir("..")
        print(os.getcwd())

    try:
        process_csv(path + file_name)
        tmp_df = pd.read_csv(path + file_name, sep=',', header=0, usecols=[0, 1, 2, 3, 4, ],
                             encoding='cp1252')  # load csv

        todo = 0

        for i, row in tmp_df.iterrows():
            if (int(row[1][3:5]) == todo and int(  # some magic for missing values.
                    row[1][0:2]) >= start and int(row[1][0:2]) < end):
                df[index][0:9] = np.array([row[0][0:2], row[0][3:5], row[0][6:8],  # date
                                           row[1][0:2], row[1][3:5], row[1][6:8],  # time
                                           row[2],  # temp
                                           row[3],  # humidity
                                           row[4]])  # ghi  # set csv data to df
                index += 1
                todo += step
                if (todo == 60):
                    todo = 0

        return df.astype(int)
    except:
        print('FAILED PROCESSING:')
        print(file_name)
        return None


def plot_per_month(start, end, step):
    times0, avg_temp0, var_temp0, avg_ghi0, var_ghi0, var_ghi0, avg_hum0, var_hum0, tick_times0 = ([] for i in range(9))
    avg_int0, var_int0, avg_cld0, var_cld0, avg_corn0, var_corn0, avg_edge0, var_edge0 = ([] for i in range(8))

    months = [8,9,10,11,12]
    for mon in months:
        # df = get_df_csv_month(mon, start, end, step)
        df = get_features_month(start,end, mon)
        hours = list(range(start, end))
        minutes = list(range(0, 60, step))
        times, avg_temp, var_temp, avg_ghi, var_ghi, var_ghi, avg_hum, var_hum, tick_times = ([] for i in range(9))
        avg_int, var_int, avg_cld, var_cld, avg_corn, var_corn, avg_edge, var_edge = ([] for i in range(8))

        for h in hours:
            tick_times.append(time(h, 0, 0))  # round hours
            tick_times.append(time(h, 30, 0))  # half hours
            for m in minutes:
                tmp_avg, tmp_var = get_avg_var_by_minute(df, h, m)

                tmp_time = time(h, m, 0)
                times.append(tmp_time)

                avg_temp.append(tmp_avg[0])
                var_temp.append(tmp_var[0])

                avg_hum.append(tmp_avg[1])
                var_hum.append(tmp_var[1])

                avg_ghi.append(tmp_avg[2])
                var_ghi.append(tmp_var[2])

                avg_int.append(tmp_avg[3])
                var_int.append(tmp_var[3])

                avg_cld.append(tmp_avg[4])
                var_cld.append(tmp_var[4])

                avg_corn.append(tmp_avg[5])
                var_corn.append(tmp_var[5])

                avg_edge.append(tmp_avg[6])
                var_edge.append(tmp_var[6])

        times0.append(times)
        tick_times0.append(tick_times)

        avg_temp0.append(avg_temp)
        var_temp0.append(var_temp)

        avg_ghi0.append(avg_ghi)
        var_ghi0.append(var_ghi)

        avg_hum0.append(avg_hum)
        var_hum0.append(var_hum)

        avg_int0.append(avg_int)
        var_int0.append(var_int)

        avg_cld0.append(avg_cld)
        var_cld0.append(var_cld)

        avg_corn0.append(avg_corn)
        var_corn0.append(var_corn)

        avg_edge0.append(avg_edge)
        var_edge0.append(var_edge)

        #avg_temp0, var_temp0, avg_ghi0, var_ghi0, var_ghi0, avg_hum0, var_hum0, tick_times0

    labels = ['August', 'September', 'October', 'November', 'December']

    # # plot data
    # data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, avg_temp0, labels, 'time', 'Temp. in Celsius', 'avg. Temp. in months')
    # data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, var_temp0, labels,  'time', 'Variance temp. Celsius.', 'var. Temp. in months')
    #
    # data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, avg_ghi0, labels, 'time', 'GHI in W/m^2', 'avg. GHI in months')
    # data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, var_ghi0, labels, 'time', 'Variance GHI', 'var. GHI in months')
    #
    # data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, avg_hum0, labels,  'time', 'Humidity', 'avg. Humidity in months')
    # data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, var_hum0, labels,  'time', 'Variance Humidity', 'var. Humidity in months')

    data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, avg_int0, labels, 'time', 'Intensity feature',
                                          'avg. intensity in months')
    data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, var_int0, labels, 'time', 'Variance intensity feature',
                                          'var. intensity in months')


    data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, avg_cld0, labels, 'time', '#CloudPixel feature',
                                          'avg. #CloudPixel in months')
    data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, var_cld0, labels, 'time', 'Variance #CloudPixel feature',
                                          'var. #CloudPixel in months')

    data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, avg_corn0, labels, 'time', '#Corners feature',
                                          'avg. #Corners in months')
    data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, var_corn0, labels, 'time', 'Variance #Corners feature',
                                          'var. #Corners in months')

    data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, avg_edge0, labels, 'time', '#Edges feature',
                                          'avg. #Edges in months')
    data.data_visuals.plot_time_avg_multi(tick_times0[0], times0, var_edge0, labels, 'time', 'Variance #Edges feature',
                                          'var. #Edges in months')

def plot_day(day, month, start, end, step):
    df = get_df_csv_day_RP(month, day, start, end, step)
    hours = list(range(start, end))
    minutes = list(range(0, 60, step))
    times, temp, ghi, tick_times = ([] for i in range(4))

    ghi_clear_sky = pvlib_playground.PvLibPlayground.get_clear_sky_irradiance(pvlib_playground.PvLibPlayground.get_times(2019,
                                                                                       int(month),
                                                                                       int(day),
                                                                                       start,
                                                                                       end))
    for h in hours:
        tick_times.append(time(h, 0, 0))  # round hours
        tick_times.append(time(h, 30, 0))  # half hours
        for m in minutes:
            rows = get_ghi_temp_by_minute(df, h, m)
            tmp_time = time(h, m, 0)
            old_ghi, old_temp = 0, 0

            if (len(rows) > 0):
                tmp_temp, old_temp = rows[0][6], rows[0][6]
                tmp_ghi, old_ghi = rows[0][8], rows[0][8]

                times.append(tmp_time)
                temp.append(tmp_temp)
                ghi.append(tmp_ghi)
            else:
                times.append(tmp_time)
                temp.append(old_temp)
                ghi.append(old_ghi)

    # plot data
    data.data_visuals.plot_time_avg(tick_times, times, temp, '', 'time', 'temp. in celsius',
                  'temp. in day: ' + str(day) + ' month: ' + str(month))
    data.data_visuals.plot_time_avg(tick_times, times, ghi, 'GHI measured', 'time', 'GHI in W/m^2',
                  'GHI in day: ' + str(day) + ' month: ' + str(month), ghi_clear_sky, 'Clear sky GHI')

def plot_persistence_day(day, month, start, end, step):

    df_truth = get_df_csv_day_RP(month, day, start, end, step)
    df_pred = get_df_csv_day_RP(month, day, start-1, end, step)
    copy = df_pred.copy()

    for i in range(0, len(df_pred)):
        if i < 20:
            continue
        else:
            df_pred[i][8] = copy[i - 20][8]

    df_pred = df_pred[60:]

    hours = list(range(start, end))
    minutes = list(range(0, 60, step))
    times, ghi_pred, ghi_truth, tick_times = ([] for i in range(4))

    for h in hours:
        tick_times.append(time(h, 0, 0))  # round hours
        tick_times.append(time(h, 30, 0))  # half hours
        for m in minutes:
            rows_truth = get_ghi_temp_by_minute(df_truth, h, m)
            rows_pred = get_ghi_temp_by_minute(df_pred, h, m)
            tmp_time = time(h, m, 0)

            # sometimes data is missing then skip.
            if (len(rows_truth) > 0 and len(rows_pred) > 0):
                ghi_truth_tmp = rows_truth[0][8]
                ghi_pred_tmp = rows_pred[0][8]
                times.append(tmp_time)
                ghi_truth.append(ghi_truth_tmp)
                ghi_pred.append(ghi_pred_tmp)

    # plot data
    data.data_visuals.plot_2_models(tick_times, times, ghi_truth, ghi_pred, 'time', 'GHI in W/m^2',
                  'GHI at day: ' + str(day) + ' month: ' + str(month))




def get_persistence_df(month, day, start, end, pred_hor):
    name = str(month) + str(day) + str(start) + str(end) + str(pred_hor)

    additions = ['pred', 'truth', 'times']

    if os.path.isfile('persistence/' + name+additions[0]):
        # print('from cache')
        ghi_pred, ghi_truth, times = load_list(name+additions[0]), load_list(name+additions[1]), load_list(name+additions[2])
        return ghi_pred, ghi_truth, times


    df_truth = get_df_csv_day_RP(month, day, start, end+1, 1)
    df_pred = get_df_csv_day_RP(month, day, start-1, end, 1)
    copy = df_pred.copy()


    for i in range(0, len(df_pred)):
        if i < (pred_hor):
            # print(i)
            continue
        else:
            df_pred[i][8] = copy[i - pred_hor][8]

    df_pred = df_pred[60:]
    hours = list(range(start, end))
    minutes = list(range(0, 60, 1))
    times, ghi_pred, ghi_truth = ([] for i in range(3))

    for h in hours:
        for m in minutes:
            rows_truth = get_ghi_temp_by_minute(df_truth, h, m)
            rows_pred = get_ghi_temp_by_minute(df_pred, h, m)

            timestamp = datetime.datetime(year=2019, month=month , day=day, hour=h, minute=m)
            matplotlib_timestamp = matplotlib.dates.date2num(timestamp)

            if (len(rows_truth) > 0 and len(rows_pred) > 0):  # sometimes data is missing then skip.
                ghi_truth_tmp = rows_truth[0][8]
                ghi_pred_tmp = rows_pred[0][8]
                times.append(matplotlib_timestamp)
                ghi_truth.append(ghi_truth_tmp)
                ghi_pred.append(ghi_pred_tmp)


    dump_list(name+additions[0], ghi_pred)
    dump_list(name + additions[1], ghi_truth)
    dump_list(name + additions[2], times)

    return ghi_pred, ghi_truth, times

def get_persistence_dates(tups, start, end, pred_hor, offset=0):
    actual = []
    pred = []
    times = []
    for tup in tups:
        # print(tup)
        # print(offset)
        p, a, tm = get_persistence_df(tup[0], tup[1], start, end+1, pred_hor)

        # if p == 0 or a == 0:
        #     continue
        # else:

        actual.extend(a[offset+pred_hor-1:780+pred_hor-1])
        pred.extend(p[offset+pred_hor-1:780+pred_hor-1])
        times.extend(tm[offset+pred_hor-1:780+pred_hor-1])



    return actual, pred, times

def get_thesis_test_days(in_sunny=True, in_cloudy=True, in_parcloudy=True):  # test days personal research
    sunny = [(9, 15), (10, 15), (11, 15), (12, 15)]
    pcloudy = [(10, 21), (11, 17), (12, 16)]
    cloudy = [(10, 22), (12, 3)]
    total = []

    if in_sunny:
        total.extend(sunny)
    if in_parcloudy:
        total.extend(pcloudy)
    if in_cloudy:
        total.extend(cloudy)

    if len(total) < 1:
        print('Include atleast 1 weather circumstance')

    return total

def get_all_days():  # copernicus test days
    total = []
    s = [24, 29, 30]
    o = [1, 5, 6, 7, 8, 10, 12, 17, 18, 20, 23, 26, 29, 30 ]
    n = [1, 6, 7, 8, 14, 18, 19, 20, 21, 26, 28]

    for i in s:
        total.append((9,i))
    for j in o:
        total.append((10,j))
    for k in n:
        total.append((11, k))

    return total


def plot_history(settings, num, history):
    plt.figure()
    axes = plt.gca()

    # axes.set_ylim([0, 100000])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss ' + str(settings))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig(str(num) + '.png')

    plt.clf()
    plt.close()




def get_error_month(month, start, end, step):
    y_observed = []
    y_predicted = []

    for i in range(1, 30):
        df_truth = get_df_csv_day_RP(month, i, start, end, step)
        df_pred = get_df_csv_day_RP(month, i, start - 1, end, step)
        copy = df_pred.copy()

        for i in range(0, len(df_pred)):
            if i < 20:
                continue
            else:
                df_pred[i][8] = copy[i - 20][8]

        df_pred = df_pred[60:]

        for idx, val in enumerate(df_truth):
            y_observed.append(val[8])
            y_predicted.append(df_pred[idx][8])

    print('RMSE')
    print(metrics.Metrics.rmse(y_observed, y_predicted))
    print('MAE')
    print(metrics.Metrics.mae(y_observed, y_predicted))
    print('MAPE')
    print(metrics.Metrics.mape(y_observed, y_predicted))


def search_weather_circ_days():
# sunny > 75
# coudy < 25
# p.cloude > 25 < 75
    months = [8,9,10,11,12]

    sunny_days = [(8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (8, 21), (8, 22), (8, 23), (8, 24), (8, 25), (8, 26), (8, 27), (8, 28), (8, 29), (8, 30), (8, 31), (9, 1), (9, 2), (9, 3), (9, 4), (9, 6), (9, 10), (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 22), (9, 23), (9, 24), (9, 25), (9, 26), (9, 27), (9, 28), (9, 29), (9, 30), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 23), (10, 24), (10, 25), (10, 26), (10, 27), (10, 28), (10, 29), (10, 30), (10, 31), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 12), (11, 13), (11, 15), (11, 16), (11, 18), (11, 19), (11, 20), (11, 23), (11, 24), (11, 26), (11, 27), (11, 28), (11, 29), (11, 30), (12, 1), (12, 4), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 14), (12, 15), (12, 17), (12, 18), (12, 21), (12, 22), (12, 23), (12, 24), (12, 25), (12, 26), (12, 27), (12, 28), (12, 29), (12, 30)]
    pcloudy_days = [(8, 11), (8, 12), (9, 5), (9, 7), (9, 8), (9, 9), (9, 12), (9, 13), (9, 20), (9, 21), (10, 20), (10, 21), (11, 1), (11, 2), (11, 10), (11, 11), (11, 14), (11, 17), (11, 21), (11, 22), (11, 25), (12, 2), (12, 5), (12, 12), (12, 13), (12, 16), (12, 19), (12, 20), (12, 31)]
    cloudy_days = [(9, 11), (10, 22), (12, 3)]

    # ALREADY CALCULATED
    # for m in months:
    #     days = list(range(1, calendar.monthrange(2019, m)[1] + 1))
    #     for d in days:
    #         ghi = get_df_csv_day_RP(m, d, 11, 15, 1)[:,8]
    #         ghi_clr = PvLibPlayground.get_clear_sky_irradiance( PvLibPlayground.get_times(2019,
    #                                                                                    m,  # month
    #                                                                                    d,  # day
    #                                                                                    11,# start time
    #                                                                                    15))  # end time)
    #         csi = PvLibPlayground.calc_clear_sky(ghi, ghi_clr).mean()
    #         if csi > 0.75:
    #             sunny_days.append((m,d))
    #         elif csi > 0.25 and csi < 0.75:
    #             pcloudy_days.append((m,d))
    #         elif csi < 0.25:
    #             cloudy_days.append((m,d))


    print('sunny days: ' + str(len(sunny_days)))
    print(sunny_days)

    print('pcloudy days: ' + str(len(pcloudy_days)))
    print(pcloudy_days)

    print('Cloudy days: ' + str(len(cloudy_days)))
    print(cloudy_days)


def get_smart_persistence_dates(tups, start, end, pred_hor):
    actual = []
    pred = []
    times = []
    for tup in tups:
        # print(tup)
        # print(offset)
        observed, predicted, _ = get_smart_persistence(tup[0], tup[1], start, end, pred_hor)

        actual.extend(observed)
        pred.extend(predicted)
        # times.extend(tm[offset+pred_hor-1:780+pred_hor-1])

    return actual, pred, 0

def get_smart_persistence(month, day, start, end, pred_hor):
    # def get_smart_persistence_df(month, day, start, end, pred_hor):
    name = 'smart/' + str(month) + str(day) + str(start) + str(end) + str(pred_hor)

    additions = ['pred', 'truth', 'times']

    if os.path.isfile('persistence/' + name + additions[0]):
        # print('from cache')
        ghi_pred, ghi_truth = load_list(name + additions[0]), load_list(name + additions[1])
        return ghi_pred, ghi_truth, 0

    df_copy = get_df_csv_day_RP(month, day, start, end, 1)
    df_true = get_df_csv_day_RP(month, day, start, end+1, 1)
    times = pvlib_playground.PvLibPlayground.get_times(2019, month, day, start, end+1)
    df_dni = pvlib_playground.PvLibPlayground.get_clear_sky_irradiance(times)
    csi_ls = pvlib_playground.PvLibPlayground.calc_clear_sky_ls(df_copy[:,8], df_dni[0:len(df_copy)])

    res = []

    for idx, val in enumerate(csi_ls):
        res.append(val * df_dni[idx + pred_hor])


    dump_list(name+additions[0], res)
    dump_list(name + additions[1], df_true[pred_hor: len(res)+ pred_hor,8])

    return df_true[pred_hor: len(res)+ pred_hor,8], res, 0

