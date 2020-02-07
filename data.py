import ftplib
from builtins import enumerate
import os

import pandas as pd
import numpy as np
import time
from os import listdir, path
import cv2
from data_visuals import plot_time_avg, plot_freq, plot_2_models
from metrics import Metrics
from pvlib_playground import PvLibPlayground
from features import get_image_by_date_time, int_to_str, extract_features, show_img
from tqdm import tqdm
from sklearn.preprocessing import *

enable_print = True

def printf(str):
    if enable_print:
        print(str)

def get_df_csv_day_RP(month, day, start, end,
                      step):  # replaces missing values with value of 15 seconds later.

    path = 'asi_16124/2019' + int_to_str(month) + int_to_str(day) + '/'
    file_name = 'peridata_16124_' + month_to_year(month) + int_to_str(month) + int_to_str(
        day) + '.csv'  # todo make 2020 ready
    index = 0

    # data frame
    queries = int(((end - start) * 60 / step))
    df = np.empty([queries, 9])  # create df

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

    # print('filled queries: ' + str(index) + ' out of: ' + str(queries))
    return df.astype(int)


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
        cam_url = "/asi16_data/asi_16124/"
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
        cam_url = "/asi16_data/asi_16124/"
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
    return np.mean(rows[:, 6:9], axis=0), np.var(rows[:, 6:9], axis=0)

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

    plot_freq(start_dict, 'Frequency start times')
    plot_freq(stop_dict, 'Frequency stop times')


def get_df_csv_month(month, start, end,
                     step):  # get data frame for a month with start and end time not inc. image
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


def get_df_csv_day_RP(month, day, start, end,
                      step, cam=1):  # replaces missing values with value of 15 seconds later.

    if cam == 1:
        path = 'asi_16124/2019' + int_to_str(month) + int_to_str(day) + '/'
        file_name = 'peridata_16124_' + month_to_year(month) + int_to_str(month) + int_to_str(day) + '.csv'
    elif cam == 2:
        # print('CAM 2 DATA')
        path = 'asi_16133/2019' + int_to_str(month) + int_to_str(day) + '/'
        file_name = 'peridata_16133_' + month_to_year(month) + int_to_str(month) + int_to_str(day) + '.csv'

    index = 0

    # data frame
    queries = int(((end - start) * 60 / step))
    df = np.empty([queries, 9])  # create df
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

def plot_per_month(month, start, end, step):
    df = get_df_csv_month(month, start, end, step)
    hours = list(range(start, end))
    hours = list(range(start, end))
    minutes = list(range(0, 60, step))
    times, avg_temp, var_temp, avg_ghi, var_ghi, var_ghi, tick_times = ([] for i in range(7))

    for h in hours:
        tick_times.append(time(h, 0, 0))  # round hours
        tick_times.append(time(h, 30, 0))  # half hours
        for m in minutes:
            tmp_avg, tmp_var = get_avg_var_by_minute(df, h, m)
            tmp_time = time(h, m, 0)
            times.append(tmp_time)
            avg_temp.append(tmp_avg[0])
            var_temp.append(tmp_var[0])
            # todo tmp[1] = humidity
            avg_ghi.append(tmp_avg[2])
            var_ghi.append(tmp_var[2])
    # plot data
    plot_time_avg(tick_times, times, avg_temp, 'time', 'Temp. in celsius', 'avg. Temp. in month ' + str(month))
    plot_time_avg(tick_times, times, var_temp, 'time', 'Variance temp.', 'var. Temp. in month ' + str(month))
    plot_time_avg(tick_times, times, avg_ghi, 'time', 'GHI in W/m^2', 'avg. GHI in month ' + str(month))
    plot_time_avg(tick_times, times, var_ghi, 'time', 'Varian'
                                                      'ce GHI', 'var. GHI in month ' + str(month))


def plot_day(day, month, start, end, step):
    df = get_df_csv_day_RP(month, day, start, end, step)
    hours = list(range(start, end))
    minutes = list(range(0, 60, step))
    times, temp, ghi, tick_times = ([] for i in range(4))

    ghi_clear_sky = PvLibPlayground.get_clear_sky_irradiance(PvLibPlayground.get_times(2019,
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
    plot_time_avg(tick_times, times, temp, '', 'time', 'temp. in celsius',
                  'temp. in day: ' + str(day) + ' month: ' + str(month))
    plot_time_avg(tick_times, times, ghi, 'GHI measured', 'time', 'GHI in W/m^2',
                  'GHI in day: ' + str(day) + ' month: ' + str(month), ghi_clear_sky, 'Clear sky GHI')

def plot_persistence_day(day, month, start, end, step):

    df_truth = get_df_csv_day_RP(month, day, start, end, step)
    previous_day = '0' + str(int(day) - 1)  # cao ni mam fix this
    df_pred = get_df_csv_day_RP(month, get_prev_day(day), start, end, step)

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
    plot_2_models(tick_times, times, ghi_truth, ghi_pred, 'time', 'GHI in W/m^2',
                  'GHI at day: ' + str(day) + ' month: ' + str(month))

def get_error_month(month, start, end, step):
    y_observed = []
    y_predicted = []

    for i in range(2, 30):

        day = str(i)
        if len(day) == 1:
            day = '0' + day

        df_truth = get_df_csv_day_RP(month, day, start, end, step)
        df_pred = get_df_csv_day_RP(month, get_prev_day(day), start, end, step)  # todo include prev month

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
                    ghi_truth_tmp = rows_truth[0][7]
                    ghi_pred_tmp = rows_pred[0][7]
                    times.append(tmp_time)
                    ghi_truth.append(ghi_truth_tmp)
                    ghi_pred.append(ghi_pred_tmp)

        y_observed.append(ghi_truth)
        y_predicted.append(ghi_pred)

    y_observed = flatten(y_observed)
    y_predicted = flatten(y_predicted)

    print('RMSE')
    print(Metrics.rmse(y_observed, y_predicted))
    print('MAE')
    print(Metrics.mae(y_observed, y_predicted))
    print('MAPE')
    print(Metrics.mape(y_observed, y_predicted))