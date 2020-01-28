import ftplib
from builtins import enumerate
import os

import numpy as np
import pandas as pd
from os import listdir, path
import cv2
from datetime import time
from data_visuals import *
from metrics import Metrics
from pvlib_playground import PvLibPlayground
from features import get_image_by_date_time, int_to_str, extract_features
import calendar
from datetime import date
from tqdm import tqdm
from sklearn.preprocessing import *

enable_print = True


def printf(str):
    if enable_print:
        print(str)

def month_to_year(month):
    if month < 4:
        return '2020'
    else:
        return '2019'

def process_csv(csv_name):
    # print(csv_name)

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


class Data:

    mega_df = []
    extra_df = []

    train_df = []
    test_df = []

    months = []
    size_of_row = 9  # row default size 8 + 1 for label
    size_meteor_data = 8  # amount of columns from meteor data
    img_idx = 9  # index of column were image data starts
    queries_per_day = 0
    pred_horizon = 0

    def __init__(self, meteor_data=False, images=False, debug=False):
        self.start = 0
        self.end = 0
        self.step = 0

        self.meteor_data = meteor_data
        self.images = images
        self.debug = debug

        self.x_train = 0
        self.y_train = 0
        self.x_test = 0
        self.y_test = 0

        if self.meteor_data:  # adjusting df row length according to amount of data
            self.size_of_row += self.size_meteor_data
        if self.images:
            self.size_of_row += 4  # change if it gets bigger
            if self.meteor_data:
                self.img_idx += self.size_meteor_data

        print('Building Df with meteor data: ' + str(self.meteor_data) + ', image data: ' + str(
            self.images) + ', debug: ' + str(self.debug) + '..')
        print('size of a row: ' + str(self.size_of_row))



    def download_data(self, cam=1, overwrite=False, process=True):  # download data
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
                # else:
                #     print('file ' + file_name + " exists.. no overwrite")

    def process_all_csv(self, cam = 1):
        print("downloading all csv files")
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
                if not path.isfile(tmp_name):
                    if not os.path.exists(tmp_path):
                        os.mkdir(tmp_path)
                    if '.jpg' in i:  # if image
                        pass
                    elif '.csv' in i:
                        csv = open(tmp_name, 'wb')
                        ftp.retrbinary('RETR ' + file_name, csv.write, 1024)
                        csv.close()
                        try:  # some have wrong encoding..
                            process_csv(tmp_name)
                        except:
                            print('Error processing: ' + file_name)


    def extract_time(self, time_str):
        return (time_str[8:14])

    def exract_formatted_time(self, time_str):
        s = time_str[8:14]
        return s[0:2] + ':' + s[2:4] + ':' + s[4:6]

    def extract_time_less_accurate(self, time_str):
        return (time_str[8:12])

    def wordListToFreqDict(self, wordlist):
        wordfreq = [wordlist.count(p) for p in wordlist]
        return dict(zip(wordlist, wordfreq))

    def resize_image(self, img, height, width):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    def get_avg_var_by_minute(self, df, hour, minute):
        rows = df[df[:, 3] == hour]  # filter on hours
        rows = rows[rows[:, 4] == minute]  # filter on minutes
        return np.mean(rows[:, 6:9], axis=0), np.var(rows[:, 6:9], axis=0)

    def get_ghi_temp_by_minute(self, df, hour, minute):

        rows = df[np.where(df[:, 3] == hour)]
        rows = rows[rows[:, 4] == minute]  # filter on minutes
        return rows

    def plot_per_month(self, month, start, end, step):
        df = self.get_df_csv_month(month, start, end, step)
        hours = list(range(start, end))
        minutes = list(range(0, 60, step))
        times, avg_temp, var_temp, avg_ghi, var_ghi, var_ghi, tick_times = ([] for i in range(7))

        for h in hours:
            tick_times.append(time(h, 0, 0))  # round hours
            tick_times.append(time(h, 30, 0))  # half hours
            for m in minutes:
                tmp_avg, tmp_var = self.get_avg_var_by_minute(df, h, m)
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

    def plot_day(self, day, month, start, end, step):
        # df = self.get_df_csv_day(month, day, start, end, step)
        df = self.get_df_csv_day_RP(month, day, start, end, step)
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
                rows = self.get_ghi_temp_by_minute(df, h, m)
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

    def plot_persistence_day(self, day, month, start, end, step):

        df_truth = self.get_df_csv_day(month, day, start, end, step)
        previous_day = '0' + str(int(day) - 1)  # cao ni mam fix this
        df_pred = self.get_df_csv_day(month, self.get_prev_day(day), start, end, step)

        hours = list(range(start, end))
        minutes = list(range(0, 60, step))
        times, ghi_pred, ghi_truth, tick_times = ([] for i in range(4))

        for h in hours:
            tick_times.append(time(h, 0, 0))  # round hours
            tick_times.append(time(h, 30, 0))  # half hours
            for m in minutes:
                rows_truth = self.get_ghi_temp_by_minute(df_truth, h, m)
                rows_pred = self.get_ghi_temp_by_minute(df_pred, h, m)
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

    def get_prev_day(self, day):
        previous_day = str(int(day) - 1)  # cao ni mam fix this
        if len(previous_day) == 1:
            previous_day = '0' + previous_day
        return previous_day

    def get_error_month(self, month, start, end, step):
        y_observed = []
        y_predicted = []

        for i in range(2, 30):

            day = str(i)
            if len(day) == 1:
                day = '0' + day

            df_truth = self.get_df_csv_day(month, day, start, end, step)
            df_pred = self.get_df_csv_day(month, self.get_prev_day(day), start, end, step)  # todo include prev month

            hours = list(range(start, end))
            minutes = list(range(0, 60, step))
            times, ghi_pred, ghi_truth, tick_times = ([] for i in range(4))

            for h in hours:
                tick_times.append(time(h, 0, 0))  # round hours
                tick_times.append(time(h, 30, 0))  # half hours
                for m in minutes:
                    rows_truth = self.get_ghi_temp_by_minute(df_truth, h, m)
                    rows_pred = self.get_ghi_temp_by_minute(df_pred, h, m)
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

        y_observed = self.flatten(y_observed)
        y_predicted = self.flatten(y_predicted)

        print('RMSE')
        print(Metrics.rmse(y_observed, y_predicted))
        print('MAE')
        print(Metrics.mae(y_observed, y_predicted))
        print('MAPE')
        print(Metrics.mape(y_observed, y_predicted))

    def flatten(self, l):
        f = [item for sublist in l for item in sublist]
        return f

    def images_information(self):
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
            start_times.append(self.extract_time_less_accurate(f_[0]))
            stop_times.append(self.extract_time_less_accurate(f_[-2]))

            done += 1
            print(str(done) + '/' + str(todo))

        start_dict = self.wordListToFreqDict(sorted(start_times))
        stop_dict = self.wordListToFreqDict(sorted(stop_times))

        print(start_dict)
        print(stop_dict)

        plot_freq(start_dict, 'Frequency start times')
        plot_freq(stop_dict, 'Frequency stop times')

    def get_df_csv_month(self, month, start, end,
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

    def get_df_csv_day_RP(self, month, day, start, end,
                          step):  # replaces missing values with value of 15 seconds later.

        path = 'asi_16124/2019' + int_to_str(month) + int_to_str(day) + '/'
        file_name = 'peridata_16124_' + month_to_year(month) + int_to_str(month) + int_to_str(day) + '.csv'  # todo make 2020 ready
        index = 0

        # data frame
        queries = int(((end - start) * 60 / step))
        df = np.empty([queries, 9])  # create df

        process_csv(path + file_name)
        tmp_df = pd.read_csv(path + file_name, sep=',', header=0, usecols=[0, 1, 2, 3, 4, ],  encoding='cp1252')  # load csv
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
                if(todo == 60):
                    todo = 0

        # print('filled queries: ' + str(index) + ' out of: ' + str(queries))
        return df.astype(int)

    def sample_from_df(self, sample_size):  # sample random rows. returns new df with indexes.
        random_idx = np.random.randint(self.mega_df.shape[0], size=sample_size)
        return self.mega_df[random_idx, :, :], random_idx


    def split_data_set_DEPRICATED(self, train_percentage=0.8):  # todo validation set and in/ex clusions for training set.
        print('Splitting with train: ' + str(train_percentage) + '...')
        np.random.shuffle(self.mega_df)
        tmp_size = int(train_percentage * self.mega_df.shape[0])
        self.train_df = self.mega_df[0:tmp_size]
        self.test_df = self.mega_df[tmp_size:self.mega_df.shape[0]]
        print('done')

    def split_data_set(self, m, d):
        printf('Splitting with train until month: ' + str(m) + ', day: ' + str(d) + '...')

        day_idx = 0
        for idx, day in enumerate(self.mega_df):  #find index month
            if int(day[0][1]) == m and int(day[0][2]) == d and day_idx == 0:
                day_idx = idx
                print('found: ' + str(day_idx))
                break

        self.train_df = self.mega_df[0:day_idx]
        self.test_df = self.mega_df[day_idx]

        printf('done')

    def flatten_data_set_CNN(self):
        printf('Flattening..')

        self.train_df = self.train_df.reshape((self.train_df.shape[0] * self.train_df.shape[1], -1))

        self.x_train = self.train_df[:, 3: self.train_df.shape[1]]
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 400, 400, 3))  # reshaping for tf
        self.y_train = self.train_df[:, 0]

        self.x_test = self.test_df[:, 3:self.test_df.shape[1]]
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 400, 400, 3))  # reshaping for tf
        self.y_test = self.test_df[:, 0]


    def flatten_data_set(self):  # this is needed to use it as input for models
        printf('Flattening..')

        self.train_df = self.train_df.reshape((self.train_df.shape[0] * self.train_df.shape[1], -1))
        # self.test_df = self.test_df.reshape((self.test_df.shape[0] * self.test_df.shape[1], -1))

        self.x_train = self.train_df[:, 0: self.train_df.shape[1] - 1]
        self.y_train = self.train_df[:, -1]

        self.x_test = self.test_df[:, 0:self.test_df.shape[1] - 1]
        self.y_test = self.test_df[:, -1]

        printf('done')

    def normalize_data_sets(self):
        #0 year, 1 month, 2 day, 3 hour, 4 minute, 5 seconds, 6 temp, 7 humidity,
        #8 current ghi, 9 future ghi (y) , 10 csi, 11 azimuth, 12 zenith, 13 intensity,
        #14 cloudpixel, 15 harris, 16 edges

        colums_to_normalize = [3,4,5,6,7,8]
        if(self.meteor_data):
            colums_to_normalize.extend([9,15])
        if(self.images):
            colums_to_normalize.extend([13, 14, 15, 16])

        printf('normalzing for: ' + str(colums_to_normalize))

        self.x_train[:, colums_to_normalize] = normalize(self.x_train[:, colums_to_normalize], axis=0, norm='l2')
        self.x_test[:, colums_to_normalize] = normalize(self.x_test[:, colums_to_normalize], axis=0, norm='l2')
        # print('done')

    def build_df_for_cnn(self, start, end, step, months):
        self.start = start
        self.end = end
        self.step = step
        self.months = months
        self.queries_per_day = int(((end - start) * 60 / step))  # amount of data in one day

        days = 0
        day_index = -1

        printf('Processing months for CNN DF: ' + str(months))

        for m in months:
            if(self.debug):  # debug
                days += 3
            elif m == 7:
                days += 8
            else:
                year = int(month_to_year(m))
                days += calendar.monthrange(year, m)[1]


        #image res + label + month + day
        size_of_row = 3+(400*400*3)
        self.mega_df = np.zeros((days, self.queries_per_day, size_of_row ), dtype=np.uint16)

        print(months)
        for m in tqdm(months, total=len(months), unit='Month progress'):
            if self.debug:  # debug
                days = [26, 27, 28]
            elif m == 7:  # different for july..
                days = [24, 25, 26, 27, 28, 29, 30, 31]
            else:
                days = list(range(1, calendar.monthrange(2019, m)[1] + 1))  # create an array with days for that month

            print(days)

            for d in tqdm(days, total=len(days), unit='Day progress'):
                # todo add sunrise/sundown for start end hour? half hour?
                day_data = self.get_df_csv_day_RP(m, d, start, end, step).astype(int)
                day_index += 1

                for idx, data in enumerate(day_data):

                    self.mega_df[day_index][idx][0] = data[8]  # label
                    self.mega_df[day_index][idx][1] = data[1]  # month

                    self.mega_df[day_index][idx][2] = data[2]  # day
                    year, month, day, hour, minute, seconds = int(data[0]), int(data[1]), int(data[2]), int(
                        data[3]), int(data[4]), int(data[5])

                    img = get_image_by_date_time(year, month, day, hour, minute, seconds)  # get image
                    self.mega_df[day_index][idx][3:size_of_row] = img.ravel()


    def build_df(self, start, end, step, months):

        self.start = start
        self.end = end
        self.step = step
        self.months = months
        self.queries_per_day = int(((end - start) * 60 / step))  # amount of data in one day

        days = 0
        day_index = -1

        printf('Processing months: ' + str(months))


        for m in months:
            if m == 7:
                days += 8
            else:
                year = int(month_to_year(m))
                days += calendar.monthrange(year, m)[1]
        if(self.debug):  # debug
            days = 3
        self.mega_df = np.zeros((days, self.queries_per_day, self.size_of_row), dtype=np.float)


        for m in tqdm(months, total=len(months), unit=' Month progress for ' + str(self.pred_horizon)):
            if self.debug:  # debug
                days = [25, 26, 27]
            elif m == 7:  # different for july..
                days = [24, 25, 26, 27, 28, 29, 30, 31]
            else:
                days = list(range(1, calendar.monthrange(2019, m)[1] + 1) ) # create an array with days for that month

            for d in days:
                # todo add sunrise/sundown for start end hour? half hour?
                day_data = self.get_df_csv_day_RP(m, d, start, end, step).astype(int)
                day_index += 1

                for idx, data in enumerate(day_data):
                    data[6] = Metrics.celsius_to_kelvin(data[6])  # convert to kelvin
                    self.mega_df[day_index][idx][0:9] = data  # adding data from csv
                    if self.images:
                        year, month, day, hour, minute, seconds = int(data[0]), int(data[1]), int(data[2]), int(
                            data[3]), int(data[4]), int(data[5])
                        img = get_image_by_date_time(year, month, day, hour, minute, seconds)  #  get image
                        self.mega_df[day_index][idx][self.img_idx:(self.size_of_row-1)] = extract_features(img)  # get features from image
                        del img  # delete img for mem

        # YEAR, MONTH, DAY, HOURS, MINUTES, SECONDS, TEMP, IRRADIANCE, IMAGE
        printf('Building done')

    def set_pred_horizon_virtual(self, prediction_horizon):  # This function just changes the variable but doesnt label accoringly. This is needed for persistence model.
        self.pred_horizon = prediction_horizon  # set prediction horizon
        self.label_df()

    def set_prediction_horizon(self, prediction_horizon):  # only for model 1
        printf('Setting prediction horizon to: ' + str(prediction_horizon) + '...')
        self.pred_horizon = prediction_horizon  # set prediction horizon

        days = 0
        day_index = -1

        for m in self.months:
            if m == 7:
                days += 8
            else:
                year = int(month_to_year(m))
                days += calendar.monthrange(year, m)[1]

        self.extra_df = np.zeros((days, self.pred_horizon, 1), dtype=np.float)  # make space for labels

        for m in self.months:

            if m == 7:  # different for july..
                days = [24, 25, 26, 27, 28, 29, 30, 31]
            else:
                days = range(1, calendar.monthrange(2019, m)[1])  # create an array with days for that month

            if(self.debug):  # debug
                days = [25, 26, 27]

            for d in days:
                # todo add sunrise/sundown for start end
                extra = self.get_df_csv_day_RP(m, d, self.end, self.end+1, self.step).astype(int)
                day_index += 1

                for idx, data in enumerate(extra):  # getting label data for predictions
                    if idx < self.pred_horizon:
                        self.extra_df[day_index][idx][0] = data[8]
                    else:
                        continue

        self.label_df()
        printf('done')

    def label_df(self):
        for idx_day, day in enumerate(self.mega_df):
            tmp_cnt = 0
            m = int(day[0][1])
            d = int(day[0][2])
            start = self.start
            end = self.end

            csi, azimuth, zenith = [], [], []

            if self.meteor_data:
                csi, azimuth, zenith, sun_earth_dis, ephemeris = PvLibPlayground.get_meteor_data(m,
                                                                       d,
                                                                       PvLibPlayground.get_times(2019,
                                                                                                 m,  # month
                                                                                                 d,  # day
                                                                                                 start,  # start time
                                                                                                 end,
                                                                                                 offset=int(self.pred_horizon)))  # end time


            for idx_timeslot, time_slot in enumerate(day):
                if self.meteor_data:
                    self.mega_df[idx_day][idx_timeslot][9] = csi[idx_timeslot]
                    self.mega_df[idx_day][idx_timeslot][10] = azimuth[idx_timeslot]
                    self.mega_df[idx_day][idx_timeslot][11] = zenith[idx_timeslot]
                    self.mega_df[idx_day][idx_timeslot][11] = sun_earth_dis[idx_timeslot]
                    self.mega_df[idx_day][idx_timeslot][12:16] = ephemeris[idx_timeslot]

                if (self.queries_per_day - self.pred_horizon) > idx_timeslot:
                    self.mega_df[idx_day][idx_timeslot][self.size_of_row-1] = self.mega_df[idx_day][idx_timeslot + self.pred_horizon][8]
                    tmp_cnt +=1
                else:
                    self.mega_df[idx_day][idx_timeslot][self.size_of_row - 1] = self.extra_df[idx_day][idx_timeslot-tmp_cnt][0]

    def process_save_image_data(self, filename):
        today = str(date.today())
        np.savez_compressed('mega_df_' + filename + '_' + today, self.mega_df)

    def load_prev_mega_df(self, filename):
        self.mega_df = np.load('mega_df_' + filename)

# ## build df for model 1
# data = Data(meteor_data=True, images=False, debug=True)
# data.build_df(10, 17, 1, months=[7])
# # data.process_save_image_data('mega_df')
#
# data.set_prediction_horizon(5)
# data.split_data_set(7, 26)
# data.flatten_data_set()
# data.normalize_data_sets()
# np.set_printoptions(precision=3)
# print(data.mega_df)
#

## downloading and processing stuff
# d.download_data(1, True)
# process_csv("peridata_16124_20190723.csv")
# d.images_information()
# d.plot_per_month(9, 5, 19)
# d.plot_day('01','08', 7, 19, 1)
# d.plot_persistence_day('02', '09', 6, 19, 1)


## DF for CNN
# data = Data(meteor_data=False, images=False, debug=True)
# data.build_df_for_cnn(10,17,1, months=[8])
# data.split_data_set(8, 28)
# data.flatten_data_set_CNN()

#
# data = Data(meteor_data=False, images=False, debug=False)
# data.build_df(10, 17, 1, months=[7, 8])
# data.set_prediction_horizon(5)
# data.split_data_set(8, 11)
# data.flatten_data_set()
#
# np.set_printoptions(precision=3)
# print(data.y_test)
# print(data.y_train)
#
# print(data.x_test)
# print(data.x_train)
