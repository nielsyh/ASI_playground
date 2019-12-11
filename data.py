import ftplib
import csv
import numpy as np
import pandas as pd
from os import listdir, path
import cv2
from datetime import time
from data_visuals import *
from PIL import Image
from metrics import Metrics
from pvlib_playground import PvLibPlayground


def int_to_str(i):
    s = str(i)
    if(len(s)) == 2:
        return s
    else:
        return '0' + s


def process_csv(csv_name):
    tmp = pd.read_csv(csv_name, sep=';', header=None)
    if(len(tmp.columns) > 1):
        arr = pd.read_csv(csv_name, sep=';', header=None, usecols=[0, 1, 2, 4, 7, 15])  # colums = ["DATE", "TIME", "RHUA", "TMPA", "PIRA"]

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

    def __init__(self):
        pass

    def download_data(self, cam = 1, overwrite = False):  # download data
        data = 0
        cam_url = 0
        file_url = 0

        if(cam  == 1):
            cam_url = "/asi16_data/asi_16124/"
            file_url = "asi_16124/"
            #todo add second

        server, username, passwd = get_credentials()
        ftp = ftplib.FTP(server)
        ftp.login(user=username, passwd=passwd)

        ftp.cwd(cam_url)
        files = ftp.nlst()

        for f in files:
            print("f:" + f)
            ftp.cwd((cam_url + str(f)))
            tmp_path = file_url + f + "/"
            print("path: " + tmp_path)

            f_ = ftp.nlst()
            for i in f_:
                file_name = (str(i))
                tmp_name = (tmp_path + str(i))
                if not path.isfile(tmp_name) or overwrite:  # check if file exists
                    if('.jpg' in i): # if image
                        image = open(tmp_name, 'wb')
                        ftp.retrbinary('RETR ' + file_name, image.write, 1024)
                        image.close()
                        # #TODO now you can do pre_processing
                    elif('.csv' in i):
                        csv = open(tmp_name, 'wb')
                        ftp.retrbinary('RETR ' + file_name, csv.write, 1024)
                        csv.close()
                        try:  # some have wrong encoding..
                            process_csv(tmp_name)
                        except:
                            print('Error processing: ' + file_name)
                else:
                    print('file ' + file_name + " exists.. no overwrite")

        return data

    def extract_time(self, time_str):
        return(time_str[8:14])

    def exract_formatted_time(self, time_str):
        s = time_str[8:14]
        return s[0:2] + ':' + s[2:4] + ':' + s[4:6]

    def extract_time_less_accurate(self, time_str):
        return(time_str[8:12])

    def wordListToFreqDict(self, wordlist):
        wordfreq = [wordlist.count(p) for p in wordlist]
        return dict(zip(wordlist, wordfreq))

    def resize_image(self,img, height, width):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    def get_avg_var_by_minute(self, df, hour, minute):
        rows = df[df[:, 3] == hour]  # filter on hours
        rows = rows[rows[:, 4] == minute]  # filter on minutes
        return np.mean(rows[:,6:9], axis=0), np.var(rows[:,6:9], axis=0)

    def get_ghi_temp_by_minute(self,df, hour, minute):

        print(df)
        rows = df[df[:, 3] == hour]  # filter on hours
        rows = df[np.where(df[:, 3] == hour)]
        print(rows)
        rows = rows[rows[:, 4] == minute]  # filter on minutes
        return rows

    def plot_per_month(self, month, start, end, step):
        df = self.get_df_csv_month(month, start, end, step)
        hours = list(range(start, end))
        minutes = list(range(0, 60, step))
        times, avg_temp, var_temp, avg_ghi, var_ghi, var_ghi, tick_times = ([] for i in range(7))

        for h in hours:
            tick_times.append(time(h, 0, 0))  #round hours
            tick_times.append(time(h, 30, 0))  # half hours
            for m in minutes:
                tmp_avg, tmp_var = self.get_avg_var_by_minute(df, h, m)
                tmp_time = time(h, m, 0)
                times.append(tmp_time)
                avg_temp.append(tmp_avg[0])
                var_temp.append(tmp_var[0])
                #todo tmp[1] = humidity
                avg_ghi.append(tmp_avg[2])
                var_ghi.append(tmp_var[2])
        # plot data
        plot_time_avg(tick_times, times, avg_temp, 'time', 'Temp. in celsius', 'avg. Temp. in month ' + str(month))
        plot_time_avg(tick_times, times, var_temp, 'time', 'Variance temp.','var. Temp. in month ' + str(month))
        plot_time_avg(tick_times, times, avg_ghi, 'time', 'GHI in W/m^2', 'avg. GHI in month ' + str(month))
        plot_time_avg(tick_times, times, var_ghi, 'time', 'Variance GHI', 'var. GHI in month ' + str(month))

    def plot_day(self, day, month, start, end, step):
        df = self.get_df_csv_day(month,day, start, end, step)
        hours = list(range(start, end))
        minutes = list(range(0, 60, step))
        times, temp, ghi, tick_times = ([] for i in range(4))

        ghi_clear_sky = PvLibPlayground.get_clear_sky_irradiance(int(month), int(day), start, end)

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

        #plot data
        plot_time_avg(tick_times, times, temp, '', 'time', 'temp. in celsius', 'temp. in day: ' + str(day) + ' month: ' + str(month))
        plot_time_avg(tick_times, times, ghi, 'GHI measured', 'time', 'GHI in W/m^2', 'GHI in day: ' + str(day) + ' month: ' + str(month), ghi_clear_sky, 'Clear sky GHI')

    def plot_persistence_day(self, day, month, start, end, step):

        df_truth = self.get_df_csv_day(month, day, start, end, step)
        previous_day = '0' + str(int(day) - 1) #cao ni mam fix this
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

                #sometimes data is missing then skip.
                if( len(rows_truth) > 0 and len(rows_pred) > 0):
                    ghi_truth_tmp = rows_truth[0][8]
                    ghi_pred_tmp = rows_pred[0][8]
                    times.append(tmp_time)
                    ghi_truth.append(ghi_truth_tmp)
                    ghi_pred.append(ghi_pred_tmp)

        # plot data
        plot_2_models(tick_times, times,ghi_truth, ghi_pred, 'time', 'GHI in W/m^2',
                      'GHI at day: ' + str(day) + ' month: ' + str(month))

    def get_prev_day(self, day):
        previous_day = str(int(day) - 1)  # cao ni mam fix this
        if len(previous_day) == 1:
            previous_day = '0' + previous_day
        return previous_day

    def get_error_month(self, month, start, end, step):
        y_observed = []
        y_predicted = []

        for i in range(2,30):

            day = str(i)
            if len(day) == 1:
                day = '0' + day

            df_truth = self.get_df_csv_day(month, day, start, end, step)
            df_pred = self.get_df_csv_day(month, self.get_prev_day(day), start, end, step) #todo include prev month

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


    def flatten(self,l):
        f = [item for sublist in l for item in sublist]
        return f

    def images_information(self):
        server, username, passwd = get_credentials()
        ftp = ftplib.FTP(server)
        ftp.login(user=username, passwd=passwd)

        ftp.cwd("/asi16_data/asi_16124")  # cam 1
        files = ftp.nlst()
        del files[0] #data not valid

        start_times, stop_times, times = ([] for i in range(3)) #times not used. too much data. unable to plot..
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


    def get_df_csv_month(self, month, start, end, step): #get data frame for a month with start and end time not inc. image
        folders = listdir('asi_16124')  # select cam
        del folders[0:3]  # first 3 are bad data
        index = 0
        queries = int(31 * ((end - start)*60/step))
        df = np.empty([queries, 9]) #create df

        for folder in folders: #fill df
            if(int(folder[4:6]) == month): #only check for month

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

                        if(df[index][6] == 0):
                            df[index][6] = tmp_temp
                        else:
                            tmp_temp = df[index][6]

                        index += 1
                        # print(index)
        # # YEAR, MONTH, DAY, HOURS, MINUTES, SECONDS, TEMP, IRRADIANCE, IMAGE
        print('filled queries: ' + str(index) + 'out of: ' + str(queries))
        return df.astype(int)

    def get_df_csv_day(self, month, day, start, end, step):

        path = 'asi_16124/2019' + int_to_str(month) + int_to_str(day) + '/'
        files = listdir(path)
        index = 0

        #data frame
        queries = int(((end - start) * 60 / step))
        df = np.empty([queries, 9])
        print(df.shape)

        #               dtype=[('a', 'i4'), ('b', 'i4'), ('c', 'i4'), ('d', 'i4'), ('e', 'i4'), ('f', 'i4'), ('g', 'i4'), ('h', 'i4'), ('i', 'i4')] )  # create df
        process_csv(path + files[-1])
        tmp_df = pd.read_csv(path + files[-1], sep=',', header=0, usecols=[0, 1, 2, 3, 4])  # load csv

        for row in tmp_df.iterrows():
            # check seconds 0, check step
            # todo add more data from solar pos

            if (int(row[1].values[1][6:8]) == 0 and int(row[1].values[1][3:5]) % step == 0 and int(
                    row[1].values[1][0:2]) >= start and int(row[1].values[1][0:2]) < end):
                df[index][0:9] = np.array([row[1].values[0][0:2], row[1].values[0][3:5], row[1].values[0][6:8],  # date
                                           row[1].values[1][0:2], row[1].values[1][3:5], row[1].values[1][6:8],  # time
                                           row[1].values[2],  # temp
                                           row[1].values[3],  # humidity
                                           row[1].values[4]])  # ghi  # set csv data to df
                index += 1

        print('filled queries: ' + str(index) + 'out of: ' + str(queries))
        print(df.dtype)
        return df[0:index]

    def get_df_csv_day_RP(self, month, day, start, end, step): #  replaces missing values with value of 15 seconds later.
        path = 'asi_16124/2019' + int_to_str(month) + int_to_str(day) + '/'
        files = listdir(path)
        index = 0

        #data frame
        queries = int(((end - start) * 60 / step))
        df = np.empty([queries, 9])  # create df

        process_csv(path + files[-1])
        tmp_df = pd.read_csv(path + files[-1], sep=',', header=0, usecols=[0, 1, 2, 3, 4])  # load csv

        min_idx = 0

        for i, row in tmp_df.iterrows():
            # todo add more data from solar pos
            if(index <= (i - min_idx) / 4 and int(row[1][3:5]) % step == 0 and int(   # some magic for missing values.
                    row[1][0:2]) >= start and int(row[1][0:2]) < end):
                df[index][0:9] = np.array([row[0][0:2], row[0][3:5], row[0][6:8],  # date
                                           row[1][0:2], row[1][3:5], row[1][6:8],  # time
                                           row[2],  # temp
                                           row[3],  # humidity
                                           row[4]])  # ghi  # set csv data to df
                index += 1
                if index == 1:  # some magic for missing values.
                    min_idx = i

        print('filled queries: ' + str(index) + ' out of: ' + str(queries))
        return df

    def sample_from_df(self, df, sample_size):  # sample random rows. returns new df with indexes.
        random_idx = np.random.randint(df.shape[0], size=sample_size)
        return df[random_idx, :], random_idx

    def build_df(self, days, start, end, step, images=False):
        # size 0 means all images
        index = 0

        if images:
            tmp_img = cv2.imread('asi_16124/20191006/20191006060800_11.jpg')  # random image assume here they are all the same.
            height, width, channels = tmp_img.shape
            size_of_row = 9 + height*width*channels
        else:
            size_of_row = 9

        queries_per_day = int(((end - start) * 60 / step))
        mega_df = np.empty((days, queries_per_day, size_of_row))

        months = [8]
        days = [1]

        for m in months:
            for d in days:
                mega_df[index] = self.get_df_csv_day_RP(m, d, start, end, step)
                index += 1


        #YEAR, MONTH, DAY, HOURS, MINUTES, SECONDS, TEMP, IRRADIANCE, IMAGE
        # print('filled queries: ' + str(index) + 'out of: ' + str(queries))
        return mega_df


#
d = Data()
# arr = d.build_df(1, 5, 19, 1)
# print(arr[0])
# d.download_data(1, True)
# d.process_csv("asi_16124/20190712/peridata_16124_20190712.csv")
# df = d.build_df(2)
# d.images_information()
# d.plot_per_month(9, 5, 19)
# d.plot_per_month(9, 5, 19, 5)
# d.plot_day('28', '08', 5 , 19, 1)
# d.plot_day('04', '09', 14, 15, 1)
# d.plot_persistence_day('02', '09', 6, 19, 1)
# d.plot_persistence_day('03', '09', 6, 19, 1)
# d.plot_persistence_day('04', '09', 6, 19, 1)
# d.plot_persistence_day('05', '09', 6, 19, 1)
# d.get_error_month('09', 6, 19, 1)

d.plot_day('05', '10', 5, 19, 1)
# df = d.get_df_csv_day('10','05',5,19)
# print(df[0][0:12])
# print(df[-1][0:12])
# # d.download_data()

#make df ready for future predictions
#filter unnesacasru dtuff out
#normalization
#simple features?
#save df


