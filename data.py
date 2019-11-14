import ftplib
import csv
import numpy as np
import pandas as pd
from os import listdir
import numpy as np
import cv2
from datetime import time
from data_visuals import *
from PIL import Image

class Data:

    def __init__(self):
        pass

    def download_data(self):
        data = 0

        server, username, passwd = self.get_credentials()
        ftp = ftplib.FTP(server)
        ftp.login(user=username, passwd=passwd)

        ftp.cwd("/asi16_data/asi_16124") #cam 1
        files  = ftp.nlst()

        for f in files:
            print(f)
            ftp.cwd(("/asi16_data/asi_16124/" + str(f)))

            f_ = ftp.nlst()
            for i in f_:
                file_name = (str(i))
                #if image
                if('.jpg' in i):
                    tmp_name = ('image/' + str(i))
                    image = open(tmp_name, 'wb')
                    ftp.retrbinary('RETR ' + file_name, image.write, 1024)
                    image.close()
                    #TODO now you can do pre_processing

                elif('.csv' in i):
                    tmp_name = ('csv/' + str(i))
                    csv = open(tmp_name, 'wb')
                    ftp.retrbinary('RETR ' + file_name, csv.write, 1024)
                    self.process_csv(tmp_name)

        return data

    def process_csv(self, csv_name):
        # colums = ["SQNR", "DATE", "TIME", "TMPA", "PIRA"]
        tmp = pd.read_csv(csv_name, sep=';', header=None)
        if(len(tmp.columns) > 1):
            arr = pd.read_csv(csv_name, sep=';', header=None, usecols=[0, 1, 2, 4, 15])

            #remove rows containing c,v,r not sure what it means..
            arr = arr[arr[0] != 'V-----']
            arr = arr[arr[0] != 'C-----']
            arr = arr[arr[0] != 'R-----']

            c = [0, 1, 2, 4, 15]
            for index, row in arr.iterrows():
                for i in c:
                    try:
                        vals = row[i].split('=')
                        row[i] = vals[1]
                    except:
                        print(row[i])

            arr.to_csv(csv_name)

        del tmp

    def get_credentials(self):
        f = open('cred.txt', 'r')
        lines = f.read().split(',')
        f.close()
        # print(lines)
        return lines[0], lines[1], lines[2]

    def extract_time(self, time_str):
        return(time_str[8:14])

    def exract_formatted_time(self, time_str):
        s = time_str[8:14]
        return s[0:2] + ':' + s[2:4] + ':' + s[4:6]

    def extract_time_less_accurate(self, time_str):
        return(time_str[8:12])

    def pre_process(self):
        pass

    def sample(self):
        pass

    def wordListToFreqDict(self, wordlist):
        wordfreq = [wordlist.count(p) for p in wordlist]
        return dict(zip(wordlist, wordfreq))

    def resize_image(self,img, height, width):
        # setting dim of the resize
        # name = ('image/20190716084100_11.jpg')
        # img = cv2.imread(name)
        # image = cv2.resize(np.float32(img), (800, 800), interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(name + str('._prep.jpg'), image)
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    def get_avg_var_by_minute(self, df, hour, minute):
        rows = df[df[:, 3] == hour] # filter on hours
        rows = rows[rows[:, 4] == minute] # filter on minutes
        return np.mean(rows[:,6:8], axis=0), np.var(rows[:,6:8], axis=0)

    def get_ghi_temp_by_minute(self,df, hour, minute):
        rows = df[df[:, 3] == hour]  # filter on hours
        rows = rows[rows[:, 4] == minute]  # filter on minutes
        return rows


    def plot_per_month(self, month, start, end, step):
        df = self.get_df_csv_month(month, start, end, step)
        hours = list(range(start, end))
        minutes = list(range(0, 60, step))
        times, avg_temp, var_temp, avg_ghi, var_ghi, var_ghi, tick_times = ([] for i in range(7))

        for h in hours:
            tick_times.append(time(h, 0, 0)) #round hours
            tick_times.append(time(h, 30, 0)) # half hours
            for m in minutes:
                tmp_avg, tmp_var = self.get_avg_var_by_minute(df, h, m)
                tmp_time = time(h, m, 0)
                times.append(tmp_time)
                avg_temp.append(tmp_avg[0])
                var_temp.append(tmp_var[0])
                avg_ghi.append(tmp_avg[1])
                var_ghi.append(tmp_var[1])
        #plot data
        plot_time_avg(tick_times, times, avg_temp, 'time', 'Temp. in celsius', 'avg. Temp. in month ' + str(month))
        plot_time_avg(tick_times, times, var_temp, 'time', 'Variance temp.','var. Temp. in month ' + str(month))
        plot_time_avg(tick_times, times, avg_ghi, 'time', 'GHI in W/m^2', 'avg. GHI in month ' + str(month))
        plot_time_avg(tick_times, times, var_ghi, 'time', 'Variance GHI', 'var. GHI in month ' + str(month))


    def plot_day(self, day, month, start, end, step):
        df = self.get_df_csv_day(month,day, start, end, step)
        hours = list(range(start, end))
        minutes = list(range(0, 60, step))
        times, temp, ghi, tick_times = ([] for i in range(4))

        for h in hours:
            tick_times.append(time(h, 0, 0)) #round hours
            tick_times.append(time(h, 30, 0)) # half hours
            for m in minutes:
                rows = self.get_ghi_temp_by_minute(df, h, m)
                tmp_time = time(h, m, 0)
                # print('---------')
                # print(rows)
                # print(str(rows) + '' + str(m))
                tmp_temp = rows[0][6]
                tmp_ghi = rows[0][7]
                # print(tmp_temp)
                # print(tmp_ghi)

                times.append(tmp_time)
                temp.append(tmp_temp)
                ghi.append(tmp_ghi)

        #plot data
        plot_time_avg(tick_times, times, temp, 'time', 'temp. in celsius', 'temp. in day: ' + str(day) + ' month: ' + str(month))
        plot_time_avg(tick_times, times, ghi, 'time', 'GHI in W/m^2', 'GHI in day: ' + str(day) + ' month: ' + str(month))



    def images_information(self):
        server, username, passwd = self.get_credentials()
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
        df = np.empty([queries, 8]) #create df

        for folder in folders: #fill df
            if(int(folder[4:6]) == month): #only check for month

                path = 'asi_16124/' + str(folder) + '/'
                files = listdir(path)

                self.process_csv(path + files[-1])  # process csv
                tmp_df = pd.read_csv(path + files[-1], sep=',', header=0, usecols=[2, 3, 4, 5])  # load csv
                tmp_temp = None

                for row in tmp_df.iterrows():
                    if( int(row[1].values[1][6:8]) == 0 and int(row[1].values[1][3:5])%step == 0 and int(row[1].values[1][0:2]) >= start and int(row[1].values[1][0:2]) < end):
                        df[index][0:8] = np.array([row[1].values[0][0:2], row[1].values[0][3:5], row[1].values[0][6:8],
                                                   row[1].values[1][0:2], row[1].values[1][3:5], row[1].values[1][6:8],
                                                   row[1].values[2], row[1].values[3]])  # set csv data to df

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
        path = 'asi_16124/2019' + month + day + '/'
        files = listdir(path)
        index = 0
        queries = int(((end - start) * 60 / step))
        df = np.empty([queries, 8])  # create df
        self.process_csv(path + files[-1])
        tmp_df = pd.read_csv(path + files[-1], sep=',', header=0, usecols=[2, 3, 4, 5])  # load csv

        for row in tmp_df.iterrows():
            if (int(row[1].values[1][6:8]) == 0 and int(row[1].values[1][3:5]) % step == 0 and int(
                    row[1].values[1][0:2]) >= start and int(row[1].values[1][0:2]) < end):
                df[index][0:8] = np.array([row[1].values[0][0:2], row[1].values[0][3:5], row[1].values[0][6:8],
                                           row[1].values[1][0:2], row[1].values[1][3:5], row[1].values[1][6:8],
                                           row[1].values[2], row[1].values[3]])  # set csv data to df
                index += 1

        print('filled queries: ' + str(index) + 'out of: ' + str(queries))
        return df.astype(int)



    def build_df(self, queries):
        # size 0 means al images
        index = 0
        tmp_img = cv2.imread('asi_16124/20191006/20191006060800_11.jpg')  # random image assume here they are all the same
        height, width, channels = tmp_img.shape
        size_of_row = 8 + height*width*channels
        df = np.empty([queries, size_of_row])

        folders = listdir('asi_16124')#select cam
        del folders[0:3] #first 3 are bad data

        for folder in folders:
            if index == queries:
                break

            path = 'asi_16124/' + str(folder) + '/'
            files = listdir(path)

            self.process_csv(path + files[-1])  # process csv
            tmp_df = pd.read_csv(path + files[-1], sep=',', header=0, usecols=[2,3,4,5])  #load csv
            # print(tmp_df)
            for file in files:
                if index == queries: #cancel if dataframe is full
                    break

                arr = tmp_df[tmp_df['2'] == self.exract_formatted_time(file)] #find image in cvs by time
                if(arr.empty): #means if image is not in csv file
                    continue

                data = arr.to_numpy().flatten()
                #index in csv, seconds, year, month, day, hours, minutes, seconds, temp, irriadiance. then image
                df[index][0:8] = np.array([data[0][0:2], data[0][3:5], data[0][6:8], data[1][0:2], data[1][3:5], data[1][6:8], data[2], data[3]]) #set csv data to df
                img  = cv2.imread(path + file)
                df[index][8:] = img.flatten() #set img data to df

                index += 1
                print(index)

        #YEAR, MONTH, DAY, HOURS, MINUTES, SECONDS, TEMP, IRRADIANCE, IMAGE
        print('filled queries: ' + str(index) + 'out of: ' + str(queries))
        return df.astype(int)


#
d = Data()
# df = d.build_df(2)
# d.images_information()
# d.plot_per_month(9, 5, 19)
# d.plot_per_month(9, 5, 19, 5)
# d.plot_day('05', '09', 5 , 19)
d.plot_day('07', '09', 5, 19, 2)
# d.plot_day('05', '10', 5 , 19)
# df = d.get_df_csv_day('10','05',5,19)
# print(df[0][0:12])
# print(df[-1][0:12])
# # d.download_data()

#make df ready for future predictions
#filter unnesacasru dtuff out
#normalization
#simple features?
#save df

