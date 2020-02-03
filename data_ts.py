import numpy as np
from data import month_to_year, PvLibPlayground, int_to_str, process_csv
import calendar
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import normalize

# from models_ts.ann_model import ANN_Predictor
from models_ts import ann_model


class Data_TS:

    meteor_data = True
    mega_df_x = None
    mega_df_y = None

    train_x_df = None
    test_x_df = None
    val_x_df = None

    train_y_df = None
    test_y_df = None
    val_y_df = None

    def __init__(self, debug, pred_horizon):
        self.debug = debug
        self.pred_horizon = pred_horizon

    def build_ts_df(self, start, end, months, lenth_tm):
        self.start = start
        self.end = end
        self.months = months

        time_steps = int((((end - start) / lenth_tm) * 60) - 60)

        days = 0
        day_index = -1

        print('Processing months: ' + str(months))

        for m in months:
            if self.debug:  # debug
                days += 4
            elif m == 7:
                days += 6
            else:
                year = int(month_to_year(m))
                days += calendar.monthrange(year, m)[1]

        print('timesteps: ', time_steps)

        self.mega_df_x = np.zeros((days, int(time_steps), 60, 17), dtype=np.float)
        self.mega_df_y = np.zeros((days, int(time_steps), 1), dtype=np.float)

        for m in tqdm(months, total=len(months), unit='Month progress'):

            days = list(range(1, calendar.monthrange(2019, m)[1] + 1))  # create an array with days for that month
            if self.debug:
                days = [26,27,28,29]

            for d in tqdm(days, total=len(days), unit='Days progress'):
                # todo add sunrise/sundown for start end hour? half hour?
                day_data = self.get_df_csv_day_RP(m, d, start, end+1, 1).astype(int)
                if self.meteor_data:
                    csi, azimuth, zenith, sun_earth_dis, ephemeris = \
                        PvLibPlayground.get_meteor_data(m, d, PvLibPlayground.get_times(2019,
                                                                                       m,  # month
                                                                                       d,  # day
                                                                                       start,# start time
                                                                                       end))  # end time
                day_index += 1

                for i in range(0, time_steps):
                    minutes = 60
                    variables = 17

                    ts = np.zeros((minutes, variables))
                    for v in range(variables):
                        if v < 9:
                            ts[0:60, v] = day_data[i:i + 60, v]
                        elif v == 9:
                            ts[0:60, v] = csi[i:i + 60]
                        elif v == 10:
                            ts[0:60, v] = azimuth[i:i + 60]
                        elif v == 11:
                            ts[0:60, v] = zenith[i:i + 60]
                        elif v == 12:
                            ts[0:60, v] = sun_earth_dis[i:i + 60]
                        elif v > 12:
                            ts[0:60, v] = [item[v - 13] for item in ephemeris[i:i+60]]

                    self.mega_df_x[day_index, i] = ts
                    pred = i + 59 + self.pred_horizon
                    self.mega_df_y[day_index, i] = day_data[pred, 8]

    def label_df(self):
        pass

    def split_data_set(self, m, d):
        print('Splitting with train until month: ' + str(m) + ', day: ' + str(d) + '...')
        self.month_split = m
        self.day_split = d

        day_idx = 0
        for idx, day in enumerate(self.mega_df_x):  # find index month
            if int(day[0][0][1]) == m and int(day[0][0][2]) == d and day_idx == 0:
                day_idx = idx
                print('found: ' + str(day_idx))
                break

        self.train_x_df = np.copy(self.mega_df_x[0:day_idx])
        self.test_x_df = np.copy(self.mega_df_x[day_idx])
        self.val_x_df = np.copy(self.mega_df_x[day_idx+1:self.mega_df_x.shape[0]])

        self.train_y_df = np.copy(self.mega_df_y[0:day_idx])
        self.test_y_df = np.copy(self.mega_df_y[day_idx])
        self.val_y_df = np.copy(self.mega_df_y[day_idx+1:self.mega_df_x.shape[0]])


        print('done')

    def flatten_data_set(self):  # this is needed to use it as input for models
        print('Flattening..')

        self.train_x_df = self.train_x_df.reshape(self.train_x_df.shape[0] * self.train_x_df.shape[1], 60, 17)
        self.train_x_df = self.train_x_df.reshape(self.train_x_df.shape[0], 60*17)

        # self.test_x_df = self.test_x_df.reshape(self.test_x_df.shape[0] * self.test_x_df.shape[1], 60, 17)
        self.test_x_df = self.test_x_df.reshape(self.test_x_df.shape[0], 60 * 17)

        self.val_x_df = self.val_x_df.reshape(self.val_x_df.shape[0] * self.val_x_df.shape[1], 60, 17)
        self.val_x_df = self.val_x_df.reshape(self.val_x_df.shape[0], 60 * 17)

        self.train_y_df.reshape(self.train_y_df.shape[0] * self.train_y_df.shape[1])
        self.train_y_df = self.train_y_df.reshape(self.train_y_df.shape[0]* self.train_y_df.shape[1])

        # self.test_y_df.reshape(self.test_y_df.shape[0] * self.test_y_df.shape[1])
        self.val_y_df.reshape(self.val_y_df.shape[0] * self.val_y_df.shape[1])
        self.val_y_df = self.val_y_df.reshape(self.val_y_df.shape[0]* self.val_y_df.shape[1])

        print('done')

    def normalize_data_sets(self, colums_to_normalize = [6,7,8], metoer_to_normalize= [9,12,16]):
        colums_to_normalize = colums_to_normalize
        if (self.meteor_data):
            colums_to_normalize.extend(metoer_to_normalize)

        print('normalzing for: ' + str(colums_to_normalize))


        self.train_x_df[:, colums_to_normalize] = normalize(self.train_x_df[:,colums_to_normalize], axis=0, norm='l2')
        self.test_x_df[:,colums_to_normalize] = normalize(self.test_x_df[:, colums_to_normalize], axis=0, norm='l2')
        self.val_x_df[:,colums_to_normalize] = normalize(self.val_x_df[:, colums_to_normalize], axis=0, norm='l2')


    def get_df_csv_day_RP(self, month, day, start, end,
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

data = Data_TS(False, 20)
data.build_ts_df(9,18,[8,9,10,11,12],1)
data.split_data_set(8,28)
data.flatten_data_set()
data.normalize_data_sets()

ann = ann_model.ANN(data, 200, 100)
ann.get_model()
ann.run_experiment()
# ann.train(50, 50)