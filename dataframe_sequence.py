import numpy as np
from data import month_to_year, PvLibPlayground, int_to_str, process_csv, get_df_csv_day_RP
import calendar
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import normalize

# from models_ts.ann_model import ANN_Predictor
from models_ts import ann_model, lstm_model


class DataFrameSequence:

    meteor_data = True
    mega_df_x_1 = None
    mega_df_y_1 = None

    mega_df_x_2 = None
    mega_df_y_2 = None

    train_x_df = None
    test_x_df = None
    val_x_df = None

    train_y_df = None
    test_y_df = None
    val_y_df = None

    number_of_features = 18
    sequence_len_minutes = 60

    cams = 1

    def __init__(self, debug, pred_horizon):
        self.debug = debug
        self.pred_horizon = pred_horizon

    def build_ts_df(self, start, end, months, lenth_tm, cams):
        self.start = start
        self.end = end
        self.months = months
        self.cams = cams
        self.sequence_len_minutes = lenth_tm

        print('BUILDING SEQUENCE DF: ')
        print('start: ' + str(start) + ' end: ' + str(end) + ' months: ' + str(months)
              + ' lenght: ' + str(lenth_tm) + ' cams: ' + str(cams))

        # time_steps = int((((end - start) / lenth_tm) * 60) - self.sequence_len_minutes)
        time_steps = int((end-start)*60 - lenth_tm) + 1
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

        self.mega_df_x_1 = np.zeros((days, int(time_steps), self.sequence_len_minutes, self.number_of_features), dtype=np.float)
        self.mega_df_y_1 = np.zeros((days, int(time_steps), 1), dtype=np.float)

        if cams == 2:
            self.mega_df_x_2 = np.zeros((days, int(time_steps), self.sequence_len_minutes, self.number_of_features), dtype=np.float)
            self.mega_df_y_2 = np.zeros((days, int(time_steps), 1), dtype=np.float)

        for m in tqdm(months, total=len(months), unit='Month progress'):
            days = list(range(1, calendar.monthrange(2019, m)[1] + 1))  # create an array with days for that month
            if self.debug:
                days = [26,27,28,29]
            elif m == 7:
                days = [26,27,28,29,30]

            for d in tqdm(days, total=len(days), unit='Days progress'):
                # todo add sunrise/sundown for start end hour? half hour?
                day_data = get_df_csv_day_RP(m, d, start, end+1, 1).astype(int)
                if cams == 2:
                    day_data_2 = get_df_csv_day_RP(m, d, start, end+1, 1, cam=2).astype(int)

                if self.meteor_data:
                    PvLibPlayground.set_cam(1)
                    csi, azimuth, zenith, sun_earth_dis, ephemeris = \
                        PvLibPlayground.get_meteor_data(m, d, PvLibPlayground.get_times(2019,
                                                                                       m,  # month
                                                                                       d,  # day
                                                                                       start,# start time
                                                                                       end))  # end time
                    if cams == 2:
                        PvLibPlayground.set_cam(2)
                        csi2, azimuth2, zenith2, sun_earth_dis2, ephemeris2 = \
                            PvLibPlayground.get_meteor_data(m, d, PvLibPlayground.get_times(2019,
                                                                                            m,  # month
                                                                                            d,  # day
                                                                                            start,  # start time
                                                                                            end))  # end time

                day_index += 1

                for i in range(0, time_steps):
                    minutes = self.sequence_len_minutes
                    variables = self.number_of_features

                    ts = np.zeros((minutes, variables))
                    if cams == 2:
                        ts2 = np.zeros((minutes, variables))

                    for v in range(variables):
                        if v < 9:
                            ts[0:minutes, v] = day_data[i:i + minutes, v]
                        elif v == 9:
                            ts[0:minutes, v] = csi[i:i + minutes]
                        elif v == 10:
                            a = day_data[i:i+minutes, 8]
                            b = csi[i:i + minutes]
                            ts[0:minutes, v] = [x/y for x, y in zip(a, b)]  # clear sky index
                        elif v == 11:
                            ts[0:minutes, v] = azimuth[i:i + minutes]
                        elif v == 12:
                            ts[0:minutes, v] = zenith[i:i + minutes]
                        elif v == 13:
                            ts[0:minutes, v] = sun_earth_dis[i:i + minutes]
                        elif v > 14:
                            ts[0:minutes, v] = [item[v - 14] for item in ephemeris[i:i+minutes]]

                        if cams == 2:
                            if v < 9:
                                ts2[0:minutes, v] = day_data_2[i:i + minutes, v]
                            elif v == 9:
                                ts2[0:minutes, v] = csi2[i:i + minutes]
                            elif v == 10:
                                a = day_data_2[i:i + minutes, 8]
                                b = csi2[i:i + minutes]
                                ts[0:minutes, v] = [x / y for x, y in zip(a, b)]  # clear sky index
                            elif v == 11:
                                ts2[0:minutes, v] = azimuth2[i:i + minutes]
                            elif v == 12:
                                ts2[0:minutes, v] = zenith2[i:i + minutes]
                            elif v == 13:
                                ts2[0:minutes, v] = sun_earth_dis2[i:i + minutes]
                            elif v > 14:
                                ts2[0:minutes, v] = [item[v - 14] for item in ephemeris2[i:i + minutes]]

                    self.mega_df_x_1[day_index, i] = ts
                    pred = i + (minutes-1) + self.pred_horizon
                    self.mega_df_y_1[day_index, i] = day_data[pred, 8]

                    if cams == 2:
                        self.mega_df_x_2[day_index, i] = ts2
                        self.mega_df_y_2[day_index, i] = day_data_2[pred, 8]


    def label_df(self):
        pass

    def split_data_set(self, m, d):
        print('Splitting with train until month: ' + str(m) + ', day: ' + str(d) + '...')
        self.month_split = m
        self.day_split = d

        day_idx = 0
        for idx, day in enumerate(self.mega_df_x_1):  # find index month
            if int(day[0][0][1]) == m and int(day[0][0][2]) == d and day_idx == 0:
                day_idx = idx
                print('found: ' + str(day_idx))
                break

        if self.cams == 1:
            self.train_x_df = np.copy(self.mega_df_x_1[0:day_idx])
            self.train_y_df = np.copy(self.mega_df_y_1[0:day_idx])

        else:  # double training data
            self.train_x_df = np.concatenate((np.copy(self.mega_df_x_1[0:day_idx]), np.copy(self.mega_df_x_2[0:day_idx])))
            self.train_y_df = np.concatenate((np.copy(self.mega_df_y_1[0:day_idx]), np.copy(self.mega_df_y_2[0:day_idx])))

        self.test_x_df = np.copy(self.mega_df_x_1[day_idx])
        self.val_x_df = np.copy(self.mega_df_x_1[day_idx + 1:self.mega_df_x_1.shape[0]])

        self.test_y_df = np.copy(self.mega_df_y_1[day_idx])
        self.val_y_df = np.copy(self.mega_df_y_1[day_idx + 1:self.mega_df_x_1.shape[0]])

        print('done')

    def flatten_data_set(self):  # this is needed to use it as input for models
        print('Flattening..')

        self.train_x_df = self.train_x_df.reshape(self.train_x_df.shape[0] * self.train_x_df.shape[1], self.sequence_len_minutes, self.number_of_features)
        self.train_x_df = self.train_x_df.reshape(self.train_x_df.shape[0], self.sequence_len_minutes*self.number_of_features)

        # self.test_x_df = self.test_x_df.reshape(self.test_x_df.shape[0] * self.test_x_df.shape[1], 60, 17)
        self.test_x_df = self.test_x_df.reshape(self.test_x_df.shape[0], self.sequence_len_minutes * self.number_of_features)

        self.val_x_df = self.val_x_df.reshape(self.val_x_df.shape[0] * self.val_x_df.shape[1], self.sequence_len_minutes, self.number_of_features)
        self.val_x_df = self.val_x_df.reshape(self.val_x_df.shape[0], self.sequence_len_minutes * self.number_of_features)

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

    def flatten_data_set_to_3d(self):
        print('Flattening..')

        self.train_x_df = self.train_x_df.reshape(self.train_x_df.shape[0] * self.train_x_df.shape[1], self.sequence_len_minutes, self.number_of_features)
        # self.train_x_df = self.train_x_df.reshape(self.train_x_df.shape[0], 60*17)

        # self.test_x_df = self.test_x_df.reshape(self.test_x_df.shape[0] * self.test_x_df.shape[1], 60, 17)
        # self.test_x_df = self.test_x_df.reshape(self.test_x_df.shape[0], 60 * 17)

        self.val_x_df = self.val_x_df.reshape(self.val_x_df.shape[0] * self.val_x_df.shape[1], self.sequence_len_minutes, self.number_of_features)
        # self.val_x_df = self.val_x_df.reshape(self.val_x_df.shape[0], 60 * 17)

        self.train_y_df.reshape(self.train_y_df.shape[0] * self.train_y_df.shape[1])
        self.train_y_df = self.train_y_df.reshape(self.train_y_df.shape[0]* self.train_y_df.shape[1])

        # self.test_y_df.reshape(self.test_y_df.shape[0] * self.test_y_df.shape[1])
        self.val_y_df.reshape(self.val_y_df.shape[0] * self.val_y_df.shape[1])
        self.val_y_df = self.val_y_df.reshape(self.val_y_df.shape[0]* self.val_y_df.shape[1])

        print('done')

    def save_df(self):
        name = 'mega_df'
        np.save(name + 'x', self.mega_df_x)
        np.save(name + 'y', self.mega_df_y)


    def load_prev_mega_df(self):  # todo get data from df
        self.start = 9
        self.end = 18
        self.step = 1
        self.months = [7,8,9,10,11,12]
        # self.queries_per_day = int(((self.end - self.start) * 60 / self.step))  # amount of data in one day
        self.mega_df_x = np.load('mega_dfx.npy')
        self.mega_df_x = np.load('mega_dfy.npy')
        print('loaded')

#
# data = DataFrameSequence(False, 20)
# data.build_ts_df(10,13,[9],45,2)
# data.split_data_set(9,27)

# data = DataFrameSequence(False, 20)
# data.build_ts_df(9,18,[7,8,9,10],1)
# # data.save_df()
# # data.split_data_set(8,25)
# # data.flatten_data_set_to_3d()
# # data.normalize_data_sets()
# # data.load_prev_mega_df()
# ann = ann_model.ANN(data, 50, 50)
# ann.run_experiment()
# # lstm = lstm_model.LSTM_predictor(data, 400,100)
# # lstm.get_model()
# # lstm.run_experiment()
