import numpy as np

from data.data_helper import month_to_year, get_df_csv_day_RP
from pvlib_playground import PvLibPlayground
import data.data_helper
import calendar
from tqdm import tqdm
from sklearn.preprocessing import normalize
from features import get_features_by_day

class DataFrameSequenceMulti:

    meteor_data = True
    mega_df_x_1 = None
    df_norm = None
    mega_df_y_1 = None
    mega_df_label = None

    mega_df_x_2 = None
    mega_df_y_2 = None

    train_x_df = None
    test_x_df = None
    test_x_norm = None
    val_x_df = None

    train_y_df = None
    test_y_df = None
    test_label_df = None
    val_y_df = None

    features = None

    number_of_features = 6
    onsite_features  = 3
    meteor_features = 9
    img_features = 4
    sequence_len_minutes = 60
    gradients = False
    cams = 1

    def __init__(self, debug, onsite_data, img_data, meteor_data, gradients=True):
        self.debug = debug
        self.onsite_data  = onsite_data
        self.img_data = img_data
        self.meteor_data = meteor_data
        self.clear_sky_label = False

        if gradients:
            print('Gradient enabled')
            self.onsite_features = 3*2
            self.img_features = 4*2
            self.gradients = True


        # first onsite
        if onsite_data:
            self.number_of_features += self.onsite_features
        if meteor_data:
            self.meteor_idx = self.number_of_features
            self.number_of_features = self.number_of_features + self.meteor_features
            # self.index_img = 18
        if img_data:
            self.img_idx = self.number_of_features
            self.number_of_features = self.number_of_features + self.img_features
            self.load_features()
            print('DF with images')

        if img_data and self.cams == 2:
            raise ValueError('Cams 2 and images not possible')

        print('Total size: ' + str(self.number_of_features))

    def load_features(self):
        data.data_helper.fix_directory()
        self.features = np.load('x_22_d6to19_m7to12.npy')

    def get_feature_data(self, month, day, start, end):
        start_time_idx = (start - 6)*60
        end_time_idx = (end - 6)*60
        previous_days = 0

        month_list = list(range(7,month+1))
        for i in month_list:
            if i == 7:
                continue
            if i == 8:
                previous_days += 5
            else:
                previous_days += calendar.monthrange(2019, i-1)[1]
        if month == 7:
            day_idx = day - 26
        else:
            day_idx = previous_days + day
        return self.features[day_idx, start_time_idx:end_time_idx]


    def build_ts_df(self, start, end, months, lenth_tm, cams=1, clear_sky_label = False):

        self.start = start
        self.end = end
        self.months = months
        self.cams = cams
        self.sequence_len_minutes = lenth_tm
        self.clear_sky_label = clear_sky_label

        print('BUILDING SEQUENCE DF: ')
        print('start: ' + str(start) + ' end: ' + str(end) + ' months: ' + str(months)
              + ' lenght: ' + str(lenth_tm) + ' cams: ' + str(cams))

        time_steps = int(((end - start) * 60 - lenth_tm))
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
        self.mega_df_y_1 = np.zeros((days, int(time_steps), 20), dtype=np.float)
        self.mega_df_label = np.zeros((days, int(time_steps), 20), dtype=np.float)

        if cams == 2:
            self.mega_df_x_2 = np.zeros((days, int(time_steps), self.sequence_len_minutes, self.number_of_features), dtype=np.float)
            self.mega_df_y_2 = np.zeros((days, int(time_steps), 20), dtype=np.float)

        for m in tqdm(months, total=len(months), unit='Month progress'):
            days = list(range(1, calendar.monthrange(2019, m)[1] + 1))  # create an array with days for that month
            if self.debug:
                days = [26,27,28,29]
            elif m == 7:
                days = [26,27,28,29,30,31]

            for d in tqdm(days, total=len(days), unit='Days progress'):
                # todo add sunrise/sundown for start end hour? half hour?
                day_data = get_df_csv_day_RP(m, d, start, end+1, 1).astype(int)
                if cams == 2:
                    day_data2 = get_df_csv_day_RP(m, d, start, end+1, 1, cam=2).astype(int)

                if self.img_data:
                    # intensity, cloudpixels, corners, edges = get_features_by_day(m, d, start, end+1)
                    f = self.get_feature_data(m ,d, start, end+1)
                    if cams == 2:
                        f2 = self.get_feature_data(m ,d, start, end+1)


                if self.meteor_data:  # get metoer data for 1st location
                    PvLibPlayground.set_cam(1)
                    csi, azimuth, zenith, sun_earth_dis, ephemeris = \
                        PvLibPlayground.get_meteor_data(m, d, PvLibPlayground.get_times(2019,
                                                                                        m,  # month
                                                                                        d,  # day
                                                                                        start,# start time
                                                                                        end+1))  # end time
                    if cams == 2:
                        PvLibPlayground.set_cam(2)
                        csi2, azimuth2, zenith2, sun_earth_dis2, ephemeris2 = \
                            PvLibPlayground.get_meteor_data(m, d, PvLibPlayground.get_times(2019,
                                                                                            m,  # month
                                                                                            d,  # day
                                                                                            start,  # start time
                                                                                            end+1))  # end time

                day_index += 1

                for i in range(0, time_steps):
                    minutes = self.sequence_len_minutes
                    variables = self.number_of_features

                    ts = np.zeros((minutes, variables))
                    if cams == 2:
                        ts2 = np.zeros((minutes, variables))

                    for v in range(variables):

                        if v < 6:  # always day data 0:5
                            ts[0:minutes, 0:6] = day_data[i:i + minutes, 0:6]

                        elif self.onsite_data and v < 6 + self.onsite_features:  # onsite features
                            ts[0:minutes, 6:9] = day_data[i:i + minutes, 6:9]
                            if self.gradients:
                                ts[0:minutes, 9:12] = np.gradient(day_data[i:i + minutes, 6:9])[0]

                        if self.meteor_data and v >= self.meteor_idx and v < self.meteor_idx + self.meteor_features:
                            if v == self.meteor_idx:
                                ts[0:minutes, self.meteor_idx] = csi[i:i + minutes]

                            elif v == self.meteor_idx + 1:
                                a = day_data[i:i + minutes, 8]
                                b = csi[i:i + minutes]
                                c = []
                                for x,y in zip(a,b):
                                    if y == 0:
                                        c.append(0)
                                    else:
                                        c.append(x/y)
                                ts[0:minutes, v] = c  # clear sky index

                            elif v == self.meteor_idx + 2:
                                ts[0:minutes, v] = azimuth[i:i + minutes]
                            elif v == self.meteor_idx + 3:
                                ts[0:minutes, v] = zenith[i:i + minutes]
                            elif v == self.meteor_idx + 4:
                                ts[0:minutes, v] = sun_earth_dis[i:i + minutes]
                            elif v > self.meteor_idx + 4 and v < self.meteor_idx + 9:
                                x = 14
                                if self.gradients:
                                    x = 17
                                ts[0:minutes, v] = [item[v - x] for item in ephemeris[i:i+minutes]]

                        # if img
                        if self.img_data:
                            if v == self.img_idx:
                                ts[0:minutes, v:v+4] = f[i:i + minutes, 18:22]
                                if self.gradients:
                                    ts[0:minutes, v+4:v+8] = np.gradient(f[i:i + minutes, 18:22])[0]


                        if cams == 2:
                            for v in range(variables):

                                if v < 6:  # always day data 0:5
                                    ts2[0:minutes, 0:6] = day_data2[i:i + minutes, 0:6]

                                elif self.onsite_data and v < 9:  # onsite features
                                    ts2[0:minutes, 6:9] = day_data2[i:i + minutes, 6:9]
                                    if self.gradients:
                                        ts2[0:minutes, 9:12] = np.gradient(day_data2[i:i + minutes, 6:9])[0]

                                if self.meteor_data and v >= self.meteor_idx and v < self.meteor_idx + self.meteor_features:
                                    if v == self.meteor_idx:
                                        ts2[0:minutes, self.meteor_idx] = csi2[i:i + minutes]

                                    elif v == self.meteor_idx + 1:
                                        a = day_data2[i:i + minutes, 8]
                                        b = csi2[i:i + minutes]
                                        c = []
                                        for x, y in zip(a, b):
                                            if y == 0:
                                                c.append(0)
                                            else:
                                                c.append(x / y)
                                        ts2[0:minutes, v] = c  # clear sky index

                                    elif v == self.meteor_idx + 2:
                                        ts2[0:minutes, v] = azimuth2[i:i + minutes]
                                    elif v == self.meteor_idx + 3:
                                        ts2[0:minutes, v] = zenith2[i:i + minutes]
                                    elif v == self.meteor_idx + 4:
                                        ts2[0:minutes, v] = sun_earth_dis2[i:i + minutes]
                                    elif v > self.meteor_idx + 4 and v < self.meteor_idx + 9:
                                        x = 14
                                        if self.gradients:
                                            x = 17
                                        ts2[0:minutes, v] = [item[v - x] for item in ephemeris2[i:i + minutes]]

                                # if img
                                if self.img_data:
                                    if v == self.img_idx:
                                        ts2[0:minutes, v:v + 4] = f2[i:i + minutes, 18:22]
                                        if self.gradients:
                                            ts2[0:minutes, v + 4:v + 8] = np.gradient(f2[i:i + minutes, 18:22])[0]


                    self.mega_df_x_1[day_index, i] = ts
                    first = i + (minutes-1) + 1
                    last = first + 20
                    if not clear_sky_label:
                        self.mega_df_y_1[day_index, i] = day_data[first:last, 8]
                    else:
                        self.mega_df_y_1[day_index, i] = PvLibPlayground.calc_clear_sky_ls(day_data[first:last, 8],csi[first:last])
                        self.mega_df_label[day_index, i] = day_data[first:last, 8]

                    if cams == 2:
                        self.mega_df_x_2[day_index, i] = ts2
                        if not clear_sky_label:
                            self.mega_df_y_2[day_index, i] = day_data2[first:last, 8]


    def label_df(self):
        pass

    def get_prem_days(self):  # prem test days
        return [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]

    def get_all_test_days(self):  # copernicus test days
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

    def get_thesis_test_days(self):  # test days personal research
        sunny = [(9, 15), (10, 15), (11, 15), (12, 15)]
        pcloudy = [(10, 21), (11, 17), (12, 16)]
        cloudy = [(10, 22), (12, 3)]
        total = []
        total.extend(sunny)
        total.extend(pcloudy)
        total.extend(cloudy)
        return total

    def split_data_set_EXPRMTL(self, m, d, val_days):
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
            # self.train_x_df = np.copy(self.df_norm[0:day_idx-val_days])
            self.train_x_df = np.copy(self.mega_df_x_1[0:day_idx - val_days])
            self.train_y_df = np.copy(self.mega_df_y_1[0:day_idx-val_days])

        elif self.cams == 2:  # double training data
            self.train_x_df = np.concatenate((np.copy(self.mega_df_x_1[0:day_idx-val_days]), np.copy(self.mega_df_x_2[0:day_idx-val_days])))
            self.train_y_df = np.concatenate((np.copy(self.mega_df_y_1[0:day_idx-val_days]), np.copy(self.mega_df_y_2[0:day_idx-val_days])))

        self.test_x_df = np.copy(self.mega_df_x_1[day_idx])
        # self.test_x_norm = np.copy(self.df_norm[day_idx])
        # self.val_x_df = np.copy(self.df_norm[day_idx-val_days:day_idx])
        self.val_x_df = np.copy(self.mega_df_x_1[day_idx - val_days:day_idx])

        self.test_y_df = np.copy(self.mega_df_y_1[day_idx])
        self.val_y_df = np.copy(self.mega_df_y_1[day_idx-val_days:day_idx])

        if self.clear_sky_label:
            self.test_label_df = np.copy(self.mega_df_label[day_idx])

        print('done')

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

        elif self.cams == 2:  # double training data
            self.train_x_df = np.concatenate((np.copy(self.mega_df_x_1[0:day_idx]), np.copy(self.mega_df_x_2[0:day_idx])))
            self.train_y_df = np.concatenate((np.copy(self.mega_df_y_1[0:day_idx]), np.copy(self.mega_df_y_2[0:day_idx])))
        self.test_x_df = np.copy(self.mega_df_x_1[day_idx])
        self.val_x_df = np.copy(self.mega_df_x_1[day_idx + 1:self.mega_df_x_1.shape[0]])

        self.test_y_df = np.copy(self.mega_df_y_1[day_idx])
        self.val_y_df = np.copy(self.mega_df_y_1[day_idx + 1:self.mega_df_x_1.shape[0]])

        if self.clear_sky_label:
            self.test_label_df = np.copy(self.mega_df_label[day_idx])

        print('done')

    def scale_mega(self, model='ann'):
        from sklearn import preprocessing

        if model == 'ann':
            ctn = [0,self.img_idx, self.img_idx +1 , self.img_idx+ 2, self.img_idx + 3]
        if model == 'lstm':
            # ctn = [0, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,25]
            ctn = [0, 6, 7, 8]

        print('scaling for :')
        print(ctn)

        shape = self.mega_df_x_1.shape
        a = np.copy(self.mega_df_x_1.reshape(shape[0] * shape[1] * shape[2], shape[3]))


        # a[:, ctn] = preprocessing.scale(a[:, ctn])
        min_max_scaler = preprocessing.MinMaxScaler()
        a[:, ctn] = min_max_scaler.fit_transform(a[:, ctn])

        self.mega_df_x_1 = a.reshape(shape)



    def normalize_mega_EXPRTML(self, norm=True):
        from sklearn.preprocessing import StandardScaler  # for normalization
        from sklearn.preprocessing import MinMaxScaler

        shape = self.mega_df_x_1.shape
        a = np.copy(self.mega_df_x_1.reshape(shape[0]* shape[1]* shape[2], shape[3]))

        if norm:
            ctn = [0,1,2,3,4,5,6,7,8,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
            ctn = [0]
            a[:,ctn] = normalize(a[:, ctn], axis=0, norm='l2')

        self.df_norm = a.reshape(shape)

    def normalize_mega_df(self):
        # norm_onsite = [1, 2, 3, 4, 5, 6, 7, 8]
        norm_onsite = [6,7,8]
        norm_onsite = [0]
        if self.meteor_data:
            norm_metoer = []
            # norm_metoer = [self.meteor_idx,self.meteor_idx+1, self.meteor_idx + 4, self.meteor_idx + 5, self.meteor_idx + 6, self.meteor_idx + 7, self.meteor_idx + 8] #[9, 13, 17]
        if self.img_data:
            norm_img = [self.img_idx, self.img_idx +1 , self.img_idx+ 2, self.img_idx + 3] #, self.img_idx + 4, self.img_idx + 5 , self.img_idx +6, self.img_idx +7]
            norm_img = []

        colums_to_normalize = []
        if self.onsite_data:
            colums_to_normalize.extend(norm_onsite)
        if self.meteor_data:
            colums_to_normalize.extend(norm_metoer)
        if self.img_data:
            colums_to_normalize.extend(norm_img)

        print('Mega normalzing for: ' + str(colums_to_normalize))

        shape = self.mega_df_x_1.shape
        a = np.copy(self.mega_df_x_1.reshape(shape[0] * shape[1] * shape[2], shape[3]))
        # a = self.mega_df_x_1.reshape(shape[2], shape[0]*shape[1]*shape[3])
        a[:, colums_to_normalize]  = normalize(a[:, colums_to_normalize], axis=0, norm='l2')
        self.mega_df_x_1 = a.reshape(shape)

        if self.cams == 2:
            shape = self.mega_df_x_2.shape
            a = self.mega_df_x_2.reshape(shape[2], shape[0] * shape[1] * shape[3])
            a[:, colums_to_normalize]  = normalize(a[:, colums_to_normalize], axis=0, norm='l2')
            self.mega_df_x_2 = a.reshape(shape)

    def flatten_data_set_to_3d(self):
        print('Flattening..')
        self.train_x_df = self.train_x_df.reshape(self.train_x_df.shape[0] * self.train_x_df.shape[1], self.sequence_len_minutes, self.number_of_features)
        self.val_x_df = self.val_x_df.reshape(self.val_x_df.shape[0] * self.val_x_df.shape[1], self.sequence_len_minutes, self.number_of_features)
        self.train_y_df = self.train_y_df.reshape(self.train_y_df.shape[0]* self.train_y_df.shape[1],20)
        self.val_y_df = self.val_y_df.reshape(self.val_y_df.shape[0]* self.val_y_df.shape[1], 20)
        print('done')

    def flatten_data_set(self):  # this is needed to use it as input for models
        print('Flattening..')
        self.train_x_df = self.train_x_df.reshape(self.train_x_df.shape[0] * self.train_x_df.shape[1],
                                                  self.sequence_len_minutes, self.number_of_features)
        self.train_x_df = self.train_x_df.reshape(self.train_x_df.shape[0],
                                                  self.sequence_len_minutes * self.number_of_features)

        self.test_x_df = self.test_x_df.reshape(self.test_x_df.shape[0],
                                                self.sequence_len_minutes * self.number_of_features)

        # self.test_x_norm = self.test_x_norm.reshape(self.test_x_norm.shape[0],
        #                                         self.sequence_len_minutes * self.number_of_features)


        self.val_x_df = self.val_x_df.reshape(self.val_x_df.shape[0] * self.val_x_df.shape[1],
                                              self.sequence_len_minutes, self.number_of_features)
        self.val_x_df = self.val_x_df.reshape(self.val_x_df.shape[0],
                                              self.sequence_len_minutes * self.number_of_features)

        self.train_y_df = self.train_y_df.reshape(self.train_y_df.shape[0] * self.train_y_df.shape[1], 20)
        self.val_y_df = self.val_y_df.reshape(self.val_y_df.shape[0] * self.val_y_df.shape[1], 20)
        print('done')

    def save_df(self):
        name = 'mega_df'
        np.save(name + 'x', self.mega_df_x_1)
        np.save(name + 'y', self.mega_df_y_1)

        if self.cams == 2:
            np.save(name + 'x2', self.mega_df_x_2)
            np.save(name + 'y2', self.mega_df_y_2)


    def load_prev_mega_df(self):  # todo get data from df
        self.start = 6
        self.end = 20
        self.step = 1
        self.months = [7,8,9,10,11,12]
        # self.queries_per_day = int(((self.end - self.start) * 60 / self.step))  # amount of data in one day
        self.mega_df_x_1 = np.load('mega_dfx.npy')
        self.mega_df_y_1 = np.load('mega_dfy.npy')
        if self.cams == 2:
            self.mega_df_x_2 = np.load('mega_dfx2.npy')
            self.mega_df_y_2 = np.load('mega_dfy2.npy')

        print('loaded')


#
# data = DataFrameSequence(False, 20)
# data.build_ts_df(10,13,[9,10],45,2)
# data.split_data_set(9,27)

# data = DataFrameSequence(False, 20)
# data.build_ts_df(9,18,[7,8,9,10],1)
# # data.save_df()
# data.split_data_set(8,25)
# data.flatten_data_set_to_3d()
# data



# # data.normalize_data_sets()
# # data.load_prev_mega_df()
# ann = ann_model.ANN(data, 50, 50)
# ann.run_experiment()
# # lstm = lstm_model.LSTM_predictor(data, 400,100)
# # lstm.get_model()
# # lstm.run_experiment()
