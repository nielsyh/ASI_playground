import ftplib
from builtins import enumerate

from data_visuals import *
from metrics import Metrics
from pvlib_playground import PvLibPlayground
from features import get_image_by_date_time, int_to_str, extract_features, show_img
import calendar
from datetime import date
from tqdm import tqdm
from sklearn.preprocessing import *
from data import month_to_year, PvLibPlayground, int_to_str, process_csv, get_df_csv_day_RP


class DataFrameNormal:
    mega_df = []
    extra_df = []
    train_df = []
    test_df = []
    months = []
    size_of_row = 10  # row default size 9 + 1 for label
    size_meteor_data = 18  # amount of columns from meteor data
    img_idx = 9  # index of column were image data starts
    queries_per_day = 0
    pred_horizon = 0

    month_split = 0
    day_split = 0

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
        self.x_val = 0
        self.y_val = 0

        if self.meteor_data:  # adjusting df row length according to amount of data
            self.size_of_row += self.size_meteor_data
        if self.images:
            self.size_of_row += 4  # change if it gets bigger
            if self.meteor_data:
                self.img_idx += self.size_meteor_data

        print('Building Df with meteor data: ' + str(self.meteor_data) + ', image data: ' + str(
            self.images) + ', debug: ' + str(self.debug) + '..')
        print('size of a row: ' + str(self.size_of_row))


    def flatten_data_set_CNN(self):
        print('Flattening..')

        self.train_df = self.train_df.reshape((self.train_df.shape[0] * self.train_df.shape[1], -1))

        self.x_train = self.train_df[:, 3: self.train_df.shape[1]]
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 400, 400, 3))  # reshaping for tf
        self.y_train = self.train_df[:, 0]

        self.x_test = self.test_df[:, 3:self.test_df.shape[1]]
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 400, 400, 3))  # reshaping for tf
        self.y_test = self.test_df[:, 0]

    def flatten_data_set(self):  # this is needed to use it as input for models
        print('Flattening..')

        self.train_df = self.train_df.reshape((self.train_df.shape[0] * self.train_df.shape[1], -1))
        # self.test_df = self.test_df.reshape((self.test_df.shape[0] * self.test_df.shape[1], -1))
        self.val_df = self.val_df.reshape((self.val_df.shape[0] * self.val_df.shape[1], -1))

        self.x_train = self.train_df[:, 0: self.train_df.shape[1] - 1]
        self.y_train = self.train_df[:, -1]

        self.x_test = self.test_df[:, 0:self.test_df.shape[1] - 1]
        self.y_test = self.test_df[:, -1]

        self.x_val = self.val_df[:, 0:self.val_df.shape[1] - 1]
        self.y_val = self.val_df[:, -1]

        self.x_train = np.concatenate((self.x_train, self.x_val))
        self.y_train = np.concatenate((self.y_train, self.y_val))

        print('done')

    def normalize_data_sets(self, colums_to_normalize = [6,7,8], metoer_to_normalize= [9,10,13,17,18,19,22,26]):
        # 0 year, 1 month, 2 day, 3 hour, 4 minute, 5 seconds, 6 temp, 7 humidity,
        # 8 current ghi, 9 future ghi (y) , 10 csi, 11 azimuth, 12 zenith, 13 intensity,
        # 14 cloudpixel, 15 harris, 16 edges

        colums_to_normalize = colums_to_normalize
        if (self.meteor_data):
            colums_to_normalize.extend(metoer_to_normalize)
        if (self.images):
            colums_to_normalize.extend([27,28,29,30])
        print('normalzing for: ' + str(colums_to_normalize))

        self.x_train[:, colums_to_normalize] = normalize(self.x_train[:, colums_to_normalize], axis=0, norm='l2')
        self.x_test[:, colums_to_normalize] = normalize(self.x_test[:, colums_to_normalize], axis=0, norm='l2')
        self.x_val[:, colums_to_normalize] = normalize(self.x_val[:, colums_to_normalize], axis=0, norm='l2')
        # print('done')

    def drop_columns(self, colums_to_drop):
        self.x_train = np.delete(self.x_train,colums_to_drop,1)

    def build_df_for_cnn(self, start, end, step, months):
        self.start = start
        self.end = end
        self.step = step
        self.months = months
        self.queries_per_day = int(((end - start) * 60 / step))  # amount of data in one day

        days = 0
        day_index = -1

        print('Processing months for CNN DF: ' + str(months))

        for m in months:
            if (self.debug):  # debug
                days += 3
            elif m == 7:
                days += 6
            else:
                year = int(month_to_year(m))
                days += calendar.monthrange(year, m)[1]

        # image res + label + month + day
        size_of_row = 3 + (400 * 400 * 3)
        self.mega_df = np.zeros((days, self.queries_per_day, size_of_row), dtype=np.uint16)

        print(months)
        for m in tqdm(months, total=len(months), unit='Month progress'):
            if self.debug:  # debug
                days = [26, 27, 28]
            elif m == 7:  # different for july..
                days = [26, 27, 28, 29, 30, 31]
            else:
                days = list(range(1, calendar.monthrange(2019, m)[1] + 1))  # create an array with days for that month


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
                    try:
                        self.mega_df[day_index][idx][3:size_of_row] = img.ravel()
                    except:
                        print('Broken image: ')
                        print(month, day, minute, seconds)
                        continue

    def build_df(self, start, end, step, months):

        self.start = start
        self.end = end
        self.step = step
        self.months = months
        self.queries_per_day = int(((end - start) * 60 / step))  # amount of data in one day

        days = 0
        day_index = -1

        print('Processing months: ' + str(months))

        for m in months:
            if self.debug:  # debug
                days += 3
            elif m == 7:
                days += 6
            else:
                year = int(month_to_year(m))
                days += calendar.monthrange(year, m)[1]

        self.mega_df = np.zeros((days, self.queries_per_day, self.size_of_row), dtype=np.float)

        for m in tqdm(months, total=len(months), unit='Month progress'):
            if self.debug:  # debug
                days = [26, 27, 28]
            elif m == 7:  # different for july..
                # days = [24, 25, 26, 27, 28, 29, 30, 31]
                days = [26, 27, 28, 29, 30, 31]
            else:
                days = list(range(1, calendar.monthrange(2019, m)[1] + 1))  # create an array with days for that month

            for d in tqdm(days, total=len(days), unit='Days progress'):
                # todo add sunrise/sundown for start end hour? half hour?
                day_data = get_df_csv_day_RP(m, d, start, end, step).astype(int)
                if self.meteor_data:
                    csi, azimuth, zenith, sun_earth_dis, ephemeris = PvLibPlayground.get_meteor_data(m,
                                                                                                         d,
                                                                                                         PvLibPlayground.get_times(
                                                                                                             2019,
                                                                                                             m,  # month
                                                                                                             d,  # day
                                                                                                             start,
                                                                                                             # start time
                                                                                                             end))  # end time
                day_index += 1
                for idx, data in enumerate(day_data):
                    data[6] = Metrics.celsius_to_kelvin(data[6])  # convert to kelvin
                    self.mega_df[day_index][idx][0:9] = data  # adding data from csv

                    if self.meteor_data:
                        self.mega_df[day_index][idx][9] = csi[idx]
                        self.mega_df[day_index][idx][10] = ( self.mega_df[day_index][idx][9] - self.mega_df[day_index][idx][8])  # csi - ghi
                        self.mega_df[day_index][idx][11] = azimuth[idx]
                        self.mega_df[day_index][idx][12] = zenith[idx]
                        self.mega_df[day_index][idx][13] = sun_earth_dis[idx]
                        self.mega_df[day_index][idx][14:18] = ephemeris[idx]

                    if self.images:
                        year, month, day, hour, minute, seconds = int(data[0]), int(data[1]), int(data[2]), int(
                            data[3]), int(data[4]), int(data[5])
                        try:
                            img = get_image_by_date_time(year, month, day, hour, minute, seconds)  # get image
                            self.mega_df[day_index][idx][self.img_idx:(self.size_of_row - 1)] = extract_features(
                                img)  # get features from image
                            del img  # delete img for mem
                        except:
                            print("IMAGE NOT FOUND")
                            print(month, day, minute, seconds)
                            self.mega_df[day_index][idx][self.img_idx:(self.size_of_row - 1)] = np.zeros(4)

        # YEAR, MONTH, DAY, HOURS, MINUTES, SECONDS, TEMP, IRRADIANCE, IMAGE
        print('Building done')

    def set_pred_horizon_virtual(self,
                                 prediction_horizon):  # This function just changes the variable but doesnt label accoringly. This is needed for persistence model.
        self.pred_horizon = prediction_horizon  # set prediction horizon
        self.label_df()

    def set_prediction_horizon(self, prediction_horizon):  # only for model 1
        print('Setting prediction horizon to: ' + str(prediction_horizon) + '...')
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

            if (self.debug):  # debug
                days = [25, 26, 27]

            for d in days:
                # todo add sunrise/sundown for start end
                extra = self.get_df_csv_day_RP(m, d, self.end, self.end + 1, self.step).astype(int)
                day_index += 1

                for idx, data in enumerate(extra):  # getting label data for predictions
                    if idx < self.pred_horizon:
                        self.extra_df[day_index][idx][0] = data[8]
                    else:
                        continue

        self.label_df()
        print('done')

    def label_df(self):
        for idx_day, day in enumerate(self.mega_df):
            tmp_cnt = 0
            m = int(day[0][1])
            d = int(day[0][2])
            start = self.start
            end = self.end

            csi, azimuth, zenith = [], [], []

            if m == 0:
                print(idx_day, day)

            if self.meteor_data:
                csi, azimuth, zenith, sun_earth_dis, ephemeris = PvLibPlayground.get_meteor_data(m,
                                                                                                 d,
                                                                                                 PvLibPlayground.get_times(
                                                                                                     2019,
                                                                                                     m,  # month
                                                                                                     d,  # day
                                                                                                     start,
                                                                                                     # start time
                                                                                                     end,
                                                                                                     offset=int(
                                                                                                         self.pred_horizon)))  # end time

            for idx_timeslot, time_slot in enumerate(day):
                if self.meteor_data:
                    self.mega_df[idx_day][idx_timeslot][18] = csi[idx_timeslot]
                    self.mega_df[idx_day][idx_timeslot][19] = (self.mega_df[idx_day][idx_timeslot][9] - self.mega_df[idx_day][idx_timeslot][8])  #csi - ghi
                    self.mega_df[idx_day][idx_timeslot][20] = azimuth[idx_timeslot]
                    self.mega_df[idx_day][idx_timeslot][21] = zenith[idx_timeslot]
                    self.mega_df[idx_day][idx_timeslot][22] = sun_earth_dis[idx_timeslot]
                    self.mega_df[idx_day][idx_timeslot][23:27] = ephemeris[idx_timeslot]

                if (self.queries_per_day - self.pred_horizon) > idx_timeslot:
                    self.mega_df[idx_day][idx_timeslot][self.size_of_row - 1] = \
                    self.mega_df[idx_day][idx_timeslot + self.pred_horizon][8]
                    tmp_cnt += 1
                else:
                    self.mega_df[idx_day][idx_timeslot][self.size_of_row - 1] = \
                    self.extra_df[idx_day][idx_timeslot - tmp_cnt][0]

    def save_df(self):
        name = str(self.size_of_row) + '_' + str(self.images) + '_' + str(self.meteor_data)
        np.save('mega_df_' + name + '_' , self.mega_df)

    def save_df_cnn(self):
        np.save('mega_df_CNN', self.mega_df)

    def load_prev_mega_df(self, filename):  # todo get data from df
        self.start = 10
        self.end = 17
        self.step = 1
        self.months = [7,8,9,10,11,12]
        self.queries_per_day = int(((self.end - self.start) * 60 / self.step))  # amount of data in one day
        self.mega_df = np.load(filename)
        print('loaded')

## build df for model 1
data = Data(meteor_data=True, images=False, debug=True)
data.build_df(10, 17, 1, months=[7])
# data.save_df()
#
# data.set_prediction_horizon(5)
# data.split_data_set(7, 27)
# data.flatten_data_set()
# data.normalize_data_sets()
# np.set_printoptions(precision=3)
# print(data.mega_df)


## downloading and processing stuff
# d.download_data(1, True)F
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
