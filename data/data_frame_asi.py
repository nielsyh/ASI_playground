import numpy as np
from data.data_helper import month_to_year, PvLibPlayground, int_to_str, process_csv, get_df_csv_day_RP
import calendar
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import normalize
from features import get_features_by_day
from optical_flow import generate_img_for_cnn

class DataFrameIMG:


    mega_df_x = None
    mega_df_y = None
    mega_df_times = None

    train_x_df = None
    test_x_df = None
    val_x_df = None

    train_y_df = None
    test_y_df = None
    val_y_df = None

    def __init__(self, debug, pred_horizon ):
        self.debug = debug
        self.pred_horizon = pred_horizon


    def build_img_df(self, start, end, month, day):
        self.start = start
        self.end = end
        self.month = month
        self.day = day


        print('BUILDING IMG DF: ')
        print('start: ' + str(start) + ' end: ' + str(end) + ' months: ' + str(month))
        time_steps = int((end - start) * 60)
        print('timesteps: ', time_steps)

        self.mega_df_x = np.zeros((time_steps, 400, 400, 3), dtype=np.float)
        self.mega_df_y = np.zeros((time_steps), dtype=np.float)
        self.mega_df_times = np.zeros((time_steps, 5))

        day_data = get_df_csv_day_RP(self.month, self.day, start, end + 1, 1).astype(int)

        for i in tqdm(range(0, time_steps), total=time_steps, unit='Timesteps progress'):
            year, month, day, hour, minute, ghi = day_data[i][0], day_data[i][1], day_data[i][2], day_data[i][3], day_data[i][4], day_data[i+self.pred_horizon][8]
            img = generate_img_for_cnn(month, day, hour, minute, 0, self.pred_horizon)
            self.mega_df_x[i] = img
            self.mega_df_y[i] = ghi
            self.mega_df_times[i] = [year, month, day, hour, minute]


    def split_data_set(self, m, d):
        pass

    def flatten_data_set(self):  # this is needed to use it as input for models
        pass


    def normalize_mega_df(self, ctn = [6,7,8], metoer_to_normalize= [9,12,16]):
        pass


    def save_df(self):
        pass


    def load_prev_mega_df(self):  # todo get data from df
        pass

# t = DataFrameIMG(False, 20)
# t.build_img_df(10,12,9,1)