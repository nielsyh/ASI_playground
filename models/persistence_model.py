import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from data import *
from metrics import Metrics
import pickle
import calendar
import pandas as pd


class Persistence_predictor_a:  # predict value as day before

    def __init__(self, data):
        self.data = data
        self.model = 0


    def train(self):
        print('No training involved..')


    def predict(self):
        print('Persistence a: Predicting..')
        day_data = 0
        last_day = 0
        y_pred = []
        for x in self.data.x_test:
            month, day, hour, minute, second = int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5])
            year = int(month_to_year(month))
            minute += self.data.pred_horizon

            if month > 12 or month < 1:
                print('cao ni ma')
                continue

            current = pd.Timestamp(year=year, month=month, day=day, hour=hour, second=second)
            horizon = current + pd.Timedelta(minutes=self.data.pred_horizon)
            persistence = horizon - pd.Timedelta(days=1)

            if(last_day != persistence.day):
                day_data = self.data.get_df_csv_day_RP(persistence.month, persistence.day, self.data.start, (self.data.end + self.data.pred_horizon + 1), self.data.step)
                last_day = persistence.day

            #lookup
            rows = day_data[np.where(day_data[:, 3] == persistence.hour)]
            rows = rows[np.where(rows[:, 4] == persistence.minute)]
            y_pred.append(rows[0][8])

        rmse, mae, mape = Metrics.get_error(self.data.y_test, y_pred)
        return rmse, mae, mape

class Persistence_predictor_b:  # predict value as minute before

    def __init__(self, data):
        self.data = data
        self.model = 0


    def train(self):
        print('No training involved..')

    def predict(self):
        print('Persistence b: Predicting..')
        y_pred = []
        for x in self.data.x_test:
            y_pred.append(int(x[8]))

        rmse, mae, mape = Metrics.get_error(self.data.y_test, y_pred)
        return rmse, mae, mape



