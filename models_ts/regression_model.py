import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from metrics import Metrics
from models.model_template import Predictor_template
import pickle
import calendar
import sys

class Regression_predictor():
    day_month_to_predict = []

    def __init__(self, data, name):
        self.data = data
        self.model = 0
        self.name = name

    def run_experiment(self):
        self.day_month_to_predict = []

        for m in self.data.months:
            last_day = calendar.monthrange(2019, m)[1]
            if m < 9:
                continue
            elif m == 9:
                days = list(range(11, last_day + 1)) #  Predict from 11 september
            else:
                days = list(range(1, last_day + 1))

            for d in days:
                self.day_month_to_predict.append((m, d))

        prem = [(10,5), (10,6), (10,7), (10,8), (10,20)]
        self.day_month_to_predict = prem

        for exp in self.day_month_to_predict:
            sys.stdout.write('REG: ' + str(exp) + ', horizon: ' + str(self.data.pred_horizon))
            self.data.split_data_set(exp[0], exp[1])
            self.data.flatten_data_set()
            self.train()
            y_pred, rmse, mae, mape = self.predict()

            Metrics.write_results_SVR(str(self.name), self.data.test_x_df.reshape(
                (self.data.test_x_df.shape[0],
                 self.data.sequence_len_minutes,
                 self.data.number_of_features)),
                                      self.data.test_y_df, y_pred,
                                      self.data.pred_horizon)


    def train(self):
        print('REG: Training..')
        self.logisticRegr = LogisticRegression(max_iter=100)
        self.logisticRegr.fit(self.data.train_x_df, self.data.train_y_df)
        print('done..')

    def predict(self):
        print('REG: Predicting..')
        y_pred = self.logisticRegr.predict(self.data.test_x_df)
        rmse, mae, mape = Metrics.get_error(self.data.test_y_df, y_pred)
        sys.stdout.write(str(rmse))
        return y_pred, rmse, mae, mape


# a = Regression_predictor()
# a.train()
# a.predict()



