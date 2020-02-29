import numpy as np
import matplotlib.pyplot as plt
from metrics import Metrics
from sklearn.ensemble import RandomForestRegressor
import pickle
import calendar
import sys

class RF_predictor():
    day_month_to_predict = []

    def __init__(self, data, name):
        self.data = data
        self.model = 0
        self.name = name

    def set_days(self, days):
        self.day_month_to_predict = days

    def run_experiment(self):
        for exp in self.day_month_to_predict:
            sys.stdout.write('RF: ' + str(exp) + ', horizon: ' + str(self.data.pred_horizon))
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
        print('RF: Training..')
        self.model = RandomForestRegressor(n_estimators = 200,max_depth=100, min_samples_leaf=1, random_state = 0, n_jobs=-1)
        self.model.fit(self.data.train_x_df, self.data.train_y_df)
        print('done..')

    def predict(self):
        print('RF: Predicting..')
        y_pred = self.model.predict(self.data.test_x_df)
        rmse, mae, mape = Metrics.get_error(self.data.test_y_df, y_pred)
        sys.stdout.write(str(rmse))
        return y_pred, rmse, mae, mape


# a = Regression_predictor()
# a.train()
# a.predict()



