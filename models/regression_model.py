import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from metrics import Metrics
from models.model_template import Predictor_template
import pickle
import calendar


class Regression_predictor(Predictor_template):
    day_month_to_predict = []

    def __init__(self, data):
        self.data = data
        self.model = 0

    def run_experiment(self):
        self.day_month_to_predict = []

        for m in self.data.months:
            last_day = calendar.monthrange(2019, m)[1]
            if m < 9:
                continue
            elif m == 9:
                days = list(range(11, last_day + 1))  # Predict from 11 september
            else:
                days = list(range(1, last_day + 1))

            for d in days:
                self.day_month_to_predict.append((m, d))

        for exp in self.day_month_to_predict:
            print('Regression: ' + str(exp))
            self.data.split_data_set(exp[0], exp[1])
            self.data.flatten_data_set()
            self.data.normalize_data_sets()

            self.train()
            y_pred, rmse, mae, mape = self.predict()

            name = 0
            if self.data.debug:
                name = name + '_debug'
            if self.data.images:
                name = name + '_images'
            if self.data.meteor_data:
                name = name + '_meteor'

            Metrics.write_results('Regression predictor' + str(name), self.data.x_test, self.data.y_test, y_pred, self.data.pred_horizon)


    def train(self):
        print('REG: Training..')
        self.logisticRegr = LogisticRegression(max_iter=1000)
        self.logisticRegr.fit(self.data.x_train, self.data.y_train)
        print('done..')

    def predict(self):
        print('REG: Predicting..')
        y_pred = self.logisticRegr.predict(self.data.x_test)
        rmse, mae, mape = Metrics.get_error(self.data.y_test, y_pred)
        print(rmse)
        return y_pred, rmse, mae, mape

    def save(self, name):
        with open(name, 'wb') as file:
            pickle.dump(self.model, file)

    def load(self, name):
        with open(name, 'rb') as file:
            self.model = pickle.load(file)

# a = Regression_predictor()
# a.train()
# a.predict()



