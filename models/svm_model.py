import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from data import Data
from metrics import Metrics
import pickle
import calendar


class SVM_predictor:
    day_month_to_predict = []

    def __init__(self, data, model_name):
        self.data = data
        self.model = 0
        self.model_name = model_name

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

        for exp in self.day_month_to_predict:
            print('SVM: ' + str(exp) + ', horizon: ' + str(self.data.pred_horizon))
            self.data.split_data_set(exp[0], exp[1])
            self.data.flatten_data_set()
            self.data.normalize_data_sets()

            self.train()
            y_pred, rmse, mae, mape = self.predict()


            name = str(self.model_name)
            name = name + '_horizon_' + str(self.data.pred_horizon)
            if self.data.debug:
                name = name + '_debug'
            if self.data.images:
                name = name + '_images'
            if self.data.meteor_data:
                name = name + '_meteor'

            Metrics.write_results(str(name), self.data.x_test, self.data.y_test, y_pred, self.data.pred_horizon)

    def train(self):
        # print('SVM: Training..')
        self.svclassifier = SVC(kernel='rbf', gamma='auto')
        self.model = self.svclassifier.fit(self.data.x_train, self.data.y_train)
        # print('done..')

    def predict(self):
        # print('SVM: Predicting..')
        y_pred = self.model.predict(self.data.x_test)
        rmse, mae, mape = Metrics.get_error(self.data.y_test, y_pred)
        print(rmse)
        return y_pred, rmse, mae, mape

    def save(self, name):
        with open(name, 'wb') as file:
            pickle.dump(self.model, file)

    def load(self, name):
        with open(name, 'rb') as file:
            self.model = pickle.load(file)


