import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from data import Data
from metrics import Metrics
from models.model_template import Predictor_template
import pickle


class Regression_predictor(Predictor_template):

    def __init__(self, data):
        self.data = data

        self.x_train = self.data.train_df[:, 0: self.data.train_df.shape[1] - 1]
        self.y_train = self.data.train_df[:, -1]

        self.x_test = self.data.test_df[:, 0:self.data.test_df.shape[1] - 1]
        self.y_test = self.data.test_df[:, -1]

    def train(self):
        print('REG: Training..')
        self.logisticRegr = LogisticRegression()
        self.logisticRegr.fit(self.x_train, self.y_train)
        print('done..')

    def predict(self):
        print('REG: Predicting..')
        y_pred = self.logisticRegr.predict(self.x_train)
        rmse, mae, mape = Metrics.get_error(self.y_test, y_pred)
        return rmse, mae, mape

    def save(self, name):
        with open(name, 'wb') as file:
            pickle.dump(self.model, file)

    def load(self, name):
        with open(name, 'rb') as file:
            self.model = pickle.load(file)

# a = Regression_predictor()
# a.train()
# a.predict()



