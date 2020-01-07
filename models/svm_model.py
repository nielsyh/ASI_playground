import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from data import Data
from metrics import Metrics
import pickle


class SVM_predictor:

    def __init__(self, data):
        self.data = data

        self.x_train = self.data.train_df[:, 0: self.data.train_df.shape[1] - 1]
        self.y_train = self.data.train_df[:, -1]

        self.x_test = self.data.test_df[:, 0:self.data.test_df.shape[1] - 1]
        self.y_test = self.data.test_df[:, -1]

        self.model = 0

    def train(self):
        print('SVM: Training..')
        self.svclassifier = SVC(kernel='rbf', gamma='auto')
        self.model = self.svclassifier.fit(self.x_train, self.y_train)
        print('done..')

    def predict(self):
        print('SVM: Predicting..')
        y_pred = self.model.predict(self.x_test)
        rmse, mae, mape = Metrics.get_error(self.y_test, y_pred)
        return rmse, mae, mape

    def save(self, name):
        with open(name, 'wb') as file:
            pickle.dump(self.model, file)

    def load(self, name):
        with open(name, 'rb') as file:
            self.model = pickle.load(file)

# a = SVM_predictor()
# a.train()
# a.predict()
