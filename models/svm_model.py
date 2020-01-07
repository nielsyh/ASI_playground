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
        self.model = 0

    def train(self):
        print('SVM: Training..')
        self.svclassifier = SVC(kernel='rbf', gamma='auto')
        self.model = self.svclassifier.fit(self.data.x_train, self.data.y_train)
        print('done..')

    def predict(self):
        print('SVM: Predicting..')
        y_pred = self.model.predict(self.data.x_test)
        rmse, mae, mape = Metrics.get_error(self.data.y_test, y_pred)
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
