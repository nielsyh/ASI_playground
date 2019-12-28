import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from data import Data
from metrics import Metrics


class SVM_predictor:

    def __init__(self, data):
        self.data = data

        self.x_train = self.data.train_df[:, 0: self.data.train_df.shape[1] - 1]
        self.y_train = self.data.train_df[:, -1]

        self.x_test = self.data.test_df[:, 0:self.data.test_df.shape[1] - 1]
        self.y_test = self.data.test_df[:, -1]

    def train(self):
        print('SVM: Training..')
        self.svclassifier = SVC(kernel='rbf', gamma='auto')
        self.svclassifier.fit(self.x_train, self.y_train)
        print('done..')

    def predict(self):
        print('SVM: Predicting..')
        y_pred = self.svclassifier.predict(self.x_train)
        Metrics.print_error(self.y_test, y_pred)


# a = SVM_predictor()
# a.train()
# a.predict()