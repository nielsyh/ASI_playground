import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from data import Data
from metrics import Metrics

class SVM_predictor:

    def __init__(self, pred_horzion=10, meteor_data=False):
        self.data = Data(pred_horzion=pred_horzion, meteor_data=meteor_data)

        self.data.build_df(7, 19, 1, months=[9])
        self.data.label_df()
        self.data.split_data_set()
        self.data.flatten_data_set()

        self.x_train = self.data.train_df[:, 0: self.data.train_df.shape[1] - 1]
        self.y_train = self.data.train_df[:, -1]

        self.x_test = self.data.test_df[:, 0:self.data.test_df.shape[1] - 1]
        self.y_test = self.data.test_df[:, -1]


    def train_svm(self):
        self.svclassifier = SVC(kernel='rbf', gamma='auto')
        self.svclassifier.fit(self.x_train, self.y_train)

    def predict_svm(self):
        y_pred = self.svclassifier.predict(self.x_train)
        self.print_error(self.y_test, y_pred)

    def print_error(self, y_observed, y_predicted):
        print('RMSE')
        print(Metrics.rmse(y_observed, y_predicted))
        print('MAE')
        print(Metrics.mae(y_observed, y_predicted))
        print('MAPE')
        print(Metrics.mape(y_observed, y_predicted))

a = SVM_predictor()
a.train_svm()
a.predict_svm()



