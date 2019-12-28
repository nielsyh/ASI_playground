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




