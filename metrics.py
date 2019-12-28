from sklearn.metrics import *
# from sklearn.utils import check_arrays
from math import sqrt
import numpy as np

class Metrics:

    def __init__(self):
        pass

    @staticmethod
    def rmse(y_observed, y_predicted):
        return sqrt(mean_squared_error(y_observed, y_predicted))

    @staticmethod
    def mae(y_observed, y_predicted):
        return mean_absolute_error(y_observed, y_predicted)

    @staticmethod
    def mape(y_observed, y_predicted):
        y_observed, y_pred = np.array(y_observed), np.array(y_predicted)
        return np.mean(np.abs((y_observed - y_predicted) / (y_observed +1) )) * 100

    @staticmethod
    def print_error(y_observed, y_predicted):
        print('RMSE')
        print(Metrics.rmse(y_observed, y_predicted))
        print('MAE')
        print(Metrics.mae(y_observed, y_predicted))
        print('MAPE')
        print(Metrics.mape(y_observed, y_predicted))

