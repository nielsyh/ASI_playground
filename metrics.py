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
    def get_error(y_observed, y_predicted):
        print('RMSE')
        rmse = Metrics.rmse(y_observed, y_predicted)
        print(rmse)
        print('MAE')
        mae = Metrics.mae(y_observed, y_predicted)
        print(mae)
        print('MAPE')
        mape = Metrics.mape(y_observed, y_predicted)
        print(mape)
        return rmse, mae, mape

    @staticmethod
    def print_error(y_observed, y_predicted):
        print('RMSE')
        print(Metrics.rmse(y_observed, y_predicted))
        print('MAE')
        print(Metrics.mae(y_observed, y_predicted))
        print('MAPE')
        print(Metrics.mape(y_observed, y_predicted))

    @staticmethod
    def write_results(model, x_test, y_actual, y_pred, horizon):
        for idx, value in x_test:
            Metrics.write_to_file(model, x_test[1], x_test[2], x_test[3], x_test[4], horizon, y_actual[idx], y_pred[idx])

    @staticmethod
    def write_to_file(model, month, hour, day, minute, horizon, actual, predicted):
        f = open(str(model) + ".txt", "w+")
        f.write(str(month) + " " + str(day) + " " + str(minute) + " " + str(horizon) + " " + str(actual) + " " + str(predicted))
        f.close()

