from sklearn.metrics import *
# from sklearn.utils import check_arrays
from math import sqrt
import numpy as np

class Metrics:

    def __init__(self):
        pass

    @staticmethod
    def celsius_to_kelvin(c):
        return c + 273.15

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
        # print('RMSE')
        rmse = Metrics.rmse(y_observed, y_predicted)
        # print(rmse)
        # print('MAE')
        mae = Metrics.mae(y_observed, y_predicted)
        # print(mae)
        # print('MAPE')
        mape = Metrics.mape(y_observed, y_predicted)
        # print(mape)
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
    def write_results_NN(model, x_test, y_actual, y_pred, horizon):
        y_actual = y_actual.ravel()
        y_pred = y_pred.ravel()
        f = open(str(model) + ".txt", "a")
        for idx, value in enumerate(x_test):
            Metrics.write_to_file(f, value[-1,1], value[-1,2], value[-1,3], value[-1,4], horizon, y_actual[idx], y_pred[idx])
        f.close()

    @staticmethod
    def write_results_multi(model, x_test, y_actual, y_pred):
        f = open(str(model) + ".txt", "a")
        for idx, value in enumerate(x_test):
            for p in list(range(0,20)):
                Metrics.write_to_file(f, value[-1,1], value[-1,2], value[-1,3], value[-1,4], p+1, y_actual[idx][p], y_pred[idx][p])
        f.close()


    @staticmethod
    def write_results_SVR(model, x_test, y_actual, y_pred, horizon):
        y_actual = y_actual.ravel()
        # y_pred = y_pred.ravel()
        f = open(str(model) + ".txt", "a")
        for idx, value in enumerate(x_test):
            Metrics.write_to_file(f, value[-1,1], value[-1,2], value[-1,3], value[-1,4], horizon, y_actual[idx], y_pred[idx])
        f.close()

    @staticmethod
    def write_results(model, x_test, y_actual, y_pred, horizon):
        f = open(str(model) + ".txt", "a")
        for idx, value in enumerate(x_test):
            Metrics.write_to_file(f, value[1], value[2], value[3], value[4], horizon, y_actual[idx], y_pred[idx])
        f.close()

    @staticmethod
    def write_results_sequence(model, x_test, y_actual, y_pred, horizon):
        f = open(str(model) + ".txt", "a")
        for idx, value in enumerate(x_test):
            Metrics.write_to_file(f, value[1], value[2], value[3], value[4], horizon, y_actual[idx], y_pred[idx])
        f.close()

    @staticmethod
    def write_to_file(f, month, day, hour, minute, horizon, actual, predicted):
        f.write(str(month) + "," + str(day) + "," + str(hour) + "," + str(minute) + "," + str(horizon) + "," + str(actual) + "," + str(predicted))
        f.write('\n')

    @staticmethod
    # stolen from https://github.com/johntwk/Diebold-Mariano-Test
    def dm_test(actual_lst, pred1_lst, pred2_lst, h=1, crit="MSE", power=2):
        # Routine for checking errors
        def error_check():
            rt = 0
            msg = ""
            # Check if h is an integer
            if (not isinstance(h, int)):
                rt = -1
                msg = "The type of the number of steps ahead (h) is not an integer."
                return (rt, msg)
            # Check the range of h
            if (h < 1):
                rt = -1
                msg = "The number of steps ahead (h) is not large enough."
                return (rt, msg)
            len_act = len(actual_lst)
            len_p1 = len(pred1_lst)
            len_p2 = len(pred2_lst)
            # Check if lengths of actual values and predicted values are equal
            if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
                rt = -1
                msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
                return (rt, msg)
            # Check range of h
            if (h >= len_act):
                rt = -1
                msg = "The number of steps ahead is too large."
                return (rt, msg)
            # Check if criterion supported
            if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
                rt = -1
                msg = "The criterion is not supported."
                return (rt, msg)
                # Check if every value of the input lists are numerical values
            from re import compile as re_compile
            comp = re_compile("^\d+?\.\d+?$")

            def compiled_regex(s):
                """ Returns True is string is a number. """
                if comp.match(s) is None:
                    return s.isdigit()
                return True

            for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
                is_actual_ok = compiled_regex(str(abs(actual)))
                is_pred1_ok = compiled_regex(str(abs(pred1)))
                is_pred2_ok = compiled_regex(str(abs(pred2)))
                if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):
                    msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                    rt = -1
                    return (rt, msg)
            return (rt, msg)

        # Error check
        error_code = error_check()
        # Raise error if cannot pass error check
        if (error_code[0] == -1):
            raise SyntaxError(error_code[1])
            return
        # Import libraries
        from scipy.stats import t
        import collections
        import pandas as pd
        import numpy as np

        # Initialise lists
        e1_lst = []
        e2_lst = []
        d_lst = []

        # convert every value of the lists into real values
        actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
        pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
        pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()

        # Length of lists (as real numbers)
        T = float(len(actual_lst))

        # construct d according to crit
        if (crit == "MSE"):
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                e1_lst.append((actual - p1) ** 2)
                e2_lst.append((actual - p2) ** 2)
            for e1, e2 in zip(e1_lst, e2_lst):
                d_lst.append(e1 - e2)
        elif (crit == "MAD"):
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                e1_lst.append(abs(actual - p1))
                e2_lst.append(abs(actual - p2))
            for e1, e2 in zip(e1_lst, e2_lst):
                d_lst.append(e1 - e2)
        elif (crit == "MAPE"):
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                e1_lst.append(abs((actual - p1) / actual))
                e2_lst.append(abs((actual - p2) / actual))
            for e1, e2 in zip(e1_lst, e2_lst):
                d_lst.append(e1 - e2)
        elif (crit == "poly"):
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                e1_lst.append(((actual - p1)) ** (power))
                e2_lst.append(((actual - p2)) ** (power))
            for e1, e2 in zip(e1_lst, e2_lst):
                d_lst.append(e1 - e2)

                # Mean of d
        mean_d = pd.Series(d_lst).mean()

        # Find autocovariance and construct DM test statistics
        def autocovariance(Xi, N, k, Xs):
            autoCov = 0
            T = float(N)
            for i in np.arange(0, N - k):
                autoCov += ((Xi[i + k]) - Xs) * (Xi[i] - Xs)
            return (1 / (T)) * autoCov

        gamma = []
        for lag in range(0, h):
            gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))  # 0, 1, 2
        V_d = (gamma[0] + 2 * sum(gamma[1:])) / T
        DM_stat = V_d ** (-0.5) * mean_d
        harvey_adj = ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** (0.5)
        DM_stat = harvey_adj * DM_stat
        # Find p-value
        p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)

        # Construct named tuple for return
        dm_return = collections.namedtuple('dm_return', 'DM p_value')

        rt = dm_return(DM=DM_stat, p_value=p_value)

        return rt
