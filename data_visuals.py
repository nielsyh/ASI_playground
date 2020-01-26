

import matplotlib.pyplot as plt
import matplotlib.style as style
from metrics import *

style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')



def plot_error(model_a, model_b, model_c, title, xl, yl, prediction_horizons, option):

    plt.plot(prediction_horizons, [i[option] for i in model_a], linestyle='-', label='SVM')
    plt.plot(prediction_horizons, [i[option] for i in model_b], linestyle='-', label='Log. res.')
    plt.plot(prediction_horizons, [i[option] for i in model_c], linestyle='-', label='Persistence')

    plt.legend()
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)

    plt.show()
    plt.savefig('plot_' + str(title) +'.png')
    plt.close()


def plot_time_avg(tick_times, times, values, values_label, lx, ly, title, values_2 = [], values_2_label = ''):

    plt.xticks(tick_times)
    plt.xticks(rotation=45)
    plt.plot(times, values, linestyle='-', label=values_label)

    if(len(values_2) > 1):
        plt.plot(times, values_2, linestyle='-', label = values_2_label)

    plt.legend()
    plt.title(title)
    plt.xlabel(lx)
    plt.ylabel(ly)

    plt.show()


def plot_2_models(tick_times, times, values_m1, values_m2, lx, ly, title):

    plt.xticks(tick_times)
    plt.xticks(rotation=45)
    plt.plot(times, values_m1 ,linestyle='-', label = 'Observed')
    plt.plot(times, values_m2, linestyle='-', label = 'Predicted')
    plt.legend()
    plt.title(title)
    plt.xlabel(lx)
    plt.ylabel(ly)

    plt.show()


def plot_freq(dict, title):
    ax = plt.axes()
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='x-small')
    plt.bar(dict.keys(), dict.values(), 0.75, color='b')

    plt.title(title)
    plt.xlabel('times')
    plt.ylabel('frequency')
    plt.tight_layout()

    plt.show()

def plot_error_per_month(errors, title, date_tuples):
    ax = plt.axes()
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='x-small')
    plt.plot(date_tuples, errors, linestyle='-', label='label')


    plt.legend()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='x-small')
    plt.show()
    plt.close()


def read_plot_file(file_name, model_name):
    total_rmse, total_mae, total_mape = [], [], []
    date_tuples = []
    with open(file_name) as fp:

        y_true = []
        y_pred = []
        last_day = 11.0  # first day is 11/9/2019


        for line in fp:
            l = line.split(',')
            day = float(l[1])
            month = float(l[0])
            if day != last_day:
                rmse, mae, mape = Metrics.get_error(y_true, y_pred)
                total_rmse.append(rmse)
                total_mae.append(mae)
                total_mape.append(mape)
                date_tuples.append(str((month, day)))
                y_true = []
                y_pred = []
                last_day = day

            y_true.append(float(l[5]))
            y_pred.append(float(l[6]))


    plot_error_per_month(total_rmse, 'TITLE', date_tuples)


read_plot_file('results/meteor_ghi_norm/SVM predictor_horizon_5_meteor.txt', 'SVM')