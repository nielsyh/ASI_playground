import matplotlib.pyplot as plt
import matplotlib.style as style
import data.data_visuals

from data import data_helper
from metrics import *
from datetime import time, timedelta
import datetime
import matplotlib.pyplot
import data.data_helper
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
from matplotlib.ticker import FuncFormatter, MaxNLocator

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

def plot_time_avg_multi(tick_times, times, values, values_label, lx, ly, title):
    plt.xticks(tick_times)
    plt.xticks(rotation=45)

    for idx, val in enumerate(values):
        plt.plot(times[idx], val, linestyle='-', label=values_label[idx])

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


def plot_error_per_month(errors, names, title, yl, xl = 'Days'):
    ax = plt.axes()
    # plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='x-small')

    for idx, i in enumerate(errors):
        plt.plot(i, linestyle='--', label=names[idx])

    plt.legend()
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()
    plt.close()

def plot_prediction_per_day(predicts, names, title, yl, xl = 'Days'):
    ax = plt.axes()
    # plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='x-small')

    for idx, i in enumerate(predicts):
        plt.plot(i, linestyle='--', label=names[idx])

    plt.legend()
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()
    plt.close()

def days_plot():
    m = 10
    for i in list(range(1, 10)):
        predicted, actual, times = data.data_visuals.file_to_dates('results/Persistence_b_horizon_20.txt', m, i, 0)
        predicted2, actual2, times2 = data.data_visuals.file_to_dates('results/ANN_BETA_SEQUENCE_1CAM_30Minutes_.txt', m, i, 30)
        # predicted22, actual22, times22 = file_to_dates('results/ANN_BETA_SEQUENCE_2CAM_30Minutes_.txt', m, i, 30)

        predicted4, actual4, times4 = data.data_visuals.file_to_dates('results/ANN_BETA_SEQUENCE_1CAM_45Minutes_.txt', m, i, 45)
        # predicted42, actual42, times42 = file_to_dates('results/ANN_BETA_SEQUENCE_2CAM_45Minutes_.txt', m, i, 45)

        predicted5, actual5, times5 = data.data_visuals.file_to_dates('results/ANN_BETA_SEQUENCE_1CAM_60Minutes_.txt', m, i, 60)
        predicted52, actual52, times52 = data.data_visuals.file_to_dates('results/ANN_BETA_SEQUENCE_2CAM_60Minutes_.txt', m, i, 60)

        # predicted6, actual6, times6 = file_to_dates('results/ANN_BETA_SEQUENCE_IMG1CAM_60Minutes_.txt', m, i, 0)


        names = ['Truth', 'Persistence', 'ANN 1 30', 'ANN 2 30', 'ANN 1 45','ANN 2 45', 'ANN 1 60', 'ANN 2 60', 'ANN 60 img']
        data.data_visuals.plot_with_times([actual2, predicted, predicted2, predicted4, predicted5, predicted52],
                        [times2, times, times2, times4, times5, times52 ],
                        names, 'GHI forecast ' + str(i) + '/' + str(m), 'GHI in W/m2', xl='Time of day')