import matplotlib.pyplot as plt
import matplotlib.style as style
from metrics import *
from datetime import time, timedelta
import datetime
import matplotlib.pyplot

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

def plot_with_times(predicts, times, names, title, yl, xl = 'Days'):
    ax = plt.axes()
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    for idx, i in enumerate(predicts):
        if idx == 0:
            plt.plot_date(times[idx], i, linestyle='-', marker='None', label=names[idx])
        else:
            plt.plot_date(times[idx], i, linestyle='dotted',marker='None', label=names[idx])

    plt.legend()
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()
    plt.close()

def plot_with_months(predicts, times, names, title, yl, xl='Months'):
    ax = plt.axes()
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    for idx, i in enumerate(predicts):
        if idx == 0:
            plt.plot_date(times[idx], i, linestyle='-', marker='None', label=names[idx])
        else:
            plt.plot_date(times[idx], i, linestyle='dotted', marker='None', label=names[idx])

    plt.legend()
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()
    plt.close()


def file_to_dates(file, month, day, offset):
    predicted, actual = [],[]
    times = []
    with open(file) as fp:
        for line in fp:
            l = line.split(',')
            if float(l[0]) == month and float(l[1]) == day:
                month = int(float(l[0]))
                day = int(float(l[1]))
                hour = int(float(l[2]))
                minute = int(float(l[3]))

                if l[5][0] == '[':
                    true = float(l[5][1:-2])
                else:
                    true = float(l[5])
                if l[6][0] == '[':
                    pred = float(l[6][1:-2])
                else:
                    pred = float(l[6])

                if offset > 0:
                    a = time(hour=hour, minute=minute, second=0)
                    b = (datetime.datetime.combine(datetime.date(1, 1, 1), a) + datetime.timedelta(minutes=offset)).time()
                    a = datetime.datetime(year=2019, month=int(month), day=int(day), hour=b.hour, minute=b.minute)
                else:
                    a = datetime.datetime(year=2019, month=int(month), day=int(day), hour=int(hour), minute=int(minute))

                a = matplotlib.dates.date2num(a)

                predicted.append(pred)
                actual.append(true)
                times.append(a)
    return predicted, actual, times

def file_to_months(file, offset):
    predicted, actual = [],[]
    times = []
    with open(file) as fp:
        for line in fp:
            l = line.split(',')

            month = int(float(l[0]))
            day = int(float(l[1]))
            hour = int(float(l[2]))
            minute = int(float(l[3]))

            if l[5][0] == '[':
                true = float(l[5][1:-2])
            else:
                true = float(l[5])
            if l[6][0] == '[':
                pred = float(l[6][1:-2])
            else:
                pred = float(l[6])

            if offset > 0:
                a = time(hour=hour, minute=minute, second=0)
                b = (datetime.datetime.combine(datetime.date(1, 1, 1), a) + datetime.timedelta(minutes=offset)).time()
                a = datetime.datetime(year=2019, month=int(month), day=int(day), hour=b.hour, minute=b.minute)
            else:
                a = datetime.datetime(year=2019, month=int(month), day=int(day), hour=int(hour), minute=int(minute))

            a = matplotlib.dates.date2num(a)
            predicted.append(pred)
            actual.append(true)
            times.append(a)
    return predicted, actual, times




def days_plot():
    m = 10
    for i in list(range(1, 10)):
        predicted, actual, times = file_to_dates('Persistence_b_horizon_20.txt', m, i, 0)
        predicted2, actual2, times2 = file_to_dates('ANN_BETA_SEQUENCE_1CAM_30Minutes_.txt', m, i, 30)
        predicted22, actual22, times22 = file_to_dates('ANN_BETA_SEQUENCE_2CAM_30Minutes_.txt', m, i, 30)

        predicted4, actual4, times4 = file_to_dates('ANN_BETA_SEQUENCE_1CAM_45Minutes_.txt', m, i, 45)
        predicted42, actual42, times42 = file_to_dates('ANN_BETA_SEQUENCE_2CAM_45Minutes_.txt', m, i, 45)

        predicted5, actual5, times5 = file_to_dates('ANN_BETA_SEQUENCE_1CAM_60Minutes_.txt', m, i, 60)
        predicted52, actual52, times52 = file_to_dates('ANN_BETA_SEQUENCE_2CAM_60Minutes_.txt', m, i, 60)

        names = ['Truth', 'Persistence', 'ANN 1 30', 'ANN 2 30', 'ANN 1 45','ANN 2 45', 'ANN 1 60', 'ANN 2 60']
        plot_with_times([actual2, predicted, predicted2, predicted22, predicted4, predicted42, predicted5, predicted52],
                        [times2, times, times2, times22, times4,times42, times5, times52],
                        names, 'GHI forecast ' + str(i) + '/' + str(m), 'GHI in W/m2', xl='Time of day')

def file_to_day_error(file_name):  # returns errors per day
    total_rmse, total_mae, total_mape = [], [], []
    times = []
    with open(file_name) as fp:

        y_true = []
        y_pred = []
        last_day = 11.0  # first day is 11/9/2019

        for line in fp:
            l = line.split(',')
            day = float(l[1])
            month = float(l[0])

            if len(y_true) == 0:
                last_day = day

            if day != last_day:
                rmse, mae, mape = Metrics.get_error(y_true, y_pred)
                total_rmse.append(rmse)
                total_mae.append(mae)
                total_mape.append(mape)

                a = datetime.datetime(year=2019, month=int(month), day=int(day))
                a = matplotlib.dates.date2num(a)
                times.append(a)

                y_true = []
                y_pred = []
                last_day = day

            if l[5][0] == '[':
                l[5] = float(l[5][1:-2])
            if l[6][0] == '[':
                l[6] = float(l[6][1:-2])

            y_true.append(float(l[5]))
            y_pred.append(float(l[6]))

    return times, total_rmse, total_mae, total_mape


def plot_months_error_day():
    times, rmse, _, _ = file_to_day_error('ANN_BETA_SEQUENCE_1CAM_150Minutes_.txt')
    times1, rmse1, _, _ = file_to_day_error('Persistence_b_horizon_20.txt')
    times2, rmse2, _, _ = file_to_day_error('ANN_BETA_SEQUENCE_1CAM_60Minutes_.txt')

    names = ['Persistence', 'ANN 150', 'ANN 60']
    plot_with_months([rmse1, rmse, rmse2], [times1, times, times2],names, 'title', 'yl')

# test_plot()
# plot_months_error_day()