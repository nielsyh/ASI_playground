import matplotlib.pyplot as plt
import matplotlib.style as style

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

def plot_error_per_horizons(errors, horizons, names, title, xl, yl):
    ax = plt.axes()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='x-small')

    lines = ['-','-','-','-','-','-','-','--','--','--','--','--','--','--']

    for idx, i in enumerate(errors):
        plt.plot(horizons, i, linestyle=lines[idx], label=names[idx])

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

def get_all_TP(file):
    predicted, actual = [], []
    with open(file) as fp:
        for line in fp:
            l = line.split(',')

            #TODO CONSIDER TIMES
            month = int(float(l[0]))
            day = int(float(l[1]))
            hour = int(float(l[2]))
            minute = int(float(l[3]))

            true = float(l[5])
            pred = float(l[6])

            if int(true) == 0 or int(pred) == 0:
                continue
            else:
                predicted.append(pred)
                actual.append(true)

    return actual, predicted

def get_all_TP_multi(file):
    actual = [[] for x in range(20)]
    predicted = [[] for x in range(20)]
    times = [[] for x in range(20)]

    data_helper.fix_directory()

    with open(file) as fp:
        for line in fp:

            l = line.split(',')
#           10.0,5.0,6.0,19.0,1,14.0,0.30910182

            month = int(float(l[0]))
            day = int(float(l[1]))
            hour = int(float(l[2]))
            minute = int(float(l[3]))
            horizon = int(float(l[4])) -1

            a = datetime.datetime(year=2019, month=int(month), day=int(day), hour=int(hour), minute=int(minute))
            a = a + timedelta(minutes=horizon)
            a = matplotlib.dates.date2num(a)
            times[horizon].append(a)

            true = float(l[5])
            pred = float(l[6])

            actual[horizon].append(true)
            predicted[horizon].append(pred)

    return actual, predicted, times

def file_to_values(file, prediction_horizon = 0, times_=False):
    predicted, actual = [],[]
    times = []
    with open(file) as fp:
        for line in fp:
            l = line.split(',')

            month = int(float(l[0]))
            day = int(float(l[1]))
            hour = int(float(l[2]))
            minute = int(float(l[3]))
            pred_hor = int(float(l[4]))

            if prediction_horizon > 0 and pred_hor != prediction_horizon:
                continue

            if l[5][0] == '[':
                true = float(l[5][1:-2])
            else:
                true = float(l[5])
            if l[6][0] == '[':
                pred = float(l[6][1:-2])
            else:
                pred = float(l[6])

            if int(true) == 0 or int(pred) == 0:
                continue
            else:
                predicted.append(pred)
                actual.append(true)
                times.append((hour, minute))
                # a = datetime.datetime(hour=int(hour), minute=int(minute))
                # a = matplotlib.dates.date2num(a)
                # times.append(a)

    if times_:
        print('TIMES TRUE')
        return predicted, actual, times
    else:
        return predicted, actual


def file_to_dates(file, month, day, offset):
    data_helper.fix_directory()
    predicted, actual = [],[]
    times = []
    with open(file) as fp:
        for line in fp:
            l = line.split(',')
            # print(l)
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

def get_files_best_multi():

    files = ['persistence',
             'ann/ANN_SEQUENCE_MULTIepochs_40_sqnc_20data_all data.txt',
             'lstm/LSTM_SEQUENCE_MULTIepochs_50_sqnc_5data_all data.txt',
             'lstm/LSTM_SEQUENCE_MULTIepochs_50_sqnc_5data_onsite_only.txt',
             'rf/RF SEQUENCE multi_sqnc_30data_all data.txt'
             ]

    names = ['Persistence',
             'ANN 20 alldata',
             'LSTM 5 all data',
             'LSTM 5 onsite only',
             'RF 30 all data']

    return files, names

def get_files_ann_multi():
    files = ['persistence',
             'ann/ANN_SEQUENCE_MULTIepochs_40_sqnc_20data_all data.txt',
             'ann/ANN_SEQUENCE_MULTIepochs_40_sqnc_20data_img only.txt',
             'ann/ANN_SEQUENCE_MULTIepochs_40_sqnc_20data_meteor only.txt',
             'ann/ANN_SEQUENCE_MULTIepochs_40_sqnc_20data_onsite_only.txt',
             'ann/ANN_SEQUENCE_MULTIepochs_40_sqnc_40data_all data.txt',
             'ann/ANN_SEQUENCE_MULTIepochs_40_sqnc_40data_img only.txt',
             'ann/ANN_SEQUENCE_MULTIepochs_40_sqnc_40data_onsite_only.txt'
             ]

    names = ['persistence',
             'ANN 20 all', 'ANN 20 img', 'ANN 20 metoer', 'ANN 20 onsite',
             'ANN 40 all', 'ANN 40 img', 'ANN 40 onsite']

    return files, names


def get_files_lstm_multi():
    #
    #
    files = [ 'persistence',
              'lstm/LSTM_SEQUENCE_MULTIepochs_50_sqnc_5data_all data.txt',
             'lstm/LSTM_SEQUENCE_MULTIepochs_50_sqnc_5data_meteor only.txt',
             'lstm/LSTM_SEQUENCE_MULTIepochs_50_sqnc_5data_img only.txt',
             'lstm/LSTM_SEQUENCE_MULTIepochs_50_sqnc_5data_onsite_only.txt',
             'lstm/LSTM_SEQUENCE_MULTIepochs_50_sqnc_10data_all data.txt',
              'lstm/LSTM_SEQUENCE_MULTIepochs_50_sqnc_3data_onsite,img.txt',
              'lstm/LSTM_SEQUENCE_MULTIepochs_50_sqnc_5data_onsite,img.txt',
              'lstm/LSTM_SEQUENCE_MULTIepochs_50_sqnc_10data_onsite,img.txt',
              'lstm/LSTM_SEQUENCE_MULTIepochs_50_sqnc_5data_all data 2 cam.txt'
             ]



    # 'persistence',
    names = ['Persistence',
             'LSTM 5 all',
             'LSTM 5 metoer only',
             'LSTM 5 img only',
             'LSTM 5 on-site only',
             'LSTM 10 all',
             'LSTM 3 onsite/img',
             'LSTM 5 onsite/img',
             'LSTM 10 onsite/img',
             'LSTM 5 all data 2CAM'
             ]

    return files, names

def get_files_rf_multi():
    files = ['persistence',
             'rf/RF SEQUENCE multi_sqnc_30data_all data.txt',
             'rf/RF SEQUENCE multi_sqnc_30data_img only.txt',
             'rf/RF SEQUENCE multi_sqnc_30data_meteor only.txt',
             'rf/RF SEQUENCE multi_sqnc_60data_all data.txt',
             'rf/RF SEQUENCE multi_sqnc_120data_all data.txt'
             ]

    names = ['persistence',
             'RF 30 all',
             'RF 30 img only',
             'RF 30 meteor only',
             'RF 60 all',
             'RF 120 all']

    return files, names

def get_files_all_results():
    files = ['RF SEQUENCE multi_sqnc_30data_all.txt',
             'LSTM_SEQUENCE_MULTI_alldata_epochs_50_sqnc_5data_all.txt'
             ]

    names = ['RF 30 all data','LSTM 5 all data']

    return files, names

def get_files_test_set():

    files = ['persistence','RF SEQUENCE multi testset_sqnc_30data_all.txt',
             'ANN_SEQUENCE_MULTI_testsetepochs_40_sqnc_5data_all.txt',
             'LSTM_SEQUENCE_MULTI_testsetepochs_50_sqnc_5data_all.txt',
             'LSTM_SEQUENCE_MULTIepochs_50_sqnc_5data_all data 2 cam.txt',
             ''
             ]

    names = ['Persistence', 'rf 30', 'ann 20', 'lstm 5', 'LSTM 5 2cam']

    return files, names



def construct_hybrid():
    files, names = get_files_all_results()
    t = data_helper.get_all_days()


    actual1, pred1, times1 = get_all_TP_multi(str(files[0]))
    actual2, pred2, times2 = get_all_TP_multi(str(files[1]))
    actual3, pred3, times3 = get_all_TP_multi(str(files[1]))

    names.append('Hybrid')
    names.append('Persistence')
    trmse = [[],[],[],[]]


    for i in range(0, 20):
        rmse1, mae1, mape1 = Metrics.get_error(actual1[i], pred1[i])
        rmse2, mae2, mape2 = Metrics.get_error(actual2[i], pred2[i])

        actual, pred, _ = data_helper.get_persistence_dates(t, 6, 19, i + 1)
        rmse_p, mae_p, mape_p = Metrics.get_error(actual, pred)

        if i < 5:  #take file 1 RF
            rmse_h, mae_h, mape_h = Metrics.get_error(actual1[i], pred1[i])
        else:
            rmse_h, mae_h, mape_h = Metrics.get_error(actual2[i], pred2[i])


        trmse[0].append(rmse1)
        trmse[1].append(rmse2)
        trmse[2].append(rmse_h)
        trmse[3].append(rmse_p)


    predictions = list(range(1, 21))

    print(trmse, predictions, names)

    plot_error_per_horizons(trmse, predictions, names,
                            'RMSE Error per prediction horizon (multi)', 'Prediction Horizon in minutes',
                            'Error in RMSE')

    # plot_error_per_horizons(tmae, predictions, names,
    #                         'MAE Error per prediction horizon (multi)', 'Prediction Horizon in minutes', 'Error in MAE')
    #
    # plot_error_per_horizons(tmape, predictions, names,
    #                         'MAPE Error per prediction horizon (multi)', 'Prediction Horizon in minutes',
    #                         'Error in MAPE')



def plot_day_multi(offset):
    files, names = get_files_best_multi()
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]
    add = 'prem results multi/'
    actual, pred, times = data_helper.get_persistence_dates(t, 6, 19, 20, offset=offset) #24 for 5 lstm
    actual2, pred2, times2 = get_all_TP_multi(str(add + files[2]))

    plt.plot(actual, linestyle='-', label='1')
    plt.plot(actual2[19], linestyle='-', label='2')
    plt.legend()
    plt.show()
    plt.close()

def get_statistical_sig():
    files, names = get_files_best_multi()
    add = 'prem results multi/'
    print(str(add + files[0]))
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]
    actual, pred, _ = data_helper.get_persistence_dates(t, 6, 19, 20, 24)
    actual2, pred2, _ = get_all_TP_multi(str(add + files[2]))


    print(Metrics.dm_test(actual, pred, pred2[19], h=20, crit="MSE", power=2))


def plot_err_all_days():
    files, names = get_files_all_results()
    file, name = files[1], names[1]
    errors = []
    days = data_helper.get_all_days()

    for day in days:
        predicted, actual, _ = file_to_dates(file, day[0], day[1], 0)
        r, _, _ = Metrics.get_error(actual, predicted)
        errors.append(r)

    plt.plot(errors, linestyle='-', label='errors')
    plt.legend()
    plt.title('title')
    plt.xlabel('days')
    plt.ylabel('RMSE')

    plt.show()
    plt.close()


def plot_err_hor_multi(model):
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]

    if model == 'ann':
        files, names = get_files_ann_multi()
    elif model == 'rf':
        files, names = get_files_rf_multi()
    elif model == 'lstm':
        files, names = get_files_lstm_multi()
    elif model == 'best':
        files, names = get_files_best_multi()
    elif model == 'test':
        files, names = get_files_test_set()
        t = data_helper.get_thesis_test_days()

    trmse = []
    tmae = []
    tmape = []

    for file in files:
        tmp_rmse = []
        tmp_mae = []
        tmp_mape = []
        if file != 'persistence' and model != 'test':
            add = 'prem results multi/'
            actual, pred, _ = get_all_TP_multi(add + file)
        elif model == 'test' and file != 'persistence':
            actual, pred, _ = get_all_TP_multi(file)

        for i in range(0,20):
            if file == 'persistence':
                actual, pred, _ = data_helper.get_persistence_dates(t, 6, 19, i+1)
                rmse, mae, mape = Metrics.get_error(actual, pred)
            else:
                rmse, mae, mape = Metrics.get_error(actual[i], pred[i])

            # print(rmse)
            tmp_rmse.append(rmse)
            tmp_mae.append(mae)
            tmp_mape.append(mape)

        trmse.append(tmp_rmse)
        tmae.append(tmp_mae)
        tmape.append(tmp_mape)

    predictions = list(range(1, 21))

    print(trmse, predictions, names)

    plot_error_per_horizons(trmse, predictions, names,
                            'RMSE Error per prediction horizon (multi)', 'Prediction Horizon in minutes', 'Error in RMSE')

    plot_error_per_horizons(tmae, predictions, names,
                            'MAE Error per prediction horizon (multi)', 'Prediction Horizon in minutes', 'Error in MAE')

    plot_error_per_horizons(tmape, predictions, names,
                            'MAPE Error per prediction horizon (multi)', 'Prediction Horizon in minutes', 'Error in MAPE')


def get_folders_ann():
    folders = ['persistence',
               'prem results/ANN 5 IMG/ANN_SEQUENCE_epochs_40_sequence_5CAM_1_img_Truepredhor_',
           'prem results/ANN 5 NOIMG/ANN_SEQUENCE_epochs_40_sequence_5CAM_1_img_Falsepredhor_',
           'prem results/ANN 10 IMG/ANN_SEQUENCE_epochs_40_sequence_10CAM_1_img_Truepredhor_',
           'prem results/ANN 20 IMG/ANN_SEQUENCE_epochs_40_sequence_20CAM_1_img_Truepredhor_',
           'prem results/ANN 20 NOIMG/ANN_SEQUENCE_epochs_40_sequence_20CAM_1_img_Falsepredhor_',
           'prem results/ANN 50 IMG/ANN_SEQUENCE_epochs_40_sequence_50CAM_1_img_Truepredhor_',
           'prem results/ANN 60 NOIMG NOMETEOR/ANN_SEQUENCE_NOMETEORepochs_40_sequence_60CAM_1_img_Falsepredhor_'
               ]

    return folders


def get_folders_rf():
    folders = ['persistence',
               'prem results/RF 5 IMG/RF SEQUENCE PREM__sequence_5CAM_1_img_Truepredhor_',
               'prem results/RF 5 NOIMG/RF SEQUENCE PREM__sequence_5CAM_1_img_Falsepredhor_',
               'prem results/RF 10 IMG/RF SEQUENCE PREM__sequence_10CAM_1_img_Truepredhor_',
               'prem results/RF 10 NOIMG/RF SEQUENCE PREM__sequence_10CAM_1_img_Falsepredhor_',
               'prem results/RF 20 IMG/RF SEQUENCE PREM__sequence_20CAM_1_img_Truepredhor_',
               'prem results/RF 20 NOIMG/RF SEQUENCE PREM__sequence_20CAM_1_img_Falsepredhor_',
               'prem results/RF 30 IMG/RF SEQUENCE PREM__sequence_30CAM_1_img_Truepredhor_',
               'prem results/RF 30 NOIMG/RF SEQUENCE PREM__sequence_30CAM_1_img_Falsepredhor_',
               'prem results/RF 60 IMG/RF SEQUENCE PREM__sequence_60CAM_1_img_Truepredhor_',
               'prem results/RF 60 NOIMG/RF SEQUENCE PREM__sequence_60CAM_1_img_Falsepredhor_',
               'prem results/RF 120 IMG/RF SEQUENCE PREM__sequence_120CAM_1_img_Truepredhor_',
               'prem results/RF 120 NOIMG/RF SEQUENCE PREM__sequence_120CAM_1_img_Falsepredhor_']

    return folders

def get_folders_lstm():
    folders = ['persistence',
               'prem results/LSTM 5 IMG/LSTM_SEQUENCE_epochs_40_sequence_5CAM_1_img_Truepredhor_',
               'prem results/LSTM 10 IMG/LSTM_SEQUENCE_epochs_40_sequence_10CAM_1_img_Truepredhor_',
               'prem results/LSTM 10 NOIMG/LSTM_BETA_SEQUENCE_epochs_40CAM_1_sequence_10predhor_',
               'prem results/LSTM 20 IMG/LSTM_SEQUENCE_epochs_40_sequence_20CAM_1_img_Truepredhor_',
               'prem results/LSTM 20 NOIMG/LSTM_SEQUENCE_epochs_40_sequence_20CAM_1_img_Falsepredhor_']

    return folders

def get_folders_best():
    folders = ['persistence',
               'prem results/LSTM 5 IMG/LSTM_SEQUENCE_epochs_40_sequence_5CAM_1_img_Truepredhor_',
               'prem results/LSTM 10 IMG/LSTM_SEQUENCE_epochs_40_sequence_10CAM_1_img_Truepredhor_',
               'prem results/LSTM 10 NOIMG/LSTM_BETA_SEQUENCE_epochs_40CAM_1_sequence_10predhor_',
               'prem results/RF 60 IMG/RF SEQUENCE PREM__sequence_60CAM_1_img_Truepredhor_',
               'prem results/RF 60 NOIMG/RF SEQUENCE PREM__sequence_60CAM_1_img_Falsepredhor_',
               'prem results/RF 120 IMG/RF SEQUENCE PREM__sequence_120CAM_1_img_Truepredhor_',
               'prem results/RF 120 NOIMG/RF SEQUENCE PREM__sequence_120CAM_1_img_Falsepredhor_'
               ]

    return folders

def plot_err_hor(model):
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]

    if model == 'ann':
       folders = get_folders_ann()
    elif model == 'rf':
        folders = get_folders_rf()
    elif model == 'lstm':
        folders = get_folders_lstm()
    elif model == 'best':
        folders = get_folders_best()

    # 'Persistence',
    names = ['Persistence']

    extension = '.txt'
    predictions = list(range(1, 21))
    # predictions = [1,2,5,10,15,20]
    trmse, tmae, tmape = [], [], []

    for f in folders:
        if f != 'persistence':
            names.append(f[f.find('/') + 1:-1][0:(f[f.find('/') + 1:-1]).find('/')])
        rmse, mae, mape = [], [], []
        for i in predictions:

            if f == 'persistence':
                actual, pred = data.data_helper.get_persistence_dates(t, 6, 20, i)
            else:
                file =  f + str(i) + extension
                actual, pred = get_all_TP(file)

            if len(pred) > 0:
                a, b, c = Metrics.get_error(actual, pred)
                rmse.append(a)
                mae.append(b)
                mape.append(c)
            else:
                rmse.append(0)
                mae.append(0)
                mape.append(0)

        trmse.append(rmse)
        tmae.append(mae)
        tmape.append(mape)

    plot_error_per_horizons(trmse, predictions, names,
                            'RMSE Error per prediction horizon', 'Prediction Horizon in minutes', 'Error in RMSE')

    plot_error_per_horizons(tmae, predictions, names,
                            'MAE Error per prediction horizon', 'Prediction Horizon in minutes', 'Error in MAE')

    plot_error_per_horizons(tmape, predictions, names,
                            'MAPE Error per prediction horizon', 'Prediction Horizon in minutes', 'Error in MAPE')

def print_error_prem_day():
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]
    for tup in t:
        # print(tup)
        pred1, actual1, times1 = data_helper.get_persistence_df(tup[0], tup[1], 6, 20, 20)
        pred2, actual2, times2 = file_to_dates('prem results/ANN_SEQUENCE_40epoch_pred60_1CAM_20Minutes_.txt', tup[0], tup[1], 20)
        pred3, actual3, times3 = file_to_dates('prem results/ANN PREM 20 min 1 cam/ANN_SEQUENCE_40epoch_pred20_1CAM_20Minutes_.txt', tup[0], tup[1], 20)
        pred4, actual4, times4 = file_to_dates('prem results/ANN PREM 120min 1 cam/ANN_SEQUENCE_40epoch_pred20_1CAM_120Minutes_.txt', tup[0], tup[1], 20)

        pred5, actual5, times5 = file_to_dates('LSTM_BETA_SEQUENCE_CAM_1_sequence_10predhor_20.txt', tup[0], tup[1], 20)
        pred6, actual6, times6 = file_to_dates('LSTM_BETA_SEQUENCE_CAM_1_sequence_5predhor_20.txt', tup[0], tup[1], 20)

        pred7, actual7, times7 = file_to_dates('prem results/ANN PREM 10min 1cam/ANN_SEQUENCE_40epoch_pred20_CAMs_1sq_10.txt', tup[0], tup[1], 20)

        names = ['true', 'persistence', 'ann 60', 'ann 20', 'ann 120', 'lstm 10', 'lstm 5', 'ann 10']

        print('--------------------------------------------------------------------------------------------------')
        print(tup)
        print('RMSE, MAE, MAPE')
        print('Persistence:')
        rmse, mae, mape = Metrics.get_error(actual1, pred1)
        print(rmse, mae, mape)

        print('ann 120')
        rmse, mae, mape = Metrics.get_error(actual4, pred4)
        print(rmse, mae, mape)

        print('ann 60')
        rmse, mae, mape = Metrics.get_error(actual2, pred2)
        print(rmse, mae, mape)

        print('ann 20')
        rmse, mae, mape = Metrics.get_error(actual3, pred3)
        print(rmse, mae, mape)

        print('ann 10')
        rmse, mae, mape = Metrics.get_error(actual7, pred7)
        print(rmse, mae, mape)

        print('lstm 10')
        rmse, mae, mape = Metrics.get_error(actual5, pred5)
        print(rmse, mae, mape)

        print('lstm 5')
        rmse, mae, mape = Metrics.get_error(actual6, pred6)
        print(rmse, mae, mape)


def plot_prem_day_folder():
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]

    folders = ['prem results/ANN PREM 10min 1cam/ANN_SEQUENCE_40epoch_sq10_1cam_pred',
               'prem results/ann prem 20 img/ANN_SEQUENCE_epochs_40_sequence_20CAM_1_img_Truepredhor_',
               'prem results/ann Prem 20 noimg/ANN_SEQUENCE_epochs_40_sequence_20CAM_1_img_Falsepredhor_',
               'prem results/ANN 10 img/ANN_SEQUENCE_epochs_40_sequence_10CAM_1_img_Truepredhor_',
               'prem results/ANN 5 noimg/ANN_SEQUENCE_epochs_40_sequence_5CAM_1_img_Falsepredhor_']

    extension = '.txt'
    predictions = list(range(1,21))


    for f in folders:
        for i in predictions:
            file = f + str(i) + extension

            for tup in t:
                pred, actual, times = file_to_dates(file, tup[0], tup[1], 20)
                name = str(tup) + str(file)
                plot_with_times([actual, pred],
                                [times, times],
                                ['actual', 'pred'], name, 'GHI in W/m2',
                                xl='Time of day')






def plot_prem_day():
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]
    for tup in t:
        # pred0, actual0, times0 = file_to_dates('prem results/SVR SEQUENCE PREM_1CAM_60Minutes__pred_hor_20.txt', tup[0], tup[1],0)
        # pred0, actual0, times0 = get_persistence_df(tup[0], tup[1], 6, 20, 20)
        # pred1, actual1, times1 = file_to_dates('results/Persistence_b_horizon_20.txt', tup[0], tup[1], 0)

        pred1, actual1, times1 = data.get_persistence_df(tup[0], tup[1], 6, 20, 20)
        pred2, actual2, times2 = file_to_dates('prem results/ANN_SEQUENCE_40epoch_pred60_1CAM_20Minutes_.txt', tup[0], tup[1], 20)
        pred3, actual3, times3 = file_to_dates('prem results/ANN PREM 20 min 1 cam/ANN_SEQUENCE_40epoch_pred20_1CAM_20Minutes_.txt', tup[0], tup[1], 20)
        pred4, actual4, times4 = file_to_dates('prem results/ANN PREM 120min 1 cam/ANN_SEQUENCE_40epoch_pred20_1CAM_120Minutes_.txt', tup[0], tup[1], 20)

        pred5, actual5, times5 = file_to_dates('LSTM_BETA_SEQUENCE_CAM_1_sequence_10predhor_20.txt', tup[0], tup[1], 20)
        pred6, actual6, times6 = file_to_dates('LSTM_BETA_SEQUENCE_CAM_1_sequence_5predhor_20.txt', tup[0], tup[1], 20)

        pred7, actual7, times7 = file_to_dates('prem results/ANN PREM 10min 1cam/ANN_SEQUENCE_40epoch_pred20_CAMs_1sq_10.txt', tup[0], tup[1], 20)



        names = ['true', 'persistence', 'ann 60', 'ann 20', 'ann 120', 'lstm 10', 'lstm 5', 'ANN 10']

        plot_with_times([actual1, pred1, pred2, pred3, pred4, pred5, pred6, pred7],
                        [times1, times1, times2, times3, times4,times5,times6, times7],
                        names, 'GHI forecast ' + str(tup[1]) + '/' + str(tup[0]), 'GHI in W/m2', xl='Time of day')

        # names = ['persist', 'true p', 'lstm', 'true l']
        # plot_with_times([pred1, actual1, pred6, actual6],
        #                 [times1, times1, times6, times6],
        #                 names, 'GHI forecast ' + str(tup[1]) + '/' + str(tup[0]), 'GHI in W/m2', xl='Time of day')

def days_plot():
    m = 10
    for i in list(range(1, 10)):
        predicted, actual, times = file_to_dates('results/Persistence_b_horizon_20.txt', m, i, 0)
        predicted2, actual2, times2 = file_to_dates('results/ANN_BETA_SEQUENCE_1CAM_30Minutes_.txt', m, i, 30)
        # predicted22, actual22, times22 = file_to_dates('results/ANN_BETA_SEQUENCE_2CAM_30Minutes_.txt', m, i, 30)

        predicted4, actual4, times4 = file_to_dates('results/ANN_BETA_SEQUENCE_1CAM_45Minutes_.txt', m, i, 45)
        # predicted42, actual42, times42 = file_to_dates('results/ANN_BETA_SEQUENCE_2CAM_45Minutes_.txt', m, i, 45)

        predicted5, actual5, times5 = file_to_dates('results/ANN_BETA_SEQUENCE_1CAM_60Minutes_.txt', m, i, 60)
        predicted52, actual52, times52 = file_to_dates('results/ANN_BETA_SEQUENCE_2CAM_60Minutes_.txt', m, i, 60)

        # predicted6, actual6, times6 = file_to_dates('results/ANN_BETA_SEQUENCE_IMG1CAM_60Minutes_.txt', m, i, 0)


        names = ['Truth', 'Persistence', 'ANN 1 30', 'ANN 2 30', 'ANN 1 45','ANN 2 45', 'ANN 1 60', 'ANN 2 60', 'ANN 60 img']
        plot_with_times([actual2, predicted, predicted2, predicted4, predicted5, predicted52],
                        [times2, times, times2, times4, times5, times52 ],
                        names, 'GHI forecast ' + str(i) + '/' + str(m), 'GHI in W/m2', xl='Time of day')

def plot_day(m, d):
    files = ['prem results/LSTM 10 IMG/LSTM_SEQUENCE_epochs_40_sequence_10CAM_1_img_Truepredhor_20.txt',
             'LSTM_TEST 915.txt']
    names = ['true', 'lstm 10', 'lstm 10 CSI']
    predicted, actual, times = data.get_persistence_df(m, d, 6, 19, 20)
    predicted2, actual2, times2 = file_to_dates(files[1], m, d, 10)

    plot_with_times([actual, predicted, predicted2],
                    [times, times, times2],
                    names, 'GHI forecast ' + str(d) + '/' + str(m), 'GHI in W/m2', xl='Time of day')


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
    times, rmse, _, _ = file_to_day_error('SVR SEQUENCE PREM_1CAM_60Minutes__pred_hor_20.txt')
    times1, rmse1, _, _ = file_to_day_error('results/Persistence_b_horizon_20.txt')
    times2, rmse2, _, _ = file_to_day_error('results/ANN_BETA_SEQUENCE_1CAM_60Minutes_.txt')

    names = ['Persistence', 'SVR', 'ANN 60']
    plot_with_months([rmse1, rmse, rmse2], [times1, times, times2],names, 'title', 'yl')

# test_plot()
# plot_months_error_day()
# plot_err_hor_multi('rf')
# plot_err_hor_multi('ann')
# plot_err_hor_multi('lstm')
# plot_err_hor_multi('best')