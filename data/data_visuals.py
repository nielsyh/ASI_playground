import matplotlib.pyplot as plt
import matplotlib.style as style
import metrics
from datetime import time, timedelta
import datetime
import matplotlib.pyplot
import data.data_helper
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
from matplotlib.ticker import FuncFormatter, MaxNLocator

def plot_per_weather_circ(errors, horizons, names, title, xl, yl, save_as='none'):
    ax = plt.axes()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.ylim(0, 150)


    plt.plot(horizons, errors[0], 'r', linestyle='-', label=names[0])
    plt.plot(horizons, errors[1], 'r', linestyle=':', label=names[1])
    plt.plot(horizons, errors[2], 'r', linestyle='--', label=names[2])

    plt.plot(horizons, errors[3], 'b', linestyle='-', label=names[3])
    plt.plot(horizons, errors[4], 'b', linestyle=':', label=names[4])
    plt.plot(horizons, errors[5], 'b', linestyle='--', label=names[5])


    plt.legend()
    plt.title(title, fontsize=16)
    plt.xlabel(xl, fontsize=16)
    plt.ylabel(yl, fontsize=16)

    if save_as != 'none':
        data.data_helper.fix_directory()
        plt.savefig(save_as)
    else:
        plt.show()

    plt.close()



def plot_error_per_horizons(errors, horizons, names, title, xl, yl, save_as='none', y_lim=100):
    ax = plt.axes()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(0, y_lim)

    plt.plot(horizons, errors[0], color='grey', linewidth=2, alpha=0.5)
    plt.plot(horizons, errors[0], 'r', linestyle='-', label=names[0], linewidth=4, alpha=0.5)

    plt.plot(horizons, errors[1], color='grey', linewidth=2, alpha=0.5)
    plt.plot(horizons, errors[1], 'b', linestyle='-', label=names[1], linewidth=4, alpha=0.5)

    for idx, i in enumerate(errors):
        if idx > 1:
            plt.plot(horizons, i, color=data.data_helper.getColor_racket(len(errors)-1, idx-1), linestyle='-', label=names[idx], linewidth=2)

    plt.legend()
    fz = 20
    plt.title(title, fontsize=fz)
    plt.xlabel(xl, fontsize=fz)
    plt.ylabel(yl, fontsize=fz)

    if save_as != 'none':
        data.data_helper.fix_directory()
        plt.savefig(save_as)
    else:
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
    data.data_helper.fix_directory()
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

def get_all_TP_multi(file, split=False, md_split=False, md_list_split=False):

    if split:
        print('SPLIT')
    if md_split:
        print('MD SPLIT: ' + str(md_split))

    actual = [[] for x in range(20)]
    predicted = [[] for x in range(20)]
    times = [[] for x in range(20)]

    data.data_helper.fix_directory()

    with open(file) as fp:
        for line in fp:

            l = line.split(',')
#           10.0,5.0,6.0,19.0,1,14.0,0.30910182

            month = int(float(l[0]))
            day = int(float(l[1]))
            hour = int(float(l[2]))
            minute = int(float(l[3]))
            horizon = int(float(l[4])) -1

            if split:
                if hour < split[0] or hour > split[1]:
                    continue
            if md_split:
                if md_split[0] != month:
                    continue
                if md_split[1] != day:
                    continue
            if md_list_split:
                current = (month,day)
                if current not in md_list_split:
                    continue

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

            if int(true) == 0:
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
    data.data_helper.fix_directory()
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

def construct_hybrid():
    files, names = data.data_helper.get_files_all_results()
    t = data.data_helper.get_all_days()


    actual1, pred1, times1 = get_all_TP_multi(str(files[0]))
    actual2, pred2, times2 = get_all_TP_multi(str(files[1]))
    actual3, pred3, times3 = get_all_TP_multi(str(files[1]))

    names.append('Hybrid')
    names.append('Persistence')
    trmse = [[],[],[],[]]


    for i in range(0, 20):
        rmse1, mae1, mape1 = metrics.Metrics.get_error(actual1[i], pred1[i])
        rmse2, mae2, mape2 = metrics.Metrics.get_error(actual2[i], pred2[i])

        actual, pred, _ = data.data_helper.get_persistence_dates(t, 6, 19, i + 1)
        rmse_p, mae_p, mape_p = metrics.Metrics.get_error(actual, pred)

        if i < 5:  #take file 1 RF
            rmse_h, mae_h, mape_h = metrics.Metrics.get_error(actual1[i], pred1[i])
        else:
            rmse_h, mae_h, mape_h = metrics.Metrics.get_error(actual2[i], pred2[i])


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

def plot_multiple_days():
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]
    files, names = data.data_helper.get_files_lstm_multi()

    actual, pred, times = data.data_helper.get_persistence_dates(t, 6, 19, 20,0)  # 24 for 5 lstm
    actual2, pred2, times2 = get_all_TP_multi(files[0])

    print(names[0])
    plot_with_times([actual, pred, actual2[19], pred2[19]], [times,times,times2[19],times2[19]],
                    ['true', 'persistence', 'tr', names[0]], 'title', 'GHI', xl = 'Days')


def plot_day_multi(offset):
    files, names = data.data_helper.get_files_lstm_test()
    t = data.data_helper.get_thesis_test_days()
    actual, pred, times = data.data_helper.get_persistence_dates(t, 6, 19, 20, offset=offset) #24 for 5 lstm
    actual2, pred2, times2 = get_all_TP_multi(files[0])

    plt.plot(actual, linestyle='-', label='Persistece')
    plt.plot(actual2[19], linestyle='-', label='Competing model')
    plt.legend()
    plt.show()
    plt.close()

def get_statistical_sig():
    files, names = data.data_helper.get_files_lstm_test()
    t = data.data_helper.get_thesis_test_days()
    actual, pred, _ = data.data_helper.get_persistence_dates(t, 6, 19, 20, 20)
    actual2, pred2, _ = get_all_TP_multi(files[0])

    print(len(actual))
    print(len(actual2[19]))

    plot_day_multi(24)
    print(metrics.Metrics.dm_test(actual, pred, pred2[19], h=20, crit="MSE", power=2))


def plot_err_all_days():
    files, names = data.data_helper.get_files_all_results()
    file, name = files[1], names[1]
    errors = []
    days = data.data_helper.get_all_days()

    for day in days:
        predicted, actual, _ = file_to_dates(file, day[0], day[1], 0)
        r, _, _ = metrics.Metrics.get_error(actual, predicted)
        errors.append(r)

    plt.plot(errors, linestyle='-', label='errors')
    plt.legend()
    plt.title('title')
    plt.xlabel('days')
    plt.ylabel('RMSE')

    plt.show()
    plt.close()

def plot_err_day_split(model, prediction_horizon):
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]
    hours = list(range(6,19))
    split = []
    times = []

    trmse = []

    for i in hours:
        split.append((i, i+1))
        times.append(str(i) + '-' + str(i+1))

    if model == 'ann':
        files, names = data.data_helper.get_files_ann_multi()
    elif model == 'rf':
        files, names = data.data_helper.get_files_rf_multi()
    elif model == 'lstm':
        files, names = data.data_helper.get_files_lstm_multi()
    elif model == 'best':
        files, names = data.data_helper.get_files_best_multi()
    elif model == 'test':
        files, names = data.data_helper.get_files_test_set()
        t = data.data_helper.get_thesis_test_days()

    for file in files:
        tmp_rmse = []
        for idx, s in enumerate(split):
            if file == 'persistence':
                actual, pred, _ = data.data_helper.get_persistence_dates(t, s[0], s[1], prediction_horizon)
                rmse, mae, mape = metrics.Metrics.get_error(actual, pred)
            else:
                actual, pred, _ = get_all_TP_multi(file, s)
                rmse, mae, mape = metrics.Metrics.get_error(actual[prediction_horizon-1], pred[prediction_horizon-1])

            tmp_rmse.append(rmse)
        trmse.append(tmp_rmse)

    plot_error_per_horizons(trmse, times, names, 'AVG RMSE per hour', 'hours', 'avg error in RMSE')





def plot_err_hor_all(model, max_models=8, save=0):
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]

    if model == 'ann':
        files, names = data.data_helper.get_files_ann_multi()
        folders, names_ = data.data_helper.get_folders_ann()
    elif model == 'rf':
        files, names = data.data_helper.get_files_rf_multi()
        folders, names_ = data.data_helper.get_folders_rf()
    elif model == 'lstm':
        files, names = data.data_helper.get_files_lstm_multi()
        folders, names_ = data.data_helper.get_folders_lstm()
    elif model == 'best':
        files, names = data.data_helper.get_files_best_multi()
        folders, names_ = data.data_helper.get_folders_best()
    elif model == 'cnn':
        # files, names = [],[]
        folders = data.data_helper.get_files_cnn()

    #merge names
    names = names + names_
    trmse = []
    tmae = []
    tmape = []
    predictions = list(range(1, 21))

    # get persistence errors:
    rmse_persistence, mae_persistence, mape_persistence = [],[],[]
    rmse_spersistence, mae_spersistence, mape_spersistence = [], [], []

    for i in range(0, 20):
        actual, pred, _ = data.data_helper.get_persistence_dates(t, 6, 19, i + 1)
        actuals, preds, _ = data.data_helper.get_smart_persistence_dates(t, 6, 19, i + 1)


        rmse, mae, mape,_ = metrics.Metrics.get_error(actual, pred)
        rmse_persistence.append(rmse)
        mae_persistence.append(mae)
        mape_persistence.append(mape)

        rmse, mae, mape,_ = metrics.Metrics.get_error(actuals, preds)
        rmse_spersistence.append(rmse)
        mae_spersistence.append(mae)
        mape_spersistence.append(mape)

    for file in files:  # get multi model data
        tmp_rmse = []
        tmp_mae = []
        tmp_mape = []
        actual, pred, _ = get_all_TP_multi(file)

        for i in range(0,20):
            rmse, mae, mape,_ = metrics.Metrics.get_error(actual[i], pred[i])
            tmp_rmse.append(rmse)
            tmp_mae.append(mae)
            tmp_mape.append(mape)

        trmse.append(tmp_rmse)
        tmae.append(tmp_mae)
        tmape.append(tmp_mape)

    extension = '.txt'
    for f in folders:  # get single model data
        rmse, mae, mape = [], [], []
        for i in predictions:
            file =  f + str(i) + extension
            actual, pred = get_all_TP(file)
            if len(pred) > 0:
                a, b, c, _ = metrics.Metrics.get_error(actual, pred)
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

    id = 0
    for i in range(0, len(trmse), max_models):
        if save == 0:
            save_as = ['none','none','none']
        else:
            name_rmse = 'final_plots_val/' + model + '_prem_rmse_' + str(id) + '.jpg'
            name_mae = 'final_plots_val/' + model + '_prem_mae_' + str(id) + '.jpg'
            name_mape = 'final_plots_val/' + model + '_prem_mape_' + str(id) + '.jpg'
            save_as = [name_rmse, name_mae, name_mape]

        plot_error_per_horizons([rmse_persistence] + [rmse_spersistence] + trmse[i:i+max_models], predictions, ['Persistence', 'Smart-persistence'] + names[i:i+max_models],
                                'RMSE per prediction horizon', 'Prediction horizon in minutes', 'Root mean squared error', save_as[0], y_lim=140)

        plot_error_per_horizons([mae_persistence] + [mae_spersistence] + tmae[i:i+max_models], predictions, ['Persistence', 'Smart-persistence'] + names[i:i+max_models],
                                'MAE per prediction horizon', 'Prediction horizon in minutes', 'Mean average error', save_as[1], y_lim=140)

        plot_error_per_horizons([mape_persistence] + [mape_spersistence] + tmape[i:i+max_models], predictions, ['Persistence', 'Smart-persistence'] + names[i:i+max_models],
                                'MAPE per prediction horizon', 'Prediction horizon in minutes', 'Mean average percentage error', save_as[2], y_lim=140)
        id = id + 1

def plot_err_hor_multi(model):
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]

    if model == 'ann':
        files, names = data.data_helper.get_files_ann_multi()
    elif model == 'rf':
        files, names = data.data_helper.get_files_rf_multi()
    elif model == 'lstm':
        files, names = data.data_helper.get_files_lstm_multi()
    elif model == 'best':
        files, names = data.data_helper.get_files_best_multi()
    elif model == 'test':
        files, names = data.data_helper.get_files_test_set()
        t = data.data_helper.get_thesis_test_days()

    trmse = []
    tmae = []
    tmape = []

    for file in files:
        tmp_rmse = []
        tmp_mae = []
        tmp_mape = []

        if file != 'persistence':
            actual, pred, _ = get_all_TP_multi(file)

        for i in range(0,20):
            if file == 'persistence':
                actual, pred, _ = data.data_helper.get_persistence_dates(t, 6, 19, i+1)
                rmse, mae, mape = metrics.Metrics.get_error(actual, pred)
            else:
                rmse, mae, mape = metrics.Metrics.get_error(actual[i], pred[i])

            # print(rmse)
            tmp_rmse.append(rmse)
            tmp_mae.append(mae)
            tmp_mape.append(mape)

        trmse.append(tmp_rmse)
        tmae.append(tmp_mae)
        tmape.append(tmp_mape)

    predictions = list(range(1, 21))
    plot_error_per_horizons(trmse, predictions, names,
                            'RMSE Error per prediction horizon (multi)', 'Prediction Horizon in minutes', 'Error in RMSE')

    plot_error_per_horizons(tmae, predictions, names,
                            'MAE Error per prediction horizon (multi)', 'Prediction Horizon in minutes', 'Error in MAE')

    plot_error_per_horizons(tmape, predictions, names,
                            'MAPE Error per prediction horizon (multi)', 'Prediction Horizon in minutes', 'Error in MAPE')

def plot_err_hor(model):
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]

    if model == 'ann':
       folders = data.data_helper.get_folders_ann()
    elif model == 'rf':
        folders = data.data_helper.get_folders_rf()
    elif model == 'lstm':
        folders = data.data_helper.get_folders_lstm()
    elif model == 'best':
        folders = data.data_helper.get_folders_best()
    elif model == 'cnn':
        folders, names = data.data_helper.get_files_cnn()

    # 'Persistence',


    extension = '.txt'
    predictions = list(range(1, 21))
    # predictions = [1,2,5,10,15,20]
    trmse, tmae, tmape = [], [], []

    for f in folders:
        rmse, mae, mape = [], [], []
        for i in predictions:

            if f == 'persistence':
                actual, pred, _ = data.data_helper.get_persistence_dates(t, 6, 20, i)
            else:
                file =  f + str(i) + extension
                actual, pred = get_all_TP(file)

            if len(pred) > 0:
                a, b, c = metrics.Metrics.get_error(actual, pred)
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
                            'Average error flow models', 'Prediction Horizon in minutes', 'Root mean squared error',y_lim=500)

    # plot_error_per_horizons(tmae, predictions, names,
    #                         'MAE Error per prediction horizon', 'Prediction Horizon in minutes', 'Error in MAE')
    #
    # plot_error_per_horizons(tmape, predictions, names,
    #                         'MAPE Error per prediction horizon', 'Prediction Horizon in minutes', 'Error in MAPE')

def print_error_prem_day():
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]
    for tup in t:
        # print(tup)
        pred1, actual1, times1 = data.data_helper.get_persistence_df(tup[0], tup[1], 6, 20, 20)
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
        rmse, mae, mape = metrics.Metrics.get_error(actual1, pred1)
        print(rmse, mae, mape)

        print('ann 120')
        rmse, mae, mape = metrics.Metrics.get_error(actual4, pred4)
        print(rmse, mae, mape)

        print('ann 60')
        rmse, mae, mape = metrics.Metrics.get_error(actual2, pred2)
        print(rmse, mae, mape)

        print('ann 20')
        rmse, mae, mape = metrics.Metrics.get_error(actual3, pred3)
        print(rmse, mae, mape)

        print('ann 10')
        rmse, mae, mape = metrics.Metrics.get_error(actual7, pred7)
        print(rmse, mae, mape)

        print('lstm 10')
        rmse, mae, mape = metrics.Metrics.get_error(actual5, pred5)
        print(rmse, mae, mape)

        print('lstm 5')
        rmse, mae, mape = metrics.Metrics.get_error(actual6, pred6)
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


def plot_prem_day(pred_hor=20):
    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]
    for tup in t:
        pred1, actual1, times1 = data.data_helper.get_persistence_df(tup[0], tup[1], 6, 18, pred_hor)
        pred2, actual2, times2 = file_to_dates('results_18march/cnn/farneback/CNN_predhor1.txt', tup[0], tup[1], 1)
        pred3, actual3, times3 = file_to_dates('results_18march/cnn/lukaskanade/CNN_predhor_LK1.txt', tup[0], tup[1], 1)

        names = ['true', 'persistence', 'CNN FB', 'CNN LK']

        plot_with_times([actual1, pred1, pred2, pred3],
                        [times1, times1, times2, times3],
                        names, 'GHI forecast ' + str(tup[1]) + '/' + str(tup[0]), 'GHI in W/m2', xl='Time of day')


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
                rmse, mae, mape = metrics.Metrics.get_error(y_true, y_pred)
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

