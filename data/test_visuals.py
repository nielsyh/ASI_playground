from data.data_helper import getColor_binairy
from metrics import *
from data import data_visuals, data_helper
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')

def mean(a):
    return sum(a) / len(a)

def avg_lol(lol):
    return list(map(mean, zip(*lol)))

def proces_lists_stack(lst):
    ls = lst
    for i in ls:
        for x in range(0, 19):
            i[x + 1] = i[x + 1] - i[x]
            if i[x] < 0:
                i[x] = 0

    return ls

def get_files_names(model):
    if model == 'ann':
        files, names = data_helper.get_files_ann_test()
    elif model == 'rf':
        files, names = data_helper.get_files_rf_test()
    elif model == 'lstm':
        files, names = data_helper.get_files_lstm_test()
    elif model == 'p':
        return ['p'], ['Persistence']

    return files, names

def plot_bar(model, save_as=0):
    days = data_helper.get_thesis_test_days()
    files, names = get_files_names(model)

    sunny = [(9, 15), (10, 15), (11, 15), (12, 15)]
    pcloudy = [(10, 21), (11, 17), (12, 16)]
    cloudy = [(10, 22), (12, 3)]

    for idx, file in enumerate(files):

        errors_sunny = []
        errors_pcloudy = []
        errors_cloudy = []

        for t in days:
            # merge names
            trmse, tmae, tmape = [[] for x in range(3)]
            predictions = list(range(1, 21))

            if model == 'p':
                # get persistence errors:
                rmse_persistence, mae_persistence, mape_persistence = [[] for x in range(3)]
                # get persistence
                for i in range(0, 20):
                    actual, pred, _ = data_helper.get_persistence_df(t[0], t[1], 7, 17, i + 1)
                    rmse, mae, mape = Metrics.get_error(actual, pred)
                    rmse_persistence.append(rmse)
                    mae_persistence.append(mae)
                    mape_persistence.append(mape)

                if t in sunny:
                    errors_sunny.append(rmse_persistence)
                elif t in pcloudy:
                    errors_pcloudy.append(rmse_persistence)
                elif t in cloudy:
                    errors_cloudy.append(rmse_persistence)
            else:
                tmp_rmse = []
                tmp_mae = []
                tmp_mape = []
                actual, pred, _ = data_visuals.get_all_TP_multi(file, md_split=t)

                for i in range(0, 20):
                    rmse, mae, mape = Metrics.get_error(actual[i], pred[i])
                    tmp_rmse.append(rmse)
                    tmp_mae.append(mae)
                    tmp_mape.append(mape)
                if t in sunny:
                    errors_sunny.append(tmp_rmse)
                elif t in pcloudy:
                    errors_pcloudy.append(tmp_rmse)
                elif t in cloudy:
                    errors_cloudy.append(tmp_rmse)

        labels = ['Sunny', 'Par. cloudy', 'Cloudy']
        r = list(range(0, len(labels)))

        # average weather
        errors_sunny = avg_lol(errors_sunny)
        errors_pcloudy = avg_lol(errors_pcloudy)
        errors_cloudy = avg_lol(errors_cloudy)

        errors = [errors_sunny] + [errors_pcloudy] + [errors_cloudy]
        # process for bar
        proces_lists_stack(errors)

        plt.bar(r, [item[0] for item in errors], color=data_helper.getColor_binairy(25, 3), edgecolor='white', width=1, label='1')
        for i in range(1, 20):
            plt.bar(r, [item[i] for item in errors], bottom=[item[i - 1] for item in errors], color=data_helper.getColor_binairy(25, i+3),
                    edgecolor='white', width=1, label=str(i + 1))

        # Custom X axis
        plt.xticks(r, labels, fontweight='bold')
        plt.xlabel("Weather circumstances")
        handles, labels = plt.axes().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], title='Prediction horizon', fancybox=True)

        plt.ylim(0, 100)
        plt.ylabel('Error RMSE')
        plt.title('Average error per weather circumstance for '+ str(names[idx]))

        if save_as != 0:
            data_helper.fix_directory()
            dir = 'Results test set/'
            name = names[idx]
            plt.savefig(dir + name)
        else:
            plt.show()

        plt.close()

def plot_days_sep_bar(model):
    days = data_helper.get_thesis_test_days()
    files, names = get_files_names(model)
    file = files[0]
    name = names[0]

    for t in days:
        # merge names
        trmse, tmae, tmape = [[] for x in range(3)]
        predictions = list(range(1, 21))

        if model == 'p':
            # get persistence errors:
            rmse_persistence, mae_persistence, mape_persistence = [[] for x in range(3)]
            # get persistence
            for i in range(0, 20):
                actual, pred, _ = data_helper.get_persistence_df(t[0], t[1], 7, 17, i + 1)
                rmse, mae, mape = Metrics.get_error(actual, pred)
                rmse_persistence.append(rmse)
                mae_persistence.append(mae)
                mape_persistence.append(mape)

        else:
            tmp_rmse = []
            tmp_mae = []
            tmp_mape = []
            actual, pred, _ = data_visuals.get_all_TP_multi(file, md_split=t)

            for i in range(0, 20):
                rmse, mae, mape = Metrics.get_error(actual[i], pred[i])
                tmp_rmse.append(rmse)
                tmp_mae.append(mae)
                tmp_mape.append(mape)

            # process for bar
            tmp_rmse = proces_lists_stack(tmp_rmse)
            plt.bar(r, [item[0] for item in tmp_rmse], color=getColor_binairy(30, 5), edgecolor='white', width=1, label='PH 1')
            for i in range(1, 20):
                plt.bar(r, [item[i] for item in tmp_rmse], bottom=[item[i - 1] for item in tmp_rmse], color=getColor_binairy(30, i+5),
                        edgecolor='white', width=1, label='PH ' + str(i + 1))

            labels = ['model']
            y_pos = np.arange(len(labels))
            r = list(range(0, len(labels)))
            # Custom X axis
            plt.xticks(r, labels, fontweight='bold')
            plt.xlabel("Weather circumstances")
            plt.legend()

            plt.ylim(0, 100)
            plt.ylabel('Error RMSE')
            plt.title('Average error per weather circumstance for '+ str(name))
            plt.show()


def plot_err_hor_test(model, max_models=9, save=0):
    t = data_helper.get_thesis_test_days()
    files, names = get_files_names(model)

    # merge names
    trmse = []
    tmae = []
    tmape = []
    predictions = list(range(1, 21))

    # get persistence errors:
    rmse_persistence = []
    mae_persistence = []
    mape_persistence = []

    for i in range(0, 20):

        actual, pred, _ = data_helper.get_persistence_dates(t, 7, 17, i + 1)
        rmse, mae, mape = Metrics.get_error(actual, pred)
        rmse_persistence.append(rmse)
        mae_persistence.append(mae)
        mape_persistence.append(mape)

    for file in files:  # get multi model data
        tmp_rmse = []
        tmp_mae = []
        tmp_mape = []
        actual, pred, _ = data_visuals.get_all_TP_multi(file)

        for i in range(0, 20):

            rmse, mae, mape = Metrics.get_error(actual[i], pred[i])
            tmp_rmse.append(rmse)
            tmp_mae.append(mae)
            tmp_mape.append(mape)

        trmse.append(tmp_rmse)
        tmae.append(tmp_mae)
        tmape.append(tmp_mape)

    id = 0
    for i in range(0, len(trmse), max_models):
        if save == 0:
            save_as = ['none', 'none', 'none']
        else:
            name_rmse = 'final_plots_test/' + model + '_prem_rmse_' + str(id) + '.jpg'
            name_mae = 'final_plots_test/' + model + '_prem_mae_' + str(id) + '.jpg'
            name_mape = 'final_plots_test/' + model + '_prem_mape_' + str(id) + '.jpg'
            save_as = [name_rmse, name_mae, name_mape]

        data_visuals.plot_error_per_horizons([rmse_persistence] + trmse[i:i + max_models], predictions,
                                ['Persistence'] + names[i:i + max_models],
                                'RMSE per prediction horizon', 'Prediction Horizon in minutes', 'Root mean squared error',
                                save_as[0])

        data_visuals.plot_error_per_horizons([mae_persistence] + tmae[i:i + max_models], predictions,
                                ['Persistence'] + names[i:i + max_models],
                                'MAE per prediction horizon', 'Prediction Horizon in minutes', 'Mean average error',
                                save_as[1])

        data_visuals.plot_error_per_horizons([mape_persistence] + tmape[i:i + max_models], predictions,
                                ['Persistence'] + names[i:i + max_models],
                                'MAPE per prediction horizon', 'Prediction Horizon in minutes', 'Mean average percentage error',
                                save_as[2])
        id = id + 1


def plot_day_multi(model, ph):
    t = data_helper.get_thesis_test_days()
    files, names = get_files_names(model)

    for idx, file in enumerate(files):
        if '5' in names[idx]:
            offset = 5
        if '10' in names[idx]:
            offset = 10
        if '20' in names[idx]:
            offset = 20


        actual, pred, times = data_helper.get_persistence_dates(t, 6, 19, ph, offset=offset)
        actual2, pred2, times2 = data_visuals.get_all_TP_multi(file)
        plt.plot(actual, linestyle='-', label='Actual')
        plt.plot(pred, linestyle=':', label='Persistence')
        plt.plot(pred2[ph-1], linestyle=':', label=names[idx])
        plt.legend()
        plt.show()
        plt.close()

def get_statistical_sig(model, ph):
    t = data_helper.get_thesis_test_days()
    files, names = get_files_names(model)

    # print(files)


    for idx, file in enumerate(files):

        print(names[idx])

        if '5' in names[idx]:
            offset = 5
        if '10' in names[idx]:
            offset = 10
        if '20' in names[idx]:
            offset = 20

        actual, pred, _ = data_helper.get_persistence_dates(t, 6, 19, ph, offset=offset)
        actual2, pred2, _ = data_visuals.get_all_TP_multi(file)

        # remove 0
        remove_list = []
        first = 0
        last = 0
        for idxs, val in enumerate(actual):
            if val == 0:
                first = idxs
            else:
                break

        for idxs, val in enumerate(reversed(actual)):
            if val == 0:
                last = idxs
            else:
                break

        last = len(actual) - last
        # print(actual[first:last])

        actual = actual[first:last]
        actual2[ph - 1] = actual2[ph-1][first:last]
        pred = pred[first:last]
        pred2[ph - 1] = pred2[ph-1][first:last]


        if len(actual) != len(actual2[ph-1]):
            print('LEN ERROR')
            print(len(actual))
            print(len(actual2[19]))

        if actual[0:10] != [int(x) for x in actual2[ph-1][0:10]]:
            print('SYNC ERROR')

        # print(names[idx])
        sig = Metrics.dm_test(actual, pred, pred2[ph-1], h=ph, crit="MSE", power=2)
        print(sig)

        # return sig, names[idx]

def normal_bar_plot(model, save_as='none'):
    days = data_helper.get_thesis_test_days()
    files, names = get_files_names(model)

    sunny = [(9, 15), (10, 15), (11, 15), (12, 15)]
    pcloudy = [(10, 21), (11, 17), (12, 16)]
    cloudy = [(10, 22), (12, 3)]

    for idx, file in enumerate(files):
        print(file)

        errors_sunny, errors_pcloudy, errors_cloudy = [[] for x in range(3)]
        perrors_sunny, perrors_pcloudy, perrors_cloudy = [[] for x in range(3)]

        for t in days:
            # get persistence errors:
            rmse_persistence, mae_persistence, mape_persistence = [[] for x in range(3)]
            # get persistence
            for i in range(0, 20):
                actual, pred, _ = data_helper.get_persistence_df(t[0], t[1], 7, 17, i + 1)
                rmse, mae, mape = Metrics.get_error(actual, pred)
                rmse_persistence.append(rmse)
                mae_persistence.append(mae)
                mape_persistence.append(mape)

            if t in sunny:
                perrors_sunny.append(rmse_persistence)
            elif t in pcloudy:
                perrors_pcloudy.append(rmse_persistence)
            elif t in cloudy:
                perrors_cloudy.append(rmse_persistence)

             #MODELS
            tmp_rmse = []
            tmp_mae = []
            tmp_mape = []
            actual, pred, _ = data_visuals.get_all_TP_multi(file, md_split=t)

            for i in range(0, 20):
                rmse, mae, mape = Metrics.get_error(actual[i], pred[i])
                tmp_rmse.append(rmse)
                tmp_mae.append(mae)
                tmp_mape.append(mape)
            if t in sunny:
                errors_sunny.append(tmp_rmse)
            elif t in pcloudy:
                errors_pcloudy.append(tmp_rmse)
            elif t in cloudy:
                errors_cloudy.append(tmp_rmse)

        labels = list(range(1, 21))

        r = np.arange(20)
        errors_sunny = avg_lol(errors_sunny)
        errors_pcloudy = avg_lol(errors_pcloudy)
        errors_cloudy = avg_lol(errors_cloudy)

        perrors_sunny = avg_lol(perrors_sunny)
        perrors_pcloudy = avg_lol(perrors_pcloudy)
        perrors_cloudy = avg_lol(perrors_cloudy)

        plt.bar(r, errors_sunny, width=0.15, label='Sunny ' + names[idx] )
        plt.bar(r+0.10, perrors_sunny, width=0.15, label='Sunny persistence', hatch=".")

        plt.bar(r + 0.3, errors_cloudy, width=0.15, label='Cloudy ' + names[idx])
        plt.bar(r + 0.4, perrors_cloudy, width=0.15, label='Cloudy persistence', hatch=".")

        plt.bar(r + 0.6, errors_pcloudy, width=0.15, label='Partially cloudy ' + names[idx])
        plt.bar(r + 0.7, perrors_pcloudy, width=0.15, label='Partially cloudy persistence', hatch=".")

        tmp_errors = [perrors_sunny, perrors_pcloudy, perrors_cloudy, errors_sunny, errors_pcloudy, errors_cloudy]
        tmp_names = ['Sunny Peristence', 'Partially cloudy Peristence', 'Cloudy Peristence', 'Sunny '+names[idx], 'Partially cloudy ' + names[idx], 'Cloudy '+ names[idx]]




        # data_visuals.plot_per_weather_circ(tmp_errors, r,
        #                                      tmp_names,
        #                                      'RMSE per prediction horizon', 'Prediction Horizon in minutes',
        #                                      'Error in RMSE',
        #                                      'none')

        plt.xticks(r, labels, fontweight='bold')
        plt.xlabel("Prediction horizon in minutes")
        plt.legend(title='Weather circumstances')

        plt.ylim(0, 100)
        plt.ylabel('Root mean squared error')
        plt.title('Average error per weather circumstance')


        save_name = model + '_bar_plot_' + str(idx) + '.jpg'
        if save_as != 'none':
            plt.savefig('final bar plots/' + save_name)

        else:
            plt.show()
        plt.close()

