import data.data_helper
import data.data_visuals
import data.test_visuals
import metrics
import data.test_visuals
from data import data_helper


def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1

def round_s(flt, rnd):
    res = round(flt, rnd)
    if res == 0.0:
        return 'n/a'
    else:
        return res

def avg_res(res):
    return list(map(sum, zip(*res)))

def round_list(ls):
    avgls = [n/20 for n in ls]
    l = []
    for i in avgls:
        l.append(round(i, 2))
    return l
def calc_ss(res):
    result = ['NA']
    for i in range(1, len(res)):
        result.append(round(1 - (res[i] / res[0]),2))

    return result


def print_table(model_names, rmse, mae, mape, ss_rmse, ss_mae, ss_mape, caption, label):
    print('\\begin{table}[!htb]')
    print('\\resizebox{\\textwidth}{!}{%')
    print('\\begin{tabular}{l|llllll}')

    print('\\textbf{Model} & \\textbf{RMSE} $\\downarrow$     & \\textbf{MAE} $\\downarrow$      & \\textbf{MAPE} $\\downarrow$   & \\textbf{SS-RMSE} $\\uparrow$  & \\textbf{SS-MAE} $\\uparrow$    & \\textbf{SS-MAPE} $\\uparrow$   \\\\ ')

    for idx, name in enumerate(model_names):
        print(listToString(name + ' & ' +str(rmse[idx]) + ' & ' + str(mae[idx]) + ' & ' + str(mape[idx]) + ' & ' + str(ss_rmse[idx]) + ' & ' + str(ss_mae[idx]) + ' & ' + str(ss_mape[idx]) + '    \\\\ ' ))

    print('\\end{tabular}%')
    print('}')
    print('\\caption{ ' + str(caption) + '}')
    print('\\label{tab:'+str(label)+'}')
    print('\\end{table}')


def print_sig_table(model):
    prediction_horizons = list(range(1, 21))
    # prediction_horizons = [1,5,15,20]

    sigs, names = data.test_visuals.get_statistical_sig(model, 20)
    columns = listToString(['l|'] + ['l' for x in range(len(names))])

    days_cloudy = data_helper.get_thesis_test_days(in_cloudy=True, in_parcloudy=False, in_sunny=False)
    days_pcloudy = data_helper.get_thesis_test_days(in_cloudy=False, in_parcloudy=True, in_sunny=False)
    days_sunny = data_helper.get_thesis_test_days(in_cloudy=False, in_parcloudy=False, in_sunny=True)
    weather_circumstances = [days_sunny, days_pcloudy]
    weather_names = ['sunny', 'partially cloudy']

    # weather_circumstances = [days_sunny]
    # weather_names = ['sunny']

    for weather_idx, weather in enumerate(weather_circumstances):

        print('\\begin{table}[!htb]')
        print('\\resizebox{\\textwidth}{!}{%')
        print('\\begin{tabular}{' + str(columns) + '}')
        # print('\hline')

        next_line = listToString(['\\textbf{Prediction horizon} &'] + ['\\textbf{' + names[x] + '}' if (x == len(names) -1) else '\\textbf{' + names[x] + '} &' for x in range(len(names))] + ['\\\\'])

        # print('\\textbf{Model} & \\textbf{RMSE} \\downarrow     & \\textbf{MAE} \\downarrow      & \\textbf{MAPE} \\downarrow   & \\textbf{SS-RMSE} \\uparrow  & \\textbf{SS-MAE} \\uparrow    & \\textbf{SS-MAPE} \\uparrow    \\\\ \\hline')
        print(str(next_line))

        # print start table

        for i in prediction_horizons:
            sigs, names = data.test_visuals.get_statistical_sig(model, i, days=weather)
            next_line = listToString([str(i) + '&']+[str(sigs[x]) + '    \\\\' if (x == len(sigs) - 1) else str(sigs[x]) + ' & ' for x in
                         range(len(sigs))])
            print(str(next_line))

        caption = str(model).upper() + ': Diabold-Mariono P value per horizon, compared with baseline model Persistence. Weather circumstance ' + weather_names[weather_idx] + '. (n/a implies competing model is not performing better than the baseline).'
        label = 'tab:' + str(model).upper() + '.' + str(weather_names[weather_idx])

        print('\\end{tabular}%')
        print('}')
        print('\\caption{' + str(caption) + '}')
        print('\\label{' + str(label) + '}')
        print('\\end{table}')
    #     print end table


def print_results_val(model, set='val'):
    prediction_horizons = list(range(1,21))
    # prediction_horizons = [20]
    armse = []
    amae = []
    amape = []
    ass_rmse = ['NA']
    ass_mae = ['NA']
    ass_mape = ['NA']
    all_model_names = []
    # prediction_horizons = [20]

    if set == 'val':
        t = [[(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]]
        if model == 'ann':
            files, names = data.data_helper.get_files_ann_multi()
            folders, names_ = data.data_helper.get_folders_ann()
        elif model == 'rf':
            files, names = data.data_helper.get_files_rf_multi()
            folders, names_ = data.data_helper.get_folders_rf()
        elif model == 'lstm':
            files, names = data.data_helper.get_files_lstm_multi()
            folders, names_ = data.data_helper.get_folders_lstm()
    else:
        # t = data.data_helper.get_thesis_test_days(include_cloudy=False)
        files, names = data.test_visuals.get_files_names(model)

    days_cloudy = data_helper.get_thesis_test_days(in_cloudy=True, in_parcloudy=False, in_sunny=False)
    days_pcloudy = data_helper.get_thesis_test_days(in_cloudy=False, in_parcloudy=True, in_sunny=False)
    days_sunny = data_helper.get_thesis_test_days(in_cloudy=False, in_parcloudy=False, in_sunny=True)
    weather_circumstances = [days_sunny, days_pcloudy, days_cloudy]
    weather_names = ['sunny', 'partially cloudy', 'cloudy']

    for weather_idx, t in enumerate(weather_circumstances):
        for i in prediction_horizons:
            model_names = []
            rmse = []
            mae = []
            mape = []
            ss_rmse = ['NA']
            ss_mae = ['NA']
            ss_mape = ['NA']

            # Persistence results
            # actual, pred, _ = data_helper.get_persistence_dates(t, 7, 17, i + 1)
            actual, pred, _ = data.data_helper.get_persistence_dates(t, 6, 19, i+1)


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
            actual = actual[first:last]
            pred = pred[first:last]

            trmse, tmae, tmape = metrics.Metrics.get_error(actual, pred)
            model_names.append("Persistence")
            rmse.append(round(trmse, 2))
            mae.append(round(tmae, 2))
            mape.append(round(tmape, 2))

            if set == 'val':
                for idx, folder in enumerate(folders):
                    extension = '.txt'
                    file = folder + str(i) + extension
                    predicted, actual =  data.data_visuals.file_to_values(file)
                    trmse, tmae, tmape = metrics.Metrics.get_error(actual, predicted)
                    modelname = names_[idx]

                    model_names.append(modelname)
                    rmse.append(round(trmse, 2))
                    mae.append(round(tmae, 2))
                    mape.append(round(tmape, 2))
                    ss_rmse.append(round(1 - (trmse / rmse[0]), 2))
                    ss_mae.append(round(1 - (tmae / mae[0]), 2))
                    ss_mape.append(round(1 - (tmape / mape[0]), 2))

            for idx, file in enumerate(files):
                predicted, actual = data.data_visuals.file_to_values(file, i)
                predicted, actual, _ = data.data_visuals.get_all_TP_multi(file, md_list_split=t)
                trmse, tmae, tmape = metrics.Metrics.get_error(actual[i-1], predicted[i-1])
                modelname = names[idx]

                model_names.append(modelname)
                rmse.append(round(trmse, 2))
                mae.append(round(tmae, 2))
                mape.append(round(tmape, 2))
                ss_rmse.append(round(1 - (trmse / rmse[0]), 2))
                ss_mae.append(round(1 - (tmae / mae[0]), 2))
                ss_mape.append(round(1 - (tmape / mape[0]), 2))

            if i == 20:
                print_table(model_names, rmse, mae, mape, ss_rmse, ss_mae, ss_mape, 'Performance evaluation test set (' + weather_names[weather_idx] + ') with prediction horizon: ' + str(i), 'prem.' + str(i))

            armse.append(rmse)
            amae.append(mae)
            amape.append(mape)
            ass_rmse.append(ss_rmse)
            ass_mae.append(ss_mae)
            ass_mape.append(ss_mape)
            all_model_names = model_names

        print_table( all_model_names, round_list(avg_res(armse)), round_list(avg_res(amae)),
                     round_list(avg_res(amape)), calc_ss(avg_res(armse)), calc_ss(avg_res(amae)),
                     calc_ss(avg_res(amape)), str(model) + '. Average performance evaluation test set (' + weather_names[weather_idx] + ').', str(model) + '.test.avg.' + str(weather_names[weather_idx]))


# print_results(folders_rf + folders_rf_multi)
# print_results(folders_ann + folders_ann_multi)
# print_results(folders_lstm + folders_lstm_multi)