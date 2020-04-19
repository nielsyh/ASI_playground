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

def round_list(ls, div=True):

    if div:
        avgls = [n/20 for n in ls]
    else:
        avgls = ls

    l = []
    for i in avgls:
        l.append(round(i, 2))
    return l

def calc_ss(res):
    result = ['NA','NA']
    for i in range(2, len(res)):
        result.append(round(1 - (res[i] / res[0]),2))

    return result

def format_results(res_list, pref='small'):
    if pref == 'small':
        val = min(res_list[2:])
    else:
        val = max(res_list[2:])

    for i in range(0, len(res_list)):
        if res_list[i] == val:
            res_list[i] = '\\textbf{' + str(res_list[i]) + '}'

    return res_list


def print_table(model_names, rmse, mae, mape, ramp, ss_rmse, ss_mae, ss_mape, ss_ramp, caption, label):

    rmse, mae, mape, ramp = format_results(rmse), format_results(mae), format_results(mape), format_results(ramp)
    ss_rmse, ss_mae, ss_mape, ss_ramp = format_results(ss_rmse, pref='big'), format_results(ss_mae, pref='big'), format_results(ss_mape, pref='big'), format_results(ss_ramp, pref='big')

    print('\\begin{table}[!htb]')
    print('\\resizebox{\\textwidth}{!}{%')
    print('\\begin{tabular}{l|llllllll}')

    print('\\textbf{Model} & \\textbf{RMSE} $\\downarrow$     & \\textbf{MAE} $\\downarrow$      & \\textbf{MAPE} $\\downarrow$ & \\textbf{Ramp-score} $\\downarrow$   & \\textbf{SS-RMSE} $\\uparrow$  & \\textbf{SS-MAE} $\\uparrow$    & \\textbf{SS-MAPE} $\\uparrow$ & \\textbf{SS-RAMP} $\\uparrow$   \\\\ ')

    for idx, name in enumerate(model_names):
        print(listToString(name + ' & ' + str(rmse[idx]) + ' & ' + str(mae[idx]) + ' & ' + str(mape[idx]) + ' & ' + str(ramp[idx]) + ' & ' + str(ss_rmse[idx]) + ' & ' + str(ss_mae[idx]) + ' & ' + str(ss_mape[idx]) + ' & ' + str(ss_ramp[idx]) + '   \\\\ ' ))

    print('\\end{tabular}%')
    print('}')
    print('\\caption{' + str(caption) + '}')
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

        next_line = listToString(['\\textbf{Prediction horizon} &'] + ['\\textbf{' + names[x] + '}' if (x == len(names) -1) else '\\textbf{' + names[x] + '} &' for x in range(len(names))] + ['\\\\'])

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


def print_results_val(model, set='val', default_hor=True):

    if default_hor:
        prediction_horizons = list(range(1,21))
    else:
        prediction_horizons = [20]

    if set == 'val':
        t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]
        weather_circumstances = [t]
        weather_names = ['All weather']
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

        armse, amae, amape, aramp, all_model_names = [[] for x in range(5)]
        ass_rmse, ass_mae, ass_mape, ass_ramp = [['NA','NA'] for x in range(4)]

        for i in prediction_horizons:
            model_names, rmse, mae, mape, ramp = [[] for x in range(5)]
            ss_rmse, ss_mae, ss_mape, ss_ramp = [['NA','NA'] for x in range(4)]

            # Persistence results
            actual, pred, _ = data.data_helper.get_persistence_dates(t, 6, 19, i)
            sactual, spred, _ = data.data_helper.get_smart_persistence_dates(t, 6, 19, i)

            actual, pred = data_helper.filter_0_list(actual,pred)
            sactual, spred = data_helper.filter_0_list(sactual, spred)

            #persistence
            trmse, tmae, tmape, tramp = metrics.Metrics.get_error(actual, pred)
            model_names.append("Persistence")
            rmse.append(round(trmse, 2))
            mae.append(round(tmae, 2))
            mape.append(round(tmape, 2))
            ramp.append(round(tramp, 2))

            #smart-persistence
            trmse, tmae, tmape, tramp = metrics.Metrics.get_error(sactual, spred)
            model_names.append("Smart-persistence")
            rmse.append(round(trmse, 2))
            mae.append(round(tmae, 2))
            mape.append(round(tmape, 2))
            ramp.append(round(tramp, 2))

            if set == 'val':
                for idx, folder in enumerate(folders):
                    extension = '.txt'
                    file = folder + str(i) + extension
                    predicted, actual =  data.data_visuals.file_to_values(file)
                    trmse, tmae, tmape, tramp = metrics.Metrics.get_error(actual, predicted)
                    modelname = names_[idx]

                    model_names.append(modelname)
                    rmse.append(round(trmse, 2))
                    mae.append(round(tmae, 2))
                    mape.append(round(tmape, 2))
                    ramp.append(round(tramp, 2))

                    ss_rmse.append(round(1 - (trmse / rmse[0]), 2))
                    ss_mae.append(round(1 - (tmae / mae[0]), 2))
                    ss_mape.append(round(1 - (tmape / mape[0]), 2))
                    ss_ramp.append(round(1 - (tramp / ramp[0]), 2))

            for idx, file in enumerate(files):
                predicted, actual, _ = data.data_visuals.get_all_TP_multi(file, md_list_split=t)
                actual, predicted = data_helper.filter_0_list_LS(actual, predicted)
                trmse, tmae, tmape, tramp = metrics.Metrics.get_error(actual[i-1], predicted[i-1])
                modelname = names[idx]

                model_names.append(modelname)
                rmse.append(round(trmse, 2))
                mae.append(round(tmae, 2))
                mape.append(round(tmape, 2))
                ramp.append(round(tramp, 2))
                ss_rmse.append(round(1 - (trmse / rmse[0]), 2))
                ss_mae.append(round(1 - (tmae / mae[0]), 2))
                ss_mape.append(round(1 - (tmape / mape[0]), 2))
                ss_ramp.append(round(1 - (tramp / ramp[0]), 2))

            # if i == 20:
            #     print_table(model_names, rmse, mae, mape, ramp, ss_rmse, ss_mae, ss_mape, ss_ramp, 'Performance evaluation test set, for weather circumstance' + weather_names[weather_idx] + ', with prediction horizon: ' + str(i), 'prem.' + str(i))

            armse.append(rmse)
            amae.append(mae)
            amape.append(mape)
            aramp.append(ramp)
            ass_rmse.append(ss_rmse)
            ass_mae.append(ss_mae)
            ass_mape.append(ss_mape)
            ass_ramp.append(ss_ramp)
            all_model_names = model_names

        if default_hor:
            title = str(model).upper() + '. Average performance on test-set, with weather circumstance ' + weather_names[weather_idx] + '.'
            label =  str(model).upper() + '.test.avg.' + str(weather_names[weather_idx])

            print_table(all_model_names, round_list(avg_res(armse)), round_list(avg_res(amae)),
                        round_list(avg_res(amape)), round_list(avg_res(aramp)), calc_ss(avg_res(armse)),
                        calc_ss(avg_res(amae)),
                        calc_ss(avg_res(amape)), calc_ss(avg_res(aramp)), title, label)
        else:
            title = str(model).upper() + '. Average performance on test-set with prediction horizon 20, with weather circumstance ' + weather_names[weather_idx] + '.'
            label =  str(model).upper() + '.test.avg.20' + str(weather_names[weather_idx])

            print_table(all_model_names, round_list(avg_res(armse), div=False), round_list(avg_res(amae), div=False),
                        round_list(avg_res(amape), div=False), round_list(avg_res(aramp), div=False), calc_ss(avg_res(armse)),
                        calc_ss(avg_res(amae)),
                        calc_ss(avg_res(amape)), calc_ss(avg_res(aramp)), title, label)



