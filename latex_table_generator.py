import data.data_helper
import data.data_visuals
import metrics

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
        result.append(round(1 - (res[i] / res[0]),4))

    return result


def print_table(model_names, rmse, mae, mape, ss_rmse, ss_mae, ss_mape, caption, label):
    print('\\begin{table}[]')
    print('\\resizebox{\\textwidth}{!}{%')
    print('\\begin{tabular}{|l|l|l|l|l|l|l|}')
    print('\hline')


    print('\\textbf{Model} & \\textbf{RMSE} \\downarrow     & \\textbf{MAE} \\downarrow      & \\textbf{MAPE} \\downarrow   & \\textbf{SS-RMSE} \\uparrow  & \\textbf{SS-MAE} \\uparrow    & \\textbf{SS-MAPE} \\uparrow    \\\\ \\hline')

    for idx, name in enumerate(model_names):
        print(name + ' & ' +str(rmse[idx]) + ' & ' + str(mae[idx]) + ' & ' + str(mape[idx]) + ' & ' + str(ss_rmse[idx]) + ' & ' + str(ss_mae[idx]) + ' & ' + str(ss_mape[idx]) + '    \\\\ \\hline' )

    print('\\end{tabular}%')
    print('}')
    print('\\caption{ ' + str(caption) + '}')
    print('\\label{tab:'+str(label)+'}')
    print('\\end{table}')

folders_ann = ['prem results/ANN 5 IMG/ANN_SEQUENCE_epochs_40_sequence_5CAM_1_img_Truepredhor_',
           'prem results/ANN 5 NOIMG/ANN_SEQUENCE_epochs_40_sequence_5CAM_1_img_Falsepredhor_',
           'prem results/ANN 10 IMG/ANN_SEQUENCE_epochs_40_sequence_10CAM_1_img_Truepredhor_',
           'prem results/ANN 20 IMG/ANN_SEQUENCE_epochs_40_sequence_20CAM_1_img_Truepredhor_',
           'prem results/ANN 20 NOIMG/ANN_SEQUENCE_epochs_40_sequence_20CAM_1_img_Falsepredhor_',
           'prem results/ANN 50 IMG/ANN_SEQUENCE_epochs_40_sequence_50CAM_1_img_Truepredhor_',
           'prem results/ANN 60 NOIMG NOMETEOR/ANN_SEQUENCE_NOMETEORepochs_40_sequence_60CAM_1_img_Falsepredhor_'
               ]

folders_rf = ['prem results/RF 5 IMG/RF SEQUENCE PREM__sequence_5CAM_1_img_Truepredhor_',
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
          'prem results/RF 120 NOIMG/RF SEQUENCE PREM__sequence_120CAM_1_img_Falsepredhor_'
              ]

folders_lstm = ['prem results/LSTM 5 IMG/LSTM_SEQUENCE_epochs_40_sequence_5CAM_1_img_Truepredhor_',
           'prem results/LSTM 10 IMG/LSTM_SEQUENCE_epochs_40_sequence_10CAM_1_img_Truepredhor_',
            'prem results/LSTM 10 NOIMG/LSTM_BETA_SEQUENCE_epochs_40CAM_1_sequence_10predhor_',
           'prem results/LSTM 20 IMG/LSTM_SEQUENCE_epochs_40_sequence_20CAM_1_img_Truepredhor_',
           'prem results/LSTM 20 NOIMG/LSTM_SEQUENCE_epochs_40_sequence_20CAM_1_img_Falsepredhor_']

armse = []
amae = []
amape = []
ass_rmse = ['NA']
ass_mae = ['NA']
ass_mape = ['NA']
all_model_names = []
# for i in prediction_horizons:
prediction_horizons = list(range(1,21))
for i in prediction_horizons:
    model_names = []
    rmse = []
    mae = []
    mape = []
    ss_rmse = ['NA']
    ss_mae = ['NA']
    ss_mape = ['NA']

    t = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]
    actual, pred = data.data_helper.get_persistence_dates(t, 7, 19, i)
    trmse, tmae, tmape = metrics.Metrics.get_error(actual, pred)
    model_names.append("Persistence")
    rmse.append(round(trmse, 4))
    mae.append(round(tmae, 4))
    mape.append(round(tmape, 4))


    for folder in folders_lstm + folders_rf + folders_ann:
        extension = '.txt'
        file = folder + str(i) + extension
        predicted, actual = data.data_visuals.file_to_values(file)
        trmse, tmae, tmape = metrics.Metrics.get_error(actual, predicted)
        modelname = folder[folder.find('/')+1:-1][0:(folder[folder.find('/')+1:-1]).find('/')]

        model_names.append(modelname)
        rmse.append(round(trmse, 2))
        mae.append(round(tmae, 2))
        mape.append(round(tmape, 2))

        ss_rmse.append(round(1 - (trmse / rmse[0]), 2))
        ss_mae.append(round(1 - (tmae / mae[0]), 2))
        ss_mape.append(round(1 - (tmape / mape[0]),2))

    # print('L: '  + str(len(model_names) + 1))
    # print('W: 7' )

    # print_table(model_names, rmse, mae, mape, ss_rmse, ss_mae, ss_mape, 'Performance evaluation Prem. days with prediction horizon: ' + str(i), 'prem.' + str(i))

    armse.append(rmse)
    amae.append(mae)
    amape.append(mape)
    ass_rmse.append(ss_rmse)
    ass_mae.append(ss_mae)
    ass_mape.append(ss_mape)
    all_model_names = model_names

print_table( all_model_names, round_list(avg_res(armse)), round_list(avg_res(amae)),
             round_list(avg_res(amape)), calc_ss(avg_res(armse)), calc_ss(avg_res(amae)),
             calc_ss(avg_res(amape)), 'Average performance evaluation Prem. days', 'prem.avg')