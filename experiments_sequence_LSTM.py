from data.dataframe_sequence import DataFrameSequence
from metrics import Metrics
from models.models_ts import ann_model, lstm_model, svr_model
from keras import optimizers
from data.data import plot_history

init_epochs = 40
epochs = 40
start = 6
end = 19
prediction_horizons = list(range(1, 21))

res = []
sets = []
min_vals = []
min_loss = []

def run_lstm_experiments():
    sqs = [5, 10, 20]
    stages = [1,2]
    for st in stages:
        for s in sqs:
            for i in prediction_horizons:
                LSTM_experiment(i, s, 1, st)

def LSTM_experiment(prediction_horizon, minutes_sequence, cams, st):
    if st == 1:
        data = DataFrameSequence(False, prediction_horizon, False, True)
    if st == 2:
        data = DataFrameSequence(False, prediction_horizon, True, True)

    data.build_ts_df(start, end, [7,8,9,10,11,12], minutes_sequence, cams)
    data.normalize_mega_df()
    name_epoch = 'epochs_' + str(epochs)
    name_time = '_sqnc_' + str(minutes_sequence)
    name_cam = 'CAM_' + str(cams)
    name_stage = 'stg_' + str(st)
    name_pred = 'ph_' + str(prediction_horizon)
    lstm = lstm_model.LSTM_predictor(data, epochs, epochs, 'LSTM_SEQUENCE' + name_epoch + name_time + name_cam  + name_stage + name_pred)
    lstm.set_days(data.get_thesis_test_days())
    lstm.run_experiment()

def LSTM_test():
    data = DataFrameSequence(False, 20, True, True)
    data.build_ts_df(7, 19, [8,9], 10, 1, clear_sky_label=True)
    lstm = lstm_model.LSTM_predictor(data, 100, 50, 'LSTM_TEST', pred_csi=True)
    data.normalize_mega_df()
    data.split_data_set(9, 15)
    data.flatten_data_set_to_3d()
    lstm.get_model()
    lstm.train(40)
    y_pred, rmse, mae, mape = lstm.predict()
    plot_history('s1', 1, lstm.history)
    Metrics.write_results_SVR('LSTM_TEST 915', data.test_x_df, data.test_y_df, y_pred, data.pred_horizon)

def optimize():
    data = DataFrameSequence(False, 20, True, False)
    data.build_ts_df(6, 18, [7,8,9,10,11,12], 60, 1)
    data.normalize_mega_df()
    data.split_data_set(11,15)
    data.flatten_data_set_to_3d()

    seq_l = [3,5,10]
    nodes =  [(50,25,10),(60,30,15),(80,40,20)]
    activations = ['relu', 'sigmoid']
    opts = ['Adam', 'RMSprop']
    learning_rate = [0.001, 0.01, 0.1]

    lstm = lstm_model.LSTM_predictor(data, 3, 3, 'LSTM_TEST')
    num = 0
    for s in seq_l:
        data.build_ts_df(6, 18, [7, 8, 9, 10, 11, 12], s, 1)
        data.normalize_mega_df()
        data.split_data_set(11, 15)
        data.flatten_data_set_to_3d()
        for n in nodes:
            for a in activations:
                for o in opts:
                    for lr in learning_rate:

                        if o == 'Adam':
                            opt = optimizers.Adam(lr=lr)
                        else:
                            opt = optimizers.RMSprop(lr=lr)

                        lstm.set_model(n, a, opt)
                        out = lstm.train(100)
                        res.append(out)
                        settings = 'nodes: ' + str(n) + ' activation: ' + str(a) + ' optimizer: ' + str(o) + ' lr: ' + str(lr) + " seq_l: " + str(s)
                        sets.append(settings)
                        plot_history(settings, num)
                        min_loss.append(min(out.history['loss']))
                        min_vals.append(min(out.history['val_loss']))
                        num = num + 1

    best_val_loss = min_vals.index(min(min_vals))
    print('BEST VAL LOSS: ')
    print(sets[best_val_loss])
    print('val loss: ' + str(min(min_vals)))
    print('epoch: ')
    print(res[best_val_loss].history['val_loss'].index(min(res[best_val_loss].history['val_loss'])))

    best_loss = min_loss.index(min(min_loss))
    print('BEST Train LOSS: ')
    print(sets[best_loss])
    print('train loss: ' + str(min(min_loss)))
    print('epoch: ')
    print(res[best_loss].history['loss'].index(min(res[best_loss].history['loss'])))



# # LSTM_test()
# minutes_sequence = int(sys.argv[1])
# cams = int(sys.argv[2])
# img = int(sys.argv[3])
#
# if img == 1:
#     img = True
# else:
#     img = False
#
#
# print('Minutes sequence: ' + str(minutes_sequence))
# print('Cams: ' + str(cams))
# print('IMG: ' + str(img))
run_lstm_experiments()

