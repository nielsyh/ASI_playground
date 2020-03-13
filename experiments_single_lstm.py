from data.dataframe_sequence import DataFrameSequence
from metrics import Metrics
from models.models_ts import lstm_model
from keras import optimizers
from data.data_helper import plot_history

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
    permutations = [(True, True, True), (True, False, False), (False, True, False), (False, False, True)]
    permutations_names = ['all data', 'onsite_only', 'img only', 'meteor only']
    for idx, p in enumerate(permutations):
        for s in sqs:
            for i in prediction_horizons:
                LSTM_experiment(i, s, p, permutations_names[idx])

def LSTM_experiment(prediction_horizon, minutes_sequence, data, data_name):
    data = DataFrameSequence(False, prediction_horizon, data[0], data[1], data[2])
    data.build_ts_df(start, end, [7,8,9,10,11,12], minutes_sequence)
    data.normalize_mega_df()

    name_epoch = 'epochs_' + str(epochs)
    name_time = '_sqnc_' + str(minutes_sequence)
    name_data = 'data_' + str(data_name)
    name_pred = 'ph_' + str(prediction_horizon)

    lstm = lstm_model.LSTM_predictor(data, epochs, epochs, 'LSTM_SEQUENCE' + name_epoch + name_time + name_data + name_pred)
    lstm.set_days(data.get_prem_days())
    lstm.run_experiment()

def LSTM_test():
    # debug, pred_horizon, onsite_data, img_data, meteor_data
    data = DataFrameSequence(False, 20, True, True, True)
    # start, end, months, lenth_tm
    data.build_ts_df(12, 15, [8,9], 5)
    lstm = lstm_model.LSTM_predictor(data, 50, 50, 'LSTM_TEST')
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


run_lstm_experiments()