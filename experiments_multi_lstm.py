from data.dataframe_sequence_multi import DataFrameSequenceMulti
from metrics import Metrics
from models.models_ts_multi import lstm_model_multi
import threading
import sys
from keras import optimizers
from data.data_helper import plot_history

epochs = 50
start = 6
end = 19

res = []
sets = []
min_vals = []
min_loss = []

def run_final_all_days():
    data = DataFrameSequenceMulti(False, True, True, True)
    data.build_ts_df(start, end, [7, 8, 9, 10, 11, 12], 5)
    data.normalize_mega_df()

    name_time = '_sqnc_' + str(5)
    name_data = 'data_' + 'all'
    name_epoch = 'epochs_' + str(epochs)

    lstm = lstm_model_multi.LSTM_predictor(data, epochs,
                                           'LSTM_SEQUENCE_MULTI' + name_epoch + name_time + name_data)
    lstm.set_days(data.get_all_test_days())
    lstm.run_experiment()


def run_lstm_experiment():
    sqs = [3, 5, 10]
    # permutations = [(True, True, True), (True, False, False), (False, True, False), (False, False, True)]
    permutations = [(True, True, False)]
    # permutations_names = ['all data', 'onsite_only', 'img only', 'meteor only']
    permutations_names = ['onsite,img']
    for pidx, p in enumerate(permutations):
        for s in sqs:
            data = DataFrameSequenceMulti(False, p[0], p[1], p[2])
            data.build_ts_df(start, end, [7, 8, 9, 10, 11, 12], s)
            data.normalize_mega_df()

            name_time = '_sqnc_' + str(s)
            name_data = 'data_' + permutations_names[pidx]
            name_epoch = 'epochs_' + str(epochs)

            lstm = lstm_model_multi.LSTM_predictor(data, epochs,
                                            'LSTM_SEQUENCE_MULTI' + name_epoch + name_time + name_data)
            lstm.set_days(data.get_prem_days())
            lstm.run_experiment()


def LSTM_test():
    data = DataFrameSequenceMulti(False, True, True, True)
    data.build_ts_df(6, 19, [8,9,10], 10, 1)
    lstm = lstm_model_multi.LSTM_predictor(data, 100, 'LSTM_TEST')
    data.normalize_mega_df()
    data.split_data_set(9, 15)
    data.flatten_data_set_to_3d()
    lstm.get_model()
    lstm.train(50)
    y_pred, rmse, mae, mape = lstm.predict()
    plot_history('s1', 1, lstm.history)
    Metrics.write_results_multi('LSTM_TEST_MULTI', data.test_x_df.reshape(
        (data.test_x_df.shape[0],
         data.sequence_len_minutes,
         data.number_of_features)),
                                data.test_y_df, y_pred)

    print(rmse)


def optimize():
    data = lstm_model_multi(False, 20, True, False)
    data.build_ts_df(6, 18, [7,8,9,10,11,12], 60, 1)
    data.normalize_mega_df()
    data.split_data_set(11,15)
    data.flatten_data_set_to_3d()

    seq_l = [3,5,10]
    nodes =  [(50,25,10),(60,30,15),(80,40,20)]
    activations = ['relu', 'sigmoid']
    opts = ['Adam', 'RMSprop']
    learning_rate = [0.001, 0.01, 0.1]

    lstm = lstm_model_multi.LSTM_predictor(data, 3, 3, 'LSTM_TEST')
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


# run_lstm_experiment()
# LSTM_test()
run_final_all_days
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
# run_lstm_experiments()

