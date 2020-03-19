from data.dataframe_sequence_multi import DataFrameSequenceMulti
from metrics import Metrics
from models.models_ts_multi import lstm_model_multi
import threading
import sys
from keras import optimizers
from data.data_helper import plot_history

epochs = 100
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
                                           'LSTM_SEQUENCE_MULTI alldays' + name_epoch + name_time + name_data)
    lstm.set_days(data.get_all_test_days())
    lstm.run_experiment()

def run_final_test_days():
    sqs = [5, 10]
    cams = [1,2]
    permutations = [(True, True, True), (True, False, False), (False, True, False)]
    permutations_names = ['all data', 'onsite_only', 'img only']

    for pidx, p in enumerate(permutations):
        for s in sqs:
            for c in cams:
                data = DataFrameSequenceMulti(False, p[0], p[1], p[2])
                data.build_ts_df(start, end, [7, 8, 9, 10, 11, 12], s, cams=c, clear_sky_label=False)
                data.normalize_mega_df()

                name_time = '_sqnc_' + str(s)
                name_data = 'data_' + permutations_names[pidx]
                name_epoch = '_epochs_' + str(epochs)
                name_cam = '_cams_' + str(c)

                lstm = lstm_model_multi.LSTM_predictor(data, epochs,
                                                'LSTM_MULTI_TESTSET gradient' + name_epoch + name_time + name_data + name_cam, pred_csi=False)
                lstm.set_days(data.get_thesis_test_days())
                lstm.run_experiment()


def run_lstm_experiment():
    sqs = [3,5,10]
    cams = [1,2]
    permutations = [(True, True, True), (True, False, False), (False, True, False), (False, False, True)]
    # permutations = [(True, True, True)]
    permutations_names = ['all data', 'onsite_only', 'img only', 'meteor only']
    # permutations_names = ['all data clrsky']

    for pidx, p in enumerate(permutations):
        for s in sqs:
            for c in cams:
                data = DataFrameSequenceMulti(False, p[0], p[1], p[2])
                data.build_ts_df(start, end, [7, 8, 9, 10, 11, 12], s, cams=c, clear_sky_label=False)
                data.normalize_mega_df()

                name_time = '_sqnc_' + str(s)
                name_data = 'data_' + permutations_names[pidx]
                name_epoch = '_epochs_' + str(epochs)
                name_cam = '_cams_' + str(c)

                lstm = lstm_model_multi.LSTM_predictor(data, epochs,
                                                'LSTM_SEQUENCE_MULTI' + name_epoch + name_time + name_data + name_cam, pred_csi=False)
                lstm.set_days(data.get_prem_days())
                lstm.run_experiment()


def LSTM_test():
    data = DataFrameSequenceMulti(False, True, True, True, gradients=True)
    data.build_ts_df(6, 19, [8,9,10], 5)
    lstm = lstm_model_multi.LSTM_predictor(data, 100, 'LSTM_TEST')
    data.normalize_mega_df()
    data.split_data_set(9, 27)
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
    # data.build_ts_df(6, 19, [8, 9, 10,11,12], 10, cams=1, clear_sky_label=False)
    # data.normalize_mega_df()
    # data.split_data_set(10,15)
    # data.flatten_data_set_to_3d()
    #
    # seq_l = [3,5,10]
    # nodes =  [(50,25,10),(60,30,15),(80,40,20)]
    # activations = ['relu', 'sigmoid']
    # opts = ['Adam', 'RMSprop']
    # learning_rate = [0.001, 0.01, 0.1]


    seq_l = [5]
    nodes =  [(50,25,10)]
    activations = ['relu']
    opts = ['Adam']
    learning_rate = [0.001]

    data = DataFrameSequenceMulti(False, True, True, True)
    lstm = lstm_model_multi.LSTM_predictor(data, 50, 'LSTM_TEST')
    num = 0
    for s in seq_l:
        data.build_ts_df(6, 19, [7,8,9,10,11,12], s, 1)
        data.normalize_mega_df()
        data.split_data_set(10, 15)
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
                        plot_history(settings, num, out)
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


# run_final_test_days()
# LSTM_test()
# optimize()
# run_lstm_experiment()
run_final_test_days()
# LSTM_test()