from dataframe_sequence import DataFrameSequence
from metrics import Metrics
from models_ts import ann_model, lstm_model, svr_model
import threading
import sys
from keras import optimizers

init_epochs = 100
epochs = 50
start = 6
end = 20
prediction_horizons = list(range(1, 21))

res = []
sets = []
min_vals = []
min_loss = []

def LSTM_experiment(minutes_sequence, cams):
    data = DataFrameSequence(False, 20, True, False)
    data.build_ts_df(start ,end, [7,8,9,10,11,12], minutes_sequence, cams)
    name_cam = str(cams) + 'CAM_'
    name_time = str(minutes_sequence) + 'Minutes_'
    lstm = lstm_model.LSTM_predictor(data, 100, 50, 'LSTM_BETA_SEQUENCE_' + str(name_cam) + str(name_time))
    lstm.run_experiment()

def LSTM_test():
    data = DataFrameSequence(True, 20, True, False)
    data.build_ts_df(6, 12, [9], 60, 1)
    lstm = lstm_model.LSTM_predictor(data, 100, 50, 'LSTM_TEST')
    data.normalize_mega_df()
    data.split_data_set(9, 27)
    data.flatten_data_set_to_3d()
    lstm.get_model()
    lstm.train(3)
    y_pred, rmse, mae, mape = lstm.predict()
    Metrics.write_results_NN('LSTM_TEST', data.test_x_df, data.test_y_df, y_pred, data.pred_horizon)

def optimize():
    data = DataFrameSequence(False, 20, True, False)
    data.build_ts_df(6, 18, [7,8,9,10,11,12], 60, 1)
    data.normalize_mega_df()
    data.split_data_set(11,15)
    data.flatten_data_set_to_3d()

    nodes =  [(50,25),(60,30),(80,40)]
    activations = ['relu', 'sigmoid']
    opts = ['Adam', 'RMSprop']
    learning_rate = [0.001, 0.01, 0.1]

    lstm = lstm_model.LSTM_predictor(data, 3, 3, 'LSTM_TEST')
    num = 0
    for n in nodes:
        for a in activations:
            for o in opts:
                for lr in learning_rate:

                    if o == 'Adam':
                        opt = optimizers.Adam(lr=lr)
                    else:
                        opt = optimizers.RMSprop(lr=lr)

                    lstm.set_model(n, a, opt)
                    out = lstm.train(3)
                    res.append(out)
                    settings = 'nodes: ' + str(n) + ' activation: ' + str(a) + ' optimizer: ' + str(o) + ' lr: ' + str(lr)
                    sets.append(settings)
                    lstm.plot_history(settings, num)
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

# minutes_sequence = int(sys.argv[1])
# cams = int(sys.argv[2])
# print('Minutes sequence: ' + str(minutes_sequence))
# print('Cams: ' + str(cams))
# LSTM_experiment(minutes_sequence, cams)
optimize()

# LSTM_test()
