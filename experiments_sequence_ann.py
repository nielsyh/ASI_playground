from dataframe_sequence import DataFrameSequence
from metrics import Metrics
from models_ts import ann_model, lstm_model
import sys
from keras import optimizers

init_epochs = 200
epochs = 50
start = 6
end = 20

res = []
sets = []
min_vals = []
min_loss = []

def ann_experiment(minutes_sequence, cams):
    data = DataFrameSequence(False, 20,True, True)
    data.build_ts_df(start, end, [7, 8, 9, 10, 11, 12], minutes_sequence, cams)
    data.load_prev_mega_df()
    data.normalize_mega_df()
    name_cam = str(cams) + 'CAM_'
    name_time = str(minutes_sequence) + 'Minutes_'
    ann = ann_model.ANN(data, init_epochs, epochs, 'ANN_BETA_SEQUENCE_IMG' + name_cam + name_time)
    ann.run_experiment()

def ann_test():
    data = DataFrameSequence(True, 20, True, False)
    data.build_ts_df(5, 10, [9], 60, 1)
    data.normalize_mega_df()
    ann = ann_model.ANN(data, 3, 3, 'ANN_BETA_SEQUENCE_TEST')
    data.split_data_set(9, 27)
    data.flatten_data_set()
    ann.get_model()
    ann.train(3)
    y_pred, rmse, mae, mape = ann.predict()
    Metrics.write_results_NN('ANN_TEST', data.test_x_df.reshape(
        (data.test_x_df.shape[0], data.sequence_len_minutes, data.number_of_features)),
                              data.test_y_df, y_pred, data.pred_horizon)

def optimize():
    data = DataFrameSequence(False, 20, True, False)
    data.build_ts_df(6, 18, [7,8,9,10,11,12], 60, 1)
    data.normalize_mega_df()
    data.split_data_set(12, 1)
    data.flatten_data_set()

    # nodes = [(32,64,32),(64, 128, 64), (128, 256, 128)]
    nodes = [(64, 128, 64), (128, 256, 128)]
    # activations = ['relu', 'sigmoid']
    activations = ['relu']
    opts = ['Adam', 'RMSprop']
    drop_out = [0.1, 0]
    learning_rate = [0.001, 0.01, 0.1]

    ann = ann_model.ANN(data, 3, 3, 'ANN_BETA_SEQUENCE_TEST')

    for n in nodes:
        for a in activations:
            for o in opts:
                for d in drop_out:
                    for lr in learning_rate:

                        if o == 'Adam':
                            opt = optimizers.Adam(lr=lr)
                        else:
                            opt = optimizers.RMSprop(lr=lr)

                        ann.set_model(n, a, opt, d)
                        out = ann.train(40)
                        print(out)
                        res.append(out)
                        settings = 'nodes: ' + str(n) + ' activation: ' + str(a) + ' optimizer: ' + str(o) + ' dropout: ' + str(d) + ' lr: ' + str(lr)
                        sets.append(settings)
                        ann.plot_history(settings)
                        min_loss.append(min(out.history['loss']))
                        min_vals.append(min(out.history['val_loss']))


# minutes_sequence = int(sys.argv[1])
# cams = int(sys.argv[2])
# print('Minutes sequence: ' + str(minutes_sequence))
# print('Cams: ' + str(cams))
# ann_experiment(minutes_sequence, cams)
# # ann_test()


