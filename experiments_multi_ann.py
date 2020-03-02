from data.dataframe_sequence_multi import DataFrameSequenceMulti
from metrics import Metrics
from models.models_ts_multi import ann_model_multi
import sys
from keras import optimizers
import threading
from data.data_helper import plot_history

init_epochs = 40
epochs = 40
start = 6
end = 19

res = []
sets = []
min_vals = []
min_loss = []
prediction_horizons = list(range(1, 21))

#BONUSSES/TODO
#forecast metoeor data
#data 24 hours ago
#

def run_ann_experiments():
    sqs = [20, 40, 60]
    permutations = [(True, True, True), (True, False, False), (False, True, False), (False, False, True)]
    permutations_names = ['all data', 'onsite_only', 'img only', 'meteor only']
    for pidx, p in enumerate(permutations):
        for s in sqs:
            data = DataFrameSequenceMulti(False, p[0], p[1], p[2])
            data.build_ts_df(start, end, [7, 8, 9, 10, 11, 12], s)
            data.normalize_mega_df()

            name_time = '_sqnc_' + str(s)
            name_data = 'data_' + permutations_names[pidx]
            name_epoch = 'epochs_' + str(epochs)

            ann = ann_model_multi.ANN_Multi(data, epochs, 'ANN_SEQUENCE_MULTI' + name_epoch + name_time + name_data)
            ann.set_days(data.get_prem_days())
            ann.run_experiment()

def ann_test():
    data = DataFrameSequenceMulti(False, True, True, True)
    data.build_ts_df(6, 19, [8,9], 20)
    data.normalize_mega_df()
    ann = ann_model_multi.ANN_Multi(data, 3, 3, 'ANN_BETA_SEQUENCE_MUTLI_TEST')
    data.split_data_set(9, 27)
    data.flatten_data_set()
    ann.get_model()
    ann.train(50)

    plot_history('s1',1, ann.history)
    y_pred, rmse, mae, mape = ann.predict()

    Metrics.write_results_multi('ANN_TEST_MULTI', data.test_x_df.reshape(
        (data.test_x_df.shape[0],
         data.sequence_len_minutes,
         data.number_of_features)),
                                data.test_y_df, y_pred)

    print(rmse)

def optimize():
    data = ann_model_multi(False, 20, False, False)
    data.build_ts_df(6, 18, [7,8,9,10,11,12], 60, 1)
    data.normalize_mega_df()
    data.split_data_set(11,15)
    data.flatten_data_set()

    nodes = [(50,10),(60,20), (40,20)]
    activations = ['relu']
    opts = ['Adam', 'RMSprop']
    drop_out = [0, 0.1, 0.5]
    learning_rate = [0.001, 0.01, 0.1]

    ann = ann_model_multi.ANN(data, 3, 3, 'ANN_BETA_SEQUENCE_TEST')
    num = 0
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
                        out = ann.train(10)
                        res.append(out)
                        settings = 'nodes: ' + str(n) + ' activation: ' + str(a) + ' optimizer: ' + str(o) + ' dropout: ' + str(d) + ' lr: ' + str(lr)
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


# minutes_sequence = int(sys.argv[1])
# cams = int(sys.argv[2])
# img = int(sys.argv[3])
#
# if img == 1:
#     img = True
# else:
#     img = False
#
# print('Minutes sequence: ' + str(minutes_sequence))
# print('cams: ' + str(cams))
# print('IMG: ' + str(img))

run_ann_experiments()
# ann_test()
# optimize()


