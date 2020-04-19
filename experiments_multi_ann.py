from data.dataframe_sequence_multi import DataFrameSequenceMulti
from metrics import Metrics
from models.models_ts_multi import ann_model_multi
import sys
from keras import optimizers
import threading
from data.data_helper import plot_history

# init_epochs = 40
epochs = 100 #36
start = 6
end = 19

res = []
sets = []
min_vals = []
min_loss = []
prediction_horizons = list(range(1, 21))


def run_final_all_days():
    # onsite
    data = DataFrameSequenceMulti(False, True, False, False)
    data.build_ts_df(start, end, [7, 8, 9, 10, 11, 12], 5)
    data.scale_mega(model='ann')

    name_time = '_sqnc_' + str(5)
    name_data = 'data_' + 'all'
    name_epoch = 'epochs_' + str(epochs)

    ann = ann_model_multi.ANN_Multi(data, epochs,
                                           'ANN_SEQUENCE_MULTI_alldays' + name_epoch + name_time + name_data)
    ann.set_days(data.get_all_test_days())
    ann.run_experiment()

def run_final_test_days():
    sqs = [10, 20, 40, 60]
    cams = [1,2]
    permutations = [(True, True, True), (True, False, False), (False, True, False)]
    permutations_names = ['all data', 'onsite_only', 'img only', 'meteor only']
    for pidx, p in enumerate(permutations):
        for s in sqs:
            for c in cams:
                data = DataFrameSequenceMulti(False, p[0], p[1], p[2])
                data.build_ts_df(start, end, [7, 8, 9, 10, 11, 12], s, cams=c)
                # data.normalize_mega_df()
                data.scale_mega(model='ann')

                name_time = '_sqnc_' + str(s)
                name_data = 'data_' + permutations_names[pidx]
                name_epoch = 'epochs_' + str(epochs)
                name_cam = '_cams_' + str(c)

                ann = ann_model_multi.ANN_Multi(data, epochs, 'ANN MUTLI TSET grad' + name_time + name_data + name_cam )
                ann.set_days(data.get_thesis_test_days())
                ann.run_experiment()

def run_ann_experiments():
    sqs = [5,10,20, 40, 60]
    cams = [1,2]
    permutations = [(True, True, True), (True, False, False), (False, True, False), (False, False, True)]
    permutations_names = ['all data', 'onsite_only', 'img only', 'meteor only']
    for pidx, p in enumerate(permutations):
        for s in sqs:
            for c in cams:
                data = DataFrameSequenceMulti(False, p[0], p[1], p[2])
                data.build_ts_df(start, end, [7, 8, 9, 10, 11, 12], s, cams=c)
                data.scale_mega('ann')

                name_time = '_sqnc_' + str(s)
                name_data = 'data_' + permutations_names[pidx]
                name_epoch = 'epochs_' + str(epochs)
                name_cam = '_cams_' + str(c)

                ann = ann_model_multi.ANN_Multi(data, epochs, 'ANN_MULTI_PREM' + name_epoch + name_time + name_data + name_cam )
                ann.set_days(data.get_prem_days())
                ann.run_experiment()

def ann_test():
    data = DataFrameSequenceMulti(False, True, False, False)
    data.build_ts_df(6, 19, [7,8,9,10,11,12], 10, cams=1)
    # data.normalize_mega_EXPRTML(norm=True)
    data.split_data_set_EXPRMTL(12, 15, 10)
    data.scale_mega(model='ann')
    ann = ann_model_multi.ANN_Multi(data, 100, 'ANN_BETA_SEQUENCE_MUTLI_TEST')
    data.flatten_data_set()
    ann.get_model()
    ann.train(100)

    plot_history('s1',1, ann.history)
    y_pred, rmse = ann.predict()


    import matplotlib.pyplot as plt
    from matplotlib.lines import lineStyles
    plt.plot(ann.history.history['loss'])
    plt.plot(ann.history.history['val_loss'], linestyle=':')
    ymin = min(ann.history.history['val_loss'])
    xpos = ann.history.history['val_loss'].index(ymin)
    xmin = ann.history.history['val_loss'][xpos]
    plt.annotate('Minimum validation loss', size=20,  xy=(xpos, ymin), xytext=(xpos , ymin + 10000),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=5, headwidth=20),
                 horizontalalignment='center', verticalalignment='top',
                 )

    plt.title('ANN M 10 all data 15 September 2019',size=20)
    plt.ylabel('Mean squared error',size=20)
    plt.xlabel('Epochs',size=20)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    Metrics.write_results_multi('ANN_TEST_MULTI', data.test_x_df.reshape(
        (data.test_x_df.shape[0],
         data.sequence_len_minutes,
         data.number_of_features)),
                                data.test_y_df, y_pred)

    print(rmse)

def optimize():
    data = DataFrameSequenceMulti(False, True, True, True)
    data.build_ts_df(6, 19, [7,8,9,10,11,12], 20, 1)
    data.normalize_mega_df()
    data.split_data_set(10,15)
    data.flatten_data_set()

    # nodes = [(50,10),(60,20), (40,20)]
    activations = ['relu']
    opts = ['Adam']
    # drop_out = [0, 0.1, 0.5]
    # learning_rate = [0.001, 0.01, 0.1]

    ann = ann_model_multi.ANN_Multi(data, 100, 'ANN_BETA_SEQUENCE_TEST')
    ann.get_model()
    out = ann.train(100)
    res.append(out)
    # settings = 'nodes: ' + str(n) + ' activation: ' + str(a) + ' optimizer: ' + str(o) + ' dropout: ' + str(d) + ' lr: ' + str(lr)
    settings = 'settings'
    sets.append(settings)
    plot_history(settings, 0, out)
    min_loss.append(min(out.history['loss']))
    min_vals.append(min(out.history['val_loss']))

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

# optimize()
# run_ann_experiments()
# run_final_all_days()
ann_test()