from dataframe_sequence import DataFrameSequence
from metrics import Metrics
from models_ts import ann_model, lstm_model
import sys

init_epochs = 200
epochs = 50
start = 6
end = 20

def ann_experiment(minutes_sequence, cams):
    data = DataFrameSequence(False, 20, True)
    data.build_ts_df(start, end, [7, 8, 9, 10, 11, 12], minutes_sequence, cams)
    data.save_df()
    data.normalize_mega_df()
    name_cam = str(cams) + 'CAM_'
    name_time = str(minutes_sequence) + 'Minutes_'
    ann = ann_model.ANN(data, init_epochs, epochs, 'ANN_BETA_SEQUENCE_IMG' + name_cam + name_time)
    ann.run_experiment()

def ann_test():
    data = DataFrameSequence(True, 20)
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

minutes_sequence = int(sys.argv[1])
cams = int(sys.argv[2])
print('Minutes sequence: ' + str(minutes_sequence))
print('Cams: ' + str(cams))
ann_experiment(minutes_sequence, cams)
# ann_test()


