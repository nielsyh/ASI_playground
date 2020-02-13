from dataframe_sequence import DataFrameSequence
from metrics import Metrics
from models_ts import ann_model, lstm_model, svr_model
import threading
import sys

init_epochs = 100
epochs = 50
start = 6
end = 20
prediction_horizons = list(range(1, 21))

def LSTM_experiment(minutes_sequence, cams):
    data = DataFrameSequence(False, 20, False)
    data.build_ts_df(start,end,[7,8,9,10,11,12],minutes_sequence,cams)
    name_cam = str(cams) + 'CAM_'
    name_time = str(minutes_sequence) + 'Minutes_'
    lstm = lstm_model.LSTM_predictor(data, 100, 50, 'LSTM_BETA_SEQUENCE_' + str(name_cam) + str(name_time))
    lstm.run_experiment()

def LSTM_test():
    data = DataFrameSequence(True, 20)
    data.build_ts_df(6, 12, [9], 60, 1)
    lstm = lstm_model.LSTM_predictor(data, 100, 50, 'LSTM_TEST')
    data.normalize_mega_df()
    data.split_data_set(9, 27)
    data.flatten_data_set_to_3d()
    lstm.get_model()
    lstm.train(3)
    y_pred, rmse, mae, mape = lstm.predict()
    Metrics.write_results_NN('LSTM_TEST', data.test_x_df, data.test_y_df, y_pred, data.pred_horizon)

minutes_sequence = int(sys.argv[1])
cams = int(sys.argv[2])
print('Minutes sequence: ' + str(minutes_sequence))
print('Cams: ' + str(cams))
LSTM_experiment(minutes_sequence, cams)

