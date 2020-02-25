from dataframe_sequence import DataFrameSequence
from metrics import Metrics
from models_ts import svr_model
import threading
import sys


start = 6
end = 19
prediction_horizons = list(range(1, 21))


def SVR_experiment_thread(prediction_horizon, minutes_sequence, cams):
    print('start SVR: ' + str(prediction_horizon))
    data = DataFrameSequence(False,prediction_horizon, True, False)
    data.build_ts_df(start,end,[7,8,9,10,11,12],minutes_sequence,cams)
    data.normalize_mega_df()
    name_cam = str(cams) + 'CAM_'
    name_time = str(minutes_sequence) + 'Minutes_'
    svr = svr_model.SVM_predictor(data, 'SVR SEQUENCE PREM_' + str(name_cam) + str(name_time) + '_pred_hor_' + str(prediction_horizon))
    svr.run_experiment()
    print('Finish SVR: ' + str(prediction_horizon))

def SVR_test():
    data = DataFrameSequence(False, 20, True, True)
    data.build_ts_df(start, end, [8,9], 5, 1)
    svr = svr_model.SVM_predictor(data, 'SVR SEQ TEST')
    svr.data.normalize_mega_df()
    svr.data.split_data_set(9, 11)
    svr.data.flatten_data_set()
    svr.train()
    y_pred, rmse, mae, mape = svr.predict()
    print(rmse)
    print(y_pred)
    # Metrics.write_results_SVR('SVR SEQ TEST', data.test_x_df.reshape(
    #     (data.test_x_df.shape[0], data.sequence_len_minutes, data.number_of_features)), data.test_y_df, y_pred, data.pred_horizon)

def run_svm_multi_thread(minutes_sequence, cams):
    for i in prediction_horizons:
        x = threading.Thread(target=SVR_experiment_thread, args=(i,minutes_sequence,cams))
        x.start()

def optimize():
    data = DataFrameSequence(False, 20, True, False)
    data.build_ts_df(6, 18, [7,8,9,10,11,12], 60, 1)
    data.normalize_mega_df()
    data.split_data_set(12, 1)
    data.flatten_data_set()
    svr = svr_model.SVM_predictor(data, 'SVR optimze')
    svr.optimize()

# minutes_sequence = int(sys.argv[1])
# cams = int(sys.argv[2])
# print('Minutes sequence: ' + str(minutes_sequence))
# print('Cams: ' + str(cams))
# run_svm_multi_thread(minutes_sequence, cams)
# # optimize()


data = DataFrameSequence(False, 20, True, False)
data.build_ts_df(10, 14, [8,9], 1, 1)
data.normalize_mega_df(ctn = [6,7,8], metoer_to_normalize= [9,10,13])
data.split_data_set(9, 15)
data.flatten_data_set_to_3d()

data.train_x_df = data.train_x_df.reshape(data.train_x_df.shape[0], data.train_x_df.shape[1] * data.train_x_df.shape[2])
data.test_x_df = data.test_x_df.reshape(data.test_x_df.shape[0], data.test_x_df.shape[1] * data.test_x_df.shape[2])
data.val_x_df= data.val_x_df.reshape(data.val_x_df.shape[0], data.val_x_df.shape[1] * data.val_x_df.shape[2])

from models import regression_model
reg = regression_model.Regression_predictor(data)
reg.train(data.train_x_df, data.train_y_df)
reg.predict(data.test_x_df, data.test_y_df)