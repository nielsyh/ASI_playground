from dataframe_sequence import DataFrameSequence
from models_ts import ann_model, lstm_model, svr_model
import sys

init_epochs = 100
epochs = 50
start = 8
end = 19

def LSTM_experiment(minutes_sequence, cams):
    data = DataFrameSequence(False, 20)
    data.build_ts_df(start,end,[7,8,9,10,11,12],minutes_sequence,cams)
    data.normalize_mega_df()
    # data.load_prev_mega_df()
    name_cam = str(cams) + 'CAM_'
    name_time = str(minutes_sequence) + 'Minutes_'
    lstm = lstm_model.LSTM_predictor(data, 100, 50, 'LSTM_BETA_SEQUENCE_' + str(name_cam) + str(name_time))
    lstm.run_experiment()


def SVR_experiment(minutes_sequence, cams):
    data = DataFrameSequence(False,20)
    data.build_ts_df(start,end,[7,8,9],minutes_sequence,cams)
    data.normalize_mega_df()
    # data.load_prev_mega_df()
    name_cam = str(cams) + 'CAM_'
    name_time = str(minutes_sequence) + 'Minutes_'
    svr = svr_model.SVM_predictor(data, 'SVR SEQUENCE_' + str(name_cam) + str(name_time))
    svr.run_experiment()

minutes_sequence = int(sys.argv[1])
cams = int(sys.argv[2])
print('Minutes sequence: ' + str(minutes_sequence))
print('Cams: ' + str(cams))
LSTM_experiment(minutes_sequence, cams)