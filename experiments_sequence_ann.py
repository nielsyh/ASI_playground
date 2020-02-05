from dataframe_sequence import DataFrameSequence
from models_ts import ann_model, lstm_model
import sys

init_epochs = 200
epochs = 50
start = 8
end = 19

def ann_experiment(minutes_sequence, cams):
    data = DataFrameSequence(False, 20)
    data.build_ts_df(start, end, [7, 8, 9, 10, 11, 12], minutes_sequence, cams)
    data.normalize_mega_df()
    name_cam = str(cams) + 'CAM_'
    name_time = str(minutes_sequence) + 'Minutes_'
    ann = ann_model.ANN(data, init_epochs, epochs, 'ANN_BETA_SEQUENCE_' + name_cam + name_time)
    ann.run_experiment()

def ann_test():
    data = DataFrameSequence(True, 20)
    data.build_ts_df(start, end, [7, 8, 9], 45, 1)
    data.normalize_mega_df()
    ann = ann_model.ANN(data, init_epochs, epochs, 'ANN_BETA_SEQUENCE_TEST')

    data.split_data_set(9, 27)
    data.flatten_data_set()
    ann.get_model()

    ann.train(100)
    y_pred, rmse, mae, mape = ann.predict()
    print(rmse)


# ann_test()
minutes_sequence = int(sys.argv[1])
cams = int(sys.argv[2])
print('Minutes sequence: ' + str(minutes_sequence))
print('Cams: ' + str(cams))
ann_experiment(minutes_sequence, cams)



