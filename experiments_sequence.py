from dataframe_sequence import DataFrameSequence
from models_ts import ann_model, lstm_model

# data.save_df()
# data.split_data_set(8,25)
# data.flatten_data_set_to_3d()
# data.normalize_data_sets()
# data.load_prev_mega_df()

# lstm = lstm_model.LSTM_predictor(data, 400,100)
# lstm.get_model()
# lstm.run_experiment()

def ann_expiriment_2_cam():
    data = DataFrameSequence(False, 20)
    data.build_ts_df(7,20,[7,8,9,10,11,12],45,2)
    ann = ann_model.ANN(data, 50, 50, 'ANN_BETA_SEQUENCE_2cam_45min')
    ann.run_experiment()

ann_expiriment_2_cam()