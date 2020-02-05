from dataframe_sequence import DataFrameSequence
from models_ts import ann_model, lstm_model

def LSTM_experiment():
    data = DataFrameSequence(False, 20)
    data.build_ts_df(9,18,[7,8,9,10,11,12],30,2)
    lstm = lstm_model.LSTM_predictor(data, 400, 100, 'LSTM_BETA_SEQUENCE 2cam 30')
    lstm.run_experiment()
