from data.dataframe_sequence_multi import DataFrameSequenceMulti
from models.models_ts_multi import ann_model_multi, rf_model_multi, lstm_model_multi
import time

epochs_ann = 36
epochs_lstm = 56

data = DataFrameSequenceMulti(False, True, True, True)
data.build_ts_df(6, 19, [8, 9], 10)
data.normalize_mega_df()
data.split_data_set(9, 27)
data.flatten_data_set()

start_time = time.time()
rf = rf_model_multi.RF_predictor(data, 'RF SEQUENCE multi PREM_ NO METOER')
rf.train()
print("RF TRAINING --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
y_pred, rmse, mae, mape = rf.predict()
print("RF PREDICTION --- %s seconds ---" % (time.time() - start_time))

ann = ann_model_multi.ANN_Multi(data, 3, 'ANN_BETA_SEQUENCE_MUTLI_TEST')
ann.get_model()
start_time = time.time()
ann.train(epochs_ann)
print("ANN TRAINING--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
y_pred, rmse, mae, mape = ann.predict()
print("ANN prediction--- %s seconds ---" % (time.time() - start_time))


data = DataFrameSequenceMulti(False, True, True, True)
data.build_ts_df(6, 19, [8, 9], 10)
lstm = lstm_model_multi.LSTM_predictor(data, 100, 'LSTM_TEST', pred_csi=True)
data.normalize_mega_df()
data.split_data_set(9, 27)
data.flatten_data_set_to_3d()
lstm.get_model()

start_time = time.time()
lstm.train(epochs_lstm)
print("LSTM TRAINING--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
y_pred, rmse, mae, mape = lstm.predict()
print("LSM prediction--- %s seconds ---" % (time.time() - start_time))
