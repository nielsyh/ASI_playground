from keras.models import load_model
import metrics
from data.data_frame_asi import DataFrameIMG
from data.dataframe_normal import DataFrameNormal
from tqdm import tqdm
from models import cnn_model
import numpy as np


def predict(data, model):
        y_pred =  model.predict(data.mega_df_x)
        rmse, mae, mape = metrics.Metrics.get_error(data.mega_df_y, y_pred)
        return y_pred, rmse, mae, mape

def experiment(prediction_horizon):
    prem = [(10, 5), (10, 6), (10, 7), (10, 8), (10, 20)]
    for tup in tqdm(prem, total=len(prem), unit='Days to predict'):
        data = DataFrameIMG(False, prediction_horizon)
        data.build_img_df(8, 17, tup[0], tup[1])
        cnn = cnn_model.resnet50(data)
        cnn.get_model(400)
        cnn.load_model_files(str(tup[0] + str(tup[1])))
        y_pred, rmse, mae, mape = cnn.predict()
        name = 'CNN_predhor' + str(prediction_horizon)
        metrics.Metrics.write_results(name, data.mega_df_times, data.mega_df_y, y_pred.ravel(), prediction_horizon)

def cnn_test():
    prem = [(10, 5)]
    for tup in tqdm(prem, total=len(prem), unit='Days to predict'):
        data = DataFrameIMG(False, 10)
        data.build_img_df(10, 12, tup[0], tup[1])
        cnn = cnn_model.resnet50(data)
        cnn.get_model(400)
        cnn.load_model_files(str(tup[0]) + str(tup[1]))
        y_pred, rmse, mae, mape = cnn.predict()
        name = 'CNN_predhor' + str(10)
        metrics.Metrics.write_results(name, data.mega_df_times, data.mega_df_y, y_pred.ravel(), 10)

def train():
    data = DataFrameNormal()
    data.build_df_for_cnn(6, 20, 1, [7,8,9,10,11,12])
    cnn = cnn_model.resnet50(data, 200, 200)
    cnn.build_prem_models()

prediction_horizons = list(range(1,21))
for i in prediction_horizons:
    experiment(i)
    print('done: ' + str(i))

# cnn_test()
# train()