from dataframe_normal import DataFrameNormal
from dataframe_sequence import DataFrameSequence
from models import persistence_model, regression_model, svm_model, cnn_model, ann_model
from models_ts import ann_model as aans
import time
from data_visuals import plot_error
from tqdm import tqdm
import numpy as np
import logging
import threading
import time
import sys

arg = sys.argv
# 1 model 1=svm 2=ann 3=persistence
# 2 debug 1=yes, default no
# 3 prediction horizon 0 default


prediction_horizons = list(range(1, 21))
print(prediction_horizons)

def SVM_experiment_thread(prediction_horizon):
    print("Thread %s: starting", prediction_horizon)

    data = DataFrameSequence(meteor_data=True, images=True, debug=False)
    # data.build_df(10, 17, 1, months=[7,8,9,10,11,12])
    data.load_prev_mega_df('mega_df_32_True_True_.npy')
    data.set_prediction_horizon(prediction_horizon)
    svm = svm_model.SVM_predictor(data, model_name='SVM norm: default + images + metoer')
    svm.run_experiment()
    print("Thread %s: finishing", prediction_horizon)

def run_svm_multi_thread():
    for i in prediction_horizons:
        x = threading.Thread(target=SVM_experiment_thread, args=(i,))
        x.start()

def persistence_b_thread(prediction_horizon):
    logging.info("PERSISTENCE b Thread %s: starting", prediction_horizon)
    data = DataFrameNormal(meteor_data=False, images=False, debug=False)
    data.build_df(9, 18, 1, months=[9, 10, 11, 12])
    data.set_prediction_horizon(prediction_horizon)
    persistence_b = persistence_model.Persistence_predictor_b(data)
    persistence_b.run_experiment()

    logging.info("Thread %s: finishing", prediction_horizon)

def run_persistenceB_multi_thread():
    prediction_horizons = [20]
    for i in prediction_horizons:
        x = threading.Thread(target=persistence_b_thread, args=(i,))
        x.start()

def train_cnn(prediction_horizon, months = [7,8,9,10,11,12]):
    data = DataFrameSequence(meteor_data=False, images=False, debug=False)
    data.build_df_for_cnn(12, 13, 1, months=months)
    # data.save_df_cnn()
    # data.set_prediction_horizon(prediction_horizon)
    cnn = cnn_model.resnet50(data, init_epochs=3, epochs=3)
    cnn.get_model(400)
    cnn.run_experiment()

def train_ann(prediction_horizon, months = [7,8,9,10,11,12]):
    print(months)
    data = DataFrameSequence(meteor_data=True, images=True, debug=False)
    # data.build_df(9, 17, 1, months=months)
    data.load_prev_mega_df('mega_df_32_True_True_.npy')
    data.set_prediction_horizon(prediction_horizon)
    ann = ann_model.ANN_Predictor(data, init_epochs=200, epochs=200)
    ann.get_model()
    ann.run_experiment()

# run_persistenceB_multi_thread()

data = DataFrameNormal(meteor_data=False, images=False, debug=False)
data.build_df(9, 18, 1, months=[9, 10, 11, 12])
data.set_prediction_horizon(20)
persistence_b = persistence_model.Persistence_predictor_b(data)
persistence_b.run_experiment()


