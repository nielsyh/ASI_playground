from data import Data
from models import persistence_model, regression_model, svm_model, cnn_model, ann_model
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
# prediction_horizons = [1]
print(prediction_horizons)

def SVM_experiment_thread(prediction_horizon):
    logging.info("Thread %s: starting", prediction_horizon)

    data = Data(meteor_data=True, images=False, debug=False)
    data.build_df(10, 17, 1, months=[7,8,9,10,11,12])
    data.set_prediction_horizon(prediction_horizon)

    svm = svm_model.SVM_predictor(data, model_name='SVM norm: 3-7,16')
    svm.run_experiment()

    logging.info("Thread %s: finishing", prediction_horizon)

def run_svm_multi_thread():
    for i in prediction_horizons:
        x = threading.Thread(target=SVM_experiment_thread, args=(i,))
        x.start()

def persistence_b_thread(prediction_horizon):
    logging.info("PERSISTENCE b Thread %s: starting", prediction_horizon)

    data = Data(meteor_data=False, images=False, debug=False)
    data.build_df(10, 17, 1, months=[7, 8, 9, 10, 11, 12])
    data.set_prediction_horizon(prediction_horizon)

    persistence_b = persistence_model.Persistence_predictor_b(data)
    persistence_b.run_experiment()

    logging.info("Thread %s: finishing", prediction_horizon)

def run_persistenceB_multi_thread():
    for i in prediction_horizons:
        x = threading.Thread(target=persistence_b_thread, args=(i,))
        x.start()

def train_cnn(month, day, months = [7,8,9,10,11,12]):
    data = Data(meteor_data=False, images=False, debug=False)
    data.build_df_for_cnn(10, 17, 1, months=months)
    data.split_data_set(month, day)
    data.flatten_data_set_CNN()
    cnn = cnn_model.resnet50(400, data)
    model = cnn.get_model(400)
    model.train(model.get_model(400), data.x_train, data.y_train)

def train_ann(prediction_horizon, months = [7,8,9,10,11,12]):
    data = Data(meteor_data=True, images=False, debug=False)
    data.build_df(10, 17, 1, months=months)
    data.set_prediction_horizon(prediction_horizon)
    ann = ann_model.ANN_Predictor(data, init_epochs=500, epochs=100)
    ann.get_model()
    ann.run_experiment()

# run_persistenceB_multi_thread()
# run_svm_multi_thread()
# train_ann(20, months=[7,8,9,10,11,12])
train_ann(20, months=[7,8,9])