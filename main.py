from data import Data
from models import persistence_model, regression_model, svm_model
import time
from data_visuals import plot_error
from tqdm import tqdm
import numpy as np
import logging
import threading
import time


prediction_horizons = list(range(1, 21))
print(prediction_horizons)

def run_experiment_thread(prediction_horizon):
    logging.info("Thread %s: starting", prediction_horizon)
    data = Data(meteor_data=True, images=False, debug=False)
    data.build_df(10, 17, 1, months=[7, 8, 9, 10, 11, 12])
    data.set_prediction_horizon(prediction_horizon)

    svm = svm_model.SVM_predictor(data)
    svm.run_experiment()

    logging.info("Thread %s: finishing", prediction_horizon)

def run_svm_multi_thread():
    for i in prediction_horizons:
        x = threading.Thread(target=run_experiment_thread, args=(i,))
        x.start()

def run_persistence_b():
    print('Running persistence b expiriments')
    data = Data(meteor_data=False, images=False, debug=False)
    data.build_df(10, 17, 1, months=[7, 8, 9, 10, 11, 12])

    persistence_b = persistence_model.Persistence_predictor_b(data)
    persistence_b.run_experiment()


run_persistence_b()


# data = Data(meteor_data=True, images=False, debug=False)
# for i in tqdm(prediction_horizons, total=len(prediction_horizons)):
#
#     # SETUP
#     # data.process_all_csv()
#     data.build_df(10, 17, 1, months=[7, 8, 9, 10, 11, 12])
#     data.set_prediction_horizon(i)
#
#     # MODELS
#     svm = svm_model.SVM_predictor(data)
#     svm.run_experiment()
#
#     # reg = regression_model.Regression_predictor(data)
#     # reg.run_experiment()
#     #
#     # persistence_a = persistence_model.Persistence_predictor_a(data)
#     # persistence_a.run_experiment()
#     #
#     # persistence_b = persistence_model.Persistence_predictor_b(data)
#     # persistence_b.run_experiment()


