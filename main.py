from data import Data
from models import persistence_model, regression_model, svm_model
import time
from data_visuals import plot_error
from tqdm import tqdm
import numpy as np

prediction_horizons = list(range(1, 21))
# prediction_horizons = [1,5]
print(prediction_horizons)


data = Data(meteor_data=True, images=False, debug=False)
for i in tqdm(prediction_horizons, total=len(prediction_horizons)):

    # SETUP
    # data.download_data()
    data.build_df(10, 17, 1, months=[7, 8, 9, 10, 11, 12])
    data.set_prediction_horizon(i)

    # MODELS
    svm = svm_model.SVM_predictor(data)
    svm.run_experiment()

    reg = regression_model.Regression_predictor(data)
    reg.run_experiment()

    persistence_a = persistence_model.Persistence_predictor_a(data)
    persistence_a.run_experiment()

    persistence_b = persistence_model.Persistence_predictor_b(data)
    persistence_b.run_experiment()

