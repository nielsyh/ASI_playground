from data import Data
from models import persistence_model, regression_model, svm_model
import time
from data_visuals import plot_error
from tqdm import tqdm
import numpy as np

# prediction_horizons = list(range(1, 31))
prediction_horizons = [1,3,5,10,15,20,30]
print(prediction_horizons)

errors_svm = []
errors_log_reg = []
errors_persistence = []

for i in tqdm(prediction_horizons, total=len(prediction_horizons)):

    # SETUP
    data = Data(pred_horzion=i, meteor_data=True, images=False, debug=False)
    # data.download_data()
    data.build_df(10, 17, 1, months=[9])
    data.label_df()
    data.split_data_set()
    data.flatten_data_set()
    data.normalize_data_sets()


    # MODELS
    svm = svm_model.SVM_predictor(data)
    start_time = time.time()
    svm.train()
    print("--- %s seconds ---" % (time.time() - start_time))

    rmse_s, mae_s, mape_s = svm.predict()
    errors_svm.append((rmse_s, mae_s, mape_s))

    reg = regression_model.Regression_predictor(data)
    start_time = time.time()
    reg.train()
    print("--- %s seconds ---" % (time.time() - start_time))

    rmse_reg, mae_reg, mape_reg = reg.predict()
    errors_log_reg.append((rmse_reg, mae_reg, mape_reg))

    persistence = persistence_model.Persistence_predictor(data)
    rmse_p, mae_p, mape_p = persistence.predict()
    errors_persistence.append((rmse_p, mae_p, mape_p))

print(errors_svm)
print(errors_log_reg)
print(errors_persistence)

np.savetxt('errors_svm', errors_svm, delimiter=',')
np.savetxt('errors_log_reg', errors_log_reg, delimiter=',')
np.savetxt('errors_persistence', errors_persistence, delimiter=',')

plot_error(errors_svm, errors_log_reg, errors_persistence, 'Error RMSE', 'Prediction horizon in minutes', 'Error in RMSE', prediction_horizons, 0) #rmse
plot_error(errors_svm, errors_log_reg, errors_persistence, 'Error MAE', 'Prediction horizon in minutes', 'Error in MAE', prediction_horizons, 1) #mae
plot_error(errors_svm, errors_log_reg, errors_persistence, 'Error MAPE', 'Prediction horizon in minutes', 'Error in MAPE', prediction_horizons, 2) #mape
#

