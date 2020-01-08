from data import Data
from models import persistence_model, regression_model, svm_model
import time
from data_visuals import plot_error
from tqdm import tqdm
import numpy as np

# prediction_horizons = list(range(1, 31))
prediction_horizons = [1,3,5,10,20,30]
print(prediction_horizons)

errors_svm = []
errors_log_reg = []

for i in tqdm(prediction_horizons, total=len(prediction_horizons)):
    data = Data(pred_horzion=i, meteor_data=False, images=False, debug=False)
    # data.download_data()
    data.build_df(10, 17, 1, months=[9,10,11,12])
    data.label_df()
    data.split_data_set()
    data.flatten_data_set()
    data.normalize_data_sets()

    svm = svm_model.SVM_predictor(data)
    start_time = time.time()
    svm.train()
    print("--- %s seconds ---" % (time.time() - start_time))

    rmse, mae, mape = svm.predict()
    errors_svm.append((rmse, mae, mape))

    reg = regression_model.Regression_predictor(data)
    start_time = time.time()
    reg.train()
    print("--- %s seconds ---" % (time.time() - start_time))

    rmse2, mae2, mape2 = reg.predict()
    errors_log_reg.append((rmse2, mae2, mape2))

print(errors_svm)
np.savetxt('errors_svm', errors_svm, delimiter=',')
print(errors_log_reg)
np.savetxt('errors_log_reg', errors_log_reg, delimiter=',')

plot_error(errors_svm, errors_log_reg, 'Error RMSE', 'Prediction horizon in minutes', 'Error in RMSE', prediction_horizons, 0) #rmse
plot_error(errors_svm, errors_log_reg, 'Error MAE', 'Prediction horizon in minutes', 'Error in MAE', prediction_horizons, 1) #mae
plot_error(errors_svm, errors_log_reg, 'Error MAPE', 'Prediction horizon in minutes', 'Error in MAPE', prediction_horizons, 2) #mape


