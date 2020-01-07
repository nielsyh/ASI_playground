from data import Data
from models import persistence_model, regression_model, svm_model
import time

# prediction_horizons = list(range(1, 31))
prediction_horizons = [1, 5, 10, 20]
print(prediction_horizons)

errors_svm = []
errors_log_reg = []

for i in prediction_horizons:
    data = Data(pred_horzion=i, meteor_data=False, images=False, debug=True)
    # data.download_data()
    data.build_df(7, 19, 1, months=[9])
    data.label_df()
    data.split_data_set()
    data.flatten_data_set()

    svm = svm_model.SVM_predictor(data)
    start_time = time.time()
    svm.train()
    print("--- %s seconds ---" % (time.time() - start_time))

    rmse, mae, mape = svm.predict()
    errors_svm.append((rmse, mae, mape))
    #
    # svm.save('test_model_save_' + str(i))

    reg = regression_model.Regression_predictor(data)
    start_time = time.time()
    reg.train()
    print("--- %s seconds ---" % (time.time() - start_time))

    rmse, mae, mape = reg.predict()
    errors_log_reg.append((rmse, mae, mape))

print(errors_svm)
print(errors_log_reg)
