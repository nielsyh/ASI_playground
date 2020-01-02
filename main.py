from data import Data
from models import persistence_model, regression_model, svm_model

# prediction_horizons = list(range(1, 31))
prediction_horizons = [1,5,10,20]
print(prediction_horizons)


errors_svm = []
errors_log_reg = []

for i in prediction_horizons:
    data = Data(pred_horzion=i, meteor_data=False, debug=True)
    data.build_df(7, 19, 1, months=[9])
    data.label_df()
    data.split_data_set()
    data.flatten_data_set()

    svm = svm_model.SVM_predictor(data)
    svm.train()
    rmse, mae, mape = svm.predict()
    errors_svm.append((rmse, mae, mape))

    reg = regression_model.Regression_predictor(data)
    reg.train()
    rmse, mae, mape = reg.predict()
    errors_log_reg.append((rmse, mae, mape))


print(errors_svm)
print(errors_log_reg)