from data import Data
from models import persistence_model, regression_model, svm_model

data = Data(pred_horzion=10, meteor_data=False, debug=False)
data.build_df(7, 19, 1, months=[9])
data.label_df()
data.split_data_set()
data.flatten_data_set()

svm = svm_model.SVM_predictor(data)
svm.train()
svm.predict()

reg = regression_model.Regression_predictor(data)
reg.train()
reg.predict()

